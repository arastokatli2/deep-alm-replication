"""Layer 3: move the bank forward one month at a time.

Takes x0 from Layer 2 and a set of yield-curve scenarios from Layer 1, and
walks the balance sheet through H monthly steps along every scenario at once.

Order inside one step, following [M] section 3.3 / [E] section 2.3:

    roll forward   realise last month's cash flows; loans and deposits roll
                   over and grow; bond portfolios shift up by one month
    decide         the policy picks how much to invest and to borrow
    restructure    the new bonds join the portfolios, cash pays for them
    annual step    every 12th month: pay the dividend, check the yearly return

Step 0 has no roll-forward: the starting balance sheet is already settled
([M] section 3.3.1).  The last step is a roll-forward with no decision.

Everything is vectorised over scenarios.  Arrays are (S, N) for one cash-flow
vector, (S, 4, N) for deposits, which are tracked per reference maturity
because the rolling scheme needs to know which one a maturing amount came from.

Units: CHF millions.  Rates are decimals.  Equations cited as [E] / [M].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from balance_sheet import InitialState, N, REF_MATURITIES
import constraints as cs

DT = 1.0 / 12.0


# ====================================================================== setup

@dataclass(frozen=True)
class Params:
    """Every constant the transition needs.  Defaults are the papers' values;
    those the papers withhold are marked and calibrated or assumed."""

    # --- loans ---------------------------------------------------------
    loan_growth: float = 0.03               # rho_L,  [E] (8)
    loan_spread: float = float("nan")       # kappa_L, withheld (D11) -> calibrated
    enterprise_new_months: tuple = (1, 2, 3)        # [E] 2.2.2: 1m-3m, equally
    mortgage_new_months: tuple = tuple(range(24, 145, 12))   # [E] 2.2.2: 2y-12y
    mortgage_new_weights: tuple = ()        # withheld (D12); empty -> fitted, see below
    depreciation_trigger: float = 0.02      # [E] 2.2.2: 6m rate up by more than 2 %

    # --- deposits ------------------------------------------------------
    nmd_growth: float = 0.04                # rho_SD, [E] (10)
    term_growth: float = 0.01               # rho_SF, [M] (31)
    nmd_rate_factor: float = 0.60           # [E] (11a)
    nmd_rate_cap: float = 0.03
    term_rate_factor: float = 0.85          # [E] (11b)
    term_rate_spread: float = 0.0025
    term_rate_cap: float = 0.05

    # --- bonds ---------------------------------------------------------
    invest_maturities: tuple = tuple(range(36, 181, 12))      # 3y-15y, b_B = 13
    finance_maturities: tuple = (3,) + tuple(range(12, 181, 12))   # 3m, 1y-15y, b_K = 16
    invest_spread: float = -0.0015          # kappa_B = -15 bp, [E] 2.2.1
    finance_spread: float = 0.0015          # kappa_K = +15 bp

    # --- costs and payout ----------------------------------------------
    personnel_growth: float = 0.02          # [E] 2.2.4
    dividend_share: float = 0.5             # [M] (41)

    # --- cash penalty ----------------------------------------------------
    reserve_exemption: float = 30.0         # 30 x MR, [E] (14)

    # --- switches on the two ambiguous points ----------------------------
    deposit_cash_rule: str = "reinvest"     # B6, see RESOLVED below
    cash_penalty_sign: str = "corrected"    # B7, see RESOLVED below
    deposit_rate_floor: float | None = None  # B8: None = the paper as printed


RESOLVED = {
    "B6": "[E] (13) subtracts maturing deposits from cash while [E] 2.2.3 says the "
          "same amounts are reinvested. Taken literally the deposit base would run "
          "off completely and the bank would fail, so the two cannot both hold. We "
          "reinvest (deposit_cash_rule='reinvest'): maturing deposits move no cash, "
          "credited interest raises the liability and lowers equity, and only the "
          "growth is new cash. 'literal' reproduces the printed formula.",
    "B7": "[E] (14) defines the cash penalty as (C-30MR)+ (min(1/D,1)-1), which is "
          "zero or NEGATIVE, and (15) then subtracts it, so negative rates would pay "
          "the bank. The sign inside is wrong. We use (C-30MR)+ (1-min(1/D,1)) >= 0 "
          "('corrected'); 'literal' reproduces the printed formula.",
    "B8": "[E] (11a)-(11b) cap customer deposit rates but do not floor them, so on "
          "a start date with negative rates the bank charges its depositors. [E] 1 "
          "itself notes that Swiss customer rates 'were often floored at 0%'. We "
          "keep the formulas as printed (deposit_rate_floor=None) and leave the "
          "floor as an option, because the omission is the paper's, not a typo. "
          "It is worth about 17 mCHF a year at the start and fades as rates rise.",
}


class Policy(Protocol):
    """Chooses how much to invest and to borrow, per scenario, per step."""

    def __call__(self, step: int, state: "SimState", ctx: "Context"
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Return (a_B, a_K), shapes (S, b_B) and (S, b_K), both >= 0.

        An entry is a number of par bonds, so it is also the cash amount."""


@dataclass
class Context:
    """Everything about the current month that a policy may look at."""
    yields: np.ndarray          # (S, N) current yield curve
    discount: np.ndarray        # (S, N) current discount factors
    params: Params
    values: dict                # economic values of every slot, (S,) arrays
    last_constraints: dict      # the previous month's ratios, as [E] 3.2.1.4 does
    matured_B: np.ndarray = None    # (S,) investment cash that came in this month
    matured_K: np.ndarray = None    # (S,) financing cash that was repaid this month


# ====================================================================== state

@dataclass
class SimState:
    """The bank, for every scenario at once."""
    C: np.ndarray               # (S,)
    B: np.ndarray               # (S, N)
    K: np.ndarray               # (S, N)
    RP: np.ndarray              # (S, N) mortgage principal
    RE: np.ndarray              # (S, N) enterprise-loan principal
    IRP: np.ndarray             # (S, N) monthly interest each mortgage tranche pays
    IRE: np.ndarray             # (S, N) same for enterprise loans
    SD: np.ndarray              # (S, 4, N) non-maturing deposits by reference maturity
    SF: np.ndarray              # (S, 4, N) term deposits
    E_last_annual: np.ndarray   # (S,) equity at the previous annual step
    y6m_history: list = field(default_factory=list)   # 6m yield, most recent last

    @property
    def R(self) -> np.ndarray:
        return self.RP + self.RE

    @property
    def S_total(self) -> np.ndarray:
        return self.SD.sum(axis=1) + self.SF.sum(axis=1)

    def values(self, D: np.ndarray) -> dict:
        v = {"C": self.C,
             "B": np.einsum("sn,sn->s", D, self.B),
             "RP": np.einsum("sn,sn->s", D, self.RP),
             "RE": np.einsum("sn,sn->s", D, self.RE),
             "SD": np.einsum("sn,san->s", D, self.SD),
             "SF": np.einsum("sn,san->s", D, self.SF),
             "K": np.einsum("sn,sn->s", D, self.K)}
        v["A"] = v["C"] + v["B"] + v["RP"] + v["RE"]
        v["L"] = v["SD"] + v["SF"] + v["K"]
        v["E"] = v["A"] - v["L"]
        return v


def _shift_up(x: np.ndarray) -> np.ndarray:
    """The operator U of [E] (5): drop the first entry, append a zero."""
    out = np.zeros_like(x)
    out[..., :-1] = x[..., 1:]
    return out


def _reinvest_shape(m: int) -> np.ndarray:
    """A maturing deposit of reference maturity m months is put back in equal
    monthly tranches over 1..m ([E] 2.2.3). Flat, exactly as printed.

    Note this is NOT the shape such a portfolio settles at: that is the straight
    line down of balance_sheet._steady_shape, which is what x0 starts from."""
    v = np.zeros(N)
    v[:m] = 1.0 / m
    return v


REINVEST = np.vstack([_reinvest_shape(m) for m in REF_MATURITIES])   # (4, N)


# ============================================================ calibration

def loan_rate(y: np.ndarray, spread: float) -> np.ndarray:
    """Monthly interest on one unit of loan, [E] (9) / [M] (30).

    r = (exp((Y(tau) + kappa_L)/12) - 1)+ ; never negative to the customer."""
    return np.maximum(np.exp((y + spread) / 12.0) - 1.0, 0.0)


def calibrate_loan_spread(x0: InitialState, y0: np.ndarray,
                          annual_interest_target: float) -> float:
    """Pick kappa_L so the legacy book pays what the bank actually earned.

    [E] withholds kappa_L (open item D11).  Legacy loans are priced off Y0
    ([E] 2.2.2), so the model's first-year loan interest is a function of
    kappa_L alone, and it is increasing in it.  One bisection.
    """
    R = x0.RP + x0.RE
    y = np.asarray(y0)[:N]

    def annual(sp: float) -> float:
        return 12.0 * float(R @ loan_rate(y, sp))

    lo, hi = -0.05, 0.20
    if annual(hi) < annual_interest_target:
        raise ValueError("target above what kappa_L = 20 % can pay")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if annual(mid) < annual_interest_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_mortgage_mix(RP0: np.ndarray, months: tuple, smooth: float = 0.3) -> np.ndarray:
    """New-mortgage maturity split, [E] 2.2.2, withheld by the bank (D12).

    A portfolio fed every month by the split w, and rolled by U, settles at a
    profile proportional to the reverse cumulative sum of w.  We pick the w
    over the allowed maturities whose settled profile best matches the bank's
    actual mortgage profile, by non-negative least squares.  That keeps the
    replication on TKB's own data instead of an invented split.
    """
    from scipy.optimize import nnls
    months = np.asarray(months)
    # column j: settled profile from putting one unit at maturity months[j]
    A = np.zeros((N, len(months)))
    for j, m in enumerate(months):
        A[:m, j] = 1.0
    target = np.asarray(RP0, dtype=float)
    # A small penalty on second differences keeps the split from collapsing onto
    # one or two maturities, which plain NNLS tends to do and no real bank does.
    n = len(months)
    D2 = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    scale = smooth * np.linalg.norm(A) / max(np.linalg.norm(D2), 1e-12)
    w, _ = nnls(np.vstack([A, scale * D2]),
                np.concatenate([target, np.zeros(D2.shape[0])]))
    if w.sum() <= 0:
        return np.full(len(months), 1.0 / len(months))
    return w / w.sum()


# ============================================================ bond pricing

def _coupon_months(m: int) -> list[int]:
    """Semi-annual coupon dates counted back from maturity ([E] (6))."""
    out = []
    k = m
    while k >= 1:
        out.append(k)
        k -= 6
    return sorted(out)


def par_bonds(D: np.ndarray, maturities: tuple, spread: float) -> np.ndarray:
    """Cash-flow vector of one unit of each available bond, after the spread.

    The coupon makes the bond trade at par, [E] (7):
        alpha = (1 - D[m]) / sum of D over the coupon dates.
    The bank then receives (investments) or pays (financing) a coupon moved by
    the annualised spread, pro-rated over the coupon period ([E] 2.2.1).

    D : (S, N) discount factors.  Returns (S, n_bonds, N).
    """
    S = D.shape[0]
    out = np.zeros((S, len(maturities), N))
    for j, m in enumerate(maturities):
        cm = _coupon_months(m)
        idx = np.array(cm) - 1
        denom = D[:, idx].sum(axis=1)
        alpha = (1.0 - D[:, m - 1]) / denom                      # (S,)
        period = np.diff([0] + cm) / 12.0                        # years per coupon
        adj = alpha[:, None] + spread * period[None, :]
        out[:, j, idx] = adj
        out[:, j, m - 1] += 1.0
    return out


# ============================================================ the step parts

def deposit_rates(y6m_avg: np.ndarray, y6m_now: np.ndarray, p: Params
                  ) -> tuple[np.ndarray, np.ndarray]:
    """[E] (11a)-(11b): what the bank pays on each kind of deposit."""
    uD = np.minimum(p.nmd_rate_factor * y6m_avg, p.nmd_rate_cap)
    uF = np.minimum(np.maximum(p.term_rate_factor * y6m_avg,
                               y6m_now - p.term_rate_spread), p.term_rate_cap)
    if p.deposit_rate_floor is not None:                             # B8
        uD = np.maximum(uD, p.deposit_rate_floor)
        uF = np.maximum(uF, p.deposit_rate_floor)
    return uD, uF


def roll_deposits(dep: np.ndarray, u: np.ndarray, growth: float, p: Params
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One month of the deposit rolling scheme, [E] 2.2.3.

    dep : (S, 4, N).  Returns the new portfolio and the cash it brings in.

    What matures is put straight back over 1..tau.  Credited interest
    ([E] (12)) is put back the same way: it raises what the bank owes and
    costs equity, and moves no cash.  New money (the growth) is cash in.
    """
    matured = dep[:, :, 0]                                       # (S, 4)
    rolled = _shift_up(dep)
    total = dep.sum(axis=(1, 2))                                 # (S,)
    interest_rate = np.exp(u / 12.0) - 1.0                       # (S,)

    weights = dep.sum(axis=2)                                    # (S, 4)
    share = np.divide(weights, weights.sum(axis=1, keepdims=True),
                      out=np.zeros_like(weights),
                      where=weights.sum(axis=1, keepdims=True) > 0)
    interest = (interest_rate * total)[:, None] * share          # (S, 4)
    new_money = (growth / 12.0 * total)[:, None] * share         # (S, 4)

    if p.deposit_cash_rule == "literal":                         # B6
        put_back = interest + new_money
        cash_in = new_money.sum(axis=1) + interest.sum(axis=1) - matured.sum(axis=1)
    else:
        put_back = matured + interest + new_money
        cash_in = new_money.sum(axis=1)

    return rolled + put_back[:, :, None] * REINVEST[None, :, :], cash_in


def roll_loans(RP, RE, IRP, IRE, y: np.ndarray, p: Params, mortgage_mix: np.ndarray
               ) -> tuple:
    """One month of the loan book, [E] (8).

    Maturing principal comes back as cash, new loans go out as cash, and every
    outstanding tranche pays its monthly interest ([E] (9)).  New loans are
    priced off today's curve and keep that rate until they mature.
    """
    interest_in = IRP.sum(axis=1) + IRE.sum(axis=1)              # r_t
    repaid = RP[:, 0] + RE[:, 0]

    new = {}
    for name, R, months, weights in (
            ("P", RP, p.mortgage_new_months, mortgage_mix),
            ("E", RE, p.enterprise_new_months,
             np.full(len(p.enterprise_new_months), 1.0 / len(p.enterprise_new_months)))):
        size = R[:, 0] + p.loan_growth / 12.0 * R.sum(axis=1)    # [E] (8)
        vec = np.zeros_like(R)
        irv = np.zeros_like(R)
        for w, m in zip(weights, months):
            amount = size * w
            vec[:, m - 1] += amount
            irv[:, m - 1] += amount * loan_rate(y[:, m - 1], p.loan_spread)
        new[name] = (vec, irv, size)

    RP2 = _shift_up(RP) + new["P"][0]
    RE2 = _shift_up(RE) + new["E"][0]
    IRP2 = _shift_up(IRP) + new["P"][1]
    IRE2 = _shift_up(IRE) + new["E"][1]
    granted = new["P"][2] + new["E"][2]
    return RP2, RE2, IRP2, IRE2, interest_in, repaid, granted


def cash_penalty(C: np.ndarray, MR: np.ndarray, D_1m: np.ndarray, p: Params
                 ) -> np.ndarray:
    """[E] (14), sign corrected (B7).

    Cash above 30 x the minimum reserve earns the market short rate, which only
    bites when that rate is negative.  Returns a non-negative charge.
    """
    excess = np.maximum(C - p.reserve_exemption * MR, 0.0)
    factor = 1.0 - np.minimum(1.0 / D_1m, 1.0)                   # >= 0 iff rate < 0
    if p.cash_penalty_sign == "literal":
        factor = np.minimum(1.0 / D_1m, 1.0) - 1.0
    return excess * factor


def monthly_cost(step: int, x0: InitialState, p: Params) -> float:
    """[E] 2.2.4: material flat, personnel up 2 % a year."""
    years = step // 12
    return (x0.monthly_material_cost
            + x0.monthly_personnel_cost * (1.0 + p.personnel_growth) ** years)


# ============================================================ policies

class NoAction:
    """Invest and borrow nothing.  The reference path: it shows what the
    starting balance sheet does on its own."""

    def __call__(self, step, state, ctx):
        S = state.C.shape[0]
        return (np.zeros((S, len(ctx.params.invest_maturities))),
                np.zeros((S, len(ctx.params.finance_maturities))))


@dataclass
class RollOver:
    """A plain benchmark in the spirit of [E] BM_E: put a fixed share of the
    month's free cash into bonds, spread equally over the maturities, and
    borrow back whatever matured out of the financing book."""
    invest_share: float = 0.25
    refinance: bool = True

    def __call__(self, step, state, ctx):
        S = state.C.shape[0]
        nB, nK = len(ctx.params.invest_maturities), len(ctx.params.finance_maturities)
        # cash above the central-bank exemption is idle, so a share of it is invested
        exempt = ctx.params.reserve_exemption * 0.025 * (ctx.values["SD"] + ctx.values["SF"])
        spare = np.maximum(state.C - exempt, 0.0)
        aB = (self.invest_share * spare / nB)[:, None] * np.ones((1, nB))
        aK = np.zeros((S, nK))
        if self.refinance:
            aK = (ctx.matured_K / nK)[:, None] * np.ones((1, nK))
        return aB, aK


# ============================================================ the simulation

def simulate(x0: InitialState, yields: np.ndarray, p: Params,
             policy: Policy = NoAction(), *, mortgage_mix: np.ndarray | None = None,
             record_scenarios: int = 0, record_books: bool = False) -> dict:
    """Walk every scenario forward.

    yields : (S, H+1, N) simulated yield curves from Layer 1, month by month.
    Returns per-step summary statistics plus the final equity of each scenario.
    """
    yields = np.asarray(yields, dtype=float)
    S, steps, _ = yields.shape
    H = steps - 1
    if np.isnan(p.loan_spread):
        raise ValueError("Params.loan_spread is unset; calibrate it first")
    if mortgage_mix is None:
        mortgage_mix = fit_mortgage_mix(x0.RP, p.mortgage_new_months)

    tau = np.arange(1, N + 1) / 12.0
    disc = np.exp(-tau[None, None, :] * yields)                  # (S, H+1, N)
    # the +100 bp curve of [E] (25) only multiplies the discount factors
    disc_shift = disc * np.exp(-tau * cs.IRS_SHIFT)[None, None, :]

    def tile(v):
        return np.tile(np.asarray(v, dtype=float), (S, 1))

    y0 = yields[:, 0, :]
    st = SimState(
        C=np.full(S, float(x0.C)),
        B=tile(x0.B), K=tile(x0.K), RP=tile(x0.RP), RE=tile(x0.RE),
        IRP=tile(x0.RP) * loan_rate(y0, p.loan_spread),
        IRE=tile(x0.RE) * loan_rate(y0, p.loan_spread),
        SD=np.tile(np.asarray(x0.SD)[None, :, :], (S, 1, 1)),
        SF=np.tile(np.asarray(x0.SF)[None, :, :], (S, 1, 1)),
        E_last_annual=np.zeros(S),
    )
    st.y6m_history = [yields[:, 0, 5]] * 3
    y6m_year_ago = [yields[:, 0, 5]] * 12

    v = st.values(disc[:, 0, :])
    st.E_last_annual = v["E"].copy()
    D_shift = disc_shift[:, 0, :]
    last_con = cs.all_constraints(v, st.values(D_shift), yields[:, 0, 0])
    history = []
    recorded = []

    def snapshot(step, v, con, extra):
        row = {"step": step}
        for k in ("C", "B", "RP", "RE", "SD", "SF", "K", "E", "A"):
            row[k] = v[k].copy()
        row.update({k: np.asarray(x).copy() for k, x in con.items()})
        row.update(extra)
        if record_books:
            # The loan and deposit books themselves, for Layer 5's features.
            # Nothing here depends on the policy ([E] 2.2), so Layer 5 lifts
            # these straight out of a NoAction run instead of recomputing them.
            row["R_vec"] = (st.RP + st.RE).astype(np.float32)
            row["S_vec"] = (st.SD.sum(axis=1) + st.SF.sum(axis=1)).astype(np.float32)
            row["V_RP_shift"] = np.einsum("sn,sn->s", D_shift, st.RP)
            row["V_RE_shift"] = np.einsum("sn,sn->s", D_shift, st.RE)
            row["V_SD_shift"] = np.einsum("sn,san->s", D_shift, st.SD)
            row["V_SF_shift"] = np.einsum("sn,san->s", D_shift, st.SF)
        history.append(row)

    def restructure(step, st, y, D, matured_B, matured_K, last_con):
        """The policy acts, and its bonds join the books ([M] 3.3.2)."""
        ctx = Context(yields=y, discount=D, params=p, values=st.values(D),
                      last_constraints=last_con,
                      matured_B=matured_B, matured_K=matured_K)
        raw_B, raw_K = policy(step, st, ctx)
        aB = np.maximum(np.asarray(raw_B, dtype=float), 0.0)
        aK = np.maximum(np.asarray(raw_K, dtype=float), 0.0)
        ZB = par_bonds(D, p.invest_maturities, p.invest_spread)       # (S, bB, N)
        ZK = par_bonds(D, p.finance_maturities, p.finance_spread)
        return replace(st,
                       B=st.B + np.einsum("sj,sjn->sn", aB, ZB),      # [E] (18a)
                       K=st.K + np.einsum("sj,sjn->sn", aK, ZK),      # [E] (18b)
                       C=st.C - aB.sum(axis=1) + aK.sum(axis=1))      # [E] (19)

    # t = 0 has no roll-forward, but the bank does decide ([M] 3.3.1-3.3.2).
    # [M] 5 notes the model's "extensive initial borrowing", so this matters.
    zero = np.zeros(S)
    st = restructure(0, st, y0, disc[:, 0, :], zero, zero, last_con)
    v = st.values(disc[:, 0, :])
    st.E_last_annual = v["E"].copy()
    snapshot(0, v, last_con, {"CF": zero.copy(), "MR": zero.copy(),
                              "D1m_prev": np.ones(S), "matured_B": zero.copy(),
                              "matured_K": zero.copy()})

    for step in range(1, H + 1):
        y, D = yields[:, step, :], disc[:, step, :]

        # ---------------------------------------------------- roll forward
        y6m_avg = sum(st.y6m_history[-3:]) / 3.0
        uD, uF = deposit_rates(y6m_avg, yields[:, step - 1, 5], p)
        SD2, cash_D = roll_deposits(st.SD, uD, p.nmd_growth, p)
        SF2, cash_F = roll_deposits(st.SF, uF, p.term_growth, p)
        RP2, RE2, IRP2, IRE2, interest_in, repaid, granted = roll_loans(
            st.RP, st.RE, st.IRP, st.IRE, yields[:, step - 1, :], p, mortgage_mix)

        cost = monthly_cost(step - 1, x0, p)
        CF = repaid - granted + interest_in + (cash_D + cash_F) - cost   # [E] (13)

        MR = 0.025 * (np.einsum("sn,san->s", disc[:, step - 1, :], st.SD)
                      + np.einsum("sn,san->s", disc[:, step - 1, :], st.SF))
        cp = cash_penalty(st.C, MR, disc[:, step - 1, 0], p)

        matured_B, matured_K = st.B[:, 0].copy(), st.K[:, 0].copy()
        C_pre = st.C + CF + matured_B - matured_K - cp                # [E] (15)
        st = replace(st, C=C_pre, B=_shift_up(st.B), K=_shift_up(st.K),
                     RP=RP2, RE=RE2, IRP=IRP2, IRE=IRE2, SD=SD2, SF=SF2)
        st.y6m_history = (st.y6m_history + [y[:, 5]])[-3:]
        y6m_year_ago.append(y[:, 5])

        # -------------------------------------------- decide and restructure
        # No decision at the horizon: [M] 3.3.3 ends the model on a roll-forward.
        if step < H:
            st = restructure(step, st, y, D, matured_B, matured_K, last_con)

        # ------------------------------------------------------ annual step
        extra = {}
        if step % 12 == 0:
            # loan depreciation is an end-of-year event ([E] 2.2.2), every year
            rise = y[:, 5] - y6m_year_ago[-13]
            frac = np.maximum(rise - p.depreciation_trigger, 0.0)
            st = replace(st, RE=st.RE * (1.0 - frac)[:, None],
                         IRE=st.IRE * (1.0 - frac)[:, None])
            extra["depreciation_fraction"] = frac.copy()

        # ------------------------------------------- read the books, then pay
        # [E] Fig. 10's own note: "the equity value is recorded at the end of
        # the restructure step; but the annual step is performed after the
        # balance sheet restructuring."  So the constraints -- and the penalty
        # that Layer 4 builds from them -- are read before the dividend leaves.
        v = st.values(D)
        D_shift = disc_shift[:, step, :]
        last_con = cs.all_constraints(v, st.values(D_shift), y[:, 0])
        extra["cash_penalty"] = cp.copy()
        extra["loan_interest"] = interest_in.copy()
        extra["CF"] = CF.copy()
        extra["MR"] = MR.copy()
        extra["D1m_prev"] = disc[:, step - 1, 0].copy()
        extra["matured_B"] = matured_B.copy()
        extra["matured_K"] = matured_K.copy()

        if cs.is_annual_step(step, H):
            extra["EYR"] = cs.eyr(v["E"], st.E_last_annual)           # [E] (26)
            delta = p.dividend_share * np.maximum(v["E"] - st.E_last_annual, 0.0)
            st = replace(st, C=st.C - delta)                          # [E] (20)
            extra["dividend"] = delta.copy()
            st.E_last_annual = (v["E"] - delta).copy()

        snapshot(step, v, last_con, extra)
        if record_scenarios:
            recorded.append({k: st.__dict__[k][:record_scenarios].copy()
                             for k in ("C", "B", "K", "RP", "RE")})

    return {"history": history, "state": st, "mortgage_mix": mortgage_mix,
            "recorded": recorded, "params": p}


# ============================================================ reporting

def summarise(out: dict) -> dict:
    """Median path of each quantity, plus how often each limit is broken."""
    h = out["history"]
    keys = [k for k in h[0] if k != "step"]
    med = {k: [float(np.median(row[k])) for row in h if k in row] for k in keys}
    H = h[-1]["step"]
    viol = {}
    # [E] Alg. 1 checks the constraints at the H decision steps t = 0..H-1;
    # the terminal roll-forward is outside the loop and is not checked.
    for name in ("CMR", "LCR", "NSFR", "E_over_RWA", "IRS"):
        bad = np.zeros_like(h[0]["C"], dtype=bool)
        for row in h:
            if row["step"] < H:
                bad |= cs.violation(name, row[name]) > 0
        viol[name] = float(bad.mean())
    eyr_rows = [row["EYR"] for row in h if "EYR" in row]
    if eyr_rows:
        viol["EYR"] = float((np.minimum.reduce(eyr_rows) < 0).mean())
    E0, ET = h[0]["E"], h[-1]["E"]
    return {"median": med, "violation_share": viol,
            "E0": float(np.median(E0)),
            "ET": {"median": float(np.median(ET)),
                   "q05": float(np.quantile(ET, 0.05)),
                   "q95": float(np.quantile(ET, 0.95)),
                   "mean": float(ET.mean())},
            "ET_over_E0": float(np.median(ET / E0))}