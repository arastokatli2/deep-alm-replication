"""Layer 2: the bank's starting balance sheet x0, built from TKB's 2022 annual report.

Source: Thurgauer Kantonalbank, Geschaeftsbericht 2022 ("GB").  Page numbers
are the printed ones.  Units in this file: kCHF in the raw tables (as printed),
CHF millions everywhere else (the model's unit).

Model slots, following [E] Table 1 / [M] Table 4:
    C    cash
    B    investments           (cash-flow vector incl. coupons)
    RP   mortgages             (principal vector)
    RE   loans to enterprises  (principal vector)
    SD   non-maturing deposits (principal vector, split by reference maturity)
    SF   term deposits         (principal vector, split by reference maturity)
    K    borrowings            (cash-flow vector incl. coupons)
    E    equity = A - L, a residual (never an input)

Each vector has N = 180 monthly cells; cell i is a cash flow i months after
31.12.2022.  Value is V(X) = <D0, X> with D0 from the Layer 1 starting curve.

Every number below is either copied from the GB (page given) or produced by a
named assumption (A1, A2, ...) listed in ASSUMPTIONS.  Nothing else.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field, asdict

import numpy as np

from yieldcurve import svensson_spot

N = 180
VALUATION_DATE = _dt.date(2022, 12, 31)
MONTH = 365.25 / 12.0
REF_MATURITIES = (1, 2, 12, 120)          # months; [E] section 2.2.3: {1/12, 1/6, 1, 10} y

# ============================================================== raw data (kCHF)
# Maturity table, GB note 21 p.107 ("Faelligkeitsstruktur der Finanzinstrumente").
# Buckets: sight, callable, <=3m, 3-12m, 1-5y, >5y.  Rows sum to the balance
# sheet within 1 kCHF (the GB rounds each cell).
BUCKETS = ("sight", "callable", "m3", "m12", "y5", "over5")
BUCKET_MONTHS = {"sight": (1, 1), "callable": (1, 1), "m3": (1, 3),
                 "m12": (4, 12), "y5": (13, 60)}      # over5: (61, max), see A3

BALANCE_SHEET = {          # GB p.70, 31.12.2022, kCHF.  (side, German label)
    "cash":            ("A", "Fluessige Mittel",                         6_138_689),
    "due_from_banks":  ("A", "Forderungen gegenueber Banken",              228_249),
    "customer_loans":  ("A", "Forderungen gegenueber Kunden",            1_497_109),
    "mortgages":       ("A", "Hypothekarforderungen",                   23_122_881),
    "trading":         ("A", "Handelsgeschaeft",                             1_832),
    "pos_repl_values": ("A", "Positive Wiederbeschaffungswerte",           134_298),
    "fin_investments": ("A", "Finanzanlagen",                            1_977_089),
    "accrued_assets":  ("A", "Aktive Rechnungsabgrenzungen",                47_887),
    "participations":  ("A", "Beteiligungen",                               33_134),
    "fixed_assets":    ("A", "Sachanlagen",                                 95_716),
    "other_assets":    ("A", "Sonstige Aktiven",                             8_868),
    "due_to_banks":    ("L", "Verpflichtungen gegenueber Banken",        1_691_865),
    "repo":            ("L", "Verpflichtungen aus Wertpapierfinanzierung", 1_522_000),
    "customer_deposits": ("L", "Verpflichtungen aus Kundeneinlagen",    18_704_865),
    "neg_repl_values": ("L", "Negative Wiederbeschaffungswerte",            29_497),
    "cash_bonds":      ("L", "Kassenobligationen",                         239_527),
    "bonds_pfandbrief": ("L", "Anleihen und Pfandbriefdarlehen",         8_173_000),
    "accrued_liab":    ("L", "Passive Rechnungsabgrenzungen",              100_907),
    "other_liab":      ("L", "Sonstige Passiven",                          217_883),
    "provisions":      ("L", "Rueckstellungen",                             44_607),
    "reserves_gbr":    ("E", "Reserven fuer allgemeine Bankrisiken",     1_042_476),
    "share_capital":   ("E", "Gesellschaftskapital",                       400_000),
    "legal_reserve":   ("E", "Gesetzliche Gewinnreserve",                  968_550),
    "carried_forward": ("E", "Gewinnvortrag",                                2_726),
    "profit":          ("E", "Jahresgewinn",                               147_848),
}
TOTAL_ASSETS_KCHF = 33_285_751

MATURITY_TABLE = {         # GB p.107, kCHF: sight, callable, <=3m, 3-12m, 1-5y, >5y
    "cash":              (6_138_689, 0, 0, 0, 0, 0),
    "due_from_banks":    (58_142, 0, 50_000, 120_107, 0, 0),
    "customer_loans":    (2_973, 50_796, 692_084, 142_385, 354_463, 254_408),
    "mortgages":         (1_712, 27_228, 3_686_517, 1_869_009, 9_781_131, 7_757_283),
    "fin_investments":   (0, 0, 34_311, 74_804, 396_758, 1_471_217),
    "due_to_banks":      (116_695, 0, 1_329_309, 175_860, 65_000, 5_000),
    "repo":              (0, 0, 1_522_000, 0, 0, 0),
    "customer_deposits": (13_629_632, 1_922_665, 1_893_661, 761_674, 417_233, 80_000),
    "cash_bonds":        (0, 0, 58_090, 57_331, 119_194, 4_912),
    "bonds_pfandbrief":  (0, 0, 257_000, 397_000, 2_370_000, 5_149_000),
}

# GB note 13 p.100: individual bond issues (coupon %, maturity, kCHF).  Sum 2,560,000.
BOND_ISSUES = [
    (0.600, "2023-01-13", 50_000), (0.670, "2023-01-25", 100_000),
    (0.660, "2023-02-14", 50_000), (0.680, "2023-02-14", 50_000),
    (0.950, "2023-06-16", 30_000), (1.375, "2023-08-28", 150_000),
    (1.750, "2024-09-30", 250_000), (0.375, "2025-03-31", 60_000),
    (0.375, "2025-03-31", 140_000), (0.500, "2026-11-30", 200_000),
    (1.500, "2027-03-04", 200_000), (0.375, "2028-02-08", 250_000),
    (0.500, "2029-02-16", 120_000), (1.250, "2029-08-29", 200_000),
    (0.700, "2030-03-22", 210_000), (0.375, "2031-05-20", 250_000),
    (0.125, "2032-05-21", 250_000),
]
# GB note 13 p.100: Pfandbriefdarlehen by maturity year, kCHF; average rate 0.751 %.
PFANDBRIEF_BY_YEAR = {2023: 224_000, 2024: 291_000, 2025: 317_000, 2026: 395_000,
                      2027: 517_000, 2028: 473_000, 2029: 523_000, 2030: 747_000,
                      2031: 320_000, 2032: 491_000, 2033: 186_000, 2034: 318_000,
                      2035: 293_000, 2036: 141_000, 2037: 71_000, 2040: 204_000,
                      2042: 102_000}
PFANDBRIEF_RATE = 0.00751
# First maturity year only: kCHF due in months 1-3 and 4-12, so that the
# maturity table is matched exactly (A6).  None spreads that year evenly too.
PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF = (7_000, 217_000)

FIN_INVESTMENTS_FAIR_VALUE_KCHF = 1_751_966    # GB note 5 p.94 (book 1,977,089)
# SNB Svensson parameters for the fair value's own date, 30.12.2022 (cube rendopar).
FAIR_VALUE_CURVE = np.array([0.00890401, 0.000353016, 0.00377662,
                             0.0223517, 1.08667, 8.8053])
INTEREST_EXPENSE_CASH_BONDS_KCHF = 686          # GB note 25 p.108
PERSONNEL_COST_KCHF = 112_818                   # GB p.72, 2022
MATERIAL_COST_KCHF = 58_931                     # GB p.72, 2022

# Offenlegung 2022, table IRRBBA1 (mCHF; average repricing term in years).
# Volumes are IRRBB volumes and differ somewhat from the balance sheet; we use
# the TERMS as calibration targets and the VOLUMES only where stated (D1).
IRRBBA1 = {
    "fixed_mortgages":     (22_961, 3.90),
    "money_mkt_mortgages": (1_245, 0.01),     # SARON / Geldmarkthypotheken
    "variable_mortgages":  (30, 4.83),        # undetermined repricing
    "customer_loans":      (1_475, 1.35),
    "fin_investments":     (2_061, 8.34),
    "due_from_banks":      (175, 0.41),
    "due_to_banks":        (2_274, 0.20),
    "customer_term":       (4_046, 0.64),
    "cash_bonds":          (242, 1.59),
    "bonds_pfandbrief":    (8_568, 6.47),
    "nmd_sight":           (9_077, 2.31),     # Privat- und Kontokorrentkonti
    "nmd_savings":         (6_406, 1.69),     # Spargelder
    "irs_receive_leg":     (1_575, 2.21),
    "irs_pay_leg":         (1_525, 4.82),
    "total":               (61_878, 3.40),
}
NMD_TARGET_Y = ((IRRBBA1["nmd_sight"][0] * IRRBBA1["nmd_sight"][1] +
                 IRRBBA1["nmd_savings"][0] * IRRBBA1["nmd_savings"][1]) /
                (IRRBBA1["nmd_sight"][0] + IRRBBA1["nmd_savings"][0]))      # 2.06 y

# Observables the model state is checked against (Offenlegung 2022, kCHF).
TARGETS = {
    "RWA": 13_742_722, "CET1": 2_488_665, "E_over_RWA": 0.1811,
    "HQLA": 6_168_829, "NO30": 4_513_024, "LCR": 1.3669,
    "ASF": 25_481_039, "RSF": 18_667_681, "NSFR": 1.3650,
    "dEVE_up150": -227_269, "dEVE_down150": 256_256,
}

# ============================================================== assumptions
ASSUMPTIONS = {
    "A1": "Inside a maturity bucket, principal is spread evenly over its months "
          "(sight and callable -> month 1). The GB gives buckets only.",
    "A2": "Non-maturing deposits are spread over the [E] reference maturities 1m and "
          "10y in steady-state rolling shape (uniform over 1..m months); the 10y weight "
          "makes the average maturity equal TKB's disclosed NMD repricing term "
          "(IRRBBA1: sight 2.31y on 9.1bn, savings 1.69y on 6.4bn -> 2.06y). Max 10y "
          "matches TKB's disclosed modelling maximum.",
    "A3": "The '>5y' bucket runs from month 61 to a last month M. M is solved so the "
          "line's average maturity equals its IRRBBA1 repricing term, capped at 180 "
          "(the model grid). Where no M can reach the target (term too short for the "
          "bucket mass), M = 61, the closest possible, and the miss is reported. "
          "Lines without a target use 120 (10y).",
    "A4": "Term deposits get a reference maturity for rolling over: <=3m -> 2m, "
          "3-12m -> 1y, >1y -> 10y (nearest of [E]'s {1m, 2m, 1y, 10y}).",
    "A5": "Investments (Finanzanlagen, all bonds per GB note 5) pay one annual coupon "
          "rate c on book principal; c is solved so that V(B0) equals the disclosed "
          "fair value 1,751,966 kCHF.",
    "A6": "Bond issues: annual coupons on the maturity anniversary (Swiss market "
          "standard). Pfandbriefdarlehen: average rate 0.751 % for all, maturities "
          "spread evenly within the year; 2023 split 7 / 217 mCHF between <=3m and "
          "3-12m so that the maturity table is matched exactly.",
    "A7": "Cash flows after month 180 (Pfandbrief 2040/2042) are moved to month 180 "
          "at equal present value under the starting curve.",
    "A8": "Short money-market items (interbank, repo, interbank placements) are "
          "zero-coupon at the starting curve, so V = book at t0.",
    "A9": "Monthly costs = 2022 cost / 12 (personnel 112.8, material 58.9 mCHF/year).",
    "A10": "SARON mortgages (1,245 mCHF, IRRBBA1) sit in the <=3m bucket; with the "
           "sight + callable mortgages (28.9 mCHF ~ IRRBBA1 'variable' 30) they are "
           "loans that reprice monthly, so they go to RE (D1).",
}


@dataclass(frozen=True)
class Options:
    """The open mapping decisions, each with its default."""
    saron_mortgages_to: str = "RE"        # D1: SARON + variable mortgages -> "RE" or "RP"
    cash_bonds_to: str = "SF"             # D2: Kassenobligationen -> "SF" or "K"
    money_market_in_K: bool = True        # D3: interbank + repo funding in K (else excluded)
    calibrate_max_months: bool = True     # A3: solve M from IRRBBA1 (else use fallback)
    fallback_max_month: int = 120         # A3


@dataclass(frozen=True)
class InitialState:
    """x0 in CHF millions.  Arrays are read-only."""
    C: float
    B: np.ndarray
    RP: np.ndarray
    RE: np.ndarray
    SD: np.ndarray              # (4, N): one row per reference maturity
    SF: np.ndarray              # (4, N)
    K: np.ndarray
    monthly_personnel_cost: float
    monthly_material_cost: float
    investment_coupon: float
    nmd_weights: tuple
    options: Options
    max_months: dict = field(default_factory=dict)
    mapping: dict = field(default_factory=dict)

    @property
    def SD_total(self): return self.SD.sum(axis=0)

    @property
    def SF_total(self): return self.SF.sum(axis=0)


# ============================================================== helpers
def _month_of(date_str: str) -> int:
    d = _dt.date.fromisoformat(date_str)
    return max(1, int(np.ceil((d - VALUATION_DATE).days / MONTH)))


def _spread(buckets: dict, max_month: int, length: int = N) -> np.ndarray:
    """A1: bucket amounts -> monthly vector, even within each bucket."""
    v = np.zeros(length)
    for b, amount in buckets.items():
        if amount == 0:
            continue
        lo, hi = BUCKET_MONTHS.get(b, (61, max_month))
        v[lo - 1:hi] += amount / (hi - lo + 1)
    return v


def _row(line: str, keep=BUCKETS) -> dict:
    return {b: MATURITY_TABLE[line][i] / 1000.0 for i, b in enumerate(BUCKETS) if b in keep}


def _fold(long: np.ndarray, D: np.ndarray) -> np.ndarray:
    """A7: move cash flows beyond N to month N at equal present value."""
    out = long[:N].copy()
    out[N - 1] += np.dot(long[N:], D[N:len(long)]) / D[N - 1]
    return out


def _coupon_bond(maturity_month: int, principal: float, rate: float, length: int) -> np.ndarray:
    v = np.zeros(length)
    m = maturity_month
    while m >= 1:
        v[m - 1] += principal * rate
        m -= 12
    v[maturity_month - 1] += principal
    return v


def _zero_coupon(principal_vec: np.ndarray, D: np.ndarray) -> np.ndarray:
    """A8: grow principal at the starting curve so V = principal."""
    return principal_vec / D[:len(principal_vec)]


def discount_factors(y0: np.ndarray, shift: float = 0.0) -> np.ndarray:
    tau = np.arange(1, len(y0) + 1) / 12.0
    return np.exp(-tau * (y0 + shift))


def nmd_weights(target_years: float | None = None) -> dict:
    """A2: 1m / 10y mix whose average maturity is target_years
    (default: NMD_TARGET_Y, read when called so that other data can be used)."""
    if target_years is None:
        target_years = NMD_TARGET_Y
    mean = {m: (m + 2) / 3 / 12 for m in REF_MATURITIES}   # steady shape, see below
    w10 = (target_years - mean[1]) / (mean[120] - mean[1])
    return {1: 1 - w10, 2: 0.0, 12: 0.0, 120: w10}


def _steady_shape(m: int) -> np.ndarray:
    """A2: the profile a deposit tranche of reference maturity m settles at.

    [E] 2.2.3 puts a maturing amount back in equal parts over 1..m months. A
    portfolio fed that way is NOT flat: solving s = U s + s_1 * uniform(1..m)
    gives s_j proportional to (m - j + 1), a straight line down to zero, whose
    average maturity is (m + 2)/3 months. Starting the bank on the flat shape
    instead would make the deposit book drift for no economic reason, so the
    opening balance sheet uses the settled shape. The reinvestment itself stays
    flat, exactly as printed: see simulator.REINVEST.
    """
    v = np.zeros(N)
    v[:m] = np.arange(m, 0, -1)
    return v / v.sum()


def mean_maturity(vec: np.ndarray) -> float:
    """Principal-weighted average maturity in years (the IRRBBA1 measure)."""
    t = np.arange(1, len(vec) + 1) / 12.0
    return float((t * vec).sum() / vec.sum())


# Which balance-sheet buckets make up each calibrated line, and its IRRBBA1 target.
def line_buckets(opt: Options) -> dict:
    mort = _row("mortgages")
    saron = IRRBBA1["money_mkt_mortgages"][0] if opt.saron_mortgages_to == "RE" else 0.0
    fixed_mort = {**mort, "sight": 0.0, "callable": 0.0, "m3": mort["m3"] - saron}
    var_mort = {"sight": mort["sight"], "callable": mort["callable"], "m3": saron}
    if opt.saron_mortgages_to != "RE":
        fixed_mort, var_mort = mort, {}
    dep = _row("customer_deposits", ("m3", "m12", "y5", "over5"))
    return {
        "fixed_mortgages": (fixed_mort, IRRBBA1["fixed_mortgages"][1]),
        "saron_mortgages": (var_mort, None),
        "customer_loans":  (_row("customer_loans"), IRRBBA1["customer_loans"][1]),
        "fin_investments": (_row("fin_investments"), IRRBBA1["fin_investments"][1]),
        "customer_term":   (dep, IRRBBA1["customer_term"][1]),
        "cash_bonds":      (_row("cash_bonds"), IRRBBA1["cash_bonds"][1]),
        "due_to_banks":    (_row("due_to_banks"), IRRBBA1["due_to_banks"][1]),
        "repo":            (_row("repo"), None),
    }


def solve_max_month(buckets: dict, target: float | None, opt: Options) -> tuple[int, str]:
    """A3: last month of the >5y bucket so that the mean maturity hits target."""
    over = buckets.get("over5", 0.0)
    if over == 0:
        return opt.fallback_max_month, "no >5y mass"
    if target is None or not opt.calibrate_max_months:
        return opt.fallback_max_month, "fallback"
    total = sum(buckets.values())
    rest = sum(a * (BUCKET_MONTHS[b][0] + BUCKET_MONTHS[b][1]) / 24.0
               for b, a in buckets.items() if b != "over5")
    need = (target * total - rest) / over               # required mean of >5y part, years
    M = int(round(24.0 * need - 61))
    if M < 61:          # target unreachable: take the end that comes closest
        return 61, f"closest possible (target needs >5y mean {need:.2f}y)"
    if M > N:
        return N, f"capped at {N} (target needs {M})"
    return M, "calibrated"


def max_months(opt: Options) -> dict:
    return {k: solve_max_month(bk, tgt, opt) for k, (bk, tgt) in line_buckets(opt).items()}


# ============================================================== builders
def build_K(y0: np.ndarray, opt: Options) -> tuple[np.ndarray, dict]:
    D = discount_factors(y0)
    L = len(y0)
    lb, mm = line_buckets(opt), max_months(opt)
    parts = {}
    v = np.zeros(L)
    for c, date, amt in BOND_ISSUES:                                    # A6
        v += _coupon_bond(_month_of(date), amt / 1000.0, c / 100.0, L)
    parts["bond_issues"] = v

    v = np.zeros(L)
    year0 = VALUATION_DATE.year + 1                       # first maturity year
    for year, amt in PFANDBRIEF_BY_YEAR.items():                        # A6
        first = 12 * (year - year0) + 1
        if year == year0 and PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF is not None:
            q, r = (x / 1000.0 for x in PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF)
            split = {**{m: q / 3 for m in (1, 2, 3)}, **{m: r / 9 for m in range(4, 13)}}
        else:
            split = {m: amt / 1000.0 / 12 for m in range(first, first + 12)}
        for m, p in split.items():
            v += _coupon_bond(m, p, PFANDBRIEF_RATE, L)
    parts["pfandbrief"] = v

    if opt.money_market_in_K:                                           # A8
        for line in ("due_to_banks", "repo"):
            pv = _spread(lb[line][0], mm[line][0])
            parts[line] = np.concatenate([_zero_coupon(pv, D), np.zeros(L - N)])
    if opt.cash_bonds_to == "K":
        rate = INTEREST_EXPENSE_CASH_BONDS_KCHF / BALANCE_SHEET["cash_bonds"][2]  # 0.29 %
        pv = _spread(lb["cash_bonds"][0], mm["cash_bonds"][0])
        v = np.zeros(L)
        for m in np.nonzero(pv)[0]:
            v += _coupon_bond(m + 1, pv[m], rate, L)
        parts["cash_bonds"] = v
    folded = {k: _fold(p, D) for k, p in parts.items()}
    return sum(folded.values()), folded


def build_B(y0: np.ndarray, opt: Options) -> tuple[np.ndarray, float, dict]:
    D = discount_factors(y0)[:N]
    lb, mm = line_buckets(opt), max_months(opt)
    principal = _spread(lb["fin_investments"][0], mm["fin_investments"][0])
    # coupon dates: each monthly principal tranche pays annually back from maturity
    coupon_unit = np.zeros(N)
    for m in np.nonzero(principal)[0]:
        coupon_unit += _coupon_bond(m + 1, principal[m], 1.0, N) - \
            _coupon_bond(m + 1, principal[m], 0.0, N)
    # A5: the fair value is a 31.12.2022 observation, so the coupon is solved on
    # THAT day's curve, whatever curve we then value the portfolio on.  Solving it
    # on a lower curve (e.g. 15.07.2022) would force a negative coupon.
    target = FIN_INVESTMENTS_FAIR_VALUE_KCHF / 1000.0
    D_fv = discount_factors(svensson_spot(FAIR_VALUE_CURVE, np.arange(1, N + 1) / 12.0))
    c = (target - D_fv @ principal) / (D_fv @ coupon_unit)              # linear in c
    fin = principal + c * coupon_unit
    placements = _zero_coupon(_spread(_row("due_from_banks", ("m3", "m12")), 0), D)  # A8
    return fin + placements, float(c), {"fin_investments": fin, "bank_placements": placements,
                                        "fin_principal": principal}


def build_loans(opt: Options) -> tuple[np.ndarray, np.ndarray]:
    lb, mm = line_buckets(opt), max_months(opt)
    fixed = _spread(lb["fixed_mortgages"][0], mm["fixed_mortgages"][0])
    saron = _spread(lb["saron_mortgages"][0], opt.fallback_max_month)
    cust = _spread(lb["customer_loans"][0], mm["customer_loans"][0])
    return fixed, cust + saron                                          # D1 inside line_buckets


def build_deposits(opt: Options) -> tuple[np.ndarray, np.ndarray, dict]:
    lb, mm = line_buckets(opt), max_months(opt)
    nmd = (MATURITY_TABLE["customer_deposits"][0] + MATURITY_TABLE["customer_deposits"][1]) / 1000.0
    w = nmd_weights()
    SD = np.vstack([w[m] * nmd * _steady_shape(m) for m in REF_MATURITIES])
    SF = np.zeros((4, N))
    ref_of = {"m3": 2, "m12": 12, "y5": 120, "over5": 120}                # A4
    lines = ["customer_term"] + (["cash_bonds"] if opt.cash_bonds_to == "SF" else [])
    for line in lines:
        bk, M = lb[line][0], mm[line][0]
        for b, ref in ref_of.items():
            if b in bk:
                SF[REF_MATURITIES.index(ref)] += _spread({b: bk[b]}, M)
    return SD, SF, w


def mapping_table(opt: Options) -> dict:
    """Every balance-sheet line -> slot(s) or 'excluded', with the reason."""
    mm = "K" if opt.money_market_in_K else "excluded"
    return {
        "cash": ("C", "cash, SNB giro"),
        "due_from_banks": ("C + B", "sight part to cash, term placements to B (A8)"),
        "customer_loans": ("RE", "loans to companies and individuals"),
        "mortgages": ("RP + RE" if opt.saron_mortgages_to == "RE" else "RP",
                      "D1: SARON (1,245) + variable (28.9) mortgages reprice monthly -> RE; "
                      "fixed-rate mortgages -> RP"),
        "fin_investments": ("B", "all bonds, held to maturity (GB note 5)"),
        "customer_deposits": ("SD + SF", "sight + callable -> SD; fixed-term buckets -> SF"),
        "cash_bonds": (opt.cash_bonds_to, "D2: customer savings certificates"),
        "bonds_pfandbrief": ("K", "own bonds and Pfandbrief loans, exact from note 13"),
        "due_to_banks": (mm, "D3: short interbank funding as 3m-type borrowing"),
        "repo": (mm, "D3: repo funding, all <=3m"),
        "trading": ("excluded", "trading book, not banking book"),
        "pos_repl_values": ("excluded", "derivatives (IRS 1.51 bn notional); replication is bond-only"),
        "neg_repl_values": ("excluded", "derivatives; replication is bond-only"),
        "accrued_assets": ("excluded", "accounting accrual, no economic cash-flow slot"),
        "accrued_liab": ("excluded", "accounting accrual, no economic cash-flow slot"),
        "participations": ("excluded", "equity stakes, no slot"),
        "fixed_assets": ("excluded", "buildings/IT, no slot"),
        "other_assets": ("excluded", "no slot"),
        "other_liab": ("excluded", "no slot"),
        "provisions": ("excluded", "no slot"),
        "reserves_gbr": ("E", "equity is a residual; used for validation only"),
        "share_capital": ("E", "residual"), "legal_reserve": ("E", "residual"),
        "carried_forward": ("E", "residual"), "profit": ("E", "residual"),
    }


def build_initial_state(y0: np.ndarray, opt: Options = Options()) -> InitialState:
    if len(y0) < 240:
        raise ValueError("need the 240-tenor starting curve from fit_sqrt.npz")
    C = (BALANCE_SHEET["cash"][2] + MATURITY_TABLE["due_from_banks"][0]) / 1000.0
    B, c, _ = build_B(y0, opt)
    RP, RE = build_loans(opt)
    SD, SF, w = build_deposits(opt)
    K, _ = build_K(y0, opt)
    arrays = dict(B=B, RP=RP, RE=RE, SD=SD, SF=SF, K=K)
    for a in arrays.values():
        a.setflags(write=False)
    return InitialState(C=C, **arrays,
                        monthly_personnel_cost=PERSONNEL_COST_KCHF / 1000.0 / 12,
                        monthly_material_cost=MATERIAL_COST_KCHF / 1000.0 / 12,
                        investment_coupon=c, nmd_weights=tuple(w[m] for m in REF_MATURITIES),
                        options=opt, max_months=max_months(opt), mapping=mapping_table(opt))


# ============================================================== valuation & checks
def values(x: InitialState, D: np.ndarray) -> dict:
    D = D[:N]
    v = dict(C=x.C, B=D @ x.B, RP=D @ x.RP, RE=D @ x.RE,
             SD=D @ x.SD_total, SF=D @ x.SF_total, K=D @ x.K)
    v["A"] = v["C"] + v["B"] + v["RP"] + v["RE"]
    v["L"] = v["SD"] + v["SF"] + v["K"]
    v["E"] = v["A"] - v["L"]
    return v


def constraints(v: dict, y_1m: float) -> dict:
    """[E] eqs. (21)-(24) at t0, via the settled formulas in constraints.py."""
    import constraints as cs
    return dict(RWA=cs.rwa(v), E_over_RWA=v["E"] / cs.rwa(v), LCR=cs.lcr(v),
                NSFR=cs.nsfr(v), CMR=cs.cmr(v, y_1m))


def duration(vec: np.ndarray, D: np.ndarray) -> float:
    t = np.arange(1, N + 1) / 12.0
    pv = vec * D[:N]
    return float((t * pv).sum() / pv.sum())


def book_by_slot(opt: Options) -> dict:
    """Book (accounting) amounts that went into each slot, mCHF, and the excluded rest."""
    bs = {k: v[2] / 1000.0 for k, v in BALANCE_SHEET.items()}
    mt = {k: np.array(v) / 1000.0 for k, v in MATURITY_TABLE.items()}
    saron = (IRRBBA1["money_mkt_mortgages"][0] + mt["mortgages"][:2].sum()) \
        if opt.saron_mortgages_to == "RE" else 0.0
    out = dict(C=bs["cash"] + mt["due_from_banks"][0],
               B=bs["fin_investments"] + mt["due_from_banks"][2:].sum(),
               RP=bs["mortgages"] - saron, RE=bs["customer_loans"] + saron,
               SD=mt["customer_deposits"][:2].sum(),
               SF=mt["customer_deposits"][2:].sum() + (bs["cash_bonds"] if opt.cash_bonds_to == "SF" else 0),
               K=bs["bonds_pfandbrief"] + (bs["cash_bonds"] if opt.cash_bonds_to == "K" else 0)
                 + ((bs["due_to_banks"] + bs["repo"]) if opt.money_market_in_K else 0))
    m = mapping_table(opt)
    out["excluded_assets"] = sum(bs[k] for k, (sl, _) in m.items()
                                 if sl == "excluded" and BALANCE_SHEET[k][0] == "A")
    out["excluded_liabilities"] = sum(bs[k] for k, (sl, _) in m.items()
                                      if sl == "excluded" and BALANCE_SHEET[k][0] == "L")
    out["equity"] = sum(bs[k] for k in bs if BALANCE_SHEET[k][0] == "E")
    return out


def report(x: InitialState, y0: np.ndarray) -> dict:
    D = discount_factors(y0)
    v = values(x, D)
    k = constraints(v, float(y0[0]))
    shocks = {bp: values(x, discount_factors(y0, bp / 1e4))["E"] - v["E"]
              for bp in (-150, -100, 100, 150)}
    book = book_by_slot(x.options)
    t = {kk: (vv / 1000.0 if abs(vv) > 100 else vv) for kk, vv in TARGETS.items()}
    slots = ("C", "B", "RP", "RE", "SD", "SF", "K")
    # equity bridge: book equity -> model E0
    bridge = {"book equity": book["equity"],
              "- excluded assets": -book["excluded_assets"],
              "+ excluded liabilities": book["excluded_liabilities"]}
    for s_ in slots:
        sign = 1 if s_ in ("C", "B", "RP", "RE") else -1
        bridge[f"{'+' if sign > 0 else '-'} (V - book) {s_}"] = sign * (v[s_] - book[s_])
    bridge["= model E0"] = sum(bridge.values())
    # mean maturities vs IRRBBA1
    lb = line_buckets(x.options)
    mm = x.max_months
    vec = {"fixed_mortgages": x.RP,
           "customer_loans": _spread(lb["customer_loans"][0], mm["customer_loans"][0]),
           "fin_investments": _spread(lb["fin_investments"][0], mm["fin_investments"][0]),
           "customer_term": _spread(lb["customer_term"][0], mm["customer_term"][0]),
           "cash_bonds": _spread(lb["cash_bonds"][0], mm["cash_bonds"][0]),
           "due_to_banks": _spread(lb["due_to_banks"][0], mm["due_to_banks"][0])}
    K_principal = _bond_principal_vector()
    repricing = {kk: dict(model=mean_maturity(vv), irrbba1=IRRBBA1[kk][1], last_month=mm[kk][0],
                          how=mm[kk][1]) for kk, vv in vec.items()}
    repricing["bonds_pfandbrief"] = dict(model=mean_maturity(K_principal),
                                         irrbba1=IRRBBA1["bonds_pfandbrief"][1],
                                         last_month=None, how="exact data (note 13)")
    repricing["nmd"] = dict(model=mean_maturity(x.SD_total), irrbba1=round(NMD_TARGET_Y, 4),
                            last_month=120, how="A2 1m/10y mix")
    return {
        "options": asdict(x.options),
        "nominal_cash_flows_mCHF": {s_: float(np.sum(getattr(x, s_ if s_ not in ("SD", "SF") else s_)))
                                    for s_ in slots},
        "book_mCHF": {s_: book[s_] for s_ in slots},
        "values_mCHF": v,
        "shares_of_assets": {s_: v[s_] / v["A"] for s_ in slots + ("E",)},
        "equity_bridge_mCHF": bridge,
        "book_equity_mCHF": book["equity"], "CET1_mCHF": t["CET1"],
        "constraints_t0": k,
        "targets": {kk: t[kk] for kk in ("RWA", "E_over_RWA", "HQLA", "NO30", "LCR",
                                           "ASF", "RSF", "NSFR")},
        "IRS_100bp": shocks[100] / v["E"],          # B4: signed, +100 bp only
        "dE_mCHF": shocks,
        "dEVE_tkb_mCHF": {150: t["dEVE_up150"], -150: t["dEVE_down150"]},
        "duration_y": dict(loans=duration(x.RP + x.RE, D), deposits=duration(x.SD_total + x.SF_total, D),
                           B=duration(x.B, D), K=duration(x.K, D)),
        "repricing_vs_irrbba1": repricing,
        "investment_coupon": x.investment_coupon,
        "nmd_weights": dict(zip(map(str, REF_MATURITIES), x.nmd_weights)),
        "monthly_cost_mCHF": dict(personnel=x.monthly_personnel_cost, material=x.monthly_material_cost),
    }


def _bond_principal_vector() -> np.ndarray:
    v = np.zeros(300)
    for _, date, amt in BOND_ISSUES:
        v[_month_of(date) - 1] += amt / 1000.0
    year0 = VALUATION_DATE.year + 1
    for year, amt in PFANDBRIEF_BY_YEAR.items():
        first = 12 * (year - year0)
        if year == year0 and PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF is not None:
            q, r = (x / 1000.0 for x in PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF)
            v[0:3] += q / 3
            v[3:12] += r / 9
        else:
            v[first:first + 12] += amt / 1000.0 / 12
    return v


def save(x: InitialState, path: str) -> None:
    np.savez(path, C=x.C, B=x.B, RP=x.RP, RE=x.RE, SD=x.SD, SF=x.SF, K=x.K,
             ref_maturities=np.array(REF_MATURITIES),
             monthly_personnel_cost=x.monthly_personnel_cost,
             monthly_material_cost=x.monthly_material_cost,
             valuation_date=np.array(VALUATION_DATE.isoformat()),
             options=np.array(json.dumps(asdict(x.options))),
             mapping=np.array(json.dumps(x.mapping)),
             max_months=np.array(json.dumps(x.max_months)),
             assumptions=np.array(json.dumps(ASSUMPTIONS)))
