"""Layer 4 - the objective: penalties, target loss, utility, and the metrics.

[E] 2.5 / [M] 3.5.  Three pieces, all per scenario:

    P_i,t  = (1 + violation)^2 - 1          each breach               [E] (30)
    p      = sum_i sigma_i sum_t P_i,t      accumulated               [E] (31b)
    l_p    = (1 + p)^2 - 1                  penalty loss              [E] (31a)
    l_t    = [ (E_T - (1+mu)^T E_0)^- ]^2   target loss, asymmetric   [E] (47)
    l      = l_t + lambda * l_p             total loss                [E] (32)

The CRRA utility loss [E] (28) is not trained on -- [E] 3.2.3 says every model
was trained on the target loss -- but it is reported, so it is here too.

Which side of a bound counts as a breach follows constraints.py (E9), not
[E] (30) as printed; see RESOLVED below.

Inputs are economic values in CHF millions, as everywhere else.
"""
from __future__ import annotations

import numpy as np

import constraints as cs

# ---------------------------------------------------------------- [E] Table 2
# Table 2 names the weights in the order (LCR, NSFR, CMR, E/RWA, IRS, EYR),
# which is NOT the order of [E] 2.4 that eq. (30) and (31b) index by
# (E/RWA, LCR, NSFR, CMR, IRS, EYR).  Reading Table 2's list positionally
# against 2.4 would put 1.0 on the leverage ratio instead of 2.5.  Keyed here
# so the order cannot be got wrong.
WEIGHTS = {"LCR": 1.0, "NSFR": 0.2, "CMR": 1.0,
           "E_over_RWA": 2.5, "IRS": 2.0, "EYR": 0.002}

MU_EVAL = 0.0406          # [E] Table 3 footnote, for the four 5y models
MU_RANGE = (0.02, 0.07)   # [E] Table 2, sampled per path during training
LAMBDA_EVAL = 3.5         # [E] Table 2
LAMBDA_RANGE = (0.05, 25.0)
GAMMA_EVAL = 10.0         # [E] Table 3, evaluation only
EPS_OVER_E0 = 1e-6        # [E] (28): 0 < eps/E_0 << 1

# B9, see RESOLVED: the target loss is measured in units of E_0 / TARGET_SCALE.
TARGET_SCALE = 10.0

RESOLVED = {
    "E9": "[E] (30) prints the two cases the wrong way round: it uses "
          "(x - beta)+ for i in {1,2,3,4,6} (E/RWA, LCR, NSFR, CMR, EYR), "
          "which are lower bounds, and (beta - x)+ for i = 5 (IRS), the only "
          "upper bound.  [E] Table 4 settles it: the CMR values that count as "
          "violations are 0.92-0.94 against a bound of 1.00, the EYR ones are "
          "-2.99 and -23.8 against 0, and the IRS ones are 0.09 against 0.085. "
          "So a breach is beta - x for the five and |x| - beta for IRS. "
          "[M] (50) prints the same error.  We follow Table 4.",
    "B9": "Neither paper states the scale of the target loss, but [E] Table 3 "
          "pins it down.  l_t equals 100 x E[(E_T/E_0 - (1+mu)^T)^2] for all "
          "four 5y models to within 2%: e.g. BM_C has mean equity ratio 1.124 "
          "and sd 0.026, giving 100 x (0.0963^2 + 0.026^2) = 0.995 against a "
          "reported 0.997.  So equity is measured in units of E_0/10, i.e. "
          "TARGET_SCALE = 10 on the ratio.  Without it lambda = 3.5 would "
          "weigh the penalties 100 times more heavily than the authors did.",
    "B10": "[E] Table 3's l_t row is the SYMMETRIC target loss (29), not the "
           "asymmetric (47) the text says every model was trained on: l_t "
           "there equals 100 x (bias^2 + variance) exactly, and the total loss "
           "l is smaller than l_t alone in every column, which is impossible "
           "for l = l_t + lambda l_p with lambda, l_p >= 0.  The gap between "
           "the two closes monotonically as the skew of the equity ratio falls "
           "(MM 1.68 -> 18% gap, BM_E -0.42 -> 8%), which is exactly how the "
           "symmetric and asymmetric losses differ.  We train and report the "
           "asymmetric one and give the symmetric one for comparability.",
    "B11": "[E] Alg. 1 accumulates the penalty right after Restructure and "
           "before AnnualStep, for i = 0..H-1, so the terminal roll-forward "
           "contributes no penalty and the starting balance sheet does.  "
           "Layer 3 now records each row before the dividend leaves, which is "
           "what [E] Fig. 10's note describes.",
}


# ------------------------------------------------------------------ penalties
def breach(name: str, x) -> np.ndarray:
    """[E] (30): (1 + violation)^2 - 1.  Zero when the bound holds."""
    v = cs.violation(name, x)
    return (1.0 + v) ** 2 - 1.0


def accumulated_penalty(history, horizon: int | None = None,
                        weights: dict | None = None) -> np.ndarray:
    """[E] (31b): p = sum_i sigma_i sum_t P_i,t, one number per scenario.

    history : the rows from simulator.simulate()["history"].
    horizon : H.  Rows at step >= H are the terminal roll-forward and carry no
              penalty ([E] Alg. 1 loops i = 0..H-1).  Defaults to the last step.
    """
    w = WEIGHTS if weights is None else weights
    H = history[-1]["step"] if horizon is None else horizon
    p = np.zeros_like(np.asarray(history[0]["E"], dtype=float))
    for row in history:
        if row["step"] >= H:
            continue
        for name in ("LCR", "NSFR", "CMR", "E_over_RWA", "IRS"):
            p = p + w[name] * breach(name, row[name])
        if "EYR" in row:                       # annual steps only, [E] (26)
            p = p + w["EYR"] * breach("EYR", row["EYR"])
    return p


def penalty_loss(p) -> np.ndarray:
    """[E] (31a): l_p = (1 + p)^2 - 1."""
    p = np.asarray(p, dtype=float)
    return (1.0 + p) ** 2 - 1.0


# --------------------------------------------------------------- the two goals
def target_loss(ET, E0, mu: float = MU_EVAL, years: float = 5.0, *,
                asymmetric: bool = True, scale: float = TARGET_SCALE):
    """[E] (47) asymmetric, or (29) symmetric.  See B9 for `scale`."""
    gap = scale * (np.asarray(ET, dtype=float) / np.asarray(E0, dtype=float)
                   - (1.0 + mu) ** years)
    if asymmetric:
        gap = np.minimum(gap, 0.0)             # only shortfalls are penalised
    return gap ** 2


def utility(x, gamma: float = GAMMA_EVAL):
    """[E] (27): CRRA."""
    x = np.asarray(x, dtype=float)
    if gamma == 1.0:
        return np.log(x)
    return (x ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def utility_loss(ET, E0, gamma: float = GAMMA_EVAL,
                 eps_over_E0: float = EPS_OVER_E0):
    """[E] (28): minus the utility of the equity ratio, floored at eps/E_0."""
    ratio = np.maximum(np.asarray(ET, dtype=float), 0.0) \
        / np.asarray(E0, dtype=float) + eps_over_E0
    return -utility(ratio, gamma)


def total_loss(ET, p, E0, *, mu: float = MU_EVAL, lam: float = LAMBDA_EVAL,
               years: float = 5.0, asymmetric: bool = True):
    """[E] (32) with the target loss in place of the utility loss ([E] 3.2.3)."""
    return target_loss(ET, E0, mu, years, asymmetric=asymmetric) \
        + lam * penalty_loss(p)


# ------------------------------------------------------------------ risk stats
def _var(x, q: float) -> float:
    """[E] (50d)/(51f): the q-quantile measured from the mean."""
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, q) - x.mean())


def _es(x, q: float, upper: bool) -> float:
    """[E] (50e)/(51g): the mean of the tail beyond that quantile, from the mean."""
    x = np.asarray(x, dtype=float)
    cut = np.quantile(x, q)
    tail = x[x >= cut] if upper else x[x <= cut]
    return float(tail.mean() - x.mean())


# -------------------------------------------------------------------- Table 3
def evaluate(out: dict, *, mu: float = MU_EVAL, lam: float = LAMBDA_EVAL,
             gamma: float = GAMMA_EVAL, alpha: float = 0.95) -> dict:
    """Every number in [E] Table 3, for one run of simulator.simulate()."""
    h = out["history"]
    H = h[-1]["step"]
    years = H / 12.0
    E0 = np.asarray(h[0]["E"], dtype=float)
    ET = np.asarray(h[-1]["E"], dtype=float)

    p = accumulated_penalty(h, H)
    lp = penalty_loss(p)
    lt = target_loss(ET, E0, mu, years)
    lt_sym = target_loss(ET, E0, mu, years, asymmetric=False)
    lu = utility_loss(ET, E0, gamma)
    tot = lt + lam * lp

    er = ET / E0
    m = er.mean()
    cen = er - m
    m2, m3, m4 = (cen ** 2).mean(), (cen ** 3).mean(), (cen ** 4).mean()

    dividends = sum(np.asarray(row["dividend"], dtype=float)
                    for row in h if "dividend" in row)
    if np.ndim(dividends) == 0:
        dividends = np.zeros_like(E0)

    return {
        "loss": {
            "total": float(tot.mean()),
            "target": float(lt.mean()),
            "target_symmetric": float(lt_sym.mean()),
            "penalty": float(lp.mean()),
            f"penalty_VaR{alpha}": _var(lp, alpha),
            f"penalty_ES{alpha}": _es(lp, alpha, upper=True),
            "utility": float(lu.mean()),
            "mu": mu, "lambda": lam, "gamma": gamma,
        },
        "equity_ratio": {
            "mean": float(m),
            "sd": float(np.sqrt(m2)),
            "skew": float(m3 / m2 ** 1.5),
            "excess_kurtosis": float(m4 / m2 ** 2 - 3.0),
            f"VaR{alpha}": _var(er, 1.0 - alpha),
            f"ES{alpha}": _es(er, 1.0 - alpha, upper=False),
        },
        "annualised": {
            "mu_hat": float((er ** (1.0 / years) - 1.0).mean()),
            "dividend_yield": float((dividends / E0).mean() / (years - 1.0)),
        },
        "penalty_share": {
            name: float((w * sum(breach(name, row[name]) for row in h
                                 if row["step"] < H and name in row)).mean())
            for name, w in WEIGHTS.items()
        },
    }
