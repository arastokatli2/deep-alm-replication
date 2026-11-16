"""The six constraints, as settled in decisions.md (B1-B5, E2, E9, CMR).

Rule used to settle: follow the journal paper [E]; where [E]'s printed
formula contradicts [E]'s own reported results, follow the results,
because the goal is to reproduce what the authors actually ran.

Inputs are economic values in CHF millions (see balance_sheet.values()).
"""
from __future__ import annotations

BOUNDS = {"CMR": 1.00, "LCR": 1.05, "NSFR": 1.05, "E_over_RWA": 0.17,
          "IRS": 0.085, "EYR": 0.0}
UPPER_BOUNDED = {"IRS"}             # E9: the only upper bound
IRS_SHIFT = 0.01                    # B4: +100 bp parallel shift of Y, upward only
EYR_BUFFER = 6.0                    # mCHF
EYR_DENOMINATOR = 6.0               # B3: [E] Table 4 values imply 6, not E_{t-1}


def rwa(v):                         # [E] (21b) = [M] (42b)
    return 0.10 * v["B"] + 0.35 * v["RP"] + 1.00 * v["RE"]


def lcr(v):                         # [E] (22b-c); [M] (43d) typo -> B2
    hqla = 0.71 * v["C"] + 0.89 * v["B"]
    no30 = 0.176 * v["SD"] + 0.130 * v["SF"] + 0.010 * v["K"]
    return hqla / no30


def nsfr(v):                        # [E] (23b-c); B1: 0.95 on SD, not [M]'s 0.6
    asf = 0.95 * v["SD"] + 0.90 * v["SF"] + 0.60 * v["K"] + 1.00 * v["E"]
    rsf = 0.12 * v["B"] + 0.71 * (v["RP"] + v["RE"])
    return asf / rsf


def cmr(v, y_1m):                   # [E] (24b): the 80 % relief applies to both deposit types
    import numpy as np
    relief = np.where(np.asarray(y_1m) >= 0.0, 0.8, 0.0)
    out = v["C"] / (0.025 * (v["SD"] + v["SF"]) * (1.0 - relief))
    return out if np.ndim(out) else float(out)


def irs(E, E_shifted):              # B4: signed; constraint on |IRS|
    return (E_shifted - E) / E


def eyr(E, E_prev):                 # B3
    return ((E - E_prev) - EYR_BUFFER) / EYR_DENOMINATOR


def violation(name, x):
    """E9: amount by which x is on the wrong side of its bound (>= 0).

    Works on a scalar or on an array of scenarios."""
    import numpy as np
    b = BOUNDS[name]
    gap = np.abs(x) - b if name in UPPER_BOUNDED else b - np.asarray(x)
    out = np.maximum(gap, 0.0)
    return out if np.ndim(out) else float(out)


def is_annual_step(i: int, horizon: int | None = None) -> bool:
    """E2: the dividend-and-EYR step.

    [E] Alg. 1 says (i mod 12) = 0, but [E] (26) restricts the EYR check to
    t in (T \\ {T}) and [M] 3.3.3 says the dividend that would fall due at the
    horizon "is not included in the model".  So both i = 0 and i = horizon are
    excluded: with H = 60 the annual steps are {12, 24, 36, 48}.
    """
    return i > 0 and i % 12 == 0 and (horizon is None or i < horizon)


def all_constraints(v, v_shifted, y_1m):
    """The five checked after every restructuring (EYR is annual, separate)."""
    return {"CMR": cmr(v, y_1m), "LCR": lcr(v), "NSFR": nsfr(v),
            "E_over_RWA": v["E"] / rwa(v), "IRS": irs(v["E"], v_shifted["E"])}
