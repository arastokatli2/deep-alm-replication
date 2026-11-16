"""
Yield curve layer for the Deep ALM replication.

Implements the HJM-PCA generator of Englisch et al. (2023), Section 2.7,
on Swiss National Bank Svensson parameters (data cube `rendopar`).

Modules, in build order:
  1. load_rendopar      - parse the SNB CSV
  2. svensson_spot      - six parameters -> spot yield curve
  3. svensson_forward   - six parameters -> instantaneous forward curve
     svensson_interval_forward - interval-average forwards (what the simulator uses)
  4. converters         - yields <-> forwards <-> discount factors
  5. fit_hjm_pca        - historical fit of the volatility matrix V
  6. simulate           - forward curve paths, optional term premium
  7. fit_curve_features - the 3-number curve compression for the network
  8. load_scenarios     - rebuild scenarios exactly from a saved fit + seed

Conventions
-----------
Rates are decimals (0.015 = 1.5%), continuously compounded.
Tenors are in years. The model grid is monthly: tau_k = k/12, k = 1..N.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PARAMS = ("b0", "b1", "b2", "b3", "t1", "t2")


# --------------------------------------------------------------------------
# 1. Loader
# --------------------------------------------------------------------------

def load_rendopar(path: str) -> pd.DataFrame:
    """Parse the SNB `rendopar` CSV into a wide DataFrame of Svensson parameters.

    The file is the SNB data portal's long format: two metadata lines, a
    header row `Date;d0;Value`, then one row per (date, parameter).
    A date carries either all six parameters or none; blanks are Swiss
    public holidays on which no fixing was published.

    Returns a frame indexed by date with columns b0, b1, b2, b3, t1, t2,
    rates converted from percent to decimal, holiday rows dropped.
    """
    records: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        in_body = False
        for row in reader:
            if not row:
                continue
            if row[0] == "Date":
                in_body = True
                continue
            if not in_body or len(row) < 3:
                continue
            date, param, value = row[0], row[1], row[2].strip()
            if value == "":
                continue
            records.setdefault(date, {})[param] = float(value)

    frame = pd.DataFrame.from_dict(records, orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.dropna(subset=list(PARAMS))[list(PARAMS)]

    # SNB publishes beta coefficients in percent; tau1/tau2 are already years.
    frame[["b0", "b1", "b2", "b3"]] /= 100.0
    return frame


# --------------------------------------------------------------------------
# 2-3. Svensson evaluation
# --------------------------------------------------------------------------

def svensson_spot(params: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Continuously-compounded spot yields from Svensson parameters.

    This is Englisch et al. (2023) eq. (37) / Mueller (2023) eq. (16):

        R(m) = b0
             + b1 * (1 - exp(-m/t1)) / (m/t1)
             + b2 * [(1 - exp(-m/t1)) / (m/t1) - exp(-m/t1)]
             + b3 * [(1 - exp(-m/t2)) / (m/t2) - exp(-m/t2)]

    params : (..., 6) array ordered b0, b1, b2, b3, t1, t2
    tenors : (M,) array of maturities in years, all > 0
    returns: (..., M) array of yields as decimals
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    m = np.asarray(tenors, dtype=float)[None, :]
    b0, b1, b2, b3, t1, t2 = (params[:, i : i + 1] for i in range(6))

    x1, x2 = m / t1, m / t2
    e1, e2 = np.exp(-x1), np.exp(-x2)
    g1 = (1.0 - e1) / x1
    g2 = (1.0 - e2) / x2

    out = b0 + b1 * g1 + b2 * (g1 - e1) + b3 * (g2 - e2)
    return out.squeeze()


def svensson_forward(params: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Instantaneous forward rates from Svensson parameters, analytically.

    The Svensson spot formula is by construction the running average of

        f(m) = b0 + b1*exp(-m/t1) + b2*(m/t1)*exp(-m/t1) + b3*(m/t2)*exp(-m/t2)

    i.e. R(m) = (1/m) * integral_0^m f(u) du.

    This is the INSTANTANEOUS forward at each tenor.  It is the right object
    for theory and for plotting, but NOT for the simulator's grid: see
    `svensson_interval_forward`, which the simulator uses.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    m = np.asarray(tenors, dtype=float)[None, :]
    b0, b1, b2, b3, t1, t2 = (params[:, i : i + 1] for i in range(6))

    x1, x2 = m / t1, m / t2
    out = b0 + b1 * np.exp(-x1) + b2 * x1 * np.exp(-x1) + b3 * x2 * np.exp(-x2)
    return out.squeeze()


def svensson_interval_forward(params: np.ndarray, n_tenors: int,
                              dtau: float) -> np.ndarray:
    """Average forward rate over each grid interval ((k-1)dtau, k dtau].

    THIS is the forward curve the simulator uses.  Eq. (42) of Englisch et
    al. sums forwards times dtau to get discount factors, so each entry must
    be the AVERAGE forward over its interval for the sum to equal the
    integral.  Because the Svensson spot curve is a closed-form formula,

        f_k = (tau_k R(tau_k) - tau_{k-1} R(tau_{k-1})) / dtau

    is the exact interval average: no noise is amplified by differencing.

    Using the instantaneous forward at the grid point instead (as an earlier
    version did) makes eq. (42) a right-Riemann sum and leaves the day-0
    simulator curve up to 2.2 bp above the SNB curve at 1 month.

    params : (..., 6); returns (..., n_tenors) on tenors dtau, 2dtau, ...
    """
    tenors = np.arange(1, n_tenors + 1) * dtau
    return yields_to_forwards(svensson_spot(params, tenors), dtau)


# --------------------------------------------------------------------------
# 4. Converters
# --------------------------------------------------------------------------

def forwards_to_yields(fwd: np.ndarray, dtau: float) -> np.ndarray:
    """Yields from forward rates: R_k = (1/tau_k) * sum_{j<=k} f_j * dtau.

    de Jonge & Govaert (2025) eq. (11).  A yield is the running average of
    the forward curve.
    """
    fwd = np.asarray(fwd, dtype=float)
    cum = np.cumsum(fwd, axis=-1) * dtau
    tau = np.arange(1, fwd.shape[-1] + 1) * dtau
    return cum / tau


def yields_to_forwards(yld: np.ndarray, dtau: float) -> np.ndarray:
    """Forward rates from yields: f_k = (tau_k*R_k - tau_{k-1}*R_{k-1}) / dtau.

    de Jonge & Govaert (2025) eq. (4).  Exact inverse of `forwards_to_yields`
    on the same grid.  Assumes forwards are constant within each interval.
    """
    yld = np.asarray(yld, dtype=float)
    tau = np.arange(1, yld.shape[-1] + 1) * dtau
    weighted = yld * tau
    prev = np.concatenate(
        [np.zeros(weighted.shape[:-1] + (1,)), weighted[..., :-1]], axis=-1
    )
    return (weighted - prev) / dtau


def forwards_to_discount(fwd: np.ndarray, dtau: float) -> np.ndarray:
    """Discount factors: D_k = exp(-sum_{j<=k} f_j * dtau).

    Englisch et al. (2023) eq. (42), after Glasserman (2003, ch. 3).
    """
    return np.exp(-np.cumsum(np.asarray(fwd, dtype=float), axis=-1) * dtau)


def yields_to_discount(yld: np.ndarray, dtau: float) -> np.ndarray:
    """Discount factors straight from yields: D(tau) = exp(-tau * R(tau)).

    Mueller (2023) eq. (25).  Independent of the forward-rate route, so
    comparing the two is a genuine cross-check.
    """
    yld = np.asarray(yld, dtype=float)
    tau = np.arange(1, yld.shape[-1] + 1) * dtau
    return np.exp(-yld * tau)


# --------------------------------------------------------------------------
# 5. HJM-PCA fit
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HJMFit:
    """Fitted volatility structure and the settings that produced it."""

    V: np.ndarray                 # (M, d) smoothed volatility loadings
    V_raw: np.ndarray             # (M, d) before polynomial smoothing
    eigenvalues: np.ndarray       # (d,) annualised, largest first
    explained: np.ndarray         # (d,) share of total variance
    tenors: np.ndarray            # (M,) years
    scaling: str                  # "sqrt" or "eigenvalue"
    poly_degree: int
    n_observations: int
    sample_start: str
    sample_end: str
    meta: dict = field(default_factory=dict)


def fit_hjm_pca(
    fwd_history: np.ndarray,
    tenors: np.ndarray,
    *,
    periods_per_year: int = 52,
    n_components: int = 3,
    poly_degree: int = 3,
    scaling: str = "sqrt",
    sample_start: str = "",
    sample_end: str = "",
) -> HJMFit:
    """Fit the tenor-dependent volatility matrix V.

    Englisch et al. (2023) Section 2.7.5, five steps:
      1. covariance of forward-curve changes, annualised
      2. eigendecomposition
      3. keep the first `n_components`, scale them
      4. smooth each with a polynomial in the tenor
      5. those become the columns of V

    `scaling` selects step 3:
      "sqrt"       - V_j = sqrt(lambda_j) * Q_j   (standard factorisation;
                     reproduces the magnitudes in the published Figure 3)
      "eigenvalue" - V_j = lambda_j * Q_j         (as literally printed in
                     both papers; retained so the two can be compared)
    """
    if scaling not in ("sqrt", "eigenvalue"):
        raise ValueError(f"unknown scaling {scaling!r}")

    changes = np.diff(np.asarray(fwd_history, dtype=float), axis=0)
    cov = np.cov(changes, rowvar=False) * periods_per_year

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    total = eigvals[eigvals > 0].sum()
    keep_val = eigvals[:n_components]
    keep_vec = eigvecs[:, :n_components]

    # Sign convention: make the largest-magnitude loading of each component
    # positive, so repeated fits are comparable and plots are stable.
    for j in range(n_components):
        if keep_vec[np.argmax(np.abs(keep_vec[:, j])), j] < 0:
            keep_vec[:, j] *= -1.0

    factor = np.sqrt(np.maximum(keep_val, 0.0)) if scaling == "sqrt" else keep_val
    v_raw = keep_vec * factor[None, :]

    # Step 4: regularise each loading with a polynomial in the tenor.
    v_smooth = np.empty_like(v_raw)
    for j in range(n_components):
        coeffs = np.polyfit(tenors, v_raw[:, j], poly_degree)
        v_smooth[:, j] = np.polyval(coeffs, tenors)

    return HJMFit(
        V=v_smooth,
        V_raw=v_raw,
        eigenvalues=keep_val,
        explained=keep_val / total,
        tenors=np.asarray(tenors, dtype=float),
        scaling=scaling,
        poly_degree=poly_degree,
        n_observations=changes.shape[0],
        sample_start=sample_start,
        sample_end=sample_end,
    )


def hjm_drift(V: np.ndarray, dtau: float) -> np.ndarray:
    """No-arbitrage drift A(tau) = sigma(tau) . integral_0^tau sigma(s) ds.

    The inner integral is taken with the trapezoidal rule, as Englisch et al.
    (2023) specify.  `hjm_drift_riemann` gives the left-Riemann variant used
    by de Jonge & Govaert (2025) eq. (8), for cross-checking.
    """
    V = np.asarray(V, dtype=float)
    m = V.shape[0]
    integral = np.empty_like(V)
    integral[0, :] = 0.5 * V[0, :] * dtau          # 0 -> tau_1, f(0) taken as V[0]
    for k in range(1, m):
        integral[k, :] = integral[k - 1, :] + 0.5 * (V[k - 1, :] + V[k, :]) * dtau
    return np.sum(V * integral, axis=1)


def hjm_drift_riemann(V: np.ndarray, dtau: float) -> np.ndarray:
    """Left-Riemann drift, de Jonge & Govaert (2025) eq. (8)."""
    V = np.asarray(V, dtype=float)
    integral = np.cumsum(V, axis=0) * dtau
    return np.sum(V * integral, axis=1)


def expected_yield_curve(
    f0: np.ndarray,
    drift: np.ndarray,
    *,
    n_steps: int,
    dt: float,
    dtau: float,
    n_output: int,
) -> np.ndarray:
    """The curve the simulator expects after `n_steps`, with no randomness.

    Runs eq. (41) with the shock term switched off.  Because the shock has
    zero mean and the dynamics are linear in F, this is exactly the mean of
    the simulated forward curves.  Returns yields on the first `n_output`
    tenors.
    """
    f = np.asarray(f0, dtype=float).copy()
    drift = np.asarray(drift, dtype=float)
    for _ in range(n_steps):
        slope = np.empty_like(f)
        slope[:-1] = np.diff(f) / dtau
        slope[-1] = slope[-2]
        f = (f + (drift[: f.shape[0]] + slope) * dt)[:-1]
    return forwards_to_yields(f[:n_output], dtau)


def calibrate_term_premium(
    f0: np.ndarray,
    y0: np.ndarray,
    fit: HJMFit,
    *,
    n_steps: int,
    dt: float,
    dtau: float,
    n_output: int,
) -> dict:
    """Market price of risk such that the expected curve stays where it is.

    Under P = Q the simulator treats today's forward curve as its forecast,
    which flattens the expected curve (Englisch et al. assume this; Section
    2.7.4 notes it "could be relaxed by exogenously specifying the market
    price of risk").  This chooses a constant phi so that the expected yield
    curve after `n_steps` equals today's yield curve, by least squares over
    all `n_output` maturities.  That is the "no expected change" benchmark:
    the best forecast of future yields is today's yields.

    The expected curve is linear in phi, so the response matrix is exact
    and the solve is a single least-squares problem.

    phi is calibrated to ONE starting curve.  A different valuation date
    needs a fresh calibration.

    Returns phi, the expected curve it produces, and the fit error in bp.
    """
    A = hjm_drift(fit.V, dtau)
    kw = dict(n_steps=n_steps, dt=dt, dtau=dtau, n_output=n_output)
    target = np.asarray(y0, dtype=float)[:n_output]

    base = expected_yield_curve(f0, A, **kw)
    J = np.column_stack([
        expected_yield_curve(f0, A + fit.V[:, j], **kw) - base
        for j in range(fit.V.shape[1])
    ])
    phi, *_ = np.linalg.lstsq(J, target - base, rcond=None)

    expected = expected_yield_curve(f0, A + fit.V @ phi, **kw)
    err = np.abs(expected - target) * 1e4
    return {
        "phi": phi,
        "expected_curve": expected,
        "expected_curve_paper": base,
        "max_error_bp": float(err.max()),
        "mean_error_bp": float(err.mean()),
    }


# --------------------------------------------------------------------------
# 6. Simulation
# --------------------------------------------------------------------------

def simulate_forward_paths(
    f0: np.ndarray,
    fit: HJMFit,
    *,
    n_scenarios: int,
    n_steps: int,
    dt: float,
    dtau: float,
    n_output: int,
    seed: int | None = None,
    drift: str = "trapezoid",
    market_price_of_risk: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate forward-curve paths.

    Englisch et al. (2023) eq. (41):

        F_{t+dt} = F_t + (A + dF_t/dtau) * dt + V * (W_{t+dt} - W_t)

    With `market_price_of_risk = None` this is exactly the paper: the
    real-world and risk-neutral measures coincide (P = Q).

    Passing a d-vector phi relaxes that assumption, as Englisch et al.
    Section 2.7.4 allows ("could be relaxed by exogenously specifying the
    market price of risk").  The drift becomes A + V @ phi, which is the
    standard HJM change of measure with a constant market price of risk.

    The grid starts longer than `n_output` so the forward difference is
    always defined; one tenor is dropped per step, as the paper's footnote
    describes.  Requires len(f0) >= n_output + n_steps.

    Returns (n_scenarios, n_steps + 1, n_output) forward curves.
    """
    f0 = np.asarray(f0, dtype=float)
    m0 = f0.shape[0]
    if m0 < n_output + n_steps:
        raise ValueError(
            f"initial grid has {m0} tenors; need at least {n_output + n_steps}"
        )

    A_full = (hjm_drift(fit.V, dtau) if drift == "trapezoid"
              else hjm_drift_riemann(fit.V, dtau))
    if market_price_of_risk is not None:
        A_full = A_full + fit.V @ np.asarray(market_price_of_risk, dtype=float)

    rng = np.random.default_rng(seed)
    d = fit.V.shape[1]

    out = np.empty((n_scenarios, n_steps + 1, n_output), dtype=float)
    curve = np.tile(f0, (n_scenarios, 1))
    out[:, 0, :] = curve[:, :n_output]

    for step in range(1, n_steps + 1):
        m = curve.shape[1]
        slope = np.empty_like(curve)
        slope[:, :-1] = np.diff(curve, axis=1) / dtau
        slope[:, -1] = slope[:, -2]                      # unused: tenor dropped below

        dW = rng.standard_normal((n_scenarios, d)) * np.sqrt(dt)
        shock = dW @ fit.V[:m, :].T

        curve = curve + (A_full[:m] + slope) * dt + shock
        curve = curve[:, :-1]                            # drop the last tenor
        out[:, step, :] = curve[:, :n_output]

    return out


# --------------------------------------------------------------------------
# 7. Curve features for the decision network
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CurveFeatureMap:
    """Projects a 180-maturity yield curve onto 3 numbers.

    Mueller (2023) Section 4.2.1: "To reduce input dimensionality, we use PCA
    to project the high-dimensional yield curve into R^3. The PCA is
    performed before training on a subset of yield curves from the training
    data set."

    This is a DIFFERENT PCA from the one in `fit_hjm_pca`.  That one runs on
    historical forward-curve CHANGES and defines how the simulator moves.
    This one runs on simulated yield-curve LEVELS and defines what the
    network sees.
    """

    mean: np.ndarray          # (M,) average curve of the fitting set
    components: np.ndarray    # (k, M) orthonormal directions, largest first
    explained: np.ndarray     # (k,) share of variance per component
    n_curves: int             # number of curves the map was fitted on

    def transform(self, yields: np.ndarray) -> np.ndarray:
        """(..., M) yield curves -> (..., k) features."""
        return (np.asarray(yields, dtype=float) - self.mean) @ self.components.T

    def inverse(self, features: np.ndarray) -> np.ndarray:
        """(..., k) features -> (..., M) reconstructed yield curves."""
        return self.mean + np.asarray(features, dtype=float) @ self.components


def fit_curve_features(yields: np.ndarray, n_components: int = 3) -> CurveFeatureMap:
    """Fit the 3-number curve compression on a set of yield curves.

    yields : (n_curves, M) - pass simulated TRAINING curves, all time steps
             pooled, never validation curves.

    Features are centred but not standardised; [M] does not say either way,
    and the network's first layer can rescale them.  The sign of each
    component is fixed so its largest loading is positive, making repeated
    fits comparable.
    """
    Y = np.asarray(yields, dtype=float)
    mean = Y.mean(axis=0)
    _, svals, vt = np.linalg.svd(Y - mean, full_matrices=False)
    comps = vt[:n_components].copy()
    for j in range(n_components):
        if comps[j, np.argmax(np.abs(comps[j]))] < 0:
            comps[j] *= -1.0
    var = svals ** 2
    return CurveFeatureMap(mean=mean, components=comps,
                           explained=var[:n_components] / var.sum(),
                           n_curves=Y.shape[0])


# --------------------------------------------------------------------------
# 8. Scenario sets - regenerate instead of storing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSet:
    """What the ALM model consumes, for one batch of scenarios.

    All arrays are (n_scenarios, n_steps + 1, n_output); index 0 on the
    time axis is the valuation date, identical across scenarios.
    """

    forwards: np.ndarray
    yields: np.ndarray
    discount: np.ndarray
    variant: str
    seed: int
    valuation_date: str

    @property
    def n_scenarios(self) -> int:
        return self.forwards.shape[0]


def load_scenarios(fit_path: str, *, n_scenarios: int, seed: int,
                   variant: str = "term_premium") -> ScenarioSet:
    """Rebuild a scenario set exactly from a saved fit plus a seed.

    The generator is deterministic given (V, f0, phi, seed, n_scenarios), so
    storing the 19 KB fit file replaces storing ~175 MB of paths per 4,000
    scenarios.  Note: the SAME n_scenarios must be used to reproduce a set,
    because the random draws are taken in blocks of n_scenarios per step.

    variant : "term_premium" (what we train on) or "sqrt" (paper, P = Q)
    """
    if variant not in ("term_premium", "sqrt"):
        raise ValueError(f"unknown variant {variant!r}")
    z = np.load(fit_path)
    fit = HJMFit(V=z["V"], V_raw=z["V_raw"], eigenvalues=z["eigenvalues"],
                 explained=np.full(z["V"].shape[1], np.nan),
                 tenors=z["tenors"], scaling="sqrt", poly_degree=3,
                 n_observations=0, sample_start="", sample_end="")
    dtau = float(z["dtau"]); n_steps = int(z["n_steps"]); n_out = int(z["n_output"])
    mpr = z["term_premium_phi"] if variant == "term_premium" else None
    paths = simulate_forward_paths(
        z["f0"], fit, n_scenarios=n_scenarios, n_steps=n_steps,
        dt=dtau, dtau=dtau, n_output=n_out, seed=seed,
        market_price_of_risk=mpr)
    return ScenarioSet(forwards=paths,
                       yields=forwards_to_yields(paths, dtau),
                       discount=forwards_to_discount(paths, dtau),
                       variant=variant, seed=seed,
                       valuation_date=str(z["valuation_date"]))


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def check_forward_consistency(params: np.ndarray, tenors: np.ndarray,
                              dtau: float) -> dict:
    """Cross-check the analytic forward curve against the discrete inverse.

    Route A: Svensson spot -> yields_to_forwards (de Jonge & Govaert eq. 4)
    Route B: analytic instantaneous forwards (svensson_forward)

    These are not identical by construction: route A returns the average
    forward over each monthly interval, route B the instantaneous rate at
    the interval end.  They agree to O(dtau).  The simulator uses route A
    (`svensson_interval_forward`), because eq. (42) needs interval averages.
    """
    spot = svensson_spot(params, tenors)
    fwd_discrete = yields_to_forwards(spot, dtau)
    fwd_analytic = svensson_forward(params, tenors)
    diff = np.abs(fwd_discrete - fwd_analytic)
    return {
        "max_abs_diff_bp": float(diff.max() * 1e4),
        "mean_abs_diff_bp": float(diff.mean() * 1e4),
    }


def check_discount_consistency(params: np.ndarray, tenors: np.ndarray,
                               dtau: float) -> dict:
    """Cross-check discount factors built two independent ways.

    Route A: D = exp(-tau * R(tau))                 (Mueller eq. 25)
    Route B: D = exp(-sum f_j * dtau)               (Englisch et al. eq. 42)

    Using the discrete forwards of route A's own grid, these agree to
    machine precision, because the cumulative sum of those forwards is
    exactly tau*R by construction.
    """
    spot = svensson_spot(params, tenors)
    d_from_yields = yields_to_discount(spot, dtau)
    fwd = yields_to_forwards(spot, dtau)
    d_from_forwards = forwards_to_discount(fwd, dtau)
    diff = np.abs(d_from_yields - d_from_forwards)
    return {
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float((diff / d_from_yields).max()),
    }