"""Regression tests for the yield curve layer.

Two kinds:
  * unit tests  - need no data, check the maths on synthetic inputs
  * data tests  - pin the numbers in layer1_report.md on the real SNB file

Run all of them:
    RENDOPAR_CSV=path/to/snb-data-rendopar-....csv  python -m pytest -q

Without RENDOPAR_CSV the data tests are skipped, not failed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import yieldcurve as yc

DTAU = 1.0 / 12.0
TEN = np.arange(1, 241) * DTAU
# A realistic parameter set: the SNB fit for 2022-12-30, as decimals.
P_2022 = np.array([0.00890401, 0.000353016, 0.00377662, 0.0223517, 1.08667, 8.8053])


# =========================================================== unit tests

def test_svensson_spot_is_average_of_forward():
    """R(m) must equal (1/m) * integral_0^m f(u) du — the identity the
    closed-form forward curve relies on."""
    fine = np.linspace(0.0, 15.0, 150_001)
    f = yc.svensson_forward(P_2022, fine)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(fine))])
    for m in (0.5, 2.0, 5.0, 10.0, 15.0):
        i = np.searchsorted(fine, m)
        assert abs(cum[i] / fine[i] - yc.svensson_spot(P_2022, np.array([fine[i]]))) < 1e-10


def test_interval_forwards_reproduce_snb_curve_exactly():
    """The simulator's day-0 curve must BE the SNB curve.  Instantaneous
    point forwards miss it by up to ~2.2 bp at 1 month; interval averages
    hit it to machine precision."""
    y_snb = yc.svensson_spot(P_2022, TEN)
    f_iv = yc.svensson_interval_forward(P_2022, 240, DTAU)
    assert np.max(np.abs(yc.forwards_to_yields(f_iv, DTAU) - y_snb)) < 1e-14
    d_snb = np.exp(-TEN * y_snb)
    assert np.max(np.abs(yc.forwards_to_discount(f_iv, DTAU) - d_snb)) < 1e-14
    f_pt = yc.svensson_forward(P_2022, TEN)
    assert np.max(np.abs(yc.forwards_to_yields(f_pt, DTAU) - y_snb)) * 1e4 > 1.0


def test_yield_forward_roundtrip_is_exact():
    y = yc.svensson_spot(P_2022, TEN)
    back = yc.forwards_to_yields(yc.yields_to_forwards(y, DTAU), DTAU)
    assert np.max(np.abs(back - y)) < 1e-14


def test_discount_two_routes_agree():
    out = yc.check_discount_consistency(P_2022, TEN, DTAU)
    assert out["max_abs_diff"] < 1e-14


def test_discount_factors_decrease_with_maturity():
    d = yc.forwards_to_discount(yc.svensson_forward(P_2022, TEN), DTAU)
    assert np.all(np.diff(d) < 0) and np.all(d > 0)


def test_drift_variants_close():
    v = np.column_stack([0.006 * np.ones(240), np.linspace(0.003, -0.002, 240),
                         np.sin(TEN / 3) * 0.002])
    a, b = yc.hjm_drift(v, DTAU), yc.hjm_drift_riemann(v, DTAU)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 0.01


def _toy_fit(V):
    return yc.HJMFit(V=V, V_raw=V, eigenvalues=np.ones(V.shape[1]),
                     explained=np.ones(V.shape[1]), tenors=TEN, scaling="sqrt",
                     poly_degree=3, n_observations=0, sample_start="", sample_end="")


def test_zero_volatility_simulation_equals_expected_curve():
    """With no shocks the simulator must reproduce expected_yield_curve
    exactly — this is what makes the term premium calibration valid."""
    V = np.zeros((240, 3))
    f0 = yc.svensson_interval_forward(P_2022, 240, DTAU)
    paths = yc.simulate_forward_paths(f0, _toy_fit(V), n_scenarios=2, n_steps=60,
                                      dt=DTAU, dtau=DTAU, n_output=180, seed=0)
    sim = yc.forwards_to_yields(paths[0, -1], DTAU)
    exp = yc.expected_yield_curve(f0, np.zeros(240), n_steps=60, dt=DTAU,
                                  dtau=DTAU, n_output=180)
    assert np.max(np.abs(sim - exp)) < 1e-14


def test_simulation_mean_matches_expected_curve():
    """With shocks, the average over many scenarios converges to the
    expected curve (shocks have zero mean and the dynamics are linear)."""
    V = np.column_stack([0.006 * np.ones(240), np.linspace(0.003, -0.002, 240),
                         np.zeros(240)])
    f0 = yc.svensson_interval_forward(P_2022, 240, DTAU)
    fit = _toy_fit(V)
    paths = yc.simulate_forward_paths(f0, fit, n_scenarios=20_000, n_steps=60,
                                      dt=DTAU, dtau=DTAU, n_output=180, seed=3)
    mean = yc.forwards_to_yields(paths[:, -1], DTAU).mean(axis=0)
    exp = yc.expected_yield_curve(f0, yc.hjm_drift(V, DTAU), n_steps=60, dt=DTAU,
                                  dtau=DTAU, n_output=180)
    assert np.max(np.abs(mean - exp)) * 1e4 < 2.0          # under 2 bp


def test_term_premium_calibration_hits_target():
    V = np.column_stack([0.006 * np.ones(240), np.linspace(0.003, -0.002, 240),
                         np.sin(TEN / 3) * 0.002])
    f0 = yc.svensson_interval_forward(P_2022, 240, DTAU)
    y0 = yc.svensson_spot(P_2022, TEN)
    tp = yc.calibrate_term_premium(f0, y0, _toy_fit(V), n_steps=60, dt=DTAU,
                                   dtau=DTAU, n_output=180)
    paper_err = np.abs(tp["expected_curve_paper"] - y0[:180]).max()
    assert tp["max_error_bp"] < paper_err * 1e4             # strictly better than P = Q


def test_curve_features_reconstruct_three_factor_curves():
    rng = np.random.default_rng(0)
    basis = np.vstack([np.ones(180), np.linspace(-1, 1, 180),
                       np.linspace(-1, 1, 180) ** 2])
    Y = 0.01 + rng.standard_normal((5000, 3)) @ basis * 0.005
    fmap = yc.fit_curve_features(Y, 3)
    assert fmap.explained.sum() > 0.999999
    assert np.max(np.abs(fmap.inverse(fmap.transform(Y)) - Y)) < 1e-12


def test_loader_handles_holidays_and_percent(tmp_path):
    csv = tmp_path / "mini.csv"
    rows = ['﻿"CubeId";"rendopar"', '"PublishingDate";"2025-09-01 14:30"', "",
            '"Date";"d0";"Value"']
    for p, v in zip(yc.PARAMS, ("1.0", "-0.5", "0.2", "0.3", "1.5", "8.0")):
        rows.append(f'"2022-12-29";"{p}";"{v}"')
    for p in yc.PARAMS:
        rows.append(f'"2022-12-30";"{p}";')                 # a holiday: blank
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df = yc.load_rendopar(str(csv))
    assert len(df) == 1                                      # holiday dropped
    assert df["b0"].iloc[0] == pytest.approx(0.01)           # percent -> decimal
    assert df["t1"].iloc[0] == pytest.approx(1.5)            # tau left in years


# =========================================================== data tests

CSV = os.environ.get("RENDOPAR_CSV", "")
needs_data = pytest.mark.skipif(not os.path.isfile(CSV),
                                reason="set RENDOPAR_CSV to the SNB file")


@pytest.fixture(scope="module")
def real():
    params = yc.load_rendopar(CSV)
    weekly = params.loc["2005-01-01":"2022-07-15"].resample("W-WED").last().dropna()
    F = yc.svensson_interval_forward(weekly.values, 240, DTAU)
    fit = yc.fit_hjm_pca(F, TEN, scaling="sqrt")
    row = params.loc[params.index <= pd.Timestamp("2022-12-30")].iloc[-1]
    f0 = yc.svensson_interval_forward(row.values, 240, DTAU)
    y0 = yc.svensson_spot(row.values, TEN)
    tp = yc.calibrate_term_premium(f0, y0, fit, n_steps=60, dt=DTAU, dtau=DTAU,
                                   n_output=180)
    return dict(params=params, weekly=weekly, fit=fit, f0=f0, y0=y0, tp=tp)


def _inverted_share(real, mpr):
    paths = yc.simulate_forward_paths(real["f0"], real["fit"], n_scenarios=4000,
                                      n_steps=60, dt=DTAU, dtau=DTAU, n_output=180,
                                      seed=20221230, market_price_of_risk=mpr)
    y = yc.forwards_to_yields(paths, DTAU)
    return float((y[:, -1, 179] < y[:, -1, 0]).mean())


@needs_data
def test_data_counts(real):
    assert len(real["params"]) == 7534
    assert len(real["weekly"]) == 916


@needs_data
def test_starting_curve(real):
    got = {m: round(float(real["y0"][int(m * 12) - 1]) * 100, 4) for m in (1, 5, 10, 15)}
    assert got == {1: 1.1276, 5: 1.4141, 10: 1.5533, 15: 1.5867}
    assert round(float(real["y0"][0]) * 100, 4) == 0.9487


@needs_data
def test_fitted_volatility(real):
    np.testing.assert_allclose(real["fit"].eigenvalues,
                               [0.00797887, 0.00137517, 0.00089326], rtol=1e-5)
    assert real["fit"].V.min() == pytest.approx(-0.00279858, abs=1e-7)
    assert real["fit"].V.max() == pytest.approx(0.00652758, abs=1e-7)


@needs_data
def test_loadings_inside_published_figure_3(real):
    assert -0.005 < real["fit"].V.min() and real["fit"].V.max() < 0.008


@needs_data
def test_term_premium_phi(real):
    np.testing.assert_allclose(real["tp"]["phi"], [-0.0870, -0.1344, -0.3755],
                               atol=1e-4)
    assert real["tp"]["max_error_bp"] == pytest.approx(8.77, abs=0.01)


@needs_data
def test_inversion_rates(real):
    assert _inverted_share(real, None) == pytest.approx(0.4490, abs=1e-4)
    assert _inverted_share(real, real["tp"]["phi"]) == pytest.approx(0.2943, abs=1e-4)


@needs_data
def test_load_scenarios_reproduces_stored_sample(tmp_path, real):
    """A saved fit plus a seed must rebuild the exact scenarios."""
    fp = tmp_path / "fit.npz"
    np.savez(fp, V=real["fit"].V, V_raw=real["fit"].V_raw,
             eigenvalues=real["fit"].eigenvalues, tenors=TEN, f0=real["f0"],
             y0=real["y0"], term_premium_phi=real["tp"]["phi"],
             valuation_date=np.array("2022-12-30"), dtau=np.array(DTAU),
             n_steps=np.array(60), n_output=np.array(180))
    a = yc.load_scenarios(str(fp), n_scenarios=500, seed=11)
    b = yc.load_scenarios(str(fp), n_scenarios=500, seed=11)
    assert np.array_equal(a.forwards, b.forwards)
    assert a.forwards.shape == (500, 61, 180)
    assert np.max(np.abs(a.yields[:, 0] - real["y0"][:180])) < 1e-14   # all start ON Y0
