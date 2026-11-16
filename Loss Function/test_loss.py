"""Tests for Layer 4 (loss.py).

    python -m pytest -q test_loss.py
"""
import os

import numpy as np
import pytest

import balance_sheet as bs
import constraints as cs
import loss as ls
import simulator as sim
import yieldcurve as yc

FIT = os.environ.get("FIT_NPZ", "output_paper/fit_sqrt.npz")
needs_fit = pytest.mark.skipif(not os.path.isfile(FIT), reason="run run_fit.py first")


# ------------------------------------------------------- the printed formulas
def test_no_breach_means_no_penalty():
    for name, b in cs.BOUNDS.items():
        ok = b + 0.5 if name not in cs.UPPER_BOUNDED else 0.0
        assert ls.breach(name, ok) == 0.0, name


def test_breach_is_squared_and_grows_faster_than_linearly():
    # LCR 0.05 below its bound, then 0.10 below
    one = ls.breach("LCR", cs.BOUNDS["LCR"] - 0.05)
    two = ls.breach("LCR", cs.BOUNDS["LCR"] - 0.10)
    assert one == pytest.approx((1.05) ** 2 - 1.0)
    assert two > 2.0 * one                      # "more than linearly worse"


def test_the_sides_of_the_bounds_follow_table_4_not_equation_30():
    """E9: [E] Table 4's own violating values settle which side is a breach."""
    assert ls.breach("CMR", 0.92) > 0 and ls.breach("CMR", 1.50) == 0
    assert ls.breach("EYR", -2.99) > 0 and ls.breach("EYR", 21.17) == 0
    assert ls.breach("IRS", 0.09) > 0 and ls.breach("IRS", -0.03) == 0
    assert ls.breach("IRS", -0.09) > 0          # the limit is on |IRS| (B4)


def test_penalty_loss_squares_again():
    assert ls.penalty_loss(0.0) == 0.0
    assert ls.penalty_loss(3.0) == pytest.approx(15.0)


def test_weights_are_keyed_not_positional():
    """[E] Table 2 lists them in a different order than (31b) indexes."""
    assert ls.WEIGHTS["E_over_RWA"] == 2.5      # not 1.0, which is LCR's
    assert ls.WEIGHTS["EYR"] == 0.002


# ----------------------------------------------------------- the target loss
def test_target_loss_ignores_overshoot_but_not_shortfall():
    E0, target = 2000.0, (1 + ls.MU_EVAL) ** 5
    assert ls.target_loss(target * E0 * 1.10, E0) == 0.0
    assert ls.target_loss(target * E0 * 0.90, E0) > 0.0
    assert ls.target_loss(target * E0, E0) == pytest.approx(0.0)


def test_symmetric_target_loss_reproduces_table_3():
    """B9 + B10: l_t = 100 x E[(ER - (1+mu)^T)^2] for [E] Table 3's BM_C,
    whose equity ratio has mean 1.124 and sd 0.026."""
    rng = np.random.default_rng(0)
    er = 1.124 + 0.026 * rng.standard_normal(400_000)
    got = ls.target_loss(er * 2000.0, 2000.0, asymmetric=False).mean()
    assert got == pytest.approx(0.997, abs=0.02)


def test_target_loss_scale_is_ten_on_the_ratio():
    E0 = 2000.0
    ET = ((1 + ls.MU_EVAL) ** 5 - 0.1) * E0     # ratio 0.1 short of the target
    assert ls.target_loss(ET, E0) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- the utility
def test_crra_matches_table_3():
    """[E] Table 3: l_u(gamma = 10) = -0.079 for MM, whose mean ER is 1.153."""
    assert ls.utility_loss(1.153 * 2000.0, 2000.0) == pytest.approx(-0.0803, abs=5e-4)


def test_crra_is_increasing_and_concave():
    x = np.array([0.8, 1.0, 1.2, 1.5])
    u = ls.utility(x)
    assert np.all(np.diff(u) > 0)
    assert np.all(np.diff(np.diff(u)) < 0)


def test_utility_loss_survives_zero_equity():
    assert np.isfinite(ls.utility_loss(0.0, 2000.0))
    assert np.isfinite(ls.utility_loss(-500.0, 2000.0))


def test_total_loss_is_target_plus_weighted_penalty():
    E0, ET, p = 2000.0, 2100.0, 0.4
    assert ls.total_loss(ET, p, E0, lam=3.5) == pytest.approx(
        ls.target_loss(ET, E0) + 3.5 * ls.penalty_loss(p))


# ----------------------------------------------------------- against Layer 3
@pytest.fixture(scope="module")
def run():
    y0 = np.load(FIT)["y0"]
    x0 = bs.build_initial_state(y0)
    p = sim.Params(loan_spread=sim.calibrate_loan_spread(x0, y0, 303.354))
    sc = yc.load_scenarios(FIT, n_scenarios=200, seed=20221230, variant="sqrt")
    return sim.simulate(x0, sc.yields, p, sim.RollOver())


@needs_fit
def test_the_terminal_roll_forward_carries_no_penalty(run):
    """B11: [E] Alg. 1 loops i = 0..H-1, so the last row is excluded."""
    h = run["history"]
    H = h[-1]["step"]
    full = ls.accumulated_penalty(h, H)
    plus_last = ls.accumulated_penalty(h, H + 1)
    assert np.all(plus_last >= full - 1e-12)
    assert not np.allclose(plus_last, full)     # the last row does breach


@needs_fit
def test_penalty_is_zero_only_when_nothing_is_breached(run):
    h = run["history"]
    H = h[-1]["step"]
    p = ls.accumulated_penalty(h, H)
    clean = np.ones_like(p, dtype=bool)
    for row in h:
        if row["step"] >= H:
            continue
        for name in ("LCR", "NSFR", "CMR", "E_over_RWA", "IRS"):
            clean &= cs.violation(name, row[name]) <= 0
        if "EYR" in row:
            clean &= cs.violation("EYR", row["EYR"]) <= 0
    np.testing.assert_allclose(p[clean], 0.0, atol=1e-12)
    assert np.all(p[~clean] > 0.0)


@needs_fit
def test_evaluate_has_every_table_3_row(run):
    r = ls.evaluate(run)
    for k in ("total", "target", "penalty", "utility", "penalty_ES0.95"):
        assert k in r["loss"]
    for k in ("mean", "sd", "skew", "excess_kurtosis", "VaR0.95", "ES0.95"):
        assert k in r["equity_ratio"]
    assert r["loss"]["penalty"] >= 0.0
    assert r["loss"]["target_symmetric"] >= r["loss"]["target"]


@needs_fit
def test_dividend_yield_uses_four_annual_steps(run):
    """[E] (52b) divides by T - 1, which with T = 5 is the number of payouts."""
    h = run["history"]
    paid = [row for row in h if "dividend" in row]
    assert [row["step"] for row in paid] == [12, 24, 36, 48]
    r = ls.evaluate(run)
    E0 = h[0]["E"]
    total = sum(row["dividend"] for row in paid)
    assert r["annualised"]["dividend_yield"] == pytest.approx(
        float((total / E0).mean()) / 4.0, abs=1e-12)


@needs_fit
def test_constraints_are_read_before_the_dividend():
    """[E] Fig. 10's note: equity is recorded at the end of the restructure
    step, so the payout shows up in the next row, not this one.

    Switching the dividend off must therefore leave the row at t = 12
    untouched and change the row at t = 13."""
    y0 = np.load(FIT)["y0"]
    x0 = bs.build_initial_state(y0)
    kappa = sim.calibrate_loan_spread(x0, y0, 303.354)
    sc = yc.load_scenarios(FIT, n_scenarios=50, seed=20221230, variant="sqrt")

    rows = {}
    for share in (0.5, 0.0):
        p = sim.Params(loan_spread=kappa, dividend_share=share)
        out = sim.simulate(x0, sc.yields, p, sim.RollOver())
        rows[share] = {r["step"]: r for r in out["history"]}

    paid = rows[0.5][12]["dividend"]
    assert float(paid.max()) > 0                       # some year-1 profit
    np.testing.assert_allclose(rows[0.5][12]["E"], rows[0.0][12]["E"], atol=1e-9)
    np.testing.assert_allclose(rows[0.5][12]["LCR"], rows[0.0][12]["LCR"], atol=1e-9)
    assert not np.allclose(rows[0.5][13]["E"], rows[0.0][13]["E"])
