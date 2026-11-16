"""Timing tests: when the bank is allowed to act, and when it is not.

[M] 3.3.1 : no roll-forward at t = 0, but the bank does decide there.
[M] 3.3.3 : the last step is a roll-forward, so there is no decision at T.
"""
import os

import numpy as np
import pytest

import balance_sheet as bs
import simulator as sim
import yieldcurve as yc

FIT = os.environ.get("FIT_NPZ", "output_paper/fit_sqrt.npz")
needs_fit = pytest.mark.skipif(not os.path.isfile(FIT), reason="run run_fit.py first")


class Recorder:
    """Borrows a fixed amount whenever it is asked, and logs when it was asked."""

    def __init__(self):
        self.steps = []

    def __call__(self, step, state, ctx):
        self.steps.append(step)
        S = state.C.shape[0]
        nB = len(ctx.params.invest_maturities)
        nK = len(ctx.params.finance_maturities)
        aK = np.zeros((S, nK))
        aK[:, 5] = 10.0
        return np.zeros((S, nB)), aK


@pytest.fixture(scope="module")
def setup():
    y0 = np.load(FIT)["y0"]
    x0 = bs.build_initial_state(y0)
    p = sim.Params(loan_spread=sim.calibrate_loan_spread(x0, y0, 303.354))
    sc = yc.load_scenarios(FIT, n_scenarios=20, seed=20221230, variant="sqrt")
    return x0, y0, p, sc


@needs_fit
def test_the_bank_decides_at_t0_and_not_at_the_horizon(setup):
    x0, y0, p, sc = setup
    rec = Recorder()
    sim.simulate(x0, sc.yields, p, rec)
    assert rec.steps == list(range(60))          # 0 .. H-1, never H


@needs_fit
def test_a_t0_decision_shows_up_in_the_t0_snapshot(setup):
    """[M] 3.3.2: equity is recorded at the end of the restructure step, so a
    decision made at t = 0 must already be in the first row."""
    x0, y0, p, sc = setup
    quiet = sim.simulate(x0, sc.yields, p, sim.NoAction())["history"][0]
    busy = sim.simulate(x0, sc.yields, p, Recorder())["history"][0]
    assert float(np.median(busy["C"] - quiet["C"])) == pytest.approx(10.0, abs=1e-9)
    assert float(np.median(busy["K"] - quiet["K"])) > 0


@needs_fit
def test_no_roll_forward_happens_before_the_first_snapshot(setup):
    """With no action, t = 0 must be exactly the balance sheet Layer 2 froze:
    no interest, no maturities, no costs have been applied yet."""
    x0, y0, p, sc = setup
    row = sim.simulate(x0, sc.yields, p, sim.NoAction())["history"][0]
    ref = bs.report(x0, y0)["values_mCHF"]
    for k in ("C", "B", "RP", "RE", "SD", "SF", "K", "E"):
        assert float(np.median(row[k])) == pytest.approx(ref[k], abs=1e-6), k
