"""Tests for Layer 3 (simulator.py).

    python -m pytest -q test_simulator.py

The tests that need scenarios are skipped unless output_paper/fit_sqrt.npz
exists (run run_fit.py --valuation-date 2022-07-15 --model paper first).
"""
import os

import numpy as np
import pytest

import balance_sheet as bs
import constraints as cs
import simulator as sim
import yieldcurve as yc

FIT = os.environ.get("FIT_NPZ", "output_paper/fit_sqrt.npz")
needs_fit = pytest.mark.skipif(not os.path.isfile(FIT), reason="run run_fit.py first")
N = sim.N


# --------------------------------------------------------------- mechanics
def test_shift_drops_the_first_entry_and_pads():
    x = np.arange(1.0, 6.0).reshape(1, 5)
    np.testing.assert_allclose(sim._shift_up(x), [[2, 3, 4, 5, 0]])


def test_loan_rate_never_charges_the_customer_a_negative_rate():
    y = np.array([-0.05, -0.001, 0.0, 0.02])
    r = sim.loan_rate(y, spread=0.0)
    assert np.all(r >= 0.0)
    assert r[-1] == pytest.approx(np.exp(0.02 / 12) - 1)


def test_par_bond_is_worth_exactly_one_before_the_spread():
    D = np.exp(-np.arange(1, N + 1) / 12.0 * 0.015)[None, :]
    for m in (3, 12, 36, 180):
        Z = sim.par_bonds(D, (m,), spread=0.0)[0, 0]
        assert float(D[0] @ Z) == pytest.approx(1.0, abs=1e-12)


def test_spread_makes_investments_cheaper_and_borrowing_dearer():
    D = np.exp(-np.arange(1, N + 1) / 12.0 * 0.015)[None, :]
    inv = sim.par_bonds(D, (60,), -0.0015)[0, 0]
    fin = sim.par_bonds(D, (60,), +0.0015)[0, 0]
    assert float(D[0] @ inv) < 1.0 < float(D[0] @ fin)


def test_coupon_dates_are_semi_annual_counted_back_from_maturity():
    assert sim._coupon_months(36) == [6, 12, 18, 24, 30, 36]
    assert sim._coupon_months(3) == [3]            # the 3m financing bond
    assert sim._coupon_months(9) == [3, 9]


# ---------------------------------------------------------------- deposits
def _settled_deposits(total=100.0):
    """A deposit book already in the steady state of the rolling scheme."""
    d = np.zeros((1, 4, N))
    for j, m in enumerate(sim.REF_MATURITIES):
        d[0, j] = total / 4 * bs._steady_shape(m)
    return d


def test_the_settled_shape_is_the_fixed_point_of_the_reinvestment_rule():
    """balance_sheet._steady_shape must be exactly what simulator.REINVEST
    leaves unchanged; otherwise Layer 2 and Layer 3 disagree."""
    p = sim.Params(loan_spread=0.0)
    for j, m in enumerate(sim.REF_MATURITIES):
        d = np.zeros((1, 4, N))
        d[0, j] = bs._steady_shape(m)
        out, _ = sim.roll_deposits(d, np.zeros(1), 0.0, p)
        np.testing.assert_allclose(out[0, j], d[0, j], atol=1e-14)


def test_deposits_keep_their_shape_and_grow_at_the_stated_rate():
    """With no interest, a settled portfolio must grow by exactly rho/12."""
    p = sim.Params(loan_spread=0.0)
    d = _settled_deposits()
    out, cash = sim.roll_deposits(d, np.zeros(1), 0.04, p)
    assert out.sum() == pytest.approx(100.0 * (1 + 0.04 / 12))
    assert float(cash[0]) == pytest.approx(100.0 * 0.04 / 12)
    for j in range(4):      # new money enters flat, so the shape barely moves
        np.testing.assert_allclose(out[0, j] / out[0, j].sum(),
                                   d[0, j] / d[0, j].sum(), atol=1e-3)


def test_reinvested_interest_costs_equity_and_moves_no_cash():
    """B6: crediting interest raises what the bank owes without paying cash."""
    p = sim.Params(loan_spread=0.0)
    d = _settled_deposits()
    u = np.array([0.012])                                 # 1.2 % a year
    out, cash = sim.roll_deposits(d, u, 0.0, p)
    credited = 100.0 * (np.exp(0.001) - 1)
    assert out.sum() - 100.0 == pytest.approx(credited)
    assert float(cash[0]) == pytest.approx(0.0)


def test_literal_deposit_rule_runs_the_deposit_base_off():
    """The alternative reading of [E] (13), kept only for comparison."""
    p = sim.Params(loan_spread=0.0, deposit_cash_rule="literal")
    d = _settled_deposits()
    out, cash = sim.roll_deposits(d, np.zeros(1), 0.04, p)
    assert out.sum() < 100.0
    assert float(cash[0]) < 0.0


def test_negative_deposit_rate_floor_is_optional():
    lo = sim.deposit_rates(np.array([-0.004]), np.array([-0.004]),
                           sim.Params(loan_spread=0.0))
    assert lo[0][0] < 0                                   # B8: as printed
    floored = sim.deposit_rates(np.array([-0.004]), np.array([-0.004]),
                                sim.Params(loan_spread=0.0, deposit_rate_floor=0.0))
    assert floored[0][0] == 0.0


# ------------------------------------------------------------------- loans
def test_new_loans_match_the_growth_formula():
    p = sim.Params(loan_spread=0.01)
    RP = np.zeros((1, N)); RP[0, :120] = 10.0
    RE = np.zeros((1, N)); RE[0, :3] = 5.0
    y = np.full((1, N), 0.01)
    mix = np.full(len(p.mortgage_new_months), 1.0 / len(p.mortgage_new_months))
    _, _, _, _, _, repaid, granted = sim.roll_loans(RP, RE, RP * 0, RE * 0, y, p, mix)
    expected = (RP[0, 0] + 0.03 / 12 * RP.sum()) + (RE[0, 0] + 0.03 / 12 * RE.sum())
    assert float(granted[0]) == pytest.approx(expected)      # [E] (8)
    assert float(repaid[0]) == pytest.approx(15.0)


def test_loan_interest_is_charged_on_every_outstanding_tranche():
    p = sim.Params(loan_spread=0.01)
    RP = np.zeros((1, N)); RP[0, :12] = 100.0
    IRP = RP * 0.001
    y = np.zeros((1, N))
    mix = np.full(len(p.mortgage_new_months), 1.0 / len(p.mortgage_new_months))
    *_, interest, _, _ = sim.roll_loans(RP, RP * 0, IRP, RP * 0, y, p, mix)[:7]
    assert float(interest[0]) == pytest.approx(12 * 100.0 * 0.001)


def test_mortgage_mix_is_a_distribution_over_the_allowed_maturities():
    RP = np.zeros(N); RP[:100] = 1.0
    w = sim.fit_mortgage_mix(RP, tuple(range(24, 145, 12)))
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert len(w) == 11


# ----------------------------------------------------------- cash penalty
def test_cash_penalty_only_bites_on_excess_cash_at_negative_rates():
    p = sim.Params(loan_spread=0.0)
    MR = np.array([10.0])
    D_neg = np.array([np.exp(0.01 / 12)])        # 1m rate of -1 %
    D_pos = np.array([np.exp(-0.01 / 12)])
    assert float(sim.cash_penalty(np.array([1000.0]), MR, D_neg, p)[0]) == pytest.approx(
        (1000.0 - 300.0) * (1 - np.exp(-0.01 / 12)))
    assert float(sim.cash_penalty(np.array([1000.0]), MR, D_pos, p)[0]) == 0.0
    assert float(sim.cash_penalty(np.array([200.0]), MR, D_neg, p)[0]) == 0.0   # under 30xMR


def test_literal_cash_penalty_has_the_wrong_sign():
    """B7: as printed, negative rates would pay the bank."""
    p = sim.Params(loan_spread=0.0, cash_penalty_sign="literal")
    out = sim.cash_penalty(np.array([1000.0]), np.array([10.0]),
                           np.array([np.exp(0.01 / 12)]), p)
    assert float(out[0]) < 0


# ------------------------------------------------------------ annual step
def test_dividend_is_paid_in_years_one_to_four_only():
    assert [i for i in range(61) if cs.is_annual_step(i, 60)] == [12, 24, 36, 48]


def test_costs_grow_two_percent_a_year_on_personnel_only():
    x0 = type("X", (), {"monthly_material_cost": 6.0, "monthly_personnel_cost": 10.0})()
    p = sim.Params(loan_spread=0.0)
    assert sim.monthly_cost(0, x0, p) == pytest.approx(16.0)
    assert sim.monthly_cost(12, x0, p) == pytest.approx(6.0 + 10.2)
    assert sim.monthly_cost(11, x0, p) == pytest.approx(16.0)


# ------------------------------------------------------- end-to-end checks
@pytest.fixture(scope="module")
def setup():
    y0 = np.load(FIT)["y0"]
    x0 = bs.build_initial_state(y0)
    kl = sim.calibrate_loan_spread(x0, y0, 303.354)
    sc = yc.load_scenarios(FIT, n_scenarios=200, seed=20221230, variant="sqrt")
    return x0, y0, sim.Params(loan_spread=kl), sc


@needs_fit
def test_loan_spread_calibration_hits_the_banks_interest_income(setup):
    x0, y0, p, _ = setup
    R = np.asarray(x0.RP) + np.asarray(x0.RE)
    assert 12 * float(R @ sim.loan_rate(y0[:N], p.loan_spread)) == pytest.approx(
        303.354, abs=1e-6)
    assert 0.0 < p.loan_spread < 0.03            # a plausible retail margin


@needs_fit
def test_step_zero_reproduces_layer_two(setup):
    """The simulator must start exactly on the balance sheet Layer 2 froze."""
    x0, y0, p, sc = setup
    out = sim.simulate(x0, sc.yields[:, :2, :], p, sim.NoAction())
    first = out["history"][0]
    ref = bs.report(x0, y0)
    for k in ("C", "B", "RP", "RE", "SD", "SF", "K", "E"):
        assert float(np.median(first[k])) == pytest.approx(ref["values_mCHF"][k], abs=1e-6)
    assert float(np.median(first["IRS"])) == pytest.approx(ref["IRS_100bp"], abs=1e-6)


@needs_fit
def test_balance_sheet_balances_at_every_step(setup):
    x0, y0, p, sc = setup
    out = sim.simulate(x0, sc.yields, p, sim.RollOver())
    for row in out["history"]:
        assets = row["C"] + row["B"] + row["RP"] + row["RE"]
        np.testing.assert_allclose(assets, row["A"], atol=1e-8,
                                   err_msg=f"assets, step {row['step']}")
        np.testing.assert_allclose(row["A"] - (row["SD"] + row["SF"] + row["K"]),
                                   row["E"], atol=1e-8,
                                   err_msg=f"equity, step {row['step']}")


@needs_fit
def test_nothing_is_created_from_nothing_with_no_action(setup):
    """With no decisions the bond books can only shrink, never grow."""
    x0, y0, p, sc = setup
    out = sim.simulate(x0, sc.yields, p, sim.NoAction())
    med = sim.summarise(out)["median"]
    assert all(med["K"][i] >= med["K"][i + 1] - 1e-9 for i in range(60))
    assert all(med["B"][i] >= med["B"][i + 1] - 1e-9 for i in range(60))


@needs_fit
def test_borrowing_raises_cash_one_for_one_before_spreads(setup):
    """Issuing a par bond must bring in exactly its face amount."""
    x0, y0, p, sc = setup

    class BorrowOnce:
        def __call__(self, step, state, ctx):
            nB = len(ctx.params.invest_maturities)
            nK = len(ctx.params.finance_maturities)
            S = state.C.shape[0]
            aK = np.zeros((S, nK))
            if step == 1:
                aK[:, 5] = 100.0
            return np.zeros((S, nB)), aK

    a = sim.simulate(x0, sc.yields[:, :3, :], p, sim.NoAction())
    b = sim.simulate(x0, sc.yields[:, :3, :], p, BorrowOnce())
    assert float(np.median(b["history"][1]["C"] - a["history"][1]["C"])) == pytest.approx(
        100.0, abs=1e-9)
    # the bank pays the 15 bp spread, so equity falls a little
    assert float(np.median(b["history"][1]["E"] - a["history"][1]["E"])) < 0


@needs_fit
def test_deposits_and_loans_grow_at_the_stated_annual_rates(setup):
    x0, y0, p, sc = setup
    med = sim.summarise(sim.simulate(x0, sc.yields, p, sim.NoAction()))["median"]
    loans = (med["RP"][12] + med["RE"][12]) / (med["RP"][0] + med["RE"][0])
    assert loans == pytest.approx(1.0 + p.loan_growth, abs=0.015)
    assert med["SF"][12] / med["SF"][0] == pytest.approx(1.0 + p.term_growth, abs=0.01)
