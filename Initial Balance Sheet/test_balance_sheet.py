"""Tests for Layer 2 (balance_sheet.py).

    python -m pytest -q test_balance_sheet.py

Needs output/fit_sqrt.npz (from run_fit.py) for the starting curve; the
tests that use it are skipped without it.
"""
import os
import numpy as np
import pytest

import balance_sheet as bs

FIT = os.environ.get("FIT_NPZ", "output/fit_sqrt.npz")
needs_fit = pytest.mark.skipif(not os.path.isfile(FIT), reason="run run_fit.py first")


# ------------------------------------------------------- the transcribed data
def test_balance_sheet_balances():
    a = sum(v for s, _, v in bs.BALANCE_SHEET.values() if s == "A")
    le = sum(v for s, _, v in bs.BALANCE_SHEET.values() if s in "LE")
    assert le == bs.TOTAL_ASSETS_KCHF
    assert abs(a - bs.TOTAL_ASSETS_KCHF) <= 1          # GB rounds each line


def test_maturity_rows_match_balance_sheet():
    for line, row in bs.MATURITY_TABLE.items():
        assert abs(sum(row) - bs.BALANCE_SHEET[line][2]) <= 1, line


def test_bond_note_totals_and_buckets():
    assert sum(a for _, _, a in bs.BOND_ISSUES) == 2_560_000
    assert sum(bs.PFANDBRIEF_BY_YEAR.values()) == 5_613_000
    # bonds + Pfandbrief, bucketed by our month rule, reproduce note 21 exactly
    v = bs._bond_principal_vector() * 1000
    got = (v[:3].sum(), v[3:12].sum(), v[12:60].sum(), v[60:].sum())
    np.testing.assert_allclose(got, bs.MATURITY_TABLE["bonds_pfandbrief"][2:], atol=1e-6)


def test_every_line_mapped_once():
    m = bs.mapping_table(bs.Options())
    assert set(m) == set(bs.BALANCE_SHEET)


# ------------------------------------------------------- construction rules
def test_nmd_mix_hits_disclosed_term():
    w = bs.nmd_weights()
    x = sum(w[m] * bs._steady_shape(m) for m in bs.REF_MATURITIES)
    assert bs.mean_maturity(x) == pytest.approx(bs.NMD_TARGET_Y, abs=1e-12)
    assert bs.NMD_TARGET_Y == pytest.approx(2.0535, abs=1e-4)
    assert w[120] == pytest.approx(0.596, abs=0.001)   # settled shape, not flat


def test_fixed_mortgage_end_month_is_calibrated():
    M, how = bs.max_months(bs.Options())["fixed_mortgages"]
    assert (M, how) == (105, "calibrated")


@needs_fit
def test_nominal_is_conserved():
    """Spreading buckets over months must not create or lose principal."""
    y0 = np.load(FIT)["y0"]
    x = bs.build_initial_state(y0)
    book = bs.book_by_slot(x.options)
    for s in ("RP", "RE", "SD", "SF"):
        total = getattr(x, s).sum()
        assert total == pytest.approx(book[s], abs=0.002), s   # GB 1 kCHF rounding


@needs_fit
def test_investments_valued_at_fair_value():
    y0 = np.load(FIT)["y0"]
    _, _, parts = bs.build_B(y0, bs.Options())
    D = bs.discount_factors(y0)[:bs.N]
    assert D @ parts["fin_investments"] == pytest.approx(1751.966, abs=1e-9)


@needs_fit
def test_money_market_items_valued_at_book():
    y0 = np.load(FIT)["y0"]
    _, parts = bs.build_K(y0, bs.Options())
    D = bs.discount_factors(y0)[:bs.N]
    assert D @ parts["repo"] == pytest.approx(1522.0, abs=1e-9)
    assert D @ parts["due_to_banks"] == pytest.approx(1691.865, abs=0.002)


@needs_fit
def test_folding_keeps_present_value():
    y0 = np.load(FIT)["y0"]
    D = bs.discount_factors(y0)
    long = np.zeros(240)
    long[[50, 200, 239]] = 1.0
    assert D[:180] @ bs._fold(long, D) == pytest.approx(D @ long, abs=1e-14)


@needs_fit
def test_state_is_frozen():
    x = bs.build_initial_state(np.load(FIT)["y0"])
    with pytest.raises(ValueError):
        x.RP[0] = 1.0
    with pytest.raises(Exception):
        x.C = 0.0


@needs_fit
def test_equity_bridge_closes():
    y0 = np.load(FIT)["y0"]
    r = bs.report(bs.build_initial_state(y0), y0)
    assert r["equity_bridge_mCHF"]["= model E0"] == pytest.approx(r["values_mCHF"]["E"], abs=0.01)


# ------------------------------------------------------- pinned results (layer2_report.md)
@needs_fit
def test_pinned_numbers():
    y0 = np.load(FIT)["y0"]
    r = bs.report(bs.build_initial_state(y0), y0)
    v, k = r["values_mCHF"], r["constraints_t0"]
    assert v["E"] == pytest.approx(2037.3, abs=0.5)
    assert k["E_over_RWA"] == pytest.approx(0.2004, abs=0.0005)
    assert k["LCR"] == pytest.approx(1.9044, abs=0.0005)
    assert k["NSFR"] == pytest.approx(1.5444, abs=0.0005)
    assert k["CMR"] == pytest.approx(67.06, abs=0.05)
    assert r["IRS_100bp"] == pytest.approx(-0.0644, abs=0.0005)
    assert r["dE_mCHF"][150] == pytest.approx(-195.7, abs=0.5)
    assert r["investment_coupon"] == pytest.approx(0.00011, abs=0.00001)


@needs_fit
def test_pinned_numbers_on_the_paper_start_date():
    """The settled setup: Y0 from 15.07.2022 (output_paper/fit_sqrt.npz)."""
    path = "output_paper/fit_sqrt.npz"
    if not os.path.isfile(path):
        pytest.skip("run run_fit.py --valuation-date 2022-07-15 --model paper")
    y0 = np.load(path)["y0"]
    r = bs.report(bs.build_initial_state(y0), y0)
    v, k = r["values_mCHF"], r["constraints_t0"]
    assert v["E"] == pytest.approx(2216.7, abs=0.5)
    assert k["E_over_RWA"] == pytest.approx(0.2107, abs=0.0005)
    assert k["LCR"] == pytest.approx(1.8977, abs=0.0005)
    assert k["NSFR"] == pytest.approx(1.5332, abs=0.0005)
    assert k["CMR"] == pytest.approx(13.14, abs=0.05)
    assert r["IRS_100bp"] == pytest.approx(-0.0619, abs=0.0005)