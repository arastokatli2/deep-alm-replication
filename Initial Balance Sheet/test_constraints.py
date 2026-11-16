"""Tests for constraints.py (the settled B1-B5 formulas)."""
import numpy as np
import pytest
import constraints as cs

V = dict(C=6196.8, B=1922.1, RP=20688.4, RE=2735.1, SD=15113.1, SF=3355.9, K=11023.4)
V["E"] = V["C"] + V["B"] + V["RP"] + V["RE"] - V["SD"] - V["SF"] - V["K"]


def test_b1_nsfr_uses_095_on_non_maturing_deposits():
    lower = dict(V, SD=V["SD"] - 100.0)
    rsf = 0.12 * V["B"] + 0.71 * (V["RP"] + V["RE"])
    assert (cs.nsfr(V) - cs.nsfr(lower)) * rsf == pytest.approx(95.0)


def test_b2_lcr_term_deposits_enter_with_013():
    more = dict(V, SF=V["SF"] + 100.0)
    no30 = lambda v: 0.71 * v["C"] / cs.lcr(v) + 0.89 * v["B"] / cs.lcr(v)
    assert no30(more) - no30(V) == pytest.approx(13.0)


def test_b3_eyr_reproduces_journal_table4_scale():
    # [E] Table 4: median EYR 21.17 <=> equity growth of about 133 mCHF a year
    assert cs.eyr(E=2133.0, E_prev=2000.0) == pytest.approx(21.17, abs=0.01)
    assert cs.eyr(2006.0, 2000.0) == 0.0


def test_b4_irs_signed_and_two_sided_limit():
    assert cs.irs(2000.0, 1860.0) == pytest.approx(-0.07)
    assert cs.violation("IRS", -0.09) == pytest.approx(0.005)
    assert cs.violation("IRS", 0.08) == 0.0


def test_e9_lower_bounds():
    assert cs.violation("LCR", 1.00) == pytest.approx(0.05)
    assert cs.violation("LCR", 1.20) == 0.0


def test_e2_four_annual_steps_in_five_years():
    assert [i for i in range(60) if cs.is_annual_step(i)] == [12, 24, 36, 48]


def test_cmr_relief_only_when_1m_rate_non_negative():
    assert cs.cmr(V, -0.0022) * 5 == pytest.approx(cs.cmr(V, 0.0095))
