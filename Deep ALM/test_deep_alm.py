"""Tests for Layer 5 (deep_alm.py).

The important ones check that the differentiable simulator is the same bank as
the numpy reference of Layer 3.  Everything else is only allowed to be wrong in
ways the reference would be wrong too.

    python -m pytest -q test_deep_alm.py
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import balance_sheet as bs                                        # noqa: E402
import deep_alm as da                                             # noqa: E402
import loss as ls                                                 # noqa: E402
import simulator as sim                                           # noqa: E402
import yieldcurve as yc                                           # noqa: E402

FIT = os.environ.get("FIT_NPZ", "output_paper/fit_sqrt.npz")
needs_fit = pytest.mark.skipif(not os.path.isfile(FIT), reason="run run_fit.py first")
S = 64


@pytest.fixture(scope="module")
def setup():
    y0 = np.load(FIT)["y0"]
    x0 = bs.build_initial_state(y0)
    p = sim.Params(loan_spread=sim.calibrate_loan_spread(x0, y0, 303.354))
    sc = yc.load_scenarios(FIT, n_scenarios=S, seed=4242, variant="sqrt")
    mix = sim.fit_mortgage_mix(x0.RP, p.mortgage_new_months)
    return x0, sc.yields, p, mix, da.prepare(x0, sc.yields, p, mix)


def _path(pre, strategy, theta=None):
    b = pre.take(np.arange(pre.S))
    one = jnp.ones(pre.S)
    roll = da.make_rollout(pre, strategy)
    out = roll(strategy.init() if theta is None else theta,
               b, 0.04 * one, 3.5 * one)
    # path comes back stacked (H+1, S); hand back a row per month
    out["rows"] = [{k: v[t] for k, v in out["path"].items()}
                   for t in range(pre.H + 1)]
    return out


# ------------------------------------------------ the two simulators agree
@needs_fit
@pytest.mark.parametrize("name", ["none", "rollover"])
def test_the_differentiable_bank_is_the_same_bank(setup, name):
    x0, yields, p, mix, pre = setup
    npol, jpol = {"none": (sim.NoAction(), da.Zero(p)),
                  "rollover": (sim.RollOver(), da.RollOver(p))}[name]
    ref = sim.simulate(x0, yields, p, npol, mortgage_mix=mix)["history"]
    got = _path(pre, jpol)["rows"]
    assert len(got) == len(ref)
    for r, g in zip(ref, got):
        # tolerances are float32 rounding: JAX runs in single precision,
        # the reference in double.  0.02 mCHF is 20 thousand francs on a
        # 27 billion balance sheet.
        for k in ("C", "E"):
            np.testing.assert_allclose(np.asarray(g[k]), r[k], rtol=1e-4,
                                       atol=0.02,
                                       err_msg=f"{k} at step {r['step']}")
        for k in da.CON_NAMES:
            np.testing.assert_allclose(np.asarray(g[k]), r[k], rtol=1e-3,
                                       atol=2e-5,
                                       err_msg=f"{k} at step {r['step']}")


@needs_fit
def test_the_penalty_matches_layer_4(setup):
    x0, yields, p, mix, pre = setup
    ref = sim.simulate(x0, yields, p, sim.RollOver(), mortgage_mix=mix)
    want = ls.accumulated_penalty(ref["history"], pre.H)
    got = np.asarray(_path(pre, da.RollOver(p))["penalty"])
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-5)


@needs_fit
def test_the_prepared_books_are_the_simulator_s_own(setup):
    x0, yields, p, mix, pre = setup
    ref = sim.simulate(x0, yields, p, sim.NoAction(), mortgage_mix=mix)["history"]
    for t, row in enumerate(ref):
        for key, col in (("RP", pre.V_RP), ("RE", pre.V_RE),
                         ("SD", pre.V_SD), ("SF", pre.V_SF)):
            np.testing.assert_allclose(col[:, t], row[key], rtol=1e-6)
    assert pre.E0 == pytest.approx(float(ref[0]["E"][0]), abs=1e-9)


@needs_fit
def test_par_bonds_match_the_numpy_version(setup):
    _, _, p, _, pre = setup
    D = jnp.asarray(pre.disc[:, 7, :])
    pat = da.bond_patterns(p.invest_maturities, p.invest_spread)
    a = np.zeros((pre.S, len(p.invest_maturities)), dtype=np.float32)
    a[:, 3] = 2.0
    got = np.asarray(da.add_bonds(jnp.asarray(a), D, pat))
    want = 2.0 * sim.par_bonds(pre.disc[:, 7, :].astype(float),
                               p.invest_maturities, p.invest_spread)[:, 3, :]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


# -------------------------------------------------------------- strategies
@needs_fit
def test_untouched_benchmark_rolls_the_maturing_amount_over(setup):
    """[E] (43) with theta = 0: invest and borrow exactly what matured."""
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=True, uniform=True)
    pth = _path(pre, bm)["path"]
    for t in range(pre.H):
        np.testing.assert_allclose(np.asarray(pth["aB"][t]),
                                   np.asarray(pth["matured_B"][t]),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(pth["aK"][t]),
                                   np.asarray(pth["matured_K"][t]),
                                   rtol=1e-5, atol=1e-6)
    # and nothing at all in the terminal month: no decision at t = H
    assert float(np.abs(np.asarray(pth["aB"][pre.H])).max()) == 0.0
    assert float(np.abs(np.asarray(pth["aK"][pre.H])).max()) == 0.0


@needs_fit
def test_the_three_benchmarks_have_the_paper_s_shapes(setup):
    _, _, p, _, pre = setup
    e = da.Benchmark(p, pre.H, shared=True, uniform=True)
    c = da.Benchmark(p, pre.H, shared=True, uniform=False)
    d = da.Benchmark(p, pre.H, shared=False, uniform=False)
    assert da.n_params(e.init()) == 2                      # [E] 3.1.1: BM_E
    assert da.n_params(c.init()) == 2 + 13 + 16            # BM_C
    assert da.n_params(d.init()) == pre.H * (2 + 13 + 16)   # BM_D


def _books(p, pre, n=8):
    one = jnp.ones(n)
    return {"step": jnp.int32(3), "tfrac": 0.05 * one, "C": 6000.0 * one,
            "B": jnp.asarray(pre.B0) * jnp.ones((n, 1)),
            "K": jnp.asarray(pre.K0) * jnp.ones((n, 1)),
            "matured_B": 10.0 * one, "matured_K": 20.0 * one,
            "R_vec": jnp.asarray(pre.R_vec[:n, 3, :], jnp.float32),
            "S_vec": jnp.asarray(pre.S_vec[:n, 3, :], jnp.float32),
            "pc": jnp.asarray(pre.pc[:n, 3, :]),
            "last_con": {k: 1.2 * one for k in da.CON_NAMES},
            "last_eyr": 20.0 * one, "mu": 0.04 * one, "lam": 3.5 * one,
            "VB": 2000.0 * one, "VK": 11000.0 * one, "A": 30000.0 * one,
            "E": 2200.0 * one, "V_S": 19000.0 * one}


@needs_fit
def test_feature_vector_is_145_long_like_table_2(setup):
    """[E] Table 2 gives d = 145: 4 x 32 encodings + 3 curve + 5 ratios
    + 6 constraints + time + mu + lambda."""
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=False, uniform=False)
    net = da.DecisionNet(p, pre.H, bm, bm.init(), a0=pre.A0)
    assert net.feature_dim == 145
    theta = net.init(jax.random.PRNGKey(0))
    x = net.features(theta, _books(p, pre))
    assert x.shape == (8, 145)
    assert bool(jnp.isfinite(x).all())


@needs_fit
def test_the_network_starts_at_its_benchmark(setup):
    """[E] (46): the net learns a deviation, so an untrained one with small
    output weights must sit close to BM_D."""
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=False, uniform=False)
    bt = bm.init()
    net = da.DecisionNet(p, pre.H, bm, bt, a0=pre.A0)
    base = np.asarray(_path(pre, bm, bt)["ET"])
    got = np.asarray(_path(pre, net, net.init(jax.random.PRNGKey(0)))["ET"])
    assert abs(np.median(got) / np.median(base) - 1.0) < 0.05


@needs_fit
def test_gradients_reach_every_parameter(setup):
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=True, uniform=False)
    theta = bm.init()
    roll = da.make_rollout(pre, bm)
    one = jnp.ones(pre.S)
    b = pre.take(np.arange(pre.S))
    g = jax.grad(lambda th: da.path_loss(roll(th, b, 0.04 * one, 3.5 * one),
                                        0.04 * one, 3.5 * one,
                                        pre.H / 12.0, pre.E0))(theta)
    for k, v in g.items():
        assert float(jnp.abs(v).sum()) > 0, k


@needs_fit
def test_the_frozen_baseline_takes_no_gradient(setup):
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=False, uniform=False)
    bt = bm.init()
    net = da.DecisionNet(p, pre.H, bm, bt, a0=pre.A0)
    theta = net.init(jax.random.PRNGKey(0))
    roll = da.make_rollout(pre, net)
    one = jnp.ones(pre.S)
    b = pre.take(np.arange(pre.S))
    g = jax.grad(lambda th: da.path_loss(roll(th, b, 0.04 * one, 3.5 * one),
                                        0.04 * one, 3.5 * one,
                                        pre.H / 12.0, pre.E0))(theta)
    assert set(g) == set(theta)                 # and nothing from the baseline
    assert float(jnp.abs(g["scl_B"]).sum()) > 0
    assert float(jnp.abs(g["encB_1"]).sum()) > 0   # encoders do train, [E] Fig. 5


@needs_fit
def test_decisions_are_never_negative(setup):
    """[E] 2.1.4: long-only, no short sales."""
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=False, uniform=False)
    bt = jax.tree.map(lambda a: a - 500.0, bm.init())
    net = da.DecisionNet(p, pre.H, bm, bt, a0=pre.A0)
    for scale in (1.0, 20.0, 200.0):
        theta = jax.tree.map(lambda a: a * scale,
                             net.init(jax.random.PRNGKey(3)))
        aB, aK = net.apply(theta, _books(p, pre))
        assert float(aB.min()) >= 0.0 and float(aK.min()) >= 0.0
    out = _path(pre, net, net.init(jax.random.PRNGKey(3)))
    assert float(np.asarray(out["path"]["aB"]).min()) >= 0.0
    assert float(np.asarray(out["path"]["aK"]).min()) >= 0.0


@needs_fit
def test_training_lowers_the_loss(setup):
    _, _, p, _, pre = setup
    bm = da.Benchmark(p, pre.H, shared=True, uniform=False)
    start = bm.init()
    theta, hist = da.train(pre, bm, start, epochs=8, batch_size=32,
                           sample_params=False, log=lambda *a: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    before = da.evaluate(pre, bm, start, batch_size=32)
    after = da.evaluate(pre, bm, theta, batch_size=32)
    assert after["total"] < before["total"]


@needs_fit
def test_evaluate_reports_the_table_3_rows(setup):
    _, _, p, _, pre = setup
    z = da.Zero(p)
    r = da.evaluate(pre, z, z.init(), batch_size=32)
    for k in ("total", "target", "penalty", "utility", "ER_mean", "ER_sd",
              "ER_skew", "annual_return", "clean"):
        assert k in r and np.isfinite(r[k])
    ref = sim.summarise(sim.simulate(bs.build_initial_state(np.load(FIT)["y0"]),
                                     setup[1], p, sim.NoAction(),
                                     mortgage_mix=setup[3]))
    assert r["ER_mean"] == pytest.approx(
        float(np.mean(np.asarray(ref["median"]["E"][-1]) / pre.E0)), rel=0.05)
