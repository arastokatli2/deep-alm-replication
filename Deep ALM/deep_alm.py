"""Layer 5 - Deep ALM: a differentiable bank, the benchmarks, and the network.

[E] 3 / [M] 4.  Three things live here:

1. `prepare()` lifts every decision-independent quantity out of one NoAction run
   of the Layer 3 simulator.  [E] 2.3 and 3.2.4 do the same, "outside of the
   training loop".  Because the numbers come from the numpy simulator itself
   rather than being re-derived, the two simulators cannot drift apart.
2. `make_rollout()` builds the decision-dependent part -- cash, the two bond
   books, and everything computed from them -- in JAX, so gradients flow
   through all 60 months.  `test_deep_alm.py` checks it against the numpy
   reference month by month.
3. The strategies: `Benchmark` ([E] 3.1: BM_E / BM_C / BM_D) and `DecisionNet`
   ([E] 3.2.2), both trained by `train()`.

JAX rather than PyTorch only because this machine has no CUDA libraries and the
PyPI torch wheel will not load without them; the maths is the paper's.

Everything is in CHF millions, as in Layers 2-4.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import balance_sheet as bs
import constraints as cs
import loss as ls
import simulator as sim

N = bs.N
BATCH_KEYS = ("disc", "disc_shift", "y1m", "CF", "MR", "D1m_prev",
              "V_RP", "V_RE", "V_SD", "V_SF", "V_R_shift", "V_S_shift",
              "R_vec", "S_vec", "pc")
SHIFT = np.exp(-np.arange(1, N + 1) / 12.0 * cs.IRS_SHIFT).astype(np.float32)

# Decision scales are learnt in units of mCHF 100.  [E] reaches volumes in the
# hundreds of millions with a learning rate of 5e-3 only because it takes
# ~125,000 optimiser steps (40,000 paths, batch 32, 100 epochs); we take about
# 2,000, so without this the scale parameters could never leave zero.  It is a
# reparametrisation, not a change to the model.
SCALE_UNIT = 100.0


# ===================================================== decision-independent

@dataclass
class Prepared:
    """One frozen bundle of scenarios, with the policy-free part already run."""
    disc: np.ndarray          # (S, H+1, N) float32
    y1m: np.ndarray           # (S, H+1)
    CF: np.ndarray            # cash flow from loans, deposits and costs
    MR: np.ndarray            # minimum reserve, previous month
    D1m_prev: np.ndarray
    V_RP: np.ndarray
    V_RE: np.ndarray
    V_SD: np.ndarray
    V_SF: np.ndarray
    V_R_shift: np.ndarray     # loans under the +100 bp curve
    V_S_shift: np.ndarray     # deposits likewise
    R_vec: np.ndarray         # (S, H+1, N) float16, features only
    S_vec: np.ndarray
    pc: np.ndarray            # (S, H+1, 3) the curve in 3 dims, [E] 3.2.1.1
    C0: float
    B0: np.ndarray            # (N,)
    K0: np.ndarray
    E0: float                 # equity of the starting balance sheet
    A0: float
    params: sim.Params
    pca: np.ndarray = None
    pca_mean: np.ndarray = None

    @property
    def H(self) -> int:
        return self.disc.shape[1] - 1

    @property
    def S(self) -> int:
        return self.disc.shape[0]

    def take(self, idx) -> dict:
        """One minibatch, as JAX arrays."""
        def get(k):
            a = getattr(self, k)
            return jnp.asarray(np.ascontiguousarray(a[idx]), dtype=jnp.float32)
        out = {k: get(k) for k in BATCH_KEYS if k != "disc_shift"}
        out["disc_shift"] = out["disc"] * jnp.asarray(SHIFT)
        return out


def prepare(x0: bs.InitialState, yields: np.ndarray, p: sim.Params,
            mortgage_mix: np.ndarray | None = None,
            pca_from: "Prepared | None" = None) -> Prepared:
    """Run the policy-free part of Layer 3 once and keep the result."""
    out = sim.simulate(x0, yields, p, sim.NoAction(),
                       mortgage_mix=mortgage_mix, record_books=True)
    h = out["history"]
    col = lambda k: np.stack([row[k] for row in h], axis=1).astype(np.float32)
    y = np.asarray(yields, dtype=np.float32)

    pre = Prepared(
        disc=np.exp(-np.arange(1, N + 1)[None, None, :] / 12.0 * y).astype(np.float32),
        y1m=y[:, :, 0].copy(),
        CF=col("CF"), MR=col("MR"), D1m_prev=col("D1m_prev"),
        V_RP=col("RP"), V_RE=col("RE"), V_SD=col("SD"), V_SF=col("SF"),
        V_R_shift=col("V_RP_shift") + col("V_RE_shift"),
        V_S_shift=col("V_SD_shift") + col("V_SF_shift"),
        R_vec=np.stack([row["R_vec"] for row in h], axis=1).astype(np.float16),
        S_vec=np.stack([row["S_vec"] for row in h], axis=1).astype(np.float16),
        pc=None,
        C0=float(x0.C), B0=np.asarray(x0.B, dtype=np.float32),
        K0=np.asarray(x0.K, dtype=np.float32),
        E0=float(h[0]["E"][0]), A0=float(h[0]["A"][0]), params=p)
    if pca_from is not None:
        pre.pca, pre.pca_mean = pca_from.pca, pca_from.pca_mean
    else:
        pre.pca, pre.pca_mean = _fit_pca(np.asarray(yields, dtype=np.float64))
    pre.pc = ((y - pre.pca_mean) @ pre.pca).astype(np.float32)
    return pre


def _fit_pca(yields: np.ndarray, k: int = 3, cap: int = 20_000):
    """[E] 3.2.1.1: project the 180-dim curve into R^3, fitted before training."""
    flat = yields.reshape(-1, yields.shape[-1])
    if flat.shape[0] > cap:
        flat = flat[np.linspace(0, flat.shape[0] - 1, cap).astype(int)]
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return vt[:k].T.astype(np.float32), mean.astype(np.float32)


# ======================================================== par bonds, in JAX

def bond_patterns(maturities, spread: float):
    """Split [E] (6)-(7) into scenario-free matrices.

    One unit of the bond maturing in m months pays
        Z = principal + alpha * coupon_dates + spread * periods,
    and only alpha depends on the curve.  So a whole decision a (B, n) becomes
    a @ base + (a * alpha) @ cdates -- two matmuls, never a (B, n, N) tensor.
    """
    n = len(maturities)
    base = np.zeros((n, N), dtype=np.float32)
    cdates = np.zeros((n, N), dtype=np.float32)
    for j, m in enumerate(maturities):
        cm = sim._coupon_months(m)
        idx = np.array(cm) - 1
        base[j, idx] = spread * (np.diff([0] + cm) / 12.0)
        base[j, m - 1] += 1.0
        cdates[j, idx] = 1.0
    return (jnp.asarray(base), jnp.asarray(cdates),
            jnp.asarray(np.array(maturities) - 1))


def add_bonds(a, D, pat):
    """The cash-flow vector that a decision `a` adds to a bond book."""
    base, cdates, mat_idx = pat
    alpha = (1.0 - D[:, mat_idx]) / (D @ cdates.T)       # [E] (7), at par
    return a @ base + (a * alpha) @ cdates


# ================================================================= rollout

def _shift_up(x):
    return jnp.concatenate([x[:, 1:], jnp.zeros_like(x[:, :1])], axis=1)


CON_NAMES = ("LCR", "NSFR", "CMR", "E_over_RWA", "IRS")


def make_rollout(pre: Prepared, strategy):
    """Build the differentiable 60-month walk for one strategy.

    Mirrors simulator.simulate() step for step: roll forward, decide,
    restructure, record, then the annual step.

    The months go through one `lax.scan` rather than an unrolled loop, so the
    gradient graph is one month deep whatever the horizon.  Unrolling 60 months
    and reverse-differentiating the result exhausts the compiler's memory on a
    small machine.  The three places where months differ -- no roll-forward at
    t = 0, no decision at t = H, a dividend every twelfth month -- become masks
    carried alongside the month's data instead of Python branches.
    """
    p, H = pre.params, pre.H
    patB = bond_patterns(p.invest_maturities, p.invest_spread)
    patK = bond_patterns(p.finance_maturities, p.finance_spread)
    B0, K0 = jnp.asarray(pre.B0), jnp.asarray(pre.K0)

    steps = np.arange(H + 1)
    flags = {
        "step": jnp.asarray(steps),
        "is_roll": jnp.asarray((steps > 0).astype(np.float32)),
        "is_act": jnp.asarray((steps < H).astype(np.float32)),
        "is_annual": jnp.asarray(np.array(
            [cs.is_annual_step(int(t), H) for t in steps], dtype=np.float32)),
        "tfrac": jnp.asarray((steps / H).astype(np.float32)),
    }

    def ratios(m, C, Bk, Kk):
        D, Ds = m["disc"], m["disc_shift"]
        VB, VK = (D * Bk).sum(1), (D * Kk).sum(1)
        V_SD, V_SF, V_RP, V_RE = m["V_SD"], m["V_SF"], m["V_RP"], m["V_RE"]
        A = C + VB + V_RP + V_RE
        E = A - (V_SD + V_SF + VK)
        E_sh = (C + (Ds * Bk).sum(1) + m["V_R_shift"] - m["V_S_shift"]
                - (Ds * Kk).sum(1))
        relief = jnp.where(m["y1m"] >= 0.0, 0.8, 0.0)
        con = {
            "CMR": C / (0.025 * (V_SD + V_SF) * (1.0 - relief)),
            "LCR": (0.71 * C + 0.89 * VB)
                   / (0.176 * V_SD + 0.130 * V_SF + 0.010 * VK),
            "NSFR": (0.95 * V_SD + 0.90 * V_SF + 0.60 * VK + E)
                    / (0.12 * VB + 0.71 * (V_RP + V_RE)),
            "E_over_RWA": E / (0.10 * VB + 0.35 * V_RP + 1.00 * V_RE),
            "IRS": (E_sh - E) / E,
        }
        return {"VB": VB, "VK": VK, "A": A, "E": E, "V_S": V_SD + V_SF}, con

    def rollout(theta, b, mu, lam):
        n = b["CF"].shape[0]
        months = {k: jnp.moveaxis(b[k], 1, 0) for k in BATCH_KEYS}
        months.update({k: jnp.broadcast_to(v[:, None], (H + 1, n))
                       if v.ndim == 1 and v.shape[0] == H + 1 else v
                       for k, v in flags.items()})
        months["step"] = flags["step"]

        zero = jnp.zeros((n,), dtype=jnp.float32)
        val0, con0 = ratios({k: months[k][0] for k in BATCH_KEYS},
                            jnp.full((n,), pre.C0), B0 * jnp.ones((n, 1)),
                            K0 * jnp.ones((n, 1)))

        def month(carry, m):
            C, Bk, Kk, pen, con, last_eyr, E_prev = carry
            roll, act, ann = m["is_roll"], m["is_act"], m["is_annual"]

            # ------------------------------------------------- roll forward
            excess = jnp.maximum(C - p.reserve_exemption * m["MR"], 0.0)
            factor = 1.0 - jnp.minimum(1.0 / m["D1m_prev"], 1.0)
            matured_B, matured_K = Bk[:, 0] * roll, Kk[:, 0] * roll
            C = C + roll * (m["CF"] + Bk[:, 0] - Kk[:, 0] - excess * factor)
            Bk = jnp.where(roll[:, None] > 0, _shift_up(Bk), Bk)
            Kk = jnp.where(roll[:, None] > 0, _shift_up(Kk), Kk)

            # ------------------------------------------------------- decide
            pre_val, _ = ratios(m, C, Bk, Kk)
            books = {"step": m["step"], "tfrac": m["tfrac"], "C": C, "B": Bk,
                     "K": Kk, "matured_B": matured_B, "matured_K": matured_K,
                     "R_vec": m["R_vec"], "S_vec": m["S_vec"], "pc": m["pc"],
                     "last_con": con, "last_eyr": last_eyr,
                     "mu": mu, "lam": lam, **pre_val}
            aB, aK = strategy.apply(theta, books)
            aB, aK = aB * act[:, None], aK * act[:, None]
            Bk = Bk + add_bonds(aB, m["disc"], patB)
            Kk = Kk + add_bonds(aK, m["disc"], patK)
            C = C - aB.sum(1) + aK.sum(1)

            # ---------------------------------------- record, then pay out
            val, con = ratios(m, C, Bk, Kk)
            for name in CON_NAMES:
                v = con[name]
                gap = (jnp.abs(v) - cs.BOUNDS[name] if name in cs.UPPER_BOUNDED
                       else cs.BOUNDS[name] - v)
                pen = pen + act * ls.WEIGHTS[name] * (
                    (1.0 + jnp.maximum(gap, 0.0)) ** 2 - 1.0)

            # `con` is what next month sees as x_{t-1} ([E] 3.2.1.4), and the
            # books are read before the dividend leaves ([E] Fig. 10's note).
            eyr = (val["E"] - E_prev - cs.EYR_BUFFER) / cs.EYR_DENOMINATOR
            out = {"C": C, "E": val["E"], "aB": aB.sum(1), "aK": aK.sum(1),
                   "matured_B": matured_B, "matured_K": matured_K,
                   "EYR": jnp.where(ann > 0, eyr, jnp.inf), **con}
            pen = pen + ann * ls.WEIGHTS["EYR"] * (
                (1.0 + jnp.maximum(-eyr, 0.0)) ** 2 - 1.0)
            last_eyr = jax.lax.stop_gradient(
                jnp.where(ann > 0, eyr, last_eyr))
            delta = ann * p.dividend_share * jnp.maximum(val["E"] - E_prev, 0.0)
            C = C - delta
            E_prev = jax.lax.stop_gradient(
                jnp.where(ann > 0, val["E"] - delta, E_prev))
            return (C, Bk, Kk, pen, con, last_eyr, E_prev), out

        init = (jnp.full((n,), pre.C0, dtype=jnp.float32),
                B0 * jnp.ones((n, 1)), K0 * jnp.ones((n, 1)), zero, con0,
                zero, jax.lax.stop_gradient(val0["E"]))
        (C, Bk, Kk, pen, _, _, _), ys = jax.lax.scan(month, init, months)
        return {"ET": ys["E"][-1], "penalty": pen, "path": ys}

    return rollout


# ============================================================== strategies

def _glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


class Benchmark:
    """[E] 3.1: roll the maturing amount over, plus a learnt excess, spread by
    a learnt distribution.  Eq. (43): ||a_t||_1 = pi(1)(B_pre_t) + theta_t.

    shared=True,  uniform=True   -> BM_E  (2 parameters)
    shared=True,  uniform=False  -> BM_C  (one distribution for every month)
    shared=False, uniform=False  -> BM_D  (a distribution per month)
    """

    def __init__(self, p: sim.Params, horizon: int, *, shared: bool = True,
                 uniform: bool = False):
        self.nB, self.nK = len(p.invest_maturities), len(p.finance_maturities)
        self.T = 1 if shared else horizon
        self.shared, self.uniform = shared, uniform

    def init(self, key=None) -> dict:
        th = {"theta_B": jnp.zeros(self.T), "theta_K": jnp.zeros(self.T)}
        if not self.uniform:
            th["logit_B"] = jnp.zeros((self.T, self.nB))
            th["logit_K"] = jnp.zeros((self.T, self.nK))
        return th

    def _row(self, x, step):
        """step may be a traced integer, so index with take, not Python."""
        if self.shared:
            return x[0]
        return jnp.take(x, jnp.clip(step, 0, self.T - 1), axis=0)

    def apply(self, theta, b) -> tuple:
        step = b["step"]
        sB = jnp.maximum(b["matured_B"]
                         + SCALE_UNIT * self._row(theta["theta_B"], step), 0.0)
        sK = jnp.maximum(b["matured_K"]
                         + SCALE_UNIT * self._row(theta["theta_K"], step), 0.0)
        if self.uniform:
            dB = jnp.full((self.nB,), 1.0 / self.nB)
            dK = jnp.full((self.nK,), 1.0 / self.nK)
        else:
            dB = jax.nn.softmax(self._row(theta["logit_B"], step))
            dK = jax.nn.softmax(self._row(theta["logit_K"], step))
        return sB[:, None] * dB[None, :], sK[:, None] * dK[None, :]


class DecisionNet:
    """[E] 3.2.2 and eq. (46): the pre-trained benchmark plus a learnt deviation.

    The deviation is split into a scale and a distribution; the raw features are
    detached first ([E] (48)), the encoder weights are not.  Hidden widths
    default below the paper's [512, 512, 256, 128] because this box has two
    cores; `widths` restores them.
    """

    def __init__(self, p: sim.Params, horizon: int, baseline: Benchmark,
                 baseline_theta: dict, a0: float = 1.0,
                 widths=(256, 256, 128), enc: int = 32, enc_hidden: int = 64):
        self.nB, self.nK = len(p.invest_maturities), len(p.finance_maturities)
        self.baseline, self.baseline_theta = baseline, baseline_theta
        self.widths, self.enc, self.enc_hidden = tuple(widths), enc, enc_hidden
        self.horizon, self.a0 = horizon, a0
        self.feature_dim = 4 * enc + 3 + 5 + 6 + 1 + 2   # 145 at enc = 32

    def init(self, key) -> dict:
        keys = jax.random.split(key, 20)
        th, i = {}, 0
        for nm in ("B", "K", "R", "S"):
            th[f"enc{nm}_1"] = _glorot(keys[i], (N, self.enc_hidden)); i += 1
            th[f"enc{nm}_1b"] = jnp.zeros(self.enc_hidden)
            th[f"enc{nm}_2"] = _glorot(keys[i], (self.enc_hidden, self.enc)); i += 1
            th[f"enc{nm}_2b"] = jnp.zeros(self.enc)
        prev = self.feature_dim
        for j, w in enumerate(self.widths):
            th[f"fc{j}"] = _glorot(keys[i], (prev, w)); i += 1
            th[f"fc{j}b"] = jnp.zeros(w)
            prev = w
        for nm, out in (("scl_B", 1), ("dist_B", self.nB),
                        ("scl_K", 1), ("dist_K", self.nK)):
            # start as a small deviation from the benchmark
            th[nm] = _glorot(keys[i], (prev, out)) * 0.01; i += 1
            th[f"{nm}b"] = jnp.zeros(out)
        return th

    def _encode(self, theta, nm, x):
        h = jax.nn.elu(x / 100.0 @ theta[f"enc{nm}_1"] + theta[f"enc{nm}_1b"])
        return h @ theta[f"enc{nm}_2"] + theta[f"enc{nm}_2b"]

    def features(self, theta, b) -> jnp.ndarray:
        sg = jax.lax.stop_gradient                        # [E] (48)
        A = jnp.maximum(sg(b["A"]), 1.0)
        bounded = [sg(b["last_con"][k]) - cs.BOUNDS[k]
                   for k in ("LCR", "NSFR", "CMR", "E_over_RWA")]
        return jnp.concatenate([
            self._encode(theta, "B", sg(b["B"])),
            self._encode(theta, "K", sg(b["K"])),
            self._encode(theta, "R", sg(b["R_vec"])),
            self._encode(theta, "S", sg(b["S_vec"])),
            sg(b["pc"]) / 0.01,
            jnp.stack([A / self.a0, sg(b["E"]) / A, sg(b["C"]) / A,
                       sg(b["VB"]) / A, sg(b["VK"]) / A], axis=1),
            jnp.stack(bounded + [sg(b["last_con"]["IRS"]), sg(b["last_eyr"])],
                      axis=1),
            b["tfrac"][:, None],
            b["mu"][:, None], b["lam"][:, None] / 10.0,
        ], axis=1)

    def apply(self, theta, b) -> tuple:
        aB0, aK0 = self.baseline.apply(
            jax.lax.stop_gradient(self.baseline_theta), b)
        x = self.features(theta, b)
        for j in range(len(self.widths)):
            y = jax.nn.elu(x @ theta[f"fc{j}"] + theta[f"fc{j}b"])
            x = x + y if y.shape[-1] == x.shape[-1] else y   # [E] 3.2.3.2
        dB = SCALE_UNIT * (x @ theta["scl_B"] + theta["scl_Bb"]) \
            * jax.nn.softmax(x @ theta["dist_B"] + theta["dist_Bb"], axis=-1)
        dK = SCALE_UNIT * (x @ theta["scl_K"] + theta["scl_Kb"]) \
            * jax.nn.softmax(x @ theta["dist_K"] + theta["dist_Kb"], axis=-1)
        return jnp.maximum(aB0 + dB, 0.0), jnp.maximum(aK0 + dK, 0.0)


def n_params(theta) -> int:
    return int(sum(np.size(a) for a in jax.tree.leaves(theta)))


# ================================================================= the loss

def path_loss(out, mu, lam, years: float, E0: float):
    """[E] (32) with the asymmetric target loss (47), averaged over the batch."""
    gap = ls.TARGET_SCALE * (out["ET"] / E0 - (1.0 + mu) ** years)
    lt = jnp.minimum(gap, 0.0) ** 2
    lp = (1.0 + out["penalty"]) ** 2 - 1.0
    return (lt + lam * lp).mean()


# ================================================================= training

def train(pre: Prepared, strategy, theta: dict, *, epochs: int = 30,
          batch_size: int = 256, lr=(5e-4, 5e-3), clip: float = 0.2,
          seed: int = 0, log=print, val: "Prepared | None" = None,
          sample_params: bool = True):
    """Minimise the mean total loss by backpropagating through all H months."""
    rng = np.random.default_rng(seed)
    nb = max(1, pre.S // batch_size)
    years = pre.H / 12.0
    roll = make_rollout(pre, strategy)

    def objective(th, b, mu, lam):
        return path_loss(roll(th, b, mu, lam), mu, lam, years, pre.E0)

    grad_fn = jax.jit(jax.value_and_grad(objective))
    sched = optax.linear_onecycle_schedule(
        transition_steps=max(2, epochs * nb), peak_value=lr[1],
        div_factor=lr[1] / lr[0], final_div_factor=lr[1] / lr[0])
    opt = optax.chain(optax.clip(clip), optax.radam(sched))   # [E] Table 2
    state = opt.init(theta)

    # Batches are contiguous blocks visited in a random order.  The scenarios
    # are already an i.i.d. sample from Layer 1, so a block is as random as a
    # gather, and slicing a 350 MB array beats fancy-indexing it: the gather,
    # not the gradient, was most of each epoch.
    history, best = [], None
    for ep in range(epochs):
        order = rng.permutation(nb)
        tot = 0.0
        for k in order:
            idx = slice(int(k) * batch_size, (int(k) + 1) * batch_size)
            b = pre.take(idx)
            if sample_params:          # [E] 3.2.3.1, one draw per path
                mu = jnp.asarray(rng.uniform(*ls.MU_RANGE, batch_size), jnp.float32)
                lam = jnp.asarray(rng.uniform(*ls.LAMBDA_RANGE, batch_size),
                                  jnp.float32)
            else:
                mu = jnp.full(batch_size, ls.MU_EVAL, jnp.float32)
                lam = jnp.full(batch_size, ls.LAMBDA_EVAL, jnp.float32)
            L, g = grad_fn(theta, b, mu, lam)
            upd, state = opt.update(g, state, theta)
            theta = optax.apply_updates(theta, upd)
            tot += float(L)
        row = {"epoch": ep, "train_loss": tot / nb}
        if val is not None:
            v = evaluate(val, strategy, theta)
            row.update({"val_total": v["total"], "val_penalty": v["penalty"],
                        "val_ER": v["ER_mean"]})
            if best is None or v["total"] < best[0]:      # [E] 3.2.3: early stop
                best = (v["total"], ep, jax.tree.map(np.asarray, theta))
            log(f"  epoch {ep + 1:3d}/{epochs}  train {row['train_loss']:12.3f}"
                f"   val {v['total']:12.3f}   ER {v['ER_mean']:.4f}"
                f"   clean {v['clean']:5.1%}"
                + ("  *" if best[1] == ep else ""))
        else:
            log(f"  epoch {ep + 1:3d}/{epochs}  train {row['train_loss']:12.3f}")
        history.append(row)
    if best is not None:
        log(f"  keeping epoch {best[1] + 1} (validation loss {best[0]:.3f})")
        theta = jax.tree.map(jnp.asarray, best[2])
    return theta, history


PATH_KEYS = ("C", "E", "aB", "aK", "EYR") + CON_NAMES


def run(pre: Prepared, strategy, theta: dict, *, mu: float = ls.MU_EVAL,
        lam: float = ls.LAMBDA_EVAL, batch_size: int = 400, path: bool = False):
    """Final equity and accumulated penalty for every scenario in `pre`."""
    roll = jax.jit(make_rollout(pre, strategy))
    ET, pen, rows = [], [], []
    for k in range(0, pre.S, batch_size):
        idx = np.arange(k, min(k + batch_size, pre.S))
        b = pre.take(idx)
        one = jnp.ones(len(idx), jnp.float32)
        out = roll(theta, b, mu * one, lam * one)
        ET.append(np.asarray(out["ET"]))
        pen.append(np.asarray(out["penalty"]))
        if path:
            rows.append({k2: np.asarray(out["path"][k2]) for k2 in PATH_KEYS})
    ET, pen = np.concatenate(ET), np.concatenate(pen)
    if not path:
        return ET, pen
    full = {k2: np.concatenate([r[k2] for r in rows], axis=1) for k2 in PATH_KEYS}
    return ET, pen, full


def summarise_path(full: dict, H: int) -> dict:
    """[E] Table 4: the median path and how often each limit is broken.

    Checked at the H decision steps t = 0..H-1, as [E] Alg. 1 does."""
    med = {k: [float(np.median(v[t])) for t in range(v.shape[0])]
           for k, v in full.items()}
    viol = {}
    for name in CON_NAMES:
        x = full[name][:H]
        gap = (np.abs(x) - cs.BOUNDS[name] if name in cs.UPPER_BOUNDED
               else cs.BOUNDS[name] - x)
        viol[name] = float((gap > 0).any(axis=0).mean())
    # EYR is inf on the months where it is not checked ([E] (26): annual only)
    eyr = full["EYR"][:H]
    viol["EYR"] = float((eyr < 0.0).any(axis=0).mean())
    med["EYR"] = [float(np.median(v)) if np.isfinite(v).all() else float("nan")
                  for v in eyr] + [float("nan")] * (full["EYR"].shape[0] - H)
    return {"median": med, "violation_share": viol}


def evaluate(pre: Prepared, strategy, theta: dict, *, mu: float = ls.MU_EVAL,
             lam: float = ls.LAMBDA_EVAL, gamma: float = ls.GAMMA_EVAL,
             batch_size: int = 400, path: bool = False) -> dict:
    """Every number of [E] Table 3, on a held-out set."""
    got = run(pre, strategy, theta, mu=mu, lam=lam, batch_size=batch_size,
              path=path)
    ET, pen = got[0], got[1]
    years, E0 = pre.H / 12.0, pre.E0
    lp = ls.penalty_loss(pen)
    lt = ls.target_loss(ET, E0, mu, years)
    er = ET / E0
    cen = er - er.mean()
    m2, m3, m4 = (cen ** 2).mean(), (cen ** 3).mean(), (cen ** 4).mean()
    extra = summarise_path(got[2], pre.H) if path else {}
    return {
        **extra,
        "total": float((lt + lam * lp).mean()),
        "target": float(lt.mean()),
        "target_symmetric": float(
            ls.target_loss(ET, E0, mu, years, asymmetric=False).mean()),
        "penalty": float(lp.mean()),
        "penalty_ES0.95": ls._es(lp, 0.95, upper=True),
        "utility": float(ls.utility_loss(ET, E0, gamma).mean()),
        "ER_mean": float(er.mean()), "ER_sd": float(np.sqrt(m2)),
        "ER_skew": float(m3 / m2 ** 1.5),
        "ER_excess_kurtosis": float(m4 / m2 ** 2 - 3.0),
        "ER_VaR0.95": ls._var(er, 0.05),
        "ER_ES0.95": ls._es(er, 0.05, upper=False),
        "annual_return": float((er ** (1.0 / years) - 1.0).mean()),
        "clean": float((pen <= 1e-9).mean()),
    }


# -------------------------------------------- the two untrained strategies

class Zero:
    """NoAction, in JAX: the reference path."""

    def __init__(self, p: sim.Params):
        self.nB, self.nK = len(p.invest_maturities), len(p.finance_maturities)

    def init(self, key=None):
        return {}

    def apply(self, theta, b):
        n = b["C"].shape[0]
        return jnp.zeros((n, self.nB)), jnp.zeros((n, self.nK))


class RollOver:
    """simulator.RollOver, in JAX: invest a share of idle cash, refinance."""

    def __init__(self, p: sim.Params, invest_share: float = 0.25):
        self.p, self.share = p, invest_share
        self.nB, self.nK = len(p.invest_maturities), len(p.finance_maturities)

    def init(self, key=None):
        return {}

    def apply(self, theta, b):
        exempt = self.p.reserve_exemption * 0.025 * b["V_S"]
        spare = jnp.maximum(b["C"] - exempt, 0.0)
        aB = jnp.broadcast_to((self.share * spare / self.nB)[:, None],
                              (b["C"].shape[0], self.nB))
        aK = jnp.broadcast_to((b["matured_K"] / self.nK)[:, None],
                              (b["C"].shape[0], self.nK))
        return aB, aK
