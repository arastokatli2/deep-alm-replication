"""Score the saved strategies side by side on one validation set (Layer 5).

    python score_deep_alm.py --fit output_paper/fit_sqrt.npz --out output_paper

Reads the pickles that run_deep_alm.py wrote and reports, for every strategy,
[E] Table 3's loss and equity statistics together with [E] Table 4's median
ratios and violation shares.  Writes <out>/comparison.json.
"""
import argparse
import json
import os
import pickle

import jax
import numpy as np

import balance_sheet as bs
import deep_alm as da
import loss as ls
import simulator as sim
import yieldcurve as yc

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="output_paper/fit_sqrt.npz")
ap.add_argument("--out", default="output_paper")
ap.add_argument("--val-scenarios", type=int, default=1600)
ap.add_argument("--val-seed", type=int, default=20221230)
ap.add_argument("--variant", default="sqrt", choices=["sqrt", "term_premium"])
ap.add_argument("--interest-income", type=float, default=303.354)
ap.add_argument("--widths", default="512,512,256,128")
A = ap.parse_args()

y0 = np.load(A.fit)["y0"]
x0 = bs.build_initial_state(y0)
kappa_L = sim.calibrate_loan_spread(x0, y0, A.interest_income)
p = sim.Params(loan_spread=kappa_L)
mix = sim.fit_mortgage_mix(x0.RP, p.mortgage_new_months)
val = da.prepare(x0, yc.load_scenarios(A.fit, n_scenarios=A.val_scenarios,
                                       seed=A.val_seed,
                                       variant=A.variant).yields, p, mix)
H = val.H
load = lambda n: pickle.load(open(os.path.join(A.out, f"deep_alm_{n}.pkl"), "rb"))

strategies = {"nothing": (da.Zero(p), {}), "rollover": (da.RollOver(p), {})}
for name, kw in (("BM_E", dict(shared=True, uniform=True)),
                 ("BM_C", dict(shared=True, uniform=False)),
                 ("BM_D", dict(shared=False, uniform=False))):
    path = os.path.join(A.out, f"deep_alm_{name}.pkl")
    if os.path.isfile(path):
        strategies[name] = (da.Benchmark(p, H, **kw), load(name))
if os.path.isfile(os.path.join(A.out, "deep_alm_DeepALM.pkl")):
    base = da.Benchmark(p, H, shared=False, uniform=False)
    strategies["DeepALM"] = (
        da.DecisionNet(p, H, base, jax.tree.map(jax.numpy.asarray, load("BM_D")),
                       a0=val.A0,
                       widths=tuple(int(w) for w in A.widths.split(","))),
        load("DeepALM"))

out = {}
for name, (strategy, theta) in strategies.items():
    th = jax.tree.map(jax.numpy.asarray, theta)
    out[name] = da.evaluate(val, strategy, th, path=True)
    print(f"scored {name}")

with open(os.path.join(A.out, "comparison.json"), "w") as f:
    json.dump({"setup": {"E0_mCHF": val.E0, "val_scenarios": A.val_scenarios,
                         "val_seed": A.val_seed, "variant": A.variant,
                         "mu": ls.MU_EVAL, "lambda": ls.LAMBDA_EVAL,
                         "gamma": ls.GAMMA_EVAL},
               "results": out}, f, indent=1, default=float)

order = [k for k in ("nothing", "rollover", "BM_E", "BM_C", "BM_D", "DeepALM")
         if k in out]
print("\nLoss and equity ([E] Table 3)")
print(" " * 22 + "".join(f"{k:>12}" for k in order))
for r in ("total", "target", "penalty", "penalty_ES0.95", "utility", "ER_mean",
          "ER_sd", "ER_skew", "ER_VaR0.95", "ER_ES0.95", "annual_return"):
    print(f"  {r:<20}" + "".join(f"{out[k][r]:12.4f}" for k in order))

print("\nMedian ratio at 5y ([E] Table 4)")
print(" " * 22 + "".join(f"{k:>12}" for k in order))
for r in da.CON_NAMES:
    print(f"  {r:<20}" + "".join(f"{out[k]['median'][r][H]:12.4f}" for k in order))
print(f"  {'equity mCHF':<20}" + "".join(f"{out[k]['median']['E'][H]:12.0f}"
                                         for k in order))
print(f"  {'cash mCHF':<20}" + "".join(f"{out[k]['median']['C'][H]:12.0f}"
                                       for k in order))
for t in (12, 24, 36, 48):
    print(f"  {'EYR at year ' + str(t // 12):<20}"
          + "".join(f"{out[k]['median']['EYR'][t]:12.2f}" for k in order))

print("\nWhat the strategy actually does, mCHF a month, median over months")
print(" " * 22 + "".join(f"{k:>12}" for k in order))
for lbl, key in (("invest volume", "aB"), ("borrow volume", "aK")):
    print(f"  {lbl:<20}" + "".join(
        f"{float(np.median(out[k]['median'][key][:H])):12.1f}" for k in order))

print("\nScenarios breaking each limit at least once")
print(" " * 22 + "".join(f"{k:>12}" for k in order))
for r in da.CON_NAMES:
    print(f"  {r:<20}" + "".join(f"{out[k]['violation_share'][r]:11.1%}"
                                 for k in order))
print(f"  {'none at all':<20}" + "".join(f"{out[k]['clean']:11.1%}"
                                         for k in order))
print(f"\nSaved {A.out}/comparison.json")
