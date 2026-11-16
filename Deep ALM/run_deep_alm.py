"""Train and score every ALM strategy on the same scenarios (Layer 5).

    python run_deep_alm.py --fit output_paper/fit_sqrt.npz --out output_paper

Trains [E] 3.1's three benchmarks and then the Deep ALM decision network on top
of BM_D, and scores all of them -- plus doing nothing and the hand-written
roll-over rule -- on a held-out validation set, with [E] Table 3's metrics.

Writes <out>/deep_alm_report.json and <out>/deep_alm_<name>.npz
"""
import argparse
import json
import os
import pickle
import time

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
ap.add_argument("--train-scenarios", type=int, default=8000)
ap.add_argument("--val-scenarios", type=int, default=1600)
ap.add_argument("--train-seed", type=int, default=20220715)
ap.add_argument("--val-seed", type=int, default=20221230)
ap.add_argument("--variant", default="sqrt", choices=["sqrt", "term_premium"])
ap.add_argument("--bm-epochs", type=int, default=25)
ap.add_argument("--nn-epochs", type=int, default=60)
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--widths", default="512,512,256,128",
                help="hidden widths; [E] Table 2 used 512,512,256,128")
ap.add_argument("--interest-income", type=float, default=303.354)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only", default="", help="comma-separated subset to score")
A = ap.parse_args()

# ------------------------------------------------------------------ the bank
y0 = np.load(A.fit)["y0"]
x0 = bs.build_initial_state(y0)
kappa_L = sim.calibrate_loan_spread(x0, y0, A.interest_income)
p = sim.Params(loan_spread=kappa_L)
mix = sim.fit_mortgage_mix(x0.RP, p.mortgage_new_months)

print(f"kappa_L = {kappa_L * 1e4:.0f} bp;  preparing "
      f"{A.train_scenarios} training and {A.val_scenarios} validation scenarios")
t0 = time.time()
train_y = yc.load_scenarios(A.fit, n_scenarios=A.train_scenarios,
                            seed=A.train_seed, variant=A.variant).yields
train = da.prepare(x0, train_y, p, mix)
del train_y
val_y = yc.load_scenarios(A.fit, n_scenarios=A.val_scenarios,
                          seed=A.val_seed, variant=A.variant).yields
val = da.prepare(x0, val_y, p, mix, pca_from=train)
del val_y
print(f"  ready in {time.time() - t0:.0f}s;  E_0 = {train.E0:,.1f} mCHF, "
      f"H = {train.H} months")

widths = tuple(int(w) for w in A.widths.split(","))
wanted = [s.strip() for s in A.only.split(",") if s.strip()]
want = lambda n: not wanted or n in wanted
results, histories, thetas = {}, {}, {}


def score(name, strategy, theta):
    r = da.evaluate(val, strategy, theta)
    results[name] = r
    print(f"  {name:<9} loss {r['total']:11.3f}  penalty {r['penalty']:10.3f}"
          f"  ER {r['ER_mean']:.4f} +- {r['ER_sd']:.4f}"
          f"  clean {r['clean']:6.1%}")


# ---------------------------------------- the two strategies with no training
print("\nuntrained strategies")
for name, strat in (("nothing", da.Zero(p)), ("rollover", da.RollOver(p))):
    if want(name):
        score(name, strat, strat.init())

# ------------------------------------------------------------- the benchmarks
specs = {"BM_E": dict(shared=True, uniform=True),
         "BM_C": dict(shared=True, uniform=False),
         "BM_D": dict(shared=False, uniform=False)}
for name, kw in specs.items():
    if not want(name) and name != "BM_D":
        continue                              # BM_D is the network's baseline
    bm = da.Benchmark(p, train.H, **kw)
    th = bm.init()
    print(f"\n{name}: {da.n_params(th)} trainable parameters")
    t0 = time.time()
    th, histories[name] = da.train(train, bm, th, epochs=A.bm_epochs,
                                   batch_size=A.batch_size, val=val,
                                   seed=A.seed + 1)
    print(f"  trained in {time.time() - t0:.0f}s")
    thetas[name] = (bm, th)
    if want(name):
        score(name, bm, th)

# ----------------------------------------------------------------- Deep ALM
if want("DeepALM"):
    base_bm, base_th = thetas["BM_D"]
    net = da.DecisionNet(p, train.H, base_bm, base_th, a0=train.A0, widths=widths)
    th = net.init(jax.random.PRNGKey(A.seed))
    print(f"\nDeepALM: {da.n_params(th):,} trainable parameters, "
          f"hidden {widths}, {net.feature_dim} features")
    t0 = time.time()
    th, histories["DeepALM"] = da.train(train, net, th, epochs=A.nn_epochs,
                                        batch_size=A.batch_size, val=val,
                                        seed=A.seed + 2)
    print(f"  trained in {time.time() - t0:.0f}s")
    thetas["DeepALM"] = (net, th)
    score("DeepALM", net, th)

# -------------------------------------------------------------------- output
os.makedirs(A.out, exist_ok=True)
for name, (strategy, th) in thetas.items():
    with open(os.path.join(A.out, f"deep_alm_{name}.pkl"), "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, th), f)

report = {
    "setup": {
        "fit": A.fit, "variant": A.variant, "E0_mCHF": train.E0,
        "train_scenarios": A.train_scenarios, "val_scenarios": A.val_scenarios,
        "train_seed": A.train_seed, "val_seed": A.val_seed,
        "loan_spread_bp": round(kappa_L * 1e4, 2), "widths": list(widths),
        "bm_epochs": A.bm_epochs, "nn_epochs": A.nn_epochs,
        "batch_size": A.batch_size, "mu_eval": ls.MU_EVAL,
        "lambda_eval": ls.LAMBDA_EVAL, "gamma_eval": ls.GAMMA_EVAL,
        "mu_range": ls.MU_RANGE, "lambda_range": ls.LAMBDA_RANGE,
        "paper_hyperparameters": {"widths": [512, 512, 256, 128],
                                  "train_scenarios": 40000, "epochs": 100,
                                  "batch_size": 32},
    },
    "results": results,
    "training": histories,
}
with open(os.path.join(A.out, "deep_alm_report.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)

rows = ["total", "target", "penalty", "penalty_ES0.95", "utility", "ER_mean",
        "ER_sd", "ER_skew", "ER_excess_kurtosis", "ER_VaR0.95", "ER_ES0.95",
        "annual_return", "clean"]
order = [k for k in ("nothing", "rollover", "BM_E", "BM_C", "BM_D", "DeepALM")
         if k in results]
print("\n" + " " * 22 + "".join(f"{k:>12}" for k in order))
for r in rows:
    print(f"  {r:<20}" + "".join(f"{results[k][r]:12.4f}" for k in order))
print(f"\nSaved {A.out}/deep_alm_report.json")
