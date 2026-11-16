"""Walk the bank forward 5 years along every yield-curve scenario (Layer 3).

    python run_simulator.py --fit output_paper/fit_sqrt.npz --out output_paper

Needs fit_sqrt.npz from run_fit.py.  Writes
    <out>/simulator_report.json    every number quoted in layer3_report.md
"""
import argparse
import json
import os

import numpy as np

import balance_sheet as bs
import loss as ls
import simulator as sim
import yieldcurve as yc

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="output_paper/fit_sqrt.npz")
ap.add_argument("--out", default="output_paper")
ap.add_argument("--scenarios", type=int, default=4000)
ap.add_argument("--seed", type=int, default=20221230)
ap.add_argument("--variant", default="sqrt", choices=["sqrt", "term_premium"],
                help="'sqrt' is the paper as published; 'term_premium' is Version B")
ap.add_argument("--policy", default="none", choices=["none", "rollover"],
                help="'none' = invest and borrow nothing, the reference path")
ap.add_argument("--interest-income", type=float, default=303.354,
                help="the bank's 2022 interest income, mCHF, used to calibrate kappa_L")
ap.add_argument("--mu", type=float, default=ls.MU_EVAL,
                help="annual return target for the Layer 4 target loss")
ap.add_argument("--penalty-weight", type=float, default=ls.LAMBDA_EVAL,
                help="lambda, the weight on the penalty loss")
A = ap.parse_args()

y0 = np.load(A.fit)["y0"]
x0 = bs.build_initial_state(y0)
kappa_L = sim.calibrate_loan_spread(x0, y0, A.interest_income)
p = sim.Params(loan_spread=kappa_L)
mix = sim.fit_mortgage_mix(x0.RP, p.mortgage_new_months)

scen = yc.load_scenarios(A.fit, n_scenarios=A.scenarios, seed=A.seed, variant=A.variant)
policy = {"none": sim.NoAction(), "rollover": sim.RollOver()}[A.policy]
out = sim.simulate(x0, scen.yields, p, policy, mortgage_mix=mix)
r = sim.summarise(out)
r["loss"] = ls.evaluate(out, mu=A.mu, lam=A.penalty_weight)

r["setup"] = {
    "fit": A.fit, "variant": A.variant, "policy": A.policy,
    "scenarios": A.scenarios, "seed": A.seed,
    "valuation_date": str(np.load(A.fit)["valuation_date"]),
    "loan_spread_bp": round(kappa_L * 1e4, 2),
    "mortgage_mix": {f"{m // 12}y": round(float(w), 4)
                     for m, w in zip(p.mortgage_new_months, mix)},
    "resolved": sim.RESOLVED,
}
os.makedirs(A.out, exist_ok=True)
with open(os.path.join(A.out, "simulator_report.json"), "w") as f:
    json.dump(r, f, indent=1, default=float)

yrs = [0, 12, 24, 36, 48, 60]
print(f"policy: {A.policy}   kappa_L = {kappa_L * 1e4:.0f} bp   "
      f"{A.scenarios} scenarios from {r['setup']['valuation_date']}\n")
print("Median path, CHF millions          " + "".join(f"{t // 12:>10}y" for t in yrs))
for k in ("C", "B", "RP", "RE", "SD", "SF", "K", "E"):
    print(f"  {k:<33}" + "".join(f"{r['median'][k][t]:>11,.0f}" for t in yrs))
print("\nMedian path, ratios")
for k in ("E_over_RWA", "LCR", "NSFR", "CMR", "IRS"):
    print(f"  {k:<33}" + "".join(f"{r['median'][k][t]:>11.3f}" for t in yrs))
print("\nScenarios breaking each limit at least once")
for k, v in r["violation_share"].items():
    print(f"  {k:<12}{100 * v:6.1f}%")
print(f"\nEquity after 5 years: median {r['ET']['median']:,.0f}  "
      f"(5-95%: {r['ET']['q05']:,.0f} to {r['ET']['q95']:,.0f}), "
      f"{r['ET_over_E0']:.3f} x the starting value")

L, ER = r["loss"]["loss"], r["loss"]["equity_ratio"]
print(f"\nLayer 4 loss (mu = {A.mu:.2%}, lambda = {A.penalty_weight})")
print(f"  total                            {L['total']:11.3f}")
print(f"  target (asymmetric)              {L['target']:11.3f}")
print(f"  penalty                          {L['penalty']:11.3f}")
print(f"  penalty ES 95%                   {L['penalty_ES0.95']:11.3f}")
print(f"  utility (gamma = 10)             {L['utility']:11.3f}")
print("Equity ratio E_T/E_0")
for k in ("mean", "sd", "skew", "excess_kurtosis", "VaR0.95", "ES0.95"):
    print(f"  {k:<33}{ER[k]:11.3f}")
print(f"  annualised return                "
      f"{r['loss']['annualised']['mu_hat']:11.2%}")
print(f"  annual dividend yield            "
      f"{r['loss']['annualised']['dividend_yield']:11.2%}")
print("Penalty contribution by constraint")
for k, v in sorted(r["loss"]["penalty_share"].items(), key=lambda kv: -kv[1]):
    print(f"  {k:<33}{v:11.4f}")
print(f"\nSaved {A.out}/simulator_report.json")