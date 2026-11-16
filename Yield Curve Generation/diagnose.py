"""Post-fit diagnostics: does the simulator reproduce historical structure?

Checks the generator against properties nobody asked it to match, which is
where a scenario generator usually fails.
"""
import argparse, os
import numpy as np
import pandas as pd
import yieldcurve as yc

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="output")
ap.add_argument("--variant", default="term_premium",
                help='"sqrt" (paper, P = Q) or "term_premium"')
A = ap.parse_args()

DTAU = 1/12
N_OUT, N_EXTRA = 180, 60
TEN = np.arange(1, N_OUT + N_EXTRA + 1) * DTAU

params = yc.load_rendopar(A.csv)
weekly = params.loc["2005-01-01":"2022-07-15"].resample("W-WED").last().dropna()

# ---- historical yield volatility term structure -------------------------
hist_y = np.vstack([yc.svensson_spot(r, TEN[:N_OUT]) for r in weekly.values])
hist_vol = hist_y[1:].__sub__(hist_y[:-1]).std(axis=0) * np.sqrt(52) * 100

# ---- simulated yield volatility term structure --------------------------
sq = np.load(os.path.join(A.out, "fit_sqrt.npz"))
paths = np.load(os.path.join(A.out, f"sample_{A.variant}.npz"))["forwards"].astype(float)
print(f"variant: {A.variant}\n")
sim_y = yc.forwards_to_yields(paths, DTAU)
sim_vol = np.diff(sim_y, axis=1).std(axis=(0, 1)) * np.sqrt(12) * 100

idx = [0, 5, 11, 23, 59, 119, 179]
lab = ["1m", "6m", "1y", "2y", "5y", "10y", "15y"]
print("Annualised volatility of yield changes (percentage points)")
print(f"{'maturity':>9} {'historical':>11} {'simulated':>10} {'ratio':>7}")
for i, l in zip(idx, lab):
    print(f"{l:>9} {hist_vol[i]:11.3f} {sim_vol[i]:10.3f} {sim_vol[i]/hist_vol[i]:7.2f}")

# ---- curve shape distribution -------------------------------------------
print("\nCurve shape (15y yield minus 1m yield, percentage points)")
hist_slope = (hist_y[:, 179] - hist_y[:, 0]) * 100
sim_slope_T = (sim_y[:, -1, 179] - sim_y[:, -1, 0]) * 100
sim_slope_all = (sim_y[:, :, 179] - sim_y[:, :, 0]).ravel() * 100
for nm, s in (("historical 2005-2022", hist_slope),
              ("simulated, at T=5y", sim_slope_T),
              ("simulated, all steps", sim_slope_all)):
    print(f"  {nm:<22} mean {s.mean():6.2f}  sd {s.std():5.2f} "
          f" inverted {100*(s<0).mean():5.1f}%  "
          f"[{np.quantile(s,.05):6.2f}, {np.quantile(s,.95):5.2f}]")

# ---- cross-maturity correlation -----------------------------------------
print("\nCorrelation of yield changes, 1m vs other maturities")
h = np.diff(hist_y, axis=0)
s = np.diff(sim_y, axis=1).reshape(-1, N_OUT)
print(f"{'maturity':>9} {'historical':>11} {'simulated':>10}")
for i, l in zip(idx[1:], lab[1:]):
    print(f"{l:>9} {np.corrcoef(h[:,0],h[:,i])[0,1]:11.3f} "
          f"{np.corrcoef(s[:,0],s[:,i])[0,1]:10.3f}")

# ---- negative rates ------------------------------------------------------
neg = (sim_y[:, -1, :] < 0).any(axis=1).mean()
print(f"\nScenarios with any negative yield at T: {100*neg:.1f}%")
print(f"Lowest simulated 1m yield at T: {sim_y[:,-1,0].min()*100:.2f}%")
print(f"Highest simulated 15y yield at T: {sim_y[:,-1,179].max()*100:.2f}%")
