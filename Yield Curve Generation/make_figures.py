"""Figures for the yield curve layer report."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import yieldcurve as yc

ap = argparse.ArgumentParser(description="Figures for the yield curve layer.")
ap.add_argument("--out", default="output", help="directory written by run_fit.py")
ap.add_argument("--paths", default="sample",
                help='"sample" (default) or "scenarios", matching run_fit.py')
ARGS = ap.parse_args()
OUT, KIND = ARGS.out, ARGS.paths

DTAU = 1.0 / 12.0
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

sq = np.load(os.path.join(OUT, "fit_sqrt.npz"))
eg = np.load(os.path.join(OUT, "fit_eig.npz"))
tenors, f0, y0 = sq["tenors"], sq["f0"], sq["y0"]

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
for ax, data, title in (
    (axes[0], sq, r"scaled by $\sqrt{\lambda_j}$"),
    (axes[1], eg, r"scaled by $\lambda_j$"),
):
    for j, c in enumerate(["#1f4e79", "#c0504d", "#4f8a3d"]):
        ax.plot(tenors[:240], data["V_raw"][:, j], lw=0.8, alpha=0.45, color=c)
        ax.plot(tenors[:240], data["V"][:, j], lw=1.8, color=c,
                label=f"$V^{{({j+1})}}$")
    ax.axhspan(-0.005, 0.008, color="#888888", alpha=0.10, zorder=0)
    ax.set_xlabel("Tenor (years)")
    ax.set_title(title)
    ax.set_xlim(0, 20)
axes[0].set_ylabel("Volatility loading")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_ylim(-0.006, 0.009)
axes[1].set_ylim(-0.006, 0.009)
axes[1].text(10, 0.0065, "shaded band = axis range\nof the published Figure 3",
             fontsize=7.5, ha="center", color="#444444")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_loadings.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, tag, title in (
    (axes[0], "sqrt", r"$\sqrt{\lambda_j}$ scaling"),
    (axes[1], "eigenvalue", r"$\lambda_j$ scaling"),
):
    paths = np.load(os.path.join(OUT, f"{KIND}_{tag}.npz"))["forwards"][:400]
    ylds = yc.forwards_to_yields(paths.astype(float), DTAU)
    short = ylds[:, :, 0] * 100
    t = np.arange(short.shape[1]) / 12
    ax.fill_between(t, np.quantile(short, .05, 0), np.quantile(short, .95, 0),
                    color="#1f4e79", alpha=0.15, lw=0)
    ax.fill_between(t, np.quantile(short, .25, 0), np.quantile(short, .75, 0),
                    color="#1f4e79", alpha=0.30, lw=0)
    ax.plot(t, np.median(short, 0), color="#1f4e79", lw=1.8)
    ax.axhline(short[0, 0], color="#c0504d", ls="--", lw=1,
               label="initial 1m yield")
    ax.axhline(-1.0, color="#777777", ls=":", lw=1, label="$-1\\%$ floor")
    ax.set_xlabel("Model period (years)")
    ax.set_title(title)
axes[0].set_ylabel("1-month yield (%)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_shortrate.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
paths = np.load(os.path.join(OUT, f"{KIND}_sqrt.npz"))["forwards"][:400].astype(float)
ylds = yc.forwards_to_yields(paths, DTAU)
final = ylds[:, -1, :] * 100
m = tenors[:180]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
ax = axes[0]
for i in range(60):
    ax.plot(m, final[i], color="#1f4e79", lw=0.4, alpha=0.25)
ax.plot(m, y0[:180] * 100, color="#c0504d", lw=2, label="initial curve")
ax.plot(m, final.mean(0), color="#1f4e79", lw=2, label="mean after 5y")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (%)")
ax.set_title("Simulated curves in 5 years")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.hist(ylds[:, -1, 0] * 100, bins=45, color="#1f4e79", alpha=0.75,
        edgecolor="white", lw=0.3)
ax.axvline(y0[0] * 100, color="#c0504d", ls="--", lw=1.2, label="initial")
ax.axvline(-1.0, color="#777777", ls=":", lw=1.2, label="$-1\\%$")
ax.set_xlabel("1-month yield after 5 years (%)")
ax.set_ylabel("Scenarios")
ax.set_title("Distribution at the horizon")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_curves.pdf"), bbox_inches="tight")
plt.close(fig)

print(f"figures written to {OUT}/")
