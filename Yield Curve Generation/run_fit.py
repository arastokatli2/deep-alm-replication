"""Fit the HJM-PCA volatility structure and run the validation checks."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import yieldcurve as yc

ap = argparse.ArgumentParser(description="Fit the HJM-PCA yield curve layer.")
ap.add_argument("--csv", required=True,
                help="path to the SNB rendopar CSV")
ap.add_argument("--out", default="output",
                help="directory for fits, scenarios and the report")
ap.add_argument("--scenarios", type=int, default=4000,
                help="number of scenarios to simulate")
ap.add_argument("--seed", type=int, default=20221230)
ap.add_argument("--valuation-date", default="2022-12-30",
                help="start date of the simulated curves (paper: 2022-07-15)")
ap.add_argument("--model", default="term_premium", choices=["paper", "term_premium"],
                help="which scenarios the curve features are fitted on: "
                     "'paper' = P = Q exactly as published, 'term_premium' = Version B")
ap.add_argument("--feature-scenarios", type=int, default=1000,
                help="training scenarios used to fit the 3-number curve features")
ap.add_argument("--feature-seed", type=int, default=1,
                help="seed for those training scenarios; must differ from --seed")
ap.add_argument("--save-paths", action="store_true",
                help="also write the simulated forward curves to disk. Off by "
                     "default: paths are fully reproducible from the saved fit "
                     "plus the seed, and 4,000 scenarios is ~175 MB.")
ARGS = ap.parse_args()

CSV = ARGS.csv
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)

DTAU = 1.0 / 12.0      # monthly tenor spacing          (decision D2)
N_OUTPUT = 180         # model horizon in cash-flow buckets
N_EXTRA = 60           # extra tenors for the roll-down (decision D4)
N_GRID = N_OUTPUT + N_EXTRA
SAMPLE_START = "2005-01-01"
SAMPLE_END = "2022-07-15"
VALUATION_DATE = ARGS.valuation_date
FEATURE_VARIANT = "sqrt" if ARGS.model == "paper" else "term_premium"

TENORS = np.arange(1, N_GRID + 1) * DTAU


def main() -> dict:
    report: dict = {}

    # ---- 1. load -------------------------------------------------------
    params = yc.load_rendopar(CSV)
    report["file"] = {
        "rows_after_cleaning": int(len(params)),
        "first_date": str(params.index[0].date()),
        "last_date": str(params.index[-1].date()),
    }

    # ---- 2. sanity check on the Svensson evaluator ---------------------
    day = params.loc[params.index <= pd.Timestamp(VALUATION_DATE)].iloc[-1]
    probe = np.array([1 / 12, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    spot_probe = yc.svensson_spot(day.values, probe)
    report["spot_check"] = {
        "date": str(day.name.date()),
        "params": {k: float(v) for k, v in day.items()},
        "yields_pct": {f"{m:g}y": round(float(r) * 100, 4)
                       for m, r in zip(probe, spot_probe)},
    }

    # ---- 3. converter cross-checks (T1, T2) ----------------------------
    report["check_discount"] = yc.check_discount_consistency(
        day.values, TENORS, DTAU)
    report["check_forward_routes"] = yc.check_forward_consistency(
        day.values, TENORS, DTAU)

    # round-trip forwards -> yields -> forwards
    f_iv = yc.svensson_interval_forward(day.values, N_GRID, DTAU)
    rt = yc.yields_to_forwards(yc.forwards_to_yields(f_iv, DTAU), DTAU)
    report["check_roundtrip_bp"] = float(np.abs(rt - f_iv).max() * 1e4)

    # ---- 4. weekly history --------------------------------------------
    window = params.loc[SAMPLE_START:SAMPLE_END]
    weekly = window.resample("W-WED").last().dropna()
    report["history"] = {
        "daily_observations": int(len(window)),
        "weekly_observations": int(len(weekly)),
        "first": str(weekly.index[0].date()),
        "last": str(weekly.index[-1].date()),
    }

    # interval-average forwards: what eq. (42) needs to be exact
    fwd_hist = yc.svensson_interval_forward(weekly.values, N_GRID, DTAU)
    report["history"]["forward_curve_shape"] = list(fwd_hist.shape)

    # ---- 5. fit both scalings (decision D1) ----------------------------
    fits = {}
    for scaling in ("sqrt", "eigenvalue"):
        fit = yc.fit_hjm_pca(
            fwd_hist, TENORS,
            periods_per_year=52, n_components=3, poly_degree=3,
            scaling=scaling,
            sample_start=str(weekly.index[0].date()),
            sample_end=str(weekly.index[-1].date()),
        )
        fits[scaling] = fit
        report[f"fit_{scaling}"] = {
            "eigenvalues": [float(x) for x in fit.eigenvalues],
            "sqrt_eigenvalues": [float(np.sqrt(x)) for x in fit.eigenvalues],
            "explained_variance": [round(float(x), 4) for x in fit.explained],
            "V_min": float(fit.V.min()),
            "V_max": float(fit.V.max()),
            "V_abs_max": float(np.abs(fit.V).max()),
        }

    report["figure3_axis_published"] = {"low": -0.005, "high": 0.008}

    # ---- 6. initial curve ----------------------------------------------
    v_row = params.loc[params.index <= pd.Timestamp(VALUATION_DATE)].iloc[-1]
    f0 = yc.svensson_interval_forward(v_row.values, N_GRID, DTAU)
    y0 = yc.svensson_spot(v_row.values, TENORS)
    # the simulator's own day-0 curve must equal the SNB curve
    day0_gap_bp = float(np.abs(yc.forwards_to_yields(f0, DTAU) - y0).max() * 1e4)
    report["initial_curve"] = {
        "date": str(v_row.name.date()),
        "simulator_vs_snb_gap_bp": day0_gap_bp,
        "y_1m_pct": round(float(y0[0]) * 100, 4),
        "y_1y_pct": round(float(y0[11]) * 100, 4),
        "y_5y_pct": round(float(y0[59]) * 100, 4),
        "y_10y_pct": round(float(y0[119]) * 100, 4),
        "y_15y_pct": round(float(y0[179]) * 100, 4),
    }

    # ---- 7. term premium (Version B) -----------------------------------
    # The paper sets P = Q, so the simulator treats today's forward curve as
    # its forecast. On this date that flattens the expected curve and about
    # half of scenarios end up inverted. Version B adds a constant market
    # price of risk, chosen so the expected curve in 5 years equals today's.
    tp = yc.calibrate_term_premium(
        f0, y0, fits["sqrt"],
        n_steps=60, dt=DTAU, dtau=DTAU, n_output=N_OUTPUT)
    phi = tp["phi"]
    report["term_premium"] = {
        "phi": [round(float(x), 4) for x in phi],
        "calibration_max_error_bp": round(tp["max_error_bp"], 2),
        "calibration_mean_error_bp": round(tp["mean_error_bp"], 2),
        "expected_slope_5y_pp": {
            "today": round(float(y0[179] - y0[0]) * 100, 3),
            "paper_P_eq_Q": round(float(tp["expected_curve_paper"][179]
                                        - tp["expected_curve_paper"][0]) * 100, 3),
            "term_premium": round(float(tp["expected_curve"][179]
                                        - tp["expected_curve"][0]) * 100, 3),
        },
    }

    # ---- 8. simulate all three variants --------------------------------
    #   sqrt         - the paper's model, P = Q, square-root scaling
    #   eigenvalue   - as literally printed, kept for the D1 comparison
    #   term_premium - sqrt scaling plus the Version B market price of risk
    variants = (
        ("sqrt", fits["sqrt"], None),
        ("eigenvalue", fits["eigenvalue"], None),
        ("term_premium", fits["sqrt"], phi),
    )
    for scaling, fit_v, mpr in variants:
        paths = yc.simulate_forward_paths(
            f0, fit_v,
            n_scenarios=ARGS.scenarios, n_steps=60, dt=DTAU, dtau=DTAU,
            n_output=N_OUTPUT, seed=ARGS.seed, market_price_of_risk=mpr,
        )
        yields = yc.forwards_to_yields(paths, DTAU)
        short_T = yields[:, -1, 0]
        short_0 = float(yields[0, 0, 0])
        slope_T = (yields[:, -1, N_OUTPUT - 1] - yields[:, -1, 0]) * 100
        report[f"sim_{scaling}"] = {
            "curve_shape_T": {
                "share_inverted": round(float((slope_T < 0).mean()), 4),
                "slope_mean_pp": round(float(slope_T.mean()), 3),
                "slope_sd_pp": round(float(slope_T.std()), 3),
            },
            "short_rate_t0_pct": round(short_0 * 100, 4),
            "share_above_minus1pct": round(float((short_T > -0.01).mean()), 4),
            "share_rising": round(float((short_T > short_0).mean()), 4),
            "short_T_pct": {
                "q05": round(float(np.quantile(short_T, 0.05)) * 100, 3),
                "q25": round(float(np.quantile(short_T, 0.25)) * 100, 3),
                "median": round(float(np.median(short_T)) * 100, 3),
                "q75": round(float(np.quantile(short_T, 0.75)) * 100, 3),
                "q95": round(float(np.quantile(short_T, 0.95)) * 100, 3),
            },
            "mean_curve_T_pct": {
                "1m": round(float(yields[:, -1, 0].mean()) * 100, 3),
                "5y": round(float(yields[:, -1, 59].mean()) * 100, 3),
                "15y": round(float(yields[:, -1, 179].mean()) * 100, 3),
            },
        }
        if ARGS.save_paths:
            # Forward curves only. Yields and discount factors derive from
            # these in one call each, so storing them too triples the file
            # for no information gain.
            np.savez_compressed(
                os.path.join(OUT, f"scenarios_{scaling}.npz"),
                forwards=paths.astype(np.float32),
                tenors=TENORS[:N_OUTPUT],
                valuation_date=np.array(VALUATION_DATE),
                scaling=np.array(scaling),
                seed=np.array(ARGS.seed),
                n_scenarios=np.array(ARGS.scenarios),
            )
        else:
            # A small sample, enough for the figures and eyeballing.
            np.savez_compressed(
                os.path.join(OUT, f"sample_{scaling}.npz"),
                forwards=paths[:400].astype(np.float32),
                tenors=TENORS[:N_OUTPUT],
                valuation_date=np.array(VALUATION_DATE),
                scaling=np.array(scaling),
                seed=np.array(ARGS.seed),
            )

    # ---- 9. drift variants ---------------------------------------------
    a_trap = yc.hjm_drift(fits["sqrt"].V, DTAU)
    a_riem = yc.hjm_drift_riemann(fits["sqrt"].V, DTAU)
    report["drift_variants"] = {
        "max_abs_diff": float(np.abs(a_trap - a_riem).max()),
        "max_abs_trapezoid": float(np.abs(a_trap).max()),
        "relative": float(np.abs(a_trap - a_riem).max()
                          / np.abs(a_trap).max()),
    }

    fit_path = os.path.join(OUT, "fit_sqrt.npz")
    np.savez(fit_path,
             V=fits["sqrt"].V, V_raw=fits["sqrt"].V_raw,
             eigenvalues=fits["sqrt"].eigenvalues, tenors=TENORS, f0=f0, y0=y0,
             term_premium_phi=phi, valuation_date=np.array(VALUATION_DATE),
             dtau=np.array(DTAU), n_steps=np.array(60),
             n_output=np.array(N_OUTPUT))
    np.savez(os.path.join(OUT, "fit_eig.npz"),
             V=fits["eigenvalue"].V, V_raw=fits["eigenvalue"].V_raw,
             eigenvalues=fits["eigenvalue"].eigenvalues, tenors=TENORS)

    # ---- 10. curve features for the network ([M] Section 4.2.1) --------
    # Fitted on TRAINING scenarios (their own seed), checked on the
    # validation scenarios generated above with --seed.  The two seeds
    # must differ, or the check is meaningless.
    if ARGS.feature_seed == ARGS.seed:
        raise SystemExit("--feature-seed must differ from --seed")
    train = yc.load_scenarios(fit_path, n_scenarios=ARGS.feature_scenarios,
                              seed=ARGS.feature_seed, variant=FEATURE_VARIANT)
    fmap = yc.fit_curve_features(train.yields.reshape(-1, N_OUTPUT), 3)
    valid = yc.load_scenarios(fit_path, n_scenarios=ARGS.scenarios,
                              seed=ARGS.seed, variant=FEATURE_VARIANT)
    vy = valid.yields.reshape(-1, N_OUTPUT)
    recon = fmap.inverse(fmap.transform(vy))
    err_bp = np.abs(recon - vy) * 1e4
    report["curve_features"] = {
        "fitted_on_variant": FEATURE_VARIANT,
        "fitted_on_curves": int(fmap.n_curves),
        "feature_seed": ARGS.feature_seed,
        "explained_variance": [round(float(x), 5) for x in fmap.explained],
        "explained_total": round(float(fmap.explained.sum()), 5),
        "validation_reconstruction_bp": {
            "mean": round(float(err_bp.mean()), 3),
            "p99": round(float(np.quantile(err_bp, 0.99)), 3),
            "max": round(float(err_bp.max()), 3),
        },
    }
    np.savez(os.path.join(OUT, "curve_features.npz"),
             mean=fmap.mean, components=fmap.components,
             explained=fmap.explained, n_curves=np.array(fmap.n_curves),
             feature_seed=np.array(ARGS.feature_seed))

    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps(rep, indent=2))
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    kind = "scenarios" if ARGS.save_paths else "sample"
    print(f"\nwritten to {OUT}/: report.json, fit_sqrt.npz, fit_eig.npz, "
          f"{kind}_sqrt.npz, {kind}_eigenvalue.npz, {kind}_term_premium.npz, "
          f"curve_features.npz")

    # Presentation only: an Excel copy of the files above.  Runs after
    # everything is saved, so it can never change or block a result.
    try:
        import export_layer1
        xlsx = export_layer1.write_workbook(OUT, seed=ARGS.seed,
                                            n_scenarios=ARGS.scenarios)
        print(f"written {xlsx}")
    except Exception as err:                       # noqa: BLE001
        print(f"(Excel export skipped: {err})")