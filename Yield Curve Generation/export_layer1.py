"""Layer 1 results as an Excel workbook.

Reads what run_fit.py has already written to its output folder and lays it out
for reading: no fitting happens here, and nothing here changes a result.

    python export_layer1.py --out output_paper

run_fit.py also calls write_workbook() at the end, so a fresh fit writes the
workbook by itself.  Running this file on its own makes the workbook from an
existing output folder without refitting.

The .npz files are the hand-off between layers (Layers 2, 3 and 5 load the
fitted model from them) and stay as they are.  This workbook shows what is in
them.  Scenario statistics use the same 4,000 scenarios as report.json,
rebuilt exactly from the saved fit and the seed.

Writes <out>/layer1_results.xlsx.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import yieldcurve as yc

DTAU = 1.0 / 12.0
N_OUT = 180
VARIANTS = (
    ("sqrt", "Paper model (P = Q)"),
    ("term_premium", "Version B (term premium)"),
    ("eigenvalue", "Printed formula (rejected, D1)"),
)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
SHOW_TENORS = [1, 3, 6] + [12 * k for k in range(1, 16)]   # months


# ================================================================ formatting
# Shared by the later layers' exporters, which import these two functions.

def style_sheet(ws, *, header_row: int = 1, widths: dict | None = None,
                number_format: str | None = None, freeze: str | None = "B2"):
    """Bold header, sensible column widths, a frozen header row."""
    from openpyxl.styles import Alignment, Font, PatternFill
    fill = PatternFill("solid", start_color="1F4E79")
    for cell in ws[header_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[header_row].height = 32
    for col in ws.columns:
        letter = col[0].column_letter
        longest = max((len(str(c.value)) for c in col if c.value is not None),
                      default=8)
        ws.column_dimensions[letter].width = min(max(10, longest + 2), 60)
    for letter, w in (widths or {}).items():
        ws.column_dimensions[letter].width = w
    if number_format:
        for row in ws.iter_rows(min_row=header_row + 1):
            for c in row:
                if isinstance(c.value, float):
                    c.number_format = number_format
    if freeze:
        ws.freeze_panes = freeze


def add_line_chart(ws, *, title: str, x_col: int, y_cols, first_row: int,
                   last_row: int, anchor: str, y_title: str = "",
                   x_title: str = "", width: float = 18, height: float = 9,
                   x_major: float | None = None, x_format: str = "0",
                   y_format: str = "0.0", legend: bool = True):
    """A native Excel chart of lines against a numeric x column.

    A scatter chart with lines rather than a line chart, so the x axis is a
    real number axis (one tick a year) in Excel and LibreOffice alike."""
    from openpyxl.chart import Reference, ScatterChart, Series
    ch = ScatterChart()
    ch.title, ch.width, ch.height = title, width, height
    ch.style = 2
    if y_title:                             # an empty title shows as "None"
        ch.y_axis.title = y_title
    if x_title:
        ch.x_axis.title = x_title
    if legend:
        ch.legend.position = "b"            # long series names fit underneath
    else:
        ch.legend = None
    # openpyxl 3.1 hides both axes in Excel unless told otherwise
    ch.x_axis.delete = ch.y_axis.delete = False
    ch.x_axis.tickLblPos = "low"            # labels under the plot, not at y = 0
    ch.x_axis.number_format, ch.y_axis.number_format = x_format, y_format
    if x_major:
        ch.x_axis.majorUnit = x_major
    xs = [c.value for c in ws.iter_cols(min_col=x_col, max_col=x_col,
                                        min_row=first_row, max_row=last_row).__next__()]
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = min(xs), max(xs)
    x = Reference(ws, min_col=x_col, min_row=first_row, max_row=last_row)
    for col in y_cols:
        y = Reference(ws, min_col=col, min_row=first_row - 1, max_row=last_row)
        ser = Series(y, x, title_from_data=True)
        ser.marker.symbol = "none"
        ser.smooth = False
        ch.series.append(ser)
    ws.add_chart(ch, anchor)


# ===================================================================== data

def _scenario_yields(out_dir: str, variant: str, *, n_scenarios: int,
                     seed: int) -> np.ndarray:
    """The run_fit.py scenarios, rebuilt bit for bit from the fit and the seed."""
    fit_path = os.path.join(out_dir, "fit_sqrt.npz")
    if variant in ("sqrt", "term_premium"):
        return yc.load_scenarios(fit_path, n_scenarios=n_scenarios, seed=seed,
                                 variant=variant).yields
    # the rejected variant: same simulator, the other volatility file
    z, e = np.load(fit_path), np.load(os.path.join(out_dir, "fit_eig.npz"))
    fit = yc.HJMFit(V=e["V"], V_raw=e["V_raw"], eigenvalues=e["eigenvalues"],
                    explained=np.full(e["V"].shape[1], np.nan),
                    tenors=e["tenors"], scaling="eigenvalue", poly_degree=3,
                    n_observations=0, sample_start="", sample_end="")
    paths = yc.simulate_forward_paths(
        z["f0"], fit, n_scenarios=n_scenarios, n_steps=int(z["n_steps"]),
        dt=DTAU, dtau=DTAU, n_output=int(z["n_output"]), seed=seed)
    return yc.forwards_to_yields(paths, DTAU)


def _flatten(d, prefix=""):
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            rows += _flatten(v, f"{prefix}/{k}" if prefix else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            rows += _flatten(v, f"{prefix}[{i + 1}]")
    else:
        rows.append((prefix, d))
    return rows


def _summary_rows(rep: dict) -> list:
    """The report, in plain words.  (section, measure, value, unit)"""
    g = lambda *ks: _get(rep, ks)
    r = [
        ("Data", "SNB file: rows after cleaning", g("file", "rows_after_cleaning"), "days"),
        ("Data", "SNB file: first date", g("file", "first_date"), ""),
        ("Data", "SNB file: last date", g("file", "last_date"), ""),
        ("Data", "PCA window: first date", g("history", "first"), ""),
        ("Data", "PCA window: last date", g("history", "last"), ""),
        ("Data", "PCA window: weekly observations", g("history", "weekly_observations"), "weeks"),
        ("Start", "Valuation date", g("initial_curve", "date"), ""),
        ("Start", "1-month yield", g("initial_curve", "y_1m_pct"), "%"),
        ("Start", "1-year yield", g("initial_curve", "y_1y_pct"), "%"),
        ("Start", "5-year yield", g("initial_curve", "y_5y_pct"), "%"),
        ("Start", "10-year yield", g("initial_curve", "y_10y_pct"), "%"),
        ("Start", "15-year yield", g("initial_curve", "y_15y_pct"), "%"),
        ("Start", "Model curve vs SNB curve, largest gap",
         g("initial_curve", "simulator_vs_snb_gap_bp"), "bp"),
    ]
    ev = g("fit_sqrt", "explained_variance") or []
    for j, x in enumerate(ev):
        r.append(("Volatility", f"Factor {j + 1}: share of variance explained",
                  x, "share"))
    r.append(("Volatility", "Three factors together", sum(ev) if ev else None, "share"))
    phi = g("term_premium", "phi") or []
    for j, x in enumerate(phi):
        r.append(("Version B", f"Market price of risk, factor {j + 1}", x, ""))
    r += [
        ("Version B", "Calibration error, mean", g("term_premium", "calibration_mean_error_bp"), "bp"),
        ("Version B", "Calibration error, largest", g("term_premium", "calibration_max_error_bp"), "bp"),
        ("Checks", "Discount factors: two routes agree to",
         g("check_discount", "max_abs_diff"), "absolute"),
        ("Checks", "Forward rates: two routes agree to (mean)",
         g("check_forward_routes", "mean_abs_diff_bp"), "bp"),
        ("Checks", "Yields to forwards and back", g("check_roundtrip_bp"), "bp"),
        ("Checks", "Drift: trapezoid vs Riemann, relative", g("drift_variants", "relative"), "share"),
        ("Network input", "Curve features fitted on", g("curve_features", "fitted_on_variant"), ""),
        ("Network input", "Variance kept by 3 features", g("curve_features", "explained_total"), "share"),
        ("Network input", "Reconstruction error, mean",
         g("curve_features", "validation_reconstruction_bp", "mean"), "bp"),
        ("Network input", "Reconstruction error, 99th percentile",
         g("curve_features", "validation_reconstruction_bp", "p99"), "bp"),
    ]
    return r


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ================================================================== workbook

def write_workbook(out_dir: str, *, seed: int = 20221230,
                   n_scenarios: int = 4000) -> str:
    rep = json.load(open(os.path.join(out_dir, "report.json")))
    z = np.load(os.path.join(out_dir, "fit_sqrt.npz"))
    eig = np.load(os.path.join(out_dir, "fit_eig.npz"))
    feat_path = os.path.join(out_dir, "curve_features.npz")
    feat = np.load(feat_path) if os.path.isfile(feat_path) else None
    tenors, y0, f0 = z["tenors"], z["y0"], z["f0"]
    months = np.arange(1, len(tenors) + 1)

    sims = {v: _scenario_yields(out_dir, v, n_scenarios=n_scenarios, seed=seed)
            for v, _ in VARIANTS}

    sheets = {}

    # ---- Read me
    sheets["Read me"] = pd.DataFrame({"": [
        "Layer 1 - the yield curve model (HJM-PCA, [E] 2.7 / [M] 2.2)",
        f"Valuation date {str(z['valuation_date'])}; "
        f"{n_scenarios:,} scenarios, seed {seed}, 60 months ahead.",
        "",
        "Summary - the headline numbers from report.json, in plain words.",
        "Scenario summary - the three model variants side by side.",
        "Starting curve - today's curve at every monthly tenor (figure: fig_curves).",
        "Volatility factors - the three shocks that move the curve (figure: fig_loadings).",
        "Short rate fan - the 1-month yield over 60 months (figure: fig_shortrate).",
        "Curve in 5 years - where the whole curve ends up.",
        "Example paths - ten single scenarios, to see what one future looks like.",
        "Curve features - the 3 numbers the network sees instead of 180 yields.",
        "All values - every number in report.json, unedited.",
        "",
        "Units: yields and rates in % unless stated; bp = 0.01 percentage points.",
        "Months 181-240 of the starting curve are the roll-down extension "
        "(decision D4): the simulation needs them, the bank never sees them.",
        "",
        "Source files: report.json, fit_sqrt.npz, fit_eig.npz, curve_features.npz. "
        "Scenario numbers are rebuilt from fit_sqrt.npz and the seed, so they "
        "match report.json exactly.",
    ]})

    # ---- Summary
    sheets["Summary"] = pd.DataFrame(
        _summary_rows(rep), columns=["Section", "Measure", "Value", "Unit"])

    # ---- Scenario summary (variants as columns)
    rows = []
    for v, label in VARIANTS:
        y = sims[v]
        short, slope = y[:, -1, 0] * 100, (y[:, -1, N_OUT - 1] - y[:, -1, 0]) * 100
        rows.append({
            "Variant": label,
            "1m yield at 5y, 5th pct (%)": np.quantile(short, 0.05),
            "1m yield at 5y, median (%)": np.median(short),
            "1m yield at 5y, 95th pct (%)": np.quantile(short, 0.95),
            "Share with 1m yield above -1%": (short > -1.0).mean(),
            "Share where 1m yield rose": (short > y[0, 0, 0] * 100).mean(),
            "Share of curves inverted at 5y": (slope < 0).mean(),
            "15y minus 1m at 5y, mean (pp)": slope.mean(),
            "15y minus 1m at 5y, sd (pp)": slope.std(),
            "Mean 1m yield at 5y (%)": short.mean(),
            "Mean 5y yield at 5y (%)": y[:, -1, 59].mean() * 100,
            "Mean 15y yield at 5y (%)": y[:, -1, N_OUT - 1].mean() * 100,
        })
    sheets["Scenario summary"] = pd.DataFrame(rows).set_index("Variant").T \
        .reset_index().rename(columns={"index": "Measure"})

    # ---- Starting curve
    disc = np.exp(-tenors * y0)
    sheets["Starting curve"] = pd.DataFrame({
        "Month": months, "Tenor (years)": tenors,
        "Spot yield (%)": y0 * 100, "Forward rate (%)": f0 * 100,
        "Discount factor": disc,
        "Used by the bank": np.where(months <= N_OUT, "yes", "roll-down only"),
    })

    # ---- Volatility factors
    vol = {"Month": months, "Tenor (years)": tenors}
    for j in range(z["V"].shape[1]):
        vol[f"Factor {j + 1}, used (bp/yr)"] = z["V"][:, j] * 1e4
    for j in range(z["V_raw"].shape[1]):
        vol[f"Factor {j + 1}, before smoothing (bp/yr)"] = z["V_raw"][:, j] * 1e4
    for j in range(eig["V"].shape[1]):
        vol[f"Factor {j + 1}, printed formula (bp/yr)"] = eig["V"][:, j] * 1e4
    sheets["Volatility factors"] = pd.DataFrame(vol)
    ev = rep.get("fit_sqrt", {})
    sheets["_vol_table"] = pd.DataFrame({
        "Factor": [1, 2, 3],
        "Eigenvalue (annualised)": np.asarray(z["eigenvalues"]),
        "Share of variance explained": ev.get("explained_variance", [None] * 3),
    })

    # ---- Short rate fan
    fan = {"Month": np.arange(61), "Year": np.arange(61) / 12}
    for v, label in VARIANTS[:2]:
        s = sims[v][:, :, 0] * 100
        short = "Paper" if v == "sqrt" else "Version B"
        for q in QUANTILES:
            fan[f"{short} {int(q * 100)}th pct (%)"] = np.quantile(s, q, axis=0)
        fan[f"{short} mean (%)"] = s.mean(axis=0)
    sheets["Short rate fan"] = pd.DataFrame(fan)

    # ---- Curve in 5 years
    idx = np.array(SHOW_TENORS) - 1
    cin = {"Tenor": [f"{m}m" if m < 12 else f"{m // 12}y" for m in SHOW_TENORS],
           "Tenor (years)": np.array(SHOW_TENORS) / 12.0,
           "Today (%)": y0[idx] * 100}
    for v, label in VARIANTS[:2]:
        yT = sims[v][:, -1, :][:, idx] * 100
        short = "Paper" if v == "sqrt" else "Version B"
        cin[f"{short}, mean in 5y (%)"] = yT.mean(axis=0)
        for q in (0.05, 0.50, 0.95):
            cin[f"{short}, {int(q * 100)}th pct in 5y (%)"] = np.quantile(yT, q, axis=0)
    sheets["Curve in 5 years"] = pd.DataFrame(cin)

    # ---- Example paths
    ex = {"Month": np.arange(61)}
    for i in range(10):
        ex[f"Scenario {i + 1}, 1m yield (%)"] = sims["sqrt"][i, :, 0] * 100
    for i in range(10):
        ex[f"Scenario {i + 1}, 10y yield (%)"] = sims["sqrt"][i, :, 119] * 100
    sheets["Example paths"] = pd.DataFrame(ex)

    # ---- Curve features
    if feat is not None:
        cf = {"Month": np.arange(1, N_OUT + 1), "Tenor (years)": tenors[:N_OUT],
              "Average curve (%)": feat["mean"] * 100}
        for j in range(feat["components"].shape[0]):
            cf[f"Feature {j + 1} loading"] = feat["components"][j]
        sheets["Curve features"] = pd.DataFrame(cf)

    # ---- All values
    sheets["All values"] = pd.DataFrame(_flatten(rep), columns=["Key", "Value"])

    # ================================================================ write
    path = os.path.join(out_dir, "layer1_results.xlsx")
    vol_table = sheets.pop("_vol_table")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False,
                        header=(name != "Read me"))
        vol_start = len(sheets["Volatility factors"]) + 3
        vol_table.to_excel(xw, sheet_name="Volatility factors", index=False,
                           startrow=vol_start)

        wb = xw.book
        wb["Read me"].column_dimensions["A"].width = 110
        from openpyxl.styles import Font
        wb["Read me"]["A1"].font = Font(bold=True, size=13)
        style_sheet(wb["Summary"], freeze="A2", widths={"B": 46})
        style_sheet(wb["Scenario summary"], number_format="0.000", freeze="B2",
                    widths={"A": 34, "B": 22, "C": 24, "D": 28})
        style_sheet(wb["Starting curve"], number_format="0.0000")
        style_sheet(wb["Volatility factors"], number_format="0.00")
        style_sheet(wb["Short rate fan"], number_format="0.000")
        style_sheet(wb["Curve in 5 years"], number_format="0.000")
        style_sheet(wb["Example paths"], number_format="0.000")
        if "Curve features" in wb.sheetnames:
            style_sheet(wb["Curve features"], number_format="0.0000")
        style_sheet(wb["All values"], freeze="A2", widths={"A": 60, "B": 24})

        n = len(sheets["Starting curve"])
        yrs = dict(x_major=1, x_format="0")      # one tick a year
        add_line_chart(wb["Starting curve"], title="Starting curve",
                       x_col=2, y_cols=[3, 4], first_row=2, last_row=N_OUT + 1,
                       anchor="H2", y_title="%", x_title="tenor (years)",
                       y_format="0.00", **yrs)
        add_line_chart(wb["Volatility factors"], title="Volatility factors, as used",
                       x_col=2, y_cols=[3, 4, 5], first_row=2, last_row=n + 1,
                       anchor="L2", y_title="bp per year", x_title="tenor (years)",
                       y_format="0", **yrs)
        add_line_chart(wb["Short rate fan"], title="1-month yield, paper model",
                       x_col=2, y_cols=[3, 4, 5, 6, 7], first_row=2, last_row=62,
                       anchor="Q2", y_title="%", x_title="year", **yrs)
        add_line_chart(wb["Curve in 5 years"], title="Mean curve today and in 5 years",
                       x_col=2, y_cols=[3, 4, 8], first_row=2,
                       last_row=len(SHOW_TENORS) + 1, anchor="M2", y_title="%",
                       x_title="tenor (years)", **yrs)
        add_line_chart(wb["Example paths"], title="1-month yield, ten scenarios",
                       x_col=1, y_cols=list(range(2, 12)), first_row=2,
                       last_row=62, anchor="W2", y_title="%", x_title="month",
                       x_major=12, legend=False)
        if "Curve features" in wb.sheetnames:
            add_line_chart(wb["Curve features"], title="Curve features",
                           x_col=2, y_cols=[4, 5, 6], first_row=2,
                           last_row=N_OUT + 1, anchor="H2", x_title="tenor (years)",
                           y_format="0.00", **yrs)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="output_paper",
                    help="the folder run_fit.py wrote to")
    ap.add_argument("--seed", type=int, default=20221230,
                    help="the --seed run_fit.py used")
    ap.add_argument("--scenarios", type=int, default=4000,
                    help="the --scenarios run_fit.py used")
    a = ap.parse_args()
    print(f"written {write_workbook(a.out, seed=a.seed, n_scenarios=a.scenarios)}")
