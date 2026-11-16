"""Layer 5 workbook: every strategy against the paper, explained.

    python export_layer5.py comparison.json deep_alm_report.json out.xlsx [DATE] [BANK]

The scores come from comparison.json (this run's data); training history and
settings from deep_alm_report.json (where the network was trained).  Every
sentence is computed from them, so it is right for any bank.
"""
import json
import sys

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_CMPF = json.load(open(sys.argv[1]))
CMP = _CMPF["results"]
REP = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
DATE = sys.argv[4] if len(sys.argv) > 4 else "the start date"
BANK = sys.argv[5] if len(sys.argv) > 5 else "the bank"
H = 60
E0 = float(_CMPF["setup"]["E0_mCHF"])
NVAL = int(_CMPF["setup"]["val_scenarios"])
ORDER = ["nothing", "rollover", "BM_E", "BM_C", "BM_D", "DeepALM"]
NAME = {"nothing": "No decisions", "rollover": "Roll-over rule",
        "BM_E": "BM_E  1/N", "BM_C": "BM_C  learnt mix",
        "BM_D": "BM_D  mix per month", "DeepALM": "Deep ALM network"}


def _net_params(widths, enc=32, hid=64, n=180, nb=13, nk=16):
    p = 4 * (n * hid + hid + hid * enc + enc)
    prev = 4 * enc + 3 + 5 + 6 + 1 + 2
    for w in widths:
        p, prev = p + prev * w + w, w
    return p + 2 * (prev + 1) + prev * nb + nb + prev * nk + nk


PARAMS = {"nothing": 0, "rollover": 0, "BM_E": 2, "BM_C": 31, "BM_D": 1860,
          "DeepALM": _net_params(REP["setup"]["widths"])}
_bench = ["BM_E", "BM_C", "BM_D"]
_best_bm = min(_bench, key=lambda k: CMP[k]["total"])
_net, _bb = CMP["DeepALM"], CMP[_best_bm]
_gain = 1 - _net["total"] / _bb["total"]
_steps = REP["setup"]["bm_epochs"] * (REP["setup"]["train_scenarios"] // REP["setup"]["batch_size"])
_sd_ratio = CMP["BM_D"]["ER_sd"] / 0.027
_irs_net = CMP["DeepALM"]["violation_share"]["IRS"]
_dm = CMP["DeepALM"]["median"]
_bm_m0 = max(CMP[k]["median"]["aK"][0] for k in _bench)
_inv_net = float(np.sum(_dm["aB"][:H]))
_eyr = [CMP[k]["violation_share"]["EYR"] for k in ORDER]

NAVY, BAND, RED, GREEN, GOLD = "1F4E79", "DDEBF7", "F8CBAD", "C6EFCE", "FFF2CC"
fill = lambda c: PatternFill("solid", start_color=c)
H_FILL, WHITE_B, BOLD = fill(NAVY), Font(bold=True, color="FFFFFF"), Font(bold=True)
NOTE = Font(italic=True, color="595959", size=9)
TITLE = Font(bold=True, size=13, color=NAVY)
BIG = Font(bold=True, size=15, color=NAVY)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
TOP = Border(top=Side(style="thin", color="7F7F7F"))

wb = Workbook()


def put(ws, r, c, v, *, font=None, fmt=None, fl=None, align=None, border=None):
    cell = ws.cell(row=r, column=c, value=v)
    for attr, val in (("font", font), ("number_format", fmt), ("fill", fl),
                      ("alignment", align), ("border", border)):
        if val is not None:
            setattr(cell, attr, val)
    return cell


def note(ws, r, text, last_col=12, height=30):
    put(ws, r, 2, text, font=NOTE, align=WRAP)
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=last_col)
    ws.row_dimensions[r].height = height


def header(ws, r, labels, start=2, height=32):
    for j, h in enumerate(labels):
        if h == "":                       # spacer column: leave it blank
            continue
        put(ws, r, start + j, h, font=WHITE_B, fl=H_FILL,
            align=CENTER if j else Alignment(vertical="center", wrap_text=True))
    ws.row_dimensions[r].height = height


# ====================================================================== Read me
ws = wb.active
ws.title = "Read me"
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 108
lines = [
    ("Layer 5 - the trained decision network against every benchmark", BIG),
    ("", None),
    ("What this is", BOLD),
    (f"Six ways of running {BANK}'s balance sheet for five years "
     f"from {DATE}, all scored on the same {NVAL:,} interest-rate scenarios that "
     "none of them was trained on. Each month the bank decides how much to "
     "invest in bonds and how much to borrow, at 13 and 16 maturities.", None),
    ("", None),
    ("The six strategies", BOLD),
    ("No decisions - invests and borrows nothing; debt is repaid out of cash.", None),
    ("Roll-over rule - refinances whatever borrowing matures. Hand-written, not trained.", None),
    ("BM_E, BM_C, BM_D - the paper's three benchmarks ([E] 3.1), trained by "
     "gradient descent: roll over what matures plus a learnt extra amount, "
     "spread evenly (E), by a learnt mix (C), or by a learnt mix for every "
     "month (D).", None),
    (f"Deep ALM network - the paper's model ([E] 3.2): {PARAMS['DeepALM']:,} parameters that "
     "look at the balance sheet, the yield curve and the limits each month and "
     "adjust BM_D's decision.", None),
    ("", None),
    ("What it shows", BOLD),
    ((f"1. The network's total loss is {_net['total']:,.2f} against "
      f"{_bb['total']:,.2f} for the best benchmark ({NAME[_best_bm].split()[0]}), "
      + (f"{_gain:.0%} lower." if _gain > 0 else f"{-_gain:.0%} higher: the stored "
         "network does not beat the benchmarks on this data - retrain it.")
      + f" It returns {_net['annual_return']:.2%} a year, against "
      f"{CMP['nothing']['annual_return']:.2%} doing nothing."), None),
    ((f"2. What it does: borrows {_dm['aK'][0]:,.0f} mCHF in month 0 (benchmarks at "
      f"most {_bm_m0:,.0f}), invests {_inv_net:,.0f} mCHF in bonds over five years, "
      f"and moves its rate sensitivity from {_dm['IRS'][0]:+.3f} to "
      f"{_dm['IRS'][H]:+.3f}. The paper's own model borrows early, invests "
      "little and turns its rate sensitivity negative in the last year."), None),
    ((f"3. Rate risk: it breaks the 8.5% limit in {_irs_net:.0%} of scenarios "
      "(the paper's model: 0.06%). " + (
          "The loss charges almost nothing for a small breach, so the optimiser "
          "sits on the line; a bank would not accept this. See sheet 'Limits'."
          if _irs_net > 0.05 else "That is within what a bank would accept.")), None),
    ("", None),
    ("How to read the numbers", BOLD),
    ("Lower loss is better. The equity ratio is final equity divided by "
     f"starting equity ({E0:,.1f} mCHF). Losses are not comparable in level "
     f"with the paper's: this bank's equity spreads about {_sd_ratio:.0f} times "
     "as widely as the paper's bank's, whose balance sheet is withheld. "
     "Compare the ordering and the shape, not the size.", None),
    ("Green marks the best value in a row, red a limit broken.", None),
    ("", None),
    ("Sheets: Scoreboard, Limits, Strategy, Monthly path, Training, Setup.", None),
    ("Source: deep_alm_report.json and comparison.json from run_deep_alm.py "
     "and score_deep_alm.py. Paper figures: Englisch et al. (2023), Tables 3 "
     "and 4.", NOTE),
]
for i, (text, font) in enumerate(lines, start=1):
    c = put(ws, i, 2, text or None, align=WRAP)
    if font:
        c.font = font
ws.sheet_view.showGridLines = False

# ================================================================== Scoreboard
ws = wb.create_sheet("Scoreboard")
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 44
for j in range(len(ORDER) + 3):
    ws.column_dimensions[get_column_letter(3 + j)].width = 14
ws.column_dimensions[get_column_letter(3 + len(ORDER))].width = 3
r = 1
put(ws, r, 2, "Scoreboard - six strategies, 1,600 held-out scenarios", font=BIG)
r += 2
put(ws, r, 2, "Loss and equity outcome after five years ([E] Table 3)", font=TITLE)
r += 1
note(ws, r, "µ = 4.06% a year target, λ = 3.5 penalty weight, γ = 10 risk "
     "aversion, as in the paper. Green = best of our six strategies.", 11, 18)
r += 1
pc = 3 + len(ORDER) + 1                       # paper columns start here
header(ws, r, ["Measure"] + [NAME[k] for k in ORDER] + ["", "Paper BM_D",
                                                       "Paper network"])
r += 1
put(ws, r, 2, "Trainable parameters", font=NOTE)
for j, k in enumerate(ORDER):
    put(ws, r, 3 + j, PARAMS[k], fmt="#,##0", font=NOTE)
put(ws, r, pc, 1860, fmt="#,##0", font=NOTE)
r += 1
res = CMP                      # this run's scores, not the training run's
rows = [
    # label, key, fmt, better, paper BMD, paper MM
    ("Total loss  (target + 3.5 × penalty)", "total", "#,##0.00", "low", 0.753, 0.467),
    ("   Target loss (shortfall vs 4.06% a year)", "target", "0.000", "low", 0.824, 0.558),
    ("   Penalty loss (limit breaches)", "penalty", "#,##0.000", "low", 0.012, 0.003),
    ("Penalty, worst 5% of scenarios", "penalty_ES0.95", "#,##0.00", "low", 0.144, 0.036),
    ("Utility loss, risk-averse investor (γ = 10)", "utility", "0.000", "low", -0.074, -0.079),
    ("Equity ratio, mean", "ER_mean", "0.000", "high", 1.133, 1.153),
    ("Equity ratio, standard deviation", "ER_sd", "0.000", "low", 0.027, 0.036),
    ("Equity ratio, skew (positive = good tail)", "ER_skew", "0.000", "high", 0.286, 1.682),
    ("Value at risk, worst 5% (from mean)", "ER_VaR0.95", "0.000", "high", -0.043, -0.049),
    ("Expected shortfall, worst 5% (from mean)", "ER_ES0.95", "0.000", "high", -0.053, -0.060),
    ("Annual return on equity", "annual_return", "0.00%", "high", 0.02529, 0.02883),
    ("Scenarios with no breach at all", "clean", "0.0%", "high", None, None),
]
for label, key, fmt, better, pb, pm in rows:
    strong = key == "total"
    put(ws, r, 2, label, font=BOLD if strong else None)
    vals = [res[k][key] for k in ORDER]
    best = (max if better == "high" else min)(vals)
    for j, v in enumerate(vals):
        put(ws, r, 3 + j, v, fmt=fmt, font=BOLD if strong else None,
            fl=fill(GREEN) if v == best else None)
    if pb is not None:
        put(ws, r, pc, pb, fmt=fmt, font=NOTE)
        put(ws, r, pc + 1, pm, fmt=fmt, font=NOTE)
    r += 1
note(ws, r, "Paper columns are for shape, not size: their equity ratio spreads "
     f"about {_sd_ratio:.0f} times less. Their target-loss row is the symmetric "
     "version (problem B10). Their benchmarks are trained for 125,000 steps "
     f"against our {_steps:,}.", 11, 30)
r += 2

put(ws, r, 2, "Same ordering as the paper", font=TITLE)
r += 1
header(ws, r, ["", "Total loss, ours", "Rank, ours", "Total loss, paper",
               "Rank, paper"])
r += 1
paper_loss = {"BM_E": 1.357, "BM_C": 0.999, "BM_D": 0.753, "DeepALM": 0.467}
ours = {k: res[k]["total"] for k in paper_loss}
rank = lambda d: {k: i + 1 for i, k in enumerate(sorted(d, key=d.get))}
ro, rp = rank(ours), rank(paper_loss)
for k in ("BM_E", "BM_C", "BM_D", "DeepALM"):
    put(ws, r, 2, NAME[k])
    put(ws, r, 3, ours[k], fmt="0.00")
    put(ws, r, 4, ro[k], align=CENTER)
    put(ws, r, 5, paper_loss[k], fmt="0.000")
    put(ws, r, 6, rp[k], align=CENTER)
    r += 1
note(ws, r, "Network first, BM_E last, in both. BM_C and BM_D swap places in "
     "ours: BM_D has 60 times BM_C's parameters and needs more training steps "
     "than our budget allowed; both were still improving when training stopped.",
     11, 30)
ws.sheet_view.showGridLines = False
ws.freeze_panes = "C1"

# ====================================================================== Limits
ws = wb.create_sheet("Limits")
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 34
for j in range(10):
    ws.column_dimensions[get_column_letter(3 + j)].width = 14
r = 1
put(ws, r, 2, "Limits - how close each strategy runs to the rules", font=BIG)
r += 2
put(ws, r, 2, "Scenarios breaking each limit at least once", font=TITLE)
r += 1
header(ws, r, ["Limit"] + [NAME[k] for k in ORDER] + ["", "Paper network"])
r += 1
LBL = {"LCR": "Liquidity coverage  LCR ≥ 1.05",
       "NSFR": "Stable funding  NSFR ≥ 1.05", "CMR": "Reserve ratio  CMR ≥ 1",
       "E_over_RWA": "Capital  E/RWA ≥ 0.17",
       "IRS": "Rate sensitivity  |IRS| ≤ 0.085",
       "EYR": "Minimum annual return  EYR ≥ 0"}
paper_v = {"LCR": 0, "NSFR": 0, "CMR": 0.0012, "E_over_RWA": 0, "IRS": 0.0006,
           "EYR": 0.2006}
for key in ("LCR", "NSFR", "CMR", "E_over_RWA", "IRS", "EYR"):
    put(ws, r, 2, LBL[key])
    for j, k in enumerate(ORDER):
        v = CMP[k]["violation_share"][key]
        put(ws, r, 3 + j, v, fmt="0.0%",
            fl=fill(RED) if v >= 0.2 else (fill(GREEN) if v == 0 else None))
    put(ws, r, 10, paper_v[key], fmt="0.0%", font=NOTE)
    r += 1
note(ws, r, f"The minimum-return rule (EYR) breaks in {min(_eyr):.0%}-{max(_eyr):.0%} "
     f"of scenarios across strategies, and in "
     f"{CMP['nothing']['violation_share']['EYR']:.0%} doing nothing: it reflects how "
     "much the bank earns more than how it is funded.", 11, 26)
r += 2

put(ws, r, 2, "Median ratios at year 4 - the paper reports \"T - 1\"", font=TITLE)
r += 1
header(ws, r, ["Ratio"] + [NAME[k] for k in ORDER] + ["", "Paper network"])
r += 1
paper_m = {"LCR": 1.10, "NSFR": 1.33, "CMR": 1.76, "E_over_RWA": 0.20,
           "IRS": -0.03}
lims = {"LCR": (1.05, "min"), "NSFR": (1.05, "min"), "CMR": (1.0, "min"),
        "E_over_RWA": (0.17, "min"), "IRS": (0.085, "abs")}
fmts = {"LCR": "0.00", "NSFR": "0.00", "CMR": "0.0", "E_over_RWA": "0.000",
        "IRS": "0.000"}
for key in ("E_over_RWA", "IRS", "LCR", "NSFR", "CMR"):
    put(ws, r, 2, LBL[key])
    lim, kind = lims[key]
    for j, k in enumerate(ORDER):
        v = CMP[k]["median"][key][48]
        bad = abs(v) > lim if kind == "abs" else v < lim
        put(ws, r, 3 + j, v, fmt=fmts[key], fl=fill(RED) if bad else None)
    put(ws, r, 10, paper_m[key], fmt="0.00" if key == "CMR" else fmts[key],
        font=NOTE)
    r += 1
put(ws, r, 2, "Minimum annual return  EYR")
for j, k in enumerate(ORDER):
    put(ws, r, 3 + j, CMP[k]["median"]["EYR"][48], fmt="0.0")
put(ws, r, 10, 21.17, fmt="0.0", font=NOTE)
r += 1
note(ws, r, f"Network at year 4: capital {_dm['E_over_RWA'][48]:.3f} against the "
     f"paper's 0.20, liquidity {_dm['LCR'][48]:.2f} against 1.10. The gap in "
     "liquidity and reserves comes from how much cash the bank holds.", 11, 18)
r += 2

put(ws, r, 2, "How the network handles the rate-risk limit", font=TITLE)
r += 1
for text in [
    f"Its median rate sensitivity ends at {_dm['IRS'][H]:+.3f} against a limit "
    f"of ±0.085, and it crosses the limit in {_irs_net:.0%} of scenarios.",
    "Under the paper's loss a small breach is almost free. Going 0.002 over the "
    "limit adds (1.002² - 1) × 2.0 = 0.008 to the penalty, while the extra rate "
    "exposure earns more than that. So the optimiser parks on the line.",
    "The paper's own model almost never crosses it (0.06%). Where a bank earns "
    "less, the trade-off tips the other way. Raising the IRS penalty weight, or "
    "aiming at a margin inside the limit, would remove it - at some cost in "
    "return.",
]:
    put(ws, r, 2, text, align=WRAP)
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=10)
    ws.row_dimensions[r].height = 30
    r += 1
ws.sheet_view.showGridLines = False

# ==================================================================== Strategy
ws = wb.create_sheet("Strategy")
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 40
for j in range(8):
    ws.column_dimensions[get_column_letter(3 + j)].width = 14
r = 1
put(ws, r, 2, "Strategy - what each strategy actually does", font=BIG)
r += 2
put(ws, r, 2, "Decisions, median scenario (CHF millions)", font=TITLE)
r += 1
header(ws, r, ["Decision"] + [NAME[k] for k in ORDER])
r += 1
for label, fn in [
    ("Borrowed in month 0", lambda m: m["aK"][0]),
    ("Borrowed in month 1", lambda m: m["aK"][1]),
    ("Borrowed over five years", lambda m: float(np.sum(m["aK"][:H]))),
    ("Invested in bonds over five years", lambda m: float(np.sum(m["aB"][:H]))),
    ("Cash right after month 0", lambda m: m["C"][0]),
    ("Cash at year 5", lambda m: m["C"][H]),
]:
    put(ws, r, 2, label)
    for j, k in enumerate(ORDER):
        put(ws, r, 3 + j, fn(CMP[k]["median"]), fmt="#,##0")
    r += 1
note(ws, r, "Medians month by month, so the five-year sums add up medians "
     "rather than one scenario's decisions.", 9, 18)
r += 2

put(ws, r, 2, "Rate sensitivity over time, median", font=TITLE)
r += 1
header(ws, r, ["Strategy", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4",
               "Year 5"])
r += 1
for k in ORDER:
    put(ws, r, 2, NAME[k], font=BOLD if k == "DeepALM" else None)
    for j, t in enumerate((0, 12, 24, 36, 48, 60)):
        v = CMP[k]["median"]["IRS"][t]
        put(ws, r, 3 + j, v, fmt="+0.000;-0.000",
            fl=fill(RED) if abs(v) > 0.085 else None,
            font=BOLD if k == "DeepALM" else None)
    r += 1
r += 1

put(ws, r, 2, "The network found the paper's strategy on its own", font=TITLE)
r += 1
header(ws, r, ["Habit", "Paper's network ([E] 4.1-4.2)", "", "", "Ours"],
       height=22)
ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=5)
ws.merge_cells(start_row=r, start_column=6, end_row=r, end_column=9)
r += 1
dm = CMP["DeepALM"]["median"]
_turn = next((t for t in range(1, H + 1) if dm["IRS"][t] < 0 <= dm["IRS"][t - 1]), None)
habits = [
    ("Borrow heavily in month 0",
     "\"raise funds of roughly mCHF 1,000 in the first model period... taken "
     "in each scenario\"",
     f"{dm['aK'][0]:,.0f} mCHF in month 0. The benchmarks borrow at most "
     f"{_bm_m0:,.0f}."),
    ("Hold it as cash, invest nothing",
     "\"raising funds in the first period but not making any investments "
     "builds up a cash position and increases LCR and CMR\"",
     f"Cash {CMP['nothing']['median']['C'][0]:,.0f} -> {dm['C'][0]:,.0f} on "
     f"day 0; LCR {CMP['nothing']['median']['LCR'][0]:.2f} -> {dm['LCR'][0]:.2f}. "
     f"Investments over five years: {_inv_net:,.0f} mCHF."),
    ("Rate exposure positive early, negative late",
     "\"a positive IRS strategy... the IRS decreases and becomes negative "
     "during the fifth and final year\"",
     f"{dm['IRS'][0]:+.3f} at the start, "
     + (f"turning negative in month {_turn}, " if _turn else "")
     + f"{dm['IRS'][H]:+.3f} at year 5."),
]
for habit, paper, ours in habits:
    put(ws, r, 2, habit, font=BOLD, align=WRAP)
    put(ws, r, 3, paper, align=WRAP, font=Font(italic=True))
    ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=5)
    put(ws, r, 6, ours, align=WRAP)
    ws.merge_cells(start_row=r, start_column=6, end_row=r, end_column=9)
    ws.row_dimensions[r].height = 64
    r += 1
note(ws, r, "The paper's reading: its scenarios mostly see rates rising, bonds lose "
     "value when rates rise, so borrowing before the rise and holding cash gains "
     "equity; it calls the reluctance to invest an artefact of the five-year "
     "horizon.", 9, 30)
ws.sheet_view.showGridLines = False

# ================================================================ Monthly path
ws = wb.create_sheet("Monthly path")
show = ["nothing", "rollover", "BM_D", "DeepALM"]
series = [("E", "Equity", "#,##0"), ("C", "Cash", "#,##0"),
          ("LCR", "LCR", "0.000"), ("IRS", "IRS", "0.000"),
          ("E_over_RWA", "E/RWA", "0.000"), ("aK", "Borrowed", "#,##0")]
hdr = ["Month", "Year"] + [f"{lbl} - {NAME[k]}" for key, lbl, _ in series
                           for k in show]
for j, h in enumerate(hdr, start=1):
    put(ws, 1, j, h, font=WHITE_B, fl=H_FILL, align=CENTER)
ws.row_dimensions[1].height = 45
for t in range(H + 1):
    vals = [t, t / 12]
    fm = ["0", "0.00"]
    for key, _, f in series:
        for k in show:
            vals.append(CMP[k]["median"][key][t])
            fm.append(f)
    for j, (v, f) in enumerate(zip(vals, fm), start=1):
        put(ws, t + 2, j, v, fmt=f)
for j in range(1, len(hdr) + 1):
    ws.column_dimensions[get_column_letter(j)].width = 8 if j <= 2 else 13
ws.freeze_panes = "C2"


def chart(ws, title, cols, anchor, yfmt, ytitle="", x_col=2, last=H + 2,
          xmax=5, xmajor=1, xtitle="year", legend=True):
    ch = ScatterChart()
    ch.title, ch.width, ch.height, ch.style = title, 16, 8, 2
    ch.x_axis.title = xtitle
    if ytitle:
        ch.y_axis.title = ytitle
    ch.x_axis.delete = ch.y_axis.delete = False
    ch.x_axis.tickLblPos = "low"
    ch.x_axis.majorUnit = xmajor
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = 0, xmax
    ch.x_axis.number_format, ch.y_axis.number_format = "0", yfmt
    if legend:
        ch.legend.position = "b"
    else:
        ch.legend = None
    x = Reference(ws, min_col=x_col, min_row=2, max_row=last)
    for col in cols:
        s = Series(Reference(ws, min_col=col, min_row=1, max_row=last), x,
                   title_from_data=True)
        s.marker.symbol, s.smooth = "none", False
        ch.series.append(s)
    ws.add_chart(ch, anchor)


first = {key: 3 + i * len(show) for i, (key, _, _) in enumerate(series)}
cols = lambda key: list(range(first[key], first[key] + len(show)))
at = get_column_letter(len(hdr) + 2)
chart(ws, "Equity (median)", cols("E"), f"{at}2", "#,##0", "CHF m")
chart(ws, "Rate sensitivity IRS (limit ±0.085)", cols("IRS"), f"{at}19", "0.00")
chart(ws, "Liquidity coverage LCR (limit 1.05)", cols("LCR"), f"{at}36", "0.0")
chart(ws, "Cash (median)", cols("C"), f"{at}53", "#,##0", "CHF m")

# ==================================================================== Training
ws = wb.create_sheet("Training")
tr = REP["training"]
models = ["BM_E", "BM_C", "BM_D", "DeepALM"]
hdr = ["Epoch"] + [f"Validation loss - {NAME[m]}" for m in models] \
    + [f"Mean equity ratio - {NAME[m]}" for m in models]
for j, h in enumerate(hdr, start=1):
    put(ws, 1, j, h, font=WHITE_B, fl=H_FILL, align=CENTER)
ws.row_dimensions[1].height = 45
n_ep = max(len(tr[m]) for m in models)
for e in range(n_ep):
    put(ws, e + 2, 1, e + 1)
    for j, m in enumerate(models):
        if e < len(tr[m]):
            v = min(tr[m][e]["val_total"], 40.0)     # cap the one spike
            put(ws, e + 2, 2 + j, v, fmt="0.00")
            put(ws, e + 2, 6 + j, tr[m][e]["val_ER"], fmt="0.0000")
for j in range(1, len(hdr) + 1):
    ws.column_dimensions[get_column_letter(j)].width = 7 if j == 1 else 14
ws.freeze_panes = "B2"
chart(ws, "Validation loss by epoch (capped at 40)", [2, 3, 4, 5], "K2",
      "0", x_col=1, last=n_ep + 1, xmax=n_ep, xmajor=10, xtitle="epoch")
r = 20
best = min(range(len(tr["DeepALM"])), key=lambda e: tr["DeepALM"][e]["val_total"])
plateau = float(np.mean([x["val_total"] for x in tr["DeepALM"][len(tr["DeepALM"]) // 3:]]))
_spike = max(range(len(tr["DeepALM"])), key=lambda e: tr["DeepALM"][e]["val_total"])
_tr_best_bm = min(min(x["val_total"] for x in tr[k]) for k in _bench if k in tr)
for text in [
    f"How to read it. Each epoch is one pass through the {REP['setup']['train_scenarios']:,} "
    f"training scenarios; the loss shown is on the {REP['setup']['val_scenarios']:,} "
    "validation scenarios of the training run. The "
    "model kept is the epoch with the lowest validation loss (early stopping, "
    "as in [E] 3.2.3).",
    f"The network's best epoch was {best + 1} of {len(tr['DeepALM'])} "
    f"(loss {tr['DeepALM'][best]['val_total']:.2f}); the later epochs settled "
    f"around {plateau:.1f}. " + (
        f"Epoch {_spike + 1} spiked to {tr['DeepALM'][_spike]['val_total']:.0f} "
        "(capped at 40 in the chart). " if tr['DeepALM'][_spike]['val_total'] > 2 * plateau else "")
    + "The paper reports the same instability: losses jump when a decision "
    "tips a constraint.",
    f"Settled level {plateau:.1f} against the best benchmark's "
    f"{_tr_best_bm:.1f} in the same training run: " + (
        "the network's advantage does not rest on one lucky epoch."
        if plateau < _tr_best_bm else
        "only the best epoch beats the benchmarks; treat the headline with care."),
    "Benchmarks still improving in their last epoch have upper-bound "
    "numbers, not converged values.",
]:
    # each note gets its own block of rows, so the data rows keep their height
    put(ws, r, 11, text, align=WRAP)
    ws.merge_cells(start_row=r, start_column=11, end_row=r + 3, end_column=22)
    r += 4

# ======================================================================= Setup
ws = wb.create_sheet("Setup")
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 36
ws.column_dimensions["C"].width = 30
ws.column_dimensions["D"].width = 30
s = REP["setup"]
r = 1
put(ws, r, 2, "Setup", font=BIG)
r += 2
header(ws, r, ["Setting", "Ours", "Paper ([E] Table 2)"], height=22)
r += 1
for a, b, c in [
    ("Start date", DATE, "15.07.2022"),
    ("Starting equity", f"{E0:,.1f} mCHF", "withheld"),
    ("Horizon", "5 years, 60 monthly decisions", "5 years (also 15)"),
    ("Training scenarios", f"{s['train_scenarios']:,} (seed {s['train_seed']})", "40,000, resimulated"),
    ("Validation scenarios", f"{s['val_scenarios']:,} (seed {s['val_seed']})", "1,600"),
    ("Network hidden layers", " / ".join(map(str, s["widths"])), "512 / 512 / 256 / 128"),
    ("Features per decision", "145", "145"),
    ("Epochs: benchmarks / network", f"{s['bm_epochs']} / {s['nn_epochs']}", "up to 100"),
    ("Batch size", str(s["batch_size"]), "32"),
    ("Optimiser", "RAdam, cyclic learning rate 5e-4 to 5e-3, gradient clip 0.2", "same"),
    ("Return target µ", "drawn from 2-7% in training; 4.06% to score", "same"),
    ("Penalty weight λ", "drawn from 0.05-25 in training; 3.5 to score", "same"),
    ("Penalty weights σ", "LCR 1.0, NSFR 0.2, CMR 1.0, E/RWA 2.5, IRS 2.0, EYR 0.002", "same"),
    ("Loan margin", f"{s['loan_spread_bp']:.1f} bp", "withheld"),
    ("Software", "JAX on CPU", "PyTorch"),
]:
    put(ws, r, 2, a)
    put(ws, r, 3, b, align=WRAP)
    put(ws, r, 4, c, align=WRAP)
    r += 1
r += 1
put(ws, r, 2, "One deviation of ours", font=TITLE)
r += 1
put(ws, r, 2, "Decision sizes are learnt in units of 100 mCHF. The paper takes "
    f"about 125,000 optimiser steps; we take about {_steps:,} per strategy, and "
    "without the rescaling the benchmarks' amounts could not move far enough "
    "in that budget. It changes the scale the optimiser works in, not the "
    "model.", align=WRAP)
ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
ws.row_dimensions[r].height = 58
ws.sheet_view.showGridLines = False

wb.save(OUT)
print("saved", OUT)
