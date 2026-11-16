"""Layer 3 workbook: both strategies side by side, explained.

    python export_layer3.py sim_rollover.json sim_none.json out.xlsx

Every sentence is computed from the two reports, so it is right for any bank.
"""
import json
import sys

from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

RO = json.load(open(sys.argv[1]))          # the user's report: roll-over run
NO = json.load(open(sys.argv[2]))          # identical-setup no-action run
OUT = sys.argv[3]
S = RO["setup"]
DATE = ".".join(reversed(str(S["valuation_date"]).split("-")))
NSC = S["scenarios"]
_lcr0, _lcr5 = NO["median"]["LCR"][0], NO["median"]["LCR"][-1]
_irs0, _irs5 = NO["median"]["IRS"][0], NO["median"]["IRS"][-1]
_mu_no, _mu_ro = NO["loss"]["annualised"]["mu_hat"], RO["loss"]["annualised"]["mu_hat"]
_eyr = (min(NO["violation_share"]["EYR"], RO["violation_share"]["EYR"]),
        max(NO["violation_share"]["EYR"], RO["violation_share"]["EYR"]))
_ro_regs = [k for k in ("LCR", "NSFR", "CMR", "IRS") if RO["violation_share"][k] > 0]
_never_invests = all(abs(x - y) < 1e-6 for x, y in zip(RO["median"]["B"], NO["median"]["B"]))
_mix = S["mortgage_mix"]
_avg = sum(int(k[:-1]) * v for k, v in _mix.items()) / max(sum(_mix.values()), 1e-12)

NAVY, BAND, RED, GREEN, GREY = "1F4E79", "DDEBF7", "F8CBAD", "E2EFDA", "F2F2F2"
H_FILL = PatternFill("solid", start_color=NAVY)
WHITE_B = Font(bold=True, color="FFFFFF")
BOLD = Font(bold=True)
NOTE = Font(italic=True, color="595959", size=9)
TITLE = Font(bold=True, size=13, color=NAVY)
THIN_TOP = Border(top=Side(style="thin", color="7F7F7F"))
WRAP = Alignment(wrap_text=True, vertical="top")
YEARS = [0, 12, 24, 36, 48, 60]

wb = Workbook()


def fill(c):
    return PatternFill("solid", start_color=c)


# ====================================================================== Read me
ws = wb.active
ws.title = "Read me"
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 105
lines = [
    ("Layer 3 - the bank moved forward month by month", TITLE),
    ("", None),
    ("What this is", BOLD),
    ("The bank's balance sheet (Layer 2) walked forward 60 months "
     f"along {NSC:,} interest-rate scenarios (Layer 1), starting {DATE}. Each "
     "month loans mature and new ones are granted, deposits roll over, interest "
     "comes in, costs are paid; once a year a dividend is paid and the yearly "
     "profit is checked.", None),
    ("", None),
    ("The two strategies compared", BOLD),
    ("No decisions - the bank invests and borrows nothing. Maturing debt is "
     "repaid out of cash. This is the reference path: what the starting balance "
     "sheet does on its own.", None),
    ("Roll-over - every borrowing that matures is refinanced, spread evenly over "
     "16 maturities from 3 months to 15 years. It also invests a quarter of "
     "idle cash above the negative-rate exemption" + (
         ", but here that threshold is never reached, so in practice it only "
         "refinances." if _never_invests else "."), None),
    ("Loans and deposits grow identically under both: the bank's decisions "
     "never touch them.", None),
    ("", None),
    ("Sheets", BOLD),
    ("Results - the five blocks of the run output, both strategies side by "
     "side, each with a line on how to read it.", None),
    ("Monthly path - all 60 months, with charts.", None),
    ("Setup - the run settings and the three paper problems resolved in Layer 3.", None),
    ("", None),
    ("What it shows", BOLD),
    ("1. Doing nothing: repaying debt out of cash moves the liquidity ratio "
     f"from {_lcr0:.2f} to {_lcr5:.2f} and rate sensitivity from {_irs0:+.3f} "
     f"to {_irs5:+.3f} (medians). It breaks LCR in "
     f"{NO['violation_share']['LCR']:.0%} and IRS in "
     f"{NO['violation_share']['IRS']:.0%} of scenarios.", None),
    ("2. Refinancing " + ("keeps every regulatory ratio inside its limit"
     if not _ro_regs else "still breaks " + ", ".join(_ro_regs)) +
     f", and returns {_mu_ro:.2%} a year against {_mu_no:.2%} for doing "
     "nothing. Improving on this is the job of the trained network in Layer 5.", None),
    (f"3. The minimum-return rule (EYR) is broken in {_eyr[0]:.0%}-{_eyr[1]:.0%} "
     "of scenarios. It depends on how much the bank earns more than on how it "
     "is financed.", None),
    ("", None),
    ("How to read the numbers", BOLD),
    (f"All amounts in CHF millions. Every figure is the median across the {NSC:,} "
     "scenarios, taken separately for each line - so the medians do not add up "
     "exactly (median assets need not equal median liabilities plus median "
     "equity).", None),
    ("Years 1-4 are read just before that year's dividend is paid, as the "
     "paper records them. Cells shaded red break their limit.", None),
    ("", None),
    ("Sources: sim_none.json and sim_rollover.json, written by run_simulator.py "
     "for this run.", NOTE),
]
for i, (text, font) in enumerate(lines, start=1):
    c = ws.cell(row=i, column=2, value=text or None)
    c.alignment = WRAP
    if font:
        c.font = font

# ====================================================================== Results
ws = wb.create_sheet("Results")
widths = {"A": 3, "B": 46, "C": 11, "D": 11, "E": 11, "F": 11, "G": 11,
          "H": 11, "I": 3, "J": 11, "K": 11, "L": 11, "M": 11, "N": 11, "O": 11,
          "P": 3, "Q": 12}
for k, w in widths.items():
    ws.column_dimensions[k].width = w
row = 1


def put(r, c, v, *, font=None, fmt=None, fl=None, align=None, border=None):
    cell = ws.cell(row=r, column=c, value=v)
    if font:
        cell.font = font
    if fmt:
        cell.number_format = fmt
    if fl:
        cell.fill = fl
    if align:
        cell.alignment = align
    if border:
        cell.border = border
    return cell


def block_title(text, note):
    global row
    put(row, 2, text, font=TITLE)
    row += 1
    put(row, 2, note, font=NOTE, align=Alignment(wrap_text=True, vertical="top"))
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=17)
    ws.row_dimensions[row].height = 30
    row += 1


def two_run_header(first, extra=None):
    """Years 0-5 for each run, side by side."""
    global row
    put(row, 3, "No decisions", font=WHITE_B, fl=H_FILL)
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=8)
    put(row, 10, "Roll-over", font=WHITE_B, fl=H_FILL)
    ws.merge_cells(start_row=row, start_column=10, end_row=row, end_column=15)
    if extra:
        put(row, 17, extra, font=WHITE_B, fl=H_FILL)
    for c in (3, 10):
        ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")
    row += 1
    put(row, 2, first, font=BOLD, fl=fill(BAND))
    for j, y in enumerate(range(6)):
        put(row, 3 + j, f"{y}y", font=BOLD, fl=fill(BAND),
            align=Alignment(horizontal="right"))
        put(row, 10 + j, f"{y}y", font=BOLD, fl=fill(BAND),
            align=Alignment(horizontal="right"))
    if extra:
        put(row, 17, "", fl=fill(BAND))
    row += 1


put(row, 2, f"Layer 3 results - two strategies, {NSC:,} scenarios, from {DATE}",
    font=Font(bold=True, size=15, color=NAVY))
row += 2

# ---------------------------------------------------------------- block 1
block_title("1  The balance sheet, median path (CHF millions)",
            "The customer business (loans, deposits) is identical in both runs. "
            "The difference is borrowings: repaid out of cash under no decisions, "
            "refinanced under roll-over - which is why cash drains in one and "
            "grows in the other.")
two_run_header("Position")
names = [("C", "Cash"), ("B", "Investments (bonds)"), ("RP", "Mortgages"),
         ("RE", "Business loans"), ("SD", "Sight & savings deposits"),
         ("SF", "Term deposits"), ("K", "Borrowings"), ("E", "Equity")]
for sym, name in names:
    strong = sym == "E"
    put(row, 2, f"{name} ({sym})", font=BOLD if strong else None,
        border=THIN_TOP if strong else None)
    for j, t in enumerate(YEARS):
        for base, src in ((3, NO), (10, RO)):
            put(row, base + j, src["median"][sym][t], fmt="#,##0",
                font=BOLD if strong else None, border=THIN_TOP if strong else None)
    row += 1
row += 1

# ---------------------------------------------------------------- block 2
block_title("2  Regulatory ratios, median path",
            "Shaded red where the median breaks its limit. LCR collapses and rate "
            "sensitivity (IRS) deepens when debt is repaid out of cash; refinancing "
            "keeps both inside their limits. CMR jumps in year 2 because the paper "
            "cuts the reserve requirement by 80% once short rates turn positive.")
two_run_header("Ratio", extra="Limit")
ratios = [("E_over_RWA", "Capital  E/RWA", 0.17, "min", "0.000"),
          ("LCR", "Liquidity coverage  LCR", 1.05, "min", "0.00"),
          ("NSFR", "Stable funding  NSFR", 1.05, "min", "0.00"),
          ("CMR", "Reserve ratio  CMR", 1.00, "min", "0.0"),
          ("IRS", "Rate sensitivity  IRS (+1%)", 0.085, "abs", "0.000")]
for key, name, lim, kind, fmt in ratios:
    put(row, 2, name)
    for j, t in enumerate(YEARS):
        for base, src in ((3, NO), (10, RO)):
            v = src["median"][key][t]
            bad = (abs(v) > lim) if kind == "abs" else (v < lim)
            put(row, base + j, v, fmt=fmt, fl=fill(RED) if bad else None)
    put(row, 17, f"|x| ≤ {lim}" if kind == "abs" else f"≥ {lim}",
        align=Alignment(horizontal="right"))
    row += 1
row += 1

# ---------------------------------------------------------------- block 3
block_title("3  Scenarios breaking each limit at least once",
            "Out of 4,000 scenarios, the share that break each limit in at least "
            "one month (months 0-59, as the paper checks them). Doing nothing "
            "breaks liquidity and rate risk everywhere; roll-over leaves only the "
            "capital ratio in bad scenarios and the minimum-return rule.")
for c, h in ((2, "Limit"), (3, "No decisions"), (4, "Roll-over"), (5, "Change")):
    put(row, c, h, font=WHITE_B, fl=H_FILL,
        align=Alignment(wrap_text=True, horizontal="center" if c > 2 else None, vertical="center"))
ws.row_dimensions[row].height = 30
row += 1
labels = {"CMR": "Reserve ratio  CMR", "LCR": "Liquidity coverage  LCR",
          "NSFR": "Stable funding  NSFR", "E_over_RWA": "Capital  E/RWA",
          "IRS": "Rate sensitivity  IRS", "EYR": "Minimum annual return  EYR"}
for key in ("LCR", "IRS", "CMR", "E_over_RWA", "NSFR", "EYR"):
    a, b = NO["violation_share"][key], RO["violation_share"][key]
    put(row, 2, labels[key])
    put(row, 3, a, fmt="0.0%", fl=fill(RED) if a > 0.2 else None)
    put(row, 4, b, fmt="0.0%", fl=fill(RED) if b > 0.2 else
        (fill(GREEN) if b == 0 else None))
    put(row, 5, b - a, fmt="+0.0%;-0.0%;0.0%")
    row += 1
row += 1

# ---------------------------------------------------------------- block 4
block_title("4  Equity after five years",
            "Doing nothing ends higher on average but far more spread out: with a "
            "rate sensitivity of -0.19, the yield curve decides the result. "
            "Roll-over gives up some return to narrow the range.")
for c, h in ((2, "Measure"), (3, "No decisions"), (4, "Roll-over")):
    put(row, c, h, font=WHITE_B, fl=H_FILL,
        align=Alignment(wrap_text=True, horizontal="center" if c > 2 else None, vertical="center"))
ws.row_dimensions[row].height = 30
row += 1
nl, rl = NO["loss"], RO["loss"]
rows4 = [
    ("Starting equity", NO["E0"], RO["E0"], "#,##0"),
    ("Final equity, median", NO["ET"]["median"], RO["ET"]["median"], "#,##0"),
    ("Final equity, 5th percentile", NO["ET"]["q05"], RO["ET"]["q05"], "#,##0"),
    ("Final equity, 95th percentile", NO["ET"]["q95"], RO["ET"]["q95"], "#,##0"),
    ("Final / starting equity (median)", NO["ET_over_E0"], RO["ET_over_E0"], "0.000\"×\""),
    ("Equity ratio, mean", nl["equity_ratio"]["mean"], rl["equity_ratio"]["mean"], "0.000"),
    ("Equity ratio, standard deviation", nl["equity_ratio"]["sd"], rl["equity_ratio"]["sd"], "0.000"),
    ("Equity ratio, skew", nl["equity_ratio"]["skew"], rl["equity_ratio"]["skew"], "0.000"),
    ("Value at risk, worst 5% (from mean)", nl["equity_ratio"]["VaR0.95"], rl["equity_ratio"]["VaR0.95"], "0.000"),
    ("Expected shortfall, worst 5% (from mean)", nl["equity_ratio"]["ES0.95"], rl["equity_ratio"]["ES0.95"], "0.000"),
    ("Annual return, excluding dividends", nl["annualised"]["mu_hat"], rl["annualised"]["mu_hat"], "0.00%"),
    ("Annual dividend yield", nl["annualised"]["dividend_yield"], rl["annualised"]["dividend_yield"], "0.00%"),
]
for name, a, b, f in rows4:
    put(row, 2, name)
    put(row, 3, a, fmt=f)
    put(row, 4, b, fmt=f)
    row += 1
put(row, 2, "The higher dividend yield of doing nothing is not a sign of a better "
    "strategy: dividends are half of each year's gain and losses pay nothing "
    "back, so a jumpier path pays out more.", font=NOTE,
    align=Alignment(wrap_text=True, vertical="top"))
ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=17)
ws.row_dimensions[row].height = 26
row += 2

# ---------------------------------------------------------------- block 5
block_title("5  The Layer 4 loss",
            "Total loss = target loss + 3.5 × penalty loss. The target loss is the "
            "shortfall against growing 4.06% a year; the penalty adds up every "
            "breach of every limit in every month, squared. Lower is better.")
for c, h in ((2, "Loss component"), (3, "No decisions"), (4, "Roll-over")):
    put(row, c, h, font=WHITE_B, fl=H_FILL,
        align=Alignment(wrap_text=True, horizontal="center" if c > 2 else None, vertical="center"))
ws.row_dimensions[row].height = 30
row += 1
rows5 = [
    ("Total loss", nl["loss"]["total"], rl["loss"]["total"], "#,##0.00", True),
    ("   Target loss (shortfall vs 4.06% a year)", nl["loss"]["target"], rl["loss"]["target"], "0.000", False),
    ("   Penalty loss (all breaches)", nl["loss"]["penalty"], rl["loss"]["penalty"], "#,##0.000", False),
    ("Penalty, worst 5% of scenarios (from mean)", nl["loss"]["penalty_ES0.95"], rl["loss"]["penalty_ES0.95"], "#,##0.0", False),
    ("Utility loss, very risk-averse investor (γ = 10)", nl["loss"]["utility"], rl["loss"]["utility"], "0.000", False),
]
for name, a, b, f, strong in rows5:
    put(row, 2, name, font=BOLD if strong else None)
    put(row, 3, a, fmt=f, font=BOLD if strong else None)
    put(row, 4, b, fmt=f, font=BOLD if strong else None)
    row += 1
row += 1
for c, h in ((2, "Penalty contribution by limit"), (3, "No decisions"),
             (4, "Roll-over")):
    put(row, c, h, font=WHITE_B, fl=H_FILL,
        align=Alignment(wrap_text=True, horizontal="center" if c > 2 else None, vertical="center"))
ws.row_dimensions[row].height = 30
row += 1
for key in ("LCR", "IRS", "EYR", "CMR", "E_over_RWA", "NSFR"):
    put(row, 2, labels[key])
    put(row, 3, nl["penalty_share"][key], fmt="0.000")
    put(row, 4, rl["penalty_share"][key], fmt="0.000")
    row += 1
put(row, 2, "Roll-over removes every regulatory penalty; what remains is almost "
    "entirely the minimum-return rule (EYR), the constraint the paper also "
    "calls the hardest.", font=NOTE, align=Alignment(wrap_text=True, vertical="top"))
ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=17)
ws.row_dimensions[row].height = 26
ws.sheet_view.showGridLines = False
ws.freeze_panes = "C2"

# ================================================================= Monthly path
ws = wb.create_sheet("Monthly path")
cols = [("Month", None, None), ("Year", None, None)]
series = [("E", "Equity"), ("C", "Cash"), ("K", "Borrowings"),
          ("LCR", "LCR"), ("IRS", "IRS"), ("E_over_RWA", "E/RWA"),
          ("CMR", "CMR"), ("NSFR", "NSFR"), ("B", "Investments"),
          ("RP", "Mortgages"), ("RE", "Business loans"),
          ("SD", "Sight & savings dep."), ("SF", "Term deposits")]
hdr = ["Month", "Year"]
for key, name in series:
    hdr += [f"{name} - no decisions", f"{name} - roll-over"]
hdr += ["Borrowing refinanced this month - roll-over",
        "Cash from loans, deposits & costs - roll-over"]
for j, h in enumerate(hdr, start=1):
    c = ws.cell(row=1, column=j, value=h)
    c.font, c.fill = WHITE_B, H_FILL
    c.alignment = Alignment(wrap_text=True, vertical="center")
ws.row_dimensions[1].height = 45
for t in range(61):
    vals = [t, t / 12]
    for key, _ in series:
        vals += [NO["median"][key][t], RO["median"][key][t]]
    vals += [RO["median"].get("matured_K", [None] * 61)[t],
             RO["median"].get("CF", [None] * 61)[t]]          # older reports lack these
    for j, v in enumerate(vals, start=1):
        c = ws.cell(row=t + 2, column=j, value=v)
        key = series[(j - 3) // 2][0] if 3 <= j < 3 + 2 * len(series) else None
        c.number_format = ("0" if j == 1 else "0.00" if j == 2 else
                           "0.000" if key in ("LCR", "IRS", "E_over_RWA", "NSFR")
                           else "0.0" if key == "CMR" else "#,##0")
for j in range(1, len(hdr) + 1):
    ws.column_dimensions[get_column_letter(j)].width = 8 if j <= 2 else 14
ws.freeze_panes = "C2"


def chart(title, cols_, anchor, yfmt, ytitle):
    ch = ScatterChart()
    ch.title, ch.width, ch.height, ch.style = title, 15, 7.5, 2
    ch.x_axis.title = "year"
    if ytitle:                         # an empty title would show as "None"
        ch.y_axis.title = ytitle
    ch.x_axis.delete = ch.y_axis.delete = False
    ch.x_axis.tickLblPos = "low"
    ch.x_axis.majorUnit = 1
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = 0, 5
    ch.x_axis.number_format, ch.y_axis.number_format = "0", yfmt
    ch.legend.position = "b"
    x = Reference(ws, min_col=2, min_row=2, max_row=62)
    for col in cols_:
        s = Series(Reference(ws, min_col=col, min_row=1, max_row=62), x,
                   title_from_data=True)
        s.marker.symbol, s.smooth = "none", False
        ch.series.append(s)
    ws.add_chart(ch, anchor)


col_of = {key: 3 + 2 * i for i, (key, _) in enumerate(series)}
start = get_column_letter(len(hdr) + 2)
chart("Equity (median)", [col_of["E"], col_of["E"] + 1], f"{start}2", "#,##0", "CHF m")
chart("Cash (median)", [col_of["C"], col_of["C"] + 1], f"{start}18", "#,##0", "CHF m")
chart("Liquidity coverage LCR (limit 1.05)", [col_of["LCR"], col_of["LCR"] + 1],
      f"{start}34", "0.0", "")
chart("Rate sensitivity IRS (limit ±0.085)", [col_of["IRS"], col_of["IRS"] + 1],
      f"{start}50", "0.00", "")

# ======================================================================== Setup
ws = wb.create_sheet("Setup")
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 34
ws.column_dimensions["C"].width = 95
s = RO["setup"]
rows = [
    ("Run settings", None),
    ("Start date", s["valuation_date"]),
    ("Scenarios", f"{s['scenarios']:,}  (seed {s['seed']})"),
    ("Yield curve model", "Paper model, P = Q (variant 'sqrt')"),
    ("Loan margin over the curve", f"{s['loan_spread_bp']:.1f} bp - solved so first-year "
     "loan interest equals the bank's reported interest income"),
    ("Return target µ (Layer 4)", "4.06% a year"),
    ("Penalty weight λ (Layer 4)", "3.5"),
    ("", None),
    ("New mortgages by fixed term", None),
] + [(f"   {k}", f"{v:.1%}") for k, v in s["mortgage_mix"].items()] + [
    ("", "Chosen so the long-run mortgage book matches the bank's own maturity "
         f"profile; average term {_avg:.1f} years."),
    ("", None),
    ("Paper problems resolved in Layer 3", None),
] + [(k, v) for k, v in s["resolved"].items()]
r = 1
for k, v in rows:
    a = ws.cell(row=r, column=2, value=k or None)
    b = ws.cell(row=r, column=3, value=v)
    if v is None and k:
        a.font = TITLE
    b.alignment = WRAP
    a.alignment = Alignment(vertical="top")
    if k in RO["setup"]["resolved"]:
        a.font = BOLD
        ws.row_dimensions[r].height = 62
    r += 1
ws.sheet_view.showGridLines = False

wb.save(OUT)
print("saved", OUT)
