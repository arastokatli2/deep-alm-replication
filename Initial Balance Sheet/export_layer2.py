"""Layer 2 results as an Excel workbook: a short explanation and the balance sheet.

Reads balance_sheet_report.json, which run_balance_sheet.py has already
written; nothing is recomputed and nothing here changes a result.

    python export_layer2.py --out output_paper

run_balance_sheet.py also calls write_workbook() at the end.

Writes <out>/layer2_balance_sheet.xlsx.
"""
from __future__ import annotations

import argparse
import json
import os

LINES = [  # symbol, line, side
    ("C", "Cash and central-bank deposits", "Assets"),
    ("B", "Financial investments (bonds)", "Assets"),
    ("RP", "Mortgages", "Assets"),
    ("RE", "Loans to businesses", "Assets"),
    ("SD", "Sight and savings deposits (non-maturing)", "Liabilities"),
    ("SF", "Term deposits", "Liabilities"),
    ("K", "Borrowings (bonds, Pfandbriefe, money market)", "Liabilities"),
]

README = [
    "Layer 2 - Thurgauer Kantonalbank's balance sheet on day 0",
    "",
    "What this is. The 31 December 2022 balance sheet from TKB's annual report, "
    "mapped onto the seven positions of the model ([E] Table 1) and turned into "
    "monthly cash flows over the next 15 years.",
    "Book value is what the annual report shows. Model value discounts each "
    "position's cash flows on the starting yield curve (Layer 1). Equity is "
    "what is left: assets minus liabilities, valued the same way.",
    "",
    "Why model equity differs from book equity. About 10% of the balance sheet "
    "(buildings, participations, provisions, derivatives, accruals) has no "
    "place in the model and is left out; and valuing at market rates moves "
    "every fixed-rate position. The bridge on the next sheet shows each step.",
    "",
    "How to read the checks. The ratios are computed with the papers' formulas "
    "and set against TKB's own disclosed figures. They are not expected to "
    "match exactly: the papers' weights are simplified approximations.",
    "The rate-shock line is the most important check for ALM: the change in "
    "equity if all rates rise or fall by 1.5%, against what TKB reports.",
    "",
    "All amounts in CHF millions. Source: TKB Geschäftsbericht 2022; the "
    "numbers here are copied from balance_sheet_report.json, which has every "
    "figure in full.",
]


def write_workbook(out_dir: str, *, fit_label: str = "") -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    r = json.load(open(os.path.join(out_dir, "balance_sheet_report.json")))
    book, val, nom = r["book_mCHF"], r["values_mCHF"], r["nominal_cash_flows_mCHF"]
    A = val["A"]

    wb = Workbook()
    # ------------------------------------------------------------- Read me
    ws = wb.active
    ws.title = "Read me"
    ws.column_dimensions["A"].width = 110
    for i, line in enumerate(README + ([""] + [fit_label] if fit_label else []),
                             start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws["A1"].font = Font(bold=True, size=13)

    # -------------------------------------------------------- Balance sheet
    ws = wb.create_sheet("Balance sheet")
    head_fill = PatternFill("solid", start_color="1F4E79")
    band = PatternFill("solid", start_color="DDEBF7")
    bold, white = Font(bold=True), Font(bold=True, color="FFFFFF")
    top = Border(top=Side(style="thin"))
    num = "#,##0"
    row = 1

    def title(text):
        nonlocal row
        ws.cell(row=row, column=1, value=text).font = Font(bold=True, size=12)
        row += 1

    def header(cols):
        nonlocal row
        for j, h in enumerate(cols, start=1):
            c = ws.cell(row=row, column=j, value=h)
            c.font, c.fill = white, head_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[row].height = 30
        row += 1

    def line(values, fmt=None, strong=False, fill=None, border=False):
        nonlocal row
        for j, v in enumerate(values, start=1):
            # leave blanks truly empty, so long text in column A can spill over
            c = ws.cell(row=row, column=j, value=None if v == "" else v)
            if isinstance(v, (int, float)) and fmt:
                c.number_format = fmt[j - 1] if isinstance(fmt, list) else fmt
            if strong:
                c.font = bold
            if fill:
                c.fill = fill
            if border:
                c.border = top
        row += 1

    # --- the balance sheet
    title("Balance sheet on day 0, CHF millions")
    header(["Side", "Position", "Model symbol", "Book value (annual report)",
            "Model value on day-0 curve", "Model minus book",
            "Share of model assets", "Cash flows, undiscounted"])
    fmt = [None, None, None, num, num, num, "0.0%", num]
    for side in ("Assets", "Liabilities"):
        tot_b = tot_v = tot_n = 0.0
        for sym, name, s in LINES:
            if s != side:
                continue
            tot_b, tot_v, tot_n = tot_b + book[sym], tot_v + val[sym], tot_n + nom[sym]
            line([side, name, sym, book[sym], val[sym], val[sym] - book[sym],
                  val[sym] / A, nom[sym]], fmt)
        line(["", f"Total {side.lower()}", "A" if side == "Assets" else "L",
              tot_b, tot_v, tot_v - tot_b, tot_v / A, tot_n],
             fmt, strong=True, fill=band, border=True)
    line(["Equity", "Equity = assets minus liabilities", "E", None, val["E"],
          None, val["E"] / A, None], fmt, strong=True, fill=band, border=True)
    line(["", "Book equity in the annual report", "", r["book_equity_mCHF"],
          None, None, None, None], fmt)
    line(["", "CET1 capital (regulatory)", "", r["CET1_mCHF"],
          None, None, None, None], fmt)
    row += 1

    # --- the equity bridge
    title("From book equity to model equity, CHF millions")
    header(["Step", "", "", "Effect on equity"])
    labels = {
        "book equity": "Book equity (annual report)",
        "- excluded assets": "minus assets left out of the model",
        "+ excluded liabilities": "plus liabilities left out of the model",
        "= model E0": "Model equity on day 0",
    }
    for k, v in r["equity_bridge_mCHF"].items():
        if k.startswith(("+ (V - book)", "- (V - book)")):
            sym = k.split()[-1]
            name = dict((s, n) for s, n, _ in LINES)[sym]
            sign = "assets" if k.startswith("+") else "liabilities"
            text = f"revaluation at market rates: {name} ({sign})"
        else:
            text = labels.get(k, k)
        strong = k in ("book equity", "= model E0")
        line([text, "", "", v], [None, None, None, num], strong=strong,
             fill=band if k == "= model E0" else None,
             border=k == "= model E0")
    row += 1

    # --- the checks
    title("Day-0 checks: model against TKB's disclosed figures")
    header(["Measure", "", "Limit", "Model", "TKB disclosed"])
    k, t = r["constraints_t0"], r["targets"]
    rows = [
        ("Capital ratio E/RWA", "≥ 0.17", k["E_over_RWA"], t["E_over_RWA"], "0.000"),
        ("Liquidity coverage LCR", "≥ 1.05", k["LCR"], t["LCR"], "0.00"),
        ("Stable funding NSFR", "≥ 1.05", k["NSFR"], t["NSFR"], "0.00"),
        ("Reserve ratio CMR", "≥ 1.00", k["CMR"], None, "0.0"),
        ("Rate sensitivity IRS (+1%)", "|x| ≤ 0.085", r["IRS_100bp"], None, "0.000"),
        ("Equity change if rates +1.5%", "", r["dE_mCHF"]["150"],
         r["dEVE_tkb_mCHF"]["150"], num),
        ("Equity change if rates −1.5%", "", r["dE_mCHF"]["-150"],
         r["dEVE_tkb_mCHF"]["-150"], num),
    ]
    for name, lim, m, d, f in rows:
        line([name, "", lim, m, d], [None, None, None, f, f])

    widths = {"A": 14, "B": 46, "C": 12, "D": 18, "E": 18, "F": 16, "G": 14, "H": 18}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A3"

    path = os.path.join(out_dir, "layer2_balance_sheet.xlsx")
    wb.save(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="output_paper",
                    help="the folder run_balance_sheet.py wrote to")
    a = ap.parse_args()
    print(f"written {write_workbook(a.out)}")
