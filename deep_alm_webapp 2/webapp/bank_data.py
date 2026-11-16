"""The bank's data as an Excel workbook, for the web app's Layer 2 upload.

    write_template(path)      the workbook, pre-filled with whatever bank
                              balance_sheet.py currently holds (TKB 2022)
    read_workbook(path)       -> BankData, checked
    use_bank(data)            context manager: balance_sheet.py runs on it

balance_sheet.py keeps its data in named tables at the top of the file
(BALANCE_SHEET, MATURITY_TABLE, BOND_ISSUES, ...).  use_bank() replaces those
tables for the duration of a run and puts the originals back afterwards, so
the Layer 2 code runs unchanged on uploaded figures.

The line keys ("cash", "mortgages", ...) name the rows the code maps into the
model, so they cannot be renamed; amounts can be anything, including zero.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

import balance_sheet as bs

BUCKET_LABELS = ["Sight", "Callable", "Up to 3 months", "3 to 12 months",
                 "1 to 5 years", "Over 5 years"]
SVENSSON = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
TARGET_LABELS = {
    "RWA": "Risk-weighted assets (kCHF)", "CET1": "CET1 capital (kCHF)",
    "E_over_RWA": "CET1 ratio (decimal)", "HQLA": "HQLA (kCHF)",
    "NO30": "Net outflows, 30 days (kCHF)", "LCR": "LCR (decimal)",
    "ASF": "Available stable funding (kCHF)",
    "RSF": "Required stable funding (kCHF)", "NSFR": "NSFR (decimal)",
    "dEVE_up150": "Change in EVE, rates +150 bp (kCHF)",
    "dEVE_down150": "Change in EVE, rates -150 bp (kCHF)",
}
IRRBB_LABELS = {
    "fixed_mortgages": "Fixed-rate mortgages", "money_mkt_mortgages": "SARON / money-market mortgages",
    "variable_mortgages": "Variable mortgages", "customer_loans": "Loans to customers",
    "fin_investments": "Financial investments", "due_from_banks": "Due from banks",
    "due_to_banks": "Due to banks", "customer_term": "Customer term deposits",
    "cash_bonds": "Cash bonds (Kassenobligationen)", "bonds_pfandbrief": "Bonds and Pfandbrief loans",
    "nmd_sight": "Non-maturing: sight and current accounts", "nmd_savings": "Non-maturing: savings",
    "irs_receive_leg": "Interest-rate swaps, receive leg", "irs_pay_leg": "Interest-rate swaps, pay leg",
    "total": "Total",
}


@dataclass
class BankData:
    name: str
    balance_sheet_date: dt.date
    balance_sheet: dict            # key -> (side, label, kCHF)
    total_assets_kchf: float
    maturity_table: dict           # key -> 6 kCHF amounts
    bond_issues: list              # (coupon %, ISO date, kCHF)
    pfandbrief_by_year: dict       # year -> kCHF
    pfandbrief_rate: float         # decimal
    pfandbrief_first_year_split: tuple | None
    fin_investments_fair_value_kchf: float
    fair_value_curve: np.ndarray   # Svensson, b0 b1 b2 b3 t1 t2
    interest_expense_cash_bonds_kchf: float
    personnel_cost_kchf: float
    material_cost_kchf: float
    interest_income_mchf: float    # for the Layer 3 loan margin
    irrbb: dict                    # key -> (volume mCHF, term years)
    targets: dict
    warnings: list = field(default_factory=list)

    def fingerprint(self) -> str:
        """Short hash of every number, to tell whether two banks are the same.
        Numbers are normalised first: Excel returns 6138689 as 6138689.0."""
        def norm(x):
            if isinstance(x, dict):
                return {str(k): norm(v) for k, v in x.items()}
            if isinstance(x, (list, tuple, np.ndarray)):
                return [norm(v) for v in x]
            if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool):
                return round(float(x), 8)
            return str(x)
        d = {k: norm(v) for k, v in self.__dict__.items() if k not in ("name", "warnings")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def current_bank(name: str = "Thurgauer Kantonalbank 2022") -> BankData:
    """Whatever balance_sheet.py holds right now."""
    return BankData(
        name=name, balance_sheet_date=bs.VALUATION_DATE,
        balance_sheet=dict(bs.BALANCE_SHEET), total_assets_kchf=bs.TOTAL_ASSETS_KCHF,
        maturity_table={k: tuple(v) for k, v in bs.MATURITY_TABLE.items()},
        bond_issues=list(bs.BOND_ISSUES), pfandbrief_by_year=dict(bs.PFANDBRIEF_BY_YEAR),
        pfandbrief_rate=bs.PFANDBRIEF_RATE,
        pfandbrief_first_year_split=bs.PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF,
        fin_investments_fair_value_kchf=bs.FIN_INVESTMENTS_FAIR_VALUE_KCHF,
        fair_value_curve=np.array(bs.FAIR_VALUE_CURVE, dtype=float),
        interest_expense_cash_bonds_kchf=bs.INTEREST_EXPENSE_CASH_BONDS_KCHF,
        personnel_cost_kchf=bs.PERSONNEL_COST_KCHF, material_cost_kchf=bs.MATERIAL_COST_KCHF,
        interest_income_mchf=303.354,
        irrbb={k: tuple(v) for k, v in bs.IRRBBA1.items()}, targets=dict(bs.TARGETS))


# ================================================================ apply
@contextlib.contextmanager
def use_bank(data: BankData):
    """Run balance_sheet.py on `data`; the original tables come back afterwards."""
    names = ["VALUATION_DATE", "BALANCE_SHEET", "TOTAL_ASSETS_KCHF", "MATURITY_TABLE",
             "BOND_ISSUES", "PFANDBRIEF_BY_YEAR", "PFANDBRIEF_RATE",
             "PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF", "FIN_INVESTMENTS_FAIR_VALUE_KCHF",
             "FAIR_VALUE_CURVE", "INTEREST_EXPENSE_CASH_BONDS_KCHF", "PERSONNEL_COST_KCHF",
             "MATERIAL_COST_KCHF", "IRRBBA1", "NMD_TARGET_Y", "TARGETS"]
    saved = {n: getattr(bs, n) for n in names}
    ir = data.irrbb
    nmd = ((ir["nmd_sight"][0] * ir["nmd_sight"][1] + ir["nmd_savings"][0] * ir["nmd_savings"][1])
           / max(ir["nmd_sight"][0] + ir["nmd_savings"][0], 1e-12))
    new = dict(
        VALUATION_DATE=data.balance_sheet_date, BALANCE_SHEET=data.balance_sheet,
        TOTAL_ASSETS_KCHF=data.total_assets_kchf, MATURITY_TABLE=data.maturity_table,
        BOND_ISSUES=data.bond_issues, PFANDBRIEF_BY_YEAR=data.pfandbrief_by_year,
        PFANDBRIEF_RATE=data.pfandbrief_rate,
        PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF=data.pfandbrief_first_year_split,
        FIN_INVESTMENTS_FAIR_VALUE_KCHF=data.fin_investments_fair_value_kchf,
        FAIR_VALUE_CURVE=np.asarray(data.fair_value_curve, dtype=float),
        INTEREST_EXPENSE_CASH_BONDS_KCHF=data.interest_expense_cash_bonds_kchf,
        PERSONNEL_COST_KCHF=data.personnel_cost_kchf, MATERIAL_COST_KCHF=data.material_cost_kchf,
        IRRBBA1=ir, NMD_TARGET_Y=nmd, TARGETS=data.targets)
    try:
        for n, v in new.items():
            setattr(bs, n, v)
        yield data
    finally:
        for n, v in saved.items():
            setattr(bs, n, v)


# ================================================================ write
def write_template(path: str, data: BankData | None = None) -> str:
    """The upload workbook, filled with `data` (default: what the code holds)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation  # noqa: F401

    d = data or current_bank()
    wb = Workbook()
    navy, edit = PatternFill("solid", start_color="1F4E79"), PatternFill("solid", start_color="FFF2CC")
    wf, bold = Font(bold=True, color="FFFFFF"), Font(bold=True)

    def sheet(title, headers, rows, widths, editable_from=2, numfmt=None):
        ws = wb.create_sheet(title)
        for j, h in enumerate(headers, start=1):
            c = ws.cell(row=1, column=j, value=h)
            c.font, c.fill = wf, navy
            c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[1].height = 32
        for i, row in enumerate(rows, start=2):
            for j, v in enumerate(row, start=1):
                c = ws.cell(row=i, column=j, value=v)
                if j >= editable_from:
                    c.fill = edit
                    if numfmt and isinstance(v, (int, float)):
                        c.number_format = numfmt
        for j, w in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + j)].width = w
        ws.freeze_panes = "A2"
        return ws

    ws = wb.active
    ws.title = "Read me"
    ws.column_dimensions["A"].width = 100
    text = [
        ("Bank data for the Deep ALM web app (Layer 2)", Font(bold=True, size=13)),
        ("", None),
        ("Fill in the yellow cells. Leave the grey key columns as they are: the "
         "model maps each key to a position, so a renamed key is an unknown line.", None),
        ("Amounts in thousands of CHF (kCHF) unless a column says otherwise. "
         "Rates in percent where the column says %, otherwise as decimals.", None),
        ("A line the bank does not have: put 0, do not delete the row.", None),
        ("", None),
        ("Sheets", bold),
        ("Bank - name, balance-sheet date and the single-number inputs.", None),
        ("Balance sheet - every line of the published balance sheet.", None),
        ("Maturity table - the published maturity structure of the financial "
         "instruments (in the annual report's notes).", None),
        ("Bond issues - the bank's own bonds, one row each.", None),
        ("Pfandbrief by year - Pfandbrief loans by maturity year.", None),
        ("IRRBB terms - volumes and average repricing terms from the IRRBBA1 "
         "disclosure table.", None),
        ("Disclosed figures - capital, liquidity and rate-risk figures the model "
         "is checked against (disclosure report).", None),
        ("", None),
        ("The app checks the workbook when you upload it: the balance sheet must "
         "balance, maturity rows must add up to their balance-sheet lines, and "
         "the bond issues plus Pfandbrief loans must add up to their line.", None),
        ("", None),
        ("Pre-filled with Thurgauer Kantonalbank, 31.12.2022: Geschaeftsbericht "
         "2022 and Offenlegungsbericht 2022.", Font(italic=True, color="595959")),
    ]
    for i, (t, f) in enumerate(text, start=1):
        c = ws.cell(row=i, column=1, value=t or None)
        c.alignment = Alignment(wrap_text=True)
        if f:
            c.font = f

    split = d.pfandbrief_first_year_split or (None, None)
    sheet("Bank", ["Setting", "Value", "Unit / note"], [
        ["Bank name", d.name, ""],
        ["Balance-sheet date", d.balance_sheet_date.isoformat(), "YYYY-MM-DD"],
        ["Total assets", d.total_assets_kchf, "kCHF"],
        ["Financial investments, fair value", d.fin_investments_fair_value_kchf, "kCHF (notes)"],
        ["Interest income, latest year", d.interest_income_mchf, "mCHF; sets the loan margin in Layer 3"],
        ["Personnel cost, latest year", d.personnel_cost_kchf, "kCHF"],
        ["Material cost, latest year", d.material_cost_kchf, "kCHF"],
        ["Interest expense on cash bonds", d.interest_expense_cash_bonds_kchf, "kCHF"],
        ["Pfandbrief average rate", d.pfandbrief_rate * 100, "%"],
        ["Pfandbrief due in months 1-3 of first year", split[0], "kCHF; blank = spread evenly"],
        ["Pfandbrief due in months 4-12 of first year", split[1], "kCHF; blank = spread evenly"],
    ] + [[f"Fair-value curve {n}", float(v), "SNB Svensson parameter on the balance-sheet date"]
         for n, v in zip(SVENSSON, d.fair_value_curve)],
        [42, 26, 52], editable_from=2, numfmt="#,##0.######")
    sheet("Balance sheet", ["Key", "Side (A / L / E)", "Label", "Amount (kCHF)"],
          [[k, s, lbl, amt] for k, (s, lbl, amt) in d.balance_sheet.items()],
          [20, 16, 46, 18], editable_from=3, numfmt="#,##0")
    sheet("Maturity table", ["Key"] + [f"{b} (kCHF)" for b in BUCKET_LABELS],
          [[k, *v] for k, v in d.maturity_table.items()],
          [20] + [15] * 6, editable_from=2, numfmt="#,##0")
    sheet("Bond issues", ["Coupon (%)", "Maturity date (YYYY-MM-DD)", "Amount (kCHF)"],
          [[c, m, a] for c, m, a in d.bond_issues], [14, 28, 16], editable_from=1, numfmt="#,##0.###")
    sheet("Pfandbrief by year", ["Maturity year", "Amount (kCHF)"],
          [[y, a] for y, a in d.pfandbrief_by_year.items()], [16, 18], editable_from=1, numfmt="#,##0")
    sheet("IRRBB terms", ["Key", "Item", "Volume (mCHF)", "Average repricing term (years)"],
          [[k, IRRBB_LABELS.get(k, k), v, t] for k, (v, t) in d.irrbb.items()],
          [22, 42, 16, 18], editable_from=3, numfmt="#,##0.##")
    sheet("Disclosed figures", ["Key", "Item", "Value"],
          [[k, TARGET_LABELS.get(k, k), v] for k, v in d.targets.items()],
          [16, 42, 18], editable_from=3, numfmt="#,##0.####")
    for ws in wb.worksheets[1:]:
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if c.fill != edit and c.column == 1:
                    c.fill = PatternFill("solid", start_color="EDEDED")
    wb.save(path)
    return path


# ================================================================ read
class TemplateError(ValueError):
    pass


def read_workbook(path: str) -> BankData:
    """Read and check an uploaded workbook.  Raises TemplateError with a plain
    message for anything the model cannot use; collects softer issues in
    .warnings."""
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    need = ["Bank", "Balance sheet", "Maturity table", "Bond issues",
            "Pfandbrief by year", "IRRBB terms", "Disclosed figures"]
    missing = [s for s in need if s not in wb.sheetnames]
    if missing:
        raise TemplateError(f"The workbook is missing the sheet(s): {', '.join(missing)}. "
                            "Start from the template.")

    def rows(name):
        return [r for r in wb[name].iter_rows(min_row=2, values_only=True)
                if r and any(v is not None and v != "" for v in r)]

    def num(v, where, allow_blank=False):
        if v is None or v == "":
            if allow_blank:
                return None
            raise TemplateError(f"{where}: empty cell, a number is needed.")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise TemplateError(f"{where}: '{v}' is not a number.") from None

    ref = current_bank()
    bank = {r[0]: r[1] for r in rows("Bank")}

    def get(label, allow_blank=False):
        if label not in bank:
            raise TemplateError(f"Sheet 'Bank': the row '{label}' is missing.")
        return num(bank[label], f"Bank / {label}", allow_blank)

    try:
        date = bank["Balance-sheet date"]
        date = date.date() if isinstance(date, dt.datetime) else (
            date if isinstance(date, dt.date) else dt.date.fromisoformat(str(date).strip()))
    except Exception:  # noqa: BLE001
        raise TemplateError("Bank / Balance-sheet date: write it as YYYY-MM-DD.") from None

    bsheet = {}
    for r in rows("Balance sheet"):
        key, side, label, amt = (list(r) + [None] * 4)[:4]
        if side not in ("A", "L", "E"):
            raise TemplateError(f"Balance sheet / {key}: side must be A, L or E, not '{side}'.")
        bsheet[str(key)] = (side, str(label or key), num(amt, f"Balance sheet / {key}"))
    lost = [k for k in ref.balance_sheet if k not in bsheet]
    if lost:
        raise TemplateError("Balance sheet: these lines are missing (put 0 if the bank has "
                            f"none): {', '.join(lost)}.")

    mt = {}
    for r in rows("Maturity table"):
        key = str(r[0])
        mt[key] = tuple(num(v, f"Maturity table / {key}") for v in list(r[1:7]) + [None] * (6 - len(r[1:7])))
    lost = [k for k in ref.maturity_table if k not in mt]
    if lost:
        raise TemplateError(f"Maturity table: missing rows {', '.join(lost)} (put 0s if empty).")

    bonds = []
    for i, r in enumerate(rows("Bond issues"), start=2):
        c, m, a = (list(r) + [None] * 3)[:3]
        m = m.date() if isinstance(m, dt.datetime) else m
        try:
            m = m.isoformat() if isinstance(m, dt.date) else dt.date.fromisoformat(str(m).strip()).isoformat()
        except Exception:  # noqa: BLE001
            raise TemplateError(f"Bond issues row {i}: maturity '{m}' is not YYYY-MM-DD.") from None
        if dt.date.fromisoformat(m) <= date:
            raise TemplateError(f"Bond issues row {i}: matures on {m}, not after the "
                                f"balance-sheet date {date}.")
        bonds.append((num(c, f"Bond issues row {i} coupon"), m, num(a, f"Bond issues row {i} amount")))

    pf = {}
    for r in rows("Pfandbrief by year"):
        y = int(num(r[0], "Pfandbrief year"))
        if y <= date.year:
            raise TemplateError(f"Pfandbrief by year: {y} is not after the balance-sheet year.")
        if y > date.year + 25:
            raise TemplateError(f"Pfandbrief by year: {y} is more than 25 years out.")
        pf[y] = num(r[1], f"Pfandbrief {y}")

    irrbb = {str(r[0]): (num(r[2], f"IRRBB / {r[0]} volume"), num(r[3], f"IRRBB / {r[0]} term"))
             for r in rows("IRRBB terms")}
    lost = [k for k in ref.irrbb if k not in irrbb]
    if lost:
        raise TemplateError(f"IRRBB terms: missing rows {', '.join(lost)}.")
    targets = {str(r[0]): num(r[2], f"Disclosed figures / {r[0]}") for r in rows("Disclosed figures")}
    lost = [k for k in ref.targets if k not in targets]
    if lost:
        raise TemplateError(f"Disclosed figures: missing rows {', '.join(lost)}.")

    q, r_ = (get("Pfandbrief due in months 1-3 of first year", True),
             get("Pfandbrief due in months 4-12 of first year", True))
    split = None if q is None or r_ is None else (q, r_)
    curve = np.array([get(f"Fair-value curve {n}") for n in SVENSSON])

    data = BankData(
        name=str(bank.get("Bank name") or "Uploaded bank"), balance_sheet_date=date,
        balance_sheet=bsheet, total_assets_kchf=get("Total assets"), maturity_table=mt,
        bond_issues=bonds, pfandbrief_by_year=pf,
        pfandbrief_rate=get("Pfandbrief average rate") / 100.0,
        pfandbrief_first_year_split=split,
        fin_investments_fair_value_kchf=get("Financial investments, fair value"),
        fair_value_curve=curve,
        interest_expense_cash_bonds_kchf=get("Interest expense on cash bonds"),
        personnel_cost_kchf=get("Personnel cost, latest year"),
        material_cost_kchf=get("Material cost, latest year"),
        interest_income_mchf=get("Interest income, latest year"),
        irrbb=irrbb, targets=targets)
    check(data)
    return data


def check(d: BankData) -> None:
    """The accounting identities the model relies on.  Hard errors raise;
    small rounding goes into d.warnings."""
    tol = lambda x: max(5.0, 1e-6 * abs(x))          # the reports round each cell
    a = sum(v for s, _, v in d.balance_sheet.values() if s == "A")
    le = sum(v for s, _, v in d.balance_sheet.values() if s in "LE")
    if abs(a - le) > max(tol(a), 1e-4 * abs(a)):
        raise TemplateError(f"The balance sheet does not balance: assets {a:,.0f} against "
                            f"liabilities plus equity {le:,.0f} kCHF.")
    if abs(a - d.total_assets_kchf) > max(tol(a), 1e-4 * abs(a)):
        d.warnings.append(f"Total assets {d.total_assets_kchf:,.0f} differ from the sum of "
                          f"asset lines {a:,.0f} kCHF.")
    for k, row in d.maturity_table.items():
        line = d.balance_sheet[k][2]
        if abs(sum(row) - line) > max(tol(line), 1e-3 * abs(line)):
            raise TemplateError(f"Maturity table / {k}: the buckets add up to {sum(row):,.0f} "
                                f"but the balance sheet line is {line:,.0f} kCHF.")
    bonds = sum(a for _, _, a in d.bond_issues) + sum(d.pfandbrief_by_year.values())
    line = d.balance_sheet["bonds_pfandbrief"][2]
    if abs(bonds - line) > max(tol(line), 1e-3 * abs(line)):
        raise TemplateError(f"Bond issues + Pfandbrief loans add up to {bonds:,.0f} kCHF, but "
                            f"the balance sheet line 'bonds_pfandbrief' is {line:,.0f}.")
    if d.pfandbrief_first_year_split is not None:
        y0 = d.balance_sheet_date.year + 1
        first = d.pfandbrief_by_year.get(y0, 0.0)
        if abs(sum(d.pfandbrief_first_year_split) - first) > 1.0:
            raise TemplateError(f"Bank: the first-year Pfandbrief split adds up to "
                                f"{sum(d.pfandbrief_first_year_split):,.0f} but {y0} has "
                                f"{first:,.0f} kCHF.")
    if d.fin_investments_fair_value_kchf <= 0 or d.balance_sheet["fin_investments"][2] <= 0:
        raise TemplateError("Financial investments: the book value and fair value must be positive.")
    if d.interest_income_mchf <= 0:
        raise TemplateError("Bank / Interest income must be positive.")
    if not (0 < d.fair_value_curve[4] and 0 < d.fair_value_curve[5]):
        raise TemplateError("Bank / Fair-value curve: tau1 and tau2 must be positive.")
