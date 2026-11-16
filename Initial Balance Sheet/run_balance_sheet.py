"""Build the starting balance sheet x0 (Layer 2) and check it.

    python run_balance_sheet.py --fit output/fit_sqrt.npz --out output

Needs fit_sqrt.npz from run_fit.py (for the starting curve Y0).
Writes  output/initial_state.npz        the frozen x0 (CHF millions)
        output/balance_sheet_report.json every number in layer2_report.md
"""
import argparse, json, os
import numpy as np
import balance_sheet as bs

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="output/fit_sqrt.npz")
ap.add_argument("--out", default="output")
ap.add_argument("--saron-mortgages-to", default="RE", choices=["RE", "RP"])
ap.add_argument("--cash-bonds-to", default="SF", choices=["SF", "K"])
A = ap.parse_args()

y0 = np.load(A.fit)["y0"]
opt = bs.Options(saron_mortgages_to=A.saron_mortgages_to, cash_bonds_to=A.cash_bonds_to)
x = bs.build_initial_state(y0, opt)
r = bs.report(x, y0)
os.makedirs(A.out, exist_ok=True)
bs.save(x, os.path.join(A.out, "initial_state.npz"))
with open(os.path.join(A.out, "balance_sheet_report.json"), "w") as f:
    json.dump(r, f, indent=1, default=float)

v, b = r["values_mCHF"], r["book_mCHF"]
print("Starting balance sheet, CHF millions (book -> model value at Y0)")
for s in ("C", "B", "RP", "RE", "SD", "SF", "K"):
    print(f"  {s:<3} {b[s]:10,.0f} {v[s]:10,.0f}")
print(f"  E0 (residual) {v['E']:,.0f}   book equity {r['book_equity_mCHF']:,.0f}"
      f"   CET1 {r['CET1_mCHF']:,.0f}")
print("\nConstraints at t0: model vs TKB disclosed")
k, t = r["constraints_t0"], r["targets"]
for name in ("E_over_RWA", "LCR", "NSFR"):
    print(f"  {name:<10} {k[name]:7.3f}  {t[name]:7.3f}")
print(f"  CMR        {k['CMR']:7.2f}  (min 1.00)")
print(f"  IRS(100bp) {r['IRS_100bp']:7.3f}  (limit 0.085)")
print(f"  dE +150bp  {r['dE_mCHF'][150]:7.0f}  TKB dEVE {r['dEVE_tkb_mCHF'][150]:7.0f}")
print(f"  dE -150bp  {r['dE_mCHF'][-150]:7.0f}  TKB dEVE {r['dEVE_tkb_mCHF'][-150]:7.0f}")
print(f"\nSaved {A.out}/initial_state.npz and balance_sheet_report.json")

# Presentation only: an Excel copy of the report above.  Runs after
# everything is saved, so it can never change or block a result.
try:
    import export_layer2
    fit_date = str(np.load(A.fit)["valuation_date"])
    xlsx = export_layer2.write_workbook(
        A.out, fit_label=f"Valued on the yield curve of {fit_date} ({A.fit}).")
    print(f"Saved {xlsx}")
except Exception as err:                           # noqa: BLE001
    print(f"(Excel export skipped: {err})")