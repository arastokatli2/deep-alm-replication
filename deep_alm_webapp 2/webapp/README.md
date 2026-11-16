# Deep ALM web app

A browser app that shows all five layers, with Layer 5 (the network against the benchmarks) as the main result.

## Install (once)

```bash
conda activate alm
pip install streamlit plotly openpyxl
```

## Where it goes

Put the `webapp` folder inside `Research Internship FEB`, next to your layer folders.
Also replace `balance_sheet.py` in the Initial Balance Sheet folder with the new copy.
Its results are identical (see "The one code change" below).

## Start

```bash
conda activate alm
cd ~/path/to/"Research Internship FEB"
bash webapp/start_app.sh
```

The browser opens at http://localhost:8501. To stop the app, press Ctrl+C in the terminal.

The first time it starts, the app builds a run called **TKB 2022 – paper setup** from the results you already have: your fitted curve and your trained network. The launcher finds both by itself. This takes about a minute, and every page fills in when it is done.

## Using it

- **Overview, 1–5**: one page per layer, for presenting. Layer 5 has tabs:
  - Scoreboard
  - What it does
  - Limits
  - Over time
  - Training
  - Train again
- **Data & runs**: make a new run.
  - **Layer 1**: upload an SNB curve file (the `rendopar` CSV) and choose a start date, or reuse a curve that is already fitted.
  - **Layer 2**: download the template (it comes filled with TKB 2022), change the numbers, and upload it. The app checks the file first. For example, the balance sheet must balance, and the maturity buckets must add up to each balance-sheet line.
  - Layers 1–3 are then worked out in the background, in about 1–3 minutes, followed by Layer 5 (below).
- **Layer 5 shows only trained results.** It never shows a network trained on different data.
  - **Same data as a network you already have** (for example TKB, 15.07.2022): the app uses that network straight away. No training.
  - **New data**: after Layers 1–3, the app trains the benchmarks and the network on it in the background, with the same settings as your command line (8,000 scenarios, 25 + 60 epochs). This takes about 15–30 minutes. Layer 5 shows a progress bar until then; Layers 1–4 can be presented meanwhile.
  - Every trained network is kept in `networks/`. A later run on the same data reuses it instantly.

Every page has download buttons for that layer's Excel workbook and its JSON file.

## Files

| File | What it does |
|---|---|
| `app.py` | the pages |
| `jobs.py` | runs your unchanged scripts for one run (`run_fit.py`, `run_balance_sheet.py`, `run_simulator.py`, `run_deep_alm.py`, `score_deep_alm.py`) |
| `bank_data.py` | the Layer 2 template: writes it, reads it, checks it |
| `workbooks.py`, `export_layer3.py`, `export_layer5.py` | the Excel workbooks for Layers 3 and 5 |
| `runs/` | one folder per run, made by the app |
| `networks/` | the stored network, copied from your results |

## The one code change

`balance_sheet.py` used to have TKB's 2022 values written into two functions. They are now named constants, so the app can swap in an uploaded bank:
- `PFANDBRIEF_FIRST_YEAR_SPLIT_KCHF`
- the `NMD_TARGET_Y` default
- the first bond year taken from `VALUATION_DATE`

On TKB's own data the output is identical to the old version: E0 is 2,216.717 on the July curve and 2,037.318 on the December curve, and all 89 tests pass.
