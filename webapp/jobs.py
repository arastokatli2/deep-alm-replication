"""Run one layer of the pipeline for a web-app run folder.

    python jobs.py <step> --run RUN_DIR

steps: layer1, layer2, layer3, layer5-stored, layer5-train, all

Each step runs your existing script unchanged (run_fit.py, run_balance_sheet.py,
run_simulator.py, score_deep_alm.py, run_deep_alm.py), inside use_bank() when
the run has its own bank workbook.  So a run on TKB's built-in data reproduces
your command-line results exactly.

The app starts these as separate processes and follows their logs; each step
writes <run>/logs/<step>.exit (0 = done) when it finishes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def script_next_to(module: str, script: str) -> str:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot find {module}.py: is its folder on PYTHONPATH?")
    path = os.path.join(os.path.dirname(spec.origin), script)
    if not os.path.isfile(path):
        raise RuntimeError(f"{script} is not next to {module}.py ({path})")
    return path


def load_settings(run: str) -> dict:
    with open(os.path.join(run, "settings.json")) as f:
        return json.load(f)


def bank_of(run: str):
    """The run's bank, or None for the data built into balance_sheet.py."""
    import bank_data as bd
    s = load_settings(run)
    if s.get("bank_workbook"):
        return bd.read_workbook(os.path.join(run, s["bank_workbook"]))
    return None


def bank_context(run: str):
    import bank_data as bd
    b = bank_of(run)
    return bd.use_bank(b) if b is not None else contextlib.nullcontext()


def interest_income(run: str) -> float:
    b = bank_of(run)
    return 303.354 if b is None else b.interest_income_mchf


def run_script(path: str, args: list, ctx=None) -> None:
    """Execute a script as if from the command line, in this process."""
    args = [str(a) for a in args]
    print(f"\n$ python {os.path.basename(path)} {' '.join(args)}", flush=True)
    argv = sys.argv
    sys.argv = [path] + [str(a) for a in args]
    try:
        with (ctx or contextlib.nullcontext()):
            runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = argv


def fit_fingerprint(fit_path: str) -> str:
    z = np.load(fit_path)
    h = hashlib.sha1()
    for k in ("y0", "V", "valuation_date"):
        h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:12]


def data_fingerprint(run: str) -> dict:
    import bank_data as bd
    b = bank_of(run)
    return {"bank": (b or bd.current_bank()).fingerprint(),
            "curve": fit_fingerprint(os.path.join(run, "layer1", "fit_sqrt.npz"))}


# ===================================================================== steps
def layer1(run: str) -> None:
    s = load_settings(run)
    out = os.path.join(run, "layer1")
    if not s.get("snb_csv"):
        if not os.path.isfile(os.path.join(out, "fit_sqrt.npz")):
            raise RuntimeError("This run has no SNB file and no fitted curve.")
        print("Layer 1 uses the fitted curve copied into this run; nothing to fit.")
        if not os.path.isfile(os.path.join(out, "layer1_results.xlsx")):
            import export_layer1
            print("saved", export_layer1.write_workbook(
                out, seed=s.get("seed", 20221230), n_scenarios=s.get("scenarios", 4000)))
        return
    os.makedirs(out, exist_ok=True)
    run_script(script_next_to("yieldcurve", "run_fit.py"),
               ["--csv", os.path.join(run, s["snb_csv"]), "--out", out,
                "--valuation-date", s["valuation_date"], "--model", s.get("model", "paper"),
                "--scenarios", s.get("scenarios", 4000), "--seed", s.get("seed", 20221230)])


def layer2(run: str) -> None:
    out = os.path.join(run, "layer2")
    os.makedirs(out, exist_ok=True)
    run_script(script_next_to("balance_sheet", "run_balance_sheet.py"),
               ["--fit", os.path.join(run, "layer1", "fit_sqrt.npz"), "--out", out],
               bank_context(run))


def layer3(run: str) -> None:
    s = load_settings(run)
    out = os.path.join(run, "layer3")
    os.makedirs(out, exist_ok=True)
    for policy in ("none", "rollover"):
        run_script(script_next_to("simulator", "run_simulator.py"),
                   ["--fit", os.path.join(run, "layer1", "fit_sqrt.npz"), "--out", out,
                    "--policy", policy, "--scenarios", s.get("scenarios", 4000),
                    "--seed", s.get("seed", 20221230),
                    "--interest-income", interest_income(run)],
                   bank_context(run))
        shutil.copy(os.path.join(out, "simulator_report.json"),
                    os.path.join(out, f"sim_{policy}.json"))
    try:
        import workbooks
        print("saved", workbooks.layer3(out), flush=True)
    except Exception as err:                       # noqa: BLE001
        print(f"(Layer 3 workbook skipped: {err})")


def _widths(folder: str) -> str:
    try:
        with open(os.path.join(folder, "deep_alm_report.json")) as f:
            return ",".join(map(str, json.load(f)["setup"]["widths"]))
    except Exception:                              # noqa: BLE001
        return "512,512,256,128"


def layer5_stored(run: str) -> None:
    """Score the stored network and benchmarks on this run's data."""
    s = load_settings(run)
    src = s["stored_network"]
    out = os.path.join(run, "layer5_stored")
    os.makedirs(out, exist_ok=True)
    for name in ("BM_E", "BM_C", "BM_D", "DeepALM"):
        shutil.copy(os.path.join(src, f"deep_alm_{name}.pkl"), out)
    for extra in ("deep_alm_report.json", "trained_on.json"):
        if os.path.isfile(os.path.join(src, extra)):
            shutil.copy(os.path.join(src, extra), out)
    run_script(script_next_to("deep_alm", "score_deep_alm.py"),
               ["--fit", os.path.join(run, "layer1", "fit_sqrt.npz"), "--out", out,
                "--interest-income", interest_income(run), "--widths", _widths(out)],
               bank_context(run))
    _layer5_workbook(out)


def layer5_train(run: str, *, bm_epochs: int, nn_epochs: int, train_scenarios: int,
                 widths: str) -> None:
    out = os.path.join(run, "layer5_trained")
    os.makedirs(out, exist_ok=True)
    fit = os.path.join(run, "layer1", "fit_sqrt.npz")
    ctx = bank_context(run)
    run_script(script_next_to("deep_alm", "run_deep_alm.py"),
               ["--fit", fit, "--out", out, "--bm-epochs", bm_epochs,
                "--nn-epochs", nn_epochs, "--train-scenarios", train_scenarios,
                "--widths", widths, "--interest-income", interest_income(run)], ctx)
    with open(os.path.join(out, "trained_on.json"), "w") as f:
        json.dump(data_fingerprint(run), f)
    run_script(script_next_to("deep_alm", "score_deep_alm.py"),
               ["--fit", fit, "--out", out, "--interest-income", interest_income(run),
                "--widths", widths], bank_context(run))
    _layer5_workbook(out)
    register_network(run, out)


NETWORKS = os.path.join(HERE, "networks")
PKLS = [f"deep_alm_{n}.pkl" for n in ("BM_E", "BM_C", "BM_D", "DeepALM")]


def find_network(fp: dict) -> str | None:
    """A stored network trained on exactly this data (bank + curve), if any."""
    if not os.path.isdir(NETWORKS):
        return None
    for name in sorted(os.listdir(NETWORKS)):
        d = os.path.join(NETWORKS, name)
        try:
            with open(os.path.join(d, "trained_on.json")) as f:
                if json.load(f) == fp and all(os.path.isfile(os.path.join(d, p)) for p in PKLS):
                    return d
        except (OSError, ValueError):
            continue
    return None


def register_network(run: str, folder: str) -> None:
    """Keep a freshly trained network, so a later run on the same data reuses it."""
    fp = data_fingerprint(run)
    old = find_network(fp)
    if old:
        shutil.rmtree(old, ignore_errors=True)
    dst = os.path.join(NETWORKS, f"{os.path.basename(run)}-{fp['curve']}")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(folder, dst)
    print("network kept for reuse:", dst, flush=True)


def layer5(run: str, **train_args) -> None:
    """The main result.  Same data as a network you already have: use it (seconds).
    New data: train on it (about 15-30 minutes).  Only trained-on-this-data results."""
    out = os.path.join(run, "layer5_trained")
    fp = data_fingerprint(run)
    found = find_network(fp)
    if found:
        print(f"A network trained on exactly this data exists ({os.path.basename(found)}): "
              "using it, no training needed.", flush=True)
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(found, out)
        if not os.path.isfile(os.path.join(out, "comparison.json")):
            run_script(script_next_to("deep_alm", "score_deep_alm.py"),
                       ["--fit", os.path.join(run, "layer1", "fit_sqrt.npz"), "--out", out,
                        "--interest-income", interest_income(run), "--widths", _widths(out)],
                       bank_context(run))
        if not os.path.isfile(os.path.join(out, "layer5_results.xlsx")):
            _layer5_workbook(out)
        return
    print("No network has been trained on this data yet: training one now.", flush=True)
    layer5_train(run, **train_args)


def _layer5_workbook(out: str) -> None:
    try:
        import workbooks
        print("saved", workbooks.layer5(out), flush=True)
    except Exception as err:                       # noqa: BLE001
        print(f"(Layer 5 workbook skipped: {err})")


# ====================================================================== main
STEPS = {"layer1": layer1, "layer2": layer2, "layer3": layer3,
         "layer5-stored": layer5_stored,
         # only (re)build the workbook from scores that are already there
         "layer5-book": lambda run: _layer5_workbook(os.path.join(run, "layer5_stored"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", help="comma-separated: layer1,layer2,layer3,"
                                  "layer5-stored,layer5-train, or 'all'")
    ap.add_argument("--run", required=True)
    ap.add_argument("--bm-epochs", type=int, default=25)
    ap.add_argument("--nn-epochs", type=int, default=60)
    ap.add_argument("--train-scenarios", type=int, default=8000)
    ap.add_argument("--widths", default="512,512,256,128")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    os.makedirs(os.path.join(run, "logs"), exist_ok=True)
    exit_file = os.path.join(run, "logs", "job.exit")
    if os.path.exists(exit_file):
        os.remove(exit_file)
    steps = ["layer1", "layer2", "layer3", "layer5"] if a.steps == "all" \
        else [x.strip() for x in a.steps.split(",") if x.strip()]
    targs = dict(bm_epochs=a.bm_epochs, nn_epochs=a.nn_epochs,
                 train_scenarios=a.train_scenarios, widths=a.widths)
    code = 1
    try:
        for step in steps:
            print(f"\n===== {step} =====", flush=True)
            if step == "layer5-train":
                layer5_train(run, **targs)
            elif step == "layer5":
                layer5(run, **targs)
            else:
                STEPS[step](run)
        code = 0
        print("\nDONE", flush=True)
    except SystemExit as e:                        # a script's own argparse error
        code = int(e.code or 1)
        print(f"\nFAILED (exit {code})", flush=True)
    except Exception:                              # noqa: BLE001
        traceback.print_exc()
        print("\nFAILED", flush=True)
    finally:
        with open(exit_file, "w") as f:
            f.write(str(code))
    sys.exit(code)


if __name__ == "__main__":
    main()
