"""The Layer 3 and Layer 5 workbooks for a run folder.

Thin wrappers that run export_layer3.py / export_layer5.py on the JSON files
the pipeline scripts wrote, so the workbooks can also be rebuilt by hand.
"""
from __future__ import annotations

import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script: str, args: list) -> None:
    argv = sys.argv
    sys.argv = [script] + [str(a) for a in args]
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        sys.argv = argv


def layer3(folder: str) -> str:
    out = os.path.join(folder, "layer3_results.xlsx")
    _run("export_layer3.py", [os.path.join(folder, "sim_rollover.json"),
                              os.path.join(folder, "sim_none.json"), out])
    return out


def layer5(folder: str, date: str | None = None, bank: str | None = None) -> str:
    """folder holds comparison.json and deep_alm_report.json."""
    run = os.path.dirname(os.path.abspath(folder))
    if date is None or bank is None:
        try:
            with open(os.path.join(run, "settings.json")) as f:
                s = json.load(f)
            date = date or ".".join(reversed(s.get("valuation_date", "").split("-")))
            bank = bank or s.get("bank_name", "the bank")
        except OSError:
            pass
    out = os.path.join(folder, "layer5_results.xlsx")
    _run("export_layer5.py", [os.path.join(folder, "comparison.json"),
                              os.path.join(folder, "deep_alm_report.json"), out,
                              date or "the start date", bank or "the bank"])
    return out
