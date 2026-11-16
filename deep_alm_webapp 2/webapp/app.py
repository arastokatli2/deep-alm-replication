"""Deep ALM - the five layers as a web app.

    streamlit run app.py

Layer 1 (yield curves) and Layer 2 (the bank) take uploaded data; Layers 3-5
follow from them, and Layer 5 - the trained decision network against every
benchmark - is the main result.  The heavy lifting is done by the existing
scripts, run unchanged by jobs.py in a separate process; this file only starts
those jobs and draws what they write.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("ALM_RUNS_DIR", os.path.join(HERE, "runs"))
NETWORKS = os.path.join(HERE, "networks")
JOBS = os.path.join(HERE, "jobs.py")
DEFAULT_RUN = "tkb-2022-paper"
H = 60

st.set_page_config(page_title="Deep ALM", page_icon="🏦", layout="wide")

# ================================================================ look
ORDER = ["nothing", "rollover", "BM_E", "BM_C", "BM_D", "DeepALM"]
NAME = {"nothing": "No decisions", "rollover": "Roll-over rule", "BM_E": "BM_E (1/N)",
        "BM_C": "BM_C (learnt mix)", "BM_D": "BM_D (mix per month)",
        "DeepALM": "Deep ALM network"}
COL = {"nothing": "#9E9E9E", "rollover": "#26A69A", "BM_E": "#90CAF9",
       "BM_C": "#42A5F5", "BM_D": "#1565C0", "DeepALM": "#E65100",
       "none": "#9E9E9E", "paper": "#1565C0", "termprem": "#26A69A"}
LIMITS = {"LCR": (1.05, "min"), "NSFR": (1.05, "min"), "CMR": (1.0, "min"),
          "E_over_RWA": (0.17, "min"), "IRS": (0.085, "abs")}
LIMIT_NAME = {"LCR": "Liquidity coverage LCR", "NSFR": "Stable funding NSFR",
              "CMR": "Reserve ratio CMR", "E_over_RWA": "Capital E/RWA",
              "IRS": "Rate sensitivity IRS", "EYR": "Minimum annual return EYR"}
LAYERS = [("1", "Yield curves"), ("2", "Balance sheet"), ("3", "Simulation"),
          ("4", "Objective"), ("5", "Decisions")]


def fig_base(fig, *, height=360, y=None, x=None, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0) if legend else None,
                      showlegend=legend, yaxis_title=y, xaxis_title=x,
                      font=dict(size=13))
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False})


# ================================================================ paths
def module_dir(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return os.path.dirname(spec.origin) if spec and spec.origin else None


def pythonpath() -> str:
    dirs = [HERE] + [module_dir(m) for m in ("yieldcurve", "balance_sheet", "simulator",
                                              "loss", "deep_alm", "constraints")]
    seen, out = set(), []
    for d in dirs + os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return os.pathsep.join(out)


def source_dirs() -> dict:
    """Where the command-line runs left their outputs, to seed the first run."""
    return {"fit": os.environ.get("ALM_FIT_DIR") or os.path.join(module_dir("yieldcurve") or "", "output_paper"),
            "network": os.environ.get("ALM_NETWORK_DIR") or os.path.join(module_dir("deep_alm") or "", "output_paper")}


def rp(run, *parts):
    return os.path.join(RUNS, run, *parts)


def exists(run, *parts):
    return os.path.isfile(rp(run, *parts))


@st.cache_data(show_spinner=False)
def _jload(path, mtime):
    with open(path) as f:
        return json.load(f)


def jload(path):
    return _jload(path, os.path.getmtime(path)) if os.path.isfile(path) else None


def settings(run) -> dict:
    return jload(rp(run, "settings.json")) or {}


def save_settings(run, s):
    with open(rp(run, "settings.json"), "w") as f:
        json.dump(s, f, indent=1)


def list_runs() -> list:
    if not os.path.isdir(RUNS):
        return []
    runs = [d for d in os.listdir(RUNS) if os.path.isfile(rp(d, "settings.json"))]
    return sorted(runs, key=lambda d: (d != DEFAULT_RUN, settings(d).get("created", "")))


# ================================================================ jobs
def job(run) -> dict:
    """State of the run's job: none / running / done / failed."""
    info = jload(rp(run, "logs", "job.json"))
    if not info:
        return {"state": "none"}
    ex = rp(run, "logs", "job.exit")
    if os.path.isfile(ex):
        code = open(ex).read().strip()
        info["state"] = "done" if code == "0" else "failed"
        return info
    pid = info.get("pid")
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            info["state"] = "failed"          # ended without writing its exit code
            return info
    except ChildProcessError:
        pass                                  # not our child: fall back to kill(0)
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        info["state"] = "running"
    except OSError:
        info["state"] = "failed"
    return info


def start_job(run, steps: str, extra=()):
    if job(run)["state"] == "running":
        st.warning("A job is already running for this run.")
        return
    os.makedirs(rp(run, "logs"), exist_ok=True)
    for f in ("job.exit",):
        if exists(run, "logs", f):
            os.remove(rp(run, "logs", f))
    log = open(rp(run, "logs", "job.log"), "w")
    env = dict(os.environ, PYTHONPATH=pythonpath(), PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "-u", JOBS, steps, "--run", rp(run), *map(str, extra)],
                         stdout=log, stderr=subprocess.STDOUT, cwd=rp(run), env=env,
                         start_new_session=True)
    with open(rp(run, "logs", "job.json"), "w") as f:
        json.dump({"steps": steps, "pid": p.pid, "started": dt.datetime.now().isoformat(timespec="seconds"),
                   "extra": list(map(str, extra))}, f)


def stop_job(run):
    info = job(run)
    if info["state"] == "running":
        try:
            os.killpg(info["pid"], signal.SIGTERM)
        except OSError:
            pass
        with open(rp(run, "logs", "job.exit"), "w") as f:
            f.write("stopped")


def log_tail(run, n=40) -> str:
    path = rp(run, "logs", "job.log")
    if not os.path.isfile(path):
        return ""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-n:])


def train_progress(run) -> tuple[float, str]:
    """Fraction done and a plain description, read from the training log."""
    info = jload(rp(run, "logs", "job.json")) or {}
    extra = info.get("extra", [])
    get = lambda flag, d: int(extra[extra.index(flag) + 1]) if flag in extra else d
    bm, nn = get("--bm-epochs", 25), get("--nn-epochs", 60)
    path = rp(run, "logs", "job.log")
    text = open(path, errors="replace").read() if os.path.isfile(path) else ""
    phases = [("BM_E", bm), ("BM_C", bm), ("BM_D", bm), ("DeepALM", nn)]
    total = 3 * bm + nn + 6
    done, label = 0, "preparing scenarios"
    for i, (name, n) in enumerate(phases):
        start = text.rfind(f"\n{name}:")
        if start < 0:
            break
        eps = re.findall(r"epoch\s+(\d+)/(\d+)", text[start:])
        k = int(eps[-1][0]) if eps else 0
        done = 3 + sum(m for _, m in phases[:i]) + k
        label = f"training {NAME[name]}: epoch {k} of {n}"
    if "scored DeepALM" in text:
        done, label = total, "scoring"
    elif "Saved" in text and "deep_alm_report.json" in text:
        done, label = total - 3, "scoring all six strategies"
    return min(done / total, 1.0), label


def is_training(run) -> bool:
    path = rp(run, "logs", "job.log")
    if not os.path.isfile(path):
        return False
    text = open(path, errors="replace").read()
    return "===== layer5-train" in text or "No network has been trained on this data yet" in text


# ================================================================ seeding
def fingerprint_of(run) -> dict | None:
    """What a network must have been trained on to be valid for this run."""
    if not exists(run, "layer1", "fit_sqrt.npz"):
        return None
    sys.path.insert(0, HERE)
    import jobs
    try:
        return jobs.data_fingerprint(rp(run))
    except Exception:                                   # noqa: BLE001
        return None


def ensure_default_run():
    """First start: build a run from the command-line outputs you already have."""
    if os.path.isdir(rp(DEFAULT_RUN)):
        return
    src = source_dirs()
    fit = src["fit"]
    if not os.path.isfile(os.path.join(fit, "fit_sqrt.npz")):
        return
    os.makedirs(rp(DEFAULT_RUN, "layer1"), exist_ok=True)
    for f in ("fit_sqrt.npz", "fit_eig.npz", "curve_features.npz", "report.json",
              "layer1_results.xlsx"):
        if os.path.isfile(os.path.join(fit, f)):
            shutil.copy(os.path.join(fit, f), rp(DEFAULT_RUN, "layer1"))
    z = np.load(rp(DEFAULT_RUN, "layer1", "fit_sqrt.npz"))
    s = {"title": "TKB 2022 - paper setup", "bank_name": "Thurgauer Kantonalbank",
         "valuation_date": str(z["valuation_date"]), "model": "paper", "snb_csv": None,
         "bank_workbook": None, "scenarios": 4000, "seed": 20221230,
         "created": dt.datetime.now().isoformat(timespec="seconds"),
         "note": "Seeded from your command-line outputs."}
    net = src["network"]
    pkls = [os.path.join(net, f"deep_alm_{n}.pkl") for n in ("BM_E", "BM_C", "BM_D", "DeepALM")]
    if all(os.path.isfile(p) for p in pkls):
        dst = os.path.join(NETWORKS, "tkb-2022-paper")
        os.makedirs(dst, exist_ok=True)
        for p in pkls:
            shutil.copy(p, dst)
        for f in ("deep_alm_report.json", "comparison.json"):
            if os.path.isfile(os.path.join(net, f)):
                shutil.copy(os.path.join(net, f), dst)
        save_settings(DEFAULT_RUN, s)
        fp = fingerprint_of(DEFAULT_RUN)          # trained on exactly this setup
        with open(os.path.join(dst, "trained_on.json"), "w") as f:
            json.dump(fp, f)
    save_settings(DEFAULT_RUN, s)
    # layer5 finds the network trained on exactly this data and uses it (no training)
    start_job(DEFAULT_RUN, "layer1,layer2,layer3,layer5")


# ================================================================ state
ensure_default_run()
runs = list_runs()
if "run" not in st.session_state or st.session_state["run"] not in runs:
    st.session_state["run"] = runs[0] if runs else None
RUN = st.session_state["run"]


def l5_sources(run) -> dict:
    out = {}
    for key, folder, label in (("trained", "layer5_trained", "Trained on this run's data"),):
        if exists(run, folder, "comparison.json") and l5_valid(run, folder) is not False:
            out[key] = (folder, label)
    return out


def l5_valid(run, folder) -> bool | None:
    have = jload(rp(run, folder, "trained_on.json"))
    want = fingerprint_of(run)
    if not have or not want:
        return None
    return have == want


def status(run) -> dict:
    """Which layers have results for this run."""
    return {"1": exists(run, "layer1", "report.json"),
            "2": exists(run, "layer2", "balance_sheet_report.json"),
            "3": exists(run, "layer3", "sim_none.json") and exists(run, "layer3", "sim_rollover.json"),
            "4": exists(run, "layer3", "sim_none.json") and exists(run, "layer3", "sim_rollover.json"),
            "5": bool(l5_sources(run))}


def need(layer, what="This layer"):
    """Stop the page with a helpful message if the layer has no results yet."""
    if RUN is None:
        st.info("No run yet. Go to **Data & runs** to upload data and start one.")
        st.stop()
    if not status(RUN)[layer]:
        j = job(RUN)
        if j["state"] == "running":
            st.info(f"{what} is being computed (job: {j['steps']}). This page fills in when it finishes.")
            live_job_panel(RUN, compact=True)
        else:
            st.warning(f"{what} has no results for this run yet. Start it from **Data & runs**.")
        st.stop()


def downloads(items):
    cols = st.columns(len(items))
    for c, (label, path, mime) in zip(cols, items):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                c.download_button(label, f.read(), file_name=os.path.basename(path), mime=mime,
                                  icon=":material/download:")


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ================================================================ sidebar
with st.sidebar:
    st.markdown("### Deep ALM")
    if runs:
        st.session_state["run"] = st.selectbox(
            "Run", runs, index=runs.index(RUN) if RUN in runs else 0,
            format_func=lambda r: settings(r).get("title", r))
        RUN = st.session_state["run"]
        s_ = settings(RUN)
        st.caption(f"{s_.get('bank_name', '')} · curve of "
                   f"{'.'.join(reversed(s_.get('valuation_date', '').split('-')))}")
        stt = status(RUN)
        st.markdown(" · ".join(f"{'✅' if stt[k] else '⬜'} {k}" for k, _ in LAYERS))
        j = job(RUN)
        if j["state"] == "running":
            st.info(f"Running: {j['steps']}", icon=":material/progress_activity:")
        elif j["state"] == "failed":
            st.error("Last job failed - see Data & runs.", icon=":material/error:")


# ================================================================ pages
def live_job_panel(run, compact=False):
    @st.fragment(run_every=3)
    def _panel():
        j = job(run)
        if j["state"] == "running":
            if "layer5" in j.get("steps", "") and is_training(run):
                frac, label = train_progress(run)
                st.progress(frac, text=f"Training on this run's data · {label}")
            else:
                st.progress(0.5 if not compact else 0.5, text=f"Running {j['steps']} …")
            if not compact:
                st.code(log_tail(run, 15) or "starting …", language=None)
        elif j["state"] in ("done", "failed") and st.session_state.get(f"_seen_{run}") != j.get("started"):
            st.session_state[f"_seen_{run}"] = j.get("started")
            st.rerun()
    _panel()


# ---------------------------------------------------------------- overview
def page_overview():
    st.title("Deep Asset Liability Management")
    st.markdown(
        "A bank decides each month how much to **invest** in bonds and how much to "
        "**borrow**, under six regulatory limits, for five years of uncertain "
        "interest rates. This app runs the full replication of Englisch et al. (2023) "
        "and Müller (2023) on a bank's own data: **yield curves → balance sheet → "
        "simulation → objective → decisions**.")
    if RUN is None:
        st.info("No run yet. Go to **Data & runs** to upload data and start one.")
        return
    stt = status(RUN)
    cols = st.columns(5)
    facts = overview_facts(RUN)
    for c, (k, title) in zip(cols, LAYERS):
        with c.container(border=True):
            st.markdown(f"**{k} · {title}** {'✅' if stt[k] else '⬜'}")
            st.caption(DESCR[k])
            if facts.get(k):
                st.metric(facts[k][0], facts[k][1])
    st.divider()
    src = l5_sources(RUN)
    if not src:
        need("5", "The main result (Layer 5)")
    key = "trained" if "trained" in src else "stored"
    cmp = jload(rp(RUN, src[key][0], "comparison.json"))["results"]
    st.subheader("Main result · Layer 5")
    validity_banner(RUN, src[key][0])
    hero_metrics(cmp)
    show(loss_bar(cmp))


DESCR = {"1": "SNB government-bond curves, fitted with HJM-PCA, simulated 5 years ahead.",
         "2": "The bank's published balance sheet as monthly cash flows, valued on the curve.",
         "3": "The bank walked forward 60 months on every scenario.",
         "4": "Return target, limit breaches, and how they trade off.",
         "5": "The trained network against every benchmark."}


def overview_facts(run) -> dict:
    out = {}
    r1 = jload(rp(run, "layer1", "report.json"))
    if r1:
        v = r1.get("sim_sqrt", {}).get("short_T_pct", {}).get("median")
        if v is not None:
            out["1"] = ("Median 1m rate in 5y", f"{v:.2f}%")
    r2 = jload(rp(run, "layer2", "balance_sheet_report.json"))
    if r2:
        out["2"] = ("Starting equity", f"{r2['values_mCHF']['E']:,.0f} m")
    r3 = jload(rp(run, "layer3", "sim_none.json"))
    if r3:
        out["3"] = ("Doing nothing, equity ×", f"{r3['ET_over_E0']:.3f}")
        out["4"] = ("Doing nothing, loss", f"{r3['loss']['loss']['total']:,.0f}")
    src = l5_sources(run)
    if src:
        key = "trained" if "trained" in src else "stored"
        c = jload(rp(run, src[key][0], "comparison.json"))["results"]
        out["5"] = ("Network loss", f"{c['DeepALM']['total']:,.2f}")
    return out


# ---------------------------------------------------------------- layer 1
@st.cache_data(show_spinner="Rebuilding the scenarios…")
def l1_scenarios(fit_path, mtime, n, seed):
    import yieldcurve as yc
    out = {}
    for variant in ("sqrt", "term_premium"):
        y = yc.load_scenarios(fit_path, n_scenarios=n, seed=seed, variant=variant).yields
        s = y[:, :, 0] * 100
        out[variant] = {
            "fan": {q: np.quantile(s, q, axis=0) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "T_mean": y[:, -1, :].mean(0) * 100,
            "T_q": {q: np.quantile(y[:, -1, :] * 100, q, axis=0) for q in (0.05, 0.5, 0.95)},
            "paths1m": s[:10], "paths10y": y[:10, :, 119] * 100}
        del y
    return out


def page_layer1():
    st.title("1 · Yield curves")
    st.caption("Where interest rates could go: SNB Confederation-bond curves, a 3-factor HJM-PCA "
               "model fitted to their history, and 4,000 simulated five-year futures.")
    need("1", "Layer 1")
    rep = jload(rp(RUN, "layer1", "report.json"))
    z = np.load(rp(RUN, "layer1", "fit_sqrt.npz"))
    s = settings(RUN)
    tenors, y0, f0, V = z["tenors"], z["y0"], z["f0"], z["V"]
    sims = l1_scenarios(rp(RUN, "layer1", "fit_sqrt.npz"),
                        os.path.getmtime(rp(RUN, "layer1", "fit_sqrt.npz")),
                        int(s.get("scenarios", 4000)), int(s.get("seed", 20221230)))
    ev = rep.get("fit_sqrt", {}).get("explained_variance", [])
    sim = rep.get("sim_sqrt", {})
    c = st.columns(5)
    c[0].metric("Curve date", str(z["valuation_date"]))
    c[1].metric("History used", f"{rep['history']['first'][:4]}–{rep['history']['last'][:4]}",
                help=f"{rep['history']['weekly_observations']:,} weekly curves")
    c[2].metric("3 factors explain", f"{sum(ev):.1%}" if ev else "–")
    c[3].metric("Median 1m rate in 5 years", f"{sim['short_T_pct']['median']:.2f}%",
                help=f"5–95%: {sim['short_T_pct']['q05']:.2f}% to {sim['short_T_pct']['q95']:.2f}%")
    c[4].metric("Curves inverted at 5 years", f"{sim['curve_shape_T']['share_inverted']:.0%}")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        m = tenors <= 20
        fig.add_trace(go.Scatter(x=tenors[m], y=y0[m] * 100, name="Spot yield", line=dict(color="#1565C0", width=3)))
        fig.add_trace(go.Scatter(x=tenors[m], y=f0[m] * 100, name="Forward rate", line=dict(color="#E65100", dash="dot")))
        fig.add_vrect(x0=15, x1=20, fillcolor="#EEEEEE", line_width=0, annotation_text="roll-down only",
                      annotation_position="top left")
        st.markdown("**Today's curve**")
        show(fig_base(fig, y="%", x="maturity (years)"))
    with l:
        pass
    with r:
        variant = st.segmented_control("Model", ["Paper (P = Q)", "Version B (term premium)"],
                                       default="Paper (P = Q)", key="l1v")
        sv = sims["term_premium" if variant and variant.startswith("Version") else "sqrt"]
        t = np.arange(61) / 12
        fan = sv["fan"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[fan[0.95], fan[0.05][::-1]], fill="toself",
                                 fillcolor="rgba(21,101,192,0.12)", line=dict(width=0), name="5–95%"))
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[fan[0.75], fan[0.25][::-1]], fill="toself",
                                 fillcolor="rgba(21,101,192,0.25)", line=dict(width=0), name="25–75%"))
        fig.add_trace(go.Scatter(x=t, y=fan[0.5], name="median", line=dict(color="#1565C0", width=3)))
        st.markdown("**1-month rate over the next five years**")
        show(fig_base(fig, y="%", x="years ahead"))

    l, r = st.columns(2)
    with l:
        idx = np.arange(180)
        tt = tenors[:180]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[tt, tt[::-1]], y=np.r_[sv["T_q"][0.95], sv["T_q"][0.05][::-1]],
                                 fill="toself", fillcolor="rgba(230,81,0,0.12)", line=dict(width=0), name="in 5 years, 5–95%"))
        fig.add_trace(go.Scatter(x=tt, y=sv["T_mean"], name="in 5 years, mean", line=dict(color="#E65100", width=3)))
        fig.add_trace(go.Scatter(x=tt, y=y0[idx] * 100, name="today", line=dict(color="#1565C0", width=3)))
        st.markdown("**The whole curve: today and in five years**")
        show(fig_base(fig, y="%", x="maturity (years)"))
    with r:
        fig = go.Figure()
        for j, colr in zip(range(V.shape[1]), ("#1565C0", "#E65100", "#26A69A")):
            share = f" ({ev[j]:.1%})" if ev else ""
            fig.add_trace(go.Scatter(x=tenors[:180], y=V[:180, j] * 1e4, name=f"Factor {j + 1}{share}",
                                     line=dict(color=colr, width=3)))
        st.markdown("**The three shocks that move the curve**")
        show(fig_base(fig, y="bp per year", x="maturity (years)"))

    with st.expander("Ten single scenarios, and the three model versions compared"):
        fig = go.Figure()
        for i in range(10):
            fig.add_trace(go.Scatter(x=np.arange(61), y=sv["paths1m"][i], name=f"#{i + 1}",
                                     line=dict(width=1.5)))
        show(fig_base(fig, y="1-month rate, %", x="month", legend=False, height=300))
        rows = []
        for key, label in (("sim_sqrt", "Paper (P = Q)"), ("sim_term_premium", "Version B (term premium)"),
                           ("sim_eigenvalue", "Printed formula (rejected)")):
            if key in rep:
                x = rep[key]
                rows.append({"Model": label,
                             "Median 1m in 5y (%)": x["short_T_pct"]["median"],
                             "5th pct (%)": x["short_T_pct"]["q05"], "95th pct (%)": x["short_T_pct"]["q95"],
                             "Curves inverted": x["curve_shape_T"]["share_inverted"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Curves inverted": st.column_config.NumberColumn(format="percent")})
    downloads([("Layer 1 workbook", rp(RUN, "layer1", "layer1_results.xlsx"), XLSX),
               ("report.json", rp(RUN, "layer1", "report.json"), "application/json")])


# ---------------------------------------------------------------- layer 2
SLOT = {"C": "Cash", "B": "Investments", "RP": "Mortgages", "RE": "Business loans",
        "SD": "Sight & savings deposits", "SF": "Term deposits", "K": "Borrowings"}


def page_layer2():
    st.title("2 · Balance sheet")
    s = settings(RUN)
    st.caption(f"{s.get('bank_name', 'The bank')}'s published balance sheet, mapped onto the "
               "model's seven positions and valued on the starting yield curve.")
    need("2", "Layer 2")
    r = jload(rp(RUN, "layer2", "balance_sheet_report.json"))
    v, k, t = r["values_mCHF"], r["constraints_t0"], r["targets"]
    c = st.columns(5)
    c[0].metric("Starting equity (model)", f"{v['E']:,.0f} m",
                help=f"Book equity in the annual report: {r['book_equity_mCHF']:,.0f} m")
    c[1].metric("Capital E/RWA", f"{k['E_over_RWA']:.3f}", help=f"Bank reports {t['E_over_RWA']:.3f}; limit 0.17")
    c[2].metric("Liquidity LCR", f"{k['LCR']:.2f}", help=f"Bank reports {t['LCR']:.2f}; limit 1.05")
    c[3].metric("Rate sensitivity (+1%)", f"{r['IRS_100bp']:+.3f}", help="Change in equity / equity; limit ±0.085")
    d150 = r["dE_mCHF"]["150"]
    c[4].metric("Equity if rates +1.5%", f"{d150:+,.0f} m",
                help=f"Bank's own figure: {r['dEVE_tkb_mCHF']['150']:+,.0f} m")

    l, rr = st.columns([1.1, 1])
    with l:
        fig = go.Figure()
        palette = {"C": "#90CAF9", "B": "#42A5F5", "RP": "#1565C0", "RE": "#0D47A1",
                   "SD": "#FFCC80", "SF": "#FB8C00", "K": "#E65100"}
        for sl in ("C", "B", "RP", "RE"):
            fig.add_trace(go.Bar(x=["Assets"], y=[v[sl]], name=SLOT[sl], marker_color=palette[sl]))
        for sl in ("SD", "SF", "K"):
            fig.add_trace(go.Bar(x=["Liabilities + equity"], y=[v[sl]], name=SLOT[sl], marker_color=palette[sl]))
        fig.add_trace(go.Bar(x=["Liabilities + equity"], y=[v["E"]], name="Equity", marker_color="#2E7D32"))
        fig.update_layout(barmode="stack")
        st.markdown("**The balance sheet, model values (CHF millions)**")
        show(fig_base(fig, height=420))
    with rr:
        rows = [{"Position": SLOT[sl], "Book value": round(r["book_mCHF"][sl]), "Model value": round(v[sl]),
                 "Model − book": v[sl] - r["book_mCHF"][sl], "Share of assets": v[sl] / v["A"]}
                for sl in SLOT]
        st.markdown("**Book against model value (CHF millions)**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "Book value": st.column_config.NumberColumn(format="localized"),
            "Model value": st.column_config.NumberColumn(format="localized"),
            "Model − book": st.column_config.NumberColumn(format="%+.0f"),
            "Share of assets": st.column_config.NumberColumn(format="percent")})
        st.caption("Model value discounts each position's cash flows on the starting curve; "
                   "equity is what is left.")

    l, rr = st.columns([1.1, 1])
    with l:
        br = r["equity_bridge_mCHF"]
        lab = {"book equity": "Book equity", "- excluded assets": "Assets left out",
               "+ excluded liabilities": "Liabilities left out", "= model E0": "Model equity"}
        names, vals, meas = [], [], []
        for kk, vv in br.items():
            if kk.startswith(("+ (V - book)", "- (V - book)")):
                names.append(f"Revalue {SLOT[kk.split()[-1]].lower()}")
            else:
                names.append(lab.get(kk, kk))
            vals.append(vv)
            meas.append("absolute" if kk in ("book equity",) else ("total" if kk == "= model E0" else "relative"))
        fig = go.Figure(go.Waterfall(x=names, y=vals, measure=meas,
                                     increasing=dict(marker_color="#2E7D32"),
                                     decreasing=dict(marker_color="#C62828"),
                                     totals=dict(marker_color="#1565C0")))
        st.markdown("**From book equity to model equity (CHF millions)**")
        show(fig_base(fig, height=400, legend=False))
    with rr:
        st.markdown("**Checks against the bank's own disclosures**")
        chk = pd.DataFrame([
            {"Measure": "Capital E/RWA", "Limit": "≥ 0.17", "Model": k["E_over_RWA"], "Bank": t["E_over_RWA"]},
            {"Measure": "Liquidity LCR", "Limit": "≥ 1.05", "Model": k["LCR"], "Bank": t["LCR"]},
            {"Measure": "Stable funding NSFR", "Limit": "≥ 1.05", "Model": k["NSFR"], "Bank": t["NSFR"]},
            {"Measure": "Reserve ratio CMR", "Limit": "≥ 1.0", "Model": k["CMR"], "Bank": None},
            {"Measure": "Equity, rates +1.5% (m)", "Limit": "", "Model": r["dE_mCHF"]["150"],
             "Bank": r["dEVE_tkb_mCHF"]["150"]},
            {"Measure": "Equity, rates −1.5% (m)", "Limit": "", "Model": r["dE_mCHF"]["-150"],
             "Bank": r["dEVE_tkb_mCHF"]["-150"]}])
        st.dataframe(chk, hide_index=True, column_config={
            "Model": st.column_config.NumberColumn(format="%.3f"),
            "Bank": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Ratios use the papers' simplified weights, so they are not expected to "
                   "match the bank's exactly. The rate-shock line matters most for ALM.")
    items = [("Layer 2 workbook", rp(RUN, "layer2", "layer2_balance_sheet.xlsx"), XLSX),
             ("balance_sheet_report.json", rp(RUN, "layer2", "balance_sheet_report.json"), "application/json")]
    if s.get("bank_workbook"):
        items.append(("This run's bank data", rp(RUN, s["bank_workbook"]), XLSX))
    downloads(items)


# ---------------------------------------------------------------- layer 3
def page_layer3():
    st.title("3 · Simulation")
    st.caption("The bank walked forward 60 months on every scenario: loans mature and are "
               "re-granted, deposits roll over, interest and costs are paid, a dividend each "
               "year. Two simple strategies show what the starting balance sheet does.")
    need("3", "Layer 3")
    no, ro = jload(rp(RUN, "layer3", "sim_none.json")), jload(rp(RUN, "layer3", "sim_rollover.json"))
    st.markdown("**No decisions** repays maturing debt out of cash. **Roll-over** refinances it.")
    c = st.columns(4)
    c[0].metric("Equity after 5y, no decisions", f"{no['ET_over_E0']:.3f}×",
                help=f"median {no['ET']['median']:,.0f} m; 5–95%: {no['ET']['q05']:,.0f}–{no['ET']['q95']:,.0f}")
    c[1].metric("Equity after 5y, roll-over", f"{ro['ET_over_E0']:.3f}×",
                delta=f"{ro['ET_over_E0'] - no['ET_over_E0']:+.3f}",
                help=f"median {ro['ET']['median']:,.0f} m; 5–95%: {ro['ET']['q05']:,.0f}–{ro['ET']['q95']:,.0f}")
    c[2].metric("LCR at 5y: nothing → roll-over", f"{no['median']['LCR'][-1]:.2f} → {ro['median']['LCR'][-1]:.2f}")
    c[3].metric("Rate sensitivity at 5y", f"{no['median']['IRS'][-1]:+.3f} → {ro['median']['IRS'][-1]:+.3f}")

    t = np.arange(len(no["median"]["E"])) / 12

    def two(key, title, ytitle, limit=None, fmt=None):
        fig = go.Figure()
        for lab, src, colr in (("No decisions", no, COL["nothing"]), ("Roll-over", ro, COL["rollover"])):
            fig.add_trace(go.Scatter(x=t, y=src["median"][key], name=lab, line=dict(color=colr, width=3)))
        if limit:
            lim, kind = limit
            fig.add_hline(y=lim, line_dash="dash", line_color="#C62828", annotation_text="limit")
            if kind == "abs":
                fig.add_hline(y=-lim, line_dash="dash", line_color="#C62828")
        st.markdown(f"**{title}**")
        show(fig_base(fig, y=ytitle, x="years", height=320))

    l, r = st.columns(2)
    with l:
        two("E", "Equity, median (CHF millions)", "CHF m")
        two("LCR", "Liquidity coverage LCR", "", LIMITS["LCR"])
        two("E_over_RWA", "Capital E/RWA", "", LIMITS["E_over_RWA"])
    with r:
        two("C", "Cash, median (CHF millions)", "CHF m")
        two("IRS", "Rate sensitivity IRS", "", LIMITS["IRS"])
        two("K", "Borrowings, median (CHF millions)", "CHF m")

    st.markdown("**Scenarios breaking each limit at least once**")
    keys = ["LCR", "IRS", "CMR", "E_over_RWA", "NSFR", "EYR"]
    fig = go.Figure()
    for lab, src, colr in (("No decisions", no, COL["nothing"]), ("Roll-over", ro, COL["rollover"])):
        fig.add_trace(go.Bar(x=[LIMIT_NAME[k] for k in keys], y=[src["violation_share"][k] for k in keys],
                             name=lab, marker_color=colr, text=[f"{src['violation_share'][k]:.0%}" for k in keys],
                             textposition="outside"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
    show(fig_base(fig, height=340))
    with st.expander("Year-end medians"):
        yrs = [0, 12, 24, 36, 48, 60]
        rows = []
        for key, lab in (("C", "Cash"), ("B", "Investments"), ("RP", "Mortgages"), ("RE", "Business loans"),
                         ("SD", "Sight & savings deposits"), ("SF", "Term deposits"), ("K", "Borrowings"),
                         ("E", "Equity")):
            for name, src in (("No decisions", no), ("Roll-over", ro)):
                rows.append({"Position": lab, "Strategy": name,
                             **{f"{y // 12}y": round(src["median"][key][y]) for y in yrs}})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={f"{y}y": st.column_config.NumberColumn(format="localized") for y in range(6)})
    downloads([("Layer 3 workbook", rp(RUN, "layer3", "layer3_results.xlsx"), XLSX),
               ("sim_none.json", rp(RUN, "layer3", "sim_none.json"), "application/json"),
               ("sim_rollover.json", rp(RUN, "layer3", "sim_rollover.json"), "application/json")])


# ---------------------------------------------------------------- layer 4
def page_layer4():
    st.title("4 · Objective")
    st.caption("What turns a path of the bank into one number to minimise: a shortfall against "
               "a return target, plus a penalty for every limit broken in every month.")
    need("4", "Layer 4")
    no, ro = jload(rp(RUN, "layer3", "sim_none.json")), jload(rp(RUN, "layer3", "sim_rollover.json"))
    l, r = st.columns([1, 1])
    with l:
        st.markdown("**The loss ([E] 2.5)**")
        st.latex(r"P_{i,t} = (1+v_{i,t})^2 - 1 \qquad p = \sum_i \sigma_i \sum_{t=0}^{H-1} P_{i,t}")
        st.latex(r"\ell_p = (1+p)^2 - 1 \qquad \ell_t = \Big[10\,\big(\tfrac{E_T}{E_0} - (1+\mu)^T\big)^-\Big]^2")
        st.latex(r"\ell = \ell_t + \lambda\,\ell_p")
        st.caption("v = how far a limit is broken. Squared twice: a big breach costs more than "
                   "two small ones. Only shortfalls against the target count; overshooting is free. "
                   "The factor 10 is recovered from the paper's own Table 3 (problem B9).")
        st.dataframe(pd.DataFrame({
            "Limit": [LIMIT_NAME[k] for k in ("LCR", "NSFR", "CMR", "E_over_RWA", "IRS", "EYR")],
            "Weight σ": [1.0, 0.2, 1.0, 2.5, 2.0, 0.002]}), hide_index=True)
    with r:
        lam = st.slider("Penalty weight λ", 0.05, 25.0, 3.5, 0.05,
                        help="3.5 is the paper's evaluation value; training draws λ from 0.05–25.")
        tot = {lab: src["loss"]["loss"]["target"] + lam * src["loss"]["loss"]["penalty"]
               for lab, src in (("No decisions", no), ("Roll-over", ro))}
        c = st.columns(2)
        c[0].metric("Total loss, no decisions", f"{tot['No decisions']:,.1f}")
        c[1].metric("Total loss, roll-over", f"{tot['Roll-over']:,.2f}")
        rows = []
        for lab, src in (("No decisions", no), ("Roll-over", ro)):
            L = src["loss"]["loss"]
            rows.append({"Strategy": lab, "Target loss": L["target"], "Penalty loss": L["penalty"],
                         f"Total at λ = {lam:g}": L["target"] + lam * L["penalty"],
                         "Utility loss (γ = 10)": L["utility"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            c_: st.column_config.NumberColumn(format="%.3f") for c_ in
            ("Target loss", "Penalty loss", f"Total at λ = {lam:g}", "Utility loss (γ = 10)")})
        st.markdown("**What the penalty is made of**")
        keys = ["LCR", "IRS", "EYR", "CMR", "E_over_RWA", "NSFR"]
        fig = go.Figure()
        for lab, src, colr in (("No decisions", no, COL["nothing"]), ("Roll-over", ro, COL["rollover"])):
            vals = [max(src["loss"]["penalty_share"][k], 1e-4) for k in keys]
            fig.add_trace(go.Bar(x=[LIMIT_NAME[k] for k in keys], y=vals, name=lab, marker_color=colr))
        fig.update_yaxes(type="log", title="mean contribution (log)")
        show(fig_base(fig, height=320))
    with st.expander("Four problems in the papers' objective, and what we did"):
        st.markdown(
            "- **E9 — which side of a limit is a breach.** Eq. (30) prints the two cases the wrong way "
            "round; the paper's own Table 4 settles it. We follow Table 4.\n"
            "- **B9 — the scale of the target loss.** Never stated; Table 3 implies equity in units of "
            "E₀/10. Without it λ = 3.5 would weigh penalties 100× too heavily.\n"
            "- **B10 — Table 3's target-loss row** is the symmetric loss, not the asymmetric one used "
            "for training. We train on the asymmetric one and report both.\n"
            "- **B11 — timing.** Limits are checked after each month's decisions and before the "
            "dividend, over months 0–59, as the paper's Algorithm 1 does.")


# ---------------------------------------------------------------- layer 5
def validity_banner(run, folder):
    st.success("The network and the benchmarks were trained on exactly this run's data.",
               icon=":material/verified:")


def hero_metrics(cmp):
    net = cmp["DeepALM"]
    bm = min(("BM_E", "BM_C", "BM_D"), key=lambda k: cmp[k]["total"])
    c = st.columns(4)
    c[0].metric("Total loss, network", f"{net['total']:,.2f}",
                delta=f"{net['total'] / cmp[bm]['total'] - 1:+.0%} vs best benchmark",
                delta_color="inverse")
    c[1].metric("Annual return on equity", f"{net['annual_return']:.2%}",
                delta=f"{(net['annual_return'] - cmp['nothing']['annual_return']) * 100:+.2f} pp vs doing nothing")
    c[2].metric("Equity after 5 years, mean", f"{net['ER_mean']:.3f}×",
                help=f"spread (sd) {net['ER_sd']:.3f}; worst-5% average {net['ER_ES0.95']:+.3f} from the mean")
    c[3].metric("Scenarios breaking rate-risk limit", f"{net['violation_share']['IRS']:.0%}",
                delta="paper's model: 0.06%", delta_color="off")


def loss_bar(cmp):
    fig = go.Figure(go.Bar(
        x=[NAME[k] for k in ORDER], y=[cmp[k]["total"] for k in ORDER],
        marker_color=[COL[k] for k in ORDER], text=[f"{cmp[k]['total']:,.1f}" for k in ORDER],
        textposition="outside"))
    fig.update_yaxes(type="log", title="total loss (log scale, lower is better)")
    return fig_base(fig, height=340, legend=False)


def page_layer5():
    st.title("5 · Decisions — main result")
    st.caption("Six ways of running the bank, scored on the same 1,600 scenarios none of them "
               "was trained on: doing nothing, a roll-over rule, the paper's three trained "
               "benchmarks, and the Deep ALM decision network.")
    if RUN is None:
        need("5")
    src = l5_sources(RUN)
    if not src:
        if job(RUN)["state"] == "running":
            st.info("This run's data is new, so the benchmarks and the network are being trained "
                    "on it now. The result appears here when training finishes; you can keep "
                    "presenting Layers 1–4 meanwhile.", icon=":material/model_training:")
        else:
            st.warning("No network has been trained on this run's data yet.")
        retrain_panel(RUN)
        st.stop()
    keys = list(src)
    key = keys[0]
    if len(keys) > 1:
        key = st.segmented_control("Network", keys, default="trained",
                                   format_func=lambda k: src[k][1]) or "trained"
    folder = src[key][0]
    validity_banner(RUN, folder)
    cmp = jload(rp(RUN, folder, "comparison.json"))["results"]
    rep = jload(rp(RUN, folder, "deep_alm_report.json"))
    hero_metrics(cmp)

    tabs = st.tabs(["Scoreboard", "What it does", "Limits", "Over time", "Training", "Train again"])
    with tabs[0]:
        l, r = st.columns([1, 1])
        with l:
            st.markdown("**Total loss by strategy**")
            show(loss_bar(cmp))
        with r:
            st.markdown("**Return against risk**")
            fig = go.Figure()
            for k in ORDER:
                fig.add_trace(go.Scatter(x=[-cmp[k]["ER_ES0.95"]], y=[cmp[k]["annual_return"] * 100],
                                         mode="markers", marker=dict(size=16, color=COL[k]), name=NAME[k]))
            fig.update_layout(hovermode="closest")
            show(fig_base(fig, height=340, legend=True,
                          x="risk: average shortfall in the worst 5% (equity ratio)",
                          y="annual return on equity, %"))
        rows = [("Total loss", "total", "{:,.2f}", "min"), ("Target loss", "target", "{:.3f}", "min"),
                ("Penalty loss", "penalty", "{:,.3f}", "min"), ("Utility loss (γ = 10)", "utility", "{:.3f}", "min"),
                ("Equity ratio, mean", "ER_mean", "{:.3f}", "max"), ("Equity ratio, sd", "ER_sd", "{:.3f}", "min"),
                ("Skew", "ER_skew", "{:.3f}", "max"), ("Worst-5% average (from mean)", "ER_ES0.95", "{:.3f}", "max"),
                ("Annual return", "annual_return", "{:.2f}%", "max"), ("No breach at all", "clean", "{:.1f}%", "max")]
        tab = pd.DataFrame({NAME[k]: [cmp[k][f] * (100 if f in ("annual_return", "clean") else 1)
                                      for _, f, _, _ in rows] for k in ORDER},
                           index=[r_[0] for r_ in rows])

        def best(row):
            how = dict((r_[0], r_[3]) for r_ in rows)[row.name]
            target = row.min() if how == "min" else row.max()
            return ["background-color: #C8E6C9; font-weight: 600" if v == target else "" for v in row]
        fmts = {r_[0]: r_[2] for r_ in rows}
        sty = tab.style.apply(best, axis=1).format(lambda v: v, na_rep="")
        for name, f in fmts.items():
            sty = sty.format(lambda v, f=f: f.format(v), subset=pd.IndexSlice[[name], :])
        st.markdown("**Scoreboard** (green = best in the row)")
        st.dataframe(sty)
        bm = min(("BM_E", "BM_C", "BM_D"), key=lambda k: cmp[k]["total"])
        gain = 1 - cmp["DeepALM"]["total"] / cmp[bm]["total"]
        st.info(f"The network's loss is **{gain:.0%} {'below' if gain > 0 else 'above'}** the best "
                f"benchmark ({NAME[bm]}). The paper's ranking — network first, BM_E last — "
                f"{'holds' if cmp['DeepALM']['total'] < min(cmp[k]['total'] for k in ('BM_E', 'BM_C', 'BM_D')) and cmp['BM_E']['total'] >= max(cmp[k]['total'] for k in ('BM_C', 'BM_D')) else 'does not fully hold'} here.")

    with tabs[1]:
        dm = cmp["DeepALM"]["median"]
        bench_m0 = max(cmp[k]["median"]["aK"][0] for k in ("BM_E", "BM_C", "BM_D"))
        c = st.columns(3)
        c[0].metric("Borrowed in month 0", f"{dm['aK'][0]:,.0f} m", delta=f"benchmarks ≤ {bench_m0:,.0f} m", delta_color="off")
        c[1].metric("Invested in bonds over 5 years", f"{float(np.sum(dm['aB'][:H])):,.0f} m")
        c[2].metric("Rate sensitivity: start → year 5", f"{dm['IRS'][0]:+.3f} → {dm['IRS'][H]:+.3f}")
        l, r = st.columns(2)
        with l:
            fig = go.Figure()
            for k in ("BM_D", "DeepALM"):
                fig.add_trace(go.Bar(x=np.arange(H), y=cmp[k]["median"]["aK"][:H], name=NAME[k], marker_color=COL[k]))
            st.markdown("**Borrowing per month, median (CHF millions)**")
            show(fig_base(fig, height=320, x="month", y="CHF m"))
        with r:
            fig = go.Figure()
            t = np.arange(H + 1) / 12
            for k in ("nothing", "rollover", "BM_D", "DeepALM"):
                fig.add_trace(go.Scatter(x=t, y=cmp[k]["median"]["IRS"], name=NAME[k],
                                         line=dict(color=COL[k], width=3 if k == "DeepALM" else 2)))
            fig.add_hrect(y0=-0.085, y1=0.085, fillcolor="#E8F5E9", line_width=0, layer="below")
            st.markdown("**Rate sensitivity over time (green band = allowed)**")
            show(fig_base(fig, height=320, x="years"))
        st.markdown("**The paper describes three habits of its trained model. Ours:**")
        st.dataframe(pd.DataFrame([
            {"Habit": "Borrow heavily in month 0",
             "Paper ([E] 4.1)": "“raise funds of roughly mCHF 1,000 in the first model period”",
             "This run": f"{dm['aK'][0]:,.0f} m in month 0; benchmarks at most {bench_m0:,.0f} m"},
            {"Habit": "Hold it as cash, invest little",
             "Paper ([E] 4.1)": "“not making any investments builds up a cash position”",
             "This run": f"cash {cmp['nothing']['median']['C'][0]:,.0f} → {dm['C'][0]:,.0f} m on day 0; "
                         f"{float(np.sum(dm['aB'][:H])):,.0f} m invested in 5 years"},
            {"Habit": "Rate exposure positive early, negative late",
             "Paper ([E] 4.2)": "“the IRS decreases and becomes negative during the fifth and final year”",
             "This run": f"{dm['IRS'][0]:+.3f} at the start, {dm['IRS'][H]:+.3f} at year 5"}]),
            hide_index=True)

    with tabs[2]:
        keys = ["LCR", "NSFR", "CMR", "E_over_RWA", "IRS", "EYR"]
        z = np.array([[cmp[k]["violation_share"][kk] for kk in keys] for k in ORDER])
        fig = go.Figure(go.Heatmap(z=z, x=[LIMIT_NAME[k] for k in keys], y=[NAME[k] for k in ORDER],
                                   colorscale=[[0, "#E8F5E9"], [0.2, "#FFF3E0"], [1, "#C62828"]],
                                   zmin=0, zmax=1, text=[[f"{v:.0%}" for v in row] for row in z],
                                   texttemplate="%{text}", showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.markdown("**Share of scenarios breaking each limit at least once**")
        show(fig_base(fig, height=360, legend=False))
        irs = cmp["DeepALM"]["violation_share"]["IRS"]
        if irs > 0.05:
            st.warning(
                f"The network crosses the rate-risk limit in {irs:.0%} of scenarios (the paper's model: "
                "0.06%). Under the paper's loss a small breach is almost free — 0.002 over the limit "
                "adds about 0.008 to the penalty — so the optimiser parks on the line where that pays. "
                "A higher IRS penalty weight, or a target inside the limit, would remove it.")
        st.caption("The minimum-return rule (EYR) is broken often by every strategy, doing nothing "
                   "included: it reflects how much the bank earns more than how it is funded.")

    with tabs[3]:
        t = np.arange(H + 1) / 12
        sel = st.pills("Strategies", ORDER, selection_mode="multi",
                       default=["nothing", "rollover", "BM_D", "DeepALM"], format_func=lambda k: NAME[k])
        l, r = st.columns(2)
        for col, key_, title, lim in ((l, "E", "Equity, median (CHF millions)", None),
                                      (r, "LCR", "Liquidity coverage LCR", LIMITS["LCR"]),
                                      (l, "C", "Cash, median (CHF millions)", None),
                                      (r, "E_over_RWA", "Capital E/RWA", LIMITS["E_over_RWA"])):
            with col:
                fig = go.Figure()
                for k in sel or []:
                    fig.add_trace(go.Scatter(x=t, y=cmp[k]["median"][key_], name=NAME[k],
                                             line=dict(color=COL[k], width=3 if k == "DeepALM" else 2)))
                if lim:
                    fig.add_hline(y=lim[0], line_dash="dash", line_color="#C62828")
                st.markdown(f"**{title}**")
                show(fig_base(fig, height=300, x="years"))

    with tabs[4]:
        if rep and rep.get("training"):
            tr = rep["training"]
            fig = go.Figure()
            for k in ("BM_E", "BM_C", "BM_D", "DeepALM"):
                if k in tr:
                    fig.add_trace(go.Scatter(x=[e["epoch"] + 1 for e in tr[k]],
                                             y=[min(e["val_total"], 60) for e in tr[k]], name=NAME[k],
                                             line=dict(color=COL[k], width=3 if k == "DeepALM" else 2)))
            st.markdown("**Validation loss by epoch** (capped at 60)")
            show(fig_base(fig, height=340, x="epoch", y="validation loss"))
            best_ep = min(range(len(tr["DeepALM"])), key=lambda e: tr["DeepALM"][e]["val_total"])
            s_ = rep["setup"]
            st.caption(f"Trained on {s_['train_scenarios']:,} scenarios, scored on {s_['val_scenarios']:,}. "
                       f"Network kept from epoch {best_ep + 1} of {len(tr['DeepALM'])} (lowest validation "
                       f"loss, as the paper's early stopping). Hidden layers {' / '.join(map(str, s_['widths']))}.")
        else:
            st.info("No training history for this network.")

    with tabs[5]:
        retrain_panel(RUN)

    downloads([("Layer 5 workbook", rp(RUN, folder, "layer5_results.xlsx"), XLSX),
               ("comparison.json", rp(RUN, folder, "comparison.json"), "application/json")])


def retrain_panel(run):
    st.markdown("**Train the benchmarks and the network on this run's data**")
    j = job(run)
    if j["state"] == "running":
        live_job_panel(run)
        if st.button("Stop", icon=":material/stop:"):
            stop_job(run)
            st.rerun()
        return
    bm, nn, ns, w = 25, 60, 8000, "512,512,256,128"
    st.caption("Same settings as your command-line training: 8,000 scenarios, 3 benchmarks × 25 "
               "epochs, then the network for 60 epochs. About 15–30 minutes on a laptop. "
               "New runs do this by themselves; this button trains again from scratch.")
    if not exists(run, "layer1", "fit_sqrt.npz"):
        st.info("Layer 1 must finish first.")
        return
    label = "Train again" if l5_sources(run) else "Train now"
    if st.button(label, type="primary", icon=":material/model_training:"):
        start_job(run, "layer5-train", ["--bm-epochs", bm, "--nn-epochs", nn,
                                        "--train-scenarios", ns, "--widths", w])
        st.rerun()


# ---------------------------------------------------------------- data & runs
def page_data():
    st.title("Data & runs")
    st.caption("A run is one bank on one yield-curve date. Upload new data to create a run; "
               "every layer is then computed for it, in the background.")
    with st.container(border=True):
        st.markdown("#### New run")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Layer 1 · Yield curves**")
            csv = st.file_uploader("SNB yield-curve file (CSV, cube 'rendopar')", type=["csv"],
                                   help="data.snb.ch → Interest rates → Spot interest rates on "
                                        "Confederation bond issues. Leave empty to reuse a curve "
                                        "already fitted in another run.")
            reuse = None
            if not csv:
                with_fit = [r for r in runs if exists(r, "layer1", "fit_sqrt.npz")]
                if with_fit:
                    reuse = st.selectbox("…or reuse the fitted curve of", with_fit,
                                         format_func=lambda r: settings(r).get("title", r))
            date = st.date_input("Curve date (start of the simulation)", dt.date(2022, 7, 15),
                                 disabled=not csv)
            model = st.radio("Model", ["paper", "term_premium"], horizontal=True, disabled=not csv,
                             format_func=lambda m: "Paper (P = Q)" if m == "paper" else "Version B (term premium)")
        with c2:
            st.markdown("**Layer 2 · The bank**")
            sys.path.insert(0, HERE)
            import bank_data as bd
            tmpl = os.path.join(RUNS, "bank_template.xlsx")
            os.makedirs(RUNS, exist_ok=True)
            if not os.path.isfile(tmpl):
                bd.write_template(tmpl)
            with open(tmpl, "rb") as f:
                st.download_button("Download the template (pre-filled with TKB 2022)", f.read(),
                                   file_name="bank_template.xlsx", mime=XLSX, icon=":material/download:")
            up = st.file_uploader("Bank workbook (XLSX)", type=["xlsx"],
                                  help="Leave empty to use TKB 2022, built into balance_sheet.py.")
            bank = None
            if up:
                tmp = os.path.join(RUNS, "_upload_check.xlsx")
                with open(tmp, "wb") as f:
                    f.write(up.getvalue())
                try:
                    bank = bd.read_workbook(tmp)
                    a = sum(v for s_, _, v in bank.balance_sheet.values() if s_ == "A")
                    st.success(f"**{bank.name}**, {bank.balance_sheet_date:%d.%m.%Y}: total assets "
                               f"{a / 1e6:,.1f} bn CHF. All checks passed.", icon=":material/check:")
                    for w_ in bank.warnings:
                        st.warning(w_)
                except bd.TemplateError as e:
                    st.error(str(e), icon=":material/error:")
                    bank = False
        if "new_title" not in st.session_state:
            st.session_state["new_title"] = f"Run {dt.datetime.now():%d.%m %H:%M}"
        title = st.text_input("Name of the run", key="new_title")
        ok = (csv or reuse) and bank is not False
        if st.button("Create run and compute all layers", type="primary", disabled=not ok,
                     icon=":material/play_arrow:"):
            create_run(title, csv, reuse, date, model, up if bank else None, bank)
            st.rerun()

    if RUN:
        st.markdown(f"#### This run: {settings(RUN).get('title', RUN)}")
        s = settings(RUN)
        c = st.columns(4)
        c[0].markdown(f"**Bank**  \n{s.get('bank_name')}")
        c[1].markdown(f"**Curve date**  \n{s.get('valuation_date')}")
        c[2].markdown(f"**Curve model**  \n{s.get('model')}")
        c[3].markdown(f"**Data**  \n{'uploaded workbook' if s.get('bank_workbook') else 'built-in TKB 2022'}")
        j = job(RUN)
        if j["state"] == "running":
            live_job_panel(RUN)
            if st.button("Stop the job", icon=":material/stop:"):
                stop_job(RUN)
                st.rerun()
        else:
            if j["state"] == "failed":
                st.error(f"The last job ({j.get('steps')}) failed. Its log:", icon=":material/error:")
                st.code(log_tail(RUN, 30), language=None)
            elif j["state"] == "done":
                st.success(f"Last job finished: {j.get('steps')} (started {j.get('started')}).")
            cols = st.columns(5)
            if cols[0].button("Recompute Layer 1"):
                start_job(RUN, "layer1"); st.rerun()
            if cols[1].button("Recompute Layer 2"):
                start_job(RUN, "layer2"); st.rerun()
            if cols[2].button("Recompute Layers 3–4"):
                start_job(RUN, "layer3"); st.rerun()
            if cols[3].button("Recompute Layer 5"):
                start_job(RUN, "layer5"); st.rerun()
            if cols[4].button("Recompute everything", type="primary"):
                start_job(RUN, "all"); st.rerun()
            with st.expander("Last log"):
                st.code(log_tail(RUN, 60) or "(empty)", language=None)
        if RUN != DEFAULT_RUN:
            with st.expander("Delete this run"):
                if st.button("Delete permanently", type="secondary"):
                    stop_job(RUN)
                    shutil.rmtree(rp(RUN), ignore_errors=True)
                    st.session_state.pop("run", None)
                    st.rerun()
    with st.expander("Where the app finds things"):
        src = source_dirs()
        st.markdown(f"- Runs: `{RUNS}`\n- First run seeded from: `{src['fit']}` (curve) and "
                    f"`{src['network']}` (network)\n- Code: `{pythonpath()}`")


def create_run(title, csv, reuse, date, model, up, bank):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "run"
    name, i = slug, 2
    while os.path.exists(rp(name)):
        name, i = f"{slug}-{i}", i + 1
    os.makedirs(rp(name, "inputs"))
    os.makedirs(rp(name, "layer1"))
    s = {"title": title, "model": model, "scenarios": 4000, "seed": 20221230,
         "created": dt.datetime.now().isoformat(timespec="seconds"),
         "bank_name": bank.name if bank else "Thurgauer Kantonalbank",
         "snb_csv": None, "bank_workbook": None}
    if csv:
        with open(rp(name, "inputs", "snb.csv"), "wb") as f:
            f.write(csv.getvalue())
        s["snb_csv"] = "inputs/snb.csv"
        s["valuation_date"] = date.isoformat()
    else:
        for f in ("fit_sqrt.npz", "fit_eig.npz", "curve_features.npz", "report.json", "layer1_results.xlsx"):
            if exists(reuse, "layer1", f):
                shutil.copy(rp(reuse, "layer1", f), rp(name, "layer1"))
        s["valuation_date"] = settings(reuse).get("valuation_date")
        s["model"] = settings(reuse).get("model", "paper")
    if up:
        with open(rp(name, "inputs", "bank.xlsx"), "wb") as f:
            f.write(up.getvalue())
        s["bank_workbook"] = "inputs/bank.xlsx"
    save_settings(name, s)
    steps = "layer1,layer2,layer3,layer5"
    st.session_state["run"] = name
    start_job(name, steps)


# ================================================================ navigation
pages = {
    "Presentation": [
        st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
        st.Page(page_layer1, title="1 · Yield curves", icon=":material/show_chart:", url_path="yield-curves"),
        st.Page(page_layer2, title="2 · Balance sheet", icon=":material/account_balance:", url_path="balance-sheet"),
        st.Page(page_layer3, title="3 · Simulation", icon=":material/timeline:", url_path="simulation"),
        st.Page(page_layer4, title="4 · Objective", icon=":material/functions:", url_path="objective"),
        st.Page(page_layer5, title="5 · Decisions", icon=":material/psychology:", url_path="decisions"),
    ],
    "Setup": [st.Page(page_data, title="Data & runs", icon=":material/upload_file:", url_path="data")],
}
st.navigation(pages).run()
