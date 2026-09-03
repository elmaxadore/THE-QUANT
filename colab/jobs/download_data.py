#!/usr/bin/env python3
"""THE-QUANT Colab job: download REAL Dukascopy tick data — fast + loss-proof.

Optimized for free Colab runtimes:

  SPEED
    * symbol-months fetched in parallel (month_workers x hour_workers
      concurrent requests) — Colab's datacenter IP is not rate-limited.
    * tick->bar aggregation per month; final M5 CSVs merged once at the end.

  MAXIMUM PERSISTENCE (survives runtime death at ANY point)
    * every completed month is immediately committed to the
      colab-artifacts branch as histdata_parts/<symbol>/<YYYYMM>.csv
      (small: a month of M5 bars is a few hundred KB).
    * on start, all previously persisted parts are restored, so a restarted
      job only downloads the months that are actually missing.

Params via TRAIN_PARAMS (colab_cli.py --params-json):
  {"symbols": "eurusd,gbpusd,xauusd", "start": "2025-01-01",
   "end": "2026-09-03", "month_workers": 3, "hour_workers": 4}
  ("workers" from older job specs maps to hour_workers.)
"""
import concurrent.futures as cf
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import persist  # noqa: E402

REPO_URL = "https://github.com/elmaxadore/THE-QUANT.git"
CLONE_PATH = "/content/quant_colab/repo"


def find_repo():
    for cand in [os.environ.get("COLAB_REPO_DIR"), CLONE_PATH,
                 os.path.expanduser("~/THE-QUANT"), os.getcwd()]:
        if os.path.isfile(os.path.join(cand, "python", "data",
                                       "download_dukascopy.py")):
            # always run the LATEST committed code (self-updating runtime)
            subprocess.run(["git", "-C", cand, "pull", "--ff-only",
                            "--quiet"], capture_output=True)
            return cand
    print(f"[job] repo not found; cloning {REPO_URL} -> {CLONE_PATH}")
    subprocess.run(["git", "clone", REPO_URL, CLONE_PATH], check=True)
    return CLONE_PATH


def restore_parts(repo, parts_dir):
    """Restore previously persisted month parts from colab-artifacts."""
    subprocess.run(["git", "-C", repo, "fetch", "origin",
                    persist.ARTIFACTS_BRANCH], capture_output=True)
    r = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only",
                        "FETCH_HEAD"], capture_output=True, text=True)
    n = 0
    for rel in r.stdout.split():
        if rel.startswith("histdata_parts/") and rel.endswith(".csv"):
            dest = os.path.join(parts_dir, rel[len("histdata_parts/"):])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(subprocess.run(
                    ["git", "-C", repo, "show", f"FETCH_HEAD:{rel}"],
                    capture_output=True).stdout)
            n += 1
    print(f"[job] restored {n} month part(s) from "
          f"{persist.ARTIFACTS_BRANCH}", flush=True)
    return n


def month_range(start, end):
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def write_part(parts_dir, sym, year, month, bars):
    """Persist one month of M5 bars (epoch,open,high,low,close,volume)."""
    d = os.path.join(parts_dir, sym.lower())
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{year}{month:02d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for t, o, h, l, c, n in bars:
            w.writerow([int(t), f"{o:.6f}", f"{h:.6f}",
                        f"{l:.6f}", f"{c:.6f}", n])
    return path
