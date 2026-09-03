#!/usr/bin/env python3
"""THE-QUANT Colab job: download REAL Dukascopy tick data on the Colab runtime.

Why: local home ISPs often get rate-limited (HTTP 503) by Dukascopy's WAF after
sustained parallel downloads. Colab's datacenter IP + bandwidth handles it fine.

The job:
  1. Locates (or clones) the THE-QUANT repo inside the Colab workspace.
  2. Runs python/data/download_dukascopy.py (real .bi5 tick feed -> M5 bars).
  3. Copies the resulting histdata/*.csv + a metrics.json into
     $COLAB_ARTIFACTS_DIR so `colab_cli.py collect` brings only the outputs
     back to the local machine.

Parameters come in via TRAIN_PARAMS (set by colab_cli.py --params-json):
  {"symbols": "eurusd,gbpusd,xauusd", "start": "2025-01-01",
   "end": "2026-09-03", "timeframes": "M5", "workers": 4}
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO_CANDIDATES = [
    "/content/quant_colab/repo",
    os.path.expanduser("~/THE-QUANT"),
    os.getcwd(),
]
REPO_URL = "https://github.com/elmaxadore/THE-QUANT.git"
CLONE_PATH = "/content/quant_colab/repo"


def find_repo():
    for cand in REPO_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "python", "data",
                                       "download_dukascopy.py")):
            return cand
    print(f"[job] repo not found locally; cloning {REPO_URL} -> {CLONE_PATH}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_PATH],
                   check=True)
    return CLONE_PATH


def main():
    t0 = time.time()
    try:
        params = json.loads(os.environ.get("TRAIN_PARAMS", "{}"))
    except json.JSONDecodeError:
        params = {}
    symbols = params.get("symbols", "eurusd,gbpusd,xauusd")
    start = params.get("start", "2025-01-01")
    end = params.get("end", time.strftime("%Y-%m-%d"))
    timeframes = params.get("timeframes", "M5")
    workers = str(params.get("workers", 4))

    repo = find_repo()
    downloader = os.path.join(repo, "python", "data", "download_dukascopy.py")
    histdata_dir = os.path.join(repo, "python", "data", "histdata")

    artifacts_dir = os.environ.get("COLAB_ARTIFACTS_DIR", os.getcwd())
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[job] symbols={symbols} start={start} end={end} "
          f"tf={timeframes} workers={workers}", flush=True)
    cmd = [sys.executable, downloader,
           "--symbols", symbols, "--start", start, "--end", end,
           "--timeframes", timeframes, "--workers", workers]
    print(f"[job] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[job] downloader exited with {rc}", flush=True)

    metrics = {"symbols": {}, "downloader_rc": rc,
               "runtime_seconds": 0.0}
    for path in sorted(os.listdir(histdata_dir)) \
            if os.path.isdir(histdata_dir) else []:
        if not path.endswith(".csv"):
            continue
        rows = sum(1 for _ in open(os.path.join(histdata_dir, path))) - 1
        metrics["symbols"][path] = {"bars": rows,
                                    "bytes": os.path.getsize(
                                        os.path.join(histdata_dir, path))}
        shutil.copy2(os.path.join(histdata_dir, path),
                     os.path.join(artifacts_dir, path))
        print(f"[job] artifact: {path} ({rows:,} bars)", flush=True)

    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[job] done in {metrics['runtime_seconds']}s — "
          f"{len(metrics['symbols'])} CSV artifact(s) in {artifacts_dir}",
          flush=True)
    sys.exit(0 if rc == 0 and metrics["symbols"] else 1)


if __name__ == "__main__":
    main()
