#!/usr/bin/env python3
"""THE-QUANT Colab job: run the full training pipeline on the Colab runtime.

Expects the Dukascopy CSVs to already exist in the repo's
python/data/histdata/ (run jobs/download_data.py first, or the repo clone
already contains them).

  1. Locates the THE-QUANT repo inside the Colab workspace.
  2. Runs python/research/train_pipeline.py (features -> backtests ->
     Aegis grid search -> GBDT/MLP -> ONNX export).
  3. Copies reports/*.json + models/latest.onnx into $COLAB_ARTIFACTS_DIR.

Parameters via TRAIN_PARAMS (colab_cli.py --params-json):
  {"train_args": ["--symbols", "eurusd,gbpusd,xauusd"]}
"""
import glob
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
        if os.path.isfile(os.path.join(cand, "python", "research",
                                       "train_pipeline.py")):
            return cand
    print(f"[job] repo not found; cloning {REPO_URL} -> {CLONE_PATH}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_PATH],
                   check=True)
    return CLONE_PATH


def main():
    t0 = time.time()
    try:
        params = json.loads(os.environ.get("TRAIN_PARAMS", "{}"))
    except json.JSONDecodeError:
        params = {}
    extra = list(params.get("train_args", []))

    repo = find_repo()
    pipeline = os.path.join(repo, "python", "research", "train_pipeline.py")
    artifacts_dir = os.environ.get("COLAB_ARTIFACTS_DIR", os.getcwd())
    os.makedirs(artifacts_dir, exist_ok=True)

    # GPU smoke test (informational only)
    try:
        import torch  # noqa
        print(f"[job] torch {torch.__version__} "
              f"cuda={torch.cuda.is_available()}", flush=True)
    except ImportError:
        print("[job] torch not installed — MLP stage will be skipped",
              flush=True)

    cmd = [sys.executable, pipeline] + extra
    print(f"[job] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=repo)
    print(f"[job] pipeline exited with {rc}", flush=True)

    copied = 0
    for pattern in ["reports/*.json", "models/*.onnx"]:
        for path in glob.glob(os.path.join(repo, pattern)):
            shutil.copy2(path, os.path.join(artifacts_dir,
                                            os.path.basename(path)))
            copied += 1
            print(f"[job] artifact: {os.path.basename(path)}", flush=True)

    print(f"[job] done in {round(time.time() - t0, 1)}s — "
          f"{copied} artifact(s)", flush=True)
    sys.exit(0 if rc == 0 and copied else 1)


if __name__ == "__main__":
    main()
