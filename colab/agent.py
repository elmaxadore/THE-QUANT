#!/usr/bin/env python3
"""THE-QUANT remote training agent (pull-based — no inbound tunnel needed).

Runs on ANY free compute: Google Colab, Kaggle, GitHub Codespaces, or a free
VM. Instead of hosting an HTTP server, it polls the `colab-jobs` git branch
of the repo for queued jobs, executes them, and pushes every produced
artifact to the `colab-artifacts` branch. Both branches live on GitHub, so:

  * runtimes can die at any moment — nothing is ever lost;
  * multiple runtimes can share the queue (claim/release; stale claims are
    auto-reclaimed after 2h);
  * you collect everything locally with:  python3 colab/colab_cli.py pull

Requirements: GITHUB_TOKEN env var (Contents read/write). GitHub Codespaces
provides it automatically; on Colab/Kaggle paste a fine-grained PAT.

Usage (inside Colab/Kaggle/Codespace):
  python3 colab/agent.py --interval 30 --bootstrap "pip install -q torch onnx xgboost pandas numpy requests"
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import persist  # noqa: E402

REPO_URL = "https://github.com/elmaxadore/THE-QUANT.git"
JOBS_BRANCH = "colab-jobs"
DEFAULT_CLONE = "/content/quant_colab/repo"


def find_or_clone_repo():
    for cand in [os.environ.get("COLAB_REPO_DIR"), DEFAULT_CLONE,
                 os.path.expanduser("~/THE-QUANT"), os.getcwd()]:
        if cand and os.path.isfile(os.path.join(cand, ".git")):
            subprocess.run(["git", "-C", cand, "pull", "--ff-only",
                            "--quiet"], capture_output=True)
            return cand
    subprocess.run(["git", "clone", REPO_URL, DEFAULT_CLONE], check=True)
    return DEFAULT_CLONE


def list_jobs(repo):
    """Fetch the jobs branch and return its top-level job dir names."""
    subprocess.run(["git", "-C", repo, "fetch", "origin", JOBS_BRANCH],
                   capture_output=True)
    r = subprocess.run(["git", "-C", repo, "ls-tree", "--name-only",
                        "FETCH_HEAD"], capture_output=True, text=True)
    return [n for n in r.stdout.split() if n]


def read_spec(repo, job_name):
    r = subprocess.run(["git", "-C", repo, "show",
                        f"FETCH_HEAD:{job_name}/job.json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def materialize_job(repo, job_name, workdir):
    """Check out the job dir contents into workdir."""
    tar_path = os.path.join(workdir, "_archive.tar")
    with open(tar_path, "wb") as fh:
        subprocess.run(["git", "-C", repo, "archive",
                        f"FETCH_HEAD:{job_name}"], check=True, stdout=fh)
    subprocess.run(["tar", "-xf", tar_path, "-C", workdir], check=True)
    os.remove(tar_path)


def run_job(repo, job_name, spec, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    env = os.environ.copy()
    env["TRAIN_PARAMS"] = json.dumps(spec.get("params", {}))
    env["COLAB_ARTIFACTS_DIR"] = artifacts_dir
    env["COLAB_REPO_DIR"] = repo
    env["PYTHONUNBUFFERED"] = "1"

    script = spec.get("script", "")
    if script and not os.path.isabs(script):
        script = os.path.join(repo, script)
    cmd = [sys.executable, script] if script else \
        [sys.executable, os.path.join(repo, "python", "research",
                                      "train_pipeline.py")]
    print(f"[agent] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env, cwd=repo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true",
                    help="process at most one job then exit")
    ap.add_argument("--bootstrap", default=None,
                    help="shell command run once at startup (pip install etc)")
    ap.add_argument("--repo", default=None, help="existing repo checkout")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[agent] FATAL: GITHUB_TOKEN not set. On Codespaces it is "
              "injected automatically; on Colab/Kaggle paste a PAT.")
        sys.exit(2)

    if args.bootstrap:
        print(f"[agent] bootstrap: {args.bootstrap}", flush=True)
        subprocess.call(args.bootstrap, shell=True)

    repo = args.repo or find_or_clone_repo()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    print(f"[agent] repo={repo} worker={worker_id} "
          f"polling {JOBS_BRANCH} every {args.interval}s", flush=True)

    while True:
        try:
            for job_name in list_jobs(repo):
                spec = read_spec(repo, job_name)
                if spec is None or spec.get("status") == "completed":
                    continue
                if not persist.claim_job(repo, job_name, JOBS_BRANCH,
                                         token, worker_id):
                    continue  # another live worker owns it

                with tempfile.TemporaryDirectory(prefix="dq_job_") as work:
                    materialize_job(repo, job_name, work)
                    artifacts_dir = os.path.join(work, "artifacts")
                    rc = run_job(repo, job_name, spec, artifacts_dir)

                    files = [p for p in glob.glob(
                        os.path.join(artifacts_dir, "**", "*"),
                        recursive=True) if os.path.isfile(p)]
                    note = f"rc={rc}, {len(files)} artifact(s)"
                    try:
                        persist.push_artifacts(
                            repo, files, message=f"{job_name}: {note}",
                            subdir=job_name)
                    except Exception as exc:
                        note += f"; PUSH FAILED: {exc}"
                    persist.release_job(repo, job_name, JOBS_BRANCH, token,
                                        "completed" if rc == 0 else "failed",
                                        note)
                    print(f"[agent] {job_name} -> "
                          f"{'completed' if rc == 0 else 'failed'} "
                          f"({note})", flush=True)
                if args.once:
                    return
        except KeyboardInterrupt:
            print("[agent] stopped")
            return
        except Exception as exc:
            print(f"[agent] loop error: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
