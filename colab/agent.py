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
import signal
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
PIDFILE = os.path.join(tempfile.gettempdir(), "the-quant-agent.pid")


def is_repo(path):
    """True if path is a usable git checkout (.git may be dir OR file)."""
    return bool(path) and os.path.exists(os.path.join(path, ".git"))


def repair_or_clone(dest, url=None):
    """Idempotently ensure `dest` is a fresh-enough clone of `url`.

    Re-running is ALWAYS safe:
      * valid checkout  -> pull latest, keep going (resumes month caches!)
      * broken/partial checkout (interrupted clone, missing .git) -> moved
        aside to <dest>.broken-<ts> and re-cloned
      * missing -> cloned
    """
    url = url or REPO_URL
    if os.path.exists(dest):
        if is_repo(dest) and subprocess.run(
                ["git", "-C", dest, "status", "--porcelain"],
                capture_output=True).returncode == 0:
            subprocess.run(["git", "-C", dest, "pull", "--ff-only",
                            "--quiet"], capture_output=True)
            return dest
        backup = f"{dest}.broken-{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"[agent] '{dest}' is not a valid checkout — "
              f"moving it to {backup} and re-cloning", flush=True)
        os.rename(dest, backup)
    subprocess.run(["git", "clone", url, dest], check=True)
    return dest


def find_or_clone_repo():
    for cand in [os.environ.get("COLAB_REPO_DIR"), DEFAULT_CLONE,
                 os.path.expanduser("~/THE-QUANT"), os.getcwd()]:
        if cand and is_repo(cand) and os.path.isfile(
                os.path.join(cand, "colab", "agent.py")):
            subprocess.run(["git", "-C", cand, "pull", "--ff-only",
                            "--quiet"], capture_output=True)
            return cand
    return repair_or_clone(DEFAULT_CLONE)


def _pid_alive_and_agent(pid):
    """True if pid exists AND is actually an agent process (pid-safety)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"agent.py" in f.read()
    except OSError:
        return False


def take_over_previous_agent():
    """Kill any previous agent instance so re-running the notebook cell
    always yields exactly ONE healthy worker (idempotent start).
    Returns True if a previous instance was stopped."""
    stopped = False
    if os.path.exists(PIDFILE):
        try:
            with open(PIDFILE) as f:
                pid = int(f.read().strip())
            if _pid_alive_and_agent(pid):
                os.kill(pid, signal.SIGTERM)
                print(f"[agent] took over from previous agent (pid {pid})",
                      flush=True)
                stopped = True
                time.sleep(1)
        except (ValueError, OSError):
            pass  # stale pidfile
        finally:
            try:
                os.remove(PIDFILE)
            except OSError:
                pass
    return stopped


def stop_agent():
    print("[agent] stopped previous instance"
          if take_over_previous_agent() else "[agent] no agent running")


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
    ap.add_argument("--bootstrap-on-claim", action="store_true",
                    help="defer --bootstrap until a job is claimed (CI pollers)")
    ap.add_argument("--repo", default=None, help="existing repo checkout")
    ap.add_argument("--stop", action="store_true",
                    help="stop a previously started agent and exit")
    args = ap.parse_args()

    if args.stop:
        stop_agent()
        return

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[agent] FATAL: GITHUB_TOKEN not set. On Codespaces it is "
              "injected automatically; on Colab/Kaggle paste a PAT.")
        sys.exit(2)

    # Lazy bootstrap: with --bootstrap-on-claim the (expensive) dependency
    # install only runs after a job is actually claimed — pollers on CI
    # schedulers never pay for deps when the queue is empty.
    boot = {"done": not args.bootstrap_on_claim}

    def run_bootstrap():
        if not boot["done"] and args.bootstrap:
            boot["done"] = True
            print(f"[agent] bootstrap: {args.bootstrap}", flush=True)
            subprocess.call(args.bootstrap, shell=True)

    if args.bootstrap and boot["done"]:
        run_bootstrap()

    # re-running this script is ALWAYS safe: a previous instance is stopped,
    # completed jobs are skipped, and interrupted jobs are re-claimed.
    take_over_previous_agent()
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    repo = args.repo or find_or_clone_repo()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    print(f"[agent] repo={repo} worker={worker_id} "
          f"polling {JOBS_BRANCH} every {args.interval}s", flush=True)

    try:
        while True:
            try:
                for job_name in list_jobs(repo):
                    spec = read_spec(repo, job_name)
                    if spec is None or spec.get("status") == "completed":
                        continue
                    if not persist.claim_job(repo, job_name, JOBS_BRANCH,
                                             token, worker_id):
                        continue  # another live worker owns it

                    run_bootstrap()  # deps now — we actually have work
                    with tempfile.TemporaryDirectory(
                            prefix="dq_job_") as work:
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
                        persist.release_job(
                            repo, job_name, JOBS_BRANCH, token,
                            "completed" if rc == 0 else "failed", note)
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
    finally:
        try:
            os.remove(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
