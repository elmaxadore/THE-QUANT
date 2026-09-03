#!/usr/bin/env python3
"""THE-QUANT — durable artifact persistence via a GitHub artifacts branch.

Remote runtimes (Colab / Kaggle / Codespaces) are EPHEMERAL. To guarantee
zero data loss:

  * After every job, all produced artifacts (reports/*.json, models/*.onnx,
    downloaded data CSVs, ...) are committed to the `colab-artifacts` branch
    of the repo. If the runtime dies a second later, the data is already on
    GitHub and can be restored locally with:
        python3 colab/colab_cli.py pull

  * Jobs themselves live on the `colab-jobs` branch (see colab/agent.py), so
    the queue also survives runtime restarts.

Auth: a GitHub token with repo (Contents: read/write) access must be provided
via the GITHUB_TOKEN env var. GitHub Codespaces injects this automatically;
in Colab/Kaggle the user pastes it into the notebook (it is never committed).
"""
import os
import subprocess
import sys
import tempfile
import time

ARTIFACTS_BRANCH = "colab-artifacts"
JOBS_BRANCH = "colab-jobs"


def _run(cmd, cwd=None, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> rc={r.returncode}\n"
                           f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r


def _push_url(repo_dir, token):
    origin = _run(["git", "-C", repo_dir, "remote", "get-url", "origin"]
                  ).stdout.strip()
    if "@" in origin and "://" in origin:  # strip existing credentials
        origin = origin.split("://")[0] + "://" + origin.split("@", 1)[1]
    return origin.replace("://", f"://x-access-token:{token}@", 1)


def push_artifacts(repo_dir, files, message=None, branch=ARTIFACTS_BRANCH,
                   token=None, subdir=None):
    """Commit `files` (absolute paths) to `branch`. Uses a throw-away clone
    so the running repo/worktree is never disturbed. Returns commit sha."""
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    if not token and "://" in _run(
            ["git", "-C", repo_dir, "remote", "get-url", "origin"]
    ).stdout:
        # remote URL needs credentials; local filesystem remotes don't
        raise RuntimeError("GITHUB_TOKEN not set — cannot persist artifacts")
    push_url = _push_url(repo_dir, token)

    job_label = subdir or f"job-{time.strftime('%Y%m%d-%H%M%S')}"
    with tempfile.TemporaryDirectory(prefix="dq_persist_") as tmp:
        # continue the existing artifacts branch; fall back to a fresh one
        r = _run(["git", "clone", "--depth", "1", "--branch", branch,
                  push_url, tmp], check=False)
        if r.returncode != 0:
            _run(["git", "init", "-b", branch, tmp])
            _run(["git", "-C", tmp, "remote", "add", "origin", push_url])

        copied = 0
        for f in files:
            if not os.path.isfile(f):
                continue
            rel = os.path.join(job_label, os.path.basename(f))
            dest = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(f, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
            copied += 1

        _run(["git", "-C", tmp, "add", "-A"])
        r = _run(["git", "-C", tmp, "commit",
                  "-m", message or f"artifacts: {job_label}"], check=False)
        if r.returncode != 0 and "nothing to commit" not in (r.stderr or ""):
            raise RuntimeError("artifact commit failed: " + r.stderr[-400:])
        if r.returncode != 0 or copied == 0:
            return None  # nothing new to push
        _run(["git", "-C", tmp, "push", "origin", branch])
        sha = _run(["git", "-C", tmp, "rev-parse", "HEAD"]).stdout.strip()
    print(f"[persist] pushed {copied} artifact(s) to {branch}@{sha[:8]}",
          flush=True)
    return sha


def _read_spec(tmp, job_name):
    import json as _json
    spec_path = os.path.join(tmp, job_name, "job.json")
    if not os.path.isfile(spec_path):
        return None, spec_path
    with open(spec_path) as f:
        return _json.load(f), spec_path


def claim_job(repo_dir, job_name, branch, token, worker_id):
    """Mark job.json status=claimed on the jobs branch (race-safe best
    effort). Returns True if this worker owns the job. Stale claims (>2h)
    are reclaimed so a crashed runtime never blocks the queue."""
    import json as _json
    push_url = _push_url(repo_dir, token)
    with tempfile.TemporaryDirectory(prefix="dq_claim_") as tmp:
        if _run(["git", "clone", "--depth", "1", "--branch", branch,
                 push_url, tmp], check=False).returncode != 0:
            return True  # branch gone -> nothing to race on
        spec, spec_path = _read_spec(tmp, job_name)
        if spec is None:
            return True
        status = spec.get("status", "pending")
        if status == "completed":
            return False
        if status == "claimed":
            if time.time() - spec.get("claimed_at", 0) < 2 * 3600 and \
                    spec.get("claimed_by") != worker_id:
                return False
        spec["status"] = "claimed"
        spec["claimed_by"] = worker_id
        spec["claimed_at"] = time.time()
        _json.dump(spec, open(spec_path, "w"), indent=2)
        _run(["git", "-C", tmp, "add", "-A"])
        _run(["git", "-C", tmp, "commit", "-m", f"claim {job_name}"],
             check=False)
        if _run(["git", "-C", tmp, "push", "origin", branch],
                check=False).returncode != 0:
            return False  # another worker won
    return True


def release_job(repo_dir, job_name, branch, token, status, note=""):
    """Set job.json status=completed|failed on the jobs branch."""
    import json as _json
    push_url = _push_url(repo_dir, token)
    with tempfile.TemporaryDirectory(prefix="dq_release_") as tmp:
        if _run(["git", "clone", "--depth", "1", "--branch", branch,
                 push_url, tmp], check=False).returncode != 0:
            return
        spec, spec_path = _read_spec(tmp, job_name)
        if spec is None:
            return
        spec["status"] = status
        if note:
            spec["note"] = str(note)[:500]
        _json.dump(spec, open(spec_path, "w"), indent=2)
        _run(["git", "-C", tmp, "add", "-A"])
        _run(["git", "-C", tmp, "commit", "-m", f"{status}: {job_name}"],
             check=False)
        _run(["git", "-C", tmp, "push", "origin", branch], check=False)


def enqueue_job(repo_dir, job_name, spec, branch="colab-jobs", token=None):
    """Queue a job: commit <job_name>/job.json to the jobs branch. Uses the
    user's local git credentials when GITHUB_TOKEN is not set (local CLI)."""
    import json as _json
    push_url = _push_url(repo_dir, token) if (token or
                                              os.environ.get("GITHUB_TOKEN")) \
        else _run(["git", "-C", repo_dir, "remote", "get-url", "origin"]
                  ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="dq_enqueue_") as tmp:
        if _run(["git", "clone", "--depth", "1", "--branch", branch,
                 push_url, tmp], check=False).returncode != 0:
            _run(["git", "init", "-b", branch, tmp])
            _run(["git", "-C", tmp, "remote", "add", "origin", push_url])
        os.makedirs(os.path.join(tmp, job_name), exist_ok=True)
        _json.dump(spec, open(os.path.join(tmp, job_name, "job.json"), "w"),
                   indent=2)
        _run(["git", "-C", tmp, "add", "-A"])
        _run(["git", "-C", tmp, "commit", "-m", f"enqueue {job_name}"],
             check=False)
        _run(["git", "-C", tmp, "push", "origin", branch])
    print(f"[enqueue] {job_name} queued on {branch}", flush=True)


def list_artifacts(repo_dir, branch=ARTIFACTS_BRANCH):
    """Return [(relpath, sha)] of all files on the artifacts branch."""
    r = _run(["git", "-C", repo_dir, "ls-tree", "-r", "--name-only",
              f"origin/{branch}"], check=False)
    return r.stdout.split() if r.returncode == 0 else []


if __name__ == "__main__":
    # manual usage:  python3 colab/persist.py <file> [<file> ...] -- <msg>
    args = sys.argv[1:]
    if "--" in args:
        i = args.index("--")
        files, msg = args[:i], " ".join(args[i + 1:])
    else:
        files, msg = args, None
    repo = os.environ.get("COLAB_REPO_DIR") or os.getcwd()
    print("pushed:", push_artifacts(repo, files, msg))
