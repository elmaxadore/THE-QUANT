#!/usr/bin/env python3
"""THE QUANT — mission-control coordinator (runs on the always-on server).

One authority for all cross-platform orchestration. Workers (Colab, Kaggle,
Codespaces, GitHub Actions) ONLY claim and execute jobs; this coordinator
decides what happens next:

  1. WATCH     fetch the `colab-jobs` branch and parse every job spec.
  2. CHAIN     download job completed and no training job queued
               -> enqueue the training job with the same symbols.
  3. RETRY     a failed job with < MAX_ATTEMPTS attempts, after a cool-down,
               -> re-queue it (transient 503s, dead runtimes).
  4. COLLECT   new commits on `colab-artifacts` -> `colab_cli.py pull`
               (reports -> reports/, models -> models/, data -> histdata),
               then commit reports/models to `main` and push. The server is
               the single place where artifacts are edited and committed.
  5. HEARTBEAT one log line per cycle; everything else is quiet.

Pure stdlib + git; memory footprint ~20 MB. Designed for 1 GB VPS.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "colab"))
import persist  # noqa: E402

JOBS_BRANCH = "colab-jobs"
ARTIFACTS_BRANCH = "colab-artifacts"
INTERVAL = int(os.environ.get("COORD_INTERVAL", "600"))     # seconds
MAX_ATTEMPTS = int(os.environ.get("COORD_MAX_ATTEMPTS", "3"))
RETRY_COOLDOWN = int(os.environ.get("COORD_RETRY_COOLDOWN", "3600"))
STATE_PATH = os.path.join(ROOT, "state", "coordinator.json")

DOWNLOAD_JOB = "colab/jobs/download_data.py"
TRAIN_JOB = "colab/jobs/train_data.py"


def log(msg):
    print(f"[coord] {time.strftime('%FT%TZ', time.gmtime())} {msg}", flush=True)


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"last_artifact_sha": "", "attempts": {}}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)


def git(*args, check=True):
    r = subprocess.run(["git", "-C", ROOT, *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr[-400:]}")
    return r


def fetch_branches():
    # explicit refspecs: shallow single-branch clones only track origin/main,
    # so a bare `git fetch origin <branch>` never creates origin/<branch>
    ok_jobs = git("fetch", "origin",
                  f"{JOBS_BRANCH}:{JOBS_BRANCH}", check=False).returncode == 0
    ok_art = git("fetch", "origin",
                 f"{ARTIFACTS_BRANCH}:{ARTIFACTS_BRANCH}",
                 check=False).returncode == 0
    return ok_jobs, ok_art


def read_jobs():
    """Parse every job spec from origin/colab-jobs."""
    r = git("ls-tree", "-r", "--name-only", JOBS_BRANCH, check=False)
    if r.returncode != 0:
        return {}
    jobs = {}
    for path in r.stdout.split():
        if not path.endswith("/job.json"):
            continue
        name = path.split("/")[0]
        try:
            spec = json.loads(git("show", f"{JOBS_BRANCH}:{path}").stdout)
            jobs[name] = spec
        except Exception as exc:
            log(f"WARNING: cannot parse {path}: {exc}")
    return jobs


def enqueue(name, spec):
    persist.enqueue_job(ROOT, name, spec, branch=JOBS_BRANCH,
                        token=os.environ.get("GITHUB_TOKEN"))


def has_job_with_script(jobs, script):
    return any(j.get("script") == script for j in jobs.values())


def chain(jobs, st):
    """Single-authority pipeline chaining + failure retries."""
    changed = False

    # 2. CHAIN: data done -> train next (same symbols)
    data_done = any(j.get("script") == DOWNLOAD_JOB and
                    j.get("status") == "completed" for j in jobs.values())
    if data_done and not has_job_with_script(jobs, TRAIN_JOB):
        symbols, start = "eurusd,gbpusd,xauusd", "2025-01-01"
        for j in jobs.values():
            if j.get("script") == DOWNLOAD_JOB:
                p = j.get("params", {})
                symbols = p.get("symbols", symbols)
                start = p.get("start", start)
        name = f"{time.strftime('%Y%m%d')}-train-auto"
        enqueue(name, {
            "script": TRAIN_JOB,
            "params": {"train_args": ["--symbols", symbols,
                                       "--start", start]},
            "created_by": "coordinator",
        })
        log(f"CHAINED: queued {name} (symbols={symbols})")
        changed = True

    # 3. RETRY: failed jobs with attempts left, after cool-down
    for name, spec in jobs.items():
        if spec.get("status") != "failed":
            continue
        attempts = st["attempts"].get(name, 0)
        when = spec.get("claimed_at", 0)
        if attempts >= MAX_ATTEMPTS or time.time() - when < RETRY_COOLDOWN:
            continue
        if name.endswith("-retry"):
            base, n = name.rsplit("-", 1)
            new_name = f"{base}-{int(n) + 1}"
        else:
            new_name = f"{name}-retry1"
        spec = dict(spec)
        spec["status"] = "queued"
        spec["created_by"] = "coordinator-retry"
        spec.pop("claimed_by", None)
        spec.pop("claimed_at", None)
        enqueue(new_name, spec)
        st["attempts"][name] = attempts + 1
        log(f"RETRY: {name} failed {attempts + 1}x -> requeued as {new_name}")
        changed = True
    return changed


def collect(st):
    """4. COLLECT: pull new artifacts, commit reports/models to main."""
    r = git("rev-parse", ARTIFACTS_BRANCH, check=False)
    if r.returncode != 0:
        return False
    sha = r.stdout.strip()
    if sha == st.get("last_artifact_sha"):
        return False                      # nothing new
    first = bool(st.get("last_artifact_sha"))
    log(f"new artifacts at {sha[:8]} -> pull")
    rc = subprocess.call([sys.executable,
                          os.path.join(ROOT, "colab", "colab_cli.py"),
                          "pull"], cwd=ROOT)
    st["last_artifact_sha"] = sha
    if rc == 0 and first:
        # stay current (the trading loop's backup commits also touch main)
        git("fetch", "origin", "main", check=False)
        if git("rev-parse", "--verify", "origin/main", check=False).returncode == 0:
            git("rebase", "origin/main", check=False)
        # commit whatever landed in reports/ or models/ (data CSVs stay
        # untracked on main — the trading loop reads them from disk)
        git("add", "reports", "models", check=False)
        diff = git("status", "--porcelain", "reports", "models",
                   check=False).stdout.strip()
        if diff:
            git("commit", "-m",
                f"artifacts: pulled from colab-artifacts {sha[:8]}")
            if git("push", "origin", "main", check=False).returncode == 0:
                log(f"committed pulled artifacts to main ({sha[:8]})")
            else:
                log("WARNING: push to main failed (no credentials?) — "
                    "artifacts are local; will retry next cycle")
        else:
            log("pull done; nothing new to commit on main")
    return True


def registry(jobs):
    """Build the strategy registry: per-strategy progress + platform map."""
    now = time.strftime("%FT%TZ", time.gmtime())
    data_done = any(j.get("script") == DOWNLOAD_JOB and
                    j.get("status") == "completed" for j in jobs.values())
    train_jobs = {n: j for n, j in jobs.items()
                  if j.get("script") == TRAIN_JOB}
    train_status = "none"
    workers = []
    for spec in train_jobs.values():
        if spec.get("status") in ("queued", "claimed"):
            train_status = spec["status"]
            if spec.get("claimed_by"):
                workers.append(spec["claimed_by"])

    def platform(worker_id):
        w = (worker_id or "").lower()
        if "runners" in w or "actions" in w or w.startswith("fv-az"):
            return "github-actions"
        if "colab" in w:
            return "colab"
        if "kaggle" in w or "kernel" in w:
            return "kaggle"
        if "codespace" in w:
            return "codespaces"
        return worker_id or "unknown"

    # which strategy reports already exist on main or artifacts?
    have = {}
    for pat, key in [
            ("reports/backtest_ema.json", "ema"),
            ("reports/backtest_bollinger.json", "bollinger"),
            ("reports/backtest_donchian.json", "donchian"),
            ("reports/aegis_grid.json", "aegis"),
            ("reports/gbdt_metrics.json", "gbdt"),
            ("reports/mlp_metrics.json", "mlp")]:
        have[key] = os.path.isfile(os.path.join(ROOT, pat))
        if not have[key]:
            have[key] = git("cat-file", "-e",
                            f"{ARTIFACTS_BRANCH}:{pat}",
                            check=False).returncode == 0

    def mk(sid, name, done, note=""):
        status = ("validated" if done else
                  "training" if train_status == "claimed" else
                  "queued" if train_status == "queued" else
                  "awaiting-data" if not data_done else "ready")
        progress = 100 if done else (85 if train_status == "claimed" else
                                     20 if train_status == "queued" else
                                     15 if data_done else 0)
        return {"id": sid, "name": name, "status": status,
                "progress_pct": progress,
                "worked_on_by": sorted(set(map(platform, workers))),
                "notes": note}

    strategies = [
        mk("aegis", "Aegis Hedged Pairs (grid search)", have["aegis"],
           "primary strategy"),
        mk("gbdt", "XGBoost direction model", have["gbdt"]),
        mk("mlp", "MLP (torch) direction model", have["mlp"]),
        mk("ema_cross", "EMA crossover backtest", have["ema"]),
        mk("bollinger", "Bollinger reversion backtest", have["bollinger"]),
        mk("donchian", "Donchian breakout backtest", have["donchian"]),
    ]
    return {
        "updated_at": now,
        "summary": {
            "total": len(strategies),
            "validated": sum(1 for s in strategies
                             if s["status"] == "validated"),
            "active": sum(1 for s in strategies
                          if s["status"] in ("training", "queued")),
            "data_collected": data_done,
            "queue": {n: j.get("status") for n, j in jobs.items()},
        },
        "strategies": strategies,
    }


def write_registry(jobs):
    """Write reports/strategy_registry.json; returns True when changed."""
    reg = registry(jobs)
    path = os.path.join(ROOT, "reports", "strategy_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = None
    if os.path.isfile(path):
        try:
            with open(path) as f:
                old = json.load(f)
        except Exception:
            old = None
    if old == reg:
        return False
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)
    s = reg["summary"]
    log(f"registry: {s['total']} strategies | {s['validated']} validated | "
        f"{s['active']} active | data={'yes' if s['data_collected'] else 'no'}")
    return True


def main():
    log(f"coordinator started (interval={INTERVAL}s, "
        f"max_attempts={MAX_ATTEMPTS})")
    st = load_state()
    while True:
        try:
            ok_jobs, ok_art = fetch_branches()
            jobs = read_jobs() if ok_jobs else {}
            summary = ", ".join(f"{n}:{s.get('status')}"
                                for n, s in jobs.items()) or "empty"
            if chain(jobs, st):
                save_state(st)
            if write_registry(jobs):
                git("add", "reports/strategy_registry.json", check=False)
                if git("commit", "-m", "registry: strategy progress update",
                       check=False).returncode == 0:
                    if git("push", "origin", "main",
                           check=False).returncode == 0:
                        log("registry pushed to main")
            log(f"queue: {summary}")
            if ok_art and collect(st):
                save_state(st)
        except Exception as exc:
            log(f"cycle error: {exc}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

