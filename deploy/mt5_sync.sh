#!/bin/bash
# THE QUANT — MT5 data sync: copy bars exported by the terminal EA into the
# repo, push them to colab-artifacts, and enqueue training on the real data.
# Run after the MT5 terminal (MetaQuotes-Demo login) has produced CSVs.
set -e
REPO="${REPO:-$HOME/THE-QUANT}"
MT5_FILES="${MT5_FILES:-$HOME/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files}"
HIST="$REPO/python/data/histdata"

cd "$REPO"
git pull --rebase -q || true
mkdir -p "$HIST"

copied=0
for f in "$MT5_FILES"/*.csv; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    sym=$(basename "$base" .csv)
    # keep newest-5-char alias canonical: EA writes <SYMBOL>.csv (raw name)
    cp -u "$f" "$HIST/$base"
    copied=$((copied + 1))
    echo "[mt5-sync] copied $base ($(stat -c%s "$f") bytes)"
done
[ "$copied" -gt 0 ] || { echo "[mt5-sync] nothing to copy"; exit 0; }

# push the real data to the artifacts branch so every cloud worker can use it
python3 - <<EOF
import sys, os
sys.path.insert(0, "$REPO/colab")
os.environ.setdefault("COLAB_REPO_DIR", "$REPO")
import persist
files = []
for f in os.listdir("$HIST"):
    if f.endswith(".csv"):
        files.append(os.path.join("$HIST", f))
sha = persist.push_artifacts(
    "$REPO", files,
    message="mt5-demo data: real MetaQuotes-Demo bars",
    subdir="histdata_mt5")
print("[mt5-sync] pushed", len(files), "csv(s) ->", (sha or "FAILED")[:8])
EOF

# queue training on the fresh data if not already queued/done
python3 - "$REPO" <<'EOF'
import sys, json, subprocess, time, os, glob
repo = sys.argv[1]
subprocess.run(["git", "-C", repo, "fetch", "origin",
                "colab-jobs:colab-jobs"], capture_output=True)
r = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only",
                    "colab-jobs"], capture_output=True, text=True)
train_queued = False
for p in (r.stdout or "").split():
    if p.endswith("/job.json"):
        try:
            spec = json.loads(subprocess.run(
                ["git", "-C", repo, "show", f"colab-jobs:{p}"],
                capture_output=True, text=True).stdout)
            if spec.get("script") == "colab/jobs/train_data.py" and \
                    spec.get("status") in ("queued", "claimed", "completed"):
                train_queued = True
        except Exception:
            pass
if train_queued:
    print("[mt5-sync] training already queued/running/done — not duplicating")
    sys.exit(0)
sys.path.insert(0, os.path.join(repo, "colab"))
import persist
syms = sorted(os.path.basename(f).split(".")[0]
              for f in glob.glob(os.path.join(repo, "python/data/histdata/*.csv")))
name = f"{time.strftime('%Y%m%d')}-train-mt5"
persist.enqueue_job(repo, name, {
    "script": "colab/jobs/train_data.py",
    "params": {"train_args": ["--symbols", ",".join(syms), "--download", "0"]},
    "created_by": "mt5-sync"})
print("[mt5-sync] queued", name, "symbols:", ",".join(syms))
EOF
echo "[mt5-sync] done"
