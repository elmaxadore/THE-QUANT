#!/bin/bash
# THE QUANT — VPS AUTOPILOT.
#
# One command drives EVERYTHING on the box, idempotently (safe to re-run):
#   1. sync latest code from the repo
#   2. (re)start the paper-trading loop service
#   3. (re)start the mission-control coordinator
#   4. start a local job WORKER (executes queued cloud jobs right here)
#   5. if a MetaTrader 5 terminal + MQL5/Files exist, start the MT5 sync loop
#   6. print + log a status summary
#
# Token: to give this box read/write push access, save your GitHub token
# (fine-grained PAT, Contents read+write) in ~/.thequant_token. Without it the
# box still coordinates (reads the queue) but cannot push / run a worker.
#
set -u
REPO="${REPO:-$HOME/THE-QUANT}"
LOG="$REPO/autopilot.log"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.thequant_token}"
GIT_REMOTE="https://github.com/elmaxadore/THE-QUANT.git"

ts() { date -u +%FT%TZ; }
say() { echo "[autopilot] $(ts) $*"; }
log() { echo "[autopilot] $(ts) $*" >> "$LOG"; }

log "===== start ====="

# ---- 1. code sync ----
if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 "$GIT_REMOTE" "$REPO"
fi
cd "$REPO" || exit 1
git fetch origin main -q 2>/dev/null
git pull --rebase -q 2>/dev/null || git reset --hard origin/main -q 2>/dev/null
log "code at $(git rev-parse --short HEAD)"

# ---- optional token ----
if [ -f "$TOKEN_FILE" ]; then
    export GITHUB_TOKEN="$(tr -d '\n\r' < "$TOKEN_FILE")"
    log "token loaded from $TOKEN_FILE"
fi

# ---- 2. trading loop ----
UNIT=deploy/the-quant-1gb.service
[ -f "$UNIT" ] || UNIT=deploy/the-quant.service
sudo cp -f "$UNIT" /etc/systemd/system/the-quant.service 2>/dev/null
sudo mkdir -p /etc/systemd/system/the-quant.service.d
sudo tee /etc/systemd/system/the-quant.service.d/tune.conf >/dev/null <<'UNITEOF'
[Service]
Environment=SLEEP_SECS=900
Environment=BARS_PER_CYCLE=2000
UNITEOF
sudo systemctl daemon-reload
sudo systemctl enable --now the-quant 2>/dev/null
log "trading service: $(systemctl is-active the-quant)"

# ---- 3. coordinator ----
sudo cp -f deploy/the-quant-coordinator.service \
    /etc/systemd/system/ 2>/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now the-quant-coordinator 2>/dev/null
log "coordinator: $(systemctl is-active the-quant-coordinator)"

# ---- 4. local worker (executes jobs from the shared queue) ----
# Only start a worker here if we have a URL-usable token AND python3-pip
# (a token with ':' breaks git URLs; without pip the job deps can't install.
# Otherwise EC2 stays the coordinator/collector and heavier jobs run on
# GitHub Actions / Colab / Kaggle (with their own token injections) automatically.
if [ -n "${GITHUB_TOKEN:-}" ] && ! case "$GITHUB_TOKEN" in *:* ) true;; *) false;; esac; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        log "token set but python3-pip missing — worker skipped (pip install: "
            "$(sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip 2>&1 | tail -1))"
    fi
    if python3 -m pip --version >/dev/null 2>&1; then
        mkdir -p state
        PIDFILE="$REPO/state/agent.pid"
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            log "worker already running (pid $(cat "$PIDFILE"))"
        else
            nohup python3 "$REPO/colab/agent.py" --repo "$REPO" --interval 60 \
                --bootstrap-on-claim \
                --bootstrap "python3 -m pip install -q --user pandas numpy requests \
                    xgboost scikit-learn onnx onnxruntime" \
                >> "$LOG" 2>&1 &
            echo $! > "$PIDFILE"
            log "worker started (pid $!)"
        fi
    fi
elif [ -n "${GITHUB_TOKEN:-}" ]; then
    log "token contains ':' — not URL-usable (git pushes will fail); worker skipped. Use a fine-grained PAT (ghp_/github_pat_,no colons) in $TOKEN_FILE, or the deploy key from Settings -> Deploy keys."
else
    log "no token — skipping worker (coordinator still reads the queue)"
fi

# ---- 5. MT5 terminal data sync ----
MT5_DIR="${MT5_DIR:-$HOME/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files}"
if [ -d "$MT5_DIR" ]; then
    echo "[autopilot] $(ts) mt5 sync loop" >> "$LOG"
    nohup bash "$REPO/deploy/mt5_sync_loop.sh" >> "$LOG" 2>&1 &
    log "MT5 dir detected — sync loop started"
else
    log "no MT5 terminal dir yet ($MT5_DIR)"
fi

# ---- 6. summary ----
log "===== done ====="
"$REPO/deploy/status.sh" 2>/dev/null