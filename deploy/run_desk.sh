#!/bin/bash
# THE QUANT — continuous paper-trading loop for small VPS (1 GB RAM).
# Runs N-bar trading cycles with cool-down sleeps; persists state to git.
# Memory profile: the binary peaks at ~14 MB per cycle; between cycles it
# exits, so the OS reclaims everything (no leak accumulation).
set -u
cd /home/ubuntu/THE-QUANT

BARS_PER_CYCLE="${BARS_PER_CYCLE:-5000}"
SLEEP_SECS="${SLEEP_SECS:-60}"
BACKUP_EVERY="${BACKUP_EVERY:-10}"     # backup every N cycles
UPDATE_EVERY="${UPDATE_EVERY:-1440}"   # auto-update check: once/day (~1min/cycle)

cycle=0
last_update=$(date +%s)

echo "[loop] starting: ${BARS_PER_CYCLE} bars/cycle, sleep ${SLEEP_SECS}s"
while true; do
    cycle=$((cycle + 1))
    echo "[loop] === cycle ${cycle} $(date -u +%FT%TZ) ==="

    # daily self-update check (git pull + blue-green swap)
    now=$(date +%s)
    if (( now - last_update >= UPDATE_EVERY * 60 )); then
        echo "[loop] daily update check"
        the-quant update --now || echo "[loop] update check failed (continuing)"
        last_update=$now
    fi

    the-quant paper "$BARS_PER_CYCLE" || echo "[loop] paper cycle failed (continuing)"

    # periodic state backup -> git
    if (( cycle % BACKUP_EVERY == 0 )); then
        the-quant backup || echo "[loop] backup failed (will retry)"
    fi

    sleep "$SLEEP_SECS"
done
