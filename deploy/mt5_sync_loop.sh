#!/bin/bash
# THE QUANT — MT5 data sync loop. Every X minutes:
#   - copy any bars the MT5 terminal EA has written to MQL5/Files
#   - push them to the colab-artifacts branch
#   - enqueue training on the real data (once)
set -u
REPO="${REPO:-$HOME/THE-QUANT}"
INTERVAL="${MT5_SYNC_INTERVAL:-600}"   # seconds
echo "[mt5-loop] started, every ${INTERVAL}s"
while true; do
    bash "$REPO/deploy/mt5_sync.sh" 2>&1
    sleep "$INTERVAL"
done