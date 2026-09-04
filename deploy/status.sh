#!/bin/bash
# THE QUANT — VPS STATUS SUMMARY (what the box is doing, at a glance).
# Run it on the VPS, or from your own terminal via:
#   ssh -i ~/Downloads/"Trading account.pem" ubuntu@HOST "~/THE-QUANT/deploy/status.sh"
REPO="${REPO:-$HOME/THE-QUANT}"
MT5_DIR="${MT5_DIR:-$HOME/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files}"

echo "═══ THE QUANT · VPS status · $(date -u +%FT%TZ) ═══"

# services
printf "  %-20s : %s\n" "trading loop"  "$(systemctl is-active the-quant)"
printf "  %-20s : %s\n" "coordinator"   "$(systemctl is-active the-quant-coordinator)"
pgrep -f "colab/agent.py" >/dev/null 2>&1 \
  && printf "  %-20s : %s\n" "local worker" "running (pid $(pgrep -f 'colab/agent.py' | head -1))" \
  || printf "  %-20s : %s\n" "local worker" "stopped"

# resources
mem=$(free -m | sed -n 2p | awk '{printf "%dMi free / %dMi used", $7, $3}')
swap=$(free -m | sed -n 2p | awk '{printf "%dMi", $8}')
printf "  %-20s : %s\n" "memory" "$mem (swap $swap)"
df -h "$REPO" 2>/dev/null | tail -1 | \
  awk '{printf "  %-20s : %s used of %s (avail %s, %s)\n", "disk", $3, $2, $4, $5}'

# queue + coordinator
q=$(grep "queue:" "$REPO/coordinator.log" 2>/dev/null | tail -1 | sed 's/^.*queue: //')
printf "  %-20s : %s\n" "job queue" "${q:-empty}"

# strategy registry
if [ -f "$REPO/reports/strategy_registry.json" ]; then
    python3 - "$REPO" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1] + "/reports/strategy_registry.json"))
s = r["summary"]
print(f"  {'strategies':<20} : {s['total']} total | {s['validated']} validated | {s['active']} active | data={'yes' if s['data_collected'] else 'no'}")
for st in r["strategies"]:
    bar = "#" * (st["progress_pct"] // 10)
    by = "/".join(st["worked_on_by"]) or "-"
    print(f"      {st['id']:<12} {st['status']:<14} {st['progress_pct']:>3}% [{bar}] {by}")
PYEOF
else
    printf "  %-20s : %s\n" "strategies" "(registry not written yet)"
fi

# MT5 data
n_mt5=$(find "$MT5_DIR" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
n_hist=$(find "$REPO/python/data/histdata" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
printf "  %-20s : %s csv(s)\n" "MT5 bars" "$n_mt5"
printf "  %-20s : %s csv(s)\n" "histdata" "$n_hist"

# recent trades
trades=$(tail -n 2 "$REPO"/state/trades/*.jsonl 2>/dev/null | tail -1)
[ -n "$trades" ] && printf "  %-20s : %s\n" "last trade" "$trades" \
  || printf "  %-20s : %s\n" "last trade" "(none)"

# recent coordinator activity
printf "  %-20s :\n" "recent coordinator"
tail -n 4 "$REPO/coordinator.log" 2>/dev/null | sed 's/^/      /'
echo "═══ end ═══"