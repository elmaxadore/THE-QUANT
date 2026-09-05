#!/usr/bin/env bash
# =============================================================================
# THE QUANT — in-place update helper.
# Pulls latest code, rebuilds, blue-green swaps, and restarts the service.
# The daemon also performs this automatically every 24h; this script is for a
# manual push / rollback.
# =============================================================================
set -euo pipefail

cd /opt/the-quant
echo "== update.sh =="

# Snapshot state into git so nothing is lost.
git add -A && git commit -m "pre-update snapshot" || true
git push origin HEAD || true

# Pull latest code (rebase).
git pull --rebase origin HEAD

# Rebuild with the SAME feature profile chosen at install time
# (saved by deploy/install.sh to .build-features).
FEATURE_ARGS=()
if [ -f .build-features ] && [ -s .build-features ]; then
  FEATS=$(tr -d '[:space:]' < .build-features)
  [ -n "$FEATS" ] && FEATURE_ARGS=(--features "$FEATS")
fi
cargo build --release "${FEATURE_ARGS[@]}"
BUILT=target/release/the-quant
STAGE=/tmp/the-quant-new
cp "$BUILT" "$STAGE"

# Smoke test before swapping.
if ! "$STAGE" --smoke; then
  echo "!! smoke test failed — aborting swap, keeping current binary"
  exit 1
fi

# Blue-green swap.
cp /usr/local/bin/the-quant /usr/local/bin/the-quant.prev
mv "$STAGE" /usr/local/bin/the-quant

# Restart the daemon so it runs the new binary.
sudo systemctl restart the-quant || true
echo "OK — updated and restarted."