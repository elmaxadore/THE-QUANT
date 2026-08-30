#!/usr/bin/env bash
# =============================================================================
#  THE QUANT — one-line installer & deployment
#
#  Usage (run as a normal user with sudo access — no need to be root):
#
#      curl -fsSL https://raw.githubusercontent.com/elmaxadore/THE-QUANT/main/deploy/install.sh | bash
#
#  Or from a clone:
#
#      ./deploy/install.sh
#
#  WHAT IT DOES
#  ----------------------------------------------------------------------------
#  1. Detects the machine tier from RAM/CPU (same logic as the Rust core:
#     Tier-1 <= 8 GB, Tier-2 <= 16 GB, Tier-3 <= 32 GB, Tier-4 > 32 GB).
#  2. Installs system prerequisites + the Rust toolchain (idempotent).
#  3. Clones the repo to /opt/the-quant (or reuses an existing checkout).
#  4. Picks the cargo feature set that MATCHES the tier and saves it to
#     /opt/the-quant/.build-features so every later rebuild (including the
#     24 h auto-updater) uses the same profile.
#  5. Builds the release binary, installs it to /usr/local/bin/the-quant.
#  6. Installs a tier-aware systemd unit (memory limits as % of RAM) and starts
#     the daemon (24 h auto-update scheduler).
#  7. Runs `the-quant restore` so the state dirs exist, then prints commands
#     to verify.
#
#  Overrides:
#      FEATURES="web,tui" bash deploy/install.sh   # force a feature set
#      INSTALL_DIR=/srv/the-quant bash ...          # change install location
#
#  Idempotent: safe to run again; it will update + rebuild in place.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; CYAN=$'\033[0;36m'; RED=$'\033[0;31m'; NC=$'\033[0m'
info()  { printf '%s[*]%s %s\n'  "$CYAN" "$NC" "$*"; }
ok()    { printf '%s[+]%s %s\n'  "$GREEN" "$NC" "$*"; }
warn()  { printf '%s[!]%s %s\n'  "$YELLOW" "$NC" "$*"; }
die()   { printf '%s[x]%s %s\n'  "$RED" "$NC" "$*"; exit 1; }

# run_root: execute a command as root (no-op if we already are root).
run_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

# ---------------------------------------------------------------------------
# 1. Detect machine tier (mirrors src/resource.rs)
# ---------------------------------------------------------------------------
TOTAL_RAM_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 4096)
CPUS=$(nproc 2>/dev/null || echo 2)

if   [ "$TOTAL_RAM_MB" -le  8192 ]; then TIER="Tier-1"; TIER_NAME="LEAN"
elif [ "$TOTAL_RAM_MB" -le 16384 ]; then TIER="Tier-2"; TIER_NAME="STANDARD"
elif [ "$TOTAL_RAM_MB" -le 32768 ]; then TIER="Tier-3"; TIER_NAME="ADVANCED"
else                                      TIER="Tier-4"; TIER_NAME="AGGRESSIVE"
fi

# Feature set by tier (web/tui are the Tier-2+ extras; ml/db stay opt-in:
# `ml` needs an ONNX Runtime shared library, `db` needs PostgreSQL).
case "$TIER" in
  Tier-1) DEFAULT_FEATURES="" ;;                       # lean: crypto only
  Tier-2) DEFAULT_FEATURES="web,tui" ;;
  Tier-3) DEFAULT_FEATURES="web,tui" ;;
  Tier-4) DEFAULT_FEATURES="web,tui" ;;
esac
FEATURES="${FEATURES:-$DEFAULT_FEATURES}"
INSTALL_DIR="${INSTALL_DIR:-/opt/the-quant}"

ok "Machine detected: ${TOTAL_RAM_MB} MB RAM, ${CPUS} CPUs -> ${TIER} (${TIER_NAME})"
[ -z "$FEATURES" ] && info "Feature profile: lean (default features only)"
# ---------------------------------------------------------------------------
# 2. System prerequisites + Rust toolchain (idempotent)
# ---------------------------------------------------------------------------
if [ -x "$(command -v apt-get)" ]; then
  info "Installing system prerequisites (git, curl, build tools)..."
  run_root apt-get update -y >/dev/null 2>&1 || true
  run_root apt-get install -y git curl build-essential pkg-config libssl-dev \
          libzmq3-dev ca-certificates >/dev/null 2>&1 || true
fi

if ! command -v cargo >/dev/null 2>&1; then
  info "Installing Rust toolchain (rustup, stable)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null 2>&1
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
cargo --version >/dev/null 2>&1 || die "cargo not available — install Rust first"

# ---------------------------------------------------------------------------
# 3. Source: clone the repo (or reuse an existing checkout)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || pwd)"
LOCAL_REPO=""
if [ -f "$SCRIPT_DIR/Cargo.toml" ] && [ -d "$SCRIPT_DIR/deploy" ]; then
  LOCAL_REPO="$SCRIPT_DIR"          # we were invoked from a clone (deploy/install.sh)
fi

if [ -n "$LOCAL_REPO" ]; then
  info "Using existing repository at $LOCAL_REPO"
else
  if [ -f "$INSTALL_DIR/Cargo.toml" ]; then
    info "Existing install found at $INSTALL_DIR — pulling latest"
    run_root git -C "$INSTALL_DIR" pull --ff-only origin main >/dev/null 2>&1 || warn "could not fast-forward; continuing"
    LOCAL_REPO="$INSTALL_DIR"
  else
    info "Cloning the-quant into $INSTALL_DIR ..."
    run_root mkdir -p "$INSTALL_DIR"
    run_root git clone https://github.com/elmaxadore/THE-QUANT.git "$INSTALL_DIR"
    if [ "${REPO_BRANCH:-main}" != "main" ]; then
      run_root git -C "$INSTALL_DIR" checkout "$REPO_BRANCH"
    fi
    LOCAL_REPO="$INSTALL_DIR"
  fi
fi

# Preserve the feature profile for every future rebuild (incl. auto-updater).
run_root tee "$LOCAL_REPO/.build-features" >/dev/null <<<"$FEATURES"

# ---------------------------------------------------------------------------
# 4. Build the release binary with the tier-matched features
# ---------------------------------------------------------------------------
info "Building release binary (this can take several minutes on small VPS)..."
export CARGO_TARGET_DIR="$LOCAL_REPO/target"
if [ -n "$FEATURES" ]; then
  cargo build --release --manifest-path "$LOCAL_REPO/Cargo.toml" --features "$FEATURES"
else
  cargo build --release --manifest-path "$LOCAL_REPO/Cargo.toml"
fi

run_root cp "$LOCAL_REPO/target/release/the-quant" /usr/local/bin/the-quant
run_root chmod +x /usr/local/bin/the-quant
ok "Binary installed: /usr/local/bin/the-quant"
# ---------------------------------------------------------------------------
# 5. Dedicated (non-root) runtime user
# ---------------------------------------------------------------------------
if ! id -u quant >/dev/null 2>&1; then
  info "Creating system user 'quant' (non-root)"
  run_root useradd -r -m -s /bin/bash quant
fi
run_root chown -R quant:quant "$LOCAL_REPO"

# Make sure the quant user can run cargo for the 24 h auto-update rebuild.
if [ ! -f /home/quant/.cargo/bin/cargo ]; then
  info "Installing Rust for the 'quant' service user (needed by auto-update)..."
  run_root mkdir -p /home/quant/.cargo
  run_root chown quant:quant /home/quant/.cargo
  run_root runuser -u quant -- bash -c \
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null 2>&1"
fi

# ---------------------------------------------------------------------------
# 6. systemd service (tier-aware: memory limits as % of RAM)
# ---------------------------------------------------------------------------
info "Installing systemd service..."
run_root mkdir -p /etc/the-quant
run_root cp "$LOCAL_REPO/deploy/the-quant.service" /etc/systemd/system/the-quant.service
run_root systemctl daemon-reload
run_root systemctl enable the-quant >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 7. First-run state + smoke check
# ---------------------------------------------------------------------------
/usr/local/bin/the-quant --smoke >/dev/null 2>&1 || warn "smoke check produced warnings — see output above"

# Create state dirs (restore is idempotent and run from the install dir).
run_root runuser -u quant -- bash -c \
  "cd '$LOCAL_REPO' && /usr/local/bin/the-quant restore"

# Start the daemon (24 h auto-update scheduler). Users who want to trade
# paper/live just use `the-quant paper N` directly — the daemon supervises
# updates in the background.
run_root systemctl start the-quant || warn "could not start service (running in a container? start manually with: the-quant daemon_run)"

# ---------------------------------------------------------------------------
# 8. Firewall (basic)
# ---------------------------------------------------------------------------
if command -v ufw >/dev/null 2>&1; then
  info "Configuring ufw (allow SSH, deny the rest)..."
  run_root ufw allow 22/tcp >/dev/null 2>&1 || true
  run_root ufw default deny incoming >/dev/null 2>&1 || true
  run_root ufw --force enable >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
printf '%s=========================================================%s\n' "$GREEN" "$NC"
printf '%s      THE QUANT INSTALLED  (%s / %s)%s\n'          "$GREEN" "$TIER" "$TIER_NAME" "$NC"
printf '%s=========================================================%s\n' "$GREEN" "$NC"
echo
info "Next steps:"
echo "  1) Check status:        the-quant --status"
echo "  2) Run a paper trade:   the-quant paper 20000"
echo "  3) Watch the service:   systemctl status the-quant"
echo "  4) Set master password: the-quant vault init   (see README)"
echo
info "Install location: $LOCAL_REPO"
info "Build features:   ${FEATURES:-<default>}  (saved to $LOCAL_REPO/.build-features)"
info "Auto-update:      enabled — the daemon checks the git repo every 24 h"
echo
ok "Done. Happy trading — in paper mode first!"
[ -n "$FEATURES" ] && info "Feature profile: ${FEATURES}"