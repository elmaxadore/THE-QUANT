#!/usr/bin/env bash
# =============================================================================
# THE QUANT — one-command VPS provisioning (v3.0 "zero-friction install").
# Idempotent. Safe to run more than once. Uses sudo where necessary.
#
# Usage:  sudo ./deploy/install.sh
# =============================================================================
set -euo pipefail

echo "== The Quant installer =="

# --- detect minimums --------------------------------------------------------
TOTAL_RAM_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
CPUS=$(nproc)
echo "   RAM ${TOTAL_RAM_MB} MB, ${CPUS} cpus"
if [ "$TOTAL_RAM_MB" -lt 4000 ]; then
  echo "   WARNING: <4GB RAM. The Quant will run in LEAN MODE (no lab/RL)."
fi

# --- system packages ---------------------------------------------------------
if [ -x "$(command -v apt-get)" ]; then
  sudo apt-get update -y
  sudo apt-get install -y git build-essential pkg-config libssl-dev \
    libzmq3-dev curl jq ufw logrotate || true
fi

# --- rust --------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  echo "== installing Rust toolchain =="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  . "$HOME/.cargo/env"
fi

# --- user + dirs ----------------------------------------------------------------
if ! id -u quant >/dev/null 2>&1; then
  sudo useradd -r -m -s /bin/bash quant
fi
sudo mkdir -p /opt/the-quant
sudo cp -r . /opt/the-quant
sudo chown -R quant:quant /opt/the-quant

# --- build -------------------------------------------------------------------
echo "== building release binary =="
cd /opt/the-quant
sudo -u quant bash -lc 'cargo build --release'
sudo cp target/release/the-quant /usr/local/bin/the-quant
sudo chmod +x /usr/local/bin/the-quant

# --- systemd -----------------------------------------------------------------
echo "== installing systemd service =="
sudo cp deploy/the-quant.service /etc/systemd/system/the-quant.service
sudo systemctl daemon-reload
sudo systemctl enable --now the-quant

# --- firewall -----------------------------------------------------------------------
echo "== configuring firewall (allow SSH only) =="
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow 22/tcp 2>/dev/null || true
  sudo ufw default deny incoming 2>/dev/null || true
  sudo ufw --force enable 2>/dev/null || true
fi

echo "== deployed. status: =="
sudo systemctl status the-quant --no-pager || true
echo "DONE — next steps: run 'the-quant --status' and configure master password."