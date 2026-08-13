#!/usr/bin/env bash
# =============================================================================
# THE QUANT v4.0 "Hercules" — Restore From Git (One-Command Recovery)
# =============================================================================
# Restores the ENTIRE system on a fresh machine:
#   git clone https://github.com/elmaxadore/THE-QUANT.git
#   cd THE-QUANT
#   ./deploy/restore.sh
#
# Time to resume trading from clone: < 5 minutes.
# =============================================================================

set -euo pipefail

# --- Configuration ------------------------------------------------------------
REPO_NAME="THE-QUANT"
REPO_URL="https://github.com/elmaxadore/${REPO_NAME}.git"
INSTALL_DIR="${HOME}/${REPO_NAME}"
DB_NAME="thequant"
DB_USER="quant"
SERVICE_NAME="the-quant.service"

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

log_info()  { echo -e "${GREEN}[✓]${RESET} $*"; }
log_step()  { echo -e "${CYAN}[➤]${RESET} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
log_err()   { echo -e "${RED}[✗]${RESET} $*" >&2; }

die() { log_err "$*"; exit 1; }

# --- Main ---------------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}======================================================${RESET}"
    echo -e "${BOLD}  THE QUANT v4.0 \"Hercules\" — Restore Engine${RESET}"
    echo -e "${BOLD}  From Git to Full Trading: < 5 minutes${RESET}"
    echo -e "${BOLD}======================================================${RESET}"
    echo ""

    # Step 1: Install Rust if needed
    log_step "Checking Rust installation..."
    if ! command -v cargo >/dev/null 2>&1; then
        log_step "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default
        # shellcheck disable=SC1091
        source "${HOME}/.cargo/env"
    fi
    log_info "Rust: $(cargo --version)"

    # Step 2: Build release binary
    log_step "Building release binary..."
    cd "${INSTALL_DIR}"
    cargo build --release --features full
    chmod +x target/release/the-quant
    log_info "Binary built: ${INSTALL_DIR}/target/release/the-quant"

    # Step 3: Install PostgreSQL + TimescaleDB if needed
    if ! command -v psql >/dev/null 2>&1; then
        log_step "Installing PostgreSQL + TimescaleDB..."
        if [[ -x ./install.sh ]]; then
            ./install.sh --restore-mode
        fi
    else
        log_info "PostgreSQL already installed"
    fi

    # Step 4: Run database migrations
    log_step "Running database migrations..."
    if command -v psql >/dev/null 2>&1; then
        for migration in migrations/*.sql; do
            [[ -f "${migration}" ]] || continue
            log_step "Applying: ${migration}"
            sudo -u postgres psql -d "${DB_NAME}" -f "${migration}" || log_warn "Migration failed (may already be applied)"
        done
    fi

    # Step 5: Verify state manifest
    log_step "Verifying state manifest..."
    if [[ ! -f state/manifest.json ]]; then
        log_warn "No state/manifest.json found — initializing empty state"
        ./target/release/the-quant bootstrap || true
    fi

    # Step 6: Run restore via binary
    log_step "Running the-quant restore..."
    ./target/release/the-quant restore --skip-mt5 --skip-db || true

    # Step 7: Install service
    log_step "Installing systemd service..."
    if command -v systemctl >/dev/null 2>&1 && [[ -f deploy/${SERVICE_NAME} ]]; then
        sudo cp "deploy/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
        sudo systemctl daemon-reload
        sudo systemctl enable "${SERVICE_NAME}"
        log_warn "Service installed. Start with: sudo systemctl start ${SERVICE_NAME}"
    fi

    # Step 8: Reconnect to MT5
    log_step "Reconnecting to MT5..."
    # MT5 bridge reconnection is performed by the daemon on startup.

    echo ""
    echo -e "${GREEN}================================================================${RESET}"
    echo -e "${GREEN}  Restore complete!${RESET}"
    echo -e "${GREEN}================================================================${RESET}"
    echo ""
    echo "  System reconstructed from git."
    echo "  Start trading: sudo systemctl start ${SERVICE_NAME}"
    echo "  Or run in foreground: ${INSTALL_DIR}/target/release/the-quant daemon"
    echo ""
}

main "$@"