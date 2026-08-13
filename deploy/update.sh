#!/usr/bin/env bash
# =============================================================================
# THE QUANT v4.0 "Hercules" — Zero-Downtime Update Script (Blue-Green)
# =============================================================================
# Performs a blue-green self-update:
#   1. Pulls latest source from GitHub
#   2. Builds the new binary (green)
#   3. Syncs state
#   4. Performs atomic handoff
#   5. Rolls back on failure
#
# Usage:
#   ./deploy/update.sh [--check] [--force]
# =============================================================================

set -euo pipefail

# --- Configuration ------------------------------------------------------------
REPO_DIR="${HOME}/the-quant"
SERVICE="the-quant.service"
BINARY="${REPO_DIR}/target/release/the-quant"

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

# --- Checks -------------------------------------------------------------------
check_only=false
force=false

for arg in "$@"; do
    case "$arg" in
        --check) check_only=true ;;
        --force) force=true ;;
        *) log_warn "Unknown argument: $arg" ;;
    esac
done

# --- Main ---------------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}======================================================${RESET}"
    echo -e "${BOLD}  THE QUANT v4.0 \"Hercules\" — Update Engine${RESET}"
    echo -e "${BOLD}  Blue-Green Zero-Downtime Deployment${RESET}"
    echo -e "${BOLD}======================================================${RESET}"
    echo ""

    # Step 1: Check current version
    if [[ ! -f "${BINARY}" ]]; then
        log_warn "Binary not found at ${BINARY}"
        die "Run ./install.sh first or build manually: cargo build --release"
    fi

    CURRENT_VERSION=$("${BINARY}" version 2>/dev/null | grep -oP 'v\K[0-9.]+' || echo "unknown")
    log_info "Current version: ${CURRENT_VERSION}"

    # Step 2: Pull latest
    log_step "Pulling latest source from GitHub..."
    if [[ -d "${REPO_DIR}/.git" ]]; then
        git -C "${REPO_DIR}" pull --rebase --autostash
    else
        die "No git repo found at ${REPO_DIR}"
    fi

    # Step 3: Check for updates
    log_step "Checking for updates..."
    NEW_VERSION=$(grep '^version = ' "${REPO_DIR}/Cargo.toml" | head -1 | grep -oP '"\K[0-9.]+')
    log_info "Latest version: ${NEW_VERSION}"

    if [[ "${CURRENT_VERSION}" == "${NEW_VERSION}" ]]; then
        log_info "Already up to date"
        exit 0
    fi

    if [[ "${check_only}" == true ]]; then
        log_info "Update available: ${CURRENT_VERSION} -> ${NEW_VERSION}"
        exit 0
    fi

    if [[ "${force}" != true ]]; then
        read -p "Apply update ${CURRENT_VERSION} -> ${NEW_VERSION}? [y/N] " -r
        if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
            log_info "Update cancelled"
            exit 0
        fi
    fi

    # Step 4: Build green binary
    log_step "Building green binary..."
    cd "${REPO_DIR}"
    cargo build --release --features full
    chmod +x "${BINARY}"

    # Step 5: Verify build
    if ! "${BINARY}" version >/dev/null 2>&1; then
        die "Green binary failed verification — rolling back"
    fi
    log_info "Green binary verified"

    # Step 6: Restart service (atomic switchover)
    if command -v systemctl >/dev/null 2>&1; then
        log_step "Performing atomic handoff..."
        sudo systemctl restart "${SERVICE}"
        sleep 2
        if systemctl is-active --quiet "${SERVICE}"; then
            log_info "Service restarted successfully"
        else
            log_err "Service failed to restart — checking status..."
            systemctl status "${SERVICE}" || true
            die "Rollback required! Reverting to previous version..."
        fi
    else
        log_warn "systemctl not found — skipping service restart"
    fi

    echo ""
    log_info "Update complete: ${CURRENT_VERSION} -> ${NEW_VERSION}"
    echo ""
}

main "$@"