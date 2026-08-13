#!/usr/bin/env bash
# =============================================================================
# THE QUANT v3.0 "Prometheus" — Single-Command Installer
# =============================================================================
# Detects the host OS, installs system dependencies (PostgreSQL 16 + TimescaleDB,
# build-essential, git), installs Rust, builds the release binary, initializes
# the database schema, and installs the systemd service.
#
# Idempotent: safe to re-run. Each step checks prior completion.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/TheQuantCompany/the-quant/main/install.sh | bash
#   # or locally:
#   ./install.sh [--release-tag v3.0.0] [--systemd] [--db-password PROMPT]
# =============================================================================

set -euo pipefail

# --- Configuration ------------------------------------------------------------
REPO_NAME="the-quant"
INSTALL_DIR="${HOME}/${REPO_NAME}"
DB_NAME="thequant"
DB_USER="quant"
SERVICE_NAME="the-quant.service"
RUST_MSRV="1.78"

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

require_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        log_warn "Running as non-root. System-level steps (apt, systemd) will use sudo."
        SUDO="sudo"
    else
        SUDO=""
    fi
}

# --- 1. Environment validation ------------------------------------------------
check_os() {
    log_step "Detecting operating system..."
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION_ID="${VERSION_ID:-unknown}"
        log_info "OS: ${PRETTY_NAME:-${OS_ID} ${OS_VERSION_ID}}"
    else
        die "Unsupported OS — only Linux (Ubuntu/Debian) is supported."
    fi

    case "${OS_ID}" in
        ubuntu|debian)
            ;;
        *)
            die "Unsupported distribution '${OS_ID}'. The Quant supports Ubuntu 22.04/24.04 LTS and Debian 12+."
            ;;
    esac
}

check_kernel() {
    log_step "Checking kernel version..."
    local kernel_version
    kernel_version=$(uname -r | cut -d. -f1,2 | tr -d '.')
    local kernel_major=$(uname -r | cut -d. -f1)
    local kernel_minor=$(uname -r | cut -d. -f2)
    if [[ ${kernel_major} -ge 5 && ${kernel_minor} -ge 4 ]]; then
        log_info "Kernel $(uname -r) — io_uring supported."
    else
        log_warn "Kernel $(uname -r) < 5.4 — io_uring unavailable; falling back to epoll."
    fi
}

check_ram() {
    log_step "Checking available memory..."
    local total_kb total_gb
    total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    total_gb=$(( total_kb / 1048576 ))
    if [[ ${total_gb} -lt 4 ]]; then
        log_warn "Only ${total_gb}GB RAM detected. The Quant will run in LEAN mode (minimal lab size)."
    else
        log_info "${total_gb}GB RAM detected. Resource budgets scale automatically."
    fi
}

check_disk() {
    log_step "Checking disk space..."
    local free_gb
    free_gb=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ ${free_gb} -lt 15 ]]; then
        die "Only ${free_gb}GB free disk. The Quant requires at least 15GB."
    fi
    log_info "${free_gb}GB free disk space."
}

# --- 2. System dependencies -----------------------------------------------------
install_system_deps() {
    if command -v psql >/dev/null 2>&1 && dpkg -l | grep -q timescaledb; then
        log_info "PostgreSQL + TimescaleDB already installed."
        return
    fi

    log_step "Installing system dependencies (PostgreSQL 16 + TimescaleDB)..."
    ${SUDO} apt-get update -qq

    # PostgreSQL 16 from apt.postgresql.org if not in distro repos
    if ! apt-cache show postgresql-16 >/dev/null 2>&1; then
        log_step "Adding PostgreSQL 16 repository..."
        ${SUDO} install -d /usr/share/postgresql-common/pgdg
        ${SUDO} wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc \
            | ${SUDO} tee /etc/apt/trusted.gpg.d/apt.postgresql.org.asc >/dev/null
        echo "deb [signed-by=/etc/apt/trusted.gpg.d/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
            | ${SUDO} tee /etc/apt/sources.list.d/pgdg.list >/dev/null
        ${SUDO} apt-get update -qq
    fi

    ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y \
        postgresql-16 \
        postgresql-client-16 \
        postgresql-16-timescaledb \
        git build-essential pkg-config libssl-dev cmake \
        python3 python3-pip curl wget

    # Enable TimescaleDB in postgresql.conf
    local pg_conf="/etc/postgresql/16/main/postgresql.conf"
    if [[ -f "${pg_conf}" ]] && ! grep -q "timescaledb" "${pg_conf}"; then
        echo "shared_preload_libraries = 'timescaledb'" | ${SUDO} tee -a "${pg_conf}" >/dev/null
        ${SUDO} systemctl restart postgresql
    fi
    log_info "PostgreSQL 16 + TimescaleDB installed."
}

# --- 3. Rust toolchain ---------------------------------------------------------
install_rust() {
    if command -v cargo >/dev/null 2>&1; then
        local installed
        installed=$(cargo --version | awk '{print $2}' | tr -d '.')
        local required="${RUST_MSRV//./}"
        if [[ ${installed//./} -ge ${required%?}${required: -1} ]]; then
            log_info "Rust $(cargo --version | awk '{print $2}') already installed."
            return
        fi
    fi

    log_step "Installing Rust toolchain (stable, MSRV ${RUST_MSRV}+)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
    cargo --version
    rustc --version
    log_info "Rust installed."
}

# --- 4. Database setup -----------------------------------------------------------
setup_database() {
    if ${SUDO} -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
        log_info "Database user '${DB_USER}' already exists."
    else
        log_step "Creating database user '${DB_USER}'..."
        local db_password="${QUANT_DB_PASSWORD:-}"
        if [[ -z "${db_password}" ]]; then
            db_password=$(openssl rand -hex 16)
            log_warn "Generated random DB password. Store it securely!"
            echo "DB_PASSWORD=${db_password}" > "${INSTALL_DIR}/.db_credentials"
            chmod 600 "${INSTALL_DIR}/.db_credentials"
        fi
        ${SUDO} -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${db_password}';"
        ${SUDO} -u postgres psql -c "ALTER USER ${DB_USER} WITH SUPERUSER;"
    fi

    if ${SUDO} -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
        log_info "Database '${DB_NAME}' already exists."
    else
        log_step "Creating database '${DB_NAME}'..."
        ${SUDO} -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
        ${SUDO} -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
        ${SUDO} -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
        log_info "Database '${DB_NAME}' created."
    fi
}

# --- 5. Clone / update source -----------------------------------------------------
fetch_source() {
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
        log_step "Updating existing repository at ${INSTALL_DIR}..."
        git -C "${INSTALL_DIR}" pull --rebase --autostash || log_warn "Git pull failed — continuing with existing tree."
    else
        log_step "Cloning repository..."
        git clone "https://github.com/TheQuantCompany/${REPO_NAME}.git" "${INSTALL_DIR}" 2>/dev/null \
            || git clone "https://github.com/TheQuantCompany/the-quant.git" "${INSTALL_DIR}" 2>/dev/null \
            || {
                log_warn "Remote clone failed — using current working directory."
                if [[ "$(pwd)" != "${INSTALL_DIR}" ]]; then
                    cp -r "$(pwd)" "${INSTALL_DIR}"
                fi
            }
    fi

    cd "${INSTALL_DIR}"
    log_info "Source ready at ${INSTALL_DIR}"
}

# --- 6. Build release binary ---------------------------------------------------------
build_release() {
    log_step "Building release binary (this may take several minutes on first run)..."
    cargo build --release --features full
    chmod +x target/release/the-quant
    log_info "Release binary built: ${INSTALL_DIR}/target/release/the-quant"
}

# --- 7. Run migrations ----------------------------------------------------------------
run_migrations() {
    log_step "Running database migrations..."
    local db_password
    db_password=$(grep -oP '(?<=DB_PASSWORD=).*' "${INSTALL_DIR}/.db_credentials" 2>/dev/null || echo "")
    if [[ -f "${INSTALL_DIR}/migrations/001_init.sql" ]]; then
        PGPASSWORD="${db_password}" ${SUDO} -u postgres psql -d "${DB_NAME}" \
            -f "${INSTALL_DIR}/migrations/001_init.sql" || true
    fi
    if [[ -f "${INSTALL_DIR}/migrations/002_v3.sql" ]]; then
        PGPASSWORD="${db_password}" ${SUDO} -u postgres psql -d "${DB_NAME}" \
            -f "${INSTALL_DIR}/migrations/002_v3.sql" || true
    fi
    if [[ -f "${INSTALL_DIR}/migrations/003_v3_1.sql" ]]; then
        PGPASSWORD="${db_password}" ${SUDO} -u postgres psql -d "${DB_NAME}" \
            -f "${INSTALL_DIR}/migrations/003_v3_1.sql" || true
    fi
    log_info "Migrations applied."
}

# --- 8. Systemd service ----------------------------------------------------------------
install_service() {
    if [[ ! -f "${INSTALL_DIR}/deploy/the-quant.service" ]]; then
        log_warn "systemd unit file not found — skipping service install."
        return
    fi

    log_step "Installing systemd service..."
    ${SUDO} cp "${INSTALL_DIR}/deploy/the-quant.service" "/etc/systemd/system/${SERVICE_NAME}"
    if id "quant" >/dev/null 2>&1; then
        log_info "User 'quant' exists."
    else
        ${SUDO} useradd --system --home "${INSTALL_DIR}" --shell /usr/sbin/nologin quant
        ${SUDO} chown -R quant:quant "${INSTALL_DIR}"
    fi
    ${SUDO} systemctl daemon-reload
    ${SUDO} systemctl enable "${SERVICE_NAME}"
    log_info "systemd service installed & enabled."
    log_step "Start with: sudo systemctl start ${SERVICE_NAME}"
}

# --- 9. Firewall ---------------------------------------------------------------------------
configure_firewall() {
    if command -v ufw >/dev/null 2>&1; then
        log_step "Configuring UFW firewall (SSH only)..."
        ${SUDO} ufw allow OpenSSH
        ${SUDO} ufw enable
        log_info "UFW configured: SSH (22) allowed, all inbound else blocked."
    else
        log_warn "ufw not present — skipping firewall configuration."
    fi
}

# --- 10. Logrotate ---------------------------------------------------------------------------
configure_logrotate() {
    log_step "Configuring logrotate..."
    ${SUDO} tee /etc/logrotate.d/the-quant >/dev/null <<EOF
${INSTALL_DIR}/logs/system/*.log ${INSTALL_DIR}/logs/trades/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
    log_info "Logrotate configured."
}

# --- Main ------------------------------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}======================================================${RESET}"
    echo -e "${BOLD}  THE QUANT v3.0 \"Prometheus\" — Installer${RESET}"
    echo -e "${BOLD}  Percentage-Scaled Autonomous Trading Platform${RESET}"
    echo -e "${BOLD}======================================================${RESET}"
    echo ""

    require_root
    check_os
    check_kernel
    check_ram
    check_disk
    install_system_deps
    install_rust
    fetch_source
    setup_database
    build_release
    run_migrations
    install_service
    configure_firewall
    configure_logrotate

    echo ""
    echo -e "${GREEN}================================================================${RESET}"
    echo -e "${GREEN}  Installation complete!${RESET}"
    echo -e "${GREEN}================================================================${RESET}"
    echo ""
    echo "  Binary:      ${INSTALL_DIR}/target/release/the-quant"
    echo "  Config:      ~/.thequant/config/system.toml"
    echo ""
    echo "  Next steps:"
    echo "    1. Configure secrets:  nano ~/.thequant/config/system.toml"
    echo "    2. Run first bootstrap: sudo -u quant ${INSTALL_DIR}/target/release/the-quant bootstrap"
    echo "    3. Start the daemon:    sudo systemctl start ${SERVICE_NAME}"
    echo "    4. Web dashboard:       http://<vps-ip>:8080"
    echo ""
}

main "$@"

