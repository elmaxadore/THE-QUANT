# THE QUANT v4.0 "Hercules"

**Resilience Through Updatability** — An autonomous, self-contained quantitative trading company written entirely in Rust.

## Core Principles

1. **GIT IS GOD** — The GitHub repo is the single source of truth for ALL state. The database is a cache. The binary is disposable.
2. **ACCOUNTS ARE ISLANDS** — Every account is a completely separate entity with its own strategy pool, model zoo, risk parameters, and evolution cycles.
3. **THE SYSTEM LEARNS OR IT DIES** — Strategies are born in the Lab, tested, promoted, and retired. Models retrain on every evolution cycle.
4. **UPDATES ARE NON-EVENTS** — Blue-green deployment with zero downtime. Rollback is instant. State is never lost.
5. **PRODUCTION-GRADE OR NOTHING** — Every line is audited. Every error is typed. Every state change is committed.

## Quick Start

### One-Command Install
```bash
curl -fsSL https://raw.githubusercontent.com/elmaxadore/THE-QUANT/main/deploy/install.sh | bash
```

### One-Command Restore (fresh machine)
```bash
git clone https://github.com/elmaxadore/THE-QUANT.git
cd THE-QUANT
./deploy/restore.sh
```
Time to resume trading from clone: **< 5 minutes**.

### One-Command Update
```bash
the-quant update --check    # Check for updates
the-quant update --apply    # Blue-green switchover
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `the-quant daemon` | Start the trading daemon (headless) |
| `the-quant bootstrap` | First-time setup wizard |
| `the-quant web` | Web dashboard (Axum + HTMX) |
| `the-quant tui` | Terminal UI (ratatui) |
| `the-quant health` | System health check (JSON) |
| `the-quant restore` | Rebuild system from git |
| `the-quant update` | Self-update engine (blue-green) |
| `the-quant doctor` | Full system diagnostics |
| `the-quant status` | Overview of system + all account desks |
| `the-quant account-add <name> <variant>` | Add a new trading desk |
| `the-quant lab status` | Show lab progress |
| `the-quant evolve` | Trigger manual evolution cycle |

## Architecture

```
THE-QUANT/
├── src/
│   ├── state/          # Git-centric state (manifest, store, restore)
│   ├── firm/           # Multi-account TradingDesk + QuantFirm
│   ├── update/         # Blue-green self-update engine
│   ├── github/         # Auto-commit engine + push
│   ├── account/        # Account management (v3.1)
│   ├── pipeline/       # Account pipeline (v3.1)
│   └── ...             # v3.0 modules
├── state/              # THE CRITICAL DIRECTORY — source of truth
│   ├── accounts/<uuid>/  # Per-account state
│   ├── global/           # Shared market data, regimes
│   └── manifest.json     # Master state manifest (checksums)
├── config/templates/   # Prop-firm rule templates (YAML)
├── deploy/             # install.sh, update.sh, restore.sh
└── .github/workflows/  # CI, release, deploy
```

## State Architecture

The `state/manifest.json` is the **Rosetta Stone** of the system. It maps every piece of state to its location, version, and checksum. On catastrophic failure:

1. Clone the repo to a fresh machine
2. Run `the-quant restore`
3. Reconnect to MT5
4. Resume trading with ALL historical data, models, strategies, and account states intact

## Account Types

| Type | Profit Target | Time Limit | Max DD | Daily Loss | Consistency | Shield |
|------|--------------|------------|--------|------------|-------------|--------|
| Instant | None | None | 10% | None | Yes (15%) | Yes ($50) |
| 1-Step Eval | 8% | 30-60d | 10% | 5% | Yes | No |
| 2-Step Eval | 8%+5% | 30-60d | 10% | 5% | Yes | No |
| Personal | None | None | User | User | No | No |

## Development

```bash
cargo build --release --features full
cargo test
cargo clippy
```

## License

Proprietary / Closed Source