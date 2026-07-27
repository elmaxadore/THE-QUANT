# THE QUANT — Implementation Plan

## Phase 1: Project Scaffold
- [ ] Initialize Rust project (`cargo init`)
- [ ] Create `Cargo.toml` with all dependencies from spec
- [ ] Create directory structure matching spec (src/security/, src/account/, src/data/, etc.)
- [ ] Create `config/system.toml` with default configuration
- [ ] Create systemd service file (`deploy/the-quant.service`)
- [ ] Create deploy script (`deploy/deploy_vps.sh`)

## Phase 2: Core Infrastructure
- [ ] Implement `src/lib.rs` — Main library root with public API
- [ ] Implement `src/config.rs` — Configuration loading from TOML/env
- [ ] Implement `src/error.rs` — Custom error hierarchy (QuantError)
- [ ] Implement memory management module — Percentage-scaled allocator, tracking, budgets
- [ ] Implement `src/main.rs` — CLI entry point with clap subcommands

## Phase 3: Security & Authentication (Layer 1)
- [ ] `src/security/mod.rs` — Argon2id hashing, AES-256-GCM encryption, JWT sessions
- [ ] Vault file format implementation

## Phase 4: Account Management (Layer 2)
- [ ] `src/account/mod.rs` — Account struct, AccountRules, state machine
- [ ] Rule engine with real-time monitoring

## Phase 5: Data Pipeline (Layer 3)
- [ ] `src/data/mod.rs` — MQL5 bridge (ZeroMQ), OHLCV builder, feature engineering
- [ ] SIMD-accelerated feature computation
- [ ] ZeroMQ protocol handler

## Phase 6: Regime Detection (Layer 4)
- [ ] `src/regime/mod.rs` — GMM, regime taxonomy, regime-conditioned routing

## Phase 7: Model Manager (Layer 5)
- [ ] `src/model/mod.rs` — Model zoo, training pipeline, hot-swap, manifest

## Phase 8: Strategy Engine & Lab (Layer 6)
- [ ] `src/strategy/mod.rs` — Signal generation, strategy representation
- [ ] `src/lab/mod.rs` — Genetic programming, backtesting, scoring

## Phase 9: Risk & Execution (Layer 7)
- [ ] `src/risk/mod.rs` — Position sizing, circuit breakers, correlation exposure
- [ ] `src/execution/mod.rs` — Order management, slippage tracking, trade journal

## Phase 10: Evolution Loop (Layer 8)
- [ ] `src/evolution/mod.rs` — The heartbeat cycle (7 phases)

## Phase 11: TUI Interface (Layer 9)
- [ ] `src/tui/mod.rs` — ratatui dashboard, command palette, memory dashboard

## Phase 12: GitHub Integration
- [ ] `src/github/mod.rs` — Auto-commit, repo sync, structured storage

## Phase 13: PostgreSQL Schema & Migrations
- [ ] `migrations/` — SQL scripts for all tables
- [ ] Database initialization

## Phase 14: Anti-Overfitting & Testing
- [ ] Anti-overfitting protocol implementation
- [ ] Test suite (unit, integration, property-based)

## Phase 15: Documentation & Polish
- [ ] README.md
- [ ] Architecture Decision Records
- [ ] Doc comments on all public items
