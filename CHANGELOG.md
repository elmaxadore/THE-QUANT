# Changelog

All notable changes to THE QUANT will be documented in this file.

## [4.0.0] - 2026-08-13 — "Hercules"

### Added (Resilience Through Updatability)
- **Git-centric state architecture** — `src/state/` module with:
  - `StateManifest` — master state manifest with blake3 checksums (`state/manifest.json`)
  - `StateStore` — per-account state serialization (account.toml, lifecycle.toml, consistency.toml, shield.toml, payout.toml, trades/, equity/, evolution/)
  - `RestoreEngine` — full system reconstruction from `git clone + restore`
- **Auto-commit engine** — `src/github/auto_commit.rs`:
  - Event-driven commit queueing with `[ACCOUNT:uuid] ACTION: description` format
  - Trade/Evolution/Model/AccountState/MarketData/StrategyChange/System triggers
  - Background task with rate-limited draining
- **GitHub push support** — `src/github/push.rs`:
  - Authenticated push-to-remote via git2 credential callbacks
  - Fetch support for update checks
- **Blue-Green Update Engine** — `src/update.rs`:
  - `check` / `apply` / `rollback` lifecycle
  - Git pull → build → warm-up → verify-sync → atomic-handoff phases
  - GitHub Releases API integration
- **Multi-Account Firm Orchestrator** — `src/firm.rs`:
  - `TradingDesk` — isolated strategy pool, model zoo, risk state per account
  - `QuantFirm` — manages all desks, correlation monitoring, shared state store
  - Account type matrix (`AccountTypeProfile`) matching the v4.0 spec §3.5
- **Prop-firm rule templates** — `config/templates/`:
  - blue_guardian.yaml, ftmo.yaml, the5ers.yaml, aqua_funded.yaml, goat_funded.yaml, true_forex_funds.yaml, personal.yaml
- **New CLI commands**:
  - `the-quant restore` — one-command system recovery
  - `the-quant update [--check|--apply|--force]` — blue-green self-update
  - `the-quant doctor` — full system diagnostics
  - `the-quant status` — account overview
  - `the-quant account-add <name> <variant>` — add trading desk
- **Deploy scripts**:
  - `deploy/update.sh` — zero-downtime update workflow
  - `deploy/restore.sh` — 5-minute recovery from git
- **GitHub Actions workflows** — CI, release, deploy
- **.gitattributes** — Git LFS tracking for Parquet and model binaries

### Changed
- Cargo version: 3.0.0 → 4.0.0
- CLI branding: "Prometheus" → "Hercules"

### Fixed
- GitHubSync now pushes to remote (was a TODO)
- Empty commits are skipped (no-op detection)

## [3.1.0] - "Hephaestus"
- Account pipeline manager with lifecycle rotation
- Consistency engine, guardian shield, payout engine

## [3.0.0] - "Prometheus"
- RL engine, Axum HTTP API, web_ui, microstructure, anomaly detection, changepoint
- Wizard onboarding