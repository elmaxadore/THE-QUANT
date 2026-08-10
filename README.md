# THE QUANT — Autonomous Quantitative Trading Platform v3.1 "Hephaestus"

A self-contained, autonomous quantitative trading company and prop-firm extraction engine written entirely in **Rust**. Deploys on Linux VPS, communicates with MetaTrader 5 via ZeroMQ bridge, persists state to PostgreSQL (TimescaleDB), and runs an endless evolution loop where every trade informs the next training cycle.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE QUANT — SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  DEPLOYMENT TARGET: Linux VPS (Ubuntu 22.04/24.04 LTS, x86_64)              │
│  RESOURCE MODEL: Percentage-scaled — auto-detects RAM/CPU at boot           │
│  RUST EDITION: 2021 | MSRV: 1.78+ | PROFILE: release with lto = "fat"       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         MQL5 BRIDGE EA                              │    │
│  │  • Publishes tick/bar/account/position data over ZeroMQ (bincode)   │    │
│  │  • Subscribes to trade commands: MARKET, LIMIT, STOP, CLOSE         │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │ ZeroMQ TCP/IPC                            │
│  ┌──────────────────────────────▼──────────────────────────────────────┐    │
│  │                  RUST CORE — THE QUANT DAEMON                       │    │
│  │                                                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐   │    │
│  │  │   DATA      │  │   TRADE     │  │  PORTFOLIO  │  │   RISK    │   │    │
│  │  │ COLLECTOR   │  │  EXECUTOR   │  │   MANAGER   │  │  ENGINE   │   │    │
│  │  │             │  │             │  │             │  │           │   │    │
│  │  │ • Tick /    │  │ • Pre-flight│  │ • Correl.   │  │ • Sizing  │   │    │
│  │  │   Bar stream│  │   pre-check │  │   exposure  │  │ • Shield  │   │    │
│  │  │ • Micro-    │  │ • Slippage  │  │ • Payout    │  │ • Consistency│  │
│  │  │   structure │  │   model     │  │   curves    │  │ • CVaR    │   │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘   │    │
│  │         │                │                │               │         │    │
│  │         └────────────────┴────────────────┴───────────────┘         │    │
│  │                              │                                      │    │
│  │  ┌───────────────────────────▼───────────────────────────────────┐  │    │
│  │  │            BOUNDED TOKIO & CROSSBEAM MESSAGE BUS              │  │    │
│  │  └───────────────────────────┬───────────────────────────────────┘  │    │
│  │                              │                                      │    │
│  │  ┌───────────────────────────▼───────────────────────────────────┐  │    │
│  │  │               ML TRINITY & STRATEGY ENGINE                    │  │    │
│  │  │  • Unsupervised: GMM (8 regimes) + BOCD + Isolation Forest    │  │    │
│  │  │  • Supervised: GBDT + Small Neural Net Ensemble + Attention   │  │    │
│  │  │  • Reinforcement: QuantGym + PPO + Distilled Policy (<5MB)    │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │           ACCOUNT PIPELINE & ROTATION MANAGER (v3.1)           │  │    │
│  │  │  • Lifecycle: Acquired → Warming → Trading → Extract → Retired  │  │    │
│  │  │  • Consistency Guardian (15% concentration, lot CV, frequency)│  │    │
│  │  │  • Guardian Shield (per-trade loss limit & strike counting)   │  │    │
│  │  │  • $25/day Extraction Optimization & Sigmoidal Equity Curves  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │               HYBRID INTERFACE & API LAYER                    │  │    │
│  │  │  • Web Dashboard (Axum HTTP/SSE/WS on localhost:8080)         │  │    │
│  │  │  • Terminal UI (Ratatui + Crossterm)                          │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                 │ sqlx / tokio-postgres                     │
│  ┌──────────────────────────────▼──────────────────────────────────────┐    │
│  │                    POSTGRESQL 16 + TIMESCALEDB                      │    │
│  │  • 22 Hypertables & relational schemas for ticks, OHLCV, features,  │    │
│  │    trades, regimes, RL policies, consistency, shield, & pipeline   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Three Specifications in One System

### 1. Rust System Engineering Base (v2.1 Spec)
- **Percentage-Scaled Memory Contract**: Detects total RAM at boot and computes `HARD_PROCESS_LIMIT = TOTAL_RAM × 85%`.
- **Zero-Allocation Hot Paths**: Ring buffers for tick streams, stack-allocated feature vectors, lock-free crossbeam channels.
- **Endless Evolution Loop**: 7-phase background cycle (Data update → Regime recalibration → Model retraining → Lab batch → Strategy pool update → GitHub commit → Resume trading).
- **Anti-Overfitting Gauntlet**: Temporal split only, Walk-Forward Analysis (WFA efficiency > 0.6), Monte Carlo 10,000 iterations, noise tests, parameter sensitivity checks.

### 2. "Prometheus" Architecture (v3.0 Spec)
- **One-Click Installation**: Idempotent `install.sh` script provisions Linux dependencies (PostgreSQL 16, TimescaleDB, Wine, Rust), builds release binary, and configures systemd.
- **Hybrid Interface**: Concurrent Ratatui CLI and Axum Web UI dashboard (REST, SSE, WebSockets) with real-time equity curves, regime rivers, and memory breakdown.
- **The ML Trinity**:
  - *Unsupervised*: GMM 8-regime detector, Bayesian Online Changepoint Detection (BOCD), Isolation Forest anomaly filter.
  - *Supervised*: LightGBM GBDT + Small Neural Network MLP ensemble + Temporal Attention.
  - *Reinforcement Learning*: QuantGym simulation environment, PPO offline training, distilled policy (<5MB, <2ms inference), multi-agent capital allocator.
- **Microstructure Intelligence**: Tick-level order flow features, adaptive slippage model, optimal execution (TWAP/VWAP/RL-split).
- **Math Upgrades**: Causal feature selection (PC algorithm), Kelly criterion with parameter uncertainty, 95% CVaR risk budgeting, TPE Bayesian optimization.

### 3. "Hephaestus" Survival & Extraction Engine (v3.1 Spec)
- **Consistency Rule Engine**: Enforces prop-firm consistency rules (e.g. Blue Guardian 15% max single-day profit concentration, lot size CV, trade frequency, min hold duration) with automatic position sizing caps.
- **Guardian Shield**: Per-trade loss circuit breaker with strike counting (e.g. $50 max loss per trade, strike counting, gap exception handling, 4h cooldown).
- **Payout Cap & Extraction Optimization**: Lifetime payout cap tracking, extraction velocity modeling, sigmoid target equity curves, safe daily profit target calculation.
- **Account Pipeline & Rotation System**: Manages accounts as fungible assets (`Acquired → Warming → Trading → Extracting → PayoutPending → Retired / Blown`). Auto-rotates active account on payout cap reach or blown status.
- **Prop-Firm Template Library**: Pre-configured YAML templates for Blue Guardian Instant 5K, AquaFunded Instant, The5ers High Stakes, FTMO Evaluation, and Goat Funded Trader.

---

## Quick Start & Installation

### Single-Command VPS Setup (v3.0)

```bash
curl -fsSL https://raw.githubusercontent.com/TheQuantCompany/the-quant/main/install.sh | bash
```

### Local Build & Execution

```bash
# 1. Clone repository
git clone https://github.com/TheQuantCompany/the-quant.git
cd the-quant

# 2. Build release binary with full feature set
cargo build --release --features full

# 3. First-run bootstrap setup
./target/release/the-quant bootstrap

# 4. Start the trading daemon
./target/release/the-quant daemon

# 5. Launch Terminal UI Dashboard
./target/release/the-quant tui
```

### Utility Commands

```bash
# Diagnostics & subsystem repair
the-quant doctor

# Reset options
the-quant reset --soft    # Clear caches, preserve data
the-quant reset --hard    # Purge database & local state
```

---

## Prop-Firm Account Templates

Pre-configured YAML rules live in `config/templates/`:

- `blue_guardian_instant_5k.yaml`: 10% max DD, 15% consistency, $50 Guardian Shield, $250 payout cap.
- `aquafunded_instant.yaml`: 8% max DD, 3% daily loss, 20% consistency, $60 Guardian Shield.
- `the5ers_high_stakes.yaml`: 10% max DD, 5% daily loss, 8% profit target, 3 min trading days.
- `ftmo_evaluation.yaml`: 10% max DD, 5% daily loss, 10% profit target, 4 min trading days.
- `goat_funded_trader.yaml`: 10% max DD, 8% profit target, no daily limit.

Extraction strategy configurations live in `config/extraction/`:
- `blue_guardian_instant_5k.yaml`: $25/day extraction target, 10-day projected trajectory to $250 payout cap.

---

## Percentage-Scaled Memory Contract

The Quant auto-detects system RAM at boot (`HARD_PROCESS_LIMIT = TOTAL_RAM × 85%`) and allocates module budgets as percentages:

| Module | 4GB VPS (Lean) | 8GB VPS | 16GB VPS | 32GB Workstation |
|---|---|---|---|---|
| **HARD_PROCESS_LIMIT** | **3.4 GB** | **6.8 GB** | **13.6 GB** | **27.2 GB** |
| DataCollector (25%) | 850 MB | 1.7 GB | 3.4 GB | 6.8 GB |
| FeaturePipeline (12%) | 408 MB | 816 MB | 1.6 GB | 3.2 GB |
| ModelManager (15%) | 510 MB | 1.0 GB | 2.0 GB | 4.0 GB |
| StrategyEngine (6%) | 204 MB | 408 MB | 816 MB | 1.6 GB |
| RiskEngine (3%) | 102 MB | 204 MB | 408 MB | 816 MB |
| RL / Gym (5%) | 170 MB | 340 MB | 680 MB | 1.3 GB |
| Lab (elastic) (22%) | 748 MB | 1.5 GB | 3.0 GB | 6.0 GB |
| TUI + Web UI (4%) | 136 MB | 272 MB | 544 MB | 1.08 GB |
| API / Channels (2%) | 68 MB | 136 MB | 272 MB | 544 MB |
| System Overhead (6%) | 204 MB | 408 MB | 816 MB | 1.6 GB |
| Reserve (5%) | 170 MB | 340 MB | 680 MB | 1.3 GB |

---

## Repository Structure

```
the-quant/
├── Cargo.toml                  # Cargo manifest & dependencies
├── install.sh                  # One-command VPS installer script
├── README.md                   # System documentation
├── config/
│   ├── system.toml             # Global configuration
│   ├── rules/                  # Active account rule YAMLs
│   ├── templates/              # Built-in prop firm rule templates
│   └── extraction/             # Extraction strategy configurations ($25/day model)
├── deploy/
│   ├── the-quant.service       # systemd unit configuration
│   └── deploy_vps.sh           # VPS deployment script
├── migrations/
│   ├── 001_init.sql            # Base schema (tables 1-10)
│   ├── 002_v3.sql              # Prometheus ML & microstructure schema (tables 11-16)
│   └── 003_v3_1.sql            # Hephaestus consistency, shield & pipeline schema (tables 17-22)
└── src/
    ├── main.rs                 # CLI entry point & command palette
    ├── lib.rs                  # Library entry point & QuantApp struct
    ├── bootstrap.rs            # First-run onboarding wizard & system validation
    ├── config.rs               # Hierarchical configuration parser
    ├── error.rs                # Typed QuantError definitions
    ├── memory.rs               # Percentage-scaled memory manager & allocators
    ├── account/                # Account management engine
    │   ├── mod.rs              # Account state machine & manager
    │   ├── consistency.rs      # Consistency Rule Engine (profit concentration, lot CV)
    │   ├── shield.rs           # Guardian Shield per-trade loss circuit breaker
    │   └── payout.rs           # Payout cap & sigmoid extraction curve engine
    ├── pipeline/               # Account pipeline & rotation manager
    │   ├── mod.rs
    │   └── manager.rs          # Readiness scoring & auto-rotation engine
    ├── data/                   # Data collector, ZMQ stream, OHLCV ring buffers
    ├── regime/                 # GMM market regime detection
    ├── model/                  # Model manager, GBDT, MLP, Attention, Ensemble
    ├── strategy/               # Strategy engine & debouncing
    ├── lab/                    # Strategy laboratory, genetic programming, backtester
    ├── risk/                   # Risk engine, pre-flight checks, position sizing
    ├── execution/              # Order executor, idempotency, slippage model
    ├── evolution/              # Evolution engine (7-phase loop)
    ├── github/                 # Automated GitHub repository synchronization
    ├── security/               # Argon2id, AES-256-GCM vault, JWT sessions
    ├── tui/                    # Terminal UI dashboard (Ratatui)
    ├── rl/                     # Reinforcement Learning (QuantGym, PPO, Distilled Policy)
    ├── api/                    # Axum REST, SSE, WebSocket API server
    ├── web_ui/                 # Embedded Web UI static assets
    ├── microstructure/         # Tick-level order flow features & execution opt
    ├── anomaly/                # Isolation Forest anomaly filter
    ├── changepoint/            # Bayesian Online Changepoint Detection (BOCD)
    ├── math/                   # Causal selection, Kelly with uncertainty, CVaR, TPE
    └── wizard/                 # Web-based onboarding wizard
```

---

## License

MIT License. Copyright (c) 2026 The Quant Team.
