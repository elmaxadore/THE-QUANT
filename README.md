# THE QUANT — Autonomous Quantitative Trading Platform v2.1

A self-contained, autonomous quantitative trading company written entirely in **Rust**. Deploys on Linux VPS, communicates with MetaTrader 5 via ZeroMQ bridge, persists state to PostgreSQL (TimescaleDB), and runs an endless evolution loop where every trade informs the next training cycle.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   THE QUANT — RUST CORE                      │
├───────────────┬──────────────┬──────────────┬───────────────┤
│  Data         │   Trade      │  Portfolio   │    Risk       │
│  Collector    │   Executor   │  Manager     │    Engine     │
├───────────────┴──────────────┴──────────────┴───────────────┤
│                   Tokio Message Bus (bounded)                │
├─────────────────────────────────────────────────────────────┤
│  Strategy Engine  │  Evolution Engine  │  Model Manager     │
├─────────────────────────────────────────────────────────────┤
│  Security & Auth  │  CLI/TUI (ratatui) │  GitHub Sync       │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

- **Memory Safety**: Compile-time guarantees, zero-allocation hot paths, bounded channels, memory pools
- **Speed**: Lock-free data structures, SIMD-accelerated feature engineering, async I/O
- **Percentage-Scaled**: Auto-detects RAM/CPU at boot, allocates resources as percentages of available RAM
- **Self-Healing**: Circuit breakers, graceful degradation, emergency memory reduction protocol

## Quick Start

### Prerequisites
- Linux VPS (Ubuntu 22.04/24.04 LTS, x86_64)
- Rust 1.78+
- PostgreSQL 16 + TimescaleDB
- MetaTrader 5 (local Wine or remote)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/the-quant.git
cd the-quant

# Build release binary
cargo build --release

# Run bootstrap (first-time setup)
./target/release/the-quant bootstrap

# Start trading daemon
./target/release/the-quant daemon

# Launch TUI (optional GUI)
./target/release/the-quant tui
```

### VPS Deployment (one-command)

```bash
# On a fresh VPS
./deploy/deploy_vps.sh --install-rust
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `the-quant daemon` | Start the trading daemon (headless) |
| `the-quant tui` | Launch terminal UI dashboard |
| `the-quant bootstrap` | First-time setup wizard |
| `the-quant health` | System health check |
| `the-quant version` | Version information |

## Project Structure

```
the-quant/
├── Cargo.toml              # Dependencies & build config
├── config/
│   ├── system.toml         # Global settings (RAM %, symbols, etc.)
│   └── rules/              # Per-account rule YAMLs
├── deploy/
│   ├── the-quant.service   # systemd unit
│   └── deploy_vps.sh       # One-command VPS setup
├── migrations/
│   └── 001_init.sql        # Database schema
├── src/
│   ├── main.rs             # CLI entry point
│   ├── lib.rs              # Library root
│   ├── bootstrap.rs        # First-run installer
│   ├── config.rs           # Configuration management
│   ├── error.rs            # Error types
│   ├── memory.rs           # Percentage-scaled memory manager
│   ├── account/            # Account management & rules engine
│   ├── data/               # Data pipeline (ZMQ, OHLCV, features)
│   ├── regime/             # Market regime detection (GMM)
│   ├── model/              # Model manager (GBDT, volatility, GP)
│   ├── strategy/           # Strategy engine & signal generation
│   ├── lab/                # Strategy laboratory (genetic programming)
│   ├── risk/               # Risk management & circuit breakers
│   ├── execution/          # Order execution & trade journal
│   ├── evolution/          # Evolution loop (7-phase cycle)
│   ├── github/             # GitHub synchronization
│   ├── security/           # Auth, encryption, session management
│   └── tui/                # Terminal UI (ratatui + crossterm)
├── data/                   # Time-series data (Parquet/Arrow)
├── models/                 # Trained model artifacts
├── strategies/             # Strategy configurations
└── logs/                   # Trade journals & system logs
```

## Memory Scaling

The Quant auto-detects total system RAM at boot and computes HARD_PROCESS_LIMIT = TOTAL_RAM × 85%. All module budgets, channel capacities, and buffer sizes scale as percentages:

| Module | 4GB VPS | 8GB VPS | 16GB VPS | 32GB VPS |
|--------|---------|---------|----------|----------|
| DataCollector | 850 MB | 1.7 GB | 3.4 GB | 6.8 GB |
| FeaturePipeline | 408 MB | 816 MB | 1.6 GB | 3.2 GB |
| ModelManager | 510 MB | 1.0 GB | 2.0 GB | 4.0 GB |
| Lab (elastic) | 850 MB | 1.7 GB | 3.4 GB | 6.8 GB |
| **Total Limit** | **3.4 GB** | **6.8 GB** | **13.6 GB** | **27.2 GB** |

## Tech Stack

- **Language**: Rust (edition 2021, MSRV 1.78+)
- **Async Runtime**: tokio (multi-threaded, io_uring)
- **Database**: PostgreSQL 16 + TimescaleDB
- **Message Bus**: ZeroMQ (MQL5 bridge), crossbeam channels (internal)
- **Feature Engineering**: Custom SIMD pipeline, rayon data parallelism
- **Machine Learning**: GMM (regime detection), GBDT (directional), GARCH (volatility)
- **UI**: ratatui + crossterm (terminal), optional CLI mode
- **Security**: Argon2id, AES-256-GCM, JWT sessions

## License

MIT
