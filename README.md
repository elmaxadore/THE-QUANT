# THE QUANT — v4.0 "Hercules" Hybrid Edition

<p align="center">
  <b>One command. Any machine. A self-scaling, self-updating quant desk.</b>
</p>

---

**THE QUANT** is an autonomous quantitative trading system written in Rust with a
**hybrid ML bridge**: models are trained in Python (offline), exported to ONNX,
and served inside the Rust core via ONNX Runtime — sub-2 ms inference, with
Python *never* in the live hot path.

It is engineered around four ideas:

| Idea | What it means |
|---|---|
| **Scales with the machine** | One binary runs lean on a 4 GB VPS and aggressive on a 64 GB workstation. RAM/CPU is detected at boot and every module gets a percentage budget. |
| **Self-updating** | A daemon checks the git repo **every 24 h**, builds new code in an isolated staging worktree, smoke-tests it, and blue-green swaps the binary. Any failure leaves the running system untouched. |
| **Safety first** | Drawdown hard-stops, daily loss limits, per-trade risk caps, and the prop-firm *Guardian Shield* (per-trade loss circuit breaker) gate every entry. |
| **Hybrid by design** | Rust for the low-latency core; Python only for offline training; ONNX as the bridge. |

---

## 🚀 One-line deployment

```bash
curl -fsSL https://raw.githubusercontent.com/elmaxadore/THE-QUANT/main/deploy/install.sh | bash
```

That single command **auto-detects your machine tier** and does everything:

1. Detects RAM/CPU → picks the matching **tier profile** (see table below)
2. Installs prerequisites and the Rust toolchain if missing
3. Clones the repo to `/opt/the-quant` (or fast-forwards an existing install)
4. Builds the release binary **with the feature set your tier deserves**
5. Creates a non-root `quant` service user + tier-aware systemd unit
6. Runs `the-quant restore`, starts the daemon, configures the firewall
7. Prints next steps

The script is **idempotent** — run it again any time to update + rebuild in place.

**Overrides** (optional):

```bash
FEATURES="web,tui" INSTALL_DIR=/srv/quant bash <(curl -fsSL .../install.sh)
```

> No sudo password prompts mid-script: it uses `sudo` internally only where
> needed, and never asks interactive questions.

### What gets built per tier

| Tier | RAM | Name | Features built | Runtime profile |
|---|---|---|---|---|
| Tier-1 | ≤ 8 GB | **LEAN** | default (crypto) | Paper trading, rule model, no web/TUI, minimal caches |
| Tier-2 | 8–16 GB | **STANDARD** | `web,tui` | + Axum dashboard, ratatui TUI, bigger caches |
| Tier-3 | 16–32 GB | **ADVANCED** | `web,tui` | + larger Lab budgets (RL training opt-in) |
| Tier-4 | > 32 GB | **AGGRESSIVE** | `web,tui` | Full research stack |

The chosen profile is saved to `/opt/the-quant/.build-features`, so every
rebuild — including the 24 h auto-updater — reproduces the same binary.

The same tier logic lives inside the binary (`src/resource.rs`): at boot it
detects the host and budgets each module as a **percentage** of available RAM
(DataCollector 25 %, FeaturePipeline 12 %, Lab 22 %, …), so runtimes scale even
if you rebuild with different features.

---

## ⚡ Quick start

```bash
the-quant --status          # machine tier, budgets, account, update state
the-quant paper 20000       # 20k-bar paper trade with a live P&L report
the-quant --smoke           # self-check (also used by the auto-updater)
```

A paper trade prints a coloured end-of-run report: bars processed, trades,
signals, risk denials, net P&L, return, final equity and the final market
regime — then writes the full trade journal to `state/`.

**First-run security setup** (optional, for storing broker/API credentials):

```bash
the-quant vault init        # create the Argon2id + AES-256-GCM vault
the-quant vault check       # verify the master password
the-quant vault status
```

The vault file lives at `state/vault.enc` and is **git-ignored** — it never
leaves the machine.

---

## 🧭 CLI reference

| Command | Purpose |
|---|---|
| `the-quant paper [N]` | Run an N-bar paper-trading simulation (default 10 000) |
| `the-quant --status` | Show tier, per-module memory budgets, account + update state |
| `the-quant --smoke` | Self-check used by the auto-updater before a swap |
| `the-quant update --now` | Force an auto-update check right now |
| `the-quant backup` | Commit + push state to `origin` |
| `the-quant restore` | Recreate state directories on a fresh machine |
| `the-quant vault init/check/status` | Manage the encrypted credential vault |
| `the-quant daemon` | Run the 24 h auto-update scheduler (used by systemd) |

---

## 🔄 Auto-update: how "updates never hurt"

`src/update.rs` implements **blue-green deployment for the binary itself**:

```
every 24 h:
  git fetch origin (read-only)          # live tree is NEVER rebased
  local HEAD != remote HEAD?
    ├─ no  → sleep, try again tomorrow
    └─ yes → snapshot state → git commit
             git worktree add --detach <staging> origin/main
             cargo build --release (+ saved feature profile)
             ./the-quant --smoke        # gate: binary must boot
             mv current → current.prev  # keep rollback path
             mv staged → current        # promote
             git merge --ff-only origin/main
```

* The **live working tree is never mutated** by a rebase — new code is built in
  a detached staging worktree.
* The **running binary is only replaced after a successful smoke test**.
* On **any failure** the old binary, the working tree and the state are left
  untouched; the next check retries.
* Rebuilds reuse `.build-features`, so an updated binary keeps the same
  capabilities as the one it replaces.

Manual equivalent: `deploy/update.sh`. Rollback: restore
`/usr/local/bin/the-quant.prev`.

---

## 🏗️ Architecture

```
                    ┌────────────────────────────────────────────┐
                    │            PYTHON  (offline only)          │
                    │  python/train/train_gbdt.py                │
                    │  • GBDT / MLP / (PPO later) training       │
                    │  • exports  models/latest.onnx             │
                    └──────────────────┬─────────────────────────┘
                                       │  ONNX artifact (build time)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        RUST CORE  (the live hot path)                    │
│                                                                          │
│  SimFeed/MT5 ─▶ features ─▶ regime ─▶ strategy ─▶ risk ─▶ execution      │
│      (bars)      (12 feats)  (4 regimes)  (signal+conf)  (gates)  (lots) │
│                                                                          │
│  ResourceProfile ─ percentage budgets for every module                   │
│  AutoUpdater ───── 24 h blue-green self-update (staging worktree)        │
│  Vault ─────────── Argon2id + AES-256-GCM credential store               │
│  Store/Restorer ── state journal; `restore` reconstructs on fresh box    │
│  TUI / Axum web ── dashboards (tier-dependent)                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**Per-bar pipeline** (`src/engine.rs`):
feed → `compute_features` (12 features: log-returns, realised vol, RSI-14,
ATR-14, EMA spread, Bollinger width, volume z-score, Hurst proxy, ROC) →
`RegimeDetector` (TrendingUp / TrendingDown / Ranging / HighVolatility) →
`StrategyEngine` (regime-shaped model signal) → `RiskEngine::pre_flight`
(drawdown cap, daily loss cap, per-trade risk, shield) → `ExecutionEngine`
(lot-sized fills with stop-loss / take-profit management).

**Position sizing** is coherent end-to-end: lots are sized so that the
worst-case loss at the stop equals the configured risk per trade:

```
lots = (equity × risk_pct) / (stop_distance × CONTRACT_SIZE)
```

with `CONTRACT_SIZE = 100 000`. Stop-loss and take-profit are enforced both by

---

## 🧠 Hybrid ML (optional)

The system runs out of the box on an interpretable pure-Rust **rule model**.
To swap in a trained model:

```bash
pip install -r python/requirements.txt
python3 python/train/train_gbdt.py     # trains GBDT → models/latest.onnx
cargo build --release --features ml    # ONNX Runtime backend (ort crate)
./target/release/the-quant paper       # picks up models/latest.onnx if present
```

* Training happens **offline** in the evolution lab (Tier-3+ machines can run
  the research server: `python3 python/server/train_server.py`).
* Live inference is a distilled ONNX graph — target **< 2 ms** per call.
* If the model file is missing or fails to load, the core **falls back to the
  rule model** — a broken model can never take the system down.

---

## 🔐 Security

* **Vault**: single master password → Argon2id key derivation → AES-256-GCM
  encryption of broker credentials. Salt + verifier stored in
  `state/vault.enc` (git-ignored). Wrong password → constant-time rejection.
* **Non-root service user**: the daemon runs as `quant`, never root.
* **Secrets out of git**: `state/**` and `models/*` are git-ignored; only
  code, config templates and docs are versioned.
* **systemd ceilings**: `MemoryMax=80%`, `MemoryHigh=70%`, `CPUQuota=75%` —
  percentages, so the unit file self-scales with the machine.


---

## ⚙️ Configuration reference (`config/system.toml`)

| Key | Default | Meaning |
|---|---|---|
| `system.update_interval_hours` | `24` | Auto-update check cadence |
| `system.repo_remote` | GitHub URL | Where updates come from |
| `system.symbols` | `["EURUSD"]` | Traded symbols (sim uses the first) |
| `account.type` | `PERSONAL` | `PERSONAL` / `PROP_EVALUATION` / `PROP_FUNDED` |
| `account.max_drawdown_pct` | `5.0` | Hard drawdown stop — account halts |
| `account.daily_loss_limit_pct` | `3.0` | Daily loss circuit breaker |
| `account.risk_per_trade_pct` | `1.0` | Risk per trade (% of equity) |
| `update.require_smoke_test` | `true` | Gate every update on `--smoke` |
| `ml.enabled` / `ml.model_path` | `false` / `models/latest.onnx` | ONNX model switch |


---

## 📁 Project layout

```
src/              Rust core
  resource.rs       tier detection + %-of-RAM module budgets
  engine.rs         per-bar trading pipeline (paper loop)
  risk.rs           pre-flight risk gates + Guardian Shield
  execution.rs      lot sizing, fills, SL/TP, mark-to-market
  onnx.rs           model backends: RuleModel (default) / OrtModel (feature ml)
  update.rs         24 h blue-green auto-updater
  github.rs         git sync (fetch / ff-only merge / commit+push)
  security.rs       Argon2id + AES-256-GCM vault
  state.rs          state journal + restore
  …                 config, features, regime, strategy, simfeed, tui, web, util
python/
  train/            offline GBDT trainer → models/*.onnx
  server/           offline research server (Tier-3+)
  requirements.txt
config/
  system.toml       runtime configuration
  templates/        per-firm prop-firm rule packs (YAML)
  extraction/       payout-extraction strategies
deploy/
  install.sh        the one-line tier-aware installer
  update.sh         manual blue-green update helper
  the-quant.service tier-aware systemd unit
migrations/       PostgreSQL / TimescaleDB schema (feature db)
models/           ONNX artifacts (git-ignored; produced by python/train)
state/            runtime state (git-ignored; recreated by `the-quant restore`)
```

Prop-firm rule packs live in `config/templates/` (FTMO, The5ers, Blue
Guardian, AquaFunded, Goat Funded, MyForexFunds, True Forex Funds, personal)
with per-variant drawdown / target / shield / payout rules; payout-extraction
strategies live in `config/extraction/`.


---

## 🧪 Development

```bash
cargo test                    # unit tests: tiers, vault, feed, features,
                              # strategy, risk sizing + drawdown gates
cargo build --release         # lean build (Tier-1 friendly)
cargo build --release --features ml,web,tui   # full profile
```

Tests cover the pieces that must never regress: tier percentages sum to 100,
tier boundaries gate Lab/RL, vault round-trip + wrong-password rejection,
deterministic seeded feed, feature-vector width/bounds, signal generation,
drawdown-cap denial, and risk-coherent position sizing.

Bash scripts are syntax-checked with `bash -n deploy/*.sh`; Python trainer /
server with `python3 -m py_compile`.

---

## ❓ FAQ

**Does it trade real money?**
Not by default. The shipped loop is a paper simulation against a seeded feed.
Live trading requires the MT5 ZeroMQ bridge (v2.1 spec) — stay in paper mode
until everything is validated.

**What happens if the auto-updater pulls broken code?**
The build fails or the smoke test fails → **no swap**. The old binary keeps
running and the next 24 h check retries. Manual rollback:
`cp /usr/local/bin/the-quant.prev /usr/local/bin/the-quant`.

**Where is my state?**
`state/` on the machine running the daemon. It is deliberately git-ignored;
`the-quant restore` recreates the layout, and `the-quant backup` snapshots it
to git whenever you want it versioned.

**Can I run it on a tiny VPS?**
Yes — Tier-1 (≤ 8 GB) gets a lean build with minimal budgets. The installer
picks that profile automatically.

**How do I change the feature profile later?**
Edit `/opt/the-quant/.build-features` (e.g. `web,tui`) and run
`deploy/update.sh` — or just re-run the installer.

---

## ⚠️ Risks & disclaimer

This is trading software. Run in paper/simulation first, verify every prop-firm
template against the firm's *current* rules, and never risk capital you cannot
afford to lose. The author takes no responsibility for capital losses.