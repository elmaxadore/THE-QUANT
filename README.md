# THE QUANT — v4.1 "Hercules" Hybrid Edition

<p align="center">
  <b>One command. Any machine. A self-scaling, self-updating quant desk.</b>
</p>

---

**THE QUANT** is an autonomous quantitative trading system written in Rust with a
**hybrid ML bridge**: models are trained in Python (offline), exported to ONNX,
and served inside the Rust core via ONNX Runtime — sub-2 ms inference, with
Python *never* in the live hot path.

It implements the **Aegis hedged-pairs strategy** — a directional-statistical
arbitrage system trained and validated on **real historical tick data**
(Dukascopy free feed, see [ML pipeline](#-ml-pipeline-python-offline)):

> *"Given a universe of correlated FX/metal/index pairs, when a probabilistic
> directional bias is detected on a correlated cluster, construct a hedged
> position: BUY the higher-volatility leg (long gamma) and SELL the
> lower-volatility leg (short hedge), sizing each leg so the NET position
> delta reflects the probability differential while respecting the Guardian
> Shield ($50/trade), $150/day loss, and $250 lifetime loss floor."*

| Idea | What it means |
|---|---|
| **Scales with the machine** | One binary runs lean on a 4 GB VPS and aggressive on a 64 GB workstation. RAM/CPU is detected at boot and every module gets a percentage budget. |
| **Self-updating** | A daemon checks the git repo **every 24 h**, builds new code in an isolated staging worktree, smoke-tests it, and blue-green swaps the binary. Any failure leaves the system untouched. |
| **Multi-account** | Each `[[accounts]]` entry runs as its own `TradingDesk` (isolated Aegis strategy + risk engine + P&L journal). One Tier-1 box runs 2+ tradable accounts with zero cross-contamination. |
| **Safety first** | Guardian Shield (2 strikes → halt), drawdown cap, daily/lifetime loss limits gate **every** entry — enforced at the Rust execution layer, not in strategy code. |
| **State is in git** | Every closed trade is journaled to `state/trades/*.jsonl`; account snapshots live in `state/accounts/*.json`. `the-quant backup` commits + pushes them; `the-quant restore` reconstructs them on any machine. |

---

## Table of contents

1. [Quick start](#-quick-start)
2. [Command reference](#-command-reference)
3. [Building from source](#-building-from-source)
4. [Configuration](#-configuration)
5. [Customisation guide](#%EF%B8%8F-customisation-guide)
6. [The Aegis strategy](#-the-aegis-strategy)
7. [Risk & compliance circuit breakers](#-risk--compliance-circuit-breakers)
8. [State management (git-backed)](#-state-management-git-backed)
9. [Deployment & operations](#-deployment--operations)
10. [ML pipeline (Python, offline)](#-ml-pipeline-python-offline)
11. [Testing](#-testing)
12. [Project layout](#-project-layout)
13. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick start

### One-line deployment (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/elmaxadore/THE-QUANT/main/deploy/install.sh | bash
```

That single command **auto-detects your machine tier** and does everything:

1. Detects RAM/CPU → picks the matching **tier profile** (see table below)
2. Installs prerequisites and the Rust toolchain if missing
3. Clones the repo to `/opt/the-quant` (or fast-forwards an existing install)
4. Builds the release binary **with the feature set your tier deserves**
5. Creates a **Python venv** with the training/data stack, generates starter
   market data and **trains the default rule model** (`models/latest.onnx`)
6. Installs the **MetaTrader 5 terminal** under Wine (headless) plus the
   **bar-export connector EA** (`deploy/mt5_ea.mq5`), and bridges the MT5
   shared-files directory to the config's `mt5_dir` (`/home/quant/mt5/files`)
7. Creates a non-root `quant` service user + tier-aware systemd unit
8. Runs `the-quant restore`, starts the daemon, configures the firewall
9. Prints next steps

After it finishes, all that is left is: **add accounts → collect data →
train → start trading** (the exact commands are printed at the end).

```text
  1) the-quant vault init          # master password for the secrets vault
  2) edit config/system.toml       # add [[accounts]] entries (README §5)
  3) collect data                  # MT5 EA streams live bars, or download history
  4) .venv/bin/python python/train/train_gbdt.py   # (re)train the model
  5) the-quant paper 20000         # paper trade first
  6) data_source = "mt5"           # flip to live when ready
```

The script is **idempotent** — run it again any time to update + rebuild in
place. Optional opt-outs: `INSTALL_PYTHON=0` (skip the Python env / model
training) and `INSTALL_MT5=0` (skip Wine + MetaTrader 5 — e.g. on Windows or
when you only want paper mode).

**Overrides** (optional):

```bash
FEATURES="web,tui" INSTALL_DIR=/srv/quant bash <(curl -fsSL .../install.sh)
```

### Manual build (developers)

```bash
git clone https://github.com/elmaxadore/THE-QUANT.git
cd THE-QUANT
cargo build --release            # lean profile
cargo build --release --features web,tui   # dashboard + TUI
cargo install --path .           # puts `the-quant` on your PATH
the-quant --status
the-quant paper 20000
```

### What gets built per tier

| Tier | RAM | Name | Features built | Runtime profile |
|---|---|---|---|---|
| Tier-1 | ≤ 8 GB | **LEAN** | default (crypto) | Paper trading, rule model, no web/TUI, minimal caches |
| Tier-2 | 8–16 GB | **STANDARD** | `web,tui` | + Axum dashboard, ratatui TUI, bigger caches |
| Tier-3 | 16–32 GB | **ADVANCED** | `web,tui` | + larger Lab budgets (RL training opt-in) |
| Tier-4 | > 32 GB | **AGGRESSIVE** | `web,tui` | Full research stack |

The same tier logic lives inside the binary (`src/resource.rs`): at boot it
detects the host and budgets each module as a **percentage** of available RAM
(DataCollector 25 %, FeaturePipeline 12 %, Lab 22 %, …), so runtimes scale even
if you rebuild with different features.

---

## 📖 Command reference

All commands are sub-commands of the single `the-quant` binary.

### Trading

| Command | Description |
|---|---|
| `the-quant paper [N]` | Run an **N-bar multi-desk paper simulation** (default 10 000 bars). Every `[[accounts]]` entry in the config gets its own isolated desk; each closed trade is journaled under `state/trades/` and an account snapshot is written under `state/accounts/`. Exits non-zero on error. |
| `the-quant --status` | Print system status: detected tier + resource profile, configured accounts, risk parameters, model backend. |
| `the-quant scan [N]` | Inspect the **shared knowledge base** (`state/knowledge.json`): every canonical asset learned across *all* accounts/brokers — observations, EWMA volatility & drift, trade count, win-rate and P&L. `N` limits the rows (default 20). |
| `the-quant --smoke` | Minimal boot sanity check (no trading). Used by the auto-updater and `deploy/update.sh` before a blue-green binary swap. |

### State & vault

| Command | Description |
|---|---|
| `the-quant backup` | Commit everything under `state/` to git and push to `origin` (the *state is in git* golden rule). Prints `nothing to commit` when clean. |
| `the-quant restore` | Recreate the state directory tree (`state/{trades,accounts,equity,models}`) on a fresh machine. Idempotent; run by the installer. |
| `the-quant vault init` | Create the encrypted secrets vault (API keys, MT5 credentials). Prompts for a master password (min 8 chars) with hidden TTY input. |
| `the-quant vault check` | Verify the master password unlocks the vault. |
| `the-quant vault status` | Show whether the vault exists and where. |

### Updates & daemon

| Command | Description |
|---|---|
| `the-quant update --now` | Force an auto-update check immediately (otherwise the daemon does it every 24 h). |
| `the-quant daemon_run` | Start the long-running daemon (24 h update scheduler loop). This is what the systemd unit executes; run it manually only when systemd is unavailable (e.g. containers). |
| `deploy/update.sh` | Manual in-place update: snapshot state → `git pull --rebase` → rebuild with the saved feature profile → smoke-test → blue-green swap → restart service. |

### Typical session

```bash
the-quant --status          # 1. sanity-check config + tier
the-quant paper 20000       # 2. run a 20k-bar paper simulation
the-quant backup            # 3. journal + snapshots are committed & pushed
systemctl status the-quant  # 4. confirm the update daemon is alive
```

---

## 🔨 Building from source

Requirements: **Rust 1.75+** (stable), a C linker, and (optionally) ONNX Runtime
for the `ml` feature and PostgreSQL for `db`.

```bash
cargo build --release                         # default (lean) profile
cargo build --release --features web,tui      # + dashboard & terminal UI
cargo build --release --features web,tui,ml,db  # full research stack

cargo test                                    # run the unit-test suite
cargo fmt                                     # rustfmt (CI-enforced style)
```

The installer saves the chosen feature set to `/opt/the-quant/.build-features`
so the 24 h auto-updater and `deploy/update.sh` always rebuild with the **same
profile**.

---

## ⚙️ Configuration

Everything is configured in **`config/system.toml`** (TOML). Key sections:

### `[system]` — global

| Key | Meaning |
|---|---|
| `symbols` | The tradeable universe (must cover every Aegis pair leg). |
| `state_dir` | Root of the git-backed state tree (default `state`). |
| `primary_timeframe` | Strategy timeframe (e.g. `"M5"` — 288 bars per trading day). |

### `[[accounts]]` — one per prop-firm desk

Each entry spawns an independent `TradingDesk`:

```toml
[[accounts]]
id = "paper-01"
firm = "personal"
initial_balance = 25000.0
```

### `[aegis]` — strategy + risk parameters

| Key | Default | Meaning |
|---|---|---|
| `risk_per_trade_pct` | 1.0 | Base risk per trade (% of current equity). |
| `shield_max_loss_per_trade` | 50.0 | **Guardian Shield** per-trade cap (USD). Sizing is clamped to this. |
| `shield_strike_limit` | 2 | Losing closes before the shield blows and the desk halts. |
| `daily_loss_limit` | 150.0 | Max realised loss per trading day (USD); resets at broker midnight. |
| `lifetime_loss_floor` | 250.0 | Total loss that pauses the desk until *you* intervene. |
| `max_daily_profit` | 100.0 | Daily extraction cap — stop chasing after a great day (consistency rule). |
| `consistency_threshold_pct` | 15.0 | Best day vs total profit ratio that halves sizing. |
| `max_drawdown_pct` | 10.0 | Equity drawdown that halts new entries. |
| `min_correlation` | 0.70 | Minimum Pearson correlation to consider a pair. |
| `min_volatility_ratio` | 1.2 | Long leg must be this much more volatile than the short leg. |
| `min_bias_magnitude` / `min_probability_edge` | — | Directional-bias thresholds before `decide()` can emit `Open`. |
| `max_pairs_active` | 2 | Concurrent hedged pairs per desk. |
| `time_stop_bars` | 96 | Force-close a pair after this many bars (one trading day on M5). |
| `atr_multiplier` | 2.0 | Stop distance in spread-σ units used by the sizer. |

---

## 🎛️ Customisation guide

THE QUANT is built around one principle: **every account is a separate trader,
but knowledge is shared**. You can tune each desk independently — its rules,
its universe, its strategy thresholds — and no combination of settings can
disable a circuit breaker or push a desk past a prop-firm limit (all overrides
pass through safety clamps in `src/config.rs`, enforced by unit tests).

### 5.1 Per-account rule overrides — `[[accounts.rules]`

Every knob in the global `[aegis]` section can be overridden **per account**.
Add a `[accounts.rules]` table inside any `[[accounts]]` block:

```toml
[[accounts]]
id = "ftmo-100k"
firm = "ftmo"
initial_balance = 100000.0
type = "PROP_FUNDED"

[accounts.rules]
daily_loss_limit = 500.0        # FTMO: 5% of 100k
max_drawdown_pct = 10.0         # FTMO overall limit
shield_max_loss_per_trade = 400.0
risk_per_trade_pct = 0.25
min_correlation = 0.80          # stricter pair selection
time_stop_bars = 120
```

Any key you omit is inherited from the global `[aegis]` table. Overridable
keys: `max_drawdown_pct`, `daily_loss_limit`, `lifetime_loss_floor`,
`shield_max_loss_per_trade`, `shield_strike_limit`, `risk_per_trade_pct`,
`consistency_threshold_pct`, `max_daily_profit`, `target_daily_profit`,
`min_correlation`, `min_volatility_ratio`, `min_bias_magnitude`,
`min_probability_edge`, `max_pairs_active`, `time_stop_bars`,
`atr_multiplier`, `primary_timeframe`.

**Safety clamps (cannot be bypassed):**

| Override | Clamp |
|---|---|
| `max_drawdown_pct` | 0.5 % – 50 % |
| `daily_loss_limit` | $10 – $1 000 000 |
| `lifetime_loss_floor` | ≥ $10 |
| `shield_max_loss_per_trade` | $1 – `daily_loss_limit` |
| `shield_strike_limit` | 1 – 10 |
| `risk_per_trade_pct` | 0.01 % – 5 % |
| `max_daily_profit` / `target_daily_profit` | > 0 |
| `min_correlation` | ≤ 1.0 |
| `min_volatility_ratio` | ≥ 1.0 |
| `max_pairs_active` | ≤ 8 |

Clamps only *loosen or tighten within safe bounds* — a breaker can never be
switched off (tested by `config::tests::breakers_can_never_be_disabled_by_override`).

### 5.2 Scan universe — what each desk watches

By default **every desk scans every asset its account can trade**, not just a
watchlist. Three modes per account:

```toml
[[accounts]]
universe_mode = "all_assets"   # (default) everything the feed/broker offers
# universe_mode = "watchlist"  # only [system].symbols
# universe_mode = "custom"     # only the list below
# symbols = ["EURUSD", "GBPJPY", "US30"]      # custom list / extras
# exclude_symbols = ["XAGUSD", "GASOLINE"]    # never scan or trade these
# max_scan_assets = 25         # hard cap per desk (bounds CPU, keeps focus)
# auto_pairs = true            # discover hedged pairs from measured correlation
```

* **`all_assets`** — the watchlist is scanned first (priority), then every
  other tradable asset. This is the "learn from everything" mode: the system
  builds knowledge on assets you never listed.
* **`watchlist`** — classic behaviour; only `[system].symbols`.
* **`custom`** — full manual control via the account's own `symbols` list.

`exclude_symbols` is canonicalised (aliases apply — see 5.4), so excluding
`GOLD` also excludes `XAUUSD`, `XAUUSDm`, etc.

**Dynamic pair discovery (`auto_pairs = true`)** — instead of trading only
the built-in seed universe (xau_xag, gbp_eur, …), each desk measures
correlation across its whole scanned universe and trades the most-correlated
pairs it finds (long leg = the empirically more volatile asset), capped by
`max_pairs_active`. Set `auto_pairs = false` to restrict a desk to the seed
universe only.

### 5.3 Shared knowledge & scan deduplication

All desks write what they learn into **one knowledge base**
(`state/knowledge.json`, git-backed with the rest of `state/`):

```toml
[system]
share_knowledge = true          # one "AI memory" across all accounts
scan_min_interval_bars = 10     # full re-evaluation throttle (per asset/pair)
```

* Each asset entry accumulates observations, EWMA volatility & drift, and the
  cross-account trade record (count, win-rate, P&L). Inspect it with
  `the-quant scan [N]`.
* **No repetitive scanning:** bars are deduped per *canonical asset per bar
  time* — two desks (or two brokers) streaming the same market count once.
  Expensive full re-evaluations are additionally throttled to once per
  `scan_min_interval_bars`; cheap EWMA observation still runs on every bar.
* Set `share_knowledge = false` to give each desk a private memory (e.g. to
  A/B two strategy ideas without cross-contamination).

### 5.4 Cross-broker symbol identity — `[system.symbol_aliases]`

The same market is quoted under different names by different brokers
(`GOLD`, `XAUUSD.x`, `XAUUSDm`, `NAS100`, `USTEC.r` …). THE QUANT normalises
all of them to one **canonical asset id** so knowledge and dedup merge
automatically. A large alias table is built in (GOLD, SILVER, NAS100, USTEC,
NDX, DJ30, DAX, GER30, DE40, FTSE, WTI, XBTUSD, …); broker-specific extras go
in the config:

```toml
[system.symbol_aliases]
"US30.cash" = "US30"
"EURUSD.pro" = "EURUSD"         # example — .pro suffix is already stripped
```

Normalisation also strips common broker suffixes (`.r`, `.raw`, `.pro`,
`.ecn`, `.x`, `.m`, `.c`, `-ecn`, `_m`) automatically. Aliases you add take
precedence over the built-in table.

### 5.5 Adding an account (any firm, any rules)

```toml
[[accounts]]
id = "firm-x-25k"               # unique id (used in journals/state files)
name = "Firm X 25K Phase 1"
firm = "firm_x"                 # matches config/templates/<firm>.yaml
variant = "25k_phase1"
initial_balance = 25000.0
type = "PROP_EVALUATION"        # PERSONAL | PROP_EVALUATION | PROP_FUNDED
strategy = "aegis"
# …then any universe_mode / symbols / rules overrides (see 5.1–5.2)
```

Accounts with conflicting rule-sets (e.g. one firm bans holding over the
weekend, another pays consistency bonuses) simply get different
`[accounts.rules]` tables — the desks never interact except through the
optional shared knowledge base.

### 5.6 Data source & timeframes

```toml
[system]
data_source = "sim"   # "sim" (synthetic) | "csv" (python/data/histdata) | "mt5" (live)
mt5_dir = "/home/quant/mt5/files"
```

* `primary_timeframe` (global or per account, e.g. `M5`, `M15`) drives the
  day-key bucketing for daily resets and the default bar-window horizon.
* `time_stop_bars` is in bars of that timeframe (96 × M5 ≈ one trading day).
* `csv` files in `python/data/histdata/` are **M5 bar CSVs converted locally
  from real Dukascopy tick data** by `python/data/download_dukascopy.py`
  (free, no API key; `.bi5` tick files → tick cache → M5 OHLCV). They are
  generated data, so they are **not** in git.
* Point `data_source` at `mt5` to run the identical stack live; paper and live
  share every risk gate. The one-line installer sets this up end-to-end:
  1. MT5 terminal installed under Wine (headless, `/auto`)
  2. the **connector EA** `deploy/mt5_ea.mq5` copied into `MQL5/Experts/` —
     attach it to any chart, set `InpSymbols = "EURUSD, XAUUSD, …"` to match
     `[system].symbols`; it appends every completed bar to
     `<SYMBOL>.csv` in the terminal's `MQL5/Files`
  3. `/home/quant/mt5/files` (config `mt5_dir`) symlinked to that `Files`
     directory, so `Mt5Feed` tails the same files the EA writes

  On Windows you can skip the Wine step (`INSTALL_MT5=0`) and point `mt5_dir`
  at the terminal's `MQL5\Files` folder directly; attach the same EA.

### 5.7 What you *cannot* break

* Circuit breakers run in the execution path **before every entry** — no
  configuration combination removes them.
* Sizing is always re-derived from the (clamped) risk budget; a bad override
  can make a desk trade *less*, never more than its shield allows.
* `exclude_symbols` desks refuse to even scan those assets; a desk with an
  empty universe simply idles without error.
* Every override is unit-tested (`config::tests::overrides_*`); `cargo test`
  verifies the clamps on every build.

---

## 🧭 The Aegis strategy

`src/aegis.rs` implements the hedged-pairs engine over the seed universe
(see `seed_pairs()`: `xau_xag`, `gbp_eur`, `us100_us30`, `aud_nzd`,
`gbp_aud`) plus any pairs dynamically discovered from the scanned universe
when `auto_pairs = true` (see the customisation guide). Each bar the desk:

1. **Features** — log-return series per leg, Pearson correlation, annualised
   volatility ratio, and the `ln(A)−ln(B)` spread series.
2. **Bias** — `estimate_bias()` blends short-horizon spread momentum with the
   spread z-score (standing in for the `aegis_direction_prob` ONNX graph when
   no model is loaded).
3. **Decision** — `decide()` re-validates every gate (correlation, vol ratio,
   probability edge, concurrent-pair limit) and emits `Open { hedge_ratio,
   bias, prob }` or `Hold`.
4. **Sizing** — `size_legs(risk_budget, price_a, price_b, atr, atr_mult,
   hedge_ratio)` converts the consistency-rule-adjusted risk budget into leg
   units: `notional = budget / stop_distance`, `units_a = notional / price_a`,
   `units_b = (notional / price_b) × hedge_ratio`.
5. **Management** — positions are marked to market every bar and force-closed
   at the time stop or when the pair's loss hits the shield cap.

---

## 🛡️ Risk & compliance circuit breakers

These are **hard gates in the execution path** (`DeskRisk::pre_flight` in
`src/aegis.rs`, called from `TradingDesk::step` before *every* entry), never
advisory:

| Breaker | Trigger | Effect | Resets? |
|---|---|---|---|
| **Guardian Shield** | `shield_strike_limit` losing closes (default 2) | Desk `Blown` — no entries | Never (manual) |
| **Daily loss limit** | Realised day P&L ≤ −$150 | No entries | Next trading day |
| **Lifetime floor** | Total P&L ≤ −$250 | Desk `Paused` | Never (manual) |
| **Drawdown cap** | Equity > 10 % below peak | No entries | Equity recovery |
| **Daily extraction cap** | Day profit ≥ $100 | `ExtractionCap` — stop chasing | Next trading day |
| **Consistency rule** | Best day > ~15 % of total profit | Risk budget halved | Continuously |

Daily counters (`day_pnl`, profit peak, transient halts) reset at **broker
midnight** via `DeskRisk::new_day()`, triggered by the desk's day-key rollover
(epoch-day for live feeds, timeframe-bucketed bar index for sims). Permanent
states (`Blown`, lifetime `Paused`) survive the rollover by design.

---

## 💾 State management (git-backed)

The v4.0 golden rule — **state is in git** — is enforced end-to-end:

```text
state/
├── trades/<account_id>.jsonl   # append-only journal of closed trades
├── accounts/<account_id>.json  # latest snapshot (equity, P&L, status, counts)
├── equity/                     # equity-curve history
└── models/                     # deployed ONNX models + hashes
```

* Every paper run **appends** closed trades to the per-account JSONL journal
  and refreshes the account snapshot.
* `the-quant backup` commits `state/` and pushes to `origin`.
* `the-quant restore` recreates the tree on a fresh install.
* `deploy/update.sh` snapshots state into git **before** pulling new code, so
  an update can never lose trading history.

---

## 🖥️ Deployment & operations

### systemd

`deploy/install.sh` installs `deploy/the-quant.service`: runs
`the-quant daemon_run` as the non-root `quant` user from `/opt/the-quant`, with
tier-aware ceilings (`MemoryMax=80%`, `CPUQuota=75%`) that self-scale to any
host. The `PATH` includes the `quant` user's cargo so the auto-updater can
rebuild.

```bash
systemctl status the-quant
journalctl -u the-quant -f
sudo systemctl restart the-quant
```

### Manual update / rollback

```bash
deploy/update.sh                       # pull → build → smoke → blue-green swap
sudo cp /usr/local/bin/the-quant{.prev,}   # instant rollback
```

The updater always smoke-tests (`the-quant --smoke`) before swapping and keeps
the previous binary as `the-quant.prev`.

---

## 🤖 ML pipeline (Python, offline)

Python **never** runs in the live hot path — it trains models that export ONNX:

```bash
pip install -r python/requirements.txt
# 1) pull real historical tick data (Dukascopy, free) -> M5 bars
python python/data/download_dukascopy.py --symbols eurusd,gbpusd,xauusd \
    --start 2023-01-01 --end 2025-12-31 --timeframes M5
# 2) full research pipeline: features -> PyTorch MLP (ONNX) + XGBoost compare,
#    then walk-forward backtests of Aegis / EMA / Bollinger / Donchian and an
#    Aegis hyperparameter grid. Writes models/latest.onnx + reports/*.json
python python/research/train_pipeline.py --start 2023-01-01
```

Legacy single-purpose trainers (`python/train/train_gbdt.py`,
`python/train/train_mlp.py`) still exist and train on synthetic data; the
**research pipeline above is the real-data path** and is what ships models.

### ☁️ Google Colab Cloud Training Integration

Train the heavy model + run all strategy research on a **Colab GPU**, and only
collect the output artifacts back into the repo:

1. **Launch Colab Worker** — open
   [`colab/quant_colab_runner.ipynb`](colab/quant_colab_runner.ipynb), set
   runtime accelerator to **GPU** (T4 / A100 / L4) and run all cells. It clones
   the latest repo (real-data pipeline incl. the Dukascopy downloader) and
   prints a public tunnel URL + secret token.
   > ⚠️ That URL is public — keep the runtime alive only while collecting.

2. **Connect the local repo to your runtime:**
   ```bash
   python colab/colab_cli.py connect --url https://xxxx.trycloudflare.com --token quant-colab-secret-key
   python colab/colab_cli.py info                     # show GPU / VRAM / RAM
   ```

3. **Run the full real-data pipeline on Colab and stream logs:**
   ```bash
   python colab/colab_cli.py train --type custom \
       --script python/research/train_pipeline.py \
       --params-json '{"symbols":"eurusd,gbpusd,xauusd,audusd,nzdusd,xagusd","start":"2023-01-01","epochs":80}'
   ```
   The worker downloads the Dukascopy ticks, builds features, trains the MLP,
   backtests Aegis + the other strategies, and grid-searches Aegis — all on
   your runtime.

4. **Collect only the outputs** (`.onnx` → `models/`, `.json` → `reports/`):
   ```bash
   python colab/colab_cli.py collect --job-id <id>
   ```
   Then `cargo test`, validate the ONNX, and commit the trained model + reports.

### 🛡️ Persistent agent mode — works on Colab, Kaggle & Codespaces, zero data loss

Cloud runtimes are ephemeral (Colab dies after ~12h idle, Kaggle after 12h
max). The **git job-queue agent** solves this completely — no tunnels, no
lost work, no babysitting:

- **Job queue** lives on the `colab-jobs` branch of the repo.
- **Every artifact is committed to the `colab-artifacts` branch the moment
  it is produced.** If the runtime dies mid-job, everything already
  persisted survives; a restarted agent re-claims the job automatically
  (stale claims are reclaimed after 2h).
- **Recover outputs locally at any time** — even days after the runtime is
  long gone:

  ```bash
  python3 colab/colab_cli.py pull      # artifacts -> reports/ models/ python/data/histdata/
  python3 colab/colab_cli.py status    # job queue + artifact history
  ```

**Queue work from your machine:**

```bash
# data collection (real Dukascopy ticks -> M5 bars):
python3 colab/colab_cli.py enqueue --script colab/jobs/download_data.py \
    --params-json '{"symbols":"eurusd,gbpusd,xauusd","start":"2025-01-01"}'

# full training pipeline (features -> backtests -> Aegis grid -> GBDT/MLP -> ONNX):
python3 colab/colab_cli.py enqueue --script colab/jobs/train_data.py \
    --params-json '{"train_args":["--symbols","eurusd,gbpusd,xauusd"]}'
```

**Workers that consume the queue** (any ONE of these, or all simultaneously —
claims prevent double execution):

| Channel | Free compute | Setup |
|---|---|---|
| **Google Colab** | T4 GPU, ~12h/session | Open `colab/quant_colab_runner.ipynb` → run cell **2. Persistent agent mode** (paste a GitHub PAT) |
| **Kaggle** | P100/T4, 30 GPU-h/week | Import `colab/kaggle_agent.ipynb` → enable Internet → add `GITHUB_TOKEN` secret → run |
| **GitHub Codespaces** | 2-4 core CPU, 120 core-h/month | Terminal: `export GITHUB_TOKEN=$GITHUB_TOKEN` is **automatic** → `python3 colab/agent.py --interval 30` |
| **Any free VM** (Oracle Free Tier, etc.) | 24/7 | `pip install requests` → `python3 colab/agent.py` under systemd/screen |

The agent needs `GITHUB_TOKEN` (fine-grained PAT, **Contents: read+write**
on THE-QUANT). It is never committed to the repo.

Round-trip test (no network needed):
```bash
python3 -m unittest colab.tests.test_persist -v
```

### ☁️ GitHub Codespaces (direct run)

For a 32-GB CPU Codespace instead of a Colab runtime, run the identical
pipeline directly (no tunnel needed — Codespaces exposes the repo):
```bash
pip install -r python/requirements.txt
python python/research/train_pipeline.py --start 2023-01-01 --epochs 80
```

3. **Inspect Runtime Hardware**:
   ```bash
   python colab/colab_cli.py info
   python colab/colab_cli.py runtimes
   ```

4. **Run Training & Collect ONNX Models**:
   ```bash
   python colab/colab_cli.py train --type mlp --epochs 30
   # Or run the guided interactive wizard:
   python colab/colab_cli.py interactive
   ```
   Trained `.onnx` model files (`latest.onnx`, `mlp.onnx`) and metrics are automatically downloaded into `THE-QUANT/models/` for sub-2ms Rust inference.

Drop the exported `.onnx` file under `models/` and the Rust core loads it
at boot (`onnx::load_backend`), falling back to the transparent rule-based
model when no file is present.

---

## ✅ Testing

```bash
cargo test          # 18 unit tests, zero warnings
```

The suite covers the risk engine (daily-loss / lifetime-floor / shield /
 extraction-cap gates and day rollover), pair sizing maths, day-key bucketing,
feed determinism, correlation maths, and the decision gate. Circuit-breaker
behaviour is asserted both at the `DeskRisk` level and through
`TradingDesk::step`.

---

## 📁 Project layout

```text
THE-QUANT/
├── Cargo.toml
├── config/system.toml          # all runtime configuration
├── src/
│   ├── main.rs                 # CLI entry point
│   ├── engine.rs               # multi-desk orchestration + state persistence
│   ├── desk.rs                 # TradingDesk: per-account isolation + entry gating
│   ├── universe.rs             # dynamic scan universe, cross-broker symbol identity,
│   │                           #   shared knowledge base (state/knowledge.json)
│   ├── aegis.rs                # strategy, DeskRisk circuit breakers, sizing
│   ├── risk.rs                 # RiskEngine / RiskContext (order-level risk)
│   ├── simfeed.rs              # deterministic sim / CSV / MT5 feeds
│   ├── onnx.rs                 # ONNX backend + rule-model fallback
│   ├── state.rs                # Store / Restorer (git-backed state)
│   ├── github.rs               # GitSync (backup to origin)
│   ├── update.rs               # 24 h auto-update scheduler (blue-green)
│   ├── tui.rs                  # multi-desk reporting
│   ├── resource.rs             # tier detection + per-module memory budgets
│   ├── execution.rs            # order execution scaffolding (MT5 bridge)
│   ├── features.rs, regime.rs, strategy.rs, security.rs, util.rs, web.rs
├── deploy/
│   ├── install.sh              # one-line installer (Rust + Python + MT5 + EA)
│   ├── mt5_ea.mq5              # MT5 bar-export connector (EA → shared CSV files)
│   ├── the-quant.service       # tier-aware systemd unit
│   └── update.sh               # 24 h auto-update (git pull + rebuild)
├── python/                     # offline training (torch/sklearn → ONNX)
└── state/                      # git-backed trading state
```

---

## 🩺 Troubleshooting

| Symptom | Diagnosis |
|---|---|
| `status: Blown (HALTED)` after a paper run | The Guardian Shield did its job — the desk took `shield_strike_limit` losing closes. Inspect `state/trades/<account>.jsonl`; reset is a manual operator decision. |
| `trades: 0 taken / N denied` | Every entry was refused by `pre_flight`. Check the daily loss / extraction / drawdown state in `state/accounts/<account>.json`. |
| `model load failed; falling back to rule model` | No `.onnx` under `state/models/` (or ONNX Runtime missing on a lean build). The rule model is fully functional. |
| `no git repo` on `backup` | Run from the repository root, or `git init` in the state root. |
| Desk never trades in sims | Aegis needs ≥ 30 synchronous bars per leg and correlation ≥ `min_correlation`; give the simulation more bars (e.g. `the-quant paper 20000`). |
| Update daemon silent | `journalctl -u the-quant`; it logs one line per 24 h check. Force one now with `the-quant update --now`. |

---

*Trade paper first. The circuit breakers protect the account — never weaken
them to force more trades.*
