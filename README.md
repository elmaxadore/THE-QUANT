# THE QUANT — v4.0 hybrid edition

An autonomous, self-contained quantitative trading system written in Rust with
a **hybrid ML bridge**: models are trained in Python (offline), exported to
ONNX, and served in Rust via ONNX Runtime — sub-2ms inference, Python never in
the live hot path.

This build implements the v4.0 "Hercules" requirements, optimised for
efficiency and for **scaling with the machine**:

- **Percentage-scaled resources** — the binary detects RAM/CPU at boot and
  allocates every module a percentage budget (Tier-1 lean → Tier-4 aggressive).
- **Auto-update every 24h** — checks the git repo, pulls new code, rebuilds,
  smoke-tests, and blue-green swaps the binary. Any failure rolls back
  instantly. State is never lost (`git` is the source of truth).
- **Git-centric state** — `state/` is version-controlled; `the-quant restore`
  reconstructs the system on a fresh machine.
- **Full paper-trading loop** — sim feed → features → regime → strategy →
  risk → execution → persisted journal + dashboard.

## Quick start

```bash
cargo build --release          # lean build (Tier-1 friendly, no heavy deps)
./target/release/the-quant paper 20000    # run a 20k-bar paper trade
./target/release/the-quant --status       # system + account summary
./target/release/the-quant --smoke        # self-check (used by auto-update)
./target/release/the-quant update --now   # force auto-update check now
./target/release/the-quant backup         # commit+push state to origin
```

## Optional build features (scale with your machine)

| Feature   | Adds                                        | When to enable          |
|-----------|---------------------------------------------|-------------------------|
| `crypto`  | Argon2id vault + AES-256-GCM (default)      | always                  |
| `ml`      | ONNX Runtime inference (`ort`) for `.onnx`  | Tier-2+ or when you have `models/latest.onnx` |
| `tui`     | ratatui interactive dashboard               | Tier-2+                 |
| `web`     | Axum HTTP status endpoint                   | Tier-2+                 |
| `db`      | sqlx PostgreSQL/TimescaleDB                 | when you run Postgres   |

Enable with: `cargo build --release --features ml,web,tui,db`

## Hybrid ML

```bash
pip install -r python/server/requirements.txt
python3 python/train/train_gbdt.py   # trains -> models/latest.onnx
cargo build --release --features ml
./target/release/the-quant paper     # uses ONNX model if present
```

## Auto-update (the core ask)

`AutoUpdater` (src/update.rs) runs a scheduler loop (default: every 24h):

```
fetch origin → compare HEAD → snapshot state (git commit)
→ pull --rebase → cargo build --release (staged)
→ smoke-test new binary → blue-green swap → record in state.json
```

On any failure the previous binary is preserved, state remains intact, and the
next check retries. `deploy/update.sh` is the manual equivalent.

## Layout

```
src/          Rust core (resource, config, security, state, github, update,
              simfeed, features, regime, onnx, strategy, risk, execution,
              engine, tui, web)
python/       offline training + research server (hybrid ML)
deploy/       systemd unit, install.sh, update.sh
migrations/   PostgreSQL / TimescaleDB schema
state/        git-synced source-of-truth state
config/       system.toml
models/       .onnx model artifacts
```

## Risks & disclaimer

This is trading software. Run in paper/simulation first. The author takes no
responsibility for capital losses. Prop-firm rules change; verify templates.