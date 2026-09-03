# THE-QUANT Remote Training — Full Setup Tutorial

Outsource data collection + model training to free cloud compute and only
receive the outputs locally. Every artifact is committed to a GitHub branch
the moment it is produced, so **a dying runtime never loses work**.

```
 YOUR MACHINE                     GITHUB (free storage)              FREE COMPUTE
 ─────────────                    ─────────────────────              ────────────
 colab_cli.py enqueue  ─────────►  colab-jobs branch   ─────────►  Agent (Colab /
      (queue a job)                   (the queue)                Kaggle / Codespaces /
                                                                     free VM) executes
 colab_cli.py pull     ◄─────────  colab-artifacts ◄───────────  artifacts pushed to
      (recover outputs)              (durable results)          GitHub immediately
```

You only do two things by hand: **create a GitHub token once** and **start an
agent in a browser notebook**. Everything else is terminal commands.

---

## Step 0 — Local prerequisites (already done on this machine, verify only)

```bash
cd THE-QUANT
git remote -v                                 # must show elmaxadore/THE-QUANT
git pull                                      # up to date
python3 -m unittest discover -s colab/tests   # 6 tests OK = harness healthy
```

---

## Step 1 — Create the GitHub token (the ONLY manual auth, ~2 minutes)

1. Go to <https://github.com/settings/personal-access-tokens/new>
2. **Token name:** `the-quant-remote`
3. **Expiration:** 90 days (regenerate when it expires; or custom up to 1 year)
4. **Repository access →** "Only select repositories" → pick `THE-QUANT`
5. **Permissions → Repository permissions → Contents →** `Read and write`
   (everything else stays "No access")
6. **Generate token** and copy it (starts with `github_pat_…`)

> 🔐 The token grants write access ONLY to this one repo. It is never
> committed anywhere — it lives in the notebook session / Codespace env only.
> Revoke it any time at <https://github.com/settings/personal-access-tokens>.

---

## Step 2 — Start a worker agent (pick ONE to start; add more any time)

### Option A — Google Colab (free T4 GPU) ← recommended for training

1. Open <https://colab.research.google.com> → **File → Open notebook →
   GitHub** tab → enter `elmaxadore/THE-QUANT` → choose
   `colab/quant_colab_runner.ipynb`
2. **Runtime → Change runtime type → T4 GPU → Save**
3. **Runtime → Run all.** Cell 1 installs deps; cell 2 clones the repo.
4. When you reach **"2. Persistent agent mode"**, it asks for the token →
   paste your `github_pat_…` into the small 🔑 input box.
5. You'll see: `[+] Persistent agent started — polling colab-jobs every 30s.`

Done. Leave the browser tab open (Colab needs it for keep-alive). Anything
you queue is executed and persisted automatically.

*Optional — the older tunnel mode (cells under "1.") also still works; it is
interactive/low-latency but dies with the runtime. Agent mode is better.*

### Option B — Kaggle (free P100/T4, 30 GPU-hours/week)

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import
   Notebook → GitHub URL** → `colab/kaggle_agent.ipynb` from THE-QUANT
2. Right sidebar: **Session options → Internet → ON**, **Accelerator → GPU T4**
3. **Add-ons → Secrets** → add secret named `GITHUB_TOKEN` = your PAT
4. **Run all.** The notebook joins the same queue as the Colab worker.

### Option C — GitHub Codespaces (no token paste needed — it's automatic)

1. <https://github.com/elmaxadore/THE-QUANT> → green **Code → Codespaces →
   Create codespace**
2. In its terminal:
   ```bash
   python3 colab/agent.py --interval 30
   ```
   (`GITHUB_TOKEN` is injected by Codespaces automatically.)

### Option D — any free VM / WSL / a second PC (24/7 worker)

```bash
git clone https://github.com/elmaxadore/THE-QUANT.git && cd THE-QUANT
GITHUB_TOKEN=github_pat_xxx python3 colab/agent.py --interval 30
```
Run it under `tmux`/`screen` or a systemd unit for permanence.

> Workers can run **simultaneously** — git-based claiming guarantees each
> job runs exactly once.

---

## Step 3 — Queue work (all from your terminal)

The jobs for the initial real-data + training run **are already queued** on
`colab-jobs` — see `python3 colab/colab_cli.py status`. To queue more or
different work:

```bash
cd THE-QUANT

# 1. Data collection — real Dukascopy ticks (runs on the cloud's IP, so your
#    home IP never gets rate-limited):
python3 colab/colab_cli.py enqueue --script colab/jobs/download_data.py \
    --params-json '{"symbols":"eurusd,gbpusd,xauusd","start":"2025-01-01"}'

# 2. Full training — features, EMA/Bollinger/Donchian backtests, the Aegis
#    grid search, GBDT + MLP, ONNX export:
python3 colab/colab_cli.py enqueue --script colab/jobs/train_data.py \
    --params-json '{"train_args":["--symbols","eurusd,gbpusd,xauusd"]}'

# More symbols / longer history for extra strategies:
python3 colab/colab_cli.py enqueue --script colab/jobs/download_data.py \
    --params-json '{"symbols":"usdcad,usdjpy","start":"2024-01-01"}'
```

Watch progress:
```bash
python3 colab/colab_cli.py status     # job queue + artifact history
```
Or peek inside Colab at the log cell (`!tail -15 /content/agent.log`).

---

## Step 4 — Collect the outputs

```bash
python3 colab/colab_cli.py pull
```
Routes automatically: `*.json → reports/`, `*.onnx → models/`,
`*.csv → python/data/histdata/`.

Then validate + commit locally:
```bash
cargo test
git add reports/ models/ && git commit -m "results: first remote training run"
```

---

## What happens when things go wrong (and why nothing is lost)

| Event | Consequence | Recovery |
|---|---|---|
| Colab disconnects mid-job | Artifacts made so far are already on GitHub | New agent re-claims the job after 2h and re-runs it |
| Job fails (rc≠0) | Marked `failed` on the queue with the reason | Read `status`; fix params; re-`enqueue` |
| You shut your laptop for a week | — | Outputs wait on `colab-artifacts`; `pull` whenever |
| Token expires (90 days) | Agent can't push | New PAT (Step 1), paste again |
| Local disk dies | Data/reports still on GitHub | Fresh clone → `pull` → everything back |
| Dukascopy 503s your home IP | — | That's why downloads run on the cloud worker |

Two git branches do all the work: `colab-jobs` (queue + statuses) and
`colab-artifacts` (immutable result history) — both browsable on GitHub.

## Troubleshooting

- **`401`/`403` in the agent log** → wrong/expired token, or Contents
  permission not set to Read and write.
- **Agent prints nothing** → queue is empty; check `status` locally and that
  Internet is enabled (Kaggle especially).
- **`WARNING: no CSVs on colab-artifacts`** during training → the download
  job hasn't completed yet; wait for it, then re-enqueue training.
- **Colab asks to reconnect** → normal after ~12h; run the agent cell again.
- **Download seems slow** → Dukascopy serves ~30k tiny hourly files; with 4
  workers expect roughly 1–3 hours for 3 symbols. The retry pass + month
  cache mean interruptions resume rather than restart.

## Security notes

- The PAT is scoped to one repo, Contents-only, and never stored in the repo
  or in job specs.
- The old tunnel mode is protected by a secret token; the agent needs no
  inbound access at all (outbound-only HTTPS to GitHub).
- All heavy code execution happens inside the sandboxed notebook/VM.
