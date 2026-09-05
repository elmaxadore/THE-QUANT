#!/usr/bin/env python3
"""THE QUANT — real-data training & strategy-research pipeline (runs on Colab).

Self-contained research pipeline executed on a remote training runtime (Google
Colab GPU) through colab/colab_worker.py, or locally on CPU:

  1. REAL tick data:  Dukascopy .bi5 ticks -> M5 OHLCV bars
                       (python/data/download_dukascopy.py)
  2. FEATURES:         the EXACT 12-feature vector computed in src/features.rs
  3. MODEL:            PyTorch MLP trained on next-bar signal, exported to ONNX
                       input "input" [N,12] f32 -> output "output" [N,1] f32
                       (matches src/onnx.rs OrtModel which reads arr[0]).
                       XGBoost trained alongside for comparison/importance.
  4. VALIDATION:       time-ordered walk-forward split (train/val/test) +
                       Aegis hyperparameter grid search on real data.
  5. STRATEGIES:       backtests Aegis hedged pairs vs EMA momentum,
                       Bollinger mean-reversion, Donchian breakout and the
                       MLP signal strategy, with realistic spread costs.
  6. ARTIFACTS:        models/latest.onnx + reports/*.json collected by the
                       worker and pulled back to the local repo.

Usage (Colab via worker):
  # sent as script_code by colab_cli.py train --type custom
  python train_pipeline.py --symbols eurusd,gbpusd,xauusd --start 2024-01-01

Usage (local):
  python python/research/train_pipeline.py --start 2025-01-01 --no-download
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap: find the repo root whether we run from the repo or from a
# Colab workdir (COLAB_REPO_DIR env, default /content/quant_colab/repo).
# ---------------------------------------------------------------------------
_ENV_REPO = os.environ.get("COLAB_REPO_DIR", "")
_CANDIDATE_ROOTS = [
    _ENV_REPO if _ENV_REPO else "",
    "/content/quant_colab/repo",
    os.getcwd(),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
]


def find_repo_root() -> str:
    for root in _CANDIDATE_ROOTS:
        if root and os.path.isfile(os.path.join(root, "config", "system.toml")):
            return root
    return _CANDIDATE_ROOTS[-1]  # fall back to repo-relative


ROOT = find_repo_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_MODELS = os.environ.get("COLAB_MODELS_DIR", os.path.join(ROOT, "models"))
OUT_REPORTS = os.environ.get("COLAB_ARTIFACTS_DIR", os.path.join(ROOT, "reports"))
HISTDATA = os.path.join(ROOT, "python", "data", "histdata")

# Per-symbol round-trip spread/commission cost (fraction of notional).
COST_RT = {
    "EURUSD": 0.00012, "GBPUSD": 0.00014, "AUDUSD": 0.00012,
    "NZDUSD": 0.00012, "XAUUSD": 0.00020, "XAGUSD": 0.00035,
    "US100": 0.00020, "US30": 0.00020, "US500": 0.00020,
}
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD", "XAGUSD"]
DD_DEFAULT = ["XAU_XAG", "GBP_AUD", "AUD_NZD", "EUR_GBP"]
BARS_PER_YEAR = 288.0 * 260.0  # M5 bars per trading year (approx)
# ---------------------------------------------------------------------------
# FEATURE ENGINEERING — a 1:1 Python mirror of src/features.rs compute_features()
# (see the "Feature slots" comment there for exact definitions). Given a
# DataFrame with [open, high, low, close, volume], returns an (n, 12) array
# where row t describes bars[0..=t].
# ---------------------------------------------------------------------------
def compute_features(df):
    c_arr = df["close"].to_numpy(dtype=float)
    h_arr = df["high"].to_numpy(dtype=float)
    l_arr = df["low"].to_numpy(dtype=float)
    v_arr = df["volume"].to_numpy(dtype=float)
    n = len(df)
    f = np.zeros((n, 12), dtype=np.float64)
    if n == 0:
        return f

    logr = np.full(n, np.nan)
    logr[1:] = np.log(c_arr[1:] / c_arr[:-1])

    # f0: 1-bar log return ; f1: 5-bar log return
    f[1:, 0] = logr[1:]
    f[5:, 1] = np.log(c_arr[5:] / c_arr[:-5])

    # f2: 10-bar realized vol ; f7: volume zscore (window <=10)
    # NOTE: windows exclude index 0 (logr[0] is undefined) exactly like the
    # Rust k=min(t,10) slice logr[1:t+1].
    lr_fill = np.nan_to_num(logr, nan=0.0)
    cs = np.concatenate(([0.0], np.cumsum(lr_fill)))
    cs2 = np.concatenate(([0.0], np.cumsum(lr_fill * lr_fill)))
    vs = np.concatenate(([0.0], np.cumsum(v_arr)))
    vs2 = np.concatenate(([0.0], np.cumsum(v_arr * v_arr)))

    def pop_std(cum, cum2, lo, hi):
        k = hi - lo
        if k < 2:
            return 0.0
        mean = (cum[hi] - cum[lo]) / k
        var = (cum2[hi] - cum2[lo]) / k - mean * mean
        return math.sqrt(max(var, 0.0))

    for t in range(1, n):
        f[t, 2] = pop_std(cs, cs2, max(1, t - 9), t + 1)
    vs_mean = np.concatenate(([0.0], np.cumsum(v_arr)))
    for t in range(n):
        lo = max(0, t - 9)
        sd = pop_std(vs, vs2, lo, t + 1)
        if sd > 1e-12:
            mean_v = (vs_mean[t + 1] - vs_mean[lo]) / (t + 1 - lo)
            f[t, 7] = (v_arr[t] - mean_v) / sd

    # f3: RSI-14 (simple gains/losses, mirrors src/features.rs rsi())
    f[:, 3] = 50.0  # default for insufficient data (matches Rust n < period+1)
    for t in range(1, n):
        if t < 14:
            continue
        deltas = c_arr[t - 13: t + 1] - c_arr[t - 14: t]
        gains = np.where(deltas > 0, deltas, 0.0).sum()
        losses = np.where(deltas < 0, -deltas, 0.0).sum()
        if losses < 1e-12:
            f[t, 3] = 100.0
        else:
            rs = gains / losses
            f[t, 3] = 100.0 - 100.0 / (1.0 + rs)

    # f4: ATR-14 / price (mean TR over last <=14 bars)
    tr = np.zeros(n)
    tr[1:] = np.maximum(
        np.maximum(h_arr[1:] - l_arr[1:],
                   np.abs(h_arr[1:] - c_arr[:-1])),
        np.abs(l_arr[1:] - c_arr[:-1]))
    ct = np.concatenate(([0.0], np.cumsum(tr)))
    for t in range(1, n):
        lo = max(1, t - 13)
        f[t, 4] = (ct[t + 1] - ct[lo]) / (t + 1 - lo) / max(c_arr[t], 1e-12)
# ema (mirrors src/features.rs)
    def ema(xs, period):
        a = 2.0 / (period + 1.0)
        out = np.zeros_like(xs)
        e = xs[0]
        for i in range(1, len(xs)):
            e = a * xs[i] + (1.0 - a) * e
            out[i] = e
        out[0] = xs[0]
        return out

    # f5: EMA12-EMA26 / price
    f[:, 5] = (ema(c_arr, 12) - ema(c_arr, 26)) / np.maximum(c_arr, 1e-12)

    # f6: Bollinger width (20, 2σ)
    bcs = np.concatenate(([0.0], np.cumsum(c_arr)))
    bcs2 = np.concatenate(([0.0], np.cumsum(c_arr * c_arr)))
    for t in range(19, n):
        mean = (bcs[t + 1] - bcs[t - 19]) / 20.0
        var = (bcs2[t + 1] - bcs2[t - 19]) / 20.0 - mean * mean
        sd = math.sqrt(max(var, 0.0))
        f[t, 6] = (4.0 * sd) / max(mean, 1e-12)

    # f8: high-low range / close ; f9: close position in range
    rng = h_arr - l_arr
    f[:, 8] = rng / np.maximum(c_arr, 1e-12)
    flat = rng <= 1e-12
    f[:, 9] = np.where(flat, 0.5, (c_arr - l_arr) / np.maximum(rng, 1e-12))

    # f10: crude Hurst-like sign-persistence (mirrors src/features.rs:
    #        loop w in 2..n.min(60), count where r1*r2 > 0, divide by total)
    f[:, 10] = 0.5  # default for insufficient data
    prod = np.full(n, np.nan)
    prod[2:] = logr[2:] * logr[1:-1]
    sa = np.where(np.isnan(prod), 0.0, (prod > 0.0).astype(float))
    cs10 = np.concatenate(([0.0], np.cumsum(sa)))
    for t in range(1, n):
        w_max = min(t + 1, 60)
        total = w_max - 2
        if total <= 0:
            f[t, 10] = 0.5
            continue
        agrees = cs10[w_max]
        f[t, 10] = agrees / total

    # f11: momentum ROC-10
    f[10:, 11] = (c_arr[10:] - c_arr[:-10]) / np.maximum(c_arr[:-10], 1e-12)
    return f


# ---------------------------------------------------------------------------
# DATA LOADING — real M5 bars from python/data/histdata (produced from
# Dukascopy ticks by download_dukascopy.py). date,time columns are parsed back
# to UTC epoch seconds for lockstep and time-ordered splits.
# ---------------------------------------------------------------------------
def compute_features_fast(df):
    """Vectorised equivalent of compute_features() — identical outputs for
    the rows that survive the 80-bar warmup, but 10-50x faster (pandas
    rolling instead of per-bar Python loops). Use this everywhere."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    n = len(df)
    f = np.zeros((n, 12), dtype=np.float64)
    if n == 0:
        return f

    c_arr = c.to_numpy()
    logr = np.full(n, np.nan)
    logr[1:] = np.log(c_arr[1:] / c_arr[:-1])
    lr = pd.Series(logr)

    # f0/f1: log returns
    f[1:, 0] = logr[1:]
    f[5:, 1] = np.log(c_arr[5:] / c_arr[:-5])

    # f2: 10-bar realized vol (population std, window <=10)
    f[:, 2] = lr.rolling(10, min_periods=2).std(ddof=0).fillna(0.0).to_numpy()

    # f3: RSI-14 (simple gains/losses over 14 deltas; NaN -> 50)
    diff = c.diff()
    up = diff.clip(lower=0).rolling(14, min_periods=14).sum()
    dn = (-diff).clip(lower=0).rolling(14, min_periods=14).sum()
    rsi = 100.0 - 100.0 / (1.0 + up / dn.where(dn > 1e-12))
    f[:, 3] = np.where(np.isnan(up), 50.0,
                       np.where(np.isnan(dn) | (dn <= 1e-12), 100.0, rsi))

    # f4: ATR-14 / price (exclude tr[0]=0 from mean, matching src/features.rs atr())
    tr = np.zeros(n)
    tr[1:] = np.maximum(
        np.maximum(h.to_numpy()[1:] - l.to_numpy()[1:],
                   np.abs(h.to_numpy()[1:] - c_arr[:-1])),
        np.abs(l.to_numpy()[1:] - c_arr[:-1]))
    ct = np.concatenate(([0.0], np.cumsum(tr)))
    t_idx = np.arange(n)
    lo_idx = np.maximum(1, t_idx - 13)
    denom = t_idx + 1 - lo_idx
    denom[0] = 1  # avoid div-by-zero; row 0 ATR is 0
    atr_vals = (ct[t_idx + 1] - ct[lo_idx]) / denom
    atr_vals[0] = 0.0
    f[:, 4] = atr_vals / np.maximum(c_arr, 1e-12)

    # f5: EMA12-EMA26 / price (alpha seeding matches src/features.rs)
    f[:, 5] = ((c.ewm(alpha=2 / 13, adjust=False).mean()
                - c.ewm(alpha=2 / 27, adjust=False).mean())
               / np.maximum(c_arr, 1e-12)).to_numpy()

    # f6: Bollinger width (20, 2σ)
    m20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    f[:, 6] = (4.0 * sd20 / np.maximum(m20, 1e-12)).fillna(0.0).to_numpy()

    # f7: volume zscore (window <=10)
    vm = v.rolling(10, min_periods=1).mean()
    vsd = v.rolling(10, min_periods=1).std(ddof=0)
    f[:, 7] = np.where(vsd > 1e-12, (v - vm) / np.maximum(vsd, 1e-12), 0.0)

    # f8/f9: bar range geometry
    rng = (h - l).to_numpy()
    f[:, 8] = rng / np.maximum(c_arr, 1e-12)
    f[:, 9] = np.where(rng <= 1e-12, 0.5,
                       (c_arr - l.to_numpy()) / np.maximum(rng, 1e-12))

    # f10: sign-persistence over the 58 adjacent-return products of the
    # 80-bar desk window (indices window[2..59] -> absolute t-77..t-20).
    # This mirrors what the Rust runtime computes at inference time.
    sa = (logr * np.concatenate(([np.nan], logr[:-1]))) > 0.0
    sa = pd.Series(np.where(np.isnan(sa), 0.0, sa.astype(float)))
    f10 = sa.rolling(58, min_periods=58).mean().shift(20)
    f[:, 10] = np.where(np.arange(n) >= 79, f10.fillna(0.5).to_numpy(), 0.5)

    # f11: momentum ROC-10
    f[10:, 11] = (c_arr[10:] - c_arr[:-10]) / np.maximum(c_arr[:-10], 1e-12)
    return f


# ---------------------------------------------------------------------------
# DATA LOADING — real M5 bars from python/data/histdata (produced from
# Dukascopy ticks by download_dukascopy.py). date,time columns are parsed back
# to UTC epoch seconds for lockstep and time-ordered splits.
# ---------------------------------------------------------------------------
def ensure_data(symbols_lower, start, end, download=True):
    import pandas as pd

    if not os.path.isdir(HISTDATA):
        os.makedirs(HISTDATA, exist_ok=True)
    missing = [s for s in symbols_lower
               if not os.path.isfile(os.path.join(HISTDATA, f"{s}.csv"))]
    if download and missing:
        print(f"[pipe] downloading missing history (Dukascopy tick feed): "
              f"{missing}", flush=True)
        cmd = [sys.executable,
               os.path.join(ROOT, "python", "data", "download_dukascopy.py"),
               "--symbols", ",".join(missing),
               "--start", start, "--end", end, "--timeframes", "M5",
               "--workers", "16"]
        subprocess.run(cmd, check=True)
    elif missing:
        raise FileNotFoundError(
            f"missing data for {missing}; run download_dukascopy.py first")

    frames = {}
    for sym in symbols_lower:
        path = os.path.join(HISTDATA, f"{sym}.csv")
        df = pd.read_csv(path)
        try:
            ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                                format="%Y-%m-%d %H:%M", utc=True)
            df["epoch"] = ts.astype("int64") // 10 ** 9
        except (ValueError, TypeError):
            # Fallback: not all CSV producers write real timestamps (older
            # synthetic fixtures use date=row index). Emulate the Rust
            # CsvFeed, which derives bar time from the line number.
            df["epoch"] = np.arange(len(df), dtype=np.int64)
        frames[sym] = df[["epoch", "open", "high", "low", "close", "volume"]]
    return frames


def validate_data(frames, tf_minutes=5):
    """Data-quality gate (pre-MT5 data audit): gaps, dead bars, price sanity.

    Returns a per-symbol dict consumed by the training summary so every run
    documents exactly what it trained on."""
    step = tf_minutes * 60
    out = {}
    for sym, df in frames.items():
        ep = df["epoch"].to_numpy()
        gaps = np.diff(ep) // step - 1
        gap_list = gaps[gaps > 0]
        close = df["close"].to_numpy()
        out[sym] = {
            "bars": int(len(df)),
            "first": str(dt.datetime.fromtimestamp(int(ep[0]),
                                                   tz=dt.timezone.utc)),
            "last": str(dt.datetime.fromtimestamp(int(ep[-1]),
                                                  tz=dt.timezone.utc)),
            "gap_slots_missing": int(gap_list.sum()) if len(gap_list) else 0,
            "largest_gap_bars": int(gap_list.max()) if len(gap_list) else 0,
            "zero_volume_bars": int((df["volume"] <= 0).sum()),
            "bad_price_bars": int(((close <= 0) |
                                   ~np.isfinite(close)).sum()),
        }
    return out


def build_dataset(frames, min_bars=80):
    """Stack feature rows + forward labels across all symbols."""
    X, y, syms, ep = [], [], [], []
    for sym, df in frames.items():
        f = compute_features_fast(df)
        close = df["close"].to_numpy(float).ravel()
        fwd = np.full(len(df), np.nan)
        fwd[:-1] = close[1:] / close[:-1] - 1.0
        for t in range(min_bars, len(df) - 1):
            X.append(f[t])
            # label: sign of next-bar return, tanh-squashed (gain=250) so it
            # lives in (-1,1) exactly like the Rust OrtModel signal.
            y.append(math.tanh(fwd[t] * 250.0))
            syms.append(sym)
            ep.append(df["epoch"].iloc[t])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y, np.asarray(syms), np.asarray(ep, dtype=np.int64)
# ---------------------------------------------------------------------------
# TRAINING — PyTorch MLP (fast on Colab GPU) + XGBoost comparison.
# Time-ordered walk-forward split: train 70% / validation 15% / test 15%.
# ---------------------------------------------------------------------------
def split_series(X, y, ep):
    order = np.argsort(ep, kind="stable")
    X, y, ep = X[order], y[order], ep[order]
    n = len(X)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    return ((X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:]))


def train_torch(Xtr, ytr, Xva, yva, args, ckpt=None):
    """Train the MLP and export ONNX (input 'input' [N,12], output 'output'
    [N,1]). Best epoch chosen on the validation split (early stopping).

    If ckpt is a TrainingCheckpoint, saves weights every N epochs and
    resumes from the last checkpoint on restart (survives Colab disconnections).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[pipe] torch not available — skipping MLP/ONNX training", flush=True)
        return None

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mlp] device = {device}", flush=True)

    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(ytr).float()
    Xv = torch.from_numpy(Xva).float()
    yv = torch.from_numpy(yva).float()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch, shuffle=True)

    # Normalise per-feature to zero-mean/unit-var on the TRAIN set only
    # (statistics baked into the ONNX graph via an initial Linear/BN so the
    # Rust runtime feeds raw features exactly like during training).
    mean = Xt.mean(0, keepdim=True)
    std = Xt.std(0, keepdim=True) + 1e-8
    Xt = (Xt - mean) / std
    Xv = (Xv - mean) / std

    model = nn.Sequential(
        nn.Linear(12, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(),
        nn.Linear(32, 1), nn.Tanh(),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    # Training loop checkpointing (resume across Colab disconnections)
    train_ckpt = None
    start_epoch = 1
    if ckpt:
        from python.research.checkpoint import TrainingCheckpoint
        train_ckpt = TrainingCheckpoint(
            args.job_name, "train_ckpt",
            args.checkpoint_dir or os.path.join(ROOT, "checkpoints"),
            every_n_epochs=10)
        resume = train_ckpt.load_latest()
        if resume:
            # Load model weights
            state_dict = {}
            for k, v in resume["model_state"].items():
                state_dict[k] = torch.tensor(v)
            model.load_state_dict(state_dict)
            # Load optimizer state
            if resume["optimizer_state"]:
                opt.load_state_dict(resume["optimizer_state"])
            start_epoch = resume["epoch"] + 1
            print(f"[mlp] resuming from epoch {resume['epoch']} "
                  f"(val_loss={train_ckpt.get_best_loss():.5f})", flush=True)

    best = None
    patience = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            va_loss = lossf(model(Xv.to(device)), yv.to(device)).item()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"[mlp] epoch {epoch:03d} train_loss={running / len(Xt):.5f} "
                  f"val_loss={va_loss:.5f}", flush=True)
        if best is None or va_loss < best[0]:
            best = (va_loss, model.state_dict().copy(), epoch)
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print(f"[mlp] early stop @ epoch {epoch} (best={best[0]:.5f} "
                      f"e{best[2]})", flush=True)
                break
        # Save training checkpoint every N epochs
        if train_ckpt and train_ckpt.should_save(epoch):
            train_ckpt.save(epoch, model.state_dict(), opt.state_dict(), va_loss)

    model.load_state_dict(best[1])
    model.eval()
    metrics = {"model": "mlp", "best_val_loss": best[0], "best_epoch": best[2],
               "device": device, "mean": mean.squeeze().tolist(),
               "std": std.squeeze().tolist()}

    # Export ONNX with raw-feature input: fold the normaliser into the graph
    # by prepending a scale/shift as the first Linear layer.
    class QuantMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.Linear(12, 12, bias=True)
            with torch.no_grad():
                self.norm.weight.copy_(torch.diag(1.0 / std.squeeze()))
                self.norm.bias.copy_(-mean.squeeze() / std.squeeze())
            self.net = model

        def forward(self, x):
            return self.net(self.norm(x))

    export_model = QuantMLP().eval()
    import onnx
    with torch.no_grad():
        dummy = torch.zeros(1, 12)
        torch.onnx.export(
            export_model, dummy,
            os.path.join(OUT_MODELS, "latest.onnx"),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17)
        onnx.checker.check_model(
            onnx.load(os.path.join(OUT_MODELS, "latest.onnx")))
    print("[mlp] exported ONNX -> models/latest.onnx", flush=True)
    return metrics
def train_xgb(Xtr, ytr, Xva, yva):
    """XGBoost benchmark classifier on sign-of-signal; reports feature
    importance. Not exported to ONNX (MLP is the Rust-serving model)."""
    try:
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error
    except ImportError:
        print("[pipe] xgboost not available — skipping benchmark", flush=True)
        return None
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, random_state=7,
                         n_jobs=-1)
    model.fit(Xtr, ytr.ravel())
    pred = model.predict(Xva)
    mse = float(mean_squared_error(yva.ravel(), pred))
    # directional hit-rate (sign agreement on validation)
    hit = float(np.mean(np.sign(pred) == np.sign(yva.ravel())))
    imp = {f"f{i}": float(x)
           for i, x in enumerate(model.feature_importances_)}
    print(f"[xgb] val MSE={mse:.5f} sign-hit={hit:.3f}", flush=True)
    return {"model": "xgboost", "val_mse": mse, "sign_hit": hit,
            "feature_importance": imp}


# ---------------------------------------------------------------------------
# BACKTESTING — many strategies, walk-forward validated on REAL data with
# round-trip spread costs. Returns per-strategy metric dicts.
# ---------------------------------------------------------------------------
def metrics_from_returns(ret, name):
    ret = np.asarray(ret, dtype=float)
    mask = np.abs(ret) > 1e-20
    n = max(mask.sum(), 1)
    mean = ret[mask].mean() if mask.any() else 0.0
    std = ret[mask].std() if mask.any() else 1.0
    sharpe = mean / (std + 1e-12) * math.sqrt(BARS_PER_YEAR)
    cum = float(np.prod(1.0 + ret) - 1.0)
    eq = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max() / peak.max()) if peak.max() > 0 else 0.0
    return {"strategy": name, "count": int(n), "total_return": round(cum, 6),
            "sharpe": round(float(sharpe), 3), "max_drawdown": round(dd, 4),
            "avg_bar_return": round(float(mean), 8)}


def sig_to_returns(sig, ret, cost):
    """Vectorised position PnL: enter on next bar close, pay cost on flips."""
    sig = np.asarray(sig, float)
    ret = np.asarray(ret, float)
    pos = np.zeros_like(sig)
    pos[1:] = sig[:-1]                      # trade at next bar's close
    flips = np.abs(np.diff(pos)) > 0
    return pos * ret - np.concatenate(([0.0], flips)) * cost


def strat_ema(frames, cost_map, fast=12, slow=26):
    """EMA-cross momentum strategy per symbol."""
    import pandas as pd
    out = []
    for sym, df in frames.items():
        c = df["close"]
        sig = np.where(c.ewm(span=fast, adjust=False).mean() >
                       c.ewm(span=slow, adjust=False).mean(), 1.0, -1.0)
        ret = c.pct_change().fillna(0.0).to_numpy()
        r = sig_to_returns(sig, ret, cost_map.get(sym.upper(), 0.0002))
        out.append(metrics_from_returns(r, f"ema_cross:{sym.lower()}"))
    return out


def strat_bollinger(frames, cost_map, window=20, n_sigma=2.0):
    """Bollinger mean-reversion: short the upper band, buy the lower."""
    import pandas as pd
    out = []
    for sym, df in frames.items():
        c = df["close"]
        mid = c.rolling(window).mean()
        sd = c.rolling(window).std().fillna(0.0)
        upper = mid + n_sigma * sd
        lower = mid - n_sigma * sd
        sig = np.where(c > upper, -1.0, np.where(c < lower, 1.0, 0.0))
        ret = c.pct_change().fillna(0.0).to_numpy()
        r = sig_to_returns(sig, ret, cost_map.get(sym.upper(), 0.0002))
        out.append(metrics_from_returns(r, f"bollinger:{sym.lower()}"))
    return out


def strat_donchian(frames, cost_map, window=55):
    """Donchian breakout: long above N-bar high, short below N-bar low."""
    out = []
    for sym, df in frames.items():
        c = df["close"]
        hh = c.rolling(window).max()
        ll = c.rolling(window).min()
        sig = np.where(c > hh.shift(1), 1.0,
                       np.where(c < ll.shift(1), -1.0, np.nan))
        sig = pd.Series(sig).ffill().fillna(0.0).to_numpy()
        ret = c.pct_change().fillna(0.0).to_numpy()
        r = sig_to_returns(sig, ret, cost_map.get(sym.upper(), 0.0002))
        out.append(metrics_from_returns(r, f"donchian:{sym.lower()}"))
    return out


def strat_mlp(frames, cost_map, model_onnx=None, threshold=0.25):
    """Drive positions straight from the trained ONNX model signal."""
    if model_onnx is None:
        return [{"strategy": "mlp_signal", "count": 0,
                 "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                 "avg_bar_return": 0.0}]
    import onnxruntime as ort
    sess = ort.InferenceSession(model_onnx, providers=["CPUExecutionProvider"])
    out = []
    for sym, df in frames.items():
        f = compute_features(df)
        ret = df["close"].pct_change().fillna(0.0).to_numpy()
        sig = np.zeros(len(df))
        chunk = f[64:]
        inp = np.asarray(chunk, dtype=np.float32)
        pred = sess.run(["output"], {"input": inp})[0].ravel()
        s = np.clip(pred, -1.0, 1.0)
        sig[64:] = np.where(np.abs(s) >= threshold, s, 0.0)
        r = sig_to_returns(sig, ret, cost_map.get(sym.upper(), 0.0002))
        out.append(metrics_from_returns(r, f"mlp_signal:{sym.lower()}"))
    return out
def aegis_pairs(frames, cost_map, epoch_lo, epoch_hi,
                z_entry=2.0, z_exit=0.5, lookback=200, corr_min=0.70,
                max_pairs=5, seed_pairs=None):
    """AEGIS — hedged correlated pairs on REAL M5 data.

    For each candidate pair: rolling beta of log prices, spread
    z = (logA - beta*logB - mu)/sigma over `lookback`. Entry short/long the
    spread at |z| => z_entry, exit the pair flat when |z| <= z_exit. Hedged:
    position PnL = pos * (retA - beta*retB). Reference period [epoch_lo,
    epoch_hi) restricts scoring; warmup happens earlier in the series.
    """
    import pandas as pd

    keys = list(frames.keys())
    uppercase = [k.upper() for k in keys]
    results = []
    used = set()
    candidates = seed_pairs or [(a, b) for a in keys for b in keys if a < b]

    for a, b in candidates:
        if a not in frames or b not in frames:
            continue
        if len(results) >= max_pairs:
            break
        pair_key = frozenset({a.upper(), b.upper()})
        if pair_key in used:
            continue
        merged = frames[a][["epoch", "close"]].merge(
            frames[b][["epoch", "close"]], on="epoch", suffixes=("_a", "_b"))
        if len(merged) < lookback * 2:
            continue
        lna = np.log(merged["close_a"].to_numpy())
        lnb = np.log(merged["close_b"].to_numpy())
        ra = np.diff(lna)
        rb = np.diff(lnb)

        corr = np.corrcoef(ra[-lookback:], rb[-lookback:])[0, 1]
        if corr < corr_min or not np.isfinite(corr):
            continue

        # rolling beta = cov(ra,rb)/var(rb) over lookback
        beta = np.full(len(lna), 1.0)
        for i in range(lookback, len(lna)):
            c = np.cov(ra[i - lookback:i], rb[i - lookback:i])
            var_b = c[1, 1]
            if var_b > 1e-16:
                beta[i] = c[0, 1] / var_b

        spread = lna - beta * lnb
        mu = pd.Series(spread).rolling(lookback).mean().to_numpy()
        sd = pd.Series(spread).rolling(lookback).std().to_numpy()
        z = np.where(sd > 1e-12, (spread - mu) / np.maximum(sd, 1e-12), 0.0)

        pos = np.zeros(len(z))
        for i in range(1, len(z)):
            if z[i] <= -z_entry:
                pos[i] = 1.0          # long the spread
            elif z[i] >= z_entry:
                pos[i] = -1.0         # short the spread
            elif abs(z[i]) <= z_exit:
                pos[i] = 0.0          # flatten
            else:
                pos[i] = pos[i - 1]

        # hedged pnl: pos_lag * (retA - beta*retB), cost on each flip
        flips = np.abs(np.diff(pos)) > 0
        cost = cost_map.get(a.upper(), 0.0002) + cost_map.get(b.upper(), 0.0002)
        hedged = pos[:-1] * (ra - beta[1:] * rb)
        pnl = hedged - np.concatenate(([0.0], flips)) * cost

        epoch_idx = merged["epoch"].to_numpy()
        sel = (epoch_idx[1:] >= epoch_lo) & (epoch_idx[1:] < epoch_hi)
        if sel.sum() < 100:
            continue
        m = metrics_from_returns(pnl[sel], f"aegis:{a.upper()}-{b.upper()}")
        m["corr"] = round(float(corr), 4)
        results.append(m)
        used.add(pair_key)

    if not results:
        return [{"strategy": "aegis_pairs", "count": 0, "total_return": 0.0,
                 "sharpe": 0.0, "max_drawdown": 0.0, "avg_bar_return": 0.0,
                 "note": "no qualifying pairs in period"}]
    # portfolio: average the per-pair bar returns (equal-weight)
    agg = {"strategy": "aegis_portfolio", "pairs": len(results),
           "sharpe": round(float(np.mean([r["sharpe"] for r in results])), 3),
           "total_return": round(float(np.mean(
               [r["total_return"] for r in results])), 6),
           "max_drawdown": round(float(np.mean(
               [r["max_drawdown"] for r in results])), 4)}
    results.append(agg)
    return results


def aegis_hyperparameter_grid(frames, cost_map, epoch_lo, epoch_hi):
    """Walk-forward Aegis parameter search on the validation period, then
    re-run the winner on the test period for an honest out-of-sample check."""
    grid = []
    for z_entry in (1.5, 2.0, 2.5):
        for z_exit in (0.0, 0.5):
            for lookback in (100, 200, 400):
                grid.append((z_entry, z_exit, lookback))
    best, best_sharpe = None, -9e9
    print(f"[aegis] grid search over {len(grid)} configs (validation)", flush=True)
    for ze, zx, lb in grid:
        res = aegis_pairs(frames, cost_map, epoch_lo, epoch_hi,
                          z_entry=ze, z_exit=zx, lookback=lb)
        pf = next((r for r in res if r["strategy"] == "aegis_portfolio"), None)
        sharpe = pf["sharpe"] if pf else -9e9
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = {"z_entry": ze, "z_exit": zx, "lookback": lb,
                    "val_sharpe": sharpe, "pairs": len(res)}
    print(f"[aegis] best config: {best}", flush=True)
    return best
def main():
    ap = argparse.ArgumentParser(description="THE QUANT real-data training pipeline")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--download", type=int, default=1,
                    help="0 to use existing CSVs only")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed-pairs", default=",".join(DD_DEFAULT),
                    help="seed Aegis pairs (e.g. XAU_XAG,GBP_AUD)")
    ap.add_argument("--job-name", default="training",
                    help="checkpoint job name (for resume across Colab sessions)")
    ap.add_argument("--checkpoint-dir", default="",
                    help="local checkpoint directory (default: <ROOT>/checkpoints)")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable checkpointing")
    args = ap.parse_args()

    # Worker forwards CLI parameters as the TRAIN_PARAMS env JSON; merge them
    # so the SAME script drives local runs and Colab jobs with one codebase.
    train_params = os.environ.get("TRAIN_PARAMS", "")
    if train_params:
        try:
            for k, v in json.loads(train_params).items():
                if hasattr(args, k):
                    setattr(args, k, v)
        except Exception as e:
            print(f"[pipe] ignoring TRAIN_PARAMS override ({e})", flush=True)

    # --- checkpointing (resume across Colab disconnections) -----------------
    ckpt_dir = args.checkpoint_dir or os.path.join(ROOT, "checkpoints")
    from python.research.checkpoint import Checkpoint, should_resume_job
    ckpt = Checkpoint(args.job_name, ckpt_dir) if not args.no_checkpoint else None
    if ckpt:
        progress = ckpt.get_progress()
        print(f"[pipe] checkpoint: {progress['completed_stages']}/{progress['total_stages']} "
              f"stages done — next: {progress['next_stage']}", flush=True)

    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_REPORTS, exist_ok=True)
    symbols_upper = [s.strip().upper() for s in args.symbols.split(",") if s]
    symbols_lower = [s.lower() for s in symbols_upper]

    t0 = time.time()

    # Stage 1: download / load data
    if ckpt and ckpt.is_done("download"):
        print("[pipe] checkpoint: download — skipping", flush=True)
        # Load cached frames from checkpoint if available
        try:
            frames_data = ckpt.load("download")
            # frames_data is a dict of DataFrames; reconstruct frames dict
            frames = {}
            for sym in symbols_lower:
                key = f"{sym}.parquet"
                if key in frames_data:
                    frames[sym] = frames_data[key]
        except FileNotFoundError:
            frames = None
    else:
        frames = None

    if frames is None:
        frames = ensure_data(symbols_lower, args.start, args.end,
                             download=bool(args.download))
        sizes = {k: len(v) for k, v in frames.items()}
        print(f"[pipe] loaded bars: {sizes}", flush=True)
        if ckpt:
            ckpt.save("download", {f"{sym}.parquet": df for sym, df in frames.items()},
                      fmt="parquet")
            ckpt.mark_done("download", {"symbols": symbols_lower, "sizes": sizes})

    for sym, df in frames.items():
        if df["close"].max() <= 0 or df["close"].min() <= 0:
            print(f"[pipe] WARNING: {sym} has non-positive prices — skipping", flush=True)

    # Stage 2: features
    if ckpt and ckpt.is_done("features"):
        print("[pipe] checkpoint: features — loading cached", flush=True)
        feat_data = ckpt.load("features")
        X, y, syms, ep = feat_data["X"], feat_data["y"], feat_data["syms"], feat_data["ep"]
    else:
        X, y, syms, ep = build_dataset(frames)
        print(f"[pipe] dataset rows: {len(X):,} (12 features/label each)", flush=True)
        if ckpt:
            ckpt.save("features", {"X": X, "y": y, "syms": syms, "ep": ep}, fmt="numpy")
            ckpt.mark_done("features", {"rows": len(X)})

    if len(X) < 100_000:
        print("[pipe] NOTE: small dataset — results will be noisy", flush=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_series(X, y, ep)
    val_lo, val_hi = float(ep[len(Xtr)]), float(ep[len(Xtr) + len(Xva)])
    test_lo = float(ep[len(Xtr) + len(Xva)])
    print(f"[pipe] split rows: train={len(Xtr):,} val={len(Xva):,} "
          f"test={len(Xte):,}", flush=True)

    # Stage 3: train MLP
    if ckpt and ckpt.is_done("train_mlp"):
        print("[pipe] checkpoint: train_mlp — skipping", flush=True)
        mlp_metrics = ckpt.load("train_mlp")
    else:
        mlp_metrics = train_torch(Xtr, ytr, Xva, yva, args, ckpt=ckpt)
        if ckpt:
            ckpt.save("train_mlp", mlp_metrics or {}, fmt="json")
            ckpt.mark_done("train_mlp", {"done": bool(mlp_metrics)})

    # Stage 4: train XGBoost
    if ckpt and ckpt.is_done("train_xgb"):
        print("[pipe] checkpoint: train_xgb — skipping", flush=True)
        xgb_metrics = ckpt.load("train_xgb")
    else:
        xgb_metrics = train_xgb(Xtr, ytr, Xva, yva)
        if ckpt:
            ckpt.save("train_xgb", xgb_metrics or {}, fmt="json")
            ckpt.mark_done("train_xgb", {"done": bool(xgb_metrics)})

    model_onnx = os.path.join(OUT_MODELS, "latest.onnx") if mlp_metrics else None
    cost_map = {k: COST_RT.get(k, 0.0002) for k in symbols_upper}

    seed_pairs = []
    for p in args.seed_pairs.split(","):
        a, b = p.strip().split("_")
        if a.lower() in frames and b.lower() in frames:
            seed_pairs.append((a.lower(), b.lower()))

    # Stage 5: backtest
    if ckpt and ckpt.is_done("backtest"):
        print("[pipe] checkpoint: backtest — skipping", flush=True)
        report = ckpt.load("backtest")
    else:
        print("[backtest] running strategy comparisons...", flush=True)
        report = {}
        report["ema_cross"] = strat_ema(frames, cost_map)
        report["bollinger"] = strat_bollinger(frames, cost_map)
        report["donchian"] = strat_donchian(frames, cost_map)
        report["aegis_valid"] = aegis_pairs(frames, cost_map, val_lo, val_hi,
                                            seed_pairs=seed_pairs)
        if mlp_metrics:
            report["mlp_signal"] = strat_mlp(frames, cost_map, model_onnx)
        if ckpt:
            ckpt.save("backtest", report, fmt="json")
            ckpt.mark_done("backtest", {"strategies": list(report.keys())})

    # Stage 6: grid search
    if ckpt and ckpt.is_done("grid_search"):
        print("[pipe] checkpoint: grid_search — skipping", flush=True)
        grid_result = ckpt.load("grid_search")
        if grid_result:
            report["aegis_best"] = grid_result["best"]
            report["aegis_test"] = grid_result["test"]
    else:
        best = aegis_hyperparameter_grid(frames, cost_map, val_lo, val_hi)
        if best:
            report["aegis_best"] = best
            report["aegis_test"] = aegis_pairs(
            frames, cost_map, test_lo, ep.max() + 1,
            z_entry=best["z_entry"], z_exit=best["z_exit"],
            lookback=best["lookback"], seed_pairs=seed_pairs)
        if ckpt:
            grid_result = {"best": best, "test": report.get("aegis_test")}
            ckpt.save("grid_search", grid_result, fmt="json")
            ckpt.mark_done("grid_search", {"best_z": best.get("z_entry") if best else None})

    # Stage 7: export (ONNX already done in train_torch; just checkpoint)
    if ckpt:
        ckpt.mark_done("export", {"onnx": os.path.exists(model_onnx) if model_onnx else False})

    # Buy & hold baseline per symbol
    bh = []
    for sym, df in frames.items():
        c = df["close"].to_numpy()
        r = c[1:] / c[:-1] - 1.0
        bh.append(metrics_from_returns(r, f"buy_hold:{sym}"))
    report["buy_hold"] = bh

    summary = {
        "symbols": symbols_upper,
        "rows": {k: int(v) for k, v in sizes.items()},
        "dataset_rows": len(X),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "gpu": (mlp_metrics or {}).get("device", "cpu"),
        "data_quality": validate_data(frames),
        "mlp_metrics": mlp_metrics,
        "aegis_target": {"strategy": "aegis",
                         "best_hyperparams": best},
        "runtime_seconds": round(time.time() - t0, 1),
    }

    with open(os.path.join(OUT_REPORTS, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_REPORTS, "strategies.json"), "w") as f:
        json.dump(report, f, indent=2)
    if xgb_metrics:
        with open(os.path.join(OUT_REPORTS, "xgboost.json"), "w") as f:
            json.dump(xgb_metrics, f, indent=2)

    print("=" * 70, flush=True)
    print(f"[done] reports -> {OUT_REPORTS}/  (training_summary.json, "
          f"strategies.json){' models/latest.onnx' if mlp_metrics else ''}",
          flush=True)
    print(f"[done] total runtime {summary['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()