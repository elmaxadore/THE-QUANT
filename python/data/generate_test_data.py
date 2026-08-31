#!/usr/bin/env python3
"""Generate realistic FX test data with cointegrated pairs for Aegis testing.

Creates synthetic forex data where pairs have genuine mean-reverting spreads
(cointegration), which is what the Aegis hedged pairs strategy needs to be profitable.

Statistical properties:
  - EUR/GBP: highly correlated (share European factor)
  - AUD/NZD: highly correlated (share Oceania factor)
  - Gold/Silver: correlated (commodity factor)
  - Spreads are mean-reverting (Ornstein-Uhlenbeck process)

Output: CSV files in python/data/histdata/ that the Rust CSV feed can consume.

Usage: python3 generate_test_data.py
"""
import csv
import math
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "histdata")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BARS = 15000  # ~1.5M M5 bars equivalent of training data
DT = 1.0  # 1 bar step


def ornstein_uhlenbeck(n, theta, mu, sigma, dt, seed):
    """Generate an Ornstein-Uhlenbeck process (mean-reverting)."""
    rng = random.Random(seed)
    x = mu
    out = [x]
    for _ in range(n - 1):
        dx = theta * (mu - x) * dt + sigma * math.sqrt(dt) * rng.gauss(0, 1)
        x += dx
        out.append(x)
    return out


def generate_cointegrated_pair(sym_a, sym_b, base_a, base_b, n, seed, ou_theta, ou_sigma):
    """Generate two cointegrated series: A = base_A * exp(factor + noise_A), B = base_B * exp(factor + noise_B).

    The common factor creates correlation; the OU process creates a mean-reverting spread.
    """
    rng = random.Random(seed)
    # Common market factor (shared drift)
    factor = ornstein_uhlenbeck(n, theta=0.05, mu=0.0, sigma=0.0003, dt=DT, seed=seed)
    # Idiosyncratic noise for each leg (OU process = mean reverting)
    noise_a = ornstein_uhlenbeck(n, theta=ou_theta, mu=0.0, sigma=ou_sigma, dt=DT, seed=seed + 1000)
    noise_b = ornstein_uhlenbeck(n, theta=ou_theta, mu=0.0, sigma=ou_sigma, dt=DT, seed=seed + 2000)

    bars_a = []
    bars_b = []
    price_a = base_a
    price_b = base_b

    for i in range(n):
        # Log returns: common factor + idiosyncratic
        ret_a = factor[i] * 0.8 + noise_a[i]
        ret_b = factor[i] * 0.7 + noise_b[i]

        # Small random walk component for realism
        ret_a += rng.gauss(0, 0.0001)
        ret_b += rng.gauss(0, 0.0001)

        price_a *= math.exp(ret_a)
        price_b *= math.exp(ret_b)

        # Generate OHLC from close
        vol = abs(ret_a) * price_a
        high_a = price_a + vol * 0.3
        low_a = price_a - vol * 0.3
        open_a = price_a * (1 + rng.gauss(0, 0.00005))

        vol_b = abs(ret_b) * price_b
        high_b = price_b + vol_b * 0.3
        low_b = price_b - vol_b * 0.3
        open_b = price_b * (1 + rng.gauss(0, 0.00005))

        t = i
        bars_a.append((t, sym_a, open_a, high_a, low_a, price_a))
        bars_b.append((t, sym_b, open_b, high_b, low_b, price_b))

    return bars_a, bars_b


def write_csv(filepath, bars):
    """Write bars to CSV format matching histdata.com format."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume"])
        for t, sym, o, h, l, c in bars:
            w.writerow([
                str(t),
                str(t),
                f"{o:.6f}",
                f"{h:.6f}",
                f"{l:.6f}",
                f"{c:.6f}",
                "1000",
            ])


def main():
    print(f"[data] generating {N_BARS} bars per symbol...")

    # Generate cointegrated pairs
    print("[data] EURUSD / GBPUSD (highly correlated, cointegrated)...")
    eurusd, gbpusd = generate_cointegrated_pair(
        "EURUSD", "GBPUSD", 1.08, 1.26, N_BARS, seed=42,
        ou_theta=0.15, ou_sigma=0.0004,  # theta=0.15 → spread reverts in ~7 bars
    )
    write_csv(os.path.join(OUTPUT_DIR, "eurusd.csv"), eurusd)
    write_csv(os.path.join(OUTPUT_DIR, "gbpusd.csv"), gbpusd)

    print("[data] AUDUSD / NZDUSD (highly correlated, cointegrated)...")
    audusd, nzdusd = generate_cointegrated_pair(
        "AUDUSD", "NZDUSD", 0.65, 0.60, N_BARS, seed=123,
        ou_theta=0.12, ou_sigma=0.0005,
    )
    write_csv(os.path.join(OUTPUT_DIR, "audusd.csv"), audusd)
    write_csv(os.path.join(OUTPUT_DIR, "nzdusd.csv"), nzdusd)

    print("[data] XAUUSD / XAGUSD (precious metals, correlated)...")
    xauusd, xagusd = generate_cointegrated_pair(
        "XAUUSD", "XAGUSD", 2400.0, 28.0, N_BARS, seed=456,
        ou_theta=0.10, ou_sigma=0.0006,
    )
    write_csv(os.path.join(OUTPUT_DIR, "xauusd.csv"), xauusd)
    write_csv(os.path.join(OUTPUT_DIR, "xagusd.csv"), xagusd)

    print(f"[data] done. Files in {OUTPUT_DIR}")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}: {os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
