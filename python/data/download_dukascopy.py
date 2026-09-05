#!/usr/bin/env python3
"""THE QUANT — real historical tick data from Dukascopy (free, no API key).

Downloads the official Dukascopy tick feed (LZMA-compressed .bi5 files, one per
symbol/hour) and produces:

  1. Aggregated M1/M5 OHLCV bars in python/data/histdata/<symbol>.csv
     — the exact CSV format the Rust `CsvFeed` (src/simfeed.rs) consumes for
     `data_source = "csv"` backtests and paper runs.
  2. (optional, --keep-ticks) a raw-tick cache in python/data/ticks/<symbol>/
     as per-day gzip CSVs for tick-level research/microstructure studies.

URL: https://datafeed.dukascopy.com/datafeed/<SYM>/<YYYY>/<MM-1>/<DD>/<HH>h_ticks.bi5
Record (20 bytes, big-endian): ms-offset u32, ask u32, bid u32, ask-vol f32,
bid-vol f32. Fixed-point prices: FX /1e5, metals (XAUUSD, XAGUSD) /1e3.

Usage:
  python3 python/data/download_dukascopy.py                     # 2024->now, M5
  python3 python/data/download_dukascopy.py --start 2023-01-01 --timeframes M1,M5
  python3 python/data/download_dukascopy.py --symbols eurusd,gbpusd --keep-ticks
"""
import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import lzma
import os
import struct
import sys
import threading
import time

import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
HERE = os.path.dirname(os.path.abspath(__file__))
HISTDATA_DIR = os.path.join(HERE, "histdata")
TICKS_DIR = os.path.join(HERE, "ticks")
CACHE_DIR = os.path.join(HERE, "histdata_cache")

# symbol -> fixed-point divisor (verified against known 2025 prices)
POINT_VALUE = {
    "EURUSD": 1e5, "GBPUSD": 1e5, "AUDUSD": 1e5, "NZDUSD": 1e5,
    "USDJPY": 1e3, "USDCHF": 1e5, "USDCAD": 1e5,
    "XAUUSD": 1e3, "XAGUSD": 1e3,
    "US100": 1.0, "US30": 1.0, "US500": 1.0, "DEUIDX": 1.0,
}
DEFAULT_SYMBOLS = ["eurusd", "gbpusd", "audusd", "nzdusd", "xauusd", "xagusd"]

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}
HOUR_MS = struct.Struct(">IIIff")

# One persistent TLS session per worker thread (connection reuse matters —
# a fresh handshake per file is 5-10x slower on flaky networks).
_tls = threading.local()


def _session():
    if getattr(_tls, "s", None) is None:
        s = requests.Session()
        s.headers.update({
            # Full browser UA — Dukascopy's WAF serves 503 to odd-looking agents.
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/125.0 Safari/537.36"),
            "Accept": "*/*",
            "Referer": "https://www.dukascopy.com/",
            "Accept-Encoding": "identity",
        })
        _tls.s = s
    return _tls.s


def http_get(url, retries=6, timeout=25):
    """GET with retries. Returns body bytes, None on 404/permanent failure."""
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 404:
                return None  # no data for this hour (weekend/holiday)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503, 403):
                # rate limited — exponential backoff (up to ~40s), then retry
                time.sleep(min(40, 3 * (2 ** attempt)))
            else:
                return None
        except requests.RequestException:
            time.sleep(1 + attempt)
    return None


def fetch_hour_ticks(symbol, divisor, hour):
    """Download + parse one hour of ticks. Returns [(epoch_s, mid, spread)]."""
    url = (f"{BASE}/{symbol}/{hour.year}/{hour.month - 1:02d}/"
           f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5")
    body = http_get(url)
    if not body:
        return None
    try:
        raw = lzma.decompress(body)
    except lzma.LZMAError:
        return None
    if len(raw) < HOUR_MS.size:
        return []
    base = hour.replace(minute=0, second=0, microsecond=0,
                        tzinfo=dt.timezone.utc).timestamp()
    ticks = []
    for off in range(0, len(raw) - HOUR_MS.size + 1, HOUR_MS.size):
        ms, ask, bid, _, _ = HOUR_MS.unpack_from(raw, off)
        ticks.append((base + ms / 1000.0,
                      (ask + bid) / 2.0 / divisor,
                      max(ask - bid, 0) / divisor))
    return ticks


def save_day_ticks(symbol, day, ticks):
    """Append raw ticks to the per-day gzip cache (tick-level research)."""
    d = os.path.join(TICKS_DIR, symbol.lower())
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{day:%Y%m%d}.csv.gz")
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mid", "spread"])
        for t, px, spread in ticks:
            w.writerow([f"{t:.3f}", f"{px:.6f}", f"{spread:.6f}"])


def process_symbol_month(sym, year, month, keep_ticks, workers):
    """Download one symbol-month of ticks; cache as gz CSV. Returns cache path."""
    out_dir = os.path.join(CACHE_DIR, sym.lower())
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"{year}{month:02d}.csv.gz")
    if os.path.exists(cache_path):
        return cache_path  # resume: month already fetched

    divisor = POINT_VALUE.get(sym.upper(), 1e5)
    first = dt.datetime(year, month, 1)
    nxt = dt.datetime(year + (month == 12), (month % 12) + 1, 1)
    hours = [first + dt.timedelta(hours=h)
             for h in range(int((nxt - first).total_seconds()) // 3600)]

    rows = []
    failed = list(hours)
    for pass_no in (1, 2):
        if not failed:
            break
        if pass_no == 2 and failed:
            print(f"[data] {sym.upper()} {year}-{month:02d}: "
                  f"retrying {len(failed)} failed hours", flush=True)
            time.sleep(5)
        pending, failed = failed, []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_hour_ticks, sym, divisor, h): h
                       for h in pending}
            for fut in cf.as_completed(futures):
                h = futures[fut]
                ticks = fut.result()
                if ticks is None:
                    failed.append(h)
                    continue
                rows.extend(ticks)
                if keep_ticks and ticks:
                    save_day_ticks(sym, h.date(), ticks)
    if failed:
        print(f"[data] {sym.upper()} {year}-{month:02d}: WARNING "
              f"{len(failed)} hours unrecoverable (treated as no-data)",
              flush=True)

    rows.sort(key=lambda r: r[0])
    with gzip.open(cache_path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mid", "spread"])
        for t, px, spread in rows:
            w.writerow([f"{t:.3f}", f"{px:.6f}", f"{spread:.6f}"])
    print(f"[data] {sym.upper()} {year}-{month:02d}: {len(rows):>9,} ticks", flush=True)
    return cache_path


def build_bars(cache_files, tf_minutes):
    """Re-aggregate cached mid-price ticks into OHLCV bars."""
    bars = []
    last_key = None
    step = tf_minutes * 60
    for path in cache_files:
        with gzip.open(path, "rt", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                t, px = float(row[0]), float(row[1])
                key = int(t // step) * step
                if key == last_key and bars:
                    b = bars[-1]
                    b[1] = max(b[1], px); b[2] = min(b[2], px)
                    b[3] = px; b[4] += 1
                else:
                    bars.append([key, px, px, px, px, 1])
                    last_key = key
    return bars


def write_histdata_csv(sym, bars, tf_label):
    """Write bars in the format the Rust CsvFeed expects (header + OHLCV)."""
    os.makedirs(HISTDATA_DIR, exist_ok=True)
    path = os.path.join(HISTDATA_DIR, f"{sym.lower()}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume"])
        for t, o, h, l, c, n in bars:
            ts = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
            w.writerow([f"{ts:%Y-%m-%d}", f"{ts:%H:%M}",
                        f"{o:.6f}", f"{h:.6f}", f"{l:.6f}", f"{c:.6f}", n])
    print(f"[data] wrote {path}: {len(bars):,} {tf_label} bars", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Dukascopy tick data -> M1/M5 bars")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--timeframes", default="M5", help="e.g. M1,M5")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent month-hour fetches (Dukascopy throttles "
                         "aggressive IPs -> keep low)")
    ap.add_argument("--keep-ticks", action="store_true",
                    help="also cache raw ticks under python/data/ticks/")
    args = ap.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    end = dt.datetime.strptime(args.end, "%Y-%m-%d")
    tfs = [t.strip().upper() for t in args.timeframes.split(",")]
    for tf in tfs:
        if tf not in TF_MINUTES:
            sys.exit(f"unknown timeframe {tf} (choose from {', '.join(TF_MINUTES)})")

    months = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end.date():
        months.append((cur.year, cur.month))
        cur = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)

    for sym in syms:
        print(f"[data] === {sym.upper()} ===", flush=True)
        for year, month in months:
            if dt.date(year, month, 1) > end.date():
                break
            process_symbol_month(sym, year, month, args.keep_ticks, args.workers)
        cdir = os.path.join(CACHE_DIR, sym.lower())
        cache_files = sorted(
            os.path.join(cdir, f"{y}{m:02d}.csv.gz") for y, m in months
            if os.path.exists(os.path.join(cdir, f"{y}{m:02d}.csv.gz")))
        if not cache_files:
            print(f"[data] no data for {sym}", flush=True)
            continue
        for tf in tfs:
            bars = build_bars(cache_files, TF_MINUTES[tf])
            out_sym = sym if tf == "M5" else f"{sym}_{tf.lower()}"
            write_histdata_csv(out_sym, bars, tf)
    print("[data] done.", flush=True)


if __name__ == "__main__":
    main()
