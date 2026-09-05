#!/usr/bin/env python3
"""Download free forex data from histdata.com for Aegis pairs testing.

Downloads M1 (1-minute) bar data for FX pairs and saves as CSV.
histdata.com provides free historical forex data without API key.

Pairs we need (from PAIR_UNIVERSE):
  EURUSD, GBPUSD, AUDUSD, NZDUSD (FX majors)
  XAUUSD, XAGUSD (metals - may not be available)

Usage: python3 download_histdata.py
"""
import csv
import os
import sys
import time
import zipfile
import io

import requests

# histdata.com provides free foreX data in CSV format
HISTDATA_BASE = "https://www.histdata.com/download-free-forex-data"

# Pairs we want to download (histdata.com format)
PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd"]
CURRENT_YEAR = 2025

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "histdata")


def download_pair(pair: str, year: int) -> list:
    """Download M1 data for a pair from histdata.com. Returns list of bars."""
    url = f"https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"
    print(f"[data] downloading {pair.upper()} {year} ...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes",
    })

    # Step 1: Get the page with the download form
    try:
        resp = session.get(f"https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{pair}/{year}", timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[data] failed to fetch page for {pair}: {e}")
        return []

    # Step 2: Submit the download form
    form_data = {
        "tk": _extract_input(resp.text, "tk"),
        "date": _extract_input(resp.text, "date"),
        "datetime": _extract_input(resp.text, "datetime"),
        "platform": "ascii",
        "timeframe": "M1",
        "fxpair": pair,
    }

    if not form_data["tk"]:
        print(f"[data] could not extract form token for {pair}")
        return []

    try:
        resp2 = session.post(url, data=form_data, timeout=30)
        resp2.raise_for_status()
    except Exception as e:
        print(f"[data] failed to download {pair}: {e}")
        return []

    content_type = resp2.headers.get("Content-Type", "")
    if "zip" not in content_type.lower() and not resp2.content[:2] == b"PK":
        print(f"[data] unexpected content for {pair}: {content_type}")
        return []

    # Step 3: Extract CSV from zip
    bars = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(resp2.content))
        for name in zf.namelist():
            if name.endswith(".csv") or name.endswith(".txt"):
                with zf.open(name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                    for line in text.strip().splitlines():
                        parts = line.strip().split(";")
                        if len(parts) >= 5:
                            # histdata CSV format: Date,Time,Open,High,Low,Close
                            # Date format: YYYYMMMDD
                            date_str = parts[0]
                            time_str = parts[1]
                            try:
                                o, h, l, c = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                                bars.append((date_str, time_str, o, h, l, c))
                            except (ValueError, IndexError):
                                continue
    except Exception as e:
        print(f"[data] failed to parse {pair}: {e}")
        return []

    print(f"[data] {pair.upper()}: {len(bars)} bars")
    return bars


def _extract_input(html: str, name: str) -> str:
    """Extract a hidden input value from HTML."""
    import re
    m = re.search(rf'<input[^>]*name="{name}"[^>]*value="([^"]*)"', html)
    return m.group(1) if m else ""


def save_pair_csv(pair: str, bars: list, output_dir: str):
    """Save bars as CSV for the Rust feed to consume."""
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{pair.lower()}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume"])
        for date_str, time_str, o, h, l, c in bars:
            w.writerow([date_str, time_str, f"{o:.6f}", f"{h:.6f}", f"{l:.6f}", f"{c:.6f}", "1000"])
    print(f"[data] saved {out} ({len(bars)} bars)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for pair in PAIRS:
        bars = download_pair(pair, CURRENT_YEAR)
        if bars:
            save_pair_csv(pair, bars, OUTPUT_DIR)
        else:
            print(f"[data] no data for {pair}")
        time.sleep(2)  # be polite to the server

    print(f"[data] done. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
