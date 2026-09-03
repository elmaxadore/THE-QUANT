#!/usr/bin/env python3
"""Regression tests for feature engineering parity with src/features.rs."""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_pipeline import compute_features, compute_features_fast


def _make_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 1.08 + np.cumsum(rng.standard_normal(n) * 0.001)
    highs = closes + np.abs(rng.standard_normal(n) * 0.0005)
    lows = closes - np.abs(rng.standard_normal(n) * 0.0005)
    volumes = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": volumes})


class TestFeatureParity(unittest.TestCase):
    """Verify Python feature computation matches src/features.rs."""

    def setUp(self):
        self.df = _make_df()
        self.f_loop = compute_features(self.df)
        self.f_fast = compute_features_fast(self.df)

    def test_f02_no_nan(self):
        """f02 (realized vol) must never be NaN (regression: cumsum included logr[0]=NaN)."""
        self.assertFalse(np.any(np.isnan(self.f_loop[:, 2])),
                         "loop f02 contains NaN")
        self.assertFalse(np.any(np.isnan(self.f_fast[:, 2])),
                         "fast f02 contains NaN")

    def test_f03_rsi_default(self):
        """f03 (RSI) must be 50.0 for insufficient data, including row 0."""
        self.assertEqual(self.f_loop[0, 3], 50.0,
                         "loop f03 row 0 should be 50.0")
        self.assertEqual(self.f_fast[0, 3], 50.0,
                         "fast f03 row 0 should be 50.0")
        # RSI must be in [0, 100] for all rows
        self.assertTrue(np.all((self.f_loop[:, 3] >= 0) & (self.f_loop[:, 3] <= 100)))
        self.assertTrue(np.all((self.f_fast[:, 3] >= 0) & (self.f_fast[:, 3] <= 100)))

    def test_f04_atr_excludes_tr0(self):
        """f04 (ATR) must exclude tr[0]=0 from the mean (matches Rust atr())."""
        # Manual: ATR at row 1 = tr[1] / 1 / close[1]
        c = self.df["close"].to_numpy()
        h = self.df["high"].to_numpy()
        l = self.df["low"].to_numpy()
        tr1 = max(h[1]-l[1], abs(h[1]-c[0]), abs(l[1]-c[0]))
        expected = tr1 / 1 / c[1]
        self.assertAlmostEqual(self.f_loop[1, 4], expected, places=10,
                               msg="loop f04 row 1")
        self.assertAlmostEqual(self.f_fast[1, 4], expected, places=10,
                               msg="fast f04 row 1")

    def test_f10_hurst_window(self):
        """f10 (Hurst) must use the same 58-element window as Rust for n=80."""
        # Rust: loop w in 2..min(n,60), for n=80 that's 58 iterations
        c = self.df["close"].to_numpy()
        agrees = 0
        for w in range(2, min(80, 60)):
            r1 = c[w] - c[w-1]
            r2 = c[w-1] - c[w-2]
            if r1 * r2 > 0:
                agrees += 1
        rust_f10 = agrees / 58
        self.assertAlmostEqual(self.f_loop[79, 10], rust_f10, places=10,
                               msg="loop f10 matches Rust for n=80")
        self.assertAlmostEqual(self.f_fast[79, 10], rust_f10, places=10,
                               msg="fast f10 matches Rust for n=80")

    def test_f10_default(self):
        """f10 must be 0.5 for insufficient data, including row 0."""
        self.assertEqual(self.f_loop[0, 10], 0.5)
        self.assertEqual(self.f_fast[0, 10], 0.5)

    def test_f10_matches_rust_fixed_window(self):
        """f10 (Hurst) must match Rust for fixed windows starting at 0."""
        c = self.df["close"].to_numpy()
        for w in [40, 60, 80, 100]:
            if w > len(c): break
            agrees = 0
            for i in range(2, min(w, 60)):
                r1 = c[i] - c[i-1]
                r2 = c[i-1] - c[i-2]
                if r1 * r2 > 0: agrees += 1
            rust_f10 = agrees / (min(w, 60) - 2) if w >= 20 else 0.5
            self.assertAlmostEqual(self.f_loop[w-1, 10], rust_f10, places=10,
                                   msg=f"loop f10 matches Rust for window={w}")

    def test_f10_fast_uses_sliding_window(self):
        """Fast f10 uses a 58-bar sliding window (different from loop's fixed window)."""
        # The fast version uses rolling(58).mean().shift(20), which for row t
        # computes the mean of sa[t-77:t-19] (58 elements).
        # Verify it's not NaN and within [0, 1].
        for t in range(79, 200):
            self.assertTrue(0.0 <= self.f_fast[t, 10] <= 1.0,
                            f"fast f10 at row {t} out of range")

    def test_loop_fast_match_production_window(self):
        """Loop and fast versions must match for the production window (t>=79)
        for all features except f10 (different window semantics)."""
        for i in range(12):
            if i == 10:
                continue  # f10 has intentionally different window semantics
            self.assertTrue(
                np.allclose(self.f_loop[79:, i], self.f_fast[79:, i],
                           atol=1e-10, equal_nan=True),
                f"f{i:02d} mismatch between loop and fast")

    def test_all_finite(self):
        """All features must be finite (no NaN or inf)."""
        for i in range(12):
            self.assertTrue(np.all(np.isfinite(self.f_loop[:, i])),
                            f"loop f{i:02d} has non-finite values")
            self.assertTrue(np.all(np.isfinite(self.f_fast[:, i])),
                            f"fast f{i:02d} has non-finite values")


if __name__ == "__main__":
    unittest.main()