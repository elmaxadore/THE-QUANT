//! SIMULATED FEED — synthetic OHLCV bar generator.
//!
//! The Quant is designed for a live MT5 bridge over ZeroMQ. For development,
//! testing, and lean footprints this module emits a realistic geometric random
//! walk, which keeps the whole engine testable end to end without a broker.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// An OHLCV bar (immutable, cheap to copy; f64 for speed).
#[derive(Debug, Clone)]
pub struct Bar {
    pub symbol: String,
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A configurable simulator that can be seeded for reproducible tests.
pub struct SimFeed {
    rng: StdRng,
    symbol: String,
    /// base price for the symbol (e.g. EURUSD ~1.08)
    price: f64,
    /// per-bar drift
    drift: f64,
    /// per-bar volatility (fraction of price)
    vol: f64,
    bar: i64,
}

impl SimFeed {
    pub fn new(symbol: &str, base_price: f64, seed: u64) -> Self {
        SimFeed {
            rng: StdRng::seed_from_u64(seed),
            symbol: symbol.to_string(),
            price: base_price,
            drift: 0.000_005,
            vol: base_price * 0.0009,
            bar: 0,
        }
    }

    /// Set per-bar volatility (as fraction of price). Default ~0.0009.
    pub fn with_vol(mut self, frac: f64) -> Self {
        self.vol = self.price * frac;
        self
    }

    /// Generate the next bar (random walk with occasional regime-vol shifts).
    pub fn next_bar(&mut self) -> Bar {
        // Occasionally simulate a volatility burst / trend regime.
        let burst = self.rng.gen_bool(0.02);
        let vol = if burst { self.vol * 3.0 } else { self.vol };
        let regime_drift = if self.rng.gen_bool(0.35) { self.drift * 4.0 } else { self.drift };

        let shock: f64 = self.rng.gen_range(-1.0..1.0);
        let close = self.price * (1.0 + regime_drift + shock * vol);
        let open = self.price;
        let high = open.max(close) * (1.0 + self.rng.gen_range(0.0..0.001));
        let low = open.min(close) * (1.0 - self.rng.gen_range(0.0..0.001));
        let volume = self.rng.gen_range(100.0..2000.0) * (if burst { 2.0 } else { 1.0 });

        let bar = Bar { symbol: self.symbol.clone(), time: self.bar, open, high, low, close, volume };
        self.price = close;
        self.bar += 1;
        bar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_is_deterministic_with_seed() {
        let mut a = SimFeed::new("EURUSD", 1.08, 42);
        let mut b = SimFeed::new("EURUSD", 1.08, 42);
        for _ in 0..100 {
            let (ba, bb) = (a.next_bar(), b.next_bar());
            assert_eq!(ba.close, bb.close);
            assert!((ba.high - bb.high).abs() < 1e-12);
        }
        // prices should have moved; not frozen (different seeds)
        let c0 = SimFeed::new("EURUSD", 1.08, 1).next_bar().close;
        let c1 = SimFeed::new("EURUSD", 1.08, 2).next_bar().close;
        assert_ne!(c0, c1);
    }
}