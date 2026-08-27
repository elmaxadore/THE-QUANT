//! MARKET REGIME DETECTION — Layer 4 (v2.1 §7).
//!
//! A lightweight regime classifier on top of the feature vector. Full GMM/HMM
//! lives in `can_run_lab` tiers; here we provide a deterministic, cheap
//! classifier with the same taxonomy that scales to every tier.

use crate::features::{compute_features, N_FEATURES};
use crate::simfeed::Bar;

/// The v2.1 regime taxonomy (subset used here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    TrendingUp,
    TrendingDown,
    Ranging,
    HighVolatility,
}

impl Regime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::TrendingUp => "TRENDING_UP",
            Regime::TrendingDown => "TRENDING_DOWN",
            Regime::Ranging => "RANGING",
            Regime::HighVolatility => "HIGH_VOLATILITY",
        }
    }
}

/// Classify the latest window into a regime using feature thresholds.
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    /// recent regime history (for the dashboard / meta-learner)
    history: Vec<Regime>,
    pub max_history: usize,
}

impl RegimeDetector {
    pub fn new(max_history: usize) -> Self {
        RegimeDetector { history: Vec::new(), max_history }
    }

    pub fn latest(&self) -> Option<Regime> {
        self.history.last().copied()
    }

    pub fn history(&self) -> &[Regime] {
        &self.history
    }

    /// Classify the given bar window and push to history.
    pub fn update(&mut self, bars: &[Bar]) -> Regime {
        let f: [f64; N_FEATURES] = compute_features(bars);
        // Feature indices: 2=vol, 1=5-bar log return, 0=1-bar return,
        // 5=EMA diff, 7=volume zscore.
        let vol = f[2];
        let momentum = f[1];
        let ema_trend = f[5];

        let regime = if vol > 0.004 {
            Regime::HighVolatility
        } else if ema_trend > 0.0 && r5_momentum(momentum) > 0.0 {
            Regime::TrendingUp
        } else if ema_trend < 0.0 && r5_momentum(momentum) < 0.0 {
            Regime::TrendingDown
        } else {
            Regime::Ranging
        };

        self.history.push(regime);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        regime
    }
}

fn r5_momentum(r5: f64) -> f64 {
    r5
}