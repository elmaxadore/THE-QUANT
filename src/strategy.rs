//! STRATEGY ENGINE — Layer 5 (v2.1 §8, v4.0 §4.1 discovery pipeline).
//!
//! Takes features + a model signal and converts them into concrete trade
//! intents (BUY/SELL/CLOSE) that are handed to the risk engine. Regime-aware:
//! strong trend regimes trust momentum; ranging regimes use mean reversion.

use crate::onnx::{ModelBackend, MODEL_INPUT_DIM};
use crate::regime::Regime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Buy,
    Sell,
    Close,
    /// no action / no position
    Hold,
}

impl Direction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Buy => "BUY",
            Direction::Sell => "SELL",
            Direction::Close => "CLOSE",
            Direction::Hold => "HOLD",
        }
    }
}

/// A signal emitted by the strategy engine for one symbol.
#[derive(Debug, Clone)]
pub struct Signal {
    pub direction: Direction,
    pub confidence: f64,
    pub regime: Regime,
    pub model: String,
}

/// Thresholds for acting on a model signal (configurable).
#[derive(Debug, Clone, Copy)]
pub struct StrategyParams {
    pub entry_confidence: f64, // min |signal| to open a position
    pub exit_confidence: f64,  // |signal| below this -> close
}

impl Default for StrategyParams {
    fn default() -> Self {
        StrategyParams { entry_confidence: 0.10, exit_confidence: 0.05 }
    }
}

pub struct StrategyEngine<'a> {
    model: &'a dyn ModelBackend,
    params: StrategyParams,
    // per-symbol last position direction so we can generate CLOSE.
    last_dir: Direction,
}

impl<'a> StrategyEngine<'a> {
    pub fn new(model: &'a dyn ModelBackend) -> Self {
        StrategyEngine { model, params: StrategyParams::default(), last_dir: Direction::Hold }
    }

    /// Produce a signal for the latest bar window + regime.
    pub fn evaluate(&mut self, features: &[f64; MODEL_INPUT_DIM], regime: Regime) -> Signal {
        let (raw, confidence) = self.model.predict(features);
        let model_name = self.model.name().to_string();

        // Regime-aware shaping.
        let mut signal = raw;
        match regime {
            Regime::TrendingUp => {
                // allow long bias, dampen shorts
                signal = if signal > 0.0 { signal } else { signal * 0.3 };
            }
            Regime::TrendingDown => {
                signal = if signal < 0.0 { signal } else { signal * 0.3 };
            }
            Regime::Ranging => {
                // fade extremes
                signal = -0.5 * signal;
            }
            Regime::HighVolatility => {
                // risk engine will gate this anyway; reduce conviction
                signal *= 0.5;
            }
        }

        let direction = if confidence < self.params.exit_confidence {
            Direction::Hold
        } else if confidence >= self.params.entry_confidence && signal > 0.0 {
            Direction::Buy
        } else if confidence >= self.params.entry_confidence && signal < 0.0 {
            Direction::Sell
        } else {
            Direction::Hold
        };

        let sig = Signal { direction, confidence, regime, model: model_name };
        self.last_dir = sig.direction;
        sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::RuleModel;

    #[test]
    fn strong_buy_signal_emits_buy() {
        let m = RuleModel::new();
        let mut eng = StrategyEngine::new(&m);
        let f = [0.3, 0.2, 0.001, 50.0, 0.0005, 0.02, 0.01, 0.5, 0.01, 0.6, 0.6, 0.1];
        let s = eng.evaluate(&f, Regime::TrendingUp);
        assert_eq!(s.direction, Direction::Buy);
    }
}