//! # Kelly Criterion with Uncertainty
//!
//! Implements the classic Kelly criterion for position sizing, extended to
//! handle estimation uncertainty. When the edge estimate has high variance,
//! the fraction is shrunk toward the risk-averse Kelly (fractional Kelly).
//!
//! ## Formulae
//! - **Full Kelly**: f* = p - (1-p)/b = (p(b+1) - 1) / b
//!   where p = win probability, b = win/loss ratio
//! - **Uncertainty-adjusted**: f_un = f* / (1 + κ)
//!   where κ = uncertainty scaling (based on estimation variance)
//! - **Fractional Kelly**: f_frac = q × f*
//!   where q is the conservative fraction (default 0.25–0.5)

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// A Kelly criterion calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyResult {
    /// Full Kelly fraction (can be negative if edge is negative).
    pub full_kelly: f64,
    /// Uncertainty-adjusted Kelly fraction.
    pub adjusted_kelly: f64,
    /// Recommended fractional Kelly fraction for trading.
    pub recommended: f64,
    /// Edge per trade (expected value in R-multiples).
    pub edge: f64,
    /// Whether the edge is positive (tradeable).
    pub is_positive_edge: bool,
    /// Uncertainty parameter κ.
    pub uncertainty: f64,
    /// Confidence in the estimate (0..1).
    pub confidence: f64,
}

/// Kelly calculator with uncertainty handling.
#[derive(Debug, Clone)]
pub struct KellyCalculator {
    /// Conservative fraction of Kelly to use (0..1). Default 0.5.
    pub conservative_fraction: f64,
    /// Minimum win-rate samples before the estimate is trusted.
    pub min_samples: u32,
}

impl Default for KellyCalculator {
    fn default() -> Self {
        Self {
            conservative_fraction: 0.5,
            min_samples: 30,
        }
    }
}

impl KellyCalculator {
    pub fn new(conservative_fraction: f64, min_samples: u32) -> Self {
        Self {
            conservative_fraction: conservative_fraction.clamp(0.0, 1.0),
            min_samples,
        }
    }

    /// Compute the Kelly fraction from win rate and win/loss ratio.
    ///
    /// - `win_rate`: probability of winning (p)
    /// - `win_loss_ratio`: average win / average loss (b)
    /// - `n_trades`: number of samples (for uncertainty)
    pub fn calculate(&self, win_rate: f64, win_loss_ratio: f64, n_trades: u32) -> KellyResult {
        let p = win_rate.clamp(0.0, 1.0);
        let b = win_loss_ratio.max(0.0);
        let n = n_trades.max(1);

        // Full Kelly: f* = (p(b+1) - 1) / b
        let full_kelly = if b > 0.0 {
            (p * (b + 1.0) - 1.0) / b
        } else {
            // No wins available — treat as zero edge
            2.0 * p - 1.0
        };

        // Edge per trade
        let edge = p * b - (1.0 - p);

        // Uncertainty: standard error of the win rate estimate
        let se = (p * (1.0 - p) / n as f64).sqrt();
        // κ scales with relative uncertainty; saturates at low sample counts
        let kappa = (se / p.max(1e-6)).min(10.0);

        // Uncertainty-adjusted: shrink toward risk-free
        let adjusted = if n < self.min_samples {
            // Low sample count → scale down by sample confidence
            let sample_confidence = (n as f64 / self.min_samples as f64).min(1.0);
            full_kelly * sample_confidence / (1.0 + kappa)
        } else {
            full_kelly / (1.0 + kappa * 0.5)
        };

        // Conservative fractional Kelly
        let recommended = adjusted * self.conservative_fraction;

        let confidence = (1.0 / (1.0 + kappa)).clamp(0.0, 1.0);

        KellyResult {
            full_kelly,
            adjusted_kelly: adjusted,
            recommended: recommended.max(0.0),
            edge,
            is_positive_edge: edge > 0.0,
            uncertainty: kappa,
            confidence,
        }
    }

    /// Compute position size in units of account equity.
    pub fn position_size(&self, kelly: &KellyResult, account_equity: f64) -> f64 {
        if !kelly.is_positive_edge {
            return 0.0;
        }
        account_equity * kelly.recommended
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_kelly() {
        let calc = KellyCalculator::default();
        // p = 0.55, b = 1.0 → f* = 0.10
        let result = calc.calculate(0.55, 1.0, 100);
        assert!((result.full_kelly - 0.10).abs() < 0.01);
        assert!(result.is_positive_edge);
    }

    #[test]
    fn test_negative_edge() {
        let calc = KellyCalculator::default();
        // p = 0.4, b = 1.0 → f* = -0.2 (negative edge)
        let result = calc.calculate(0.4, 1.0, 100);
        assert!(!result.is_positive_edge);
        assert!(result.full_kelly < 0.0);
        assert_eq!(result.recommended, 0.0);
    }

    #[test]
    fn test_uncertainty_reduces_size() {
        let calc = KellyCalculator::default();
        // Same edge but different sample counts
        let confident = calc.calculate(0.55, 1.0, 1000);
        let uncertain = calc.calculate(0.55, 1.0, 10);
        assert!(uncertain.adjusted_kelly < confident.adjusted_kelly);
    }

    #[test]
    fn test_position_size() {
        let calc = KellyCalculator::default();
        let result = calc.calculate(0.6, 1.5, 200);
        let size = calc.position_size(&result, 100_000.0);
        assert!(size >= 0.0);
        assert!(size <= 100_000.0);
    }
}
