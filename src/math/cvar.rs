//! # CVaR (Conditional Value at Risk)
//!
//! Computes the Conditional Value at Risk (Expected Shortfall) of a return
//! distribution. CVaR is the expected loss beyond the VaR threshold — a more
//! robust risk measure than VaR because it captures tail losses.
//!
//! Used by the risk engine to:
//!   - Set maximum acceptable per-strategy tail risk
//!   - Reject strategies with excessive tail losses
//!   - Size positions so that CVaR stays within account limits

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// A CVaR calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvarResult {
    /// The value at risk at the given confidence (as a loss fraction, positive).
    pub var: f64,
    /// Conditional value at risk (expected tail loss, positive).
    pub cvar: f64,
    /// Number of tail observations used.
    pub tail_count: usize,
    /// Mean return of the distribution.
    pub mean_return: f64,
    /// Volatility of returns.
    pub volatility: f64,
}

/// CVaR calculator.
#[derive(Debug, Clone)]
pub struct CvarCalculator {
    /// Confidence level (e.g., 0.95 = 95%). Default 0.95.
    pub confidence: f64,
}

impl Default for CvarCalculator {
    fn default() -> Self {
        Self { confidence: 0.95 }
    }
}

impl CvarCalculator {
    pub fn new(confidence: f64) -> Self {
        Self {
            confidence: confidence.clamp(0.5, 0.999),
        }
    }

    /// Compute CVaR from a series of historical returns.
    ///
    /// Returns are losses-as-positive (i.e., a return of -0.02 becomes 0.02).
    pub fn compute(&self, returns: &[f64]) -> Option<CvarResult> {
        if returns.is_empty() {
            return None;
        }

        // Convert to losses (positive = loss)
        let mut losses: Vec<f64> = returns.iter().map(|r| -r).collect();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = losses.len();
        let tail_idx = ((1.0 - self.confidence) * n as f64).ceil() as usize;
        let tail_idx = tail_idx.max(1).min(n);

        // VaR = tail_idx-th largest loss
        let var = losses[tail_idx - 1];

        // CVaR = mean of the worst (tail) losses
        let tail_slice = &losses[0..tail_idx];
        let cvar = tail_slice.iter().sum::<f64>() / tail_slice.len() as f64;

        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let volatility = if n > 1 {
            let var_sum = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>();
            (var_sum / (n - 1) as f64).sqrt()
        } else {
            0.0
        };

        Some(CvarResult {
            var,
            cvar,
            tail_count: tail_idx,
            mean_return,
            volatility,
        })
    }

    /// Compute CVaR analytically assuming normally distributed returns.
    pub fn compute_normal(&self, mean: f64, volatility: f64) -> Option<CvarResult> {
        if volatility < 0.0 {
            return None;
        }
        // Standard normal quantile at confidence
        let z = normal_quantile(self.confidence);
        let var = -(mean - z * volatility); // loss = -(mean - z*vol)
        // Expected shortfall for normal: mean + phi(z)/(1-conf) * vol, as a loss
        let phi = normal_pdf(z);
        let es = mean + (phi / (1.0 - self.confidence)) * volatility;
        let cvar = -es;

        Some(CvarResult {
            var: var.max(0.0),
            cvar: cvar.max(0.0),
            tail_count: 0,
            mean_return: mean,
            volatility,
        })
    }

    /// Compute the max position size for a target CVaR budget.
    ///
    /// Given the portfolio CVaR (as a fraction of equity), calculates the
    /// maximum fraction of equity to deploy so that tail loss stays within
    /// `cvar_budget`.
    pub fn max_position_fraction(&self, cvar: f64, cvar_budget: f64) -> f64 {
        if cvar <= 0.0 {
            return 1.0;
        }
        (cvar_budget / cvar).clamp(0.0, 1.0)
    }
}

/// Standard normal CDF inverse (Acklam's approximation).
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
    let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];

    let plow = 0.02425;
    let phigh = 1.0 - plow;

    let q = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };

    q
}

/// Standard normal PDF.
fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvar_historical() {
        let calc = CvarCalculator::default();
        let returns: Vec<f64> = vec![-0.01, 0.02, -0.03, 0.01, -0.02, 0.005, -0.04, 0.015, -0.025, 0.03];
        let result = calc.compute(&returns).unwrap();
        assert!(result.cvar > 0.0);
        assert!(result.var >= 0.0);
        assert!(result.cvar >= result.var);
    }

    #[test]
    fn test_cvar_normal() {
        let calc = CvarCalculator::new(0.95);
        let result = calc.compute_normal(0.001, 0.02).unwrap();
        assert!(result.cvar > 0.0);
        assert!(result.var >= 0.0);
    }

    #[test]
    fn test_max_position_fraction() {
        let calc = CvarCalculator::default();
        // If CVaR = 2% and budget = 1%, max fraction = 0.5
        let frac = calc.max_position_fraction(0.02, 0.01);
        assert!((frac - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_returns() {
        let calc = CvarCalculator::default();
        assert!(calc.compute(&[]).is_none());
    }
}
