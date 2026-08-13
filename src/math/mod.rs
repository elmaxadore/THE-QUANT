//! # Mathematical & Risk Upgrade Module (v3.0 "Prometheus")
//!
//! Advanced quantitative math used across the risk engine, strategy scoring,
//! and model optimization. All implementations are self-contained in pure Rust.

pub mod kelly;
pub mod cvar;
pub mod bayesian_opt;

pub use kelly::{KellyCalculator, KellyResult};
pub use cvar::{CvarCalculator, CvarResult};
pub use bayesian_opt::{BayesianOptimizer, ParamPoint, ParamSpace};

/// The math module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const MATH_MEMORY_BUDGET_PCT: f64 = 0.5;

/// A combined risk assessment using Kelly + CVaR.
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub kelly: KellyResult,
    pub cvar: Option<CvarResult>,
    /// Recommended position size as a fraction of equity.
    pub recommended_fraction: f64,
    /// Whether the trade is approved.
    pub approved: bool,
}

/// Compute a combined risk assessment for a trade.
pub fn assess_risk(
    win_rate: f64,
    win_loss_ratio: f64,
    n_trades: u32,
    returns: &[f64],
    kelly_fraction: f64,
    cvar_confidence: f64,
    max_cvar_budget: f64,
) -> RiskAssessment {
    let kelly = KellyCalculator::new(kelly_fraction, 30).calculate(win_rate, win_loss_ratio, n_trades);
    let cvar_calc = CvarCalculator::new(cvar_confidence);
    let cvar = cvar_calc.compute(returns);

    // Recommended fraction: min of Kelly fraction and CVaR-limited fraction
    let mut recommended = kelly.recommended;
    if let Some(cvar_result) = &cvar {
        let cvar_limited = cvar_calc.max_position_fraction(cvar_result.cvar, max_cvar_budget);
        recommended = recommended.min(cvar_limited);
    }

    let approved = kelly.is_positive_edge && recommended > 0.0;

    RiskAssessment {
        kelly,
        cvar,
        recommended_fraction: recommended,
        approved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_risk() {
        let returns: Vec<f64> = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.005, -0.015, 0.02];
        let assessment = assess_risk(0.6, 1.5, 100, &returns, 0.5, 0.95, 0.02);
        assert!(assessment.recommended_fraction >= 0.0);
        assert!(assessment.cvar.is_some());
    }
}
