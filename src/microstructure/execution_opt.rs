//! # Execution Optimization
//!
//! Selects an optimal execution strategy (market, TWAP, VWAP) based on order
//! size, urgency, metric, and predicted slippage. Defaults to limit orders
//! (LIMIT > STOP_LIMIT > MARKET) per the spec, with MARKET reserved for
//! emergency exits.

use crate::error::{QuantError, QuantResult};
use crate::microstructure::slippage_model::SlippagePrediction;
use serde::{Deserialize, Serialize};

/// Execution algorithm selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ExecutionAlgorithm {
    Market,
    Limit,
    StopLimit,
    Twap,
    Vwap,
    RlGuided,
}

impl ExecutionAlgorithm {
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionAlgorithm::Market => "market",
            ExecutionAlgorithm::Limit => "limit",
            ExecutionAlgorithm::StopLimit => "stop_limit",
            ExecutionAlgorithm::Twap => "twap",
            ExecutionAlgorithm::Vwap => "vwap",
            ExecutionAlgorithm::RlGuided => "rl",
        }
    }
}

/// Parameters for an execution schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSchedule {
    pub algorithm: ExecutionAlgorithm,
    /// Number of child orders to split into (for TWAP/VWAP).
    pub num_slices: u32,
    /// Slice interval in seconds (for TWAP).
    pub slice_interval_secs: u32,
    /// Urgency (0 = patient, 1 = aggressive).
    pub urgency: f64,
    /// Predicted slippage for the parent order.
    pub predicted_slippage_bps: f64,
    /// Expected total cost in bps.
    pub expected_total_cost_bps: f64,
}

/// The execution optimizer.
#[derive(Debug)]
pub struct ExecutionOptimizer {
    /// Maximum slippage threshold (bps) before switching to limit orders.
    pub max_slippage_bps: f64,
    /// Large order threshold (fraction of average depth).
    pub large_order_threshold: f64,
}

impl Default for ExecutionOptimizer {
    fn default() -> Self {
        Self {
            max_slippage_bps: 5.0,
            large_order_threshold: 0.1,
        }
    }
}

impl ExecutionOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Choose an execution algorithm for an order.
    pub fn select_algorithm(
        &self,
        order_size: f64,
        norm_depth: f64,
        slippage: &SlippagePrediction,
        urgency: f64,
        is_emergency: bool,
    ) -> ExecutionAlgorithm {
        // Emergency exits always use MARKET
        if is_emergency {
            return ExecutionAlgorithm::Market;
        }

        let size_ratio = order_size / norm_depth.max(0.001);
        let is_large_order = size_ratio > self.large_order_threshold;

        // High predicted slippage → prefer limit/patient algorithms
        if slippage.expected_slippage_bps > self.max_slippage_bps {
            if is_large_order {
                // Large + high slippage → slice it (VWAP/TWAP)
                if urgency < 0.5 {
                    ExecutionAlgorithm::Vwap
                } else {
                    ExecutionAlgorithm::Twap
                }
            } else {
                ExecutionAlgorithm::Limit
            }
        } else if is_large_order {
            // Large but manageable slippage → slice
            if urgency < 0.5 {
                ExecutionAlgorithm::Vwap
            } else {
                ExecutionAlgorithm::Twap
            }
        } else {
            // Small order with low slippage
            if urgency > 0.8 {
                ExecutionAlgorithm::Market
            } else {
                ExecutionAlgorithm::Limit
            }
        }
    }

    /// Build an execution schedule for a parent order.
    pub fn build_schedule(
        &self,
        order_size: f64,
        norm_depth: f64,
        slippage: &SlippagePrediction,
        urgency: f64,
        is_emergency: bool,
    ) -> ExecutionSchedule {
        let algorithm = self.select_algorithm(order_size, norm_depth, slippage, urgency, is_emergency);

        let (num_slices, slice_interval_secs) = match algorithm {
            ExecutionAlgorithm::Twap => (10, 60),  // 10 slices, 1 min apart
            ExecutionAlgorithm::Vwap => (24, 900), // 24 slices, 15 min apart (hourly)
            ExecutionAlgorithm::Market | ExecutionAlgorithm::Limit | ExecutionAlgorithm::StopLimit => (1, 0),
            ExecutionAlgorithm::RlGuided => (5, 300),
        };

        let expected_total_cost_bps = slippage.expected_slippage_bps
            + if num_slices > 1 { 0.5 * (num_slices as f64).ln() } else { 0.0 };

        ExecutionSchedule {
            algorithm,
            num_slices,
            slice_interval_secs,
            urgency,
            predicted_slippage_bps: slippage.expected_slippage_bps,
            expected_total_cost_bps,
        }
    }

    /// Estimate the slippage cost of market vs limit for a trade.
    pub fn market_vs_limit_cost(&self, slippage: &SlippagePrediction) -> (f64, f64) {
        // Market cost ≈ predicted slippage
        let market_cost = slippage.expected_slippage_bps;
        // Limit cost ≈ half-spread + risk of non-fill (opportunity cost)
        let limit_cost = slippage.spread_component * 0.5 + slippage.volatility_component * 0.3;
        (market_cost, limit_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microstructure::slippage_model::SlippagePrediction;

    fn make_prediction(slippage_bps: f64) -> SlippagePrediction {
        SlippagePrediction {
            expected_slippage_bps: slippage_bps,
            confidence: 0.8,
            dominant_factor: "spread".into(),
            spread_component: slippage_bps * 0.5,
            liquidity_component: 1.0,
            volatility_component: 1.0,
            size_component: 0.5,
        }
    }

    #[test]
    fn test_emergency_market() {
        let opt = ExecutionOptimizer::new();
        let slip = make_prediction(2.0);
        let algo = opt.select_algorithm(1.0, 10.0, &slip, 0.5, true);
        assert_eq!(algo, ExecutionAlgorithm::Market);
    }

    #[test]
    fn test_low_slippage_small_limit() {
        let opt = ExecutionOptimizer::new();
        let slip = make_prediction(1.0);
        let algo = opt.select_algorithm(1.0, 10.0, &slip, 0.5, false);
        assert_eq!(algo, ExecutionAlgorithm::Limit);
    }

    #[test]
    fn test_high_slippage_large_vwap() {
        let opt = ExecutionOptimizer::new();
        let slip = make_prediction(10.0);
        let algo = opt.select_algorithm(5.0, 10.0, &slip, 0.3, false);
        assert_eq!(algo, ExecutionAlgorithm::Vwap);
    }

    #[test]
    fn test_build_schedule() {
        let opt = ExecutionOptimizer::new();
        let slip = make_prediction(8.0);
        let schedule = opt.build_schedule(5.0, 10.0, &slip, 0.3, false);
        assert_eq!(schedule.algorithm, ExecutionAlgorithm::Vwap);
        assert!(schedule.num_slices >= 1);
    }
}
