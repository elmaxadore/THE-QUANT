//! # Slippage Model
//!
//! Predicts expected slippage for an order based on microstructure features,
//! order size, and market conditions. Used to:
//!   - Set realistic expectations for execution
//!   - Adjust position sizing (wider spreads → smaller size)
//!   - Choose between limit vs market orders
//!   - Log actual vs predicted slippage for continuous calibration

use crate::error::{QuantError, QuantResult};
use crate::microstructure::features::MicrostructureFeatures;
use serde::{Deserialize, Serialize};

/// A slippage prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippagePrediction {
    /// Expected slippage in basis points.
    pub expected_slippage_bps: f64,
    /// Confidence in the prediction (0..1).
    pub confidence: f64,
    /// The dominant factor driving the prediction.
    pub dominant_factor: String,
    /// Spread component (bps).
    pub spread_component: f64,
    /// Liquidity component (bps).
    pub liquidity_component: f64,
    /// Volatility component (bps).
    pub volatility_component: f64,
    /// Size component (bps).
    pub size_component: f64,
}

/// Parameters for the slippage model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModelConfig {
    /// Spread coefficient.
    pub spread_coef: f64,
    /// Liquidity coefficient (applied to inverse depth).
    pub liquidity_coef: f64,
    /// Volatility coefficient (applied to recent vol).
    pub volatility_coef: f64,
    /// Size coefficient (applied to order size relative to depth).
    pub size_coef: f64,
    /// Base slippage in bps.
    pub base_slippage_bps: f64,
    /// Minimum expected spread bps.
    pub min_spread_bps: f64,
}

impl Default for SlippageModelConfig {
    fn default() -> Self {
        Self {
            spread_coef: 0.5,
            liquidity_coef: 0.2,
            volatility_coef: 0.2,
            size_coef: 0.1,
            base_slippage_bps: 0.5,
            min_spread_bps: 0.1,
        }
    }
}

/// The slippage model.
#[derive(Debug, Clone)]
pub struct SlippageModel {
    config: SlippageModelConfig,
    /// Running calibration state (MAE tracking).
    cumulative_mae: f64,
    calibration_count: u64,
}

impl SlippageModel {
    pub fn new(config: SlippageModelConfig) -> Self {
        Self {
            config,
            cumulative_mae: 0.0,
            calibration_count: 0,
        }
    }

    pub fn default() -> Self {
        Self::new(SlippageModelConfig::default())
    }

    /// Predict slippage for an order.
    pub fn predict(
        &self,
        features: &MicrostructureFeatures,
        order_size: f64,
        norm_depth: f64,
        recent_vol_bps: f64,
    ) -> SlippagePrediction {
        let spread = features.spread_bps.max(self.config.min_spread_bps);
        let depth = norm_depth.max(0.001);
        let size_ratio = order_size / depth;

        let spread_component = self.config.spread_coef * spread;
        let liquidity_component = self.config.liquidity_coef * (1.0 / depth).min(10.0);
        let volatility_component = self.config.volatility_coef * recent_vol_bps.min(50.0);
        let size_component = self.config.size_coef * size_ratio.min(10.0);

        let total = self.config.base_slippage_bps
            + spread_component
            + liquidity_component
            + volatility_component
            + size_component;

        // Deterministic confidence based on spread stability
        let confidence = if spread > 20.0 { 0.3 } else if spread > 10.0 { 0.5 } else { 0.8 };

        // Dominant factor
        let mut factors = vec![
            ("spread", spread_component),
            ("liquidity", liquidity_component),
            ("volatility", volatility_component),
            ("size", size_component),
        ];
        factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let dominant_factor = factors[0].0.to_string();

        SlippagePrediction {
            expected_slippage_bps: total,
            confidence,
            dominant_factor,
            spread_component,
            liquidity_component,
            volatility_component,
            size_component,
        }
    }

    /// Calibrate the model with actual slippage observed.
    pub fn calibrate(&mut self, predicted_bps: f64, actual_bps: f64) -> f64 {
        let mae = (actual_bps - predicted_bps).abs();
        self.cumulative_mae += mae;
        self.calibration_count += 1;
        let avg_mae = self.cumulative_mae / self.calibration_count.max(1) as f64;
        avg_mae
    }

    /// Current mean absolute error of the model.
    pub fn current_mae(&self) -> f64 {
        if self.calibration_count > 0 {
            self.cumulative_mae / self.calibration_count as f64
        } else {
            0.0
        }
    }

    /// Serialize calibration state.
    pub fn state(&self) -> (f64, u64) {
        (self.cumulative_mae, self.calibration_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microstructure::features::{MicrostructureFeatures, TickSnapshot};

    fn make_features() -> MicrostructureFeatures {
        let snapshots: Vec<TickSnapshot> = (0..5).map(|i| TickSnapshot {
            time: i * 1000,
            bid: 100.0,
            ask: 100.1,
            bid_depth: 10.0,
            ask_depth: 10.0,
            bid_volume: 100.0,
            ask_volume: 90.0,
            volume: 1.0,
            trade_direction: Some(1),
            trade_price: Some(100.05),
        }).collect();
        MicrostructureFeatures::compute(&snapshots).unwrap()
    }

    #[test]
    fn test_predict() {
        let model = SlippageModel::default();
        let features = make_features();
        let pred = model.predict(&features, 1.0, 10.0, 5.0);
        assert!(pred.expected_slippage_bps > 0.0);
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
        assert!(!pred.dominant_factor.is_empty());
    }

    #[test]
    fn test_calibrate() {
        let mut model = SlippageModel::default();
        let mae = model.calibrate(5.0, 7.0);
        assert!(mae > 0.0);
        assert_eq!(model.current_mae(), mae);
    }

    #[test]
    fn test_larger_order_more_slippage() {
        let model = SlippageModel::default();
        let features = make_features();
        let small = model.predict(&features, 0.1, 10.0, 5.0);
        let large = model.predict(&features, 10.0, 10.0, 5.0);
        assert!(large.expected_slippage_bps > small.expected_slippage_bps);
    }
}
