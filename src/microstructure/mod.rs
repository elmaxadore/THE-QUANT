//! # Market Microstructure Module (v3.0 "Prometheus")
//!
//! Captures fine-grained order-flow and liquidity signals that are invisible
//! at the OHLCV level. These features feed the slippage model, execution
//! optimizer, and strategy feature pipeline.

pub mod features;
pub mod slippage_model;
pub mod execution_opt;

pub use features::{MicrostructureAccumulator, MicrostructureFeatures, TickSnapshot};
pub use slippage_model::{SlippageModel, SlippageModelConfig, SlippagePrediction};
pub use execution_opt::{ExecutionAlgorithm, ExecutionOptimizer, ExecutionSchedule};

/// The microstructure module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const MICROSTRUCTURE_MEMORY_BUDGET_PCT: f64 = 2.0;

/// The front-end orchestrator for the microstructure layer.
#[derive(Debug)]
pub struct MicrostructureManager {
    pub slippage_model: SlippageModel,
    pub optimizer: ExecutionOptimizer,
    pub accumulator: MicrostructureAccumulator,
}

impl MicrostructureManager {
    pub fn new() -> Self {
        Self {
            slippage_model: SlippageModel::default(),
            optimizer: ExecutionOptimizer::new(),
            accumulator: MicrostructureAccumulator::new(100),
        }
    }

    /// Feed a tick snapshot and optionally get computed features.
    pub fn feed_tick(&mut self, snapshot: TickSnapshot) -> Option<MicrostructureFeatures> {
        self.accumulator.add(snapshot)
    }

    /// Predict slippage for an order.
    pub fn predict_slippage(
        &self,
        features: &MicrostructureFeatures,
        order_size: f64,
        norm_depth: f64,
        recent_vol_bps: f64,
    ) -> SlippagePrediction {
        self.slippage_model.predict(features, order_size, norm_depth, recent_vol_bps)
    }

    /// Choose an execution algorithm.
    pub fn choose_algorithm(
        &self,
        order_size: f64,
        norm_depth: f64,
        slippage: &SlippagePrediction,
        urgency: f64,
        is_emergency: bool,
    ) -> ExecutionAlgorithm {
        self.optimizer.select_algorithm(order_size, norm_depth, slippage, urgency, is_emergency)
    }
}

impl Default for MicrostructureManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_flow() {
        let mut mgr = MicrostructureManager::new();
        let snapshot = TickSnapshot {
            time: 0,
            bid: 100.0,
            ask: 100.1,
            bid_depth: 10.0,
            ask_depth: 10.0,
            bid_volume: 100.0,
            ask_volume: 90.0,
            volume: 1.0,
            trade_direction: Some(1),
            trade_price: Some(100.05),
        };
        let _ = mgr.feed_tick(snapshot);
        let snapshot2 = TickSnapshot {
            time: 1000,
            bid: 100.1,
            ask: 100.2,
            bid_depth: 10.0,
            ask_depth: 10.0,
            bid_volume: 100.0,
            ask_volume: 90.0,
            volume: 1.0,
            trade_direction: Some(1),
            trade_price: Some(100.15),
        };
        let features = mgr.feed_tick(snapshot2);
        assert!(features.is_some());
    }
}
