//! # Market Microstructure Features
//!
//! Microstructure features capture the fine-grained order-flow and liquidity
//! characteristics of the market that are invisible at the OHLCV level. These
//! are computed from tick data (bid/ask/depth) and feed into the slippage
//! model and execution optimizer, as well as the strategy feature pipeline.

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// A refined microstructure feature vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MicrostructureFeatures {
    /// Time of the observation.
    pub time: i64,
    /// Spread in basis points (ask - bid) / mid * 10000.
    pub spread_bps: f64,
    /// Depth imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    pub depth_imbalance: f64,
    /// Order flow imbalance over the window.
    pub order_flow_imbalance: f64,
    /// Trade intensity (trades per second).
    pub trade_intensity: f64,
    /// Realized spread in basis points.
    pub realized_spread_bps: f64,
    /// Price impact over 1 second.
    pub price_impact_1s: f64,
    /// Price impact over 5 seconds.
    pub price_impact_5s: f64,
    /// Bid/ask volume ratio.
    pub bid_ask_volume_ratio: f64,
    /// Microprice: (bid*ask_vol + ask*bid_vol) / (bid_vol + ask_vol).
    pub microprice: f64,
    /// Synthetic features hash for versioning.
    pub features_hash: String,
}

impl MicrostructureFeatures {
    /// Compute features from a series of tick snapshots.
    pub fn compute(snapshots: &[TickSnapshot]) -> Option<Self> {
        if snapshots.is_empty() {
            return None;
        }

        let last = snapshots.last()?;
        let spread_bps = compute_spread_bps(last.bid, last.ask);
        let depth_imbalance = compute_depth_imbalance(last.bid_depth, last.ask_depth);
        let bid_ask_volume_ratio = if last.ask_volume > 0.0 {
            last.bid_volume / last.ask_volume
        } else {
            0.0
        };
        let microprice = compute_microprice(last);

        // Order flow imbalance over the window
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        let mut trade_count = 0.0;
        for s in snapshots {
            if s.trade_direction == Some(1) {
                buy_volume += s.volume;
            } else if s.trade_direction == Some(-1) {
                sell_volume += s.volume;
            }
            trade_count += 1.0;
        }
        let order_flow_imbalance = if (buy_volume + sell_volume) > 0.0 {
            (buy_volume - sell_volume) / (buy_volume + sell_volume)
        } else {
            0.0
        };

        // Trade intensity (trades per second)
        let duration = snapshots.last()?.time - snapshots.first()?.time;
        let trade_intensity = if duration > 0.0 {
            trade_count / (duration as f64 / 1000.0)
        } else {
            0.0
        };

        // Price impact over 1s and 5s
        let mid_now = (last.bid + last.ask) / 2.0;
        let price_impact_1s = snapshots.iter().rev()
            .find(|s| (last.time - s.time) >= 1000)
            .map(|s| {
                let mid_prev = (s.bid + s.ask) / 2.0;
                if mid_prev > 0.0 { (mid_now - mid_prev) / mid_prev } else { 0.0 }
            })
            .unwrap_or(0.0);
        let price_impact_5s = snapshots.iter().rev()
            .find(|s| (last.time - s.time) >= 5000)
            .map(|s| {
                let mid_prev = (s.bid + s.ask) / 2.0;
                if mid_prev > 0.0 { (mid_now - mid_prev) / mid_prev } else { 0.0 }
            })
            .unwrap_or(0.0);

        // Realized spread (simplified: average of (trade_price - mid) sign)
        let realized_spread_bps = snapshots.iter()
            .filter_map(|s| {
                let mid = (s.bid + s.ask) / 2.0;
                if mid > 0.0 {
                    Some((s.trade_price.unwrap_or(mid) - mid).abs() / mid * 10_000.0)
                } else {
                    None
                }
            })
            .sum::<f64>() / snapshots.len().max(1) as f64;

        let mut f = Self {
            time: last.time,
            spread_bps,
            depth_imbalance,
            order_flow_imbalance,
            trade_intensity,
            realized_spread_bps,
            price_impact_1s,
            price_impact_5s,
            bid_ask_volume_ratio,
            microprice,
            features_hash: String::new(),
        };
        f.features_hash = compute_hash(&f);
        Some(f)
    }

    /// Flatten to a feature vector for model input.
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.spread_bps,
            self.depth_imbalance,
            self.order_flow_imbalance,
            self.trade_intensity,
            self.realized_spread_bps,
            self.price_impact_1s,
            self.price_impact_5s,
            self.bid_ask_volume_ratio,
            self.microprice,
        ]
    }
}

/// A single tick snapshot with order book context.
#[derive(Debug, Clone)]
pub struct TickSnapshot {
    pub time: i64,
    pub bid: f64,
    pub ask: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub volume: f64,
    /// +1 = buy aggressor, -1 = sell aggressor, None = unknown.
    pub trade_direction: Option<i8>,
    /// Last trade price (if a trade occurred on this tick).
    pub trade_price: Option<f64>,
}

fn compute_spread_bps(bid: f64, ask: f64) -> f64 {
    let mid = (bid + ask) / 2.0;
    if mid > 0.0 {
        (ask - bid) / mid * 10_000.0
    } else {
        0.0
    }
}

fn compute_depth_imbalance(bid_depth: f64, ask_depth: f64) -> f64 {
    if bid_depth + ask_depth > 0.0 {
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
    } else {
        0.0
    }
}

fn compute_microprice(s: &TickSnapshot) -> f64 {
    let total = s.bid_volume + s.ask_volume;
    if total > 0.0 {
        (s.bid * s.ask_volume + s.ask * s.bid_volume) / total
    } else {
        (s.bid + s.ask) / 2.0
    }
}

fn compute_hash(f: &MicrostructureFeatures) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(&f.spread_bps.to_le_bytes());
    h.update(&f.depth_imbalance.to_le_bytes());
    h.update(&f.order_flow_imbalance.to_le_bytes());
    h.update(&f.trade_intensity.to_le_bytes());
    h.update(&f.realized_spread_bps.to_le_bytes());
    h.update(&f.price_impact_1s.to_le_bytes());
    h.update(&f.price_impact_5s.to_le_bytes());
    h.update(&f.bid_ask_volume_ratio.to_le_bytes());
    h.update(&f.microprice.to_le_bytes());
    h.finalize().to_hex().to_string()
}

/// Streaming accumulator for microstructure features.
#[derive(Debug)]
pub struct MicrostructureAccumulator {
    snapshots: Vec<TickSnapshot>,
    window_size: usize,
}

impl MicrostructureAccumulator {
    pub fn new(window_size: usize) -> Self {
        Self {
            snapshots: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a tick snapshot and compute features if the window is full.
    pub fn add(&mut self, snapshot: TickSnapshot) -> Option<MicrostructureFeatures> {
        self.snapshots.push(snapshot);
        if self.snapshots.len() > self.window_size {
            self.snapshots.remove(0);
        }
        if self.snapshots.len() < 2 {
            return None;
        }
        MicrostructureFeatures::compute(&self.snapshots)
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshots(n: usize) -> Vec<TickSnapshot> {
        (0..n).map(|i| TickSnapshot {
            time: i * 1000,
            bid: 100.0 + (i as f64) * 0.001,
            ask: 100.1 + (i as f64) * 0.001,
            bid_depth: 10.0 + (i as f64) * 0.1,
            ask_depth: 10.0 + (i as f64) * 0.05,
            bid_volume: 100.0 + i as f64,
            ask_volume: 90.0 + i as f64,
            volume: 1.0,
            trade_direction: Some(if i % 2 == 0 { 1 } else { -1 }),
            trade_price: Some(100.05),
        }).collect()
    }

    #[test]
    fn test_spread_bps() {
        let s = compute_spread_bps(100.0, 100.1);
        assert!((s - 9.99).abs() < 0.1);
    }

    #[test]
    fn test_compute_features() {
        let snapshots = make_snapshots(10);
        let f = MicrostructureFeatures::compute(&snapshots).unwrap();
        assert!(f.spread_bps > 0.0);
        assert!(f.bid_ask_volume_ratio > 0.0);
        assert!(f.microprice > 0.0);
        assert!(!f.features_hash.is_empty());
    }

    #[test]
    fn test_accumulator() {
        let mut acc = MicrostructureAccumulator::new(5);
        let snapshots = make_snapshots(10);
        let mut last = None;
        for s in snapshots {
            last = acc.add(s);
        }
        assert!(last.is_some());
    }

    #[test]
    fn test_to_vector() {
        let snapshots = make_snapshots(10);
        let f = MicrostructureFeatures::compute(&snapshots).unwrap();
        assert_eq!(f.to_vector().len(), 9);
    }
}
