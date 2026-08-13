//! # Distilled Policy for Live Inference
//!
//! The distilled policy is a small, deterministic MLP (32→16→11) that runs on
//! the live trading path. It is produced by distilling a trained PPO agent to
//! achieve sub-2ms inference. It is a *guest*: its outputs are always clamped
//! by the risk engine.

use crate::error::{QuantError, QuantResult};
use crate::rl::gym::Action;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// A distilled policy network (32→16→11).
///
/// Architecture:
///   input (32) → fc1 (16, ReLU) → fc2 (11) → softmax/argmax → action
///
/// The 11 output nodes map to: 4 discrete actions + 7 continuous scale bins,
/// or a single distribution over (action, scale) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledPolicy {
    pub policy_id: String,
    pub asset: String,
    pub regime: String,
    /// Flattened weights for fc1: [output][input]
    pub fc1_weights: Vec<f64>,
    /// fc1 biases
    pub fc1_bias: Vec<f64>,
    /// Flattened weights for fc2: [output][input]
    pub fc2_weights: Vec<f64>,
    /// fc2 biases
    pub fc2_bias: Vec<f64>,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (16)
    pub hidden_dim: usize,
    /// Output dimension (11)
    pub output_dim: usize,
    /// Whether the policy is active for live trading
    pub active: bool,
    /// Live performance metrics (Sharpe, win rate, etc.)
    pub metrics: PolicyMetrics,
}

/// Performance metrics for a live policy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyMetrics {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub max_drawdown_pct: f64,
    pub auto_disabled: bool,
}

impl DistilledPolicy {
    /// Create a new distilled policy with the given architecture.
    pub fn new(policy_id: String, asset: String, regime: String, input_dim: usize) -> Self {
        let hidden_dim = 16;
        let output_dim = 11;
        Self {
            policy_id,
            asset,
            regime,
            fc1_weights: vec![0.0; hidden_dim * input_dim],
            fc1_bias: vec![0.0; hidden_dim],
            fc2_weights: vec![0.0; output_dim * hidden_dim],
            fc2_bias: vec![0.0; output_dim],
            input_dim,
            hidden_dim,
            output_dim,
            active: false,
            metrics: PolicyMetrics::default(),
        }
    }

    /// Build a distilled policy from a trained PPO agent.
    pub fn from_agent(agent: &crate::rl::ppo::PpoAgent) -> Self {
        let mut policy = Self::new(
            "distilled_0".into(),
            "ALL".into(),
            "ALL".into(),
            agent.actor.input_dim,
        );
        // Copy weights from the actor's first two layers
        policy.fc1_weights = agent.actor.fc1.weight.iter().cloned().collect();
        policy.fc1_bias = agent.actor.fc1.bias.iter().cloned().collect();
        policy.fc2_weights = agent.actor.fc2.weight.iter().cloned().collect();
        policy.fc2_bias = agent.actor.fc2.bias.iter().cloned().collect();
        policy
    }

    /// Run the policy forward pass and return an action.
    pub fn predict(&self, features: &[f64]) -> Action {
        if !self.active {
            return Action::Hold;
        }

        // Ensure input dimension matches
        if features.len() != self.input_dim {
            warn!("Policy input dimension mismatch: got {}, expected {}", features.len(), self.input_dim);
            return Action::Hold;
        }

        // fc1: hidden = relu(W1 * x + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.fc1_bias[i];
            for j in 0..self.input_dim {
                sum += self.fc1_weights[i * self.input_dim + j] * features[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // fc2: out = W2 * hidden + b2
        let mut out = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = self.fc2_bias[i];
            for j in 0..self.hidden_dim {
                sum += self.fc2_weights[i * self.hidden_dim + j] * hidden[j];
            }
            out[i] = sum;
        }

        // Interpret output: first 4 nodes = discrete action, next 7 = scale bins
        let action_idx = (0..4)
            .max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap())
            .unwrap_or(0);
        let scale_bin = (0..7)
            .max_by(|&a, &b| out[4 + a].partial_cmp(&out[4 + b]).unwrap())
            .unwrap_or(3);
        let scale = (scale_bin as f64 + 0.5) / 7.0; // 0..1

        Action::from_index(action_idx, scale)
    }

    /// Activate the policy for live trading.
    pub fn activate(&mut self) {
        self.active = true;
        info!("Policy {} activated for live trading", self.policy_id);
    }

    /// Deactivate the policy (e.g., auto-disable on poor Sharpe).
    pub fn deactivate(&mut self) {
        self.active = false;
        self.metrics.auto_disabled = true;
        warn!("Policy {} deactivated (auto-disable)", self.policy_id);
    }

    /// Record a trade result and check for auto-disable.
    pub fn record_trade(&mut self, pnl: f64) {
        self.metrics.total_trades += 1;
        self.metrics.total_pnl += pnl;
        if pnl > 0.0 {
            self.metrics.winning_trades += 1;
        }
        self.metrics.win_rate = self.metrics.winning_trades as f64 / self.metrics.total_trades.max(1) as f64;

        // Auto-disable if Sharpe < 0.3 over 20 trades
        if self.metrics.total_trades >= 20 && self.metrics.sharpe_ratio < 0.3 {
            self.deactivate();
        }
    }

    /// Update the live Sharpe ratio.
    pub fn update_sharpe(&mut self, sharpe: f64) {
        self.metrics.sharpe_ratio = sharpe;
        if self.metrics.total_trades >= 20 && sharpe < 0.3 {
            self.deactivate();
        }
    }

    /// Serialize the policy to JSON.
    pub fn to_json(&self) -> QuantResult<String> {
        serde_json::to_string(self).map_err(|e| QuantError::Internal(e.to_string()))
    }

    /// Deserialize a policy from JSON.
    pub fn from_json(json: &str) -> QuantResult<Self> {
        serde_json::from_str(json).map_err(|e| QuantError::Internal(e.to_string()))
    }
}

/// Registry of distilled policies (one per asset/regime).
#[derive(Debug)]
pub struct PolicyRegistry {
    policies: Arc<RwLock<std::collections::HashMap<String, DistilledPolicy>>>,
}

impl PolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Register or replace a policy.
    pub async fn register(&self, policy: DistilledPolicy) {
        let key = format!("{}:{}", policy.asset, policy.regime);
        self.policies.write().await.insert(key, policy);
    }

    /// Get the active policy for an asset/regime.
    pub async fn get(&self, asset: &str, regime: &str) -> Option<DistilledPolicy> {
        let key = format!("{}:{}", asset, regime);
        self.policies.read().await.get(&key).cloned()
    }

    /// Get all active policies.
    pub async fn active_policies(&self) -> Vec<DistilledPolicy> {
        self.policies.read().await.values()
            .filter(|p| p.active)
            .cloned()
            .collect()
    }

    /// Deactivate all policies (emergency stop).
    pub async fn deactivate_all(&self) {
        for mut p in self.policies.write().await.values_mut() {
            p.active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_predict_hold_when_inactive() {
        let policy = DistilledPolicy::new("p1".into(), "EURUSD".into(), "TRENDING_UP".into(), 32);
        let action = policy.predict(&vec![0.0; 32]);
        assert_eq!(action, Action::Hold);
    }

    #[test]
    fn test_policy_activate_predict() {
        let mut policy = DistilledPolicy::new("p1".into(), "EURUSD".into(), "TRENDING_UP".into(), 32);
        policy.activate();
        let action = policy.predict(&vec![0.1; 32]);
        // Should return one of the valid actions
        assert!(matches!(action, Action::Hold | Action::Increase(_) | Action::Decrease(_) | Action::Close));
    }

    #[test]
    fn test_record_trade_auto_disable() {
        let mut policy = DistilledPolicy::new("p1".into(), "EURUSD".into(), "TRENDING_UP".into(), 32);
        policy.activate();
        // Simulate 20 losing trades
        for _ in 0..20 {
            policy.record_trade(-0.01);
        }
        policy.update_sharpe(0.1); // Below threshold
        assert!(policy.metrics.auto_disabled);
        assert!(!policy.active);
    }

    #[test]
    fn test_json_roundtrip() {
        let policy = DistilledPolicy::new("p1".into(), "EURUSD".into(), "TRENDING_UP".into(), 32);
        let json = policy.to_json().unwrap();
        let loaded = DistilledPolicy::from_json(&json).unwrap();
        assert_eq!(loaded.policy_id, "p1");
        assert_eq!(loaded.asset, "EURUSD");
    }
}
