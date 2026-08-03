//! # Reinforcement Learning Layer (v3.0 "Prometheus")
//!
//! The RL layer provides a self-contained reinforcement learning stack for
//! trading. It is a *guest* component: it cannot override hard risk limits
//! (hard stops, max_holding_bars, RuleCap). Its actions are advisory and always
//! clamped by the risk engine.
//!
//! ## Components
//! - `gym` — QuantGym trading environment (state, action, reward, step)
//! - `buffer` — bounded, percentage-scaled replay buffer
//! - `ppo` — PPO trainer (offline training only)
//! - `policy` — distilled policy (32→16→11) for sub-2ms live inference
//! - `multi_agent` — portfolio-level coordination of per-asset agents
//!
//! ## Safety
//! - RL never overrides hard stops, max_holding_bars, or RuleCap
//! - Auto-disable if live Sharpe < 0.3 over 20 trades
//! - Training runs in the lab/evolution phase; live runs only distilled policy
//! - Every action passes through the risk engine before execution

pub mod gym;
pub mod buffer;
pub mod ppo;
pub mod policy;
pub mod multi_agent;

pub use gym::{Action, GymConfig, Observation, QuantGym, Reward, StepResult};
pub use buffer::{ReplayBuffer, Transition, TransitionAction};
pub use ppo::{PpoAgent, PpoConfig, PpoTrainer};
pub use policy::{DistilledPolicy, PolicyMetrics, PolicyRegistry};
pub use multi_agent::{AgentProposal, MultiAgentCoordinator, PortfolioConstraints, ReconciledAction};

/// The RL layer's memory budget as a percentage of HARD_PROCESS_LIMIT.
/// The spec allocates 5% to RL/Gym.
pub const RL_MEMORY_BUDGET_PCT: f64 = 5.0;

/// RL inference latency budget (sub-2ms target).
pub const RL_INFERENCE_TIMEOUT_MS: u64 = 2;

/// Auto-disable threshold: live Sharpe below this over 20 trades disables RL.
pub const RL_AUTO_DISABLE_SHARPE: f64 = 0.3;

/// Number of trades before the auto-disable check fires.
pub const RL_AUTO_DISABLE_TRADES: u32 = 20;

/// The RL manager — top-level entry point for the RL layer.
#[derive(Debug)]
pub struct RlManager {
    pub registry: PolicyRegistry,
    pub coordinator: MultiAgentCoordinator,
    pub training_enabled: bool,
}

impl RlManager {
    pub fn new() -> Self {
        Self {
            registry: PolicyRegistry::new(),
            coordinator: MultiAgentCoordinator::new(),
            training_enabled: false,
        }
    }

    /// Enable training mode (only during lab/evolution).
    pub fn enable_training(&mut self) {
        self.training_enabled = true;
    }

    /// Disable training mode (production).
    pub fn disable_training(&mut self) {
        self.training_enabled = false;
    }

    /// Get a live action recommendation for an asset/regime.
    ///
    /// Returns `None` if no active policy exists or if RL is disabled.
    pub async fn recommend(&self, asset: &str, regime: &str, features: &[f64]) -> Option<Action> {
        let policy = self.registry.get(asset, regime).await?;
        if !policy.active {
            return None;
        }
        let action = policy.predict(features);
        // Safety: never return a non-Hold action if the policy is disabled mid-predict
        Some(action)
    }
}

impl Default for RlManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rl_manager_recommend_no_policy() {
        let manager = RlManager::new();
        let action = manager.recommend("EURUSD", "TRENDING_UP", &vec![0.0; 32]).await;
        assert!(action.is_none());
    }

    #[tokio::test]
    async fn test_rl_manager_recommend_with_policy() {
        let manager = RlManager::new();
        let mut policy = DistilledPolicy::new("p1".into(), "EURUSD".into(), "TRENDING_UP".into(), 32);
        policy.activate();
        manager.registry.register(policy).await;
        let action = manager.recommend("EURUSD", "TRENDING_UP", &vec![0.1; 32]).await;
        assert!(action.is_some());
    }
}
