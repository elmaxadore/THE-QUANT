//! # Multi-Agent RL Coordination
//!
//! Coordinates multiple RL agents (one per asset/regime) so they respect
//! portfolio-level constraints. Each agent proposes an action; the coordinator
//! reconciles them against:
//!   - Aggregate correlation exposure
//!   - Total margin/risk budget
//!   - Per-agent hard limits
//!
//! The coordinator is itself a *guest* — it cannot override the risk engine's
//! hard stops. It only de-risks (never increases risk beyond the risk engine).

use crate::error::{QuantError, QuantResult};
use crate::rl::gym::Action;
use crate::rl::policy::DistilledPolicy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// A single agent's proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProposal {
    pub agent_id: String,
    pub asset: String,
    pub action: Action,
    pub confidence: f64,
    pub current_position: f64,
    pub proposed_position: f64,
}

/// Portfolio-level constraint state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PortfolioConstraints {
    /// Maximum total exposure across all positions (fraction of equity).
    pub max_total_exposure: f64,
    /// Maximum correlated group exposure (fraction of equity).
    pub max_correlated_exposure: f64,
    /// Current total exposure.
    pub total_exposure: f64,
    /// Correlation matrix (symbol_a, symbol_b) -> correlation.
    pub correlations: HashMap<(String, String), f64>,
}

/// The reconciled decision for a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciledAction {
    pub agent_id: String,
    pub asset: String,
    pub action: Action,
    pub clamped: bool,
    pub reason: String,
}

/// Multi-agent coordinator.
#[derive(Debug)]
pub struct MultiAgentCoordinator {
    constraints: Arc<RwLock<PortfolioConstraints>>,
    /// Agent weightings (how much each agent's proposal counts).
    weights: Arc<RwLock<HashMap<String, f64>>>,
    /// Registry of active policies per agent.
    policies: Arc<RwLock<HashMap<String, DistilledPolicy>>>,
}

impl MultiAgentCoordinator {
    pub fn new() -> Self {
        Self {
            constraints: Arc::new(RwLock::new(PortfolioConstraints::default())),
            weights: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an agent policy.
    pub async fn register_agent(&self, agent_id: String, policy: DistilledPolicy) {
        self.policies.write().await.insert(agent_id, policy);
    }

    /// Update portfolio constraints.
    pub async fn update_constraints(&self, constraints: PortfolioConstraints) {
        *self.constraints.write().await = constraints;
    }

    /// Reconcile a set of agent proposals into final actions.
    ///
    /// This ensures aggregate risk stays within portfolio limits. If a proposal
    /// would push total exposure above the cap, it is scaled down (clamped).
    pub async fn reconcile(&self, proposals: Vec<AgentProposal>) -> Vec<ReconciledAction> {
        let constraints = self.constraints.read().await;
        let mut results = Vec::with_capacity(proposals.len());

        // Compute total proposed exposure
        let mut total_proposed = 0.0;
        for p in &proposals {
            total_proposed += p.proposed_position.abs();
        }

        // Scale factor if over the total exposure cap
        let scale = if total_proposed > constraints.max_total_exposure {
            constraints.max_total_exposure / total_proposed.max(1e-9)
        } else {
            1.0
        };

        for p in proposals {
            let mut final_action = p.action;
            let mut clamped = false;
            let mut reason = String::new();

            // Apply total-exposure scaling
            if scale < 1.0 {
                clamped = true;
                reason = "Total exposure cap exceeded — scaled down".to_string();
                match p.action {
                    Action::Increase(_) => {
                        let new_scale = scale;
                        final_action = Action::Increase(new_scale);
                    }
                    Action::Decrease(_) => {
                        // Decreasing is always safe, keep as-is
                    }
                    _ => {}
                }
            }

            // Check correlated exposure (simplified: use per-agent correlation)
            for (key, corr) in &constraints.correlations {
                if (key.0 == p.asset || key.1 == p.asset) && corr > &0.7 {
                    // If a correlated position would exceed the limit, reduce
                    if p.current_position + p.proposed_position > constraints.max_correlated_exposure {
                        clamped = true;
                        reason = "Correlated exposure limit exceeded — reduced".to_string();
                        final_action = Action::Decrease(0.5);
                    }
                }
            }

            results.push(ReconciledAction {
                agent_id: p.agent_id,
                asset: p.asset,
                action: final_action,
                clamped,
                reason,
            });
        }

        results
    }

    /// Reconcile a single agent's action against portfolio constraints.
    pub async fn reconcile_single(&self, proposal: AgentProposal) -> ReconciledAction {
        let reconciled = self.reconcile(vec![proposal]).await;
        reconciled.into_iter().next().unwrap_or_else(|| ReconciledAction {
            agent_id: String::new(),
            asset: String::new(),
            action: Action::Hold,
            clamped: true,
            reason: "No proposal".into(),
        })
    }

    /// Get the current weight for an agent.
    pub async fn get_weight(&self, agent_id: &str) -> f64 {
        self.weights.read().await.get(agent_id).copied().unwrap_or(1.0)
    }

    /// Set an agent's weight (e.g., based on recent performance).
    pub async fn set_weight(&self, agent_id: &str, weight: f64) {
        self.weights.write().await.insert(agent_id.to_string(), weight.clamp(0.0, 1.0));
    }

    /// Number of registered agents.
    pub async fn agent_count(&self) -> usize {
        self.policies.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reconcile_within_limits() {
        let coord = MultiAgentCoordinator::new();
        coord.update_constraints(PortfolioConstraints {
            max_total_exposure: 1.0,
            max_correlated_exposure: 0.2,
            ..Default::default()
        }).await;

        let proposals = vec![
            AgentProposal {
                agent_id: "a1".into(),
                asset: "EURUSD".into(),
                action: Action::Increase(0.5),
                confidence: 0.8,
                current_position: 0.1,
                proposed_position: 0.3,
            },
        ];

        let results = coord.reconcile(proposals).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].clamped);
    }

    #[tokio::test]
    async fn test_reconcile_clamps_total_exposure() {
        let coord = MultiAgentCoordinator::new();
        coord.update_constraints(PortfolioConstraints {
            max_total_exposure: 0.5,
            max_correlated_exposure: 0.2,
            ..Default::default()
        }).await;

        let proposals = vec![
            AgentProposal {
                agent_id: "a1".into(),
                asset: "EURUSD".into(),
                action: Action::Increase(0.8),
                confidence: 0.9,
                current_position: 0.1,
                proposed_position: 0.8, // Would exceed 0.5 cap
            },
        ];

        let results = coord.reconcile(proposals).await;
        assert!(results[0].clamped);
        assert!(results[0].reason.contains("scaled"));
    }

    #[tokio::test]
    async fn test_reconcile_correlation() {
        let coord = MultiAgentCoordinator::new();
        let mut constraints = PortfolioConstraints {
            max_total_exposure: 1.0,
            max_correlated_exposure: 0.2,
            ..Default::default()
        };
        constraints.correlations.insert(("EURUSD".into(), "GBPUSD".into()), 0.9);
        coord.update_constraints(constraints).await;

        let proposals = vec![
            AgentProposal {
                agent_id: "a1".into(),
                asset: "EURUSD".into(),
                action: Action::Increase(0.5),
                confidence: 0.8,
                current_position: 0.15,
                proposed_position: 0.3, // Exceeds correlated cap of 0.2
            },
        ];

        let results = coord.reconcile(proposals).await;
        assert!(results[0].clamped);
        assert!(results[0].reason.contains("correlated"));
    }
}
