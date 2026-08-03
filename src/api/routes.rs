//! # API Routes
//!
//! Defines the REST route handlers for the web dashboard. Each handler
//! takes the shared `ApiState` and returns a JSON `ApiResponse`.

use crate::api::server::{ApiResponse, ApiState, SystemStatus};
use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// Public route handler — returns the current health status.
pub async fn health_handler(state: &ApiState) -> ApiResponse<HealthView> {
    let health = state.health().await;
    ApiResponse::ok(HealthView {
        status: health.status,
        version: health.version,
        uptime_secs: health.uptime_secs,
        memory_pct: health.memory_pct,
        mt5_connected: health.mt5_connected,
        db_connected: health.db_connected,
        active_accounts: health.active_accounts,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthView {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
    pub memory_pct: f64,
    pub mt5_connected: bool,
    pub db_connected: bool,
    pub active_accounts: usize,
}

/// System status route handler.
pub async fn system_status_handler(state: &ApiState) -> ApiResponse<SystemStatus> {
    // TODO: Gather real metrics from memory manager and modules
    let status = SystemStatus {
        cpu_pct: 0.0,
        ram_pct: 0.0,
        disk_free_gb: 0.0,
        uptime_secs: state.app_started.elapsed().as_secs(),
        last_training: None,
        lab_status: "idle".into(),
        next_evolution: None,
        github_sync: "idle".into(),
        ram_tier: 1,
    };
    ApiResponse::ok(status)
}

/// A summary of an account for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountView {
    pub id: String,
    pub name: String,
    pub account_type: String,
    pub stage: String,
    pub status: String,
    pub equity: f64,
    pub balance: f64,
    pub drawdown_pct: f64,
    pub pnl_today: f64,
    pub total_pnl: f64,
}

/// List accounts route handler.
pub async fn list_accounts_handler(_state: &ApiState) -> ApiResponse<Vec<AccountView>> {
    // TODO: Pull from AccountManager
    ApiResponse::ok(Vec::new())
}

/// A single open position view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionView {
    pub id: String,
    pub account_id: String,
    pub symbol: String,
    pub direction: String,
    pub volume: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub pnl: f64,
    pub duration_secs: u64,
}

/// List open positions route handler.
pub async fn list_positions_handler(_state: &ApiState) -> ApiResponse<Vec<PositionView>> {
    // TODO: Pull from RiskEngine / ExecutionEngine
    ApiResponse::ok(Vec::new())
}

/// Current regime probabilities view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeView {
    pub symbol: String,
    pub dominant_regime: String,
    pub confidence: f64,
    pub probabilities: Vec<(String, f64)>,
}

/// Get current regime route handler.
pub async fn regime_handler(_state: &ApiState) -> ApiResponse<Vec<RegimeView>> {
    // TODO: Pull from RegimeDetector
    ApiResponse::ok(Vec::new())
}

/// A model manifest view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelView {
    pub model_id: String,
    pub asset: String,
    pub regime: String,
    pub algorithm: String,
    pub status: String,
    pub val_auc: f64,
    pub val_sharpe: f64,
}

/// List models route handler.
pub async fn list_models_handler(_state: &ApiState) -> ApiResponse<Vec<ModelView>> {
    // TODO: Pull from ModelManager
    ApiResponse::ok(Vec::new())
}

/// A trade journal entry view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeView {
    pub id: String,
    pub symbol: String,
    pub direction: String,
    pub volume: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub open_time: String,
    pub close_time: String,
    pub strategy_id: String,
}

/// List recent trades route handler.
pub async fn list_trades_handler(_state: &ApiState, limit: usize) -> ApiResponse<Vec<TradeView>> {
    // TODO: Pull from ExecutionEngine journal
    let _ = limit;
    ApiResponse::ok(Vec::new())
}

/// A memory breakdown view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryView {
    pub module: String,
    pub used_mb: f64,
    pub budget_mb: f64,
    pub pct_of_budget: f64,
}

/// Get memory breakdown route handler.
pub async fn memory_handler(_state: &ApiState) -> ApiResponse<Vec<MemoryView>> {
    // TODO: Pull from MemoryManager
    ApiResponse::ok(Vec::new())
}

/// A prop-firm template view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateView {
    pub template_id: String,
    pub provider: String,
    pub description: String,
    pub account_type: String,
    pub stage: String,
}

/// List available prop-firm templates.
pub async fn list_templates_handler(_state: &ApiState) -> ApiResponse<Vec<TemplateView>> {
    // TODO: Load from config/templates/
    ApiResponse::ok(Vec::new())
}

/// An RL policy view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyView {
    pub policy_id: String,
    pub asset: String,
    pub regime: String,
    pub status: String,
    pub architecture: String,
    pub is_distilled: bool,
    pub sharpe: f64,
}

/// List RL policies route handler.
pub async fn list_policies_handler(_state: &ApiState) -> ApiResponse<Vec<PolicyView>> {
    // TODO: Pull from PolicyRegistry
    ApiResponse::ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityManager;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn test_state() -> ApiState {
        ApiState::new(
            crate::api::server::ApiConfig::default(),
            Arc::new(RwLock::new(SecurityManager::new(std::path::PathBuf::from("/tmp/x")))),
        )
    }

    #[tokio::test]
    async fn test_health_handler() {
        let state = test_state();
        let resp = health_handler(&state).await;
        assert!(resp.success);
        assert_eq!(resp.data.unwrap().status, "ok");
    }

    #[tokio::test]
    async fn test_system_status_handler() {
        let state = test_state();
        let resp = system_status_handler(&state).await;
        assert!(resp.success);
    }

    #[tokio::test]
    async fn test_empty_handlers() {
        let state = test_state();
        assert!(list_accounts_handler(&state).await.success);
        assert!(list_positions_handler(&state).await.success);
        assert!(regime_handler(&state).await.success);
        assert!(list_models_handler(&state).await.success);
        assert!(list_trades_handler(&state, 10).await.success);
        assert!(memory_handler(&state).await.success);
    }
}
