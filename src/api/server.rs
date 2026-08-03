//! # Web API Module (v3.0 "Prometheus")
//!
//! Axum-based HTTP API and WebSocket streaming for the web dashboard.
//! Provides REST endpoints for accounts, positions, regimes, models, and
//! system status, plus real-time streaming via WebSocket/SSE.
//!
//! ## Security
//! - All endpoints (except login/health) require a valid JWT session token
//! - CORS restricted to configured origins
//! - Rate limiting on sensitive endpoints
//! - API p99 < 50ms target

use crate::error::{QuantError, QuantResult};
use crate::security::SecurityManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub rate_limit_per_min: u64,
    pub enable_ws: bool,
    pub enable_ssse: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_origins: vec!["http://localhost:8080".to_string()],
            rate_limit_per_min: 120,
            enable_ws: true,
            enable_ssse: true,
        }
    }
}

/// A generic API response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: i64,
}

impl<T> ApiResponse<T> {
    pub fn ok(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    pub fn err(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
    pub memory_used_mb: f64,
    pub memory_limit_mb: f64,
    pub memory_pct: f64,
    pub mt5_connected: bool,
    pub db_connected: bool,
    pub active_accounts: usize,
    pub active_policies: usize,
}

/// System status snapshot for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemStatus {
    pub cpu_pct: f64,
    pub ram_pct: f64,
    pub disk_free_gb: f64,
    pub uptime_secs: u64,
    pub last_training: Option<String>,
    pub lab_status: String,
    pub next_evolution: Option<String>,
    pub github_sync: String,
    pub ram_tier: u8,
}

/// The API server state shared across handlers.
#[derive(Debug)]
pub struct ApiState {
    pub config: ApiConfig,
    pub security: Arc<RwLock<SecurityManager>>,
    pub app_started: std::time::Instant,
    pub mt5_connected: Arc<RwLock<bool>>,
    pub db_connected: Arc<RwLock<bool>>,
    pub active_accounts: Arc<RwLock<usize>>,
}

impl ApiState {
    pub fn new(config: ApiConfig, security: Arc<RwLock<SecurityManager>>) -> Self {
        Self {
            config,
            security,
            app_started: std::time::Instant::now(),
            mt5_connected: Arc::new(RwLock::new(false)),
            db_connected: Arc::new(RwLock::new(false)),
            active_accounts: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn health(&self) -> HealthStatus {
        HealthStatus {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.app_started.elapsed().as_secs(),
            memory_used_mb: 0.0,
            memory_limit_mb: 0.0,
            memory_pct: 0.0,
            mt5_connected: *self.mt5_connected.read().await,
            db_connected: *self.db_connected.read().await,
            active_accounts: *self.active_accounts.read().await,
            active_policies: 0,
        }
    }
}

/// WebSocket event types for streaming updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsEvent {
    Quote { symbol: String, bid: f64, ask: f64, time: i64 },
    Position { account_id: String, symbol: String, pnl: f64 },
    Regime { symbol: String, regime: String, probability: f64 },
    Account { account_id: String, equity: f64, drawdown_pct: f64 },
    System { memory_pct: f64, cpu_pct: f64 },
    Anomaly { symbol: String, score: f64, anomalous: bool },
    Changepoint { symbol: String, probability: f64 },
    Trade { symbol: String, direction: String, pnl: f64 },
}

/// A subscription to a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSubscription {
    pub channel: String,
    pub symbols: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_response_ok() {
        let resp = ApiResponse::ok(42u32);
        assert!(resp.success);
        assert_eq!(resp.data, Some(42));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_api_response_err() {
        let resp: ApiResponse<u32> = ApiResponse::err("boom".into());
        assert!(!resp.success);
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_health_status() {
        let state = ApiState::new(ApiConfig::default(), Arc::new(RwLock::new(SecurityManager::new(
            std::path::PathBuf::from("/tmp/vault.enc")
        ))));
        let health = futures::executor::block_on(state.health());
        assert_eq!(health.status, "ok");
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
    }
}
