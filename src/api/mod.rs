//! # Web API Module (v3.0 "Prometheus")
//!
//! Axum-based HTTP API and WebSocket streaming for the web dashboard.

pub mod server;
pub mod routes;
pub mod auth;
pub mod ws;

pub use server::{ApiConfig, ApiResponse, ApiState, HealthStatus, SystemStatus, WsEvent};
pub use routes::{
    AccountView, HealthView, MemoryView, ModelView, PolicyView, PositionView, RegimeView,
    TemplateView, TradeView,
};
pub use auth::{ApiAuth, ApiClaims, LoginRequest, LoginResponse};
pub use ws::{WsClient, WsHub};

/// The API module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const API_MEMORY_BUDGET_PCT: f64 = 2.0;

/// The top-level API manager.
#[derive(Debug)]
pub struct ApiManager {
    pub config: ApiConfig,
    pub state: ApiState,
    pub hub: WsHub,
    pub auth: ApiAuth,
}

impl ApiManager {
    pub fn new(config: ApiConfig, security: std::sync::Arc<tokio::sync::RwLock<crate::security::SecurityManager>>) -> Self {
        let state = ApiState::new(config.clone(), security.clone());
        let auth = ApiAuth::new(security);
        Self {
            config,
            state,
            hub: WsHub::new(),
            auth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityManager;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[test]
    fn test_api_manager_new() {
        let security = Arc::new(RwLock::new(SecurityManager::new(std::path::PathBuf::from("/tmp/x"))));
        let mgr = ApiManager::new(ApiConfig::default(), security);
        assert_eq!(mgr.config.port, 8080);
    }
}
