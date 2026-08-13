//! # Onboarding Wizard Server (v3.0 "Prometheus")
//!
//! HTTP handlers for the web onboarding wizard. These are integrated into the
//! Axum server to serve the wizard UI and handle step submissions.

use crate::api::server::ApiResponse;
use crate::error::{QuantError, QuantResult};
use crate::wizard::{AccountSetup, Mt5Mode, RiskSetup, WizardManager, WizardStep};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// The wizard handler state.
#[derive(Debug)]
pub struct WizardHandler {
    pub wizard: Arc<RwLock<WizardManager>>,
}

impl WizardHandler {
    pub fn new() -> Self {
        Self {
            wizard: Arc::new(RwLock::new(WizardManager::new())),
        }
    }

    /// Start a new wizard session.
    pub async fn start(&self) -> ApiResponse<WizardView> {
        self.wizard.write().await.start();
        ApiResponse::ok(self.view().await)
    }

    /// Advance to the next step.
    pub async fn advance(&self) -> ApiResponse<WizardView> {
        let mut wizard = self.wizard.write().await;
        wizard.advance();
        drop(wizard);
        ApiResponse::ok(self.view().await)
    }

    /// Submit the master password.
    pub async fn submit_password(&self, password: String) -> ApiResponse<WizardView> {
        let result = self.wizard.write().await.set_master_password(password);
        match result {
            Ok(()) => ApiResponse::ok(self.view().await),
            Err(e) => ApiResponse::err(e.to_string()),
        }
    }

    /// Submit the MT5 connection.
    pub async fn submit_mt5(&self, mode: Mt5Mode, endpoint: String) -> ApiResponse<WizardView> {
        self.wizard.write().await.set_mt5(mode, endpoint);
        ApiResponse::ok(self.view().await)
    }

    /// Submit the account setup.
    pub async fn submit_account(&self, account: AccountSetup) -> ApiResponse<WizardView> {
        let result = self.wizard.write().await.set_account(account);
        match result {
            Ok(()) => ApiResponse::ok(self.view().await),
            Err(e) => ApiResponse::err(e.to_string()),
        }
    }

    /// Submit the risk setup.
    pub async fn submit_risk(&self, risk: RiskSetup) -> ApiResponse<WizardView> {
        self.wizard.write().await.set_risk(risk);
        ApiResponse::ok(self.view().await)
    }

    /// Get the current wizard view.
    pub async fn view(&self) -> WizardView {
        let wizard = self.wizard.read().await;
        WizardView {
            step: wizard.state.step,
            step_index: wizard.current_step(),
            total_steps: 7,
            completed: wizard.is_complete(),
            mt5_mode: wizard.state.mt5_mode,
            mt5_endpoint: wizard.state.mt5_endpoint.clone(),
            account_name: wizard.state.account.as_ref().map(|a| a.name.clone()),
            risk_per_trade_pct: wizard.state.risk.risk_per_trade_pct,
        }
    }
}

impl Default for WizardHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// A view of the wizard state for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardView {
    pub step: WizardStep,
    pub step_index: usize,
    pub total_steps: usize,
    pub completed: bool,
    pub mt5_mode: Mt5Mode,
    pub mt5_endpoint: String,
    pub account_name: Option<String>,
    pub risk_per_trade_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wizard_handler_flow() {
        let handler = WizardHandler::new();
        let start = handler.start().await;
        assert!(start.success);
        assert_eq!(start.data.unwrap().step_index, 0);

        let pw = handler.submit_password("long_enough_password".into()).await;
        assert!(pw.success);

        let advance = handler.advance().await;
        assert!(advance.success);
        assert_eq!(advance.data.unwrap().step, WizardStep::Mt5Connection);
    }
}
