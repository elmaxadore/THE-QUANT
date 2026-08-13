//! # Onboarding Wizard (v3.0 "Prometheus")
//!
//! A web-based onboarding wizard that guides the user through first-time setup:
//!   1. Set master password
//!   2. Configure MT5 connection
//!   3. Add a first account (from a prop-firm template or manual)
//!   4. Configure risk parameters
//!   5. Start the daemon
//!
//! The wizard is served by the Axum server at `/wizard`.

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// Wizard step identifiers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum WizardStep {
    Welcome,
    MasterPassword,
    Mt5Connection,
    AccountSetup,
    RiskConfig,
    Review,
    Complete,
}

impl WizardStep {
    pub fn next(&self) -> WizardStep {
        match self {
            WizardStep::Welcome => WizardStep::MasterPassword,
            WizardStep::MasterPassword => WizardStep::Mt5Connection,
            WizardStep::Mt5Connection => WizardStep::AccountSetup,
            WizardStep::AccountSetup => WizardStep::RiskConfig,
            WizardStep::RiskConfig => WizardStep::Review,
            WizardStep::Review => WizardStep::Complete,
            WizardStep::Complete => WizardStep::Complete,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            WizardStep::Welcome => 0,
            WizardStep::MasterPassword => 1,
            WizardStep::Mt5Connection => 2,
            WizardStep::AccountSetup => 3,
            WizardStep::RiskConfig => 4,
            WizardStep::Review => 5,
            WizardStep::Complete => 6,
        }
    }
}

/// The wizard session state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardState {
    pub step: WizardStep,
    pub master_password: Option<String>,
    pub mt5_mode: Mt5Mode,
    pub mt5_endpoint: String,
    pub account: Option<AccountSetup>,
    pub risk: RiskSetup,
    pub completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Mt5Mode {
    Local,
    Remote,
}

impl Default for WizardState {
    fn default() -> Self {
        Self {
            step: WizardStep::Welcome,
            master_password: None,
            mt5_mode: Mt5Mode::Local,
            mt5_endpoint: "tcp://127.0.0.1:5555".to_string(),
            account: None,
            risk: RiskSetup::default(),
            completed: false,
        }
    }
}

/// Account setup data from the wizard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSetup {
    pub name: String,
    pub account_type: String,
    pub stage: String,
    pub template_id: Option<String>,
    pub max_drawdown_pct: f64,
    pub profit_target_pct: f64,
    pub leverage: u32,
}

/// Risk setup data from the wizard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSetup {
    pub risk_per_trade_pct: f64,
    pub max_leverage: u32,
    pub max_lot_size: f64,
    pub news_trading: String,
    pub weekend_hold: bool,
}

impl Default for RiskSetup {
    fn default() -> Self {
        Self {
            risk_per_trade_pct: 0.5,
            max_leverage: 50,
            max_lot_size: 10.0,
            news_trading: "Restricted".to_string(),
            weekend_hold: false,
        }
    }
}

/// The wizard manager — tracks the onboarding session.
#[derive(Debug)]
pub struct WizardManager {
    pub state: WizardState,
    pub started: bool,
}

impl WizardManager {
    pub fn new() -> Self {
        Self {
            state: WizardState::default(),
            started: false,
        }
    }

    /// Begin the wizard.
    pub fn start(&mut self) {
        self.started = true;
        self.state = WizardState::default();
    }

    /// Advance to the next step.
    pub fn advance(&mut self) -> WizardStep {
        let next = self.state.step.next();
        self.state.step = next;
        if next == WizardStep::Complete {
            self.state.completed = true;
        }
        next
    }

    /// Set the master password.
    pub fn set_master_password(&mut self, password: String) -> QuantResult<()> {
        if password.len() < 8 {
            return Err(QuantError::AuthenticationError("Password must be at least 8 characters".into()));
        }
        self.state.master_password = Some(password);
        Ok(())
    }

    /// Set the MT5 connection.
    pub fn set_mt5(&mut self, mode: Mt5Mode, endpoint: String) {
        self.state.mt5_mode = mode;
        self.state.mt5_endpoint = endpoint;
    }

    /// Set the account setup.
    pub fn set_account(&mut self, account: AccountSetup) -> QuantResult<()> {
        if account.name.trim().is_empty() {
            return Err(QuantError::Internal("Account name is required".into()));
        }
        self.state.account = Some(account);
        Ok(())
    }

    /// Set the risk setup.
    pub fn set_risk(&mut self, risk: RiskSetup) {
        self.state.risk = risk;
    }

    /// Whether the wizard is complete.
    pub fn is_complete(&self) -> bool {
        self.state.completed
    }

    /// Current step index.
    pub fn current_step(&self) -> usize {
        self.state.step.index()
    }
}

impl Default for WizardManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wizard_flow() {
        let mut wizard = WizardManager::new();
        assert!(!wizard.started);
        wizard.start();
        assert!(wizard.started);
        assert_eq!(wizard.current_step(), 0); // Welcome

        wizard.advance();
        assert_eq!(wizard.current_step(), 1); // MasterPassword

        assert!(wizard.set_master_password("short".into()).is_err());
        assert!(wizard.set_master_password("long_enough_password".into()).is_ok());

        wizard.advance();
        assert_eq!(wizard.current_step(), 2); // Mt5Connection
        wizard.set_mt5(Mt5Mode::Local, "tcp://127.0.0.1:5555".into());

        wizard.advance();
        assert_eq!(wizard.current_step(), 3); // AccountSetup
        assert!(wizard.set_account(AccountSetup {
            name: "".into(),
            account_type: "PropFirm".into(),
            stage: "Evaluation".into(),
            template_id: None,
            max_drawdown_pct: 10.0,
            profit_target_pct: 8.0,
            leverage: 50,
        }).is_err());

        wizard.advance();
        assert_eq!(wizard.current_step(), 4); // RiskConfig
        wizard.set_risk(RiskSetup::default());

        wizard.advance();
        assert_eq!(wizard.current_step(), 5); // Review

        wizard.advance();
        assert_eq!(wizard.current_step(), 6); // Complete
        assert!(wizard.is_complete());
    }

    #[test]
    fn test_step_transitions() {
        assert_eq!(WizardStep::Welcome.next(), WizardStep::MasterPassword);
        assert_eq!(WizardStep::Complete.next(), WizardStep::Complete);
    }
}
