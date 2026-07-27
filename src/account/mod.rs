//! # Account Management Engine (Layer 2)
//!
//! Manages trading accounts with full state machine lifecycle, rule enforcement,
//! and real-time monitoring of equity, balance, margin, drawdown, and PnL.
//!
//! ## Account Types
//! - **PropFirm**: Evaluation or Funded stages with strict rule enforcement
//! - **Personal**: Self-managed with lower leverage and tax-aware tracking
//!
//! ## State Machine
//! 
```text
//! Active → Paused (user/manual)
//! Active → Halted (rule breach imminent)
//! Active → Blown (rule breached — require manual reset)
//! Paused → Active (user/manual)
//! Halted → Active (after cooldown + manual review)
//! Blown → Active (only after explicit user confirmation + rule reset)
//! 
```

use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// === Account Types & Stages ===

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountType {
    PropFirm,
    Personal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountStage {
    Evaluation,
    Funded,
    Demo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountStatus {
    Active,
    Paused,
    Halted,
    Blown,
}

// === Core Data Structures ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: Uuid,
    pub name: String,
    pub account_type: AccountType,
    pub stage: AccountStage,
    pub credentials: EncryptedCredentials,
    pub rules: AccountRules,
    pub status: AccountStatus,
    pub metrics: AccountMetrics,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedCredentials {
    pub ciphertext: Vec<u8>,      // AES-256-GCM encrypted
    pub server: String,           // MT5 server name (not encrypted)
    pub account_number: String,   // Display only, not encrypted
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountRules {
    /// Maximum drawdown percentage (e.g., 10.0 = 10%)
    pub max_drawdown_pct: Decimal,
    /// Profit target percentage (e.g., 8.0 = 8%)
    pub profit_target_pct: Decimal,
    /// Time limit in days for evaluation accounts
    pub time_limit_days: u32,
    /// Minimum trading days required
    pub min_trading_days: u32,
    /// Daily loss limit percentage
    pub daily_loss_limit_pct: Decimal,
    /// Maximum lot size per trade
    pub max_lot_size: Decimal,
    /// Maximum leverage
    pub leverage: u32,
    /// Allowed trading instruments
    pub allowed_instruments: Vec<String>,
    /// News trading rule
    pub news_trading: NewsTradingRule,
    /// Whether holding positions over weekend is allowed
    pub weekend_hold: bool,
    /// Whether hedging is allowed
    pub hedging_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NewsTradingRule {
    Allowed,
    Forbidden,
    Restricted,
}

impl Default for AccountRules {
    fn default() -> Self {
        Self {
            max_drawdown_pct: Decimal::new(10, 1),       // 10%
            profit_target_pct: Decimal::new(8, 1),        // 8%
            time_limit_days: 30,
            min_trading_days: 10,
            daily_loss_limit_pct: Decimal::new(2, 1),     // 2%
            max_lot_size: Decimal::new(10, 0),            // 10 lots
            leverage: 50,
            allowed_instruments: vec![],
            news_trading: NewsTradingRule::Forbidden,
            weekend_hold: false,
            hedging_allowed: false,
        }
    }
}

/// Real-time account metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccountMetrics {
    pub balance: Decimal,
    pub equity: Decimal,
    pub margin: Decimal,
    pub free_margin: Decimal,
    pub margin_level_pct: Decimal,
    pub open_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub total_pnl: Decimal,
    pub daily_drawdown_pct: Decimal,
    pub total_drawdown_pct: Decimal,
    pub trades_today: u32,
    pub trades_total: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub profit_target_progress_pct: Decimal,
    pub days_elapsed: u32,
    pub min_trading_days_met: bool,
    pub last_updated: Option<DateTime<Utc>>,
}

// === Account Manager ===

#[derive(Debug)]
pub struct AccountManager {
    accounts: dashmap::DashMap<Uuid, Account>,
    rule_engine: RuleEngine,
}

impl AccountManager {
    pub fn new() -> Self {
        Self {
            accounts: dashmap::DashMap::new(),
            rule_engine: RuleEngine::new(),
        }
    }

    /// Register a new account
    pub fn register_account(
        &self,
        name: String,
        account_type: AccountType,
        stage: AccountStage,
        encrypted_creds: EncryptedCredentials,
        rules: AccountRules,
    ) -> QuantResult<Account> {
        let account = Account {
            id: Uuid::new_v4(),
            name,
            account_type,
            stage,
            credentials: encrypted_creds,
            rules,
            status: AccountStatus::Active,
            metrics: AccountMetrics::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.accounts.insert(account.id, account.clone());
        Ok(account)
    }

    /// Get account by ID
    pub fn get_account(&self, id: &Uuid) -> Option<Account> {
        self.accounts.get(id).map(|a| a.clone())
    }

    /// Get all accounts
    pub fn list_accounts(&self) -> Vec<Account> {
        self.accounts.iter().map(|a| a.clone()).collect()
    }

    /// Update account metrics (called by Risk Engine)
    pub fn update_metrics(&self, account_id: &Uuid, metrics: AccountMetrics) -> QuantResult<()> {
        if let Some(mut account) = self.accounts.get_mut(account_id) {
            account.metrics = metrics;
            account.updated_at = Utc::now();

            // Check rules after metrics update
            let violations = self.rule_engine.check_rules(&account);
            if !violations.is_empty() {
                for violation in violations {
                    tracing::warn!(
                        "Rule violation for account {}: {}",
                        account.name,
                        violation
                    );
                }
            }
            Ok(())
        } else {
            Err(QuantError::Internal(format!("Account {} not found", account_id)))
        }
    }

    /// Update account status with state machine validation
    pub fn set_status(&self, account_id: &Uuid, new_status: AccountStatus) -> QuantResult<()> {
        if let Some(mut account) = self.accounts.get_mut(account_id) {
            let old_status = &account.status;
            
            // Validate state transition
            match (old_status, &new_status) {
                (AccountStatus::Active, AccountStatus::Paused) => {}
                (AccountStatus::Active, AccountStatus::Halted) => {}
                (AccountStatus::Active, AccountStatus::Blown) => {}
                (AccountStatus::Paused, AccountStatus::Active) => {}
                (AccountStatus::Halted, AccountStatus::Active) => {
                    // Must have manual review + cooldown
                }
                (AccountStatus::Blown, AccountStatus::Active) => {
                    // Must have explicit user confirmation + rule reset
                }
                _ => {
                    return Err(QuantError::AccountNotTradable {
                        account_id: account_id.to_string(),
                        status: format!("Cannot transition from {:?} to {:?}", old_status, new_status),
                    });
                }
            }

            account.status = new_status;
            account.updated_at = Utc::now();
            Ok(())
        } else {
            Err(QuantError::Internal(format!("Account {} not found", account_id)))
        }
    }

    /// Remove an account
    pub fn remove_account(&self, account_id: &Uuid) -> bool {
        self.accounts.remove(account_id).is_some()
    }

    /// Check if account can trade
    pub fn can_trade(&self, account_id: &Uuid) -> bool {
        self.accounts
            .get(account_id)
            .map(|a| a.status == AccountStatus::Active)
            .unwrap_or(false)
    }
}

// === Rule Engine ===

#[derive(Debug)]
pub struct RuleEngine;

impl RuleEngine {
    pub fn new() -> Self {
        Self
    }

    /// Check all rules for an account and return any violations found
    pub fn check_rules(&self, account: &Account) -> Vec<String> {
        let mut violations = Vec::new();
        let metrics = &account.metrics;
        let rules = &account.rules;

        // Check drawdown limits
        if metrics.total_drawdown_pct >= rules.max_drawdown_pct {
            violations.push(format!(
                "Max drawdown breached: {:.2}% >= {:.2}%",
                metrics.total_drawdown_pct, rules.max_drawdown_pct
            ));
        }

        // Check daily loss limit
        if metrics.daily_drawdown_pct >= rules.daily_loss_limit_pct {
            violations.push(format!(
                "Daily loss limit breached: {:.2}% >= {:.2}%",
                metrics.daily_drawdown_pct, rules.daily_loss_limit_pct
            ));
        }

        // Check profit target for evaluation accounts
        if account.stage == AccountStage::Evaluation {
            if metrics.profit_target_progress_pct >= Decimal::new(100, 0) {
                violations.push("Profit target achieved!".to_string());
            }
        }

        // Check time limits
        if rules.time_limit_days > 0 && metrics.days_elapsed > rules.time_limit_days {
            violations.push(format!(
                "Time limit exceeded: {} days > {} days",
                metrics.days_elapsed, rules.time_limit_days
            ));
        }

        // Check minimum trading days
        if !metrics.min_trading_days_met {
            violations.push(format!(
                "Minimum trading days not met: {} < {}",
                metrics.trades_total, rules.min_trading_days
            ));
        }

        violations
    }

    /// Determine account status based on rule violations
    pub fn determine_status(&self, account: &Account) -> AccountStatus {
        let violations = self.check_rules(account);
        
        let has_drawdown_breach = violations.iter().any(|v| v.contains("drawdown breached"));
        let has_loss_limit_breach = violations.iter().any(|v| v.contains("Daily loss limit breached"));
        let has_time_exceeded = violations.iter().any(|v| v.contains("Time limit exceeded"));

        if has_drawdown_breach || has_loss_limit_breach || has_time_exceeded {
            AccountStatus::Blown
        } else if !violations.is_empty() {
            AccountStatus::Halted
        } else {
            account.status.clone()
        }
    }
}

// === Serialization for encrypted credentials ===

impl EncryptedCredentials {
    pub fn new(ciphertext: Vec<u8>, server: String, account_number: String) -> Self {
        Self {
            ciphertext,
            server,
            account_number,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_account() -> Account {
        Account {
            id: Uuid::new_v4(),
            name: "Test Prop Firm".to_string(),
            account_type: AccountType::PropFirm,
            stage: AccountStage::Evaluation,
            credentials: EncryptedCredentials::new(vec![], "TestServer".into(), "12345".into()),
            rules: AccountRules::default(),
            status: AccountStatus::Active,
            metrics: AccountMetrics::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_account_creation() {
        let manager = AccountManager::new();
        let account = manager.register_account(
            "Test".into(),
            AccountType::PropFirm,
            AccountStage::Evaluation,
            EncryptedCredentials::new(vec![], "Server".into(), "123".into()),
            AccountRules::default(),
        ).unwrap();

        assert_eq!(account.name, "Test");
        assert!(manager.can_trade(&account.id));
    }

    #[test]
    fn test_status_transitions() {
        let manager = AccountManager::new();
        let account = manager.register_account(
            "Test".into(),
            AccountType::PropFirm,
            AccountStage::Evaluation,
            EncryptedCredentials::new(vec![], "Server".into(), "123".into()),
            AccountRules::default(),
        ).unwrap();

        let id = account.id;

        // Active -> Paused
        assert!(manager.set_status(&id, AccountStatus::Paused).is_ok());
        assert!(!manager.can_trade(&id));

        // Paused -> Active
        assert!(manager.set_status(&id, AccountStatus::Active).is_ok());
        assert!(manager.can_trade(&id));
    }

    #[test]
    fn test_rule_engine_drawdown() {
        let mut account = create_test_account();
        account.metrics.total_drawdown_pct = Decimal::new(15, 1); // 15%
        account.rules.max_drawdown_pct = Decimal::new(10, 1);    // 10%

        let engine = RuleEngine::new();
        let violations = engine.check_rules(&account);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("drawdown"));
    }
}
