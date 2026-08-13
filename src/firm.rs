//! # Firm Orchestrator (v4.0 "Hercules")
//!
//! Multi-account architecture. Every account is a COMPLETELY separate entity —
//! a `TradingDesk` with isolated strategy pool, model zoo, risk engine,
//! evolution engine, and trade journal. Market data and regime detection are
//! SHARED across desks.

use crate::account::{AccountRules, AccountStage, AccountType};
use crate::error::QuantResult;
use crate::state::{AccountStateEntry, StateManifest, StateStore};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Account type variants from the v4.0 matrix
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccountVariant {
    /// Blue Guardian Instant 5K — no profit target, no time limit, 10% DD
    Instant,
    /// 1-Step Evaluation — 8% profit target, 30-60 days, 10% DD
    OneStepEval,
    /// 2-Step Evaluation — 8% + 5% targets, 30-60 days, 10% DD
    TwoStepEval,
    /// Personal trading account — fully user-defined rules
    Personal,
}

impl AccountVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Instant => "instant",
            Self::OneStepEval => "one_step_eval",
            Self::TwoStepEval => "two_step_eval",
            Self::Personal => "personal",
        }
    }
}

/// Per-account strategy pool (isolated — no cross-account sharing)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StrategyPool {
    /// List of active strategy IDs/names
    pub active: Vec<String>,
    /// List of lab candidates
    pub lab: Vec<String>,
    /// Retired strategies with post-mortem status
    pub retired: Vec<String>,
}

/// Per-account model zoo
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelZoo {
    /// Production model IDs
    pub production: Vec<String>,
    /// Archived/previous model versions
    pub archive: Vec<String>,
}

/// Per-account risk state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeskRiskState {
    /// Maximum drawdown allowed (%)
    pub max_drawdown_pct: Decimal,
    /// Current drawdown (%)
    pub current_drawdown_pct: Decimal,
    /// Risk per trade (%)
    pub risk_per_trade_pct: Decimal,
    /// Whether the desk is in circuit-breaker
    pub circuit_breaker_active: bool,
    /// Daily PnL limit
    pub daily_pnl_limit: Decimal,
}

/// The account type matrix defaults (from the v4.0 spec §3.5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountTypeProfile {
    pub variant: AccountVariant,
    pub profit_target_pct: Option<Decimal>,
    pub time_limit_days: Option<u32>,
    pub max_drawdown_pct: Decimal,
    pub daily_loss_limit_pct: Option<Decimal>,
    pub consistency_required: bool,
    pub shield_enabled: bool,
    pub shield_amount: Option<Decimal>,
    pub payout_cap: Option<Decimal>,
    pub scaling_enabled: bool,
    pub min_trading_days: Option<u32>,
    pub leverage: u32,
    pub strategy_bias: String,
    pub risk_per_trade_pct: Decimal,
    pub target_monthly_pct: Decimal,
}

impl Default for AccountTypeProfile {
    fn default() -> Self {
        Self {
            variant: AccountVariant::Personal,
            profit_target_pct: None,
            time_limit_days: None,
            max_drawdown_pct: Decimal::new(10, 1), // 10%
            daily_loss_limit_pct: None,
            consistency_required: false,
            shield_enabled: false,
            shield_amount: None,
            payout_cap: None,
            scaling_enabled: false,
            min_trading_days: None,
            leverage: 50,
            strategy_bias: "balanced".into(),
            risk_per_trade_pct: Decimal::new(1, 1), // 1.0%
            target_monthly_pct: Decimal::new(2, 0), // 2%
        }
    }
}

impl AccountTypeProfile {
    /// Returns the default profile for each account type per the v4.0 matrix
    pub fn for_variant(variant: AccountVariant) -> Self {
        match variant {
            AccountVariant::Instant => Self {
                variant,
                profit_target_pct: None,
                time_limit_days: None,
                max_drawdown_pct: Decimal::new(10, 0),
                daily_loss_limit_pct: None,
                consistency_required: true,
                shield_enabled: true,
                shield_amount: Some(Decimal::new(50, 0)),
                payout_cap: Some(Decimal::new(250, 0)),
                scaling_enabled: true,
                min_trading_days: None,
                leverage: 100,
                strategy_bias: "conservative".into(),
                risk_per_trade_pct: Decimal::new(3, 1), // 0.3%
                target_monthly_pct: Decimal::new(3, 0), // 3-5%
            },
            AccountVariant::OneStepEval => Self {
                variant,
                profit_target_pct: Some(Decimal::new(8, 0)),
                time_limit_days: Some(45),
                max_drawdown_pct: Decimal::new(10, 0),
                daily_loss_limit_pct: Some(Decimal::new(5, 0)),
                consistency_required: true,
                shield_enabled: false,
                shield_amount: None,
                payout_cap: None,
                scaling_enabled: true,
                min_trading_days: Some(7),
                leverage: 100,
                strategy_bias: "balanced".into(),
                risk_per_trade_pct: Decimal::new(5, 1), // 0.5%
                target_monthly_pct: Decimal::new(5, 0), // 5-8%
            },
            AccountVariant::TwoStepEval => Self {
                variant,
                profit_target_pct: Some(Decimal::new(8, 0)),
                time_limit_days: Some(60),
                max_drawdown_pct: Decimal::new(10, 0),
                daily_loss_limit_pct: Some(Decimal::new(5, 0)),
                consistency_required: true,
                shield_enabled: false,
                shield_amount: None,
                payout_cap: None,
                scaling_enabled: true,
                min_trading_days: Some(7),
                leverage: 100,
                strategy_bias: "aggressive".into(),
                risk_per_trade_pct: Decimal::new(5, 1), // 0.5%
                target_monthly_pct: Decimal::new(5, 0), // 5-8%
            },
            AccountVariant::Personal => Self::default(),
        }
    }

    /// Convert to `AccountRules` for the account manager
    pub fn to_account_rules(&self) -> AccountRules {
        AccountRules {
            max_drawdown_pct: self.max_drawdown_pct,
            profit_target_pct: self.profit_target_pct.unwrap_or_default(),
            time_limit_days: self.time_limit_days.unwrap_or(0),
            min_trading_days: self.min_trading_days.unwrap_or(0),
            daily_loss_limit_pct: self.daily_loss_limit_pct.unwrap_or(Decimal::ZERO),
            leverage: self.leverage,
            ..AccountRules::default()
        }
    }
}

/// A single trading desk — a completely isolated trading entity
#[derive(Debug, Clone)]
pub struct TradingDesk {
    pub account_id: Uuid,
    pub name: String,
    pub variant: AccountVariant,
    pub strategy_pool: StrategyPool,
    pub model_zoo: ModelZoo,
    pub risk_state: DeskRiskState,
    pub trade_count: u64,
    pub total_pnl: Decimal,
    pub created_at: DateTime<Utc>,
    pub state_path: PathBuf,
}

impl TradingDesk {
    /// Create a new trading desk for an account
    pub fn new(
        account_id: Uuid,
        name: String,
        variant: AccountVariant,
        state_root: &Path,
    ) -> Self {
        let profile = AccountTypeProfile::for_variant(variant.clone());
        Self {
            account_id,
            name,
            variant,
            strategy_pool: StrategyPool::default(),
            model_zoo: ModelZoo::default(),
            risk_state: DeskRiskState {
                max_drawdown_pct: profile.max_drawdown_pct,
                current_drawdown_pct: Decimal::ZERO,
                risk_per_trade_pct: profile.risk_per_trade_pct,
                circuit_breaker_active: false,
                daily_pnl_limit: Decimal::ZERO,
            },
            trade_count: 0,
            total_pnl: Decimal::ZERO,
            created_at: Utc::now(),
            state_path: state_root.join("accounts").join(account_id.to_string()),
        }
    }

    /// Record a closed trade with PnL
    pub fn record_trade(&mut self, pnl: Decimal) {
        self.trade_count += 1;
        self.total_pnl += pnl;
    }

    /// Check if the desk is within its drawdown limits
    pub fn within_limits(&self) -> bool {
        !self.risk_state.circuit_breaker_active
            && self.risk_state.current_drawdown_pct <= self.risk_state.max_drawdown_pct
    }

    /// Apply a drawdown update and check circuit breaker
    pub fn update_drawdown(&mut self, new_dd: Decimal) -> bool {
        self.risk_state.current_drawdown_pct = new_dd;
        if new_dd >= self.risk_state.max_drawdown_pct {
            self.risk_state.circuit_breaker_active = true;
            true
        } else {
            false
        }
    }
}

/// Cross-desk correlation monitoring
#[derive(Debug, Clone, Default)]
pub struct CorrelationMonitor {
    /// Asset -> desk exposure map
    exposures: DashMap<String, Vec<(Uuid, Decimal)>>,
}

impl CorrelationMonitor {
    /// Record an open position for a desk
    pub fn record_position(&self, asset: String, desk_id: Uuid, notional: Decimal) {
        let mut entry = self.exposures.entry(asset.clone()).or_default();
        entry.retain(|(id, _)| *id != desk_id);
        entry.push((desk_id, notional));
    }

    /// Aggregate exposure to an asset across all desks
    pub fn aggregate_exposure(&self, asset: &str) -> Decimal {
        self.exposures
            .get(asset)
            .map(|e| e.value().iter().map(|(_, n)| n).sum())
            .unwrap_or(Decimal::ZERO)
    }

    /// Number of desks with open positions in an asset
    pub fn desk_count(&self, asset: &str) -> usize {
        self.exposures.get(asset).map(|e| e.value().len()).unwrap_or(0)
    }
}

/// The QuantFirm orchestrator — manages all trading desks
#[derive(Debug)]
pub struct QuantFirm {
    /// All trading desks (keyed by account UUID)
    desks: DashMap<Uuid, TradingDesk>,
    /// Shared state store
    state: StateStore,
    /// Correlation monitor
    correlation: CorrelationMonitor,
}

impl QuantFirm {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            desks: DashMap::new(),
            state: StateStore::new(repo_root),
            correlation: CorrelationMonitor::default(),
        }
    }

    /// Initialize the firm: ensure state tree + load any existing desks
    pub fn initialize(&self) -> QuantResult<()> {
        self.state.initialize()?;
        let uuids = self.state.list_account_uuids()?;
        for uuid in uuids {
            let dir = self.state.account_dir(&uuid);
            let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
            self.desks.insert(
                uuid,
                TradingDesk::new(uuid, name, AccountVariant::Personal, self.state.state_root()),
            );
        }
        Ok(())
    }

    /// Add a new trading desk (register account)
    pub fn add_desk(
        &mut self,
        name: String,
        variant: AccountVariant,
        _account_type: AccountType,
        stage: AccountStage,
    ) -> QuantResult<Uuid> {
        let profile = AccountTypeProfile::for_variant(variant.clone());
        let uuid = Uuid::new_v4();

        // Ensure state dirs exist
        self.state.ensure_account_dirs(&uuid)?;

        let desk = TradingDesk::new(uuid, name.clone(), variant, self.state.state_root());
        self.desks.insert(uuid, desk);

        // Register in the state store manifest
        let mut manifest = self.state.load_manifest().unwrap_or_default();
        manifest.upsert_account(AccountStateEntry {
            uuid,
            name: name.clone(),
            firm: profile.strategy_bias.clone(),
            variant: profile.variant.as_str().to_string(),
            stage: format!("{:?}", stage).to_lowercase(),
            lifecycle: "acquired".into(),
            state_hash: String::new(),
            last_commit: String::new(),
            paths: StateManifest::account_state_paths(self.state.state_root(), &uuid),
        });
        self.state.save_manifest(&manifest)?;

        Ok(uuid)
    }

    /// Get a desk by ID
    pub fn get_desk(&self, id: &Uuid) -> Option<TradingDesk> {
        self.desks.get(id).map(|d| d.clone())
    }

    /// Get all desk IDs
    pub fn desk_ids(&self) -> Vec<Uuid> {
        self.desks.iter().map(|e| *e.key()).collect()
    }

    /// List all desks
    pub fn list_desks(&self) -> Vec<TradingDesk> {
        self.desks.iter().map(|e| e.value().clone()).collect()
    }

    /// Record a trade on a desk
    pub fn record_trade(&self, desk_id: &Uuid, pnl: Decimal) -> bool {
        if let Some(mut desk) = self.desks.get_mut(desk_id) {
            desk.record_trade(pnl);
            true
        } else {
            false
        }
    }

    /// Update a desk's drawdown state
    pub fn update_drawdown(&self, desk_id: &Uuid, drawdown: Decimal) {
        if let Some(mut desk) = self.desks.get_mut(desk_id) {
            desk.update_drawdown(drawdown);
        }
    }

    /// Record a cross-desk position
    pub fn record_position(&self, asset: String, desk_id: Uuid, notional: Decimal) {
        self.correlation.record_position(asset, desk_id, notional);
    }

    /// Aggregate exposure across all desks
    pub fn aggregate_exposure(&self, asset: &str) -> Decimal {
        self.correlation.aggregate_exposure(asset)
    }

    /// Count of desks in an asset
    pub fn desks_in_asset(&self, asset: &str) -> usize {
        self.correlation.desk_count(asset)
    }

    /// The underlying state store
    pub fn state_store(&self) -> &StateStore {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_account_profile_matrix() {
        let instant = AccountTypeProfile::for_variant(AccountVariant::Instant);
        assert_eq!(instant.risk_per_trade_pct, Decimal::new(3, 1)); // 0.3%
        assert!(instant.consistency_required);
        assert_eq!(instant.shield_amount, Some(Decimal::new(50, 0)));

        let one_step = AccountTypeProfile::for_variant(AccountVariant::OneStepEval);
        assert_eq!(one_step.profit_target_pct, Some(Decimal::new(8, 0)));
        assert_eq!(one_step.leverage, 100);
    }

    #[test]
    fn test_desk_isolation() {
        let dir = tempdir().unwrap();
        let mut firm = QuantFirm::new(dir.path().to_path_buf());
        firm.initialize().unwrap();

        let desk1 = firm
            .add_desk(
                "BG-Instant-5K-01".into(),
                AccountVariant::Instant,
                AccountType::PropFirm,
                AccountStage::Funded,
            )
            .unwrap();
        let desk2 = firm
            .add_desk(
                "BG-Instant-5K-02".into(),
                AccountVariant::Personal,
                AccountType::Personal,
                AccountStage::Demo,
            )
            .unwrap();

        // Each desk has its own state dir
        assert!(firm.state_store().account_dir(&desk1).exists());
        assert!(firm.state_store().account_dir(&desk2).exists());

        // Trade on desk1 doesn't affect desk2
        assert!(firm.record_trade(&desk1, Decimal::new(4250, 2)));
        let d1 = firm.get_desk(&desk1).unwrap();
        let d2 = firm.get_desk(&desk2).unwrap();
        assert_eq!(d1.total_pnl, Decimal::new(4250, 2));
        assert_eq!(d2.total_pnl, Decimal::ZERO);
    }

    #[test]
    fn test_drawdown_circuit_breaker() {
        let dir = tempdir().unwrap();
        let mut firm = QuantFirm::new(dir.path().to_path_buf());
        firm.initialize().unwrap();

        let desk_id = firm
            .add_desk(
                "Test".into(),
                AccountVariant::Instant,
                AccountType::PropFirm,
                AccountStage::Funded,
            )
            .unwrap();

        assert!(!firm.get_desk(&desk_id).unwrap().risk_state.circuit_breaker_active);
        firm.update_drawdown(&desk_id, Decimal::new(12, 1)); // 12% DD, over 10% max
        let desk = firm.get_desk(&desk_id).unwrap();
        assert!(desk.risk_state.circuit_breaker_active);
        assert!(!desk.within_limits());
    }

    #[test]
    fn test_correlation_monitor() {
        let monitor = CorrelationMonitor::default();
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        monitor.record_position("XAUUSD".into(), u1, Decimal::new(100, 0));
        monitor.record_position("XAUUSD".into(), u2, Decimal::new(50, 0));
        assert_eq!(monitor.aggregate_exposure("XAUUSD"), Decimal::new(150, 0));
        assert_eq!(monitor.desk_count("XAUUSD"), 2);
    }
}