//! # Pipeline Manager (v3.1 Hephaestus)
//!
//! Handles account lifecycle transitions, auto-rotation logic on payout cap reach or blown status,
//! bench warm account readiness scoring, and $25/day extraction strategy execution.

use crate::account::{ComplianceStatus, ConsistencyState, ShieldState, ShieldStatus};
use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccountLifecycle {
    Acquired,       // Just purchased/registered, not yet configured
    Warming,        // Connected to MT5, verifying rules, paper trade check
    Trading,        // Live trading active
    Extracting,     // Near payout cap, conservative mode, preparing withdrawal
    PayoutPending,  // Withdrawal request submitted, awaiting processing
    Retired,        // Payout cap reached or manually retired — archived
    Blown,          // Rule breached, account dead
    Suspended,      // Temporary pause
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAccount {
    pub account_id: Uuid,
    pub name: String,
    pub firm_name: String,
    pub lifecycle_state: AccountLifecycle,
    pub pipeline_position: usize,
    pub readiness_score: Decimal,
    pub activated_at: Option<DateTime<Utc>>,
    pub retired_at: Option<DateTime<Utc>>,
    pub rotation_reason: Option<String>,
    pub next_account_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub target_daily_extraction: Decimal,   // e.g. $25.00 USD/day
    pub max_concurrent_trading: usize,      // default 1 for prop firm extraction
    pub auto_rotate_on_payout: bool,
    pub auto_rotate_on_blow: bool,
    pub rotation_cooldown_hours: u32,
    pub preferred_firm_sequence: Vec<String>, // e.g. ["blue_guardian", "aquafunded"]
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            target_daily_extraction: Decimal::new(25, 0), // $25/day model
            max_concurrent_trading: 1,
            auto_rotate_on_payout: true,
            auto_rotate_on_blow: true,
            rotation_cooldown_hours: 1,
            preferred_firm_sequence: vec![
                "blue_guardian".into(),
                "aquafunded".into(),
                "the5ers".into(),
            ],
        }
    }
}

pub struct ReadinessScore;

impl ReadinessScore {
    /// Computes candidate account readiness score for bench selection:
    /// readiness = (1 - dd_pct)*0.30 + consistency*0.25 + (1 - strikes/limit)*0.20 + deadline*0.15 + perf*0.10
    pub fn compute(
        current_dd_pct: Decimal,
        max_dd_limit: Decimal,
        consistency_score: Decimal,
        strikes: u32,
        strike_limit: u32,
        days_until_deadline: u32,
    ) -> Decimal {
        let dd_ratio = if max_dd_limit > Decimal::ZERO {
            (current_dd_pct / max_dd_limit).to_f64().unwrap_or(0.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dd_component = (1.0 - dd_ratio) * 0.30;

        let conc_component = consistency_score.to_f64().unwrap_or(1.0).clamp(0.0, 1.0) * 0.25;

        let strike_ratio = if strike_limit > 0 {
            (strikes as f64 / strike_limit as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let strike_component = (1.0 - strike_ratio) * 0.20;

        let deadline_component = if days_until_deadline > 0 {
            (days_until_deadline as f64 / 30.0).clamp(0.0, 1.0) * 0.15
        } else {
            0.15
        };

        let total = dd_component + conc_component + strike_component + deadline_component + 0.10;
        Decimal::from_f64_retain(total.clamp(0.0, 1.0)).unwrap_or(Decimal::ONE)
    }
}

#[derive(Debug)]
pub struct PipelineManager {
    accounts: Arc<DashMap<Uuid, ManagedAccount>>,
    active_account_id: Arc<parking_lot::RwLock<Option<Uuid>>>,
    config: PipelineConfig,
}

impl PipelineManager {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            accounts: Arc::new(DashMap::new()),
            active_account_id: Arc::new(parking_lot::RwLock::new(None)),
            config,
        }
    }

    /// Add an account to the pipeline
    pub fn add_account(&self, account_id: Uuid, name: String, firm_name: String) -> ManagedAccount {
        let position = self.accounts.len();
        let managed = ManagedAccount {
            account_id,
            name,
            firm_name,
            lifecycle_state: AccountLifecycle::Acquired,
            pipeline_position: position,
            readiness_score: Decimal::ONE,
            activated_at: None,
            retired_at: None,
            rotation_reason: None,
            next_account_id: None,
        };

        self.accounts.insert(account_id, managed.clone());

        // Set as active if no active account exists
        let mut active = self.active_account_id.write();
        if active.is_none() {
            *active = Some(account_id);
            if let Some(mut acc) = self.accounts.get_mut(&account_id) {
                acc.lifecycle_state = AccountLifecycle::Trading;
                acc.activated_at = Some(Utc::now());
            }
        }

        managed
    }

    /// Get active trading account ID
    pub fn active_account_id(&self) -> Option<Uuid> {
        *self.active_account_id.read()
    }

    /// Update an account's lifecycle state
    pub fn set_lifecycle(&self, account_id: Uuid, state: AccountLifecycle, reason: Option<String>) {
        if let Some(mut account) = self.accounts.get_mut(&account_id) {
            account.lifecycle_state = state.clone();
            account.rotation_reason = reason;

            if state == AccountLifecycle::Retired || state == AccountLifecycle::Blown {
                account.retired_at = Some(Utc::now());
            }
        }
    }

    /// Rotates from current active account to the next best warm account on the bench
    pub fn rotate(&self, reason: String) -> QuantResult<Option<Uuid>> {
        let current_active = *self.active_account_id.read();
        
        if let Some(active_id) = current_active {
            if let Some(mut acc) = self.accounts.get_mut(&active_id) {
                if acc.lifecycle_state == AccountLifecycle::Trading {
                    acc.lifecycle_state = AccountLifecycle::Extracting;
                }
                acc.rotation_reason = Some(reason.clone());
            }
        }

        // Find candidate on bench with highest readiness score
        let mut candidates: Vec<ManagedAccount> = self
            .accounts
            .iter()
            .map(|entry| entry.value().clone())
            .filter(|a| {
                a.lifecycle_state == AccountLifecycle::Acquired
                    || a.lifecycle_state == AccountLifecycle::Warming
            })
            .collect();

        candidates.sort_by(|a, b| b.readiness_score.cmp(&a.readiness_score));

        if let Some(next) = candidates.first() {
            let next_id = next.account_id;

            if let Some(mut acc) = self.accounts.get_mut(&next_id) {
                acc.lifecycle_state = AccountLifecycle::Trading;
                acc.activated_at = Some(Utc::now());
            }

            if let Some(active_id) = current_active {
                if let Some(mut acc) = self.accounts.get_mut(&active_id) {
                    acc.next_account_id = Some(next_id);
                }
            }

            *self.active_account_id.write() = Some(next_id);

            tracing::info!(
                "PIPELINE ROTATION: Swapped active account from {:?} to {} (Reason: {})",
                current_active,
                next_id,
                reason
            );

            Ok(Some(next_id))
        } else {
            tracing::warn!("PIPELINE ROTATION: No warm candidate available on bench!");
            Ok(None)
        }
    }

    /// Evaluates auto-rotation triggers (e.g. payout cap reached or account blown)
    pub fn check_auto_rotation(
        &self,
        account_id: Uuid,
        is_payout_cap_reached: bool,
        shield_status: &ShieldStatus,
        consistency_status: &ComplianceStatus,
    ) -> QuantResult<bool> {
        let active = *self.active_account_id.read();
        if active != Some(account_id) {
            return Ok(false);
        }

        if is_payout_cap_reached && self.config.auto_rotate_on_payout {
            self.set_lifecycle(account_id, AccountLifecycle::Retired, Some("Payout cap reached".into()));
            self.rotate("Payout cap reached ($25/day extraction complete)".into())?;
            return Ok(true);
        }

        if *shield_status == ShieldStatus::Blown && self.config.auto_rotate_on_blow {
            self.set_lifecycle(account_id, AccountLifecycle::Blown, Some("Guardian Shield blown".into()));
            self.rotate("Guardian Shield blown (strike limit reached)".into())?;
            return Ok(true);
        }

        if *consistency_status == ComplianceStatus::Breached {
            self.set_lifecycle(account_id, AccountLifecycle::Suspended, Some("Consistency breach".into()));
            self.rotate("Consistency breach".into())?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Lists all managed pipeline accounts
    pub fn list_accounts(&self) -> Vec<ManagedAccount> {
        self.accounts.iter().map(|entry| entry.value().clone()).collect()
    }
}
