//! # Guardian Shield — Per-Trade Loss Circuit Breaker (v3.1 Hephaestus)
//!
//! Enforces per-trade loss limits with strike counting (e.g. Blue Guardian Instant 5K $50 per-trade max loss).
//! Pre-flight validation guarantees no order is submitted that could breach the per-trade limit.

use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShieldStatus {
    Active,     // No strikes, full trading
    Warning,    // 1 strike, reduced size if configured
    Critical,   // At strike_limit - 1, minimum size only
    Blown,      // strike_limit reached — account halted
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardianShield {
    pub enabled: bool,
    pub max_loss_per_trade: Decimal,      // e.g. $50.00 USD
    pub max_loss_pct_per_trade: Decimal,  // alternative: % of account
    pub strike_limit: u32,                // e.g. 2 strikes
    pub current_strikes: u32,
    pub cooldown_bars: u32,               // bars to wait after strike
    pub auto_reduce_size_after_strike: bool,
    pub reduction_factor: Decimal,        // e.g. 0.5 (halve size)
}

impl Default for GuardianShield {
    fn default() -> Self {
        Self {
            enabled: true,
            max_loss_per_trade: Decimal::new(50, 0), // $50 default
            max_loss_pct_per_trade: Decimal::new(1, 0), // 1%
            strike_limit: 2,
            current_strikes: 0,
            cooldown_bars: 0,
            auto_reduce_size_after_strike: true,
            reduction_factor: Decimal::new(5, 1), // 0.5
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldState {
    pub account_id: Uuid,
    pub strikes: u32,
    pub last_strike_time: Option<DateTime<Utc>>,
    pub shield_status: ShieldStatus,
    pub trade_count_since_last_strike: u32,
}

#[derive(Debug, Default)]
pub struct ShieldEngine;

impl ShieldEngine {
    pub fn new() -> Self {
        Self
    }

    /// Pre-flight validation before trade submission. Rejects trades exceeding worst-case loss limits.
    pub fn pre_flight_check(
        &self,
        worst_case_loss: Decimal,
        shield: &GuardianShield,
        state: &ShieldState,
    ) -> QuantResult<()> {
        if !shield.enabled {
            return Ok(());
        }

        if state.shield_status == ShieldStatus::Blown {
            return Err(QuantError::AccountNotTradable {
                account_id: state.account_id.to_string(),
                status: "Guardian Shield blown (strike limit reached)".into(),
            });
        }

        // Hard cap check
        if worst_case_loss > shield.max_loss_per_trade {
            return Err(QuantError::RuleBreachImminent {
                rule: format!(
                    "Guardian Shield: Proposed worst-case loss ${:.2} exceeds per-trade limit ${:.2}",
                    worst_case_loss, shield.max_loss_per_trade
                ),
            });
        }

        // Critical status tighter check
        if state.shield_status == ShieldStatus::Critical
            && worst_case_loss > (shield.max_loss_per_trade * Decimal::new(5, 1))
        {
            return Err(QuantError::RuleBreachImminent {
                rule: format!(
                    "Guardian Shield Critical Mode: Loss ${:.2} exceeds conservative cap ${:.2}",
                    worst_case_loss,
                    shield.max_loss_per_trade * Decimal::new(5, 1)
                ),
            });
        }

        Ok(())
    }

    /// Post-trade evaluation of realized loss for strike detection
    pub fn evaluate_closed_trade(
        &self,
        realized_loss: Decimal,
        shield: &GuardianShield,
        state: &mut ShieldState,
        is_gap_exception: bool,
    ) -> bool {
        if !shield.enabled || realized_loss <= shield.max_loss_per_trade {
            state.trade_count_since_last_strike += 1;
            return false;
        }

        // Mechanical market gap exception does not increment strike
        if is_gap_exception {
            tracing::warn!(
                "Guardian Shield: Loss ${:.2} exceeded limit ${:.2} but marked as GAP EXCEPTION",
                realized_loss,
                shield.max_loss_per_trade
            );
            return false;
        }

        state.strikes += 1;
        state.last_strike_time = Some(Utc::now());
        state.trade_count_since_last_strike = 0;

        if state.strikes >= shield.strike_limit {
            state.shield_status = ShieldStatus::Blown;
        } else if state.strikes == shield.strike_limit - 1 {
            state.shield_status = ShieldStatus::Critical;
        } else {
            state.shield_status = ShieldStatus::Warning;
        }

        tracing::error!(
            "Guardian Shield STRIKE #{} for account {}: Loss ${:.2} > Limit ${:.2}. Status: {:?}",
            state.strikes,
            state.account_id,
            realized_loss,
            shield.max_loss_per_trade,
            state.shield_status
        );

        true
    }
}
