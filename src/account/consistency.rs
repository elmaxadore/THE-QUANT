//! # Consistency Rule Engine (v3.1 Hephaestus)
//!
//! Native enforcement of prop-firm consistency rules (e.g. Blue Guardian 15% consistency).
//! Tracks profit concentration, lot size consistency, trade frequency, and duration consistency.
//! Generates auto-position-sizing adjustments and consistency score metrics.

use crate::error::QuantResult;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsistencyRuleType {
    ProfitConcentration,
    LotSizeConsistency,
    TradeFrequency,
    DurationConsistency,
    DailyProfitCap,
    WinRateConsistency,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnforcementMode {
    Hard,
    Soft,
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComplianceStatus {
    Compliant,
    Warning,
    Breached,
    ShieldActive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyProfile {
    pub rule_type: ConsistencyRuleType,
    pub threshold_pct: Decimal,           // e.g. 15.0 for 15% max single-day concentration
    pub lookback_days: u32,
    pub enforcement_mode: EnforcementMode,
    pub auto_adjust_sizing: bool,
}

impl Default for ConsistencyProfile {
    fn default() -> Self {
        Self {
            rule_type: ConsistencyRuleType::ProfitConcentration,
            threshold_pct: Decimal::new(15, 0), // 15%
            lookback_days: 30,
            enforcement_mode: EnforcementMode::Hard,
            auto_adjust_sizing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyState {
    pub account_id: Uuid,
    pub profit_concentration_max: Decimal,   // max single-day % of total profit
    pub lot_size_cv: Decimal,                // coefficient of variation (std_dev / mean)
    pub trade_frequency_7d: u32,
    pub avg_hold_time_minutes: Decimal,
    pub daily_profit_high_water: Decimal,
    pub consistency_score: Decimal,          // 0.0 – 1.0 composite
    pub compliance_status: ComplianceStatus,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub pnl: Decimal,
    pub lot_size: Decimal,
    pub duration_minutes: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Default)]
pub struct ConsistencyEngine;

impl ConsistencyEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the consistency state of an account based on trade history and profile
    pub fn evaluate(
        &self,
        account_id: Uuid,
        trades: &[TradeRecord],
        cumulative_profit: Decimal,
        profile: &ConsistencyProfile,
    ) -> ConsistencyState {
        if trades.is_empty() || cumulative_profit <= Decimal::ZERO {
            return ConsistencyState {
                account_id,
                profit_concentration_max: Decimal::ZERO,
                lot_size_cv: Decimal::ZERO,
                trade_frequency_7d: trades.len() as u32,
                avg_hold_time_minutes: Decimal::ZERO,
                daily_profit_high_water: Decimal::ZERO,
                consistency_score: Decimal::ONE,
                compliance_status: ComplianceStatus::Compliant,
                last_updated: Utc::now(),
            };
        }

        // 1. Calculate Profit Concentration
        let max_trade_pnl = trades.iter().map(|t| t.pnl).max().unwrap_or(Decimal::ZERO);
        let profit_concentration_max = if cumulative_profit > Decimal::ZERO {
            (max_trade_pnl / cumulative_profit) * Decimal::new(100, 0)
        } else {
            Decimal::ZERO
        };

        // 2. Calculate Lot Size Coefficient of Variation (CV)
        let lot_sizes: Vec<f64> = trades.iter().map(|t| t.lot_size.to_f64().unwrap_or(0.0)).collect();
        let mean_lot = lot_sizes.iter().sum::<f64>() / lot_sizes.len() as f64;
        let var_lot = if lot_sizes.len() > 1 && mean_lot > 0.0 {
            lot_sizes.iter().map(|l| (l - mean_lot).powi(2)).sum::<f64>() / (lot_sizes.len() as f64 - 1.0)
        } else {
            0.0
        };
        let std_dev_lot = var_lot.sqrt();
        let cv_f64 = if mean_lot > 0.0 { std_dev_lot / mean_lot } else { 0.0 };
        let lot_size_cv = Decimal::from_f64_retain(cv_f64).unwrap_or(Decimal::ZERO);

        // 3. Average Hold Time
        let total_hold: Decimal = trades.iter().map(|t| t.duration_minutes).sum();
        let avg_hold_time_minutes = total_hold / Decimal::from(trades.len() as u64);

        // 4. Score components (0.0 to 1.0)
        let thresh_f64 = profile.threshold_pct.to_f64().unwrap_or(15.0);
        let conc_f64 = profit_concentration_max.to_f64().unwrap_or(0.0);

        let s_profit = (1.0 - (conc_f64 / thresh_f64).clamp(0.0, 1.0)).max(0.0);
        let s_lot = (1.0 - (cv_f64 / 0.5).clamp(0.0, 1.0)).max(0.0);
        let s_freq = if trades.len() >= 5 { 1.0 } else { trades.len() as f64 / 5.0 };
        let s_time = if avg_hold_time_minutes >= Decimal::new(2, 0) { 1.0 } else { 0.5 };

        // Composite harmonic mean with weights: profit=0.4, lot=0.25, freq=0.2, time=0.15
        let weighted_inv = 0.4 / (s_profit + 1e-6)
            + 0.25 / (s_lot + 1e-6)
            + 0.20 / (s_freq + 1e-6)
            + 0.15 / (s_time + 1e-6);
        let score_f64 = (1.0 / weighted_inv).clamp(0.0, 1.0);
        let consistency_score = Decimal::from_f64_retain(score_f64).unwrap_or(Decimal::ONE);

        // Compliance status
        let compliance_status = if profit_concentration_max > profile.threshold_pct {
            ComplianceStatus::Breached
        } else if profit_concentration_max > profile.threshold_pct * Decimal::new(8, 1) {
            ComplianceStatus::Warning
        } else if score_f64 < 0.8 {
            ComplianceStatus::ShieldActive
        } else {
            ComplianceStatus::Compliant
        };

        ConsistencyState {
            account_id,
            profit_concentration_max,
            lot_size_cv,
            trade_frequency_7d: trades.len() as u32,
            avg_hold_time_minutes,
            daily_profit_high_water: max_trade_pnl,
            consistency_score,
            compliance_status,
            last_updated: Utc::now(),
        }
    }

    /// Computes position sizing cap to preserve consistency rules
    pub fn compute_position_sizing_cap(
        &self,
        base_size: Decimal,
        state: &ConsistencyState,
        profile: &ConsistencyProfile,
    ) -> Decimal {
        if !profile.auto_adjust_sizing {
            return base_size;
        }

        let mut multiplier = state.consistency_score;
        if state.compliance_status == ComplianceStatus::Warning {
            multiplier = multiplier * Decimal::new(7, 1); // 70% of base
        } else if state.compliance_status == ComplianceStatus::Breached {
            multiplier = Decimal::ZERO;
        }

        base_size * multiplier
    }
}
