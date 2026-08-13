//! # Payout Cap & Extraction Optimization (v3.1 Hephaestus)
//!
//! Models lifetime payout caps (e.g. $250 on a $5K instant account), models extraction velocity,
//! computes optimal extraction curves via sigmoid trajectories, and calculates safe daily profit targets.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PayoutSchedule {
    OnDemand,
    BiWeekly,
    Monthly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutCap {
    pub enabled: bool,
    pub total_cap: Decimal,               // e.g. $250.00 USD
    pub already_paid_out: Decimal,        // historical payouts
    pub remaining_cap: Decimal,           // total_cap - already_paid_out
    pub min_days_before_first_payout: u32, // e.g. 0 for instant
    pub payout_schedule: PayoutSchedule,
    pub payout_fee_pct: Decimal,
}

impl Default for PayoutCap {
    fn default() -> Self {
        Self {
            enabled: true,
            total_cap: Decimal::new(250, 0),
            already_paid_out: Decimal::ZERO,
            remaining_cap: Decimal::new(250, 0),
            min_days_before_first_payout: 0,
            payout_schedule: PayoutSchedule::OnDemand,
            payout_fee_pct: Decimal::ZERO,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionState {
    pub account_id: Uuid,
    pub current_equity: Decimal,
    pub current_profit: Decimal,
    pub days_traded: u32,
    pub avg_daily_profit: Decimal,
    pub projected_payout_date: Option<DateTime<Utc>>,
    pub extraction_velocity: Decimal,     // profit per day
    pub payout_progress_pct: Decimal,     // profit / total_cap
    pub economic_value_remaining: Decimal, // EV of remaining cap
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionCurvePoint {
    pub day: u32,
    pub target_equity: Decimal,
    pub actual_equity: Decimal,
    pub deviation_pct: Decimal,
}

#[derive(Debug, Default)]
pub struct PayoutEngine;

impl PayoutEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates current extraction status and progress toward payout cap
    pub fn evaluate_extraction(
        &self,
        account_id: Uuid,
        initial_balance: Decimal,
        current_equity: Decimal,
        days_traded: u32,
        cap: &PayoutCap,
    ) -> ExtractionState {
        let current_profit = (current_equity - initial_balance).max(Decimal::ZERO);
        let remaining_cap = (cap.total_cap - cap.already_paid_out - current_profit).max(Decimal::ZERO);
        let days = days_traded.max(1);
        let avg_daily_profit = current_profit / Decimal::from(days as u64);
        let extraction_velocity = avg_daily_profit;

        let progress_f64 = if cap.total_cap > Decimal::ZERO {
            (current_profit / cap.total_cap).to_f64().unwrap_or(0.0) * 100.0
        } else {
            0.0
        };
        let payout_progress_pct = Decimal::from_f64_retain(progress_f64).unwrap_or(Decimal::ZERO);

        let projected_days_left = if avg_daily_profit > Decimal::ZERO {
            (remaining_cap / avg_daily_profit).to_f64().unwrap_or(0.0).ceil() as i64
        } else {
            30
        };

        let projected_payout_date = Some(Utc::now() + chrono::Duration::days(projected_days_left));

        ExtractionState {
            account_id,
            current_equity,
            current_profit,
            days_traded,
            avg_daily_profit,
            projected_payout_date,
            extraction_velocity,
            payout_progress_pct,
            economic_value_remaining: remaining_cap,
        }
    }

    /// Sigmoid curve computation for smooth target profit trajectory:
    /// TargetEquity(t) = InitialBalance + (Cap * Sigmoid(t / T))
    pub fn compute_sigmoid_target(
        &self,
        initial_balance: Decimal,
        cap: Decimal,
        current_day: u32,
        expected_total_days: u32,
    ) -> Decimal {
        if expected_total_days == 0 {
            return initial_balance + cap;
        }

        let x = (current_day as f64 / expected_total_days as f64) * 6.0 - 3.0; // range [-3, 3]
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let cap_f64 = cap.to_f64().unwrap_or(250.0);
        let target_profit = cap_f64 * sigmoid;

        initial_balance + Decimal::from_f64_retain(target_profit).unwrap_or(Decimal::ZERO)
    }

    /// Calculates safe daily profit target incorporating rule headroom
    pub fn compute_safe_daily_target(
        &self,
        cap: &PayoutCap,
        current_profit: Decimal,
        days_traded: u32,
        min_days_to_payout: u32,
        dd_headroom: Decimal,
        consistency_headroom: Decimal,
        shield_headroom: Decimal,
    ) -> Decimal {
        let remaining_profit = (cap.total_cap - cap.already_paid_out - current_profit).max(Decimal::ZERO);
        let days_left = if min_days_to_payout > days_traded {
            min_days_to_payout - days_traded
        } else {
            10
        };
        let required_daily = remaining_profit / Decimal::from(days_left as u64);

        let safe_target = required_daily * dd_headroom * consistency_headroom * shield_headroom;
        safe_target.max(Decimal::ZERO)
    }
}
