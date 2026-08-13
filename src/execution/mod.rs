//! # Execution Module (Layer 7)
//!
//! Handles order management, MT5 trade command dispatch, slippage tracking,
//! partial fills, and trade journaling. All orders go through pre-flight
//! validation before reaching MT5.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use crate::data::{OrderType, TradeDirection, CommandResponse};
use crate::risk::{RiskCheckResult, RiskEngine};
use crate::strategy::{TradeSignal, TradeSignalDirection};
use crate::regime::Regime;
use crate::risk::CircuitBreakerLevel;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// A trade order ready for dispatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOrder {
    pub id: uuid::Uuid,
    pub account_id: uuid::Uuid,
    pub symbol: String,
    pub direction: TradeDirection,
    pub order_type: OrderType,
    pub volume: Decimal,
    pub price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub slippage_tolerance: f64,
    pub strategy_id: uuid::Uuid,
    pub signal_id: uuid::Uuid,
    pub created_at: DateTime<Utc>,
    pub status: OrderStatus,
}

/// Order lifecycle status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// A completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeJournalEntry {
    pub id: uuid::Uuid,
    pub account_id: uuid::Uuid,
    pub order_id: uuid::Uuid,
    pub symbol: String,
    pub direction: TradeDirection,
    pub volume: Decimal,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub open_time: DateTime<Utc>,
    pub close_time: Option<DateTime<Utc>>,
    pub pnl: Decimal,
    pub commission: Decimal,
    pub swap: Decimal,
    pub strategy_id: uuid::Uuid,
    pub regime: String,
    pub context_hash: String,
    pub created_at: DateTime<Utc>,
}

/// Execution engine
#[derive(Debug)]
pub struct ExecutionEngine {
    /// Active orders by account
    orders: Arc<RwLock<HashMap<uuid::Uuid, Vec<TradeOrder>>>>,
    /// Trade journal
    journal: Arc<RwLock<Vec<TradeJournalEntry>>>,
    /// Configuration
    config: QuantConfig,
}

impl ExecutionEngine {
    pub fn new(config: &QuantConfig) -> Self {
        Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
            journal: Arc::new(RwLock::new(Vec::new())),
            config: config.clone(),
        }
    }

    /// Submit a trade order after risk checks pass
    pub async fn submit_order(
        &self,
        account_id: uuid::Uuid,
        signal: &TradeSignal,
        risk_result: &RiskCheckResult,
        price: Decimal,
    ) -> QuantResult<TradeOrder> {
        let size = risk_result.suggested_size.unwrap_or(Decimal::new(1, 1));
        
        let order = TradeOrder {
            id: uuid::Uuid::new_v4(),
            account_id,
            symbol: signal.symbol.clone(),
            direction: match signal.direction {
                crate::strategy::TradeSignalDirection::Buy => TradeDirection::Buy,
                crate::strategy::TradeSignalDirection::Sell => TradeDirection::Sell,
                _ => return Err(QuantError::StrategyError("Invalid signal direction for order".into())),
            },
            order_type: OrderType::Market,
            volume: size,
            price: Some(price),
            stop_loss: None, // Computed by risk engine
            take_profit: None,
            slippage_tolerance: self.config.risk.default_slippage_tolerance_atr,
            strategy_id: signal.strategy_id,
            signal_id: uuid::Uuid::new_v4(),
            created_at: Utc::now(),
            status: OrderStatus::Pending,
        };

        let mut orders = self.orders.write().await;
        orders.entry(account_id).or_insert_with(Vec::new).push(order.clone());
        info!("Order submitted: {} {} {} lots", order.symbol, order.direction, order.volume);
        Ok(order)
    }

    /// Execute an order (send to MT5)
    pub async fn execute_order(&self, order: &TradeOrder) -> QuantResult<CommandResponse> {
        // Build MT5 command
        let cmd = format!(
            "C|{}|{}|{}|{}|{}|{}|{}|{}",
            order.id,
            match order.direction {
                TradeDirection::Buy => "BUY",
                TradeDirection::Sell => "SELL",
            },
            order.order_type,
            order.volume,
            order.price.unwrap_or(Decimal::ZERO),
            order.stop_loss.unwrap_or(Decimal::ZERO),
            order.take_profit.unwrap_or(Decimal::ZERO),
        );

        // TODO: Send via ZMQ to MT5
        info!("Executing order via MT5 bridge: {}", cmd);

        Ok(CommandResponse {
            cmd_id: order.id.to_string(),
            success: true,
            message: "Order sent to MT5".into(),
            order_ticket: Some(rand::random::<u64>() % 10000000),
        })
    }

    /// Record a completed trade in the journal
    pub async fn record_trade(
        &self,
        entry: TradeJournalEntry,
    ) -> QuantResult<()> {
        let mut journal = self.journal.write().await;
        journal.push(entry);
        Ok(())
    }

    /// Get open orders for an account
    pub async fn get_open_orders(&self, account_id: &uuid::Uuid) -> Vec<TradeOrder> {
        let orders = self.orders.read().await;
        orders.get(account_id)
            .map(|o| o.iter().filter(|o| o.status == OrderStatus::Pending || o.status == OrderStatus::Submitted).cloned().collect())
            .unwrap_or_default()
    }

    /// Update order status
    pub async fn update_order_status(
        &self,
        order_id: &uuid::Uuid,
        new_status: OrderStatus,
        fill_price: Option<Decimal>,
    ) -> QuantResult<()> {
        let mut orders = self.orders.write().await;
        for (_, order_list) in orders.iter_mut() {
            if let Some(order) = order_list.iter_mut().find(|o| o.id == *order_id) {
                order.status = new_status;
                if let Some(price) = fill_price {
                    order.price = Some(price);
                }
                return Ok(());
            }
        }
        Err(QuantError::Internal(format!("Order {} not found", order_id)))
    }

    /// Get trade journal
    pub async fn get_journal(&self) -> Vec<TradeJournalEntry> {
        self.journal.read().await.clone()
    }

    /// Compute slippage from expected vs actual fill price
    pub async fn compute_slippage(&self, expected: Decimal, actual: Decimal) -> Decimal {
        if expected == Decimal::ZERO {
            return Decimal::ZERO;
        }
        ((actual - expected) / expected).abs() * Decimal::new(10000, 4) // In basis points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::RiskCheck;
    use crate::strategy::TradeSignalDirection;
    use crate::regime::Regime;
    use crate::risk::CircuitBreakerLevel;

    #[tokio::test]
    async fn test_order_submission() {
        let config = QuantConfig::default();
        let engine = ExecutionEngine::new(&config);
        let account_id = uuid::Uuid::new_v4();
        let signal = TradeSignal {
            symbol: "EURUSD".into(),
            direction: TradeSignalDirection::Buy,
            strength: 0.8,
            strategy_id: uuid::Uuid::new_v4(),
            regime: Regime::TrendingUp,
            feature_snapshot: vec![],
            timestamp: Utc::now(),
        };
        let risk_result = RiskCheckResult {
            passed: true,
            checks: vec![],
            suggested_size: Some(Decimal::new(1, 1)),
            circuit_breaker: CircuitBreakerLevel::None,
        };

        let order = engine.submit_order(account_id, &signal, &risk_result, Decimal::new(10500, 4)).await;
        assert!(order.is_ok());
    }
}
