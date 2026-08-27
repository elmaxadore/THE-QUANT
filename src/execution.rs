//! EXECUTION ENGINE — Layer 6 (MT5 bridge command side).
//!
//! Converts approved intents into fills at the market price of the next bar.
//! In a live deployment these are the `C|` commands over ZeroMQ to the MQL5
//! Bridge EA; here they are applied locally against the simulated feed.

use crate::state::PositionState;
use crate::strategy::Direction;
use crate::util::now_iso8601;

/// Units of base currency per 1.0 "lot" (standard FX/crypto convention).
/// Using this keeps P&L and risk sizing coherent:
///   loss at stop = stop_dist * lots * CONTRACT_SIZE
///   P&L          = (price - entry) * lots * CONTRACT_SIZE * direction
pub const CONTRACT_SIZE: f64 = 100_000.0;

/// Compute realised P&L for a position given the exit price, in account currency.
fn pnl_for(pos: &PositionState, exit_price: f64) -> f64 {
    let dir = if pos.direction == "BUY" { 1.0 } else { -1.0 };
    (exit_price - pos.entry_price) * pos.volume * CONTRACT_SIZE * dir
}

#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub direction: Direction,
    pub volume: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
}

impl OrderRequest {
    pub fn new(symbol: &str, direction: Direction, volume: f64, sl: f64, tp: f64) -> Self {
        OrderRequest { symbol: symbol.into(), direction, volume, stop_loss: sl, take_profit: tp }
    }
}

/// The trader's margin/position keeper.
#[derive(Debug, Clone)]
pub struct ExecutionEngine {
    /// symbol -> open position
    pub open: Option<PositionState>,
    pub net_pnl: f64,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        ExecutionEngine { open: None, net_pnl: 0.0 }
    }

    pub fn has_position(&self) -> bool {
        self.open.is_some()
    }

    /// Execute an order against a bar's close price (market fill).
    ///
    /// Returns the filled Order if a position was opened, or the closed
    /// PositionState if the order closed an existing position.
    pub fn execute(&mut self, order: OrderRequest, price: f64) -> OpenOrClose {
        match order.direction {
            Direction::Buy | Direction::Sell => {
                // Open (or replace) a position.
                let pos = PositionState {
                    symbol: order.symbol.clone(),
                    direction: order.direction.as_str().to_string(),
                    volume: order.volume,
                    entry_price: price,
                    stop_loss: order.stop_loss,
                    take_profit: order.take_profit,
                    open_time: now_iso8601(),
                    unrealized_pnl: 0.0,
                };
                let prev = self.open.replace(pos);
                OpenOrClose {
                    opened: Some(prev.map(|_| true).unwrap_or(false)),
                    closed: None,
                }
            }
            Direction::Close | Direction::Hold => {
                if let Some(p) = self.open.take() {
                    let pnl = pnl_for(&p, price);
                    self.net_pnl += pnl;
                    OpenOrClose { opened: None, closed: Some(p.to_trade(pnl, price)) }
                } else {
                    OpenOrClose { opened: None, closed: None }
                }
            }
        }
    }

    /// Revalue the open position using the current price.
    pub fn mark_to_market(&mut self, price: f64) -> f64 {
        if let Some(p) = self.open.as_mut() {
            p.unrealized_pnl = pnl_for(p, price);
            p.unrealized_pnl
        } else {
            0.0
        }
    }
}

/// Result of an execution: whether a position was opened and/or one closed.
#[derive(Debug, Clone)]
pub struct OpenOrClose {
    pub opened: Option<bool>,
    pub closed: Option<PositionState>,
}

impl ExecutionEngine {
    pub fn default() -> Self {
        Self::new()
    }
}

trait ToTrade {
    fn to_trade(self, pnl: f64, exit_price: f64) -> PositionState;
}

impl ToTrade for PositionState {
    fn to_trade(mut self, pnl: f64, exit_price: f64) -> PositionState {
        self.unrealized_pnl = pnl;
        self.take_profit = exit_price; // reuse field to carry exit for simplicity
        self
    }
}