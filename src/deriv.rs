//! DERIV API INTEGRATION — Layer 6 (Deriv.com bridge).
//!
//! Provides live market data and order execution via Deriv's WebSocket API.
//! Deriv supports synthetic indices, forex, commodities, and cryptocurrencies
//! with 24/7 trading on synthetic indices and regular market hours for others.
//!
//! References:
//!   - Deriv API Docs: https://developers.deriv.com/api
//!   - WebSocket Endpoint: wss://ws.derivws.com/websockets/v3
//!   - Authentication: OAuth2 or app_id + token
//!
//! Features:
//!   * Real-time ticks via `ticks` stream
//!   * OHLC candles via `ohlc` stream  
//!   * Order placement (buy/sell contracts)
//!   * Account balance and position queries
//!   * Auto-reconnect with exponential backoff
//!   * Rate limiting compliance (1 request/second default)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Deriv API application credentials.
/// Register your app at: https://api.deriv.com/
#[derive(Debug, Clone)]
pub struct DerivCredentials {
    pub app_id: u32,
    pub api_token: Option<String>, // Optional for read-only; required for trading
}

/// Supported Deriv markets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DerivMarket {
    Forex,
    SyntheticIndices,
    Commodities,
    Cryptocurrencies,
    Stocks,
    Indices,
}

/// Deriv-specific symbol mapping (broker symbol -> canonical asset).
pub fn deriv_symbol_map(symbol: &str) -> String {
    match symbol.to_uppercase().as_str() {
        "R_10" | "FR10" => "SYNTHETIC_10".into(),
        "R_25" | "FR25" => "SYNTHETIC_25".into(),
        "R_50" | "FR50" => "SYNTHETIC_50".into(),
        "R_75" | "FR75" => "SYNTHETIC_75".into(),
        "R_100" | "FR100" => "SYNTHETIC_100".into(),
        "BOOM1000" => "BOOM_1000".into(),
        "CRASH1000" => "CRASH_1000".into(),
        "VOL10" => "VOLATILITY_10".into(),
        "VOL25" => "VOLATILITY_25".into(),
        "VOL50" => "VOLATILITY_50".into(),
        "VOL75" => "VOLATILITY_75".into(),
        "VOL100" => "VOLATILITY_100".into(),
        "1HZ10V" => "VOLATILITY_10_1HZ".into(),
        "1HZ25V" => "VOLATILITY_25_1HZ".into(),
        "1HZ50V" => "VOLATILITY_50_1HZ".into(),
        "1HZ75V" => "VOLATILITY_75_1HZ".into(),
        "1HZ100V" => "VOLATILITY_100_1HZ".into(),
        "WOLF" => "WOLF_WHEEL".into(),
        "MUL1000" => "MULTIPLIER_UP".into(),
        "frxEURUSD" => "EURUSD".into(),
        "frxGBPUSD" => "GBPUSD".into(),
        "frxUSDJPY" => "USDJPY".into(),
        "frxAUDUSD" => "AUDUSD".into(),
        "frxUSDCAD" => "USDCAD".into(),
        "frxUSDCHF" => "USDCHF".into(),
        "frxNZDUSD" => "NZDUSD".into(),
        "XAUUSD" => "XAUUSD".into(),
        "XAGUSD" => "XAGUSD".into(),
        "BTCUSD" => "BTCUSD".into(),
        "ETHUSD" => "ETHUSD".into(),
        _ => symbol.to_string(),
    }
}

/// WebSocket message types from Deriv API.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DerivMessage {
    Authorize(AuthorizeResponse),
    Ticks(TicksResponse),
    Ohlc(OhlcResponse),
    Proposal(ProposalResponse),
    Buy(BuyResponse),
    Balance(BalanceResponse),
    Error(ErrorResponse),
    Ping(PingResponse),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthorizeResponse {
    pub msg_type: String,
    pub authorize: Option<AuthorizeData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthorizeData {
    pub email: String,
    pub currency: String,
    pub balance: f64,
    pub loginid: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TicksResponse {
    pub msg_type: String,
    pub tick: Option<TickData>,
    pub subscription: Option<SubscriptionData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TickData {
    pub symbol: String,
    pub price: f64,
    pub epoch: i64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubscriptionData {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OhlcResponse {
    pub msg_type: String,
    pub ohlc: Option<OhlcData>,
    pub subscription: Option<SubscriptionData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OhlcData {
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub epoch: i64,
    pub granularity: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProposalResponse {
    pub msg_type: String,
    pub proposal: Option<ProposalData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProposalData {
    pub id: String,
    pub ask_price: f64,
    pub payout: f64,
    pub currency: String,
    pub underlying: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BuyResponse {
    pub msg_type: String,
    pub buy: Option<BuyData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BuyData {
    pub transaction_id: u64,
    pub buy_price: f64,
    pub payout: f64,
    pub currency: String,
    pub underlying: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BalanceResponse {
    pub msg_type: String,
    pub balance: Option<BalanceData>,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BalanceData {
    pub balance: f64,
    pub currency: String,
    pub loginid: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub msg_type: String,
    pub error: Option<ErrorData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorData {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PingResponse {
    pub msg_type: String,
    pub ping: u64,
}

/// Convert Deriv tick to internal Bar format.
pub fn tick_to_bar(tick: &TickData, symbol: &str) -> crate::simfeed::Bar {
    crate::simfeed::Bar {
        symbol: symbol.to_string(),
        time: tick.epoch,
        open: tick.price,
        high: tick.price,
        low: tick.price,
        close: tick.price,
        volume: 0.0,
    }
}

/// Convert Deriv OHLC to internal Bar format.
pub fn ohlc_to_bar(ohlc: &OhlcData, symbol: &str) -> crate::simfeed::Bar {
    crate::simfeed::Bar {
        symbol: symbol.to_string(),
        time: ohlc.epoch,
        open: ohlc.open,
        high: ohlc.high,
        low: ohlc.low,
        close: ohlc.close,
        volume: 0.0,
    }
}

/// Deriv feed configuration.
#[derive(Debug, Clone)]
pub struct DerivFeedConfig {
    pub app_id: u32,
    pub api_token: Option<String>,
    pub symbols: Vec<String>,
    pub granularity: u32, // in seconds (e.g., 300 for M5)
    pub use_ticks: bool,  // true = tick stream, false = OHLC stream
}

impl Default for DerivFeedConfig {
    fn default() -> Self {
        DerivFeedConfig {
            app_id: 1089, // Default public app ID (for testing only)
            api_token: None,
            symbols: vec!["R_50".to_string()],
            granularity: 300,
            use_ticks: true,
        }
    }
}

/// Rate limiter for Deriv API (1 req/sec default, adjust per API tier).
pub struct RateLimiter {
    min_interval: Duration,
    last_call: Instant,
}

impl RateLimiter {
    pub fn new(requests_per_second: u32) -> Self {
        let interval = if requests_per_second > 0 {
            Duration::from_secs(1) / requests_per_second
        } else {
            Duration::from_millis(100)
        };
        RateLimiter {
            min_interval: interval,
            last_call: Instant::now() - interval,
        }
    }

    pub async fn wait(&mut self) {
        let elapsed = self.last_call.elapsed();
        if elapsed < self.min_interval {
            tokio::time::sleep(self.min_interval - elapsed).await;
        }
        self.last_call = Instant::now();
    }
}

/// Connection state for Deriv WebSocket.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Authorized,
    Subscribed,
    Error(String),
}

/// Deriv WebSocket client wrapper.
/// 
/// Note: This uses tokio-tungstenite for WebSocket communication.
/// The actual implementation would need to be wired into the engine's
/// feed system similar to Mt5Feed in simfeed.rs.
pub struct DerivClient {
    config: DerivFeedConfig,
    connection_state: ConnectionState,
    rate_limiter: RateLimiter,
    subscriptions: HashMap<String, String>, // symbol -> subscription_id
    last_error: Option<String>,
    reconnect_attempts: u32,
    max_reconnect_attempts: u32,
}

impl DerivClient {
    pub fn new(config: DerivFeedConfig) -> Self {
        DerivClient {
            config,
            connection_state: ConnectionState::Disconnected,
            rate_limiter: RateLimiter::new(1),
            subscriptions: HashMap::new(),
            last_error: None,
            reconnect_attempts: 0,
            max_reconnect_attempts: 5,
        }
    }

    /// Get current connection state.
    pub fn state(&self) -> &ConnectionState {
        &self.connection_state
    }

    /// Get last error message.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Build WebSocket URL for Deriv API.
    pub fn ws_url(&self) -> String {
        format!(
            "wss://ws.derivws.com/websockets/v3?app_id={}",
            self.config.app_id
        )
    }

    /// Build authorize request.
    pub fn authorize_request(&self) -> String {
        if let Some(ref token) = self.config.api_token {
            serde_json::json!({
                "authorize": token
            })
            .to_string()
        } else {
            // Read-only mode
            serde_json::json!({
                "balance": 1
            })
            .to_string()
        }
    }

    /// Build ticks subscription request.
    pub fn ticks_subscribe(&self, symbol: &str) -> String {
        serde_json::json!({
            "ticks": symbol,
            "subscribe": 1
        })
        .to_string()
    }

    /// Build OHLC subscription request.
    pub fn ohlc_subscribe(&self, symbol: &str, granularity: u32) -> String {
        serde_json::json!({
            "ohlc": symbol,
            "granularity": granularity,
            "subscribe": 1
        })
        .to_string()
    }

    /// Build proposal request for buying a contract.
    pub fn proposal_request(
        &self,
        symbol: &str,
        amount: f64,
        direction: &str,
        duration: u32,
        duration_unit: &str,
    ) -> String {
        serde_json::json!({
            "proposal": amount,
            "symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
            "contract_type": if direction == "CALL" { "CALL" } else { "PUT" }
        })
        .to_string()
    }

    /// Build buy request.
    pub fn buy_request(&self, proposal_id: &str, amount: f64) -> String {
        serde_json::json!({
            "buy": proposal_id,
            "price": amount
        })
        .to_string()
    }

    /// Build balance request.
    pub fn balance_request(&self) -> String {
        serde_json::json!({
            "balance": 1,
            "subscribe": 1
        })
        .to_string()
    }

    /// Build ping request.
    pub fn ping_request(&self) -> String {
        serde_json::json!({
            "ping": 1
        })
        .to_string()
    }

    /// Parse incoming WebSocket message.
    pub fn parse_message(&self, text: &str) -> Result<DerivMessage, String> {
        serde_json::from_str(text)
            .map_err(|e| format!("Failed to parse Deriv message: {}", e))
    }

    /// Handle authorization response.
    pub fn handle_authorize(&mut self, resp: &AuthorizeResponse) -> Result<(), String> {
        if let Some(ref auth) = resp.authorize {
            log::info!("Authorized as {} (balance: {} {})", 
                auth.loginid, auth.balance, auth.currency);
            self.connection_state = ConnectionState::Authorized;
            self.reconnect_attempts = 0;
            Ok(())
        } else if let Some(ref err) = resp.error {
            self.last_error = Some(err.message.clone());
            self.connection_state = ConnectionState::Error(err.message.clone());
            Err(err.message.clone())
        } else {
            Err("Unknown authorization response".into())
        }
    }

    /// Handle tick data.
    pub fn handle_tick(&self, resp: &TicksResponse) -> Option<crate::simfeed::Bar> {
        if let Some(ref tick) = resp.tick {
            let canonical = deriv_symbol_map(&tick.symbol);
            Some(tick_to_bar(tick, &canonical))
        } else {
            None
        }
    }

    /// Handle OHLC data.
    pub fn handle_ohlc(&self, resp: &OhlcResponse) -> Option<crate::simfeed::Bar> {
        if let Some(ref ohlc) = resp.ohlc {
            let canonical = deriv_symbol_map(&ohlc.symbol);
            Some(ohlc_to_bar(ohlc, &canonical))
        } else {
            None
        }
    }

    /// Reset connection state for reconnection.
    pub fn reset_connection(&mut self) {
        self.connection_state = ConnectionState::Disconnected;
        self.subscriptions.clear();
    }

    /// Increment reconnect attempts and check limit.
    pub fn should_reconnect(&mut self) -> bool {
        self.reconnect_attempts += 1;
        self.reconnect_attempts <= self.max_reconnect_attempts
    }
}

/// Deriv-specific order types for execution.
#[derive(Debug, Clone)]
pub enum DerivOrderType {
    /// Rise/Fall contracts (vanilla options)
    RiseFall {
        amount: f64,
        duration: u32,
        duration_unit: String, // "t"=ticks, "m"=minutes, "h"=hours
    },
    /// Higher/Lower contracts
    HigherLower {
        amount: f64,
        duration: u32,
        duration_unit: String,
        barrier: f64,
    },
    /// In/Out barriers
    TouchNoTouch {
        amount: f64,
        duration: u32,
        duration_unit: String,
        barrier: f64,
    },
}

/// Deriv execution result.
#[derive(Debug, Clone)]
pub struct DerivExecutionResult {
    pub transaction_id: u64,
    pub buy_price: f64,
    pub payout: f64,
    pub currency: String,
    pub underlying: String,
    pub status: String, // "won", "lost", "open"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_mapping() {
        assert_eq!(deriv_symbol_map("R_50"), "SYNTHETIC_50");
        assert_eq!(deriv_symbol_map("frxEURUSD"), "EURUSD");
        assert_eq!(deriv_symbol_map("XAUUSD"), "XAUUSD");
        assert_eq!(deriv_symbol_map("UNKNOWN"), "UNKNOWN");
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10);
        assert_eq!(limiter.min_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_client_creation() {
        let config = DerivFeedConfig::default();
        let client = DerivClient::new(config);
        assert_eq!(client.state(), &ConnectionState::Disconnected);
        assert!(client.ws_url().contains("app_id=1089"));
    }

    #[test]
    fn test_message_building() {
        let config = DerivFeedConfig {
            app_id: 12345,
            api_token: Some("test_token".into()),
            ..Default::default()
        };
        let client = DerivClient::new(config);
        
        let auth = client.authorize_request();
        assert!(auth.contains("test_token"));
        
        let tick_sub = client.ticks_subscribe("R_50");
        assert!(tick_sub.contains("R_50"));
        assert!(tick_sub.contains("subscribe"));
    }
}
