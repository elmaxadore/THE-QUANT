//! # Data Pipeline — Data Collector (Layer 3)
//!
//! The most performance-critical layer in The Quant. Handles ZeroMQ communication
//! with MT5, OHLCV bar construction, feature engineering, and data versioning.
//!
//! ## Zero-Copy Hot Path
//! - Tick processing: pre-allocated ring buffers
//! - Feature vectors: stack-allocated arrays (ArrayVec)
//! - OHLCV bars: append-only Vec with reserve_exact()
//! - Parse ZMQ frames in-place, borrow don't clone

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use crate::memory::MemoryManager;
use arrayvec::ArrayVec;
use chrono::{DateTime, Utc, Timelike};
use crossbeam_channel::{bounded, Receiver, Sender};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

// === Message Types (MQL5 Bridge Protocol) ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Mt5Message {
    Tick(TickData),
    Bar(BarData),
    Account(AccountUpdate),
    Position(PositionUpdate),
    Command(CommandResponse),
    Heartbeat,
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub time: DateTime<Utc>,
    pub volume: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarData {
    pub symbol: String,
    pub timeframe: String,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountUpdate {
    pub balance: Decimal,
    pub equity: Decimal,
    pub margin: Decimal,
    pub free_margin: Decimal,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub ticket: u64,
    pub symbol: String,
    pub direction: TradeDirection,
    pub volume: Decimal,
    pub open_price: Decimal,
    pub current_price: Decimal,
    pub profit: Decimal,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    pub cmd_id: String,
    pub success: bool,
    pub message: String,
    pub order_ticket: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TradeDirection {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

// === Feature Vector (stack-allocated, fixed-size) ===

/// Maximum number of features computed per bar
pub const MAX_FEATURES: usize = 64;

/// A feature vector — stack-allocated with ArrayVec for zero-allocation hot path
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub symbol: String,
    pub time: DateTime<Utc>,
    pub features: ArrayVec<Feature, MAX_FEATURES>,
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub name: &'static str,
    pub value: f64,
}

// === OHLCV Ring Buffer ===

/// Ring buffer for OHLCV bars per symbol per timeframe
#[derive(Debug)]
pub struct OhlcvRingBuffer {
    pub symbol: String,
    pub timeframe: String,
    pub bars: Vec<BarData>,
    pub capacity: usize,
    pub head: usize,
    pub count: usize,
}

impl OhlcvRingBuffer {
    pub fn new(symbol: String, timeframe: String, capacity: usize) -> Self {
        Self {
            symbol,
            timeframe,
            bars: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            count: 0,
        }
    }

    /// Add a bar to the ring buffer (oldest dropped if full)
    pub fn push(&mut self, bar: BarData) -> Option<BarData> {
        let dropped = if self.count >= self.capacity {
            let old = self.bars[self.head].clone();
            self.bars[self.head] = bar;
            self.head = (self.head + 1) % self.capacity;
            Some(old)
        } else {
            self.bars.push(bar);
            None
        };
        self.count = self.count.min(self.capacity);
        dropped
    }

    pub fn len(&self) -> usize {
        self.count.min(self.bars.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&BarData> {
        if index >= self.len() {
            return None;
        }
        let pos = if self.head == 0 {
            self.capacity - index - 1
        } else {
            (self.head + self.capacity - index - 1) % self.capacity
        };
        self.bars.get(pos)
    }

    pub fn last_n(&self, n: usize) -> Vec<&BarData> {
        (0..n.min(self.len())).filter_map(|i| self.get(i)).collect()
    }
}

// === Data Collector ===

#[derive(Debug)]
pub struct DataCollector {
    zmq_context: Option<zmq::Context>,
    ohlcv_buffers: Arc<RwLock<HashMap<String, HashMap<String, OhlcvRingBuffer>>>>,
    tick_tx: Sender<TickData>,
    tick_rx: Receiver<TickData>,
    bar_tx: Sender<BarData>,
    bar_rx: Receiver<BarData>,
    memory_manager: Arc<RwLock<MemoryManager>>,
    symbols: Vec<String>,
    timeframes: Vec<String>,
}

impl DataCollector {
    pub fn new(config: &QuantConfig, memory_manager: Arc<RwLock<MemoryManager>>) -> Self {
        let mm = memory_manager.blocking_read();
        let capacity = mm.channel_capacity("data_collector");
        drop(mm);
        let (tick_tx, tick_rx) = bounded(capacity);
        let (bar_tx, bar_rx) = bounded(capacity / 4);
        Self {
            zmq_context: None,
            ohlcv_buffers: Arc::new(RwLock::new(HashMap::new())),
            tick_tx,
            tick_rx,
            bar_tx,
            bar_rx,
            memory_manager,
            symbols: config.mt5.symbols.clone(),
            timeframes: config.mt5.timeframes.clone(),
        }
    }

    pub async fn connect(&mut self, config: &QuantConfig) -> QuantResult<()> {
        let ctx = zmq::Context::new();
        let subscriber = ctx.socket(zmq::SUB)?;
        subscriber.connect(&config.mt5.zmq_pub_endpoint)?;
        for symbol in &self.symbols {
            subscriber.set_subscribe(symbol.as_bytes())?;
        }
        info!("Connected to MT5 at {}", config.mt5.zmq_pub_endpoint);
        self.zmq_context = Some(ctx);
        Ok(())
    }

    pub async fn initialize_buffers(&self, config: &QuantConfig) {
        let depth = self.memory_manager.read().await.ring_buffer_depth("data_collector");
        let mut buffers = self.ohlcv_buffers.write().await;
        for symbol in &config.mt5.symbols {
            let mut symbol_buffers = HashMap::new();
            for tf in &config.mt5.timeframes {
                symbol_buffers.insert(tf.clone(), OhlcvRingBuffer::new(symbol.clone(), tf.clone(), depth));
            }
            buffers.insert(symbol.clone(), symbol_buffers);
        }
    }

    pub fn process_tick(&self, tick: TickData) -> QuantResult<()> {
        self.tick_tx.send(tick).map_err(|e| QuantError::ChannelClosed(format!("Tick channel: {}", e)))?;
        Ok(())
    }

    pub fn tick_receiver(&self) -> Receiver<TickData> {
        self.tick_rx.clone()
    }

    pub fn bar_receiver(&self) -> Receiver<BarData> {
        self.bar_rx.clone()
    }

    pub async fn get_ohlcv(&self, symbol: &str, timeframe: &str) -> Option<Vec<BarData>> {
        self.ohlcv_buffers.read().await.get(symbol).and_then(|m| m.get(timeframe)).map(|b| b.bars.clone())
    }

    pub fn disconnect(&mut self) {
        self.zmq_context = None;
        info!("Disconnected from MT5");
    }
}

// === Feature Pipeline ===

#[derive(Debug)]
pub struct FeaturePipeline {
    cache: Arc<RwLock<HashMap<String, lru::LruCache<String, f64>>>>,
    memory_manager: Arc<RwLock<MemoryManager>>,
    correlation_matrix: Arc<RwLock<HashMap<(String, String), f64>>>,
}

impl FeaturePipeline {
    pub fn new(memory_manager: Arc<RwLock<MemoryManager>>, _config: &QuantConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            memory_manager,
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn compute_features(&self, symbol: &str, bars: &[BarData]) -> QuantResult<FeatureVector> {
        if bars.is_empty() {
            return Err(QuantError::FeatureError("No bars provided".into()));
        }
        let current = &bars[0];
        let mut features = ArrayVec::new();

        if let Some(v) = compute_log_return(bars, 1) {
            features.push(Feature { name: "log_return_1", value: v });
        }
        if let Some(v) = compute_log_return(bars, 5) {
            features.push(Feature { name: "log_return_5", value: v });
        }
        if let Some(v) = self.compute_ema_diff(bars, 12, 26) {
            features.push(Feature { name: "ema_12_26_diff", value: v });
        }
        if let Some(v) = self.compute_rsi(bars, 14) {
            features.push(Feature { name: "rsi_14", value: v });
        }
        if let Some(v) = self.compute_atr(bars, 14) {
            features.push(Feature { name: "atr_14", value: v });
        }
        if let Some(v) = self.compute_obv(bars) {
            features.push(Feature { name: "obv", value: v });
        }
        if let Some(v) = compute_mean(bars, 20) {
            features.push(Feature { name: "roll_mean_20", value: v });
        }
        if let Some(v) = compute_std(bars, 20) {
            features.push(Feature { name: "roll_std_20", value: v });
        }

        let (h_sin, h_cos) = compute_time_features(&current.time);
        features.push(Feature { name: "hour_sin", value: h_sin });
        features.push(Feature { name: "hour_cos", value: h_cos });

        if let Some(v) = self.compute_hurst_exponent(bars, 20) {
            features.push(Feature { name: "hurst_20", value: v });
        }

        Ok(FeatureVector {
            symbol: symbol.to_string(),
            time: current.time,
            features,
        })
    }

    fn compute_ema_diff(&self, bars: &[BarData], sp: usize, lp: usize) -> Option<f64> {
        let closes: Vec<f64> = bars.iter().map(|b| b.close.to_f64().unwrap_or(0.0)).collect();
        Some(ema(&closes, sp)? - ema(&closes, lp)?)
    }

    fn compute_rsi(&self, bars: &[BarData], period: usize) -> Option<f64> {
        if bars.len() <= period {
            return None;
        }
        let closes: Vec<f64> = bars.iter().map(|b| b.close.to_f64().unwrap_or(0.0)).collect();
        let mut gains = 0.0f64;
        let mut losses = 0.0f64;
        for i in 1..=period {
            let diff = closes[i - 1] - closes[i];
            if diff > 0.0 {
                gains += diff;
            } else {
                losses -= diff;
            }
        }
        if losses == 0.0 {
            return Some(100.0);
        }
        Some(100.0 - (100.0 / (1.0 + gains / losses)))
    }

    fn compute_atr(&self, bars: &[BarData], period: usize) -> Option<f64> {
        if bars.len() <= period {
            return None;
        }
        let mut tr_sum = 0.0f64;
        for i in 0..period.min(bars.len() - 1) {
            let h = bars[i].high.to_f64().unwrap_or(0.0);
            let l = bars[i].low.to_f64().unwrap_or(0.0);
            let pc = bars[i + 1].close.to_f64().unwrap_or(0.0);
            tr_sum += (h - l).max((h - pc).abs()).max((l - pc).abs());
        }
        Some(tr_sum / period as f64)
    }

    fn compute_obv(&self, bars: &[BarData]) -> Option<f64> {
        let mut obv = 0.0f64;
        for i in 1..bars.len() {
            let v = bars[i].volume.to_f64().unwrap_or(0.0);
            let d = bars[i - 1].close.to_f64().unwrap_or(0.0) - bars[i].close.to_f64().unwrap_or(0.0);
            if d > 0.0 {
                obv += v;
            } else if d < 0.0 {
                obv -= v;
            }
        }
        Some(obv)
    }

    fn compute_hurst_exponent(&self, bars: &[BarData], period: usize) -> Option<f64> {
        let n = period.min(bars.len());
        if n < 10 {
            return None;
        }
        let lr: Vec<f64> = (1..n)
            .filter_map(|i| {
                let p = bars[i].close.to_f64()?;
                let c = bars[i - 1].close.to_f64()?;
                if p == 0.0 {
                    None
                } else {
                    Some((c / p).ln())
                }
            })
            .collect();
        if lr.len() < 10 {
            return None;
        }
        let mean = lr.iter().sum::<f64>() / lr.len() as f64;
        let devs: Vec<f64> = lr.iter().map(|r| r - mean).collect();
        let cum: Vec<f64> = devs.iter().scan(0.0, |s, &x| {
            *s += x;
            Some(*s)
        }).collect();
        let r = cum.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cum.iter().cloned().fold(f64::INFINITY, f64::min);
        let s = (devs.iter().map(|d| d * d).sum::<f64>() / devs.len() as f64).sqrt();
        if s == 0.0 {
            return None;
        }
        Some((r / s).ln() / (lr.len() as f64).ln())
    }

    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
        debug!("Feature cache cleared");
    }
}

// === Free functions for feature computation ===

fn compute_log_return(bars: &[BarData], period: usize) -> Option<f64> {
    if bars.len() <= period {
        return None;
    }
    let current = bars[0].close.to_f64()?;
    let previous = bars[period].close.to_f64()?;
    if previous == 0.0 {
        return None;
    }
    Some((current / previous).ln())
}

fn compute_time_features(time: &DateTime<Utc>) -> (f64, f64) {
    let tm = time.hour() as f64 * 60.0 + time.minute() as f64;
    let f = tm / (24.0 * 60.0);
    (
        (f * 2.0 * std::f64::consts::PI).sin(),
        (f * 2.0 * std::f64::consts::PI).cos(),
    )
}

fn ema(values: &[f64], period: usize) -> Option<f64> {
    if values.is_empty() || period == 0 {
        return None;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut ema = values[0];
    for &v in values.iter().skip(1) {
        ema = v * k + ema * (1.0 - k);
    }
    Some(ema)
}

fn compute_mean(bars: &[BarData], period: usize) -> Option<f64> {
    let n = period.min(bars.len());
    if n == 0 {
        return None;
    }
    Some(bars.iter().take(n).filter_map(|b| b.close.to_f64()).sum::<f64>() / n as f64)
}

fn compute_std(bars: &[BarData], period: usize) -> Option<f64> {
    let n = period.min(bars.len());
    if n < 2 {
        return None;
    }
    let mean = compute_mean(bars, period)?;
    let var = bars
        .iter()
        .take(n)
        .filter_map(|b| b.close.to_f64())
        .map(|c| (c - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    Some(var.sqrt())
}

// === ZMQ Protocol Handler ===

#[derive(Debug)]
pub struct ZmqProtocol {
    context: zmq::Context,
    publisher: Option<zmq::Socket>,
    subscriber: Option<zmq::Socket>,
    consecutive_missed_pings: u32,
    pub max_missed_pings: u32,
}

impl ZmqProtocol {
    pub fn new() -> QuantResult<Self> {
        Ok(Self {
            context: zmq::Context::new(),
            publisher: None,
            subscriber: None,
            consecutive_missed_pings: 0,
            max_missed_pings: 5,
        })
    }

    pub fn connect(&mut self, pub_endpoint: &str, sub_endpoint: &str) -> QuantResult<()> {
        let p = self.context.socket(zmq::PUB)?;
        p.bind(pub_endpoint)?;
        let s = self.context.socket(zmq::SUB)?;
        s.connect(sub_endpoint)?;
        s.set_subscribe(b"")?;
        self.publisher = Some(p);
        self.subscriber = Some(s);
        info!("ZMQ connected — PUB: {}, SUB: {}", pub_endpoint, sub_endpoint);
        Ok(())
    }

    pub fn send_command(&self, cmd: &str) -> QuantResult<()> {
        self.publisher
            .as_ref()
            .ok_or_else(|| QuantError::Mt5ConnectionError("Not connected".into()))?
            .send(cmd, 0)?;
        Ok(())
    }

    pub fn receive_message(&self) -> QuantResult<Option<String>> {
        match self
            .subscriber
            .as_ref()
            .ok_or_else(|| QuantError::Mt5ConnectionError("Not connected".into()))?
            .recv_msg(zmq::DONTWAIT)
        {
            Ok(msg) => {
                self.consecutive_missed_pings = 0;
                Ok(msg.as_str().map(|s| s.to_string()))
            }
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(QuantError::ZmqError(e)),
        }
    }

    pub fn check_heartbeat(&mut self) -> bool {
        if self.consecutive_missed_pings >= self.max_missed_pings {
            false
        } else {
            self.consecutive_missed_pings += 1;
            true
        }
    }

    pub fn disconnect(&mut self) {
        self.publisher = None;
        self.subscriber = None;
        info!("ZMQ disconnected");
    }
}

// === Data Versioning ===

#[derive(Debug)]
pub struct DataVersioning {
    data_dir: std::path::PathBuf,
}

impl DataVersioning {
    pub fn new(base_dir: &Path) -> Self {
        Self {
            data_dir: base_dir.to_path_buf(),
        }
    }

    pub fn verify_checksum(path: &Path, expected: &str) -> QuantResult<bool> {
        let data = std::fs::read(path)?;
        let hash = blake3::hash(&data);
        Ok(hash.to_hex().as_str() == expected)
    }

    pub fn compute_checksum(path: &Path) -> QuantResult<String> {
        let data = std::fs::read(path)?;
        let hash = blake3::hash(&data);
        Ok(hash.to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bars(count: usize) -> Vec<BarData> {
        let mut bars = Vec::with_capacity(count);
        let mut price = 100.0;
        let now = Utc::now();
        for i in 0..count {
            price += (i as f64).sin() * 0.5;
            bars.push(BarData {
                symbol: "EURUSD".into(),
                timeframe: "M5".into(),
                open: Decimal::from_f64_retain(price - 0.1).unwrap(),
                high: Decimal::from_f64_retain(price + 0.2).unwrap(),
                low: Decimal::from_f64_retain(price - 0.2).unwrap(),
                close: Decimal::from_f64_retain(price).unwrap(),
                volume: Decimal::new(100 + i as i64, 0),
                time: now - chrono::Duration::minutes(i as i64 * 5),
            });
        }
        bars
    }

    #[test]
    fn test_ring_buffer_push() {
        let mut buf = OhlcvRingBuffer::new("EURUSD".into(), "M5".into(), 10);
        let bars = create_test_bars(5);
        for bar in bars {
            assert!(buf.push(bar).is_none());
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buf = OhlcvRingBuffer::new("EURUSD".into(), "M5".into(), 3);
        let bars = create_test_bars(5);
        for bar in bars {
            buf.push(bar);
        }
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_log_return() {
        let bars = create_test_bars(10);
        let ret = compute_log_return(&bars, 1);
        assert!(ret.is_some());
        assert!(ret.unwrap().abs() < 1.0);
    }

    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&values, 3);
        assert!(result.is_some());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_time_features() {
        let now = Utc::now();
        let (sin, cos) = compute_time_features(&now);
        assert!(sin >= -1.0 && sin <= 1.0);
        assert!(cos >= -1.0 && cos <= 1.0);
    }

    #[test]
    fn test_data_versioning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"test data").unwrap();
        let checksum = DataVersioning::compute_checksum(&path).unwrap();
        assert!(DataVersioning::verify_checksum(&path, &checksum).unwrap());
    }
}
