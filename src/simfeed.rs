//! SIMULATED FEED — synthetic OHLCV bar generator.
//!
//! The Quant is designed for a live MT5 bridge over ZeroMQ. For development,
//! testing, and lean footprints this module emits a realistic geometric random
//! walk, which keeps the whole engine testable end to end without a broker.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// An OHLCV bar (immutable, cheap to copy; f64 for speed).
#[derive(Debug, Clone)]
pub struct Bar {
    pub symbol: String,
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A configurable simulator that can be seeded for reproducible tests.
pub struct SimFeed {
    rng: StdRng,
    symbol: String,
    /// base price for the symbol (e.g. EURUSD ~1.08)
    price: f64,
    /// per-bar drift
    drift: f64,
    /// per-bar volatility (fraction of price)
    vol: f64,
    bar: i64,
}

impl SimFeed {
    pub fn new(symbol: &str, base_price: f64, seed: u64) -> Self {
        SimFeed {
            rng: StdRng::seed_from_u64(seed),
            symbol: symbol.to_string(),
            price: base_price,
            drift: 0.000_005,
            // per-bar volatility as a FRACTION of price (the walk applies it
            // relatively: close = price * (1 + drift + shock * vol))
            vol: 0.0009,
            bar: 0,
        }
    }

    /// Set per-bar volatility (as fraction of price). Default ~0.0009.
    pub fn with_vol(mut self, frac: f64) -> Self {
        self.vol = frac;
        self
    }

    /// Generate the next bar (random walk with occasional regime-vol shifts).
    pub fn next_bar(&mut self) -> Bar {
        // Occasionally simulate a volatility burst / trend regime.
        let burst = self.rng.gen_bool(0.02);
        let vol = if burst { self.vol * 3.0 } else { self.vol };
        let regime_drift = if self.rng.gen_bool(0.35) {
            self.drift * 4.0
        } else {
            self.drift
        };

        let shock: f64 = self.rng.gen_range(-1.0..1.0);
        let close = self.price * (1.0 + regime_drift + shock * vol);
        let open = self.price;
        let high = open.max(close) * (1.0 + self.rng.gen_range(0.0..0.001));
        let low = open.min(close) * (1.0 - self.rng.gen_range(0.0..0.001));
        let volume = self.rng.gen_range(100.0..2000.0) * (if burst { 2.0 } else { 1.0 });

        let bar = Bar {
            symbol: self.symbol.clone(),
            time: self.bar,
            open,
            high,
            low,
            close,
            volume,
        };
        self.price = close;
        self.bar += 1;
        bar
    }

    /// Symbol this feed produces.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Multi-symbol feed: steps one `SimFeed` per symbol in lockstep, returning all
/// bars for one time tick. Used by the Aegis pairs strategy which needs
/// synchronous bars on both legs of every pair.
pub struct MultiFeed {
    feeds: Vec<SimFeed>,
}

impl MultiFeed {
    /// Build a feed. `symbols` is `[(symbol, base_price), ...]`. Each leg gets
    /// a unique seed derived from the base seed so pairs co-move but differ.
    pub fn new(symbols: &[(&str, f64)], seed: u64) -> Self {
        let feeds = symbols
            .iter()
            .enumerate()
            .map(|(i, (sym, base))| SimFeed::new(sym, *base, seed.wrapping_add(i as u64 * 7919)))
            .collect();
        MultiFeed { feeds }
    }

    /// Advance every symbol one bar. Returns the collected bars for this tick.
    pub fn next_bars(&mut self) -> Vec<Bar> {
        self.feeds.iter_mut().map(|f| f.next_bar()).collect()
    }

    /// Symbols this feed covers.
    pub fn symbols(&self) -> Vec<&str> {
        self.feeds.iter().map(|f| f.symbol.as_str()).collect()
    }
}

/// Correlated multi-symbol feed.
///
/// Generates symbols that share a common market factor + idiosyncratic noise,
/// so pairs like EURUSD/GBPUSD genuinely co-move (corr > 0.70). This gives the
/// Aegis pairs strategy real alpha to work with in simulation.
///
/// `betas` is the factor loading for each symbol (higher = more correlated).
/// `idiovol` is the idiosyncratic noise fraction per bar.
pub struct CorrelatedFeed {
    rng: StdRng,
    symbols: Vec<String>,
    prices: Vec<f64>,
    betas: Vec<f64>,
    idiovol: f64,
    drift: f64,
    bar: i64,
}

impl CorrelatedFeed {
    pub fn new(symbols: &[(&str, f64)], betas: &[f64], seed: u64, idiovol: f64) -> Self {
        CorrelatedFeed {
            rng: StdRng::seed_from_u64(seed),
            symbols: symbols.iter().map(|(s, _)| s.to_string()).collect(),
            prices: symbols.iter().map(|(_, p)| *p).collect(),
            betas: betas.to_vec(),
            idiovol,
            drift: 0.000_005,
            bar: 0,
        }
    }

    /// Advance every symbol one bar using a shared market factor.
    pub fn next_bars(&mut self) -> Vec<Bar> {
        // Common market factor (shared by all symbols).
        let market_shock: f64 = self.rng.gen_range(-1.0..1.0);
        let market_return = self.drift + market_shock * 0.0009;

        let mut bars = Vec::new();
        for i in 0..self.symbols.len() {
            let idio_shock: f64 = self.rng.gen_range(-1.0..1.0);
            let idio_return = idio_shock * self.idiovol;
            let total_return = self.betas[i] * market_return + idio_return;

            let close = self.prices[i] * (1.0 + total_return);
            let open = self.prices[i];
            let high = open.max(close) * (1.0 + self.rng.gen_range(0.0..0.001));
            let low = open.min(close) * (1.0 - self.rng.gen_range(0.0..0.001));
            let volume = self.rng.gen_range(100.0..2000.0);

            bars.push(Bar {
                symbol: self.symbols[i].clone(),
                time: self.bar,
                open,
                high,
                low,
                close,
                volume,
            });
            self.prices[i] = close;
        }
        self.bar += 1;
        bars
    }

    pub fn symbols(&self) -> Vec<&str> {
        self.symbols.iter().map(|s| s.as_str()).collect()
    }
}

impl Feed for CorrelatedFeed {
    fn next_bars(&mut self) -> Option<Vec<Bar>> {
        Some(CorrelatedFeed::next_bars(self))
    }
    fn symbols(&self) -> Vec<&str> {
        CorrelatedFeed::symbols(self)
    }
    fn is_exhausted(&self) -> bool {
        false
    }
}

/// Unified feed interface — the engine is feed-agnostic.
/// Any feed type (sim, csv, mt5) implements this so the engine can swap
/// data sources based on `config/system.toml` → `[system.data_source]`.
pub trait Feed {
    /// Return the next tick (one Bar per covered symbol), or None when exhausted.
    fn next_bars(&mut self) -> Option<Vec<Bar>>;
    /// Symbols this feed covers.
    fn symbols(&self) -> Vec<&str>;
    /// Whether the feed has finished replaying its data.
    fn is_exhausted(&self) -> bool {
        false
    }
}

/// MT5 shared-files live feed.
///
/// In production the MT5 terminal runs a small Expert Advisor
/// (`deploy/mt5_ea.mq5`) that writes completed M5 bars to CSV files in the MT5
/// `Files` directory (shared between the MT5 process and the Rust binary, even
/// under Wine). Each symbol gets its own `<SYMBOL>.csv` file, appended in real
/// time. `Mt5Feed` polls that directory, tracks the read offset per file, and
/// returns new bars only when **all** symbols for a given timestamp have
/// arrived — guaranteeing lockstep bars for the Aegis pairs strategy.
///
/// Directory layout expected:
/// ```text
/// <mt5_dir>/EURUSD.csv   (appended live by the EA)
/// <mt5_dir>/GBPUSD.csv
/// <mt5_dir>/XAUUSD.csv
/// ...
/// ```
///
/// CSV columns (written by the EA): time,symbol,open,high,low,close,volume
pub struct Mt5Feed {
    /// symbols to track (kept in stable order)
    symbols: Vec<String>,
    /// directory polled for per-symbol CSV files
    dir: String,
    /// per-symbol read offset (handles file rotation / MT5 restart)
    offsets: std::collections::HashMap<String, u64>,
    /// pending bars keyed by timestamp, waiting for all symbols to arrive
    pending: std::collections::HashMap<i64, std::collections::HashMap<String, Bar>>,
    /// how many consecutive polls found no data (→ MT5 likely offline)
    missing_count: u32,
    /// terminal has gone quiet (too many missing polls)
    offline: bool,
}

impl Mt5Feed {
    /// Create a new feed polling `dir` for `symbols` CSV files.
    pub fn new(dir: &str, symbols: &[&str]) -> Result<Self, String> {
        Ok(Mt5Feed {
            symbols: symbols.iter().map(|s| s.to_string()).collect(),
            dir: dir.to_string(),
            offsets: std::collections::HashMap::new(),
            pending: std::collections::HashMap::new(),
            missing_count: 0,
            offline: false,
        })
    }

    /// Poll the MT5 files directory for new bars. Returns a lockstep batch when
    /// all tracked symbols have a bar for some timestamp, else None.
    pub fn poll(&mut self) -> Option<Vec<Bar>> {
        if self.offline {
            return None;
        }
        let mut new_bars: Vec<(String, Bar)> = Vec::new();

        for sym in &self.symbols {
            let path = format!("{}/{}.csv", self.dir, sym);
            let content = match std::fs::read(&path) {
                Ok(c) => c,
                Err(_) => {
                    self.missing_count += 1;
                    if self.missing_count > 10 {
                        self.offline = true;
                    }
                    continue;
                }
            };
            self.missing_count = 0;

            let offset = *self.offsets.get(sym).unwrap_or(&0) as usize;
            let new_data: &[u8] = if content.len() > offset {
                &content[offset..]
            } else {
                // file was rotated/truncated (MT5 restart) — re-read from start
                &content[..]
            };
            self.offsets.insert(sym.clone(), content.len() as u64);

            let text = String::from_utf8_lossy(new_data);
            for line in text.lines() {
                if line.starts_with("time,") || line.trim().is_empty() {
                    continue;
                }
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() < 6 {
                    continue;
                }
                let time: i64 = match parts[0].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let open: f64 = parts[2].parse().unwrap_or(0.0);
                let high: f64 = parts[3].parse().unwrap_or(0.0);
                let low: f64 = parts[4].parse().unwrap_or(0.0);
                let close: f64 = parts[5].parse().unwrap_or(0.0);
                let vol: f64 = parts.get(6).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                new_bars.push((
                    sym.clone(),
                    Bar {
                        symbol: sym.clone(),
                        time,
                        open,
                        high,
                        low,
                        close,
                        volume: vol,
                    },
                ));
            }
        }

        // Index by timestamp so we can wait for lockstep
        for (sym, bar) in &new_bars {
            let row = self.pending.entry(bar.time).or_default();
            row.insert(sym.clone(), bar.clone());
        }

        // Return timestamps where ALL symbols have a bar
        let complete: Vec<i64> = self
            .pending
            .iter()
            .filter(|(_, m)| self.symbols.iter().all(|s| m.contains_key(s)))
            .map(|(t, _)| *t)
            .collect();

        if complete.is_empty() {
            return None;
        }

        // Return the oldest complete tick
        let t = complete.into_iter().min()?;
        let row = self.pending.remove(&t)?;
        let bars: Vec<Bar> = self
            .symbols
            .iter()
            .filter_map(|s| row.get(s).cloned())
            .collect();
        Some(bars)
    }
}

impl Feed for Mt5Feed {
    fn next_bars(&mut self) -> Option<Vec<Bar>> {
        // Retry with a short backoff so we always return lockstep bars
        for _ in 0..100 {
            if let Some(bars) = self.poll() {
                return Some(bars);
            }
            if self.offline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        None
    }
    fn symbols(&self) -> Vec<&str> {
        self.symbols.iter().map(|s| s.as_str()).collect()
    }
    fn is_exhausted(&self) -> bool {
        self.offline
    }
}

/// CSV-based feed that loads real or generated historical data from files.
///
/// Reads bars from CSV files (one per symbol) and replays them in lockstep.
/// This lets the Aegis strategy be tested against realistic cointegrated data
/// without a live broker connection.
///
/// CSV format: date,time,open,high,low,close,volume (header row required).
pub struct CsvFeed {
    /// symbol -> remaining bars to replay
    series: std::collections::HashMap<String, Vec<Bar>>,
    symbols: Vec<String>,
    time: i64,
    exhausted: bool,
}

impl CsvFeed {
    /// Load CSV feeds from a directory. Each file `<symbol>.csv` becomes one symbol.
    pub fn load(dir: &str, symbols: &[&str]) -> Result<Self, String> {
        let mut series = std::collections::HashMap::new();
        let mut loaded = Vec::new();

        for &sym in symbols {
            let path = format!("{}/{}.csv", dir, sym.to_lowercase());
            let bars = Self::read_csv(&path, sym)?;
            if bars.is_empty() {
                return Err(format!("no bars in {}", path));
            }
            loaded.push(sym.to_string());
            series.insert(sym.to_string(), bars);
        }

        Ok(CsvFeed {
            series,
            symbols: loaded,
            time: 0,
            exhausted: false,
        })
    }

    /// Load from the default test-data directory (python/data/histdata).
    pub fn load_default() -> Result<Self, String> {
        Self::load(
            "python/data/histdata",
            &["eurusd", "gbpusd", "audusd", "nzdusd", "xauusd", "xagusd"],
        )
    }

    fn read_csv(path: &str, symbol: &str) -> Result<Vec<Bar>, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        let mut bars = Vec::new();
        for (lineno, line) in data.lines().enumerate() {
            if lineno == 0 {
                continue; // skip header
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 6 {
                continue;
            }
            let open: f64 = parts[2]
                .parse()
                .map_err(|e| format!("parse open {}:{}", path, e))?;
            let high: f64 = parts[3]
                .parse()
                .map_err(|e| format!("parse high {}:{}", path, e))?;
            let low: f64 = parts[4]
                .parse()
                .map_err(|e| format!("parse low {}:{}", path, e))?;
            let close: f64 = parts[5]
                .parse()
                .map_err(|e| format!("parse close {}:{}", path, e))?;
            let volume: f64 = parts.get(6).and_then(|v| v.parse().ok()).unwrap_or(1000.0);
            let time = lineno as i64 - 1;
            bars.push(Bar {
                symbol: symbol.to_uppercase(),
                time,
                open,
                high,
                low,
                close,
                volume,
            });
        }
        Ok(bars)
    }

    /// Advance every symbol one bar. Returns None when all series are exhausted.
    pub fn next_bars(&mut self) -> Option<Vec<Bar>> {
        if self.exhausted {
            return None;
        }
        let mut bars = Vec::new();
        for sym in &self.symbols {
            if let Some(series) = self.series.get_mut(sym) {
                if series.is_empty() {
                    self.exhausted = true;
                    return None;
                }
                let mut bar = series.remove(0);
                bar.time = self.time;
                bars.push(bar);
            }
        }
        self.time += 1;
        Some(bars)
    }

    pub fn symbols(&self) -> Vec<&str> {
        self.symbols.iter().map(|s| s.as_str()).collect()
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

impl Feed for CsvFeed {
    fn next_bars(&mut self) -> Option<Vec<Bar>> {
        CsvFeed::next_bars(self)
    }
    fn symbols(&self) -> Vec<&str> {
        CsvFeed::symbols(self)
    }
    fn is_exhausted(&self) -> bool {
        CsvFeed::is_exhausted(self)
    }
}

impl Feed for MultiFeed {
    fn next_bars(&mut self) -> Option<Vec<Bar>> {
        Some(MultiFeed::next_bars(self))
    }
    fn symbols(&self) -> Vec<&str> {
        MultiFeed::symbols(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_is_deterministic_with_seed() {
        let mut a = SimFeed::new("EURUSD", 1.08, 42);
        let mut b = SimFeed::new("EURUSD", 1.08, 42);
        for _ in 0..100 {
            let (ba, bb) = (a.next_bar(), b.next_bar());
            assert_eq!(ba.close, bb.close);
            assert!((ba.high - bb.high).abs() < 1e-12);
        }
        // prices should have moved; not frozen (different seeds)
        let c0 = SimFeed::new("EURUSD", 1.08, 1).next_bar().close;
        let c1 = SimFeed::new("EURUSD", 1.08, 2).next_bar().close;
        assert_ne!(c0, c1);
    }
}
