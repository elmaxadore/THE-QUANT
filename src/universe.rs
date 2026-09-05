//! UNIVERSE — dynamic asset universe, cross-broker symbol identity and the
//! shared knowledge base (v4.1).
//!
//! Design goals (product spec):
//!   * Every account is an ISOLATED trader (own risk engine, own strategy
//!     config, own journal) but KNOWLEDGE is shared: anything any desk learns
//!     about an asset feeds the same knowledge base.
//!   * Assets are scanned whether or not they are on a watchlist. Each desk
//!     scans the full asset list of its account (`universe_mode`).
//!   * The same asset quoted by different brokers under different symbol
//!     names (GOLD / XAUUSD.x / XAUUSDm ...) maps to ONE canonical asset id,
//!     so the system never repetitively scans/learns the same market twice.
//!   * Scan dedup: an asset (or pair) is fully re-evaluated at most once per
//!     `scan_min_interval_bars`; cheap EWMA observation still happens on
//!     every bar.
//!   * Pairs are DISCOVERED from measured correlation across the whole
//!     scanned universe (auto_pairs), not only from the hard-coded seed list.

use crate::aegis::{correlation, returns_for, vol_annual, PairMeta};
use crate::simfeed::Bar;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Canonical symbol identity (cross-broker dedup)
// ---------------------------------------------------------------------------

/// Built-in broker/venue alias table. Values are canonical asset ids.
/// Extend from `config/system.toml` -> `[system.symbol_aliases]`.
pub fn builtin_aliases() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("GOLD", "XAUUSD"),
        ("XAUUSD.X", "XAUUSD"),
        ("XAUUSD-ECN", "XAUUSD"),
        ("SILVER", "XAGUSD"),
        ("XAGUSD.X", "XAGUSD"),
        ("NAS100", "US100"),
        ("USTEC", "US100"),
        ("NDX", "US100"),
        ("SPX500", "US500"),
        ("US500", "US500"),
        ("DJ30", "US30"),
        ("DOW", "US30"),
        ("DAX", "GER40"),
        ("GER30", "GER40"),
        ("DE40", "GER40"),
        ("FTSE", "UK100"),
        ("UKX", "UK100"),
        ("NIKKEI", "J225"),
        ("CABLE", "GBPUSD"),
        ("FIBER", "EURUSD"),
        ("SWISSY", "USDCHF"),
        ("AUSSIE", "AUDUSD"),
        ("KIWI", "NZDUSD"),
        ("LOONIE", "USDCAD"),
        ("GAH", "GASOLINE"),
        ("WTI", "USOIL"),
        ("UKOIL", "BRENT"),
        ("BTCUSD", "BTCUSD"),
        ("XBTUSD", "BTCUSD"),
    ])
}

/// Map any broker-specific symbol to its canonical asset id.
///
/// Normalisation: trim, uppercase, strip common broker suffixes
/// (`.r`, `.raw`, `.pro`, `.ecn`, `.x`, `.m`, `.c`, `-ecn`, `_m`, `m`/`c`
/// endings on 6+ char FX symbols), then apply the alias table (built-in plus
/// any user-supplied `[system.symbol_aliases]`).
pub fn canonical_symbol(sym: &str, extra_aliases: &HashMap<String, String>) -> String {
    let mut s = sym.trim().to_uppercase();
    for suf in [
        ".RAW", ".PRO", ".ECN", ".SPOT", ".X", ".R", ".M", ".C", "-ECN", "_M", "_C",
    ] {
        if s.ends_with(suf) {
            s.truncate(s.len() - suf.len());
        }
    }
    // Broker-style short suffixes only when the base is long enough to be a
    // real symbol (avoids mangling e.g. "US30M" style oddities we care about).
    if s.len() >= 6 {
        for suf in ["M", "C", "Z"] {
            if s.ends_with(suf) {
                s.truncate(s.len() - 1);
                break;
            }
        }
    }
    if let Some(a) = extra_aliases.get(&s) {
        return a.to_uppercase();
    }
    for (k, v) in builtin_aliases() {
        if s == k {
            return v.to_string();
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Knowledge base — shared across every desk/account (the "AI memory")
// ---------------------------------------------------------------------------

/// What the system knows about ONE canonical asset. Learned passively from
/// every bar any desk sees and actively from every closed trade.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssetKnowledge {
    /// bars observed (deduped across brokers/desks)
    pub observations: u64,
    /// bar time of the last observation (dedup marker)
    pub last_bar_time: i64,
    /// EWMA of squared log-returns (per-bar variance proxy)
    pub ewma_var: f64,
    /// EWMA of log-returns (per-bar drift proxy)
    pub ewma_drift: f64,
    pub last_close: f64,
    /// realised trades on this asset across ALL accounts
    pub trades: u64,
    pub wins: u64,
    pub realized_pnl: f64,
    /// bar time of the last full re-scan (dedup gate)
    pub last_scanned: i64,
}

impl AssetKnowledge {
    /// Annualised-ish volatility proxy from the EWMA variance.
    pub fn vol(&self) -> f64 {
        self.ewma_var.max(0.0).sqrt() * (252.0f64 * 288.0).sqrt()
    }
    pub fn win_rate(&self) -> f64 {
        if self.trades == 0 {
            0.0
        } else {
            self.wins as f64 / self.trades as f64
        }
    }
}

/// Shared cross-account knowledge. Keyed by CANONICAL asset id, so the same
/// market quoted by two brokers under different symbols merges into one
/// entry — the core of the no-repetitive-scanning requirement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub assets: BTreeMap<String, AssetKnowledge>,
    /// per-pair last full evaluation (dedup gate for pair scanning)
    #[serde(default)]
    pub pair_scans: BTreeMap<String, i64>,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        KnowledgeBase::default()
    }

    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => KnowledgeBase::new(),
        }
    }

    pub fn save(&self, path: &std::path::Path) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Observe bars (passive learning). Deduped per canonical asset per bar
    /// time: two desks receiving the same bar (or two brokers quoting the
    /// same asset) update the entry only once.
    pub fn observe(&mut self, bars: &[Bar], extra_aliases: &HashMap<String, String>) {
        for bar in bars {
            let canon = canonical_symbol(&bar.symbol, extra_aliases);
            let k = self.assets.entry(canon).or_default();
            if k.observations > 0 && k.last_bar_time == bar.time {
                continue; // already learned from this bar (other desk/broker)
            }
            if k.last_close > 0.0 && bar.close > 0.0 {
                let r = (bar.close / k.last_close).ln();
                const ALPHA: f64 = 0.06; // ~16-bar EWMA half-life
                k.ewma_var = (1.0 - ALPHA) * k.ewma_var + ALPHA * r * r;
                k.ewma_drift = (1.0 - ALPHA) * k.ewma_drift + ALPHA * r;
            }
            k.last_close = bar.close;
            k.last_bar_time = bar.time;
            k.observations += 1;
        }
    }

    /// Dedup gate for FULL re-evaluation of an asset. Cheap observation
    /// (see `observe`) still happens every bar; this only throttles the
    /// expensive correlation/bias/decision pass.
    pub fn should_scan_asset(&mut self, asset: &str, bar_time: i64, min_interval: i64) -> bool {
        let k = self.assets.entry(asset.to_string()).or_default();
        if k.last_scanned == 0 || bar_time - k.last_scanned >= min_interval {
            k.last_scanned = bar_time;
            true
        } else {
            false
        }
    }

    /// Dedup gate for full pair evaluation keyed by the (unordered) pair id.
    pub fn should_scan_pair(&mut self, pair_id: &str, bar_time: i64, min_interval: i64) -> bool {
        match self.pair_scans.get(pair_id) {
            Some(t) if bar_time - *t < min_interval => false,
            _ => {
                self.pair_scans.insert(pair_id.to_string(), bar_time);
                true
            }
        }
    }

    /// Record a closed trade outcome on both legs (active learning shared by
    /// every account that trades the pair).
    pub fn record_trade(
        &mut self,
        legs: &[&str],
        pnl: f64,
        extra_aliases: &HashMap<String, String>,
    ) {
        for leg in legs {
            let canon = canonical_symbol(leg, extra_aliases);
            let k = self.assets.entry(canon).or_default();
            k.trades += 1;
            if pnl > 0.0 {
                k.wins += 1;
            }
            k.realized_pnl += pnl / legs.len() as f64;
        }
    }

    /// Human-readable knowledge summary (used by `the-quant scan`).
    pub fn summary(&self, top: usize) -> String {
        let mut rows: Vec<_> = self
            .assets
            .iter()
            .filter(|(_, k)| k.observations > 0)
            .collect();
        rows.sort_by_key(|(_, k)| std::cmp::Reverse(k.observations));
        let mut out =
            String::from("asset          obs      vol%   drift       trades  win%   pnl$\n");
        for (name, k) in rows.iter().take(top) {
            out.push_str(&format!(
                "{:<14} {:>6} {:>8.2} {:>+9.6} {:>7} {:>6.1} {:>8.2}\n",
                name,
                k.observations,
                k.vol() * 100.0,
                k.ewma_drift,
                k.trades,
                k.win_rate() * 100.0,
                k.realized_pnl
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Universe construction + dynamic pair discovery
// ---------------------------------------------------------------------------

/// Build the asset universe one desk scans.
///
/// * `mode`      — account `universe_mode`:
///     - "all_assets" (default): EVERY asset the account can trade (`available`
///       = the feed/broker symbol list), watchlist prioritised first.
///     - "watchlist": only the config watchlist (classic behaviour).
///     - "custom": only `custom` (account.symbols).
/// * `watchlist` — `[system].symbols` (config watchlist)
/// * `available` — every asset the account CAN trade (broker/feed symbols)
/// * `custom`    — account `symbols` (include-list / extra assets)
/// * `exclude`   — account `exclude_symbols` (never scanned/traded)
/// * `cap`       — account `max_scan_assets` hard ceiling (safety: bounds CPU
///   and keeps the learner focused).
pub fn build_universe(
    mode: &str,
    watchlist: &[String],
    available: &[String],
    custom: &[String],
    exclude: &[String],
    cap: usize,
    extra_aliases: &HashMap<String, String>,
) -> Vec<String> {
    let ex: HashSet<String> = exclude
        .iter()
        .map(|s| canonical_symbol(s, extra_aliases))
        .collect();
    let canon = |xs: &[String]| -> Vec<String> {
        xs.iter()
            .map(|s| canonical_symbol(s, extra_aliases))
            .filter(|c| !ex.contains(c))
            .collect()
    };
    let mut ordered: Vec<String> = match mode {
        "watchlist" => canon(watchlist),
        "custom" => canon(custom),
        // all_assets (default): watchlist first, then every tradable asset.
        _ => {
            let mut v = canon(watchlist);
            for a in canon(available).into_iter().chain(canon(custom)) {
                if !v.contains(&a) {
                    v.push(a);
                }
            }
            v
        }
    };
    ordered.dedup();
    ordered.truncate(cap.max(1));
    ordered
}

/// Discover hedged pairs from MEASURED correlation across the scanned
/// universe. Returns at most `max_pairs`, sorted by |correlation| desc.
/// Each pair's long leg is the empirically more volatile asset.
pub fn discover_pairs(
    universe: &[String],
    canon_raw: &HashMap<String, String>,
    windows: &HashMap<String, Vec<Bar>>,
    min_corr: f64,
    max_pairs: usize,
) -> Vec<PairMeta> {
    // canonical asset -> return series (from the desk's own windows)
    let mut series: HashMap<&str, Vec<f64>> = HashMap::new();
    for asset in universe {
        if let Some(raw) = canon_raw.get(asset) {
            if let Some(w) = windows.get(raw) {
                let r = returns_for(w, raw);
                if r.len() >= 30 {
                    series.insert(asset.as_str(), r);
                }
            }
        }
    }
    let vols: HashMap<&str, f64> = series
        .iter()
        .map(|(a, r)| (*a, vol_annual(r).max(1e-9)))
        .collect();
    let mut assets: Vec<&&str> = series.keys().collect();
    assets.sort();

    let mut out: Vec<(f64, PairMeta)> = Vec::new();
    for i in 0..assets.len() {
        for j in (i + 1)..assets.len() {
            let ca = assets[i];
            let cb = assets[j];
            let corr = correlation(&series[ca], &series[cb]);
            if corr < min_corr {
                continue;
            }
            let (long, short) = if vols[ca] >= vols[cb] {
                (ca, cb)
            } else {
                (cb, ca)
            };
            let ratio = vols[long] / vols[short];
            let id = format!("{}_{}", long, short).to_lowercase();
            out.push((
                corr,
                PairMeta::new(&id, long, short, ratio, out.len() as u32 + 100),
            ));
        }
    }
    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(max_pairs);
    out.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalises_broker_symbols() {
        let extra = HashMap::new();
        assert_eq!(canonical_symbol("xauusd", &extra), "XAUUSD");
        assert_eq!(canonical_symbol("GOLD", &extra), "XAUUSD");
        assert_eq!(canonical_symbol("XAUUSDm", &extra), "XAUUSD");
        assert_eq!(canonical_symbol("xauusd.pro", &extra), "XAUUSD");
        assert_eq!(canonical_symbol("NAS100", &extra), "US100");
        assert_eq!(canonical_symbol("USTEC.r", &extra), "US100");
        assert_eq!(canonical_symbol("eurusd", &extra), "EURUSD");
    }

    #[test]
    fn user_aliases_override_builtins() {
        let mut extra = HashMap::new();
        extra.insert("GOLD".to_string(), "XAUUSD_MICRO".to_string());
        assert_eq!(canonical_symbol("gold", &extra), "XAUUSD_MICRO");
    }

    #[test]
    fn universe_modes_filter_and_cap() {
        let feed = vec![
            "EURUSD".to_string(),
            "XAUUSD".to_string(),
            "NAS100".to_string(),
            "GER40".to_string(),
        ];
        let watch = vec!["XAUUSD".to_string(), "EURUSD".to_string()];
        // all_assets: everything the feed offers, watchlist first, capped.
        // (NAS100 canonicalises to US100 — broker aliasing applies.)
        let u = build_universe("all_assets", &watch, &feed, &[], &[], 3, &HashMap::new());
        assert_eq!(u, vec!["XAUUSD", "EURUSD", "US100"]);
        // watchlist mode only scans the watchlist.
        let u = build_universe("watchlist", &watch, &feed, &[], &[], 10, &HashMap::new());
        assert_eq!(u, vec!["XAUUSD", "EURUSD"]);
        // custom mode + exclusions.
        let custom = vec![
            "EURUSD".to_string(),
            "GOLD".to_string(),
            "GER40".to_string(),
        ];
        let u = build_universe(
            "custom",
            &[],
            &feed,
            &custom,
            &["GOLD".to_string()],
            10,
            &HashMap::new(),
        );
        assert_eq!(u, vec!["EURUSD", "GER40"]);
    }

    #[test]
    fn knowledge_base_dedups_across_brokers() {
        let mut kb = KnowledgeBase::new();
        let extra = HashMap::new();
        // The same market, two brokers, same bar time.
        let mk = |sym: &str, t: i64| Bar {
            symbol: sym.into(),
            time: t,
            open: 2400.0,
            high: 2410.0,
            low: 2390.0,
            close: 2400.0,
            volume: 100.0,
        };
        kb.observe(&[mk("XAUUSD", 5), mk("GOLD", 5)], &extra);
        let k = kb.assets.get("XAUUSD").unwrap();
        assert_eq!(k.observations, 1, "same bar via two brokers must dedup");
        kb.observe(&[mk("XAUUSD.x", 6)], &extra);
        assert_eq!(kb.assets.get("XAUUSD").unwrap().observations, 2);
    }

    #[test]
    fn scan_gates_throttle_re_evaluation() {
        let mut kb = KnowledgeBase::new();
        assert!(kb.should_scan_pair("a_b", 100, 10));
        assert!(!kb.should_scan_pair("a_b", 105, 10));
        assert!(kb.should_scan_pair("a_b", 110, 10));
    }
}
