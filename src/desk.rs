//! DESK — per-account trading isolation (v4.0 HERCULES core).
//!
//! Each managed account (`config/system.toml` `[[accounts]]`) runs as its own
//! `TradingDesk` with independent:
//!   - Aegis strategy state (bar windows, pair positions)
//!   - risk engine (`DeskRisk` — Guardian Shield, daily/lifetime limits)
//!   - P&L journal
//!
//! The engine steps every desk on each bar, so one binary serves 2+ accounts
//! on a single Tier-1 box with no cross-account contamination.
//!
//! COMPLIANCE (Aegis spec §1.1 — non-negotiable, enforced HERE at the Rust
//! execution layer as hard circuit breakers):
//!   * `DeskRisk::pre_flight` gates EVERY entry (shield cap, $150/day loss,
//!     $250 lifetime floor, 10% drawdown, daily extraction cap).
//!   * Sizing uses `DeskRisk::risk_budget` (consistency-rule adjusted).
//!   * Daily counters reset at broker midnight via `DeskRisk::new_day`.

use crate::aegis::{
    correlation, decide, estimate_bias, returns_for, seed_pairs, vol_annual, AegisAction, AegisCfg,
    DeskRisk, PairMeta, PairPosition,
};
use crate::config::AccountDef;
use crate::simfeed::{Bar, CorrelatedFeed};
use crate::state::Trade;
use crate::universe::{self, KnowledgeBase};
use crate::util::now_iso8601;
use std::collections::HashMap;

/// Rolling bar window per symbol for one desk.
const WIN: usize = 80;

/// One managed account's complete trading state.
pub struct TradingDesk {
    pub account: AccountDef,
    pub risk: DeskRisk,
    pub positions: Vec<PairPosition>,
    /// symbol -> rolling bar window (keyed by the broker's RAW symbol)
    windows: std::collections::HashMap<String, Vec<Bar>>,
    /// Current trading-day key; a change triggers the daily reset (§1.1).
    current_day: i64,
    pub realized_pnl: f64,
    pub trade_count: u64,
    pub denied_count: u64,
    /// CANONICAL assets this desk scans (v4.1 universe — see `configure`).
    pub universe: Vec<String>,
    /// Effective strategy/risk config: global `[aegis]` merged with this
    /// account's `rules` overrides (safety-clamped — see `config::AegisCfg`).
    pub eff_cfg: AegisCfg,
    /// Broker symbol aliases in effect (raw broker symbol -> canonical asset).
    pub aliases: HashMap<String, String>,
    /// Minimum bars between full re-evaluations of an asset/pair (scan dedup).
    pub scan_interval: i64,
}

impl TradingDesk {
    pub fn new(account: AccountDef) -> Self {
        let start = account.initial_balance;
        TradingDesk {
            account,
            risk: DeskRisk::new(start),

            positions: Vec::new(),
            windows: std::collections::HashMap::new(),
            current_day: i64::MIN,
            realized_pnl: 0.0,
            trade_count: 0,
            denied_count: 0,
            universe: Vec::new(),
            eff_cfg: AegisCfg::default(),
            aliases: HashMap::new(),
            scan_interval: 10,
        }
    }

    /// Apply the account's customisations on top of the global config (v4.1):
    ///   * `rules` overrides merged into the effective Aegis config
    ///     (safety-clamped so no override can break the risk engine),
    ///   * the scan universe built from `universe_mode` (all assets of the
    ///     account by default — not just the watchlist),
    ///   * shield strike limit synced from the effective config.
    pub fn configure(&mut self, cfg: &crate::config::Config, available: &[String]) {
        self.eff_cfg = cfg.aegis.merged_with(&self.account.rules);
        self.risk.shield_strike_limit = self.eff_cfg.shield_strike_limit;
        self.aliases = cfg.system.symbol_aliases.clone();
        self.scan_interval = cfg.system.scan_min_interval_bars.max(1);
        self.universe = universe::build_universe(
            &self.account.universe_mode,
            &cfg.system.symbols,
            available,
            &self.account.symbols,
            &self.account.exclude_symbols,
            self.account.max_scan_assets,
            &self.aliases,
        );
    }

    /// Canonical asset id for a broker-raw symbol.
    pub fn canon(&self, raw: &str) -> String {
        universe::canonical_symbol(raw, &self.aliases)
    }

    /// Feed one multi-symbol bar-tick to this desk. Updates windows, marks
    /// positions, closes expired ones, and looks for new entries. `kb` is the
    /// (shared) knowledge base: closed trades are recorded into it and the
    /// pair scan-dedup gate throttles re-evaluation.
    pub fn step(&mut self, bars: &[Bar], cfg: &AegisCfg, kb: &mut KnowledgeBase) -> Vec<Trade> {
        let mut trades = Vec::new();

        // 0. Daily rollover (spec §1.1): the $150 daily loss ceiling, the
        //    consistency tracker and transient halts reset at broker midnight.
        //    Permanent states (Blown / lifetime-floor Paused) survive.
        if let Some(b) = bars.first() {
            let day = day_key(b.time, &cfg.primary_timeframe);
            if self.current_day != day {
                self.current_day = day;
                self.risk.new_day();
            }
        }

        // 1. Collect latest prices + update rolling windows. Windows and the
        //    price map are keyed by CANONICAL asset id, so the same market
        //    quoted by different brokers under different symbols (GOLD /
        //    XAUUSD.x / XAUUSDm) merges into ONE learned series.
        let mut latest: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for bar in bars {
            let sym = self.canon(&bar.symbol);
            latest.insert(sym.clone(), bar.close);
            let w = self.windows.entry(sym).or_insert_with(Vec::new);
            w.push(bar.clone());
            if w.len() > WIN {
                w.remove(0);
            }
        }

        // 2. Mark to market + increment bars-held + check time-stop closes.
        for pos in &mut self.positions {
            if let (Some(&pa), Some(&pb)) = (latest.get(&pos.pair.a), latest.get(&pos.pair.b)) {
                pos.mark(pa, pb);
            }
            pos.bars_held += 1;
        }

        // 3. Close positions that hit the time stop or the shield loss cap.
        let mut to_close = Vec::new();
        for (i, pos) in self.positions.iter().enumerate() {
            if pos.force_close() {
                to_close.push(i);
            }
        }
        for &i in to_close.iter().rev() {
            let pos = self.positions.remove(i);
            let pnl = pos.unrealized;
            self.risk.register_close(pnl);
            // Active learning: record the outcome on BOTH legs in the shared
            // knowledge base so every desk/account benefits (spec: knowledge
            // is shared, strategies are per-account).
            kb.record_trade(&[&pos.long_sym, &pos.short_sym], pnl, &self.aliases);
            self.realized_pnl += pnl;
            self.trade_count += 1;
            trades.push(Trade {
                id: format!("{}-t{}", self.account.id, self.trade_count),
                account_id: self.account.id.clone(),
                symbol: pos.pair.id.to_string(),
                direction: "HEDGED".into(),
                volume: pos.long_units,
                entry_price: pos.entry_price_a,
                exit_price: pos.entry_price_a, // hedged pair: pnl captured in `pnl`
                pnl,
                open_time: pos.open_time.clone(),
                close_time: now_iso8601(),
                strategy: "aegis".into(),
            });
        }

        // 4. Risk gate — hard circuit breakers before EVERY open (spec §1.1).
        if !self.risk.available() {
            return trades;
        }

        // 5. Look for new Aegis entries. Candidate pairs come from the desk's
        //    scanned universe: dynamically DISCOVERED pairs (auto_pairs =
        //    measured correlation across every scanned asset) merged with the
        //    built-in seed universe as a guaranteed fallback. A full pair
        //    re-evaluation is gated by the shared knowledge base so the same
        //    pair is scanned at most once per `scan_min_interval_bars` — even
        //    when several accounts/brokers quote the same market.
        let bar_time = bars.first().map(|b| b.time).unwrap_or(0);
        let mut pairs: Vec<PairMeta> = Vec::new();
        if self.account.auto_pairs {
            let canon_raw: HashMap<String, String> = self
                .windows
                .keys()
                .map(|raw| (self.canon(raw), raw.clone()))
                .collect();
            pairs.extend(universe::discover_pairs(
                &self.universe,
                &canon_raw,
                &self.windows,
                cfg.min_correlation,
                cfg.max_pairs_active,
            ));
        }
        // Seed-universe fallback: include any seed pair whose legs are in this
        // desk's universe and that discovery did not already produce.
        for p in seed_pairs() {
            if !pairs.iter().any(|x| x.id == p.id)
                && self.universe.contains(&p.a)
                && self.universe.contains(&p.b)
            {
                pairs.push(p);
            }
        }
        pairs.sort_by_key(|p| p.priority);

        for pair in &pairs {
            if self.positions.len() >= cfg.max_pairs_active {
                break;
            }
            if self.positions.iter().any(|p| p.pair.id == pair.id) {
                continue;
            }
            // Cross-account scan dedup (shared knowledge base).
            if !kb.should_scan_pair(&pair.id, bar_time, self.scan_interval) {
                continue;
            }
            if let Some(action) = self.evaluate_pair(pair.clone(), cfg, &latest) {
                if let AegisAction::Open {
                    pair: p,
                    hedge_ratio,
                    bias,
                    prob,
                } = action
                {
                    // Budget under the consistency rule, capped by the
                    // Guardian Shield per-trade maximum ($50 on BG 5K).
                    let budget = self.risk.risk_budget(cfg);
                    // PRE-FLIGHT: shield / daily / lifetime / drawdown gate.
                    if self.risk.pre_flight(cfg, budget).is_err() {
                        self.denied_count += 1;
                        continue;
                    }
                    let price_a = match latest.get(&p.a) {
                        Some(v) => *v,
                        None => continue,
                    };
                    let price_b = match latest.get(&p.b) {
                        Some(v) => *v,
                        None => continue,
                    };
                    let atr = self.spread_std(pair.clone()).unwrap_or(price_a * 0.001);
                    let (units_a, units_b) = crate::aegis::size_legs(
                        budget,
                        price_a,
                        price_b,
                        atr,
                        cfg.atr_multiplier,
                        hedge_ratio,
                    );
                    if units_a <= 0.0 {
                        self.denied_count += 1;
                        continue;
                    }
                    let pos = PairPosition::new(
                        p,
                        price_a,
                        price_b,
                        units_a,
                        units_b,
                        hedge_ratio,
                        bias,
                        prob,
                        cfg.time_stop_bars,
                        cfg.shield_max_loss_per_trade,
                    );
                    self.positions.push(pos);
                }
            }
        }

        trades
    }

    /// Compute Aegis features for a pair and return an action (or Hold).
    fn evaluate_pair(
        &self,
        pair: PairMeta,
        cfg: &AegisCfg,
        _latest: &std::collections::HashMap<String, f64>,
    ) -> Option<AegisAction> {
        let wa = self.windows.get(&pair.a)?;
        let wb = self.windows.get(&pair.b)?;
        if wa.len() < 30 || wb.len() < 30 {
            return None;
        }
        let ra = returns_for(wa, &pair.a);
        let rb = returns_for(wb, &pair.b);
        if ra.len() < 30 {
            return None;
        }
        let corr = correlation(&ra, &rb);
        if corr < cfg.min_correlation {
            return None;
        }
        let va = vol_annual(&ra);
        let vb = vol_annual(&rb);
        let vr = va / vb.max(1e-9);
        if vr < cfg.min_volatility_ratio {
            return None;
        }
        let spread = build_spread(wa, wb, &pair.a, &pair.b)?;
        let (bias, prob) = estimate_bias(&spread);
        if bias.abs() < cfg.min_bias_magnitude {
            return None;
        }
        // prob must be outside the "no edge" zone [1-min_edge, min_edge]
        if prob > 1.0 - cfg.min_probability_edge && prob < cfg.min_probability_edge {
            return None;
        }
        Some(decide(cfg, &self.risk, pair, corr, va, vb, bias, prob))
    }

    /// Std-dev of the spread series (used as ATR proxy for sizing).
    fn spread_std(&self, pair: PairMeta) -> Option<f64> {
        let wa = self.windows.get(&pair.a)?;
        let wb = self.windows.get(&pair.b)?;
        let s = build_spread(wa, wb, &pair.a, &pair.b)?;
        Some(crate::util::stddev(&s).max(1e-9))
    }
}

/// Map a bar timestamp to a day key.
///
/// Live feeds (MT5) carry epoch seconds; sim/csv feeds carry a bar index. For
/// bar indices we bucket by `bars_per_day` derived from the primary timeframe
/// (M5 -> 288 bars per trading day), which reproduces broker-midnight resets.
fn day_key(time: i64, timeframe: &str) -> i64 {
    if time > 100_000_000 {
        time / 86_400 // epoch seconds -> UTC day number
    } else {
        time / bars_per_day(timeframe)
    }
}

/// Bars per 24h for the configured timeframe.
fn bars_per_day(timeframe: &str) -> i64 {
    match timeframe {
        "M1" => 1440,
        "M5" => 288,
        "M15" => 96,
        "M30" => 48,
        "H1" => 24,
        "H4" => 6,
        "D1" => 1,
        _ => 288,
    }
}

/// Build the ln(A)−ln(B) spread series from two synchronous bar windows.
fn build_spread(wa: &[Bar], wb: &[Bar], a: &str, b: &str) -> Option<Vec<f64>> {
    let ma: std::collections::HashMap<i64, f64> = wa
        .iter()
        .filter(|x| x.symbol == a)
        .map(|b| (b.time, b.close))
        .collect();
    let mb: std::collections::HashMap<i64, f64> = wb
        .iter()
        .filter(|x| x.symbol == b)
        .map(|b| (b.time, b.close))
        .collect();
    let mut out = Vec::new();
    for (&t, &ca) in &ma {
        if let Some(&cb) = mb.get(&t) {
            if ca > 0.0 && cb > 0.0 {
                out.push((ca / cb).ln());
            }
        }
    }
    if out.len() >= 30 {
        Some(out)
    } else {
        None
    }
}

/// The multi-symbol feed covering every leg in the Aegis universe.
/// Uses a correlated feed so pairs genuinely co-move (Aegis needs corr > 0.70).
///
/// Betas are tuned so that:
/// - EURUSD/GBPUSD/AUDUSD/NZDUSD (all USD-quoted) share a strong USD factor → high cross-correlation
/// - XAUUSD/XAGUSD (precious metals) share a commodity factor
/// - US100/US30 (US indices) share an equity factor
/// This gives the Aegis pairs strategy real alpha to exploit.
pub fn aegis_feed(seed: u64) -> CorrelatedFeed {
    let symbols = [
        ("EURUSD", 1.08),
        ("GBPUSD", 1.26),
        ("AUDUSD", 0.65),
        ("NZDUSD", 0.60),
        ("XAUUSD", 2400.0),
        ("XAGUSD", 28.0),
        ("US100", 19000.0),
        ("US30", 38000.0),
    ];
    // Beta: how much each symbol loads on the common market factor.
    // Higher beta = more correlated with other high-beta symbols.
    let betas = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55];
    // Idiosyncratic noise (lower = cleaner correlation).
    let idiovol = 0.0004;
    CorrelatedFeed::new(&symbols, &betas, seed, idiovol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aegis::size_legs;
    use crate::config::AegisCfg;

    fn test_cfg() -> AegisCfg {
        AegisCfg::default()
    }

    fn synth_bars(symbol: &str, start_time: i64, n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| Bar {
                symbol: symbol.to_string(),
                time: start_time + i as i64,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn daily_loss_limit_blocks_entries_then_resets_next_day() {
        let cfg = test_cfg();
        let mut desk = TradingDesk::new(AccountDef::default());
        // Simulate a bad day: -$150 realized → daily ceiling reached.
        desk.risk.register_close(-150.0);
        assert_eq!(desk.risk.day_pnl, -150.0);
        // Pre-flight must refuse any new entry.
        assert!(desk.risk.pre_flight(&cfg, 20.0).is_err());

        // A new trading day resets the daily counters (§1.1).
        let bars = synth_bars("XAUUSD", 0, 1);
        let mut kb = KnowledgeBase::new();
        desk.step(&bars, &cfg, &mut kb); // triggers rollover (day key changes)
        assert_eq!(desk.risk.day_pnl, 0.0);
        assert!(desk.risk.pre_flight(&cfg, 20.0).is_ok());
    }

    #[test]
    fn lifetime_floor_pauses_and_survives_rollover() {
        let cfg = test_cfg();
        let mut risk = DeskRisk::new(5000.0);
        risk.register_close(-250.0);
        assert!(risk.pre_flight(&cfg, 10.0).is_err());
        assert_eq!(risk.status, "Paused");
        // A new day must NOT un-pause a lifetime-floor pause.
        risk.new_day();
        assert!(risk.pre_flight(&cfg, 10.0).is_err());
    }

    #[test]
    fn shield_blows_after_two_losing_closes() {
        let cfg = test_cfg();
        let mut desk = TradingDesk::new(AccountDef::default());
        desk.risk.register_close(-45.0);
        desk.risk.register_close(-45.0);
        assert_eq!(desk.risk.shield_strikes, 2);
        assert_eq!(desk.risk.status, "Blown");
        assert!(desk.risk.pre_flight(&cfg, 20.0).is_err());
        // A blown desk never opens new positions no matter how many bars pass.
        for chunk_start in (0..600).step_by(50) {
            let bars = synth_bars("XAUUSD", chunk_start, 50);
            let mut kb = KnowledgeBase::new();
            desk.step(&bars, &cfg, &mut kb);
        }
        assert!(desk.positions.is_empty());
    }

    #[test]
    fn pre_flight_rejects_oversized_risk() {
        let cfg = test_cfg();
        let mut risk = DeskRisk::new(5000.0);
        // Shield cap is $50/trade: a $51 risk must be rejected.
        assert!(risk.pre_flight(&cfg, 51.0).is_err());
        assert!(risk.pre_flight(&cfg, 50.0).is_ok());
    }

    #[test]
    fn extraction_cap_stops_chasing_after_big_day() {
        let cfg = test_cfg();
        let mut risk = DeskRisk::new(5000.0);
        risk.register_close(cfg.max_daily_profit); // hit the daily profit cap
        assert!(risk.pre_flight(&cfg, 10.0).is_err());
        assert_eq!(risk.status, "ExtractionCap");
        // …and a new trading day re-enables trading.
        risk.new_day();
        assert!(risk.pre_flight(&cfg, 10.0).is_ok());
    }

    #[test]
    fn size_legs_scales_with_budget_not_equity() {
        // $25 budget, $50 stop distance → $0.5 notional per price unit.
        let (ua, ub) = size_legs(25.0, 100.0, 50.0, 25.0, 2.0, 0.5);
        assert!((ua - 0.005).abs() < 1e-9, "units_a={ua}");
        assert!((ub - 0.005).abs() < 1e-9, "units_b={ub}");
        // Zero/negative inputs must never panic or produce garbage.
        let (z, _) = size_legs(100.0, 100.0, 50.0, 0.0, 2.0, 0.5);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn day_key_buckets_bar_indices_by_timeframe() {
        assert_eq!(day_key(0, "M5"), 0);
        assert_eq!(day_key(287, "M5"), 0);
        assert_eq!(day_key(288, "M5"), 1);
        assert_eq!(day_key(95, "M15"), 0);
        assert_eq!(day_key(96, "M15"), 1);
        // Values above the epoch/index threshold bucket by UTC day.
        assert_eq!(day_key(100_000_001, "M5"), 100_000_001 / 86_400);
        assert_eq!(day_key(1_700_000_000, "M5"), 1_700_000_000 / 86_400);
    }
}
