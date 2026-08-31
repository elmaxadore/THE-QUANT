//! AEGIS — hedged correlated pairs strategy (v4.0 module).
//!
//! Implements the hypothesis from `Aegis_Hedged_Pairs_Strategy_Spec.txt`:
//!
//!   "Given a universe of correlated FX/metal/index pairs, when a probabilistic
//!    directional bias is detected on a correlated cluster, construct a hedged
//!    position: BUY the higher-volatility leg (long gamma) and SELL the
//!    lower-volatility leg (short hedge), sizing each leg so the NET position
//!    delta reflects the probability differential while respecting the
//!    Guardian Shield ($50/trade), $150/day loss, $250 lifetime floor and
//!    10% max drawdown."
//!
//! This is DIRECTIONAL-hedged, not market-neutral: the lower-vol leg is a
//! dynamic hedge whose ratio is `corr × (1 − |2·bias − 1|)`, so the hedge
//! weakens as directional confidence grows (net delta follows the bias).

pub use crate::config::AegisCfg;
use crate::simfeed::Bar;

/// Static metadata for one hedged pair. Legs are owned strings so pairs can
/// be discovered dynamically from the scanned universe (not just the built-in
/// seed universe).
#[derive(Debug, Clone, PartialEq)]
pub struct PairMeta {
    pub id: String,
    pub a: String,
    pub b: String,
    pub exp_vol_ratio: f64,
    pub priority: u32,
}

impl PairMeta {
    pub fn new(id: &str, a: &str, b: &str, exp_vol_ratio: f64, priority: u32) -> Self {
        PairMeta {
            id: id.to_string(),
            a: a.to_string(),
            b: b.to_string(),
            exp_vol_ratio,
            priority,
        }
    }
}

/// The Aegis SEED pair universe (spec §8.2). `a` = higher-vol leg. These are
/// always available as a guaranteed fallback; with `auto_pairs = true` the
/// desks additionally discover pairs dynamically from the scanned universe
/// (see `universe::discover_pairs`).
pub fn seed_pairs() -> Vec<PairMeta> {
    vec![
        PairMeta::new("xau_xag", "XAUUSD", "XAGUSD", 1.2, 1),
        PairMeta::new("gbp_eur", "GBPUSD", "EURUSD", 1.15, 2),
        PairMeta::new("us100_us30", "US100", "US30", 1.25, 3),
        PairMeta::new("aud_nzd", "AUDUSD", "NZDUSD", 1.1, 4),
        PairMeta::new("gbp_aud", "GBPUSD", "AUDUSD", 1.2, 5),
    ]
}

/// A trading decision for one pair: open a hedged position or do nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum AegisAction {
    /// Open the hedged position: long leg A, short leg B.
    /// `hedge_ratio` is the short-leg amount as a fraction of the long leg.
    Open {
        pair: PairMeta,
        hedge_ratio: f64,
        bias: f64,
        prob: f64,
    },
    Hold,
}

/// Live risk state for one managed desk (Blue Guardian Instant 5K spec §1.1).
#[derive(Debug, Clone)]
pub struct DeskRisk {
    pub starting_equity: f64,
    pub peak_equity: f64,
    pub equity: f64,
    /// cumulative realized P&L (account lifetime)
    pub cum_pnl: f64,
    /// realized P&L today
    pub day_pnl: f64,
    pub day_profit_peak: f64,
    /// today's largest single-day-profit (for consistency ratio)
    pub max_daily_profit: f64,
    pub shield_strikes: u32,
    /// losses allowed before the Guardian Shield blows (settable from cfg)
    pub shield_strike_limit: u32,
    pub status: String, // Active | Strike | Paused | Blown | ExtractionCap
}

impl DeskRisk {
    pub fn new(starting_equity: f64) -> Self {
        DeskRisk {
            starting_equity,
            peak_equity: starting_equity,
            equity: starting_equity,
            cum_pnl: 0.0,
            day_pnl: 0.0,
            day_profit_peak: 0.0,
            max_daily_profit: 0.0,
            shield_strikes: 0,
            shield_strike_limit: 2,
            status: "Active".into(),
        }
    }

    pub fn available(&self) -> bool {
        self.status == "Active"
    }

    /// Reset daily counters at broker midnight (spec §1.1: the $150 daily
    /// loss ceiling, consistency tracker and transient halts reset per day).
    /// Permanent states ("Blown", "Paused" — lifetime floor) survive the
    /// rollover and require operator action.
    pub fn new_day(&mut self) {
        self.day_pnl = 0.0;
        self.max_daily_profit = 0.0;
        self.day_profit_peak = 0.0;
        if self.status == "Strike" || self.status == "ExtractionCap" {
            self.status = "Active".into();
        }
    }

    /// Hard circuit breakers (spec §1.1). Called before every open attempt.
    pub fn pre_flight(&mut self, cfg: &AegisCfg, risk_usd: f64) -> Result<(), String> {
        if self.status == "Blown" {
            return Err("account halted (blown)".into());
        }
        if self.shield_strikes >= cfg.shield_strike_limit {
            self.status = "Blown".into();
            return Err("Guardian Shield blown (2 strikes) — account halted".into());
        }
        // Lifetime loss floor → pause requires manual reset (spec).
        if self.cum_pnl <= -cfg.lifetime_loss_floor {
            self.status = "Paused".into();
            return Err("lifetime loss floor reached — paused".into());
        }
        // Daily loss ceiling (realized).
        if self.day_pnl <= -cfg.daily_loss_limit {
            self.status = "Strike".into();
            return Err("daily loss limit reached — no new trades".into());
        }
        // Max drawdown (peak-to-trough on equity).
        let dd = if self.peak_equity > 0.0 {
            (self.peak_equity - self.equity) / self.peak_equity * 100.0
        } else {
            0.0
        };
        if dd >= cfg.max_drawdown_pct {
            self.status = "Blown".into();
            return Err(format!(
                "drawdown {dd:.1}% >= {}% — account blown",
                cfg.max_drawdown_pct
            ));
        }
        // Extraction soft-stop: never chase beyond the daily profit cap.
        if self.day_pnl >= cfg.max_daily_profit {
            self.status = "ExtractionCap".into();
            return Err(format!(
                "daily profit cap {:.0} hit — good day, stop",
                cfg.max_daily_profit
            ));
        }
        // Per-trade shield: rejected up-front if risk would exceed the cap.
        if risk_usd > cfg.shield_max_loss_per_trade {
            return Err(format!(
                "shield: {risk_usd:.2} exceeds ${:.0} per trade",
                cfg.shield_max_loss_per_trade
            ));
        }
        Ok(())
    }

    /// The allowed risk budget for the next opening, respecting the
    /// consistency rule (15% max day/total) by halving size when near it.
    pub fn risk_budget(&self, cfg: &AegisCfg) -> f64 {
        let base = self.equity * cfg.risk_per_trade_pct / 100.0;
        let mut budget = base.min(cfg.shield_max_loss_per_trade);
        // Consistency guard: if today's profit approaches the threshold share
        // of total realised profit, halve size (spec: 15% consistency rule).
        let total = self.cum_pnl.max(0.0) + 1e-9;
        let ratio = self.max_daily_profit / total;
        if ratio > cfg.consistency_threshold_pct / 100.0 * 0.8 {
            budget *= 0.5;
        }
        budget
    }

    pub fn register_close(&mut self, pnl: f64) {
        self.cum_pnl += pnl;
        self.day_pnl += pnl;
        if self.day_pnl > self.max_daily_profit {
            self.max_daily_profit = self.day_pnl;
        }
        if pnl < 0.0 {
            self.shield_strikes += 1; // a losing pair-close counts as a strike
            if self.shield_strikes >= self.shield_strike_limit {
                self.status = "Blown".into(); // halt NOW, not at next pre-flight
            }
        }
        self.equity = self.starting_equity + self.cum_pnl;
        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
        self.day_profit_peak = self.day_profit_peak.max(self.day_pnl);
    }
}

/// A hedged position currently held by one desk.
#[derive(Debug, Clone)]
pub struct PairPosition {
    pub pair: PairMeta,
    pub long_sym: String,
    pub short_sym: String,
    pub long_units: f64,
    pub short_units: f64,
    pub hedge_ratio: f64,
    pub bias: f64,
    pub prob: f64,
    pub entry_price_a: f64,
    pub entry_price_b: f64,
    pub bars_held: usize,
    /// Max bars to hold before forced close (time stop).
    pub time_stop_bars: usize,
    pub max_loss: f64,
    /// unrealized P&L of the pair if closed now (both legs marked)
    pub unrealized: f64,
    /// ISO-8601 timestamp when the position was opened.
    pub open_time: String,
}

impl PairPosition {
    /// Mark both legs. P&L ≈ (pa−entryA)·long_units + (entryB−pb)·short_units.
    pub fn mark(&mut self, pa: f64, pb: f64) -> f64 {
        let long_pnl = (pa - self.entry_price_a) * self.long_units;
        let short_pnl = (self.entry_price_b - pb) * self.short_units;
        self.unrealized = long_pnl + short_pnl;
        self.unrealized
    }

    pub fn force_close(&self) -> bool {
        self.unrealized <= -self.max_loss || self.bars_held >= self.time_stop_bars
    }
}

impl PairPosition {
    /// Open a new hedged pair position.
    pub fn new(
        pair: PairMeta,
        price_a: f64,
        price_b: f64,
        units_a: f64,
        units_b: f64,
        hedge_ratio: f64,
        bias: f64,
        prob: f64,
        time_stop_bars: usize,
        max_loss: f64,
    ) -> Self {
        PairPosition {
            long_sym: pair.a.clone(),
            short_sym: pair.b.clone(),
            pair,
            long_units: units_a,
            short_units: units_b,
            hedge_ratio,
            bias,
            prob,
            entry_price_a: price_a,
            entry_price_b: price_b,
            bars_held: 0,
            time_stop_bars,
            max_loss, // Guardian Shield per-trade cap (cfg)
            unrealized: 0.0,
            open_time: crate::util::now_iso8601(),
        }
    }
}

/// Size the two legs of a hedged pair. Returns (units_a, units_b).
///
/// `risk_budget` is the max loss allowed for this pair (from `DeskRisk::risk_budget`).
/// `atr` is the spread std-dev used as the stop distance. `hedge_ratio` scales
/// the short leg so net delta follows the directional bias.
pub fn size_legs(
    risk_budget: f64,
    price_a: f64,
    price_b: f64,
    atr: f64,
    atr_multiplier: f64,
    hedge_ratio: f64,
) -> (f64, f64) {
    let stop_dist = atr * atr_multiplier;
    if stop_dist <= 0.0 || price_a <= 0.0 || price_b <= 0.0 {
        return (0.0, 0.0);
    }
    // notional = budget / stop_dist (spread units → USD notional).
    let notional = risk_budget / stop_dist;
    // units_a = notional / price_a (convert to units of currency A).
    let units_a = notional / price_a;
    // units_b = notional / price_b * hedge_ratio (hedged short leg).
    let units_b = (notional / price_b) * hedge_ratio;
    (units_a, units_b)
}
/// The statistical heart — the Aegis hypothesis gate.
///
/// `bias` is the probabilistic directional bias of a pair's cluster (−1..1,
/// >0 → long A / short B), `prob` its confidence (0..1). We only trade when:
///   corr ≥ min_correlation, vol ratio ≥ min_volatility_ratio,
///   prob outside [min_edge, 1−min_edge], |bias| ≥ min_bias_magnitude.
pub fn decide(
    cfg: &AegisCfg,
    risk: &DeskRisk,
    meta: PairMeta,
    corr: f64,
    vol_a: f64,
    vol_b: f64,
    bias: f64,
    prob: f64,
) -> AegisAction {
    if !risk.available() {
        return AegisAction::Hold;
    }
    if corr < cfg.min_correlation {
        return AegisAction::Hold; // not correlated enough for a hedge
    }
    let vr = (vol_a / vol_b).max(vol_b / vol_a);
    if vr < cfg.min_volatility_ratio {
        return AegisAction::Hold; // no vol differential → no gamma edge
    }
    if prob > 1.0 - cfg.min_probability_edge && prob < cfg.min_probability_edge {
        return AegisAction::Hold; // probability edge not strong enough (prob is in the no-edge zone)
    }
    if bias.abs() < cfg.min_bias_magnitude {
        return AegisAction::Hold;
    }
    // Hedge ratio: full hedge at low confidence, disappearing as confidence
    // → 1 (net delta follows the probability differential). Clamp ≥ 0.2 so a
    // position always has *some* downside protection.
    let hedge_ratio = (corr * (1.0 - (2.0 * prob - 1.0).abs())).max(0.2);
    AegisAction::Open {
        pair: meta,
        hedge_ratio,
        bias,
        prob,
    }
}

/// Rolling Pearson correlation between two assets' log-return series.
pub fn correlation(ra: &[f64], rb: &[f64]) -> f64 {
    let n = ra.len().min(rb.len()).min(60);
    if n < 3 {
        return 0.0;
    }
    let ra = &ra[ra.len() - n..];
    let rb = &rb[rb.len() - n..];
    let ma = crate::util::mean(ra);
    let mb = crate::util::mean(rb);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (xa, xb) = (ra[i] - ma, rb[i] - mb);
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        (num / den).clamp(-1.0, 1.0)
    }
}

/// Realised volatility (annualised proxy) from a return series.
pub fn vol_annual(returns: &[f64]) -> f64 {
    crate::util::stddev(returns) * (252.0f64).sqrt()
}

/// Log-return series of one symbol within a (possibly multi-symbol) window.
pub fn returns_for(bars: &[Bar], symbol: &str) -> Vec<f64> {
    let mut out = Vec::with_capacity(bars.len().saturating_sub(1));
    let mut prev: Option<f64> = None;
    for b in bars {
        if b.symbol != symbol {
            continue;
        }
        if let Some(p) = prev {
            if p > 0.0 && b.close > 0.0 && b.close.is_finite() {
                out.push((b.close / p).ln());
            }
        }
        prev = Some(b.close);
    }
    out
}

/// A cheap, transparent directional-bias estimator standing in for the
/// `aegis_direction_prob` ONNX graph. It blends short-horizon momentum of the
/// (A−B) spread with mean-reversion of the spread z-score:
///   spread   = ln(A) − ln(B)
///   momentum = slope of spread over the last `w` bars
///   reversion = −z(spread)
///   prob     = 1/(1+exp(−gain·(momentum·α + reversion·β + c)))
pub fn estimate_bias(spread: &[f64]) -> (f64, f64) {
    let n = spread.len();
    if n < 10 {
        return (0.0, 0.5);
    }
    let w = n.min(20);
    let recent = &spread[n - w..];
    let m0 = crate::util::mean(recent);
    let slope = (recent[w - 1] - recent[0]) / w as f64;
    let sd = crate::util::stddev(recent).max(1e-9);
    let z = (spread[n - 1] - m0) / sd;
    let momentum = slope * 1000.0; // scale to bars
    let reversion = -z; // fade extreme spread
    let bias_raw = 0.6 * momentum + 0.4 * reversion;
    let prob = 1.0 / (1.0 + (-3.0 * bias_raw).exp());
    let bias = bias_raw.clamp(-1.0, 1.0);
    (bias, prob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simfeed::SimFeed;

    fn bars_for(sym: &str, base: f64, seed: u64, n: usize) -> Vec<Bar> {
        let mut f = SimFeed::new(sym, base, seed);
        (0..n).map(|_| f.next_bar()).collect()
    }

    #[test]
    fn correlation_is_bounded_and_positive_for_linked_feeds() {
        let x = bars_for("XAUUSD", 2400.0, 7, 200);
        let y = bars_for("XAGUSD", 28.0, 7, 200); // same seed → co-moving
        let rx = returns_for(&x, "XAUUSD");
        let ry = returns_for(&y, "XAGUSD");
        let c = correlation(&rx, &ry);
        assert!((-1.0..=1.0).contains(&c));
        assert!(c > 0.2, "linked feeds should correlate, got {c}");
    }

    #[test]
    fn decides_only_on_strong_edge() {
        let cfg = crate::config::Config::load()
            .map(|c| c.aegis)
            .unwrap_or_default();
        let risk = DeskRisk::new(5000.0);
        let p0 = seed_pairs()[0].clone();
        let strong = decide(&cfg, &risk, p0.clone(), 0.9, 30.0, 18.0, 0.3, 0.70);
        assert!(matches!(strong, AegisAction::Open { .. }));
        let weak_p = decide(&cfg, &risk, p0.clone(), 0.9, 30.0, 18.0, 0.3, 0.55);
        assert_eq!(weak_p, AegisAction::Hold);
        let low_corr = decide(&cfg, &risk, p0, 0.3, 30.0, 18.0, 0.3, 0.70);
        assert_eq!(low_corr, AegisAction::Hold);
    }

    #[test]
    fn shield_halts_after_two_strikes() {
        let cfg = crate::config::Config::load()
            .map(|c| c.aegis)
            .unwrap_or_default();
        let mut risk = DeskRisk::new(5000.0);
        assert!(risk.pre_flight(&cfg, 40.0).is_ok());
        risk.register_close(-45.0);
        risk.register_close(-40.0);
        assert_eq!(risk.shield_strikes, 2);
        assert!(risk.pre_flight(&cfg, 40.0).is_err());
        assert_eq!(risk.status, "Blown");
    }
}
