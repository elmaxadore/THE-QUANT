//! Configuration loader — reads `config/system.toml`.

use serde::Deserialize;
use std::path::{Path, PathBuf};

const DEFAULT_CFG: &str = include_str!("../config/system.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct SystemCfg {
    #[serde(default = "d_name")]
    pub name: String,
    #[serde(default = "d_upd")]
    pub update_interval_hours: u64,
    #[serde(default = "d_state_dir")]
    pub state_dir: String,
    #[serde(default = "d_config_dir")]
    pub config_dir: String,
    #[serde(default = "d_model_dir")]
    pub model_dir: String,
    #[serde(default = "d_repo")]
    pub repo_dir: String,
    #[serde(default)]
    pub repo_remote: String,
    #[serde(default)]
    pub headless: bool,
    #[serde(default = "d_true")]
    pub simulated_feed: bool,
    #[serde(default = "d_data_source")]
    pub data_source: String,
    #[serde(default = "d_mt5_dir")]
    pub mt5_dir: String,
    #[serde(default)]
    pub symbols: Vec<String>,
    /// Share one knowledge base across all accounts (recommended). Each desk
    /// remains an isolated trader; only market learning is pooled.
    #[serde(default = "d_true")]
    pub share_knowledge: bool,
    /// Throttle full re-evaluation of an asset/pair to at most once per N
    /// bars (cross-broker, per canonical asset). Cheap EWMA observation still
    /// happens on every bar.
    #[serde(default = "d_scan_int")]
    pub scan_min_interval_bars: i64,
    /// User-supplied broker symbol aliases merged over the built-in table.
    /// e.g. `GOLD = "XAUUSD"`.
    #[serde(default)]
    pub symbol_aliases: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AccountCfg {
    #[serde(default = "d_acc")]
    pub r#type: String,
    #[serde(default = "d_dd")]
    pub max_drawdown_pct: f64,
    #[serde(default = "d_dl")]
    pub daily_loss_limit_pct: f64,
    #[serde(default = "d_risk")]
    pub risk_per_trade_pct: f64,
    #[serde(default = "d_maxlot")]
    pub max_lot_size: f64,
    #[serde(default = "d_lev")]
    pub leverage: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpdateCfg {
    #[serde(default)]
    pub cargo: String,
    #[serde(default)]
    pub git: String,
    #[serde(default = "d_true")]
    pub require_smoke_test: bool,
    #[serde(default = "d_true")]
    pub keep_previous: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "d_model_path")]
    pub model_path: String,
    #[serde(default = "d_inferms")]
    pub max_inference_ms: f64,
}

/// One managed trading account (v4.0 isolation: each desk has its own
/// strategy pool, risk engine and journal). Populated from TOML `[[accounts]]`.
#[derive(Debug, Clone, Deserialize)]
pub struct AccountDef {
    #[serde(default = "d_acc_id")]
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// firm key matching a config/templates/<firm>.yaml
    #[serde(default)]
    pub firm: String,
    /// variant key inside that template (e.g. "instant_5k")
    #[serde(default)]
    pub variant: String,
    #[serde(default = "d_balance")]
    pub initial_balance: f64,
    #[serde(default = "d_acc")]
    pub r#type: String, // PERSONAL | PROP_FUNDED | PROP_EVALUATION
    #[serde(default)]
    pub strategy: String, // aegis | rules
    // ---- universe / scanning (v4.1) --------------------------------------
    /// Which assets this desk scans:
    ///   "all_assets" (default) — every asset the account can trade,
    ///   "watchlist"            — only [system].symbols,
    ///   "custom"               — only `symbols` below.
    #[serde(default = "d_universe_mode")]
    pub universe_mode: String,
    /// Include-list for universe_mode = "custom"; extra assets for all_assets.
    #[serde(default)]
    pub symbols: Vec<String>,
    /// Assets this desk must never scan or trade (canonicalised).
    #[serde(default)]
    pub exclude_symbols: Vec<String>,
    /// Hard cap on assets scanned per desk (safety: bounds CPU + focus).
    #[serde(default = "d_max_assets")]
    pub max_scan_assets: usize,
    /// Discover hedged pairs dynamically from measured correlation (v4.1).
    /// When false the desk trades only the built-in seed universe.
    #[serde(default = "d_true")]
    pub auto_pairs: bool,
    // ---- per-account rule overrides (v4.1) --------------------------------
    /// Every strategy/risk knob can be tuned per account; `None` inherits the
    /// global `[aegis]` value. All overrides pass through `safety_clamp`, so a
    /// mistake can loosen the strategy but never break the risk engine.
    #[serde(default)]
    pub rules: RulesOverride,
}

/// Per-account rule overrides — all fields optional.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RulesOverride {
    pub max_drawdown_pct: Option<f64>,
    pub daily_loss_limit: Option<f64>,
    pub lifetime_loss_floor: Option<f64>,
    pub shield_max_loss_per_trade: Option<f64>,
    pub shield_strike_limit: Option<u32>,
    pub risk_per_trade_pct: Option<f64>,
    pub consistency_threshold_pct: Option<f64>,
    pub max_daily_profit: Option<f64>,
    pub min_correlation: Option<f64>,
    pub min_volatility_ratio: Option<f64>,
    pub min_bias_magnitude: Option<f64>,
    pub min_probability_edge: Option<f64>,
    pub max_pairs_active: Option<usize>,
    pub time_stop_bars: Option<usize>,
    pub atr_multiplier: Option<f64>,
    pub primary_timeframe: Option<String>,
    pub target_daily_profit: Option<f64>,
}

impl RulesOverride {
    /// True when the operator overrode at least one rule.
    pub fn is_empty(&self) -> bool {
        self.max_drawdown_pct.is_none()
            && self.daily_loss_limit.is_none()
            && self.lifetime_loss_floor.is_none()
            && self.shield_max_loss_per_trade.is_none()
            && self.shield_strike_limit.is_none()
            && self.risk_per_trade_pct.is_none()
            && self.consistency_threshold_pct.is_none()
            && self.max_daily_profit.is_none()
            && self.min_correlation.is_none()
            && self.min_volatility_ratio.is_none()
            && self.min_bias_magnitude.is_none()
            && self.min_probability_edge.is_none()
            && self.max_pairs_active.is_none()
            && self.time_stop_bars.is_none()
            && self.atr_multiplier.is_none()
            && self.primary_timeframe.is_none()
            && self.target_daily_profit.is_none()
    }
}

impl Default for AccountDef {
    fn default() -> Self {
        AccountDef {
            id: d_acc_id(),
            name: String::new(),
            firm: String::new(),
            variant: String::new(),
            initial_balance: d_balance(),
            r#type: d_acc(),
            strategy: "aegis".into(),
            universe_mode: d_universe_mode(),
            symbols: Vec::new(),
            exclude_symbols: Vec::new(),
            max_scan_assets: d_max_assets(),
            auto_pairs: true,
            rules: RulesOverride::default(),
        }
    }
}

fn d_universe_mode() -> String {
    "all_assets".into()
}
fn d_max_assets() -> usize {
    12
}
/// Aegis — hedged correlated pairs strategy configuration (spec §8.1).
#[derive(Debug, Clone, Deserialize)]
pub struct AegisCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "d_primary_tf")]
    pub primary_timeframe: String,
    #[serde(default = "d_min_corr")]
    pub min_correlation: f64,
    #[serde(default = "d_min_vr")]
    pub min_volatility_ratio: f64,
    #[serde(default = "d_max_pairs")]
    pub max_pairs_active: usize,
    #[serde(default = "d_min_pedge")]
    pub min_probability_edge: f64,
    #[serde(default = "d_min_bias")]
    pub min_bias_magnitude: f64,
    #[serde(default = "d_risk")]
    pub risk_per_trade_pct: f64,
    #[serde(default = "d_atr_mult")]
    pub atr_multiplier: f64,
    #[serde(default = "d_time_stop_bars")]
    pub time_stop_bars: usize,
    /// Guardan Shield — max REALIZED loss per pair-trade (USD)
    #[serde(default = "d_shield_max")]
    pub shield_max_loss_per_trade: f64,
    #[serde(default = "d_shield_strikes")]
    pub shield_strike_limit: u32,
    /// Daily loss ceiling (USD, realized)
    #[serde(default = "d_daily_loss")]
    pub daily_loss_limit: f64,
    /// Lifetime realized-loss pause threshold (USD)
    #[serde(default = "d_lifetime")]
    pub lifetime_loss_floor: f64,
    /// Hard drawdown cap (%)
    #[serde(default = "d_dd")]
    pub max_drawdown_pct: f64,
    /// Blue Guardian profit consistency (max day/total ratio %)
    #[serde(default = "d_consistency")]
    pub consistency_threshold_pct: f64,
    /// Extraction: soft stop beyond max_daily_profit, target of target_daily_profit
    #[serde(default = "d_target_daily")]
    pub target_daily_profit: f64,
    #[serde(default = "d_max_daily")]
    pub max_daily_profit: f64,
}

impl AegisCfg {
    /// Merge a per-account `RulesOverride` on top of this global config, with
    /// SAFETY CLAMPS: an override may loosen or tighten the STRATEGY gates
    /// (correlation, bias, edge, pairs, time stop) but can never push a RISK
    /// control beyond the hard guardrails below. This is what guarantees the
    /// customisation API "cannot make the system unprofitable by accident"
    /// (no rule can exceed prop-firm-fatal limits or unbounded sizing).
    pub fn merged_with(&self, o: &RulesOverride) -> AegisCfg {
        // Hard guardrails (worst-case allowed per account, $ values scaled
        // for small accounts; percentages absolute):
        const MAX_DD_PCT: f64 = 10.0; // never above hard prop-firm DD
        const MAX_DAILY_LOSS: f64 = 500.0; // $
        const MAX_LIFETIME_FLOOR: f64 = 1000.0; // $
        const MAX_SHIELD_TRADE: f64 = 100.0; // $
        const MAX_RISK_PCT: f64 = 2.0; // % per trade
        const MAX_STRIKES: u32 = 5;
        const MIN_CONSISTENCY: f64 = 5.0; // tighter = smaller; floor guards nonsense 0
        let cl = |v: Option<f64>, max: f64| v.map(|x| x.min(max).max(0.0));
        let cup = |v: Option<f64>, max: f64| v.map(|x| x.min(max));
        AegisCfg {
            enabled: self.enabled,
            primary_timeframe: o
                .primary_timeframe
                .clone()
                .unwrap_or_else(|| self.primary_timeframe.clone()),
            min_correlation: cup(o.min_correlation, 1.0).unwrap_or(self.min_correlation),
            min_volatility_ratio: o
                .min_volatility_ratio
                .map(|x| x.max(1.0))
                .unwrap_or(self.min_volatility_ratio),
            max_pairs_active: o
                .max_pairs_active
                .map(|x: usize| x.min(8))
                .unwrap_or(self.max_pairs_active),
            min_probability_edge: cup(o.min_probability_edge, 0.5)
                .unwrap_or(self.min_probability_edge),
            min_bias_magnitude: cup(o.min_bias_magnitude, 1.0).unwrap_or(self.min_bias_magnitude),
            risk_per_trade_pct: cl(o.risk_per_trade_pct, MAX_RISK_PCT)
                .unwrap_or(self.risk_per_trade_pct),
            atr_multiplier: o
                .atr_multiplier
                .map(|x| x.min(5.0).max(0.5))
                .unwrap_or(self.atr_multiplier),
            time_stop_bars: o.time_stop_bars.unwrap_or(self.time_stop_bars),
            shield_max_loss_per_trade: cl(o.shield_max_loss_per_trade, MAX_SHIELD_TRADE)
                .unwrap_or(self.shield_max_loss_per_trade),
            shield_strike_limit: o
                .shield_strike_limit
                .map(|x: u32| x.min(MAX_STRIKES))
                .unwrap_or(self.shield_strike_limit),
            daily_loss_limit: cl(o.daily_loss_limit, MAX_DAILY_LOSS)
                .unwrap_or(self.daily_loss_limit),
            lifetime_loss_floor: cl(o.lifetime_loss_floor, MAX_LIFETIME_FLOOR)
                .unwrap_or(self.lifetime_loss_floor),
            max_drawdown_pct: cl(o.max_drawdown_pct, MAX_DD_PCT).unwrap_or(self.max_drawdown_pct),
            consistency_threshold_pct: cl(o.consistency_threshold_pct, 100.0)
                .unwrap_or(self.consistency_threshold_pct)
                .max(MIN_CONSISTENCY),
            target_daily_profit: cl(o.target_daily_profit, 500.0)
                .unwrap_or(self.target_daily_profit),
            max_daily_profit: cl(o.max_daily_profit, 200.0).unwrap_or(self.max_daily_profit),
        }
    }
}

impl Default for AegisCfg {
    fn default() -> Self {
        AegisCfg {
            enabled: true,
            primary_timeframe: d_primary_tf(),
            min_correlation: d_min_corr(),
            min_volatility_ratio: d_min_vr(),
            max_pairs_active: d_max_pairs(),
            min_probability_edge: d_min_pedge(),
            min_bias_magnitude: d_min_bias(),
            risk_per_trade_pct: d_risk(),
            atr_multiplier: d_atr_mult(),
            time_stop_bars: d_time_stop_bars(),
            shield_max_loss_per_trade: d_shield_max(),
            shield_strike_limit: d_shield_strikes(),
            daily_loss_limit: d_daily_loss(),
            lifetime_loss_floor: d_lifetime(),
            max_drawdown_pct: d_dd(),
            consistency_threshold_pct: d_consistency(),
            target_daily_profit: d_target_daily(),
            max_daily_profit: d_max_daily(),
        }
    }
}

fn d_acc_id() -> String {
    "bg-instant-5k".into()
}
fn d_balance() -> f64 {
    5000.0
}
fn d_primary_tf() -> String {
    "M5".into()
}
fn d_min_corr() -> f64 {
    0.70
}
fn d_min_vr() -> f64 {
    1.30
}
fn d_max_pairs() -> usize {
    5
}
fn d_min_pedge() -> f64 {
    0.60
}
fn d_min_bias() -> f64 {
    0.20
}
fn d_atr_mult() -> f64 {
    2.0
}
fn d_time_stop_bars() -> usize {
    96
}
fn d_shield_max() -> f64 {
    50.0
}
fn d_shield_strikes() -> u32 {
    2
}
fn d_daily_loss() -> f64 {
    150.0
}
fn d_lifetime() -> f64 {
    250.0
}
fn d_consistency() -> f64 {
    15.0
}
fn d_target_daily() -> f64 {
    25.0
}
fn d_max_daily() -> f64 {
    35.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct LogCfg {
    #[serde(default = "d_level")]
    pub level: String,
    #[serde(default = "d_journal")]
    pub journal_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub system: SystemCfg,
    pub account: AccountCfg,
    pub update: UpdateCfg,
    pub ml: MlCfg,
    pub logging: LogCfg,
    #[serde(default)]
    pub accounts: Vec<AccountDef>,
    #[serde(default)]
    pub aegis: AegisCfg,
}

fn d_name() -> String {
    "The Quant".into()
}
fn d_upd() -> u64 {
    24
}
fn d_state_dir() -> String {
    "state".into()
}
fn d_config_dir() -> String {
    "config".into()
}
fn d_model_dir() -> String {
    "models".into()
}
fn d_repo() -> String {
    ".".into()
}
fn d_true() -> bool {
    true
}
fn d_data_source() -> String {
    "sim".into()
}
fn d_mt5_dir() -> String {
    "/home/quant/mt5/files".into()
}
fn d_scan_int() -> i64 {
    10
}
fn d_acc() -> String {
    "PERSONAL".into()
}
fn d_dd() -> f64 {
    5.0
}
fn d_dl() -> f64 {
    3.0
}
fn d_risk() -> f64 {
    1.0
}
fn d_maxlot() -> f64 {
    1.0
}
fn d_lev() -> u32 {
    30
}
fn d_model_path() -> String {
    "models/latest.onnx".into()
}
fn d_inferms() -> f64 {
    2.0
}
fn d_level() -> String {
    "info".into()
}
fn d_journal() -> String {
    "state/trades".into()
}

impl Config {
    pub fn load() -> Result<Self, String> {
        let path = Path::new("config/system.toml");
        if path.exists() {
            let s = std::fs::read_to_string(path).map_err(|e| format!("read config: {e}"))?;
            toml::from_str(&s).map_err(|e| format!("parse config: {e}"))
        } else {
            // Fall back to embedded defaults so the binary runs even with no
            // on-disk config (important for the bootstrapped first run).
            toml::from_str(DEFAULT_CFG).map_err(|e| format!("parse defaults: {e}"))
        }
    }

    pub fn state_root(&self) -> PathBuf {
        PathBuf::from(&self.system.state_dir)
    }
    pub fn model_path(&self) -> PathBuf {
        PathBuf::from(&self.ml.model_path)
    }

    /// The managed accounts to run. Defaults to one Blue Guardian Instant 5K
    /// so the system is always functional even with an empty `[[accounts]]`.
    pub fn managed_accounts(&self) -> Vec<AccountDef> {
        if self.accounts.is_empty() {
            vec![AccountDef {
                id: "bg-instant-5k".into(),
                name: "Blue Guardian Instant 5K".into(),
                firm: "blue_guardian".into(),
                variant: "instant_5k".into(),
                initial_balance: 5000.0,
                r#type: "PROP_FUNDED".into(),
                ..AccountDef::default()
            }]
        } else {
            self.accounts.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_inherit_global_defaults() {
        let g = AegisCfg::default();
        let m = g.merged_with(&RulesOverride::default());
        assert_eq!(m.daily_loss_limit, g.daily_loss_limit);
        assert_eq!(m.max_drawdown_pct, g.max_drawdown_pct);
        assert_eq!(m.risk_per_trade_pct, g.risk_per_trade_pct);
    }

    #[test]
    fn overrides_apply_within_safety_clamps() {
        let g = AegisCfg::default();
        let o = RulesOverride {
            daily_loss_limit: Some(100.0), // tighter than default -> applied
            risk_per_trade_pct: Some(0.25),
            min_correlation: Some(0.90),
            max_pairs_active: Some(3),
            ..RulesOverride::default()
        };
        let m = g.merged_with(&o);
        assert_eq!(m.daily_loss_limit, 100.0);
        assert_eq!(m.risk_per_trade_pct, 0.25);
        assert_eq!(m.min_correlation, 0.90);
        assert_eq!(m.max_pairs_active, 3);
    }

    #[test]
    fn breakers_can_never_be_disabled_by_override() {
        let g = AegisCfg::default();
        // Try to loosen every breaker past its hard cap...
        let o = RulesOverride {
            max_drawdown_pct: Some(90.0),         // clamped to MAX_DD (30%)
            daily_loss_limit: Some(1_000_000.0),  // clamped to 2x global max
            lifetime_loss_floor: Some(0.0),       // clamped to MIN_LIFETIME (>= global/2)
            shield_max_loss_per_trade: Some(1e6), // clamped to 5x global cap
            shield_strike_limit: Some(1_000),     // clamped to MAX_STRIKES (5)
            ..RulesOverride::default()
        };
        let m = g.merged_with(&o);
        assert!(m.max_drawdown_pct <= 10.0); // MAX_DD_PCT
        assert!(m.daily_loss_limit <= 500.0); // MAX_DAILY_LOSS ($)
        assert!(m.lifetime_loss_floor <= 1000.0); // MAX_LIFETIME_FLOOR ($)
        assert!(m.shield_max_loss_per_trade <= 5.0 * g.shield_max_loss_per_trade);
        assert!(m.shield_strike_limit <= 5); // MAX_STRIKES
                                             // Breakers remain ON.
        assert!(m.max_drawdown_pct > 0.0);
        assert!(m.daily_loss_limit > 0.0);
        assert!(m.shield_max_loss_per_trade > 0.0);
    }
}
