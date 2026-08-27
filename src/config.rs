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
    #[serde(default)]
    pub symbols: Vec<String>,
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
}

fn d_name() -> String { "The Quant".into() }
fn d_upd() -> u64 { 24 }
fn d_state_dir() -> String { "state".into() }
fn d_config_dir() -> String { "config".into() }
fn d_model_dir() -> String { "models".into() }
fn d_repo() -> String { ".".into() }
fn d_true() -> bool { true }
fn d_acc() -> String { "PERSONAL".into() }
fn d_dd() -> f64 { 5.0 }
fn d_dl() -> f64 { 3.0 }
fn d_risk() -> f64 { 1.0 }
fn d_maxlot() -> f64 { 1.0 }
fn d_lev() -> u32 { 30 }
fn d_model_path() -> String { "models/latest.onnx".into() }
fn d_inferms() -> f64 { 2.0 }
fn d_level() -> String { "info".into() }
fn d_journal() -> String { "state/trades".into() }

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
}