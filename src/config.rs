//! # Configuration Module
//!
//! Hierarchical configuration loaded from TOML files and environment variables.
//! The system.toml file defines all tunable parameters including memory budgets,
//! channel capacities, and module-specific settings.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::info;

/// Top-level system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantConfig {
    /// System identification
    pub system: SystemConfig,
    /// Memory management (percentage-scaled)
    pub memory: MemoryConfig,
    /// MT5 bridge configuration
    pub mt5: Mt5Config,
    /// PostgreSQL database configuration
    pub database: DatabaseConfig,
    /// Account management settings
    pub account: AccountConfig,
    /// Risk management parameters
    pub risk: RiskConfig,
    /// Strategy engine settings
    pub strategy: StrategyConfig,
    /// Evolution loop configuration
    pub evolution: EvolutionConfig,
    /// Lab (strategy laboratory) configuration
    pub lab: LabConfig,
    /// Model manager settings
    pub model: ModelConfig,
    /// GitHub integration configuration
    pub github: GitHubConfig,
    /// TUI/CLI configuration
    pub ui: UiConfig,
    /// Data pipeline settings
    pub data: DataConfig,
    /// Security configuration
    pub security: SecurityConfig,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            memory: MemoryConfig::default(),
            mt5: Mt5Config::default(),
            database: DatabaseConfig::default(),
            account: AccountConfig::default(),
            risk: RiskConfig::default(),
            strategy: StrategyConfig::default(),
            evolution: EvolutionConfig::default(),
            lab: LabConfig::default(),
            model: ModelConfig::default(),
            github: GitHubConfig::default(),
            ui: UiConfig::default(),
            data: DataConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    /// Data directory (~/.thequant by default)
    pub data_dir: PathBuf,
    /// Config directory
    pub config_dir: PathBuf,
    /// Log directory
    pub log_dir: PathBuf,
    /// Number of CPU cores to reserve for OS
    pub reserved_cpus: usize,
    /// Worker thread ratio (applied to available CPUs)
    pub worker_thread_ratio: f64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let base = PathBuf::from(&home).join(".thequant");
        Self {
            data_dir: base.join("data"),
            config_dir: base.join("config"),
            log_dir: base.join("logs"),
            reserved_cpus: 1,
            worker_thread_ratio: 0.8,
        }
    }
}

/// Percentage-scaled memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Percentage of total system RAM that The Quant may use (default: 85%)
    pub process_cap_pct: f64,
    /// Soft limit as percentage of HARD_PROCESS_LIMIT (default: 75%)
    pub soft_limit_pct: f64,
    /// Hard limit as percentage of HARD_PROCESS_LIMIT (default: 90%)
    pub hard_limit_pct: f64,
    /// Emergency reduction threshold (default: 90%)
    pub emergency_pct: f64,
    /// Module budget allocations (percentages of HARD_PROCESS_LIMIT)
    pub module_budgets: ModuleBudgets,
    /// Channel base capacities
    pub channel_base_capacity: usize,
    /// Ring buffer depth base
    pub ring_buffer_base: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            process_cap_pct: 85.0,
            soft_limit_pct: 75.0,
            hard_limit_pct: 90.0,
            emergency_pct: 90.0,
            module_budgets: ModuleBudgets::default(),
            channel_base_capacity: 1024,
            ring_buffer_base: 1000,
        }
    }
}

/// Per-module memory budgets as percentages of HARD_PROCESS_LIMIT
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModuleBudgets {
    pub data_collector: f64,       // 25%
    pub feature_pipeline: f64,     // 12%
    pub model_manager: f64,        // 15%
    pub strategy_engine: f64,      // 6%
    pub risk_engine: f64,          // 3%
    pub lab: f64,                  // 25%
    pub tui: f64,                  // 3%
    pub system_overhead: f64,      // 6%
    pub reserve: f64,              // 5%
}

impl Default for ModuleBudgets {
    fn default() -> Self {
        Self {
            data_collector: 25.0,
            feature_pipeline: 12.0,
            model_manager: 15.0,
            strategy_engine: 6.0,
            risk_engine: 3.0,
            lab: 25.0,
            tui: 3.0,
            system_overhead: 6.0,
            reserve: 5.0,
        }
    }
}

/// MT5 ZeroMQ bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Mt5Config {
    pub zmq_pub_endpoint: String,
    pub zmq_sub_endpoint: String,
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub reconnect_backoff: Vec<u64>,
    pub symbols: Vec<String>,
    pub timeframes: Vec<String>,
    pub max_buffered_ticks: usize,
}

impl Default for Mt5Config {
    fn default() -> Self {
        Self {
            zmq_pub_endpoint: "tcp://127.0.0.1:5555".to_string(),
            zmq_sub_endpoint: "tcp://127.0.0.1:5556".to_string(),
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_ms: 5000,
            reconnect_backoff: vec![1, 2, 5, 10, 30],
            symbols: vec!["EURUSD".into(), "GBPUSD".into(), "XAUUSD".into()],
            timeframes: vec!["M1".into(), "M5".into(), "M15".into(), "H1".into(), "H4".into(), "D1".into()],
            max_buffered_ticks: 1000,
        }
    }
}

/// PostgreSQL database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
    pub pool_size: u32,
    pub ssl_mode: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 5432,
            database: "thequant".to_string(),
            username: "quant".to_string(),
            password: "".to_string(),
            max_connections: 10,
            pool_size: 5,
            ssl_mode: "prefer".to_string(),
        }
    }
}

/// Account management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AccountConfig {
    pub risk_per_trade_pct: f64,
    pub max_correlated_group_exposure_pct: f64,
    pub default_daily_loss_limit_pct: f64,
    pub default_max_drawdown_pct: f64,
}

impl Default for AccountConfig {
    fn default() -> Self {
        Self {
            risk_per_trade_pct: 0.5,
            max_correlated_group_exposure_pct: 2.0,
            default_daily_loss_limit_pct: 2.0,
            default_max_drawdown_pct: 10.0,
        }
    }
}

/// Risk management parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    pub max_leverage: u32,
    pub max_lot_size: f64,
    pub default_slippage_tolerance_atr: f64,
    pub correlation_window_days: u32,
    pub max_correlation_threshold: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_leverage: 50,
            max_lot_size: 10.0,
            default_slippage_tolerance_atr: 0.5,
            correlation_window_days: 20,
            max_correlation_threshold: 0.7,
        }
    }
}

/// Strategy engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategyConfig {
    pub min_signal_quality: f64,
    pub signal_debounce_seconds: u64,
    pub max_active_strategies: usize,
    pub strategy_correlation_threshold: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_signal_quality: 0.5,
            signal_debounce_seconds: 30,
            max_active_strategies: 50,
            strategy_correlation_threshold: 0.7,
        }
    }
}

/// Evolution loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvolutionConfig {
    pub min_trades_before_evolution: u32,
    pub max_days_between_evolutions: u32,
    pub performance_decay_threshold: f64,
    pub regime_instability_threshold: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            min_trades_before_evolution: 20,
            max_days_between_evolutions: 7,
            performance_decay_threshold: 0.5,
            regime_instability_threshold: 0.6,
        }
    }
}

/// Lab configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LabConfig {
    pub population_size: usize,
    pub max_generations: u32,
    pub validation_split_pct: f64,
    pub backtest_monte_carlo_iterations: u32,
    pub min_backtest_months: u32,
    pub promotion_sharpe_threshold: f64,
    pub promotion_walkforward_efficiency: f64,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            population_size: 1000,
            max_generations: 20,
            validation_split_pct: 0.15,
            backtest_monte_carlo_iterations: 10_000,
            min_backtest_months: 24,
            promotion_sharpe_threshold: 1.0,
            promotion_walkforward_efficiency: 0.7,
        }
    }
}

/// Model manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub inference_timeout_ms: u64,
    pub stale_prediction_max_seconds: u64,
    pub gmm_max_components: usize,
    pub gmm_retrain_interval_days: u32,
    pub gbdt_max_trees: u32,
    pub gbdt_max_depth: u32,
    pub gbdt_num_leaves: u32,
    pub max_models_per_asset: usize,
    pub hot_swap_enabled: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            inference_timeout_ms: 50,
            stale_prediction_max_seconds: 30,
            gmm_max_components: 8,
            gmm_retrain_interval_days: 7,
            gbdt_max_trees: 1000,
            gbdt_max_depth: 6,
            gbdt_num_leaves: 31,
            max_models_per_asset: 5,
            hot_swap_enabled: true,
        }
    }
}

/// GitHub integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitHubConfig {
    pub enabled: bool,
    pub repo_url: String,
    pub private_repo: bool,
    pub auto_commit: bool,
    pub auto_commit_interval_mins: u32,
    pub pat: String,
    pub branch: String,
}

impl Default for GitHubConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            repo_url: String::new(),
            private_repo: true,
            auto_commit: true,
            auto_commit_interval_mins: 60,
            pat: String::new(),
            branch: "main".to_string(),
        }
    }
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub tui_enabled: bool,
    pub refresh_interval_ms: u64,
    pub auto_lock_minutes: u64,
    pub color_scheme: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            tui_enabled: true,
            refresh_interval_ms: 1000,
            auto_lock_minutes: 5,
            color_scheme: "dark".to_string(),
        }
    }
}

/// Data pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    pub tick_buffer_size: usize,
    pub ohlcv_buffer_capacity: usize,
    pub feature_cache_size_items: usize,
    pub parquet_compression: String,
    pub raw_data_retention_days: u32,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            tick_buffer_size: 10_000,
            ohlcv_buffer_capacity: 10_000,
            feature_cache_size_items: 1000,
            parquet_compression: "zstd".to_string(),
            raw_data_retention_days: 7,
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub argon2_time_cost: u32,
    pub argon2_memory_cost_kib: u32,
    pub argon2_parallelism: u32,
    pub jwt_expiry_hours: u32,
    pub session_inactivity_lock_minutes: u32,
    pub vault_path: PathBuf,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        Self {
            argon2_time_cost: 3,
            argon2_memory_cost_kib: 65536,
            argon2_parallelism: 4,
            jwt_expiry_hours: 1,
            session_inactivity_lock_minutes: 5,
            vault_path: PathBuf::from(&home).join(".thequant").join("vault.enc"),
        }
    }
}

impl QuantConfig {
    /// Load configuration from system.toml, with environment variable overrides
    pub fn load() -> Result<Self, crate::QuantError> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let config_path = PathBuf::from(&home).join(".thequant").join("config").join("system.toml");

        let config = if config_path.exists() {
            info!("Loading config from {:?}", config_path);
            config::Config::builder()
                .add_source(config::File::from(config_path.as_path()))
                .add_source(config::Environment::with_prefix("QUANT").separator("_"))
                .build()?
                .try_deserialize()?
        } else {
            info!("No system.toml found, using default configuration");
            QuantConfig::default()
        };

        Ok(config)
    }

    /// Load configuration from a specific path
    pub fn load_from(path: &Path) -> Result<Self, crate::QuantError> {
        let config: QuantConfig = config::Config::builder()
            .add_source(config::File::from(path))
            .add_source(config::Environment::with_prefix("QUANT").separator("_"))
            .build()?
            .try_deserialize()?;
        Ok(config)
    }

    /// Save current configuration to default path
    pub fn save(&self) -> Result<(), crate::QuantError> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let config_dir = PathBuf::from(&home).join(".thequant").join("config");
        std::fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join("system.toml");
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, toml_string)?;
        tracing::info!("Configuration saved to {:?}", config_path);
        Ok(())
    }
}
