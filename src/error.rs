//! # Error Handling Module
//!
//! Custom error hierarchy for The Quant system. Every error is typed, logged,
//! and handled. No unwrap() in production code.

use thiserror::Error;

/// Top-level error type for all Quant operations
#[derive(Error, Debug)]
pub enum QuantError {
    // === Connection Errors ===
    #[error("MT5 connection failed: {0}")]
    Mt5ConnectionError(String),

    #[error("MT5 heartbeat timeout after {0}s")]
    Mt5HeartbeatTimeout(u64),

    #[error("ZeroMQ error: {0}")]
    ZmqError(#[from] zmq::Error),

    // === Account & Risk Errors ===
    #[error("Account rule breach imminent: {rule}")]
    RuleBreachImminent { rule: String },

    #[error("Account rule breached: {rule} — {detail}")]
    RuleBreached { rule: String, detail: String },

    #[error("Account {account_id} is {status} — cannot trade")]
    AccountNotTradable { account_id: String, status: String },

    #[error("Position sizing error: {0}")]
    PositionSizingError(String),

    // === Memory Errors ===
    #[error("Memory limit exceeded: {used_pct:.1}% / {limit_pct:.1}%")]
    MemoryLimitExceeded { used_pct: f64, limit_pct: f64 },

    #[error("Emergency memory reduction triggered at {used_pct:.1}%")]
    EmergencyMemoryReduction { used_pct: f64 },

    #[error("Module {module} exceeded budget: {used_mb:.0}MB / {budget_mb:.0}MB")]
    ModuleBudgetExceeded { module: String, used_mb: f64, budget_mb: f64 },

    // === Model Errors ===
    #[error("Model inference failed: {0}")]
    InferenceError(String),

    #[error("Model {model_id} not found for asset {asset}")]
    ModelNotFound { model_id: String, asset: String },

    #[error("Model training failed: {0}")]
    TrainingError(String),

    #[error("Model hot-swap failed: {0}")]
    HotSwapError(String),

    // === Data Pipeline Errors ===
    #[error("Feature computation error: {0}")]
    FeatureError(String),

    #[error("OHLCV builder error: {0}")]
    OhlcvError(String),

    #[error("Data versioning error: {0}")]
    DataVersioningError(String),

    // === Database Errors ===
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    #[error("Database connection pool exhausted")]
    DatabasePoolExhausted,

    #[error("Migration error: {0}")]
    MigrationError(String),

    // === Security Errors ===
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Encryption error: {0}")]
    CryptoError(String),

    #[error("Session expired or invalid")]
    SessionExpired,

    #[error("Vault error: {0}")]
    VaultError(String),

    // === Configuration Errors ===
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Environment validation failed: {0}")]
    EnvValidationError(String),

    // === GitHub Errors ===
    #[error("GitHub sync failed: {0}")]
    GitHubError(String),

    #[error("Git operation failed: {0}")]
    GitError(String),

    // === Strategy & Lab Errors ===
    #[error("Strategy error: {0}")]
    StrategyError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Lab error: {0}")]
    LabError(String),

    // === Evolution Errors ===
    #[error("Evolution cycle failed at phase {phase}: {0}")]
    EvolutionError { phase: String, detail: String },

    // === TUI Errors ===
    #[error("TUI error: {0}")]
    TuiError(String),

    // === State & Update Errors (v4.0 "Hercules") ===
    #[error("State error: {0}")]
    StateError(String),

    #[error("State manifest verification failed: {0}")]
    ManifestError(String),

    #[error("Update error: {0}")]
    UpdateError(String),

    #[error("Blue-green handoff failed at phase {phase}: {0}")]
    HandoffError { phase: String, detail: String },

    #[error("Restore failed at step {step}: {0}")]
    RestoreError { step: String, detail: String },

    // === I/O Errors ===
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    // === Serialization Errors ===
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("TOML serialization error: {0}")]
    TomlError(#[from] toml::ser::Error),

    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),

    // === General Errors ===
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Channel closed: {0}")]
    ChannelClosed(String),
}

impl From<config::ConfigError> for QuantError {
    fn from(err: config::ConfigError) -> Self {
        QuantError::ConfigError(err.to_string())
    }
}

impl From<anyhow::Error> for QuantError {
    fn from(err: anyhow::Error) -> Self {
        QuantError::Internal(err.to_string())
    }
}

impl From<argon2::password_hash::Error> for QuantError {
    fn from(err: argon2::password_hash::Error) -> Self {
        QuantError::CryptoError(err.to_string())
    }
}

impl From<aes_gcm::Error> for QuantError {
    fn from(err: aes_gcm::Error) -> Self {
        QuantError::CryptoError(err.to_string())
    }
}

impl From<jsonwebtoken::errors::Error> for QuantError {
    fn from(err: jsonwebtoken::errors::Error) -> Self {
        QuantError::AuthenticationError(err.to_string())
    }
}

impl From<reqwest::Error> for QuantError {
    fn from(err: reqwest::Error) -> Self {
        QuantError::Internal(err.to_string())
    }
}

impl From<git2::Error> for QuantError {
    fn from(err: git2::Error) -> Self {
        QuantError::GitError(err.to_string())
    }
}

impl From<octocrab::Error> for QuantError {
    fn from(err: octocrab::Error) -> Self {
        QuantError::GitHubError(err.to_string())
    }
}

impl From<sysinfo::Error> for QuantError {
    fn from(err: sysinfo::Error) -> Self {
        QuantError::Internal(err.to_string())
    }
}

impl From<crossbeam_channel::RecvError> for QuantError {
    fn from(err: crossbeam_channel::RecvError) -> Self {
        QuantError::ChannelClosed(err.to_string())
    }
}

impl From<crossbeam_channel::SendError<std::vec::Vec<u8>>> for QuantError {
    fn from(err: crossbeam_channel::SendError<std::vec::Vec<u8>>) -> Self {
        QuantError::ChannelClosed(err.to_string())
    }
}

/// Result type alias for Quant operations
pub type QuantResult<T> = Result<T, QuantError>;
