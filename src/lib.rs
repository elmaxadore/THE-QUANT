//! # The Quant — Autonomous Quantitative Trading Platform
//!
//! A percentage-scaled, zero-copy quantitative trading system written entirely in Rust.
//! Deploys on Linux VPS, communicates with MetaTrader 5 via ZeroMQ bridge,
//! persists state to PostgreSQL, and runs an endless evolution loop.
//!
//! ## Architecture
//!
//! - **Layer 0**: Bootstrap & Installer
//! - **Layer 1**: Security & Authentication
//! - **Layer 2**: Account Management
//! - **Layer 3**: Data Pipeline (Data Collector)
//! - **Layer 4**: Market Regime Detection
//! - **Layer 5**: Model Manager
//! - **Layer 6**: Strategy Laboratory
//! - **Layer 7**: Risk Management & Execution
//! - **Layer 8**: Evolution Loop
//! - **Layer 9**: CLI/TUI Interface
//! - **Layer 10**: PostgreSQL Schema (TimescaleDB)
//! - **Layer 11**: GitHub Integration

pub mod config;
pub mod error;
pub mod memory;
pub mod bootstrap;
pub mod security;
pub mod account;
pub mod data;
pub mod regime;
pub mod model;
pub mod strategy;
pub mod lab;
pub mod risk;
pub mod execution;
pub mod evolution;
pub mod github;
pub mod tui;

// Re-export commonly used types at crate level
pub use config::QuantConfig;
pub use error::QuantError;
pub use memory::MemoryManager;

use tracing::{info, warn};
use std::sync::Arc;
use tokio::sync::RwLock;

/// The core application state shared across all modules
#[derive(Debug)]
pub struct QuantApp {
    pub config: QuantConfig,
    pub memory_manager: Arc<RwLock<MemoryManager>>,
    pub running: Arc<std::sync::atomic::AtomicBool>,
}

impl QuantApp {
    /// Create a new application instance, loading config and initializing the memory manager
    pub async fn new() -> Result<Self, QuantError> {
        let config = QuantConfig::load()?;
        let memory_manager = Arc::new(RwLock::new(MemoryManager::new(&config)));
        
        info!(
            "The Quant v{} initialized — detected {} GB RAM, using {} GB",
            env!("CARGO_PKG_VERSION"),
            memory_manager.read().await.total_ram_gb(),
            memory_manager.read().await.hard_limit_gb()
        );

        Ok(Self {
            config,
            memory_manager,
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Start all subsystems and run the main event loop
    pub async fn run(&self) -> Result<(), QuantError> {
        info!("The Quant is starting all subsystems...");

        // TODO: spawn all subsystem tasks
        // - Data Collector (ZeroMQ listener)
        // - Feature Pipeline
        // - Strategy Engine
        // - Evolution Engine
        // - TUI Dashboard
        // - GitHub Sync
        // - Health Monitor

        while self.running.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            // Memory monitoring every 5 seconds
            // Check resource budgets and trigger backpressure if needed
        }

        Ok(())
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) {
        info!("Shutting down The Quant gracefully...");
        self.running.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
