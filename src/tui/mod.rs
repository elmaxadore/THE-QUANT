//! # Terminal UI Interface (Layer 9)
//!
//! Optional GUI built with ratatui + crossterm. Provides a real-time dashboard
//! with account status, market regimes, open positions, system health, and a
//! command palette. Auto-locks after 5 minutes of inactivity.
//!
//! The TUI is optional — the daemon runs headless without it. Users can install
//! the TUI feature later via `cargo build --features tui`.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Dashboard panel identifiers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DashboardPanel {
    Accounts,
    MarketRegimes,
    OpenPositions,
    SystemStatus,
    MemoryDashboard,
    TrainingProgress,
}

/// Command palette action
#[derive(Debug, Clone)]
pub enum TuiCommand {
    Start(String),
    Stop(String),
    Train(String),
    LabStatus,
    LabRun,
    Evolve,
    Status,
    Config,
    Logs(String),
    Report,
    Backup,
    Memory,
    Exit,
}

/// TUI application state
#[derive(Debug)]
pub struct TuiState {
    pub active_panel: DashboardPanel,
    pub is_locked: bool,
    pub last_activity: DateTime<Utc>,
    pub auto_lock_minutes: u64,
    pub command_history: Vec<String>,
}

impl TuiState {
    pub fn new(auto_lock_minutes: u64) -> Self {
        Self {
            active_panel: DashboardPanel::Accounts,
            is_locked: false,
            last_activity: Utc::now(),
            auto_lock_minutes,
            command_history: Vec::new(),
        }
    }

    pub fn record_activity(&mut self) {
        self.last_activity = Utc::now();
        self.is_locked = false;
    }

    pub fn check_lock(&mut self) {
        let elapsed = (Utc::now() - self.last_activity).num_minutes();
        if elapsed >= self.auto_lock_minutes as i64 {
            self.is_locked = true;
        }
    }
}

/// TUI manager — handles the terminal interface lifecycle
#[derive(Debug)]
pub struct TuiManager {
    state: Arc<RwLock<TuiState>>,
    config: QuantConfig,
    is_running: Arc<RwLock<bool>>,
}

impl TuiManager {
    pub fn new(config: &QuantConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(TuiState::new(config.ui.auto_lock_minutes))),
            config: config.clone(),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the TUI event loop
    pub async fn run(&self) -> QuantResult<()> {
        if !self.config.ui.tui_enabled {
            info!("TUI disabled in configuration");
            return Ok(());
        }

        *self.is_running.write().await = true;
        info!("TUI started");

        // Main event loop
        while *self.is_running.read().await {
            self.tick().await?;
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(())
    }

    /// Single tick of the TUI event loop
    async fn tick(&self) -> QuantResult<()> {
        let mut state = self.state.write().await;
        state.check_lock();

        if state.is_locked {
            // Show lock screen
            return Ok(());
        }

        // TODO: Render dashboard panels using ratatui
        // This is a placeholder — actual rendering requires crossterm terminal setup
        // and ratatui Frame rendering with widgets

        Ok(())
    }

    /// Process a command from the palette
    pub async fn process_command(&self, command: TuiCommand) -> QuantResult<String> {
        let mut state = self.state.write().await;
        state.record_activity();

        let response = match command {
            TuiCommand::Start(account) => format!("Starting trading for account: {}", account),
            TuiCommand::Stop(account) => format!("Stopping trading for account: {}", account),
            TuiCommand::Train(asset) => format!("Triggering training for asset: {}", asset),
            TuiCommand::LabStatus => "Lab status: idle".into(),
            TuiCommand::LabRun => "Starting lab batch...".into(),
            TuiCommand::Evolve => "Triggering manual evolution...".into(),
            TuiCommand::Status => "System status: running".into(),
            TuiCommand::Config => "Opening configuration...".into(),
            TuiCommand::Logs(filter) => format!("Showing logs matching: {}", filter),
            TuiCommand::Report => "Generating performance report...".into(),
            TuiCommand::Backup => "Forcing GitHub sync...".into(),
            TuiCommand::Memory => "Memory: see memory dashboard".into(),
            TuiCommand::Exit => {
                *self.is_running.write().await = false;
                "Shutting down...".into()
            }
        };

        state.command_history.push(response.clone());
        Ok(response)
    }

    /// Stop the TUI
    pub async fn stop(&self) {
        *self.is_running.write().await = false;
        info!("TUI stopped");
    }

    /// Check if TUI is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get current state
    pub async fn get_state(&self) -> TuiState {
        self.state.read().await.clone()
    }
}

/// Render a memory usage bar (color-coded)
pub fn memory_bar(used_pct: f64) -> String {
    let bar_length = 20;
    let filled = (used_pct * bar_length as f64).round() as usize;
    let empty = bar_length - filled;

    let color = if used_pct < 0.5 {
        "\x1b[32m" // Green
    } else if used_pct < 0.75 {
        "\x1b[33m" // Yellow
    } else if used_pct < 0.9 {
        "\x1b[31m" // Red
    } else {
        "\x1b[35m" // Magenta
    };

    format!(
        "{}[{}{}]\x1b[0m {:.1}%",
        color,
        "█".repeat(filled),
        "░".repeat(empty),
        used_pct * 100.0
    )
}

/// Format a decimal value as a currency string
pub fn format_currency(value: rust_decimal::Decimal) -> String {
    format!("${:.2}", value)
}

/// Format a percentage
pub fn format_pct(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bar() {
        let bar = memory_bar(0.5);
        assert!(bar.contains("50.0%"));
        assert!(bar.contains("\x1b[33m")); // Yellow at 50%
    }

    #[test]
    fn test_memory_bar_green() {
        let bar = memory_bar(0.3);
        assert!(bar.contains("\x1b[32m")); // Green below 50%
    }

    #[test]
    fn test_memory_bar_red() {
        let bar = memory_bar(0.8);
        assert!(bar.contains("\x1b[31m")); // Red above 75%
    }

    #[test]
    fn test_memory_bar_magenta() {
        let bar = memory_bar(0.95);
        assert!(bar.contains("\x1b[35m")); // Magenta above 90%
    }

    #[test]
    fn test_format_currency() {
        let d = rust_decimal::Decimal::new(123456, 2);
        assert_eq!(format_currency(d), "$1234.56");
    }

    #[test]
    fn test_tui_state_lock() {
        let mut state = TuiState::new(1); // 1 minute auto-lock
        assert!(!state.is_locked);
        
        // Simulate 2 minutes passing
        state.last_activity = Utc::now() - chrono::Duration::minutes(2);
        state.check_lock();
        assert!(state.is_locked);
    }

    #[test]
    fn test_tui_state_activity_resets_lock() {
        let mut state = TuiState::new(1);
        state.last_activity = Utc::now() - chrono::Duration::minutes(2);
        state.check_lock();
        assert!(state.is_locked);
        
        state.record_activity();
        assert!(!state.is_locked);
    }
}
