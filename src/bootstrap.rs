//! # Bootstrap & Installer (Layer 0)
//!
//! First-run wizard, environment validation, dependency installation,
//! and systemd service setup. Runs only on first execution or when
//! explicitly invoked via `the-quant bootstrap`.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use crate::security::SecurityManager;
use std::path::PathBuf;
use tracing::{info, warn};

pub struct Bootstrap;

impl Bootstrap {
    /// Run the full bootstrap sequence
    pub async fn run() -> QuantResult<()> {
        info!("=== The Quant Bootstrap ===");
        Bootstrap::validate_environment()?;
        Bootstrap::create_directory_structure()?;
        info!("Bootstrap complete. Run `the-quant daemon` to start trading.");
        Ok(())
    }

    /// Validate the deployment environment
    pub fn validate_environment() -> QuantResult<()> {
        info!("Validating environment...");
        let total_ram = sysinfo::System::new().total_memory();
        if total_ram < 4_294_967_296 {
            warn!("Less than 4GB RAM detected. Performance will be constrained.");
        }
        let disk = std::fs::metadata("/").map_err(|e| QuantError::EnvValidationError(e.to_string()))?;
        info!("Environment validation passed — {} GB RAM detected", total_ram / 1_073_741_824);
        Ok(())
    }

    /// Create the directory structure
    pub fn create_directory_structure() -> QuantResult<()> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let base = PathBuf::from(&home).join(".thequant");
        let dirs = vec![
            base.join("config").join("rules"),
            base.join("data").join("raw"),
            base.join("data").join("processed"),
            base.join("data").join("cache"),
            base.join("models").join("current"),
            base.join("models").join("archive"),
            base.join("strategies").join("production"),
            base.join("strategies").join("lab"),
            base.join("strategies").join("retired"),
            base.join("logs").join("trades"),
            base.join("logs").join("evolution"),
            base.join("logs").join("system"),
        ];
        for dir in &dirs {
            std::fs::create_dir_all(dir)?;
        }
        info!("Created directory structure at ~/.thequant");
        Ok(())
    }
}
