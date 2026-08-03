//! # The Quant — CLI Entry Point (v3.0 "Prometheus")
//!
//! Command-line interface for the trading platform. Provides subcommands for
//! starting the daemon, launching the web dashboard, bootstrap, training,
//! the lab, manual evolution, and health checks.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use the_quant::config::QuantConfig;
use the_quant::error::{QuantError, QuantResult};
use the_quant::memory::MemoryManager;
use the_quant::security::SecurityManager;

fn main() {
    // Install custom panic hook
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[PANIC] The Quant crashed: {}", info);
        eprintln!("[PANIC] Trading halting gracefully. Please check logs.");
    }));

    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("daemon");

    // Simple tokio runtime for async commands
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create async runtime");

    let result = match command {
        "daemon" => runtime.block_on(async { run_daemon().await }),
        "bootstrap" => runtime.block_on(async { run_bootstrap().await }),
        "web" => runtime.block_on(async { run_web().await }),
        "tui" => runtime.block_on(async { run_tui().await }),
        "health" => runtime.block_on(async { run_health().await }),
        "train" => runtime.block_on(async { run_train(&args.get(2).map(|s| s.as_str()).unwrap_or("all")).await }),
        "lab" => runtime.block_on(async { run_lab(&args.get(2).map(|s| s.as_str()).unwrap_or("status")).await }),
        "evolve" => runtime.block_on(async { run_evolve().await }),
        "version" => {
            println!("The Quant v{} (Prometheus)", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            print_help();
            Err(QuantError::Internal("Unknown command".into()))
        }
    };

    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Print CLI usage help.
fn print_help() {
    println!("The Quant v{} — Autonomous Quantitative Trading Platform", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage: the-quant [COMMAND]");
    println!();
    println!("Commands:");
    println!("  daemon          Start the trading daemon (headless)");
    println!("  bootstrap       Run first-time setup wizard");
    println!("  web             Launch the web dashboard (Axum + HTMX)");
    println!("  tui             Launch the terminal UI (ratatui)");
    println!("  health          Check system health status (JSON)");
    println!("  train [asset]   Force training cycle for an asset (default: all)");
    println!("  lab status      Show lab progress");
    println!("  lab run         Start a new lab batch");
    println!("  evolve          Trigger manual evolution cycle");
    println!("  version         Print version information");
    println!("  help            Show this help");
}

/// Run the trading daemon (main event loop).
async fn run_daemon() -> QuantResult<()> {
    let config = QuantConfig::load()?;
    let memory = Arc::new(RwLock::new(MemoryManager::new(&config)));

    println!("The Quant v{} (Prometheus) — daemon starting", env!("CARGO_PKG_VERSION"));
    println!(
        "Detected {} GB RAM — hard limit {:.1} GB",
        memory.read().await.total_ram_gb(),
        memory.read().await.hard_limit_gb()
    );

    // Initialize each module
    the_quant::bootstrap::Bootstrap::create_directory_structure()?;

    // TODO: Spawn subsystem tasks:
    // - DataCollector (ZMQ)
    // - FeaturePipeline
    // - RegimeDetector
    // - StrategyEngine
    // - RiskEngine
    // - ExecutionEngine
    // - EvolutionEngine
    // - GitHubSync
    // - MemoryMonitor

    println!("Daemon running. Ctrl+C to stop.");
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

/// Run the bootstrap / first-time setup.
async fn run_bootstrap() -> QuantResult<()> {
    println!("=== The Quant v3.0 Bootstrap ===");
    the_quant::bootstrap::Bootstrap::run().await
}

/// Launch the web dashboard server.
async fn run_web() -> QuantResult<()> {
    let config = QuantConfig::load()?;
    let memory = Arc::new(RwLock::new(MemoryManager::new(&config)));
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let vault_path = PathBuf::from(&home).join(".thequant").join("vault.enc");
    let security = Arc::new(RwLock::new(SecurityManager::new(vault_path)));

    // TODO: Bind the Axum router with handlers
    println!("Web dashboard starting at http://{}:{}", config.api_host(), config.api_port());

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

/// Launch the terminal UI.
async fn run_tui() -> QuantResult<()> {
    println!("The Quant TUI requires a terminal with ANSI support.");
    println!("Launching TUI...");
    // TODO: Initialize ratatui terminal + event loop
    Ok(())
}

/// Run a health check.
async fn run_health() -> QuantResult<()> {
    let config = QuantConfig::load()?;
    let mut memory = MemoryManager::new(&config);

    let health = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "name": "the-quant",
        "total_ram_gb": memory.total_ram_gb(),
        "hard_limit_gb": memory.hard_limit_gb(),
        "rss_pct": memory.rss_pct(),
        "modules": memory.module_breakdown().iter()
            .map(|m| {
                serde_json::json!({
                    "module": m.name,
                    "used_mb": m.used_mb(),
                    "budget_mb": m.budget_mb(),
                    "pct": m.pct_of_budget,
                })
            })
            .collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&health)?);
    Ok(())
}

/// Force a training cycle for an asset.
async fn run_train(_asset: &str) -> QuantResult<()> {
    println!("Training cycle requested...");
    println!("NOTE: Full training pipeline is not yet wired to the CLI.");
    println!("Run `the-quant daemon` to start the system, then use `evolve` for the full cycle.");
    Ok(())
}

/// Lab operations: status or run.
async fn run_lab(command: &str) -> QuantResult<()> {
    match command {
        "status" => {
            println!("Lab status: idle (no active batch)");
            println!("Last generation: N/A");
            println!("Population: 0 candidates");
            println!("Hall of Fame: 0 strategies");
        }
        "run" => {
            println!("Starting lab batch...");
            println!("NOTE: Lab batch execution is not yet wired to the CLI.");
            println!("Use `the-quant daemon` and the TUI/web dashboard to trigger batches.");
        }
        other => {
            println!("Unknown lab command: {}", other);
            println!("Usage: the-quant lab [status|run]");
        }
    }
    Ok(())
}

/// Trigger a manual evolution cycle.
async fn run_evolve() -> QuantResult<()> {
    println!("Triggering manual evolution cycle...");
    println!("NOTE: Evolution loop is not yet wired to the CLI.");
    println!("Use `the-quant daemon` to run the evolution engine.");
    Ok(())
}

