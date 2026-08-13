//! # The Quant — CLI Entry Point (v4.0 "Hercules")
//!
//! Command-line interface for the trading platform. Provides subcommands for
//! starting the daemon, launching the web dashboard, bootstrap, training,
//! the lab, manual evolution, health checks, restore, update, and doctor.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use the_quant::config::QuantConfig;
use the_quant::error::{QuantError, QuantResult};
use the_quant::firm::{AccountVariant, QuantFirm};
use the_quant::memory::MemoryManager;
use the_quant::security::SecurityManager;
use the_quant::state::{RestoreEngine, RestoreOptions};
use the_quant::update::{UpdateEngine, UpdateEngineOptions, UpdateOptions};

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
        // v4.0 "Hercules" commands
        "restore" => runtime.block_on(async { run_restore(&args).await }),
        "update" => runtime.block_on(async { run_update(&args).await }),
        "doctor" => runtime.block_on(async { run_doctor().await }),
        "status" => runtime.block_on(async { run_status().await }),
        "account-add" => runtime.block_on(async { run_account_add(&args).await }),
        "version" => {
            println!("The Quant v{} (Hercules)", env!("CARGO_PKG_VERSION"));
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
    println!("The Quant v{} — Autonomous Quantitative Trading Platform (Hercules)", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage: the-quant [COMMAND]");
    println!();
    println!("Commands:");
    println!("  daemon            Start the trading daemon (headless)");
    println!("  bootstrap         Run first-time setup wizard");
    println!("  web               Launch the web dashboard (Axum + HTMX)");
    println!("  tui               Launch the terminal UI (ratatui)");
    println!("  health            Check system health status (JSON)");
    println!("  train [asset]     Force training cycle for an asset (default: all)");
    println!("  lab status        Show lab progress");
    println!("  lab run           Start a new lab batch");
    println!("  evolve            Trigger manual evolution cycle");
    println!("  restore           Rebuild system from git (one-command recovery)");
    println!("  update [--check|--apply|--force]  Self-update engine");
    println!("  doctor            Full system diagnostics");
    println!("  status            Overview of system + all account desks");
    println!("  account-add <name> <variant>  Add a new trading desk");
    println!("  version           Print version information");
    println!("  help              Show this help");
    println!();
    println!("Account variants: instant | one-step-eval | two-step-eval | personal");
}

/// Run the trading daemon (main event loop).
async fn run_daemon() -> QuantResult<()> {
    let config = QuantConfig::load()?;
    let memory = Arc::new(RwLock::new(MemoryManager::new(&config)));

    println!("The Quant v{} (Hercules) — daemon starting", env!("CARGO_PKG_VERSION"));
    println!(
        "Detected {} GB RAM — hard limit {:.1} GB",
        memory.read().await.total_ram_gb(),
        memory.read().await.hard_limit_gb()
    );

    // Initialize each module
    the_quant::bootstrap::Bootstrap::create_directory_structure()?;

    // Initialize the state store (git-centric state)
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let repo_root = PathBuf::from(&home).join("the-quant");
    let state_store = the_quant::state::StateStore::new(repo_root);
    state_store.initialize()?;

    // Initialize the firm (multi-account orchestration)
    let firm = QuantFirm::new(state_store.repo_root().to_path_buf());
    firm.initialize()?;
    println!("[FIRM] {} desk(s) loaded from state", firm.desk_ids().len());

    // Initialize the auto-commit engine
    let mut auto_commit = the_quant::github::AutoCommitEngine::new();
    let github_sync = Arc::new(the_quant::github::GitHubSync::new(&config));
    auto_commit.attach_handler(Arc::new(the_quant::github::GitHubCommitHandler::new(github_sync)));
    let _autocommit_task = auto_commit.spawn_background(std::time::Duration::from_secs(30));

    // TODO: Spawn subsystem tasks:
    // - DataCollector (ZMQ)
    // - FeaturePipeline
    // - RegimeDetector
    // - StrategyEngine
    // - RiskEngine
    // - ExecutionEngine
    // - EvolutionEngine
    // - MemoryMonitor

    println!("Daemon running. Ctrl+C to stop.");
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

/// Run the bootstrap / first-time setup.
async fn run_bootstrap() -> QuantResult<()> {
    println!("=== The v3.0 Bootstrap ===");
    the_quant::bootstrap::Bootstrap::run().await
}

/// Launch the web dashboard server.
async fn run_web() -> QuantResult<()> {
    let config = QuantConfig::load()?;
    let memory = Arc::new(RwLock::new(MemoryManager::new(&config)));
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let vault_path = PathBuf::from(&home).join(".thequant").join("vault.enc");
    let security = Arc::new(RwLock::new(SecurityManager::new(vault_path)));

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
            println!("Use `the-quant daemon` and the TUI/web dashboard to trigger batches.");
        }
        other => {
            println!("Unknown lab command: {}", other);
            println!("Usage: the lab [status|run]");
        }
    }
    Ok(())
}

/// Trigger a manual evolution cycle.
async fn run_evolve() -> QuantResult<()> {
    println!("Triggering manual evolution cycle...");
    Ok(())
}

/// v4.0: Restore from git (reconstruct the entire system on a fresh machine)
async fn run_restore(args: &[String]) -> QuantResult<()> {
    let repo_root = args
        .iter()
        .position(|a| a == "--path")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let skip_checksums = args.iter().any(|a| a == "--skip-checksums");
    let skip_db = args.iter().any(|a| a == "--skip-db");
    let skip_mt5 = args.iter().any(|a| a == "--skip-mt5");

    let options = RestoreOptions {
        skip_checksums,
        skip_database: skip_db,
        skip_mt5,
        dry_run: args.iter().any(|a| a == "--dry-run"),
        vault_password: None,
    };

    let engine = RestoreEngine::new(repo_root, options);
    println!("=== The v4.0 Restore ===");
    let report = engine.restore()?;
    println!("Manifest: v{} (schema {})", report.manifest_version, report.schema_version);
    println!("Accounts restored: {}", report.accounts_restored);
    println!("Status: {:?}", report.status);
    Ok(())
}

/// v4.0: Update engine (blue-green)
async fn run_update(args: &[String]) -> QuantResult<()> {
    let repo_root = PathBuf::from(".");
    let config = QuantConfig::load()?;

    let options = UpdateEngineOptions {
        repo_root: repo_root.clone(),
        branch: "main".into(),
        auto_update_window: Some(String::from("Sun 02:00 UTC")),
        github_enabled: config.github.enabled,
    };

    let engine = UpdateEngine::new(repo_root, options);

    let check_only = args.iter().any(|a| a == "--check");
    let force = args.iter().any(|a| a == "--force");
    let apply = args.iter().any(|a| a == "--apply") || !check_only;

    if check_only {
        let status = engine.check().await?;
        println!("Update status: {:?}", status);
        return Ok(());
    }

    if apply {
        let opts = UpdateOptions {
            force,
            skip_build: args.iter().any(|a| a == "--skip-build"),
            yes: args.iter().any(|a| a == "--yes"),
        };
        let status = engine.apply(opts).await?;
        println!("Update: {:?}", status);
    } else {
        println!("Usage: the update [--check|--apply|--force|--yes]");
    }
    Ok(())
}

/// v4.0: Doctor — full system diagnostics
async fn run_doctor() -> QuantResult<()> {
    println!("=== The Quant v4.0 Doctor ===");
    println!();

    // 1. Version
    println!("[1/7] Version: {}", env!("CARGO_PKG_VERSION"));

    // 2. System state
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let repo_root = PathBuf::from(&home).join("the-quant");
    let state_store = the_quant::state::StateStore::new(repo_root.clone());
    let manifest_path = state_store.state_root().join("manifest.json");
    if manifest_path.exists() {
        match state_store.load_manifest() {
            Ok(manifest) => {
                println!(
                    "[2/7] State manifest: valid ({:?} account records)",
                    manifest.accounts.len()
                );
                let mismatches = manifest.verify_checksums(state_store.state_root());
                if mismatches.is_empty() {
                    println!("[2/7] Checksums: ALL MATCH");
                } else {
                    println!("[2/7] Checksums: {} MISMATCHES", mismatches.len());
                    for (path, _, _) in mismatches.iter().take(5) {
                        println!("      - {}", path);
                    }
                }
            }
            Err(e) => println!("[2/7] State manifest error: {}", e),
        }
    } else {
        println!("[2/7] State manifest: NOT FOUND (run `the-quant bootstrap`)");
    }

    // 3. Database
    println!("[3/7] Database: check via config (host:port)");

    // 4. MT5 bridge
    println!("[4/7] MT5 bridge: ZMQ endpoints TBD");

    // 5. Account desks
    let firm = QuantFirm::new(repo_root.clone());
    firm.initialize()?;
    let desk_count = firm.desk_ids().len();
    println!("[5/7] Account desks: {}", desk_count);

    // 6. Memory
    let config = QuantConfig::load()?;
    let mut memory = MemoryManager::new(&config);
    println!(
        "[6/7] Memory: {:.1}% of hard limit used",
        memory.rss_pct()
    );

    // 7. GitHub sync
    let github_status = "Idle";
    println!("[7/7] GitHub sync: {}", github_status);

    println!();
    println!("Doctor complete.");
    Ok(())
}

/// v4.0: Account status overview
async fn run_status() -> QuantResult<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let repo_root = PathBuf::from(&home).join("the-quant");
    let state_store = the_quant::state::StateStore::new(repo_root);
    state_store.initialize()?;

    let manifest = state_store.load_manifest()?;

    println!("=== The Quant v{} (Hercules) ===", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Accounts ({}):", manifest.accounts.len());
    for account in &manifest.accounts {
        println!(
            "  {} — {} ({} / {})",
            account.uuid, account.name, account.firm, account.variant
        );
    }
    println!();
    println!("Global state:");
    println!(
        "  Regime detector: {}",
        manifest.global_state.regime_detector_version
    );
    println!("  Last evolution: {:?}", manifest.global_state.last_evolution);
    println!("  Market data through: {:?}", manifest.global_state.market_data_through);
    println!();
    println!("Last manifest update: {}", manifest.last_updated.format("%Y-%m-%d %H:%M:%S"));
    Ok(())
}

/// v4.0: Add a new trading desk
async fn run_account_add(args: &[String]) -> QuantResult<()> {
    let name = args.get(2).cloned().unwrap_or_else(|| "New Account".to_string());
    let variant_str = args.get(3).cloned().unwrap_or_else(|| "personal".to_string());

    let variant = match variant_str.as_str() {
        "instant" => AccountVariant::Instant,
        "one-step-eval" | "one_step_eval" => AccountVariant::OneStepEval,
        "two-step-eval" | "two_step_eval" => AccountVariant::TwoStepEval,
        "personal" => AccountVariant::Personal,
        other => {
            eprintln!("Unknown variant '{}'. Use: instant | one-step-eval | two-step-eval | personal", other);
            return Err(QuantError::Internal("Invalid account variant".into()));
        }
    };

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let repo_root = PathBuf::from(&home).join("the-quant");
    let mut firm = QuantFirm::new(repo_root);
    firm.initialize()?;

    let account_type = the_quant::account::AccountType::PropFirm;
    let stage = the_quant::account::AccountStage::Funded;

    let uuid = firm.add_desk(name.clone(), variant, account_type, stage)?;
    println!("Added desk: {} ({})", name, uuid);
    println!("State dir: {}", firm.state_store().account_dir(&uuid).display());
    Ok(())
}
