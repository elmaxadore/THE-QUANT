//! THE QUANT — entry point.
//!
//! Usage:
//!   the-quant paper [bars]      run a paper-trading simulation
//!   the-quant backup            commit + push state to origin
//!   the-quant update --now      force an auto-update check immediately
//!   the-quant --status          print system + account status
//!   the-quant --smoke           minimal smoke test (used by auto-update)
//!   the-quant --help            this help
//!
//! The daemon (systemd) runs `the-quant daemon` which starts the update
//! scheduler loop in the background.

mod config;
mod engine;
mod execution;
mod features;
mod github;
mod onnx;
mod regime;
mod resource;
mod risk;
mod security;
mod simfeed;
mod state;
mod strategy;
mod update;
mod util;

mod tui;
#[cfg(feature = "web")]
mod web;

use config::Config;
use onnx::ModelBackend;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verb = args.first().map(|s| s.as_str()).unwrap_or("paper");

    if verb == "--help" || verb == "-h" || verb == "help" {
        print_help();
        return;
    }

    let cfg = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(2);
        }
    };

    let profile = resource::ResourceProfile::detect();

    match verb {
        "--smoke" => {
            // Lightweight boot sanity check used by the auto-update engine.
            println!("the-quant smoke ok (v4.0 hybrid)");
        }
        "--status" => {
            print_status(&cfg, &profile);
        }
        "paper" => {
            let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
            let model = onnx::load_backend(&cfg).unwrap_or_else(|e| {
                eprintln!("model load failed ({e}); falling back to rule model");
                Box::new(onnx::RuleModel::new())
            });
            run_paper(&cfg, &profile, &*model, n);
        }
        "backup" | "--backup" => {
            backup(&cfg);
        }
        "update" => {
            update_now(&cfg);
        }
        "restore" | "--restore" => {
            let r = state::Restorer::new(&cfg);
            match r.restore() {
                Ok(()) => println!("state restored/reconstructed under {}/", cfg.system.state_dir),
                Err(e) => eprintln!("restore failed: {e}"),
            }
        }
        "daemon_run" => {
            let p2 = profile.clone();
            daemon(&cfg, &p2);
        }
        "vault" => {
            vault_cmd(&cfg, &args);
        }
        other => {
            eprintln!("unknown command: {other}");
            print_help();
        }
    }
}

fn print_help() {
    println!(
        "THE QUANT — hybrid self-scaling self-updating quant desk\n\
         \n\
         commands:\n\
         \x20 the-quant paper [N]       run N-bar paper simulation\n\
         \x20 the-quant --status         show system + account status\n\
         \x20 the-quant backup           commit+push state to git\n\
         \x20 the-quant update --now     check git and apply update now\n\
         \x20 the-quant restore          recreate state dirs on a fresh machine\n\
         \x20 the-quant vault init       create the encrypted master vault\n\
         \x20 the-quant vault check      verify the master password\n\
         \x20 the-quant vault status     show vault state\n\
         \x20 the-quant daemon           run scheduler (systemd)\n\
         \x20 the-quant --smoke          self-check used by auto-update"
    );
}

fn print_status(cfg: &Config, profile: &resource::ResourceProfile) {
    println!("{}", profile.summary());
    println!("account type: {}  account.drawdown cap {}%  risk/trade {}%",
             cfg.account.r#type, cfg.account.max_drawdown_pct, cfg.account.risk_per_trade_pct);
    println!("state dir: {} (git sync backend)", cfg.system.state_dir);
    let upd = update::AutoUpdater::new(cfg.clone());
    match &upd.git {
        Some(g) => match g.current_head() {
            Ok(h) => println!("repo HEAD: {h}"),
            Err(e) => println!("repo HEAD unavailable: {e}"),
        },
        None => println!("auto-update: disabled (no git repo found)"),
    }
    println!("update interval: {}h", cfg.system.update_interval_hours);
    println!("ml: model={} enabled={}", cfg.ml.model_path, cfg.ml.enabled);
}

fn run_paper(cfg: &Config, profile: &resource::ResourceProfile, model: &dyn ModelBackend, n: usize) {
    println!("[the-quant] booted {} ({})", profile.tier, model.name());
    let stats = engine::run_paper_trade(cfg, profile, model, n).unwrap_or_else(|e| {
        eprintln!("paper trade error: {e}");
        std::process::exit(1);
    });
    tui::report(cfg, &stats);
}

fn backup(cfg: &Config) {
    match github::GitSync::discover(cfg) {
        Ok(g) => match g.commit_state("state backup") {
            Ok(true) => {
                println!("committed state");
                let _ = g.push();
            }
            Ok(false) => println!("nothing to commit"),
            Err(e) => eprintln!("commit failed: {e}"),
        },
        Err(e) => eprintln!("no git repo: {e}"),
    }
}

fn vault_cmd(cfg: &Config, args: &[String]) {
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let vault = security::Vault::new(cfg);
    match sub {
        "init" => {
            // `the-quant vault init` — prompt for a master password and create
            // the encrypted vault. Reads from TTY to avoid echo of the password.
            eprint!("New master password: ");
            let pw = rpassword_prompt();
            if pw.len() < 8 {
                eprintln!("error: master password must be at least 8 characters");
                std::process::exit(1);
            }
            match vault.init(&pw) {
                Ok(_) => println!("vault created at {}", vault.path.display()),
                Err(e) => eprintln!("vault init failed: {e}"),
            }
        }
        "check" => {
            eprint!("Master password: ");
            let pw = rpassword_prompt();
            match vault.open(&pw) {
                Ok(_) => println!("vault unlocked ok"),
                Err(e) => eprintln!("vault check failed: {e}"),
            }
        }
        "status" => {
            if vault.path.exists() {
                println!("vault: present ({})", vault.path.display());
            } else {
                println!("vault: not initialised — run `the-quant vault init`");
            }
        }
        _ => {
            println!("vault commands:\n  the-quant vault init     create the encrypted vault\n  the-quant vault check    verify the master password\n  the-quant vault status   show vault state");
        }
    }
}

/// Read the master password. Hidden input when attached to a TTY; a plain
/// stdin read otherwise (so scripted/piped use never blocks on a terminal).
fn rpassword_prompt() -> String {
    #[cfg(feature = "crypto")]
    {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            if let Ok(pw) = rpassword::read_password() {
                return pw.trim_end().to_string();
            }
        }
    }
    let mut pw = String::new();
    let _ = std::io::stdin().read_line(&mut pw);
    pw.trim_end().to_string()
}

fn update_now(cfg: &Config) {
    let mut store = state::Store::new(cfg).expect("store");
    let upd = update::AutoUpdater::new(cfg.clone());
    let _ = upd.check_now(&mut store);
}

fn daemon(cfg: &Config, _profile: &resource::ResourceProfile) {
    let mut store = state::Store::new(cfg).expect("store");
    let upd = update::AutoUpdater::new(cfg.clone());
    let stop = Arc::new(AtomicBool::new(false));
    eprintln!("[the-quant] daemon: update scheduler every {}h", upd.interval_hours);
    upd.scheduler_loop(&mut store, stop);
}
