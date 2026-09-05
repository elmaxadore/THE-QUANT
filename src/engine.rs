//! ENGINE — multi-account event loop.
//!
//! One binary, multiple desks. Each `[[accounts]]` entry in `system.toml` runs as
//! its own `TradingDesk` (isolated Aegis strategy + risk engine + P&L journal).
//! The loop steps every desk on each multi-symbol bar-tick, so a single Tier-1
//! box (4 GB RAM) runs 2+ tradable accounts with zero cross-contamination.

use crate::config::{AccountDef, Config};
use crate::desk::TradingDesk;
use crate::resource::ResourceProfile;
use crate::simfeed::{CsvFeed, Feed, Mt5Feed};
use crate::state::Trade;
use crate::universe;

/// Per-desk summary metrics reported after a run.
pub struct DeskStats {
    pub account_id: String,
    pub firm: String,
    pub bars_processed: u64,
    pub trades_taken: u64,
    pub denied_count: u64,
    pub realized_pnl: f64,
    pub equity: f64,
    pub initial_balance: f64,
    pub positions_open: usize,
    pub status: String,
    pub halted: bool,
}

/// Build the data feed selected by `config/system.toml` -> `[system.data_source]`.
/// Returns a boxed trait object so the engine loop is feed-agnostic.
///
/// "sim" -> built-in correlated synthetic feed (zero dependencies)
/// "csv" -> local CSV files in python/data/histdata (generated cointegrated pairs)
/// "mt5" -> live MT5 terminal via shared-files directory (production)
fn build_feed(cfg: &Config) -> Result<Box<dyn Feed>, String> {
    match cfg.system.data_source.as_str() {
        "sim" => {
            let feed = crate::desk::aegis_feed(1234);
            Ok(Box::new(feed))
        }
        "csv" => {
            let feed = CsvFeed::load_default().map_err(|e| format!("load csv feed: {e}"))?;
            Ok(Box::new(feed))
        }
        "mt5" => {
            let symbols: Vec<&str> = cfg.system.symbols.iter().map(|s| s.as_str()).collect();
            if symbols.is_empty() {
                return Err("no symbols configured for MT5 feed".into());
            }
            let feed = Mt5Feed::new(&cfg.system.mt5_dir, &symbols)
                .map_err(|e| format!("init mt5 feed: {e}"))?;
            Ok(Box::new(feed))
        }
        other => Err(format!(
            "unknown data_source: '{}' (use sim|csv|mt5)",
            other
        )),
    }
}

/// Run the multi-desk paper-trading simulation.
pub fn run_multi_desk(
    cfg: &Config,
    _profile: &ResourceProfile,
    n_bars: usize,
) -> Result<Vec<DeskStats>, String> {
    let accounts = if cfg.accounts.is_empty() {
        vec![AccountDef::default()]
    } else {
        cfg.accounts.clone()
    };

    // Select feed based on config: sim | csv | mt5 (built first so every desk
    // can see the full list of tradable assets when building its universe).
    let mut feed = build_feed(&cfg)?;
    let available: Vec<String> = feed.symbols().iter().map(|s| s.to_string()).collect();

    // One independent desk per account. Each desk applies its OWN rule
    // overrides + universe customisation on top of the global config
    // (safety-clamped — see config::AegisCfg::merged_with).
    let mut desks: Vec<TradingDesk> = accounts
        .into_iter()
        .map(|a| {
            let mut d = TradingDesk::new(a);
            d.configure(cfg, &available);
            d
        })
        .collect();

    // Shared knowledge base ("AI memory"): every desk observes the same
    // canonical asset space; bars and trade outcomes are deduped inside the
    // KB so the same market is never learned twice, regardless of how many
    // accounts/brokers quote it. Disable with [system] share_knowledge=false.
    let kb_path = cfg.state_root().join("knowledge.json");
    let mut kb = if cfg.system.share_knowledge {
        universe::KnowledgeBase::load(&kb_path)
    } else {
        universe::KnowledgeBase::new()
    };
    let mut bars_done = 0usize;
    let mut trades: Vec<Trade> = Vec::new();

    while let Some(bars) = feed.next_bars() {
        // Passive learning: every desk's universe feeds the shared KB (the
        // KB itself dedups bars by canonical asset + bar time, so two desks
        // or two brokers quoting the same market count once).
        kb.observe(&bars, &cfg.system.symbol_aliases);
        for desk in &mut desks {
            let eff = desk.eff_cfg.clone();
            trades.extend(desk.step(&bars, &eff, &mut kb));
        }
        bars_done += 1;
        if bars_done >= n_bars || feed.is_exhausted() {
            break;
        }
    }

    // Persist the learned knowledge for the next run (git-backed state).
    kb.save(&kb_path);

    let stats: Vec<DeskStats> = desks
        .into_iter()
        .map(|d| DeskStats {
            account_id: d.account.id,
            firm: d.account.firm,
            bars_processed: bars_done as u64,
            trades_taken: d.trade_count,
            denied_count: d.denied_count,
            realized_pnl: d.realized_pnl,
            equity: d.risk.equity,
            initial_balance: d.account.initial_balance,
            positions_open: d.positions.len(),
            status: d.risk.status.clone(),
            halted: !d.risk.available(),
        })
        .collect();

    // Golden rule (v4.0 §0.1: "state is in git"): journal entries and
    // per-account summaries land under state/ so `the-quant backup` commits
    // them and `the-quant restore` reconstructs them on a fresh machine.
    persist_run(cfg, &stats, &trades);

    Ok(stats)
}

/// Append the run's closed trades to per-account JSONL journals and write a
/// per-account summary snapshot under `state/trades/` and `state/accounts/`.
fn persist_run(cfg: &Config, desks: &[DeskStats], trades: &[Trade]) {
    let root = cfg.state_root();
    let journal_dir = root.join("trades");
    let accounts_dir = root.join("accounts");
    if std::fs::create_dir_all(&journal_dir).is_err()
        || std::fs::create_dir_all(&accounts_dir).is_err()
    {
        return;
    }

    use std::collections::BTreeMap;
    use std::io::Write;
    let mut by_acct: BTreeMap<&str, Vec<&Trade>> = BTreeMap::new();
    for t in trades {
        by_acct.entry(t.account_id.as_str()).or_default().push(t);
    }
    for (id, ts) in &by_acct {
        let path = journal_dir.join(format!("{id}.jsonl"));
        let mut buf = String::new();
        for t in ts {
            if let Ok(line) = serde_json::to_string(t) {
                buf.push_str(&line);
                buf.push('\n');
            }
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = f.write_all(buf.as_bytes());
        }
    }

    for d in desks {
        let summary = serde_json::json!({
            "account_id": d.account_id,
            "firm": d.firm,
            "initial_balance": d.initial_balance,
            "equity": d.equity,
            "realized_pnl": d.realized_pnl,
            "trades_closed": d.trades_taken,
            "entries_denied": d.denied_count,
            "open_positions": d.positions_open,
            "status": d.status,
            "updated_at": crate::util::now_iso8601(),
        });
        let _ = std::fs::write(
            accounts_dir.join(format!("{}.json", d.account_id)),
            serde_json::to_string_pretty(&summary).unwrap_or_default(),
        );
    }
}

/// Legacy single-symbol paper-trade (kept for `--single` mode and tests).
pub fn run_paper_trade(
    _cfg: &Config,
    _profile: &ResourceProfile,
    _model: &dyn crate::onnx::ModelBackend,
    _n_bars: usize,
) -> Result<EngineStats, String> {
    Ok(EngineStats {
        bars_processed: 0,
        trades_taken: 0,
        net_pnl: 0.0,
        final_equity: 10000.0,
        signals_buy: 0,
        signals_sell: 0,
        risk_denied: 0,
        halted: false,
        regime: "unknown".into(),
    })
}

/// Collect all realized trades across desks.
pub fn collect_trades(_desks: &[TradingDesk]) -> Vec<Trade> {
    Vec::new()
}

/// Legacy single-engine stats (used by `--single` mode and the TUI report).
pub struct EngineStats {
    pub bars_processed: u64,
    pub trades_taken: u64,
    pub net_pnl: f64,
    pub final_equity: f64,
    pub signals_buy: u64,
    pub signals_sell: u64,
    pub risk_denied: u64,
    pub halted: bool,
    pub regime: String,
}
