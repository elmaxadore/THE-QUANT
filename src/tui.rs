//! DASHBOARD — a lightweight ANSI report (the lean default).
//!
//! The `tui` cargo feature can upgrade this to a full ratatui interactive UI;
//! the default build uses a clean, dependency-free textual dashboard so it runs
//! on any Tier-1 machine and inside SSH.

use crate::engine::EngineStats;

/// Print the end-of-run report using ANSI colours.
pub fn report(cfg: &crate::config::Config, stats: &EngineStats) {
    let start_balance = 10_000.0;
    let ret = if start_balance > 0.0 { stats.net_pnl / start_balance * 100.0 } else { 0.0 };
    let per_trade = if stats.trades_taken > 0 {
        stats.net_pnl / stats.trades_taken as f64
    } else {
        0.0
    };
    let (c_open, c_close) = color_pair(stats.net_pnl);
    println!();
    println!("  THE QUANT — end of paper trade");
    println!("  ------------------------------");
    println!("  bars processed : {}", stats.bars_processed);
    println!("  trades taken   : {}", stats.trades_taken);
    println!(
        "  signals        : {} buy / {} sell / {} risk-denied",
        stats.signals_buy, stats.signals_sell, stats.risk_denied
    );
    println!("  net P&L        : {c_open}{:+.2}{c_close}   (per trade {per_trade:+.2})", stats.net_pnl);
    println!("  return         : {c_open}{:+.2}%{c_close}", ret);
    println!("  final equity   : {:.2}", stats.final_equity);
    match stats.regime {
        Some(r) => println!("  final regime   : {}", r.as_str()),
        None => println!("  final regime   : n/a"),
    }
    println!("  state written  : {}", cfg.system.state_dir);
}

fn color_pair(v: f64) -> (&'static str, &'static str) {
    if v >= 0.0 {
        ("\x1b[32m", "\x1b[0m")
    } else {
        ("\x1b[31m", "\x1b[0m")
    }
}