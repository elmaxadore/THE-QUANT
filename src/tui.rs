//! DASHBOARD — a lightweight ANSI report (the lean default).
//!
//! The `tui` cargo feature can upgrade this to a full ratatui interactive UI;
//! the default build uses a clean, dependency-free textual dashboard so it runs
//! on any Tier-1 machine and inside SSH.

use crate::engine::{DeskStats, EngineStats};

/// Print the end-of-run report using ANSI colours.
pub fn report(cfg: &crate::config::Config, stats: &EngineStats) {
    let start_balance = 10_000.0;
    let ret = if start_balance > 0.0 {
        stats.net_pnl / start_balance * 100.0
    } else {
        0.0
    };
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
    if stats.halted {
        println!("  account status : \x1b[33mHALTED (drawdown cap reached)\x1b[0m");
    } else {
        println!("  account status : ACTIVE");
    }
    println!(
        "  net P&L        : {c_open}{:+.2}{c_close}   (per trade {per_trade:+.2})",
        stats.net_pnl
    );
    println!("  return         : {c_open}{:+.2}%{c_close}", ret);
    println!("  final equity   : {:.2}", stats.final_equity);
    if stats.regime.is_empty() {
        println!("  final regime   : n/a");
    } else {
        println!("  final regime   : {}", stats.regime);
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

/// Multi-desk report — one block per account.
pub fn report_multi(cfg: &crate::config::Config, desks: &[DeskStats]) {
    println!();
    println!("  THE QUANT — multi-desk paper trade");
    println!("  ----------------------------------");
    println!("  desks          : {}", desks.len());
    println!("  state written  : {}", cfg.system.state_dir);
    println!();

    let mut total_pnl = 0.0;
    let mut total_trades = 0u64;
    let mut total_denied = 0u64;
    let mut halted_count = 0usize;

    for d in desks {
        total_pnl += d.realized_pnl;
        total_trades += d.trades_taken;
        total_denied += d.denied_count;
        if d.halted {
            halted_count += 1;
        }

        let (c_open, c_close) = color_pair(d.realized_pnl);
        let ret = if d.initial_balance > 0.0 {
            d.realized_pnl / d.initial_balance * 100.0
        } else {
            0.0
        };
        let status = if d.halted {
            format!("{} (HALTED)", d.status)
        } else {
            d.status.clone()
        };

        println!("  [{}] {}", d.account_id, d.firm);
        println!(
            "    trades: {} taken / {} denied | P&L: {}{:+.2}{} ({}{:+.2}%{})",
            d.trades_taken, d.denied_count, c_open, d.realized_pnl, c_close, c_open, ret, c_close,
        );
        println!(
            "    open positions: {} | equity: {:.2} | status: {}",
            d.positions_open, d.equity, status,
        );
        println!();
    }

    let (tc_open, tc_close) = color_pair(total_pnl);
    println!("  === TOTAL ===");
    println!(
        "    trades: {} taken / {} denied | P&L: {}{:+.2}{}",
        total_trades, total_denied, tc_open, total_pnl, tc_close,
    );
    println!("    desks halted: {}/{}", halted_count, desks.len(),);
}
