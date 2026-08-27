//! ENGINE — the event loop that ties the whole system together.
//!
//! Per-bar pipeline:
//!   feed.next_bar()  -> Bar
//!   compute_features -> [f64; N_FEATURES]
//!   regime.update    -> Regime
//!   strategy.evaluate-> Signal
//!   risk.pre_flight  -> Allow/Deny
//!   execution.execute-> fill / close
//!   state.save       -> durable store (source of truth for git)
//!
//! The same loop runs lean on a 4 GB VPS or full on a 64 GB workstation; the
//! `ResourceProfile` only decides class of service / memory budgets.

use crate::config::Config;
use crate::execution::{ExecutionEngine, OrderRequest};
use crate::features::compute_features;
use crate::onnx::ModelBackend;
use crate::regime::{Regime, RegimeDetector};
use crate::resource::ResourceProfile;
use crate::risk::{RiskContext, RiskDecision, RiskEngine};
use crate::simfeed::{Bar, SimFeed};
use crate::state::{AccountState, Store, Trade};
use crate::strategy::{Direction, StrategyEngine};

/// Summary metrics reported after a run.
pub struct EngineStats {
    pub bars_processed: u64,
    pub trades_taken: u64,
    pub net_pnl: f64,
    pub final_equity: f64,
    pub regime: Option<Regime>,
    pub signals_buy: u64,
    pub signals_sell: u64,
    pub risk_denied: u64,
}

/// Run the paper-trading simulation for `n_bars` ticks.
pub fn run_paper_trade<'a>(
    cfg: &Config,
    _profile: &ResourceProfile,
    model: &'a dyn ModelBackend,
    n_bars: usize,
) -> Result<EngineStats, String> {
    let mut store = Store::new(cfg)?;
    let mut root = store.load()?;
    if root.account.balance <= 0.0 {
        root.account = AccountState::new("paper", "PERSONAL", 10_000.0);
    }

    let mut regime_det = RegimeDetector::new(200);
    let mut strategy = StrategyEngine::new(model);
    let mut risk = RiskEngine::new(cfg.account.clone());
    let mut exec = ExecutionEngine::new();

    let symbols: Vec<String> = if cfg.system.symbols.is_empty() {
        vec!["EURUSD".to_string()]
    } else {
        cfg.system.symbols.clone()
    };
    let base = [1.08, 2400.0, 1.26];
    let mut feeds: Vec<SimFeed> = symbols
        .iter()
        .enumerate()
        .map(|(i, sym)| SimFeed::new(sym.as_str(), base[i % base.len()], 1234 + i as u64))
        .collect();

    const WIN: usize = 60;
    let mut window: Vec<Bar> = Vec::with_capacity(WIN + 1);
    let mut bars_done = 0usize;
    let mut trades_taken = 0u64;
    let mut net_pnl = 0.0f64;
    let mut last_regime: Option<Regime> = None;
    let mut signals_buy = 0u64;
    let mut signals_sell = 0u64;
    let mut risk_denied = 0u64;
    let mut matched_trades = 0u64;
    // Main loop.
    for bar_idx in 0..n_bars {
        let feed_idx = bar_idx % feeds.len();
        let feed = &mut feeds[feed_idx];
        let bar = feed.next_bar();

        // Rolling window.
        if window.len() >= WIN {
            window.remove(0);
        }
        window.push(bar.clone());
        if window.len() < WIN {
            continue;
        }

        let f = compute_features(&window);
        let reg = regime_det.update(&window);
        last_regime = Some(reg);
        let sig = strategy.evaluate(&f, reg);

        // Revalue any open position and update drawdown high-water mark.
        let unrealized = if let Some(p) = exec.open.as_ref() {
            let dir = if p.direction == "BUY" { 1.0 } else { -1.0 };
            (bar.close - p.entry_price) * p.volume * dir * crate::execution::CONTRACT_SIZE
        } else {
            0.0
        };
        let equity = root.account.balance + unrealized;
        risk.observe_equity(equity);

        match sig.direction {
            Direction::Buy => signals_buy += 1,
            Direction::Sell => signals_sell += 1,
            _ => {}
        }
        // Act on the strategy signal.
        match sig.direction {
            Direction::Buy | Direction::Sell => {
                if !exec.has_position() {
                    let stop_dist = (bar.close / 1000.0).max(1e-4);
                    let volume = risk.position_size(equity, stop_dist);
                    // risk_amount = worst case loss at the stop, in dollars.
                    let risk_amount = volume * stop_dist * crate::execution::CONTRACT_SIZE;
                    let ctx = RiskContext {
                        balance: root.account.balance,
                        equity,
                        open_positions: 0,
                        drawdown_pct: risk.drawdown_pct(equity),
                        daily_pnl: root.account.daily_pnl,
                        day_start_equity: root.account.balance,
                    };
                    match risk.pre_flight(&ctx, risk_amount) {
                        RiskDecision::Allow => {
                            let (sl, tp) = match sig.direction {
                                Direction::Buy => (bar.close - stop_dist, bar.close + 2.0 * stop_dist),
                                _ => (bar.close + stop_dist, bar.close - 2.0 * stop_dist),
                            };
                            let _ = exec.execute(
                                OrderRequest::new(bar.symbol.as_str(), sig.direction, volume, sl, tp),
                                bar.close,
                            );
                            if matched_trades < 8 {
                                eprintln!(
                                    "[diag] ENTRY {} vol={:.4} lots stop={:.6} risk={:.2}",
                                    sig.direction.as_str(), volume, stop_dist, risk_amount
                                );
                                matched_trades += 1;
                            }
                        }
                        RiskDecision::Deny(_) => risk_denied += 1,
                    }
                }
            }
            Direction::Close | Direction::Hold => {
                if exec.has_position() {
                    let res = exec.execute(
                        OrderRequest::new(bar.symbol.as_str(), Direction::Close, 0.0, 0.0, 0.0),
                        bar.close,
                    );
                    if let Some(pos) = res.closed {
                        root.account.balance += pos.unrealized_pnl;
                        root.account.daily_pnl += pos.unrealized_pnl;
                        risk.register_loss(pos.unrealized_pnl);
                        net_pnl += pos.unrealized_pnl;
                        trades_taken += 1;
                        root.trades.push(Trade {
                            id: format!("t-{bar_idx}"),
                            account_id: root.account.account_id.clone(),
                            symbol: pos.symbol.clone(),
                            direction: pos.direction.clone(),
                            volume: pos.volume,
                            entry_price: pos.entry_price,
                            exit_price: bar.close,
                            pnl: pos.unrealized_pnl,
                            open_time: pos.open_time.clone(),
                            close_time: crate::util::now_iso8601(),
                            strategy: sig.model.clone(),
                        });
                    }
                }
            }
        }

        // Stop-loss / take-profit forcing.
        if let Some(p) = exec.open.clone() {
            let hit_sl = if p.direction == "BUY" {
                bar.low <= p.stop_loss
            } else {
                bar.high >= p.stop_loss
            };
            let hit_tp = if p.direction == "BUY" {
                bar.high >= p.take_profit
            } else {
                bar.low <= p.take_profit
            };
            if hit_sl || hit_tp {
                let exit_price = if hit_sl { p.stop_loss } else { p.take_profit };
                let close_req = OrderRequest::new(bar.symbol.as_str(), Direction::Close, 0.0, 0.0, 0.0);
                if let Some(pos) = exec.execute(close_req, exit_price).closed {
                    root.account.balance += pos.unrealized_pnl;
                    root.account.daily_pnl += pos.unrealized_pnl;
                    risk.register_loss(pos.unrealized_pnl);
                    net_pnl += pos.unrealized_pnl;
                    trades_taken += 1;
                    root.trades.push(Trade {
                        id: format!("t-{bar_idx}-sl"),
                        account_id: root.account.account_id.clone(),
                        symbol: pos.symbol.clone(),
                        direction: pos.direction.clone(),
                        volume: pos.volume,
                        entry_price: pos.entry_price,
                        exit_price,
                        pnl: pos.unrealized_pnl,
                        open_time: pos.open_time.clone(),
                        close_time: crate::util::now_iso8601(),
                        strategy: "stop-mgmt".into(),
                    });
                }
            }
        }

        bars_done += 1;
        if bar_idx % 200 == 0 {
            let _ = store.save(&root);
        }
    }

    let final_equity = root.account.balance
        + exec.open.as_ref().map(|p| p.unrealized_pnl).unwrap_or(0.0);
    let _ = store.save(&root);

    // Diagnostic: how many entries were rejected per reason.
    eprintln!(
        "[engine] risk denials — dd:{} daily:{} risk:{} shield:{}",
        risk.deny_dd, risk.deny_daily, risk.deny_risk, risk.deny_shield
    );

    Ok(EngineStats {
        bars_processed: bars_done as u64,
        trades_taken,
        net_pnl,
        final_equity,
        regime: last_regime,
        signals_buy,
        signals_sell,
        risk_denied,
    })
}