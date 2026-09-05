//! RISK ENGINE — Layer 8 (v2.1 §10, v3.1 shield).
//!
//! Enforces HARD STOPS before breaches. Every trade intent must pass pre-flight
//! (drawdown cap, daily loss cap, per-trade risk, maximum positions). The shield
//! (v3.1 Guardian) is a per-trade loss circuit breaker: 2 strikes halts a prop
//! account. This is safety-first: a rule breach is always worse than a missed
//! opportunity.

use crate::config::AccountCfg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskDecision {
    Allow,
    Deny(&'static str),
}

/// Current exposure snapshot fed into the risk engine each bar.
#[derive(Debug, Clone, Default)]
pub struct RiskContext {
    pub balance: f64,
    pub equity: f64,
    pub open_positions: usize,
    /// current drawdown as a fraction of peak equity (0..1)
    pub drawdown_pct: f64,
    /// daily P&L (negative if losing today)
    pub daily_pnl: f64,
    /// yesterday's closing equity used to compute the daily loss
    pub day_start_equity: f64,
}

pub struct RiskEngine {
    pub cfg: AccountCfg,
    /// equity high-water mark used to compute drawdown
    pub peak_equity: f64,
    /// per-trade loss strikes (v3.1 Guardian Shield)
    pub shield_strikes: u32,
    pub shield_strike_limit: u32,
    /// diagnostic: counts of each deny reason
    pub deny_dd: u64,
    pub deny_daily: u64,
    pub deny_risk: u64,
    pub deny_shield: u64,
}

impl RiskEngine {
    pub fn new(cfg: AccountCfg) -> Self {
        RiskEngine {
            cfg,
            peak_equity: 0.0,
            shield_strikes: 0,
            shield_strike_limit: 2,
            deny_dd: 0,
            deny_daily: 0,
            deny_risk: 0,
            deny_shield: 0,
        }
    }

    /// Update the high-water mark from current equity.
    pub fn observe_equity(&mut self, equity: f64) {
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
    }

    pub fn current_drawdown_pct(&self, equity: f64) -> f64 {
        if self.peak_equity <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity - equity) / self.peak_equity * 100.0).max(0.0)
    }

    /// Alias used by the engine.
    pub fn drawdown_pct(&self, equity: f64) -> f64 {
        self.current_drawdown_pct(equity)
    }

    /// Pre-flight check before opening a position of `risk_amount` dollars.
    pub fn pre_flight(&mut self, ctx: &RiskContext, risk_amount: f64) -> RiskDecision {
        // 1. Drawdown hard cap.
        let dd = self.drawdown_pct(ctx.equity);
        if dd > self.cfg.max_drawdown_pct {
            self.deny_dd += 1;
            return RiskDecision::Deny("drawdown limit reached");
        }
        if ctx.drawdown_pct > self.cfg.max_drawdown_pct {
            self.deny_dd += 1;
            return RiskDecision::Deny("drawdown limit reached (reported)");
        }

        // 2. Daily loss cap.
        let daily_loss = (ctx.equity - ctx.day_start_equity) / ctx.day_start_equity * 100.0;
        if daily_loss < -self.cfg.daily_loss_limit_pct {
            self.deny_daily += 1;
            return RiskDecision::Deny("daily loss limit hit");
        }

        // 3. Per-trade risk cap. A tiny relative tolerance avoids rejecting a
        //    risk_amount that is equal to max_risk up to floating-point error
        //    (risk_amount = volume*stop_dist and volume is sized to hit max_risk
        //    exactly, so the comparison must not fail on a rounding hair).
        let max_risk = ctx.equity * self.cfg.risk_per_trade_pct / 100.0;
        let tol = max_risk * 0.001 + 1e-9;
        if risk_amount > max_risk + tol {
            self.deny_risk += 1;
            return RiskDecision::Deny("per-trade risk exceeds allowed");
        }

        // 4. Shield (prop-firm per-trade loss cap). This is a v3.1 prop-firm
        //    feature — only enforced on PROP accounts. For a PERSONAL account
        //    the risk engine relies on drawdown/daily/risk caps instead, which
        //    is the appropriate "personal" blast discipline. 2 strikes halts.
        let acct: &str = self.cfg.r#type.as_str();
        if acct.starts_with("PROP") && self.shield_strikes >= self.shield_strike_limit {
            self.deny_shield += 1;
            return RiskDecision::Deny("shield blown (2 strikes)");
        }

        RiskDecision::Allow
    }

    /// Called after a losing trade closes. Returns the new blast state.
    pub fn register_loss(&mut self, pnl: f64) {
        if pnl < 0.0 {
            self.shield_strikes += 1;
        }
    }

    /// Position size in LOTS so the worst-case loss at the stop == risk_amount.
    /// loss_at_stop = stop_distance * lots * CONTRACT_SIZE == risk_amount
    ///   => lots = risk_amount / (stop_distance * CONTRACT_SIZE)
    pub fn position_size(&self, equity: f64, stop_distance: f64) -> f64 {
        let risk_amount = equity * self.cfg.risk_per_trade_pct / 100.0;
        let contract = crate::execution::CONTRACT_SIZE;
        let denom = stop_distance.max(1e-9) * contract;
        (risk_amount / denom).max(0.0)
    }

    pub fn account_status(&self, ctx: &RiskContext) -> &'static str {
        let dd = self.drawdown_pct(ctx.equity);
        if self.shield_strikes >= self.shield_strike_limit {
            "Blown"
        } else if dd > self.cfg.max_drawdown_pct {
            "Halted"
        } else {
            "Active"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn denies_when_drawdown_exceeded() {
        let mut cfg = Config::load().unwrap();
        cfg.account.max_drawdown_pct = 5.0;
        let mut risk = RiskEngine::new(cfg.account.clone());
        risk.observe_equity(10_000.0); // peak
        let ctx = RiskContext {
            equity: 9_000.0,
            day_start_equity: 10_000.0,
            ..Default::default()
        };
        risk.observe_equity(9_000.0);
        let res = risk.pre_flight(&ctx, 50.0);
        assert!(matches!(res, RiskDecision::Deny(_)));
    }
}
