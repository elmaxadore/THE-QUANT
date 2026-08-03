//! # Reinforcement Learning Layer — QuantGym Environment
//!
//! A self-contained RL environment for trading. The RL agent is a *guest*:
//! it cannot override hard stops, `max_holding_bars`, or `RuleCap`. Its actions
//! are advisory position adjustments that are always clamped by the risk engine.
//!
//! ## Safety Contract
//! - RL never overrides hard risk limits (hard stops, max holding, RuleCap)
//! - Auto-disable if live Sharpe < 0.3 over 20 trades
//! - Distilled policy (32→16→11) runs live; full PPO only trains offline
//! - Every action is sanitized by the risk engine before reaching the executor

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Standardized observation vector for the RL agent.
/// Features are normalized to roughly [-1, 1] for stable training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// OHLCV & technical features (normalized)
    pub features: Vec<f64>,
    /// Current regime index (from regime detector)
    pub regime_index: usize,
    /// Position state: {current_volume, entry_price, unrealized_pnl}
    pub position: Vec<f64>,
    /// Account state: {equity, balance, free_margin, daily_pnl}
    pub account: Vec<f64>,
    /// Time features: {hour_sin, hour_cos, day_of_week}
    pub time: Vec<f64>,
}

impl Observation {
    /// Flatten into a single vector for the neural network input layer.
    pub fn flatten(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(
            self.features.len() + self.position.len() + self.account.len() + self.time.len() + 1,
        );
        out.extend_from_slice(&self.features);
        out.push(self.regime_index as f64);
        out.extend_from_slice(&self.position);
        out.extend_from_slice(&self.account);
        out.extend_from_slice(&self.time);
        out
    }

    pub fn state_dim(&self) -> usize {
        self.flatten().len()
    }
}

/// Action space for the RL agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Action {
    /// Do nothing / hold current position.
    Hold,
    /// Increase position size (scaled by a fraction 0..1).
    Increase(f64),
    /// Decrease position size (scaled by a fraction 0..1).
    Decrease(f64),
    /// Close the entire position.
    Close,
}

impl Action {
    /// Convert to a discrete index for the action layer.
    pub fn to_index(&self) -> usize {
        match self {
            Action::Hold => 0,
            Action::Increase(_) => 1,
            Action::Decrease(_) => 2,
            Action::Close => 3,
        }
    }

    pub fn from_index(index: usize, scale: f64) -> Self {
        match index {
            0 => Action::Hold,
            1 => Action::Increase(scale.clamp(0.0, 1.0)),
            2 => Action::Decrease(scale.clamp(0.0, 1.0)),
            3 => Action::Close,
            _ => Action::Hold,
        }
    }
}

/// Reward signal for the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reward {
    pub immediate: f64,
    pub risk_adjusted: f64,
    pub milestone_bonus: f64,
    pub penalty: f64,
}

impl Reward {
    pub fn total(&self) -> f64 {
        self.immediate + self.risk_adjusted + self.milestone_bonus - self.penalty
    }
}

/// A single step in the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub observation: Observation,
    pub reward: f64,
    pub done: bool,
    pub info: StepInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepInfo {
    pub action: Option<Action>,
    pub position: f64,
    pub equity: f64,
    pub pnl: f64,
    pub drawdown: f64,
    pub sharpe_window: f64,
    pub clamped: bool,
}

/// Configuration for the QuantGym environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GymConfig {
    /// Maximum position size (in units, enforced by RuleCap).
    pub max_position: f64,
    /// Risk per trade as fraction of equity (0.005 = 0.5%).
    pub risk_per_trade: f64,
    /// Maximum holding bars before forced close.
    pub max_holding_bars: u32,
    /// Reward scaling factor.
    pub reward_scale: f64,
    /// Maximum drawdown fraction before episode ends (e.g. 0.10 = 10%).
    pub max_drawdown_frac: f64,
    /// Number of steps per episode.
    pub episode_length: u32,
    /// Whether to clamp actions to risk limits (always true in production).
    pub enforce_safety: bool,
}

impl Default for GymConfig {
    fn default() -> Self {
        Self {
            max_position: 100.0,
            risk_per_trade: 0.005,
            max_holding_bars: 48,
            reward_scale: 1.0,
            max_drawdown_frac: 0.10,
            episode_length: 1000,
            enforce_safety: true,
        }
    }
}

/// Internal state of the trading environment.
#[derive(Debug, Clone)]
struct EnvState {
    step: u32,
    position: f64,
    entry_price: f64,
    holding_bars: u32,
    equity: f64,
    balance: f64,
    peak_equity: f64,
    returns: Vec<f64>,
    daily_pnl: f64,
    current_price: f64,
}

impl Default for EnvState {
    fn default() -> Self {
        Self {
            step: 0,
            position: 0.0,
            entry_price: 0.0,
            holding_bars: 0,
            equity: 100_000.0, // starting capital
            balance: 100_000.0,
            peak_equity: 100_000.0,
            returns: Vec::new(),
            daily_pnl: 0.0,
            current_price: 1.0,
        }
    }
}

/// QuantGym — RL environment for trading.
///
/// This is a *simulator* used for training. It consumes a stream of price bars
/// (or synthetic ones) and produces observations + rewards. The *live* system
/// uses the `DistilledPolicy` for inference only; the full simulator runs in
/// the lab/evolution phase.
#[derive(Debug)]
pub struct QuantGym {
    config: GymConfig,
    state: EnvState,
    price_series: Vec<f64>,
    price_idx: usize,
    rng: Arc<RwLock<rand::rngs::StdRng>>,
}

impl QuantGym {
    pub fn new(config: GymConfig, price_series: Vec<f64>) -> Self {
        let seed = rand::random::<u64>();
        Self {
            config,
            state: EnvState::default(),
            price_series,
            price_idx: 0,
            rng: Arc::new(RwLock::new(rand::rngs::StdRng::seed_from_u64(seed))),
        }
    }

    /// Reset the environment to the initial state.
    pub fn reset(&mut self) -> Observation {
        self.state = EnvState::default();
        self.price_idx = 0;
        self.observation()
    }

    /// Step the environment with an action.
    pub fn step(&mut self, action: Action) -> StepResult {
        let mut state = self.state.clone();
        state.step += 1;

        // Sample next price
        let next_price = self.next_price();
        let prev_price = state.current_price;
        let ret = (next_price / prev_price) - 1.0;

        // Apply action (with safety clamping)
        let mut clamped = false;
        match action {
            Action::Hold => {}
            Action::Increase(f) => {
                let target = state.position + self.config.max_position * f;
                if self.config.enforce_safety {
                    let cap = self.config.max_position;
                    if target > cap {
                        clamped = true;
                        state.position = cap;
                    } else {
                        state.position = target;
                    }
                } else {
                    state.position = target;
                }
                if state.position > 0.0 && state.entry_price == 0.0 {
                    state.entry_price = next_price;
                }
            }
            Action::Decrease(f) => {
                let target = state.position - self.config.max_position * f;
                if target < 0.0 {
                    clamped = true;
                    state.position = 0.0;
                    state.entry_price = 0.0;
                } else {
                    state.position = target;
                }
            }
            Action::Close => {
                state.position = 0.0;
                state.entry_price = 0.0;
            }
        }

        // Track holding bars
        if state.position > 0.0 {
            state.holding_bars += 1;
        } else {
            state.holding_bars = 0;
        }

        // Compute PnL
        let pnl = if state.position > 0.0 {
            state.position * (next_price - state.entry_price) / state.entry_price
        } else {
            0.0
        };
        state.equity = state.balance + pnl;
        state.peak_equity = state.peak_equity.max(state.equity);
        state.returns.push(ret);
        state.daily_pnl += pnl;

        // Compute reward
        let reward = self.compute_reward(&state, pnl);

        // Determine done
        let drawdown = if state.peak_equity > 0.0 {
            (state.peak_equity - state.equity) / state.peak_equity
        } else {
            0.0
        };
        let holding_breach = state.holding_bars >= self.config.max_holding_bars && state.position > 0.0;
        let dd_breach = drawdown >= self.config.max_drawdown_frac;
        let episode_end = state.step >= self.config.episode_length;
        let done = holding_breach || dd_breach || episode_end;

        self.state = state;
        let info = StepInfo {
            action: Some(action),
            position: self.state.position,
            equity: self.state.equity,
            pnl,
            drawdown,
            sharpe_window: self.compute_window_sharpe(&self.state.returns),
            clamped,
        };

        StepResult {
            observation: self.observation(),
            reward,
            done,
            info,
        }
    }

    /// Build the current observation vector.
    fn observation(&self) -> Observation {
        let s = &self.state;
        let features = vec![
            self.price_series.get(self.price_idx).copied().unwrap_or(1.0).ln() * 0.1,
            (s.current_price - s.entry_price).abs() * 0.01,
            s.returns.last().copied().unwrap_or(0.0) * 10.0,
        ];
        Observation {
            features,
            regime_index: 0,
            position: vec![s.position, s.entry_price, s.equity - s.balance],
            account: vec![s.equity, s.balance, 0.0, s.daily_pnl],
            time: vec![0.0, 0.0, 0.0],
        }
    }

    /// Compute the reward for a step.
    fn compute_reward(&self, state: &EnvState, pnl: f64) -> f64 {
        let mut reward = 0.0;

        // Immediate PnL reward
        reward += pnl * self.config.reward_scale;

        // Risk-adjusted: penalty for large drawdowns
        if state.peak_equity > 0.0 {
            let dd = (state.peak_equity - state.equity) / state.peak_equity;
            if dd > self.config.max_drawdown_frac * 0.5 {
                reward -= (dd - self.config.max_drawdown_frac * 0.5) * 5.0;
            }
        }

        // Holding penalty: encourage decisive action
        if state.holding_bars > self.config.max_holding_bars / 2 {
            reward -= 0.0001 * state.holding_bars as f64;
        }

        reward
    }

    /// Compute a rolling Sharpe ratio over recent returns.
    fn compute_window_sharpe(&self, returns: &[f64]) -> f64 {
        let window = returns.iter().rev().take(20).copied().collect::<Vec<_>>();
        if window.len() < 2 {
            return 0.0;
        }
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let var = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
        if var <= 0.0 {
            0.0
        } else {
            mean / var.sqrt() * (252.0_f64).sqrt()
        }
    }

    /// Get the next price from the series (wraps around or random-walks).
    fn next_price(&mut self) -> f64 {
        if self.price_series.is_empty() {
            // Synthetic random walk
            let mut rng = self.rng.write();
            let shock = (rand::Rng::sample::<f64, _>(&mut *rng, rand::distributions::Standard) - 0.5) * 0.01;
            self.state.current_price * (1.0 + shock)
        } else {
            let p = self.price_series[self.price_idx % self.price_series.len()];
            self.price_idx += 1;
            p
        }
    }

    /// Current environment state info (for debugging).
    pub fn state_info(&self) -> (f64, f64, f64) {
        (self.state.equity, self.state.position, self.state.daily_pnl)
    }

    /// Whether the current episode is done.
    pub fn is_done(&self) -> bool {
        self.state.step >= self.config.episode_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_prices(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64).sin() * 2.0).collect()
    }

    #[test]
    fn test_gym_reset() {
        let prices = synthetic_prices(100);
        let mut gym = QuantGym::new(GymConfig::default(), prices);
        let obs = gym.reset();
        assert_eq!(obs.state_dim(), 3 + 1 + 3 + 4 + 3);
    }

    #[test]
    fn test_gym_step() {
        let prices = synthetic_prices(100);
        let mut gym = QuantGym::new(GymConfig::default(), prices);
        gym.reset();
        let result = gym.step(Action::Increase(0.5));
        assert!(result.reward.is_finite());
        assert!(result.info.position > 0.0);
    }

    #[test]
    fn test_action_clamping() {
        let config = GymConfig {
            max_position: 100.0,
            enforce_safety: true,
            ..Default::default()
        };
        let prices = synthetic_prices(100);
        let mut gym = QuantGym::new(config, prices);
        gym.reset();
        // Try to over-increase; should clamp to max_position
        let result = gym.step(Action::Increase(2.0));
        assert!(result.info.position <= 100.0);
    }

    #[test]
    fn test_close_action() {
        let prices = synthetic_prices(100);
        let mut gym = QuantGym::new(GymConfig::default(), prices);
        gym.reset();
        gym.step(Action::Increase(0.5));
        let result = gym.step(Action::Close);
        assert_eq!(result.info.position, 0.0);
    }
}
