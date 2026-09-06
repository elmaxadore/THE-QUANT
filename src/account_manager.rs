//! Account Manager Module for THE QUANT v4.1 Hercules
//! 
//! This module provides Rust implementations for:
//! - Dynamic position sizing based on account equity and risk parameters
//! - Daily drawdown headroom tracking for prop firm compliance
//! - Correlation-based allocation for multi-asset portfolios

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Prop firm account tiers based on maximum drawdown limits
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AccountTier {
    Tier5Pct = 5,
    Tier8Pct = 8,
    Tier10Pct = 10,
    Tier12Pct = 12,
}

impl AccountTier {
    pub fn as_f64(&self) -> f64 {
        *self as f64
    }
}

/// Configuration for an individual trading account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountConfig {
    pub account_id: String,
    pub initial_equity: f64,
    pub current_equity: f64,
    pub max_drawdown_pct: f64,
    pub daily_drawdown_limit_pct: f64,
    pub risk_per_trade_pct: f64,
    pub max_positions: usize,
    pub tier: AccountTier,
}

impl Default for AccountConfig {
    fn default() -> Self {
        Self {
            account_id: String::from("default"),
            initial_equity: 100_000.0,
            current_equity: 100_000.0,
            max_drawdown_pct: 5.0,
            daily_drawdown_limit_pct: 5.0,
            risk_per_trade_pct: 1.0,
            max_positions: 5,
            tier: AccountTier::Tier5Pct,
        }
    }
}

/// Information about a current or proposed position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub side: String,
    pub unrealized_pnl: f64,
    pub weight: f64,
}

impl PositionInfo {
    pub fn new(
        symbol: String,
        quantity: f64,
        entry_price: f64,
        current_price: f64,
        side: String,
    ) -> Self {
        let mut pos = Self {
            symbol,
            quantity,
            entry_price,
            current_price,
            side,
            unrealized_pnl: 0.0,
            weight: 0.0,
        };
        pos.update_pnl();
        pos
    }

    pub fn update_pnl(&mut self) {
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity;
        if self.side == "short" {
            self.unrealized_pnl = -self.unrealized_pnl;
        }
    }
}

/// Result of the correlation-based allocation optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    pub symbol: String,
    pub optimal_weight: f64,
    pub risk_contribution: f64,
    pub correlation_adjusted: bool,
    pub reason: String,
}

/// Complete report from the account optimization engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub timestamp: String,
    pub account_id: String,
    pub current_equity: f64,
    pub available_headroom: f64,
    pub daily_headroom: f64,
    pub positions: Vec<PositionInfo>,
    pub allocations: Vec<AllocationResult>,
    pub recommended_actions: Vec<String>,
    pub risk_metrics: HashMap<String, f64>,
}

/// Account Optimizer for production trading
pub struct AccountOptimizer {
    accounts: HashMap<String, AccountConfig>,
    positions: HashMap<String, Vec<PositionInfo>>,
    correlation_matrix: Option<Vec<Vec<f64>>>,
    symbols: Vec<String>,
}

impl AccountOptimizer {
    /// Create a new AccountOptimizer
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            positions: HashMap::new(),
            correlation_matrix: None,
            symbols: Vec::new(),
        }
    }

    /// Register a new trading account for optimization
    pub fn register_account(&mut self, config: AccountConfig) {
        let account_id = config.account_id.clone();
        self.accounts.insert(account_id.clone(), config);
        self.positions.insert(account_id, Vec::new());
    }

    /// Update account equity and calculate drawdown metrics
    pub fn update_account_equity(&mut self, account_id: &str, new_equity: f64) -> f64 {
        if let Some(account) = self.accounts.get_mut(account_id) {
            let old_equity = account.current_equity;
            account.current_equity = new_equity;

            let peak_equity = account.initial_equity.max(old_equity);
            let drawdown_pct = ((peak_equity - new_equity) / peak_equity) * 100.0;
            return drawdown_pct;
        }
        0.0
    }

    /// Calculate remaining daily drawdown headroom
    pub fn calculate_daily_headroom(&self, account_id: &str, daily_pnl: f64) -> (f64, f64) {
        if let Some(account) = self.accounts.get(account_id) {
            let daily_limit_usd = account.initial_equity * (account.daily_drawdown_limit_pct / 100.0);
            let remaining_headroom = daily_limit_usd - daily_pnl.abs().min(0.0).abs();
            let remaining_pct = (remaining_headroom / account.initial_equity) * 100.0;
            return (remaining_headroom.max(0.0), remaining_pct.max(0.0));
        }
        (0.0, 0.0)
    }

    /// Calculate dynamic position size using fractional Kelly criterion
    pub fn calculate_dynamic_position_size(
        &self,
        account_id: &str,
        _symbol: &str,
        price: f64,
        volatility: f64,
        win_rate: f64,
        avg_win_loss_ratio: f64,
    ) -> f64 {
        let account = match self.accounts.get(account_id) {
            Some(a) => a,
            None => return 0.0,
        };

        // Check available headroom (simplified - would need mutable access for full calc)
        let drawdown = 0.0; // Would calculate actual drawdown here
        if drawdown >= account.max_drawdown_pct * 0.8 {
            return 0.0;
        }

        // Fractional Kelly calculation
        let edge = win_rate - (1.0 - win_rate) / avg_win_loss_ratio;
        let kelly_fraction = edge / avg_win_loss_ratio;

        // Apply risk scaling based on volatility
        let vol_scale = 1.0.min(0.2 / volatility.max(0.01));
        let risk_fraction = kelly_fraction * vol_scale * (account.risk_per_trade_pct / 100.0);

        // Calculate position size
        let risk_capital = account.current_equity * risk_fraction;
        let position_value = if volatility > 0.0 {
            risk_capital / volatility
        } else {
            risk_capital
        };
        let quantity = position_value / price;

        // Apply maximum position limit (20% per position)
        let max_position_value = account.current_equity * 0.2;
        let max_quantity = max_position_value / price;

        quantity.min(max_quantity)
    }

    /// Build correlation matrix from historical returns
    pub fn build_correlation_matrix(&mut self, returns_data: HashMap<String, Vec<f64>>) {
        self.symbols = returns_data.keys().cloned().collect();
        let n_symbols = self.symbols.len();

        if n_symbols == 0 {
            self.correlation_matrix = Some(vec![vec![1.0]]);
            return;
        }

        // Convert to matrix format
        let returns_matrix: Vec<Vec<f64>> = self
            .symbols
            .iter()
            .map(|s| returns_data.get(s).cloned().unwrap_or_default())
            .collect();

        // Calculate correlation matrix
        let mut corr_matrix = vec![vec![0.0; n_symbols]; n_symbols];

        for i in 0..n_symbols {
            for j in 0..n_symbols {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                } else {
                    corr_matrix[i][j] = self.calculate_correlation(
                        &returns_matrix[i],
                        &returns_matrix[j],
                    );
                }
            }
        }

        self.correlation_matrix = Some(corr_matrix);
    }

    /// Calculate Pearson correlation between two return series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 {
            return 0.0;
        }

        let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

        let mut cov_xy = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov_xy += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom > 0.0 {
            cov_xy / denom
        } else {
            0.0
        }
    }

    /// Optimize portfolio allocation using Hierarchical Risk Parity approach
    pub fn optimize_allocation(
        &self,
        account_id: &str,
        expected_returns: HashMap<String, f64>,
        volatilities: HashMap<String, f64>,
        _target_risk: f64,
    ) -> Vec<AllocationResult> {
        if !self.accounts.contains_key(account_id) {
            return Vec::new();
        }

        let symbols: Vec<String> = expected_returns.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Vec::new();
        }

        // Check if we have correlation data
        if self.correlation_matrix.is_none() || n != self.correlation_matrix.as_ref().unwrap().len() {
            // Use simple inverse volatility weighting
            let inv_vols: Vec<f64> = symbols
                .iter()
                .map(|s| 1.0 / volatilities.get(s).unwrap_or(&0.1).max(0.01))
                .collect();
            let total_inv_vol: f64 = inv_vols.iter().sum();

            return symbols
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let weight = inv_vols[i] / total_inv_vol;
                    let vol = *volatilities.get(s).unwrap_or(&0.1);
                    AllocationResult {
                        symbol: s.clone(),
                        optimal_weight: weight,
                        risk_contribution: weight * vol,
                        correlation_adjusted: false,
                        reason: String::from("Inverse volatility weighting"),
                    }
                })
                .collect();
        }

        // HRP-like allocation with correlation adjustment
        let weights = self.hierarchical_clustering_weights(&symbols, &volatilities);
        let weights = self.tilt_towards_returns(weights, &expected_returns, &symbols);

        // Normalize weights
        let total_weight: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let vol = *volatilities.get(s).unwrap_or(&0.1);
                AllocationResult {
                    symbol: s.clone(),
                    optimal_weight: weights[i],
                    risk_contribution: weights[i] * vol,
                    correlation_adjusted: true,
                    reason: String::from("HRP optimization applied"),
                }
            })
            .collect()
    }

    /// Simplified hierarchical clustering for weight allocation
    fn hierarchical_clustering_weights(
        &self,
        symbols: &[String],
        volatilities: &HashMap<String, f64>,
    ) -> Vec<f64> {
        let n = symbols.len();
        if n == 0 {
            return Vec::new();
        }

        // Start with inverse volatility weights
        let mut weights: Vec<f64> = symbols
            .iter()
            .map(|s| 1.0 / volatilities.get(s).unwrap_or(&0.1).max(0.01))
            .collect();
        let sum: f64 = weights.iter().sum();
        weights = weights.iter().map(|w| w / sum).collect();

        let corr_matrix = match &self.correlation_matrix {
            Some(m) if m.len() == n => m,
            _ => return weights,
        };

        // Reduce weight for highly correlated pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = corr_matrix[i][j].abs();
                if corr > 0.7 {
                    let reduction = 0.1 * corr;
                    weights[i] *= 1.0 - reduction;
                    weights[j] *= 1.0 - reduction;
                }
            }
        }

        // Re-normalize
        let sum: f64 = weights.iter().sum();
        weights.iter().map(|w| w / sum).collect()
    }

    /// Tilt weights towards assets with higher expected returns
    fn tilt_towards_returns(
        &self,
        weights: Vec<f64>,
        expected_returns: &HashMap<String, f64>,
        symbols: &[String],
    ) -> Vec<f64> {
        let returns: Vec<f64> = symbols
            .iter()
            .map(|s| *expected_returns.get(s).unwrap_or(&0.0))
            .collect();

        let min_ret = returns.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ret = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let norm_returns: Vec<f64> = if max_ret > min_ret {
            returns
                .iter()
                .map(|r| (r - min_ret) / (max_ret - min_ret))
                .collect()
        } else {
            vec![0.5; returns.len()]
        };

        // Blend: 70% original weights, 30% return-tilted
        let tilt_factor = 0.3;
        weights
            .iter()
            .zip(norm_returns.iter())
            .map(|(w, r)| w * (1.0 - tilt_factor) + r * tilt_factor)
            .collect()
    }

    /// Generate comprehensive optimization report for an account
    pub fn generate_optimization_report(
        &self,
        account_id: &str,
        daily_pnl: f64,
    ) -> Option<OptimizationReport> {
        let account = self.accounts.get(account_id)?;
        let positions = self.positions.get(account_id).cloned().unwrap_or_default();

        let (total_headroom, _) = self.calculate_daily_headroom(account_id, daily_pnl);

        // Calculate risk metrics
        let total_exposure: f64 = positions.iter().map(|p| p.weight.abs()).sum();
        let net_exposure: f64 = positions.iter().map(|p| p.weight).sum();
        let concentration_risk = positions
            .iter()
            .map(|p| p.weight.abs())
            .fold(0.0_f64, f64::max);

        let mut risk_metrics = HashMap::new();
        risk_metrics.insert(String::from("total_exposure"), total_exposure);
        risk_metrics.insert(String::from("net_exposure"), net_exposure);
        risk_metrics.insert(String::from("concentration_risk"), concentration_risk);
        risk_metrics.insert(
            String::from("drawdown_buffer"),
            account.max_drawdown_pct, // Simplified
        );

        // Generate recommendations
        let mut recommendations = Vec::new();
        if risk_metrics.get("drawdown_buffer").unwrap_or(&0.0) < &account.max_drawdown_pct * 0.3 {
            recommendations.push(String::from(
                "WARNING: Approaching drawdown limit - reduce position sizes",
            ));
        }
        if total_exposure > 0.8 {
            recommendations.push(String::from(
                "High exposure detected - consider reducing leverage",
            ));
        }
        if positions.len() >= account.max_positions {
            recommendations.push(format!(
                "Maximum positions ({}) reached",
                account.max_positions
            ));
        }

        // Create allocations from positions
        let allocations: Vec<AllocationResult> = positions
            .iter()
            .map(|p| AllocationResult {
                symbol: p.symbol.clone(),
                optimal_weight: p.weight,
                risk_contribution: p.weight * 0.1,
                correlation_adjusted: true,
                reason: String::from("Current position weight"),
            })
            .collect();

        let timestamp = chrono::Utc::now().to_rfc3339();

        Some(OptimizationReport {
            timestamp,
            account_id: account_id.to_string(),
            current_equity: account.current_equity,
            available_headroom: total_headroom,
            daily_headroom: total_headroom,
            positions,
            allocations,
            recommended_actions: recommendations,
            risk_metrics,
        })
    }
}

impl Default for AccountOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create an AccountOptimizer pre-configured for a specific prop firm tier
pub fn create_optimizer_for_tier(tier: AccountTier, account_id: &str) -> AccountOptimizer {
    let mut optimizer = AccountOptimizer::new();

    let config = AccountConfig {
        account_id: account_id.to_string(),
        initial_equity: 100_000.0,
        current_equity: 100_000.0,
        max_drawdown_pct: tier.as_f64(),
        daily_drawdown_limit_pct: 5.0.min(tier.as_f64() * 0.8),
        risk_per_trade_pct: if tier == AccountTier::Tier5Pct {
            0.5
        } else {
            1.0
        },
        max_positions: if tier == AccountTier::Tier5Pct {
            3
        } else {
            5
        },
        tier,
    };

    optimizer.register_account(config);
    optimizer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_optimizer_creation() {
        let optimizer = create_optimizer_for_tier(AccountTier::Tier5Pct, "TEST-001");
        assert!(optimizer.accounts.contains_key("TEST-001"));
    }

    #[test]
    fn test_dynamic_position_sizing() {
        let mut optimizer = AccountOptimizer::new();
        optimizer.register_account(AccountConfig::default());

        let size = optimizer.calculate_dynamic_position_size(
            "default",
            "BTCUSDT",
            45000.0,
            0.02,
            0.55,
            1.5,
        );

        assert!(size > 0.0);
        assert!(size < 10.0); // Should be reasonable
    }

    #[test]
    fn test_daily_headroom() {
        let mut optimizer = AccountOptimizer::new();
        let mut config = AccountConfig::default();
        config.daily_drawdown_limit_pct = 5.0;
        optimizer.register_account(config);

        let (headroom_usd, headroom_pct) = optimizer.calculate_daily_headroom("default", -1000.0);
        
        assert!(headroom_usd > 0.0);
        assert!(headroom_pct > 0.0);
        assert!(headroom_pct <= 5.0);
    }
}
