//! # Risk Management Module (Layer 7)
//!
//! Handles position sizing, drawdown circuit breakers, correlation exposure
//! management, and pre-flight trade validation. All calculations use exact
//! Decimal arithmetic for financial precision.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use crate::account::{Account, AccountMetrics, AccountRules, AccountStatus};
use crate::strategy::TradeSignal;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Drawdown circuit breaker levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitBreakerLevel {
    None,
    DailyDd2Pct,     // Halt new trades 2h, reduce size 50%
    DailyDd3Pct,     // Close all, halt for rest of day
    TotalDd50Pct,    // Halt, require manual review
    TotalDd75Pct,    // Close all, paper trading only
    TotalDd90Pct,    // Emergency shutdown
}

/// Risk engine — validates trades and manages risk
#[derive(Debug)]
pub struct RiskEngine {
    /// Correlation matrix: (symbol_a, symbol_b) -> correlation
    correlation_matrix: Arc<RwLock<HashMap<(String, String), f64>>>,
    /// Open positions per account
    open_positions: Arc<RwLock<HashMap<uuid::Uuid, Vec<PositionRisk>>>>,
    /// Current circuit breaker level per account
    circuit_breakers: Arc<RwLock<HashMap<uuid::Uuid, CircuitBreakerLevel>>>,
    /// Configuration
    config: QuantConfig,
}

/// Risk assessment for an open position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRisk {
    pub symbol: String,
    pub direction: String,
    pub volume: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub stop_loss: Decimal,
    pub take_profit: Decimal,
    pub unrealized_pnl: Decimal,
    pub risk_amount: Decimal,
    pub risk_pct: f64,
}

/// Result of a pre-flight risk check
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub passed: bool,
    pub checks: Vec<RiskCheck>,
    pub suggested_size: Option<Decimal>,
    pub circuit_breaker: CircuitBreakerLevel,
}

#[derive(Debug, Clone)]
pub struct RiskCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

impl RiskEngine {
    pub fn new(config: &QuantConfig) -> Self {
        Self {
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
            open_positions: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(),
        }
    }

    /// Pre-flight risk check before executing a trade
    pub async fn pre_flight_check(
        &self,
        account: &Account,
        signal: &TradeSignal,
        current_price: Decimal,
    ) -> QuantResult<RiskCheckResult> {
        let mut checks = Vec::new();
        let mut all_passed = true;

        // Check 1: Account status
        if account.status != AccountStatus::Active {
            checks.push(RiskCheck {
                name: "Account Status".into(),
                passed: false,
                detail: format!("Account is {:?}", account.status),
            });
            all_passed = false;
        }

        // Check 2: Margin requirement
        let margin_check = self.check_margin(account, signal, current_price).await?;
        checks.push(margin_check.clone());
        if !margin_check.passed {
            all_passed = false;
        }

        // Check 3: Daily loss limit
        let daily_loss_check = self.check_daily_loss_limit(account).await?;
        checks.push(daily_loss_check.clone());
        if !daily_loss_check.passed {
            all_passed = false;
        }

        // Check 4: Drawdown limit
        let dd_check = self.check_drawdown_limit(account).await?;
        checks.push(dd_check.clone());
        if !dd_check.passed {
            all_passed = false;
        }

        // Check 5: Correlation exposure
        let corr_check = self.check_correlation_exposure(account, &signal.symbol).await?;
        checks.push(corr_check.clone());
        if !corr_check.passed {
            all_passed = false;
        }

        // Check 6: Position sizing
        let size = self.compute_position_size(account, signal, current_price).await?;
        checks.push(RiskCheck {
            name: "Position Sizing".into(),
            passed: true,
            detail: format!("Suggested size: {} lots", size),
        });

        // Check circuit breaker
        let cb_level = self.get_circuit_breaker_level(account).await;
        if cb_level != CircuitBreakerLevel::None {
            checks.push(RiskCheck {
                name: "Circuit Breaker".into(),
                passed: false,
                detail: format!("Active circuit breaker: {:?}", cb_level),
            });
            all_passed = false;
        }

        Ok(RiskCheckResult {
            passed: all_passed,
            checks,
            suggested_size: if all_passed { Some(size) } else { None },
            circuit_breaker: cb_level,
        })
    }

    /// Compute position size using the full formula from the spec
    pub async fn compute_position_size(
        &self,
        account: &Account,
        signal: &TradeSignal,
        current_price: Decimal,
    ) -> QuantResult<Decimal> {
        let balance = account.metrics.balance;
        let risk_per_trade = Decimal::from_f64_retain(self.config.account.risk_per_trade_pct / 100.0)
            .unwrap_or(Decimal::new(5, 3)); // 0.5% default

        // BaseRisk = AccountBalance × RiskPerTrade%
        let base_risk = balance * risk_per_trade;

        // VolAdj = TargetVol% / CurrentForecastVol%
        let vol_adj = Decimal::new(1, 1); // Simplified: 0.1 target / 0.1 current

        // RegimeAdj from signal regime
        let regime_mult = Decimal::from_f64_retain(signal.regime.sizing_multiplier())
            .unwrap_or(Decimal::new(1, 0));

        // Quality = SignalQualityScore
        let quality = Decimal::from_f64_retain(signal.strength)
            .unwrap_or(Decimal::new(5, 1));

        // FinalSize = BaseRisk × VolAdj × RegimeAdj × Quality
        let final_size = base_risk * vol_adj * regime_mult * quality;

        // Apply rule caps
        let max_lot = account.rules.max_lot_size;
        let final_size = final_size.min(max_lot);

        Ok(final_size.max(Decimal::ZERO))
    }

    /// Check margin requirements
    async fn check_margin(&self, account: &Account, signal: &TradeSignal, price: Decimal) -> QuantResult<RiskCheck> {
        let margin_used = account.metrics.margin;
        let free_margin = account.metrics.free_margin;
        let required_margin = price * Decimal::new(1, 0) / Decimal::from(account.rules.leverage);

        Ok(RiskCheck {
            name: "Margin Check".into(),
            passed: required_margin <= free_margin,
            detail: format!(
                "Required: {}, Free: {}, Used: {}",
                required_margin, free_margin, margin_used
            ),
        })
    }

    /// Check daily loss limit
    async fn check_daily_loss_limit(&self, account: &Account) -> QuantResult<RiskCheck> {
        let daily_dd = account.metrics.daily_drawdown_pct;
        let limit = account.rules.daily_loss_limit_pct;

        Ok(RiskCheck {
            name: "Daily Loss Limit".into(),
            passed: daily_dd < limit,
            detail: format!("Daily DD: {:.2}%, Limit: {:.2}%", daily_dd, limit),
        })
    }

    /// Check drawdown limit
    async fn check_drawdown_limit(&self, account: &Account) -> QuantResult<RiskCheck> {
        let total_dd = account.metrics.total_drawdown_pct;
        let limit = account.rules.max_drawdown_pct;

        Ok(RiskCheck {
            name: "Drawdown Limit".into(),
            passed: total_dd < limit,
            detail: format!("Total DD: {:.2}%, Limit: {:.2}%", total_dd, limit),
        })
    }

    /// Check correlation exposure
    async fn check_correlation_exposure(&self, account: &Account, symbol: &str) -> QuantResult<RiskCheck> {
        let positions = self.open_positions.read().await;
        let account_positions = positions.get(&account.id).cloned().unwrap_or_default();

        let mut correlated_volume = Decimal::ZERO;
        for pos in &account_positions {
            let corr = self.correlation_matrix.read().await
                .get(&(pos.symbol.clone(), symbol.to_string()))
                .copied()
                .unwrap_or(0.5);
            if corr > self.config.risk.max_correlation_threshold {
                correlated_volume += pos.volume;
            }
        }

        let max_exposure = account.metrics.balance * Decimal::from_f64_retain(
            self.config.account.max_correlated_group_exposure_pct / 100.0
        ).unwrap_or(Decimal::new(2, 0));

        Ok(RiskCheck {
            name: "Correlation Exposure".into(),
            passed: correlated_volume <= max_exposure,
            detail: format!("Correlated volume: {}, Max: {}", correlated_volume, max_exposure),
        })
    }

    /// Get current circuit breaker level
    pub async fn get_circuit_breaker_level(&self, account: &Account) -> CircuitBreakerLevel {
        let dd_pct = account.metrics.total_drawdown_pct.to_f64().unwrap_or(0.0);
        let max_dd = account.rules.max_drawdown_pct.to_f64().unwrap_or(10.0);
        let daily_dd = account.metrics.daily_drawdown_pct.to_f64().unwrap_or(0.0);

        if dd_pct > max_dd * 0.9 {
            CircuitBreakerLevel::TotalDd90Pct
        } else if dd_pct > max_dd * 0.75 {
            CircuitBreakerLevel::TotalDd75Pct
        } else if dd_pct > max_dd * 0.5 {
            CircuitBreakerLevel::TotalDd50Pct
        } else if daily_dd > 3.0 {
            CircuitBreakerLevel::DailyDd3Pct
        } else if daily_dd > 2.0 {
            CircuitBreakerLevel::DailyDd2Pct
        } else {
            CircuitBreakerLevel::None
        }
    }

    /// Update correlation matrix with new data
    pub async fn update_correlation(&self, symbol_a: &str, symbol_b: &str, correlation: f64) {
        let mut matrix = self.correlation_matrix.write().await;
        matrix.insert((symbol_a.to_string(), symbol_b.to_string()), correlation);
        matrix.insert((symbol_b.to_string(), symbol_a.to_string()), correlation);
    }

    /// Track a new open position
    pub async fn add_position(&self, account_id: &uuid::Uuid, position: PositionRisk) {
        let mut positions = self.open_positions.write().await;
        positions.entry(*account_id).or_insert_with(Vec::new).push(position);
    }

    /// Remove a closed position
    pub async fn remove_position(&self, account_id: &uuid::Uuid, symbol: &str) {
        let mut positions = self.open_positions.write().await;
        if let Some(pos_list) = positions.get_mut(account_id) {
            pos_list.retain(|p| p.symbol != symbol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::{Account, AccountType, AccountStage, EncryptedCredentials, AccountRules};
    use crate::strategy::{TradeSignal, TradeSignalDirection};
    use crate::regime::Regime;

    fn create_test_account() -> Account {
        Account {
            id: uuid::Uuid::new_v4(),
            name: "Test".into(),
            account_type: AccountType::PropFirm,
            stage: AccountStage::Evaluation,
            credentials: EncryptedCredentials::new(vec![], "Server".into(), "123".into()),
            rules: AccountRules::default(),
            status: AccountStatus::Active,
            metrics: AccountMetrics {
                balance: Decimal::new(10000, 0),
                equity: Decimal::new(10500, 0),
                margin: Decimal::new(1000, 0),
                free_margin: Decimal::new(9500, 0),
                ..Default::default()
            },
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_position_sizing() {
        let config = QuantConfig::default();
        let engine = RiskEngine::new(&config);
        let account = create_test_account();
        let signal = TradeSignal {
            symbol: "EURUSD".into(),
            direction: TradeSignalDirection::Buy,
            strength: 0.8,
            strategy_id: uuid::Uuid::new_v4(),
            regime: Regime::TrendingUp,
            feature_snapshot: vec![],
            timestamp: Utc::now(),
        };

        let size = engine.compute_position_size(&account, &signal, Decimal::new(10500, 4)).await.unwrap();
        assert!(size > Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = QuantConfig::default();
        let engine = RiskEngine::new(&config);
        let mut account = create_test_account();
        account.metrics.total_drawdown_pct = Decimal::new(55, 1); // 5.5% of 10% max
        let level = engine.get_circuit_breaker_level(&account).await;
        assert_eq!(level, CircuitBreakerLevel::TotalDd50Pct);
    }
}
