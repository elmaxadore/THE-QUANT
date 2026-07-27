//! # Evolution Loop (Layer 8)
//!
//! The heartbeat of The Quant. Runs an endless 7-phase evolution cycle:
//! Data Update → Regime Re-calibration → Model Retraining → Lab Batch →
//! Strategy Pool Update → GitHub Commit → Resume Trading.
//!
//! ## Trigger Conditions (OR logic)
//! 1. N trades since last evolution (default: 20)
//! 2. T hours elapsed (default: 168 = 7 days)
//! 3. Performance decay: live Sharpe < 50% of backtest Sharpe for 10+ trades
//! 4. Regime instability: confidence < 60% for 24 hours
//! 5. Manual trigger through CLI

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Phase of the evolution cycle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvolutionPhase {
    Idle,
    DataUpdate,
    RegimeRecalibration,
    ModelRetraining,
    LabBatch,
    StrategyPoolUpdate,
    GitHubCommit,
    ResumeTrading,
}

/// Status of the evolution engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStatus {
    pub phase: EvolutionPhase,
    pub cycle_number: u64,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub triggered_by: String,
    pub progress_pct: f64,
    pub error: Option<String>,
}

impl EvolutionStatus {
    pub fn idle() -> Self {
        Self {
            phase: EvolutionPhase::Idle,
            cycle_number: 0,
            started_at: None,
            completed_at: None,
            triggered_by: String::new(),
            progress_pct: 0.0,
            error: None,
        }
    }
}

/// Evolution engine — runs the 7-phase cycle
#[derive(Debug)]
pub struct EvolutionEngine {
    status: Arc<RwLock<EvolutionStatus>>,
    last_evolution: Arc<RwLock<Option<DateTime<Utc>>>>,
    trade_count_since_last: Arc<RwLock<u64>>,
    config: QuantConfig,
}

impl EvolutionEngine {
    pub fn new(config: &QuantConfig) -> Self {
        Self {
            status: Arc::new(RwLock::new(EvolutionStatus::idle())),
            last_evolution: Arc::new(RwLock::new(None)),
            trade_count_since_last: Arc::new(RwLock::new(0)),
            config: config.clone(),
        }
    }

    /// Check if evolution should be triggered
    pub async fn should_evolve(&self) -> bool {
        let trade_count = *self.trade_count_since_last.read().await;
        if trade_count >= self.config.evolution.min_trades_before_evolution as u64 {
            return true;
        }

        if let Some(last) = *self.last_evolution.read().await {
            let elapsed = Utc::now() - last;
            if elapsed.num_hours() >= self.config.evolution.max_days_between_evolutions as i64 * 24 {
                return true;
            }
        } else {
            // Never evolved — trigger immediately
            return true;
        }

        false
    }

    /// Run a full evolution cycle
    pub async fn run_cycle(&self, trigger: &str) -> QuantResult<EvolutionStatus> {
        info!("=== Evolution Cycle Started (trigger: {}) ===", trigger);

        let mut status = self.status.write().await;
        status.phase = EvolutionPhase::DataUpdate;
        status.triggered_by = trigger.to_string();
        status.started_at = Some(Utc::now());
        status.cycle_number += 1;
        status.progress_pct = 0.0;
        drop(status);

        // Phase 1: Data Update (5-15 min)
        self.run_phase(EvolutionPhase::DataUpdate, 0.0, 0.15).await?;

        // Phase 2: Regime Re-calibration (10-30 min)
        self.run_phase(EvolutionPhase::RegimeRecalibration, 0.15, 0.30).await?;

        // Phase 3: Model Retraining (30 min - 2 hours)
        self.run_phase(EvolutionPhase::ModelRetraining, 0.30, 0.55).await?;

        // Phase 4: Lab Batch (2-6 hours)
        self.run_phase(EvolutionPhase::LabBatch, 0.55, 0.80).await?;

        // Phase 5: Strategy Pool Update (5 min)
        self.run_phase(EvolutionPhase::StrategyPoolUpdate, 0.80, 0.90).await?;

        // Phase 6: GitHub Commit (2 min)
        self.run_phase(EvolutionPhase::GitHubCommit, 0.90, 0.95).await?;

        // Phase 7: Resume Trading
        self.run_phase(EvolutionPhase::ResumeTrading, 0.95, 1.0).await?;

        // Mark complete
        let mut status = self.status.write().await;
        status.phase = EvolutionPhase::Idle;
        status.completed_at = Some(Utc::now());
        status.progress_pct = 1.0;
        *self.last_evolution.write().await = Some(Utc::now());
        *self.trade_count_since_last.write().await = 0;

        info!(
            "=== Evolution Cycle {} Complete ===",
            status.cycle_number
        );

        Ok(status.clone())
    }

    /// Run a single phase of the evolution cycle
    async fn run_phase(&self, phase: EvolutionPhase, start_pct: f64, end_pct: f64) -> QuantResult<()> {
        let phase_name = format!("{:?}", phase);
        info!("Phase: {} ({:.0}% - {:.0}%)", phase_name, start_pct * 100.0, end_pct * 100.0);

        let mut status = self.status.write().await;
        status.phase = phase;
        status.progress_pct = start_pct;
        drop(status);

        match phase {
            EvolutionPhase::DataUpdate => {
                // TODO: Incremental download from MT5
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::RegimeRecalibration => {
                // TODO: Re-fit GMM with new data
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::ModelRetraining => {
                // TODO: Warm-start retrain all models
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::LabBatch => {
                // TODO: Run lab batch
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::StrategyPoolUpdate => {
                // TODO: Re-weight strategies, retire underperformers
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::GitHubCommit => {
                // TODO: Commit all models, configs, logs
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::ResumeTrading => {
                // TODO: Load new models, update risk params
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            EvolutionPhase::Idle => {}
        }

        let mut status = self.status.write().await;
        status.progress_pct = end_pct;
        drop(status);

        info!("Phase {} complete", phase_name);
        Ok(())
    }

    /// Record a trade (increments counter for evolution triggering)
    pub async fn record_trade(&self) {
        let mut count = self.trade_count_since_last.write().await;
        *count += 1;
    }

    /// Get current evolution status
    pub async fn get_status(&self) -> EvolutionStatus {
        self.status.read().await.clone()
    }

    /// Force trigger an evolution cycle
    pub async fn force_evolve(&self) -> QuantResult<EvolutionStatus> {
        self.run_cycle("manual").await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_trigger() {
        let config = QuantConfig::default();
        let engine = EvolutionEngine::new(&config);
        
        // Should evolve on first check (never evolved)
        assert!(engine.should_evolve().await);
    }

    #[tokio::test]
    async fn test_evolution_cycle() {
        let config = QuantConfig::default();
        let engine = EvolutionEngine::new(&config);
        
        let status = engine.run_cycle("test").await.unwrap();
        assert_eq!(status.phase, EvolutionPhase::Idle);
        assert!(status.completed_at.is_some());
        assert_eq!(status.cycle_number, 1);
    }

    #[tokio::test]
    async fn test_trade_counting() {
        let config = QuantConfig::default();
        let engine = EvolutionEngine::new(&config);
        
        // Record trades
        for _ in 0..25 {
            engine.record_trade().await;
        }
        
        // Should trigger after 20 trades
        assert!(engine.should_evolve().await);
    }
}
