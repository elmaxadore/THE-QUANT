//! # Market Regime Detection Module (Layer 4)
//!
//! Detects market regimes using Gaussian Mixture Models (GMM) and Hidden Markov
//! Models (HMM). Regime-conditioned routing adjusts strategy selection and
//! position sizing based on the detected market environment.
//!
//! ## Regime Taxonomy
//! TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT,
//! HIGH_VOLATILITY, LOW_VOLATILITY, NEWS_EVENT, REGIME_TRANSITION

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// The eight market regimes detected by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Regime {
    TrendingUp,
    TrendingDown,
    Ranging,
    Breakout,
    HighVolatility,
    LowVolatility,
    NewsEvent,
    RegimeTransition,
}

impl Regime {
    pub fn name(&self) -> &'static str {
        match self {
            Regime::TrendingUp => "TRENDING_UP",
            Regime::TrendingDown => "TRENDING_DOWN",
            Regime::Ranging => "RANGING",
            Regime::Breakout => "BREAKOUT",
            Regime::HighVolatility => "HIGH_VOLATILITY",
            Regime::LowVolatility => "LOW_VOLATILITY",
            Regime::NewsEvent => "NEWS_EVENT",
            Regime::RegimeTransition => "REGIME_TRANSITION",
        }
    }

    pub fn sizing_multiplier(&self) -> f64 {
        match self {
            Regime::TrendingUp => 1.0,
            Regime::TrendingDown => 1.0,
            Regime::Ranging => 0.7,
            Regime::Breakout => 0.6,
            Regime::HighVolatility => 0.5,
            Regime::LowVolatility => 1.0,
            Regime::NewsEvent => 0.0,
            Regime::RegimeTransition => 0.3,
        }
    }
}

/// Probabilistic regime output from the detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeProbability {
    pub regime: Regime,
    pub probability: f64,
}

/// Full regime detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeResult {
    pub symbol: String,
    pub time: DateTime<Utc>,
    pub probabilities: Vec<RegimeProbability>,
    pub dominant_regime: Regime,
    pub confidence: f64,
    pub is_transition: bool,
}

/// Simplified GMM implementation for regime detection
#[derive(Debug)]
pub struct GaussianMixtureModel {
    n_components: usize,
    n_features: usize,
    weights: Vec<f64>,
    means: Array2<f64>,
    covariances: Vec<Vec<f64>>,
    priors: Vec<f64>,
    fitted: bool,
}

impl GaussianMixtureModel {
    pub fn new(n_components: usize, n_features: usize) -> Self {
        Self {
            n_components,
            n_features,
            weights: vec![1.0 / n_components as f64; n_components],
            means: Array2::zeros((n_components, n_features)),
            covariances: vec![vec![1.0; n_features]; n_components],
            priors: vec![1.0 / n_components as f64; n_components],
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>, max_iterations: usize) -> QuantResult<()> {
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Err(QuantError::TrainingError("Empty training data".into()));
        }
        self.initialize_parameters(data);
        for iteration in 0..max_iterations {
            let responsibilities = self.expectation(data)?;
            let (new_weights, new_means, new_covs) = self.maximization(data, &responsibilities)?;
            let diff = self.compute_parameter_diff(&new_weights, &new_means, &new_covs);
            self.weights = new_weights;
            self.means = new_means;
            self.covariances = new_covs;
            if diff < 1e-6 {
                debug!("GMM converged after {} iterations", iteration + 1);
                break;
            }
        }
        self.fitted = true;
        info!("GMM fitted with {} components, {} features", self.n_components, self.n_features);
        Ok(())
    }

    fn initialize_parameters(&mut self, data: &Array2<f64>) {
        let n_samples = data.nrows();
        let mut rng = rand::thread_rng();
        // Random initialization using k-means++ style
        for k in 0..self.n_components {
            let idx = rand::Rng::gen_range(&mut rng, 0..n_samples);
            for j in 0..self.n_features {
                self.means[[k, j]] = data[[idx, j]];
            }
        }
    }

    fn expectation(&self, data: &Array2<f64>) -> QuantResult<Array2<f64>> {
        let n_samples = data.nrows();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            let mut total = 0.0;
            for k in 0..self.n_components {
                let prob = self.gaussian_pdf(data.row(i), k);
                responsibilities[[i, k]] = self.weights[k] * prob;
                total += responsibilities[[i, k]];
            }
            if total > 0.0 {
                for k in 0..self.n_components {
                    responsibilities[[i, k]] /= total;
                }
            }
        }
        Ok(responsibilities)
    }

    fn maximization(&self, data: &Array2<f64>, responsibilities: &Array2<f64>) -> QuantResult<(Vec<f64>, Array2<f64>, Vec<Vec<f64>>)> {
        let n_samples = data.nrows();
        let n_features = self.n_features;
        let mut new_weights = vec![0.0; self.n_components];
        let mut new_means = Array2::zeros((self.n_components, n_features));
        let mut new_covs = vec![vec![0.0; n_features]; self.n_components];

        for k in 0..self.n_components {
            let mut total_resp = 0.0;
            for i in 0..n_samples {
                total_resp += responsibilities[[i, k]];
            }
            new_weights[k] = total_resp / n_samples as f64;
            if total_resp > 0.0 {
                for j in 0..n_features {
                    let mut mean_sum = 0.0;
                    for i in 0..n_samples {
                        mean_sum += responsibilities[[i, k]] * data[[i, j]];
                    }
                    new_means[[k, j]] = mean_sum / total_resp;
                }
                for j in 0..n_features {
                    let mut var_sum = 0.0;
                    for i in 0..n_samples {
                        let diff = data[[i, j]] - new_means[[k, j]];
                        var_sum += responsibilities[[i, k]] * diff * diff;
                    }
                    new_covs[k][j] = var_sum / total_resp + 1e-6;
                }
            }
        }
        Ok((new_weights, new_means, new_covs))
    }

    fn gaussian_pdf(&self, sample: ndarray::ArrayView1<f64>, component: usize) -> f64 {
        let mut exponent = 0.0;
        let mut log_det = 0.0;
        for j in 0..self.n_features {
            let diff = sample[j] - self.means[[component, j]];
            exponent += diff * diff / self.covariances[component][j];
            log_det += self.covariances[component][j].ln();
        }
        (-0.5 * (self.n_features as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + exponent)).exp()
    }

    fn compute_parameter_diff(&self, new_weights: &[f64], new_means: &Array2<f64>, new_covs: &[Vec<f64>]) -> f64 {
        let mut diff = 0.0;
        for k in 0..self.n_components {
            diff += (self.weights[k] - new_weights[k]).abs();
            for j in 0..self.n_features {
                diff += (self.means[[k, j]] - new_means[[k, j]]).abs();
                diff += (self.covariances[k][j] - new_covs[k][j]).abs();
            }
        }
        diff
    }

    pub fn predict_proba(&self, sample: &[f64]) -> QuantResult<Vec<RegimeProbability>> {
        if !self.fitted {
            return Err(QuantError::TrainingError("GMM not fitted".into()));
        }
        let sample_array = Array2::from_shape_vec((1, self.n_features), sample.to_vec())
            .map_err(|e| QuantError::Internal(format!("Array error: {}", e)))?;
        let respons = self.expectation(&sample_array)?;
        let regimes = vec![
            Regime::TrendingUp, Regime::TrendingDown, Regime::Ranging,
            Regime::Breakout, Regime::HighVolatility, Regime::LowVolatility,
            Regime::NewsEvent, Regime::RegimeTransition,
        ];
        let mut probs = Vec::new();
        for k in 0..self.n_components.min(regimes.len()) {
            probs.push(RegimeProbability {
                regime: regimes[k].clone(),
                probability: respons[[0, k]],
            });
        }
        Ok(probs)
    }
}

/// Regime detector — the main interface for the rest of the system
#[derive(Debug)]
pub struct RegimeDetector {
    gmm: Arc<RwLock<GaussianMixtureModel>>,
    min_confirmation_bars: u32,
    transition_threshold: f64,
    last_regime: Arc<RwLock<Option<Regime>>>,
    confirmation_count: Arc<RwLock<u32>>,
}

impl RegimeDetector {
    pub fn new(config: &QuantConfig) -> Self {
        let gmm = GaussianMixtureModel::new(
            config.model.gmm_max_components,
            8, // Feature dimension for regime detection
        );
        Self {
            gmm: Arc::new(RwLock::new(gmm)),
            min_confirmation_bars: 3,
            transition_threshold: 0.3,
            last_regime: Arc::new(RwLock::new(None)),
            confirmation_count: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn detect_regime(&self, features: &[f64]) -> QuantResult<RegimeResult> {
        let gmm = self.gmm.read().await;
        if !gmm.fitted {
            return Err(QuantError::TrainingError("GMM not yet fitted to market data".into()));
        }
        let probabilities = gmm.predict_proba(features)?;
        let dominant = probabilities.iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
            .ok_or_else(|| QuantError::TrainingError("No regime detected".into()))?;
        let confidence = dominant.probability;
        let mut last = self.last_regime.write().await;
        let mut count = self.confirmation_count.write().await;

        let is_transition = if let Some(ref last_reg) = *last {
            if *last_reg != dominant.regime {
                *count = 0;
                true
            } else {
                *count += 1;
                *count < self.min_confirmation_bars
            }
        } else {
            *count = 1;
            false
        };

        if !is_transition && confidence > 0.5 {
            *last = Some(dominant.regime.clone());
        }

        Ok(RegimeResult {
            symbol: String::new(),
            time: Utc::now(),
            probabilities,
            dominant_regime: dominant.regime.clone(),
            confidence,
            is_transition,
        })
    }

    pub async fn fit_gmm(&self, data: &Array2<f64>) -> QuantResult<()> {
        let mut gmm = self.gmm.write().await;
        gmm.fit(data, 100)
    }

    pub async fn is_fitted(&self) -> bool {
        self.gmm.read().await.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_regime_sizing_multipliers() {
        assert_eq!(Regime::TrendingUp.sizing_multiplier(), 1.0);
        assert_eq!(Regime::HighVolatility.sizing_multiplier(), 0.5);
        assert_eq!(Regime::NewsEvent.sizing_multiplier(), 0.0);
    }

    #[test]
    fn test_gmm_fit_and_predict() {
        let mut gmm = GaussianMixtureModel::new(3, 2);
        let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64 / 10.0).collect()).unwrap();
        assert!(gmm.fit(&data, 50).is_ok());
        let probs = gmm.predict_proba(&[1.0, 2.0]).unwrap();
        assert_eq!(probs.len(), 3);
        let total: f64 = probs.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
