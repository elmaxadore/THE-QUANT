//! # Bayesian Online Changepoint Detection (BOCD)
//!
//! Detects structural changes in the return distribution of a price series.
//! When a changepoint is detected, the regime detector re-calibrates and the
//! strategy engine may pause or re-route signals.
//!
//! Implements the Adams & MacKay (2007) algorithm (BOCPD) with a Student-t
//! predictive distribution for returns.

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};

/// Parameters for the BOCD model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BocdConfig {
    /// Hazard rate (probability of a changepoint at each step).
    pub hazard: f64,
    /// Prior degrees of freedom for the Student-t predictive.
    pub nu: f64,
    /// Prior variance scale.
    pub sigma2: f64,
    /// Prior mean.
    pub mu: f64,
    /// Prior precision (kappa).
    pub kappa: f64,
    /// Maximum run length to track.
    pub max_run_length: usize,
}

impl Default for BocdConfig {
    fn default() -> Self {
        Self {
            hazard: 0.01,
            nu: 3.0,
            sigma2: 0.01,
            mu: 0.0,
            kappa: 1.0,
            max_run_length: 250,
        }
    }
}

/// A single changepoint detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangepointResult {
    pub run_length: usize,
    pub change_probability: f64,
    pub is_changepoint: bool,
    pub z_score: f64,
}

/// The BOCD detector.
#[derive(Debug)]
pub struct BocdDetector {
    config: BocdConfig,
    /// Run length posterior probabilities.
    run_length_probs: Vec<f64>,
    /// Sufficient statistics for each run length.
    /// (count, mean, second_moment)
    stats: Vec<(f64, f64, f64)>,
    /// Total observations processed.
    observations: u64,
    /// Whether a changepoint was detected on the last step.
    last_changepoint: bool,
}

impl BocdDetector {
    pub fn new(config: BocdConfig) -> Self {
        let max_run = config.max_run_length.max(1);
        Self {
            run_length_probs: vec![1.0],
            stats: vec![(0.0, 0.0, 0.0)],
            config,
            observations: 0,
            last_changepoint: false,
        }
    }

    pub fn default() -> Self {
        Self::new(BocdConfig::default())
    }

    /// Process a new observation (return value) and update the run-length distribution.
    pub fn update(&mut self, value: f64) -> ChangepointResult {
        self.observations += 1;
        let hazard = self.config.hazard;
        let max_run = self.config.max_run_length;

        // Current run length
        let t = self.run_length_probs.len();
        let new_run_len = t.min(max_run);

        // Predictive probability for each run length
        let mut pred_probs = vec![0.0; new_run_len];
        for r in 0..new_run_len {
            pred_probs[r] = self.student_t_pdf(value, &self.stats[r]);
        }

        // Growth probability
        let mut growth_probs = vec![0.0; new_run_len];
        for r in 0..new_run_len {
            growth_probs[r] = self.run_length_probs[r] * pred_probs[r] * (1.0 - hazard);
        }

        // Changepoint probability (reset to run length 0)
        let mut cp_prob = 0.0;
        for r in 0..new_run_len {
            cp_prob += self.run_length_probs[r] * pred_probs[r] * hazard;
        }

        // New run-length posterior
        let mut new_probs = vec![0.0; new_run_len + 1];
        new_probs[0] = cp_prob;
        for r in 0..new_run_len {
            new_probs[r + 1] = growth_probs[r];
        }

        // Normalize
        let total: f64 = new_probs.iter().sum();
        if total > 0.0 {
            for p in new_probs.iter_mut() {
                *p /= total;
            }
        }

        // Update sufficient statistics
        let mut new_stats = vec![(0.0, 0.0, 0.0); new_run_len + 1];
        // Run length 0: fresh start
        new_stats[0] = (1.0, value, value * value);
        for r in 0..new_run_len {
            let (count, mean, m2) = self.stats[r];
            let new_count = count + 1.0;
            let delta = value - mean;
            let new_mean = mean + delta / new_count;
            let new_m2 = m2 + delta * (value - new_mean);
            new_stats[r + 1] = (new_count, new_mean, new_m2);
        }

        // Truncate to max_run_length
        if new_stats.len() > max_run {
            new_stats.truncate(max_run);
            new_probs.truncate(max_run);
        }
        // Handle the case where truncation drops the changepoint index
        if new_probs.len() > 1 && new_probs[0] < 0.5 {
            // ensure the array has at least index for the changepoint
        }

        self.run_length_probs = new_probs;
        self.stats = new_stats;

        // Determine most likely run length
        let max_idx = self.run_length_probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let change_probability = self.run_length_probs[0];
        let is_changepoint = change_probability > 0.5;
        self.last_changepoint = is_changepoint;

        // Compute a z-score of the recent observation
        let z_score = self.compute_z_score(value);

        ChangepointResult {
            run_length: max_idx,
            change_probability,
            is_changepoint,
            z_score,
        }
    }

    /// Student-t predictive distribution for a new observation.
    fn student_t_pdf(&self, value: f64, stats: &(f64, f64, f64)) -> f64 {
        let (count, mean, m2) = *stats;
        if count == 0.0 {
            // Prior: use the prior parameters
            return self.prior_pdf(value);
        }
        let nu = self.config.nu + count;
        let kappa = self.config.kappa + count;
        let mu = mean;
        let var = if count > 1.0 {
            m2 / (count - 1.0)
        } else {
            self.config.sigma2
        };
        let scale = var * (1.0 + 1.0 / kappa);

        let z = (value - mu) / scale.sqrt();
        let t = (nu).powf(0.5) * (1.0 + z * z / nu).powf(-(nu + 1.0) / 2.0);
        let norm = (nu * std::f64::consts::PI).sqrt() * Beta::beta(nu / 2.0, 0.5);
        let pdf = t / norm;
        pdf.max(1e-12)
    }

    fn prior_pdf(&self, value: f64) -> f64 {
        // Use the prior Student-t
        let nu = self.config.nu;
        let mu = self.config.mu;
        let scale = self.config.sigma2.sqrt();
        let z = (value - mu) / scale;
        let t = (nu).powf(0.5) * (1.0 + z * z / nu).powf(-(nu + 1.0) / 2.0);
        let norm = (nu * std::f64::consts::PI).sqrt() * Beta::beta(nu / 2.0, 0.5);
        (t / norm).max(1e-12)
    }

    fn compute_z_score(&self, value: f64) -> f64 {
        let (count, mean, m2) = self.stats[0];
        if count < 2.0 {
            return 0.0;
        }
        let var = m2 / (count - 1.0);
        if var <= 0.0 {
            return 0.0;
        }
        (value - mean) / var.sqrt()
    }

    /// Most likely run length since the last changepoint.
    pub fn most_likely_run_length(&self) -> usize {
        self.run_length_probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Total observations processed.
    pub fn observation_count(&self) -> u64 {
        self.observations
    }

    /// Whether the last update flagged a changepoint.
    pub fn last_was_changepoint(&self) -> bool {
        self.last_changepoint
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.run_length_probs = vec![1.0];
        self.stats = vec![(0.0, 0.0, 0.0)];
        self.observations = 0;
        self.last_changepoint = false;
    }
}

/// Minimal beta function helper.
struct Beta;

impl Beta {
    fn beta(a: f64, b: f64) -> f64 {
        // Using the gamma function approximation
        let ln_gamma_a = ln_gamma(a);
        let ln_gamma_b = ln_gamma(b);
        let ln_gamma_ab = ln_gamma(a + b);
        (ln_gamma_a + ln_gamma_b - ln_gamma_ab).exp()
    }
}

/// Lanczos approximation of the natural log of the gamma function.
fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        let t = x + g + 0.5;
        for i in 1..9 {
            a += c[i] / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_changepoint_stable() {
        let mut detector = BocdDetector::default();
        let mut last_result = None;
        for i in 0..50 {
            let value = 0.001 * (i as f64);
            last_result = Some(detector.update(value));
        }
        let result = last_result.unwrap();
        assert!(!result.is_changepoint);
    }

    #[test]
    fn test_changepoint_detection() {
        let mut detector = BocdDetector::new(BocdConfig {
            hazard: 0.05,
            max_run_length: 50,
            ..Default::default()
        });
        // Stable regime
        for _ in 0..30 {
            detector.update(0.001);
        }
        // Shift in mean
        let mut detected = false;
        for _ in 0..30 {
            let result = detector.update(0.3);
            if result.is_changepoint {
                detected = true;
                break;
            }
        }
        assert!(detected);
    }

    #[test]
    fn test_reset() {
        let mut detector = BocdDetector::default();
        for _ in 0..10 {
            detector.update(0.001);
        }
        assert!(detector.observation_count() > 0);
        detector.reset();
        assert_eq!(detector.observation_count(), 0);
    }

    #[test]
    fn test_ln_gamma() {
        // ln(1) = 0, ln(2) = ln(2)
        assert!((ln_gamma(1.0)).abs() < 1e-6);
        assert!((ln_gamma(2.0) - 2.0f64.ln()).abs() < 1e-6);
    }
}
