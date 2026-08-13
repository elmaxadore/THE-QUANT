//! # Bayesian Optimization
//!
//! A lightweight Bayesian optimization implementation for hyperparameter
//! tuning of models and strategies. Uses a Gaussian Process surrogate with
//! Expected Improvement (EI) acquisition.
//!
//! This is a *self-contained* implementation (no external GP library) that
//! scales to small/medium parameter spaces.

use crate::error::{QuantError, QuantResult};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A candidate parameter point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamPoint {
    /// Normalized parameters in [0, 1].
    pub params: Vec<f64>,
    /// Observed objective value (higher is better).
    pub value: f64,
}

/// A bounded parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpace {
    /// Lower bounds for each parameter.
    pub lower: Vec<f64>,
    /// Upper bounds for each parameter.
    pub upper: Vec<f64>,
}

impl ParamSpace {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(lower.len(), upper.len(), "bounds must have equal length");
        Self { lower, upper }
    }

    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    /// Normalize a raw parameter value to [0, 1].
    pub fn normalize(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter().enumerate().map(|(i, v)| {
            let span = self.upper[i] - self.lower[i];
            if span > 0.0 {
                ((v - self.lower[i]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }).collect()
    }

    /// Denormalize from [0, 1] to the raw range.
    pub fn denormalize(&self, norm: &[f64]) -> Vec<f64> {
        norm.iter().enumerate().map(|(i, v)| {
            self.lower[i] + v * (self.upper[i] - self.lower[i])
        }).collect()
    }
}

/// A simple Gaussian Process surrogate.
#[derive(Debug, Clone)]
struct GaussianProcess {
    /// Observed points (normalized).
    points: Vec<Vec<f64>>,
    /// Observed values.
    values: Vec<f64>,
    /// Length scale (squared exponential kernel).
    length_scale: f64,
    /// Signal variance.
    signal_var: f64,
    /// Noise variance.
    noise_var: f64,
}

impl GaussianProcess {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            values: Vec::new(),
            length_scale: 0.5,
            signal_var: 1.0,
            noise_var: 1e-4,
        }
    }

    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
        self.signal_var * (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }

    /// Compute the posterior mean and variance at a query point.
    fn predict(&self, query: &[f64]) -> (f64, f64) {
        let n = self.points.len();
        if n == 0 {
            return (0.0, self.signal_var);
        }

        // Build K matrix
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel(&self.points[i], &self.points[j]);
            }
            k[i][i] += self.noise_var;
        }

        // Solve K\u2081·k = K⁻¹ (simplified: use a small linear solve)
        // For a self-contained impl, invert via Gaussian elimination
        let k_inv = invert_matrix(&k);

        // k* vector
        let k_star: Vec<f64> = self.points.iter().map(|p| self.kernel(p, query)).collect();

        // mean = k*ᵀ K⁻¹ y
        let mut mean = 0.0;
        for i in 0..n {
            for j in 0..n {
                mean += k_star[i] * k_inv[i][j] * self.values[j];
            }
        }

        // var = k(x,x) - k*ᵀ K⁻¹ k*
        let mut var = self.signal_var;
        for i in 0..n {
            for j in 0..n {
                var -= k_star[i] * k_inv[i][j] * k_star[j];
            }
        }

        (mean, var.max(1e-9))
    }

    fn add(&mut self, point: Vec<f64>, value: f64) {
        self.points.push(point);
        self.values.push(value);
    }
}

/// Gaussian elimination matrix inversion.
fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    // Augment with identity
    let mut aug: Vec<Vec<f64>> = a.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.extend(std::iter::repeat(0.0).take(n));
        r[n + i] = 1.0;
        r
    }).collect();

    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        for row in col + 1..n {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        aug.swap(col, pivot);

        let pivot_val = aug[col][col];
        if pivot_val.abs() < 1e-12 {
            continue;
        }
        for j in 0..2 * n {
            aug[col][j] /= pivot_val;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    aug.iter().map(|row| row[n..].to_vec()).collect()
}

/// Expected Improvement acquisition function.
fn expected_improvement(gp: &GaussianProcess, query: &[f64], best_value: f64) -> f64 {
    let (mean, var) = gp.predict(query);
    let std = var.sqrt();
    if std <= 0.0 {
        return 0.0;
    }
    let z = (mean - best_value) / std;
    let phi = normal_pdf(z);
    let cdf = normal_cdf(z);
    (mean - best_value) * cdf + std * phi
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(z: f64) -> f64 {
    // Abramowitz-Stegun approximation
    0.5 * (1.0 + (z / std::f64::consts::SQRT_2).erf())
}

/// The Bayesian optimizer.
#[derive(Debug)]
pub struct BayesianOptimizer {
    gp: GaussianProcess,
    space: ParamSpace,
    best_value: f64,
    best_params: Vec<f64>,
    iterations: u64,
}

impl BayesianOptimizer {
    pub fn new(space: ParamSpace) -> Self {
        Self {
            gp: GaussianProcess::new(),
            space,
            best_value: f64::NEG_INFINITY,
            best_params: vec![0.0; space.dim()],
            iterations: 0,
        }
    }

    /// Record an observed evaluation.
    pub fn observe(&mut self, raw_params: &[f64], value: f64) {
        let norm = self.space.normalize(raw_params);
        self.gp.add(norm, value);
        if value > self.best_value {
            self.best_value = value;
            self.best_params = raw_params.to_vec();
        }
        self.iterations += 1;
    }

    /// Suggest the next parameter point to evaluate.
    pub fn suggest(&self) -> Vec<f64> {
        // Random candidate sampling + EI optimization (simple multi-start)
        let n_candidates = 100;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_norm = vec![0.0; self.space.dim()];
        let mut rng = rand::thread_rng();

        for _ in 0..n_candidates {
            let candidate: Vec<f64> = (0..self.space.dim())
                .map(|_| rng.gen_range(0.0..1.0))
                .collect();
            let score = expected_improvement(&self.gp, &candidate, self.best_value);
            if score > best_score {
                best_score = score;
                best_norm = candidate;
            }
        }

        self.space.denormalize(&best_norm)
    }

    /// Current best value found.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Current best parameters (raw space).
    pub fn best_params(&self) -> &[f64] {
        &self.best_params
    }

    /// Number of iterations performed.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Whether the optimizer has any observations.
    pub fn has_observations(&self) -> bool {
        self.iterations > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_finds_peak() {
        // Maximize f(x) = -(x-0.5)^2 on [0,1]
        let space = ParamSpace::new(vec![0.0], vec![1.0]);
        let mut opt = BayesianOptimizer::new(space);

        // Initial samples
        for x in [0.0, 0.2, 0.8, 1.0] {
            let value = -(x - 0.5).powi(2);
            opt.observe(&[x], value);
        }

        // Suggest should be near the peak (0.5)
        let suggestion = opt.suggest();
        assert!(suggestion[0] > 0.3 && suggestion[0] < 0.7);
    }

    #[test]
    fn test_optimizer_tracks_best() {
        let space = ParamSpace::new(vec![0.0], vec![10.0]);
        let mut opt = BayesianOptimizer::new(space);
        opt.observe(&[1.0], 0.1);
        opt.observe(&[5.0], 0.9);
        opt.observe(&[9.0], 0.5);
        assert!((opt.best_params()[0] - 5.0).abs() < 1e-6);
        assert!((opt.best_value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_param_space_roundtrip() {
        let space = ParamSpace::new(vec![0.0, 100.0], vec![10.0, 200.0]);
        let norm = space.normalize(&[5.0, 150.0]);
        assert!((norm[0] - 0.5).abs() < 1e-6);
        assert!((norm[1] - 0.5).abs() < 1e-6);
        let raw = space.denormalize(&norm);
        assert!((raw[0] - 5.0).abs() < 1e-6);
        assert!((raw[1] - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_inversion() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let inv = invert_matrix(&a);
        assert!((inv[0][0] - 0.5).abs() < 1e-6);
        assert!((inv[1][1] - 0.25).abs() < 1e-6);
    }
}
