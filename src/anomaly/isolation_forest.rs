//! # Isolation Forest Anomaly Detection
//!
//! A self-contained implementation of the Isolation Forest algorithm for
//! detecting anomalous market events (flash crashes, flash spikes, liquidity
//! collapses, data glitches). It is used to:
//!   - Flag anomalous ticks/bars for the risk engine
//!   - Suppress signals generated during anomalous conditions
//!   - Feed the anomaly_scores persistence table

use crate::error::{QuantError, QuantResult};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single node in an isolation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TreeNode {
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf {
        size: usize,
    },
}

/// An isolation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationTree {
    root: Option<TreeNode>,
    max_depth: usize,
}

impl IsolationTree {
    /// Build a tree from a subsample of data.
    fn build(data: &[Vec<f64>], indices: &[usize], depth: usize, max_depth: usize, feature_count: usize) -> Self {
        let root = if indices.len() <= 1 || depth >= max_depth {
            Some(TreeNode::Leaf { size: indices.len() })
        } else {
            Self::build_internal(data, indices, depth, max_depth, feature_count)
        };
        Self { root, max_depth }
    }

    fn build_internal(
        data: &[Vec<f64>],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        feature_count: usize,
    ) -> TreeNode {
        let mut rng = rand::thread_rng();

        // Choose a random feature
        let feature = rng.gen_range(0..feature_count);

        // Collect values for this feature across the sample
        let values: Vec<f64> = indices.iter().map(|&i| data[i][feature]).collect();
        let (min_v, max_v) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });

        if max_v == min_v {
            return TreeNode::Leaf { size: indices.len() };
        }

        // Random split point between min and max
        let threshold = rng.gen_range(min_v..max_v);

        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        for &i in indices {
            if data[i][feature] < threshold {
                left_idx.push(i);
            } else {
                right_idx.push(i);
            }
        }

        if left_idx.is_empty() || right_idx.is_empty() {
            return TreeNode::Leaf { size: indices.len() };
        }

        let left = Box::new(Self::build(data, &left_idx, depth + 1, max_depth, feature_count).root.unwrap());
        let right = Box::new(Self::build(data, &right_idx, depth + 1, max_depth, feature_count).root.unwrap());

        TreeNode::Internal {
            feature,
            threshold,
            left,
            right,
        }
    }

    /// Compute the path length of a sample through the tree.
    fn path_length(&self, sample: &[f64]) -> f64 {
        let mut node = self.root.as_ref();
        let mut depth = 0.0;
        while let Some(n) = node {
            match n {
                TreeNode::Internal { feature, threshold, left, right } => {
                    if sample[*feature] < *threshold {
                        node = Some(left.as_ref());
                    } else {
                        node = Some(right.as_ref());
                    }
                    depth += 1.0;
                }
                TreeNode::Leaf { size } => {
                    // External node term: average path length of a random BST
                    let c = if *size <= 1 {
                        0.0
                    } else {
                        2.0 * (size - 1).ln() + 0.5772156649 - (2.0 * (size - 1) as f64) / (*size as f64)
                    };
                    return depth + c;
                }
            }
        }
        depth
    }
}

/// The Isolation Forest model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    n_trees: usize,
    sample_size: usize,
    max_depth: usize,
    /// Mean path length for normalization (computed at fit time).
    mean_path_length: f64,
    /// Auto-detected anomaly threshold.
    threshold: f64,
    /// Number of features expected.
    feature_count: usize,
    fitted: bool,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            trees: Vec::new(),
            n_trees: 100,
            sample_size: 256,
            max_depth: 8,
            mean_path_length: 0.0,
            threshold: 0.5,
            feature_count: 0,
            fitted: false,
        }
    }
}

impl IsolationForest {
    pub fn new(n_trees: usize, sample_size: usize, max_depth: usize) -> Self {
        Self {
            n_trees,
            sample_size,
            max_depth,
            ..Default::default()
        }
    }

    /// Fit the forest on a dataset.
    pub fn fit(&mut self, data: &[Vec<f64>]) -> QuantResult<()> {
        if data.is_empty() {
            return Err(QuantError::TrainingError("Empty training data".into()));
        }
        self.feature_count = data[0].len();
        let n = data.len();

        let mut rng = rand::thread_rng();
        self.trees.clear();

        for _ in 0..self.n_trees {
            // Sample a random subsample
            let sample_size = self.sample_size.min(n);
            let mut indices: Vec<usize> = (0..n).collect();
            // Shuffle and take first sample_size
            for i in (0..n).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);

            let tree = IsolationTree::build(data, &indices, 0, self.max_depth, self.feature_count);
            self.trees.push(tree);
        }

        // Average path length of a random BST with n nodes
        let c = self.average_path_length(n);
        self.mean_path_length = c;
        self.fitted = true;

        // Auto-set threshold from the data (e.g., top 5% anomaly scores)
        let scores: Vec<f64> = data.iter().map(|sample| self.inlier_score(sample)).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() as f64 * 0.05) as usize;
        self.threshold = sorted.get(idx).copied().unwrap_or(0.5);

        Ok(())
    }

    fn average_path_length(&self, n: usize) -> f64 {
        if n <= 1 {
            0.0
        } else {
            2.0 * (n - 1).ln() + 0.5772156649 - (2.0 * (n - 1) as f64) / (n as f64)
        }
    }

    /// Compute the anomaly score. 0 = normal, 1 = anomalous.
    pub fn anomaly_score(&self, sample: &[f64]) -> f64 {
        if !self.fitted || self.trees.is_empty() {
            return 0.0;
        }
        let avg_path = self.trees.iter()
            .map(|t| t.path_length(sample))
            .sum::<f64>() / self.trees.len() as f64;

        let n = self.sample_size;
        let c = self.average_path_length(n);
        let score = 2.0_f64.powf(-(avg_path / c.max(1e-9)));
        score.clamp(0.0, 1.0)
    }

    /// Inlier score (1 - anomaly score), used for threshold calibration.
    fn inlier_score(&self, sample: &[f64]) -> f64 {
        1.0 - self.anomaly_score(sample)
    }

    /// Check if a sample is anomalous.
    pub fn is_anomaly(&self, sample: &[f64]) -> bool {
        self.anomaly_score(sample) >= self.threshold
    }

    /// Get the current anomaly threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Serialize the forest to JSON.
    pub fn to_json(&self) -> QuantResult<String> {
        serde_json::to_string(self).map_err(|e| QuantError::Internal(e.to_string()))
    }

    /// Deserialize a forest from JSON.
    pub fn from_json(json: &str) -> QuantResult<Self> {
        serde_json::from_str(json).map_err(|e| QuantError::Internal(e.to_string()))
    }
}

/// An anomaly detection result for a single sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub score: f64,
    pub is_anomaly: bool,
    pub threshold: f64,
    pub contributing_features: Vec<(String, f64)>,
}

/// The anomaly detector wrapper.
#[derive(Debug)]
pub struct AnomalyDetector {
    forest: IsolationForest,
    /// Feature name mapping for interpretability.
    feature_names: Vec<String>,
}

impl AnomalyDetector {
    pub fn new(forest: IsolationForest, feature_names: Vec<String>) -> Self {
        Self { forest, feature_names }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) -> QuantResult<()> {
        self.forest.fit(data)
    }

    /// Analyze a sample and return a structured result.
    pub fn analyze(&self, sample: &[f64]) -> AnomalyResult {
        let score = self.forest.anomaly_score(sample);
        let is_anomaly = self.forest.is_anomaly(sample);

        // Determine contributing features (highest deviations)
        let mut features: Vec<(String, f64)> = Vec::new();
        for (i, name) in self.feature_names.iter().enumerate() {
            if i < sample.len() {
                features.push((name.clone(), sample[i].abs()));
            }
        }
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        features.truncate(5);

        AnomalyResult {
            score,
            is_anomaly,
            threshold: self.forest.threshold(),
            contributing_features: features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_and_score() {
        let mut forest = IsolationForest::new(50, 32, 6);
        // Normal data around 0
        let data: Vec<Vec<f64>> = (0..200).map(|i| {
            vec![
                (i as f64 * 0.01).sin() * 0.1,
                (i as f64 * 0.02).cos() * 0.1,
            ]
        }).collect();
        forest.fit(&data).unwrap();
        assert!(forest.is_fitted());

        // Normal sample
        let normal_score = forest.anomaly_score(&[0.0, 0.0]);
        // Anomalous sample (far outside)
        let anomaly_score = forest.anomaly_score(&[10.0, -10.0]);
        assert!(anomaly_score > normal_score);
    }

    #[test]
    fn test_threshold_detection() {
        let mut forest = IsolationForest::new(50, 32, 6);
        let mut data: Vec<Vec<f64>> = (0..200).map(|i| vec![(i as f64 * 0.01).sin() * 0.1]).collect();
        // Inject anomalies
        data.push(vec![20.0]);
        data.push(vec![-20.0]);
        forest.fit(&data).unwrap();

        assert!(forest.is_anomaly(&[20.0]));
        assert!(!forest.is_anomaly(&[0.0]));
    }

    #[test]
    fn test_json_roundtrip() {
        let forest = IsolationForest::new(10, 10, 5);
        let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 10.0]).collect();
        let mut forest = forest;
        forest.fit(&data).unwrap();
        let json = forest.to_json().unwrap();
        let loaded = IsolationForest::from_json(&json).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_anomaly_detector() {
        let mut forest = IsolationForest::new(50, 32, 6);
        let data: Vec<Vec<f64>> = (0..200).map(|i| vec![(i as f64 * 0.01).sin() * 0.1]).collect();
        forest.fit(&data).unwrap();
        let detector = AnomalyDetector::new(forest, vec!["feature_0".into()]);
        let result = detector.analyze(&[0.0]);
        assert!(!result.contributing_features.is_empty());
    }
}
