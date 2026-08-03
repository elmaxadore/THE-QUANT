//! # Anomaly Detection Module (v3.0 "Prometheus")
//!
//! Detects anomalous market conditions using Isolation Forest. Anomalies are
//! flagged for the risk engine to suppress signals and for the anomaly_scores
//! persistence table.

pub mod isolation_forest;

pub use isolation_forest::{AnomalyDetector, AnomalyResult, IsolationForest};

/// The anomaly module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const ANOMALY_MEMORY_BUDGET_PCT: f64 = 1.0;

/// The anomaly detector manager.
#[derive(Debug)]
pub struct AnomalyManager {
    pub detector: Option<AnomalyDetector>,
    pub enabled: bool,
}

impl AnomalyManager {
    pub fn new() -> Self {
        Self {
            detector: None,
            enabled: false,
        }
    }

    /// Build a detector from fitted data.
    pub fn build_detector(&mut self, feature_names: Vec<String>, n_trees: usize, sample_size: usize, max_depth: usize) {
        let forest = IsolationForest::new(n_trees, sample_size, max_depth);
        self.detector = Some(AnomalyDetector::new(forest, feature_names));
        self.enabled = true;
    }

    /// Analyze a feature vector for anomalies.
    pub fn analyze(&self, sample: &[f64]) -> Option<AnomalyResult> {
        self.detector.as_ref().map(|d| d.analyze(sample))
    }

    /// Whether anomaly detection is active and blocking signals.
    pub fn is_active(&self) -> bool {
        self.enabled && self.detector.is_some()
    }
}

impl Default for AnomalyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_manager() {
        let mut mgr = AnomalyManager::new();
        assert!(!mgr.is_active());
        mgr.build_detector(vec!["f0".into()], 20, 10, 4);
        assert!(mgr.is_active());
        let result = mgr.analyze(&[0.0]);
        assert!(result.is_some());
    }
}
