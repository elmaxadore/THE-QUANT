//! # Changepoint Detection Module (v3.0 "Prometheus")
//!
//! Uses Bayesian Online Changepoint Detection (BOCD) to detect structural
//! changes in the return distribution. Results feed the regime detector and
//! the changepoints persistence table.

pub mod bocd;

pub use bocd::{BocdConfig, BocdDetector, ChangepointResult};

/// The changepoint module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const CHANGEPOINT_MEMORY_BUDGET_PCT: f64 = 0.5;

/// The changepoint detector manager.
#[derive(Debug)]
pub struct ChangepointManager {
    pub detector: BocdDetector,
    pub enabled: bool,
}

impl ChangepointManager {
    pub fn new(config: BocdConfig) -> Self {
        Self {
            detector: BocdDetector::new(config),
            enabled: true,
        }
    }

    pub fn default() -> Self {
        Self::new(BocdConfig::default())
    }

    /// Process a return and return the changepoint result.
    pub fn update(&mut self, return_value: f64) -> ChangepointResult {
        self.detector.update(return_value)
    }

    /// Whether a changepoint is currently active.
    pub fn has_changepoint(&self) -> bool {
        self.detector.last_was_changepoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_update() {
        let mut mgr = ChangepointManager::default();
        let result = mgr.update(0.001);
        assert!(result.change_probability >= 0.0);
    }
}
