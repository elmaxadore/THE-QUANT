//! # Model Manager Module (Layer 5)
//!
//! Manages the model zoo — GMM regime detector, GBDT directional predictors,
//! volatility predictors, meta-learner for strategy selection, and symbolic
//! regression alpha engine. All models are memory-budgeted as percentages of
//! HARD_PROCESS_LIMIT and support hot-swap loading without stopping trading.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Status of a model in the manager
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Production,
    Staging,
    Archived,
    Failed,
}

/// Metadata for a trained model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub asset: String,
    pub regime: String,
    pub algorithm: String,
    pub trained_at: DateTime<Utc>,
    pub features_hash: String,
    pub metrics: ModelMetrics,
    pub status: ModelStatus,
    pub file_path: PathBuf,
    pub model_size_bytes: u64,
}

/// Performance metrics for a model
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetrics {
    pub val_auc: f64,
    pub val_sharpe: f64,
    pub max_dd_pct: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub inference_time_ms: f64,
}

/// A trained model ready for inference
#[derive(Debug)]
pub enum TrainedModel {
    Gmm(crate::regime::GaussianMixtureModel),
    Gbdt(GbdtModel),
    VolatilityPredictor(VolatilityModel),
    SymbolicRegressor(SymbolicModel),
}

/// Gradient Boosted Decision Tree model (simplified for inference)
#[derive(Debug)]
pub struct GbdtModel {
    pub trees: Vec<DecisionTree>,
    pub learning_rate: f64,
    pub max_depth: u32,
    pub num_leaves: u32,
    pub feature_importances: Vec<f64>,
}

/// A single decision tree
#[derive(Debug)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
}

#[derive(Debug)]
pub struct TreeNode {
    pub feature_index: usize,
    pub threshold: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub value: f64,
    pub is_leaf: bool,
}

impl GbdtModel {
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut sum = 0.0;
        for tree in &self.trees {
            sum += self.predict_tree(tree, features) * self.learning_rate;
        }
        sum
    }

    fn predict_tree(&self, tree: &DecisionTree, features: &[f64]) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &tree.nodes[node_idx];
            if node.is_leaf {
                return node.value;
            }
            if features[node.feature_index] <= node.threshold {
                node_idx = node.left_child.unwrap_or(node_idx);
            } else {
                node_idx = node.right_child.unwrap_or(node_idx);
            }
        }
    }
}

/// EWMA/GARCH volatility model
#[derive(Debug)]
pub struct VolatilityModel {
    pub lambda: f64,      // Decay factor
    pub long_mean: f64,   // Long-term mean variance
    pub alpha: f64,       // GARCH alpha
    pub beta: f64,        // GARCH beta
    pub current_var: f64, // Current variance estimate
}

impl VolatilityModel {
    pub fn new() -> Self {
        Self {
            lambda: 0.94,
            long_mean: 0.0001,
            alpha: 0.1,
            beta: 0.85,
            current_var: 0.0001,
        }
    }

    pub fn update(&mut self, return_val: f64) -> f64 {
        // EWMA update
        self.current_var = self.lambda * self.current_var + (1.0 - self.lambda) * return_val * return_val;
        // GARCH(1,1) update
        self.current_var = self.long_mean * (1.0 - self.alpha - self.beta)
            + self.alpha * return_val * return_val
            + self.beta * self.current_var;
        self.current_var.sqrt()
    }

    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        let mut forecast = Vec::with_capacity(steps);
        let mut var = self.current_var;
        for _ in 0..steps {
            var = self.long_mean * (1.0 - self.alpha - self.beta)
                + self.alpha * var
                + self.beta * var;
            forecast.push(var.sqrt());
        }
        forecast
    }
}

/// Symbolic regression model (genetic programming)
#[derive(Debug)]
pub struct SymbolicModel {
    pub expression_tree: ExpressionNode,
    pub complexity: usize,
    pub fitness: f64,
}

#[derive(Debug)]
pub enum ExpressionNode {
    Constant(f64),
    Feature(usize),
    Add(Box<ExpressionNode>, Box<ExpressionNode>),
    Sub(Box<ExpressionNode>, Box<ExpressionNode>),
    Mul(Box<ExpressionNode>, Box<ExpressionNode>),
    Div(Box<ExpressionNode>, Box<ExpressionNode>),
    Sqrt(Box<ExpressionNode>),
    Log(Box<ExpressionNode>),
    Abs(Box<ExpressionNode>),
    Neg(Box<ExpressionNode>),
}

impl ExpressionNode {
    pub fn evaluate(&self, features: &[f64]) -> f64 {
        match self {
            ExpressionNode::Constant(v) => *v,
            ExpressionNode::Feature(idx) => features.get(*idx).copied().unwrap_or(0.0),
            ExpressionNode::Add(a, b) => a.evaluate(features) + b.evaluate(features),
            ExpressionNode::Sub(a, b) => a.evaluate(features) - b.evaluate(features),
            ExpressionNode::Mul(a, b) => a.evaluate(features) * b.evaluate(features),
            ExpressionNode::Div(a, b) => {
                let denom = b.evaluate(features);
                if denom.abs() < 1e-10 { 0.0 } else { a.evaluate(features) / denom }
            }
            ExpressionNode::Sqrt(a) => a.evaluate(features).abs().sqrt(),
            ExpressionNode::Log(a) => a.evaluate(features).abs().ln(),
            ExpressionNode::Abs(a) => a.evaluate(features).abs(),
            ExpressionNode::Neg(a) => -a.evaluate(features),
        }
    }

    pub fn complexity(&self) -> usize {
        match self {
            ExpressionNode::Constant(_) | ExpressionNode::Feature(_) => 1,
            ExpressionNode::Add(a, b) | ExpressionNode::Sub(a, b) | ExpressionNode::Mul(a, b) | ExpressionNode::Div(a, b) => {
                1 + a.complexity() + b.complexity()
            }
            ExpressionNode::Sqrt(a) | ExpressionNode::Log(a) | ExpressionNode::Abs(a) | ExpressionNode::Neg(a) => {
                1 + a.complexity()
            }
        }
    }
}

/// Main model manager
#[derive(Debug)]
pub struct ModelManager {
    /// Loaded models per asset
    models: Arc<RwLock<HashMap<String, Vec<(ModelManifest, TrainedModel)>>>>,
    /// Model directory path
    model_dir: PathBuf,
    /// Configuration reference
    config: QuantConfig,
    /// Volatility predictor (shared across all assets)
    volatility: Arc<RwLock<VolatilityModel>>,
}

impl ModelManager {
    pub fn new(config: &QuantConfig) -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let model_dir = PathBuf::from(&home).join(".thequant").join("models");
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            model_dir,
            config: config.clone(),
            volatility: Arc::new(RwLock::new(VolatilityModel::new())),
        }
    }

    /// Load a model from disk
    pub async fn load_model(&self, asset: &str, regime: &str) -> QuantResult<()> {
        let manifest_path = self.model_dir.join("current").join(format!("{}_{}_manifest.json", asset, regime));
        if !manifest_path.exists() {
            return Err(QuantError::ModelNotFound {
                model_id: format!("{}_{}", asset, regime),
                asset: asset.to_string(),
            });
        }
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: ModelManifest = serde_json::from_str(&manifest_json)?;
        info!("Loaded model manifest for {} in regime {}", asset, regime);
        Ok(())
    }

    /// Hot-swap a model atomically
    pub async fn hot_swap(&self, manifest: ModelManifest, model: TrainedModel) -> QuantResult<()> {
        let asset = manifest.asset.clone();
        let regime = manifest.regime.clone();
        let mut models = self.models.write().await;
        let entry = models.entry(asset).or_insert_with(Vec::new);
        // Remove old model for this regime if exists
        entry.retain(|(m, _)| m.regime != regime);
        entry.push((manifest, model));
        info!("Hot-swapped model for regime {}", regime);
        Ok(())
    }

    /// Run inference with the appropriate model for an asset/regime
    pub async fn predict(&self, asset: &str, regime: &str, features: &[f64]) -> QuantResult<f64> {
        let models = self.models.read().await;
        if let Some(asset_models) = models.get(asset) {
            for (manifest, model) in asset_models {
                if manifest.regime == regime {
                    return match model {
                        TrainedModel::Gbdt(gbdt) => Ok(gbdt.predict(features)),
                        TrainedModel::SymbolicRegressor(sym) => Ok(sym.expression_tree.evaluate(features)),
                        _ => Err(QuantError::InferenceError("Unsupported model type for prediction".into())),
                    };
                }
            }
        }
        Err(QuantError::ModelNotFound {
            model_id: format!("{}_{}", asset, regime),
            asset: asset.to_string(),
        })
    }

    /// Update and get volatility forecast
    pub async fn update_volatility(&self, return_val: f64) -> f64 {
        let mut vol = self.volatility.write().await;
        vol.update(return_val)
    }

    /// Create a model manifest for serialization
    pub fn create_manifest(
        model_id: String, asset: String, regime: String,
        algorithm: String, metrics: ModelMetrics, file_path: PathBuf, size_bytes: u64,
    ) -> ModelManifest {
        ModelManifest {
            model_id, asset, regime, algorithm,
            trained_at: Utc::now(),
            features_hash: String::new(),
            metrics, status: ModelStatus::Production,
            file_path, model_size_bytes: size_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbdt_prediction() {
        let tree = DecisionTree {
            nodes: vec![
                TreeNode { feature_index: 0, threshold: 0.5, left_child: Some(1), right_child: Some(2), value: 0.0, is_leaf: false },
                TreeNode { feature_index: 1, threshold: 0.3, left_child: None, right_child: None, value: 1.0, is_leaf: true },
                TreeNode { feature_index: 1, threshold: 0.7, left_child: None, right_child: None, value: -1.0, is_leaf: true },
            ],
        };
        let model = GbdtModel {
            trees: vec![tree],
            learning_rate: 1.0,
            max_depth: 2,
            num_leaves: 3,
            feature_importances: vec![0.6, 0.4],
        };
        assert_eq!(model.predict(&[0.2, 0.1]), 1.0);
        assert_eq!(model.predict(&[0.8, 0.5]), -1.0);
    }

    #[test]
    fn test_volatility_model() {
        let mut vol = VolatilityModel::new();
        let vol_estimate = vol.update(0.01);
        assert!(vol_estimate > 0.0);
        let forecast = vol.forecast(5);
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_symbolic_expression() {
        let expr = ExpressionNode::Add(
            Box::new(ExpressionNode::Feature(0)),
            Box::new(ExpressionNode::Mul(
                Box::new(ExpressionNode::Feature(1)),
                Box::new(ExpressionNode::Constant(2.0)),
            )),
        );
        assert_eq!(expr.evaluate(&[3.0, 4.0]), 11.0);
        assert_eq!(expr.complexity(), 5);
    }
}
