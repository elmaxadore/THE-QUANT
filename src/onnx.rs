//! HYBRID ML BRIDGE — the core innovation of the "most efficient" build.
//!
//! Training happens in Python (offline, in the evolution lab) and produces
//! `.onnx` models. Rust loads them at startup through the `ort` crate for
//! sub-2ms inference — Python is never in the live hot path.
//!
//! Two backends behind one `ModelTrait`:
//!   * `RuleModel`        — pure-Rust fallback (no heavy deps), used when the
//!                          binary is a lean build or no .onnx is present.
//!   * `OrtModel`         — real ONNX Runtime inference behind the `ml` feature.
//!
//! The lean Tier-1 default build uses `RuleModel` so it compiles and runs with
//! zero external ML dependencies.

use crate::config::Config;

/// Width of the input the model expects — matches `N_FEATURES`.
pub const MODEL_INPUT_DIM: usize = 12;

/// A single scalar prediction (e.g. expected log-return or trade confidence).
pub type Prediction = f64;

/// Directional signal between -1 (short) and +1 (long); 0 = neutral.
pub type Signal = f64;

/// The interface every model backend implements.
pub trait ModelBackend {
    /// Predict for a feature vector. Returns (directional_signal, confidence).
    /// signal > 0 => net long bias, < 0 => net short bias, ~0 => neutral.
    fn predict(&self, features: &[f64; MODEL_INPUT_DIM]) -> (Signal, f64);
    fn name(&self) -> &str;
}

/// A pure-Rust baseline model: weighted linear readout over the features.
///
/// This is intentionally simple and interpretable. In production it is replaced
/// at runtime by the ONNX model trained in Python.
pub struct RuleModel {
    weights: [f64; MODEL_INPUT_DIM],
    bias: f64,
}

impl RuleModel {
    pub fn new() -> Self {
        // A rationally-chosen sign pattern: follow momentum/trend, fade the
        // overbought/oversold. Coefficients are hyper-parameters the Python
        // trainer would refine into a real .onnx.
        RuleModel {
            weights: [
                0.6,  /* log_return_1 */
                0.8,  /* log_return_5 */
                -0.4, /* realized_vol */
                0.0,  /* rsi */
                0.0,  /* atr */
                0.9,  /* ema_diff */
                -0.2, /* bb_width */
                0.1,  /* volume_zscore */
                0.0,  /* high_low_range */
                0.0,  /* close_position */
                0.3,  /* hurst */
                0.7,  /* roc_10 */
            ],
            bias: 0.0,
        }
    }
}

impl Default for RuleModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBackend for RuleModel {
    fn predict(&self, features: &[f64; MODEL_INPUT_DIM]) -> (Signal, f64) {
        let mut raw = self.bias;
        for i in 0..MODEL_INPUT_DIM {
            raw += self.weights[i] * features[i];
        }
        // The input features are small (log-returns ~1e-4, EMA differences
        // ~1e-3), so we amplify with an on-the-fly normaliser so the signal is
        // meaningfully populated. This mimics what proper feature scaling in
        // the Python trainer (and a real .onnx) would do.
        let gain = 250.0;
        // squash to (-1,1); confidence = magnitude.
        let signal = (raw * gain).tanh();
        (signal, signal.abs())
    }

    fn name(&self) -> &'static str {
        "rule-model"
    }
}

/// ONNX Runtime backend — activated with `--features ml` and a `.onnx` file.
#[cfg(feature = "ml")]
pub struct OrtModel {
    session: ort::Session,
}

#[cfg(feature = "ml")]
impl OrtModel {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let session = ort::Session::builder()
            .map_err(|e| format!("onnx builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("onnx load: {e}"))?;
        Ok(OrtModel { session })
    }
}

#[cfg(feature = "ml")]
impl ModelBackend for OrtModel {
    fn predict(&self, features: &[f64; MODEL_INPUT_DIM]) -> (Signal, f64) {
        // Build a [1, MODEL_INPUT_DIM] f32 input tensor.
        let input: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let out = match self.session.run(ort::inputs!["input" => input.as_slice()]) {
            Ok(v) => v,
            Err(_) => return (0.0, 0.0),
        };
        // Grab the first output tensor as a 1-d slice; take element 0.
        let first = out[0];
        let winner = match first.try_extract_array::<f32>() {
            Ok(Ok(arr)) => Some(arr),
            _ => None,
        };
        match winner {
            Some(arr) => {
                let v = if arr.len() > 0 { arr[0] } else { 0.0 } as f64;
                (v, v.abs())
            }
            None => (0.0, 0.0),
        }
    }

    fn name(&self) -> &'static str {
        "onnx-runtime"
    }
}

/// A thin factory that picks the best available backend.
pub fn load_backend(_cfg: &Config) -> Result<Box<dyn ModelBackend>, String> {
    #[cfg(feature = "ml")]
    {
        if cfg.ml.enabled && std::path::Path::new(&cfg.ml.model_path).exists() {
            return Ok(Box::new(OrtModel::load(std::path::Path::new(
                &cfg.ml.model_path,
            ))?));
        }
    }
    Ok(Box::new(RuleModel::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_model_produces_finite_signal() {
        let m = RuleModel::new();
        let f = [
            0.05, 0.03, 0.002, 50.0, 0.001, 0.01, 0.02, 0.5, 0.01, 0.5, 0.6, 0.04,
        ];
        let (sig, conf) = m.predict(&f);
        assert!(sig.is_finite());
        assert!((0.0..=1.0).contains(&conf));
    }
}
