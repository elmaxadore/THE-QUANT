//! FEATURE ENGINEERING PIPELINE — Layer 3 (v2.1 §6).
//!
//! Turns a rolling window of bars into a fixed-size feature vector that both
//! the classical strategies and the ONNX model can consume. In production these
//! features are the output of a zero-allocation hot path; here we compute a
//! representative, numerically stable subset with clear math in comments.

use crate::simfeed::Bar;

/// Maximum features we produce; kept small enough to be a fast-to-compute,
/// cache-friendly fixed-size window (the model input width).
pub const N_FEATURES: usize = 12;

/// Compute the feature vector for the given window of bars (oldest first).
/// Returns a slice `[f64; N_FEATURES]`.
///
/// Feature slots:
///   0  log_return_1          (close_t / close_{t-1}) - 1, log
///   1  log_return_5          5-bar log return
///   2  realized_vol_10       10-bar rolling volatility
///   3  rsi_14                Relative Strength Index (14)
///   4  atr_14                Average True Range / price (normalised)
///   5  ema_diff_12_26        (EMA12 - EMA26)/price
///   6  bb_width              (upper-lower)/mid Bollinger width
///   7  volume_zscore         (v - mean(v))/std(v)
///   8  high_low_range        (high-low)/close
///   9  close_position        (close-low)/(high-low), 0.5 if flat
///  10  hurst_approx          crude Hurst-like persistence estimate
///  11  momentum_roc_10       rate of change over 10 bars
pub fn compute_features(bars: &[Bar]) -> [f64; N_FEATURES] {
    let mut f = [0.0; N_FEATURES];
    let n = bars.len();
    if n == 0 {
        return f;
    }

    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let last = closes[n - 1];

    // 0: 1-bar log return
    if n >= 2 {
        f[0] = (closes[n - 1] / closes[n - 2]).ln();
    }
    // 1: 5-bar log return
    if n >= 6 {
        f[1] = (closes[n - 1] / closes[n - 6]).ln();
    }
    // 2: 10-bar realized vol
    if n >= 2 {
        let mut rets = Vec::with_capacity(n - 1);
        for w in 1..n {
            rets.push((closes[w] / closes[w - 1]).ln());
        }
        let k = rets.len().min(10);
        let tail = &rets[rets.len() - k..];
        f[2] = crate::util::stddev(tail);
    }
    // 3: RSI-14
    f[3] = rsi(&closes, 14);
    // 4: ATR-14 normalised by price
    f[4] = atr(bars, 14) / last.max(1e-12);
    // 5: EMA diff
    if n >= 26 {
        let e12 = ema(&closes, 12);
        let e26 = ema(&closes, 26);
        f[5] = (e12 - e26) / last.max(1e-12);
    }
    // 6: Bollinger width (20, 2σ)
    if n >= 20 {
        let k = 20;
        let tail = &closes[n - k..];
        let m = crate::util::mean(tail);
        let sd = crate::util::stddev(tail);
        let mid = m;
        let hi = m + 2.0 * sd;
        let lo = m - 2.0 * sd;
        f[6] = (hi - lo) / mid.max(1e-12);
    }
    // 7: volume zscore (10)
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    {
        let k = volumes.len().min(10);
        let tail = &volumes[volumes.len() - k..];
        let m = crate::util::mean(tail);
        let sd = crate::util::stddev(tail);
        let lastv = *volumes.last().unwrap_or(&0.0);
        f[7] = if sd > 1e-12 { (lastv - m) / sd } else { 0.0 };
    }
    // 8: high-low range
    let lastbar = bars[n - 1].clone();
    f[8] = (lastbar.high - lastbar.low) / lastbar.close.max(1e-12);
    // 9: close position in range
    let rng = lastbar.high - lastbar.low;
    f[9] = if rng > 1e-12 { (lastbar.close - lastbar.low) / rng } else { 0.5 };
    // 10: crude Hurst-like persistence (sign-correlation of consecutive returns)
    if n >= 20 {
        let mut agrees = 0;
        let mut total = 0;
        for w in 2..n.min(60) {
            let r1 = closes[w] - closes[w - 1];
            let r2 = closes[w - 1] - closes[w - 2];
            if r1 * r2 > 0.0 {
                agrees += 1;
            }
            total += 1;
        }
        f[10] = if total > 0 { agrees as f64 / total as f64 } else { 0.5 };
    }
    // 11: momentum ROC-10
    if n >= 11 {
        f[11] = (closes[n - 1] - closes[n - 11]) / closes[n - 11].max(1e-12);
    }
    f
}

fn ema(xs: &[f64], period: usize) -> f64 {
    let a = 2.0 / (period as f64 + 1.0);
    let mut e = xs[0];
    for x in xs.iter().skip(1) {
        e = a * x + (1.0 - a) * e;
    }
    e
}

fn rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 50.0;
    }
    let mut gains = 0.0;
    let mut losses = 0.0;
    let start = closes.len() - period;
    for i in start..closes.len() {
        let delta = closes[i] - closes[i - 1];
        if delta > 0.0 {
            gains += delta;
        } else {
            losses -= delta;
        }
    }
    if losses.abs() < 1e-12 {
        return 100.0;
    }
    let rs = gains / losses;
    100.0 - 100.0 / (1.0 + rs)
}

fn atr(bars: &[Bar], period: usize) -> f64 {
    if bars.len() < 2 {
        return 0.0;
    }
    let start = bars.len().saturating_sub(period).max(1);
    let mut trues = Vec::new();
    for i in start..bars.len() {
        let b = &bars[i];
        let prev_close = bars[i - 1].close;
        let tr = (b.high - b.low)
            .max((b.high - prev_close).abs())
            .max((b.low - prev_close).abs());
        trues.push(tr);
    }
    crate::util::mean(&trues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simfeed::SimFeed;

    #[test]
    fn feature_vector_has_correct_width_and_bounds() {
        let mut feed = SimFeed::new("EURUSD", 1.08, 7);
        let mut bars = Vec::new();
        for _ in 0..60 {
            bars.push(feed.next_bar());
        }
        let f = compute_features(&bars);
        assert_eq!(f.len(), N_FEATURES);
        for v in f {
            assert!(v.is_finite(), "feature must be finite");
        }
        // RSI within [0,100]
        assert!((0.0..=100.0).contains(&f[3]));
    }
}