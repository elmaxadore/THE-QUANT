//! Small shared helpers (timestamps, formatting, Math helpers).

use chrono::{SecondsFormat, Utc};

/// RFC3339-ish UTC timestamp with milliseconds, e.g. `2026-08-26T10:39:05.123Z`.
pub fn now_iso8601() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Format a `f64` price with `dp` decimals, e.g. for display.
pub fn fmt_price(v: f64, dp: usize) -> String {
    format!("{v:.dp$}")
}

/// Simple mean of a slice.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Standard deviation (population) of a slice.
pub fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    v.sqrt()
}

/// Linear interpolation of `x` from [x0,x1] into [y0,y1].
pub fn lerp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() < f64::EPSILON {
        return y0;
    }
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

/// Clamp `v` into [lo, hi].
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}
