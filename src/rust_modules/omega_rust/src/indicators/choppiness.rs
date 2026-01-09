//! Choppiness Index implementation.
//!
//! The Choppiness Index measures whether the market is trending or ranging.
//!
//! ## Formula
//!
//! ```text
//! CI = 100 × log10(sum(ATR_n) / (max_high - min_low)) / log10(n)
//! ```
//!
//! ## Interpretation
//!
//! - CI > 61.8: Choppy/ranging market
//! - CI < 38.2: Trending market
//! - Range: 0-100

use crate::error::{OmegaError, Result};
use crate::indicators::atr::atr_impl;

/// Calculate Choppiness Index.
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Lookback period (typically 14)
///
/// # Returns
///
/// Vector of Choppiness Index values (0-100 scale).
/// First `period - 1` values will be NaN.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0
/// - Arrays have different lengths
/// - Any array is empty
pub fn choppiness_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 0".to_string(),
        });
    }

    let n = high.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    if low.len() != n || close.len() != n {
        return Err(OmegaError::InvalidParameter {
            reason: format!(
                "Array length mismatch: high={}, low={}, close={}",
                n,
                low.len(),
                close.len()
            ),
        });
    }

    // Calculate ATR first (we'll use it for sum of True Ranges)
    let _atr = atr_impl(high, low, close, period)?;

    let mut result = vec![f64::NAN; n];

    if n < period {
        return Ok(result);
    }

    let log_period = (period as f64).log10();

    // Calculate True Ranges for summing
    let mut tr = vec![f64::NAN; n];

    // First bar
    if high[0].is_finite() && low[0].is_finite() {
        tr[0] = (high[0] - low[0]).abs();
    }

    // Subsequent bars
    for i in 1..n {
        let h = high[i];
        let l = low[i];
        let c = close[i - 1];

        if h.is_finite() && l.is_finite() && c.is_finite() {
            let hl = (h - l).abs();
            let hc = (h - c).abs();
            let lc = (l - c).abs();
            tr[i] = hl.max(hc).max(lc);
        }
    }

    // Calculate Choppiness Index
    for i in (period - 1)..n {
        let start = i - period + 1;
        let window_tr = &tr[start..=i];
        let window_high = &high[start..=i];
        let window_low = &low[start..=i];

        // Check for NaN in windows
        if window_tr.iter().any(|&v| v.is_nan())
            || window_high.iter().any(|&v| v.is_nan())
            || window_low.iter().any(|&v| v.is_nan())
        {
            continue;
        }

        // Sum of True Ranges
        let sum_tr: f64 = window_tr.iter().sum();

        // Range (max high - min low)
        let max_high = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_low = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_high - min_low;

        if range > 0.0 && log_period > 0.0 {
            result[i] = 100.0 * (sum_tr / range).log10() / log_period;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_choppiness_basic() {
        let high = vec![110.0, 112.0, 111.0, 113.0, 115.0, 114.0, 116.0];
        let low = vec![105.0, 108.0, 107.0, 109.0, 111.0, 110.0, 112.0];
        let close = vec![108.0, 110.0, 109.0, 111.0, 113.0, 112.0, 114.0];

        let result = choppiness_impl(&high, &low, &close, 3).unwrap();

        assert_eq!(result.len(), 7);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Values should be between 0 and 100
        for &v in &result[2..] {
            if v.is_finite() {
                assert!(v >= 0.0 && v <= 100.0, "Choppiness {} out of range", v);
            }
        }
    }

    #[test]
    fn test_choppiness_trending() {
        // Strong uptrend should have lower choppiness
        let high: Vec<f64> = (100..120).map(|x| x as f64 + 5.0).collect();
        let low: Vec<f64> = (100..120).map(|x| x as f64).collect();
        let close: Vec<f64> = (100..120).map(|x| x as f64 + 3.0).collect();

        let result = choppiness_impl(&high, &low, &close, 5).unwrap();

        // In a strong trend, later values should be relatively low
        let last_valid = result.iter().rev().find(|&&v| v.is_finite());
        if let Some(&v) = last_valid {
            assert!(v < 80.0); // Trending markets usually below 61.8
        }
    }

    #[test]
    fn test_choppiness_invalid_period() {
        let high = vec![110.0, 112.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0];

        let result = choppiness_impl(&high, &low, &close, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_choppiness_length_mismatch() {
        let high = vec![110.0, 112.0, 111.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0, 109.0];

        let result = choppiness_impl(&high, &low, &close, 2);
        assert!(result.is_err());
    }
}
