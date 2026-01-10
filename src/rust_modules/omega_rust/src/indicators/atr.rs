//! Average True Range (ATR) implementation.
//!
//! ATR is a volatility indicator that measures the degree of price movement.
//! This implementation uses the Wilder smoothing method (same as Bloomberg/TradingView).
//!
//! ## Formula
//!
//! ```text
//! True Range = max(
//!     |High - Low|,
//!     |High - Previous Close|,
//!     |Low - Previous Close|
//! )
//!
//! ATR_0 = SMA(TR[0:period])
//! ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
//! ```
//!
//! ## Performance Target
//!
//! 50x speedup over Python baseline (954ms → ≤19ms for 50k bars)
//!
//! ## Reference
//!
//! - FFI Specification: `docs/ffi/indicator_cache.md`
//! - Migration Plan: `docs/WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md`

use crate::error::{OmegaError, Result};

/// Calculate True Range for a single bar.
///
/// True Range = max(|High - Low|, |High - Prev Close|, |Low - Prev Close|)
#[inline]
fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    let hl = (high - low).abs();
    let hc = (high - prev_close).abs();
    let lc = (low - prev_close).abs();
    hl.max(hc).max(lc)
}

/// Calculate Average True Range using Wilder smoothing.
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - ATR period (typically 14)
///
/// # Returns
///
/// Vector of ATR values with the same length as input.
/// First `period - 1` values will be NaN.
/// NaN values in input are carried forward.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0
/// - Arrays have different lengths
/// - Any array is empty
///
/// # Performance
///
/// - Time complexity: O(n)
/// - Space complexity: O(n)
/// - SIMD-friendly true range calculation
pub fn atr_impl(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
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

    // Pre-allocate result with NaN
    let result = vec![f64::NAN; n];

    // Calculate True Range for all bars
    let mut tr = vec![f64::NAN; n];

    // First TR uses High-Low only (no previous close)
    if !high[0].is_nan() && !low[0].is_nan() {
        tr[0] = (high[0] - low[0]).abs();
    }

    // Remaining TRs use previous close
    for i in 1..n {
        let h = high[i];
        let l = low[i];
        let pc = close[i - 1];

        if h.is_nan() || l.is_nan() || pc.is_nan() {
            // NaN propagation: if any input is NaN, TR is NaN
            tr[i] = f64::NAN;
        } else {
            tr[i] = true_range(h, l, pc);
        }
    }

    // Find first valid index with enough data for initial SMA
    let mut first_valid = None;
    for i in 0..n {
        if !tr[i].is_nan() {
            first_valid = Some(i);
            break;
        }
    }

    let first_valid = match first_valid {
        Some(idx) => idx,
        None => return Ok(result), // All NaN, return early
    };

    // Check if we have enough data for initial SMA
    if first_valid + period > n {
        return Ok(result); // Not enough data
    }

    // Check if initial window has all finite values
    let init_window = &tr[first_valid..first_valid + period];
    if !init_window.iter().all(|&v| v.is_finite()) {
        // Find a later starting point where we have period consecutive finite values
        let mut found_start = None;
        'outer: for start in first_valid..n.saturating_sub(period) {
            for j in 0..period {
                if !tr[start + j].is_finite() {
                    continue 'outer;
                }
            }
            found_start = Some(start);
            break;
        }

        match found_start {
            Some(start) => {
                // Recalculate with new starting point
                return atr_from_index(tr, start, period, result);
            }
            None => return Ok(result), // No valid window found
        }
    }

    atr_from_index(tr, first_valid, period, result)
}

/// Calculate ATR starting from a specific index.
fn atr_from_index(
    tr: Vec<f64>,
    first_valid: usize,
    period: usize,
    mut result: Vec<f64>,
) -> Result<Vec<f64>> {
    let n = tr.len();

    // Calculate initial ATR as SMA of first `period` TRs
    let init_window = &tr[first_valid..first_valid + period];
    let atr_init: f64 = init_window.iter().sum::<f64>() / period as f64;

    result[first_valid + period - 1] = atr_init;
    let mut atr_prev = atr_init;

    // Wilder smoothing for remaining values
    let period_f64 = period as f64;
    let period_minus_1 = (period - 1) as f64;

    for i in (first_valid + period)..n {
        let tr_i = tr[i];

        if tr_i.is_nan() {
            // Carry forward ATR when TR is NaN (missing data)
            result[i] = atr_prev;
        } else {
            // Wilder smoothing: ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
            let atr_new = (atr_prev * period_minus_1 + tr_i) / period_f64;
            result[i] = atr_new;
            atr_prev = atr_new;
        }
    }

    Ok(result)
}

/// Calculate ATR with pre-computed True Range values.
///
/// This is useful when TR is already available (e.g., from DMI calculation).
pub fn atr_from_tr(tr: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 0".to_string(),
        });
    }

    let n = tr.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let result = vec![f64::NAN; n];
    let tr_owned = tr.to_vec();

    // Find first valid index
    let first_valid = tr.iter().position(|&v| v.is_finite());
    let first_valid = match first_valid {
        Some(idx) => idx,
        None => return Ok(result),
    };

    if first_valid + period > n {
        return Ok(result);
    }

    atr_from_index(tr_owned, first_valid, period, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_true_range() {
        // Simple case: TR = High - Low
        assert_relative_eq!(true_range(110.0, 100.0, 105.0), 10.0, epsilon = 1e-10);

        // Gap up case: TR = High - Previous Close
        assert_relative_eq!(true_range(120.0, 115.0, 100.0), 20.0, epsilon = 1e-10);

        // Gap down case: TR = Previous Close - Low
        assert_relative_eq!(true_range(95.0, 90.0, 110.0), 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_atr_basic() {
        let high = vec![110.0, 112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 115.0];
        let low = vec![105.0, 108.0, 107.0, 109.0, 111.0, 110.0, 112.0, 111.0];
        let close = vec![108.0, 110.0, 109.0, 111.0, 113.0, 112.0, 114.0, 113.0];

        let result = atr_impl(&high, &low, &close, 3).unwrap();

        assert_eq!(result.len(), 8);
        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value should be valid (initial SMA)
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_atr_with_nan() {
        let high = vec![110.0, f64::NAN, 111.0, 113.0, 115.0];
        let low = vec![105.0, f64::NAN, 107.0, 109.0, 111.0];
        let close = vec![108.0, f64::NAN, 109.0, 111.0, 113.0];

        let result = atr_impl(&high, &low, &close, 2).unwrap();

        assert_eq!(result.len(), 5);
        // NaN should propagate to TR, but ATR should still be computed
        // where possible
    }

    #[test]
    fn test_atr_invalid_period() {
        let high = vec![110.0, 112.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0];

        let result = atr_impl(&high, &low, &close, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_atr_empty_input() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];

        let result = atr_impl(&high, &low, &close, 14);
        assert!(result.is_err());
    }

    #[test]
    fn test_atr_length_mismatch() {
        let high = vec![110.0, 112.0, 111.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0, 109.0];

        let result = atr_impl(&high, &low, &close, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_atr_wilder_smoothing() {
        // Test that Wilder smoothing is correctly applied
        // Using known values for verification
        let high = vec![10.0; 10];
        let low = vec![5.0; 10];
        let close = vec![7.5; 10];

        let result = atr_impl(&high, &low, &close, 3).unwrap();

        // TR should be constant at 5.0 (high - low)
        // Initial ATR (SMA) = 5.0
        // Subsequent ATRs should also be ~5.0 due to constant TR
        for i in 2..10 {
            assert_relative_eq!(result[i], 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_atr_from_tr() {
        let tr = vec![5.0, 4.0, 6.0, 5.5, 4.5, 5.0];
        let result = atr_from_tr(&tr, 3).unwrap();

        assert_eq!(result.len(), 6);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Initial ATR = (5 + 4 + 6) / 3 = 5.0
        assert_relative_eq!(result[2], 5.0, epsilon = 1e-10);
    }
}
