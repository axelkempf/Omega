//! Directional Movement Index (DMI) implementation.
//!
//! DMI measures trend strength and direction using +DI, -DI, and ADX.
//! This implementation uses Wilder smoothing (same as Bloomberg/TradingView).
//!
//! ## Components
//!
//! - **+DI (Positive Directional Indicator)**: Measures upward price movement
//! - **-DI (Negative Directional Indicator)**: Measures downward price movement
//! - **ADX (Average Directional Index)**: Measures trend strength (0-100)
//!
//! ## Interpretation
//!
//! - +DI > -DI: Uptrend
//! - -DI > +DI: Downtrend
//! - ADX > 25: Strong trend
//! - ADX < 20: Weak/no trend
//!
//! ## Performance Target
//!
//! 20x speedup over Python baseline (65ms → ≤3.3ms for 50k bars)
//!
//! ## Reference
//!
//! - FFI Specification: `docs/ffi/indicator_cache.md`

use crate::error::{OmegaError, Result};

/// DMI result containing +DI, -DI, and ADX.
pub struct DmiResult {
    /// Positive Directional Indicator
    pub plus_di: Vec<f64>,
    /// Negative Directional Indicator
    pub minus_di: Vec<f64>,
    /// Average Directional Index
    pub adx: Vec<f64>,
}

/// Calculate True Range for DMI.
#[inline]
fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    let hl = (high - low).abs();
    let hc = (high - prev_close).abs();
    let lc = (low - prev_close).abs();
    hl.max(hc).max(lc)
}

/// Apply Wilder smoothing (EMA with alpha = 1/period).
///
/// This is equivalent to pandas ewm(alpha=1/period, adjust=False).mean()
fn wilder_smooth(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];
    let alpha = 1.0 / period as f64;
    let one_minus_alpha = 1.0 - alpha;

    // Find first valid value
    let first_valid = values.iter().position(|&v| v.is_finite());
    let first_valid = match first_valid {
        Some(idx) => idx,
        None => return result,
    };

    // Initialize with first valid value
    let mut ema = values[first_valid];
    result[first_valid] = ema;

    // Apply EMA smoothing
    for i in (first_valid + 1)..n {
        let v = values[i];
        if v.is_nan() {
            // Carry forward
            result[i] = ema;
        } else {
            ema = alpha * v + one_minus_alpha * ema;
            result[i] = ema;
        }
    }

    result
}

/// Calculate Directional Movement Index.
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - DMI period (typically 14)
///
/// # Returns
///
/// `DmiResult` containing +DI, -DI, and ADX series.
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
pub fn dmi_impl(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<DmiResult> {
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

    // Pre-allocate arrays
    let mut plus_dm = vec![f64::NAN; n];
    let mut minus_dm = vec![f64::NAN; n];
    let mut tr = vec![f64::NAN; n];

    // First bar: no previous data
    if !high[0].is_nan() && !low[0].is_nan() {
        tr[0] = (high[0] - low[0]).abs();
    }
    plus_dm[0] = 0.0;
    minus_dm[0] = 0.0;

    // Calculate DM and TR for each bar
    for i in 1..n {
        let h = high[i];
        let l = low[i];
        let c = close[i - 1];
        let prev_h = high[i - 1];
        let prev_l = low[i - 1];

        // True Range
        if h.is_nan() || l.is_nan() || c.is_nan() {
            tr[i] = f64::NAN;
        } else {
            tr[i] = true_range(h, l, c);
        }

        // Directional Movement
        if h.is_nan() || l.is_nan() || prev_h.is_nan() || prev_l.is_nan() {
            plus_dm[i] = f64::NAN;
            minus_dm[i] = f64::NAN;
        } else {
            let up_move = h - prev_h;
            let down_move = prev_l - l;

            // +DM: up move > down move AND up move > 0
            plus_dm[i] = if up_move > down_move && up_move > 0.0 {
                up_move
            } else {
                0.0
            };

            // -DM: down move > up move AND down move > 0
            minus_dm[i] = if down_move > up_move && down_move > 0.0 {
                down_move
            } else {
                0.0
            };
        }
    }

    // Apply Wilder smoothing
    let atr = wilder_smooth(&tr, period);
    let smooth_plus_dm = wilder_smooth(&plus_dm, period);
    let smooth_minus_dm = wilder_smooth(&minus_dm, period);

    // Calculate +DI and -DI
    let mut plus_di = vec![f64::NAN; n];
    let mut minus_di = vec![f64::NAN; n];
    let mut dx = vec![f64::NAN; n];

    for i in 0..n {
        let atr_val = atr[i];
        let plus_dm_val = smooth_plus_dm[i];
        let minus_dm_val = smooth_minus_dm[i];

        if atr_val.is_finite() && atr_val != 0.0 {
            plus_di[i] = 100.0 * plus_dm_val / atr_val;
            minus_di[i] = 100.0 * minus_dm_val / atr_val;

            // DX = |+DI - -DI| / (+DI + -DI) * 100
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum != 0.0 {
                dx[i] = ((plus_di[i] - minus_di[i]).abs() / di_sum) * 100.0;
            }
        }
    }

    // ADX = Wilder smooth of DX
    let adx = wilder_smooth(&dx, period);

    Ok(DmiResult {
        plus_di,
        minus_di,
        adx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wilder_smooth() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wilder_smooth(&values, 3);

        assert_eq!(result.len(), 5);
        // First value should be 1.0
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        // Values should increase gradually
        for i in 1..5 {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_dmi_basic() {
        let high = vec![110.0, 112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 115.0];
        let low = vec![105.0, 108.0, 107.0, 109.0, 111.0, 110.0, 112.0, 111.0];
        let close = vec![108.0, 110.0, 109.0, 111.0, 113.0, 112.0, 114.0, 113.0];

        let result = dmi_impl(&high, &low, &close, 3).unwrap();

        assert_eq!(result.plus_di.len(), 8);
        assert_eq!(result.minus_di.len(), 8);
        assert_eq!(result.adx.len(), 8);

        // Values should be in [0, 100] range (or NaN)
        for &v in &result.plus_di {
            if v.is_finite() {
                assert!(v >= 0.0 && v <= 100.0);
            }
        }
    }

    #[test]
    fn test_dmi_invalid_period() {
        let high = vec![110.0, 112.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0];

        let result = dmi_impl(&high, &low, &close, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dmi_length_mismatch() {
        let high = vec![110.0, 112.0, 111.0];
        let low = vec![105.0, 108.0];
        let close = vec![108.0, 110.0, 109.0];

        let result = dmi_impl(&high, &low, &close, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_dmi_with_nan() {
        let high = vec![110.0, f64::NAN, 111.0, 113.0, 115.0];
        let low = vec![105.0, f64::NAN, 107.0, 109.0, 111.0];
        let close = vec![108.0, f64::NAN, 109.0, 111.0, 113.0];

        let result = dmi_impl(&high, &low, &close, 2).unwrap();

        // Should handle NaN gracefully
        assert_eq!(result.plus_di.len(), 5);
    }
}
