//! Simple Moving Average (SMA) implementation.
//!
//! SMA is the unweighted mean of the previous n data points.
//!
//! ## Formula
//!
//! ```text
//! SMA_t = (P_t + P_{t-1} + ... + P_{t-n+1}) / n
//! ```

use crate::error::{OmegaError, Result};

/// Calculate Simple Moving Average.
///
/// # Arguments
///
/// * `values` - Input values (typically close prices)
/// * `period` - SMA period (window size)
///
/// # Returns
///
/// Vector of SMA values with the same length as input.
/// First `period - 1` values will be NaN.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0
/// - `values` is empty
pub fn sma_impl(values: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 0".to_string(),
        });
    }

    let n = values.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let mut result = vec![f64::NAN; n];

    if n < period {
        return Ok(result);
    }

    let period_f64 = period as f64;

    // Calculate first SMA
    let mut sum: f64 = values[..period].iter().sum();
    if sum.is_finite() {
        result[period - 1] = sum / period_f64;
    }

    // Rolling sum for subsequent SMAs
    for i in period..n {
        let old_val = values[i - period];
        let new_val = values[i];

        // Handle NaN values
        if old_val.is_nan() || new_val.is_nan() {
            // Recalculate sum from scratch when NaN is involved
            let window = &values[(i - period + 1)..=i];
            let valid_sum: f64 = window.iter().filter(|v| v.is_finite()).sum();
            let valid_count = window.iter().filter(|v| v.is_finite()).count();

            if valid_count == period {
                result[i] = valid_sum / period_f64;
            }
        } else {
            sum = sum - old_val + new_val;
            if sum.is_finite() {
                result[i] = sum / period_f64;
            }
        }
    }

    Ok(result)
}

/// Calculate rolling standard deviation.
///
/// Uses a numerically stable two-pass algorithm.
///
/// # Arguments
///
/// * `values` - Input values
/// * `period` - Window size
/// * `ddof` - Delta degrees of freedom (1 for sample std, 0 for population std)
///
/// # Returns
///
/// Vector of standard deviation values.
/// First `period - 1` values will be NaN.
pub fn rolling_std_impl(values: &[f64], period: usize, ddof: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 0".to_string(),
        });
    }

    if period <= ddof {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than ddof".to_string(),
        });
    }

    let n = values.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let mut result = vec![f64::NAN; n];

    if n < period {
        return Ok(result);
    }

    let divisor = (period - ddof) as f64;

    for i in (period - 1)..n {
        let window = &values[(i - period + 1)..=i];

        // Check for NaN values
        if window.iter().any(|&v| v.is_nan()) {
            continue;
        }

        // Two-pass algorithm for numerical stability
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / divisor;

        result[i] = variance.sqrt();
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sma_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma_impl(&values, 3).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10); // (2+3+4)/3
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10); // (3+4+5)/3
    }

    #[test]
    fn test_sma_with_nan() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let result = sma_impl(&values, 2).unwrap();

        assert!(result[0].is_nan());
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
        assert!(result[2].is_nan()); // NaN in window
        assert!(result[3].is_nan()); // NaN in window
        assert_relative_eq!(result[4], 4.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sma_invalid_period() {
        let values = vec![1.0, 2.0, 3.0];
        let result = sma_impl(&values, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_std_basic() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = rolling_std_impl(&values, 3, 1).unwrap();

        assert_eq!(result.len(), 8);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Window [2,4,4]: mean=3.33, std≈1.155
        assert!(result[2].is_finite());
    }
}
