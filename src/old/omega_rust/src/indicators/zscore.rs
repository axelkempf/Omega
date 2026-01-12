//! Z-Score (Standard Score) implementation.
//!
//! Z-Score measures how many standard deviations a value is from the mean.
//!
//! ## Formula
//!
//! ```text
//! z = (x - μ) / σ
//! where μ = rolling mean, σ = rolling standard deviation
//! ```
//!
//! ## Interpretation
//!
//! - z > 2: Significantly above average (potential overbought)
//! - z < -2: Significantly below average (potential oversold)
//! - |z| < 1: Within normal range

use crate::error::{OmegaError, Result};

/// Calculate rolling Z-Score.
///
/// # Arguments
///
/// * `values` - Input values
/// * `period` - Window size for mean and std calculation
/// * `ddof` - Delta degrees of freedom for std (typically 1)
///
/// # Returns
///
/// Vector of Z-Score values.
/// First `period - 1` values will be NaN.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0 or ≤ `ddof`
/// - `values` is empty
pub fn zscore_impl(values: &[f64], period: usize, ddof: usize) -> Result<Vec<f64>> {
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

        // Check for NaN
        if window.iter().any(|&v| v.is_nan()) {
            continue;
        }

        let current = values[i];

        // Calculate mean
        let mean: f64 = window.iter().sum::<f64>() / period as f64;

        // Calculate standard deviation
        let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / divisor;

        let std = variance.sqrt();

        // Calculate z-score
        if std > 0.0 {
            result[i] = (current - mean) / std;
        } else {
            result[i] = 0.0; // All values equal, z-score is 0
        }
    }

    Ok(result)
}

/// Calculate z-score normalized to [-1, 1] range using tanh.
///
/// Useful when z-scores need to be bounded for neural networks or
/// when extreme values should be dampened.
///
/// # Arguments
///
/// * `values` - Input values
/// * `period` - Window size
/// * `ddof` - Delta degrees of freedom
/// * `scale` - Scaling factor before tanh (default 1.0)
pub fn zscore_normalized_impl(
    values: &[f64],
    period: usize,
    ddof: usize,
    scale: f64,
) -> Result<Vec<f64>> {
    let zscore = zscore_impl(values, period, ddof)?;

    let result: Vec<f64> = zscore
        .iter()
        .map(|&z| {
            if z.is_nan() {
                f64::NAN
            } else {
                (z * scale).tanh()
            }
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zscore_basic() {
        // Standard normal-ish data
        let values = vec![10.0, 11.0, 9.0, 10.5, 10.0];
        let result = zscore_impl(&values, 3, 1).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_zscore_constant() {
        // All values equal → z-score = 0
        let values = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = zscore_impl(&values, 3, 1).unwrap();

        for i in 2..5 {
            assert_relative_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zscore_extreme() {
        // Last value is extreme
        let values = vec![10.0, 10.0, 10.0, 10.0, 100.0];
        let result = zscore_impl(&values, 3, 1).unwrap();

        // z-score of 100 in [10, 10, 100] should be positive and large
        assert!(result[4] > 1.0);
    }

    #[test]
    fn test_zscore_normalized() {
        let values = vec![10.0, 10.0, 10.0, 10.0, 100.0];
        let result = zscore_normalized_impl(&values, 3, 1, 1.0).unwrap();

        // Tanh bounds output to [-1, 1]
        for &v in &result {
            if v.is_finite() {
                assert!(v >= -1.0 && v <= 1.0);
            }
        }
    }

    #[test]
    fn test_zscore_invalid_period() {
        let values = vec![1.0, 2.0, 3.0];
        let result = zscore_impl(&values, 0, 1);
        assert!(result.is_err());

        let result = zscore_impl(&values, 1, 1);
        assert!(result.is_err()); // period <= ddof
    }
}
