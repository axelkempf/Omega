//! Statistical functions for financial time series.
//!
//! Provides rolling window calculations commonly used in
//! quantitative finance and risk analysis.

use pyo3::prelude::*;

use crate::error::{OmegaError, Result};

/// Calculate rolling standard deviation.
///
/// # Arguments
///
/// * `values` - Vector of values
/// * `window` - Rolling window size
///
/// # Returns
///
/// Vector of rolling standard deviation values.
/// The first `window - 1` values will be NaN.
///
/// # Example
///
/// ```python
/// from omega._rust import rolling_std
///
/// returns = [0.01, -0.02, 0.015, -0.005, 0.02, ...]
/// volatility = rolling_std(returns, window=20)
/// ```
#[pyfunction]
#[pyo3(signature = (values, window))]
pub fn rolling_std(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    rolling_std_impl(&values, window).map_err(Into::into)
}

/// Internal rolling standard deviation implementation.
///
/// Uses Welford's online algorithm for numerical stability.
pub fn rolling_std_impl(values: &[f64], window: usize) -> Result<Vec<f64>> {
    // Validate inputs
    if window < 2 {
        return Err(OmegaError::InvalidParameter {
            reason: "window must be at least 2".to_string(),
        });
    }

    if values.len() < window {
        return Err(OmegaError::InsufficientData {
            required: window,
            actual: values.len(),
        });
    }

    let n = values.len();
    let mut result = vec![f64::NAN; n];

    // Calculate for each valid window position
    for i in (window - 1)..n {
        let window_start = i + 1 - window;
        let window_values = &values[window_start..=i];

        // Calculate mean
        let mean: f64 = window_values.iter().sum::<f64>() / window as f64;

        // Calculate variance using two-pass algorithm for stability
        let variance: f64 = window_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (window - 1) as f64;

        result[i] = variance.sqrt();
    }

    Ok(result)
}

/// Calculate rolling mean (Simple Moving Average).
///
/// # Arguments
///
/// * `values` - Vector of values
/// * `window` - Rolling window size
///
/// # Returns
///
/// Vector of rolling mean values.
/// The first `window - 1` values will be NaN.
#[allow(dead_code)]
pub fn rolling_mean_impl(values: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "window must be greater than 0".to_string(),
        });
    }

    if values.len() < window {
        return Err(OmegaError::InsufficientData {
            required: window,
            actual: values.len(),
        });
    }

    let n = values.len();
    let mut result = vec![f64::NAN; n];
    let window_f64 = window as f64;

    // Initialize with first window
    let mut sum: f64 = values[..window].iter().sum();
    result[window - 1] = sum / window_f64;

    // Rolling calculation
    for i in window..n {
        sum = sum - values[i - window] + values[i];
        result[i] = sum / window_f64;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rolling_std_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_std_impl(&values, 3).unwrap();

        assert_eq!(result.len(), 5);

        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // std([1, 2, 3]) = 1.0 (sample std)
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10);

        // std([2, 3, 4]) = 1.0
        assert_relative_eq!(result[3], 1.0, epsilon = 1e-10);

        // std([3, 4, 5]) = 1.0
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rolling_std_constant() {
        let values = vec![5.0; 10];
        let result = rolling_std_impl(&values, 3).unwrap();

        // Std of constant values should be 0
        for val in result.iter().skip(2) {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rolling_std_invalid_window() {
        let values = vec![1.0, 2.0, 3.0];
        let result = rolling_std_impl(&values, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_std_insufficient_data() {
        let values = vec![1.0, 2.0];
        let result = rolling_std_impl(&values, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_mean_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean_impl(&values, 3).unwrap();

        assert_eq!(result.len(), 5);

        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // mean([1, 2, 3]) = 2.0
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);

        // mean([2, 3, 4]) = 3.0
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10);

        // mean([3, 4, 5]) = 4.0
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10);
    }
}
