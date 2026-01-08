//! Exponential Moving Average (EMA) implementation.
//!
//! The EMA gives more weight to recent prices, making it more responsive
//! to new information compared to a simple moving average.
//!
//! ## Formula
//!
//! ```text
//! EMA_t = α * Price_t + (1 - α) * EMA_{t-1}
//! where α = 2 / (period + 1)
//! ```

use pyo3::prelude::*;

use crate::error::{OmegaError, Result};

/// Calculate Exponential Moving Average.
///
/// # Arguments
///
/// * `prices` - Vector of price values
/// * `period` - EMA period (smoothing window)
///
/// # Returns
///
/// Vector of EMA values with the same length as input.
/// The first `period - 1` values use a cumulative average for initialization.
///
/// # Example
///
/// ```python
/// from omega._rust import ema
///
/// prices = [100.0, 101.5, 99.8, 102.3, 103.1]
/// result = ema(prices, period=3)
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0
/// - `period` is greater than the length of `prices`
/// - `prices` is empty
#[pyfunction]
#[pyo3(signature = (prices, period))]
#[allow(clippy::needless_pass_by_value)]
pub fn ema(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    exponential_moving_average_impl(&prices, period).map_err(Into::into)
}

/// Full name alias for EMA function.
///
/// Identical to [`ema`], provided for API clarity.
///
/// # Errors
///
/// See [`ema`].
#[pyfunction]
#[pyo3(signature = (prices, period))]
#[allow(clippy::needless_pass_by_value)]
pub fn exponential_moving_average(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    exponential_moving_average_impl(&prices, period).map_err(Into::into)
}

/// Internal EMA implementation.
///
/// Separated from `PyO3` wrapper for easier testing and potential reuse.
pub fn exponential_moving_average_impl(prices: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
    if period == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 0".to_string(),
        });
    }

    if prices.is_empty() {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    if period > prices.len() {
        return Err(OmegaError::InsufficientData {
            required: period,
            actual: prices.len(),
        });
    }

    let period_u32 = u32::try_from(period).map_err(|_| OmegaError::InvalidParameter {
        reason: "period is too large".to_string(),
    })?;

    // Calculate smoothing factor (alpha)
    let alpha = 2.0 / (f64::from(period_u32) + 1.0);

    // Pre-allocate result vector
    let mut result = Vec::with_capacity(prices.len());

    // Initialize with first price
    let mut ema_value = prices[0];
    result.push(ema_value);

    // Calculate EMA for remaining prices
    for &price in &prices[1..] {
        ema_value = alpha.mul_add(price, (1.0 - alpha) * ema_value);
        result.push(ema_value);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ema_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = exponential_moving_average_impl(&prices, 3).unwrap();

        assert_eq!(result.len(), 5);
        // First value should be the first price
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        // Values should increase monotonically for this input
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_ema_single_value() {
        let prices = vec![100.0];
        let result = exponential_moving_average_impl(&prices, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_constant_prices() {
        let prices = vec![50.0, 50.0, 50.0, 50.0, 50.0];
        let result = exponential_moving_average_impl(&prices, 3).unwrap();

        // EMA of constant values should remain constant
        for &val in &result {
            assert_relative_eq!(val, 50.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ema_invalid_period_zero() {
        let prices = vec![1.0, 2.0, 3.0];
        let result = exponential_moving_average_impl(&prices, 0);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OmegaError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn test_ema_period_too_large() {
        let prices = vec![1.0, 2.0, 3.0];
        let result = exponential_moving_average_impl(&prices, 10);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OmegaError::InsufficientData { .. }
        ));
    }

    #[test]
    fn test_ema_empty_prices() {
        let prices: Vec<f64> = vec![];
        let result = exponential_moving_average_impl(&prices, 3);

        assert!(result.is_err());
    }

    #[test]
    fn test_ema_known_values() {
        // Test against known EMA calculation
        // Period 3: alpha = 2/(3+1) = 0.5
        let prices = vec![10.0, 12.0, 14.0, 16.0];
        let result = exponential_moving_average_impl(&prices, 3).unwrap();

        // Manual calculation:
        // EMA[0] = 10.0
        // EMA[1] = 0.5 * 12.0 + 0.5 * 10.0 = 11.0
        // EMA[2] = 0.5 * 14.0 + 0.5 * 11.0 = 12.5
        // EMA[3] = 0.5 * 16.0 + 0.5 * 12.5 = 14.25
        assert_relative_eq!(result[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 11.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 12.5, epsilon = 1e-10);
        assert_relative_eq!(result[3], 14.25, epsilon = 1e-10);
    }
}
