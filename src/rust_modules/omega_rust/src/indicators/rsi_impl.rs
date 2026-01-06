//! Relative Strength Index (RSI) implementation.
//!
//! RSI is a momentum oscillator that measures the speed and magnitude
//! of recent price changes to evaluate overbought or oversold conditions.
//!
//! ## Formula
//!
//! ```text
//! RS = Average Gain / Average Loss
//! RSI = 100 - (100 / (1 + RS))
//! ```
//!
//! ## Interpretation
//!
//! - RSI > 70: Potentially overbought
//! - RSI < 30: Potentially oversold
//! - RSI = 50: Neutral

use pyo3::prelude::*;

use crate::error::{OmegaError, Result};

/// Calculate Relative Strength Index.
///
/// # Arguments
///
/// * `prices` - Vector of price values
/// * `period` - RSI period (typically 14)
///
/// # Returns
///
/// Vector of RSI values. The first `period` values will be NaN
/// as they require a lookback period for calculation.
///
/// # Example
///
/// ```python
/// from omega._rust import rsi
///
/// prices = [44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, ...]
/// result = rsi(prices, period=14)
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - `period` is less than 2
/// - `prices` has fewer than `period + 1` elements
#[pyfunction]
#[pyo3(signature = (prices, period = 14))]
#[allow(clippy::needless_pass_by_value)]
pub fn rsi(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    rsi_impl(&prices, period).map_err(Into::into)
}

/// Internal RSI implementation.
pub fn rsi_impl(prices: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
    if period < 2 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be at least 2".to_string(),
        });
    }

    let min_required = period + 1;
    if prices.len() < min_required {
        return Err(OmegaError::InsufficientData {
            required: min_required,
            actual: prices.len(),
        });
    }

    let period_u32 = u32::try_from(period).map_err(|_| OmegaError::InvalidParameter {
        reason: "period is too large".to_string(),
    })?;
    let period_as_f64 = f64::from(period_u32);

    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    // Calculate price changes
    let mut gains = Vec::with_capacity(n - 1);
    let mut losses = Vec::with_capacity(n - 1);

    for i in 1..n {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Calculate initial average gain and loss (SMA for first period)
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period_as_f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period_as_f64;

    // Calculate first RSI value
    result[period] = calculate_rsi_value(avg_gain, avg_loss);

    // Calculate remaining RSI values using smoothed averages
    let smoothing_factor = (period_as_f64 - 1.0) / period_as_f64;
    let new_value_factor = 1.0 / period_as_f64;

    for i in period..gains.len() {
        avg_gain = avg_gain.mul_add(smoothing_factor, gains[i] * new_value_factor);
        avg_loss = avg_loss.mul_add(smoothing_factor, losses[i] * new_value_factor);
        result[i + 1] = calculate_rsi_value(avg_gain, avg_loss);
    }

    Ok(result)
}

/// Calculate RSI value from average gain and loss.
#[inline]
fn calculate_rsi_value(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss == 0.0 {
        if avg_gain == 0.0 {
            50.0 // No movement, neutral RSI
        } else {
            100.0 // Only gains, maximum RSI
        }
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rsi_basic() {
        // Generate a simple trending series
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + f64::from(i)).collect();
        let result = rsi_impl(&prices, 14).unwrap();

        assert_eq!(result.len(), 20);

        // First 14 values should be NaN
        for (idx, val) in result.iter().take(14).enumerate() {
            assert!(val.is_nan(), "Expected NaN at index {idx}");
        }

        // RSI should be 100 for pure uptrend
        for val in result.iter().take(20).skip(14) {
            assert_relative_eq!(*val, 100.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rsi_downtrend() {
        // Generate a simple downtrending series
        let prices: Vec<f64> = (0..20).map(|i| 200.0 - f64::from(i)).collect();
        let result = rsi_impl(&prices, 14).unwrap();

        // RSI should be 0 for pure downtrend
        for val in result.iter().take(20).skip(14) {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rsi_flat() {
        // Constant prices
        let prices = vec![100.0; 20];
        let result = rsi_impl(&prices, 14).unwrap();

        // RSI should be 50 for no movement
        for val in result.iter().take(20).skip(14) {
            assert_relative_eq!(*val, 50.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rsi_range() {
        // Mixed movement
        let prices = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5, 45.0, 44.75, 44.5,
            44.25, 44.0, 44.5,
        ];
        let result = rsi_impl(&prices, 14).unwrap();

        // RSI should be in valid range [0, 100]
        for val in result.iter().skip(14) {
            assert!(*val >= 0.0 && *val <= 100.0, "RSI out of range: {val}");
        }
    }

    #[test]
    fn test_rsi_invalid_period() {
        let prices = vec![1.0; 20];
        let result = rsi_impl(&prices, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_rsi_insufficient_data() {
        let prices = vec![1.0; 10];
        let result = rsi_impl(&prices, 14);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OmegaError::InsufficientData { .. }
        ));
    }
}
