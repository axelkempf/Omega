//! Rate of Change (ROC) implementation.
//!
//! ROC measures the percentage change between the current value
//! and a value n periods ago.
//!
//! ## Formula
//!
//! ```text
//! ROC = (current - previous) / previous × 100
//! ```

use crate::error::{OmegaError, Result};

/// Calculate Rate of Change.
///
/// # Arguments
///
/// * `values` - Input values (typically close prices)
/// * `period` - Number of periods to look back
///
/// # Returns
///
/// Vector of ROC values as percentages.
/// First `period` values will be NaN.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0
/// - `values` is empty
pub fn roc_impl(values: &[f64], period: usize) -> Result<Vec<f64>> {
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

    for i in period..n {
        let current = values[i];
        let previous = values[i - period];

        if current.is_finite() && previous.is_finite() && previous != 0.0 {
            result[i] = ((current - previous) / previous) * 100.0;
        }
    }

    Ok(result)
}

/// Calculate Momentum (absolute change).
///
/// Unlike ROC, this returns the absolute difference, not percentage.
///
/// # Arguments
///
/// * `values` - Input values
/// * `period` - Number of periods to look back
///
/// # Returns
///
/// Vector of momentum values (current - previous).
pub fn momentum_impl(values: &[f64], period: usize) -> Result<Vec<f64>> {
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

    for i in period..n {
        let current = values[i];
        let previous = values[i - period];

        if current.is_finite() && previous.is_finite() {
            result[i] = current - previous;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roc_basic() {
        let values = vec![100.0, 102.0, 104.0, 103.0, 105.0];
        let result = roc_impl(&values, 1).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan()); // First value is NaN
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10); // (102-100)/100 * 100 = 2%
        assert_relative_eq!(result[2], (104.0 - 102.0) / 102.0 * 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_period_2() {
        let values = vec![100.0, 110.0, 120.0, 115.0, 125.0];
        let result = roc_impl(&values, 2).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 20.0, epsilon = 1e-10); // (120-100)/100 * 100 = 20%
    }

    #[test]
    fn test_roc_with_zero() {
        let values = vec![0.0, 100.0, 200.0];
        let result = roc_impl(&values, 1).unwrap();

        // Division by zero should result in NaN
        assert!(result[1].is_nan());
    }

    #[test]
    fn test_momentum_basic() {
        let values = vec![100.0, 102.0, 104.0, 103.0, 105.0];
        let result = momentum_impl(&values, 1).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10); // 102 - 100
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10); // 104 - 102
        assert_relative_eq!(result[3], -1.0, epsilon = 1e-10); // 103 - 104
    }

    #[test]
    fn test_roc_invalid_period() {
        let values = vec![1.0, 2.0, 3.0];
        let result = roc_impl(&values, 0);
        assert!(result.is_err());
    }
}
