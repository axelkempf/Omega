//! Exponential Moving Average (EMA) with stepwise support.
//!
//! This module provides EMA implementations including:
//! - Standard EMA with configurable span
//! - Stepwise EMA for HTF-bar aligned calculations
//!
//! ## Formula
//!
//! ```text
//! EMA_t = α × P_t + (1 - α) × EMA_{t-1}
//! where α = 2 / (span + 1)
//! ```
//!
//! ## Performance Target
//!
//! 20x speedup for stepwise variant over Python baseline
//!
//! ## Reference
//!
//! - FFI Specification: `docs/ffi/indicator_cache.md`

use crate::error::{OmegaError, Result};

/// Calculate Exponential Moving Average.
///
/// Uses pandas-compatible EMA formula with adjust=False.
///
/// # Arguments
///
/// * `values` - Input values (typically close prices)
/// * `span` - EMA span (period)
/// * `start_idx` - Optional index to start calculation from (deferred start)
///
/// # Returns
///
/// Vector of EMA values. Values before start_idx will be NaN.
///
/// # Errors
///
/// Returns an error if:
/// - `span` is 0
/// - `values` is empty
pub fn ema_impl(values: &[f64], span: usize, start_idx: Option<usize>) -> Result<Vec<f64>> {
    if span == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "span must be greater than 0".to_string(),
        });
    }

    let n = values.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let start = start_idx.unwrap_or(0);
    let mut result = vec![f64::NAN; n];

    if start >= n {
        return Ok(result);
    }

    let alpha = 2.0 / (span as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Find first valid value at or after start
    let first_valid = values[start..].iter().position(|&v| v.is_finite());
    let first_valid = match first_valid {
        Some(idx) => start + idx,
        None => return Ok(result),
    };

    // Initialize EMA with first valid value
    let mut ema = values[first_valid];
    result[first_valid] = ema;

    // Calculate EMA for remaining values
    for i in (first_valid + 1)..n {
        let v = values[i];
        if v.is_nan() {
            // Carry forward previous EMA
            result[i] = ema;
        } else {
            ema = alpha * v + one_minus_alpha * ema;
            result[i] = ema;
        }
    }

    Ok(result)
}

/// Calculate EMA with stepwise HTF-bar semantics.
///
/// This variant calculates EMA only at "new" bar indices, then forward-fills
/// to prevent artificial drift from repeated carry_forward values.
///
/// # Arguments
///
/// * `values` - Values aligned to primary raster
/// * `new_bar_indices` - Indices where a new HTF bar starts
/// * `span` - EMA span
///
/// # Returns
///
/// Vector with EMA values computed only at new_bar_indices and forward-filled.
pub fn ema_stepwise_impl(
    values: &[f64],
    new_bar_indices: &[usize],
    span: usize,
) -> Result<Vec<f64>> {
    if span == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "span must be greater than 0".to_string(),
        });
    }

    let n = values.len();
    if n == 0 {
        return Ok(vec![]);
    }

    if new_bar_indices.is_empty() {
        return Ok(vec![f64::NAN; n]);
    }

    // Extract values only at new bar indices
    let reduced: Vec<f64> = new_bar_indices
        .iter()
        .filter_map(|&idx| if idx < n { Some(values[idx]) } else { None })
        .collect();

    if reduced.is_empty() {
        return Ok(vec![f64::NAN; n]);
    }

    // Calculate EMA on reduced series
    let reduced_ema = ema_impl(&reduced, span, None)?;

    // Expand to full length with forward-fill
    let result = expand_and_ffill(&reduced_ema, new_bar_indices, n);

    Ok(result)
}

/// Calculate Double EMA (DEMA).
///
/// DEMA = 2 × EMA - EMA(EMA)
///
/// # Arguments
///
/// * `values` - Input values
/// * `span` - EMA span
pub fn dema_impl(values: &[f64], span: usize) -> Result<Vec<f64>> {
    let ema1 = ema_impl(values, span, None)?;
    let ema2 = ema_impl(&ema1, span, None)?;

    let n = values.len();
    let mut result = vec![f64::NAN; n];

    for i in 0..n {
        let e1 = ema1[i];
        let e2 = ema2[i];
        if e1.is_finite() && e2.is_finite() {
            result[i] = 2.0 * e1 - e2;
        }
    }

    Ok(result)
}

/// Calculate Triple EMA (TEMA).
///
/// TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
///
/// # Arguments
///
/// * `values` - Input values
/// * `span` - EMA span
pub fn tema_impl(values: &[f64], span: usize) -> Result<Vec<f64>> {
    let ema1 = ema_impl(values, span, None)?;
    let ema2 = ema_impl(&ema1, span, None)?;
    let ema3 = ema_impl(&ema2, span, None)?;

    let n = values.len();
    let mut result = vec![f64::NAN; n];

    for i in 0..n {
        let e1 = ema1[i];
        let e2 = ema2[i];
        let e3 = ema3[i];
        if e1.is_finite() && e2.is_finite() && e3.is_finite() {
            result[i] = 3.0 * e1 - 3.0 * e2 + e3;
        }
    }

    Ok(result)
}

/// Expand a reduced series to full length and forward-fill.
fn expand_and_ffill(reduced: &[f64], indices: &[usize], n: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; n];

    // Place reduced values at their indices
    for (i, &idx) in indices.iter().enumerate() {
        if idx < n && i < reduced.len() {
            result[idx] = reduced[i];
        }
    }

    // Forward fill
    let mut last_valid = f64::NAN;
    for v in &mut result {
        if v.is_finite() {
            last_valid = *v;
        } else if last_valid.is_finite() {
            *v = last_valid;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ema_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema_impl(&values, 3, None).unwrap();

        assert_eq!(result.len(), 5);
        // First value = 1.0
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        // EMA increases as prices increase
        for i in 1..5 {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_ema_deferred_start() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema_impl(&values, 2, Some(2)).unwrap();

        // First two values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value should be the first value in the deferred range
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema_with_nan() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let result = ema_impl(&values, 2, None).unwrap();

        // NaN should cause carry-forward
        assert!(result[2].is_finite()); // carry forward from index 1
    }

    #[test]
    fn test_ema_stepwise() {
        let values = vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0];
        let new_bar_indices = vec![0, 2, 4];

        let result = ema_stepwise_impl(&values, &new_bar_indices, 2).unwrap();

        assert_eq!(result.len(), 6);
        // Values at odd indices should be forward-filled
        assert_relative_eq!(result[1], result[0], epsilon = 1e-10);
        assert_relative_eq!(result[3], result[2], epsilon = 1e-10);
        assert_relative_eq!(result[5], result[4], epsilon = 1e-10);
    }

    #[test]
    fn test_ema_invalid_span() {
        let values = vec![1.0, 2.0, 3.0];
        let result = ema_impl(&values, 0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dema_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = dema_impl(&values, 3).unwrap();

        // DEMA should be more responsive than EMA
        assert!(result[7].is_finite());
    }

    #[test]
    fn test_tema_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = tema_impl(&values, 3).unwrap();

        // TEMA should be more responsive than DEMA
        assert!(result[7].is_finite());
    }
}
