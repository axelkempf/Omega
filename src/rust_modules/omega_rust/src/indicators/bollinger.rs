//! Bollinger Bands implementation.
//!
//! Bollinger Bands consist of:
//! - Middle Band: SMA of close prices
//! - Upper Band: Middle Band + (std_factor × standard deviation)
//! - Lower Band: Middle Band - (std_factor × standard deviation)
//!
//! ## Variants
//!
//! - `bollinger`: Standard calculation on all bars
//! - `bollinger_stepwise`: HTF-bar aware calculation to prevent carry_forward drift
//!
//! ## Performance Target
//!
//! 20x speedup over Python baseline for stepwise variant
//!
//! ## Reference
//!
//! - FFI Specification: `docs/ffi/indicator_cache.md`

use crate::error::{OmegaError, Result};
use crate::indicators::sma::{rolling_std_impl, sma_impl};

/// Bollinger Bands result containing upper, middle, and lower bands.
pub struct BollingerResult {
    /// Upper band (middle + std_factor × std)
    pub upper: Vec<f64>,
    /// Middle band (SMA)
    pub middle: Vec<f64>,
    /// Lower band (middle - std_factor × std)
    pub lower: Vec<f64>,
}

/// Calculate Bollinger Bands.
///
/// # Arguments
///
/// * `close` - Close prices
/// * `period` - SMA period (typically 20)
/// * `std_factor` - Standard deviation multiplier (typically 2.0)
///
/// # Returns
///
/// `BollingerResult` containing upper, middle, and lower bands.
///
/// # Errors
///
/// Returns an error if:
/// - `period` is 0 or 1
/// - `close` is empty
pub fn bollinger_impl(close: &[f64], period: usize, std_factor: f64) -> Result<BollingerResult> {
    if period <= 1 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 1".to_string(),
        });
    }

    let n = close.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    // Calculate middle band (SMA)
    let middle = sma_impl(close, period)?;

    // Calculate rolling standard deviation (sample std, ddof=1)
    let std = rolling_std_impl(close, period, 1)?;

    // Calculate upper and lower bands
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in 0..n {
        let mid = middle[i];
        let s = std[i];

        if mid.is_finite() && s.is_finite() {
            let band_width = std_factor * s;
            upper[i] = mid + band_width;
            lower[i] = mid - band_width;
        }
    }

    Ok(BollingerResult {
        upper,
        middle,
        lower,
    })
}

/// Calculate Bollinger Bands with stepwise HTF-bar semantics.
///
/// This variant calculates Bollinger Bands only at "new" bar indices
/// (where the candle object changed), then forward-fills to the primary raster.
/// This prevents artificial "drift" caused by carry_forward repetitions.
///
/// # Arguments
///
/// * `close` - Close prices (aligned to primary raster)
/// * `new_bar_indices` - Indices where a new HTF bar starts
/// * `period` - SMA period
/// * `std_factor` - Standard deviation multiplier
///
/// # Returns
///
/// `BollingerResult` with values computed only at new_bar_indices and forward-filled.
pub fn bollinger_stepwise_impl(
    close: &[f64],
    new_bar_indices: &[usize],
    period: usize,
    std_factor: f64,
) -> Result<BollingerResult> {
    if period <= 1 {
        return Err(OmegaError::InvalidParameter {
            reason: "period must be greater than 1".to_string(),
        });
    }

    let n = close.len();
    if n == 0 {
        return Ok(BollingerResult {
            upper: vec![],
            middle: vec![],
            lower: vec![],
        });
    }

    if new_bar_indices.is_empty() {
        return Ok(BollingerResult {
            upper: vec![f64::NAN; n],
            middle: vec![f64::NAN; n],
            lower: vec![f64::NAN; n],
        });
    }

    // Extract reduced series (one close per HTF bar)
    let reduced_close: Vec<f64> = new_bar_indices
        .iter()
        .filter_map(|&idx| if idx < n { Some(close[idx]) } else { None })
        .collect();

    if reduced_close.is_empty() {
        return Ok(BollingerResult {
            upper: vec![f64::NAN; n],
            middle: vec![f64::NAN; n],
            lower: vec![f64::NAN; n],
        });
    }

    // Calculate Bollinger Bands on reduced series
    let reduced_result = bollinger_impl(&reduced_close, period, std_factor)?;

    // Expand to full length with forward-fill
    let upper = expand_and_ffill(&reduced_result.upper, new_bar_indices, n);
    let middle = expand_and_ffill(&reduced_result.middle, new_bar_indices, n);
    let lower = expand_and_ffill(&reduced_result.lower, new_bar_indices, n);

    Ok(BollingerResult {
        upper,
        middle,
        lower,
    })
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
    fn test_bollinger_basic() {
        let close = vec![20.0, 21.0, 22.0, 21.5, 22.5, 23.0, 22.0, 23.5, 24.0, 23.0];
        let result = bollinger_impl(&close, 3, 2.0).unwrap();

        assert_eq!(result.upper.len(), 10);
        assert_eq!(result.middle.len(), 10);
        assert_eq!(result.lower.len(), 10);

        // First 2 values should be NaN
        assert!(result.upper[0].is_nan());
        assert!(result.upper[1].is_nan());

        // Third value should be valid
        assert!(result.upper[2].is_finite());
        assert!(result.middle[2].is_finite());
        assert!(result.lower[2].is_finite());

        // Upper > Middle > Lower
        for i in 2..10 {
            if result.middle[i].is_finite() {
                assert!(result.upper[i] >= result.middle[i]);
                assert!(result.middle[i] >= result.lower[i]);
            }
        }
    }

    #[test]
    fn test_bollinger_constant_prices() {
        let close = vec![100.0; 10];
        let result = bollinger_impl(&close, 3, 2.0).unwrap();

        // For constant prices, std = 0, so upper = middle = lower
        for i in 2..10 {
            assert_relative_eq!(result.upper[i], 100.0, epsilon = 1e-10);
            assert_relative_eq!(result.middle[i], 100.0, epsilon = 1e-10);
            assert_relative_eq!(result.lower[i], 100.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bollinger_stepwise() {
        let close = vec![20.0, 20.0, 21.0, 21.0, 22.0, 22.0, 23.0, 23.0];
        let new_bar_indices = vec![0, 2, 4, 6]; // Every 2nd bar is "new"

        let result = bollinger_stepwise_impl(&close, &new_bar_indices, 2, 2.0).unwrap();

        assert_eq!(result.upper.len(), 8);

        // Values at indices 1, 3, 5, 7 should be forward-filled from 0, 2, 4, 6
        if result.middle[0].is_finite() {
            assert_relative_eq!(result.middle[1], result.middle[0], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bollinger_invalid_period() {
        let close = vec![1.0, 2.0, 3.0];
        let result = bollinger_impl(&close, 0, 2.0);
        assert!(result.is_err());

        let result = bollinger_impl(&close, 1, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_and_ffill() {
        let reduced = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 3, 6];
        let result = expand_and_ffill(&reduced, &indices, 9);

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10); // forward-filled
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10); // forward-filled
        assert_relative_eq!(result[3], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 2.0, epsilon = 1e-10); // forward-filled
        assert_relative_eq!(result[5], 2.0, epsilon = 1e-10); // forward-filled
        assert_relative_eq!(result[6], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[7], 3.0, epsilon = 1e-10); // forward-filled
        assert_relative_eq!(result[8], 3.0, epsilon = 1e-10); // forward-filled
    }
}
