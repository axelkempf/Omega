//! MACD (Moving Average Convergence Divergence) implementation.
//!
//! MACD shows the relationship between two EMAs of prices.
//!
//! ## Components
//!
//! - **MACD Line**: fast_ema - slow_ema (typically 12-day EMA - 26-day EMA)
//! - **Signal Line**: EMA of MACD Line (typically 9-day)
//! - **Histogram**: MACD Line - Signal Line
//!
//! ## Interpretation
//!
//! - MACD > Signal: Bullish momentum
//! - MACD < Signal: Bearish momentum
//! - Histogram shows strength of momentum

use crate::error::{OmegaError, Result};
use crate::indicators::ema_extended::ema_impl;

/// MACD result containing MACD line, signal line, and histogram.
pub struct MacdResult {
    /// MACD Line (fast_ema - slow_ema)
    pub macd: Vec<f64>,
    /// Signal Line (EMA of MACD)
    pub signal: Vec<f64>,
    /// Histogram (MACD - Signal)
    pub histogram: Vec<f64>,
}

/// Calculate MACD.
///
/// # Arguments
///
/// * `close` - Close prices
/// * `fast_span` - Fast EMA span (typically 12)
/// * `slow_span` - Slow EMA span (typically 26)
/// * `signal_span` - Signal line EMA span (typically 9)
///
/// # Returns
///
/// `MacdResult` containing MACD line, signal line, and histogram.
///
/// # Errors
///
/// Returns an error if:
/// - Any span is 0
/// - `fast_span` >= `slow_span`
/// - `close` is empty
pub fn macd_impl(
    close: &[f64],
    fast_span: usize,
    slow_span: usize,
    signal_span: usize,
) -> Result<MacdResult> {
    if fast_span == 0 || slow_span == 0 || signal_span == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "All spans must be greater than 0".to_string(),
        });
    }

    if fast_span >= slow_span {
        return Err(OmegaError::InvalidParameter {
            reason: format!(
                "fast_span ({}) must be less than slow_span ({})",
                fast_span, slow_span
            ),
        });
    }

    let n = close.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    // Calculate fast and slow EMAs
    let fast_ema = ema_impl(close, fast_span, None)?;
    let slow_ema = ema_impl(close, slow_span, None)?;

    // Calculate MACD line
    let mut macd = vec![f64::NAN; n];
    for i in 0..n {
        let f = fast_ema[i];
        let s = slow_ema[i];
        if f.is_finite() && s.is_finite() {
            macd[i] = f - s;
        }
    }

    // Calculate Signal line
    let signal = ema_impl(&macd, signal_span, None)?;

    // Calculate Histogram
    let mut histogram = vec![f64::NAN; n];
    for i in 0..n {
        let m = macd[i];
        let s = signal[i];
        if m.is_finite() && s.is_finite() {
            histogram[i] = m - s;
        }
    }

    Ok(MacdResult {
        macd,
        signal,
        histogram,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_basic() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = macd_impl(&close, 12, 26, 9).unwrap();

        assert_eq!(result.macd.len(), 30);
        assert_eq!(result.signal.len(), 30);
        assert_eq!(result.histogram.len(), 30);

        // For uptrending data, MACD should be positive
        let last_macd = result.macd[29];
        assert!(last_macd.is_finite());
        assert!(last_macd > 0.0);
    }

    #[test]
    fn test_macd_invalid_spans() {
        let close = vec![1.0, 2.0, 3.0];

        // Zero span
        let result = macd_impl(&close, 0, 26, 9);
        assert!(result.is_err());

        // fast >= slow
        let result = macd_impl(&close, 26, 12, 9);
        assert!(result.is_err());
    }

    #[test]
    fn test_macd_constant_price() {
        let close = vec![100.0; 30];
        let result = macd_impl(&close, 12, 26, 9).unwrap();

        // For constant price, MACD = 0 (both EMAs equal)
        let last_macd = result.macd[29];
        if last_macd.is_finite() {
            assert!((last_macd).abs() < 1e-10);
        }
    }
}
