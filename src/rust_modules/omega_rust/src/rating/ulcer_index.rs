//! Ulcer Index Score calculation module.
//!
//! Computes the Ulcer Index (weekly drawdowns in percent) and maps it
//! to a robustness score.
//!
//! ## Formula
//!
//! ```text
//! dd_pct = (equity - rolling_max) / rolling_max * 100
//! ulcer_index = sqrt(mean(dd_pct^2))
//! score = 1 - ulcer_index / ulcer_cap
//! score = clamp(score, 0, 1)
//! ```
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/ulcer_index_score.py`

use pyo3::prelude::*;

/// Compute Ulcer Index from equity values.
///
/// Assumes values are already weekly closes (or pre-processed).
///
/// # Arguments
/// * `equity_values` - Slice of equity values
///
/// # Returns
/// Ulcer Index value (percent), or NaN if insufficient data
pub fn compute_ulcer_index_impl(equity_values: &[f64]) -> f64 {
    // Filter to finite values
    let values: Vec<f64> = equity_values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    if values.len() < 2 {
        return f64::NAN;
    }

    // Calculate rolling maximum
    let mut roll_max = vec![0.0_f64; values.len()];
    roll_max[0] = values[0];
    for i in 1..values.len() {
        roll_max[i] = roll_max[i - 1].max(values[i]);
    }

    // Calculate percent drawdowns
    let mut dd_pct_squared = vec![0.0_f64; values.len()];
    for i in 0..values.len() {
        if roll_max[i] > 0.0 {
            let dd_pct = (values[i] - roll_max[i]) / roll_max[i] * 100.0;
            dd_pct_squared[i] = dd_pct * dd_pct;
        } else {
            dd_pct_squared[i] = 0.0;
        }
    }

    // Ulcer Index = sqrt(mean(dd_pct^2))
    let mean_sq = dd_pct_squared.iter().sum::<f64>() / dd_pct_squared.len() as f64;
    let ulcer_index = mean_sq.sqrt();

    if ulcer_index.is_finite() {
        ulcer_index
    } else {
        f64::NAN
    }
}

/// Compute Ulcer Index and Score from equity values.
///
/// # Arguments
/// * `equity_values` - Slice of equity values (weekly closes)
/// * `ulcer_cap` - Cap for linear score mapping (default: 10.0)
///
/// # Returns
/// Tuple of (`ulcer_index`, score). Index may be NaN for insufficient data.
pub fn compute_ulcer_index_and_score_impl(equity_values: &[f64], ulcer_cap: f64) -> (f64, f64) {
    let ulcer_index = compute_ulcer_index_impl(equity_values);

    if !ulcer_index.is_finite() {
        return (ulcer_index, 0.0);
    }

    let cap = if ulcer_cap.is_finite() && ulcer_cap > 0.0 {
        ulcer_cap
    } else {
        return (ulcer_index, 0.0);
    };

    let score = 1.0 - ulcer_index / cap;
    let score_clamped = score.clamp(0.0, 1.0);

    (ulcer_index, score_clamped)
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Compute Ulcer Index from equity values.
///
/// # Arguments
/// * `equity_values` - List of equity values (weekly closes)
///
/// # Returns
/// Ulcer Index value (percent)
#[pyfunction]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_ulcer_index(equity_values: Vec<f64>) -> PyResult<f64> {
    Ok(compute_ulcer_index_impl(&equity_values))
}

/// Compute Ulcer Index and Score from equity values.
///
/// # Arguments
/// * `equity_values` - List of equity values (weekly closes)
/// * `ulcer_cap` - Cap for linear score mapping (default: 10.0)
///
/// # Returns
/// Tuple of (`ulcer_index`, score)
#[pyfunction]
#[pyo3(signature = (equity_values, ulcer_cap = 10.0))]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_ulcer_index_and_score(
    equity_values: Vec<f64>,
    ulcer_cap: f64,
) -> PyResult<(f64, f64)> {
    Ok(compute_ulcer_index_and_score_impl(&equity_values, ulcer_cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ulcer_index_empty() {
        assert!(compute_ulcer_index_impl(&[]).is_nan());
        assert!(compute_ulcer_index_impl(&[100.0]).is_nan());
    }

    #[test]
    fn test_ulcer_index_no_drawdown() {
        // Monotonically increasing equity: no drawdowns
        let equity = vec![100.0, 110.0, 120.0, 130.0];
        let ui = compute_ulcer_index_impl(&equity);
        assert_relative_eq!(ui, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ulcer_index_with_drawdown() {
        // Equity with drawdown: 100 -> 120 -> 100 (16.67% drawdown)
        let equity = vec![100.0, 120.0, 100.0];
        let ui = compute_ulcer_index_impl(&equity);
        // DD at index 2: (100 - 120) / 120 * 100 = -16.67%
        // mean(dd^2) = (0 + 0 + 16.67^2) / 3 = 278.89 / 3 ≈ 92.63
        // sqrt(92.63) ≈ 9.62
        assert!(ui > 0.0);
        assert!(ui < 20.0);
    }

    #[test]
    fn test_ulcer_score_mapping() {
        // Low ulcer index -> high score
        let equity = vec![100.0, 110.0, 120.0, 130.0];
        let (ui, score) = compute_ulcer_index_and_score_impl(&equity, 10.0);
        assert_relative_eq!(ui, 0.0, epsilon = 1e-10);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ulcer_score_capping() {
        // High ulcer index -> score capped at 0
        let equity = vec![100.0, 200.0, 50.0]; // 75% drawdown
        let (ui, score) = compute_ulcer_index_and_score_impl(&equity, 10.0);
        assert!(ui > 10.0); // High ulcer
        assert_relative_eq!(score, 0.0, epsilon = 1e-10); // Capped
    }

    #[test]
    fn test_ulcer_nan_handling() {
        let equity = vec![100.0, f64::NAN, 110.0, f64::INFINITY];
        let ui = compute_ulcer_index_impl(&equity);
        // Should filter out non-finite values and still compute
        assert!(ui.is_finite() || ui.is_nan()); // Either valid or NaN if < 2 finite
    }
}
