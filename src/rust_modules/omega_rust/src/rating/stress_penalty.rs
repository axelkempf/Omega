//! Stress penalty calculation module.
//!
//! Provides the shared stress penalty computation used by multiple rating
//! scores (`cost_shock`, `trade_dropout`, `timing_jitter`, etc.).
//!
//! ## Formula
//!
//! ```text
//! penalty = mean(
//!     for each stress_metric:
//!         (rel_drop(profit) + rel_increase(drawdown) + rel_drop(sharpe)) / 3
//! )
//! penalty = clamp(penalty, 0, penalty_cap)
//! score = 1 - penalty
//! ```
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/stress_penalty.py`

use pyo3::prelude::*;

use crate::rating::common::{nan_mean, rel_drop, rel_increase, to_finite};

/// Metrics required for stress penalty calculation.
#[derive(Debug, Clone, Default)]
pub struct StressMetrics {
    pub profit: f64,
    pub drawdown: f64,
    pub sharpe: f64,
}

impl StressMetrics {
    /// Create new metrics from values, converting to finite.
    #[inline]
    pub fn new(profit: f64, drawdown: f64, sharpe: f64) -> Self {
        Self {
            profit: to_finite(profit, 0.0),
            drawdown: to_finite(drawdown, 0.0),
            sharpe: to_finite(sharpe, 0.0),
        }
    }
}

/// Compute stress penalty from base metrics and stressed metrics.
///
/// Uses profit drop, drawdown increase, and sharpe drop as penalty factors.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `stress_metrics` - Slice of stressed metrics to compare
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Penalty value in `[0.0, penalty_cap]`
pub fn compute_penalty_profit_drawdown_sharpe_impl(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    stress_metrics: &[StressMetrics],
    penalty_cap: f64,
) -> f64 {
    let cap = penalty_cap.max(0.0);
    if cap == 0.0 {
        return 0.0;
    }

    let base_profit_safe = to_finite(base_profit, 0.0);
    let base_drawdown_safe = to_finite(base_drawdown, 0.0);
    let base_sharpe_safe = to_finite(base_sharpe, 0.0);

    let mut penalties: Vec<f64> = Vec::with_capacity(stress_metrics.len());

    for metrics in stress_metrics {
        let p = rel_drop(base_profit_safe, metrics.profit);
        let d = rel_increase(base_drawdown_safe.max(1e-9), metrics.drawdown);
        let s = rel_drop(base_sharpe_safe.max(1e-9), metrics.sharpe);

        let pen = (p + d + s) / 3.0;
        if pen.is_finite() {
            penalties.push(pen);
        }
    }

    if penalties.is_empty() {
        return 0.0;
    }

    let mut penalty = nan_mean(&penalties);
    if !penalty.is_finite() {
        penalty = cap;
    }

    penalty.max(0.0).min(cap)
}

/// Convert penalty to score.
///
/// # Arguments
/// * `penalty` - Penalty value
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0] where score = 1 - penalty
pub fn score_from_penalty_impl(penalty: f64, penalty_cap: f64) -> f64 {
    let cap = penalty_cap.max(0.0);
    let pen = to_finite(penalty, cap);
    let pen_clamped = pen.max(0.0).min(cap);
    (1.0 - pen_clamped).max(0.0)
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Compute stress penalty from base metrics and stressed metrics.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `stress_profits` - List of stressed profit values
/// * `stress_drawdowns` - List of stressed drawdown values
/// * `stress_sharpes` - List of stressed sharpe ratios
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Penalty value in `[0.0, penalty_cap]`
#[pyfunction]
#[pyo3(signature = (base_profit, base_drawdown, base_sharpe, stress_profits, stress_drawdowns, stress_sharpes, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_penalty_profit_drawdown_sharpe(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    stress_profits: Vec<f64>,
    stress_drawdowns: Vec<f64>,
    stress_sharpes: Vec<f64>,
    penalty_cap: f64,
) -> PyResult<f64> {
    // Build stress metrics from parallel arrays
    let len = stress_profits
        .len()
        .min(stress_drawdowns.len())
        .min(stress_sharpes.len());

    let stress_metrics: Vec<StressMetrics> = (0..len)
        .map(|i| StressMetrics::new(stress_profits[i], stress_drawdowns[i], stress_sharpes[i]))
        .collect();

    Ok(compute_penalty_profit_drawdown_sharpe_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        &stress_metrics,
        penalty_cap,
    ))
}

/// Convert penalty to score.
///
/// # Arguments
/// * `penalty` - Penalty value
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
#[pyfunction]
#[pyo3(signature = (penalty, penalty_cap = 0.5))]
#[allow(clippy::missing_errors_doc)]
pub fn score_from_penalty(penalty: f64, penalty_cap: f64) -> PyResult<f64> {
    Ok(score_from_penalty_impl(penalty, penalty_cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stress_metrics_new() {
        let m = StressMetrics::new(100.0, 50.0, 1.5);
        assert_relative_eq!(m.profit, 100.0, epsilon = 1e-10);
        assert_relative_eq!(m.drawdown, 50.0, epsilon = 1e-10);
        assert_relative_eq!(m.sharpe, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_stress_metrics_nan_handling() {
        let m = StressMetrics::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY);
        assert_relative_eq!(m.profit, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m.drawdown, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m.sharpe, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_penalty_no_stress() {
        let penalty =
            compute_penalty_profit_drawdown_sharpe_impl(1000.0, 100.0, 1.5, &[], 0.5);
        assert_relative_eq!(penalty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_penalty_zero_cap() {
        let stress = vec![StressMetrics::new(500.0, 200.0, 0.5)];
        let penalty =
            compute_penalty_profit_drawdown_sharpe_impl(1000.0, 100.0, 1.5, &stress, 0.0);
        assert_relative_eq!(penalty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_penalty_calculation() {
        // Base: profit=1000, drawdown=100, sharpe=1.0
        // Stress: profit=800, drawdown=150, sharpe=0.8
        // rel_drop(profit) = (1000-800)/1000 = 0.2
        // rel_increase(drawdown) = (150-100)/100 = 0.5
        // rel_drop(sharpe) = (1.0-0.8)/1.0 = 0.2
        // pen = (0.2 + 0.5 + 0.2) / 3 = 0.3
        let stress = vec![StressMetrics::new(800.0, 150.0, 0.8)];
        let penalty =
            compute_penalty_profit_drawdown_sharpe_impl(1000.0, 100.0, 1.0, &stress, 0.5);
        assert_relative_eq!(penalty, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_penalty_capping() {
        // Very large degradation should be capped
        let stress = vec![StressMetrics::new(0.0, 1000.0, 0.0)];
        let penalty =
            compute_penalty_profit_drawdown_sharpe_impl(1000.0, 100.0, 1.0, &stress, 0.5);
        assert_relative_eq!(penalty, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_score_from_penalty() {
        assert_relative_eq!(score_from_penalty_impl(0.0, 0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(score_from_penalty_impl(0.3, 0.5), 0.7, epsilon = 1e-10);
        assert_relative_eq!(score_from_penalty_impl(0.5, 0.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(score_from_penalty_impl(1.0, 0.5), 0.5, epsilon = 1e-10); // Capped
    }

    #[test]
    fn test_score_from_penalty_nan() {
        // NaN penalty should use cap as default, resulting in 1 - cap
        assert_relative_eq!(
            score_from_penalty_impl(f64::NAN, 0.5),
            0.5,
            epsilon = 1e-10
        );
    }
}
