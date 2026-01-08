//! Robustness Score 1 calculation module.
//!
//! Computes parameter jitter robustness score based on relative drops
//! in profit, `avg_r`, winrate, and drawdown increase.
//!
//! ## Formula
//!
//! ```text
//! For each jitter repeat:
//!     drop = mean(
//!         pct_drop(profit),
//!         pct_drop(avg_r),
//!         pct_drop(winrate),
//!         pct_drop(drawdown, invert=True)
//!     )
//!
//! penalty = mean(drops)
//! penalty = clamp(penalty, 0, penalty_cap)
//! score = 1 - penalty
//! ```
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/robustness_score_1.py`

use pyo3::prelude::*;

use crate::rating::common::{nan_mean, pct_drop, to_finite};

/// Metrics required for robustness score calculation.
#[derive(Debug, Clone, Default)]
pub struct RobustnessMetrics {
    pub profit: f64,
    pub avg_r: f64,
    pub winrate: f64,
    pub drawdown: f64,
}

impl RobustnessMetrics {
    /// Create new metrics from values, converting to finite.
    #[inline]
    pub fn new(profit: f64, avg_r: f64, winrate: f64, drawdown: f64) -> Self {
        Self {
            profit: to_finite(profit, 0.0),
            avg_r: to_finite(avg_r, 0.0),
            winrate: to_finite(winrate, 0.0),
            drawdown: to_finite(drawdown, 0.0),
        }
    }
}

/// Compute robustness score 1 (parameter jitter).
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_avg_r` - Base average R-multiple
/// * `base_winrate` - Base win rate
/// * `base_drawdown` - Base drawdown
/// * `jitter_metrics` - Slice of jittered metrics
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
pub fn compute_robustness_score_1_impl(
    base_profit: f64,
    base_avg_r: f64,
    base_winrate: f64,
    base_drawdown: f64,
    jitter_metrics: &[RobustnessMetrics],
    penalty_cap: f64,
) -> f64 {
    let cap = penalty_cap.max(0.0);
    if cap == 0.0 {
        return 1.0;
    }

    let base_profit_safe = to_finite(base_profit, 0.0);
    let base_avg_r_safe = to_finite(base_avg_r, 0.0);
    let base_winrate_safe = to_finite(base_winrate, 0.0);
    let base_drawdown_safe = to_finite(base_drawdown, 0.0);

    if jitter_metrics.is_empty() {
        return (1.0 - cap).max(0.0);
    }

    let mut drops: Vec<f64> = Vec::with_capacity(jitter_metrics.len());

    for m in jitter_metrics {
        let profit_drop = pct_drop(base_profit_safe, m.profit, false);
        let avg_r_drop = pct_drop(base_avg_r_safe, m.avg_r, false);
        let winrate_drop = pct_drop(base_winrate_safe, m.winrate, false);
        let drawdown_increase = pct_drop(base_drawdown_safe, m.drawdown, true);

        let d = (profit_drop + avg_r_drop + winrate_drop + drawdown_increase) / 4.0;
        if d.is_finite() {
            drops.push(d);
        }
    }

    let penalty = if drops.is_empty() {
        cap
    } else {
        let mean = nan_mean(&drops);
        if mean.is_finite() {
            mean.max(0.0).min(cap)
        } else {
            cap
        }
    };

    (1.0 - penalty).max(0.0)
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Compute robustness score 1 (parameter jitter).
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_avg_r` - Base average R-multiple
/// * `base_winrate` - Base win rate
/// * `base_drawdown` - Base drawdown
/// * `jitter_profits` - List of jittered profit values
/// * `jitter_avg_rs` - List of jittered `avg_r` values
/// * `jitter_winrates` - List of jittered winrate values
/// * `jitter_drawdowns` - List of jittered drawdown values
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
#[pyfunction]
#[pyo3(signature = (base_profit, base_avg_r, base_winrate, base_drawdown, jitter_profits, jitter_avg_rs, jitter_winrates, jitter_drawdowns, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_robustness_score_1(
    base_profit: f64,
    base_avg_r: f64,
    base_winrate: f64,
    base_drawdown: f64,
    jitter_profits: Vec<f64>,
    jitter_avg_rs: Vec<f64>,
    jitter_winrates: Vec<f64>,
    jitter_drawdowns: Vec<f64>,
    penalty_cap: f64,
) -> PyResult<f64> {
    // Build jitter metrics from parallel arrays
    let len = jitter_profits
        .len()
        .min(jitter_avg_rs.len())
        .min(jitter_winrates.len())
        .min(jitter_drawdowns.len());

    let jitter_metrics: Vec<RobustnessMetrics> = (0..len)
        .map(|i| {
            RobustnessMetrics::new(
                jitter_profits[i],
                jitter_avg_rs[i],
                jitter_winrates[i],
                jitter_drawdowns[i],
            )
        })
        .collect();

    Ok(compute_robustness_score_1_impl(
        base_profit,
        base_avg_r,
        base_winrate,
        base_drawdown,
        &jitter_metrics,
        penalty_cap,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_robustness_metrics_new() {
        let m = RobustnessMetrics::new(1000.0, 1.5, 55.0, 200.0);
        assert_relative_eq!(m.profit, 1000.0, epsilon = 1e-10);
        assert_relative_eq!(m.avg_r, 1.5, epsilon = 1e-10);
        assert_relative_eq!(m.winrate, 55.0, epsilon = 1e-10);
        assert_relative_eq!(m.drawdown, 200.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_empty_jitter() {
        // Empty jitter metrics should return 1 - cap
        let score = compute_robustness_score_1_impl(1000.0, 1.5, 55.0, 200.0, &[], 0.5);
        assert_relative_eq!(score, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_zero_cap() {
        let jitter = vec![RobustnessMetrics::new(500.0, 0.5, 40.0, 400.0)];
        let score = compute_robustness_score_1_impl(1000.0, 1.5, 55.0, 200.0, &jitter, 0.0);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_identical_metrics() {
        // No change from base should give score of 1.0
        let jitter = vec![RobustnessMetrics::new(1000.0, 1.5, 55.0, 200.0)];
        let score = compute_robustness_score_1_impl(1000.0, 1.5, 55.0, 200.0, &jitter, 0.5);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_calculation() {
        // Base: profit=1000, avg_r=1.0, winrate=50, drawdown=100
        // Jitter: profit=800, avg_r=0.8, winrate=40, drawdown=150
        // pct_drop(profit) = (1000-800)/1000 = 0.2
        // pct_drop(avg_r) = (1.0-0.8)/1.0 = 0.2
        // pct_drop(winrate) = (50-40)/50 = 0.2
        // pct_drop(drawdown, invert) = (150-100)/100 = 0.5
        // d = (0.2 + 0.2 + 0.2 + 0.5) / 4 = 0.275
        // score = 1 - 0.275 = 0.725
        let jitter = vec![RobustnessMetrics::new(800.0, 0.8, 40.0, 150.0)];
        let score = compute_robustness_score_1_impl(1000.0, 1.0, 50.0, 100.0, &jitter, 0.5);
        assert_relative_eq!(score, 0.725, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_penalty_capping() {
        // Severe degradation should be capped
        let jitter = vec![RobustnessMetrics::new(0.0, 0.0, 0.0, 1000.0)];
        let score = compute_robustness_score_1_impl(1000.0, 1.0, 50.0, 100.0, &jitter, 0.5);
        assert_relative_eq!(score, 0.5, epsilon = 1e-10); // 1 - 0.5 (cap)
    }

    #[test]
    fn test_robustness_multiple_jitters() {
        // Two jitter runs: one good, one bad
        let jitter = vec![
            RobustnessMetrics::new(900.0, 0.9, 48.0, 120.0), // ~10% drops
            RobustnessMetrics::new(700.0, 0.7, 35.0, 200.0), // ~30% drops
        ];
        let score = compute_robustness_score_1_impl(1000.0, 1.0, 50.0, 100.0, &jitter, 0.5);
        // Average penalty should be between individual penalties
        assert!(score > 0.5);
        assert!(score < 1.0);
    }
}
