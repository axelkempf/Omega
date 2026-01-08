//! Cost Shock Score calculation module.
//!
//! Computes cost shock robustness score based on degradation in performance
//! metrics when execution costs are increased.
//!
//! ## Formula
//!
//! Uses the stress penalty mechanism:
//! ```text
//! penalty = compute_penalty_profit_drawdown_sharpe(base, shocked)
//! score = score_from_penalty(penalty)
//! ```
//!
//! For multiple shocks: average of individual scores.
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/cost_shock_score.py`

use pyo3::prelude::*;

use crate::rating::stress_penalty::{
    compute_penalty_profit_drawdown_sharpe_impl, score_from_penalty_impl, StressMetrics,
};

/// Compute cost shock score for a single shocked scenario.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `shocked_profit` - Shocked profit value
/// * `shocked_drawdown` - Shocked drawdown value
/// * `shocked_sharpe` - Shocked sharpe ratio
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
pub fn compute_cost_shock_score_impl(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    shocked_profit: f64,
    shocked_drawdown: f64,
    shocked_sharpe: f64,
    penalty_cap: f64,
) -> f64 {
    let shocked_metrics = vec![StressMetrics::new(
        shocked_profit,
        shocked_drawdown,
        shocked_sharpe,
    )];

    let penalty = compute_penalty_profit_drawdown_sharpe_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        &shocked_metrics,
        penalty_cap,
    );

    score_from_penalty_impl(penalty, penalty_cap)
}

/// Compute multi-factor cost shock score.
///
/// Aggregates scores across multiple shocked scenarios.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `shocked_metrics` - Slice of shocked metrics
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Mean score across all shocks, or 1.0 if no shocks provided
pub fn compute_multi_factor_cost_shock_score_impl(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    shocked_metrics: &[StressMetrics],
    penalty_cap: f64,
) -> f64 {
    if shocked_metrics.is_empty() {
        return 1.0;
    }

    let scores: Vec<f64> = shocked_metrics
        .iter()
        .map(|sm| {
            compute_cost_shock_score_impl(
                base_profit,
                base_drawdown,
                base_sharpe,
                sm.profit,
                sm.drawdown,
                sm.sharpe,
                penalty_cap,
            )
        })
        .collect();

    scores.iter().sum::<f64>() / scores.len() as f64
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Compute cost shock score for a single shocked scenario.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `shocked_profit` - Shocked profit value
/// * `shocked_drawdown` - Shocked drawdown value
/// * `shocked_sharpe` - Shocked sharpe ratio
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
#[pyfunction]
#[pyo3(signature = (base_profit, base_drawdown, base_sharpe, shocked_profit, shocked_drawdown, shocked_sharpe, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
pub fn compute_cost_shock_score(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    shocked_profit: f64,
    shocked_drawdown: f64,
    shocked_sharpe: f64,
    penalty_cap: f64,
) -> PyResult<f64> {
    Ok(compute_cost_shock_score_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        shocked_profit,
        shocked_drawdown,
        shocked_sharpe,
        penalty_cap,
    ))
}

/// Compute multi-factor cost shock score.
///
/// Aggregates scores across multiple shocked scenarios.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `shocked_profits` - List of shocked profit values
/// * `shocked_drawdowns` - List of shocked drawdown values
/// * `shocked_sharpes` - List of shocked sharpe ratios
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Mean score across all shocks
#[pyfunction]
#[pyo3(signature = (base_profit, base_drawdown, base_sharpe, shocked_profits, shocked_drawdowns, shocked_sharpes, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_multi_factor_cost_shock_score(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    shocked_profits: Vec<f64>,
    shocked_drawdowns: Vec<f64>,
    shocked_sharpes: Vec<f64>,
    penalty_cap: f64,
) -> PyResult<f64> {
    let len = shocked_profits
        .len()
        .min(shocked_drawdowns.len())
        .min(shocked_sharpes.len());

    let shocked_metrics: Vec<StressMetrics> = (0..len)
        .map(|i| {
            StressMetrics::new(shocked_profits[i], shocked_drawdowns[i], shocked_sharpes[i])
        })
        .collect();

    Ok(compute_multi_factor_cost_shock_score_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        &shocked_metrics,
        penalty_cap,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cost_shock_no_change() {
        // Identical metrics should give score of 1.0
        let score =
            compute_cost_shock_score_impl(1000.0, 100.0, 1.5, 1000.0, 100.0, 1.5, 0.5);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cost_shock_degradation() {
        // Some degradation should give score < 1.0
        let score =
            compute_cost_shock_score_impl(1000.0, 100.0, 1.5, 800.0, 150.0, 1.2, 0.5);
        assert!(score > 0.5);
        assert!(score < 1.0);
    }

    #[test]
    fn test_cost_shock_severe() {
        // Severe degradation should hit minimum score (0.5 for cap=0.5)
        let score = compute_cost_shock_score_impl(1000.0, 100.0, 1.5, 0.0, 500.0, 0.0, 0.5);
        assert_relative_eq!(score, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_factor_empty() {
        let score = compute_multi_factor_cost_shock_score_impl(1000.0, 100.0, 1.5, &[], 0.5);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_factor_calculation() {
        // Two shocks: one mild, one severe
        let shocked_metrics = vec![
            StressMetrics::new(900.0, 120.0, 1.3), // Mild degradation
            StressMetrics::new(700.0, 200.0, 0.8), // More severe
        ];

        let score = compute_multi_factor_cost_shock_score_impl(
            1000.0,
            100.0,
            1.5,
            &shocked_metrics,
            0.5,
        );

        // Average should be between individual scores
        let score1 =
            compute_cost_shock_score_impl(1000.0, 100.0, 1.5, 900.0, 120.0, 1.3, 0.5);
        let score2 =
            compute_cost_shock_score_impl(1000.0, 100.0, 1.5, 700.0, 200.0, 0.8, 0.5);
        let expected_avg = (score1 + score2) / 2.0;

        assert_relative_eq!(score, expected_avg, epsilon = 1e-10);
    }
}
