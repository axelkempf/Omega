//! Trade Dropout Score calculation module.
//!
//! Computes trade dropout robustness by simulating random trade removal
//! and measuring degradation in performance metrics.
//!
//! Uses `ChaCha8` RNG for deterministic, reproducible random selection.
//!
//! ## Formula
//!
//! ```text
//! 1. Randomly remove dropout_frac of trades
//! 2. Recompute profit, drawdown, sharpe from remaining trades
//! 3. Apply stress penalty scoring
//! ```
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/trade_dropout_score.py`

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::rating::stress_penalty::{
    compute_penalty_profit_drawdown_sharpe_impl, score_from_penalty_impl, StressMetrics,
};

/// Trade data for dropout simulation.
#[derive(Debug, Clone)]
pub struct TradeData {
    /// Net `PnL` (result - fees)
    pub pnl: f64,
    /// R-multiple for Sharpe calculation
    pub r_multiple: f64,
}

/// Calculate drawdown from a series of `PnL` results.
fn drawdown_from_results(results: &[f64]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    // Cumulative sum with leading zero
    let mut cum_pnl = vec![0.0_f64; results.len() + 1];
    for (i, &pnl) in results.iter().enumerate() {
        cum_pnl[i + 1] = cum_pnl[i] + pnl;
    }

    // Running maximum
    let mut peaks = vec![0.0_f64; cum_pnl.len()];
    peaks[0] = cum_pnl[0];
    for i in 1..cum_pnl.len() {
        peaks[i] = peaks[i - 1].max(cum_pnl[i]);
    }

    // Maximum drawdown
    let mut max_dd = 0.0_f64;
    for i in 0..cum_pnl.len() {
        let dd = peaks[i] - cum_pnl[i];
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate Sharpe ratio from R-multiples.
fn sharpe_from_r_multiples(r_multiples: &[f64]) -> f64 {
    // Filter finite values
    let valid: Vec<f64> = r_multiples.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.len() < 2 {
        return 0.0;
    }

    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;

    // Sample standard deviation (ddof=1)
    let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();

    if std > 0.0 {
        mean / std
    } else {
        0.0
    }
}

/// Simulate trade dropout and compute metrics.
///
/// # Arguments
/// * `trades` - Slice of trade data
/// * `dropout_frac` - Fraction of trades to remove (0.0-1.0)
/// * `seed` - Seed for deterministic random selection
///
/// # Returns
/// Tuple of (profit, drawdown, sharpe) from remaining trades
pub fn simulate_trade_dropout_metrics_impl(
    trades: &[TradeData],
    dropout_frac: f64,
    seed: u64,
) -> (f64, f64, f64) {
    if trades.is_empty() || dropout_frac <= 0.0 {
        // No dropout, return from all trades
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let r_mults: Vec<f64> = trades.iter().map(|t| t.r_multiple).collect();
        let profit: f64 = pnls.iter().sum();
        let dd = drawdown_from_results(&pnls);
        let sharpe = sharpe_from_r_multiples(&r_mults);
        return (profit, dd, sharpe);
    }

    let n = trades.len();
    let n_drop = ((n as f64 * dropout_frac).ceil() as usize).clamp(1, n);

    // Create RNG and shuffle indices to select drops
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    // Take first n_drop as dropped indices
    let drop_set: std::collections::HashSet<usize> = indices.iter().take(n_drop).copied().collect();

    // Keep remaining trades
    let kept_trades: Vec<&TradeData> = trades
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop_set.contains(i))
        .map(|(_, t)| t)
        .collect();

    if kept_trades.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pnls: Vec<f64> = kept_trades.iter().map(|t| t.pnl).collect();
    let r_mults: Vec<f64> = kept_trades.iter().map(|t| t.r_multiple).collect();

    let profit: f64 = pnls.iter().sum();
    let dd = drawdown_from_results(&pnls);
    let sharpe = sharpe_from_r_multiples(&r_mults);

    (profit, dd, sharpe)
}

/// Compute trade dropout score.
///
/// Uses stress penalty to compare base metrics with dropout metrics.
pub fn compute_trade_dropout_score_impl(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    dropout_profit: f64,
    dropout_drawdown: f64,
    dropout_sharpe: f64,
    penalty_cap: f64,
) -> f64 {
    let dropout_metrics = vec![StressMetrics::new(
        dropout_profit,
        dropout_drawdown,
        dropout_sharpe,
    )];

    let penalty = compute_penalty_profit_drawdown_sharpe_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        &dropout_metrics,
        penalty_cap,
    );

    score_from_penalty_impl(penalty, penalty_cap)
}

/// Compute multi-run trade dropout score.
///
/// Averages scores across multiple dropout simulation runs.
pub fn compute_multi_run_trade_dropout_score_impl(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    dropout_metrics: &[StressMetrics],
    penalty_cap: f64,
) -> f64 {
    if dropout_metrics.is_empty() {
        return 1.0;
    }

    let scores: Vec<f64> = dropout_metrics
        .iter()
        .map(|dm| {
            compute_trade_dropout_score_impl(
                base_profit,
                base_drawdown,
                base_sharpe,
                dm.profit,
                dm.drawdown,
                dm.sharpe,
                penalty_cap,
            )
        })
        .collect();

    scores.iter().sum::<f64>() / scores.len() as f64
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Simulate trade dropout and compute metrics.
///
/// # Arguments
/// * `pnls` - List of trade `PnL` values (net of fees)
/// * `r_multiples` - List of R-multiples (parallel to pnls)
/// * `dropout_frac` - Fraction of trades to remove
/// * `seed` - Seed for deterministic random selection
///
/// # Returns
/// Tuple of (profit, drawdown, sharpe)
#[pyfunction]
#[pyo3(signature = (pnls, r_multiples, dropout_frac, seed = 987_654_321))]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn simulate_trade_dropout_metrics(
    pnls: Vec<f64>,
    r_multiples: Vec<f64>,
    dropout_frac: f64,
    seed: u64,
) -> PyResult<(f64, f64, f64)> {
    let len = pnls.len().min(r_multiples.len());

    let trades: Vec<TradeData> = (0..len)
        .map(|i| TradeData {
            pnl: pnls[i],
            r_multiple: r_multiples[i],
        })
        .collect();

    Ok(simulate_trade_dropout_metrics_impl(&trades, dropout_frac, seed))
}

/// Compute trade dropout score.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `dropout_profit` - Dropout simulation profit
/// * `dropout_drawdown` - Dropout simulation drawdown
/// * `dropout_sharpe` - Dropout simulation sharpe
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Score value in [0.0, 1.0]
#[pyfunction]
#[pyo3(signature = (base_profit, base_drawdown, base_sharpe, dropout_profit, dropout_drawdown, dropout_sharpe, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
pub fn compute_trade_dropout_score(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    dropout_profit: f64,
    dropout_drawdown: f64,
    dropout_sharpe: f64,
    penalty_cap: f64,
) -> PyResult<f64> {
    Ok(compute_trade_dropout_score_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        dropout_profit,
        dropout_drawdown,
        dropout_sharpe,
        penalty_cap,
    ))
}

/// Compute multi-run trade dropout score.
///
/// # Arguments
/// * `base_profit` - Base profit value
/// * `base_drawdown` - Base drawdown value
/// * `base_sharpe` - Base sharpe ratio
/// * `dropout_profits` - List of dropout simulation profits
/// * `dropout_drawdowns` - List of dropout simulation drawdowns
/// * `dropout_sharpes` - List of dropout simulation sharpes
/// * `penalty_cap` - Maximum penalty value (default: 0.5)
///
/// # Returns
/// Mean score across all runs
#[pyfunction]
#[pyo3(signature = (base_profit, base_drawdown, base_sharpe, dropout_profits, dropout_drawdowns, dropout_sharpes, penalty_cap = 0.5))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_multi_run_trade_dropout_score(
    base_profit: f64,
    base_drawdown: f64,
    base_sharpe: f64,
    dropout_profits: Vec<f64>,
    dropout_drawdowns: Vec<f64>,
    dropout_sharpes: Vec<f64>,
    penalty_cap: f64,
) -> PyResult<f64> {
    let len = dropout_profits
        .len()
        .min(dropout_drawdowns.len())
        .min(dropout_sharpes.len());

    let dropout_metrics: Vec<StressMetrics> = (0..len)
        .map(|i| {
            StressMetrics::new(dropout_profits[i], dropout_drawdowns[i], dropout_sharpes[i])
        })
        .collect();

    Ok(compute_multi_run_trade_dropout_score_impl(
        base_profit,
        base_drawdown,
        base_sharpe,
        &dropout_metrics,
        penalty_cap,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_drawdown_calculation() {
        // Simple winning series: no drawdown
        let results = vec![100.0, 100.0, 100.0];
        let dd = drawdown_from_results(&results);
        assert_relative_eq!(dd, 0.0, epsilon = 1e-10);

        // Simple drawdown
        let results = vec![100.0, -50.0, 50.0];
        let dd = drawdown_from_results(&results);
        assert_relative_eq!(dd, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sharpe_calculation() {
        // Consistent positive returns
        let r = vec![1.0, 1.0, 1.0, 1.0];
        let sharpe = sharpe_from_r_multiples(&r);
        // All same values -> std = 0 -> sharpe = 0
        assert_relative_eq!(sharpe, 0.0, epsilon = 1e-10);

        // Variable returns
        let r = vec![1.0, 2.0, 1.0, 2.0];
        let sharpe = sharpe_from_r_multiples(&r);
        // Mean = 1.5, std = 0.577... -> sharpe ≈ 2.598
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_simulate_dropout_deterministic() {
        let trades = vec![
            TradeData { pnl: 100.0, r_multiple: 1.0 },
            TradeData { pnl: 200.0, r_multiple: 2.0 },
            TradeData { pnl: -50.0, r_multiple: -0.5 },
            TradeData { pnl: 150.0, r_multiple: 1.5 },
        ];

        // Same seed should give same results
        let (p1, d1, s1) = simulate_trade_dropout_metrics_impl(&trades, 0.25, 42);
        let (p2, d2, s2) = simulate_trade_dropout_metrics_impl(&trades, 0.25, 42);

        assert_relative_eq!(p1, p2, epsilon = 1e-10);
        assert_relative_eq!(d1, d2, epsilon = 1e-10);
        assert_relative_eq!(s1, s2, epsilon = 1e-10);
    }

    #[test]
    fn test_simulate_dropout_different_seeds() {
        let trades = vec![
            TradeData { pnl: 100.0, r_multiple: 1.0 },
            TradeData { pnl: 200.0, r_multiple: 2.0 },
            TradeData { pnl: -50.0, r_multiple: -0.5 },
            TradeData { pnl: 150.0, r_multiple: 1.5 },
        ];

        // Different seeds should give different results (usually)
        let (p1, _, _) = simulate_trade_dropout_metrics_impl(&trades, 0.25, 42);
        let (p2, _, _) = simulate_trade_dropout_metrics_impl(&trades, 0.25, 123);

        // Not guaranteed to be different, but very likely with different drops
        // At least verify both run without error
        assert!(p1.is_finite());
        assert!(p2.is_finite());
    }

    #[test]
    fn test_dropout_score_no_change() {
        let score = compute_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, 1000.0, 100.0, 1.5, 0.5,
        );
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_score_degradation() {
        let score = compute_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, 800.0, 150.0, 1.2, 0.5,
        );
        assert!(score > 0.5);
        assert!(score < 1.0);
    }

    #[test]
    fn test_multi_run_empty() {
        let score = compute_multi_run_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, &[], 0.5,
        );
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_run_average() {
        let metrics = vec![
            StressMetrics::new(900.0, 120.0, 1.4),
            StressMetrics::new(800.0, 150.0, 1.2),
        ];

        let score = compute_multi_run_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, &metrics, 0.5,
        );

        let s1 = compute_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, 900.0, 120.0, 1.4, 0.5,
        );
        let s2 = compute_trade_dropout_score_impl(
            1000.0, 100.0, 1.5, 800.0, 150.0, 1.2, 0.5,
        );

        assert_relative_eq!(score, (s1 + s2) / 2.0, epsilon = 1e-10);
    }
}
