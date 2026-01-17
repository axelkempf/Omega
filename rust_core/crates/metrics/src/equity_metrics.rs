//! Helpers for equity-curve derived metrics.

use std::collections::BTreeMap;

use omega_types::EquityPoint;

/// Nanoseconds per day for UTC day grouping.
const NS_PER_DAY: i64 = 86_400_000_000_000;

/// Computes maximum drawdown (relative, absolute) and duration in bars.
#[must_use]
pub fn compute_drawdown(equity: &[EquityPoint]) -> (f64, f64, u64) {
    if equity.is_empty() {
        return (0.0, 0.0, 0);
    }

    let mut high_water: f64 = equity[0].equity;
    let mut max_dd_rel: f64 = 0.0;
    let mut max_dd_abs: f64 = 0.0;
    let mut current_dd_start = 0usize;
    let mut max_dd_duration = 0u64;
    let mut in_drawdown = false;

    for (idx, point) in equity.iter().enumerate() {
        if point.equity >= high_water {
            // Recovery or new high - not in drawdown
            if in_drawdown {
                let duration = (idx - current_dd_start) as u64;
                max_dd_duration = max_dd_duration.max(duration);
                in_drawdown = false;
            }
            high_water = point.equity;
        } else if high_water > 0.0 {
            // Actual drawdown (equity < high_water)
            if !in_drawdown {
                current_dd_start = idx;
                in_drawdown = true;
            }

            let dd_abs: f64 = high_water - point.equity;
            let dd_rel: f64 = dd_abs / high_water;

            max_dd_abs = max_dd_abs.max(dd_abs);
            max_dd_rel = max_dd_rel.max(dd_rel);
        }
    }

    if in_drawdown {
        let duration = (equity.len() - current_dd_start) as u64;
        max_dd_duration = max_dd_duration.max(duration);
    }

    max_dd_rel = max_dd_rel.clamp(0.0, 1.0);

    (max_dd_rel, max_dd_abs, max_dd_duration)
}

/// Computes daily returns from an equity curve.
/// Groups equity points by UTC day and calculates day-over-day returns.
#[must_use]
pub fn compute_daily_returns(equity: &[EquityPoint]) -> Vec<f64> {
    if equity.len() < 2 {
        return Vec::new();
    }

    // Group by UTC day, taking the last equity value of each day
    let mut daily_equity: BTreeMap<i64, f64> = BTreeMap::new();
    for point in equity {
        let day = point.timestamp_ns / NS_PER_DAY;
        daily_equity.insert(day, point.equity);
    }

    // Calculate daily returns
    let equities: Vec<f64> = daily_equity.values().copied().collect();
    let mut returns = Vec::with_capacity(equities.len().saturating_sub(1));

    for window in equities.windows(2) {
        let prev = window[0];
        let curr = window[1];
        if prev > 0.0 {
            returns.push((curr - prev) / prev);
        }
    }

    returns
}

/// Computes the Sharpe ratio for a series of returns.
/// Returns 0.0 if fewer than 2 samples or if standard deviation is 0.
/// Risk-free rate is assumed to be 0.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Return counts never exceed f64 mantissa precision
pub fn sharpe_ratio(returns: &[f64], annualization_factor: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;

    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0); // Sample variance with Bessel's correction

    let std_dev = variance.sqrt();

    if std_dev <= 0.0 || !std_dev.is_finite() {
        return 0.0;
    }

    (mean / std_dev) * annualization_factor
}

/// Computes the Sortino ratio for a series of returns.
/// Returns 0.0 if fewer than 2 samples or if downside deviation is 0.
/// Uses target return of 0 (downside = negative returns only).
#[must_use]
#[allow(clippy::cast_precision_loss)] // Return counts never exceed f64 mantissa precision
pub fn sortino_ratio(returns: &[f64], annualization_factor: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;

    // Downside deviation: sqrt of mean squared negative returns
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside_returns.is_empty() {
        // No negative returns, Sortino is undefined (infinite)
        // Return 0 as per n/a policy (will be handled upstream)
        return 0.0;
    }

    let downside_variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n; // Divide by total n, not downside count (semi-deviation convention)

    let downside_dev = downside_variance.sqrt();

    if downside_dev <= 0.0 || !downside_dev.is_finite() {
        return 0.0;
    }

    (mean / downside_dev) * annualization_factor
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unreadable_literal)]
mod tests {
    use super::*;

    fn make_equity_point(timestamp_ns: i64, equity: f64) -> EquityPoint {
        EquityPoint {
            timestamp_ns,
            equity,
            balance: equity,
            drawdown: 0.0,
            high_water: equity,
        }
    }

    // ============ compute_drawdown tests ============

    #[test]
    fn test_drawdown_empty_equity() {
        let (dd_rel, dd_abs, duration) = compute_drawdown(&[]);
        assert_eq!(dd_rel, 0.0);
        assert_eq!(dd_abs, 0.0);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_drawdown_constant_equity() {
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 10000.0),
            make_equity_point(2000, 10000.0),
        ];

        let (dd_rel, dd_abs, duration) = compute_drawdown(&equity);
        assert_eq!(dd_rel, 0.0);
        assert_eq!(dd_abs, 0.0);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_drawdown_monotonic_increase() {
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 10100.0),
            make_equity_point(2000, 10200.0),
        ];

        let (dd_rel, dd_abs, duration) = compute_drawdown(&equity);
        assert_eq!(dd_rel, 0.0);
        assert_eq!(dd_abs, 0.0);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_drawdown_simple() {
        // Peak at 10000, drop to 9000, recover to 10000
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 9000.0),
            make_equity_point(2000, 10000.0),
        ];

        let (dd_rel, dd_abs, duration) = compute_drawdown(&equity);
        assert!((dd_rel - 0.1).abs() < 0.0001); // 10% drawdown
        assert!((dd_abs - 1000.0).abs() < 0.01);
        assert_eq!(duration, 1); // 1 bar in drawdown before recovery
    }

    #[test]
    fn test_drawdown_no_recovery() {
        // Peak at 10000, drop and never recover
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 9500.0),
            make_equity_point(2000, 9000.0),
            make_equity_point(3000, 8500.0),
        ];

        let (dd_rel, dd_abs, duration) = compute_drawdown(&equity);
        assert!((dd_rel - 0.15).abs() < 0.0001); // 15% drawdown
        assert!((dd_abs - 1500.0).abs() < 0.01);
        assert_eq!(duration, 3); // 3 bars until end (no recovery)
    }

    #[test]
    fn test_drawdown_multiple_drawdowns() {
        // Two drawdowns: first 5%, second 10%
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 9500.0),  // -5%
            make_equity_point(2000, 10500.0), // New high
            make_equity_point(3000, 9450.0),  // -10%
            make_equity_point(4000, 10500.0), // Recovery
        ];

        let (dd_rel, dd_abs, _duration) = compute_drawdown(&equity);
        assert!((dd_rel - 0.1).abs() < 0.0001); // Max 10% drawdown
        assert!((dd_abs - 1050.0).abs() < 0.01);
    }

    // ============ compute_daily_returns tests ============

    #[test]
    fn test_daily_returns_empty() {
        let returns = compute_daily_returns(&[]);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_daily_returns_single_point() {
        let equity = vec![make_equity_point(0, 10000.0)];
        let returns = compute_daily_returns(&equity);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_daily_returns_same_day() {
        // Multiple points on the same day should be grouped
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(1000, 10100.0),
            make_equity_point(2000, 10200.0),
        ];
        let returns = compute_daily_returns(&equity);
        // All on day 0, so only one day value (last one: 10200)
        assert!(returns.is_empty()); // Need at least 2 days for returns
    }

    #[test]
    fn test_daily_returns_two_days() {
        let ns_per_day = 86_400_000_000_000_i64;
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(ns_per_day, 10100.0),
        ];
        let returns = compute_daily_returns(&equity);
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.01).abs() < 0.0001); // 1% return
    }

    // ============ sharpe_ratio tests ============

    #[test]
    fn test_sharpe_insufficient_samples() {
        assert_eq!(sharpe_ratio(&[], 1.0), 0.0);
        assert_eq!(sharpe_ratio(&[0.01], 1.0), 0.0);
    }

    #[test]
    fn test_sharpe_zero_std_dev() {
        // All same returns -> zero std dev
        let returns = vec![0.01, 0.01, 0.01];
        assert_eq!(sharpe_ratio(&returns, 1.0), 0.0);
    }

    #[test]
    fn test_sharpe_positive() {
        let returns = vec![0.02, 0.01, 0.03, 0.02, 0.015];
        let sharpe = sharpe_ratio(&returns, 1.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sharpe_annualization() {
        let returns = vec![0.02, 0.01, 0.03, 0.02, 0.015];
        let sharpe_raw = sharpe_ratio(&returns, 1.0);
        let sharpe_annualized = sharpe_ratio(&returns, 252.0_f64.sqrt());
        assert!((sharpe_annualized - sharpe_raw * 252.0_f64.sqrt()).abs() < 0.0001);
    }

    // ============ sortino_ratio tests ============

    #[test]
    fn test_sortino_insufficient_samples() {
        assert_eq!(sortino_ratio(&[], 1.0), 0.0);
        assert_eq!(sortino_ratio(&[0.01], 1.0), 0.0);
    }

    #[test]
    fn test_sortino_no_downside() {
        // All positive returns -> no downside deviation
        let returns = vec![0.01, 0.02, 0.015];
        assert_eq!(sortino_ratio(&returns, 1.0), 0.0);
    }

    #[test]
    fn test_sortino_with_downside() {
        let returns = vec![0.02, -0.01, 0.03, -0.005, 0.015];
        let sortino = sortino_ratio(&returns, 1.0);
        assert!(sortino > 0.0); // Positive mean with some downside
    }

    #[test]
    fn test_sortino_vs_sharpe_with_downside() {
        // Sortino should generally be higher than Sharpe when positive mean
        // and downside is less than total volatility
        let returns = vec![0.02, -0.01, 0.03, -0.005, 0.015, 0.02];
        let sharpe = sharpe_ratio(&returns, 1.0);
        let sortino = sortino_ratio(&returns, 1.0);
        // Both should be positive with this data
        assert!(sharpe > 0.0);
        assert!(sortino > 0.0);
    }
}
