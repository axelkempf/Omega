//! Metric computation entrypoints.

use omega_types::{EquityPoint, MetricValue, Metrics, Trade};

use crate::definitions::MetricDefinitions;
use crate::equity_metrics::{compute_daily_returns, compute_drawdown, sharpe_ratio, sortino_ratio};
use crate::output::{MetricsOutput, round_metrics};
use crate::trade_metrics::{active_days, count_trades, gross_profit_loss, win_loss_stats};

/// Annualization factor for daily returns (trading days per year).
const ANNUALIZATION_FACTOR: f64 = 252.0;

/// Nanoseconds per year for time calculations.
const NS_PER_YEAR: f64 = 365.25 * 24.0 * 60.0 * 60.0 * 1_000_000_000.0;
const NA_VALUE: &str = "n/a";

/// Computes performance metrics for a completed backtest.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Trade counts will never exceed f64 mantissa precision
pub fn compute_metrics(
    trades: &[Trade],
    equity_curve: &[EquityPoint],
    fees_total: f64,
    risk_per_trade: f64,
) -> MetricsOutput {
    let mut metrics = Metrics::default();

    let (total_trades, wins, losses) = count_trades(trades);
    metrics.total_trades = total_trades;
    metrics.wins = wins;
    metrics.losses = losses;

    let total_trades_f = total_trades as f64;
    let wins_f = wins as f64;

    metrics.win_rate = if total_trades > 0 {
        wins_f / total_trades_f
    } else {
        0.0
    };

    metrics.profit_gross = trades.iter().map(|trade| trade.result).sum();
    metrics.fees_total = fees_total;
    metrics.profit_net = metrics.profit_gross - metrics.fees_total;

    let (max_dd, max_dd_abs, max_dd_duration) = compute_drawdown(equity_curve);
    metrics.max_drawdown = max_dd;
    metrics.max_drawdown_abs = max_dd_abs;
    metrics.max_drawdown_duration_bars = max_dd_duration;

    // R-multiples computation
    let r_multiples: Vec<f64> = if risk_per_trade.is_finite() && risk_per_trade > 0.0 {
        trades
            .iter()
            .map(|trade| trade.result / risk_per_trade)
            .collect()
    } else {
        Vec::new()
    };

    metrics.avg_r_multiple = if r_multiples.is_empty() {
        0.0
    } else {
        r_multiples.iter().sum::<f64>() / r_multiples.len() as f64
    };

    metrics.avg_trade_pnl = if total_trades > 0 {
        metrics.profit_net / total_trades_f
    } else {
        0.0
    };
    metrics.expectancy = metrics.avg_r_multiple;

    metrics.active_days = active_days(trades);
    let active_days_f = metrics.active_days as f64;
    metrics.trades_per_day = if metrics.active_days > 0 {
        total_trades_f / active_days_f
    } else {
        0.0
    };

    let (gross_profit, gross_loss) = gross_profit_loss(trades);
    metrics.profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };

    // Win/loss statistics
    let (avg_win, avg_loss, largest_win, largest_loss) = win_loss_stats(trades);
    metrics.avg_win = avg_win;
    metrics.avg_loss = avg_loss;
    metrics.largest_win = largest_win;
    metrics.largest_loss = largest_loss;

    // Trade-based Sharpe/Sortino (R-multiples, no annualization)
    metrics.sharpe_trade_r = metric_value_or_na(r_multiples.len(), sharpe_ratio(&r_multiples, 1.0));
    metrics.sortino_trade_r =
        metric_value_or_na(r_multiples.len(), sortino_ratio(&r_multiples, 1.0));

    // Equity-based Sharpe/Sortino (daily returns, annualized with sqrt(252))
    let daily_returns = compute_daily_returns(equity_curve);
    let annualization = ANNUALIZATION_FACTOR.sqrt();
    metrics.sharpe_equity_daily = metric_value_or_na(
        daily_returns.len(),
        sharpe_ratio(&daily_returns, annualization),
    );
    metrics.sortino_equity_daily = metric_value_or_na(
        daily_returns.len(),
        sortino_ratio(&daily_returns, annualization),
    );

    // Calmar ratio: annualized return / max_drawdown
    metrics.calmar_ratio = compute_calmar_ratio(equity_curve, max_dd);

    let definitions = MetricDefinitions::definitions();
    MetricsOutput::new(round_metrics(metrics), definitions)
}

/// Computes the Calmar ratio (annualized return / max drawdown).
/// Returns 0.0 if `max_drawdown` is 0 or equity curve is too short.
#[allow(clippy::cast_precision_loss)] // Timestamp differences never exceed f64 mantissa precision
fn compute_calmar_ratio(equity_curve: &[EquityPoint], max_drawdown: f64) -> f64 {
    if equity_curve.len() < 2 || max_drawdown <= 0.0 {
        return 0.0;
    }

    let first = &equity_curve[0];
    let last = &equity_curve[equity_curve.len() - 1];

    if first.equity <= 0.0 {
        return 0.0;
    }

    // Calculate total return
    let total_return = (last.equity - first.equity) / first.equity;

    // Calculate time span in years (nanoseconds to years)
    let time_span_ns = (last.timestamp_ns - first.timestamp_ns) as f64;
    let years = time_span_ns / NS_PER_YEAR;

    if years <= 0.0 {
        return 0.0;
    }

    // Annualized return (CAGR formula simplified for short periods)
    let annualized_return = if years >= 1.0 {
        (1.0 + total_return).powf(1.0 / years) - 1.0
    } else {
        // For periods less than a year, extrapolate linearly
        total_return / years
    };

    annualized_return / max_drawdown
}

fn metric_value_or_na(samples: usize, value: f64) -> MetricValue {
    if samples < 2 {
        MetricValue::Text(NA_VALUE.to_string())
    } else {
        MetricValue::Number(value)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unreadable_literal)]
mod tests {
    use super::*;
    use omega_types::{signal::Direction, trade::ExitReason};

    fn make_trade(result: f64, entry_time_ns: i64) -> Trade {
        Trade {
            entry_time_ns,
            exit_time_ns: entry_time_ns + 60_000_000_000, // 1 minute later
            direction: Direction::Long,
            symbol: "EURUSD".to_string(),
            entry_price: 1.1000,
            exit_price: if result > 0.0 { 1.1050 } else { 1.0950 },
            stop_loss: 1.0950,
            take_profit: 1.1100,
            size: 0.1,
            result,
            r_multiple: result / 50.0,
            reason: if result > 0.0 {
                ExitReason::TakeProfit
            } else {
                ExitReason::StopLoss
            },
            scenario_id: 1,
            meta: serde_json::json!({}),
        }
    }

    fn make_equity_point(timestamp_ns: i64, equity: f64) -> EquityPoint {
        EquityPoint {
            timestamp_ns,
            equity,
            balance: equity,
            drawdown: 0.0,
            high_water: equity,
        }
    }

    #[test]
    fn test_compute_metrics_zero_trades() {
        let trades: Vec<Trade> = vec![];
        let equity = vec![make_equity_point(0, 10000.0)];

        let output = compute_metrics(&trades, &equity, 0.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.total_trades, 0);
        assert_eq!(m.wins, 0);
        assert_eq!(m.losses, 0);
        assert_eq!(m.win_rate, 0.0);
        assert_eq!(m.avg_trade_pnl, 0.0);
        assert_eq!(m.profit_factor, 0.0);
        assert_eq!(m.avg_win, 0.0);
        assert_eq!(m.avg_loss, 0.0);
    }

    #[test]
    fn test_sharpe_sortino_na_policy() {
        let trades = vec![make_trade(100.0, 0)];
        let equity = vec![make_equity_point(0, 10000.0)];

        let output = compute_metrics(&trades, &equity, 0.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.sharpe_trade_r, MetricValue::Text("n/a".to_string()));
        assert_eq!(m.sortino_trade_r, MetricValue::Text("n/a".to_string()));
        assert_eq!(m.sharpe_equity_daily, MetricValue::Text("n/a".to_string()));
        assert_eq!(m.sortino_equity_daily, MetricValue::Text("n/a".to_string()));
    }

    #[test]
    fn test_compute_metrics_only_wins() {
        let ns_per_day = 86_400_000_000_000_i64;
        let trades = vec![
            make_trade(100.0, ns_per_day),
            make_trade(50.0, ns_per_day * 2),
            make_trade(75.0, ns_per_day * 3),
        ];
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(ns_per_day, 10100.0),
            make_equity_point(ns_per_day * 2, 10150.0),
            make_equity_point(ns_per_day * 3, 10225.0),
        ];

        let output = compute_metrics(&trades, &equity, 0.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.total_trades, 3);
        assert_eq!(m.wins, 3);
        assert_eq!(m.losses, 0);
        assert_eq!(m.win_rate, 1.0);
        assert_eq!(m.profit_factor, 0.0); // No losses -> profit_factor = 0 (edge case)
        assert!(m.avg_win > 0.0);
        assert_eq!(m.avg_loss, 0.0);
        assert_eq!(m.largest_win, 100.0);
        assert_eq!(m.largest_loss, 0.0);
    }

    #[test]
    fn test_compute_metrics_only_losses() {
        let ns_per_day = 86_400_000_000_000_i64;
        let trades = vec![
            make_trade(-100.0, ns_per_day),
            make_trade(-50.0, ns_per_day * 2),
        ];
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(ns_per_day, 9900.0),
            make_equity_point(ns_per_day * 2, 9850.0),
        ];

        let output = compute_metrics(&trades, &equity, 0.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.total_trades, 2);
        assert_eq!(m.wins, 0);
        assert_eq!(m.losses, 2);
        assert_eq!(m.win_rate, 0.0);
        assert_eq!(m.avg_win, 0.0);
        assert!(m.avg_loss > 0.0);
        assert_eq!(m.largest_win, 0.0);
        assert_eq!(m.largest_loss, 100.0);
    }

    #[test]
    fn test_compute_metrics_mixed_trades() {
        let ns_per_day = 86_400_000_000_000_i64;
        let trades = vec![
            make_trade(100.0, ns_per_day),
            make_trade(-50.0, ns_per_day * 2),
            make_trade(75.0, ns_per_day * 3),
            make_trade(-25.0, ns_per_day * 4),
        ];
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(ns_per_day, 10100.0),
            make_equity_point(ns_per_day * 2, 10050.0),
            make_equity_point(ns_per_day * 3, 10125.0),
            make_equity_point(ns_per_day * 4, 10100.0),
        ];

        let output = compute_metrics(&trades, &equity, 0.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.total_trades, 4);
        assert_eq!(m.wins, 2);
        assert_eq!(m.losses, 2);
        assert_eq!(m.win_rate, 0.5);

        // profit_factor = (100 + 75) / (50 + 25) = 175 / 75
        let expected_pf = 175.0 / 75.0;
        assert!((m.profit_factor - expected_pf).abs() < 0.000001);

        // avg_win = (100 + 75) / 2 = 87.5
        assert_eq!(m.avg_win, 87.5);

        // avg_loss = (50 + 25) / 2 = 37.5
        assert_eq!(m.avg_loss, 37.5);
    }

    #[test]
    fn test_compute_metrics_with_fees() {
        let ns_per_day = 86_400_000_000_000_i64;
        let trades = vec![make_trade(100.0, ns_per_day)];
        let equity = vec![
            make_equity_point(0, 10000.0),
            make_equity_point(ns_per_day, 10090.0), // After 10 fees
        ];

        let output = compute_metrics(&trades, &equity, 10.0, 50.0);
        let m = &output.metrics;

        assert_eq!(m.profit_gross, 100.0);
        assert_eq!(m.fees_total, 10.0);
        assert_eq!(m.profit_net, 90.0);
        assert_eq!(m.avg_trade_pnl, 90.0);
    }

    #[test]
    fn test_definitions_present_for_all_metrics() {
        let output = compute_metrics(&[], &[], 0.0, 50.0);

        // Check that definitions exist for key metrics
        assert!(output.definitions.contains_key("total_trades"));
        assert!(output.definitions.contains_key("win_rate"));
        assert!(output.definitions.contains_key("profit_net"));
        assert!(output.definitions.contains_key("max_drawdown"));
        assert!(output.definitions.contains_key("avg_win"));
        assert!(output.definitions.contains_key("avg_loss"));
        assert!(output.definitions.contains_key("sharpe_trade_r"));
        assert!(output.definitions.contains_key("sortino_trade_r"));
        assert!(output.definitions.contains_key("sharpe_equity_daily"));
        assert!(output.definitions.contains_key("sortino_equity_daily"));
        assert!(output.definitions.contains_key("calmar_ratio"));
    }

    #[test]
    fn test_definitions_sorted_deterministically() {
        let output1 = compute_metrics(&[], &[], 0.0, 50.0);
        let output2 = compute_metrics(&[], &[], 0.0, 50.0);

        // Keys should be in the same order (BTreeMap guarantees this)
        let keys1: Vec<_> = output1.definitions.keys().collect();
        let keys2: Vec<_> = output2.definitions.keys().collect();
        assert_eq!(keys1, keys2);

        // Keys should be sorted alphabetically
        let mut sorted_keys = keys1.clone();
        sorted_keys.sort();
        assert_eq!(keys1, sorted_keys);
    }
}
