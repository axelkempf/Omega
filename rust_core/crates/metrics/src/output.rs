//! Output formatting helpers for metrics.

use std::collections::BTreeMap;

use omega_types::{MetricDefinition, MetricValue, Metrics};
use serde::{Deserialize, Serialize};

const CURRENCY_DECIMALS: u32 = 2;
const RATIO_DECIMALS: u32 = 6;

/// Output payload for metrics including definitions.
/// Uses `BTreeMap` for deterministic (sorted) key order in JSON output.
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsOutput {
    /// Computed metric values.
    pub metrics: Metrics,
    /// Metric definition catalog keyed by metric name (sorted for determinism).
    pub definitions: BTreeMap<String, MetricDefinition>,
}

impl MetricsOutput {
    /// Creates a new output payload for metrics and definitions.
    #[must_use]
    pub fn new(metrics: Metrics, definitions: BTreeMap<String, MetricDefinition>) -> Self {
        Self {
            metrics,
            definitions,
        }
    }
}

/// Rounds metrics according to the output contract.
/// Currency values: 2 decimal places, Ratios: 6 decimal places.
#[must_use]
pub fn round_metrics(mut metrics: Metrics) -> Metrics {
    // Currency values (2 decimal places)
    metrics.profit_gross = round_to_decimals(metrics.profit_gross, CURRENCY_DECIMALS);
    metrics.profit_net = round_to_decimals(metrics.profit_net, CURRENCY_DECIMALS);
    metrics.fees_total = round_to_decimals(metrics.fees_total, CURRENCY_DECIMALS);
    metrics.max_drawdown_abs = round_to_decimals(metrics.max_drawdown_abs, CURRENCY_DECIMALS);
    metrics.avg_trade_pnl = round_to_decimals(metrics.avg_trade_pnl, CURRENCY_DECIMALS);
    metrics.avg_win = round_to_decimals(metrics.avg_win, CURRENCY_DECIMALS);
    metrics.avg_loss = round_to_decimals(metrics.avg_loss, CURRENCY_DECIMALS);
    metrics.largest_win = round_to_decimals(metrics.largest_win, CURRENCY_DECIMALS);
    metrics.largest_loss = round_to_decimals(metrics.largest_loss, CURRENCY_DECIMALS);

    // Ratio values (6 decimal places)
    metrics.win_rate = round_to_decimals(metrics.win_rate, RATIO_DECIMALS);
    metrics.max_drawdown = round_to_decimals(metrics.max_drawdown, RATIO_DECIMALS);
    metrics.avg_r_multiple = round_to_decimals(metrics.avg_r_multiple, RATIO_DECIMALS);
    metrics.profit_factor = round_to_decimals(metrics.profit_factor, RATIO_DECIMALS);
    metrics.expectancy = round_to_decimals(metrics.expectancy, RATIO_DECIMALS);
    metrics.trades_per_day = round_to_decimals(metrics.trades_per_day, RATIO_DECIMALS);
    metrics.sharpe_trade_r = round_metric_value(metrics.sharpe_trade_r, RATIO_DECIMALS);
    metrics.sortino_trade_r = round_metric_value(metrics.sortino_trade_r, RATIO_DECIMALS);
    metrics.sharpe_equity_daily =
        round_metric_value(metrics.sharpe_equity_daily, RATIO_DECIMALS);
    metrics.sortino_equity_daily =
        round_metric_value(metrics.sortino_equity_daily, RATIO_DECIMALS);
    metrics.calmar_ratio = round_to_decimals(metrics.calmar_ratio, RATIO_DECIMALS);

    metrics
}

#[allow(clippy::cast_possible_wrap)] // decimals is always small (< 10)
fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    (value * factor).round() / factor
}

fn round_metric_value(value: MetricValue, decimals: u32) -> MetricValue {
    match value {
        MetricValue::Number(number) => MetricValue::Number(round_to_decimals(number, decimals)),
        MetricValue::Text(text) => MetricValue::Text(text),
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unreadable_literal, clippy::approx_constant)]
mod tests {
    use super::*;
    use omega_types::MetricValue;

    #[test]
    fn test_round_to_decimals_currency() {
        // Currency: 2 decimal places
        assert_eq!(round_to_decimals(123.456, 2), 123.46);
        assert_eq!(round_to_decimals(123.454, 2), 123.45);
        assert_eq!(round_to_decimals(123.455, 2), 123.46); // Round half up
        assert_eq!(round_to_decimals(0.0, 2), 0.0);
        assert_eq!(round_to_decimals(-123.456, 2), -123.46);
    }

    #[test]
    fn test_round_to_decimals_ratio() {
        // Ratio: 6 decimal places
        assert_eq!(round_to_decimals(0.123456789, 6), 0.123457);
        assert_eq!(round_to_decimals(0.123456123, 6), 0.123456);
        assert_eq!(round_to_decimals(1.0, 6), 1.0);
        assert_eq!(round_to_decimals(0.0, 6), 0.0);
    }

    #[test]
    fn test_round_metrics_currency_fields() {
        let metrics = Metrics {
            profit_gross: 1234.5678,
            profit_net: 1134.5678,
            fees_total: 100.0001,
            max_drawdown_abs: 500.999,
            avg_trade_pnl: 56.7891,
            avg_win: 100.1234,
            avg_loss: 50.5678,
            largest_win: 250.9999,
            largest_loss: 125.0001,
            ..Metrics::default()
        };

        let rounded = round_metrics(metrics);

        // All currency fields should be rounded to 2 decimal places
        assert_eq!(rounded.profit_gross, 1234.57);
        assert_eq!(rounded.profit_net, 1134.57);
        assert_eq!(rounded.fees_total, 100.0);
        assert_eq!(rounded.max_drawdown_abs, 501.0);
        assert_eq!(rounded.avg_trade_pnl, 56.79);
        assert_eq!(rounded.avg_win, 100.12);
        assert_eq!(rounded.avg_loss, 50.57);
        assert_eq!(rounded.largest_win, 251.0);
        assert_eq!(rounded.largest_loss, 125.0);
    }

    #[test]
    fn test_round_metrics_ratio_fields() {
        let metrics = Metrics {
            win_rate: 0.5234567890,
            max_drawdown: 0.1234567890,
            avg_r_multiple: 1.2345678901,
            profit_factor: 2.3456789012,
            expectancy: 0.9876543210,
            trades_per_day: 3.1415926535,
            sharpe_trade_r: MetricValue::Number(1.4142135623),
            sortino_trade_r: MetricValue::Number(1.7320508075),
            sharpe_equity_daily: MetricValue::Number(2.2360679774),
            sortino_equity_daily: MetricValue::Number(2.6457513110),
            calmar_ratio: 3.1415926535,
            ..Metrics::default()
        };

        let rounded = round_metrics(metrics);

        // All ratio fields should be rounded to 6 decimal places
        assert_eq!(rounded.win_rate, 0.523457);
        assert_eq!(rounded.max_drawdown, 0.123457);
        assert_eq!(rounded.avg_r_multiple, 1.234568);
        assert_eq!(rounded.profit_factor, 2.345679);
        assert_eq!(rounded.expectancy, 0.987654);
        assert_eq!(rounded.trades_per_day, 3.141593);
        assert_eq!(rounded.sharpe_trade_r, MetricValue::Number(1.414214));
        assert_eq!(rounded.sortino_trade_r, MetricValue::Number(1.732051));
        assert_eq!(rounded.sharpe_equity_daily, MetricValue::Number(2.236068));
        assert_eq!(rounded.sortino_equity_daily, MetricValue::Number(2.645751));
        assert_eq!(rounded.calmar_ratio, 3.141593);
    }

    #[test]
    fn test_round_metrics_exact_values_no_change() {
        let metrics = Metrics {
            profit_gross: 100.0,
            win_rate: 0.5,
            ..Metrics::default()
        };

        let rounded = round_metrics(metrics);

        assert_eq!(rounded.profit_gross, 100.0);
        assert_eq!(rounded.win_rate, 0.5);
    }
}
