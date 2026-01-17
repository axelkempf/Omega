//! Output formatting helpers for metrics.

use std::collections::HashMap;

use omega_types::{MetricDefinition, Metrics};
use serde::{Deserialize, Serialize};

const CURRENCY_DECIMALS: u32 = 2;
const RATIO_DECIMALS: u32 = 6;

/// Output payload for metrics including definitions.
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsOutput {
    /// Computed metric values.
    pub metrics: Metrics,
    /// Metric definition catalog keyed by metric name.
    pub definitions: HashMap<String, MetricDefinition>,
}

impl MetricsOutput {
    /// Creates a new output payload for metrics and definitions.
    #[must_use]
    pub fn new(metrics: Metrics, definitions: HashMap<String, MetricDefinition>) -> Self {
        Self { metrics, definitions }
    }
}

/// Rounds metrics according to the output contract.
#[must_use]
pub fn round_metrics(mut metrics: Metrics) -> Metrics {
    metrics.profit_gross = round_to_decimals(metrics.profit_gross, CURRENCY_DECIMALS);
    metrics.profit_net = round_to_decimals(metrics.profit_net, CURRENCY_DECIMALS);
    metrics.fees_total = round_to_decimals(metrics.fees_total, CURRENCY_DECIMALS);
    metrics.max_drawdown_abs = round_to_decimals(metrics.max_drawdown_abs, CURRENCY_DECIMALS);
    metrics.avg_trade_pnl = round_to_decimals(metrics.avg_trade_pnl, CURRENCY_DECIMALS);
    metrics.avg_win = round_to_decimals(metrics.avg_win, CURRENCY_DECIMALS);
    metrics.avg_loss = round_to_decimals(metrics.avg_loss, CURRENCY_DECIMALS);
    metrics.largest_win = round_to_decimals(metrics.largest_win, CURRENCY_DECIMALS);
    metrics.largest_loss = round_to_decimals(metrics.largest_loss, CURRENCY_DECIMALS);

    metrics.win_rate = round_to_decimals(metrics.win_rate, RATIO_DECIMALS);
    metrics.max_drawdown = round_to_decimals(metrics.max_drawdown, RATIO_DECIMALS);
    metrics.avg_r_multiple = round_to_decimals(metrics.avg_r_multiple, RATIO_DECIMALS);
    metrics.profit_factor = round_to_decimals(metrics.profit_factor, RATIO_DECIMALS);
    metrics.expectancy = round_to_decimals(metrics.expectancy, RATIO_DECIMALS);
    metrics.trades_per_day = round_to_decimals(metrics.trades_per_day, RATIO_DECIMALS);
    metrics.sharpe_ratio = round_to_decimals(metrics.sharpe_ratio, RATIO_DECIMALS);
    metrics.sortino_ratio = round_to_decimals(metrics.sortino_ratio, RATIO_DECIMALS);
    metrics.calmar_ratio = round_to_decimals(metrics.calmar_ratio, RATIO_DECIMALS);

    metrics
}

#[allow(clippy::cast_possible_wrap)] // decimals is always small (< 10)
fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    (value * factor).round() / factor
}
