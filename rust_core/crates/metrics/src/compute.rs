//! Metric computation entrypoints.

use omega_types::{EquityPoint, Metrics, Trade};

use crate::definitions::MetricDefinitions;
use crate::equity_metrics::compute_drawdown;
use crate::output::{MetricsOutput, round_metrics};
use crate::trade_metrics::{active_days, count_trades, gross_profit_loss};

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

    metrics.avg_r_multiple =
        if total_trades > 0 && risk_per_trade.is_finite() && risk_per_trade > 0.0 {
        trades
            .iter()
            .map(|trade| trade.result / risk_per_trade)
            .sum::<f64>()
            / total_trades_f
    } else {
        0.0
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

    let definitions = MetricDefinitions::definitions();
    MetricsOutput::new(round_metrics(metrics), definitions)
}
