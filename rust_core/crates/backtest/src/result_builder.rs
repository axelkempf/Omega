//! Backtest result assembly helpers.

use omega_metrics::compute::compute_metrics;
use omega_types::{BacktestResult, EquityPoint, ResultMeta, Trade};

/// Builds a successful backtest result payload.
#[must_use]
pub(crate) fn build_result(
    trades: Vec<Trade>,
    equity_curve: Vec<EquityPoint>,
    fees_total: f64,
    risk_per_trade: f64,
    meta: ResultMeta,
    profiling: Option<serde_json::Value>,
) -> BacktestResult {
    let metrics_output = compute_metrics(&trades, &equity_curve, fees_total, risk_per_trade);

    BacktestResult {
        ok: true,
        error: None,
        trades: Some(trades),
        metrics: Some(metrics_output.metrics),
        metric_definitions: Some(metrics_output.definitions),
        equity_curve: Some(equity_curve),
        profiling,
        meta: Some(meta),
    }
}

/// Builds metadata for a backtest result.
#[must_use]
pub(crate) fn build_meta(
    timestamps: &[i64],
    warmup_bars: usize,
    runtime_seconds: f64,
    extra: serde_json::Value,
) -> ResultMeta {
    let start_timestamp = timestamps.get(warmup_bars).copied();
    let end_timestamp = timestamps.last().copied();
    let candles_processed = timestamps.len().saturating_sub(warmup_bars) as u64;

    ResultMeta {
        runtime_seconds,
        candles_processed,
        start_timestamp,
        end_timestamp,
        extra,
    }
}
