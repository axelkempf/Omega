//! Backtest result assembly helpers.

use omega_types::{BacktestResult, EquityPoint, ResultMeta, Trade};

/// Builds a successful backtest result payload.
#[must_use]
pub(crate) fn build_result(
    trades: Vec<Trade>,
    equity_curve: Vec<EquityPoint>,
    meta: ResultMeta,
) -> BacktestResult {
    BacktestResult {
        ok: true,
        error: None,
        trades: Some(trades),
        metrics: None,
        equity_curve: Some(equity_curve),
        meta: Some(meta),
    }
}

/// Builds metadata for a backtest result.
#[must_use]
pub(crate) fn build_meta(
    timestamps: &[i64],
    warmup_bars: usize,
    runtime_seconds: f64,
) -> ResultMeta {
    let start_timestamp = timestamps.get(warmup_bars).copied();
    let end_timestamp = timestamps.last().copied();
    let candles_processed = timestamps.len().saturating_sub(warmup_bars) as u64;

    ResultMeta {
        runtime_seconds,
        candles_processed,
        start_timestamp,
        end_timestamp,
        extra: serde_json::json!({}),
    }
}
