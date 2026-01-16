//! Event loop driver.

use crate::engine::BacktestEngine;

/// Runs the main backtest event loop.
pub fn run_event_loop(engine: &mut BacktestEngine) {
    let warmup = engine.warmup_bars();
    let len = engine.primary_len();

    tracing::info!(
        "Starting backtest: {} bars ({} warmup, {} trading)",
        len,
        warmup,
        len.saturating_sub(warmup)
    );

    for idx in warmup..len {
        engine.process_bar(idx);
    }
}
