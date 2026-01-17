//! Result serialization helpers.

use omega_backtest::BacktestError;
use omega_types::{BacktestResult, ErrorResult};

/// Serializes a backtest error into JSON response.
///
/// Creates a `BacktestResult` with `ok: false` and the error details,
/// then serializes it to JSON.
///
/// # Arguments
///
/// * `err` - The backtest error to serialize.
///
/// # Returns
///
/// JSON string representing the error result.
pub fn serialize_error(err: BacktestError) -> String {
    let error_result = BacktestResult {
        ok: false,
        error: Some(ErrorResult::from(err)),
        trades: None,
        metrics: None,
        metric_definitions: None,
        equity_curve: None,
        meta: None,
    };

    serde_json::to_string(&error_result).unwrap_or_else(|_| {
        r#"{"ok":false,"error":{"category":"runtime","message":"serialization_failed"}}"#
            .to_string()
    })
}
