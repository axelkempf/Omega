//! Python function entry points.

use omega_backtest::runner::run_backtest_from_json;
use pyo3::prelude::*;

use crate::error::handle_backtest_error;

/// Runs a backtest from JSON config and returns the result JSON.
///
/// # Arguments
///
/// * `config_json` - JSON string containing backtest configuration.
///
/// # Returns
///
/// JSON string containing `BacktestResult` on success or error.
///
/// # Errors
///
/// Returns `PyValueError` if config is invalid JSON or fails validation.
/// Runtime errors are returned as JSON with `ok: false`.
#[pyfunction]
pub fn run_backtest(config_json: &str) -> PyResult<String> {
    match run_backtest_from_json(config_json) {
        Ok(result_json) => Ok(result_json),
        Err(err) => handle_backtest_error(err),
    }
}
