//! Python bindings for Omega V2 backtests.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macro-generated code

use omega_backtest::runner::run_backtest_from_json;
use omega_metrics as _;
use omega_types::{BacktestResult, ErrorResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python module definition for the Omega backtest engine.
#[pymodule]
fn omega_bt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}

/// Runs a backtest from JSON config and returns the result JSON.
#[pyfunction]
fn run_backtest(config_json: &str) -> PyResult<String> {
    match run_backtest_from_json(config_json) {
        Ok(result_json) => Ok(result_json),
        Err(err) if err.is_config_error() => Err(PyValueError::new_err(err.to_string())),
        Err(err) => {
            let error_result = BacktestResult {
                ok: false,
                error: Some(ErrorResult::from(err)),
                trades: None,
                metrics: None,
                metric_definitions: None,
                equity_curve: None,
                meta: None,
            };
            let error_json = serde_json::to_string(&error_result).unwrap_or_else(|_| {
                "{\"ok\":false,\"error\":{\"category\":\"runtime\",\"message\":\"serialization_failed\"}}".to_string()
            });
            Ok(error_json)
        }
    }
}
