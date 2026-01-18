//! FFI error handling and Python exception mapping.

use omega_backtest::BacktestError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::result::serialize_error;

/// Handles backtest errors, returning appropriate Python responses.
///
/// Config errors (invalid JSON, validation failures) raise `PyValueError`.
/// Runtime errors (data loading, execution) return error JSON with `ok: false`.
///
/// # Arguments
///
/// * `err` - The backtest error to handle.
///
/// # Returns
///
/// `PyResult<String>` - Either raises `PyValueError` or returns error JSON.
pub fn handle_backtest_error(err: BacktestError) -> PyResult<String> {
    if err.is_config_error() {
        return Err(PyValueError::new_err(err.to_string()));
    }

    Ok(serialize_error(err))
}
