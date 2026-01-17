//! Python bindings for Omega V2 backtests.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macro-generated code

mod entry;
mod error;
mod result;

use omega_metrics as _;
use pyo3::prelude::*;

/// Python module definition for the Omega backtest engine.
#[pymodule]
fn omega_bt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entry::run_backtest, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use serde_json::Value as JsonValue;

    use crate::entry::run_backtest;

    static INIT: Once = Once::new();

    fn init_python() {
        INIT.call_once(pyo3::prepare_freethreaded_python);
    }

    fn valid_config_with_missing_data() -> String {
        let config = serde_json::json!({
            "schema_version": "2",
            "strategy_name": "mean_reversion_z_score",
            "symbol": "MISSINGDATA",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-02T00:00:00Z",
            "run_mode": "dev",
            "data_mode": "candle",
            "execution_variant": "v2",
            "timeframes": {
                "primary": "M1",
                "additional": [],
                "additional_source": "separate_parquet"
            },
            "warmup_bars": 1,
            "rng_seed": 42,
            "costs": {"enabled": false},
            "strategy_parameters": {
                "ema_length": 2,
                "atr_length": 1,
                "atr_mult": 1.0,
                "window_length": 2,
                "z_score_long": -0.5,
                "z_score_short": 0.5,
                "htf_filter": "none",
                "extra_htf_filter": "none",
                "enabled_scenarios": []
            }
        });

        serde_json::to_string(&config).expect("config json")
    }

    #[test]
    fn test_config_error_returns_python_exception() {
        init_python();
        Python::with_gil(|py| {
            let err = run_backtest("{invalid_json").unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn test_runtime_error_returns_error_json_category() {
        init_python();
        Python::with_gil(|_| {
            let config_json = valid_config_with_missing_data();
            let result_json = run_backtest(&config_json).expect("error json");
            let parsed: JsonValue = serde_json::from_str(&result_json).expect("parsed json");

            assert_eq!(parsed["ok"], JsonValue::Bool(false));
            assert_eq!(
                parsed["error"]["category"],
                JsonValue::String("market_data".to_string())
            );
        });
    }
}
