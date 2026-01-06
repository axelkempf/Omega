//! # Omega Rust Extensions
//!
//! High-performance Rust implementations for the Omega Trading System.
//!
//! This crate provides Python bindings via `PyO3` for performance-critical
//! numerical algorithms, including:
//!
//! - Technical indicators (EMA, RSI, etc.)
//! - Statistical calculations
//! - Event simulation (future)
//!
//! ## Usage from Python
//!
//! ```python
//! from omega._rust import ema, rsi, rolling_std
//!
//! prices = [100.0, 101.5, 99.8, 102.3, 103.1]
//! ema_values = ema(prices, period=3)
//! ```
//!
//! ## Performance
//!
//! Rust implementations typically achieve 10-50x speedup over pure Python
//! for numerical computations, especially with large datasets.

use pyo3::prelude::*;

pub mod error;
pub mod indicators;

use indicators::{ema, exponential_moving_average, rolling_std, rsi};

/// Omega Rust Extension Module
///
/// This module is the entry point for Python bindings.
/// All public functions are exposed to Python via `PyO3`.
#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register indicator functions
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(exponential_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Axel Kempf")?;

    Ok(())
}
