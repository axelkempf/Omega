//! # Omega Rust Extensions
//!
//! High-performance Rust implementations for the Omega Trading System.
//!
//! This crate provides Python bindings via `PyO3` for performance-critical
//! numerical algorithms, including:
//!
//! - Technical indicators (EMA, RSI, etc.)
//! - Statistical calculations
//! - Cost calculations (Slippage, Fee) - Wave 0 Pilot
//! - Portfolio management - Wave 2
//!
//! ## Usage from Python
//!
//! ```python
//! from omega._rust import ema, rsi, rolling_std
//!
//! prices = [100.0, 101.5, 99.8, 102.3, 103.1]
//! ema_values = ema(prices, period=3)
//!
//! # Wave 0: Slippage & Fee calculations
//! from omega._rust import calculate_slippage, calculate_fee
//!
//! adjusted_price = calculate_slippage(1.10000, 1, 0.0001, 0.5, 1.0, seed=42)
//! fee = calculate_fee(1.0, 1.10000, 100_000.0, 30.0, 0.01)
//!
//! # Wave 2: Portfolio management
//! from omega_rust import PortfolioRust, PositionRust
//!
//! portfolio = PortfolioRust(initial_balance=100000.0)
//! pos = PositionRust(1704067200000000, 1, "EURUSD", 1.1, 1.099, 1.102, 1.0, 100.0)
//! portfolio.register_entry(pos)
//! ```
//!
//! ## Performance
//!
//! Rust implementations typically achieve 10-50x speedup over pure Python
//! for numerical computations, especially with large datasets.

use pyo3::prelude::*;

pub mod costs;
pub mod error;
pub mod indicators;
pub mod portfolio;

use costs::{calculate_fee, calculate_fee_batch, calculate_slippage, calculate_slippage_batch};
use error::get_error_code_constants;
use indicators::{ema, exponential_moving_average, rolling_std, rsi};
use portfolio::{PortfolioRust, PositionRust};

/// Omega Rust Extension Module
///
/// This module is the entry point for Python bindings.
/// All public functions are exposed to Python via `PyO3`.
///
/// The module is named `omega_rust` (not `_rust`) so it can be installed as a
/// standalone package and imported directly. The main `omega` package re-exports
/// this as `omega._rust` for a cleaner API.
#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register indicator functions
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(exponential_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;

    // Register cost functions (Wave 0 Pilot)
    m.add_function(wrap_pyfunction!(calculate_slippage, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_slippage_batch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fee, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fee_batch, m)?)?;

    // Register Portfolio classes (Wave 2)
    m.add_class::<PortfolioRust>()?;
    m.add_class::<PositionRust>()?;

    // Register error code constants for cross-language verification
    m.add_function(wrap_pyfunction!(get_error_code_constants, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Axel Kempf")?;

    Ok(())
}
