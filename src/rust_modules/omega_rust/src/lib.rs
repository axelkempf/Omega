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
//! - Rating scores (Robustness, Stability, etc.) - Wave 1
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
//! # Wave 1: Rating score calculations
//! from omega._rust import compute_robustness_score_1, compute_stability_score
//!
//! score = compute_stability_score([2020, 2021, 2022], [10000.0, 12000.0, 11000.0])
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
pub mod rating;

use costs::{calculate_fee, calculate_fee_batch, calculate_slippage, calculate_slippage_batch};
use error::get_error_code_constants;
use indicators::{ema, exponential_moving_average, rolling_std, rsi};
use rating::{
    compute_cost_shock_score, compute_multi_factor_cost_shock_score,
    compute_multi_run_trade_dropout_score, compute_penalty_profit_drawdown_sharpe,
    compute_robustness_score_1, compute_stability_score, compute_stability_score_and_wmape,
    compute_trade_dropout_score, compute_ulcer_index, compute_ulcer_index_and_score,
    score_from_penalty, simulate_trade_dropout_metrics,
};

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

    // Register rating score functions (Wave 1)
    m.add_function(wrap_pyfunction!(compute_penalty_profit_drawdown_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(score_from_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(compute_robustness_score_1, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stability_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stability_score_and_wmape, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cost_shock_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multi_factor_cost_shock_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_trade_dropout_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multi_run_trade_dropout_score, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_trade_dropout_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ulcer_index, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ulcer_index_and_score, m)?)?;

    // Register error code constants for cross-language verification
    m.add_function(wrap_pyfunction!(get_error_code_constants, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Axel Kempf")?;

    Ok(())
}
