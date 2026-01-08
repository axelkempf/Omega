//! Rating score modules for strategy evaluation.
//!
//! This module provides high-performance Rust implementations of the
//! rating/robustness score calculations used in the Omega backtest system.
//!
//! ## Wave 1 Migration
//!
//! These modules are part of the Wave 1 Rust migration, following the
//! Wave 0 pilot (slippage/fee). All functions maintain numerical parity
//! with their Python counterparts.
//!
//! ## Available Modules
//!
//! - [`common`] - Shared helper functions
//! - [`stress_penalty`] - Base stress penalty calculation
//! - [`robustness`] - Parameter jitter robustness score
//! - [`stability`] - Yearly profit stability score
//! - [`cost_shock`] - Cost shock robustness score
//! - [`trade_dropout`] - Trade dropout robustness score
//! - [`ulcer_index`] - Ulcer Index score
//!
//! ## Feature Flag
//!
//! The Python wrapper uses `OMEGA_USE_RUST_RATING` environment variable
//! to enable/disable Rust implementations. Default is `true`.
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_1_RATING_MODULE_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/rating_modules.md`
//! - Python Baseline: `src/backtest_engine/rating/`

pub mod common;
pub mod cost_shock;
pub mod robustness;
pub mod stability;
pub mod stress_penalty;
pub mod trade_dropout;
pub mod ulcer_index;

// Re-export key functions for easier access
pub use cost_shock::{
    compute_cost_shock_score, compute_multi_factor_cost_shock_score,
};
pub use robustness::compute_robustness_score_1;
pub use stability::{compute_stability_score, compute_stability_score_and_wmape};
pub use stress_penalty::{compute_penalty_profit_drawdown_sharpe, score_from_penalty};
pub use trade_dropout::{
    compute_multi_run_trade_dropout_score, compute_trade_dropout_score,
    simulate_trade_dropout_metrics,
};
pub use ulcer_index::{compute_ulcer_index, compute_ulcer_index_and_score};
