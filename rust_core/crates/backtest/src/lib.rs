//! Omega Backtest
//!
//! Orchestrates data loading, indicator precomputation, execution,
//! portfolio management, and strategy evaluation for backtesting.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

pub mod context;
pub mod engine;
pub mod error;
pub mod event_loop;
pub mod result_builder;
pub mod runner;
pub mod warmup;

pub use engine::BacktestEngine;
pub use error::BacktestError;
pub use runner::run_backtest_from_json;
