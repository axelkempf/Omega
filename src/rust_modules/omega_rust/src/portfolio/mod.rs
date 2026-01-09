//! Portfolio module for high-performance backtesting.
//!
//! Provides Rust implementations of portfolio state management for the Omega Trading System.
//! This is **Wave 2** of the Rust migration.
//!
//! ## Available Types
//!
//! - [`PositionRust`] - Single trading position representation
//! - [`PortfolioRust`] - Portfolio state manager with position tracking
//! - [`BatchOperation`] - Enum for batch processing operations
//! - [`BatchResult`] - Result of batch processing
//!
//! ## Available Structs (internal)
//!
//! - [`PortfolioState`] - Internal state tracking
//! - [`EquityPoint`] - Equity curve data point
//! - [`FeeLogEntry`] - Fee logging entry
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_2_PORTFOLIO_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/portfolio.md`

mod batch;
mod position;
#[allow(clippy::module_inception)]
mod portfolio;
mod state;

pub use batch::{BatchOperation, BatchResult};
pub use position::{PositionRust, DIRECTION_LONG, DIRECTION_SHORT};
pub use portfolio::PortfolioRust;
pub use state::{EquityPoint, FeeLogEntry, PortfolioState};
