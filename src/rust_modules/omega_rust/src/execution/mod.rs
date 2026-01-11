//! Execution Simulator module for trade execution in backtests.
//!
//! This module implements the core execution logic for processing trade signals,
//! managing pending orders, triggering entries, and evaluating exits.
//!
//! ## Wave 4 Migration
//!
//! This is the **Wave 4** module of the Rust migration, implementing a full
//! Rust execution simulator without Python fallback. The goal is to move the
//! hot path (signal processing, exit evaluation) entirely to Rust.
//!
//! ## Core Components
//!
//! - [`ExecutionSimulatorRust`] - Main simulator class (PyO3 binding)
//! - [`TradeSignal`] - Trade signal representation
//! - [`Position`] - Position state machine (pending → open → closed)
//! - [`SymbolSpec`] - Symbol specifications for sizing/costs
//!
//! ## Arrow IPC Integration
//!
//! All batch operations use Arrow IPC for zero-copy data transfer between
//! Python and Rust. See [`arrow`] submodule for serialization utilities.
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_4_EXECUTION_SIMULATOR_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/execution_simulator.md`
//! - Python Source: `src/backtest_engine/core/execution_simulator.py`

pub mod arrow;
pub mod position;
pub mod signal;
pub mod simulator;
pub mod sizing;
pub mod slippage;
pub mod trigger;

// Re-exports for convenience
pub use position::{Direction, OrderType, Position, PositionStatus};
pub use signal::TradeSignal;
pub use simulator::ExecutionSimulatorRust;
pub use sizing::SymbolSpec;
pub use slippage::SlippageCalculator;
