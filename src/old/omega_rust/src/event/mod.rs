//! Event Engine module for high-performance backtesting.
//!
//! Provides Rust implementations of the event loop for the Omega Trading System.
//! This is **Wave 3** of the Rust migration.
//!
//! ## Available Types
//!
//! - [`EventEngineRust`] - Single-symbol event loop implementation
//! - [`CrossSymbolEventEngineRust`] - Multi-symbol event loop (placeholder)
//! - [`EventEngineStats`] - Performance statistics
//! - [`CandleData`] - Candle representation
//! - [`SignalDirection`] - Signal direction enum
//! - [`TradeSignalRust`] - Complete trade signal
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/event_engine.md`

mod engine;
mod types;

pub use engine::{get_event_engine_backend, CrossSymbolEventEngineRust, EventEngineRust};
pub use types::{CandleData, EventEngineStats, SignalDirection, TradeSignalRust};
