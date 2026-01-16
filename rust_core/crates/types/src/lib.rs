//! Omega Types
//!
//! Core data structures for the Omega trading system.
//! This crate provides types for candles, signals, positions, trades,
//! configuration, and backtest results.

#![deny(clippy::all)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]

/// Candle data structures.
pub mod candle;
/// Configuration schema types.
pub mod config;
/// Error types shared across the core.
pub mod error;
/// Position state types.
pub mod position;
/// Price type selection helpers.
pub mod price_type;
/// Backtest result types.
pub mod result;
/// Signal and order types.
pub mod signal;
/// Timeframe parsing and conversions.
pub mod timeframe;
/// Trade record types.
pub mod trade;

// Re-export main types for convenience
pub use candle::Candle;
pub use config::{
    AccountConfig, BacktestConfig, CostsConfig, DataMode, ExecutionVariant, LoggingConfig,
    MaxHoldingTimeConfig, NewsFilterConfig, NewsImpact, RunMode, SessionConfig, StopUpdatePolicy,
    TimeframeConfig, TradeManagementConfig, TradeManagementRulesConfig,
};
pub use error::CoreError;
pub use price_type::PriceType;
pub use position::Position;
pub use result::{BacktestResult, EquityPoint, ErrorResult, Metrics, ResultMeta};
pub use signal::{Direction, OrderType, Signal};
pub use timeframe::{ParseTimeframeError, Timeframe};
pub use trade::{ExitReason, Trade};
