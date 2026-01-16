//! Omega Types
//!
//! Core data structures for the Omega trading system.
//! This crate provides types for candles, signals, positions, trades,
//! configuration, and backtest results.

#![deny(clippy::all)]

pub mod candle;
pub mod config;
pub mod error;
pub mod position;
pub mod price_type;
pub mod result;
pub mod signal;
pub mod timeframe;
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
