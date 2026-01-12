//! Technical indicators for financial time series.
//!
//! This module provides high-performance implementations of common
//! technical analysis indicators used in trading systems.
//!
//! ## Available Indicators
//!
//! ### Moving Averages
//! - [`ema`] - Exponential Moving Average
//! - [`sma`] - Simple Moving Average
//! - [`dema`] - Double Exponential Moving Average
//! - [`tema`] - Triple Exponential Moving Average
//!
//! ### Volatility
//! - [`atr`] - Average True Range (Wilder)
//! - [`bollinger`] - Bollinger Bands
//! - [`rolling_std`] - Rolling Standard Deviation
//!
//! ### Momentum
//! - [`rsi`] - Relative Strength Index
//! - [`macd`] - Moving Average Convergence Divergence
//! - [`roc`] - Rate of Change
//! - [`momentum`] - Price Momentum
//!
//! ### Trend
//! - [`dmi`] - Directional Movement Index (+DI, -DI, ADX)
//! - [`choppiness`] - Choppiness Index
//!
//! ### Statistical
//! - [`zscore`] - Rolling Z-Score
//! - [`kalman`] - Kalman Filter
//!
//! ## Performance Targets (Wave 1)
//!
//! | Indicator | Python Baseline | Rust Target | Speedup |
//! |-----------|-----------------|-------------|---------|
//! | ATR       | 954ms           | ≤19ms       | 50x     |
//! | EMA_stepwise | 45ms         | ≤2.3ms      | 20x     |
//! | Bollinger | 89ms            | ≤4.5ms      | 20x     |
//! | DMI       | 65ms            | ≤3.3ms      | 20x     |
//! | SMA       | 23ms            | ≤2.3ms      | 10x     |
//!
//! ## IndicatorCache
//!
//! The [`cache::IndicatorCache`] struct provides caching and OHLCV storage
//! for efficient repeated indicator calculations.

// =============================================================================
// Core Types
// =============================================================================
pub mod types;
pub use types::{CacheKey, IndicatorResult, OhlcvData, OhlcvKey};

// =============================================================================
// Indicator Cache
// =============================================================================
pub mod cache;
pub use cache::IndicatorCache;

// =============================================================================
// Individual Indicators
// =============================================================================

// ATR (Priority 1 - 50x target)
pub mod atr;
pub use atr::{atr_from_tr, atr_impl};

// EMA (existing + extended with stepwise)
pub mod ema_extended;
mod ema_impl;
pub use ema_extended::{dema_impl, ema_impl as ema_generic, ema_stepwise_impl, tema_impl};
pub use ema_impl::{ema, exponential_moving_average};

// SMA + Rolling Std
pub mod sma;
pub use sma::{rolling_std_impl, sma_impl};

// RSI (existing)
mod rsi_impl;
pub use rsi_impl::rsi;

// Bollinger Bands (Priority 2 - 20x target)
pub mod bollinger;
pub use bollinger::{bollinger_impl, bollinger_stepwise_impl, BollingerResult};

// DMI (Priority 2 - 20x target)
pub mod dmi;
pub use dmi::{dmi_impl, DmiResult};

// MACD
pub mod macd;
pub use macd::{macd_impl, MacdResult};

// ROC / Momentum
pub mod roc;
pub use roc::{momentum_impl, roc_impl};

// Z-Score
pub mod zscore;
pub use zscore::{zscore_impl, zscore_normalized_impl};

// Choppiness Index
pub mod choppiness;
pub use choppiness::choppiness_impl;

// Kalman Filter
pub mod kalman;
pub use kalman::{kalman_impl, kalman_zscore_impl, kalman_zscore_stepwise_impl, KalmanResult};

// GARCH Volatility (Wave 1 - Python-only indicators)
pub mod garch;
pub use garch::{
    garch_volatility_impl, garch_volatility_local_impl, garch_volatility_local_last, GarchParams,
    GarchResult,
};

// Legacy statistics (for backwards compatibility)
mod statistics;
pub use statistics::rolling_std;

// =============================================================================
// PyO3 Bindings
// =============================================================================
pub mod py_bindings;
pub use py_bindings::{register_indicator_cache, PyIndicatorCache};
