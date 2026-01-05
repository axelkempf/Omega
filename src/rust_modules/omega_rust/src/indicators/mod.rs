//! Technical indicators for financial time series.
//!
//! This module provides high-performance implementations of common
//! technical analysis indicators used in trading systems.
//!
//! ## Available Indicators
//!
//! - [`ema`] - Exponential Moving Average
//! - [`rsi`] - Relative Strength Index
//! - [`rolling_std`] - Rolling Standard Deviation
//!
//! ## Performance Notes
//!
//! All indicators are implemented with O(n) time complexity and
//! minimal memory allocation. For large datasets, consider using
//! the streaming variants (future feature).

// Internal modules (not re-exported as modules to avoid name collision)
mod ema_impl;
mod rsi_impl;
mod statistics;

// Re-export only the functions
pub use ema_impl::{ema, exponential_moving_average};
pub use rsi_impl::rsi;
pub use statistics::rolling_std;
