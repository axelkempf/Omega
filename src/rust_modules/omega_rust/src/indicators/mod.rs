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

mod ema;
mod rsi;
mod statistics;

pub use ema::{ema, exponential_moving_average};
pub use rsi::rsi;
pub use statistics::rolling_std;
