//! Omega Data
//!
//! Parquet loading, bid/ask alignment, validation, and session/news helpers.

#![deny(clippy::all)]

pub mod alignment;
pub mod error;
pub mod loader;
pub mod market_hours;
pub mod news;
pub mod store;
pub mod validation;

pub use alignment::{AlignedData, AlignmentStats, align_bid_ask};
pub use error::DataError;
pub use loader::{load_and_validate, load_and_validate_bid_ask, load_candles, resolve_data_path};
pub use market_hours::filter_by_sessions;
pub use news::{NewsEvent, load_news_calendar};
pub use store::{CandleStore, MultiTfStore};
pub use validation::{validate_candles, validate_spread};
