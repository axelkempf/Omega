//! Omega Data
//!
//! Parquet loading, bid/ask alignment, validation, and session/news helpers.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

/// Bid/ask alignment utilities.
pub mod alignment;
/// Data-layer error types.
pub mod error;
/// Parquet loading and data filters.
pub mod loader;
/// Session handling helpers.
pub mod market_hours;
/// Gap analysis and statistics.
pub mod gaps;
/// News calendar loading.
pub mod news;
/// Candle stores and multi-timeframe mapping.
pub mod store;
/// Governance validation helpers.
pub mod validation;

/// Re-export: aligned bid/ask data container.
pub use alignment::AlignedData;
/// Re-export: bid/ask alignment stats.
pub use alignment::AlignmentStats;
/// Re-export: bid/ask alignment function.
pub use alignment::align_bid_ask;
/// Re-export: data-layer error type.
pub use error::DataError;
/// Re-export: gap analysis result stats.
pub use gaps::GapStats;
/// Re-export: session-aware gap analysis.
pub use gaps::analyze_gaps;
/// Re-export: date-range filter for candles.
pub use loader::filter_by_date_range;
/// Re-export: load and validate candles.
pub use loader::load_and_validate;
/// Re-export: load, validate, align bid/ask candles.
pub use loader::load_and_validate_bid_ask;
/// Re-export: load candles from Parquet.
pub use loader::load_candles;
/// Re-export: resolve market data path.
pub use loader::resolve_data_path;
/// Re-export: session filter for candles.
pub use market_hours::filter_by_sessions;
/// Re-export: news calendar event model.
pub use news::NewsEvent;
/// Re-export: load news calendar from Parquet.
pub use news::load_news_calendar;
/// Re-export: resolve news calendar path.
pub use news::resolve_news_calendar_path;
/// Re-export: candle store data structure.
pub use store::CandleStore;
/// Re-export: multi-timeframe store data structure.
pub use store::MultiTfStore;
/// Re-export: candle validation.
pub use validation::validate_candles;
/// Re-export: bid/ask spread validation.
pub use validation::validate_spread;
