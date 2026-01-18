//! Data-layer error types.

use thiserror::Error;

/// Errors that can occur while loading or validating market/news data.
#[derive(Debug, Error)]
pub enum DataError {
    /// A required file was not found on disk.
    #[error("File not found: {0} ({1})")]
    FileNotFound(String, String),

    /// Parquet parsing or decoding failed.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// A required column is missing.
    #[error("Missing column: {0}")]
    MissingColumn(String),

    /// A column has an unexpected data type.
    #[error("Invalid column type: {0}")]
    InvalidColumnType(String),

    /// A timestamp column is missing timezone or not UTC.
    #[error("Invalid timezone for column {column}: expected UTC, got {timezone}")]
    InvalidTimezone {
        /// Name of the offending column.
        column: String,
        /// Observed timezone string ("<none>" if missing).
        timezone: String,
    },

    /// No rows were loaded after reading data.
    #[error("Empty data")]
    EmptyData,

    /// Date-range filtering removed all rows.
    #[error("Date-range filter produced empty result: start_ns={start_ns}, end_ns={end_ns}")]
    DateRangeEmpty {
        /// Inclusive start timestamp (epoch-ns).
        start_ns: i64,
        /// Inclusive end timestamp (epoch-ns).
        end_ns: i64,
    },

    /// Data violated a governance contract rule.
    #[error("Corrupt data: {0}")]
    CorruptData(String),

    /// Bid/ask spread ordering failed validation.
    #[error("Invalid spread: {0}")]
    InvalidSpread(String),

    /// Bid/ask alignment failed or exceeded loss threshold.
    #[error("Alignment failure: {0}")]
    AlignmentFailure(String),

    /// Not enough data rows to proceed.
    #[error("Insufficient data: need {required}, have {available}")]
    InsufficientData {
        /// Required number of rows.
        required: usize,
        /// Available number of rows.
        available: usize,
    },
}
