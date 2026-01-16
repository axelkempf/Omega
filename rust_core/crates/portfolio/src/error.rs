//! Error types for the portfolio crate.

use thiserror::Error;

/// Errors that can occur during portfolio operations.
#[derive(Debug, Error)]
pub enum PortfolioError {
    /// Maximum number of positions reached
    #[error("maximum positions reached: {max}")]
    MaxPositionsReached {
        /// Maximum allowed positions
        max: usize,
    },

    /// Insufficient funds for operation
    #[error("insufficient funds: required {required}, available {available}")]
    InsufficientFunds {
        /// Required amount
        required: f64,
        /// Available amount
        available: f64,
    },

    /// Position not found
    #[error("position not found: {0}")]
    PositionNotFound(u64),

    /// Invalid position modification
    #[error("invalid position modification: {0}")]
    InvalidModification(String),

    /// Invalid stop-loss value
    #[error("invalid stop-loss: {0}")]
    InvalidStopLoss(String),

    /// Invalid take-profit value
    #[error("invalid take-profit: {0}")]
    InvalidTakeProfit(String),

    /// Invalid position size
    #[error("invalid position size: {0}")]
    InvalidSize(String),

    /// Non-finite portfolio value
    #[error("portfolio value is not finite: {field}={value}")]
    NonFiniteValue {
        /// Name of the field
        field: String,
        /// Non-finite value encountered
        value: f64,
    },

    /// Portfolio consistency check failed
    #[error(
        "portfolio consistency violation: equity {equity}, expected {expected} (diff {diff}, tolerance {tolerance})"
    )]
    ConsistencyViolation {
        /// Recorded equity value
        equity: f64,
        /// Expected equity from cash + unrealized PnL
        expected: f64,
        /// Absolute difference between equity and expected value
        diff: f64,
        /// Allowed tolerance
        tolerance: f64,
    },

    /// General portfolio error
    #[error("portfolio error: {0}")]
    General(String),
}
