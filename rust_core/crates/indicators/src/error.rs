//! Indicator error types.

use thiserror::Error;

/// Errors that can occur during indicator computation or registry operations.
#[derive(Debug, Error)]
pub enum IndicatorError {
    /// Unknown indicator name requested from registry
    #[error("unknown indicator: {0}")]
    UnknownIndicator(String),

    /// Invalid parameters for the indicator
    #[error("invalid parameters: {0}")]
    InvalidParams(String),

    /// Insufficient data for computation
    #[error("insufficient data: need {required} candles, got {actual}")]
    InsufficientData {
        /// Required number of candles.
        required: usize,
        /// Actual number of candles provided.
        actual: usize,
    },

    /// Computation error (e.g., division by zero, invalid state)
    #[error("computation error: {0}")]
    ComputationError(String),

    /// Parameter out of valid range
    #[error("parameter out of range: {param} = {value} (valid: {min}..{max})")]
    ParamOutOfRange {
        /// Parameter name.
        param: String,
        /// Parameter value.
        value: f64,
        /// Minimum allowed value.
        min: f64,
        /// Maximum allowed value.
        max: f64,
    },
}

impl IndicatorError {
    /// Creates an `InvalidParams` error with a message.
    #[must_use]
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        IndicatorError::InvalidParams(msg.into())
    }

    /// Creates a `ComputationError` with a message.
    #[must_use]
    pub fn computation(msg: impl Into<String>) -> Self {
        IndicatorError::ComputationError(msg.into())
    }

    /// Creates a `ParamOutOfRange` error.
    #[must_use]
    pub fn param_out_of_range(param: impl Into<String>, value: f64, min: f64, max: f64) -> Self {
        IndicatorError::ParamOutOfRange {
            param: param.into(),
            value,
            min,
            max,
        }
    }
}
