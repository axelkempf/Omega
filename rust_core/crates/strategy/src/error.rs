//! Strategy error types

use thiserror::Error;

/// Strategy-specific errors
#[derive(Debug, Error)]
pub enum StrategyError {
    /// Unknown strategy name in registry
    #[error("Unknown strategy: {0}")]
    UnknownStrategy(String),

    /// Invalid strategy parameters
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// Missing required indicator
    #[error("Missing indicator: {0}")]
    MissingIndicator(String),

    /// Missing HTF data
    #[error("Missing HTF data for timeframe: {0}")]
    MissingHtfData(String),

    /// Invalid scenario ID
    #[error("Invalid scenario ID: {0}")]
    InvalidScenario(u8),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = StrategyError::UnknownStrategy("TestStrategy".to_string());
        assert_eq!(err.to_string(), "Unknown strategy: TestStrategy");
    }

    #[test]
    fn test_error_invalid_params() {
        let err = StrategyError::InvalidParams("z_score_long must be negative".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid parameters: z_score_long must be negative"
        );
    }
}
