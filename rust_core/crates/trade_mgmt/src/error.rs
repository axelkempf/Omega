//! Trade management error types

use thiserror::Error;

/// Trade management errors
#[derive(Debug, Error)]
pub enum TradeManagementError {
    /// Invalid rule configuration
    #[error("Invalid rule configuration: {0}")]
    InvalidConfig(String),

    /// Unknown rule type
    #[error("Unknown rule type: {0}")]
    UnknownRule(String),

    /// Position not found
    #[error("Position not found: {0}")]
    PositionNotFound(u64),

    /// Invalid action
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TradeManagementError::InvalidConfig("max_bars must be > 0".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid rule configuration: max_bars must be > 0"
        );
    }

    #[test]
    fn test_error_position_not_found() {
        let err = TradeManagementError::PositionNotFound(42);
        assert_eq!(err.to_string(), "Position not found: 42");
    }
}
