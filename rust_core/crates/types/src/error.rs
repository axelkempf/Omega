use thiserror::Error;

/// Core error types for Omega
#[derive(Debug, Error)]
pub enum CoreError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Data error
    #[error("Data error: {0}")]
    Data(String),

    /// Indicator error
    #[error("Indicator error: {0}")]
    Indicator(String),

    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Portfolio error
    #[error("Portfolio error: {0}")]
    Portfolio(String),

    /// Strategy error
    #[error("Strategy error: {0}")]
    Strategy(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::Config("Invalid parameter".to_string());
        assert_eq!(err.to_string(), "Configuration error: Invalid parameter");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let err: CoreError = io_err.into();
        assert!(matches!(err, CoreError::Io(_)));
    }
}
