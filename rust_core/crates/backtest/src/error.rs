//! Backtest error types.

use omega_types::ErrorResult;
use serde_json::json;
use thiserror::Error;

/// Errors that can occur during backtest orchestration.
#[derive(Debug, Error)]
pub enum BacktestError {
    /// JSON config parse error
    #[error("config parse error: {0}")]
    ConfigParse(String),

    /// Config validation error
    #[error("config validation error: {0}")]
    ConfigValidation(String),

    /// Result serialization error
    #[error("result serialization error: {0}")]
    ResultSerialize(String),

    /// Data loading or validation error
    #[error("data error: {0}")]
    Data(#[from] omega_data::DataError),

    /// Indicator computation error
    #[error("indicator error: {0}")]
    Indicator(#[from] omega_indicators::IndicatorError),

    /// Execution engine error
    #[error("execution error: {0}")]
    Execution(#[from] omega_execution::ExecutionError),

    /// Portfolio error
    #[error("portfolio error: {0}")]
    Portfolio(#[from] omega_portfolio::PortfolioError),

    /// Strategy error
    #[error("strategy error: {0}")]
    Strategy(#[from] omega_strategy::StrategyError),

    /// Trade management error
    #[error("trade management error: {0}")]
    TradeManagement(#[from] omega_trade_mgmt::TradeManagementError),

    /// Not enough data for warmup
    #[error("insufficient data: need {required}, have {available}")]
    InsufficientData {
        /// Required number of bars
        required: usize,
        /// Available number of bars
        available: usize,
    },

    /// Not enough HTF data for warmup
    #[error("insufficient HTF data: need {required}, have {available}")]
    InsufficientHtfData {
        /// Required number of bars
        required: usize,
        /// Available number of bars
        available: usize,
    },

    /// Date parsing error
    #[error("date parse error: {0}")]
    DateParse(String),

    /// Invalid timeframe string
    #[error("invalid timeframe: {0}")]
    InvalidTimeframe(String),

    /// Runtime error
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl BacktestError {
    /// Returns true if this is a config parse/validation error.
    #[must_use]
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            BacktestError::ConfigParse(_) | BacktestError::ConfigValidation(_)
        )
    }

    /// Returns the error category for the output contract.
    /// Categories: `config`, `market_data`, `execution`, `strategy`, `runtime`
    #[must_use]
    pub fn error_category(&self) -> &'static str {
        match self {
            // Config errors
            BacktestError::ConfigParse(_) | BacktestError::ConfigValidation(_) => "config",

            // Market data errors
            BacktestError::Data(_)
            | BacktestError::InsufficientData { .. }
            | BacktestError::InsufficientHtfData { .. }
            | BacktestError::DateParse(_)
            | BacktestError::InvalidTimeframe(_) => "market_data",

            // Execution errors (including portfolio and trade management)
            BacktestError::Execution(_)
            | BacktestError::Portfolio(_)
            | BacktestError::TradeManagement(_) => "execution",

            // Strategy errors
            BacktestError::Strategy(_) => "strategy",

            // Runtime errors (indicator, serialization, generic)
            BacktestError::Indicator(_)
            | BacktestError::ResultSerialize(_)
            | BacktestError::Runtime(_) => "runtime",
        }
    }
}

impl From<BacktestError> for ErrorResult {
    fn from(err: BacktestError) -> Self {
        Self {
            category: err.error_category().to_string(),
            message: err.to_string(),
            details: json!({}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_category_config() {
        let err = BacktestError::ConfigParse("invalid json".to_string());
        assert_eq!(err.error_category(), "config");
        assert!(err.is_config_error());

        let err = BacktestError::ConfigValidation("missing field".to_string());
        assert_eq!(err.error_category(), "config");
        assert!(err.is_config_error());
    }

    #[test]
    fn test_error_category_market_data() {
        let err = BacktestError::InsufficientData {
            required: 100,
            available: 50,
        };
        assert_eq!(err.error_category(), "market_data");
        assert!(!err.is_config_error());

        let err = BacktestError::InsufficientHtfData {
            required: 100,
            available: 50,
        };
        assert_eq!(err.error_category(), "market_data");

        let err = BacktestError::DateParse("invalid date".to_string());
        assert_eq!(err.error_category(), "market_data");

        let err = BacktestError::InvalidTimeframe("X1".to_string());
        assert_eq!(err.error_category(), "market_data");
    }

    #[test]
    fn test_error_category_runtime() {
        let err = BacktestError::ResultSerialize("json error".to_string());
        assert_eq!(err.error_category(), "runtime");
        assert!(!err.is_config_error());

        let err = BacktestError::Runtime("unexpected error".to_string());
        assert_eq!(err.error_category(), "runtime");
    }

    #[test]
    fn test_error_result_conversion_config() {
        let err = BacktestError::ConfigParse("test".to_string());
        let result: ErrorResult = err.into();
        assert_eq!(result.category, "config");
    }

    #[test]
    fn test_error_result_conversion_market_data() {
        let err = BacktestError::InsufficientData {
            required: 100,
            available: 50,
        };
        let result: ErrorResult = err.into();
        assert_eq!(result.category, "market_data");
        assert!(result.message.contains("100"));
        assert!(result.message.contains("50"));
    }

    #[test]
    fn test_error_result_conversion_runtime() {
        let err = BacktestError::Runtime("test error".to_string());
        let result: ErrorResult = err.into();
        assert_eq!(result.category, "runtime");
        assert!(result.message.contains("test error"));
    }
}
