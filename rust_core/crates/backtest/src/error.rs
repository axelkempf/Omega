//! Backtest error types.

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
        matches!(self, BacktestError::ConfigParse(_) | BacktestError::ConfigValidation(_))
    }
}
