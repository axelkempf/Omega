//! Error types for the execution crate.

use thiserror::Error;

/// Errors that can occur during order execution.
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Invalid order parameters
    #[error("invalid order: {0}")]
    InvalidOrder(String),

    /// Order was rejected (margin, risk, session, etc.)
    #[error("order rejected: {0}")]
    OrderRejected(String),

    /// Fill failed
    #[error("fill failed: {0}")]
    FillFailed(String),

    /// Invalid state transition
    #[error("invalid state transition from {from} to {to}: {reason}")]
    InvalidStateTransition {
        from: String,
        to: String,
        reason: String,
    },

    /// Costs configuration error
    #[error("costs config error: {0}")]
    CostsConfig(String),

    /// YAML parsing error
    #[error("yaml parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    /// IO error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Order not found
    #[error("order not found: {0}")]
    OrderNotFound(u64),
}
