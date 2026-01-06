//! Error types for Omega Rust extensions.
//!
//! Provides structured error handling with automatic conversion
//! to Python exceptions via `PyO3`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

/// Omega-specific errors that can occur during computation.
#[derive(Error, Debug)]
pub enum OmegaError {
    /// Invalid input parameter
    #[error("Invalid parameter: {reason}")]
    InvalidParameter { reason: String },

    /// Insufficient data for computation
    #[error("Insufficient data: need at least {required} elements, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Internal computation error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type alias for Omega operations
pub type Result<T> = std::result::Result<T, OmegaError>;

// Automatic conversion from OmegaError to PyErr
impl From<OmegaError> for PyErr {
    fn from(err: OmegaError) -> Self {
        match err {
            OmegaError::InvalidParameter { .. } | OmegaError::InsufficientData { .. } => {
                PyValueError::new_err(err.to_string())
            }
            OmegaError::NumericalError(_) | OmegaError::InternalError(_) => {
                PyRuntimeError::new_err(err.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = OmegaError::InvalidParameter {
            reason: "period must be positive".to_string(),
        };
        assert!(err.to_string().contains("period must be positive"));

        let err = OmegaError::InsufficientData {
            required: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains('5'));
    }
}
