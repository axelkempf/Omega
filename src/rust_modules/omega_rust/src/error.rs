//! Error types for Omega Rust extensions.
//!
//! Provides structured error handling with automatic conversion
//! to Python exceptions via `PyO3`.
//!
//! ## FFI Error Code Synchronization
//!
//! This module defines ErrorCode constants that are synchronized with:
//! - Python: `src/shared/error_codes.py`
//! - Julia: `src/julia_modules/omega_julia/src/error.jl` (planned)
//!
//! The primary error handling mechanism is PyO3 exceptions (see ADR-0003),
//! but ErrorCodes are included in error messages for cross-language
//! auditability and debugging.
//!
//! Reference: docs/adr/ADR-0003-error-handling.md

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;
use thiserror::Error;

// =============================================================================
// ErrorCode Constants (synchronized with src/shared/error_codes.py)
// =============================================================================
//
// Code Ranges:
//   0:          Success (no error)
//   1000-1999:  Validation Errors (recoverable)
//   2000-2999:  Computation Errors (partially recoverable)
//   3000-3999:  I/O Errors (recoverable)
//   4000-4999:  Internal Errors (not recoverable, bugs)
//   5000-5999:  FFI Errors (not recoverable)
//   6000-6999:  Resource Errors (partially recoverable)
// =============================================================================

/// FFI Error Codes for cross-language error handling.
///
/// These codes must be kept in sync with:
/// - Python: `src/shared/error_codes.py::ErrorCode`
/// - Julia: `src/julia_modules/omega_julia/src/error.jl` (planned)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    // Success
    Ok = 0,

    // Validation Errors (1000-1999)
    ValidationFailed = 1000,
    InvalidArgument = 1001,
    NullPointer = 1002,
    OutOfBounds = 1003,
    TypeMismatch = 1004,
    SchemaViolation = 1005,
    ConstraintViolation = 1006,
    InvalidState = 1007,
    MissingRequiredField = 1008,
    InvalidFormat = 1009,
    EmptyInput = 1010,
    SizeMismatch = 1011,

    // Computation Errors (2000-2999)
    ComputationFailed = 2000,
    DivisionByZero = 2001,
    Overflow = 2002,
    Underflow = 2003,
    NanResult = 2004,
    InfResult = 2005,
    ConvergenceFailed = 2006,
    NumericalInstability = 2007,
    InsufficientData = 2008,

    // I/O Errors (3000-3999)
    IoError = 3000,
    FileNotFound = 3001,
    PermissionDenied = 3002,
    SerializationFailed = 3003,
    DeserializationFailed = 3004,
    NetworkError = 3005,
    Timeout = 3006,
    DiskFull = 3007,

    // Internal Errors (4000-4999)
    InternalError = 4000,
    NotImplemented = 4001,
    AssertionFailed = 4002,
    Unreachable = 4003,
    InvariantViolated = 4004,

    // FFI Errors (5000-5999)
    FfiError = 5000,
    FfiTypeConversion = 5001,
    FfiBufferOverflow = 5002,
    FfiMemoryError = 5003,
    FfiSchemaMismatch = 5004,
    FfiPanicCaught = 5005,

    // Resource Errors (6000-6999)
    ResourceError = 6000,
    OutOfMemory = 6001,
    ResourceExhausted = 6002,
    ResourceBusy = 6003,
    ResourceLimitExceeded = 6004,
}

impl ErrorCode {
    /// Returns the numeric value of the error code.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Returns the error category name.
    pub fn category(&self) -> &'static str {
        let code = self.as_i32();
        match code {
            0 => "OK",
            1000..=1999 => "VALIDATION",
            2000..=2999 => "COMPUTATION",
            3000..=3999 => "IO",
            4000..=4999 => "INTERNAL",
            5000..=5999 => "FFI",
            6000..=6999 => "RESOURCE",
            _ => "UNKNOWN",
        }
    }

    /// Returns whether this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        let code = self.as_i32();
        matches!(code, 0 | 1000..=1999 | 3000..=3999)
            || matches!(
                self,
                ErrorCode::InsufficientData
                    | ErrorCode::ResourceBusy
                    | ErrorCode::ResourceLimitExceeded
            )
    }
}

/// Omega-specific errors that can occur during computation.
#[derive(Error, Debug)]
pub enum OmegaError {
    /// Invalid input parameter
    #[error("[{code}] Invalid parameter: {reason}", code = ErrorCode::InvalidArgument.as_i32())]
    InvalidParameter { reason: String },

    /// Insufficient data for computation
    #[error("[{code}] Insufficient data: need at least {required} elements, got {actual}",
            code = ErrorCode::InsufficientData.as_i32())]
    InsufficientData { required: usize, actual: usize },

    /// Numerical computation error
    #[error("[{code}] Numerical error: {0}", code = ErrorCode::ComputationFailed.as_i32())]
    NumericalError(String),

    /// Internal computation error
    #[error("[{code}] Internal error: {0}", code = ErrorCode::InternalError.as_i32())]
    InternalError(String),
}

impl OmegaError {
    /// Returns the ErrorCode associated with this error.
    pub fn error_code(&self) -> ErrorCode {
        match self {
            OmegaError::InvalidParameter { .. } => ErrorCode::InvalidArgument,
            OmegaError::InsufficientData { .. } => ErrorCode::InsufficientData,
            OmegaError::NumericalError(_) => ErrorCode::ComputationFailed,
            OmegaError::InternalError(_) => ErrorCode::InternalError,
        }
    }
}

/// Result type alias for Omega operations
pub type Result<T> = std::result::Result<T, OmegaError>;

// Automatic conversion from OmegaError to PyErr
// Note: Error messages include [ErrorCode] prefix for cross-language debugging
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

// =============================================================================
// Python Bindings for ErrorCode verification
// =============================================================================

/// Get all error code constants as a Python dict.
/// Used for cross-language synchronization verification.
#[pyfunction]
pub fn get_error_code_constants() -> std::collections::HashMap<String, i32> {
    let mut codes = std::collections::HashMap::new();

    // Success
    codes.insert("OK".to_string(), ErrorCode::Ok as i32);

    // Validation Errors
    codes.insert("VALIDATION_FAILED".to_string(), ErrorCode::ValidationFailed as i32);
    codes.insert("INVALID_ARGUMENT".to_string(), ErrorCode::InvalidArgument as i32);
    codes.insert("NULL_POINTER".to_string(), ErrorCode::NullPointer as i32);
    codes.insert("OUT_OF_BOUNDS".to_string(), ErrorCode::OutOfBounds as i32);
    codes.insert("TYPE_MISMATCH".to_string(), ErrorCode::TypeMismatch as i32);
    codes.insert("SCHEMA_VIOLATION".to_string(), ErrorCode::SchemaViolation as i32);
    codes.insert("CONSTRAINT_VIOLATION".to_string(), ErrorCode::ConstraintViolation as i32);
    codes.insert("INVALID_STATE".to_string(), ErrorCode::InvalidState as i32);
    codes.insert("MISSING_REQUIRED_FIELD".to_string(), ErrorCode::MissingRequiredField as i32);
    codes.insert("INVALID_FORMAT".to_string(), ErrorCode::InvalidFormat as i32);
    codes.insert("EMPTY_INPUT".to_string(), ErrorCode::EmptyInput as i32);
    codes.insert("SIZE_MISMATCH".to_string(), ErrorCode::SizeMismatch as i32);

    // Computation Errors
    codes.insert("COMPUTATION_FAILED".to_string(), ErrorCode::ComputationFailed as i32);
    codes.insert("DIVISION_BY_ZERO".to_string(), ErrorCode::DivisionByZero as i32);
    codes.insert("OVERFLOW".to_string(), ErrorCode::Overflow as i32);
    codes.insert("UNDERFLOW".to_string(), ErrorCode::Underflow as i32);
    codes.insert("NAN_RESULT".to_string(), ErrorCode::NanResult as i32);
    codes.insert("INF_RESULT".to_string(), ErrorCode::InfResult as i32);
    codes.insert("CONVERGENCE_FAILED".to_string(), ErrorCode::ConvergenceFailed as i32);
    codes.insert("NUMERICAL_INSTABILITY".to_string(), ErrorCode::NumericalInstability as i32);
    codes.insert("INSUFFICIENT_DATA".to_string(), ErrorCode::InsufficientData as i32);

    // I/O Errors
    codes.insert("IO_ERROR".to_string(), ErrorCode::IoError as i32);
    codes.insert("FILE_NOT_FOUND".to_string(), ErrorCode::FileNotFound as i32);
    codes.insert("PERMISSION_DENIED".to_string(), ErrorCode::PermissionDenied as i32);
    codes.insert("SERIALIZATION_FAILED".to_string(), ErrorCode::SerializationFailed as i32);
    codes.insert("DESERIALIZATION_FAILED".to_string(), ErrorCode::DeserializationFailed as i32);
    codes.insert("NETWORK_ERROR".to_string(), ErrorCode::NetworkError as i32);
    codes.insert("TIMEOUT".to_string(), ErrorCode::Timeout as i32);
    codes.insert("DISK_FULL".to_string(), ErrorCode::DiskFull as i32);

    // Internal Errors
    codes.insert("INTERNAL_ERROR".to_string(), ErrorCode::InternalError as i32);
    codes.insert("NOT_IMPLEMENTED".to_string(), ErrorCode::NotImplemented as i32);
    codes.insert("ASSERTION_FAILED".to_string(), ErrorCode::AssertionFailed as i32);
    codes.insert("UNREACHABLE".to_string(), ErrorCode::Unreachable as i32);
    codes.insert("INVARIANT_VIOLATED".to_string(), ErrorCode::InvariantViolated as i32);

    // FFI Errors
    codes.insert("FFI_ERROR".to_string(), ErrorCode::FfiError as i32);
    codes.insert("FFI_TYPE_CONVERSION".to_string(), ErrorCode::FfiTypeConversion as i32);
    codes.insert("FFI_BUFFER_OVERFLOW".to_string(), ErrorCode::FfiBufferOverflow as i32);
    codes.insert("FFI_MEMORY_ERROR".to_string(), ErrorCode::FfiMemoryError as i32);
    codes.insert("FFI_SCHEMA_MISMATCH".to_string(), ErrorCode::FfiSchemaMismatch as i32);
    codes.insert("FFI_PANIC_CAUGHT".to_string(), ErrorCode::FfiPanicCaught as i32);

    // Resource Errors
    codes.insert("RESOURCE_ERROR".to_string(), ErrorCode::ResourceError as i32);
    codes.insert("OUT_OF_MEMORY".to_string(), ErrorCode::OutOfMemory as i32);
    codes.insert("RESOURCE_EXHAUSTED".to_string(), ErrorCode::ResourceExhausted as i32);
    codes.insert("RESOURCE_BUSY".to_string(), ErrorCode::ResourceBusy as i32);
    codes.insert("RESOURCE_LIMIT_EXCEEDED".to_string(), ErrorCode::ResourceLimitExceeded as i32);

    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes_match_python() {
        // These values must match src/shared/error_codes.py
        assert_eq!(ErrorCode::Ok as i32, 0);
        assert_eq!(ErrorCode::InvalidArgument as i32, 1001);
        assert_eq!(ErrorCode::InsufficientData as i32, 2008);
        assert_eq!(ErrorCode::ComputationFailed as i32, 2000);
        assert_eq!(ErrorCode::InternalError as i32, 4000);
        assert_eq!(ErrorCode::FfiError as i32, 5000);
    }

    #[test]
    fn test_error_messages_contain_code() {
        let err = OmegaError::InvalidParameter {
            reason: "period must be positive".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("[1001]"), "Error message should contain error code: {}", msg);
        assert!(msg.contains("period must be positive"));

        let err = OmegaError::InsufficientData {
            required: 10,
            actual: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("[2008]"), "Error message should contain error code: {}", msg);
        assert!(msg.contains("10"));
        assert!(msg.contains('5'));
    }

    #[test]
    fn test_error_code_mapping() {
        assert_eq!(
            OmegaError::InvalidParameter { reason: "test".to_string() }.error_code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            OmegaError::InsufficientData { required: 10, actual: 5 }.error_code(),
            ErrorCode::InsufficientData
        );
        assert_eq!(
            OmegaError::NumericalError("test".to_string()).error_code(),
            ErrorCode::ComputationFailed
        );
        assert_eq!(
            OmegaError::InternalError("test".to_string()).error_code(),
            ErrorCode::InternalError
        );
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(ErrorCode::Ok.category(), "OK");
        assert_eq!(ErrorCode::InvalidArgument.category(), "VALIDATION");
        assert_eq!(ErrorCode::ComputationFailed.category(), "COMPUTATION");
        assert_eq!(ErrorCode::IoError.category(), "IO");
        assert_eq!(ErrorCode::InternalError.category(), "INTERNAL");
        assert_eq!(ErrorCode::FfiError.category(), "FFI");
        assert_eq!(ErrorCode::ResourceError.category(), "RESOURCE");
    }

    #[test]
    fn test_recoverability() {
        assert!(ErrorCode::Ok.is_recoverable());
        assert!(ErrorCode::InvalidArgument.is_recoverable());
        assert!(ErrorCode::IoError.is_recoverable());
        assert!(ErrorCode::InsufficientData.is_recoverable());
        assert!(!ErrorCode::InternalError.is_recoverable());
        assert!(!ErrorCode::FfiError.is_recoverable());
    }
}
