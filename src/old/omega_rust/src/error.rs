//! Error types for Omega Rust extensions.
//!
//! Provides structured error handling with automatic conversion
//! to Python exceptions via `PyO3`.
//!
//! ## FFI Error Code Synchronization
//!
//! This module defines `ErrorCode` constants that are synchronized with:
//! - Python: `src/shared/error_codes.py`
//! - Julia: `src/julia_modules/omega_julia/src/error.jl` (planned)
//!
//! The primary error handling mechanism is `PyO3` exceptions (see ADR-0003),
//! but `ErrorCode` values are included in error messages for cross-language
//! auditability and debugging.
//!
//! Reference: docs/adr/ADR-0003-error-handling.md

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;
use std::collections::HashMap;
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
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// Returns the error category name.
    pub const fn category(&self) -> &'static str {
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
    pub const fn is_recoverable(&self) -> bool {
        let code = self.as_i32();
        match code {
            0 | 1000..=1999 | 3000..=3999 => true,
            _ => matches!(
                self,
                Self::InsufficientData | Self::ResourceBusy | Self::ResourceLimitExceeded
            ),
        }
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

    /// Item not found
    #[error("[{code}] Not found: {item}", code = ErrorCode::InvalidState.as_i32())]
    NotFound { item: String },
}

impl OmegaError {
    /// Returns the `ErrorCode` associated with this error.
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::InvalidParameter { .. } => ErrorCode::InvalidArgument,
            Self::InsufficientData { .. } => ErrorCode::InsufficientData,
            Self::NumericalError(_) => ErrorCode::ComputationFailed,
            Self::InternalError(_) => ErrorCode::InternalError,
            Self::NotFound { .. } => ErrorCode::InvalidState,
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
            OmegaError::InvalidParameter { .. }
            | OmegaError::InsufficientData { .. }
            | OmegaError::NotFound { .. } => PyValueError::new_err(err.to_string()),
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
pub fn get_error_code_constants() -> HashMap<String, i32> {
    const SUCCESS_CODES: &[(&str, ErrorCode)] = &[("OK", ErrorCode::Ok)];
    const VALIDATION_CODES: &[(&str, ErrorCode)] = &[
        ("VALIDATION_FAILED", ErrorCode::ValidationFailed),
        ("INVALID_ARGUMENT", ErrorCode::InvalidArgument),
        ("NULL_POINTER", ErrorCode::NullPointer),
        ("OUT_OF_BOUNDS", ErrorCode::OutOfBounds),
        ("TYPE_MISMATCH", ErrorCode::TypeMismatch),
        ("SCHEMA_VIOLATION", ErrorCode::SchemaViolation),
        ("CONSTRAINT_VIOLATION", ErrorCode::ConstraintViolation),
        ("INVALID_STATE", ErrorCode::InvalidState),
        ("MISSING_REQUIRED_FIELD", ErrorCode::MissingRequiredField),
        ("INVALID_FORMAT", ErrorCode::InvalidFormat),
        ("EMPTY_INPUT", ErrorCode::EmptyInput),
        ("SIZE_MISMATCH", ErrorCode::SizeMismatch),
    ];
    const COMPUTATION_CODES: &[(&str, ErrorCode)] = &[
        ("COMPUTATION_FAILED", ErrorCode::ComputationFailed),
        ("DIVISION_BY_ZERO", ErrorCode::DivisionByZero),
        ("OVERFLOW", ErrorCode::Overflow),
        ("UNDERFLOW", ErrorCode::Underflow),
        ("NAN_RESULT", ErrorCode::NanResult),
        ("INF_RESULT", ErrorCode::InfResult),
        ("CONVERGENCE_FAILED", ErrorCode::ConvergenceFailed),
        ("NUMERICAL_INSTABILITY", ErrorCode::NumericalInstability),
        ("INSUFFICIENT_DATA", ErrorCode::InsufficientData),
    ];
    const IO_CODES: &[(&str, ErrorCode)] = &[
        ("IO_ERROR", ErrorCode::IoError),
        ("FILE_NOT_FOUND", ErrorCode::FileNotFound),
        ("PERMISSION_DENIED", ErrorCode::PermissionDenied),
        ("SERIALIZATION_FAILED", ErrorCode::SerializationFailed),
        ("DESERIALIZATION_FAILED", ErrorCode::DeserializationFailed),
        ("NETWORK_ERROR", ErrorCode::NetworkError),
        ("TIMEOUT", ErrorCode::Timeout),
        ("DISK_FULL", ErrorCode::DiskFull),
    ];
    const INTERNAL_CODES: &[(&str, ErrorCode)] = &[
        ("INTERNAL_ERROR", ErrorCode::InternalError),
        ("NOT_IMPLEMENTED", ErrorCode::NotImplemented),
        ("ASSERTION_FAILED", ErrorCode::AssertionFailed),
        ("UNREACHABLE", ErrorCode::Unreachable),
        ("INVARIANT_VIOLATED", ErrorCode::InvariantViolated),
    ];
    const FFI_CODES: &[(&str, ErrorCode)] = &[
        ("FFI_ERROR", ErrorCode::FfiError),
        ("FFI_TYPE_CONVERSION", ErrorCode::FfiTypeConversion),
        ("FFI_BUFFER_OVERFLOW", ErrorCode::FfiBufferOverflow),
        ("FFI_MEMORY_ERROR", ErrorCode::FfiMemoryError),
        ("FFI_SCHEMA_MISMATCH", ErrorCode::FfiSchemaMismatch),
        ("FFI_PANIC_CAUGHT", ErrorCode::FfiPanicCaught),
    ];
    const RESOURCE_CODES: &[(&str, ErrorCode)] = &[
        ("RESOURCE_ERROR", ErrorCode::ResourceError),
        ("OUT_OF_MEMORY", ErrorCode::OutOfMemory),
        ("RESOURCE_EXHAUSTED", ErrorCode::ResourceExhausted),
        ("RESOURCE_BUSY", ErrorCode::ResourceBusy),
        ("RESOURCE_LIMIT_EXCEEDED", ErrorCode::ResourceLimitExceeded),
    ];

    fn insert_codes(codes: &mut HashMap<String, i32>, items: &[(&str, ErrorCode)]) {
        for (name, code) in items {
            codes.insert(String::from(*name), code.as_i32());
        }
    }

    let mut codes = HashMap::new();
    insert_codes(&mut codes, SUCCESS_CODES);
    insert_codes(&mut codes, VALIDATION_CODES);
    insert_codes(&mut codes, COMPUTATION_CODES);
    insert_codes(&mut codes, IO_CODES);
    insert_codes(&mut codes, INTERNAL_CODES);
    insert_codes(&mut codes, FFI_CODES);
    insert_codes(&mut codes, RESOURCE_CODES);
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
        assert!(
            msg.contains("[1001]"),
            "Error message should contain error code: {msg}"
        );
        assert!(msg.contains("period must be positive"));

        let err = OmegaError::InsufficientData {
            required: 10,
            actual: 5,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("[2008]"),
            "Error message should contain error code: {msg}"
        );
        assert!(msg.contains("10"));
        assert!(msg.contains('5'));
    }

    #[test]
    fn test_error_code_mapping() {
        assert_eq!(
            OmegaError::InvalidParameter {
                reason: "test".to_string()
            }
            .error_code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            OmegaError::InsufficientData {
                required: 10,
                actual: 5
            }
            .error_code(),
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
