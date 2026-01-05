# ADR-0003: Fehlerbehandlungs-Konvention für FFI-Grenzen

## Status

Accepted

## Kontext

Die Migration ausgewählter Module zu Rust und Julia erfordert eine konsistente Strategie für die Fehlerbehandlung über FFI-Grenzen (Foreign Function Interface). Fehler können in verschiedenen Schichten entstehen:

1. **Python-Seite**: Input-Validierung, Config-Fehler, I/O-Fehler
2. **Rust-Seite**: Berechnungsfehler, Bounds-Violations, Panic
3. **Julia-Seite**: Domain-Errors, Type-Mismatches
4. **FFI-Grenze**: Serialisierungsfehler, Type-Mapping-Fehler

### Anforderungen

| Anforderung | Priorität | Beschreibung |
|-------------|-----------|--------------|
| Konsistenz | Kritisch | Einheitliche Fehlerbehandlung über alle Sprachen |
| Debugbarkeit | Hoch | Klare Error Messages mit Kontext |
| Performance | Hoch | Minimaler Overhead für Erfolgs-Pfade |
| Safety | Kritisch | Keine Undefined Behavior; keine Panic-Propagation über FFI |
| Recoverability | Mittel | Unterscheidung zwischen recoverable und fatal errors |

### Kräfte

- **Idiomatik vs. Konsistenz**: Jede Sprache hat eigene Fehler-Idiome (Python: Exceptions, Rust: Result<T,E>, Julia: Exceptions)
- **Performance vs. Safety**: Exception-Handling hat Overhead; Rust's Result ist zero-cost für Erfolg
- **Einfachheit vs. Vollständigkeit**: Detaillierte Error-Codes vs. einfache Fehlermeldungen

### Constraints

- Python-Code erwartet Exceptions (bestehendes Verhalten)
- Rust-Panic über FFI-Grenzen ist Undefined Behavior
- Live-Trading (`hf_engine/`) muss stabil bleiben

## Entscheidung

### Hybrid-Ansatz für Fehlerbehandlung

Wir implementieren einen **Hybrid-Ansatz**, der idiomatische Fehlerbehandlung in jeder Sprache nutzt, mit klar definierten Transformationen an FFI-Grenzen.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Error Handling Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐          FFI Boundary           ┌────────────┐             │
│  │  Python    │                                 │   Rust     │             │
│  │            │                                 │            │             │
│  │ Exception  │◄──── FFI Error Struct ─────────│ Result<T,E>│             │
│  │            │                                 │            │             │
│  │ try/except │      ┌──────────────┐          │ anyhow/    │             │
│  │            │◄─────│ FfiResult<T> │──────────│ thiserror  │             │
│  └────────────┘      │              │          └────────────┘             │
│                      │ - ok: bool   │                                      │
│                      │ - value: T   │          ┌────────────┐             │
│                      │ - error_code │          │   Julia    │             │
│                      │ - message    │          │            │             │
│                      └──────────────┘──────────│ Exception  │             │
│                                                 │            │             │
│                                                 └────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Error Categories

| Kategorie | Code-Bereich | Recoverable | Beschreibung |
|-----------|--------------|-------------|--------------|
| `OK` | 0 | - | Kein Fehler (Erfolg) |
| `VALIDATION` | 1000-1999 | Ja | Input-Validierungsfehler |
| `COMPUTATION` | 2000-2999 | Teilweise | Berechnungsfehler |
| `IO` | 3000-3999 | Ja | I/O-bezogene Fehler |
| `INTERNAL` | 4000-4999 | Nein | Interne Fehler (Bugs) |
| `FFI` | 5000-5999 | Nein | FFI-spezifische Fehler |
| `RESOURCE` | 6000-6999 | Teilweise | Ressourcen-Fehler |

### Detaillierte Error Codes

```python
# src/shared/error_codes.py

from enum import IntEnum

class ErrorCode(IntEnum):
    """FFI Error Codes für Cross-Language Fehlerbehandlung."""
    
    # Success
    OK = 0
    
    # Validation Errors (1000-1999)
    VALIDATION_FAILED = 1000
    INVALID_ARGUMENT = 1001
    NULL_POINTER = 1002
    OUT_OF_BOUNDS = 1003
    TYPE_MISMATCH = 1004
    SCHEMA_VIOLATION = 1005
    CONSTRAINT_VIOLATION = 1006
    INVALID_STATE = 1007
    MISSING_REQUIRED_FIELD = 1008
    INVALID_FORMAT = 1009
    
    # Computation Errors (2000-2999)
    COMPUTATION_FAILED = 2000
    DIVISION_BY_ZERO = 2001
    OVERFLOW = 2002
    UNDERFLOW = 2003
    NAN_RESULT = 2004
    INF_RESULT = 2005
    CONVERGENCE_FAILED = 2006
    NUMERICAL_INSTABILITY = 2007
    
    # I/O Errors (3000-3999)
    IO_ERROR = 3000
    FILE_NOT_FOUND = 3001
    PERMISSION_DENIED = 3002
    SERIALIZATION_FAILED = 3003
    DESERIALIZATION_FAILED = 3004
    NETWORK_ERROR = 3005
    TIMEOUT = 3006
    
    # Internal Errors (4000-4999)
    INTERNAL_ERROR = 4000
    NOT_IMPLEMENTED = 4001
    ASSERTION_FAILED = 4002
    UNREACHABLE = 4003
    
    # FFI Errors (5000-5999)
    FFI_ERROR = 5000
    FFI_TYPE_CONVERSION = 5001
    FFI_BUFFER_OVERFLOW = 5002
    FFI_MEMORY_ERROR = 5003
    FFI_SCHEMA_MISMATCH = 5004
    
    # Resource Errors (6000-6999)
    RESOURCE_ERROR = 6000
    OUT_OF_MEMORY = 6001
    RESOURCE_EXHAUSTED = 6002
    RESOURCE_BUSY = 6003
```

### Python-Seite: Exception Hierarchy

```python
# src/shared/exceptions.py

from typing import Any

class OmegaError(Exception):
    """Basis-Exception für alle Omega-Fehler."""
    
    def __init__(
        self,
        message: str,
        error_code: int = 4000,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message
    
    def to_ffi_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu FFI-kompatiblem Dict."""
        return {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class ValidationError(OmegaError):
    """Input-Validierungsfehler."""
    
    def __init__(self, message: str, field: str | None = None, **context: Any):
        super().__init__(
            message,
            error_code=1000,
            context={"field": field, **context},
        )


class ComputationError(OmegaError):
    """Berechnungsfehler."""
    
    def __init__(self, message: str, operation: str | None = None, **context: Any):
        super().__init__(
            message,
            error_code=2000,
            context={"operation": operation, **context},
        )


class FfiError(OmegaError):
    """FFI-spezifische Fehler."""
    
    def __init__(self, message: str, rust_error: str | None = None, **context: Any):
        super().__init__(
            message,
            error_code=5000,
            context={"rust_error": rust_error, **context},
        )
```

### Rust-Seite: Error Types

```rust
// src/rust_modules/src/error.rs

use thiserror::Error;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// FFI Error Codes (muss mit Python ErrorCode synchron sein)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Ok = 0,
    
    // Validation
    ValidationFailed = 1000,
    InvalidArgument = 1001,
    NullPointer = 1002,
    OutOfBounds = 1003,
    TypeMismatch = 1004,
    
    // Computation
    ComputationFailed = 2000,
    DivisionByZero = 2001,
    Overflow = 2002,
    NanResult = 2004,
    
    // Internal
    InternalError = 4000,
    
    // FFI
    FfiError = 5000,
    FfiTypeConversion = 5001,
}

/// Hauptfehler-Typ für Omega Rust-Module
#[derive(Error, Debug)]
pub enum OmegaError {
    #[error("Validation error: {message}")]
    Validation {
        message: String,
        code: ErrorCode,
    },
    
    #[error("Computation error: {message}")]
    Computation {
        message: String,
        code: ErrorCode,
    },
    
    #[error("FFI error: {message}")]
    Ffi {
        message: String,
        code: ErrorCode,
    },
    
    #[error("Internal error: {message}")]
    Internal {
        message: String,
    },
}

impl OmegaError {
    pub fn error_code(&self) -> i32 {
        match self {
            OmegaError::Validation { code, .. } => *code as i32,
            OmegaError::Computation { code, .. } => *code as i32,
            OmegaError::Ffi { code, .. } => *code as i32,
            OmegaError::Internal { .. } => ErrorCode::InternalError as i32,
        }
    }
}

/// Konvertierung zu Python Exception
impl From<OmegaError> for PyErr {
    fn from(err: OmegaError) -> PyErr {
        PyRuntimeError::new_err(format!(
            "[{}] {}",
            err.error_code(),
            err
        ))
    }
}

/// Result-Typ für FFI-Funktionen
pub type OmegaResult<T> = Result<T, OmegaError>;

/// FFI-safe Result Wrapper
#[repr(C)]
pub struct FfiResult<T> {
    pub ok: bool,
    pub value: Option<T>,
    pub error_code: i32,
    pub message: Option<String>,
}

impl<T> From<OmegaResult<T>> for FfiResult<T> {
    fn from(result: OmegaResult<T>) -> Self {
        match result {
            Ok(value) => FfiResult {
                ok: true,
                value: Some(value),
                error_code: 0,
                message: None,
            },
            Err(err) => FfiResult {
                ok: false,
                value: None,
                error_code: err.error_code(),
                message: Some(err.to_string()),
            },
        }
    }
}
```

### FFI Boundary: Panic Catching

```rust
// Alle FFI-exportierten Funktionen müssen Panics abfangen

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Wrapper für FFI-Funktionen die Panics abfängt
pub fn ffi_safe<F, T>(f: F) -> FfiResult<T>
where
    F: FnOnce() -> OmegaResult<T>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result.into(),
        Err(panic) => {
            let message = if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            
            FfiResult {
                ok: false,
                value: None,
                error_code: ErrorCode::InternalError as i32,
                message: Some(format!("Panic caught at FFI boundary: {}", message)),
            }
        }
    }
}

/// Beispiel FFI-Funktion
#[pyfunction]
fn calculate_ema(values: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    ffi_safe(|| {
        if period == 0 {
            return Err(OmegaError::Validation {
                message: "Period must be > 0".to_string(),
                code: ErrorCode::InvalidArgument,
            });
        }
        
        // Berechnung...
        Ok(vec![])
    }).into_py_result()
}
```

### Python FFI Wrapper Pattern

```python
# src/shared/ffi_wrapper.py

from typing import TypeVar, Callable, Any
from functools import wraps
from .exceptions import OmegaError, FfiError, ValidationError, ComputationError
from .error_codes import ErrorCode

T = TypeVar("T")


def handle_ffi_result(result: dict[str, Any]) -> Any:
    """Konvertiert FFI-Result zu Python-Value oder Exception.
    
    Args:
        result: Dict mit keys: ok, value, error_code, message
        
    Returns:
        value wenn ok=True
        
    Raises:
        OmegaError subclass basierend auf error_code
    """
    if result.get("ok", False):
        return result.get("value")
    
    error_code = result.get("error_code", 4000)
    message = result.get("message", "Unknown error")
    context = result.get("context", {})
    
    # Map error code to exception type
    if 1000 <= error_code < 2000:
        raise ValidationError(message, **context)
    elif 2000 <= error_code < 3000:
        raise ComputationError(message, **context)
    elif 5000 <= error_code < 6000:
        raise FfiError(message, **context)
    else:
        raise OmegaError(message, error_code=error_code, context=context)


def ffi_call(func: Callable[..., dict[str, Any]]) -> Callable[..., Any]:
    """Decorator für FFI-Funktionen.
    
    Konvertiert FfiResult-Dicts automatisch zu Values/Exceptions.
    
    Usage:
        @ffi_call
        def my_rust_function(x: float) -> float:
            return rust_module.my_function(x)
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return handle_ffi_result(result)
    return wrapper
```

## Konsequenzen

### Positive Konsequenzen

- **Safety**: Rust-Panics werden an FFI-Grenzen abgefangen → kein UB
- **Konsistenz**: Einheitliche Error-Codes über alle Sprachen
- **Debugbarkeit**: Strukturierte Fehler mit Kontext und Error-Codes
- **Python-Kompatibilität**: Python-Code kann weiterhin `try/except` verwenden
- **Zero-Cost Success Path**: Kein Overhead für erfolgreiche Aufrufe (Rust's Result)

### Negative Konsequenzen

- **Komplexität**: Zusätzliche Transformation an FFI-Grenzen
- **Maintenance**: Error-Codes müssen synchron gehalten werden
- **Boilerplate**: FFI-Wrapper für jede exportierte Funktion

### Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Error-Code Drift | Mittel | Mittel | Zentrale Definition; CI-Test für Synchronität |
| Stack-Trace-Verlust | Mittel | Niedrig | Rust Backtrace in Debug-Builds; context Dict |
| Performance-Overhead | Niedrig | Niedrig | catch_unwind nur an FFI-Boundary |

## Alternativen

### Alternative 1: Pure Exception Propagation (via PyO3)

- **Beschreibung**: PyO3 kann Rust-Errors direkt in Python-Exceptions konvertieren
- **Vorteile**: Weniger Boilerplate; automatische Konvertierung
- **Nachteile**: Kein einheitliches Error-Code-System; weniger Kontrolle
- **Warum nicht gewählt**: Brauchen strukturierte Fehler für Logging/Monitoring

### Alternative 2: C-Style Error Handling (Error-Codes nur)

- **Beschreibung**: Alle Funktionen geben Error-Code zurück; Out-Parameter für Werte
- **Vorteile**: Maximum Performance; einfaches FFI
- **Nachteile**: Unidiomatisch für Python; schwer zu debuggen
- **Warum nicht gewählt**: Python-Entwickler erwarten Exceptions

### Alternative 3: Result-Monad in Python

- **Beschreibung**: `Result[T, E]` Typ auch in Python verwenden
- **Vorteile**: Konsistent mit Rust; explizite Fehlerbehandlung
- **Nachteile**: Unidiomatisch für Python; Breaking Change
- **Warum nicht gewählt**: Würde bestehenden Python-Code brechen

## Implementierungs-Checkliste

- [x] Error-Code Enumeration definieren (`src/shared/error_codes.py`)
- [x] Python Exception Hierarchy (`src/shared/exceptions.py`)
- [x] Rust Error Types (`src/rust_modules/omega_rust/src/lib.rs`)
- [x] FFI Wrapper Utilities (`src/shared/ffi_wrapper.py`)
- [x] Tests für Error-Propagation
- [x] Dokumentation der Error-Codes

## Referenzen

- [PyO3 Error Handling](https://pyo3.rs/v0.20.0/function/error_handling)
- [Rust Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [thiserror Crate](https://docs.rs/thiserror/)
- [anyhow Crate](https://docs.rs/anyhow/)
- [ADR-0001: Migration Strategy](ADR-0001-migration-strategy.md)
- [ADR-0002: Serialisierungsformat](ADR-0002-serialization-format.md)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-05 | AI Agent | Initiale Version (P2-07) |
| 2026-01-05 | AI Agent | Finalisiert (P5-03); Implementierungs-Checkliste aktualisiert |
