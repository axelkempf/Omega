# =============================================================================
# Error Codes für FFI-Grenzen (Julia ↔ Python ↔ Rust)
# =============================================================================
# Task-ID: P2-07 | Phase: 2 – Core Infrastructure
#
# Diese Error-Codes müssen synchron gehalten werden mit:
# - Python: src/shared/error_codes.py (ErrorCode enum)
# - Rust: src/rust_modules/omega_rust/src/error.rs (ErrorCode enum)
#
# Referenz: docs/adr/ADR-0003-error-handling.md
# =============================================================================

"""
FFI Error Codes für Cross-Language Fehlerbehandlung.

Code-Bereiche:
    0:          Erfolg (kein Fehler)
    1000-1999:  Validation Errors (recoverable)
    2000-2999:  Computation Errors (teilweise recoverable)
    3000-3999:  I/O Errors (recoverable)
    4000-4999:  Internal Errors (nicht recoverable, Bugs)
    5000-5999:  FFI Errors (nicht recoverable)
    6000-6999:  Resource Errors (teilweise recoverable)
"""
module ErrorCodes

export ErrorCode,
    is_recoverable,
    error_category,
    FfiResult,
    ok_result,
    error_result

# =============================================================================
# Error Code Constants (synchron mit Python/Rust)
# =============================================================================

"""Error code constants matching Python's ErrorCode enum."""
module ErrorCode
    # Success
    const OK = 0

    # Validation Errors (1000-1999)
    const VALIDATION_FAILED = 1000
    const INVALID_ARGUMENT = 1001
    const NULL_POINTER = 1002
    const OUT_OF_BOUNDS = 1003
    const TYPE_MISMATCH = 1004
    const SCHEMA_VIOLATION = 1005
    const CONSTRAINT_VIOLATION = 1006
    const INVALID_STATE = 1007
    const MISSING_REQUIRED_FIELD = 1008
    const INVALID_FORMAT = 1009
    const EMPTY_INPUT = 1010
    const SIZE_MISMATCH = 1011

    # Computation Errors (2000-2999)
    const COMPUTATION_FAILED = 2000
    const DIVISION_BY_ZERO = 2001
    const OVERFLOW = 2002
    const UNDERFLOW = 2003
    const NAN_RESULT = 2004
    const INF_RESULT = 2005
    const CONVERGENCE_FAILED = 2006
    const NUMERICAL_INSTABILITY = 2007
    const INSUFFICIENT_DATA = 2008

    # I/O Errors (3000-3999)
    const IO_ERROR = 3000
    const FILE_NOT_FOUND = 3001
    const PERMISSION_DENIED = 3002
    const SERIALIZATION_FAILED = 3003
    const DESERIALIZATION_FAILED = 3004
    const NETWORK_ERROR = 3005
    const TIMEOUT = 3006
    const DISK_FULL = 3007

    # Internal Errors (4000-4999)
    const INTERNAL_ERROR = 4000
    const NOT_IMPLEMENTED = 4001
    const ASSERTION_FAILED = 4002
    const UNREACHABLE = 4003
    const INVARIANT_VIOLATED = 4004

    # FFI Errors (5000-5999)
    const FFI_ERROR = 5000
    const FFI_TYPE_CONVERSION = 5001
    const FFI_BUFFER_OVERFLOW = 5002
    const FFI_MEMORY_ERROR = 5003
    const FFI_SCHEMA_MISMATCH = 5004
    const FFI_PANIC_CAUGHT = 5005

    # Resource Errors (6000-6999)
    const RESOURCE_ERROR = 6000
    const OUT_OF_MEMORY = 6001
    const RESOURCE_EXHAUSTED = 6002
    const RESOURCE_BUSY = 6003
    const RESOURCE_LIMIT_EXCEEDED = 6004
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    is_recoverable(code::Int) -> Bool

Prüft ob ein Fehler recoverable ist (Retry sinnvoll).

# Arguments
- `code::Int`: Error-Code

# Returns
- `true` wenn Fehler recoverable ist
"""
function is_recoverable(code::Int)::Bool
    # OK ist kein Fehler
    code == 0 && return true

    # Validation und I/O sind recoverable
    1000 <= code < 2000 && return true  # Validation
    3000 <= code < 4000 && return true  # I/O

    # Teilweise recoverable: bestimmte Computation-Errors
    if 2000 <= code < 3000
        return code == ErrorCode.INSUFFICIENT_DATA
    end

    # Teilweise recoverable: bestimmte Resource-Errors
    if 6000 <= code < 7000
        return code in (ErrorCode.RESOURCE_BUSY, ErrorCode.RESOURCE_LIMIT_EXCEEDED)
    end

    # Internal, FFI: nicht recoverable
    return false
end

"""
    error_category(code::Int) -> String

Gibt die Kategorie eines Error-Codes zurück.

# Arguments
- `code::Int`: Error-Code

# Returns
- Kategorie-Name als String
"""
function error_category(code::Int)::String
    code == 0 && return "OK"
    1000 <= code < 2000 && return "VALIDATION"
    2000 <= code < 3000 && return "COMPUTATION"
    3000 <= code < 4000 && return "IO"
    4000 <= code < 5000 && return "INTERNAL"
    5000 <= code < 6000 && return "FFI"
    6000 <= code < 7000 && return "RESOURCE"
    return "UNKNOWN"
end

# =============================================================================
# FfiResult Type
# =============================================================================

"""
    FfiResult{T}

FFI-safe Result-Wrapper für Cross-Language Rückgabewerte.

Struktur entspricht dem Rust FfiResult<T> und Python-Dict-Format:
- `ok`: Bool - Erfolg-Flag
- `value`: Union{T, Nothing} - Rückgabewert (nur wenn ok=true)
- `error_code`: Int - Error-Code (nur wenn ok=false)
- `message`: Union{String, Nothing} - Fehlermeldung
- `context`: Dict{String, Any} - Zusätzlicher Kontext

# Example
```julia
result = ok_result(42.0)
result.ok  # true
result.value  # 42.0

result = error_result(ErrorCode.INVALID_ARGUMENT, "Period must be > 0")
result.ok  # false
result.error_code  # 1001
```
"""
struct FfiResult{T}
    ok::Bool
    value::Union{T, Nothing}
    error_code::Int
    message::Union{String, Nothing}
    context::Dict{String, Any}
end

"""
    ok_result(value::T) -> FfiResult{T}

Erstellt erfolgreiches FfiResult.

# Arguments
- `value`: Rückgabewert

# Returns
- FfiResult mit ok=true
"""
function ok_result(value::T)::FfiResult{T} where T
    FfiResult{T}(true, value, ErrorCode.OK, nothing, Dict{String, Any}())
end

"""
    error_result(error_code::Int, message::String; context=Dict{String,Any}()) -> FfiResult{Nothing}

Erstellt Fehler-FfiResult.

# Arguments
- `error_code`: Error-Code aus ErrorCode Modul
- `message`: Menschenlesbare Fehlermeldung
- `context`: Zusätzlicher Kontext für Debugging (optional)

# Returns
- FfiResult mit ok=false
"""
function error_result(
    error_code::Int,
    message::String;
    context::Dict{String, Any} = Dict{String, Any}()
)::FfiResult{Nothing}
    FfiResult{Nothing}(false, nothing, error_code, message, context)
end

"""
    error_result(::Type{T}, error_code::Int, message::String; context=Dict{String,Any}()) -> FfiResult{T}

Erstellt typisiertes Fehler-FfiResult für Funktionen mit explizitem Rückgabetyp.

# Arguments
- `T`: Erwarteter Rückgabetyp
- `error_code`: Error-Code aus ErrorCode Modul
- `message`: Menschenlesbare Fehlermeldung
- `context`: Zusätzlicher Kontext für Debugging (optional)

# Returns
- FfiResult{T} mit ok=false
"""
function error_result(
    ::Type{T},
    error_code::Int,
    message::String;
    context::Dict{String, Any} = Dict{String, Any}()
)::FfiResult{T} where T
    FfiResult{T}(false, nothing, error_code, message, context)
end

# =============================================================================
# Conversion to Dict (for Python interop)
# =============================================================================

"""
    to_dict(result::FfiResult) -> Dict{String, Any}

Konvertiert FfiResult zu Dict für Python-Interoperabilität.
"""
function to_dict(result::FfiResult)::Dict{String, Any}
    Dict{String, Any}(
        "ok" => result.ok,
        "value" => result.value,
        "error_code" => result.error_code,
        "message" => result.message,
        "context" => result.context
    )
end

# =============================================================================
# Safe execution wrapper
# =============================================================================

"""
    ffi_safe(f::Function, ::Type{T}) -> FfiResult{T}

Führt Funktion mit Error-Handling aus und gibt FfiResult zurück.

Fängt Julia-Exceptions ab und konvertiert sie zu FfiResult.
Sollte für alle FFI-exportierten Funktionen verwendet werden.

# Arguments
- `f`: Funktion die ausgeführt werden soll (sollte T zurückgeben)
- `T`: Erwarteter Rückgabetyp

# Returns
- FfiResult{T} mit Erfolg oder Fehler

# Example
```julia
function calculate_ema(values::Vector{Float64}, period::Int)::FfiResult{Vector{Float64}}
    ffi_safe(Vector{Float64}) do
        period <= 0 && throw(ArgumentError("Period must be > 0"))
        # ... Berechnung ...
        return result
    end
end
```
"""
function ffi_safe(f::Function, ::Type{T})::FfiResult{T} where T
    try
        result = f()
        return ok_result(result)
    catch e
        # Map Julia exceptions to error codes
        error_code, message = _exception_to_error(e)
        return error_result(T, error_code, message; context=Dict{String, Any}(
            "exception_type" => string(typeof(e)),
            "stacktrace" => sprint(showerror, e, catch_backtrace())
        ))
    end
end

"""
    _exception_to_error(e::Exception) -> Tuple{Int, String}

Mappt Julia-Exception zu Error-Code und Message.
"""
function _exception_to_error(e::Exception)::Tuple{Int, String}
    if e isa ArgumentError
        return (ErrorCode.INVALID_ARGUMENT, string(e))
    elseif e isa BoundsError
        return (ErrorCode.OUT_OF_BOUNDS, string(e))
    elseif e isa DivideError
        return (ErrorCode.DIVISION_BY_ZERO, "Division by zero")
    elseif e isa OverflowError
        return (ErrorCode.OVERFLOW, string(e))
    elseif e isa InexactError
        return (ErrorCode.TYPE_MISMATCH, string(e))
    elseif e isa DomainError
        return (ErrorCode.COMPUTATION_FAILED, string(e))
    elseif e isa SystemError
        return (ErrorCode.IO_ERROR, string(e))
    elseif e isa EOFError
        return (ErrorCode.IO_ERROR, string(e))
    elseif e isa OutOfMemoryError
        return (ErrorCode.OUT_OF_MEMORY, "Out of memory")
    else
        return (ErrorCode.INTERNAL_ERROR, string(e))
    end
end

end # module ErrorCodes
