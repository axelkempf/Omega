"""FFI Error Codes für Cross-Language Fehlerbehandlung.

Phase 2 Task: P2-07
Status: Implementiert (2026-01-05)

Diese Error-Codes werden über FFI-Grenzen hinweg verwendet (Python ↔ Rust ↔ Julia).
Sie müssen synchron gehalten werden mit:
- Rust: src/rust_modules/src/error.rs (ErrorCode enum)
- Julia: src/julia_modules/src/error.jl (ErrorCode enum)

Referenz: docs/adr/ADR-0003-error-handling.md
"""

from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    """FFI Error Codes für Cross-Language Fehlerbehandlung.

    Code-Bereiche:
        0:          Erfolg (kein Fehler)
        1000-1999:  Validation Errors (recoverable)
        2000-2999:  Computation Errors (teilweise recoverable)
        3000-3999:  I/O Errors (recoverable)
        4000-4999:  Internal Errors (nicht recoverable, Bugs)
        5000-5999:  FFI Errors (nicht recoverable)
        6000-6999:  Resource Errors (teilweise recoverable)
    """

    # =========================================================================
    # Success (0)
    # =========================================================================
    OK = 0

    # =========================================================================
    # Validation Errors (1000-1999) - Input-Validierungsfehler
    # Recoverable: Ja (Caller kann Input korrigieren)
    # =========================================================================
    VALIDATION_FAILED = 1000
    """Allgemeiner Validierungsfehler."""

    INVALID_ARGUMENT = 1001
    """Ungültiges Argument an Funktion übergeben."""

    NULL_POINTER = 1002
    """Unerwarteter Null-Pointer/None-Wert."""

    OUT_OF_BOUNDS = 1003
    """Index oder Wert außerhalb gültiger Grenzen."""

    TYPE_MISMATCH = 1004
    """Typ-Inkompatibilität (z.B. float erwartet, str erhalten)."""

    SCHEMA_VIOLATION = 1005
    """Arrow/Daten-Schema Verletzung."""

    CONSTRAINT_VIOLATION = 1006
    """Business-Constraint verletzt (z.B. SL > Entry bei Long)."""

    INVALID_STATE = 1007
    """Objekt/System in ungültigem Zustand für Operation."""

    MISSING_REQUIRED_FIELD = 1008
    """Pflichtfeld fehlt in Input-Daten."""

    INVALID_FORMAT = 1009
    """Format-Fehler (z.B. ungültiges Datumsformat)."""

    EMPTY_INPUT = 1010
    """Leere Eingabe wo Daten erwartet wurden."""

    SIZE_MISMATCH = 1011
    """Arrays/Listen haben inkompatible Größen."""

    # =========================================================================
    # Computation Errors (2000-2999) - Berechnungsfehler
    # Recoverable: Teilweise (abhängig vom Kontext)
    # =========================================================================
    COMPUTATION_FAILED = 2000
    """Allgemeiner Berechnungsfehler."""

    DIVISION_BY_ZERO = 2001
    """Division durch Null."""

    OVERFLOW = 2002
    """Numerischer Überlauf."""

    UNDERFLOW = 2003
    """Numerischer Unterlauf."""

    NAN_RESULT = 2004
    """Berechnung ergab NaN (Not a Number)."""

    INF_RESULT = 2005
    """Berechnung ergab Infinity."""

    CONVERGENCE_FAILED = 2006
    """Iterativer Algorithmus konvergierte nicht."""

    NUMERICAL_INSTABILITY = 2007
    """Numerische Instabilität erkannt."""

    INSUFFICIENT_DATA = 2008
    """Zu wenig Datenpunkte für Berechnung (z.B. EMA mit period > len)."""

    # =========================================================================
    # I/O Errors (3000-3999) - Ein/Ausgabe-Fehler
    # Recoverable: Ja (Retry möglich)
    # =========================================================================
    IO_ERROR = 3000
    """Allgemeiner I/O-Fehler."""

    FILE_NOT_FOUND = 3001
    """Datei nicht gefunden."""

    PERMISSION_DENIED = 3002
    """Zugriff verweigert."""

    SERIALIZATION_FAILED = 3003
    """Serialisierung fehlgeschlagen."""

    DESERIALIZATION_FAILED = 3004
    """Deserialisierung fehlgeschlagen."""

    NETWORK_ERROR = 3005
    """Netzwerk-Fehler."""

    TIMEOUT = 3006
    """Operation Timeout."""

    DISK_FULL = 3007
    """Festplatte voll."""

    # =========================================================================
    # Internal Errors (4000-4999) - Interne Fehler (Bugs)
    # Recoverable: Nein (Bug im Code)
    # =========================================================================
    INTERNAL_ERROR = 4000
    """Allgemeiner interner Fehler (Bug)."""

    NOT_IMPLEMENTED = 4001
    """Feature/Funktion nicht implementiert."""

    ASSERTION_FAILED = 4002
    """Interne Assertion fehlgeschlagen."""

    UNREACHABLE = 4003
    """Code erreicht der nicht erreichbar sein sollte."""

    INVARIANT_VIOLATED = 4004
    """Interne Invariante verletzt."""

    # =========================================================================
    # FFI Errors (5000-5999) - FFI-spezifische Fehler
    # Recoverable: Nein (Strukturelles Problem)
    # =========================================================================
    FFI_ERROR = 5000
    """Allgemeiner FFI-Fehler."""

    FFI_TYPE_CONVERSION = 5001
    """Typ-Konvertierung über FFI-Grenze fehlgeschlagen."""

    FFI_BUFFER_OVERFLOW = 5002
    """Buffer-Überlauf an FFI-Grenze."""

    FFI_MEMORY_ERROR = 5003
    """Memory-Fehler an FFI-Grenze."""

    FFI_SCHEMA_MISMATCH = 5004
    """Arrow-Schema Mismatch zwischen Python und Rust/Julia."""

    FFI_PANIC_CAUGHT = 5005
    """Rust-Panic an FFI-Grenze abgefangen."""

    # =========================================================================
    # Resource Errors (6000-6999) - Ressourcen-Fehler
    # Recoverable: Teilweise (Warten/Retry möglich)
    # =========================================================================
    RESOURCE_ERROR = 6000
    """Allgemeiner Ressourcen-Fehler."""

    OUT_OF_MEMORY = 6001
    """Speicher erschöpft."""

    RESOURCE_EXHAUSTED = 6002
    """Ressource erschöpft (z.B. File-Handles)."""

    RESOURCE_BUSY = 6003
    """Ressource belegt/gesperrt."""

    RESOURCE_LIMIT_EXCEEDED = 6004
    """Ressourcen-Limit überschritten."""


def is_recoverable(code: ErrorCode | int) -> bool:
    """Prüft ob ein Fehler recoverable ist.

    Args:
        code: Error-Code

    Returns:
        True wenn Fehler recoverable (Retry sinnvoll)
    """
    code_int = int(code)

    # OK ist kein Fehler
    if code_int == 0:
        return True

    # Validation und I/O sind recoverable
    if 1000 <= code_int < 2000:  # Validation
        return True
    if 3000 <= code_int < 4000:  # I/O
        return True

    # Teilweise recoverable
    if 2000 <= code_int < 3000:  # Computation
        # Nur bestimmte Computation-Errors sind recoverable
        return code_int in (
            ErrorCode.INSUFFICIENT_DATA,
        )
    if 6000 <= code_int < 7000:  # Resource
        return code_int in (
            ErrorCode.RESOURCE_BUSY,
            ErrorCode.RESOURCE_LIMIT_EXCEEDED,
        )

    # Internal, FFI: nicht recoverable
    return False


def error_category(code: ErrorCode | int) -> str:
    """Gibt die Kategorie eines Error-Codes zurück.

    Args:
        code: Error-Code

    Returns:
        Kategorie-Name als String
    """
    code_int = int(code)

    if code_int == 0:
        return "OK"
    elif 1000 <= code_int < 2000:
        return "VALIDATION"
    elif 2000 <= code_int < 3000:
        return "COMPUTATION"
    elif 3000 <= code_int < 4000:
        return "IO"
    elif 4000 <= code_int < 5000:
        return "INTERNAL"
    elif 5000 <= code_int < 6000:
        return "FFI"
    elif 6000 <= code_int < 7000:
        return "RESOURCE"
    else:
        return "UNKNOWN"


__all__ = [
    "ErrorCode",
    "is_recoverable",
    "error_category",
]
