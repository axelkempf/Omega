"""FFI Wrapper Utilities für Cross-Language Function Calls.

Phase 2 Task: P2-07
Status: Implementiert (2026-01-05)

Dieses Modul stellt Wrapper-Utilities für FFI-Calls zu Rust und Julia bereit.
Es konvertiert FfiResult-Strukturen automatisch zu Python-Values
oder Exceptions.

Referenz: docs/adr/ADR-0003-error-handling.md
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from .error_codes import ErrorCode
from .exceptions import (
    ComputationError,
    FfiError,
    InternalError,
    IoError,
    OmegaError,
    ResourceError,
    ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")
R = TypeVar("R")


def handle_ffi_result(result: Mapping[str, Any]) -> Any:
    """Konvertiert FFI-Result zu Python-Value oder Exception.

    Erwartet ein Dict/Mapping mit der FfiResult-Struktur aus Rust/Julia:
        - ok: bool - Erfolg-Flag
        - value: Any - Rückgabewert (nur wenn ok=True)
        - error_code: int - Error-Code (nur wenn ok=False)
        - message: str - Fehlermeldung (nur wenn ok=False)
        - context: dict - Zusätzlicher Kontext (optional)

    Args:
        result: FfiResult-Dict von Rust/Julia FFI-Call

    Returns:
        value wenn ok=True

    Raises:
        ValidationError: Bei Validation-Fehlern (1000-1999)
        ComputationError: Bei Berechnungsfehlern (2000-2999)
        IoError: Bei I/O-Fehlern (3000-3999)
        InternalError: Bei internen Fehlern (4000-4999)
        FfiError: Bei FFI-spezifischen Fehlern (5000-5999)
        ResourceError: Bei Ressourcen-Fehlern (6000-6999)
        OmegaError: Bei unbekannten Fehlern

    Example:
        >>> result = {"ok": True, "value": 42.0}
        >>> handle_ffi_result(result)
        42.0

        >>> result = {
        ...     "ok": False,
        ...     "error_code": 1001,
        ...     "message": "Invalid argument"
        ... }
        >>> handle_ffi_result(result)  # raises ValidationError
    """
    if result.get("ok", False):
        return result.get("value")

    error_code = result.get("error_code", ErrorCode.INTERNAL_ERROR)
    message = result.get("message", "Unknown FFI error")
    context = dict(result.get("context", {}))

    # Map error code range to exception type
    if 1000 <= error_code < 2000:
        raise ValidationError(message, error_code=error_code, **context)
    elif 2000 <= error_code < 3000:
        raise ComputationError(message, error_code=error_code, **context)
    elif 3000 <= error_code < 4000:
        raise IoError(message, error_code=error_code, **context)
    elif 4000 <= error_code < 5000:
        raise InternalError(message, error_code=error_code, **context)
    elif 5000 <= error_code < 6000:
        raise FfiError(message, error_code=error_code, **context)
    elif 6000 <= error_code < 7000:
        raise ResourceError(message, error_code=error_code, **context)
    else:
        raise OmegaError(message, error_code=error_code, context=context)


def ffi_call(func: Callable[..., Mapping[str, Any]]) -> Callable[..., Any]:
    """Decorator für FFI-Funktionen.

    Konvertiert FfiResult-Dicts automatisch zu Values/Exceptions.
    Ermöglicht transparente Integration von Rust/Julia-Funktionen
    in Python-Code mit nativer Exception-Behandlung.

    Args:
        func: FFI-Funktion die ein FfiResult-Dict zurückgibt

    Returns:
        Wrapper-Funktion die den Wert direkt zurückgibt oder
        eine Exception wirft

    Example:
        >>> @ffi_call
        ... def calculate_ema(values: list[float], period: int) -> list[float]:
        ...     return rust_module.calculate_ema(values, period)
        ...
        >>> # Aufruf sieht aus wie normale Python-Funktion:
        >>> result = calculate_ema([1.0, 2.0, 3.0], 2)
        >>> # Wirft ValidationError wenn period <= 0
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return handle_ffi_result(result)

    return wrapper


def ffi_safe_call(
    func: Callable[..., R],
    *args: Any,
    default: R | None = None,
    reraise: bool = True,
    **kwargs: Any,
) -> R | None:
    """Führt FFI-Call mit optionalem Error-Handling aus.

    Für Fälle wo Fehler erwartet werden und ein Fallback-Wert
    sinnvoll ist (z.B. optionale Optimierungen).

    Args:
        func: FFI-Funktion (bereits mit @ffi_call dekoriert)
        *args: Positionale Argumente für func
        default: Default-Wert bei Fehler (nur wenn reraise=False)
        reraise: Wenn True, werden Exceptions weitergeleitet
        **kwargs: Keyword-Argumente für func

    Returns:
        Rückgabewert von func oder default bei Fehler

    Raises:
        OmegaError: Wenn reraise=True und Fehler auftritt

    Example:
        >>> # Mit Fallback bei Fehler
        >>> result = ffi_safe_call(
        ...     rust_ema, values, period,
        ...     default=python_ema(values, period),
        ...     reraise=False
        ... )
    """
    try:
        return func(*args, **kwargs)
    except OmegaError:
        if reraise:
            raise
        return default


class FfiResultBuilder:
    """Builder für FfiResult-Dicts (Python-seitige FFI-Funktionen).

    Für Python-Funktionen die von Rust/Julia aufgerufen werden
    und ein konsistentes FfiResult-Format zurückgeben müssen.

    Example:
        >>> def python_callback(x: float) -> dict:
        ...     try:
        ...         result = expensive_computation(x)
        ...         return FfiResultBuilder.ok(result)
        ...     except ValueError as e:
        ...         return FfiResultBuilder.error(
        ...             ErrorCode.INVALID_ARGUMENT,
        ...             str(e),
        ...             {"input": x}
        ...         )
    """

    @staticmethod
    def ok(value: T) -> dict[str, Any]:
        """Erstellt erfolgreiches FfiResult.

        Args:
            value: Rückgabewert

        Returns:
            FfiResult-Dict mit ok=True
        """
        return {
            "ok": True,
            "value": value,
            "error_code": ErrorCode.OK,
            "message": None,
            "context": {},
        }

    @staticmethod
    def error(
        error_code: int | ErrorCode,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Erstellt Fehler-FfiResult.

        Args:
            error_code: Error-Code aus ErrorCode enum
            message: Menschenlesbare Fehlermeldung
            context: Zusätzlicher Kontext für Debugging

        Returns:
            FfiResult-Dict mit ok=False
        """
        return {
            "ok": False,
            "value": None,
            "error_code": int(error_code),
            "message": message,
            "context": context or {},
        }

    @staticmethod
    def from_exception(exc: Exception) -> dict[str, Any]:
        """Erstellt FfiResult aus Exception.

        Args:
            exc: Python Exception

        Returns:
            FfiResult-Dict mit Fehlerinformationen
        """
        if isinstance(exc, OmegaError):
            return FfiResultBuilder.error(
                exc.error_code,
                exc.message,
                exc.context,
            )

        # Standard Python Exceptions
        error_map: dict[type, ErrorCode] = {
            ValueError: ErrorCode.INVALID_ARGUMENT,
            TypeError: ErrorCode.TYPE_MISMATCH,
            IndexError: ErrorCode.OUT_OF_BOUNDS,
            KeyError: ErrorCode.MISSING_REQUIRED_FIELD,
            FileNotFoundError: ErrorCode.FILE_NOT_FOUND,
            PermissionError: ErrorCode.PERMISSION_DENIED,
            TimeoutError: ErrorCode.TIMEOUT,
            MemoryError: ErrorCode.OUT_OF_MEMORY,
            NotImplementedError: ErrorCode.NOT_IMPLEMENTED,
            ZeroDivisionError: ErrorCode.DIVISION_BY_ZERO,
            OverflowError: ErrorCode.OVERFLOW,
        }

        error_code = error_map.get(type(exc), ErrorCode.INTERNAL_ERROR)
        return FfiResultBuilder.error(
            error_code,
            str(exc),
            {"exception_type": type(exc).__name__},
        )


__all__ = [
    "handle_ffi_result",
    "ffi_call",
    "ffi_safe_call",
    "FfiResultBuilder",
]
