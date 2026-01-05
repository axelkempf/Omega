"""Omega Exception Hierarchy für strukturierte Fehlerbehandlung.

Phase 2 Task: P2-07
Status: Implementiert (2026-01-05)

Diese Exceptions werden an FFI-Grenzen in strukturierte Fehler konvertiert
und können von FFI-Ergebnissen zurück in Exceptions transformiert werden.

Referenz: docs/adr/ADR-0003-error-handling.md
"""

from __future__ import annotations

from typing import Any

from .error_codes import ErrorCode, error_category


class OmegaError(Exception):
    """Basis-Exception für alle Omega-Fehler.

    Alle Omega-spezifischen Exceptions erben von dieser Klasse.
    Sie unterstützt strukturierte Fehlerinformationen für:
    - Logging (error_code, context)
    - FFI-Transfer (to_ffi_dict)
    - Debugging (message mit Kontext)

    Attributes:
        message: Menschenlesbare Fehlermeldung
        error_code: Numerischer Error-Code (siehe ErrorCode enum)
        context: Dict mit zusätzlichem Kontext für Debugging

    Example:
        >>> try:
        ...     raise OmegaError("Something went wrong", error_code=4000)
        ... except OmegaError as e:
        ...     log.error(f"[{e.error_code}] {e.message}", extra=e.context)
    """

    def __init__(
        self,
        message: str,
        error_code: int | ErrorCode = ErrorCode.INTERNAL_ERROR,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialisiert OmegaError.

        Args:
            message: Fehlermeldung
            error_code: Error-Code (default: INTERNAL_ERROR)
            context: Zusätzlicher Kontext für Debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = int(error_code)
        self.context = context or {}

    def __str__(self) -> str:
        """String-Repräsentation mit Error-Code."""
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        """Debug-Repräsentation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code}, "
            f"context={self.context!r})"
        )

    @property
    def category(self) -> str:
        """Fehler-Kategorie basierend auf error_code."""
        return error_category(self.error_code)

    def to_ffi_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu FFI-kompatiblem Dict.

        Returns:
            Dict mit keys: ok, error_code, message, context, category

        Example:
            >>> e = ValidationError("Invalid value", field="price")
            >>> e.to_ffi_dict()
            {'ok': False, 'error_code': 1000, 'message': 'Invalid value',
             'context': {'field': 'price'}, 'category': 'VALIDATION'}
        """
        return {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "category": self.category,
        }

    @classmethod
    def from_ffi_dict(cls, data: dict[str, Any]) -> "OmegaError":
        """Erstellt Exception aus FFI-Dict.

        Args:
            data: Dict mit error_code, message, context

        Returns:
            Passende OmegaError Subclass basierend auf error_code
        """
        error_code = data.get("error_code", ErrorCode.INTERNAL_ERROR)
        message = data.get("message", "Unknown error")
        context = data.get("context", {})

        # Map error_code to exception type
        exception_class = _get_exception_class(error_code)
        return exception_class(message, error_code=error_code, context=context)


class ValidationError(OmegaError):
    """Input-Validierungsfehler.

    Für Fehler bei der Validierung von Eingabedaten:
    - Ungültige Argumente
    - Typ-Mismatches
    - Constraint-Verletzungen
    - Fehlende Pflichtfelder

    Attributes:
        field: Name des fehlerhaften Felds (optional)
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        error_code: int | ErrorCode = ErrorCode.VALIDATION_FAILED,
        **context: Any,
    ) -> None:
        """Initialisiert ValidationError.

        Args:
            message: Fehlermeldung
            field: Name des fehlerhaften Felds
            error_code: Spezifischer Validation-Error-Code
            **context: Zusätzlicher Kontext
        """
        if field:
            context["field"] = field
        super().__init__(message, error_code=error_code, context=context)
        self.field = field


class ComputationError(OmegaError):
    """Berechnungsfehler.

    Für Fehler bei numerischen Berechnungen:
    - Division durch Null
    - Overflow/Underflow
    - NaN/Inf Ergebnisse
    - Konvergenz-Fehler

    Attributes:
        operation: Name der fehlgeschlagenen Operation (optional)
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        error_code: int | ErrorCode = ErrorCode.COMPUTATION_FAILED,
        **context: Any,
    ) -> None:
        """Initialisiert ComputationError.

        Args:
            message: Fehlermeldung
            operation: Name der fehlgeschlagenen Operation
            error_code: Spezifischer Computation-Error-Code
            **context: Zusätzlicher Kontext
        """
        if operation:
            context["operation"] = operation
        super().__init__(message, error_code=error_code, context=context)
        self.operation = operation


class IoError(OmegaError):
    """I/O-Fehler.

    Für Ein/Ausgabe-bezogene Fehler:
    - Datei nicht gefunden
    - Netzwerk-Fehler
    - Serialisierungs-Fehler
    - Timeouts

    Attributes:
        path: Pfad oder Ressource (optional)
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        error_code: int | ErrorCode = ErrorCode.IO_ERROR,
        **context: Any,
    ) -> None:
        """Initialisiert IoError.

        Args:
            message: Fehlermeldung
            path: Betroffener Pfad oder Ressource
            error_code: Spezifischer IO-Error-Code
            **context: Zusätzlicher Kontext
        """
        if path:
            context["path"] = path
        super().__init__(message, error_code=error_code, context=context)
        self.path = path


class FfiError(OmegaError):
    """FFI-spezifische Fehler.

    Für Fehler an FFI-Grenzen:
    - Typ-Konvertierung fehlgeschlagen
    - Schema-Mismatch
    - Rust-Panic abgefangen

    Attributes:
        rust_error: Original Rust-Fehlermeldung (optional)
    """

    def __init__(
        self,
        message: str,
        rust_error: str | None = None,
        error_code: int | ErrorCode = ErrorCode.FFI_ERROR,
        **context: Any,
    ) -> None:
        """Initialisiert FfiError.

        Args:
            message: Fehlermeldung
            rust_error: Original Rust-Fehlermeldung
            error_code: Spezifischer FFI-Error-Code
            **context: Zusätzlicher Kontext
        """
        if rust_error:
            context["rust_error"] = rust_error
        super().__init__(message, error_code=error_code, context=context)
        self.rust_error = rust_error


class InternalError(OmegaError):
    """Interne Fehler (Bugs).

    Für Fehler die nicht auftreten sollten:
    - Assertion fehlgeschlagen
    - Unreachable Code erreicht
    - Invarianten verletzt

    Diese Fehler deuten auf Bugs im Code hin und sollten
    gemeldet werden.
    """

    def __init__(
        self,
        message: str,
        error_code: int | ErrorCode = ErrorCode.INTERNAL_ERROR,
        **context: Any,
    ) -> None:
        """Initialisiert InternalError.

        Args:
            message: Fehlermeldung
            error_code: Spezifischer Internal-Error-Code
            **context: Zusätzlicher Kontext
        """
        super().__init__(message, error_code=error_code, context=context)


class ResourceError(OmegaError):
    """Ressourcen-Fehler.

    Für Fehler bei Ressourcen-Erschöpfung:
    - Out of Memory
    - Ressource belegt
    - Limits überschritten

    Attributes:
        resource: Name der betroffenen Ressource (optional)
    """

    def __init__(
        self,
        message: str,
        resource: str | None = None,
        error_code: int | ErrorCode = ErrorCode.RESOURCE_ERROR,
        **context: Any,
    ) -> None:
        """Initialisiert ResourceError.

        Args:
            message: Fehlermeldung
            resource: Name der betroffenen Ressource
            error_code: Spezifischer Resource-Error-Code
            **context: Zusätzlicher Kontext
        """
        if resource:
            context["resource"] = resource
        super().__init__(message, error_code=error_code, context=context)
        self.resource = resource


def _get_exception_class(error_code: int) -> type[OmegaError]:
    """Ermittelt Exception-Klasse basierend auf Error-Code.

    Args:
        error_code: Numerischer Error-Code

    Returns:
        Passende Exception-Klasse
    """
    if 1000 <= error_code < 2000:
        return ValidationError
    elif 2000 <= error_code < 3000:
        return ComputationError
    elif 3000 <= error_code < 4000:
        return IoError
    elif 4000 <= error_code < 5000:
        return InternalError
    elif 5000 <= error_code < 6000:
        return FfiError
    elif 6000 <= error_code < 7000:
        return ResourceError
    else:
        return OmegaError


def raise_from_ffi(result: dict[str, Any]) -> None:
    """Wirft Exception wenn FFI-Result einen Fehler enthält.

    Args:
        result: FFI-Result Dict mit ok, error_code, message, context

    Raises:
        OmegaError (oder Subclass): Wenn result["ok"] == False

    Example:
        >>> result = {"ok": False, "error_code": 1001, "message": "Invalid"}
        >>> raise_from_ffi(result)  # Raises ValidationError
    """
    if result.get("ok", False):
        return

    raise OmegaError.from_ffi_dict(result)


__all__ = [
    # Base Exception
    "OmegaError",
    # Specific Exceptions
    "ValidationError",
    "ComputationError",
    "IoError",
    "FfiError",
    "InternalError",
    "ResourceError",
    # Utilities
    "raise_from_ffi",
]
