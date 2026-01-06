"""FFI Contract Tests für Cross-Language Interface Stability.

Diese Tests gewährleisten, dass:
1. ErrorCode-Werte nicht verändert werden (Rust/Julia sync)
2. ErrorCode-Kategorien stabil bleiben
3. is_recoverable() Semantik konsistent ist
4. FfiResult-Struktur kompatibel bleibt
5. Alle Error-Kategorien vollständig abgedeckt sind

⚠️  ACHTUNG: Änderungen an diesen Tests erfordern synchrone
    Updates in Rust (error.rs) und Julia (error.jl)!
"""

from __future__ import annotations

import pytest

from shared.error_codes import ErrorCode, error_category, is_recoverable
from shared.exceptions import (
    ComputationError,
    FfiError,
    InternalError,
    IoError,
    OmegaError,
    ResourceError,
    ValidationError,
)
from shared.ffi_wrapper import FfiResultBuilder, handle_ffi_result


class TestErrorCodeStability:
    """Tests that ensure ErrorCode values never change.

    CRITICAL: Any change here requires synchronized updates in:
    - src/rust_modules/omega_rust/src/error.rs
    - src/julia_modules/omega_julia/src/error.jl
    """

    def test_ok_value_is_zero(self) -> None:
        """OK must always be 0 across all languages."""
        assert ErrorCode.OK == 0, "ErrorCode.OK MUST be 0 for cross-language compat"

    def test_validation_codes_range(self) -> None:
        """Validation errors must be in 1000-1999 range."""
        validation_codes = [
            ErrorCode.VALIDATION_FAILED,
            ErrorCode.INVALID_ARGUMENT,
            ErrorCode.NULL_POINTER,
            ErrorCode.OUT_OF_BOUNDS,
            ErrorCode.TYPE_MISMATCH,
            ErrorCode.SCHEMA_VIOLATION,
            ErrorCode.CONSTRAINT_VIOLATION,
            ErrorCode.INVALID_STATE,
            ErrorCode.MISSING_REQUIRED_FIELD,
            ErrorCode.INVALID_FORMAT,
            ErrorCode.EMPTY_INPUT,
            ErrorCode.SIZE_MISMATCH,
        ]
        for code in validation_codes:
            assert 1000 <= code < 2000, f"{code.name}={code} not in validation range"

    def test_computation_codes_range(self) -> None:
        """Computation errors must be in 2000-2999 range."""
        computation_codes = [
            ErrorCode.COMPUTATION_FAILED,
            ErrorCode.DIVISION_BY_ZERO,
            ErrorCode.OVERFLOW,
            ErrorCode.UNDERFLOW,
            ErrorCode.NAN_RESULT,
            ErrorCode.INF_RESULT,
            ErrorCode.CONVERGENCE_FAILED,
            ErrorCode.NUMERICAL_INSTABILITY,
            ErrorCode.INSUFFICIENT_DATA,
        ]
        for code in computation_codes:
            assert 2000 <= code < 3000, f"{code.name}={code} not in computation range"

    def test_io_codes_range(self) -> None:
        """I/O errors must be in 3000-3999 range."""
        io_codes = [
            ErrorCode.IO_ERROR,
            ErrorCode.FILE_NOT_FOUND,
            ErrorCode.PERMISSION_DENIED,
            ErrorCode.SERIALIZATION_FAILED,
            ErrorCode.DESERIALIZATION_FAILED,
            ErrorCode.NETWORK_ERROR,
            ErrorCode.TIMEOUT,
            ErrorCode.DISK_FULL,
        ]
        for code in io_codes:
            assert 3000 <= code < 4000, f"{code.name}={code} not in IO range"

    def test_internal_codes_range(self) -> None:
        """Internal errors must be in 4000-4999 range."""
        internal_codes = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.NOT_IMPLEMENTED,
            ErrorCode.ASSERTION_FAILED,
            ErrorCode.UNREACHABLE,
            ErrorCode.INVARIANT_VIOLATED,
        ]
        for code in internal_codes:
            assert 4000 <= code < 5000, f"{code.name}={code} not in internal range"

    def test_ffi_codes_range(self) -> None:
        """FFI errors must be in 5000-5999 range."""
        ffi_codes = [
            ErrorCode.FFI_ERROR,
            ErrorCode.FFI_TYPE_CONVERSION,
            ErrorCode.FFI_BUFFER_OVERFLOW,
            ErrorCode.FFI_MEMORY_ERROR,
            ErrorCode.FFI_SCHEMA_MISMATCH,
            ErrorCode.FFI_PANIC_CAUGHT,
        ]
        for code in ffi_codes:
            assert 5000 <= code < 6000, f"{code.name}={code} not in FFI range"

    def test_resource_codes_range(self) -> None:
        """Resource errors must be in 6000-6999 range."""
        resource_codes = [
            ErrorCode.RESOURCE_ERROR,
            ErrorCode.OUT_OF_MEMORY,
            ErrorCode.RESOURCE_EXHAUSTED,
            ErrorCode.RESOURCE_BUSY,
            ErrorCode.RESOURCE_LIMIT_EXCEEDED,
        ]
        for code in resource_codes:
            assert 6000 <= code < 7000, f"{code.name}={code} not in resource range"

    def test_no_code_renumbering(self) -> None:
        """CRITICAL: Error codes must never be renumbered.

        This test locks the exact numeric values. Any change breaks FFI!
        """
        expected_values = {
            # Success
            "OK": 0,
            # Validation (1000-1999)
            "VALIDATION_FAILED": 1000,
            "INVALID_ARGUMENT": 1001,
            "NULL_POINTER": 1002,
            "OUT_OF_BOUNDS": 1003,
            "TYPE_MISMATCH": 1004,
            "SCHEMA_VIOLATION": 1005,
            "CONSTRAINT_VIOLATION": 1006,
            "INVALID_STATE": 1007,
            "MISSING_REQUIRED_FIELD": 1008,
            "INVALID_FORMAT": 1009,
            "EMPTY_INPUT": 1010,
            "SIZE_MISMATCH": 1011,
            # Computation (2000-2999)
            "COMPUTATION_FAILED": 2000,
            "DIVISION_BY_ZERO": 2001,
            "OVERFLOW": 2002,
            "UNDERFLOW": 2003,
            "NAN_RESULT": 2004,
            "INF_RESULT": 2005,
            "CONVERGENCE_FAILED": 2006,
            "NUMERICAL_INSTABILITY": 2007,
            "INSUFFICIENT_DATA": 2008,
            # I/O (3000-3999)
            "IO_ERROR": 3000,
            "FILE_NOT_FOUND": 3001,
            "PERMISSION_DENIED": 3002,
            "SERIALIZATION_FAILED": 3003,
            "DESERIALIZATION_FAILED": 3004,
            "NETWORK_ERROR": 3005,
            "TIMEOUT": 3006,
            "DISK_FULL": 3007,
            # Internal (4000-4999)
            "INTERNAL_ERROR": 4000,
            "NOT_IMPLEMENTED": 4001,
            "ASSERTION_FAILED": 4002,
            "UNREACHABLE": 4003,
            "INVARIANT_VIOLATED": 4004,
            # FFI (5000-5999)
            "FFI_ERROR": 5000,
            "FFI_TYPE_CONVERSION": 5001,
            "FFI_BUFFER_OVERFLOW": 5002,
            "FFI_MEMORY_ERROR": 5003,
            "FFI_SCHEMA_MISMATCH": 5004,
            "FFI_PANIC_CAUGHT": 5005,
            # Resource (6000-6999)
            "RESOURCE_ERROR": 6000,
            "OUT_OF_MEMORY": 6001,
            "RESOURCE_EXHAUSTED": 6002,
            "RESOURCE_BUSY": 6003,
            "RESOURCE_LIMIT_EXCEEDED": 6004,
        }

        for name, expected in expected_values.items():
            actual = getattr(ErrorCode, name)
            assert (
                actual == expected
            ), f"ErrorCode.{name} changed from {expected} to {actual}! This breaks FFI!"


class TestErrorCategoryMapping:
    """Tests that error_category() correctly maps codes to categories."""

    @pytest.mark.parametrize(
        "code,expected_category",
        [
            (ErrorCode.OK, "OK"),
            (0, "OK"),
            # Validation
            (ErrorCode.VALIDATION_FAILED, "VALIDATION"),
            (ErrorCode.INVALID_ARGUMENT, "VALIDATION"),
            (1999, "VALIDATION"),  # Edge case: max validation code
            # Computation
            (ErrorCode.COMPUTATION_FAILED, "COMPUTATION"),
            (ErrorCode.NAN_RESULT, "COMPUTATION"),
            (2999, "COMPUTATION"),  # Edge case: max computation code
            # I/O
            (ErrorCode.IO_ERROR, "IO"),
            (ErrorCode.FILE_NOT_FOUND, "IO"),
            (3999, "IO"),  # Edge case: max IO code
            # Internal
            (ErrorCode.INTERNAL_ERROR, "INTERNAL"),
            (ErrorCode.NOT_IMPLEMENTED, "INTERNAL"),
            (4999, "INTERNAL"),  # Edge case: max internal code
            # FFI
            (ErrorCode.FFI_ERROR, "FFI"),
            (ErrorCode.FFI_PANIC_CAUGHT, "FFI"),
            (5999, "FFI"),  # Edge case: max FFI code
            # Resource
            (ErrorCode.RESOURCE_ERROR, "RESOURCE"),
            (ErrorCode.OUT_OF_MEMORY, "RESOURCE"),
            (6999, "RESOURCE"),  # Edge case: max resource code
            # Unknown
            (7000, "UNKNOWN"),  # Out of range
            (-1, "UNKNOWN"),  # Negative
        ],
    )
    def test_error_category(self, code: int, expected_category: str) -> None:
        """Test category mapping for all error code ranges."""
        assert error_category(code) == expected_category


class TestIsRecoverable:
    """Tests that is_recoverable() correctly identifies recoverable errors."""

    def test_ok_is_recoverable(self) -> None:
        """OK (no error) should return True."""
        assert is_recoverable(ErrorCode.OK) is True
        assert is_recoverable(0) is True

    def test_validation_errors_are_recoverable(self) -> None:
        """All validation errors should be recoverable (fix input and retry)."""
        validation_codes = [
            ErrorCode.VALIDATION_FAILED,
            ErrorCode.INVALID_ARGUMENT,
            ErrorCode.NULL_POINTER,
            ErrorCode.OUT_OF_BOUNDS,
            ErrorCode.TYPE_MISMATCH,
            ErrorCode.SCHEMA_VIOLATION,
            ErrorCode.CONSTRAINT_VIOLATION,
            ErrorCode.INVALID_STATE,
            ErrorCode.MISSING_REQUIRED_FIELD,
            ErrorCode.INVALID_FORMAT,
            ErrorCode.EMPTY_INPUT,
            ErrorCode.SIZE_MISMATCH,
        ]
        for code in validation_codes:
            assert is_recoverable(code) is True, f"{code.name} should be recoverable"

    def test_io_errors_are_recoverable(self) -> None:
        """All I/O errors should be recoverable (retry possible)."""
        io_codes = [
            ErrorCode.IO_ERROR,
            ErrorCode.FILE_NOT_FOUND,
            ErrorCode.PERMISSION_DENIED,
            ErrorCode.SERIALIZATION_FAILED,
            ErrorCode.DESERIALIZATION_FAILED,
            ErrorCode.NETWORK_ERROR,
            ErrorCode.TIMEOUT,
            ErrorCode.DISK_FULL,
        ]
        for code in io_codes:
            assert is_recoverable(code) is True, f"{code.name} should be recoverable"

    def test_internal_errors_are_not_recoverable(self) -> None:
        """Internal errors (bugs) should NOT be recoverable."""
        internal_codes = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.NOT_IMPLEMENTED,
            ErrorCode.ASSERTION_FAILED,
            ErrorCode.UNREACHABLE,
            ErrorCode.INVARIANT_VIOLATED,
        ]
        for code in internal_codes:
            assert (
                is_recoverable(code) is False
            ), f"{code.name} should NOT be recoverable"

    def test_ffi_errors_are_not_recoverable(self) -> None:
        """FFI errors should NOT be recoverable."""
        ffi_codes = [
            ErrorCode.FFI_ERROR,
            ErrorCode.FFI_TYPE_CONVERSION,
            ErrorCode.FFI_BUFFER_OVERFLOW,
            ErrorCode.FFI_MEMORY_ERROR,
            ErrorCode.FFI_SCHEMA_MISMATCH,
            ErrorCode.FFI_PANIC_CAUGHT,
        ]
        for code in ffi_codes:
            assert (
                is_recoverable(code) is False
            ), f"{code.name} should NOT be recoverable"

    def test_computation_errors_partial_recovery(self) -> None:
        """Only INSUFFICIENT_DATA is recoverable among computation errors."""
        assert is_recoverable(ErrorCode.INSUFFICIENT_DATA) is True
        non_recoverable = [
            ErrorCode.COMPUTATION_FAILED,
            ErrorCode.DIVISION_BY_ZERO,
            ErrorCode.OVERFLOW,
            ErrorCode.UNDERFLOW,
            ErrorCode.NAN_RESULT,
            ErrorCode.INF_RESULT,
            ErrorCode.CONVERGENCE_FAILED,
            ErrorCode.NUMERICAL_INSTABILITY,
        ]
        for code in non_recoverable:
            assert (
                is_recoverable(code) is False
            ), f"{code.name} should NOT be recoverable"

    def test_resource_errors_partial_recovery(self) -> None:
        """Only RESOURCE_BUSY and RESOURCE_LIMIT_EXCEEDED are recoverable."""
        assert is_recoverable(ErrorCode.RESOURCE_BUSY) is True
        assert is_recoverable(ErrorCode.RESOURCE_LIMIT_EXCEEDED) is True
        non_recoverable = [
            ErrorCode.RESOURCE_ERROR,
            ErrorCode.OUT_OF_MEMORY,
            ErrorCode.RESOURCE_EXHAUSTED,
        ]
        for code in non_recoverable:
            assert (
                is_recoverable(code) is False
            ), f"{code.name} should NOT be recoverable"


class TestFfiResultHandling:
    """Tests for FfiResult handling (Python ↔ Rust/Julia interface)."""

    def test_ok_result_returns_value(self) -> None:
        """OK results should return the value directly."""
        result = {"ok": True, "value": 42.5}
        assert handle_ffi_result(result) == 42.5

    def test_ok_result_with_none_value(self) -> None:
        """OK results can have None as value."""
        result = {"ok": True, "value": None}
        assert handle_ffi_result(result) is None

    def test_ok_result_without_value_returns_none(self) -> None:
        """OK results without value key return None."""
        result = {"ok": True}
        assert handle_ffi_result(result) is None

    def test_validation_error_raises(self) -> None:
        """Validation errors (1000-1999) should raise ValidationError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.INVALID_ARGUMENT,
            "message": "Test validation error",
        }
        with pytest.raises(ValidationError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.INVALID_ARGUMENT
        assert "Test validation error" in str(exc_info.value)

    def test_computation_error_raises(self) -> None:
        """Computation errors (2000-2999) should raise ComputationError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.DIVISION_BY_ZERO,
            "message": "Division by zero",
        }
        with pytest.raises(ComputationError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.DIVISION_BY_ZERO

    def test_io_error_raises(self) -> None:
        """I/O errors (3000-3999) should raise IoError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.FILE_NOT_FOUND,
            "message": "File not found",
        }
        with pytest.raises(IoError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND

    def test_internal_error_raises(self) -> None:
        """Internal errors (4000-4999) should raise InternalError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.ASSERTION_FAILED,
            "message": "Internal assertion failed",
        }
        with pytest.raises(InternalError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.ASSERTION_FAILED

    def test_ffi_error_raises(self) -> None:
        """FFI errors (5000-5999) should raise FfiError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.FFI_PANIC_CAUGHT,
            "message": "Rust panic caught",
        }
        with pytest.raises(FfiError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.FFI_PANIC_CAUGHT

    def test_resource_error_raises(self) -> None:
        """Resource errors (6000-6999) should raise ResourceError."""
        result = {
            "ok": False,
            "error_code": ErrorCode.OUT_OF_MEMORY,
            "message": "Out of memory",
        }
        with pytest.raises(ResourceError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.OUT_OF_MEMORY

    def test_unknown_error_raises_omega_error(self) -> None:
        """Unknown error codes should raise OmegaError."""
        result = {
            "ok": False,
            "error_code": 9999,  # Unknown code
            "message": "Unknown error",
        }
        with pytest.raises(OmegaError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == 9999

    def test_context_passed_to_exception(self) -> None:
        """Context from FFI result should be passed to exception."""
        result = {
            "ok": False,
            "error_code": ErrorCode.INVALID_ARGUMENT,
            "message": "Invalid period",
            "context": {"param": "period", "value": -5},
        }
        with pytest.raises(ValidationError) as exc_info:
            handle_ffi_result(result)
        # Context should be in exception's context dict
        assert exc_info.value.context.get("param") == "period"
        assert exc_info.value.context.get("value") == -5


class TestFfiResultBuilder:
    """Tests for FfiResultBuilder (Python callbacks for Rust/Julia)."""

    def test_ok_creates_success_result(self) -> None:
        """ok() should create a valid success result."""
        result = FfiResultBuilder.ok(42.5)
        assert result["ok"] is True
        assert result["value"] == 42.5

    def test_ok_with_none_value(self) -> None:
        """ok() with None value should still be valid."""
        result = FfiResultBuilder.ok(None)
        assert result["ok"] is True
        assert result["value"] is None

    def test_error_creates_error_result(self) -> None:
        """error() should create a valid error result."""
        result = FfiResultBuilder.error(
            ErrorCode.INVALID_ARGUMENT, "Test error", {"param": "x"}
        )
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.INVALID_ARGUMENT
        assert result["message"] == "Test error"
        assert result["context"]["param"] == "x"

    def test_error_without_context(self) -> None:
        """error() without context should use empty dict."""
        result = FfiResultBuilder.error(ErrorCode.INTERNAL_ERROR, "Bug")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.INTERNAL_ERROR
        assert result["message"] == "Bug"
        assert result["context"] == {}

    def test_round_trip_ok(self) -> None:
        """Builder output should be handleable by handle_ffi_result."""
        result = FfiResultBuilder.ok([1.0, 2.0, 3.0])
        value = handle_ffi_result(result)
        assert value == [1.0, 2.0, 3.0]

    def test_round_trip_error(self) -> None:
        """Builder error output should raise correct exception."""
        result = FfiResultBuilder.error(ErrorCode.NAN_RESULT, "NaN detected")
        with pytest.raises(ComputationError) as exc_info:
            handle_ffi_result(result)
        assert exc_info.value.error_code == ErrorCode.NAN_RESULT


class TestErrorCodeEnumCompleteness:
    """Tests that ErrorCode enum is complete and consistent."""

    def test_all_categories_have_base_code(self) -> None:
        """Each category should have a base error code ending in 000."""
        base_codes = {
            "VALIDATION_FAILED": 1000,
            "COMPUTATION_FAILED": 2000,
            "IO_ERROR": 3000,
            "INTERNAL_ERROR": 4000,
            "FFI_ERROR": 5000,
            "RESOURCE_ERROR": 6000,
        }
        for name, expected in base_codes.items():
            code = getattr(ErrorCode, name)
            assert code == expected, f"Base code {name} should be {expected}"

    def test_no_gaps_in_sequential_codes(self) -> None:
        """Within each category, codes should be sequential (no gaps)."""
        ranges = [
            (1000, 1011),  # Validation
            (2000, 2008),  # Computation
            (3000, 3007),  # I/O
            (4000, 4004),  # Internal
            (5000, 5005),  # FFI
            (6000, 6004),  # Resource
        ]
        all_codes = {int(c) for c in ErrorCode}
        for start, end in ranges:
            for expected in range(start, end + 1):
                assert (
                    expected in all_codes
                ), f"Missing code {expected} in ErrorCode enum"

    def test_no_duplicate_values(self) -> None:
        """All ErrorCode values must be unique."""
        values = [int(c) for c in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate values in ErrorCode enum!"

    def test_all_codes_are_integers(self) -> None:
        """All ErrorCode values must be integers."""
        for code in ErrorCode:
            assert isinstance(int(code), int)
            assert int(code) >= 0, f"Error code {code.name} is negative"
