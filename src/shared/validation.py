"""Validation helpers used across module boundaries.

These utilities are primarily intended for validating inputs/outputs at
FFI boundaries (Python ↔ Rust ↔ Julia) where we want structured errors
(`ValidationError` + `ErrorCode`) instead of ad-hoc exceptions.

Referenced by: `docs/ffi/nullability-convention.md`
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from .error_codes import ErrorCode
from .exceptions import ValidationError

T = TypeVar("T")


def require_not_none(value: T | None, field: str) -> T:
    """Assert that a value is not None.

    Args:
        value: Value to validate.
        field: Field name for error context.

    Returns:
        The value if it is not None.

    Raises:
        ValidationError: If the value is None.
    """

    if value is None:
        raise ValidationError(
            f"Required field '{field}' is None",
            field=field,
            error_code=ErrorCode.NULL_POINTER,
        )

    return value


def validate_no_nan(array: np.ndarray, field: str) -> None:
    """Validate that an array does not contain NaN values.

    Args:
        array: NumPy array to validate.
        field: Field name for error context.

    Raises:
        ValidationError: If the array contains NaN values or is not a float array.
    """

    arr = np.asarray(array)

    # Empty inputs are allowed; callers typically validate emptiness separately.
    if arr.size == 0:
        return

    if not (
        np.issubdtype(arr.dtype, np.floating)
        or np.issubdtype(arr.dtype, np.complexfloating)
    ):
        raise ValidationError(
            f"Field '{field}' must be a floating array to validate NaN values (dtype={arr.dtype})",
            field=field,
            error_code=ErrorCode.TYPE_MISMATCH,
            dtype=str(arr.dtype),
        )

    nan_mask = np.isnan(arr)
    if bool(nan_mask.any()):
        nan_count = int(nan_mask.sum())
        raise ValidationError(
            f"Field '{field}' contains {nan_count} NaN values",
            field=field,
            error_code=ErrorCode.NAN_RESULT,
            nan_count=nan_count,
            size=int(arr.size),
        )
