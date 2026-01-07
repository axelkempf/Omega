from __future__ import annotations

import numpy as np
import pytest

from src.shared.error_codes import ErrorCode
from src.shared.exceptions import ValidationError
from src.shared.validation import require_not_none, validate_no_nan


def test_require_not_none_returns_value() -> None:
    assert require_not_none(123, field="x") == 123


def test_require_not_none_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        require_not_none(None, field="entry")

    err = excinfo.value
    assert err.error_code == int(ErrorCode.NULL_POINTER)
    assert err.context["field"] == "entry"


def test_validate_no_nan_allows_empty_array() -> None:
    validate_no_nan(np.array([], dtype=np.float64), field="prices")


def test_validate_no_nan_allows_float_array_without_nan() -> None:
    validate_no_nan(np.array([1.0, 2.0, 3.0], dtype=np.float64), field="prices")


def test_validate_no_nan_raises_on_nan() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_no_nan(np.array([1.0, np.nan, 3.0], dtype=np.float64), field="prices")

    err = excinfo.value
    assert err.error_code == int(ErrorCode.NAN_RESULT)
    assert err.context["field"] == "prices"
    assert err.context["nan_count"] == 1


def test_validate_no_nan_raises_on_non_float_dtype() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_no_nan(np.array([1, 2, 3], dtype=np.int64), field="prices")

    err = excinfo.value
    assert err.error_code == int(ErrorCode.TYPE_MISMATCH)
    assert err.context["field"] == "prices"
