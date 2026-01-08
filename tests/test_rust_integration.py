"""Rust FFI Integration Tests.

These tests are intentionally marker-gated (pytest -m rust_integration).
They verify *runtime importability* and a minimal functional surface of the
Rust extension module.

They default to "skip" when the Rust wheel is not installed.
CI workflows that build/install the wheel should set OMEGA_REQUIRE_RUST_FFI=1
so missing imports become hard failures.
"""

from __future__ import annotations

import os

import pytest

from shared.error_codes import ErrorCode

pytestmark = [pytest.mark.integration, pytest.mark.rust_integration]


def test_rust_module_import_and_api_surface() -> None:
    require_rust = os.environ.get("OMEGA_REQUIRE_RUST_FFI") == "1"

    try:
        from omega import _rust  # type: ignore[import-not-found]
    except ImportError as exc:
        if require_rust:
            pytest.fail(
                "Rust FFI module is required but not importable: import omega._rust failed. "
                "Build/install the wheel via maturin first.",
                pytrace=False,
            )
        pytest.skip(f"omega._rust not installed ({exc}).")
        return

    for attr in ("ema", "rsi", "rolling_std", "get_error_code_constants"):
        assert hasattr(_rust, attr), f"omega._rust missing expected symbol: {attr}"


def test_rust_error_code_constants_match_python() -> None:
    require_rust = os.environ.get("OMEGA_REQUIRE_RUST_FFI") == "1"

    try:
        from omega._rust import (
            get_error_code_constants,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        if require_rust:
            pytest.fail(
                "Rust FFI module is required but not importable: from omega._rust import get_error_code_constants failed.",
                pytrace=False,
            )
        pytest.skip(f"omega._rust not installed ({exc}).")
        return

    rust_codes = get_error_code_constants()
    assert isinstance(rust_codes, dict)

    # Compare a representative set of codes to guard the cross-language contract.
    expected = {
        "OK": ErrorCode.OK,
        "VALIDATION_FAILED": ErrorCode.VALIDATION_FAILED,
        "INVALID_ARGUMENT": ErrorCode.INVALID_ARGUMENT,
        "OUT_OF_BOUNDS": ErrorCode.OUT_OF_BOUNDS,
        "COMPUTATION_FAILED": ErrorCode.COMPUTATION_FAILED,
        "DIVISION_BY_ZERO": ErrorCode.DIVISION_BY_ZERO,
        "IO_ERROR": ErrorCode.IO_ERROR,
        "TIMEOUT": ErrorCode.TIMEOUT,
        "INTERNAL_ERROR": ErrorCode.INTERNAL_ERROR,
        "NOT_IMPLEMENTED": ErrorCode.NOT_IMPLEMENTED,
        "FFI_ERROR": ErrorCode.FFI_ERROR,
        "FFI_PANIC_CAUGHT": ErrorCode.FFI_PANIC_CAUGHT,
        "RESOURCE_ERROR": ErrorCode.RESOURCE_ERROR,
        "OUT_OF_MEMORY": ErrorCode.OUT_OF_MEMORY,
    }

    for name, expected_value in expected.items():
        assert name in rust_codes, f"Rust codes missing key {name!r}"
        assert rust_codes[name] == int(
            expected_value
        ), f"Mismatch for {name}: rust={rust_codes[name]} python={int(expected_value)}"


def test_rust_indicator_smoke() -> None:
    require_rust = os.environ.get("OMEGA_REQUIRE_RUST_FFI") == "1"

    try:
        from omega._rust import ema, rolling_std, rsi  # type: ignore[import-not-found]
    except ImportError as exc:
        if require_rust:
            pytest.fail(
                "Rust FFI module is required but not importable: omega._rust indicator functions missing.",
                pytrace=False,
            )
        pytest.skip(f"omega._rust not installed ({exc}).")
        return

    # Use enough data points for RSI (need at least period + 1)
    prices = [
        100.0,
        101.5,
        99.8,
        102.3,
        103.1,
        102.0,
        104.5,
        103.8,
        105.2,
        104.0,
        106.1,
        105.5,
        107.0,
        106.3,
        108.2,
        107.5,
    ]

    ema_out = ema(prices, period=3)
    assert len(ema_out) == len(prices)

    rsi_out = rsi(prices, period=14)
    assert len(rsi_out) == len(prices)

    returns = [0.01, -0.02, 0.015, -0.005, 0.02]
    std_out = rolling_std(returns, window=3)
    assert len(std_out) == len(returns)
