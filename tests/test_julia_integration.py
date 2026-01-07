"""Julia FFI Integration Tests.

Marker-gated via pytest -m julia_integration.

These tests verify that Julia is callable from Python via juliacall and that the
OmegaJulia package can be loaded in the configured Julia project.

CI workflows should set OMEGA_REQUIRE_JULIA_FFI=1 so missing dependencies or
failed imports become hard failures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.julia_integration]


def _require_julia() -> bool:
    return os.environ.get("OMEGA_REQUIRE_JULIA_FFI") == "1"


def test_juliacall_available_and_basic_eval() -> None:
    try:
        from juliacall import Main as jl  # type: ignore[import-not-found]
    except Exception as exc:
        if _require_julia():
            pytest.fail(
                f"Julia FFI is required but juliacall is not usable: {exc}",
                pytrace=False,
            )
        pytest.skip(f"juliacall not available/usable ({exc}).")
        return

    assert jl.seval("1 + 1") == 2


def test_omega_julia_package_loads_and_smoke_function() -> None:
    julia_project = os.environ.get("JULIA_PROJECT")
    if not julia_project:
        if _require_julia():
            pytest.fail(
                "JULIA_PROJECT is required for Julia integration tests but is not set.",
                pytrace=False,
            )
        pytest.skip("JULIA_PROJECT not set.")
        return

    if not Path(julia_project).exists():
        if _require_julia():
            pytest.fail(
                f"JULIA_PROJECT points to a non-existent path: {julia_project}",
                pytrace=False,
            )
        pytest.skip(f"JULIA_PROJECT path missing: {julia_project}")
        return

    try:
        from juliacall import Main as jl  # type: ignore[import-not-found]
    except Exception as exc:
        if _require_julia():
            pytest.fail(
                f"Julia FFI is required but juliacall is not usable: {exc}",
                pytrace=False,
            )
        pytest.skip(f"juliacall not available/usable ({exc}).")
        return

    # Ensure the package can be loaded within the active project.
    jl.seval("using OmegaJulia")

    returns = [0.01, -0.02, 0.015, -0.005, 0.02]

    # Keep this tiny: just enough to validate wiring without stressing CI.
    var_95 = jl.seval(
        "OmegaJulia.monte_carlo_var([0.01, -0.02, 0.015, -0.005, 0.02], 1000, 0.95; seed=42)"
    )

    assert isinstance(var_95, (float, int))
    assert var_95 <= 0.0

    sharpe = jl.seval(
        "OmegaJulia.rolling_sharpe([0.01, -0.02, 0.015, -0.005, 0.02], 3; annualization=252)"
    )
    # rolling_sharpe returns a vector; juliacall converts to a Python list/array-like.
    assert len(sharpe) == len(returns)
