"""Tests für Rust↔Python Parity der Slippage und Fee Module.

Diese Tests validieren, dass die Rust-Implementierung identische
Ergebnisse wie die Python-Implementierung liefert.

WICHTIG: Diese Tests werden nur ausgeführt wenn das Rust-Modul
installiert ist (omega_rust).

Reference: docs/WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from src.backtest_engine.core.slippage_and_fee import (
    FeeModel,
    SlippageModel,
    get_rust_status,
)

if TYPE_CHECKING:
    pass


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def rust_available() -> bool:
    """Check if Rust module is available."""
    status = get_rust_status()
    return bool(status["available"])


@pytest.fixture
def force_python_mode():
    """Force Python mode for testing."""
    old_value = os.environ.get("OMEGA_USE_RUST_SLIPPAGE_FEE")
    os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "false"
    yield
    if old_value is not None:
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = old_value
    else:
        os.environ.pop("OMEGA_USE_RUST_SLIPPAGE_FEE", None)


@pytest.fixture
def force_rust_mode():
    """Force Rust mode for testing (if available)."""
    old_value = os.environ.get("OMEGA_USE_RUST_SLIPPAGE_FEE")
    os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "true"
    yield
    if old_value is not None:
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = old_value
    else:
        os.environ.pop("OMEGA_USE_RUST_SLIPPAGE_FEE", None)


# ==============================================================================
# RUST STATUS TESTS
# ==============================================================================


class TestRustStatus:
    """Tests für Rust-Modul Status-Abfrage."""

    def test_get_rust_status_returns_dict(self) -> None:
        """get_rust_status muss ein Dict mit definierten Keys zurückgeben."""
        status = get_rust_status()
        assert "available" in status
        assert "enabled" in status
        assert "reason" in status

    def test_rust_status_available_is_bool(self) -> None:
        """available muss ein Boolean sein."""
        status = get_rust_status()
        assert isinstance(status["available"], bool)

    def test_force_python_mode_disables_rust(self, force_python_mode) -> None:
        """OMEGA_USE_RUST_SLIPPAGE_FEE=false muss Rust deaktivieren."""
        status = get_rust_status()
        assert status["enabled"] is False


# ==============================================================================
# SLIPPAGE PARITY TESTS
# ==============================================================================


@pytest.mark.rust_integration
class TestSlippageRustParity:
    """Tests für Rust↔Python Slippage Parity."""

    def test_slippage_parity_deterministic(
        self, rust_available, force_python_mode
    ) -> None:
        """Slippage mit Seed: Both implementations should be internally deterministic.
        
        NOTE: Python's random.Random and Rust's ChaCha8 use different RNG algorithms,
        so their outputs with the same seed will differ. What matters is:
        1. Each implementation is internally deterministic (same seed = same result)
        2. Fixed-only slippage (no random component) matches exactly
        
        When migrating to Rust, golden files need to be re-generated with Rust results.
        """
        if not rust_available:
            pytest.skip("Rust-Modul nicht verfügbar")

        model = SlippageModel(fixed_pips=0.5, random_pips=1.0)
        price = 1.10000
        pip_size = 0.0001
        seed = 42

        # Test Python determinism
        python_result1 = model._apply_python(price, "long", pip_size, seed)
        python_result2 = model._apply_python(price, "long", pip_size, seed)
        assert abs(python_result1 - python_result2) < 1e-10, "Python not deterministic"

        # Test Rust determinism
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "true"
        status = get_rust_status()
        if not status["enabled"]:
            pytest.skip("Rust mode could not be enabled")

        rust_result1 = model._apply_rust(price, "long", pip_size, seed)
        rust_result2 = model._apply_rust(price, "long", pip_size, seed)
        assert abs(rust_result1 - rust_result2) < 1e-10, "Rust not deterministic"

        # Document the expected difference (different RNG algorithms)
        diff = abs(python_result1 - rust_result1)
        assert diff > 0, "Different RNG algorithms should produce different values"
        # The difference should be bounded within the random_pips range
        assert diff < model.random_pips * pip_size, (
            f"Difference {diff} exceeds max random slippage {model.random_pips * pip_size}"
        )

    def test_slippage_fixed_only_parity(self, rust_available) -> None:
        """Slippage ohne Random muss exakt übereinstimmen."""
        if not rust_available:
            pytest.skip("Rust-Modul nicht verfügbar")

        model = SlippageModel(fixed_pips=0.5, random_pips=0.0)
        price = 1.10000
        pip_size = 0.0001

        # Python result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "false"
        python_result = model.apply(price, "long", pip_size)

        # Rust result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "true"
        status = get_rust_status()
        if not status["enabled"]:
            pytest.skip("Rust mode could not be enabled")

        rust_result = model.apply(price, "long", pip_size)

        # Should be exactly equal (no random component)
        assert abs(python_result - rust_result) < 1e-10, (
            f"Fixed Slippage Parity Fehler!\n"
            f"Python: {python_result}\n"
            f"Rust:   {rust_result}"
        )


# ==============================================================================
# FEE PARITY TESTS
# ==============================================================================


@pytest.mark.rust_integration
class TestFeeRustParity:
    """Tests für Rust↔Python Fee Parity."""

    def test_fee_parity_basic(self, rust_available) -> None:
        """Fee-Berechnung muss zwischen Python und Rust identisch sein."""
        if not rust_available:
            pytest.skip("Rust-Modul nicht verfügbar")

        model = FeeModel(per_million=30.0, lot_size=100_000, min_fee=0.01)
        volume = 1.0
        price = 1.10000
        contract_size = 100_000.0

        # Python result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "false"
        python_result = model.calculate(volume, price, contract_size)

        # Rust result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "true"
        status = get_rust_status()
        if not status["enabled"]:
            pytest.skip("Rust mode could not be enabled")

        rust_result = model.calculate(volume, price, contract_size)

        assert abs(python_result - rust_result) < 1e-10, (
            f"Fee Parity Fehler!\n"
            f"Python: {python_result}\n"
            f"Rust:   {rust_result}"
        )

    def test_fee_minimum_parity(self, rust_available) -> None:
        """Minimum Fee muss in beiden Implementierungen gleich angewandt werden."""
        if not rust_available:
            pytest.skip("Rust-Modul nicht verfügbar")

        min_fee = 1.00
        model = FeeModel(per_million=30.0, lot_size=100_000, min_fee=min_fee)
        volume = 0.001  # Very small volume to trigger min_fee
        price = 1.0
        contract_size = 100_000.0

        # Python result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "false"
        python_result = model.calculate(volume, price, contract_size)

        # Rust result
        os.environ["OMEGA_USE_RUST_SLIPPAGE_FEE"] = "true"
        status = get_rust_status()
        if not status["enabled"]:
            pytest.skip("Rust mode could not be enabled")

        rust_result = model.calculate(volume, price, contract_size)

        assert python_result == min_fee, "Python should return min_fee"
        assert rust_result == min_fee, "Rust should return min_fee"
        assert python_result == rust_result


# ==============================================================================
# BATCH OPERATION TESTS
# ==============================================================================


@pytest.mark.rust_integration
class TestBatchOperations:
    """Tests für Batch-Operationen."""

    def test_slippage_batch_python_fallback(self, force_python_mode) -> None:
        """Batch-Slippage muss im Python-Modus funktionieren."""
        model = SlippageModel(fixed_pips=0.5, random_pips=1.0)
        prices = [1.1, 1.2, 1.3]
        directions = ["long", "short", "long"]

        results = model.apply_batch(prices, directions, seed=42)

        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_fee_batch_python_fallback(self, force_python_mode) -> None:
        """Batch-Fee muss im Python-Modus funktionieren."""
        model = FeeModel(per_million=30.0, lot_size=100_000, min_fee=0.01)
        volumes = [0.01, 0.1, 1.0]
        prices = [1.1, 1.1, 1.1]

        results = model.calculate_batch(volumes, prices)

        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_slippage_batch_determinism(self, force_python_mode) -> None:
        """Batch-Slippage muss deterministisch sein."""
        model = SlippageModel(fixed_pips=0.5, random_pips=1.0)
        prices = [1.1, 1.2, 1.3]
        directions = ["long", "short", "long"]

        results_1 = model.apply_batch(prices, directions, seed=42)
        results_2 = model.apply_batch(prices, directions, seed=42)

        assert results_1 == results_2, "Batch results should be deterministic"

    def test_batch_length_mismatch_raises(self, force_python_mode) -> None:
        """Batch mit unterschiedlichen Längen muss ValueError werfen."""
        model = SlippageModel(fixed_pips=0.5, random_pips=1.0)
        prices = [1.1, 1.2, 1.3]
        directions = ["long", "short"]  # One less

        with pytest.raises(ValueError, match="same length"):
            model.apply_batch(prices, directions)


# ==============================================================================
# FEATURE FLAG TESTS
# ==============================================================================


class TestFeatureFlag:
    """Tests für Feature-Flag Verhalten."""

    def test_env_var_false_uses_python(self, force_python_mode) -> None:
        """OMEGA_USE_RUST_SLIPPAGE_FEE=false muss Python verwenden."""
        status = get_rust_status()
        assert status["enabled"] is False

    def test_env_var_auto_uses_available(self) -> None:
        """OMEGA_USE_RUST_SLIPPAGE_FEE=auto muss Verfügbarkeit prüfen."""
        os.environ.pop("OMEGA_USE_RUST_SLIPPAGE_FEE", None)  # Remove override
        status = get_rust_status()
        # enabled should match available in auto mode
        if status["available"]:
            assert status["enabled"] is True
        else:
            assert status["enabled"] is False
