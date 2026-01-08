"""
Slippage and Fee models for backtest execution simulation.

This module provides cost models for realistic trade execution:
- SlippageModel: Calculates execution price with fixed + random slippage
- FeeModel: Calculates trading fees based on notional value

Wave 0 Migration: Both classes support Rust FFI acceleration via the
OMEGA_USE_RUST_SLIPPAGE_FEE environment variable. Set to "false" to
force Python implementation for debugging or fallback.

Reference: docs/WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from typing import List

# =============================================================================
# Rust FFI Feature Flag
# =============================================================================
# Auto-detect Rust module availability with override via environment variable
# OMEGA_USE_RUST_SLIPPAGE_FEE=true|false|auto (default: auto)

_RUST_AVAILABLE: bool = False
_RUST_MODULE: object = None


def _check_rust_available() -> bool:
    """Check if Rust module is available and functional."""
    global _RUST_MODULE
    try:
        import omega_rust  # type: ignore[import-not-found]

        # Verify required functions exist
        if hasattr(omega_rust, "calculate_slippage") and hasattr(
            omega_rust, "calculate_fee"
        ):
            _RUST_MODULE = omega_rust
            return True
    except ImportError:
        pass
    return False


def _should_use_rust() -> bool:
    """Determine if Rust implementation should be used."""
    env_val = os.environ.get("OMEGA_USE_RUST_SLIPPAGE_FEE", "auto").lower()
    if env_val == "false":
        return False
    if env_val == "true":
        return _RUST_AVAILABLE
    # auto: use Rust if available
    return _RUST_AVAILABLE


# Initialize on module load
_RUST_AVAILABLE = _check_rust_available()


def get_rust_status() -> dict[str, bool | str]:
    """Get current Rust FFI status for debugging.

    Returns:
        Dict with 'available', 'enabled', and 'reason' keys.
    """
    enabled = _should_use_rust()
    env_val = os.environ.get("OMEGA_USE_RUST_SLIPPAGE_FEE", "auto").lower()
    if not _RUST_AVAILABLE:
        reason = "Rust module not installed"
    elif env_val == "false":
        reason = "Disabled via OMEGA_USE_RUST_SLIPPAGE_FEE=false"
    elif enabled:
        reason = "Rust module active"
    else:
        reason = "Unknown"
    return {
        "available": _RUST_AVAILABLE,
        "enabled": enabled,
        "reason": reason,
    }


# =============================================================================
# Direction Constants (for Rust FFI)
# =============================================================================
DIRECTION_LONG: int = 1
DIRECTION_SHORT: int = -1


def _direction_to_int(direction: str) -> int:
    """Convert direction string to integer for Rust FFI."""
    if direction == "long":
        return DIRECTION_LONG
    return DIRECTION_SHORT


class SlippageModel:
    """
    Modelliert Slippage bei Orderausführungen im Backtest.

    Supports both Python and Rust implementations for performance comparison.
    Use OMEGA_USE_RUST_SLIPPAGE_FEE environment variable to control backend.

    Args:
        fixed_pips: Feste Slippage in Pips je Trade.
        random_pips: Maximale zusätzliche, zufällige Slippage in Pips (uniform [0, random_pips]).
    """

    def __init__(self, fixed_pips: float = 0.0, random_pips: float = 0.0):
        self.fixed_pips = float(fixed_pips)
        self.random_pips = float(random_pips)

    def apply(
        self,
        price: float,
        direction: str,
        pip_size: float = 0.0001,
        seed: Optional[int] = None,
    ) -> float:
        """
        Berechnet Ausführungspreis mit Slippage.

        Args:
            price: Ursprünglicher Orderpreis.
            direction: "long" oder "short".
            pip_size: Preisinkrement pro Pip (z. B. 0.0001 FX, 0.01 JPY/Metalle/CFDs).
            seed: Optional seed for deterministic random slippage (for testing/replay).

        Returns:
            Preis inkl. Slippage.
        """
        if _should_use_rust() and _RUST_MODULE is not None:
            return self._apply_rust(price, direction, pip_size, seed)
        return self._apply_python(price, direction, pip_size, seed)

    def _apply_python(
        self,
        price: float,
        direction: str,
        pip_size: float,
        seed: Optional[int] = None,
    ) -> float:
        """Pure Python implementation of slippage calculation."""
        if seed is not None:
            # Use deterministic random for testing
            rng = random.Random(seed)
            slippage = self.fixed_pips + rng.uniform(0, self.random_pips)
        else:
            slippage = self.fixed_pips + random.uniform(0, self.random_pips)

        if direction == "long":
            return price + slippage * pip_size
        return price - slippage * pip_size

    def _apply_rust(
        self,
        price: float,
        direction: str,
        pip_size: float,
        seed: Optional[int] = None,
    ) -> float:
        """Rust FFI implementation of slippage calculation."""
        direction_int = _direction_to_int(direction)
        return _RUST_MODULE.calculate_slippage(  # type: ignore[union-attr]
            price=price,
            direction=direction_int,
            pip_size=pip_size,
            fixed_pips=self.fixed_pips,
            random_pips=self.random_pips,
            seed=seed,
        )

    def apply_batch(
        self,
        prices: Sequence[float],
        directions: Sequence[str],
        pip_size: float = 0.0001,
        seed: Optional[int] = None,
    ) -> List[float]:
        """
        Batch slippage calculation for optimizer scenarios.

        Each trade gets a unique seed derived from base_seed + index for determinism.

        Args:
            prices: List of original prices.
            directions: List of directions ("long" or "short").
            pip_size: Price increment per pip.
            seed: Base seed for deterministic results.

        Returns:
            List of adjusted prices after slippage.
        """
        if len(prices) != len(directions):
            raise ValueError(
                f"prices and directions must have same length: {len(prices)} vs {len(directions)}"
            )

        if _should_use_rust() and _RUST_MODULE is not None:
            direction_ints = [_direction_to_int(d) for d in directions]
            return list(
                _RUST_MODULE.calculate_slippage_batch(  # type: ignore[union-attr]
                    prices=list(prices),
                    directions=direction_ints,
                    pip_size=pip_size,
                    fixed_pips=self.fixed_pips,
                    random_pips=self.random_pips,
                    seed=seed,
                )
            )

        # Python fallback with per-trade seeding
        results: List[float] = []
        base_seed = seed if seed is not None else 42
        for i, (p, d) in enumerate(zip(prices, directions)):
            trade_seed = base_seed + i
            results.append(self._apply_python(p, d, pip_size, trade_seed))
        return results


class FeeModel:
    """
    Modelliert Handelsgebühren im Backtest.

    Supports both Python and Rust implementations for performance comparison.
    Use OMEGA_USE_RUST_SLIPPAGE_FEE environment variable to control backend.

    Args:
        per_million: Kommission pro 1 Mio Notional in Quote-CCY.
        lot_size: Fallback-Contract-Size (FX=100k), falls keine symbol-spezifische Größe übergeben wird.
        min_fee: Optionale Mindestgebühr pro Transaktion.
    """

    def __init__(
        self,
        per_million: float = 5.0,
        lot_size: float = 100_000.0,
        min_fee: float = 0.0,
    ):
        self.per_million = float(per_million)
        self.lot_size = float(lot_size)
        self.min_fee = float(min_fee)

    def calculate(
        self, volume_lots: float, price: float, contract_size: Optional[float] = None
    ) -> float:
        """
        Berechnet die anfallenden Gebühren für das Volumen.

        Args:
            volume_lots: Volumen in Lots.
            price: Preis pro Einheit (Quote-CCY).
            contract_size: Contract-Size pro Lot (falls vorhanden symbol-spezifisch).

        Returns:
            Absoluter Gebührenbetrag (gleiche Währung wie Preis).
        """
        cs = float(contract_size) if contract_size else self.lot_size

        if _should_use_rust() and _RUST_MODULE is not None:
            return self._calculate_rust(volume_lots, price, cs)
        return self._calculate_python(volume_lots, price, cs)

    def _calculate_python(
        self, volume_lots: float, price: float, contract_size: float
    ) -> float:
        """Pure Python implementation of fee calculation."""
        notional = float(volume_lots) * contract_size * float(price)
        fee = (notional / 1_000_000.0) * self.per_million
        if self.min_fee > 0.0:
            fee = max(fee, self.min_fee)
        return float(fee)

    def _calculate_rust(
        self, volume_lots: float, price: float, contract_size: float
    ) -> float:
        """Rust FFI implementation of fee calculation."""
        return _RUST_MODULE.calculate_fee(  # type: ignore[union-attr]
            volume_lots=volume_lots,
            price=price,
            contract_size=contract_size,
            per_million=self.per_million,
            min_fee=self.min_fee,
        )

    def calculate_batch(
        self,
        volume_lots: Sequence[float],
        prices: Sequence[float],
        contract_size: Optional[float] = None,
    ) -> List[float]:
        """
        Batch fee calculation for optimizer scenarios.

        Args:
            volume_lots: List of trade volumes.
            prices: List of prices.
            contract_size: Contract size per lot (same for all trades).

        Returns:
            List of fee amounts.
        """
        if len(volume_lots) != len(prices):
            raise ValueError(
                f"volume_lots and prices must have same length: {len(volume_lots)} vs {len(prices)}"
            )

        cs = float(contract_size) if contract_size else self.lot_size

        if _should_use_rust() and _RUST_MODULE is not None:
            return list(
                _RUST_MODULE.calculate_fee_batch(  # type: ignore[union-attr]
                    volume_lots=list(volume_lots),
                    prices=list(prices),
                    contract_size=cs,
                    per_million=self.per_million,
                    min_fee=self.min_fee,
                )
            )

        # Python fallback
        return [self._calculate_python(v, p, cs) for v, p in zip(volume_lots, prices)]

