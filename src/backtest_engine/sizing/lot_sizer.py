"""Lot sizing module for risk-based and fixed position sizing.

This module provides type-safe lot sizing calculations with proper
symbol specification lookups and FX conversion support.
"""

from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Optional

from backtest_engine.sizing.rate_provider import RateProvider
from backtest_engine.sizing.symbol_specs_registry import SymbolSpec, SymbolSpecsRegistry

if TYPE_CHECKING:
    from datetime import datetime


class LotSizer:
    """Risk-based and fixed lot sizing calculator.

    Calculates position sizes based on:
    - Risk-based sizing: Given stop loss distance and risk amount
    - Fixed sizing: Clamp and step-round externally specified lot sizes

    Attributes:
        account: Account currency code (e.g., "USD", "EUR")
        rp: Rate provider for FX conversions
        specs: Symbol specifications registry
    """

    __slots__ = ("account", "rp", "specs")

    def __init__(
        self,
        account_ccy: str,
        rate_provider: RateProvider,
        specs: SymbolSpecsRegistry,
    ) -> None:
        """Initialize lot sizer.

        Args:
            account_ccy: Account currency code (e.g., "USD")
            rate_provider: Provider for FX rate lookups
            specs: Registry containing symbol specifications
        """
        self.account: str = account_ccy
        self.rp: RateProvider = rate_provider
        self.specs: SymbolSpecsRegistry = specs

    def pip_value_acct_per_lot(
        self,
        symbol: str,
        price: float,  # noqa: ARG002 - kept for API compatibility
        t: Optional[datetime] = None,
    ) -> float:
        """Calculate pip value in account currency per standard lot.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            price: Current price (kept for API compatibility, not used)
            t: Optional timestamp for historical FX rate lookup

        Returns:
            Pip value in account currency per 1.0 lot

        Raises:
            ValueError: If symbol not found in specs registry
        """
        spec: Optional[SymbolSpec] = self.specs.get(symbol)
        if spec is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")

        # Pip value in quote currency: contract_size * pip_size
        pv_quote: float = spec.contract_size * spec.pip_size

        # Convert to account currency
        pv_acct, _ = self.rp.fx_convert(
            pv_quote, from_ccy=spec.quote_currency, to_ccy=self.account, t=t
        )
        return float(pv_acct)

    def size_risk_based(
        self,
        symbol: str,
        price: float,  # noqa: ARG002 - kept for API compatibility
        stop_pips: float,
        risk_amount_acct: float,
        t: Optional[datetime] = None,
    ) -> float:
        """Calculate lot size based on risk amount and stop loss distance.

        Uses the formula: lots = risk_amount / (stop_pips * pip_value_per_lot)
        Result is rounded conservatively (floor) to volume step.

        Args:
            symbol: Trading symbol
            price: Current price (kept for API compatibility)
            stop_pips: Stop loss distance in pips
            risk_amount_acct: Maximum risk amount in account currency
            t: Optional timestamp for historical FX rate lookup

        Returns:
            Calculated lot size, clamped to symbol's min/max limits

        Raises:
            ValueError: If symbol not found in specs registry
        """
        spec: Optional[SymbolSpec] = self.specs.get(symbol)
        if spec is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")

        pv: float = self.pip_value_acct_per_lot(symbol, price, t=t)

        # Calculate raw lot size
        raw: float
        if stop_pips <= 0:
            raw = 0.0
        else:
            raw = risk_amount_acct / (stop_pips * pv)

        # Conservative rounding: clamp → floor to step
        vmin: float = spec.volume_min
        step: float = spec.volume_step
        vmax: float = spec.volume_max

        lots: float
        if raw <= vmin:
            lots = vmin
        else:
            n: int = floor((raw - vmin + 1e-12) / step)
            lots = vmin + n * step

        return min(max(lots, vmin), vmax)

    def size_fixed(self, symbol: str, lots: float) -> float:
        """Clamp and step-round externally specified fixed lot sizes.

        Args:
            symbol: Trading symbol
            lots: Desired lot size

        Returns:
            Lot size rounded to valid step and clamped to min/max

        Raises:
            ValueError: If symbol not found in specs registry
        """
        spec: Optional[SymbolSpec] = self.specs.get(symbol)
        if spec is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")

        vmin: float = spec.volume_min
        step: float = spec.volume_step
        vmax: float = spec.volume_max

        # Clamp to valid range
        clamped: float = max(min(float(lots), vmax), vmin)

        if clamped <= vmin:
            return vmin

        # Floor to step
        n: int = floor((clamped - vmin + 1e-12) / step)
        return min(vmin + n * step, vmax)
