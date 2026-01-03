from enum import Enum
from typing import Any, Dict

from backtest_engine.sizing.rate_provider import RateProvider
from backtest_engine.sizing.symbol_specs_registry import SymbolSpecsRegistry


class Side(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    BOTH = "both"


class CommissionModel:
    def __init__(
        self,
        account_ccy: str,
        rate_provider: RateProvider,
        exec_costs: dict,
        specs: SymbolSpecsRegistry,
        *,
        multiplier: float = 1.0,
    ):
        self.account = account_ccy
        self.rp = rate_provider
        self.cfg = exec_costs  # defaults + per_symbol overrides
        self.specs = specs
        try:
            m = float(multiplier)
        except Exception:
            m = 1.0
        # Guard: negative multipliers are nonsensical; clamp to 0.0.
        self.multiplier = max(0.0, m)

    def _cfg_for(self, symbol: str) -> dict:
        defaults: Dict[str, Any] = dict(self.cfg.get("defaults", {}))
        per_sym = dict(self.cfg.get("per_symbol", {})).get(symbol, {})
        out = {**defaults, **per_sym}
        # normalize fields
        out.setdefault("schema", "per_million_notional")
        out.setdefault("side", "both")
        out.setdefault("fee_ccy", self.account)
        # numeric normalization
        for k in ("per_lot", "rate_per_million", "pct", "min_fee"):
            if k in out and out[k] is not None:
                out[k] = float(out[k])
        return out

    def fee_for_order(
        self, symbol: str, lots: float, price: float, t=None, side: Side = Side.ENTRY
    ) -> float:
        """Gibt die **Fee in Account-CCY** zurück (inkl. min_fee, side-Logik, Currency-Konvertierung)."""
        c = self._cfg_for(
            symbol
        )  # {schema, per_lot|rate_per_million|pct, fee_ccy, min_fee, side}
        cfg_side = c.get("side", "both").lower()
        # Standard: nichts berechnen
        eff_sides = 0

        if cfg_side == "both":
            # Bei Entry ODER Exit genau EINE Seite berechnen
            eff_sides = 1 if side in (Side.ENTRY, Side.EXIT) else 2
        elif cfg_side == "entry":
            eff_sides = 1 if side == Side.ENTRY else 0
        elif cfg_side == "exit":
            eff_sides = 1 if side == Side.EXIT else 0
        else:
            raise ValueError(f"Unknown cfg side: {cfg_side}")

        fee_ccy = c.get("fee_ccy", self.account)
        s = self.specs.get(symbol)

        # Notional in fee_ccy (falls benötigt)
        notional_quote = (
            float(price) * float(s.contract_size) * float(lots)
        )  # quote ccy
        notional_fee_ccy, _ = self.rp.fx_convert(
            notional_quote, from_ccy=s.quote_currency, to_ccy=fee_ccy, t=t
        )

        schema = c.get("schema", "per_million_notional").lower()
        if schema == "per_lot":
            base_fee_fee_ccy = float(c.get("per_lot", 0.0)) * float(lots)
        elif schema == "per_million_notional":
            base_fee_fee_ccy = float(c.get("rate_per_million", 0.0)) * (
                notional_fee_ccy / 1_000_000.0
            )
        elif schema == "percent_of_notional":
            base_fee_fee_ccy = float(c.get("pct", 0.0)) * notional_fee_ccy
        else:
            raise ValueError(f"Unknown fee schema: {schema}")

        # Apply global multiplier (e.g. robustness cost shock) to both base fee and min_fee.
        mult = float(self.multiplier)
        base_fee_fee_ccy *= mult
        min_fee = float(c.get("min_fee", 0.0)) * mult
        per_side_fee_fee_ccy = max(base_fee_fee_ccy, min_fee)
        total_fee_fee_ccy = per_side_fee_fee_ccy * eff_sides

        fee_acct, _ = self.rp.fx_convert(
            total_fee_fee_ccy, from_ccy=fee_ccy, to_ccy=self.account, t=t
        )
        # round to cents (configurable in future)
        return round(float(fee_acct), 2)
