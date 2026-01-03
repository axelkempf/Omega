from math import floor

from backtest_engine.sizing.rate_provider import RateProvider
from backtest_engine.sizing.symbol_specs_registry import SymbolSpecsRegistry


class LotSizer:
    def __init__(
        self, account_ccy: str, rate_provider: RateProvider, specs: SymbolSpecsRegistry
    ):
        self.account = account_ccy
        self.rp = rate_provider
        self.specs = specs

    def pip_value_acct_per_lot(self, symbol: str, price: float, t=None) -> float:
        s = self.specs.get(symbol)
        if s is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")
        # Präzise Pip-Value in Quote-CCY: contract_size * pip_size
        pv_quote = s.contract_size * s.pip_size
        # In Account-CCY normalisieren
        pv_acct, _ = self.rp.fx_convert(
            pv_quote, from_ccy=s.quote_currency, to_ccy=self.account, t=t
        )
        return pv_acct

    def size_risk_based(
        self,
        symbol: str,
        price: float,
        stop_pips: float,
        risk_amount_acct: float,
        t=None,
    ) -> float:
        s = self.specs.get(symbol)
        if s is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")
        pv = self.pip_value_acct_per_lot(symbol, price, t=t)
        raw = 0.0 if stop_pips <= 0 else (risk_amount_acct / (stop_pips * pv))
        # konservativ runden: clamp → floor to step
        vmin, step, vmax = s.volume_min, s.volume_step, s.volume_max
        if raw <= vmin:
            lots = vmin
        else:
            n = floor((raw - vmin + 1e-12) / step)
            lots = vmin + n * step
        return min(max(lots, vmin), vmax)

    def size_fixed(self, symbol: str, lots: float) -> float:
        """Clamp & step rounding for externally specified fixed lot sizes."""
        s = self.specs.get(symbol)
        if s is None:
            raise ValueError(f"Symbol-Specs für '{symbol}' nicht gefunden im Registry")
        vmin, step, vmax = s.volume_min, s.volume_step, s.volume_max
        lots = max(min(float(lots), vmax), vmin)
        if lots <= vmin:
            return vmin
        n = floor((lots - vmin + 1e-12) / step)
        return min(vmin + n * step, vmax)
