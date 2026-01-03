# core/currency/symbol_specs_registry.py
import threading
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SymbolSpec:  # kompatibel zu deinem bestehenden Dataclass
    symbol: str
    contract_size: float
    tick_size: float
    tick_value: float
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    base_currency: str = ""
    quote_currency: str = ""
    profit_currency: str = ""
    pip_size: float = 0.0001


class SymbolSpecsRegistry:
    def __init__(self, specs: Dict[str, SymbolSpec]):
        self._specs = specs
        self._lock = threading.RLock()

    def get(self, symbol: str) -> SymbolSpec:
        with self._lock:
            return self._specs[symbol]
