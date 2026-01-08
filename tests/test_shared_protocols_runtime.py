from __future__ import annotations

from backtest_engine.core.indicator_cache import IndicatorCache
from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from shared.protocols import (
    IndicatorCacheProtocol,
    MultiSymbolSliceProtocol,
    SymbolDataSliceProtocol,
)


def test_indicator_cache_is_instance_of_protocol() -> None:
    multi_candle_data = {
        "M1": {
            "bid": [
                {
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                }
            ],
            "ask": [
                {
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                }
            ],
        }
    }

    ind = IndicatorCache(multi_candle_data)

    assert isinstance(ind, IndicatorCacheProtocol)


def test_symbol_data_slice_is_instance_of_protocol() -> None:
    multi_candle_data = {
        "M1": {
            "bid": [
                {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
            ],
            "ask": [
                {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
            ],
        }
    }

    ind = IndicatorCache(multi_candle_data)
    s = SymbolDataSlice(
        multi_candle_data=multi_candle_data, index=0, indicator_cache=ind
    )

    assert isinstance(s, SymbolDataSliceProtocol)


def test_multi_symbol_slice_is_instance_of_protocol() -> None:
    ts = 123
    candle = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}

    ms = MultiSymbolSlice(
        candle_lookups={"EURUSD": {"bid": {ts: candle}, "ask": {ts: candle}}},
        timestamp=ts,
        primary_tf="M1",
    )

    assert isinstance(ms, MultiSymbolSliceProtocol)
