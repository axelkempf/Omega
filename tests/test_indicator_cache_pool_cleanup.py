import gc

import pytest

from backtest_engine.core.indicator_cache import (
    clear_indicator_cache_pool,
    get_cached_indicator_cache,
    indicator_cache_pool_size,
)


def _make_minimal_multi_candle_data(n: int = 3):
    # Minimalstruktur: Candle kann dict sein (IndicatorCache unterstützt dict oder Objekt).
    candles = [
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        for _ in range(n)
    ]
    return {"M1": {"bid": candles, "ask": candles}}


def test_indicator_cache_pool_entries_do_not_leak_after_gc():
    """Stellt sicher, dass der globale Pool keine toten Einträge akkumuliert."""

    clear_indicator_cache_pool()
    assert indicator_cache_pool_size() == 0

    multi = _make_minimal_multi_candle_data(5)
    inst = get_cached_indicator_cache(multi)
    assert inst is not None
    assert indicator_cache_pool_size() == 1

    # Referenz lösen und GC erzwingen – WeakValueDictionary sollte den Eintrag entfernen.
    del inst
    gc.collect()

    # Da das Key-Signature auf id(list) basiert, wird ohne WeakValueDictionary
    # der Pool typischerweise wachsen. Mit WeakValueDictionary muss er schrumpfen.
    assert indicator_cache_pool_size() == 0


@pytest.mark.parametrize("rounds", [10])
def test_indicator_cache_pool_stays_small_across_many_creations(rounds: int):
    clear_indicator_cache_pool()
    for _ in range(rounds):
        inst = get_cached_indicator_cache(_make_minimal_multi_candle_data(10))
        del inst
    gc.collect()
    assert indicator_cache_pool_size() == 0
