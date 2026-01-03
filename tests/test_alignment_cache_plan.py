from datetime import datetime, timedelta, timezone

from backtest_engine.data.candle import Candle
from backtest_engine.runner import _get_or_build_alignment, clear_alignment_cache


def _make_symbol_map(offset: float) -> dict:
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    base_prices = [1.0, 1.1, 1.2]
    bid_seq = [
        Candle(
            timestamp=base_ts + timedelta(minutes=i),
            open=price + offset,
            high=price + offset + 0.05,
            low=price + offset - 0.05,
            close=price + offset + 0.01,
            volume=100 + i,
            candle_type="bid",
        )
        for i, price in enumerate(base_prices)
    ]
    ask_seq = [
        Candle(
            timestamp=base_ts + timedelta(minutes=i),
            open=price + offset + 0.0005,
            high=price + offset + 0.0505,
            low=price + offset - 0.0495,
            close=price + offset + 0.0105,
            volume=120 + i,
            candle_type="ask",
        )
        for i, price in enumerate(base_prices)
    ]
    return {"TEST": {"M1": {"bid": bid_seq, "ask": ask_seq}}}


def test_alignment_cache_uses_current_candles_on_identical_timestamps():
    clear_alignment_cache()
    config = {"timeframes": {"primary": "M1", "additional": []}, "timestamp_alignment": {}}
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

    symbol_map_a = _make_symbol_map(offset=0.0)
    bid_a, ask_a, _ = _get_or_build_alignment(symbol_map_a, "M1", config, start_dt)

    symbol_map_b = _make_symbol_map(offset=0.5)
    bid_b, ask_b, _ = _get_or_build_alignment(symbol_map_b, "M1", config, start_dt)

    base_expected = [1.0, 1.1, 1.2]
    assert [c.open for c in bid_a] == base_expected
    assert [c.open for c in bid_b] == [v + 0.5 for v in base_expected]

    assert [c.open for c in ask_a] == [v + 0.0005 for v in base_expected]
    assert [c.open for c in ask_b] == [v + 0.5005 for v in base_expected]

    # Sicherheit: sicherstellen, dass Cache-Hit nicht alte Candle-Objekte reused
    assert bid_a[0] is not bid_b[0]
    assert ask_a[0] is not ask_b[0]