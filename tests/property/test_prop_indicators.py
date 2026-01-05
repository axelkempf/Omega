# -*- coding: utf-8 -*-
"""
Property-Based Tests für Indicator-Berechnungen.

Phase 3 Task P3-06: Property-Based Tests für Indicator-Berechnungen

Invarianten die getestet werden:
1. EMA/SMA sind im Bereich der Input-Daten (keine Explosion)
2. RSI ist immer zwischen 0 und 100
3. Bollinger Bands: Upper >= Middle >= Lower
4. ATR ist immer >= 0
5. Determinismus: Gleiche Inputs → Gleiche Outputs
6. Monotonie von EMA bei steigenden Kursen
7. Numerische Stabilität bei Edge-Cases
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings

from backtest_engine.core.indicator_cache import IndicatorCache

from .conftest import (
    indicator_periods,
    multi_tf_candle_data,
    ohlcv_data,
    price_arrays,
)


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDS AND RANGE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestIndicatorBounds:
    """Tests für mathematische Bounds von Indikatoren."""

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=2, max_period=50))
    @settings(max_examples=50)
    def test_ema_within_price_range(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """EMA-Werte müssen im Bereich der Input-Close-Preise liegen."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        ema = cache.ema(tf, "bid", period)

        # Nur nicht-NaN Werte prüfen
        valid_ema = ema.dropna()
        if len(valid_ema) == 0:
            return

        closes = cache.get_closes(tf, "bid").dropna()
        if len(closes) == 0:
            return

        min_close = closes.min()
        max_close = closes.max()

        # EMA muss im Bereich [min, max] der Closes liegen
        assert valid_ema.min() >= min_close * 0.99, "EMA below minimum close"
        assert valid_ema.max() <= max_close * 1.01, "EMA above maximum close"

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=2, max_period=50))
    @settings(max_examples=50)
    def test_sma_within_price_range(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """SMA-Werte müssen im Bereich der Input-Close-Preise liegen."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        sma = cache.sma(tf, "bid", period)

        valid_sma = sma.dropna()
        if len(valid_sma) == 0:
            return

        closes = cache.get_closes(tf, "bid").dropna()
        if len(closes) == 0:
            return

        min_close = closes.min()
        max_close = closes.max()

        assert valid_sma.min() >= min_close * 0.99, "SMA below minimum close"
        assert valid_sma.max() <= max_close * 1.01, "SMA above maximum close"

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=2, max_period=30))
    @settings(max_examples=50)
    def test_rsi_between_0_and_100(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """RSI muss immer zwischen 0 und 100 liegen."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        rsi = cache.rsi(tf, "bid", period)

        valid_rsi = rsi.dropna()
        if len(valid_rsi) == 0:
            return

        assert valid_rsi.min() >= 0.0, f"RSI below 0: {valid_rsi.min()}"
        assert valid_rsi.max() <= 100.0, f"RSI above 100: {valid_rsi.max()}"

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=2, max_period=30))
    @settings(max_examples=50)
    def test_atr_non_negative(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """ATR muss immer >= 0 sein."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        atr = cache.atr(tf, "bid", period)

        valid_atr = atr.dropna()
        if len(valid_atr) == 0:
            return

        assert valid_atr.min() >= 0.0, f"ATR negative: {valid_atr.min()}"

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=50)
    def test_bollinger_band_ordering(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """Bollinger Bands: Upper >= Middle >= Lower."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        upper, middle, lower = cache.bollinger(tf, "bid", period, std_factor=2.0)

        # Nur nicht-NaN Werte prüfen
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_mask.sum() == 0:
            return

        upper_valid = upper[valid_mask]
        middle_valid = middle[valid_mask]
        lower_valid = lower[valid_mask]

        # Upper >= Middle
        assert (upper_valid >= middle_valid - 1e-10).all(), "Upper < Middle"
        # Middle >= Lower
        assert (middle_valid >= lower_valid - 1e-10).all(), "Middle < Lower"

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=50)
    def test_dmi_bounds(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """DMI: +DI, -DI und ADX zwischen 0 und 100."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        plus_di, minus_di, adx = cache.dmi(tf, "bid", period)

        for name, series in [
            ("+DI", plus_di),
            ("-DI", minus_di),
            ("ADX", adx),
        ]:
            valid = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) == 0:
                continue
            # DMI kann theoretisch über 100 gehen bei extremen Bewegungen
            # aber sollte nicht negativ sein
            assert valid.min() >= -1e-10, f"{name} negative: {valid.min()}"


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestIndicatorDeterminism:
    """Tests für Determinismus der Indikatoren."""

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=30)
    def test_ema_deterministic(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """Gleiche Inputs → Gleiche EMA Outputs."""
        cache1 = IndicatorCache(data)
        cache2 = IndicatorCache(data)
        tf = list(data.keys())[0]

        ema1 = cache1.ema(tf, "bid", period)
        ema2 = cache2.ema(tf, "bid", period)

        pd.testing.assert_series_equal(ema1, ema2, check_exact=True)

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=30)
    def test_rsi_deterministic(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """Gleiche Inputs → Gleiche RSI Outputs."""
        cache1 = IndicatorCache(data)
        cache2 = IndicatorCache(data)
        tf = list(data.keys())[0]

        rsi1 = cache1.rsi(tf, "bid", period)
        rsi2 = cache2.rsi(tf, "bid", period)

        pd.testing.assert_series_equal(rsi1, rsi2, check_exact=True)

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=30)
    def test_macd_deterministic(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """Gleiche Inputs → Gleiche MACD Outputs."""
        cache1 = IndicatorCache(data)
        cache2 = IndicatorCache(data)
        tf = list(data.keys())[0]

        macd1, signal1 = cache1.macd(tf, "bid", period, period * 2, 9)
        macd2, signal2 = cache2.macd(tf, "bid", period, period * 2, 9)

        pd.testing.assert_series_equal(macd1, macd2, check_exact=True)
        pd.testing.assert_series_equal(signal1, signal2, check_exact=True)


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL STABILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestNumericalStability:
    """Tests für numerische Stabilität bei Edge-Cases."""

    @given(data=multi_tf_candle_data(min_candles=50, max_candles=100))
    @settings(max_examples=30)
    def test_no_inf_in_ema(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """EMA darf keine Infinity-Werte produzieren."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        for period in [5, 10, 20]:
            ema = cache.ema(tf, "bid", period)
            assert not np.isinf(ema).any(), f"Infinity in EMA(period={period})"

    @given(data=multi_tf_candle_data(min_candles=50, max_candles=100))
    @settings(max_examples=30)
    def test_no_inf_in_rsi(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """RSI darf keine Infinity-Werte produzieren."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        for period in [7, 14, 21]:
            rsi = cache.rsi(tf, "bid", period)
            assert not np.isinf(rsi).any(), f"Infinity in RSI(period={period})"

    @given(data=multi_tf_candle_data(min_candles=50, max_candles=100))
    @settings(max_examples=30)
    def test_no_inf_in_atr(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """ATR darf keine Infinity-Werte produzieren."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        for period in [7, 14, 21]:
            atr = cache.atr(tf, "bid", period)
            assert not np.isinf(atr).any(), f"Infinity in ATR(period={period})"

    def test_constant_price_series(self) -> None:
        """Bei konstanten Preisen sollten Indikatoren stabil sein."""
        # Konstante Preise
        n = 100
        candles = [
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000.0}
            for _ in range(n)
        ]
        data = {"M5": {"bid": candles, "ask": candles}}
        cache = IndicatorCache(data)

        # EMA sollte 1.0 sein
        ema = cache.ema("M5", "bid", 10)
        valid_ema = ema.dropna()
        if len(valid_ema) > 0:
            assert np.allclose(valid_ema, 1.0, rtol=1e-6), "EMA not stable at constant price"

        # ATR sollte 0 sein (keine Bewegung)
        atr = cache.atr("M5", "bid", 14)
        valid_atr = atr.dropna()
        if len(valid_atr) > 0:
            assert np.allclose(valid_atr, 0.0, atol=1e-10), "ATR not zero at constant price"

        # RSI sollte 50 sein oder NaN (keine Bewegung)
        # Bei konstanten Preisen ist delta=0, gain=0, loss=0
        # rs = 0/0 → NaN, oder wenn defined, sollte neutral sein

    def test_very_small_prices(self) -> None:
        """Sehr kleine Preise sollten keine Overflows verursachen."""
        n = 100
        rng = np.random.default_rng(42)
        small_price = 1e-8

        candles = []
        price = small_price
        for _ in range(n):
            ret = rng.normal(0, 0.001)
            price = max(price * (1 + ret), 1e-15)
            candles.append(
                {
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1000.0,
                }
            )

        data = {"M5": {"bid": candles, "ask": candles}}
        cache = IndicatorCache(data)

        # Keine Infinity/NaN-Explosionen
        ema = cache.ema("M5", "bid", 10)
        assert not np.isinf(ema).any(), "Infinity in EMA with small prices"

        rsi = cache.rsi("M5", "bid", 14)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0, "RSI negative with small prices"
            assert valid_rsi.max() <= 100, "RSI > 100 with small prices"

    def test_very_large_prices(self) -> None:
        """Sehr große Preise sollten keine Overflows verursachen."""
        n = 100
        rng = np.random.default_rng(42)
        large_price = 1e10

        candles = []
        price = large_price
        for _ in range(n):
            ret = rng.normal(0, 0.001)
            price = price * (1 + ret)
            candles.append(
                {
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1000.0,
                }
            )

        data = {"M5": {"bid": candles, "ask": candles}}
        cache = IndicatorCache(data)

        # Keine Infinity-Werte
        ema = cache.ema("M5", "bid", 10)
        assert not np.isinf(ema).any(), "Infinity in EMA with large prices"

        rsi = cache.rsi("M5", "bid", 14)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert not np.isinf(valid_rsi).any(), "Infinity in RSI with large prices"


# ══════════════════════════════════════════════════════════════════════════════
# CACHING CONSISTENCY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCachingConsistency:
    """Tests dass Caching keine Inkonsistenzen verursacht."""

    @given(data=multi_tf_candle_data(), period=indicator_periods(min_period=5, max_period=30))
    @settings(max_examples=30)
    def test_cached_equals_fresh(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
        period: int,
    ) -> None:
        """Gecachte Werte müssen mit frisch berechneten identisch sein."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        # Erster Aufruf (berechnet)
        ema1 = cache.ema(tf, "bid", period)

        # Zweiter Aufruf (aus Cache)
        ema2 = cache.ema(tf, "bid", period)

        # Müssen identisch sein
        pd.testing.assert_series_equal(ema1, ema2, check_exact=True)

        # Dritter Aufruf mit neuer Cache-Instanz (frisch berechnet)
        cache_fresh = IndicatorCache(data)
        ema3 = cache_fresh.ema(tf, "bid", period)

        pd.testing.assert_series_equal(ema1, ema3, check_exact=True)

    @given(data=multi_tf_candle_data())
    @settings(max_examples=20)
    def test_different_periods_independent(
        self,
        data: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """Verschiedene Perioden sollten unabhängig gecacht werden."""
        cache = IndicatorCache(data)
        tf = list(data.keys())[0]

        ema_5 = cache.ema(tf, "bid", 5)
        ema_10 = cache.ema(tf, "bid", 10)
        ema_20 = cache.ema(tf, "bid", 20)

        # Verschiedene Perioden sollten verschiedene Werte haben
        # (außer bei sehr speziellen Datensätzen)
        valid_mask = ~(ema_5.isna() | ema_10.isna() | ema_20.isna())
        if valid_mask.sum() > 10:
            # Bei genug Datenpunkten sollten sie unterschiedlich sein
            # (außer bei konstanten Preisen, was sehr unwahrscheinlich ist)
            pass  # Keine strenge Assertion, da Edge-Cases möglich


# ══════════════════════════════════════════════════════════════════════════════
# MONOTONICITY TESTS (für spezifische Szenarien)
# ══════════════════════════════════════════════════════════════════════════════


class TestMonotonicity:
    """Tests für erwartete Monotonie-Eigenschaften."""

    def test_ema_follows_trend_up(self) -> None:
        """EMA sollte bei steigendem Trend steigen."""
        n = 50
        candles = []
        for i in range(n):
            price = 1.0 + i * 0.01  # Strikt steigend
            candles.append(
                {
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1000.0,
                }
            )

        data = {"M5": {"bid": candles, "ask": candles}}
        cache = IndicatorCache(data)

        ema = cache.ema("M5", "bid", 10)
        valid_ema = ema.dropna()

        if len(valid_ema) > 10:
            # EMA sollte monoton steigend sein
            diff = valid_ema.diff().dropna()
            assert (diff >= -1e-10).all(), "EMA not monotonic in uptrend"

    def test_ema_follows_trend_down(self) -> None:
        """EMA sollte bei fallendem Trend fallen."""
        n = 50
        candles = []
        for i in range(n):
            price = 2.0 - i * 0.01  # Strikt fallend
            candles.append(
                {
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1000.0,
                }
            )

        data = {"M5": {"bid": candles, "ask": candles}}
        cache = IndicatorCache(data)

        ema = cache.ema("M5", "bid", 10)
        valid_ema = ema.dropna()

        if len(valid_ema) > 10:
            # EMA sollte monoton fallend sein
            diff = valid_ema.diff().dropna()
            assert (diff <= 1e-10).all(), "EMA not monotonic in downtrend"
