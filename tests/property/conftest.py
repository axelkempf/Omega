# -*- coding: utf-8 -*-
"""
Hypothesis Konfiguration und gemeinsame Strategien für Property-Tests.

Phase 3 Task P3-05: Hypothesis für numerische Korrektheit einrichten
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, Verbosity, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS PROFILE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default Profile für normale Test-Runs
settings.register_profile(
    "default",
    max_examples=100,
    verbosity=Verbosity.normal,
    deadline=None,  # Keine Zeitbegrenzung (numerische Tests können langsam sein)
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)

# CI Profile mit mehr Examples
settings.register_profile(
    "ci",
    max_examples=500,
    verbosity=Verbosity.quiet,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)

# Debug Profile für Fehlersuche
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
)

# Dev Profile für schnelle Iteration
settings.register_profile(
    "dev",
    max_examples=25,
    verbosity=Verbosity.normal,
    deadline=None,
)

# Load default profile
settings.load_profile("default")


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM STRATEGIES FOR TRADING DATA
# ══════════════════════════════════════════════════════════════════════════════


@st.composite
def prices(
    draw: st.DrawFn,
    min_value: float = 0.0001,
    max_value: float = 100_000.0,
) -> float:
    """Strategy für realistische Preise (Forex, Aktien, Crypto)."""
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def price_arrays(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 1000,
    min_value: float = 0.0001,
    max_value: float = 100_000.0,
) -> np.ndarray:
    """Strategy für Arrays von Preisen."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@st.composite
def ohlcv_data(
    draw: st.DrawFn,
    min_size: int = 20,
    max_size: int = 500,
    start_price: float = 1.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """
    Strategy für realistische OHLCV-Daten.

    Generiert konsistente OHLCV-Daten wobei:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Volume > 0
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # Random Walk für Close
    returns = rng.normal(0, volatility, size)
    close = start_price * np.exp(np.cumsum(returns))

    # Open = vorheriger Close (mit small noise)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # High und Low relativ zu Open/Close
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    wick_up = np.abs(rng.normal(0, volatility * 0.5, size)) * body_high
    wick_down = np.abs(rng.normal(0, volatility * 0.5, size)) * body_low

    high = body_high + wick_up
    low = body_low - wick_down
    low = np.maximum(low, 1e-10)  # Niemals negativ

    # Volume: Log-normal
    volume = np.maximum(rng.lognormal(10, 1, size), 1.0)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@st.composite
def indicator_periods(
    draw: st.DrawFn,
    min_period: int = 2,
    max_period: int = 200,
) -> int:
    """Strategy für Indikator-Perioden."""
    return draw(st.integers(min_value=min_period, max_value=max_period))


@st.composite
def percentage_values(
    draw: st.DrawFn,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    """Strategy für Prozent-Werte (0.0 - 1.0)."""
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def trade_returns(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 500,
    mean_r: float = 0.0,
    std_r: float = 2.0,
) -> np.ndarray:
    """
    Strategy für Trade-Returns (R-Multiple).

    Typische Verteilung: Leicht rechtsschief mit fat tails.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # Mixture von Normal und Gelegentlichen Outliern
    base = rng.normal(mean_r, std_r, size)
    outlier_mask = rng.random(size) < 0.05  # 5% Outliers
    outliers = rng.normal(mean_r, std_r * 3, size)
    returns = np.where(outlier_mask, outliers, base)

    return returns.astype(np.float64)


@st.composite
def yearly_profits_dict(
    draw: st.DrawFn,
    min_years: int = 2,
    max_years: int = 10,
    min_profit: float = -100_000.0,
    max_profit: float = 500_000.0,
) -> Dict[int, float]:
    """Strategy für jährliche Profit-Dicts (Stability Score Input)."""
    num_years = draw(st.integers(min_value=min_years, max_value=max_years))
    start_year = draw(st.integers(min_value=2015, max_value=2024))

    profits = {}
    for i in range(num_years):
        year = start_year + i
        profit = draw(
            st.floats(
                min_value=min_profit,
                max_value=max_profit,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        profits[year] = profit

    return profits


@st.composite
def candle_dict(
    draw: st.DrawFn,
    min_price: float = 0.0001,
    max_price: float = 100_000.0,
) -> Dict[str, float]:
    """Strategy für einzelne Candle-Dicts."""
    close = draw(
        st.floats(
            min_value=min_price,
            max_value=max_price,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    open_ = draw(
        st.floats(
            min_value=min_price,
            max_value=max_price,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    body_high = max(open_, close)
    body_low = min(open_, close)

    # High >= body_high, Low <= body_low
    high_extra = draw(st.floats(min_value=0.0, max_value=body_high * 0.1))
    low_extra = draw(st.floats(min_value=0.0, max_value=body_low * 0.1))

    high = body_high + high_extra
    low = max(body_low - low_extra, 1e-10)

    volume = draw(st.floats(min_value=1.0, max_value=1e12))

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


@st.composite
def multi_tf_candle_data(
    draw: st.DrawFn,
    min_candles: int = 50,
    max_candles: int = 300,
    timeframes: List[str] | None = None,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Strategy für Multi-Timeframe Candle-Daten (IndicatorCache Input).
    """
    if timeframes is None:
        timeframes = ["M5", "H1"]

    num_candles = draw(st.integers(min_value=min_candles, max_value=max_candles))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    result: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for tf in timeframes:
        candles: List[Dict[str, float]] = []
        price = 1.0 + rng.random() * 0.5  # Start between 1.0 and 1.5

        for _ in range(num_candles):
            ret = rng.normal(0, 0.005)
            price = max(price * (1 + ret), 0.0001)

            body = price * (1 + rng.normal(0, 0.002))
            open_ = price
            close = body

            high = max(open_, close) * (1 + abs(rng.normal(0, 0.001)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.001)))
            low = max(low, 1e-10)

            candles.append(
                {
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(max(rng.lognormal(10, 1), 1.0)),
                }
            )
            price = close

        result[tf] = {"bid": candles, "ask": candles}

    return result
