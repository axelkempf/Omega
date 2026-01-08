# -*- coding: utf-8 -*-
"""
Pytest-Benchmark Fixtures für Performance-Tests.

Phase 3 Task P3-01: pytest-benchmark Setup
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default-Größen für synthetische Testdaten
DEFAULT_CANDLE_COUNT = 10_000  # Typische Backtest-Größe
LARGE_CANDLE_COUNT = 100_000  # Stress-Test Größe
SMALL_CANDLE_COUNT = 1_000  # Schnelle Iterationen

# Seed für Reproduzierbarkeit
BENCHMARK_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def generate_synthetic_ohlcv(
    n: int = DEFAULT_CANDLE_COUNT,
    *,
    seed: int = BENCHMARK_SEED,
    start_price: float = 1.1000,
    volatility: float = 0.0002,
) -> pd.DataFrame:
    """
    Generiert synthetische OHLCV-Daten für Benchmarks.

    Args:
        n: Anzahl der Kerzen
        seed: Random seed für Reproduzierbarkeit
        start_price: Startpreis
        volatility: Preisvolatilität pro Schritt

    Returns:
        DataFrame mit columns: open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)

    # Random Walk für Close-Preise
    returns = rng.normal(0, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    # OHLC aus Close ableiten (realistischere Struktur)
    high_offset = np.abs(rng.normal(0, volatility * 0.5, n))
    low_offset = np.abs(rng.normal(0, volatility * 0.5, n))

    high = close * (1 + high_offset)
    low = close * (1 - low_offset)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # Volume: Log-normal verteilt
    volume = rng.lognormal(10, 1, n).astype(np.float64)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_synthetic_candle_list(
    n: int = DEFAULT_CANDLE_COUNT,
    *,
    seed: int = BENCHMARK_SEED,
) -> List[Dict[str, float]]:
    """
    Generiert Liste von Candle-Dicts (wie von DataHandler geliefert).
    """
    df = generate_synthetic_ohlcv(n, seed=seed)
    candles = []
    for i in range(len(df)):
        candles.append(
            {
                "open": float(df.iloc[i]["open"]),
                "high": float(df.iloc[i]["high"]),
                "low": float(df.iloc[i]["low"]),
                "close": float(df.iloc[i]["close"]),
                "volume": float(df.iloc[i]["volume"]),
            }
        )
    return candles


def generate_multi_tf_candle_data(
    n_primary: int = DEFAULT_CANDLE_COUNT,
    *,
    seed: int = BENCHMARK_SEED,
    timeframes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Generiert Multi-Timeframe Candle-Daten für IndicatorCache.

    Args:
        n_primary: Anzahl der Primary-TF Kerzen
        seed: Random seed
        timeframes: Liste der Timeframes (default: ["M5", "H1", "D1"])

    Returns:
        Dict[tf][side] -> List[Candle|None]
    """
    if timeframes is None:
        timeframes = ["M5", "H1", "D1"]

    result: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for i, tf in enumerate(timeframes):
        # Verschiedene Seeds pro Timeframe für Variation
        tf_seed = seed + i * 1000
        candles = generate_synthetic_candle_list(n_primary, seed=tf_seed)
        result[tf] = {
            "bid": candles,
            "ask": candles,  # Vereinfacht: ask = bid für Benchmarks
        }

    return result


def generate_synthetic_trades_df(
    n_trades: int = 500,
    *,
    seed: int = BENCHMARK_SEED,
    avg_r: float = 0.5,
    std_r: float = 1.5,
) -> pd.DataFrame:
    """
    Generiert synthetisches Trades-DataFrame für Rating-Benchmarks.

    Args:
        n_trades: Anzahl der Trades
        seed: Random seed
        avg_r: Durchschnittliches R-Multiple
        std_r: Standardabweichung R-Multiple

    Returns:
        DataFrame mit columns: result, r_multiple, exit_time
    """
    rng = np.random.default_rng(seed)

    r_multiples = rng.normal(avg_r, std_r, n_trades)
    results = r_multiples * 100  # Annahme: 100€ Risiko pro Trade

    # Sortierte Exit-Zeiten
    base_time = pd.Timestamp("2024-01-01")
    exit_times = [base_time + pd.Timedelta(hours=i * 4) for i in range(n_trades)]

    return pd.DataFrame(
        {
            "result": results,
            "r_multiple": r_multiples,
            "exit_time": exit_times,
        }
    )


def generate_base_metrics(
    *,
    seed: int = BENCHMARK_SEED,
    profit: float = 5000.0,
    drawdown: float = 1500.0,
    sharpe: float = 1.2,
) -> Dict[str, float]:
    """
    Generiert Base-Metrics für Rating-Score-Berechnungen.
    """
    return {
        "profit": profit,
        "drawdown": drawdown,
        "sharpe": sharpe,
        "avg_r": 0.35,
        "winrate": 0.55,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def benchmark_seed() -> int:
    """Fester Seed für alle Benchmarks."""
    return BENCHMARK_SEED


@pytest.fixture(scope="session")
def synthetic_ohlcv_small() -> pd.DataFrame:
    """Kleine OHLCV-Daten (1K Kerzen)."""
    return generate_synthetic_ohlcv(SMALL_CANDLE_COUNT)


@pytest.fixture(scope="session")
def synthetic_ohlcv_medium() -> pd.DataFrame:
    """Medium OHLCV-Daten (10K Kerzen)."""
    return generate_synthetic_ohlcv(DEFAULT_CANDLE_COUNT)


@pytest.fixture(scope="session")
def synthetic_ohlcv_large() -> pd.DataFrame:
    """Große OHLCV-Daten (100K Kerzen)."""
    return generate_synthetic_ohlcv(LARGE_CANDLE_COUNT)


@pytest.fixture(scope="session")
def multi_tf_data_small() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Multi-TF Daten für IndicatorCache (1K Kerzen)."""
    return generate_multi_tf_candle_data(SMALL_CANDLE_COUNT)


@pytest.fixture(scope="session")
def multi_tf_data_medium() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Multi-TF Daten für IndicatorCache (10K Kerzen)."""
    return generate_multi_tf_candle_data(DEFAULT_CANDLE_COUNT)


@pytest.fixture(scope="session")
def multi_tf_data_large() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Multi-TF Daten für IndicatorCache (100K Kerzen)."""
    return generate_multi_tf_candle_data(LARGE_CANDLE_COUNT)


@pytest.fixture(scope="session")
def synthetic_trades_small() -> pd.DataFrame:
    """Kleines Trades-DataFrame (100 Trades)."""
    return generate_synthetic_trades_df(100)


@pytest.fixture(scope="session")
def synthetic_trades_medium() -> pd.DataFrame:
    """Medium Trades-DataFrame (500 Trades)."""
    return generate_synthetic_trades_df(500)


@pytest.fixture(scope="session")
def synthetic_trades_large() -> pd.DataFrame:
    """Großes Trades-DataFrame (2000 Trades)."""
    return generate_synthetic_trades_df(2000)


@pytest.fixture(scope="session")
def base_metrics_fixture() -> Dict[str, float]:
    """Standard Base-Metrics für Rating-Tests."""
    return generate_base_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RESULT EXPORT
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def benchmark_output_dir() -> Path:
    """Erstellt Output-Verzeichnis für Benchmark-Ergebnisse."""
    output_dir = Path("var/results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def pytest_benchmark_generate_commit_info(config: Any) -> Dict[str, Any]:
    """
    Pytest-Benchmark-Hook zum Anreichern der Benchmark-Ergebnisse mit Metadaten.

    Dieser Hook wird von pytest-benchmark während der Ergebnis-Erfassung aufgerufen
    und soll ein JSON-serialisierbares Dict zurückgeben, das zusammen mit den
    Benchmark-Resultaten gespeichert wird (z.B. in JSON-/CSV-Exports).

    In diesem Projekt hängen wir hier statische Projektinformationen sowie einen
    Zeitstempel an, um Benchmark-Runs später eindeutig zuordnen und vergleichen
    zu können.
    """
    return {
        "project": "omega",
        "phase": "P3-migration-prep",
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM MARKERS
# ══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config: Any) -> None:
    """Registriert Custom Markers für Benchmarks."""
    config.addinivalue_line(
        "markers",
        "benchmark_indicator: Benchmarks für IndicatorCache Funktionen",
    )
    config.addinivalue_line(
        "markers",
        "benchmark_event_engine: Benchmarks für EventEngine Throughput/Latenz",
    )
    config.addinivalue_line(
        "markers",
        "benchmark_rating: Benchmarks für Rating-Score Berechnungen",
    )
    config.addinivalue_line(
        "markers",
        "benchmark_slow: Langsame Benchmarks (>1s pro Iteration)",
    )
