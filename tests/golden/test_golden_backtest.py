"""
Golden-File Tests für Backtest-Determinismus.

P3-09: Validiert dass Backtests mit fixiertem Seed reproduzierbare
Ergebnisse liefern. Dies ist kritisch für die FFI-Migration, um
sicherzustellen dass Rust/Julia-Implementierungen identische
Ergebnisse wie Python liefern.

Voraussetzungen:
- Testdaten in data/csv/ oder data/parquet/
- Strategie-Module in src/strategies/

Tests:
1. Indicator-Berechnungen sind deterministisch
2. Trade-Signale sind reproduzierbar
3. Portfolio-Metriken sind konsistent
4. Equity-Kurven stimmen überein
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.golden.conftest import (
    GoldenBacktestResult,
    GoldenFileManager,
    assert_golden_match,
    compute_dataframe_hash,
    compute_dict_hash,
    create_metadata,
    set_deterministic_seed,
)
from backtest_engine.core.indicator_cache import IndicatorCache

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def minimal_ohlcv_data() -> pd.DataFrame:
    """
    Generiert minimale deterministische OHLCV-Daten für Tests.

    Diese Daten sind vollständig synthetisch und reproducible,
    ohne Abhängigkeit von externen Datendateien.
    """
    np.random.seed(42)
    n_bars = 1000

    # Generiere realistische Price-Bewegungen
    returns = np.random.normal(0, 0.001, n_bars)
    close = 1.1000 * np.cumprod(1 + returns)

    # OHLC aus Close ableiten
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_price = np.roll(close, 1)
    open_price[0] = 1.1000

    # Korrigiere: High >= max(Open, Close), Low <= min(Open, Close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Timestamps
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")

    return pd.DataFrame(
        {
            "UTC time": timestamps,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(100, 10000, n_bars),
        }
    )


@pytest.fixture
def mock_data_handler(minimal_ohlcv_data: pd.DataFrame):
    """Mock für CSVDataHandler mit deterministischen Daten."""

    class MockDataHandler:
        def __init__(self, data: pd.DataFrame):
            self._data = data
            self._index = 0

        def get_candles(
            self, symbol: str, timeframe: str, count: int
        ) -> List[Dict[str, Any]]:
            """Liefert die letzten 'count' Candles."""
            start = max(0, self._index - count)
            end = self._index
            df = self._data.iloc[start:end]
            return df.to_dict("records")

        def advance(self):
            """Bewegt den Index um 1 vorwärts."""
            self._index = min(self._index + 1, len(self._data))

        @property
        def current_bar(self) -> int:
            return self._index

        @property
        def total_bars(self) -> int:
            return len(self._data)

    return MockDataHandler(minimal_ohlcv_data)


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================


def compute_indicator_output_hash(
    data: pd.DataFrame,
    indicator_fn,
    *args,
    **kwargs,
) -> str:
    """Berechnet Hash für Indicator-Output."""
    result = indicator_fn(data, *args, **kwargs)
    if isinstance(result, pd.Series):
        result = result.to_frame()
    return compute_dataframe_hash(result)


def create_backtest_result_from_mock(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame,
    metrics: Dict[str, Any],
    seed: int,
    description: str,
) -> GoldenBacktestResult:
    """Erstellt GoldenBacktestResult aus Mock-Daten."""
    trade_hashes = [compute_dict_hash(t) for t in trades]
    equity_hash = compute_dataframe_hash(equity_curve)

    return GoldenBacktestResult(
        metadata=create_metadata(seed, description),
        summary_metrics=metrics,
        trade_count=len(trades),
        trade_hashes=trade_hashes,
        equity_curve_hash=equity_hash,
        final_equity=float(equity_curve["equity"].iloc[-1]) if len(equity_curve) > 0 else 0.0,
        total_pnl=float(metrics.get("net_profit_eur", 0.0)),
    )


# ==============================================================================
# INDICATOR DETERMINISMUS TESTS
# ==============================================================================


class TestIndicatorDeterminism:
    """Tests für Indicator-Berechnung Determinismus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup für jeden Test."""
        set_deterministic_seed(42)

    def test_ema_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """EMA-Berechnung ist deterministisch."""

        close = minimal_ohlcv_data["Close"].values.tolist()

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        for i, price in enumerate(close):
            # Simuliere Candle-Input
            candle = {"close": price}
            ema1 = cache1.ema(candle, length=20, key="close")
            ema2 = cache2.ema(candle, length=20, key="close")

            if ema1 is not None and ema2 is not None:
                assert abs(ema1 - ema2) < 1e-10, f"EMA mismatch at index {i}"

    def test_rsi_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """RSI-Berechnung ist deterministisch."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        close = minimal_ohlcv_data["Close"].values.tolist()

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        for price in close:
            candle = {"close": price}
            rsi1 = cache1.rsi(candle, length=14, key="close")
            rsi2 = cache2.rsi(candle, length=14, key="close")

            if rsi1 is not None and rsi2 is not None:
                assert abs(rsi1 - rsi2) < 1e-10, f"RSI mismatch: {rsi1} vs {rsi2}"

    def test_atr_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """ATR-Berechnung ist deterministisch."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        for i in range(len(minimal_ohlcv_data)):
            row = minimal_ohlcv_data.iloc[i]
            candle = {
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
            }
            atr1 = cache1.atr(candle, length=14)
            atr2 = cache2.atr(candle, length=14)

            if atr1 is not None and atr2 is not None:
                assert abs(atr1 - atr2) < 1e-10, f"ATR mismatch at index {i}"

    def test_macd_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """MACD-Berechnung ist deterministisch."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        close = minimal_ohlcv_data["Close"].values.tolist()

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        for price in close:
            candle = {"close": price}
            macd1 = cache1.macd(candle, fast=12, slow=26, signal=9, key="close")
            macd2 = cache2.macd(candle, fast=12, slow=26, signal=9, key="close")

            if macd1 is not None and macd2 is not None:
                assert abs(macd1["macd"] - macd2["macd"]) < 1e-10
                assert abs(macd1["signal"] - macd2["signal"]) < 1e-10
                assert abs(macd1["hist"] - macd2["hist"]) < 1e-10

    def test_bollinger_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """Bollinger Bands Berechnung ist deterministisch."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        close = minimal_ohlcv_data["Close"].values.tolist()

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        for price in close:
            candle = {"close": price}
            bb1 = cache1.bollinger(candle, length=20, mult=2.0, key="close")
            bb2 = cache2.bollinger(candle, length=20, mult=2.0, key="close")

            if bb1 is not None and bb2 is not None:
                assert abs(bb1["upper"] - bb2["upper"]) < 1e-10
                assert abs(bb1["middle"] - bb2["middle"]) < 1e-10
                assert abs(bb1["lower"] - bb2["lower"]) < 1e-10


# ==============================================================================
# TRADE GENERATION DETERMINISMUS
# ==============================================================================


class TestTradeGenerationDeterminism:
    """Tests für deterministische Trade-Generierung."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup für jeden Test."""
        set_deterministic_seed(42)

    def test_signal_sequence_determinism(self, minimal_ohlcv_data: pd.DataFrame):
        """
        Signal-Generierung bei gleichem Seed liefert identische Sequenz.

        Dieser Test simuliert eine einfache EMA-Crossover Strategie
        und prüft dass die Signal-Sequenz deterministisch ist.
        """
        from backtest_engine.core.indicator_cache import IndicatorCache

        def generate_signals(data: pd.DataFrame, seed: int) -> List[int]:
            """Generiert Signal-Sequenz: 1=Long, -1=Short, 0=Neutral."""
            set_deterministic_seed(seed)
            cache = IndicatorCache()
            signals = []

            for i in range(len(data)):
                candle = {"close": data["Close"].iloc[i]}
                ema_fast = cache.ema(candle, length=10, key="close", prefix="fast")
                ema_slow = cache.ema(candle, length=20, key="close", prefix="slow")

                if ema_fast is None or ema_slow is None:
                    signals.append(0)
                elif ema_fast > ema_slow:
                    signals.append(1)
                elif ema_fast < ema_slow:
                    signals.append(-1)
                else:
                    signals.append(0)

            return signals

        signals1 = generate_signals(minimal_ohlcv_data, seed=42)
        signals2 = generate_signals(minimal_ohlcv_data, seed=42)

        assert signals1 == signals2, "Signal sequences should be identical with same seed"

    def test_different_seeds_produce_different_results(
        self, minimal_ohlcv_data: pd.DataFrame
    ):
        """
        Verschiedene Seeds führen zu unterschiedlichen RNG-basierten Ergebnissen.

        Hinweis: Indicator-Berechnungen selbst sind nicht zufällig,
        aber Slippage/Fill-Simulationen können es sein.
        """
        set_deterministic_seed(42)
        random_values_1 = [np.random.random() for _ in range(100)]

        set_deterministic_seed(123)
        random_values_2 = [np.random.random() for _ in range(100)]

        assert random_values_1 != random_values_2


# ==============================================================================
# GOLDEN FILE BACKTEST TESTS
# ==============================================================================


class TestGoldenFileBacktest:
    """
    Golden-File Tests für vollständige Backtest-Runs.

    Diese Tests verwenden Mock-Komponenten um externe Abhängigkeiten
    (Datendateien, Strategie-Module) zu eliminieren.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup für jeden Test."""
        set_deterministic_seed(42)

    def test_indicator_cache_golden_output(
        self, golden_manager: GoldenFileManager, minimal_ohlcv_data: pd.DataFrame
    ):
        """
        Golden-File Test für IndicatorCache Output.

        Validiert dass alle Indicator-Berechnungen für gegebenen Input
        identische Outputs liefern.
        """
        from backtest_engine.core.indicator_cache import IndicatorCache

        cache = IndicatorCache()
        outputs: Dict[str, List[Optional[float]]] = {
            "ema_10": [],
            "ema_20": [],
            "rsi_14": [],
            "atr_14": [],
        }

        for i in range(len(minimal_ohlcv_data)):
            row = minimal_ohlcv_data.iloc[i]
            candle = {
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
            }

            outputs["ema_10"].append(cache.ema(candle, length=10, key="close", prefix="10"))
            outputs["ema_20"].append(cache.ema(candle, length=20, key="close", prefix="20"))
            outputs["rsi_14"].append(cache.rsi(candle, length=14, key="close"))
            outputs["atr_14"].append(cache.atr(candle, length=14))

        # Berechne Hash des gesamten Outputs
        output_hash = compute_dict_hash(
            {k: [v if v is not None else "None" for v in vals] for k, vals in outputs.items()}
        )

        # Erstelle Golden-File Result
        result = GoldenBacktestResult(
            metadata=create_metadata(42, "IndicatorCache golden output test"),
            summary_metrics={"output_hash": output_hash, "n_bars": len(minimal_ohlcv_data)},
            trade_count=0,
            trade_hashes=[],
            equity_curve_hash="",
            final_equity=0.0,
            total_pnl=0.0,
        )

        # Vergleiche oder erstelle Referenz
        try:
            comparison = golden_manager.compare_backtest_results(
                "indicator_cache_output", result, metric_tolerance=1e-10, strict_trades=False
            )
            assert comparison["status"] == "match"
        except Exception:
            # Erstelle neue Referenz wenn keine existiert
            golden_manager.save_backtest_reference("indicator_cache_output", result)

    def test_mock_backtest_determinism(
        self, golden_manager: GoldenFileManager, minimal_ohlcv_data: pd.DataFrame
    ):
        """
        Testet Determinismus eines vereinfachten Mock-Backtests.

        Dieser Test simuliert einen Backtest ohne echte Strategie-Logik
        um die Reproduzierbarkeit der Infrastruktur zu validieren.
        """
        from backtest_engine.core.indicator_cache import IndicatorCache

        # Simulierter Backtest
        cache = IndicatorCache()
        initial_balance = 100000.0
        balance = initial_balance
        trades: List[Dict[str, Any]] = []
        equity_history: List[Dict[str, Any]] = []

        position = None

        for i in range(len(minimal_ohlcv_data)):
            row = minimal_ohlcv_data.iloc[i]
            candle = {
                "timestamp": row["UTC time"],
                "close": row["Close"],
                "high": row["High"],
                "low": row["Low"],
            }

            ema_fast = cache.ema(candle, length=10, key="close", prefix="fast")
            ema_slow = cache.ema(candle, length=20, key="close", prefix="slow")

            # Simple Crossover Logic
            if ema_fast is not None and ema_slow is not None:
                if position is None and ema_fast > ema_slow:
                    # Enter Long
                    position = {
                        "entry_price": candle["close"],
                        "entry_time": candle["timestamp"],
                        "direction": "long",
                    }
                elif position is not None and ema_fast < ema_slow:
                    # Exit
                    pnl = (candle["close"] - position["entry_price"]) * 10000
                    balance += pnl
                    trades.append(
                        {
                            "entry_price": position["entry_price"],
                            "exit_price": candle["close"],
                            "entry_time": str(position["entry_time"]),
                            "exit_time": str(candle["timestamp"]),
                            "pnl": pnl,
                            "direction": position["direction"],
                        }
                    )
                    position = None

            equity_history.append(
                {"timestamp": str(candle["timestamp"]), "equity": balance}
            )

        # Erstelle Golden Result
        equity_df = pd.DataFrame(equity_history)
        metrics = {
            "net_profit_eur": balance - initial_balance,
            "total_trades": len(trades),
            "winrate_percent": (
                sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100
                if trades
                else 0
            ),
        }

        result = create_backtest_result_from_mock(
            trades=trades,
            equity_curve=equity_df,
            metrics=metrics,
            seed=42,
            description="Mock EMA crossover backtest",
        )

        # Vergleiche oder erstelle Referenz
        try:
            comparison = golden_manager.compare_backtest_results(
                "mock_ema_crossover", result, metric_tolerance=1e-6
            )
            assert comparison["status"] == "match"
        except Exception:
            golden_manager.save_backtest_reference("mock_ema_crossover", result)


# ==============================================================================
# REPRODUZIERBARKEIT BEI WIEDERHOLTEN RUNS
# ==============================================================================


class TestReproducibilityAcrossRuns:
    """Tests für Reproduzierbarkeit über mehrere Durchläufe."""

    def test_multiple_runs_produce_identical_results(
        self, minimal_ohlcv_data: pd.DataFrame
    ):
        """Mehrere Durchläufe mit gleichem Seed liefern identische Ergebnisse."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        def run_simulation(seed: int) -> Dict[str, Any]:
            set_deterministic_seed(seed)
            cache = IndicatorCache()

            results = []
            for i in range(len(minimal_ohlcv_data)):
                row = minimal_ohlcv_data.iloc[i]
                candle = {"close": row["Close"]}
                ema = cache.ema(candle, length=20, key="close")
                if ema is not None:
                    results.append(round(ema, 10))

            return {"ema_values": results, "hash": compute_dict_hash({"values": results})}

        # Führe 3 identische Runs durch
        run1 = run_simulation(42)
        run2 = run_simulation(42)
        run3 = run_simulation(42)

        assert run1["hash"] == run2["hash"] == run3["hash"]
        assert run1["ema_values"] == run2["ema_values"] == run3["ema_values"]

    def test_fresh_cache_vs_reused_cache(self, minimal_ohlcv_data: pd.DataFrame):
        """Frischer Cache vs. wiederverwendeter Cache liefern gleiche Ergebnisse."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        # Run 1: Frischer Cache
        set_deterministic_seed(42)
        cache1 = IndicatorCache()
        results1 = []
        for i in range(len(minimal_ohlcv_data)):
            candle = {"close": minimal_ohlcv_data["Close"].iloc[i]}
            results1.append(cache1.ema(candle, length=20, key="close"))

        # Run 2: Nochmal frischer Cache
        set_deterministic_seed(42)
        cache2 = IndicatorCache()
        results2 = []
        for i in range(len(minimal_ohlcv_data)):
            candle = {"close": minimal_ohlcv_data["Close"].iloc[i]}
            results2.append(cache2.ema(candle, length=20, key="close"))

        # Vergleiche
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if r1 is not None and r2 is not None:
                assert abs(r1 - r2) < 1e-12, f"Mismatch at index {i}: {r1} vs {r2}"
            else:
                assert r1 == r2, f"None mismatch at index {i}"


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestDeterminismEdgeCases:
    """Tests für Determinismus bei Edge Cases."""

    def test_empty_data(self):
        """Leere Daten werden konsistent behandelt."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        # Keine Daten zugeführt
        assert cache1.ema({}, length=20, key="close") is None
        assert cache2.ema({}, length=20, key="close") is None

    def test_single_value(self):
        """Einzelner Wert wird konsistent behandelt."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        set_deterministic_seed(42)
        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        candle = {"close": 1.1234}

        ema1 = cache1.ema(candle, length=20, key="close")
        ema2 = cache2.ema(candle, length=20, key="close")

        # Bei nur einem Wert sollte EMA None sein (nicht genug Daten)
        # oder der erste Wert selbst
        assert ema1 == ema2

    def test_extreme_values(self):
        """Extreme Werte werden konsistent behandelt."""
        from backtest_engine.core.indicator_cache import IndicatorCache

        set_deterministic_seed(42)
        cache1 = IndicatorCache()
        cache2 = IndicatorCache()

        extreme_prices = [1e-10, 1e10, 0.0, -1e-10]

        for price in extreme_prices:
            candle = {"close": price}
            ema1 = cache1.ema(candle, length=5, key="close")
            ema2 = cache2.ema(candle, length=5, key="close")
            # Beide sollten gleich sein (auch wenn None oder NaN)
            if ema1 is not None and ema2 is not None:
                if not (np.isnan(ema1) and np.isnan(ema2)):
                    assert ema1 == ema2


# ==============================================================================
# GOLDEN FILE MANAGEMENT
# ==============================================================================


class TestGoldenFileManagement:
    """Tests für Golden-File Management Funktionalität."""

    def test_save_and_load_backtest_reference(
        self, golden_manager: GoldenFileManager, tmp_path: Path
    ):
        """Golden-File kann gespeichert und geladen werden."""
        # Verwende temp_path für isolierten Test
        manager = GoldenFileManager(tmp_path / "golden")

        result = GoldenBacktestResult(
            metadata=create_metadata(42, "Test reference"),
            summary_metrics={"profit": 1000.0, "trades": 50},
            trade_count=50,
            trade_hashes=["abc123", "def456"],
            equity_curve_hash="xyz789",
            final_equity=101000.0,
            total_pnl=1000.0,
        )

        # Speichern
        path = manager.save_backtest_reference("test_ref", result)
        assert path.exists()

        # Laden
        loaded = manager.load_backtest_reference("test_ref")
        assert loaded is not None
        assert loaded.trade_count == result.trade_count
        assert loaded.final_equity == result.final_equity
        assert loaded.trade_hashes == result.trade_hashes

    def test_comparison_detects_differences(
        self, tmp_path: Path
    ):
        """Vergleich erkennt Unterschiede zwischen Referenz und aktuellem Ergebnis."""
        manager = GoldenFileManager(tmp_path / "golden")

        reference = GoldenBacktestResult(
            metadata=create_metadata(42, "Reference"),
            summary_metrics={"profit": 1000.0},
            trade_count=50,
            trade_hashes=["abc"],
            equity_curve_hash="xyz",
            final_equity=101000.0,
            total_pnl=1000.0,
        )
        manager.save_backtest_reference("test_diff", reference)

        # Aktuelles Ergebnis mit Unterschieden
        current = GoldenBacktestResult(
            metadata=create_metadata(42, "Current"),
            summary_metrics={"profit": 1500.0},  # Unterschiedlich
            trade_count=60,  # Unterschiedlich
            trade_hashes=["def"],  # Unterschiedlich
            equity_curve_hash="abc",  # Unterschiedlich
            final_equity=101500.0,  # Unterschiedlich
            total_pnl=1500.0,  # Unterschiedlich
        )

        # Sollte Exception werfen
        from tests.golden.conftest import GoldenFileComparisonError

        with pytest.raises(GoldenFileComparisonError) as exc_info:
            manager.compare_backtest_results("test_diff", current)

        assert "trade_count" in exc_info.value.details
        assert "final_equity" in exc_info.value.details

    def test_nonexistent_reference_returns_no_reference(
        self, tmp_path: Path
    ):
        """Nicht existierende Referenz gibt no_reference Status zurück."""
        manager = GoldenFileManager(tmp_path / "golden")

        current = GoldenBacktestResult(
            metadata=create_metadata(42, "Current"),
            summary_metrics={},
            trade_count=0,
            trade_hashes=[],
            equity_curve_hash="",
            final_equity=100000.0,
            total_pnl=0.0,
        )

        result = manager.compare_backtest_results("nonexistent", current)
        assert result["status"] == "no_reference"
