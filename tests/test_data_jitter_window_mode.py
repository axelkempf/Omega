"""
Test: data_jitter_score Berechnung funktioniert auch bei preload_mode='window'
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.backtest_engine.optimizer.final_param_selector import (
    _load_or_get_preloaded_data,
)


def test_load_or_get_preloaded_data_full_mode():
    """Test: Bei preload_mode='full' wird _WORKER_PRELOADED direkt zurückgegeben."""
    test_df = pd.DataFrame({"Open": [1.0]})
    with patch(
        "src.backtest_engine.optimizer.final_param_selector._WORKER_PRELOADED",
        {("M15", "bid"): test_df},
    ):
        result = _load_or_get_preloaded_data({})
        assert ("M15", "bid") in result
        assert result[("M15", "bid")].equals(test_df)


def test_load_or_get_preloaded_data_window_mode(tmp_path):
    """Test: Bei preload_mode='window' werden Daten aus _WORKER_PATHS geladen."""
    # Erstelle Test-Parquet-Dateien
    bid_file = tmp_path / "EURUSD_M15_BID.parquet"
    ask_file = tmp_path / "EURUSD_M15_ASK.parquet"

    bid_df = pd.DataFrame(
        {
            "UTC time": pd.date_range("2024-01-01", periods=5, freq="15min"),
            "Open": [1.0800, 1.0805, 1.0810, 1.0815, 1.0820],
            "High": [1.0810, 1.0815, 1.0820, 1.0825, 1.0830],
            "Low": [1.0795, 1.0800, 1.0805, 1.0810, 1.0815],
            "Close": [1.0805, 1.0810, 1.0815, 1.0820, 1.0825],
            "Volume": [100, 100, 100, 100, 100],
        }
    )

    ask_df = bid_df.copy()
    ask_df[["Open", "High", "Low", "Close"]] += 0.0002

    bid_df.to_parquet(bid_file)
    ask_df.to_parquet(ask_file)

    # Mock _WORKER_PRELOADED (leer) und _WORKER_PATHS (gefüllt)
    with (
        patch(
            "src.backtest_engine.optimizer.final_param_selector._WORKER_PRELOADED", {}
        ),
        patch(
            "src.backtest_engine.optimizer.final_param_selector._WORKER_PATHS",
            {("M15", "bid"): bid_file, ("M15", "ask"): ask_file},
        ),
    ):
        result = _load_or_get_preloaded_data({"symbol": "EURUSD"})

        # Prüfe, dass Daten geladen wurden
        assert ("M15", "bid") in result
        assert ("M15", "ask") in result
        assert len(result[("M15", "bid")]) == 5
        assert len(result[("M15", "ask")]) == 5

        # Prüfe, dass die richtigen Daten geladen wurden
        assert result[("M15", "bid")]["Open"].iloc[0] == pytest.approx(1.0800, abs=1e-6)
        assert result[("M15", "ask")]["Open"].iloc[0] == pytest.approx(1.0802, abs=1e-6)


def test_load_or_get_preloaded_data_no_paths():
    """Test: Wenn weder _WORKER_PRELOADED noch _WORKER_PATHS gefüllt sind, leeres Dict zurückgeben."""
    with (
        patch(
            "src.backtest_engine.optimizer.final_param_selector._WORKER_PRELOADED", {}
        ),
        patch("src.backtest_engine.optimizer.final_param_selector._WORKER_PATHS", {}),
    ):
        result = _load_or_get_preloaded_data({})
        assert result == {}
