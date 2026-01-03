from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest_engine.analysis import backfill_walkforward_equity_curves as backfill


@pytest.fixture()
def sample_segments() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        (
            "2024",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31"),
        )
    ]


def _write_combined(run_dir: Path, combo_id: str) -> Path:
    final_dir = run_dir / "final_selection"
    final_dir.mkdir(parents=True, exist_ok=True)
    df_combined = pd.DataFrame(
        {
            "combo_id": [combo_id],
            "trade_dropout_score": [0.5],
            "Net Profit": [0.0],
            "Commission": [0.0],
            "Avg R-Multiple": [0.0],
            "Winrate (%)": [0.0],
            "Drawdown": [0.0],
            "Sharpe (trade)": [0.0],
            "Sortino (trade)": [0.0],
            "total_trades": [0.0],
            "active_days": [0.0],
            "profit_over_dd": [0.0],
            "comm_over_profit": [0.0],
        }
    )
    combined_path = final_dir / "05_final_scores_combined.csv"
    df_combined.to_csv(combined_path, index=False)
    return combined_path


def _write_trades(run_dir: Path, combo_id: str) -> Path:
    trades_dir = run_dir / "final_selection" / "trades" / combo_id
    trades_dir.mkdir(parents=True, exist_ok=True)
    trades = [
        {
            "result": 100.0,
            "total_fee": 0.0,
            "entry_time": "2024-06-01T00:00:00Z",
            "exit_time": "2024-06-10T00:00:00Z",
            "r_multiple": 1.0,
        }
    ]
    trades_path = trades_dir / "trades.json"
    trades_path.write_text(json.dumps(trades))
    return trades_path


def test_backfill_sets_stability_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_segments
):
    """
    Ensure stability_score is computed during backfill when yearly profits exist.
    """

    combo_id = "combo_1"
    run_dir = tmp_path / "run"
    _write_combined(run_dir, combo_id)
    _write_trades(run_dir, combo_id)

    # Patch dependencies to avoid external I/O
    monkeypatch.setattr(
        backfill,
        "load_snapshot",
        lambda *_: (
            {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "symbol": "EURUSD",
                "timeframes": {"primary": "M15"},
            },
            {},
        ),
    )
    monkeypatch.setattr(backfill, "_yearly_segments", lambda *_: sample_segments)
    monkeypatch.setattr(
        backfill, "load_primary_candle_arrays_from_parquet", lambda *_, **__: None
    )

    backfill._update_final_scores_with_backfill_metrics(run_dir, {combo_id})

    out_path = run_dir / "final_selection" / backfill.BACKFILL_COMBINED_FILENAME
    df_out = pd.read_csv(out_path)

    assert "stability_score" in df_out.columns
    # Stability score should be finite and non-null
    assert pd.notna(df_out.loc[0, "stability_score"])
    # For a single-year profit with matching duration the score should be 1.0
    assert pytest.approx(df_out.loc[0, "stability_score"], rel=1e-6) == 1.0
    # tp_sl_stress_score is computed with missing candles -> defaults to 1.0
    assert "tp_sl_stress_score" in df_out.columns
    assert pytest.approx(df_out.loc[0, "tp_sl_stress_score"], rel=1e-6) == 1.0
