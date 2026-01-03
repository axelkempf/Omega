import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_minimal_snapshot(
    path: Path, *, symbol: str, timeframe: str, direction: str
) -> None:
    path.write_text(
        json.dumps(
            {
                "base_config": {
                    "symbol": symbol,
                    "timeframes": {"primary": timeframe},
                    "strategy": {"parameters": {"direction_filter": direction}},
                }
            }
        ),
        encoding="utf-8",
    )


def _make_minimal_run(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / name
    (run_dir / "final_selection").mkdir(parents=True, exist_ok=True)
    (run_dir / "final_selection" / "05_final_scores.csv").write_text(
        "a,b\n", encoding="utf-8"
    )
    (run_dir / "final_selection" / "05_final_scores_detailed.csv").write_text(
        "a,b\n", encoding="utf-8"
    )
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    return run_dir


def test_discover_walkforward_groups_accepts_backfill_snapshot(tmp_path: Path) -> None:
    from analysis import combined_walkforward_matrix_analyzer as cwm

    run_dir = _make_minimal_run(tmp_path, "run_no_parse_1")
    _write_minimal_snapshot(
        run_dir / "baseline" / cwm.BACKFILL_SNAPSHOT_NAME,
        symbol="EURUSD",
        timeframe="M30",
        direction="short",
    )

    groups = cwm.discover_walkforward_groups(root=tmp_path)
    assert len(groups) == 1
    assert groups[0].group_id == "EURUSD_M30_short"


def test_load_metadata_from_snapshot_prefers_backfill(tmp_path: Path) -> None:
    from analysis import combined_walkforward_matrix_analyzer as cwm

    run_dir = _make_minimal_run(tmp_path, "run_no_parse_2")
    _write_minimal_snapshot(
        run_dir / "baseline" / "frozen_snapshot.json",
        symbol="SHOULDNOT",
        timeframe="H1",
        direction="long",
    )
    _write_minimal_snapshot(
        run_dir / "baseline" / cwm.BACKFILL_SNAPSHOT_NAME,
        symbol="EURUSD",
        timeframe="M30",
        direction="short",
    )

    meta = cwm.load_metadata_from_snapshot(run_dir)
    assert meta is not None
    assert meta["symbol"] == "EURUSD"
    assert meta["timeframe"] == "M30"
    assert meta["direction"] == "short"
