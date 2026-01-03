import json
from pathlib import Path

import pandas as pd


def test_persist_final_combo_artifacts_prefers_reconstruct(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression-Test: Champions-Artefakte sollen (wenn möglich) aus dem index rekonstruiert werden.

    Hintergrund: Der Monte-Carlo-Pfad kann eine Equity auf einem (konservativen) Daily-Grid erzeugen.
    Für persistierte final_combos wollen wir jedoch die vollständige Equity aus den Original-Serien
    (Union der Timestamps) schreiben. Das wird über aggregate_final_combo() erreicht.

    Dieser Test stellt sicher, dass persist_final_combo_artifacts() bei übergebenem ``index`` und
    gültigem ``groups_mapping_json`` aggregate_final_combo aufruft und den Legacy-Fallback nicht nutzt.
    """
    from analysis import combined_walkforward_matrix_analyzer as cwm

    # Isoliere IO in tmp_path
    monkeypatch.setattr(
        cwm, "COMBINED_MATRIX_DIR", tmp_path / "combined_matrix", raising=True
    )
    cwm.COMBINED_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    index = {
        "G1": [
            {
                "combo_pair_id": "CP1",
                "equity_path": tmp_path / "g1_equity.csv",
                "trades_path": tmp_path / "g1_trades.json",
            }
        ],
        "G2": [
            {
                "combo_pair_id": "CP2",
                "equity_path": tmp_path / "g2_equity.csv",
                "trades_path": tmp_path / "g2_trades.json",
            }
        ],
    }

    # selection wie in persist_final_combo_artifacts rekonstruiert
    selection = {"G1": index["G1"][0], "G2": index["G2"][0]}
    final_id = cwm._final_combo_id_from_selection(selection)

    calls = []

    def fake_aggregate_final_combo(sel, write_files: bool = True):
        calls.append(sel)
        assert write_files is True
        # Schreibe eindeutige Marker-Dateien, um zu erkennen, dass der Rekonstruktionspfad genutzt wurde.
        out_dir = (
            cwm.COMBINED_MATRIX_DIR
            / "final_combos"
            / cwm._final_combo_id_from_selection(sel)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "equity.csv").write_text(
            "timestamp,equity\n2000-01-01T00:00:00Z,123\n", encoding="utf-8"
        )
        (out_dir / "trades.json").write_text("[]", encoding="utf-8")
        return {
            "final_id": out_dir.name,
            "equity": out_dir / "equity.csv",
            "trades": out_dir / "trades.json",
        }

    monkeypatch.setattr(
        cwm, "aggregate_final_combo", fake_aggregate_final_combo, raising=True
    )

    # Legacy-Fallback-Objekte (sollten NICHT geschrieben werden)
    legacy_eq = pd.Series([999.0], index=pd.to_datetime(["1999-01-01"], utc=True))
    legacy_trades = [
        {"entry_time": "1999-01-01T00:00:00Z", "exit_time": "1999-01-02T00:00:00Z"}
    ]

    df = pd.DataFrame(
        [
            {
                "final_combo_pair_id": final_id,
                "groups_mapping_json": json.dumps({"G1": "CP1", "G2": "CP2"}),
                "_equity_internal": legacy_eq,
                "_trades_internal": legacy_trades,
            }
        ]
    )

    cwm.persist_final_combo_artifacts(df, index=index)

    assert len(calls) == 1
    assert set(calls[0].keys()) == {"G1", "G2"}

    out_dir = cwm.COMBINED_MATRIX_DIR / "final_combos" / final_id
    assert out_dir.exists()

    # Marker-Inhalt aus fake_aggregate_final_combo
    equity_text = (out_dir / "equity.csv").read_text(encoding="utf-8")
    assert "2000-01-01" in equity_text
    assert "999" not in equity_text  # nicht aus Legacy-Fallback
