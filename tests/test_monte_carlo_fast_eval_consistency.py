import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _iso_z(ts: str) -> str:
    # Ensure strict Zulu format for stable parsing across code paths.
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_equity_csv(
    path: Path, timestamps_utc: list[str], equities: list[float]
) -> None:
    df = pd.DataFrame({"timestamp": timestamps_utc, "equity": equities})
    df.to_csv(path, index=False)


def _write_trades_json(path: Path, trades: list[dict]) -> None:
    path.write_text(json.dumps(trades, indent=2), encoding="utf-8")


def _final_id_from_mapping(mapping: dict[str, str]) -> str:
    # Same semantics as combined_walkforward_matrix_analyzer.monte_carlo_portfolio_search
    parts = [f"{gid}={mapping[gid]}" for gid in sorted(mapping.keys())]
    base = "__".join(parts)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"final_{digest}"


def test_monte_carlo_fast_eval_matches_slow_metrics(tmp_path: Path) -> None:
    """Regression-Test: Fast Monte-Carlo Evaluator (Schritt 5) muss mit Slow-Metriken übereinstimmen.

    Motivation:
    - Schritt 5 verwendet prepare_monte_carlo_eval_state + NumPy-Batch-Evaluation.
    - Schritt 6 hydriert Equity/Trades erst nach Ranking.

    Dieser Test prüft für wenige synthetische Kandidaten:
    - Mapping (group_id -> combo_pair_id) ist deterministisch.
    - Fast berechnete Kennzahlen (profit, DD, avg_r, winrate, stability, comp, final) stimmen
      mit denselben Kennzahlen überein, die aus hydrierten Artefakten via compute_global_metrics
      + compute_additional_scores + compute_final_score berechnet werden.

    Der Test ist absichtlich klein und deterministisch (kein Multiprocessing), damit er stabil ist.
    """

    from backtest_engine.analysis import combined_walkforward_matrix_analyzer as cwm

    # Isoliere globale Caches (verhindert Cross-Test-Leaks)
    cwm._EQUITY_CACHE.clear()
    cwm._TRADES_CACHE.clear()
    cwm._cached_read_trades_json.cache_clear()

    # 3 Trade-Events über 3 Monate, damit stability_score_monthly sinnvoll berechnet wird.
    exit_times = [
        _iso_z("2025-01-20T12:00:00Z"),
        _iso_z("2025-02-10T12:00:00Z"),
        _iso_z("2025-03-20T12:00:00Z"),
    ]

    def make_trades(prefix: str, r_values: list[float]) -> list[dict]:
        trades = []
        for i, (et, r) in enumerate(zip(exit_times, r_values)):
            entry_time = pd.Timestamp(et, tz="UTC") - pd.Timedelta(days=1)
            trades.append(
                {
                    "trade_id": f"{prefix}_{i}",
                    "entry_time": entry_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "exit_time": et,
                    # Beide Felder: fast-path nutzt r_multiple, slow-path winrate nutzt result
                    "r_multiple": float(r),
                    "result": float(r),
                }
            )
        return trades

    # Zwei Gruppen, jeweils zwei Kandidaten
    # Kandidat-Equities: Equity muss Exit-Zeitpunkte enthalten (Correctness-Gate in prepare_monte_carlo_eval_state)
    g1_c0_eq = [100_500.0, 100_300.0, 101_100.0]
    g1_c1_eq = [100_300.0, 100_600.0, 100_900.0]
    g2_c0_eq = [100_200.0, 100_100.0, 100_200.0]
    g2_c1_eq = [99_950.0, 100_150.0, 100_250.0]

    g1_c0_eq_path = tmp_path / "g1_c0_equity.csv"
    g1_c1_eq_path = tmp_path / "g1_c1_equity.csv"
    g2_c0_eq_path = tmp_path / "g2_c0_equity.csv"
    g2_c1_eq_path = tmp_path / "g2_c1_equity.csv"

    _write_equity_csv(g1_c0_eq_path, exit_times, g1_c0_eq)
    _write_equity_csv(g1_c1_eq_path, exit_times, g1_c1_eq)
    _write_equity_csv(g2_c0_eq_path, exit_times, g2_c0_eq)
    _write_equity_csv(g2_c1_eq_path, exit_times, g2_c1_eq)

    g1_c0_tr_path = tmp_path / "g1_c0_trades.json"
    g1_c1_tr_path = tmp_path / "g1_c1_trades.json"
    g2_c0_tr_path = tmp_path / "g2_c0_trades.json"
    g2_c1_tr_path = tmp_path / "g2_c1_trades.json"

    _write_trades_json(g1_c0_tr_path, make_trades("g1c0", [0.5, -0.2, 0.8]))
    _write_trades_json(g1_c1_tr_path, make_trades("g1c1", [0.3, 0.3, 0.3]))
    _write_trades_json(g2_c0_tr_path, make_trades("g2c0", [0.2, -0.1, 0.1]))
    _write_trades_json(g2_c1_tr_path, make_trades("g2c1", [-0.05, 0.2, 0.1]))

    matrix = pd.DataFrame(
        [
            {
                "group_id": "G1",
                "combo_pair_id": "G1_C0",
                "equity_path": str(g1_c0_eq_path),
                "trades_path": str(g1_c0_tr_path),
                "symbol": "EURUSD",
                "timeframe": "M30",
                "direction": "long",
                "robustness_mean": 0.90,
            },
            {
                "group_id": "G1",
                "combo_pair_id": "G1_C1",
                "equity_path": str(g1_c1_eq_path),
                "trades_path": str(g1_c1_tr_path),
                "symbol": "EURUSD",
                "timeframe": "M30",
                "direction": "long",
                "robustness_mean": 0.80,
            },
            {
                "group_id": "G2",
                "combo_pair_id": "G2_C0",
                "equity_path": str(g2_c0_eq_path),
                "trades_path": str(g2_c0_tr_path),
                "symbol": "GBPUSD",
                "timeframe": "M15",
                "direction": "short",
                "robustness_mean": 0.70,
            },
            {
                "group_id": "G2",
                "combo_pair_id": "G2_C1",
                "equity_path": str(g2_c1_eq_path),
                "trades_path": str(g2_c1_tr_path),
                "symbol": "GBPUSD",
                "timeframe": "M15",
                "direction": "short",
                "robustness_mean": 0.60,
            },
        ]
    )

    index = cwm.build_final_combo_index(matrix)
    assert set(index.keys()) == {"G1", "G2"}
    assert len(index["G1"]) == 2
    assert len(index["G2"]) == 2

    state = cwm.prepare_monte_carlo_eval_state(
        index,
        grid="events",
        window_mode="union",
        trade_equity_tolerance_seconds=1.0,
        fail_on_trade_equity_mismatch=True,
    )

    # Evaluiere alle 2x2 Portfolios deterministisch (ohne Multiprocessing)
    selections = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    cwm._init_monte_carlo_worker(state)
    payload = cwm._evaluate_indices_batch_fast(selections)
    sel = payload.pop("selections")
    df_fast = pd.DataFrame(payload)

    # Dekodiere selection-indizes in JSON-Mapping + final_combo_pair_id
    group_ids = list(state.group_ids)
    combo_ids_by_group = state.combo_pair_ids_by_group

    mapping_jsons: list[str] = []
    final_ids: list[str] = []
    for row in sel:
        mapping = {
            group_ids[i]: str(combo_ids_by_group[i][int(row[i])])
            for i in range(len(group_ids))
        }
        mapping_jsons.append(json.dumps(mapping))
        final_ids.append(_final_id_from_mapping(mapping))

    df_fast.insert(0, "final_combo_pair_id", final_ids)
    df_fast["groups_count"] = len(group_ids)
    df_fast["groups_mapping_json"] = mapping_jsons

    # Hydration: erzeugt _equity_internal (pd.Series) und _trades_internal (list[dict])
    df_hyd = cwm.hydrate_portfolio_artifacts_for_categorical(
        df_fast, eval_state=state, index=index, chunk_size=8
    )

    # Slow-Pfad: Recompute Metriken aus hydrierten Artefakten
    df_slow = df_hyd[
        [
            "final_combo_pair_id",
            "groups_mapping_json",
            "robustness_mean",
            "_equity_internal",
            "_trades_internal",
        ]
    ].copy()

    # compute_global_metrics erwartet DataFrame oder Path – konvertiere list[dict] zu DataFrame.
    df_slow["_trades_internal"] = [
        pd.DataFrame(t) for t in df_slow["_trades_internal"].tolist()
    ]

    df_slow = cwm.compute_global_metrics(df_slow)
    df_slow = cwm.compute_additional_scores(df_slow)
    df_slow = cwm.compute_final_score(df_slow)

    merged = df_fast.merge(
        df_slow[
            [
                "final_combo_pair_id",
                "total_profit",
                "total_max_dd",
                "total_profit_over_dd",
                "total_trades",
                "avg_r",
                "winrate",
                "stability_score_monthly",
                "comp_score",
                "final_score",
                "robustness_mean",
            ]
        ],
        on="final_combo_pair_id",
        how="inner",
        suffixes=("_fast", "_slow"),
    )

    assert len(merged) == 4

    # Integer/Count Felder
    assert (
        merged["total_trades_fast"].astype(int).values
        == merged["total_trades_slow"].astype(int).values
    ).all()

    # Float-Felder (leichte Toleranz wegen float32/float64 Mischung im Fast-Path)
    float_cols = [
        "total_profit",
        "total_max_dd",
        "total_profit_over_dd",
        "avg_r",
        "winrate",
        "stability_score_monthly",
        "comp_score",
        "final_score",
        "robustness_mean",
    ]

    for col in float_cols:
        np.testing.assert_allclose(
            merged[f"{col}_fast"].to_numpy(dtype=float),
            merged[f"{col}_slow"].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-6,
            err_msg=f"Mismatch in column {col}",
        )
