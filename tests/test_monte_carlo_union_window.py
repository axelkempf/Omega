import numpy as np
import pandas as pd


def test_build_daily_index_union_vs_intersection() -> None:
    from backtest_engine.analysis import combined_walkforward_matrix_analyzer as cwm

    # Use overlapping ranges so intersection is non-empty
    starts = [
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
    ]
    ends = [pd.Timestamp("2020-12-31", tz="UTC"), pd.Timestamp("2020-09-30", tz="UTC")]

    inter = cwm._build_daily_index_utc(starts, ends, mode="intersection")
    uni = cwm._build_daily_index_utc(starts, ends, mode="union")

    assert len(inter) > 0
    assert len(uni) > 0
    # intersection starts later and ends earlier/equal
    assert inter[0] >= uni[0]
    assert inter[-1] <= uni[-1]
    # union should strictly cover more than intersection
    assert (uni[0] < inter[0]) or (uni[-1] > inter[-1])


def test_invalid_candidate_excluded_from_monte_carlo_scoring() -> None:
    """Wenn ein Kandidat nicht bis zum globalen Ende reicht, soll er in MC ausgeschlossen werden."""
    from backtest_engine.analysis import combined_walkforward_matrix_analyzer as cwm

    daily_index = pd.date_range(
        "2021-01-01", periods=3, freq="D", tz="UTC"
    ) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # 1 Gruppe, 2 Kandidaten
    pnl = np.array(
        [
            [0.0, 10.0, 0.0],  # valid
            [0.0, 50.0, 0.0],  # would look great, but invalid
        ],
        dtype=np.float32,
    )

    state = cwm.MonteCarloEvalState(
        group_ids=("G1",),
        combo_pair_ids_by_group=(("A", "B"),),
        equity_daily_pnl_by_group=(pnl,),
        trades_count_by_group=(np.array([1, 1], dtype=np.int32),),
        trades_wins_by_group=(np.array([1, 1], dtype=np.int32),),
        trades_sum_r_by_group=(np.array([1.0, 1.0], dtype=np.float32),),
        robustness_by_group=(np.array([1.0, 1.0], dtype=np.float32),),
        valid_by_group=(np.array([True, False], dtype=bool),),
        daily_index_utc=daily_index,
        month_end_positions=np.array([2], dtype=int),
        month_days_for_profits=np.array([], dtype=float),
        start_equity=100_000.0,
    )

    cwm._init_monte_carlo_worker(state)

    selections = np.array([[0], [1]], dtype=np.int32)
    out = cwm._evaluate_indices_batch_fast(selections)

    final_score = out["final_score"]
    assert np.isfinite(final_score[0])
    assert final_score[1] == -np.inf
