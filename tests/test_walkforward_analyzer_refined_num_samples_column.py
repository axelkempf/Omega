from __future__ import annotations

import pandas as pd

from analysis.walkforward_analyzer import _expand_pairs_for_display


def test_refined_display_includes_robustness_num_samples_per_leg() -> None:
    """Refined Top-50 export soll robustness_1_num_samples pro Leg-Zeile enthalten."""
    df_pairs = pd.DataFrame(
        [
            {
                "combo_pair_id": "pair_1",
                "combo_id_1": "c1",
                "source_walkforward_1": "run_a",
                "combo_id_2": "c2",
                "source_walkforward_2": "run_b",
                "robustness_1_leg_A": 0.12,
                "robustness_1_leg_B": 0.34,
                "robustness_1_num_samples_leg_A": 7,
                "robustness_1_num_samples_leg_B": 9,
            }
        ]
    )

    out = _expand_pairs_for_display(
        df_pairs,
        years=["2023"],
        base_df=pd.DataFrame(),
        include_leg_robustness=True,
    )

    assert "robustness_1_num_samples" in out.columns

    # Pro Paar: Zeile A, Zeile B, dann Leerzeile
    assert out.loc[0, "combo_leg"] == "A"
    assert out.loc[0, "robustness_1_num_samples"] == 7

    assert out.loc[1, "combo_leg"] == "B"
    assert out.loc[1, "robustness_1_num_samples"] == 9


def test_base_toplists_do_not_include_num_samples_column() -> None:
    """In den Basis-Toplisten (include_leg_robustness=False) soll die Spalte fehlen."""
    df_pairs = pd.DataFrame(
        [
            {
                "combo_pair_id": "pair_1",
                "combo_id_1": "c1",
                "source_walkforward_1": "",
                "combo_id_2": "c2",
                "source_walkforward_2": "",
                "robustness_1_num_samples_leg_A": 7,
                "robustness_1_num_samples_leg_B": 9,
            }
        ]
    )

    out = _expand_pairs_for_display(
        df_pairs,
        years=["2023"],
        base_df=pd.DataFrame(),
        include_leg_robustness=False,
    )

    assert "robustness_1_num_samples" not in out.columns
