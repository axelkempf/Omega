import pandas as pd


def test_collapse_champions_combo_pairs_to_one_liners_keeps_only_identical_columns() -> (
    None
):
    from analysis import combined_walkforward_matrix_analyzer as cwm

    # Simuliere das expandierte Champion-Format (2 Legs + eine Trennzeile)
    expanded = pd.DataFrame(
        [
            {
                "combo_pair_id": "final_abc",
                "category": "Top Performer",
                "top11_categories": "Top Performer (1)",
                "group_id": "EURUSD_H1_long",
                "combo_leg": "A",
                "final_score": 0.7,
                "total_profit": 1234.56,
                "Net Profit": 100.0,  # leg-spezifisch
                "all_empty_col": pd.NA,
            },
            {
                "combo_pair_id": "final_abc",
                "category": "Top Performer",
                "top11_categories": "Top Performer (1)",
                "group_id": "EURUSD_H1_long",
                "combo_leg": "B",
                "final_score": 0.7,
                "total_profit": 1234.56,
                "Net Profit": 50.0,  # leg-spezifisch (anders)
                "all_empty_col": pd.NA,
            },
            # Trennzeile wie im Export
            {
                "combo_pair_id": pd.NA,
                "category": pd.NA,
                "top11_categories": pd.NA,
                "group_id": pd.NA,
                "combo_leg": pd.NA,
                "final_score": pd.NA,
                "total_profit": pd.NA,
                "Net Profit": pd.NA,
                "all_empty_col": pd.NA,
            },
        ]
    )

    out = cwm._collapse_champions_combo_pairs_to_one_liners(expanded)

    assert len(out) == 1
    assert out.loc[0, "combo_pair_id"] == "final_abc"

    # Identische globale Spalten bleiben
    assert "final_score" in out.columns
    assert "total_profit" in out.columns
    assert out.loc[0, "final_score"] == 0.7

    # Leg-spezifische Spalten fliegen raus
    assert "Net Profit" not in out.columns

    # Komplett leere Spalten fliegen raus
    assert "all_empty_col" not in out.columns

    # Keine Leg-Zusammenfassungsspalte in der 1-Zeiler-Übersicht
    assert "portfolio_legs" not in out.columns


def test_collapse_champions_combo_pairs_to_one_liners_sorts_by_category_order() -> None:
    from analysis import combined_walkforward_matrix_analyzer as cwm

    expanded = pd.DataFrame(
        [
            {
                "combo_pair_id": "final_z",
                "category": "Diversifier",
                "final_score": 0.1,
                "group_id": "G1",
                "combo_leg": "A",
            },
            {
                "combo_pair_id": "final_z",
                "category": "Diversifier",
                "final_score": 0.1,
                "group_id": "G1",
                "combo_leg": "B",
            },
            {
                "combo_pair_id": "final_a",
                "category": "Top Performer",
                "final_score": 0.9,
                "group_id": "G2",
                "combo_leg": "A",
            },
            {
                "combo_pair_id": "final_a",
                "category": "Top Performer",
                "final_score": 0.9,
                "group_id": "G2",
                "combo_leg": "B",
            },
        ]
    )

    out = cwm._collapse_champions_combo_pairs_to_one_liners(expanded)
    assert list(out["category"]) == ["Top Performer", "Diversifier"]
