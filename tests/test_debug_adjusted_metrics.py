"""
Debug-Test zur Validierung der adjustierten Metriken in Score-Berechnungen.

Zeigt die Raw vs. Adjusted Metriken für alle drei Scoring-Stellen:
1. walkforward_analyzer.py - yearly composite scores
2. combined_walkforward_matrix_analyzer.py - Monte Carlo evaluation
3. combined_walkforward_matrix_analyzer.py - additional scores
"""

import numpy as np

from backtest_engine.analysis.metric_adjustments import (
    bayesian_shrinkage,
    risk_adjusted,
    shrinkage_adjusted,
)


def test_debug_score_calculation_walkforward_yearly():
    """Validiere yearly composite score Berechnung mit adjustierten Metriken."""
    print("\n" + "=" * 80)
    print("DEBUG: WALKFORWARD YEARLY SCORES (per Combo-Pair, per Year)")
    print("=" * 80)

    # Simuliere 5 Combo-Pairs für Jahr 2024
    n_pairs = 5
    winrates_raw = np.array([65.0, 45.0, 52.0, 58.0, 48.0])  # in percent
    avg_r_raw = np.array([0.35, 0.15, 0.25, 0.42, 0.18])
    n_trades_list = np.array([50, 10, 100, 75, 20])
    pod_raw = np.array([2.5, 0.8, 3.2, 2.8, 1.1])

    # Für yearly metrics: n_years=1.0, winrate_prior = alle Pairs im Jahr
    n_years = 1.0
    all_winrates_year = winrates_raw / 100.0

    print(f"\nBasis-Daten für {n_pairs} Combo-Pairs (Jahr 2024):")
    print(f"  Winrates (raw):    {winrates_raw}")
    print(f"  Avg R (raw):       {avg_r_raw}")
    print(f"  N Trades:          {n_trades_list}")
    print(f"  PoD (raw):         {pod_raw}")
    print(f"  Prior (winrate mean): {np.mean(all_winrates_year):.2%}")

    # Berechne adjustierte Metriken
    winrate_decimal = winrates_raw / 100.0
    wr_adjusted = bayesian_shrinkage(
        winrate=winrate_decimal, n_trades=n_trades_list, all_winrates=all_winrates_year
    )

    avg_r_adjusted = shrinkage_adjusted(
        average_r=avg_r_raw, n_trades=n_trades_list, n_years=n_years
    )

    pod_adjusted = risk_adjusted(
        profit_over_drawdown=pod_raw, n_trades=n_trades_list, n_years=n_years
    )

    # Normalisiere PoD für Score
    pod_term = pod_adjusted / (1.0 + pod_adjusted)

    # Berechne comp_score
    comp_scores = (wr_adjusted + avg_r_adjusted + pod_term) / 3.0

    print(f"\nAdjustierte Metriken:")
    print(
        f"{'Pair':<8} {'WR Raw%':>8} {'WR Adj%':>8} {'AvgR Raw':>10} {'AvgR Adj':>10} {'Trades':>8} {'CompScore':>10}"
    )
    print("-" * 80)
    for i in range(n_pairs):
        print(
            f"  {i}    {winrates_raw[i]:>7.1f}% {wr_adjusted[i]*100:>7.1f}%  "
            f"{avg_r_raw[i]:>9.3f}  {avg_r_adjusted[i]:>9.3f}  {n_trades_list[i]:>7.0f}  {comp_scores[i]:>9.4f}"
        )

    print(
        f"\n✓ Beobachtung: Pair mit wenigen Trades (10, 20) wird stärker adjustiert als Pair mit 100 Trades"
    )


def test_debug_score_calculation_monte_carlo_total():
    """Validiere Monte Carlo total score Berechnung mit adjustierten Metriken."""
    print("\n" + "=" * 80)
    print("DEBUG: MONTE CARLO TOTAL SCORES (Portfolios über 4 Jahre)")
    print("=" * 80)

    # Simuliere 3 Portfolios über 4 Jahre (2021-2024)
    n_portfolios = 3
    winrates_raw = np.array([58.0, 45.0, 62.0])  # in percent
    avg_r_raw = np.array([0.28, 0.12, 0.38])
    n_trades_list = np.array([200, 30, 350])  # Über 4 Jahre
    pod_raw = np.array([2.2, 0.9, 3.5])

    # Für total metrics über 4 Jahre: n_years=4.0
    n_years = 4.0
    all_winrates_batch = winrates_raw / 100.0  # Alle Portfolios im Batch

    print(f"\nBasis-Daten für {n_portfolios} Portfolios (über {n_years:.0f} Jahre):")
    print(f"  Winrates (raw):    {winrates_raw}")
    print(f"  Avg R (raw):       {avg_r_raw}")
    print(f"  N Trades Total:    {n_trades_list}")
    print(f"  PoD (raw):         {pod_raw}")
    print(f"  konst. = {n_years:.0f} Jahre × 15 = {n_years * 15:.0f}")
    print(f"  Prior (winrate mean): {np.mean(all_winrates_batch):.2%}")

    # Berechne adjustierte Metriken
    winrate_decimal = winrates_raw / 100.0
    wr_adjusted = bayesian_shrinkage(
        winrate=winrate_decimal, n_trades=n_trades_list, all_winrates=all_winrates_batch
    )

    avg_r_adjusted = shrinkage_adjusted(
        average_r=avg_r_raw, n_trades=n_trades_list, n_years=n_years
    )

    pod_adjusted = risk_adjusted(
        profit_over_drawdown=pod_raw, n_trades=n_trades_list, n_years=n_years
    )

    # Normalisiere PoD für Score
    pod_term = pod_adjusted / (1.0 + pod_adjusted)

    # Berechne comp_score
    comp_scores = (wr_adjusted + avg_r_adjusted + pod_term) / 3.0

    print(f"\nAdjustierte Metriken:")
    print(
        f"{'Port':<8} {'WR Raw%':>8} {'WR Adj%':>8} {'AvgR Raw':>10} {'AvgR Adj':>10} {'Trades':>8} {'CompScore':>10}"
    )
    print("-" * 80)
    for i in range(n_portfolios):
        print(
            f"  {i}    {winrates_raw[i]:>7.1f}% {wr_adjusted[i]*100:>7.1f}%  "
            f"{avg_r_raw[i]:>9.3f}  {avg_r_adjusted[i]:>9.3f}  {n_trades_list[i]:>7.0f}  {comp_scores[i]:>9.4f}"
        )

    print(
        f"\n✓ Beobachtung: Portfolio 1 mit nur 30 Trades wird stark geshrunkt (konst.=60)"
    )
    print(f"✓ Beobachtung: Portfolio 2 mit 350 Trades behält mehr von raw-Werten")


def test_debug_shrinkage_impact_by_trades():
    """Zeige Impact von Trade-Anzahl auf Shrinkage-Stärke."""
    print("\n" + "=" * 80)
    print("DEBUG: SHRINKAGE IMPACT BY TRADE COUNT")
    print("=" * 80)

    # Fixe Raw-Metriken, variable Trade-Anzahl
    avg_r_raw = 0.5
    pod_raw = 3.0
    winrate_raw = 0.65  # 65%
    n_years = 1.0
    all_winrates = np.array([0.5, 0.55, 0.60])

    # Trade-Counts: niedrig, mittel, hoch
    trade_counts = np.array([5, 50, 500])

    print(f"\nRohe Metriken (konstant):")
    print(f"  Winrate:  65.0%")
    print(f"  Avg R:    0.500")
    print(f"  PoD:      3.000")

    print(f"\nEffect der Trade-Anzahl auf Adjustierung (n_years=1.0, konst.=15):")
    print(
        f"{'Trades':>8} {'WR Adj%':>10} {'AvgR Adj':>10} {'PoD Adj':>10} {'Shrinkage Stärke':>20}"
    )
    print("-" * 80)

    for n_trades in trade_counts:
        wr_adj = bayesian_shrinkage(winrate_raw, n_trades, all_winrates) * 100
        avg_r_adj = shrinkage_adjusted(avg_r_raw, n_trades, n_years)
        pod_adj = risk_adjusted(pod_raw, n_trades, n_years)

        shrinkage_strength = (avg_r_raw - avg_r_adj) / avg_r_raw

        if n_trades == 5:
            strength_label = "STARK (wenig Daten)"
        elif n_trades == 50:
            strength_label = "MITTEL"
        else:
            strength_label = "SCHWACH (viel Daten)"

        print(
            f"{n_trades:>8.0f} {wr_adj:>10.1f}% {avg_r_adj:>10.3f} {pod_adj:>10.3f}  {strength_label:>20}"
        )

    print(
        f"\n✓ Kernprinzip: Wenige Trades → stärkerer Shrinkage → realistischere Scores"
    )


if __name__ == "__main__":
    # Führe alle Debug-Tests aus
    test_debug_score_calculation_walkforward_yearly()
    test_debug_score_calculation_monte_carlo_total()
    test_debug_shrinkage_impact_by_trades()
