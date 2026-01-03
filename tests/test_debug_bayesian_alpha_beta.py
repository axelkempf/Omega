"""
Debug-Test: Bayesian Shrinkage Alpha/Beta Berechnung
======================================================

Zeigt die berechneten Alpha und Beta Werte für verschiedene Winrate-Verteilungen
und deren Auswirkungen auf die Adjusted Winrate.
"""

import numpy as np
import pytest

from backtest_engine.analysis.metric_adjustments import bayesian_shrinkage


def compute_alpha_beta_manually(all_winrates):
    """Manuelle Berechnung von Alpha und Beta wie in der Produktionsfunktion."""
    valid_winrates = np.asarray(all_winrates, dtype=float)[
        np.isfinite(np.asarray(all_winrates, dtype=float))
    ]

    if len(valid_winrates) == 0:
        return None, None

    winrate_mean = np.mean(valid_winrates)
    winrate_var = np.var(valid_winrates, ddof=1) if len(valid_winrates) > 1 else 0.001

    if winrate_var <= 0:
        winrate_var = 0.001

    mean_var_ratio = (winrate_mean * (1.0 - winrate_mean)) / winrate_var
    mean_var_ratio = np.maximum(mean_var_ratio, 1.01)

    alpha = winrate_mean * (mean_var_ratio - 1.0)
    beta_param = (1.0 - winrate_mean) * (mean_var_ratio - 1.0)

    alpha = np.maximum(alpha, 0.01)
    beta_param = np.maximum(beta_param, 0.01)

    return alpha, beta_param, winrate_mean, mean_var_ratio


def test_debug_bayesian_alpha_beta_low_variance():
    """Test: Sehr homogene Winrates (alle ähnlich) = starke Prior-Prägung."""
    print("\n" + "=" * 80)
    print("SZENARIO 1: Homogene Winrate-Verteilung (alle Pairs ähnlich)")
    print("=" * 80)

    # Alle Pairs haben ähnliche Winrates (homogen)
    all_winrates_decimal = [0.52, 0.53, 0.51, 0.54, 0.52]  # Dezimal, wie in Produktion

    alpha, beta_param, mean, ratio = compute_alpha_beta_manually(all_winrates_decimal)

    print(f"\nPrior-Daten:")
    print(f"  Winrates:          {[f'{w*100:.1f}%' for w in all_winrates_decimal]}")
    print(f"  Mean:              {mean*100:.2f}%")
    print(f"  Varianz:           {np.var(all_winrates_decimal, ddof=1):.6f}")
    print(f"  Mean/Var Ratio:    {ratio:.2f}")
    print(f"  Alpha:             {alpha:.3f}")
    print(f"  Beta:              {beta_param:.3f}")
    print(f"  Alpha/Beta Ratio:  {alpha/beta_param:.3f}")

    # Test verschiedene Raw Winrates
    test_cases = [
        (0.45, 10),  # Niedrig, wenige Trades
        (0.52, 10),  # Gleich Mean, wenige Trades
        (0.60, 10),  # Hoch, wenige Trades
        (0.45, 100),  # Niedrig, viele Trades
        (0.60, 100),  # Hoch, viele Trades
    ]

    print(
        f"\nAdjustierte Winrates (homogene Prior mit alpha={alpha:.3f}, beta={beta_param:.3f}):"
    )
    print(
        f"{'Raw WR%':<10} {'Trades':<10} {'Adjusted WR%':<15} {'Change':<12} {'Effekt'}"
    )
    print("-" * 70)

    for raw_wr, n_trades in test_cases:
        adjusted = bayesian_shrinkage(raw_wr, n_trades, all_winrates_decimal)
        change_pct = (adjusted - raw_wr) * 100
        direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
        effekt = (
            f"Shrink zu {mean*100:.2f}% (homogen)"
            if n_trades == 10
            else f"Minimal (n={n_trades})"
        )

        print(
            f"{raw_wr*100:>8.1f}% {n_trades:>9} {adjusted*100:>13.2f}% {direction}{abs(change_pct):>9.2f}pp  {effekt}"
        )

    print("\n✓ Mit homogener Prior: Alle Werte werden stark zu 52% gezogen")


def test_debug_bayesian_alpha_beta_high_variance():
    """Test: Heterogene Winrates (breite Streuung) = schwächere Prior-Prägung."""
    print("\n" + "=" * 80)
    print("SZENARIO 2: Heterogene Winrate-Verteilung (breite Streuung)")
    print("=" * 80)

    # Sehr unterschiedliche Winrates (heterogen)
    all_winrates_decimal = [0.35, 0.55, 0.65, 0.45, 0.70]  # Breite Streuung

    alpha, beta_param, mean, ratio = compute_alpha_beta_manually(all_winrates_decimal)

    print(f"\nPrior-Daten:")
    print(f"  Winrates:          {[f'{w*100:.1f}%' for w in all_winrates_decimal]}")
    print(f"  Mean:              {mean*100:.2f}%")
    print(f"  Varianz:           {np.var(all_winrates_decimal, ddof=1):.6f}")
    print(f"  Mean/Var Ratio:    {ratio:.2f}")
    print(f"  Alpha:             {alpha:.3f}")
    print(f"  Beta:              {beta_param:.3f}")
    print(f"  Alpha/Beta Ratio:  {alpha/beta_param:.3f}")

    # Test verschiedene Raw Winrates
    test_cases = [
        (0.35, 10),  # Niedrig wie Min der Prior
        (0.54, 10),  # Gleich Mean, wenige Trades
        (0.70, 10),  # Hoch wie Max der Prior
        (0.35, 100),  # Niedrig, viele Trades
        (0.70, 100),  # Hoch, viele Trades
    ]

    print(
        f"\nAdjustierte Winrates (heterogene Prior mit alpha={alpha:.3f}, beta={beta_param:.3f}):"
    )
    print(
        f"{'Raw WR%':<10} {'Trades':<10} {'Adjusted WR%':<15} {'Change':<12} {'Effekt'}"
    )
    print("-" * 70)

    for raw_wr, n_trades in test_cases:
        adjusted = bayesian_shrinkage(raw_wr, n_trades, all_winrates_decimal)
        change_pct = (adjusted - raw_wr) * 100
        direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
        effekt = (
            f"Schwach zu {mean*100:.2f}% (heterogen)"
            if n_trades == 10
            else f"Minimal (n={n_trades})"
        )

        print(
            f"{raw_wr*100:>8.1f}% {n_trades:>9} {adjusted*100:>13.2f}% {direction}{abs(change_pct):>9.2f}pp  {effekt}"
        )

    print("\n✓ Mit heterogener Prior: Weniger starke Shrinkage zu Prior-Mean")


def test_debug_bayesian_compare_alpha_beta_effects():
    """Test: Vergleich wie unterschiedliche Alpha/Beta ratios die Adjustment beeinflussen."""
    print("\n" + "=" * 80)
    print("SZENARIO 3: Auswirkung von Alpha/Beta Verhältnis auf Adjustment")
    print("=" * 80)

    scenarios = {
        "Stark asymmetrisch (Bias zu niedrig)": [
            0.30,
            0.35,
            0.32,
            0.28,
            0.33,
        ],  # Mean ≈ 0.32
        "Ausgewogen (Mean ≈ 0.50)": [0.48, 0.52, 0.50, 0.49, 0.51],  # Mean ≈ 0.50
        "Stark asymmetrisch (Bias zu hoch)": [
            0.68,
            0.72,
            0.70,
            0.69,
            0.71,
        ],  # Mean ≈ 0.70
    }

    raw_wr = 0.50  # Gleichbleibender Raw Value
    n_trades = 20

    print(
        f"\nVergleich bei festem Raw Winrate={raw_wr*100:.0f}% und {n_trades} Trades:\n"
    )

    for scenario_name, all_winrates in scenarios.items():
        alpha, beta_param, mean, ratio = compute_alpha_beta_manually(all_winrates)
        adjusted = bayesian_shrinkage(raw_wr, n_trades, all_winrates)
        change_pct = (adjusted - raw_wr) * 100
        direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="

        print(f"{scenario_name}")
        print(f"  Prior Mean:       {mean*100:.1f}%")
        print(f"  Alpha:            {alpha:.3f}")
        print(f"  Beta:             {beta_param:.3f}")
        print(
            f"  α/β Ratio:        {alpha/beta_param:.3f} (>1=Bias zu Erfolg, <1=Bias zu Misserfolg)"
        )
        print(f"  Raw WR:           {raw_wr*100:.1f}%")
        print(
            f"  Adjusted WR:      {adjusted*100:.2f}% ({direction}{abs(change_pct):.2f}pp)"
        )
        print(f"  Shrinkage Ziel:   {mean*100:.1f}% (die Prior-Mitte)")
        print()

    print("✓ Das α/β Verhältnis bestimmt, wie asymmetrisch die Prior-Prägung ist")


def test_debug_prior_strength_by_sample_size():
    """Test: Zeigt wie die Prior-Stärke von der Streuung der all_winrates abhängt."""
    print("\n" + "=" * 80)
    print("SZENARIO 4: Prior-Stärke abhängig von Daten-Variabilität")
    print("=" * 80)

    # Gleiche Mean, aber unterschiedliche Variance
    test_priors = {
        "Sehr zuverlässig (σ²=0.0001)": np.random.normal(
            0.50, 0.01, 100
        ),  # Kleine Streuung
        "Mittel zuverlässig (σ²=0.001)": np.random.normal(
            0.50, 0.03, 50
        ),  # Mittlere Streuung
        "Unsicher (σ²=0.01)": np.random.normal(0.50, 0.10, 20),  # Große Streuung
    }

    raw_wr = 0.45
    n_trades = 15

    print(f"\nRaw Winrate={raw_wr*100:.0f}%, Trades={n_trades}\n")
    print(f"{'Prior Charakteristik':<30} {'α/β':<8} {'Adjusted':<12} {'Shrinkage':<12}")
    print("-" * 70)

    for desc, winrates in test_priors.items():
        winrates_clipped = np.clip(winrates, 0.0, 1.0)  # Halten im Bereich [0,1]
        alpha, beta_param, mean, ratio = compute_alpha_beta_manually(winrates_clipped)
        adjusted = bayesian_shrinkage(raw_wr, n_trades, winrates_clipped)
        change_pct = (adjusted - raw_wr) * 100

        print(
            f"{desc:<30} {alpha/beta_param:>6.2f}  {adjusted*100:>10.2f}% {change_pct:>10.2f}pp"
        )

    print("\n✓ Reliablere Prior (kleine Varianz) → stärker Shrinkage")
    print("✓ Unsichere Prior (große Varianz) → schwächer Shrinkage")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
