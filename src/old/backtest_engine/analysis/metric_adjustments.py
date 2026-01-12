"""
Metric adjustment functions for trade-count based shrinkage and Bayesian adjustments.

This module provides utilities to adjust trading metrics (Average R, Profit over Drawdown, Winrate)
based on the number of trades, applying shrinkage towards a reference to account for statistical
reliability with smaller sample sizes.
"""

from typing import List, Tuple, Union

import numpy as np

# Konfigurierbare Konstante: Anzahl Trades pro Jahr als Referenzwert
TRADES_PER_YEAR_REFERENCE = 15


def shrinkage_adjusted(
    average_r: Union[float, np.ndarray],
    n_trades: Union[int, np.ndarray],
    n_years: Union[float, np.ndarray] = 1.0,
    *,
    n_categories: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Berechnet den shrinkage-adjusted Average R basierend auf der Trade-Anzahl.

    Formel: average_r_adjust = average_r * (N / (N + konst.))

    wobei:
    - N = Anzahl der Trades
    - konst. = n_years * n_categories * TRADES_PER_YEAR_REFERENCE

    Args:
        average_r: Der rohe Average R-Multiple Wert
        n_trades: Anzahl der Trades
        n_years: Anzahl der Jahre im Backtest-Zeitraum (default: 1.0 für yearly metrics)
             Für total metrics: Gesamtanzahl Jahre des Backtests
        n_categories: Skalierungsfaktor für die Anzahl unabhängiger Kategorien/Groups
                  (z.B. Portfolio-legs). Default: 1.0 (keine Änderung gegenüber
                  historischer Logik).

    Returns:
        Adjusted Average R-Multiple
    """
    if isinstance(average_r, (int, float)) and not np.isfinite(average_r):
        return average_r

    n_years = np.asarray(n_years, dtype=float)
    n_categories = np.asarray(n_categories, dtype=float)
    n_years = np.where(np.isfinite(n_years) & (n_years > 0.0), n_years, 1.0)
    n_categories = np.where(
        np.isfinite(n_categories) & (n_categories > 0.0), n_categories, 1.0
    )

    konst = n_years * n_categories * TRADES_PER_YEAR_REFERENCE
    n_trades = np.asarray(n_trades, dtype=float)
    # Guardrail: negative trade counts are treated as zero (prevents negative/ >1 factors)
    n_trades = np.maximum(n_trades, 0.0)
    average_r = np.asarray(average_r, dtype=float)

    # Avoid division by zero
    denominator = n_trades + konst
    shrinkage_factor = np.where(denominator > 0, n_trades / denominator, 0.0)

    adjusted = average_r * shrinkage_factor

    # Preserve NaN/Inf from input
    adjusted = np.where(np.isfinite(average_r), adjusted, average_r)

    # Return scalar if input was scalar
    if adjusted.shape == ():
        return float(adjusted)
    return adjusted


def risk_adjusted(
    profit_over_drawdown: Union[float, np.ndarray],
    n_trades: Union[int, np.ndarray],
    n_years: Union[float, np.ndarray] = 1.0,
    *,
    n_categories: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Berechnet den risk-adjusted Profit over Drawdown basierend auf der Trade-Anzahl.

    Formel: profit_over_drawdown_adjust = profit_over_drawdown * sqrt(N / (N + konst.))

    wobei:
    - N = Anzahl der Trades
    - konst. = n_years * n_categories * TRADES_PER_YEAR_REFERENCE

    Args:
        profit_over_drawdown: Der rohe Profit over Drawdown Wert
        n_trades: Anzahl der Trades
        n_years: Anzahl der Jahre im Backtest-Zeitraum (default: 1.0 für yearly metrics)
             Für total metrics: Gesamtanzahl Jahre des Backtests
        n_categories: Skalierungsfaktor für die Anzahl unabhängiger Kategorien/Groups
                  (z.B. Portfolio-legs). Default: 1.0 (keine Änderung gegenüber
                  historischer Logik).

    Returns:
        Adjusted Profit over Drawdown
    """
    if isinstance(profit_over_drawdown, (int, float)) and not np.isfinite(
        profit_over_drawdown
    ):
        return profit_over_drawdown

    n_years = np.asarray(n_years, dtype=float)
    n_categories = np.asarray(n_categories, dtype=float)
    n_years = np.where(np.isfinite(n_years) & (n_years > 0.0), n_years, 1.0)
    n_categories = np.where(
        np.isfinite(n_categories) & (n_categories > 0.0), n_categories, 1.0
    )

    konst = n_years * n_categories * TRADES_PER_YEAR_REFERENCE
    n_trades = np.asarray(n_trades, dtype=float)
    # Guardrail: negative trade counts are treated as zero
    n_trades = np.maximum(n_trades, 0.0)
    profit_over_drawdown = np.asarray(profit_over_drawdown, dtype=float)

    # Avoid division by zero
    denominator = n_trades + konst
    shrinkage_factor = np.where(denominator > 0, np.sqrt(n_trades / denominator), 0.0)

    adjusted = profit_over_drawdown * shrinkage_factor

    # Preserve NaN/Inf from input
    adjusted = np.where(
        np.isfinite(profit_over_drawdown), adjusted, profit_over_drawdown
    )

    # Return scalar if input was scalar
    if adjusted.shape == ():
        return float(adjusted)
    return adjusted


def bayesian_shrinkage(
    winrate: Union[float, np.ndarray],
    n_trades: Union[int, np.ndarray],
    all_winrates: Union[List[float], np.ndarray],
    *,
    clamp_to_raw: bool = False,
) -> Union[float, np.ndarray]:
    """
    Berechnet die Bayesian shrinkage-adjusted Winrate basierend auf der Trade-Anzahl.

    Formel: winrate_adjust = (wins + alpha) / (n + alpha + beta)

    wobei:
    - wins = winrate * n_trades
    - alpha = winrate_mean * (((winrate_mean * (1 - winrate_mean)) / s^2) - 1)
    - beta = (1 - winrate_mean) * (((winrate_mean * (1 - winrate_mean)) / s^2) - 1)
    - winrate_mean = Mittelwert aller verfügbaren Winrates
    - s^2 = Varianz aller verfügbaren Winrates

    Args:
        winrate: Die rohe Winrate (als Dezimalzahl, z.B. 0.55 für 55%)
        n_trades: Anzahl der Trades
        all_winrates: Liste oder Array aller verfügbaren Winrates zur Berechnung
                      von Prior-Parametern (als Dezimalzahlen)
        clamp_to_raw: Wenn True, wird das adjustierte Ergebnis pro Element auf
                       den Rohwert nach oben gekappt (konservativer Modus)

    Returns:
        Adjusted Winrate (als Dezimalzahl)

    Note:
        - Winrate wird als Dezimalzahl erwartet (0.0 - 1.0), nicht als Prozent
        - all_winrates sollte alle relevanten Winrates enthalten (z.B. alle Pairs im gleichen Jahr
          für yearly metrics, oder alle total_winrates für total metrics)
    """
    if isinstance(winrate, (int, float)) and not np.isfinite(winrate):
        return winrate

    # Convert to numpy arrays and preserve original non-finite inputs.
    # IMPORTANT: winrate is later clipped to [0,1], so we need to restore
    # NaN/Inf from the original input (also for array inputs).
    winrate_in = np.asarray(winrate, dtype=float)
    n_trades_in = np.asarray(n_trades, dtype=float)
    winrate, n_trades = np.broadcast_arrays(winrate_in, n_trades_in)
    winrate_orig = winrate.copy()
    n_trades_orig = n_trades.copy()
    non_finite_mask = ~np.isfinite(winrate_orig) | ~np.isfinite(n_trades_orig)

    all_winrates = np.asarray(all_winrates, dtype=float)

    # Guardrails: keine negativen Trades, Winrates korrekt skalieren
    n_trades = np.maximum(n_trades, 0.0)

    def _normalize_wr(arr: np.ndarray) -> np.ndarray:
        """Toleriert %-Eingaben (0-100) und wandelt bei Bedarf in Dezimal um."""
        # ignore NaN/Inf for the check
        finite_max = (
            np.nanmax(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else np.nan
        )
        if np.isfinite(finite_max) and finite_max > 1.0 + 1e-9:
            return arr / 100.0
        return arr

    winrate = _normalize_wr(winrate)
    all_winrates = _normalize_wr(all_winrates)

    # Nach Normierung auf gültigen Bereich beschränken
    winrate = np.clip(winrate, 0.0, 1.0)
    all_winrates = np.clip(all_winrates, 0.0, 1.0)

    # Filter out non-finite values from all_winrates for computing prior
    valid_winrates = all_winrates[np.isfinite(all_winrates)]

    if len(valid_winrates) == 0:
        # No valid data for prior - return original
        out = winrate
        out = np.where(non_finite_mask, winrate_orig, out)
        if out.shape == ():
            return float(out)
        return out

    # Compute prior parameters from all available winrates
    winrate_mean = np.mean(valid_winrates)
    winrate_var = np.var(valid_winrates, ddof=1) if len(valid_winrates) > 1 else 0.001

    # Avoid division by zero in variance
    if winrate_var <= 0:
        winrate_var = 0.001  # Small epsilon to avoid numerical issues

    # Beta distribution parameters
    # alpha = winrate_mean * (((winrate_mean * (1 - winrate_mean)) / s^2) - 1)
    # beta = (1 - winrate_mean) * (((winrate_mean * (1 - winrate_mean)) / s^2) - 1)

    mean_var_ratio = (winrate_mean * (1.0 - winrate_mean)) / winrate_var

    # Ensure mean_var_ratio > 1 for valid beta distribution
    # If variance is too large, use minimum sensible value
    mean_var_ratio = np.maximum(mean_var_ratio, 1.01)

    alpha = winrate_mean * (mean_var_ratio - 1.0)
    beta_param = (1.0 - winrate_mean) * (mean_var_ratio - 1.0)

    # Ensure alpha and beta are positive
    alpha = np.maximum(alpha, 0.01)
    beta_param = np.maximum(beta_param, 0.01)

    # Calculate wins from winrate and n_trades
    wins = winrate * n_trades

    # Bayesian adjustment: (wins + alpha) / (n + alpha + beta)
    denominator = n_trades + alpha + beta_param
    adjusted = np.where(
        denominator > 0,
        (wins + alpha) / denominator,
        winrate,  # Fallback to original if denominator is zero
    )

    # Konservativer Modus: Adjusted darf nicht über den Rohwert steigen
    if clamp_to_raw:
        adjusted = np.where(
            np.isfinite(adjusted) & np.isfinite(winrate),
            np.minimum(adjusted, winrate),
            adjusted,
        )

    # Preserve NaN/Inf from original input
    adjusted = np.where(non_finite_mask, winrate_orig, adjusted)

    # Return scalar if input was scalar
    if adjusted.shape == ():
        return float(adjusted)
    return adjusted


# -----------------------------------------------------------------------------
# Wilson Score vs. Bayesian Shrinkage - Wann welche Methode verwenden?
# -----------------------------------------------------------------------------
# Wilson Score Lower Bound:
#   - Frequentistischer Ansatz: Gibt ein Konfidenzintervall an
#   - Konservative "Worst-Case" Schätzung der wahren Winrate
#   - Ideal für Risiko-averse Entscheidungen ("Was ist die minimal plausible Winrate?")
#   - Keine Prior-Annahmen nötig, nur beobachtete Daten
#   - Besonders robust bei extremen Winrates (nahe 0 oder 1)
#
# Bayesian Shrinkage:
#   - Bayesianischer Ansatz: Kombiniert Prior mit beobachteten Daten
#   - Shrinkage zum Populationsmittel (z.B. Durchschnitt aller Strategien)
#   - Ideal wenn Prior-Information verfügbar ist (z.B. historische Winrates)
#   - Glättet Schätzungen hin zum erwarteten Mittelwert
#   - Gut für Vergleiche zwischen vielen Strategien mit unterschiedlichen Sample-Sizes
# -----------------------------------------------------------------------------


def wilson_score_lower_bound(
    winrate: Union[float, np.ndarray],
    n_trades: Union[int, np.ndarray],
    confidence_level: float = 0.95,
    *,
    return_full_interval: bool = False,
) -> Union[
    float,
    np.ndarray,
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]],
]:
    """
    Berechnet die untere Grenze des Wilson Score Konfidenzintervalls für die Winrate.

    Das Wilson Score Interval ist eine robuste Methode zur Schätzung des Konfidenzintervalls
    für Binomialproportionen. Im Gegensatz zur Normal-Approximation funktioniert es auch
    bei extremen Winrates (nahe 0 oder 1) und kleinen Sample-Sizes zuverlässig.

    Formel (Wilson Score Interval):
        z = Z-Score für gewähltes Konfidenz-Level (z.B. 1.96 für 95%)

        lower_bound = (p + z²/(2n) - z * sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)
        upper_bound = (p + z²/(2n) + z * sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)

        wobei:
        - p = beobachtete Winrate
        - n = Anzahl Trades
        - z = Z-Score (1.645 für 90%, 1.96 für 95%, 2.576 für 99%)

    Args:
        winrate: Die beobachtete Winrate (als Dezimalzahl 0.0-1.0 oder Prozent 0-100)
        n_trades: Anzahl der Trades (Sample Size)
        confidence_level: Konfidenz-Level für das Intervall (default: 0.95 = 95%)
        return_full_interval: Wenn True, gibt ein Tuple zurück mit
                              (lower_bound, upper_bound, confidence_width)

    Returns:
        Wenn return_full_interval=False (default):
            Untere Grenze des Konfidenzintervalls (float oder np.ndarray)
        Wenn return_full_interval=True:
            Tuple von (lower_bound, upper_bound, confidence_width)

    Note:
        - Sehr konservative Schätzung bei kleinen Samples - die untere Grenze
          kann deutlich unter der beobachteten Winrate liegen.
        - Bei 10 Trades mit 80% Winrate liegt Lower Bound (95% CI) oft unter 50%!
          Dies reflektiert die hohe Unsicherheit bei kleinen Sample-Sizes.
        - Mathematisch robuster als Normal-Approximation für extreme Winrates
          (nahe 0% oder 100%), da es die Binomialverteilung korrekt berücksichtigt.
        - Winrate wird als Dezimalzahl erwartet (0.0 - 1.0), aber Prozent-Eingaben
          (0-100) werden automatisch erkannt und konvertiert.

    Example:
        >>> wilson_score_lower_bound(0.80, 10)  # 80% Winrate, 10 Trades
        0.4901...  # Lower Bound bei 95% Konfidenz

        >>> wilson_score_lower_bound(0.80, 100)  # 80% Winrate, 100 Trades
        0.7112...  # Höherer Lower Bound durch mehr Trades

        >>> wilson_score_lower_bound(0.80, 10, return_full_interval=True)
        (0.4901..., 0.9433..., 0.4531...)  # (lower, upper, width)
    """
    from statistics import NormalDist

    # Handle scalar non-finite input early
    if isinstance(winrate, (int, float)) and not np.isfinite(winrate):
        if return_full_interval:
            return (winrate, winrate, np.nan)
        return winrate

    # Convert to numpy arrays for vectorized operations
    winrate = np.asarray(winrate, dtype=float)
    n_trades = np.asarray(n_trades, dtype=float)

    # Store original shape to determine return type
    is_scalar = winrate.shape == () and n_trades.shape == ()

    # Broadcast arrays to same shape if needed
    winrate, n_trades = np.broadcast_arrays(winrate, n_trades)

    # Input normalization: Accept both decimal (0-1) and percent (0-100)
    def _normalize_winrate(arr: np.ndarray) -> np.ndarray:
        """Toleriert %-Eingaben (0-100) und wandelt bei Bedarf in Dezimal um."""
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) > 0:
            finite_max = np.nanmax(finite_vals)
            if finite_max > 1.0 + 1e-9:
                return arr / 100.0
        return arr

    winrate = _normalize_winrate(winrate)

    # Clip winrate to valid range [0, 1]
    winrate_clipped = np.clip(winrate, 0.0, 1.0)

    # Ensure n_trades is non-negative
    n_trades = np.maximum(n_trades, 0.0)

    # Calculate Z-Score from confidence level
    # For two-tailed interval: z = norm.ppf(1 - (1-confidence)/2) = norm.ppf((1+confidence)/2)
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be in the open interval (0, 1)")
    z = NormalDist().inv_cdf((1.0 + float(confidence_level)) / 2.0)
    if not np.isfinite(z):
        raise ValueError(
            "confidence_level results in a non-finite z-score; choose a value further from 0/1"
        )

    # Pre-compute common terms for efficiency
    z_squared = z * z
    p = winrate_clipped
    n = n_trades

    # Replace zeros with 1 to avoid division warnings (results will be masked anyway)
    n_safe = np.where(n > 0, n, 1.0)

    # Wilson Score Interval formula components
    # Numerator term: z²/(2n)
    z_sq_over_2n = z_squared / (2.0 * n_safe)

    # Denominator: 1 + z²/n
    denominator = 1.0 + z_squared / n_safe

    # Variance term under square root: p(1-p)/n + z²/(4n²)
    variance_term = (p * (1.0 - p)) / n_safe + z_squared / (4.0 * n_safe * n_safe)

    # Square root of variance term (ensure non-negative for numerical stability)
    sqrt_term = np.sqrt(np.maximum(variance_term, 0.0))

    # Center of the interval: p + z²/(2n)
    center = p + z_sq_over_2n

    # Margin: z * sqrt(variance_term)
    margin = z * sqrt_term

    # Calculate lower and upper bounds
    lower_bound = np.where(
        n > 0, (center - margin) / denominator, 0.0  # Return 0 when n_trades = 0
    )

    upper_bound = np.where(
        n > 0, (center + margin) / denominator, 0.0  # Return 0 when n_trades = 0
    )

    # Clip bounds to valid range [0, 1] for numerical stability
    lower_bound = np.clip(lower_bound, 0.0, 1.0)
    upper_bound = np.clip(upper_bound, 0.0, 1.0)

    # Confidence width
    confidence_width = upper_bound - lower_bound

    # Preserve NaN/Inf from original input
    non_finite_mask = ~np.isfinite(winrate) | ~np.isfinite(n_trades)
    lower_bound = np.where(non_finite_mask, winrate, lower_bound)
    upper_bound = np.where(non_finite_mask, winrate, upper_bound)
    confidence_width = np.where(non_finite_mask, np.nan, confidence_width)

    # Convert back to scalar if input was scalar
    if is_scalar:
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
        confidence_width = float(confidence_width)

    if return_full_interval:
        return (lower_bound, upper_bound, confidence_width)

    return lower_bound
