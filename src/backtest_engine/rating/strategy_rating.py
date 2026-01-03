from typing import Any, Dict, Optional


def rate_strategy_performance(
    summary: Dict[str, Any], thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Bewertet die Strategie-Performance anhand fester Schwellenwerte.
    Gibt Score (0–1), Deployment-Entscheidung und Fehlerschlüssel zurück.

    Args:
        summary: Dict mit den wichtigsten Metriken einer Backtest-Periode
            (Keys: "Winrate (%)", "Avg R-Multiple", "Net Profit", "profit_factor", "drawdown_eur").
        thresholds: Optionale Overrides für die Bewertungsschwellen.

    Returns:
        Dict mit:
            - Score: Anteil bestandener Kriterien (0.0–1.0)
            - Deployment: True/False, ob Deployment erlaubt ist
            - Deployment_Fails: Pipe-separierte String-Liste fehlender Kriterien
    """
    thresholds = thresholds or {
        "min_winrate": 45,
        "min_avg_r": 0.6,
        "min_profit": 500,
        "min_profit_factor": 1.2,
        "max_drawdown": 1000,
    }
    score = 0.0
    deployment = True
    checks = []

    if summary.get("Winrate (%)", 0) < thresholds["min_winrate"]:
        deployment = False
        checks.append("Winrate")
    if summary.get("Avg R-Multiple", 0) < thresholds["min_avg_r"]:
        deployment = False
        checks.append("Avg R")
    if summary.get("Net Profit", 0) < thresholds["min_profit"]:
        deployment = False
        checks.append("Net Profit")
    if summary.get("profit_factor", 0) < thresholds["min_profit_factor"]:
        deployment = False
        checks.append("Profit Factor")
    if summary.get("drawdown_eur", 0) > thresholds["max_drawdown"]:
        deployment = False
        checks.append("Drawdown")

    # Score: Anteil erfüllter Kriterien
    score = 1 - len(checks) / 5

    return {
        "Score": round(score, 2),
        "Deployment": deployment,
        "Deployment_Fails": "|".join(checks),
    }
