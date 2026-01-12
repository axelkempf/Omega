"""
Globale Settings & Defaults für Walkforward-Pipeline.
Alle Flags sind optional – ohne Aktivierung greifen Defaults ohne Effekt.
"""

from typing import Any, Dict

SELECTION_DEFAULTS: Dict[str, Any] = {
    # Auswahl-Strategien
    "use_grid_search": False,
    "use_zone_refinement": False,
    "zone_refinement_trials": 180,
    # === NEU: IS-Filter für Kandidatenauswahl ===
    "is_filter": {
        "gates": {
            "profit_min": 0.0,  # nur profitabel
            "avg_r_min": 0.05,
            # "winrate_min": 0.0,
            # "drawdown_max": float("inf"),
            # "robustness_min": 0.6,  # robustness_score ≥ 0.5
            # "min_trades": 10,  # mind. 10 Trades
        },
        "topN": 150,  # Anzahl Kandidaten, die ins OOS geschickt werden
    },
}


def get_selection_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Liefert Settings als Dict.
    Falls Overrides (z. B. aus JSON-Config) vorhanden, überschreiben diese Defaults.
    """
    cfg = dict(SELECTION_DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                merged = dict(cfg[k])
                merged.update(v)
                cfg[k] = merged
            else:
                cfg[k] = v
    return cfg
