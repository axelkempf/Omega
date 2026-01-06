import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def select_best_strategies(
    rating_files: List[str], min_score: float = 1.0, min_windows: int = 2
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregiert Deployment-Ratings aus mehreren Symbol-Rating-Dateien
    und erstellt eine Deployment-Empfehlung.

    Args:
        rating_files: Liste von Pfaden zu JSON-Dateien (z.B. "EURUSD_ratings.json").
        min_score: Mindestdurchschnitts-Score für Deployment (z.B. 0.9).
        min_windows: Mindestanzahl erfolgreicher Zeitfenster für Deployment.

    Returns:
        Dict je Symbol mit Gesamtmetriken und finaler Deployment-Empfehlung.
    """
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"scores": [], "deployments": 0}
    )

    for path in rating_files:
        symbol = os.path.basename(path).split("_")[0]
        try:
            with open(path) as f:
                entries = json.load(f)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden: {path} – {e}")
            continue

        for entry in entries:
            stats[symbol]["scores"].append(entry.get("Score", 0))
            if entry.get("Deployment") is True:
                stats[symbol]["deployments"] += 1

    results = {}
    for symbol, values in stats.items():
        count = len(values["scores"])
        avg_score = round(sum(values["scores"]) / count, 2) if count > 0 else 0.0
        deploy_count = values["deployments"]

        results[symbol] = {
            "Total Windows": count,
            "Deployable Windows": deploy_count,
            "Avg Score": avg_score,
            "Final Deployment": deploy_count >= min_windows and avg_score >= min_score,
        }

    return results


def export_deployment_summary(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "walkforward/deployment_summary.json",
) -> None:
    """
    Speichert das Ergebnis der Deployment-Selektion als JSON-Datei.

    Args:
        results: Ergebnisdict von select_best_strategies().
        output_path: Pfad zur Output-JSON.

    Prints:
        Statusmeldung beim erfolgreichen Schreiben.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Deployment-Summary gespeichert: {output_path}")
