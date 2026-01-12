from typing import Any, Dict, List

import matplotlib.pyplot as plt


def save_walkforward_plot(
    results: List[Dict[str, Any]],
    metric: str = "Avg R-Multiple",
    output_path: str = "walkforward_plot.pdf",
) -> None:
    """
    Erstellt und speichert einen Walkforward-Plot für einen bestimmten Metrik-Verlauf über alle Testfenster.

    Args:
        results: Liste von Result-Dicts (je Walkforward-Fenster).
        metric: Der zu plottende Metrik-Name (z.B. "Avg R-Multiple", "Net Profit", ...).
        output_path: Zielpfad für den PDF-Export.

    Prints:
        Statusmeldungen zum Export.
    """
    if not results:
        print("❌ Keine Walkforward-Ergebnisse vorhanden.")
        return

    timestamps = []
    values = []

    for i, r in enumerate(results):
        ts = r.get("Window") or r.get("test_window") or f"Window {i+1}"
        try:
            values.append(float(r.get(metric, 0)))
        except Exception:
            values.append(0.0)
        timestamps.append(ts)

    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, values, marker="o", linewidth=2)
    plt.title(f"Walkforward – {metric} über Testfenster")
    plt.xlabel("Zeitraum")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"✅ Walkforward-Plot gespeichert als: {output_path}")
