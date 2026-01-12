import csv
import itertools
import json
import os
from copy import deepcopy
from typing import Any, Callable


def generate_param_combinations(
    param_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """
    Erzeugt alle möglichen Parameterkombinationen aus einem Grid.

    Args:
        param_grid: Dict mit Parameternamen als Keys und Listen möglicher Werte.

    Returns:
        Liste aller Kombinations-Parameter als Dict.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def run_grid_search(
    config_template_path: str,
    param_grid: dict[str, list[Any]],
    output_dir: str = "grid_results",
    strategy_key: str = "strategy",
    result_csv: str = "grid_results/summary.csv",
    min_criteria: dict[str, float] | None = None,
    sort_by: str = "Avg R-Multiple",
    top_n: int = 5,
    base_config: dict[str, Any] | None = None,
    config_overwrite: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """
    Führt eine vollständige Grid Search mit Backtest und Metrik-Auswertung durch.

    Args:
        config_template_path: Pfad zur JSON-Config-Vorlage.
        param_grid: Dict mit Parameternamen und Wertlisten.
        output_dir: Verzeichnis für Ergebnisse.
        strategy_key: Key zur Strategie im Config-Dict.
        result_csv: Output-Pfad für die Ergebnis-CSV.
        min_criteria: Optionales Dict für Mindestmetrik-Filter (z.B. {"Winrate (%)": 50}).
        sort_by: Metrik, nach der die Top-N sortiert werden.
        top_n: Anzahl der Top-Ergebnisse für die Anzeige.
        base_config: Optionales Dict mit Config (falls nicht aus File).

    Prints:
        Fortschritt und Status für jede getestete Kombination.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lade Config-Vorlage, falls nötig
    if base_config is None:
        with open(config_template_path, "r") as f:
            base_config = json.load(f)
    # einmalig überschreiben (z.B. Symbol setzen), falls gewünscht
    if config_overwrite is not None:
        base_config = config_overwrite(deepcopy(base_config))

    combinations = generate_param_combinations(param_grid)
    print(f"🔁 Starte Grid-Search mit {len(combinations)} Kombinationen...\n")

    fieldnames = list(param_grid.keys()) + [
        "Net Profit",
        "Avg R-Multiple",
        "Winrate (%)",
        "Drawdown",
        "Total Trades",
    ]

    results: list[dict[str, Any]] = []

    for i, params in enumerate(combinations):
        print(f"🔹 [{i+1}/{len(combinations)}] Parameter: {params}")
        config = deepcopy(base_config)
        # optional pro-Kombi nochmal überschreiben (selten nötig)
        if config_overwrite is not None:
            config = config_overwrite(config)
        config[strategy_key]["parameters"].update(params)

        # Lokale Imports für speed-up und Testbarkeit
        from backtest_engine.report.metrics import calculate_metrics
        from backtest_engine.runner import run_backtest_and_return_portfolio

        try:
            portfolio, _ = run_backtest_and_return_portfolio(
                config
            )  # korrektes Unpacking
            summary = calculate_metrics(portfolio)
            # Metriken auf lesbare Spaltennamen mappen (wie im Walkforward)
            row_metrics = {
                "Net Profit": float(summary.get("net_profit_after_fees_eur", 0.0)),
                "Avg R-Multiple": float(summary.get("avg_r_multiple", 0.0)),
                "Winrate (%)": float(summary.get("winrate_percent", 0.0)),
                "Drawdown": float(summary.get("drawdown_eur", 0.0)),
                "Total Trades": int(summary.get("total_trades", 0) or 0),
            }
            row = {**params, **row_metrics}
            # Mindestkriterien (optional) – z.B. {"Net Profit": 0, "Winrate (%)": 45}
            if min_criteria:
                for k, v in min_criteria.items():
                    if k in row and not (row[k] >= v):
                        print(f"⛔️ Verworfen (Kriterium nicht erfüllt {k}>={v}): {row}")
                        break
                else:
                    results.append(row)
            else:
                results.append(row)
        except Exception as ex:
            print(f"❌ Fehler in Kombination {i+1}: {params} — {ex}")
            continue

    # Ergebnisse sortieren
    results_sorted = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=True)

    # In CSV schreiben
    with open(result_csv, mode="w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"\n📄 Grid-Search abgeschlossen. Ergebnisse in: {result_csv}")

    # Top-N anzeigen
    print(f"\n🏆 Top {top_n} nach {sort_by}:")
    for i, row in enumerate(results_sorted[:top_n]):
        print(f"{i+1}. {row}")
