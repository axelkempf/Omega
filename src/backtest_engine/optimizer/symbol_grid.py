from typing import Callable, Dict, List, Optional

from backtest_engine.optimizer.grid_searcher import run_grid_search


def run_grid_search_multi_symbol(
    symbols: List[str],
    config_template_path: str,
    param_grid: Dict,
    output_root: str = "multi_symbol_results",
    min_criteria: Optional[Dict] = None,
    sort_by: str = "Avg R-Multiple",
    top_n: int = 5,
    config_overwrite: Optional[Callable[[Dict], Dict]] = None,
) -> None:
    """
    Führt für mehrere Symbole eine Grid Search durch und speichert die Ergebnisse separat.

    Args:
        symbols: Liste aller zu optimierenden Symbole (z.B. ['EURUSD', 'GBPUSD']).
        config_template_path: Pfad zur Config-JSON.
        param_grid: Dict mit Parametern und Wertebereichen.
        output_root: Zielverzeichnis für alle Ergebnisse.
        min_criteria: Mindestkriterien für Ergebnisauswahl (z.B. Winrate etc.).
        sort_by: Metrik für Top-N Auswahl.
        top_n: Anzahl der Top-Ergebnisse pro Symbol.
        config_overwrite: Optionaler Callback zur Modifikation der Config pro Symbol.
    """
    for symbol in symbols:
        print(f"\n🔄 Starte Grid-Search für Symbol: {symbol}\n")
        output_dir = f"{output_root}/{symbol}"
        result_csv = f"{output_dir}/summary.csv"

        # Dynamische Überschreibung: Symbol in Config setzen (wird jedem Symbol individuell übergeben)
        def default_overwrite(cfg: Dict) -> Dict:
            cfg["symbol"] = symbol
            return cfg

        overwrite_fn = config_overwrite or default_overwrite

        run_grid_search(
            config_template_path=config_template_path,
            param_grid=param_grid,
            output_dir=output_dir,
            result_csv=result_csv,
            min_criteria=min_criteria,
            sort_by=sort_by,
            top_n=top_n,
            config_overwrite=overwrite_fn,
        )
