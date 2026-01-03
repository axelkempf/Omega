# log_manager.py

from pathlib import Path
from typing import Any, Optional, Tuple

from hf_engine.infra.config.paths import (
    TRADE_LOGS_DIR,  # ggf. vom log_service genutzt; belassen für Klarheit/Kompatibilität
)
from hf_engine.infra.config.paths import (
    ENTRY_LOGS_DIR,
    OPTUNA_LOGS_DIR,
)
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.log_service import log_service

# --- interne Utilities -------------------------------------------------------


def _ts() -> str:
    """Einheitlicher UTC-Zeitstempel für Dateinamen."""
    return now_utc().strftime("%Y%m%d_%H%M%S")


def _ensure_parent_dir(path: Path) -> None:
    """Stellt sicher, dass das Zielverzeichnis existiert."""
    path.parent.mkdir(parents=True, exist_ok=True)


# --- Public API ---------------------------------------------------------------


def log_entry(
    entry_logger: Any, strategy_name: str, symbol: str, suffix: Optional[str] = ""
) -> Tuple[Path, Path]:
    """
    Speichert den Entry-Log einer Strategie in CSV & JSON.

    Parameters
    ----------
    entry_logger : Any
        Objekt mit Methoden `.to_csv(PathLike)` und `.to_json(PathLike)` (z. B. pandas.DataFrame).
    strategy_name : str
    symbol : str
    suffix : Optional[str]

    Returns
    -------
    (csv_path, json_path) : Tuple[Path, Path]
        Pfade der erzeugten Dateien. Bei Fehlern werden Ausnahmen intern geloggt und erneut geworfen.
    """
    ts = _ts()
    suffix_part = f"_{suffix}" if suffix else ""
    base_name = f"{strategy_name}_{symbol}{suffix_part}_{ts}"

    out_csv = ENTRY_LOGS_DIR / f"{base_name}.csv"
    out_json = ENTRY_LOGS_DIR / f"{base_name}.json"

    try:
        _ensure_parent_dir(out_csv)
        _ensure_parent_dir(out_json)

        entry_logger.to_csv(out_csv)  # erwartet pandas-ähnliches Interface
        entry_logger.to_json(out_json)

        print(f"📝 Entry-Log gespeichert: {out_csv}")
        return out_csv, out_json

    except (OSError, AttributeError, TypeError) as e:
        # OSError: IO/FS-Probleme; AttributeError/TypeError: falsches Interface des entry_logger
        log_service.log_exception("Fehler beim Entry-Log speichern", e)
        raise


def log_trade(trade_data: dict) -> bool:
    """
    Loggt einen Trade (Delegation an LogService). Führt minimale Validierung durch.

    Parameters
    ----------
    trade_data : dict
        Erwartet mindestens ein nicht-leeres Dict; Detailvalidierung erfolgt i. d. R. im log_service.

    Returns
    -------
    ok : bool
        True bei Erfolg, False wenn log_service einen Fehler wirft (dann wird auch geloggt).
    """
    try:
        if not isinstance(trade_data, dict) or not trade_data:
            raise ValueError("trade_data muss ein nicht-leeres dict sein.")

        # Optionale, sehr leichte Plausibilitätschecks ohne Annahmen über das Schema:
        # (bewusst minimal gehalten, um bestehende Integrationen nicht zu brechen)
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = _ts()

        log_service.log_trade(trade_data)
        return True

    except Exception as e:
        log_service.log_exception("Fehler beim Trade-Logging", e)
        return False


def log_optuna_report(
    study: Any, strategy_name: str, window_id: Optional[int] = None
) -> Optional[Path]:
    """
    Erzeugt einen HTML-Report aus Optuna-Visualisierungen (History, Param-Importances, Parallel-Coordinate).

    Hinweise
    --------
    - Bei Multi-Objective-Studien wird kein Report generiert (Optuna-Defaults nicht immer sinnvoll).
    - Die Funktion importiert Optuna lazy und fängt ImportError ab.
    - Gibt den Pfad zur HTML-Datei zurück oder None, wenn kein Report erstellt wurde.

    Returns
    -------
    out_file : Optional[Path]
    """
    try:
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_param_importances,
            )
        except ImportError as e:
            log_service.log_exception(
                "Optuna nicht installiert – Report kann nicht erstellt werden.", e
            )
            return None

        # Multi-Objective? -> Report überspringen (explizit & leise)
        directions = getattr(study, "directions", None)
        if directions is not None and len(directions) > 1:
            print(
                "⚠️ Optuna-Report wird bei Multi-Objective-Studien nicht automatisch generiert."
            )
            return None

        ts = _ts()
        suffix = f"_{window_id}" if window_id is not None else ""
        out_file = OPTUNA_LOGS_DIR / f"{strategy_name}{suffix}_{ts}_optuna.html"
        _ensure_parent_dir(out_file)

        # Plots erzeugen
        figs = [
            plot_optimization_history(study),
            plot_param_importances(study),
            plot_parallel_coordinate(study),
        ]

        # Plotly-HTML-Teile bauen (mit CDN für plotlyjs)
        try:
            html_parts = [
                f.to_html(full_html=False, include_plotlyjs="cdn") for f in figs
            ]
        except Exception as e:
            log_service.log_exception("Fehler beim Rendern der Optuna-Figuren", e)
            raise

        full_html = (
            "<html>"
            "<head><meta charset='utf-8'><title>Optuna Report</title></head>"
            "<body>"
            "<h1>Optuna Optimierungsreport</h1>"
            f"{'<hr>'.join(html_parts)}"
            "</body>"
            "</html>"
        )

        try:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_html)
        except OSError as e:
            log_service.log_exception("Fehler beim Schreiben des Optuna-Reports", e)
            raise

        print(f"✅ Optuna-Report gespeichert unter: {out_file}")
        return out_file

    except Exception as e:
        log_service.log_exception("Fehler beim Optuna-Report", e)
        return None


def log_exception(message: str, exception: Exception) -> None:
    """
    Reicht Fehler an den zentralen LogService weiter.
    """
    log_service.log_exception(message, exception)
