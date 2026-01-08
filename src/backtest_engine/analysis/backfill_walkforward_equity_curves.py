from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.backtest_engine.analysis.backfill_reporting_defaults import (
    BACKFILL_REPORTING_DEFAULTS,
)

# Ensure repository root and src/ are on sys.path so local modules can be imported
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for _extra in (REPO_ROOT, SRC_ROOT):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))

try:
    from backtest_engine.analysis.walkforward_analyzer import _upgrade_base_config
    from backtest_engine.optimizer.final_param_selector import (
        _ensure_preloaded_in_worker,
        _extract_trades,
        _inject_params,
        _parameter_columns,
        _with_worker_preload,
        _yearly_segments,
    )
    from backtest_engine.rating.stability_score import (
        compute_stability_score_from_yearly_profits,
    )
    from backtest_engine.rating.tp_sl_stress_score import (
        compute_tp_sl_stress_score,
        load_primary_candle_arrays_from_parquet,
    )
    from backtest_engine.runner import run_backtest_and_return_portfolio

    IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - defensive import guard
    # Speichere Import-Fehler, um eine verständliche Fehlermeldung liefern zu können.
    IMPORT_ERROR = exc
    _upgrade_base_config = None  # type: ignore
    _ensure_preloaded_in_worker = None  # type: ignore
    _inject_params = None  # type: ignore
    _extract_trades = None  # type: ignore
    _parameter_columns = None  # type: ignore
    _with_worker_preload = None  # type: ignore
    _yearly_segments = None  # type: ignore
    compute_stability_score_from_yearly_profits = None  # type: ignore
    compute_tp_sl_stress_score = None  # type: ignore
    load_primary_candle_arrays_from_parquet = None  # type: ignore
    run_backtest_and_return_portfolio = None  # type: ignore

WALKFORWARD_ROOT_DEFAULT = Path("var/results/analysis")
EXTRA_METRIC_COLS = {
    "robustness_score_1",
    "cost_shock_score",
    "timing_jitter_score",
    "trade_dropout_score",
}
BACKFILL_SNAPSHOT_NAME = "frozen_snapshot_backfill.json"
BACKFILL_COMBINED_FILENAME = "05_final_scores_combined_backfill.csv"


def _load_backfill_reporting_defaults() -> Dict[str, Any]:
    """Liefert eine Kopie der zentral definierten Reporting-Defaults für Backfill.

    Die Defaults sind in analysis/backfill_reporting_defaults.py festgelegt und
    werden hier per deepcopy zurückgegeben, um Mutationsrisiken zu vermeiden.
    """
    return deepcopy(BACKFILL_REPORTING_DEFAULTS)


def _normalize_mean_rev_module(cfg: Dict[str, Any]) -> None:
    """
    Normalisiert Strategy-Modulnamen wie mean_reversion_z_score_X auf
    mean_reversion_z_score, damit Imports auch bei kopierten Ordnern funktionieren.
    """
    try:
        strat = cfg.get("strategy") or {}
        if not isinstance(strat, dict):
            return
        module_path = str(strat.get("module") or "")
        if not module_path:
            return
        core = module_path
        prefix = ""
        if core.startswith("strategies."):
            prefix = "strategies."
            core = core[len(prefix) :]
        first, *rest_parts = core.split(".", 1)
        rest = rest_parts[0] if rest_parts else ""
        # Match mean_reversion_z_score_<suffix> (z.B. _2, _4)
        if re.match(r"^mean_reversion_z_score_.+$", first):
            first = "mean_reversion_z_score"
            core = first + (f".{rest}" if rest else "")
            strat["module"] = prefix + core if prefix else core
            cfg["strategy"] = strat
            logging.debug("Strategy-Modul normalisiert auf %s", strat["module"])
    except Exception:
        # Kein Hard-Fail bei unerwarteten Strukturen
        return


def ensure_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise ImportError(
            "Benötigte Module konnten nicht geladen werden. "
            "Bitte Abhängigkeiten installieren (z.B. joblib) oder "
            "die Code-Basis als Paket verfügbar machen."
        ) from IMPORT_ERROR


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def find_runs(root: Path, filters: Sequence[str] | None) -> List[Path]:
    runs: List[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "combined":
            continue
        if filters and not any(frag in entry.name for frag in filters):
            continue
        scores = entry / "final_selection" / "05_final_scores.csv"
        snap = entry / "baseline" / "frozen_snapshot.json"
        if scores.exists() and snap.exists():
            runs.append(entry)
        else:
            logging.debug(
                "Überspringe %s (fehlende final_scores oder frozen_snapshot).", entry
            )
    runs.sort()
    return runs


def _resolve_snapshot_path(run_dir: Path) -> Path:
    """
    Bevorzugt einen optionalen Backfill-Snapshot gegenüber dem ursprünglichen Snapshot.
    """
    baseline_dir = run_dir / "baseline"
    backfill = baseline_dir / BACKFILL_SNAPSHOT_NAME
    if backfill.exists():
        return backfill
    return baseline_dir / "frozen_snapshot.json"


def _prepare_backfill_snapshot(
    run_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    """
    Erzeugt eine Kopie von baseline/frozen_snapshot.json als BACKFILL_SNAPSHOT_NAME
    und überschreibt dabei das Reporting mit den zentral definierten Defaults
    aus BACKFILL_REPORTING_DEFAULTS. Ebenfalls werden start_date/end_date angepasst,
    falls gesetzt.

    Optimierung: Schreibt nur, wenn sich tatsächlich etwas ändern würde (vermeidet
    unnötige FS-Writes und mtime-Churn).
    """
    snap_path = run_dir / "baseline" / "frozen_snapshot.json"
    if not snap_path.exists():
        logging.warning(
            "Kann Backfill-Snapshot nicht erstellen, Snapshot fehlt: %s", snap_path
        )
        return

    try:
        blob = json.loads(snap_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logging.error(
            "Backfill-Snapshot konnte nicht gelesen werden (%s): %s", snap_path, exc
        )
        return

    base_cfg = blob.get("base_config") or {}
    if not isinstance(base_cfg, dict):
        logging.warning("base_config im Snapshot ist ungültig: %s", snap_path)
        return

    # Prüfe, ob Backfill-Snapshot bereits existiert und bereits korrekt ist
    out_path = run_dir / "baseline" / BACKFILL_SNAPSHOT_NAME
    if out_path.exists():
        try:
            existing_blob = json.loads(out_path.read_text())
            existing_cfg = existing_blob.get("base_config") or {}
            rep_defaults = _load_backfill_reporting_defaults()

            # Check: Reporting bereits korrekt?
            reporting_match = existing_cfg.get("reporting") == rep_defaults
            # Check: Dates bereits korrekt?
            start_match = (not start_date) or (
                existing_cfg.get("start_date") == start_date
            )
            end_match = (not end_date) or (existing_cfg.get("end_date") == end_date)

            if reporting_match and start_match and end_match:
                logging.debug(
                    "Backfill-Snapshot bereits aktuell, kein Schreibvorgang nötig: %s",
                    out_path,
                )
                return
        except Exception as exc:
            # Wenn Lesen/Vergleich fehlschlägt, schreiben wir neu (defensive)
            logging.debug(
                "Existierender Backfill-Snapshot konnte nicht validiert werden (%s): %s",
                out_path,
                exc,
            )

    # Änderungen vornehmen
    changed = False
    rep_defaults = _load_backfill_reporting_defaults()
    if rep_defaults:
        if base_cfg.get("reporting") != rep_defaults:
            base_cfg["reporting"] = rep_defaults
            changed = True
    else:
        logging.warning(
            "[Backfill] Reporting-Defaults fehlen – verwende vorhandenes Reporting."
        )

    if start_date and base_cfg.get("start_date") != start_date:
        base_cfg["start_date"] = start_date
        changed = True
    if end_date and base_cfg.get("end_date") != end_date:
        base_cfg["end_date"] = end_date
        changed = True

    if not changed:
        return

    blob["base_config"] = base_cfg
    try:
        out_path.write_text(
            json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logging.info("Backfill-Snapshot geschrieben: %s", out_path)
    except Exception as exc:  # pragma: no cover - defensive
        logging.error(
            "Backfill-Snapshot konnte nicht geschrieben werden (%s): %s", out_path, exc
        )


def load_snapshot(run_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    ensure_dependencies()
    snap_path = _resolve_snapshot_path(run_dir)
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot nicht gefunden: {snap_path}")
    blob = json.loads(snap_path.read_text())
    base_cfg = blob.get("base_config") or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"base_config fehlt oder ist ungültig in {snap_path}")
    reporting_from_snapshot = (
        deepcopy(base_cfg.get("reporting"))
        if isinstance(base_cfg.get("reporting"), dict)
        else None
    )
    # Ergänze fehlende Felder (direction_filter, enabled_scenarios, Reporting ...)
    upgraded = _upgrade_base_config(base_cfg, run_id=run_dir.name)

    # Reporting ausschließlich aus dem Snapshot respektieren (kein Reload aus Template)
    if reporting_from_snapshot is not None:
        upgraded["reporting"] = reporting_from_snapshot

    _normalize_mean_rev_module(upgraded)
    param_grid = blob.get("param_grid") or {}
    if not isinstance(param_grid, dict):
        param_grid = {}
    return upgraded, param_grid


def resolve_param_columns(df: pd.DataFrame, param_grid: Dict[str, Any]) -> List[str]:
    ensure_dependencies()
    df.columns = [str(c).strip() for c in df.columns]
    grid_keys = {str(k).strip() for k in param_grid.keys()}
    if grid_keys:
        cols = [c for c in df.columns if c in grid_keys]
        if cols:
            return cols
    # Fallback auf Helper aus final_param_selector, aber ohne bekannte Metrik-Spalten
    cols = _parameter_columns(df)
    return [c for c in cols if c not in EXTRA_METRIC_COLS]


def _compute_segment_durations(
    segments: Sequence[Tuple[str, Any, Any]],
) -> Dict[int, float]:
    """
    Liefert die tatsächliche Segmentdauer (Tage) pro Jahr für die Stabilitäts-Berechnung.

    Nutzt die in den Year-Segments enthaltenen Start- und Endzeitpunkte und berechnet
    die inklusiven Tage. Falls ein Segment nicht geparst werden kann, wird es ignoriert.
    """
    durations: Dict[int, float] = {}
    for label, seg_start, seg_end in segments:
        try:
            year = int(str(label).strip())
            start_ts = pd.to_datetime(seg_start)
            end_ts = pd.to_datetime(seg_end)
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            days = float((end_ts.date() - start_ts.date()).days + 1)
            durations[year] = days
        except Exception:
            continue
    return durations


def _build_05_final_scores_combined(run_dir: Path) -> Optional[Path]:
    """
    Erstellt 05_final_scores_combined.csv aus 05_final_scores.csv und
    05_final_scores_detailed.csv, falls diese noch nicht existiert.

    Implementiert die gleiche Logik wie in final_param_selector._merge_scores_with_yearly.
    """
    final_dir = run_dir / "final_selection"
    combined_path = final_dir / "05_final_scores_combined.csv"

    # Wenn Combined bereits existiert, nichts tun
    if combined_path.exists():
        return combined_path

    p_scores = final_dir / "05_final_scores.csv"
    p_detailed = final_dir / "05_final_scores_detailed.csv"

    if not p_scores.exists() or not p_detailed.exists():
        logging.debug(
            "Kann Combined nicht erstellen in %s: fehlende scores oder detailed.",
            run_dir.name,
        )
        return None

    try:
        df_scores = pd.read_csv(p_scores)
        # Spaltennamen säubern
        df_scores.columns = [str(c).strip() for c in df_scores.columns]

        # Lese detailed ohne MultiIndex (einfacher Header)
        df_det = pd.read_csv(p_detailed)
        df_det.columns = [str(c).strip() for c in df_det.columns]

        # Extrahiere combo_id
        combo_col = None
        for c in df_det.columns:
            if "combo" in c.lower() and "id" in c.lower():
                combo_col = c
                break
        if combo_col is None:
            combo_col = df_det.columns[0]

        df_det_combo = df_det[combo_col].astype(str).str.strip()

        # Finde Jahres-Spalten und identifiziere welche Spalten-Indizes zu Jahren gehören
        # Die Struktur ist: year, Net Profit, Winrate (%), Avg R-Multiple, Drawdown, total_trades
        # Pandas benennt doppelte Spaltennamen um: year, year.1, year.2, year.3, year.4
        year_indices = (
            set()
        )  # Speichere alle Spalten-Indizes, die zu Jahres-Daten gehören
        year_data = {}

        i = 0
        while i < len(df_det.columns):
            col = df_det.columns[i]
            # Prüfe ob es "year" oder "year.N" ist
            if col.lower() == "year" or re.match(r"^year\.\d+$", col.lower()):
                # Prüfe ob genug Spalten folgen
                if i + 5 < len(df_det.columns):
                    year_val = df_det.iloc[0, i]
                    # Versuche Jahr zu extrahieren
                    try:
                        year_val = str(int(float(year_val)))
                        if re.match(r"^\d{4}$", year_val):
                            # Sammle die Year-Spalte sowie die 5 folgenden Metriken
                            # Wir fügen die Year-Spalte zuerst ein, damit die Spalten
                            # im combined CSV die Form "2025 year", "2025 Net Profit", ... haben.
                            metrics = {
                                f"{year_val} year": df_det.iloc[:, i],
                                f"{year_val} Net Profit": df_det.iloc[:, i + 1],
                                f"{year_val} Winrate (%)": df_det.iloc[:, i + 2],
                                f"{year_val} Avg R-Multiple": df_det.iloc[:, i + 3],
                                f"{year_val} Drawdown": df_det.iloc[:, i + 4],
                                f"{year_val} total_trades": df_det.iloc[:, i + 5],
                            }
                            year_data[year_val] = metrics
                            # Markiere diese 6 Spalten als Jahres-Spalten
                            for j in range(i, i + 6):
                                year_indices.add(j)
                            i += 6  # Überspringe diese 6 Spalten
                            continue
                    except Exception:
                        pass
            i += 1

        # Sortiere Jahre numerisch absteigend (2025 vor 2021)
        years_sorted = sorted(year_data.keys(), key=int, reverse=True)

        # Extrahiere Nicht-Jahres-Spalten aus detailed, aber schließe Parameter aus,
        # die bereits in df_scores vorhanden sind. Dadurch werden Parameter nur aus
        # df_scores übernommen und nicht zusätzlich aus der detailed-Datei.
        scores_cols_set = set([str(c).strip() for c in df_scores.columns])
        raw_non_year_cols = []
        for i, col in enumerate(df_det.columns):
            if i not in year_indices and col != combo_col:
                raw_non_year_cols.append(col)
        # detailed-only cols sind diejenigen, die nicht in df_scores vorkommen
        # und nicht die Spalte 'primary' (wir wollen Parameter nur aus scores)
        detailed_only_cols = [
            col
            for col in raw_non_year_cols
            if col not in scores_cols_set and str(col).strip().lower() != "primary"
        ]

        # Baue DataFrame mit combo_id und nur detailed-only Spalten (z.B. 'primary')
        df_year = pd.DataFrame({"combo_id": df_det_combo})
        for col in detailed_only_cols:
            df_year[col] = df_det[col].values

        # Füge Jahres-Spalten hinzu (sortiert nach Jahr)
        for year in years_sorted:
            for metric_name, values in year_data[year].items():
                df_year[metric_name] = values.values

        df_year = df_year.drop_duplicates(subset="combo_id", keep="first").reset_index(
            drop=True
        )

        # Stelle sicher, dass df_scores combo_id hat
        if "combo_id" not in df_scores.columns:
            if df_scores.index.name == "combo_id":
                df_scores = df_scores.reset_index()
            else:
                # Suche combo_id Spalte
                for c in df_scores.columns:
                    if "combo" in c.lower() and "id" in c.lower():
                        df_scores = df_scores.rename(columns={c: "combo_id"})
                        break
                else:
                    df_scores.insert(0, "combo_id", df_scores.iloc[:, 0].astype(str))

        df_scores["combo_id"] = df_scores["combo_id"].astype(str).str.strip()

        df_out = pd.merge(df_scores, df_year, on="combo_id", how="left")
        df_out.to_csv(combined_path, index=False)
        logging.info("05_final_scores_combined.csv erstellt: %s", combined_path)
        return combined_path

    except Exception as exc:
        logging.error(
            "Fehler beim Erstellen von 05_final_scores_combined.csv in %s: %s",
            run_dir.name,
            exc,
            exc_info=True,
        )
        return None


def _update_final_scores_with_backfill_metrics(
    run_dir: Path,
    updated_combos: Set[str],
) -> Optional[Path]:
    """
    Erstellt eine Backfill-Variante von 05_final_scores_combined.csv mit
    aktualisierten Metriken basierend auf den neu gebacktesteten Trades.

    - Alle Score-/Robustness-Spalten bleiben unverändert.
    - Globale Metriken (Net Profit, Drawdown, Winrate, Avg R, Sharpe, Sortino,
      total_trades, active_days, profit_over_dd, comm_over_profit) werden für
      die betroffenen combo_ids neu berechnet.
    - Jahresmetriken werden auf Basis der Year-Segments aus base_config
      (start_date/end_date) direkt aus den Trades abgeleitet.
    """
    if not updated_combos:
        return None

    final_dir = run_dir / "final_selection"
    combined_path = final_dir / "05_final_scores_combined.csv"

    # Stelle sicher, dass 05_final_scores_combined.csv existiert
    if not combined_path.exists():
        logging.info("05_final_scores_combined.csv fehlt, versuche zu erstellen...")
        _build_05_final_scores_combined(run_dir)
        if not combined_path.exists():
            logging.warning(
                "Kann Backfill-Combined nicht aktualisieren, Datei fehlt: %s",
                combined_path,
            )
            return None

    try:
        df_combined = pd.read_csv(combined_path)
    except Exception as exc:
        logging.error("Konnte %s nicht lesen: %s", combined_path, exc)
        return None

    if df_combined.empty:
        logging.warning("Combined-Datei %s ist leer.", combined_path)
        return None

    # Spaltennamen säubern (wichtig: vor der combo_id-Prüfung, da manche CSVs Padding enthalten)
    df_combined.columns = [str(c).strip() for c in df_combined.columns]

    # Robust: leicht abweichende combo_id-Spaltennamen erkennen
    if "combo_id" not in df_combined.columns:
        for c in list(df_combined.columns):
            cl = str(c).strip().lower()
            if "combo" in cl and "id" in cl:
                df_combined = df_combined.rename(columns={c: "combo_id"})
                break

    if "combo_id" not in df_combined.columns:
        logging.warning(
            "Combined-Datei %s ist ohne combo_id (Spalten: %s).",
            combined_path,
            list(df_combined.columns),
        )
        return None
    df_new = df_combined.copy()
    df_new["combo_id"] = df_new["combo_id"].astype(str).str.strip()

    # Snapshot mit (ggf. modifizierten) start_date/end_date laden
    try:
        base_cfg, _ = load_snapshot(run_dir)
    except Exception as exc:
        logging.error(
            "Snapshot für Backfill-Metriken konnte nicht geladen werden: %s", exc
        )
        return None

    start_date = (base_cfg or {}).get("start_date")
    end_date = (base_cfg or {}).get("end_date")
    segments: List[Tuple[str, Any, Any]] = []
    if _yearly_segments is not None and start_date and end_date:
        try:
            segments = _yearly_segments(start_date, end_date)
        except Exception as exc:
            logging.error("Fehler beim Ableiten der Year-Segments: %s", exc)

    # Segment-Dauern für Stability-Score ermitteln (wichtig bei Teiljahren)
    segment_durations = _compute_segment_durations(segments) if segments else {}

    # TP/SL Stress: Primary-TF-Candles einmal pro Run laden (Parquet)
    arrays = None
    if load_primary_candle_arrays_from_parquet is not None:
        try:
            symbol = str((base_cfg or {}).get("symbol") or "").strip()
            tf_conf = (base_cfg or {}).get("timeframes") or {}
            primary_tf = str(tf_conf.get("primary") or "").strip()
            if symbol and primary_tf:
                arrays = load_primary_candle_arrays_from_parquet(symbol, primary_tf)
        except Exception:
            arrays = None

    # Sicherstellen, dass Jahres-Spalten existieren (werden bei Bedarf erzeugt)
    yearly_metrics = [
        "Net Profit",
        "Winrate (%)",
        "Avg R-Multiple",
        "Drawdown",
        "total_trades",
    ]
    year_labels = [str(label).strip() for label, _, _ in segments] if segments else []
    for year in year_labels:
        if not year:
            continue
        base_name = f"{year} year"
        if base_name not in df_new.columns:
            df_new[base_name] = np.nan
        for metric in yearly_metrics:
            col_name = f"{year} {metric}"
            if col_name not in df_new.columns:
                df_new[col_name] = np.nan

    trades_root = final_dir / "trades"
    global_metric_cols = [
        "Net Profit",
        "Commission",
        "Avg R-Multiple",
        "Winrate (%)",
        "Drawdown",
        "Sharpe (trade)",
        "Sortino (trade)",
        "total_trades",
        "active_days",
        "profit_over_dd",
        "comm_over_profit",
    ]
    if "tp_sl_stress_score" not in df_new.columns:
        df_new["tp_sl_stress_score"] = np.nan
    if "stability_score" not in df_new.columns:
        df_new["stability_score"] = np.nan

    for combo_id in updated_combos:
        combo_id_str = str(combo_id).strip()
        if not combo_id_str:
            continue
        mask = df_new["combo_id"] == combo_id_str
        if not mask.any():
            continue

        trades_path = trades_root / combo_id_str / "trades.json"
        if not trades_path.exists():
            logging.debug(
                "Keine Trades für combo_id %s unter %s gefunden – überspringe.",
                combo_id_str,
                trades_path,
            )
            continue

        try:
            trades_blob = json.loads(trades_path.read_text())
            trades_df = pd.DataFrame(trades_blob) if trades_blob else pd.DataFrame()
        except Exception as exc:
            logging.error("Konnte Trades für %s nicht laden: %s", combo_id_str, exc)
            continue

        metrics_global = _compute_metrics_from_trades(trades_df)
        metrics_yearly = _compute_yearly_metrics_from_trades(trades_df, segments)
        tp_sl_score: Optional[float] = None
        if compute_tp_sl_stress_score is not None:
            try:
                tp_sl_score = float(compute_tp_sl_stress_score(trades_df, arrays))
            except Exception:
                tp_sl_score = None

        stability_score: Optional[float] = None
        if compute_stability_score_from_yearly_profits is not None and metrics_yearly:
            try:
                profits_by_year: Dict[int, float] = {}
                for year_label, year_metrics in metrics_yearly.items():
                    try:
                        year_int = int(str(year_label).strip())
                    except Exception:
                        continue
                    net_profit = float(year_metrics.get("Net Profit", 0.0) or 0.0)
                    profits_by_year[year_int] = net_profit

                if profits_by_year:
                    stability_score = float(
                        compute_stability_score_from_yearly_profits(
                            profits_by_year,
                            durations_by_year=segment_durations or None,
                        )
                    )
            except Exception:
                stability_score = None

        # Globale Metriken überschreiben (Scores/Robustness bleiben unverändert)
        for col in global_metric_cols:
            if col in metrics_global and col in df_new.columns:
                df_new.loc[mask, col] = metrics_global[col]
        if tp_sl_score is not None and "tp_sl_stress_score" in df_new.columns:
            df_new.loc[mask, "tp_sl_stress_score"] = round(tp_sl_score, 4)
        if stability_score is not None and "stability_score" in df_new.columns:
            df_new.loc[mask, "stability_score"] = round(stability_score, 4)

        # Jahresmetriken schreiben
        for year, m in metrics_yearly.items():
            base_col = f"{year} year"
            if base_col in df_new.columns:
                # Ensure dtype-compatibility (int64 columns complain about string assignment)
                try:
                    df_new.loc[mask, base_col] = int(year)
                except Exception:
                    df_new.loc[mask, base_col] = year
            for metric in yearly_metrics:
                col_name = f"{year} {metric}"
                if col_name in df_new.columns and metric in m:
                    df_new.loc[mask, col_name] = m[metric]

    out_path = final_dir / BACKFILL_COMBINED_FILENAME
    try:
        # Spaltenreihenfolge: tp_sl_stress_score direkt nach trade_dropout_score
        cols = list(df_new.columns)
        try:
            anchor = None
            for candidate in (
                "trade_dropout_score",
                "timing_jitter_score",
                "cost_shock_score",
                "robustness_score_1",
            ):
                if candidate in cols:
                    anchor = candidate
                    break
            if anchor is not None:
                idx = cols.index(anchor)
                for col in ("tp_sl_stress_score", "stability_score"):
                    if col in cols:
                        cols.remove(col)
                insert_at = idx + 1
                if "tp_sl_stress_score" in df_new.columns:
                    cols.insert(insert_at, "tp_sl_stress_score")
                    insert_at += 1
                if "stability_score" in df_new.columns:
                    cols.insert(insert_at, "stability_score")
                df_new = df_new[cols]
        except Exception:
            # Reihenfolge ist nur kosmetisch; niemals den Backfill blockieren
            pass
        df_new.to_csv(out_path, index=False)
        logging.info("Backfill-Combined-Datei aktualisiert: %s", out_path)
        return out_path
    except Exception as exc:
        logging.error(
            "Backfill-Combined-Datei konnte nicht geschrieben werden: %s", exc
        )
        return None


def extract_params(row: Dict[str, Any], columns: Iterable[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, np.generic):
            val = val.item()
        params[col] = val
    return params


def save_equity_curve(portfolio: Any, dest_dir: Path) -> bool:
    curve = (
        portfolio.get_equity_curve() if hasattr(portfolio, "get_equity_curve") else None
    )
    if not curve:
        return False
    rows: List[Dict[str, Any]] = []
    for ts, eq in curve:
        if ts is None:
            continue
        try:
            ts_str = ts.isoformat()
        except Exception:
            ts_str = str(ts)
        try:
            eq_val = float(eq)
        except Exception:
            continue
        rows.append({"timestamp": ts_str, "equity": eq_val})
    if not rows:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest_dir / "equity.csv", index=False)
    return True


def save_trades(trades_df: Optional[pd.DataFrame], dest_dir: Path) -> bool:
    if trades_df is None:
        return False
    try:
        if trades_df.empty:
            return False
    except Exception:
        return False
    try:
        df = trades_df.copy()
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_convert("UTC")
                df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        records = df.to_dict(orient="records")
        if not records:
            return False
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "trades.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def _parse_trade_times(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hilfsfunktion: konvertiert entry_time/exit_time nach UTC und sortiert nach exit_time.
    """
    df = trades_df.copy()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time").reset_index(drop=True)
    return df


def _sharpe_sortino_from_r(r_list: Iterable[float]) -> Tuple[float, float]:
    """
    Sharpe- und Sortino-Ratio auf R-Multiples, analog zu backtest_engine.report.metrics.
    """
    arr = np.asarray([x for x in r_list if x is not None], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0, 0.0
    excess = arr  # risk_free = 0
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    sharpe = float(mu / sigma) if sigma > 0.0 else 0.0
    downside_diff = np.minimum(excess, 0.0)  # mar = 0
    semi_dev = float(np.sqrt(np.mean(downside_diff**2)))
    sortino = float(mu / semi_dev) if semi_dev > 0.0 else 0.0
    return sharpe, sortino


def _compute_metrics_from_trades(trades_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Berechnet einfache Kennzahlen aus einem Trades-DataFrame für einen Zeitraum.

    Wichtig:
    - Net Profit wird als Summe(result) minus Summe(total_fee) approximiert.
    - Drawdown basiert auf kumulierter Equity aus Net-PnL pro Trade.
    """
    if trades_df is None:
        return {
            "Net Profit": 0.0,
            "Commission": 0.0,
            "Avg R-Multiple": 0.0,
            "Winrate (%)": 0.0,
            "Drawdown": 0.0,
            "Sharpe (trade)": 0.0,
            "Sortino (trade)": 0.0,
            "total_trades": 0.0,
            "active_days": 0.0,
            "profit_over_dd": np.nan,
            "comm_over_profit": np.nan,
        }

    df = _parse_trade_times(trades_df)
    if df.empty or "result" not in df.columns:
        return {
            "Net Profit": 0.0,
            "Commission": 0.0,
            "Avg R-Multiple": 0.0,
            "Winrate (%)": 0.0,
            "Drawdown": 0.0,
            "Sharpe (trade)": 0.0,
            "Sortino (trade)": 0.0,
            "total_trades": 0.0,
            "active_days": 0.0,
            "profit_over_dd": np.nan,
            "comm_over_profit": np.nan,
        }

    results = pd.to_numeric(df["result"], errors="coerce").fillna(0.0)
    total_fee = pd.to_numeric(df.get("total_fee", 0.0), errors="coerce").fillna(0.0)
    net_after_fees = float(results.sum() - total_fee.sum())
    commission = float(total_fee.sum())

    total_trades = int((~results.isna()).sum())
    wins = results[results > 0]
    winrate = float(len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

    r_series = pd.to_numeric(df.get("r_multiple", np.nan), errors="coerce")
    r_values = [float(x) for x in r_series.dropna().to_numpy()]
    avg_r = float(np.mean(r_values)) if r_values else 0.0
    sharpe, sortino = _sharpe_sortino_from_r(r_values)

    # Equity-/Drawdown-Berechnung auf Basis des Net-PnL pro Trade
    pnl_per_trade = results - total_fee
    equity = pnl_per_trade.cumsum()
    if equity.empty:
        max_dd = 0.0
    else:
        roll_max = equity.cummax()
        dd_series = roll_max - equity
        max_dd = float(dd_series.max() if pd.notna(dd_series.max()) else 0.0)

    days: Set[Any] = set()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            try:
                dates = df[col].dropna().dt.date
                days.update(dates.tolist())
            except Exception:
                continue
    active_days = float(len(days))

    dd = max_dd if max_dd not in (0.0, np.nan) else np.nan
    profit_over_dd = (net_after_fees / dd) if dd and np.isfinite(dd) else np.nan
    comm_over_profit = (
        (commission / net_after_fees) if net_after_fees not in (0.0, np.nan) else np.nan
    )

    return {
        "Net Profit": net_after_fees,
        "Commission": commission,
        "Avg R-Multiple": avg_r,
        "Winrate (%)": winrate,
        "Drawdown": max_dd,
        "Sharpe (trade)": sharpe,
        "Sortino (trade)": sortino,
        "total_trades": float(total_trades),
        "active_days": active_days,
        "profit_over_dd": profit_over_dd,
        "comm_over_profit": comm_over_profit,
    }


def _compute_yearly_metrics_from_trades(
    trades_df: Optional[pd.DataFrame],
    year_segments: Sequence[Tuple[str, Any, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Berechnet Jahres-Metriken auf Basis der Trades und der Year-Segments.
    """
    metrics_by_year: Dict[str, Dict[str, float]] = {}
    if trades_df is None or trades_df.empty or not year_segments:
        return metrics_by_year

    df = _parse_trade_times(trades_df)
    if "exit_time" not in df.columns:
        return metrics_by_year

    # Vergleiche auf naive Zeitzonen (UTC -> naive)
    df = df.copy()
    try:
        df["exit_time"] = df["exit_time"].dt.tz_convert(None)
    except Exception:
        pass

    for label, seg_start, seg_end in year_segments:
        label_str = str(label).strip()
        if not label_str:
            continue
        if seg_start is None or seg_end is None:
            continue
        try:
            start_ts = pd.to_datetime(seg_start)
            end_ts = pd.to_datetime(seg_end)
        except Exception:
            continue
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_convert(None)
        if getattr(end_ts, "tzinfo", None) is not None:
            end_ts = end_ts.tz_convert(None)
        mask = (df["exit_time"] >= start_ts) & (df["exit_time"] <= end_ts)
        df_year = df.loc[mask]
        if df_year.empty:
            continue
        m = _compute_metrics_from_trades(df_year)
        metrics_by_year[label_str] = m
    return metrics_by_year


def run_backtest(
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    *,
    preload_mode: str,
) -> tuple[Any, Optional[pd.DataFrame]]:
    ensure_dependencies()
    cfg = _inject_params(base_cfg, params)
    if preload_mode == "none":
        portfolio, extra = run_backtest_and_return_portfolio(cfg)
    else:
        _ensure_preloaded_in_worker(base_cfg, preload_mode)
        portfolio, extra = _with_worker_preload(cfg)
    trades_df = _extract_trades(portfolio, extra)
    return portfolio, trades_df


def _process_single_combo(
    row: Dict[str, Any],
    base_cfg: Dict[str, Any],
    param_cols: List[str],
    eq_root: Path,
    trades_root: Path,
    run_name: str,
    *,
    preload_mode: str,
    force: bool,
    dry_run: bool,
    is_backfill_run: bool = False,
) -> tuple[str, bool]:
    """
    Prozessiere eine einzelne Combo-ID (worker function für Parallelisierung).

    Args:
        is_backfill_run: Wenn True, werden bestehende Dateien ignoriert und neu erstellt
                        (wichtig für Backfill mit geänderten Datumsbereichen)
    """
    combo_id = str(row.get("combo_id", "")).strip()
    if not combo_id:
        return combo_id, False

    eq_dest_dir = eq_root / combo_id
    eq_file = eq_dest_dir / "equity.csv"
    trades_dest_dir = trades_root / combo_id
    trades_file = trades_dest_dir / "trades.json"

    # Überspringe nur wenn Dateien existieren UND es kein Backfill-Run ist UND force nicht gesetzt
    if eq_file.exists() and trades_file.exists() and not force and not is_backfill_run:
        return combo_id, False

    params = extract_params(row, param_cols)
    if dry_run:
        logging.info(
            "[Dry-Run] %s -> %s Parameter würden gesetzt.",
            combo_id,
            len(params),
        )
        return combo_id, False

    try:
        portfolio, trades_df = run_backtest(base_cfg, params, preload_mode=preload_mode)
    except Exception as exc:
        logging.error(
            "Backtest fehlgeschlagen (%s | combo=%s): %s",
            run_name,
            combo_id,
            exc,
        )
        return combo_id, False

    try:
        eq_ok = save_equity_curve(portfolio, eq_dest_dir)
        tr_ok = save_trades(trades_df, trades_dest_dir)
    except Exception as exc:
        logging.error(
            "Speicherung fehlgeschlagen (%s | combo=%s): %s",
            run_name,
            combo_id,
            exc,
        )
        return combo_id, False

    if eq_ok or tr_ok:
        logging.info(
            "Ergebnis gespeichert für %s -> equity: %s | trades: %s",
            combo_id,
            eq_file if eq_ok else "n/a",
            trades_file if tr_ok else "n/a",
        )
        return combo_id, True
    else:
        logging.warning(
            "Weder Equity-Kurve noch Trades geliefert (%s | combo=%s).",
            run_name,
            combo_id,
        )
        return combo_id, False


def process_run(
    run_dir: Path,
    combo_filter: Set[str],
    *,
    preload_mode: str,
    force: bool,
    dry_run: bool,
    n_jobs: int = 1,
    is_backfill_run: bool = False,
) -> int:
    # Stelle sicher, dass 05_final_scores_combined.csv existiert
    _build_05_final_scores_combined(run_dir)

    scores_path = run_dir / "final_selection" / "05_final_scores.csv"
    df_scores = pd.read_csv(scores_path)
    df_scores.columns = [str(c).strip() for c in df_scores.columns]
    if df_scores.empty:
        logging.warning("Keine final_scores in %s gefunden.", scores_path)
        return 0

    base_cfg, param_grid = load_snapshot(run_dir)
    param_cols = resolve_param_columns(df_scores, param_grid)
    if not param_cols:
        logging.warning("Keine Parameter-Spalten erkannt in %s.", scores_path)
        return 0

    params_snapshot = (base_cfg.get("strategy") or {}).get("parameters") or {}
    logging.info(
        "Starte Run %s | combos=%d | param_spalten=%d | direction=%s | scenarios=%s | parallel_jobs=%d",
        run_dir.name,
        len(df_scores),
        len(param_cols),
        params_snapshot.get("direction_filter"),
        params_snapshot.get("enabled_scenarios"),
        n_jobs if n_jobs > 0 else os.cpu_count() or 1,
    )

    eq_root = run_dir / "final_selection" / "equity_curves"
    eq_root.mkdir(parents=True, exist_ok=True)
    trades_root = run_dir / "final_selection" / "trades"
    trades_root.mkdir(parents=True, exist_ok=True)

    records = df_scores.to_dict(orient="records")

    # Filtere bereits bei der Vorbereitung
    filtered_records = [
        row
        for row in records
        if (combo_id := str(row.get("combo_id", "")).strip())
        and (not combo_filter or combo_id in combo_filter)
    ]

    if not filtered_records:
        logging.info("Keine zu verarbeitenden Combos nach Filterung.")
        return 0

    logging.info("Verarbeite %d Combos parallel...", len(filtered_records))

    # Erstelle joblib temp-Ordner für Parallelverarbeitung
    tmp_dir = run_dir / "final_selection" / "joblib_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Bestimme Backend: verwende loky für echte Prozess-Parallelität
    backend = os.getenv("BACKFILL_PARALLEL_BACKEND", "loky")

    # Parallele Verarbeitung aller Combos
    try:
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            temp_folder=str(tmp_dir),
            verbose=10 if logging.getLogger().level <= logging.DEBUG else 0,
        )(
            delayed(_process_single_combo)(
                row,
                base_cfg,
                param_cols,
                eq_root,
                trades_root,
                run_dir.name,
                preload_mode=preload_mode,
                force=force,
                dry_run=dry_run,
                is_backfill_run=is_backfill_run,
            )
            for row in filtered_records
        )
    except Exception as exc:
        logging.error("Parallele Verarbeitung fehlgeschlagen: %s", exc)
        return 0

    # Zähle erfolgreiche Schreibvorgänge und sammle betroffene combo_ids
    updated_combos: Set[str] = {cid for cid, success in results if success}
    written = len(updated_combos)

    if not dry_run and updated_combos:
        try:
            _update_final_scores_with_backfill_metrics(run_dir, updated_combos)
        except Exception as exc:
            logging.error(
                "Aktualisierung der Backfill-Metriken in %s fehlgeschlagen: %s",
                run_dir,
                exc,
            )
    return written


def process_run_sequential(
    run_dir: Path,
    combo_filter: Set[str],
    *,
    preload_mode: str,
    force: bool,
    dry_run: bool,
    is_backfill_run: bool = False,
) -> int:
    """Legacy sequentielle Verarbeitung (für Debugging/Fallback)."""
    # Stelle sicher, dass 05_final_scores_combined.csv existiert
    _build_05_final_scores_combined(run_dir)

    scores_path = run_dir / "final_selection" / "05_final_scores.csv"
    df_scores = pd.read_csv(scores_path)
    df_scores.columns = [str(c).strip() for c in df_scores.columns]
    if df_scores.empty:
        logging.warning("Keine final_scores in %s gefunden.", scores_path)
        return 0

    base_cfg, param_grid = load_snapshot(run_dir)
    param_cols = resolve_param_columns(df_scores, param_grid)
    if not param_cols:
        logging.warning("Keine Parameter-Spalten erkannt in %s.", scores_path)
        return 0

    params_snapshot = (base_cfg.get("strategy") or {}).get("parameters") or {}
    logging.info(
        "Starte Run %s (sequentiell) | combos=%d | param_spalten=%d | direction=%s | scenarios=%s",
        run_dir.name,
        len(df_scores),
        len(param_cols),
        params_snapshot.get("direction_filter"),
        params_snapshot.get("enabled_scenarios"),
    )

    eq_root = run_dir / "final_selection" / "equity_curves"
    eq_root.mkdir(parents=True, exist_ok=True)
    trades_root = run_dir / "final_selection" / "trades"
    trades_root.mkdir(parents=True, exist_ok=True)

    written = 0
    updated_combos: Set[str] = set()
    records = df_scores.to_dict(orient="records")
    for row in records:
        cid, success = _process_single_combo(
            row,
            base_cfg,
            param_cols,
            eq_root,
            trades_root,
            run_dir.name,
            preload_mode=preload_mode,
            force=force,
            dry_run=dry_run,
            is_backfill_run=is_backfill_run,
        )
        if success:
            written += 1
            updated_combos.add(str(cid))

    if not dry_run and updated_combos:
        try:
            _update_final_scores_with_backfill_metrics(run_dir, updated_combos)
        except Exception as exc:
            logging.error(
                "Aktualisierung der Backfill-Metriken in %s fehlgeschlagen: %s",
                run_dir,
                exc,
            )
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erstellt equity.csv Dateien für alle combo_ids aus 05_final_scores.csv "
            "unterhalb der Walkforward-Ergebnisse."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=WALKFORWARD_ROOT_DEFAULT,
        help="Wurzelpfad zu var/results/analysis (Standard: %(default)s).",
    )
    parser.add_argument(
        "--walkforward",
        action="append",
        dest="filters",
        help="Nur Runs verarbeiten, deren Ordnername diesen Teilstring enthält. Mehrfach nutzbar.",
    )
    parser.add_argument(
        "--combo-id",
        action="append",
        dest="combo_ids",
        help="Optional nur bestimmte combo_ids verarbeiten.",
    )
    parser.add_argument(
        "--preload-mode",
        choices=["full", "window", "none"],
        default="full",
        help="Datenvorladung wie in final_param_selector (Standard: full).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Anzahl paralleler Jobs für Backtest-Verarbeitung. "
            "-1 = alle verfügbaren CPU-Kerne, 1 = sequentiell (Standard: -1)."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Erzwinge sequentielle Verarbeitung (deaktiviert Parallelität).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Existierende equity.csv/trades.json Dateien überschreiben.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Logging, keine Backtests ausführen.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug-Logging aktivieren.",
    )
    parser.add_argument(
        "--backtest-start-date",
        type=str,
        default=None,
        help=(
            "Optional: Startdatum (YYYY-MM-DD) für Backfill-Backtests. "
            "Wenn gesetzt, wird pro Run ein zusätzlicher Snapshot "
            f"baseline/{BACKFILL_SNAPSHOT_NAME} mit angepasstem Startdatum erzeugt."
        ),
    )
    parser.add_argument(
        "--backtest-end-date",
        type=str,
        default=None,
        help=(
            "Optional: Enddatum (YYYY-MM-DD) für Backfill-Backtests. "
            "Wenn gesetzt, wird pro Run ein zusätzlicher Snapshot "
            f"baseline/{BACKFILL_SNAPSHOT_NAME} mit angepasstem Enddatum erzeugt."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    root = args.root.expanduser()
    if not root.exists():
        logging.error("Root-Pfad nicht gefunden: %s", root)
        return 1

    combo_filter = {c.strip() for c in (args.combo_ids or []) if c}
    runs = find_runs(root, args.filters or [])
    if not runs:
        logging.warning(
            "Keine Walkforward-Runs mit final_scores unter %s gefunden.", root
        )
        return 0

    # Bestimme ob dies ein Backfill-Run ist (mit geänderten Datumsbereichen)
    is_backfill_run = bool(args.backtest_start_date or args.backtest_end_date)
    if is_backfill_run:
        logging.info(
            "Backfill-Modus aktiviert: Bestehende Equity-Kurven werden neu erstellt "
            "(start_date=%s, end_date=%s)",
            args.backtest_start_date or "unverändert",
            args.backtest_end_date or "unverändert",
        )

    total_written = 0
    for run_dir in runs:
        try:
            # Optional: pro Run Backfill-Snapshot vorbereiten
            _prepare_backfill_snapshot(
                run_dir,
                start_date=args.backtest_start_date,
                end_date=args.backtest_end_date,
            )

            if args.sequential:
                written = process_run_sequential(
                    run_dir,
                    combo_filter,
                    preload_mode=args.preload_mode.lower(),
                    force=bool(args.force),
                    dry_run=bool(args.dry_run),
                    is_backfill_run=is_backfill_run,
                )
            else:
                written = process_run(
                    run_dir,
                    combo_filter,
                    preload_mode=args.preload_mode.lower(),
                    force=bool(args.force),
                    dry_run=bool(args.dry_run),
                    n_jobs=args.n_jobs,
                    is_backfill_run=is_backfill_run,
                )
            total_written += written
        except Exception as exc:
            logging.error("Fehler in Run %s: %s", run_dir.name, exc)

    logging.info(
        "Fertig. Neu geschriebene equity/trades Dateien (Kombos mit mind. einem neuen Artefakt): %d",
        total_written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
