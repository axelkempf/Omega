"""
Combined Walkforward Matrix Analyzer

Dieses Skript implementiert eine hierarchische Analyse-Pipeline für Walkforward-Ergebnisse:

1. Walkforward-Ergebnisordner einlesen und nach Symbol x Timeframe x Richtung gruppieren
2. Pro Kombination: walkforward_analyzer ausführen und Ergebnisse in dediziertem Ordner speichern
3. Für jede combo_pair_id: equity.csv und trades.json aggregieren (beide Legs A+B kombinieren)
3a. HART GATE: Beim Laden der Top-50 refined Liste werden alle combo_pair_ids entfernt,
    bei denen mindestens ein Leg robustness_score_1_jittered_80 < 0.8 hat.
    Dies ist ein strikter Quality-Gate für institutionelle Standards.
4. Kombinierte Matrix aus allen Kombinationen aufbauen (eine Zeile pro combo_pair_id)
5. Globale Kennzahlen pro Matrixzeile berechnen
6. Zusätzliche Scores berechnen (Stability, Robustness-Mittelwert, Composite)
7. Finalen Score berechnen

Neu (Budget-schonende Suche statt kartesischem Produkt):
    7a. Vorverarbeitung pro group_id:
        - Pareto-Filter (weiche Dominanz)
        - Clustering (Redundanzreduktion mit Return-Profile Features)
        - Top-K diverse Selection (Performance + Stability + Diversifier)
    7b. Monte Carlo Portfolio Search (Vollständige Evaluation)
        - Ziehe N = min(N_total, 10000 × num_groups) zufällige Portfolio-Kombinationen
        - Evaluiere ALLE Portfolios vollständig und parallel (Multi-Core)
        - WICHTIG: Deduplizierung nach Evaluation (siehe Punkt 8)
        - Direkt von Top-K Pruning zu Monte Carlo Search (optimiert für Performance)

8. Kategorisches Ranking erstellen und speichern (11 Champion-Kategorien)
    - Top Performer: Höchster final_score
    - Capital Efficiency: Bestes Profit-zu-Drawdown-Verhältnis pro Marktzeit
    - Defensive Low Drawdown: Niedrigster Ulcer Index bei geringer Volatilität
    - Stable Compounder: Höchster Stability Score bei konsistenten Jahresgewinnen
    - Sharpe Trader: Höchster Sharpe/Sortino-Score bei aktiver Trading-Frequenz
    - High Conviction: Höchstes durchschnittliches R-Multiple bei selektiven Trades
    - High Turnover: Bester Score für hochfrequente Trading-Strategien
    - Cost Efficient: Niedrigster Fee Drag bei aktiven Strategien
    - Return Shape: Beste Return-Verteilung (Skew, Kurtosis)
    - Independent: Geringste Trade-Überlappung zwischen Strategien
    - Diversifier: Höchster Diversifikations-Score für Portfolio-Balance

    Duplikat-Prevention: Jedes Portfolio darf maximal einmal als Champion erscheinen

WICHTIG: DEDUPLIZIERUNG IN MONTE CARLO
========================================
Nach der Monte Carlo Evaluation werden Duplikate entfernt. Dies ist **kein Fehler**, sondern
gewolltes Verhalten:
- Monte Carlo evaluiert z.B. 5.832 Portfolios
- Viele dieser 5.832 Evaluationen können zu der gleichen final_combo_pair_id führen
  (verschiedene Varianten der gleichen Kombination)
- Nach Deduplizierung (keep="first", sortiert nach final_score descending) bleiben z.B. 3.712 ein
- Dies ist eine **Qualitätsfilt<D5>rung: Die beste Variante jeder Kombination wird behalten
- Reduzierung ist typisch: 30-40% Duplikate (durchschnittlich)

Grund: Die Portfolio-Samplungslogik kann verschiedene Varianten erzeugen, weil:
- Jede Gruppe (Leg) mehrere Kandidaten enthält
- Monte Carlo randomly kombiniert diese
- Verschiedene Kombinationen der Kandidaten → selbe Legs

ARCHITEKTUR-ÄNDERUNG (Dezember 2025):
- Genetischer Algorithmus (GA) vollständig entfernt
- Beam Search vollständig entfernt
- Greedy Coordinate Ascent vollständig entfernt
- Successive Halving vollständig ersetzt durch Monte Carlo Search
- Global Marginal Screening vollständig entfernt (nicht verwendet, Performance-Overhead)
- Adaptive Samples: min(N_total, default × num_groups) mit default=10000
- Neue Pipeline: Pruning → Clustering → Top-K Pruning → Monte Carlo Search → Categorical Ranking
- Monte Carlo mit vollständiger paralleler Evaluation (Multi-Core)
- Top-10-Logik ersetzt durch kategorisches Ranking (11 Champion-Kategorien)
- Fokus: Institutional-grade Portfolio-Suche mit maximaler Transparenz und Parallelität

Autor: AI Agent
Datum: 2025-12-04
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running this file directly (e.g. `python -m backtest_engine.analysis.combined_walkforward_matrix_analyzer`)
# from the repository root without requiring the package to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # Optional dependency; falls back to greedy selection if unavailable
    from scipy.optimize import linear_sum_assignment
except (
    ImportError
):  # pragma: no cover - scipy is listed in requirements, but stay defensive
    linear_sum_assignment = None

from backtest_engine.analysis.metric_adjustments import (
    risk_adjusted,
    shrinkage_adjusted,
    wilson_score_lower_bound,
)
from backtest_engine.analysis.walkforward_analyzer import (
    BACKFILL_SNAPSHOT_NAME,
    COMBINED_DIR,
    WALKFORWARD_ROOT,
    __compute_yearly_stability,
    _combine_equity_series,
    _detect_equity_columns,
    _drawdowns_by_year_from_equity,
    _load_equity_series_for_combo,
    _load_trades_for_combo,
    _round_value,
    _safe_profit_over_dd,
    run_walkforward_analysis,
)


def _add_total_adust_metrics_to_portfolios(df: pd.DataFrame) -> pd.DataFrame:
    """Add trade-count adjusted total metrics to a portfolio DataFrame.

    Adds (requested naming):
      - winrate_adust  (percent 0-100)
      - avg_r_adust
      - profit_over_dd_adust

    Expected base columns:
      - avg_r
      - winrate (percent 0-100)
      - total_profit_over_dd
      - total_trades
      - duration_days (optional, used to infer n_years; falls back to 1.0)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    def _num_series(col: str, default: float) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(default, index=out.index, dtype=float)

    avg_r_raw = _num_series("avg_r", np.nan)
    winrate_raw_pct = _num_series("winrate", np.nan)
    pod_raw = _num_series("total_profit_over_dd", np.nan)
    n_trades = _num_series("total_trades", 0.0).fillna(0.0)

    duration_days = _num_series("duration_days", np.nan)
    n_years = np.where(
        np.isfinite(duration_days.to_numpy()) & (duration_days.to_numpy() > 0.0),
        duration_days.to_numpy() / 365.25,
        1.0,
    )
    n_years = np.where(np.isfinite(n_years) & (n_years > 0.0), n_years, 1.0)
    # Keep at least 1 year (consistent with earlier "unique years" fallback behavior)
    n_years = np.maximum(n_years, 1.0)

    groups_count = _num_series("groups_count", 1.0)
    n_categories = np.where(
        np.isfinite(groups_count.to_numpy()) & (groups_count.to_numpy() > 0.0),
        groups_count.to_numpy(),
        1.0,
    )
    n_categories = np.where(
        np.isfinite(n_categories) & (n_categories > 0.0), n_categories, 1.0
    )
    n_categories = np.maximum(n_categories, 1.0)

    avg_r_adust = shrinkage_adjusted(
        average_r=avg_r_raw.to_numpy(),
        n_trades=n_trades.to_numpy(),
        n_years=n_years,
        n_categories=n_categories,
    )

    winrate_dec = winrate_raw_pct.to_numpy() / 100.0
    winrate_adust_pct = (
        wilson_score_lower_bound(
            winrate=winrate_dec,
            n_trades=n_trades.to_numpy(),
        )
        * 100.0
    )

    pod_clipped = np.where(pod_raw.to_numpy() >= 0.0, pod_raw.to_numpy(), 0.0)
    pod_adust = risk_adjusted(
        profit_over_drawdown=pod_clipped,
        n_trades=n_trades.to_numpy(),
        n_years=n_years,
        n_categories=n_categories,
    )

    out["winrate_adust"] = winrate_adust_pct
    out["avg_r_adust"] = avg_r_adust
    out["profit_over_dd_adust"] = pod_adust
    return out


# Output-Verzeichnis für kombinierte Analyse
COMBINED_MATRIX_DIR = WALKFORWARD_ROOT / "combined_matrix"
COMBINED_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

# Champion-Listen-Output (matrix-phase)
TOP10_MATRIX_CSV = COMBINED_MATRIX_DIR / "categorical_champions_matrix.csv"
TOP10_FINAL_COMBOS_CSV = COMBINED_MATRIX_DIR / "categorical_champions_combo_pairs.csv"
CATEGORICAL_CHAMPIONS_PORTFOLIOS_ONE_LINER_CSV = (
    COMBINED_MATRIX_DIR / "categorical_champions_combo_pairs_overview.csv"
)
FINAL_BATCHES_DIR = COMBINED_MATRIX_DIR / "final_batches"
FINAL_BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# DEVELOPMENT MODE: Seed für reproducible Ergebnisse (nur für Entwicklung)
DEVELOPMENT_SEED = 42  # Kann via --development-seed überschrieben werden


@dataclass
class WalkforwardGroup:
    """Repräsentiert eine Gruppe von Walkforward-Runs mit gleicher Kombination."""

    symbol: str
    timeframe: str
    direction: str
    strategy_name: str = "MeanReversionZScoreStrategy"
    run_folders: List[Path] = None

    def __post_init__(self):
        if self.run_folders is None:
            self.run_folders = []

    @property
    def group_id(self) -> str:
        """Eindeutige ID für diese Kombination."""
        return f"{self.symbol}_{self.timeframe}_{self.direction}"

    @property
    def output_dir(self) -> Path:
        """Dedizierter Output-Ordner für diese Kombination."""
        return COMBINED_MATRIX_DIR / self.group_id


def validate_run_folder(run_folder: Path) -> bool:
    """
    Prüft, ob ein Run-Ordner die erwartete Struktur hat.

    Returns:
        True wenn die Ordnerstruktur valid ist, sonst False
    """
    final_dir = run_folder / "final_selection"
    scores_path = final_dir / "05_final_scores.csv"
    detailed_path = final_dir / "05_final_scores_detailed.csv"
    baseline_dir = run_folder / "baseline"
    snapshot_path = baseline_dir / BACKFILL_SNAPSHOT_NAME
    if not snapshot_path.exists():
        snapshot_path = baseline_dir / "frozen_snapshot.json"

    missing = []
    if not final_dir.exists():
        missing.append(str(final_dir))
    if not scores_path.exists():
        missing.append(str(scores_path))
    if not detailed_path.exists():
        missing.append(str(detailed_path))
    if not snapshot_path.exists():
        missing.append(str(snapshot_path))

    if missing:
        print(f"[Validate] Fehlende Dateien in {run_folder.name}:")
        for m in missing:
            print(f"  - {m}")
        return False

    return True


def ensure_combined_files(run_folders: List[Path]) -> None:
    """
    Stellt sicher, dass alle Runs 05_final_scores_combined.csv haben.
    Erstellt fehlende Combined-Dateien automatisch.
    """
    from backtest_engine.analysis.walkforward_analyzer import (
        _rebuild_combined_if_missing,
    )

    for run_folder in run_folders:
        final_dir = run_folder / "final_selection"
        combined_path = final_dir / "05_final_scores_combined_backfill.csv"
        if not combined_path.exists():
            combined_path = final_dir / "05_final_scores_combined.csv"

        if not combined_path.exists():
            print(f"[Combined] Erstelle combined für {run_folder.name}...")
            rebuilt = _rebuild_combined_if_missing(final_dir)

            if rebuilt:
                print(f"[Combined] ✓ Erstellt: {rebuilt}")
            else:
                print(f"[Combined] ✗ FEHLER: Konnte combined nicht erstellen!")


def extract_metadata_from_folder_name(
    folder_name: str,
) -> Optional[Tuple[str, str, str, int]]:
    """
    Extrahiert Symbol, Timeframe, Richtung und Szenario aus dem Ordnernamen.

    Beispiel:
    "41_Walkforward_Ergebnisse_EURUSD_M30_LONG_26_10_2025_2xBullish_Regime_automatisiert_Z5"
    -> ("EURUSD", "M30", "LONG", 5)

    Returns:
        (symbol, timeframe, direction, scenario) oder None bei Fehler
    """
    # Pattern: <prefix>_<SYMBOL>_<TIMEFRAME>_<DIRECTION>_<rest>_Z<scenario>
    pattern = r".*_([A-Z]{6})_(M\d+|H\d+|D\d+)_(LONG|SHORT)_.*_Z(\d+)"
    match = re.search(pattern, folder_name, re.IGNORECASE)

    if match:
        symbol = match.group(1).upper()
        timeframe = match.group(2).upper()
        direction = match.group(3).lower()
        scenario = int(match.group(4))
        return symbol, timeframe, direction, scenario

    return None


def load_metadata_from_snapshot(run_folder: Path) -> Optional[Dict[str, Any]]:
    """
    Lädt Metadaten aus frozen_snapshot.json einer Walkforward-Run.

    Returns:
        Dict mit symbol, timeframe, direction, scenario oder None
    """
    baseline_dir = run_folder / "baseline"
    snapshot_path = baseline_dir / BACKFILL_SNAPSHOT_NAME
    if not snapshot_path.exists():
        snapshot_path = baseline_dir / "frozen_snapshot.json"

    if not snapshot_path.exists():
        return None

    try:
        with snapshot_path.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)

        base_config = snapshot.get("base_config", {})

        # Symbol
        symbol = base_config.get("symbol")
        if not symbol:
            rates = base_config.get("rates", {})
            pairs = rates.get("pairs", [])
            symbol = pairs[0] if pairs else None

        # Timeframe
        timeframes = base_config.get("timeframes", {})
        timeframe = timeframes.get("primary")
        if not timeframe:
            rates = base_config.get("rates", {})
            timeframe = rates.get("timeframe")

        # Direction
        strategy = base_config.get("strategy", {})
        params = strategy.get("parameters", {})
        direction = params.get("direction_filter")

        # Scenario
        enabled_scenarios = params.get("enabled_scenarios", [])
        scenario = enabled_scenarios[0] if enabled_scenarios else None

        if symbol and timeframe and direction:
            return {
                "symbol": str(symbol).upper(),
                "timeframe": str(timeframe).upper(),
                "direction": str(direction).lower(),
                "scenario": int(scenario) if scenario is not None else None,
            }

    except Exception as e:
        print(f"Warnung: Konnte Metadaten aus {snapshot_path} nicht lesen: {e}")

    return None


def discover_walkforward_groups(
    root: Path = WALKFORWARD_ROOT,
) -> List[WalkforwardGroup]:
    """
    Durchsucht das Walkforward-Root-Verzeichnis und gruppiert Runs nach
    Symbol x Timeframe x Richtung.

    Returns:
        Liste von WalkforwardGroup-Objekten
    """
    groups_dict: Dict[str, WalkforwardGroup] = {}

    # Alle Unterordner durchsuchen
    for item in root.iterdir():
        if not item.is_dir():
            continue

        # Überspringe bekannte System-Ordner
        if item.name in ("combined", "combined_matrix"):
            continue

        # NEU: Validierung der Ordnerstruktur
        if not validate_run_folder(item):
            print(f"[Discover] Überspringe ungültigen Run: {item.name}")
            continue

        # Primär: Metadaten aus Ordnernamen extrahieren
        folder_meta = extract_metadata_from_folder_name(item.name)

        # Sekundär: Aus frozen_snapshot.json lesen
        if not folder_meta:
            snapshot_meta = load_metadata_from_snapshot(item)
            if snapshot_meta:
                folder_meta = (
                    snapshot_meta["symbol"],
                    snapshot_meta["timeframe"],
                    snapshot_meta["direction"],
                    snapshot_meta.get("scenario"),
                )

        if not folder_meta:
            print(
                f"Warnung: Konnte Metadaten für {item.name} nicht extrahieren. Überspringe."
            )
            continue

        symbol, timeframe, direction, scenario = folder_meta

        # Gruppierungs-ID (ohne Szenario, da mehrere Szenarien zusammen analysiert werden)
        group_key = f"{symbol}_{timeframe}_{direction}"

        if group_key not in groups_dict:
            groups_dict[group_key] = WalkforwardGroup(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                run_folders=[],
            )

        groups_dict[group_key].run_folders.append(item)
        print(f"  [Discover] {item.name} -> Gruppe {group_key} (Szenario {scenario})")

    groups = list(groups_dict.values())

    print(
        f"\n[Discover] Gefunden: {len(groups)} eindeutige Kombinationen (Symbol x Timeframe x Richtung)"
    )
    for group in groups:
        print(f"  - {group.group_id}: {len(group.run_folders)} Run(s)")

    return groups


def run_analyzer_for_group(group: WalkforwardGroup) -> Optional[Path]:
    """
    Führt walkforward_analyzer für eine Gruppe aus und speichert Ergebnisse
    in einem dedizierten Ordner.

    Returns:
        Path zum Output-Ordner oder None bei Fehler
    """
    print(f"\n{'='*80}")
    print(f"[Analyzer] Starte Analyse für Gruppe: {group.group_id}")
    print(f"[Analyzer] Runs: {[r.name for r in group.run_folders]}")
    print(f"{'='*80}\n")

    # Output-Ordner erstellen
    output_dir = group.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # NEU: Combined-Dateien sicherstellen
    print(f"[Analyzer] Prüfe Combined-Dateien für {group.group_id}...")
    ensure_combined_files(group.run_folders)

    try:
        # Walkforward-Analyzer ausführen für alle Runs dieser Gruppe
        # Die Funktion run_walkforward_analysis durchsucht automatisch alle
        # final_scores_combined.csv Dateien in den übergebenen Ordnern

        # WICHTIG: Globales COMBINED_DIR vor jedem Durchlauf leeren, um Race Conditions
        # und falsche Zuordnungen zwischen Gruppen zu vermeiden
        if COMBINED_DIR.exists():
            print(f"[Analyzer] Lösche vorherige Ergebnisse in {COMBINED_DIR}")
            for item in COMBINED_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # Temporäres Verzeichnis mit Symlinks zu den relevanten Runs erstellen
        temp_root = output_dir / "_temp_runs"
        temp_root.mkdir(parents=True, exist_ok=True)

        for run_folder in group.run_folders:
            link_name = temp_root / run_folder.name
            if not link_name.exists():
                # Symlink oder Copy je nach OS
                try:
                    link_name.symlink_to(run_folder.resolve())
                except (OSError, NotImplementedError):
                    # Windows: kopiere stattdessen
                    shutil.copytree(run_folder, link_name, symlinks=True)

        # Analyzer ausführen
        print(f"[Analyzer] Starte run_walkforward_analysis für {group.group_id}...")
        combined_df, pairs_df = run_walkforward_analysis(
            root=temp_root,
            save_base_combined=True,
            refine_top50=False,
            robust_workers=4,
            enable_clustering=True,
            cluster_method="interval",
        )
        print(f"[Analyzer] run_walkforward_analysis abgeschlossen für {group.group_id}")
        print(
            f"[Analyzer] combined_df Shape: {combined_df.shape}, pairs_df Shape: {pairs_df.shape}"
        )

        # Ergebnisse vom globalen COMBINED_DIR in den gruppenspezifischen Output-Ordner kopieren
        import shutil

        # Liste der zu kopierenden Dateien (inkl. verfeinerter Top-50).
        files_to_copy = [
            "combined_base.csv",
            "top_100_walkforward_combos.csv",
            "top_50_walkforward_combos.csv",
            "top_50_walkforward_combos_refined.csv",
        ]

        # Mapping von temp_runs Namen zu Original-Ordnern erstellen
        run_mapping = {run_folder.name: run_folder for run_folder in group.run_folders}

        for filename in files_to_copy:
            src_file = COMBINED_DIR / filename
            if src_file.exists():
                dest = output_dir / filename

                # CSV einlesen und source_walkforward Pfade korrigieren
                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(src_file)

                        # source_walkforward Spalte korrigieren: temp_runs -> original Pfad
                        if "source_walkforward" in df.columns:

                            def fix_source_path(path_str):
                                if pd.isna(path_str) or not path_str:
                                    return path_str
                                path_str = str(path_str)

                                # Extrahiere den Run-Namen aus dem temp_runs Pfad
                                # Format: "combined_matrix/EURUSD_M30_long/_temp_runs/41_Walkforward_..."
                                if "_temp_runs/" in path_str:
                                    parts = path_str.split("_temp_runs/")
                                    if len(parts) > 1:
                                        run_name = parts[1]

                                        # 1) Mapping auf Original-Ordner, wenn der Run in dieser Gruppe liegt
                                        if run_name in run_mapping:
                                            try:
                                                rel_path = run_mapping[
                                                    run_name
                                                ].relative_to(WALKFORWARD_ROOT)
                                                return str(rel_path)
                                            except ValueError:
                                                # Fallback: nichts tun
                                                return path_str

                                        # 2) Generischer Fallback: entferne _temp_runs-Präfix vollständig
                                        #    und liefere den reinen Run-Ordnernamen zurück. Dieser liegt unter
                                        #    WALKFORWARD_ROOT und kann von den Loadern gefunden werden.
                                        return run_name

                                return path_str

                            df["source_walkforward"] = df["source_walkforward"].apply(
                                fix_source_path
                            )

                        # VALIDIERUNG: Prüfe, ob die Daten zur aktuellen Gruppe gehören
                        if (
                            "symbol" in df.columns
                            and "timeframe" in df.columns
                            and "direction" in df.columns
                        ):
                            if not df.empty:
                                # Extrahiere einzigartige Werte
                                unique_symbols = df["symbol"].dropna().unique()
                                unique_timeframes = df["timeframe"].dropna().unique()
                                unique_directions = df["direction"].dropna().unique()

                                # Prüfe ob alle Zeilen zur erwarteten Gruppe gehören
                                expected_match = True
                                if (
                                    len(unique_symbols) > 0
                                    and group.symbol not in unique_symbols
                                ):
                                    expected_match = False
                                if (
                                    len(unique_timeframes) > 0
                                    and group.timeframe not in unique_timeframes
                                ):
                                    expected_match = False
                                if (
                                    len(unique_directions) > 0
                                    and group.direction not in unique_directions
                                ):
                                    expected_match = False

                                if not expected_match:
                                    print(
                                        f"[Analyzer] ⚠️ WARNUNG: Datei {filename} enthält Daten für falsche Gruppe!"
                                    )
                                    print(
                                        f"  Erwartet: {group.symbol} {group.timeframe} {group.direction}"
                                    )
                                    print(
                                        f"  Gefunden: {unique_symbols} {unique_timeframes} {unique_directions}"
                                    )
                                    print(f"  Diese Datei wird NICHT gespeichert!")
                                    continue

                        # Korrigierte CSV speichern
                        df.to_csv(dest, index=False)
                        print(
                            f"[Analyzer] ✓ Gespeichert (Pfade korrigiert, Validierung OK): {dest}"
                        )
                    except Exception as e:
                        print(
                            f"[Analyzer] Warnung: Fehler beim Korrigieren von {filename}: {e}"
                        )
                        shutil.copy2(src_file, dest)
                else:
                    shutil.copy2(src_file, dest)
                    print(f"[Analyzer] Gespeichert: {dest}")
            else:
                print(f"[Analyzer] Warnung: {src_file} nicht gefunden")

        # Temp-Ordner aufräumen
        shutil.rmtree(temp_root, ignore_errors=True)

        print(f"[Analyzer] Analyse für {group.group_id} abgeschlossen.\n")
        return output_dir

    except Exception as e:
        print(f"[Analyzer] FEHLER bei Gruppe {group.group_id}: {e}\n")
        import traceback

        traceback.print_exc()
        return None


def aggregate_equity_for_combo_pair(
    group: WalkforwardGroup,
    combo_pair_id: str,
    combo_rows: pd.DataFrame,
) -> Optional[Path]:
    """
    Erstellt eine aggregierte equity.csv für eine spezifische combo_pair_id.

    Die Aggregation erfolgt durch:
    - Laden der Equity-Serien der beiden Legs (A und B) der Kombination
    - Kombination der beiden Equity-Kurven

    Args:
        group: WalkforwardGroup
        combo_pair_id: Eindeutige ID des Paares
        combo_rows: DataFrame mit beiden Legs (A und B) dieser combo_pair_id

    Returns:
        Path zur aggregierten equity.csv oder None bei Fehler
    """
    try:
        equity_cache: Dict[Tuple[str, str], Optional[pd.Series]] = {}

        # Legs A und B identifizieren
        leg_a = (
            combo_rows[combo_rows["combo_leg"] == "A"].iloc[0]
            if "A" in combo_rows["combo_leg"].values
            else None
        )
        leg_b = (
            combo_rows[combo_rows["combo_leg"] == "B"].iloc[0]
            if "B" in combo_rows["combo_leg"].values
            else None
        )

        if leg_a is None or leg_b is None:
            print(f"[Aggregate] Warnung: Nicht beide Legs gefunden für {combo_pair_id}")
            return None

        # Equity-Serien laden
        combo_id_a = str(leg_a.get("combo_id", ""))
        combo_id_b = str(leg_b.get("combo_id", ""))
        src_a = str(leg_a.get("source_walkforward", ""))
        src_b = str(leg_b.get("source_walkforward", ""))

        series_a = _load_equity_series_for_combo(src_a, combo_id_a, equity_cache)
        series_b = _load_equity_series_for_combo(src_b, combo_id_b, equity_cache)

        # Kombinieren
        combined = _combine_equity_series(series_a, series_b)

        if combined is None or combined.empty:
            print(f"[Aggregate] Keine gültige kombinierte Equity für {combo_pair_id}")
            return None

        # Als DataFrame mit timestamp, equity
        df_out = combined.reset_index()
        df_out.columns = ["timestamp", "equity"]

        # Speichern im Unterordner nach combo_pair_id
        combo_dir = group.output_dir / "combo_pairs" / combo_pair_id
        combo_dir.mkdir(parents=True, exist_ok=True)

        output_path = combo_dir / "equity.csv"
        df_out.to_csv(output_path, index=False)
        print(f"[Aggregate] Equity für {combo_pair_id} gespeichert: {output_path}")

        return output_path

    except Exception as e:
        print(
            f"[Aggregate] FEHLER beim Aggregieren von Equity für {combo_pair_id}: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def aggregate_trades_for_combo_pair(
    group: WalkforwardGroup,
    combo_pair_id: str,
    combo_rows: pd.DataFrame,
) -> Optional[Path]:
    """
    Erstellt eine aggregierte trades.json für eine spezifische combo_pair_id.

    Die Aggregation erfolgt durch:
    - Laden der Trades der beiden Legs (A und B) der Kombination
    - Zusammenführen in eine gemeinsame Liste
    - Sortieren nach entry_time

    Args:
        group: WalkforwardGroup
        combo_pair_id: Eindeutige ID des Paares
        combo_rows: DataFrame mit beiden Legs (A und B) dieser combo_pair_id

    Returns:
        Path zur aggregierten trades.json oder None bei Fehler
    """
    try:
        # Legs A und B identifizieren
        leg_a = (
            combo_rows[combo_rows["combo_leg"] == "A"].iloc[0]
            if "A" in combo_rows["combo_leg"].values
            else None
        )
        leg_b = (
            combo_rows[combo_rows["combo_leg"] == "B"].iloc[0]
            if "B" in combo_rows["combo_leg"].values
            else None
        )

        if leg_a is None or leg_b is None:
            print(f"[Aggregate] Warnung: Nicht beide Legs gefunden für {combo_pair_id}")
            return None

        # Trades laden
        combo_id_a = str(leg_a.get("combo_id", ""))
        combo_id_b = str(leg_b.get("combo_id", ""))
        src_a = str(leg_a.get("source_walkforward", ""))
        src_b = str(leg_b.get("source_walkforward", ""))

        trades_a = _load_trades_for_combo(src_a, combo_id_a)
        trades_b = _load_trades_for_combo(src_b, combo_id_b)

        all_trades: List[Dict[str, Any]] = []

        if trades_a is not None and not trades_a.empty:
            all_trades.extend(trades_a.to_dict("records"))

        if trades_b is not None and not trades_b.empty:
            all_trades.extend(trades_b.to_dict("records"))

        if not all_trades:
            print(f"[Aggregate] Keine Trades für {combo_pair_id}")
            return None

        # Nach entry_time sortieren
        all_trades_df = pd.DataFrame(all_trades)
        if "entry_time" in all_trades_df.columns:
            all_trades_df["entry_time"] = pd.to_datetime(
                all_trades_df["entry_time"], utc=True, errors="coerce"
            )
            all_trades_df = all_trades_df.sort_values("entry_time")

        # Speichern im Unterordner nach combo_pair_id
        combo_dir = group.output_dir / "combo_pairs" / combo_pair_id
        combo_dir.mkdir(parents=True, exist_ok=True)

        output_path = combo_dir / "trades.json"
        trades_list = all_trades_df.to_dict("records")

        # Timestamps als ISO-String formatieren
        for trade in trades_list:
            for key in ("entry_time", "exit_time"):
                if key in trade and pd.notna(trade[key]):
                    try:
                        ts = pd.to_datetime(trade[key], utc=True)
                        trade[key] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pass

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(trades_list, f, indent=2)

        print(
            f"[Aggregate] Trades für {combo_pair_id} gespeichert: {output_path} ({len(trades_list)} Trades)"
        )

        return output_path

    except Exception as e:
        print(
            f"[Aggregate] FEHLER beim Aggregieren von Trades für {combo_pair_id}: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def aggregate_all_combo_pairs_for_group(
    group: WalkforwardGroup,
) -> Dict[str, Dict[str, Path]]:
    """
    Aggregiert Equity und Trades für alle combo_pair_ids in einer Gruppe.

    Args:
        group: WalkforwardGroup

    Returns:
        Dict mapping combo_pair_id -> {"equity": Path, "trades": Path}
    """
    # Bevorzugt die verfeinerte Top-50, fällt zurück auf Basis-Top-50
    candidates = [
        group.output_dir / "top_50_walkforward_combos_refined.csv",
        group.output_dir / "top_50_walkforward_combos.csv",
    ]
    top_csv = next((p for p in candidates if p.exists()), None)

    if not top_csv:
        print(f"[Aggregate] Keine Top-50 Datei für {group.group_id}")
        return {}

    try:
        top50 = pd.read_csv(top_csv, skip_blank_lines=True)

        if top50.empty:
            print(f"[Aggregate] Leere Top-50 Datei für {group.group_id}")
            return {}

        # HART GATE: Entferne alle combo_pair_ids mit robustness_score_1_jittered_80 < 0.8
        if "robustness_score_1_jittered_80" in top50.columns:
            initial_count = len(top50["combo_pair_id"].unique())

            # Finde combo_pair_ids, die das Gate nicht bestehen
            failed_pairs = set(
                top50[top50["robustness_score_1_jittered_80"] < 0.8][
                    "combo_pair_id"
                ].unique()
            )

            if failed_pairs:
                print(
                    f"[Aggregate] Hart Gate: {len(failed_pairs)} combo_pair_ids haben robustness_score_1_jittered_80 < 0.8 und werden entfernt"
                )
                top50 = top50[~top50["combo_pair_id"].isin(failed_pairs)]
                final_count = len(top50["combo_pair_id"].unique())
                print(
                    f"[Aggregate] Hart Gate: {initial_count} -> {final_count} combo_pair_ids verbleiben"
                )

            if top50.empty:
                print(
                    f"[Aggregate] Keine combo_pair_ids verbleiben nach Hart Gate für {group.group_id}"
                )
                return {}

        # Gruppieren nach combo_pair_id
        results = {}

        for combo_pair_id, combo_group in top50.groupby("combo_pair_id"):
            print(
                f"\n[Aggregate] Verarbeite {combo_pair_id} in {group.group_id} (Quelle: {top_csv.name})"
            )

            # Equity aggregieren
            equity_path = aggregate_equity_for_combo_pair(
                group, str(combo_pair_id), combo_group
            )

            # Trades aggregieren
            trades_path = aggregate_trades_for_combo_pair(
                group, str(combo_pair_id), combo_group
            )

            if equity_path or trades_path:
                results[str(combo_pair_id)] = {
                    "equity": equity_path,
                    "trades": trades_path,
                }

        print(
            f"\n[Aggregate] {len(results)} combo_pair_ids aggregiert für {group.group_id}"
        )
        return results

    except Exception as e:
        print(f"[Aggregate] FEHLER bei Gruppe {group.group_id}: {e}")
        import traceback

        traceback.print_exc()
        return {}


def build_combined_matrix(groups: List[WalkforwardGroup]) -> pd.DataFrame:
    """
    Baut die kombinierte Matrix aus allen Gruppen auf.

    Jede Zeile repräsentiert eine Paar-Kombination (combo_pair_id) aus den
    Top-50-Listen einer Gruppe.

    Returns:
        DataFrame mit Spalten:
        - group_id
        - combo_pair_id
        - <vollständige Parametrisierung>
        - <Pfade zu aggregierten Dateien>
    """
    rows: List[Dict[str, Any]] = []

    for group in groups:
        # Bevorzugt die verfeinerte Top-50, fällt zurück auf Basis-Top-50,
        # danach auf Legacy-Top-10 falls vorhanden.
        candidates = [
            group.output_dir / "top_50_walkforward_combos_refined.csv",
            group.output_dir / "top_50_walkforward_combos.csv",
        ]
        top50_path = next((p for p in candidates if p.exists()), None)

        if not top50_path:
            print(f"[Matrix] Keine Top-50-Datei für {group.group_id}")
            continue

        try:
            top50 = pd.read_csv(top50_path, skip_blank_lines=True)

            # HART GATE: Entferne alle combo_pair_ids mit robustness_score_1_jittered_80 < 0.8
            if "robustness_score_1_jittered_80" in top50.columns:
                initial_count = (
                    len(top50["combo_pair_id"].unique())
                    if "combo_pair_id" in top50.columns
                    else len(top50)
                )

                # Finde combo_pair_ids, die das Gate nicht bestehen
                failed_pairs = (
                    set(
                        top50[top50["robustness_score_1_jittered_80"] < 0.8][
                            "combo_pair_id"
                        ].unique()
                    )
                    if "combo_pair_id" in top50.columns
                    else set()
                )

                if failed_pairs:
                    print(
                        f"[Matrix] Hart Gate ({group.group_id}): {len(failed_pairs)} combo_pair_ids haben robustness_score_1_jittered_80 < 0.8 und werden entfernt"
                    )
                    top50 = top50[~top50["combo_pair_id"].isin(failed_pairs)]
                    final_count = (
                        len(top50["combo_pair_id"].unique())
                        if "combo_pair_id" in top50.columns
                        else len(top50)
                    )
                    print(
                        f"[Matrix] Hart Gate ({group.group_id}): {initial_count} -> {final_count} combo_pair_ids verbleiben"
                    )

            # Nur Zeilen mit combo_leg == 'A'
            if "combo_leg" in top50.columns:
                top50_a = top50[top50["combo_leg"] == "A"]
            else:
                top50_a = top50

            if top50_a.empty:
                continue

            # Pro Paar eine Zeile in der Matrix
            for _, row in top50_a.iterrows():
                combo_pair_id = str(row.get("combo_pair_id", ""))

                # Pfade zu den aggregierten Dateien dieser spezifischen combo_pair_id
                combo_dir = group.output_dir / "combo_pairs" / combo_pair_id
                equity_path = combo_dir / "equity.csv"
                trades_path = combo_dir / "trades.json"

                matrix_row = {
                    "group_id": group.group_id,
                    "symbol": group.symbol,
                    "timeframe": group.timeframe,
                    "direction": group.direction,
                    "combo_pair_id": combo_pair_id,
                    "equity_path": str(equity_path) if equity_path.exists() else "",
                    "trades_path": str(trades_path) if trades_path.exists() else "",
                }

                # Kennzeichnen, ob verfeinerte Top-Liste verwendet wurde
                matrix_row["top50_source"] = top50_path.name

                # Nur ausgewählte Robustness-Spalte übernehmen
                # tp_sl_stress_score gehört zu robustness_1_mean (per-leg robustness)
                for col in (
                    "robustness_1_mean",
                    "stability_score_combined",
                    "comp_score_combined",
                ):
                    if col in row.index and col not in matrix_row:
                        matrix_row[col] = row[col]

                rows.append(matrix_row)

        except Exception as e:
            print(f"[Matrix] FEHLER beim Verarbeiten von {group.group_id}: {e}")

    if not rows:
        print("[Matrix] Keine Daten für Matrix gefunden")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[Matrix] Matrix aufgebaut: {len(df)} Zeilen")

    return df


# =============================
# Pre-processing: Pareto + Clustering per group
# =============================


def _numeric_feature_columns(
    df: pd.DataFrame, exclude: Optional[Set[str]] = None
) -> List[str]:
    """Return numeric columns suitable as features, excluding known meta/path cols.

    Clustering-Feature-Logik:
    1) Wenn Equity-Shape Features vorhanden sind (und mind. ein non-NaN), nutze NUR diese.
    2) Sonst: nutze Standardmetriken (wenn vorhanden).
    3) Sonst: nutze alle übrigen numerischen Spalten.
    """
    if df is None or df.empty:
        return []
    exclude = exclude or set()
    # Hard exclude list of meta/path-like columns
    hard_exclude = set(
        [
            "group_id",
            "symbol",
            "timeframe",
            "direction",
            "combo_pair_id",
            "equity_path",
            "trades_path",
            "top50_source",
            "source_walkforward",
            "combo_id",
            "combo_leg",
            "strategy_name",
            "szenario",
            "final_score",
            "comp_score",
            "robustness_mean",
            "stability_score_monthly",
            "identical_trades_entry",
            "identical_trades_absolut",
        ]
    )
    exclude = exclude.union(hard_exclude)

    # 1) Equity-Shape Features (wenn vorhanden): exklusiv verwenden
    equity_shape_cols = [
        "equity_returns_skew",
        "equity_returns_kurtosis",
        "equity_returns_autocorr",
        "equity_returns_volatility",
    ]
    equity_cols = [
        c
        for c in equity_shape_cols
        if c in df.columns and c not in exclude and df[c].notna().any()
    ]
    if equity_cols:
        return equity_cols

    # 2) Standardmetriken (wenn vorhanden)
    standard_metric_cols = [
        "total_profit_over_dd",
        "avg_r",
        "winrate",
        "total_max_dd",
        "stability_score_combined",
        "comp_score_combined",
    ]
    metric_cols = [
        c
        for c in standard_metric_cols
        if c in df.columns and c not in exclude and df[c].notna().any()
    ]
    if metric_cols:
        return metric_cols

    # 3) Fallback: alle anderen numerischen Spalten
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # numeric dtypes
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().any():
                num_cols.append(c)

    return num_cols


def _pareto_dominated_mask(
    df: pd.DataFrame,
    maximize_cols: List[str],
    minimize_cols: Optional[List[str]] = None,
    eps_rel: float = 0.05,
) -> pd.Series:
    """Compute a boolean mask of dominated rows (True = dominated), using soft epsilon dominance.

    - maximize_cols: columns to maximize
    - minimize_cols: columns to minimize (internally inverted)
    - eps_rel: relative tolerance; row A is dominated by B if B is better or within eps on all metrics and strictly better (beyond eps) on at least one.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=bool)

    minimize_cols = minimize_cols or []
    # Build matrix of metrics, converting minimization to maximization by negating
    cols = list(maximize_cols) + list(minimize_cols)
    if not cols:
        return pd.Series([False] * len(df))

    M = []
    for c in maximize_cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(-np.inf).values
        M.append(vals)
    for c in minimize_cols:
        # negate to convert to max
        vals = pd.to_numeric(df[c], errors="coerce").fillna(np.inf).values
        M.append(-vals)
    if not M:
        return pd.Series([False] * len(df))
    X = np.vstack(M).T  # shape (n, k)

    n = X.shape[0]
    dominated = np.zeros(n, dtype=bool)

    # pairwise dominance check (n is small per group; typically <= ~50)
    for i in range(n):
        if dominated[i]:
            continue
        xi = X[i]
        # Skip rows with NaN or Inf values
        if not np.isfinite(xi).all():
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            xj = X[j]
            # Skip rows with NaN or Inf values
            if not np.isfinite(xj).all():
                continue
            # soft comparisons with relative eps
            # condition: j dominates i if for all dims: xj >= xi*(1 - eps_rel), and exists dim where xj > xi*(1 + eps_rel)
            # Use absolute eps scaled by |xi| to prevent zero issues
            ge_all = True
            gt_any = False
            for d in range(xi.shape[0]):
                base = xi[d]
                # Skip dimension if base is not finite
                if not np.isfinite(base):
                    ge_all = False
                    break
                tol = abs(base) * eps_rel
                if not np.isfinite(xj[d]):
                    ge_all = False
                    break
                if xj[d] + 1e-12 < base - tol:  # clearly worse beyond tol
                    ge_all = False
                    break
                if xj[d] > base + tol:
                    gt_any = True
            if ge_all and gt_any:
                dominated[i] = True
                break
    return pd.Series(dominated, index=df.index)


def apply_pareto_filter_per_group(
    matrix: pd.DataFrame,
    maximize_cols: Optional[List[str]] = None,
    minimize_cols: Optional[List[str]] = None,
    eps_rel: float = 0.01,
) -> pd.DataFrame:
    """Apply Pareto filter within each group_id to remove clearly dominated combos.

    Defaults:
      - maximize: ["comp_score_combined","robustness_mean","stability_score_combined"]
      - minimize: ["identical_trades_absolut_percentage"]

    Note: eps_rel reduced from 0.05 to 0.02 to be less aggressive
    (only removes solutions that are clearly dominated across all dimensions)
    """
    if matrix is None or matrix.empty:
        return matrix

    # Default columns, filtered by availability
    default_maximize = [
        "comp_score_combined",
        "robustness_mean",
        "stability_score_combined",
    ]
    default_minimize = ["identical_trades_absolut_percentage"]

    if maximize_cols is None:
        maximize_cols = [c for c in default_maximize if c in matrix.columns]
    if minimize_cols is None:
        minimize_cols = [c for c in default_minimize if c in matrix.columns]

    # Fallback: use any available numeric columns if defaults don't exist
    if not maximize_cols and not minimize_cols:
        print("[Pareto] Warnung: Keine Standard-Spalten gefunden, überspringe Filter")
        return matrix

    parts: List[pd.DataFrame] = []
    for gid, sub in matrix.groupby("group_id"):
        if sub.shape[0] <= 2:
            parts.append(sub)
            continue
        # Count rows with NaN values that will be skipped
        numeric_cols = [
            c for c in (maximize_cols or []) + (minimize_cols or []) if c in sub.columns
        ]
        n_with_nan = sub[numeric_cols].isna().any(axis=1).sum() if numeric_cols else 0
        if n_with_nan > 0:
            print(
                f"[Pareto] Warnung: {n_with_nan} Zeilen in group {gid} haben NaN/Inf – werden übersprungen"
            )

        mask_dom = _pareto_dominated_mask(
            sub, maximize_cols, minimize_cols, eps_rel=eps_rel
        )
        kept = sub[~mask_dom].copy()
        # Always keep at least top-1 by final_score if everything dominated accidentally
        if kept.empty:
            fallback_col = (
                "final_score"
                if "final_score" in sub.columns
                else (maximize_cols[0] if maximize_cols else sub.columns[0])
            )
            kept = sub.nlargest(1, fallback_col)
        parts.append(kept)
    out = pd.concat(parts, ignore_index=True)
    print(
        f"[Pareto] Reduktion: {len(matrix)} -> {len(out)} Zeilen (eps_rel={eps_rel}, weniger aggressiv)"
    )
    return out


def _calculate_composite_score_for_representatives(row: pd.Series) -> float:
    """
    Berechnet einen gewichteten Composite Score für Cluster-Repräsentanten.

    Score = 0.5 * comp_score_combined + 0.25 * stability_score_combined + 0.25 * robustness_mean

    Falls eine Komponente fehlt (NaN), wird sie als 0 behandelt.

    Args:
        row: Eine Pandas Series aus einem DataFrame

    Returns:
        float: Der berechnete Composite Score
    """
    comp_score = (
        float(row.get("comp_score_combined", 0.0))
        if pd.notna(row.get("comp_score_combined"))
        else 0.0
    )
    stability_score = (
        float(row.get("stability_score_combined", 0.0))
        if pd.notna(row.get("stability_score_combined"))
        else 0.0
    )
    robustness_mean = (
        float(row.get("robustness_mean", 0.0))
        if pd.notna(row.get("robustness_mean"))
        else 0.0
    )

    composite = (0.5 * comp_score) + (0.25 * stability_score) + (0.25 * robustness_mean)
    return composite


def _select_cluster_representatives(
    sub: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    score_cols_priority: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Cluster rows and keep best representative per cluster.
    Tries scikit-learn KMeans; falls back to diversity-based greedy selection if sklearn not available.
    Uses composite score (0.5*comp_score_combined + 0.25*stability_score_combined + 0.25*robustness_mean)
    as primary ranking criterion, then comp_score_combined, then total_profit_over_dd.
    """
    if sub.shape[0] <= n_clusters or n_clusters <= 1 or not feature_cols:
        return sub

    # Use existing persistent _composite_rep_score column (calculated in compute_additional_scores)
    # Score-Priorität: neuer Composite Score, dann comp_score_combined, dann total_profit_over_dd
    score_cols_priority = score_cols_priority or [
        "_composite_rep_score",
        "comp_score_combined",
        "total_profit_over_dd",
    ]

    sub = sub.copy()

    X = sub[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0).replace(0.0, 1.0)
    Z = (X - mu) / sigma

    labels = None
    try:
        # Lazy import
        from sklearn.cluster import KMeans

        k = max(1, min(n_clusters, sub.shape[0]))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Z.values)
    except Exception:
        # Fallback: greedy farthest-point sampling to ensure diversity
        k = max(1, min(n_clusters, sub.shape[0]))
        pts = Z.values
        n = pts.shape[0]

        # start with the highest score row as seed
        def _row_score(r: pd.Series) -> Tuple[float, ...]:
            vals = []
            for c in score_cols_priority:
                if c in r.index and pd.notna(r[c]):
                    vals.append(float(r[c]))
                else:
                    vals.append(-1e9)
            return tuple(vals)

        start_idx = int(
            sub.sort_values(
                score_cols_priority[0], ascending=False, na_position="last"
            ).index[0]
        )
        chosen = [list(sub.index).index(start_idx)]
        while len(chosen) < k:
            # compute distance to nearest chosen
            dmin = np.full(n, np.inf)
            for ci in chosen:
                d = np.linalg.norm(pts - pts[ci], axis=1)
                dmin = np.minimum(dmin, d)
            # avoid already chosen
            dmin[chosen] = -np.inf
            next_idx = int(np.argmax(dmin))
            if next_idx in chosen:
                break
            chosen.append(next_idx)
        # Assign labels by nearest chosen center
        centers = pts[chosen]
        dists = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

    # pick best per cluster
    sub["_cluster_label"] = labels
    reps: List[pd.DataFrame] = []
    for c, grp in sub.groupby("_cluster_label"):
        # choose by priority score
        selector = None
        for sc in score_cols_priority:
            if sc in grp.columns:
                selector = sc
                break
        if selector is None:
            reps.append(grp.head(1))
        else:
            reps.append(
                grp.sort_values(selector, ascending=False, na_position="last").head(1)
            )
    out = pd.concat(reps, ignore_index=True)
    # Drop only _cluster_label; preserve _composite_rep_score for downstream reuse
    out = out.drop(columns=["_cluster_label"], errors="ignore")
    return out


def apply_clustering_per_group(
    matrix: pd.DataFrame,
    target_cluster_size: int = 4,
) -> pd.DataFrame:
    """Cluster the remaining combos per group to reduce redundancy.

    HF-STIL CLUSTERING: Nutzt primär Equity-Shape Features (Return-Profil):
    - equity_returns_skew, kurtosis, autocorr, volatility

    WICHTIG: Wenn Equity-Shape Features vorhanden sind, wird exklusiv darauf geclustert.
    Nur wenn diese fehlen, wird auf Standardmetriken (und erst dann auf sonstige Numerik) zurückgefallen.
    Ziel: Unterschiedliche Risiko-/Return-Profile selektieren, nicht nur Parameter-Varianten.

    target_cluster_size ~ desired items per cluster; K estimated as ceil(n/target_cluster_size).
    """
    if matrix is None or matrix.empty:
        return matrix

    parts: List[pd.DataFrame] = []
    for gid, sub in matrix.groupby("group_id"):
        n = sub.shape[0]
        if n <= target_cluster_size:
            parts.append(sub)
            continue
        feature_cols = _numeric_feature_columns(sub)
        if not feature_cols:
            parts.append(sub)
            continue
        k = max(1, min(n, int(np.ceil(n / float(target_cluster_size)))))
        reps = _select_cluster_representatives(sub, feature_cols, n_clusters=k)
        parts.append(reps)
    out = pd.concat(parts, ignore_index=True)
    print(f"[Cluster] Reduktion: {len(matrix)} -> {len(out)} Zeilen (nach Clustering)")
    return out


def prune_to_topK_diverse_per_group(
    matrix: pd.DataFrame,
    K: int = 3,
    min_diversifier_score: float = 0.0,
) -> pd.DataFrame:
    """Harte Obergrenze pro group_id: Behält maximal K Elemente mit Diversifikation.

    HF-STIL SELEKTION nach Clustering:
    1. Bestes _composite_rep_score (Performance-Champion)
    2. Stabilitäts-Champion mit stability_shape_score:
       - nutzt stability_score_combined und Equity-Shape-Features
         (equity_returns_skew, equity_returns_kurtosis,
          equity_returns_autocorr, equity_returns_volatility)
       - Score-Formel (auf 0-1 normalisierten Komponenten):
         stability_shape_score =
             0.40 * stab_good
           + 0.10 * skew_good
           + 0.20 * kurt_good
           + 0.20 * vol_good
           + 0.10 * acorr_good
         wobei:
           stab_good  = stab_rank
           skew_good  = skew_rank
           kurt_good  = 1 - kurt_rank
           vol_good   = 1 - vol_rank
           acorr_good = 1 - acorr_rank
    3. Diversifier: Größter Abstand im Return-Space zum #1, aber mit Mindestscore

    Args:
        matrix: Matrix nach Clustering
        K: Maximale Anzahl Elemente pro Gruppe (Standard: 3, Minimum: 1)
        min_diversifier_score: Mindestwert für _composite_rep_score beim Diversifier

    Returns:
        Gefilterte Matrix mit maximal K Einträgen pro group_id
    """
    if matrix is None or matrix.empty:
        return matrix

    # Validiere K (mindestens 1)
    K = max(1, K)

    parts: List[pd.DataFrame] = []

    # Equity-Shape Features für Distanzberechnung
    equity_shape_cols = [
        "equity_returns_skew",
        "equity_returns_kurtosis",
        "equity_returns_autocorr",
        "equity_returns_volatility",
    ]

    for gid, sub in matrix.groupby("group_id"):
        n = sub.shape[0]

        # Wenn <= K Elemente: alle behalten
        if n <= K:
            parts.append(sub)
            continue

        # Sortiere nach _composite_rep_score (absteigend)
        sub = sub.sort_values(
            "_composite_rep_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

        selected_indices = []

        # 1. Bestes _composite_rep_score
        selected_indices.append(0)

        # 2. Stabilitäts-Champion via stability_shape_score (falls != #1)
        if (
            K >= 2
            and "stability_score_combined" in sub.columns
            and all(c in sub.columns for c in equity_shape_cols)
        ):
            # Robuste, rangbasierte Normalisierung der Komponenten innerhalb der Gruppe
            def _rank_01(series: pd.Series) -> pd.Series:
                s = pd.to_numeric(series, errors="coerce")
                if s.notna().sum() <= 1:
                    return pd.Series(np.zeros(len(s)), index=s.index)
                ranks = s.rank(method="average", pct=True)
                return ranks.fillna(0.0)

            stab_rank = _rank_01(sub["stability_score_combined"])

            skew_raw = pd.to_numeric(sub["equity_returns_skew"], errors="coerce").clip(
                lower=0.0
            )
            skew_rank = _rank_01(skew_raw)

            kurt_rank = _rank_01(sub["equity_returns_kurtosis"])
            vol_rank = _rank_01(sub["equity_returns_volatility"])

            acorr_abs = pd.to_numeric(
                sub["equity_returns_autocorr"], errors="coerce"
            ).abs()
            acorr_rank = _rank_01(acorr_abs)

            # Gute Komponenten (je höher, desto besser)
            stab_good = stab_rank
            skew_good = skew_rank
            kurt_good = 1.0 - kurt_rank
            vol_good = 1.0 - vol_rank
            acorr_good = 1.0 - acorr_rank

            # Stability-Shape-Score gemäß Gewichtung
            stability_shape_score = (
                0.50 * stab_good
                + 0.13 * skew_good
                + 0.13 * kurt_good
                + 0.12 * vol_good
                + 0.12 * acorr_good
            )

            best_stab_idx = int(stability_shape_score.idxmax())
            if best_stab_idx not in selected_indices:
                selected_indices.append(best_stab_idx)

        # 3. Diversifier: Größter Abstand im Return-Space zum #1
        if K >= 3 and len(selected_indices) < K:
            # Return-Space Features extrahieren
            available_cols = [c for c in equity_shape_cols if c in sub.columns]

            if available_cols:
                # Erstelle Feature-Matrix
                X = sub[available_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # Standardisierung
                mu = X.mean(axis=0)
                sigma = X.std(axis=0).replace(0.0, 1.0)
                Z = (X - mu) / sigma

                # Referenzpunkt: #1 (bestes _composite_rep_score)
                ref_point = Z.iloc[selected_indices[0]].values

                # Berechne Distanzen zu allen anderen Punkten
                distances = []
                for i in range(len(Z)):
                    if i in selected_indices:
                        distances.append(-np.inf)  # Bereits ausgewählt
                    else:
                        # Euklidische Distanz im Return-Space
                        dist = float(np.linalg.norm(Z.iloc[i].values - ref_point))

                        # Prüfe Mindestscore
                        score = pd.to_numeric(
                            sub.iloc[i]["_composite_rep_score"], errors="coerce"
                        )
                        if pd.isna(score) or score < min_diversifier_score:
                            distances.append(-np.inf)  # Score zu niedrig
                        else:
                            distances.append(dist)

                # Wähle Index mit größter Distanz
                if distances and max(distances) > -np.inf:
                    diversifier_idx = int(np.argmax(distances))
                    if diversifier_idx not in selected_indices:
                        selected_indices.append(diversifier_idx)

        # Falls weniger als K Elemente gefunden: Fülle mit nächstbesten _composite_rep_score auf
        while len(selected_indices) < min(K, n):
            for i in range(n):
                if i not in selected_indices:
                    selected_indices.append(i)
                    break

        # Behalte nur ausgewählte Zeilen
        selected = sub.iloc[selected_indices].copy()
        parts.append(selected)

    out = pd.concat(parts, ignore_index=True)
    print(
        f"[Prune-K] Reduktion: {len(matrix)} -> {len(out)} Zeilen (max {K} pro Gruppe)"
    )

    return out


def _load_equity_series_from_path(path: Path) -> Optional[pd.Series]:
    """Hilfsfunktion: Equity-CSV laden und als Series mit Timestamp-Index zurückgeben."""
    try:
        if not path.exists():
            return None
        # CSV-Load kann sehr häufig aufgerufen werden – beschleunige durch
        # dtype-Hints und nur benötigte Spalten laden, wenn möglich
        df = pd.read_csv(path)
        ts_col, eq_col = _detect_equity_columns(df)
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        eq_vals = pd.to_numeric(df[eq_col], errors="coerce")
        mask = ts.notna() & eq_vals.notna()
        if not mask.any():
            return None
        series = pd.Series(eq_vals[mask].values, index=ts[mask]).sort_index()
        return series
    except Exception:
        return None


def build_final_combo_index(matrix: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Baue einen Index: group_id -> Liste verfügbarer combo_pair_id Einträge mit Pfaden.
    Nur Einträge mit existierenden equity/trades Pfaden werden berücksichtigt.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    if matrix is None or matrix.empty:
        return index
    for _, row in matrix.iterrows():
        group_id = str(row.get("group_id", ""))
        combo_pair_id = str(row.get("combo_pair_id", ""))
        equity_path = Path(str(row.get("equity_path", "")))
        trades_path = Path(str(row.get("trades_path", "")))
        if not equity_path.exists() and not trades_path.exists():
            continue
        index.setdefault(group_id, []).append(
            {
                "combo_pair_id": combo_pair_id,
                "equity_path": equity_path,
                "trades_path": trades_path,
                "symbol": row.get("symbol", ""),
                "timeframe": row.get("timeframe", ""),
                "direction": row.get("direction", ""),
                # Wird in Schritt 4.5/5 berechnet; für Monte Carlo Ranking wichtig
                "robustness_mean": row.get("robustness_mean", np.nan),
            }
        )
    return index


def generate_final_combos(
    index: Dict[str, List[Dict[str, Any]]], max_combos: int = 1000
) -> List[Dict[str, Any]]:
    """
    Generiere Kombinationen, die für jede group_id genau eine combo_pair_id enthalten.
    Achtung: Vollständiges kartesisches Produkt kann explodieren (≈ 10^G). Daher limitierbar.

    Returns: Liste von Auswahl-Mappings {group_id: entry_dict}
    """
    from itertools import product

    groups = list(index.keys())
    if not groups:
        return []
    lists = [index[g] for g in groups]
    # Produkt iterator
    combos = []
    count = 0
    for tpl in product(*lists):
        selection = {groups[i]: tpl[i] for i in range(len(groups))}
        combos.append(selection)
        count += 1
        if max_combos and count >= max_combos:
            break
    return combos


def estimate_total_combinations(index: Dict[str, List[Dict[str, Any]]]) -> int:
    """
    Berechne die Gesamtanzahl möglicher finaler Kombinationen.
    """
    if not index:
        return 0
    total = 1
    for entries in index.values():
        total *= len(entries)
    return total


def compute_adaptive_level1_samples(
    index: Dict[str, List[Dict[str, Any]]],
    default_level1_samples: int = 10000,
) -> int:
    """
    Berechne adaptive Monte Carlo Samples: min(N_total, default * num_groups).

    Rational:
    - N_total = Gesamtzahl möglicher Kombinationen (aus Index)
    - x = Anzahl Gruppen
    - monte_carlo_samples = min(N_total, default * x)

    Dies stellt sicher, dass genug Samples gezogen werden, um den
    Portfolio-Raum angemessen abzudecken, skaliert mit Anzahl Gruppen.

    Args:
        index: Dict mapping group_id -> Liste von Kandidaten
        default_level1_samples: Standard-Default (normalerweise 10000)

    Returns:
        Adaptive Anzahl der Monte Carlo Samples
    """
    if not index:
        return max(1, default_level1_samples)

    num_groups = len(index)
    n_total = estimate_total_combinations(index)

    # Berechne: default * num_groups
    scaled_default = default_level1_samples * num_groups

    # Nimm das Minimum: min(N_total, scaled_default)
    adaptive_samples = min(n_total, scaled_default)

    print(
        f"[Adaptive] Groups={num_groups}, N_total={n_total}, "
        f"Default×Groups={scaled_default}, → monte_carlo_samples={adaptive_samples}"
    )

    return max(1, adaptive_samples)  # Mindestens 1


def _final_combo_id_from_selection(selection: Dict[str, Dict[str, Any]]) -> str:
    """Erzeuge eine deterministische final_combo_pair_id aus group_id:combo_pair_id Paaren."""
    parts = []
    for group_id in sorted(selection.keys()):
        parts.append(f"{group_id}={selection[group_id]['combo_pair_id']}")
    base = "__".join(parts)
    # Hash zur Kürzung
    import hashlib

    # SHA1 nur für nicht-kryptografische ID-Generierung verwendet
    digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :16
    ]  # nosec B324
    return f"final_{digest}"


# Global caches für massive Beschleunigung von Schritt 5
# OPTIMIERT: Shared Memory für echte Multi-Core Parallelität
_EQUITY_CACHE: Dict[str, Optional[pd.Series]] = {}
_TRADES_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Locks for thread-safety when accessing the global caches
_EQUITY_LOCK = Lock()
_TRADES_LOCK = Lock()

# Shared cache manager für Multi-Processing (wird in preload_equity_and_trades_cache initialisiert)
_SHARED_MANAGER: Optional[Any] = None
_SHARED_EQUITY_CACHE: Optional[Any] = None
_SHARED_TRADES_CACHE: Optional[Any] = None


def get_equity_cached(path_str: str) -> Optional[pd.Series]:
    """Thread-safe get-or-load for equity series cached in _EQUITY_CACHE.

    OPTIMIERT: Nutzt shared cache falls verfügbar (Multi-Processing),
    sonst lokalen cache (Single-Processing oder Thread-basiert).
    """
    # Fast-path: local cache (including negative cache entries: path -> None)
    if path_str in _EQUITY_CACHE:
        return _EQUITY_CACHE[path_str]

    # Priorität 1: Shared cache (für Multi-Processing) → unpickle once, then promote to local cache.
    if _SHARED_EQUITY_CACHE is not None:
        try:
            raw = _SHARED_EQUITY_CACHE.get(path_str)
        except Exception:
            raw = None
        if raw is not None:
            try:
                # Pickle von vertrauenswürdigen internen Daten (eigene Snapshots)
                s = pickle.loads(raw)  # nosec B301
            except Exception:
                s = None
            with _EQUITY_LOCK:
                _EQUITY_CACHE[path_str] = s
            return s

    # Load if missing (thread-safe). Store None as negative cache to prevent repeated disk hits.
    with _EQUITY_LOCK:
        if path_str in _EQUITY_CACHE:
            return _EQUITY_CACHE[path_str]
        s = _load_equity_series_from_path(Path(path_str))
        _EQUITY_CACHE[path_str] = s
        return s


def get_trades_cached(path_str: str) -> List[Dict[str, Any]]:
    """Thread-safe get-or-load for trades cached in _TRADES_CACHE.

    OPTIMIERT: Nutzt shared cache falls verfügbar (Multi-Processing),
    sonst lokalen cache (Single-Processing oder Thread-basiert).
    """
    # Fast-path: local cache (including empty lists)
    if path_str in _TRADES_CACHE:
        return _TRADES_CACHE[path_str]

    # Priorität 1: Shared cache (für Multi-Processing) → unpickle once, then promote to local cache.
    if _SHARED_TRADES_CACHE is not None:
        try:
            raw = _SHARED_TRADES_CACHE.get(path_str)
        except Exception:
            raw = None
        if raw is not None:
            try:
                # Pickle von vertrauenswürdigen internen Daten (eigene Snapshots)
                t = pickle.loads(raw)  # nosec B301
            except Exception:
                t = []
            with _TRADES_LOCK:
                _TRADES_CACHE[path_str] = t
            return t

    # Load if missing (thread-safe).
    with _TRADES_LOCK:
        if path_str in _TRADES_CACHE:
            return _TRADES_CACHE[path_str]
        t = _cached_read_trades_json(path_str)
        _TRADES_CACHE[path_str] = t
        return t


@lru_cache(maxsize=4096)
def _cached_read_trades_json(path_str: str) -> List[Dict[str, Any]]:
    """Kleiner Cache-Layer für Trades-JSON Lesevorgänge."""
    p = Path(path_str)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


@lru_cache(maxsize=1024)
def _load_top50_df_for(group_id: str, top50_source: str) -> Optional[pd.DataFrame]:
    """Thread-safe cached loader for a group's Top-50 CSV (uses lru_cache).
    Returns a DataFrame or None if not found/failed.
    """
    group_dir = COMBINED_MATRIX_DIR / group_id
    top50_path = group_dir / top50_source
    if not top50_path.exists():
        print(
            f"[Scores] Warnung: Top-50-Datei nicht gefunden für {group_id}: {top50_path}"
        )
        return None
    try:
        return pd.read_csv(top50_path, skip_blank_lines=True)
    except Exception as e:
        print(f"[Scores] Warnung: Konnte Top-50-Datei für {group_id} nicht laden: {e}")
        return None


def get_pair_robustness_mean_from_top50(row: pd.Series) -> Optional[float]:
    """
    Fallback: Wenn keine robustness_1_mean vorhanden ist, berechne
    robustness_mean als Mittelwert der robustness_score_1 über alle Legs
    (typisch Leg A und B) aus der ursprünglichen Top-50-CSV.
    """
    group_id = str(row.get("group_id", "")).strip()
    combo_pair_id = str(row.get("combo_pair_id", "")).strip()
    top50_source = str(row.get("top50_source", "")).strip()

    if not group_id or not combo_pair_id or not top50_source:
        return None

    top50_df = _load_top50_df_for(group_id, top50_source)
    if (
        top50_df is None
        or top50_df.empty
        or "robustness_score_1" not in top50_df.columns
    ):
        return None

    pair_rows = top50_df[top50_df["combo_pair_id"] == combo_pair_id]
    if pair_rows.empty:
        return None

    vals = pd.to_numeric(pair_rows["robustness_score_1"], errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        return None

    return float(vals.mean())


def preload_equity_and_trades_cache(
    matrix: pd.DataFrame, use_shared: bool = True
) -> None:
    """Lädt alle Equity- und Trades-Dateien aus der Matrix vorab in Speicher-Cache.

    OPTIMIERT: Erstellt shared memory cache für Multi-Processing wenn use_shared=True.

    Massiver Performance-Boost für Schritt 5: Statt jede Datei 100+ mal zu lesen,
    wird sie nur einmal geladen und im RAM gehalten (und über Prozesse geteilt).

    Args:
        matrix: DataFrame mit equity_path und trades_path Spalten
        use_shared: Wenn True, erstelle shared memory cache (für Multi-Processing)
    """
    global _SHARED_MANAGER, _SHARED_EQUITY_CACHE, _SHARED_TRADES_CACHE

    print("[Cache] Pre-loading equity und trades in RAM...")
    equity_paths = set()
    trades_paths = set()

    for _, row in matrix.iterrows():
        eq_path = row.get("equity_path", "")
        tr_path = row.get("trades_path", "")
        if eq_path and Path(eq_path).exists():
            equity_paths.add(eq_path)
        if tr_path and Path(tr_path).exists():
            trades_paths.add(tr_path)

    if use_shared:
        # Initialisiere shared memory manager
        _SHARED_MANAGER = Manager()
        _SHARED_EQUITY_CACHE = _SHARED_MANAGER.dict()
        _SHARED_TRADES_CACHE = _SHARED_MANAGER.dict()
        print(f"[Cache] Shared memory manager initialisiert für {cpu_count()} cores")

    # Equity laden
    for path_str in tqdm(equity_paths, desc="[Cache] Loading equity", unit="file"):
        s = _load_equity_series_from_path(Path(path_str))
        if s is not None:
            _EQUITY_CACHE[path_str] = s
            if use_shared and _SHARED_EQUITY_CACHE is not None:
                # Serialize und speichere in shared dict
                _SHARED_EQUITY_CACHE[path_str] = pickle.dumps(s)

    # Trades laden
    for path_str in tqdm(trades_paths, desc="[Cache] Loading trades", unit="file"):
        t = _cached_read_trades_json(path_str)
        if t:
            _TRADES_CACHE[path_str] = t
            if use_shared and _SHARED_TRADES_CACHE is not None:
                # Serialize und speichere in shared dict
                _SHARED_TRADES_CACHE[path_str] = pickle.dumps(t)

    cache_type = "shared" if use_shared else "local"
    print(
        f"[Cache] ✓ Geladen ({cache_type}): {len(_EQUITY_CACHE)} equity, {len(_TRADES_CACHE)} trades"
    )


@dataclass(frozen=True, slots=True)
class MonteCarloEvalState:
    """Vorbereiteter Zustand für schnelle Monte-Carlo-Evaluation (Schritt 5).

    Design-Ziel:
    - Monte Carlo Search soll NICHT pro Portfolio Equity/Trades-Objekte serialisieren.
    - Stattdessen werden pro combo_pair_id *einmal* kompakte Arrays/Aggregate gebaut
      und Portfolios dann batchweise in NumPy bewertet.
    """

    group_ids: Tuple[str, ...]
    combo_pair_ids_by_group: Tuple[Tuple[str, ...], ...]
    equity_daily_pnl_by_group: Tuple[
        np.ndarray, ...
    ]  # each: (K_g, T) float32 on evaluation grid
    trades_count_by_group: Tuple[np.ndarray, ...]  # each: (K_g,) int32
    trades_wins_by_group: Tuple[np.ndarray, ...]  # each: (K_g,) int32
    trades_sum_r_by_group: Tuple[np.ndarray, ...]  # each: (K_g,) float32
    robustness_by_group: Tuple[np.ndarray, ...]  # each: (K_g,) float32
    valid_by_group: Tuple[
        np.ndarray, ...
    ]  # each: (K_g,) bool; invalid candidates are excluded (final_score=-inf)
    daily_index_utc: (
        pd.DatetimeIndex
    )  # length T, tz-aware UTC (evaluation grid; may be event-based)
    month_end_positions: np.ndarray  # positions in daily_index_utc
    month_days_for_profits: np.ndarray  # length = len(month_end_positions)-1, float64
    start_equity: float = 100_000.0


def _trade_r_multiple(trade: Dict[str, Any]) -> Optional[float]:
    """Extract R-multiple from a trade record (robust to schema drift)."""
    if not isinstance(trade, dict):
        return None
    for key in ("r_multiple", "r", "R", "r_mult", "rMultiple"):
        if key in trade:
            try:
                v = float(trade[key])
                return v if np.isfinite(v) else None
            except Exception:
                return None
    return None


def _trade_aggregates(trades: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Return (n_trades, wins, sum_r) for a trade list."""
    if not trades:
        return 0, 0, 0.0
    n = 0
    wins = 0
    sum_r = 0.0
    for tr in trades:
        r = _trade_r_multiple(tr)
        if r is None:
            continue
        n += 1
        sum_r += float(r)
        if r > 0.0:
            wins += 1
    return n, wins, float(sum_r)


def _trade_exit_times_utc(trades: List[Dict[str, Any]]) -> pd.DatetimeIndex:
    """Extract exit times (UTC) from a list of trade dicts.

    We treat exit_time as the canonical timestamp that should appear in equity.csv,
    because equity updates are recorded when a trade is closed.
    """
    if not trades:
        return pd.DatetimeIndex([], tz="UTC")
    vals: List[Any] = []
    for tr in trades:
        if not isinstance(tr, dict):
            continue
        v = tr.get("exit_time") or tr.get("close_time")
        if v is None or v == "":
            continue
        vals.append(v)
    if not vals:
        return pd.DatetimeIndex([], tz="UTC")
    ts = pd.to_datetime(pd.Series(vals), utc=True, errors="coerce").dropna()
    if ts.empty:
        return pd.DatetimeIndex([], tz="UTC")
    # Ensure unique & sorted
    return pd.DatetimeIndex(ts.sort_values().unique(), tz="UTC")


def _trade_entry_min_utc(trades: List[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    """Best-effort earliest entry_time (UTC)."""
    if not trades:
        return None
    vals: List[Any] = []
    for tr in trades:
        if not isinstance(tr, dict):
            continue
        v = tr.get("entry_time") or tr.get("open_time")
        if v is None or v == "":
            continue
        vals.append(v)
    if not vals:
        return None
    ts = pd.to_datetime(pd.Series(vals), utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.min())


def _trade_exit_max_utc(trades: List[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    """Best-effort latest exit_time (UTC)."""
    ex = _trade_exit_times_utc(trades)
    return pd.Timestamp(ex.max()) if len(ex) else None


def _equity_matches_trade_exits(
    equity: pd.Series,
    trades: List[Dict[str, Any]],
    *,
    tolerance: pd.Timedelta,
) -> Tuple[bool, int]:
    """Return (ok, missing_count) for exit_time -> equity timestamp matching.

    We require that every trade exit timestamp has a corresponding equity timestamp
    (within tolerance). Additionally, equity must extend to the last exit.
    """
    if equity is None or equity.empty:
        ex = _trade_exit_times_utc(trades)
        return (len(ex) == 0), int(len(ex))

    ex = _trade_exit_times_utc(trades)
    if len(ex) == 0:
        return True, 0

    eq_idx = pd.DatetimeIndex(equity.index).tz_convert("UTC")
    eq_idx = eq_idx.sort_values()

    # 1) Hard end-coverage check (within tolerance)
    try:
        eq_end = pd.Timestamp(eq_idx.max())
    except Exception:
        return False, int(len(ex))
    ex_end = pd.Timestamp(ex.max())
    if (eq_end + tolerance) < ex_end:
        # If equity ends before the last trade exit, we definitely have missing equity updates.
        return False, int(len(ex))

    # 2) For each exit_time, ensure a matching equity timestamp exists (nearest within tolerance)
    # Requires monotonic index
    idx = eq_idx.get_indexer(ex, method="nearest", tolerance=tolerance)
    missing = int(np.sum(idx < 0))
    return missing == 0, missing


def _build_event_index_utc(
    equity_indices: List[pd.DatetimeIndex],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_month_ends: bool = True,
) -> pd.DatetimeIndex:
    """Build an event-based evaluation grid from the union of equity timestamps.

    Additionally, we inject explicit month-end EOD timestamps so monthly stability
    metrics remain well-defined even in months without trades.
    """
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.DatetimeIndex([], tz="UTC")

    # Collect all timestamps as int64 ns for fast unique/sort.
    parts: List[np.ndarray] = []
    for idx in equity_indices:
        if idx is None or len(idx) == 0:
            continue
        try:
            ii = pd.DatetimeIndex(idx).tz_convert("UTC")
            parts.append(ii.asi8)
        except Exception:
            continue

    if include_month_ends:
        # Month-end at 23:59:59 UTC
        start_m = pd.Timestamp(start).tz_convert("UTC").normalize()
        end_m = pd.Timestamp(end).tz_convert("UTC").normalize()
        me = pd.date_range(start=start_m, end=end_m, freq="ME", tz="UTC")
        me = me + pd.Timedelta(hours=23, minutes=59, seconds=59)
        if len(me):
            parts.append(me.asi8)

    # Always include the global start and end day EOD to stabilize endpoints
    s_eod = pd.Timestamp(start).tz_convert("UTC")
    e_eod = pd.Timestamp(end).tz_convert("UTC")
    parts.append(np.array([s_eod.value, e_eod.value], dtype=np.int64))

    if not parts:
        return pd.DatetimeIndex([], tz="UTC")

    arr = np.unique(np.concatenate(parts))
    arr.sort()
    out = pd.DatetimeIndex(arr, tz="UTC")
    # Clip to [start,end]
    return out[(out >= s_eod) & (out <= e_eod)]


def _build_daily_index_utc(
    starts: List[pd.Timestamp],
    ends: List[pd.Timestamp],
    *,
    mode: str = "intersection",
) -> pd.DatetimeIndex:
    """Build a common daily (EOD UTC) evaluation grid.

    Modes:
    - "intersection": conservative common window (start=max(starts), end=min(ends))
    - "union": full window (start=min(starts), end=max(ends))

    Note: This grid is used for Monte-Carlo evaluation and later hydration. When using
    "union", candidates with insufficient end-coverage should be handled explicitly
    (see ``valid_by_group``).
    """
    if not starts or not ends:
        # Fallback: empty
        return pd.DatetimeIndex([], tz="UTC")
    mode = str(mode or "intersection").strip().lower()
    if mode not in ("intersection", "union"):
        raise ValueError(f"Unbekannter mode für _build_daily_index_utc: {mode}")

    if mode == "intersection":
        start = max(starts)
        end = min(ends)
    else:
        start = min(starts)
        end = max(ends)
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.DatetimeIndex([], tz="UTC")
    # Use end-of-day timestamps for stable daily snapshots (better matches "last" semantics).
    start_day = pd.Timestamp(start).tz_convert("UTC").normalize()
    end_day = pd.Timestamp(end).tz_convert("UTC").normalize()
    days = pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC")
    return days + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _month_end_positions(index_utc: pd.DatetimeIndex) -> np.ndarray:
    if index_utc is None or len(index_utc) == 0:
        return np.array([], dtype=int)
    # Avoid pandas warning: converting tz-aware DatetimeIndex to PeriodIndex drops tz info.
    # We only need month boundaries, so use an integer month key (YYYYMM) instead.
    month_keys = (index_utc.year * 100 + index_utc.month).to_numpy()
    change = month_keys[:-1] != month_keys[1:]
    positions = np.where(change)[0]
    # month-end = last day of each month in the index
    return np.append(positions, len(month_keys) - 1).astype(int)


def prepare_monte_carlo_eval_state(
    index: Dict[str, List[Dict[str, Any]]],
    *,
    start_equity: float = 100_000.0,
    window_mode: str = "union",
    grid: str = "events",
    trade_equity_tolerance_seconds: float = 60.0,
    fail_on_trade_equity_mismatch: bool = True,
) -> MonteCarloEvalState:
    """Precompute compact per-candidate arrays for fast Monte-Carlo evaluation.

    Notes (Correctness-first):
    - Default grid is event-based (union of equity timestamps), not daily. This avoids
        under/over-stating drawdowns when multiple closes happen within a day.
    - Strategies may stop trading due to regime → that's fine: equity stays flat via ffill.
    - Critical invariant: equity timestamps must match trade exit timestamps, because
        equity updates are recorded when trades are closed. If trades extend beyond equity
        (or exit-times can't be matched), we abort by default (or mark candidate invalid).
    """
    group_ids = tuple(sorted(index.keys()))
    if not group_ids:
        return MonteCarloEvalState(
            group_ids=tuple(),
            combo_pair_ids_by_group=tuple(),
            equity_daily_pnl_by_group=tuple(),
            trades_count_by_group=tuple(),
            trades_wins_by_group=tuple(),
            trades_sum_r_by_group=tuple(),
            robustness_by_group=tuple(),
            valid_by_group=tuple(),
            daily_index_utc=pd.DatetimeIndex([], tz="UTC"),
            month_end_positions=np.array([], dtype=int),
            month_days_for_profits=np.array([], dtype=float),
            start_equity=float(start_equity),
        )

    # Load all candidate equities once to define the common date range / event grid
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    trade_starts: List[pd.Timestamp] = []
    trade_ends: List[pd.Timestamp] = []
    equity_series_by_group: List[List[Optional[pd.Series]]] = []
    equity_indices_all: List[pd.DatetimeIndex] = []
    for gid in group_ids:
        series_list: List[Optional[pd.Series]] = []
        for entry in index.get(gid, []):
            s = get_equity_cached(str(entry.get("equity_path", "")))
            if s is None or s.empty:
                series_list.append(None)
                continue
            s = s.sort_index()
            series_list.append(s)
            starts.append(pd.Timestamp(s.index.min()))
            ends.append(pd.Timestamp(s.index.max()))
            try:
                equity_indices_all.append(pd.DatetimeIndex(s.index))
            except Exception:
                pass

            # Trade start/end bounds help define a correct global window
            trades = get_trades_cached(str(entry.get("trades_path", "")))
            t0 = _trade_entry_min_utc(trades)
            t1 = _trade_exit_max_utc(trades)
            if t0 is not None:
                trade_starts.append(pd.Timestamp(t0))
            if t1 is not None:
                trade_ends.append(pd.Timestamp(t1))
        equity_series_by_group.append(series_list)

    # Global window bounds
    all_starts = [*starts, *trade_starts]
    all_ends = [*ends, *trade_ends]
    if not all_starts or not all_ends:
        daily_index = pd.DatetimeIndex([], tz="UTC")
    else:
        global_start = pd.Timestamp(min(all_starts))
        global_end = pd.Timestamp(max(all_ends))

        grid = str(grid or "events").strip().lower()
        if grid not in ("events", "daily"):
            raise ValueError(f"Unbekannter Monte-Carlo grid: {grid}")

        if grid == "daily":
            daily_index = _build_daily_index_utc(all_starts, all_ends, mode=window_mode)
        else:
            daily_index = _build_event_index_utc(
                equity_indices_all,
                start=global_start,
                end=global_end,
                include_month_ends=True,
            )

    if len(daily_index) < 2:
        # Too little data → keep empty, downstream will handle gracefully
        daily_index = pd.DatetimeIndex([], tz="UTC")

    tol = pd.Timedelta(seconds=float(trade_equity_tolerance_seconds))

    month_end_pos = _month_end_positions(daily_index)
    if len(month_end_pos) >= 2:
        month_end_dates = daily_index[month_end_pos]
        # Profit months correspond to month_end_dates[1:], weights by that month
        month_days_for_profits = np.asarray(
            [pd.Period(ts, freq="M").days_in_month for ts in month_end_dates[1:]],
            dtype=float,
        )
    else:
        month_days_for_profits = np.array([], dtype=float)

    combo_pair_ids_by_group: List[Tuple[str, ...]] = []
    equity_daily_pnl_by_group: List[np.ndarray] = []
    trades_count_by_group: List[np.ndarray] = []
    trades_wins_by_group: List[np.ndarray] = []
    trades_sum_r_by_group: List[np.ndarray] = []
    robustness_by_group: List[np.ndarray] = []
    valid_by_group: List[np.ndarray] = []

    for gid, series_list in zip(group_ids, equity_series_by_group):
        entries = index.get(gid, [])
        combo_ids = tuple(str(e.get("combo_pair_id", "")) for e in entries)
        combo_pair_ids_by_group.append(combo_ids)

        k = len(entries)
        if k == 0 or len(daily_index) == 0:
            equity_daily_pnl_by_group.append(np.zeros((k, 0), dtype=np.float32))
            trades_count_by_group.append(np.zeros((k,), dtype=np.int32))
            trades_wins_by_group.append(np.zeros((k,), dtype=np.int32))
            trades_sum_r_by_group.append(np.zeros((k,), dtype=np.float32))
            robustness_by_group.append(np.full((k,), np.nan, dtype=np.float32))
            valid_by_group.append(np.zeros((k,), dtype=bool))
            continue

        pnl = np.zeros((k, len(daily_index)), dtype=np.float32)
        t_count = np.zeros((k,), dtype=np.int32)
        t_wins = np.zeros((k,), dtype=np.int32)
        t_sum_r = np.zeros((k,), dtype=np.float32)
        rob = np.full((k,), np.nan, dtype=np.float32)
        valid = np.ones((k,), dtype=bool)

        for i, (entry, s) in enumerate(zip(entries, series_list)):
            # Robustness (already computed earlier in the pipeline)
            try:
                rob[i] = float(
                    pd.to_numeric(entry.get("robustness_mean", np.nan), errors="coerce")
                )
            except Exception:
                rob[i] = np.nan

            # Trades aggregates (cheap + additive)
            trades = get_trades_cached(str(entry.get("trades_path", "")))
            n_tr, n_w, sum_r = _trade_aggregates(trades)
            t_count[i] = int(n_tr)
            t_wins[i] = int(n_w)
            t_sum_r[i] = float(sum_r)

            # Equity PnL on evaluation grid
            if s is None or s.empty:
                valid[i] = False
                continue
            s = s.sort_index()

            # Trade↔Equity consistency:
            # equity timestamps must match exit_time timestamps (within tolerance).
            ok, missing = _equity_matches_trade_exits(s, trades, tolerance=tol)
            if not ok:
                msg = (
                    f"Trade/Equity-Mismatch in Monte Carlo: group_id={gid} combo_pair_id={entry.get('combo_pair_id')} "
                    f"missing_exit_matches={missing} tol={tol} "
                    f"equity_path={entry.get('equity_path')} trades_path={entry.get('trades_path')}"
                )
                if fail_on_trade_equity_mismatch:
                    raise ValueError(
                        msg
                        + "\n"
                        + "Equity.csv muss Exit-Zeitpunkte der Trades abdecken/treffen, da Equity-Einträge beim Close entstehen."
                    )
                valid[i] = False
                continue

            aligned = s.reindex(daily_index, method="ffill")
            if aligned.isna().all():
                valid[i] = False
                continue
            # Correct baseline before first close: flat start_equity.
            aligned = aligned.astype(float).ffill().fillna(float(start_equity))
            if len(aligned) > 0:
                aligned.iloc[0] = float(start_equity)
            diff = aligned.diff().fillna(0.0).to_numpy(dtype=np.float32, na_value=0.0)
            pnl[i, :] = diff

        equity_daily_pnl_by_group.append(pnl)
        trades_count_by_group.append(t_count)
        trades_wins_by_group.append(t_wins)
        trades_sum_r_by_group.append(t_sum_r)
        robustness_by_group.append(rob)
        valid_by_group.append(valid)

    return MonteCarloEvalState(
        group_ids=group_ids,
        combo_pair_ids_by_group=tuple(combo_pair_ids_by_group),
        equity_daily_pnl_by_group=tuple(equity_daily_pnl_by_group),
        trades_count_by_group=tuple(trades_count_by_group),
        trades_wins_by_group=tuple(trades_wins_by_group),
        trades_sum_r_by_group=tuple(trades_sum_r_by_group),
        robustness_by_group=tuple(robustness_by_group),
        valid_by_group=tuple(valid_by_group),
        daily_index_utc=daily_index,
        month_end_positions=month_end_pos,
        month_days_for_profits=month_days_for_profits,
        start_equity=float(start_equity),
    )


_MC_EVAL_STATE: Optional[MonteCarloEvalState] = None


def _init_monte_carlo_worker(state: MonteCarloEvalState) -> None:
    # Set global state in worker (spawn-safe; each process receives state once).
    global _MC_EVAL_STATE
    _MC_EVAL_STATE = state


def _evaluate_indices_batch_fast(selections: np.ndarray) -> Dict[str, Any]:
    """Worker: evaluate a batch of portfolios given integer-coded selections.

    Returns a dict of numpy arrays (one entry per portfolio), plus the selections array.
    """
    state = _MC_EVAL_STATE
    if state is None:
        raise RuntimeError(
            "MonteCarloEvalState ist nicht initialisiert (worker initializer fehlt)."
        )
    if selections.ndim != 2 or selections.shape[1] != len(state.group_ids):
        raise ValueError("Ungültige selections-Form für Monte-Carlo-Evaluation.")

    b = selections.shape[0]
    t = len(state.daily_index_utc)

    # Candidate validity: exclude portfolios that include any invalid candidate
    valid_portfolio = np.ones((b,), dtype=bool)
    for gi in range(len(state.group_ids)):
        idxs = selections[:, gi].astype(np.int64, copy=False)
        if gi < len(state.valid_by_group):
            v = state.valid_by_group[gi]
            if v is not None and len(v) > 0:
                valid_portfolio &= v[idxs].astype(bool, copy=False)

    # Aggregate trades & robustness (cheap)
    total_trades = np.zeros((b,), dtype=np.int64)
    total_wins = np.zeros((b,), dtype=np.int64)
    sum_r = np.zeros((b,), dtype=np.float64)
    rob_sum = np.zeros((b,), dtype=np.float64)
    rob_count = np.zeros((b,), dtype=np.int64)

    # Aggregate daily PnL
    pnl = np.zeros((b, t), dtype=np.float32)

    for gi in range(len(state.group_ids)):
        idxs = selections[:, gi].astype(np.int64, copy=False)

        pnl += state.equity_daily_pnl_by_group[gi][idxs]

        tc = state.trades_count_by_group[gi][idxs].astype(np.int64, copy=False)
        tw = state.trades_wins_by_group[gi][idxs].astype(np.int64, copy=False)
        sr = state.trades_sum_r_by_group[gi][idxs].astype(np.float64, copy=False)
        total_trades += tc
        total_wins += tw
        sum_r += sr

        rvals = state.robustness_by_group[gi][idxs].astype(np.float64, copy=False)
        mask = np.isfinite(rvals)
        rob_sum += np.where(mask, rvals, 0.0)
        rob_count += mask.astype(np.int64)

    robustness_mean = np.where(rob_count > 0, rob_sum / rob_count, np.nan)

    # Invalidate aggregates for invalid portfolios (keeps downstream formulas stable)
    if not np.all(valid_portfolio):
        total_trades = np.where(valid_portfolio, total_trades, 0)
        total_wins = np.where(valid_portfolio, total_wins, 0)
        sum_r = np.where(valid_portfolio, sum_r, np.nan)
        robustness_mean = np.where(valid_portfolio, robustness_mean, np.nan)

    # Equity from daily PnL
    if t == 0:
        total_profit = np.full((b,), np.nan, dtype=np.float64)
        max_dd = np.full((b,), np.nan, dtype=np.float64)
        profit_over_dd = np.full((b,), np.nan, dtype=np.float64)
        equity = np.zeros((b, 0), dtype=np.float64)
    else:
        equity = state.start_equity + np.cumsum(pnl.astype(np.float64), axis=1)
        total_profit = equity[:, -1] - equity[:, 0]
        roll_max = np.maximum.accumulate(equity, axis=1)
        max_dd = np.max(roll_max - equity, axis=1)
        profit_over_dd = np.asarray(
            [_safe_profit_over_dd(p, d) for p, d in zip(total_profit, max_dd)],
            dtype=np.float64,
        )

    if not np.all(valid_portfolio):
        total_profit = np.where(valid_portfolio, total_profit, np.nan)
        max_dd = np.where(valid_portfolio, max_dd, np.nan)
        profit_over_dd = np.where(valid_portfolio, profit_over_dd, np.nan)

    # Trades-derived metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_r = np.where(total_trades > 0, sum_r / total_trades, np.nan)
        winrate = np.where(
            total_trades > 0, (100.0 * total_wins / total_trades), np.nan
        )
        winrate_norm = winrate / 100.0

    if not np.all(valid_portfolio):
        avg_r = np.where(valid_portfolio, avg_r, np.nan)
        winrate = np.where(valid_portfolio, winrate, np.nan)
        winrate_norm = np.where(valid_portfolio, winrate_norm, np.nan)

    # Stability score (monthly WMAPE) on the daily equity grid
    stability_score_monthly = np.full((b,), np.nan, dtype=np.float64)
    mep = state.month_end_positions
    if t > 0 and mep.size >= 3 and state.month_days_for_profits.size == (mep.size - 1):
        monthly_equity = equity[:, mep]
        monthly_profits = monthly_equity[:, 1:] - monthly_equity[:, :-1]
        weights = state.month_days_for_profits.astype(np.float64, copy=False)
        d_total = float(np.sum(weights))
        if d_total > 0.0:
            p_total = np.sum(monthly_profits, axis=1)
            mu = p_total / d_total
            s_min = np.maximum(100.0, 0.02 * np.abs(p_total))
            expected = mu[:, None] * weights[None, :]
            denom = np.maximum(np.abs(expected), s_min[:, None])
            r = np.abs(monthly_profits - expected) / denom
            w = (weights / d_total)[None, :]
            wmape = np.sum(r * w, axis=1)
            stability_score_monthly = np.where(
                np.isfinite(wmape), 1.0 / (1.0 + wmape), np.nan
            )

    # Composite + final score (same formula as compute_additional_scores/compute_final_score)
    # but with trade-count adjusted metrics

    # Calculate n_years from daily_index for total metrics
    n_years = 1.0  # fallback
    if len(state.daily_index_utc) > 0:
        years_in_data = state.daily_index_utc.year.unique()
        n_years = float(len(years_in_data)) if len(years_in_data) > 0 else 1.0

    # Portfolio breadth scaling (groups_count)
    n_categories = float(len(state.group_ids)) if hasattr(state, "group_ids") else 1.0
    if not np.isfinite(n_categories) or n_categories <= 0.0:
        n_categories = 1.0

    # Apply adjustments (total metrics: n_years based on data)
    avg_r_adjusted = shrinkage_adjusted(
        average_r=avg_r,
        n_trades=total_trades,
        n_years=n_years,
        n_categories=n_categories,
    )

    # Winrate: Wilson Score Lower Bound adjustment (conservative CI)
    winrate_adjusted = wilson_score_lower_bound(
        winrate=winrate_norm, n_trades=total_trades
    )

    profit_over_dd_adjusted = risk_adjusted(
        profit_over_drawdown=np.where(profit_over_dd >= 0.0, profit_over_dd, 0.0),
        n_trades=total_trades,
        n_years=n_years,
        n_categories=n_categories,
    )

    # Normalize adjusted PoD for score
    pod_term = profit_over_dd_adjusted / (1.0 + profit_over_dd_adjusted)
    pod_term = np.where(np.isfinite(pod_term), pod_term, 0.0)

    comp_score = np.where(
        np.isfinite(avg_r_adjusted)
        & np.isfinite(winrate_adjusted)
        & np.isfinite(pod_term),
        (avg_r_adjusted + winrate_adjusted + pod_term) / 3.0,
        np.nan,
    )

    # DEBUG: Log first portfolio to validate adjustments
    if len(selections) > 0 and np.isfinite(comp_score[0]):
        print(
            f"  [MC] Portfolio 0/Batch: "
            f"wr_raw={winrate[0]:.1f}% → adj={winrate_adjusted[0]*100:.1f}%, "
            f"avg_r_raw={avg_r[0]:.3f} → adj={avg_r_adjusted[0]:.3f}, "
            f"pod_raw={profit_over_dd[0]:.3f} → adj={profit_over_dd_adjusted[0]:.3f}, "
            f"comp_score={comp_score[0]:.4f}"
        )
    comp0 = np.nan_to_num(comp_score, nan=0.0)
    stab0 = np.nan_to_num(stability_score_monthly, nan=0.0)
    rob0 = np.nan_to_num(robustness_mean, nan=0.0)
    final_score = 0.5 * comp0 + 0.25 * stab0 + 0.25 * rob0

    # Hard exclude invalid portfolios
    if not np.all(valid_portfolio):
        final_score = np.where(valid_portfolio, final_score, -np.inf)

    return {
        "selections": selections,
        "final_score": final_score.astype(np.float64),
        "comp_score": comp_score.astype(np.float64),
        "stability_score_monthly": stability_score_monthly.astype(np.float64),
        "robustness_mean": robustness_mean.astype(np.float64),
        "total_profit": total_profit.astype(np.float64),
        "total_max_dd": max_dd.astype(np.float64),
        "total_profit_over_dd": profit_over_dd.astype(np.float64),
        "avg_r": avg_r.astype(np.float64),
        "winrate": winrate.astype(np.float64),
        "total_trades": total_trades.astype(np.int64),
    }


# =============================
# Search utilities (Sampling, Beam, GA)
# =============================


def portfolio_sample_generator(
    index: Dict[str, List[Dict[str, Any]]],
    n_samples: int,
    rng: np.random.Generator,
    unique_only: bool = True,
) -> None:  # type: ignore
    """QW4 OPTIMIERT: Generator-Pattern statt Liste — O(1) Speicher für Samples!

    Streamt zufällige Portfolio-Samples statt sie alle im RAM zu speichern.
    Mit 10.000 Samples: -50 MB RAM (10 MB Liste → 0.1 MB aktuel).

    NEU (unique_only=True):
    - Trackt bereits generierte Kombinationen (als Tuple von combo_pair_ids)
    - Garantiert, dass jeder Sample **eindeutig** ist
    - Eliminiert 36% Duplikate in der Monte Carlo Evaluation
    - Nur minimal slower: O(n_samples) statt O(n_samples) mit lokalem Set

    Args:
        index: Dict mapping group_id -> List[combo_pair_entry]
        n_samples: Anzahl zufälliger Samples
        rng: NumPy Generator für Reproduzierbarkeit
        unique_only: Wenn True, generiere nur eindeutige Kombinationen (empfohlen)

    Yields:
        Dict[str, Dict[str, Any]]: Portfolio-Selection pro Iteration
    """
    if not unique_only:
        # Legacy Mode: Keine Deduplication bei Generierung
        for _ in range(n_samples):
            yield _random_selection(index, rng)
    else:
        # NEU: Unique-Only Mode mit lokalem Tracking
        seen_combinations: Set[Tuple[Tuple[str, str], ...]] = set()
        attempts = 0
        max_attempts = n_samples * 10  # Limit um Infinite Loop zu vermeiden
        yielded = 0

        while yielded < n_samples and attempts < max_attempts:
            selection = _random_selection(index, rng)
            # Kombinations-Signatur: (group_id, combo_pair_id) in deterministischer Reihenfolge.
            # Wichtig: combo_pair_id ist nicht zwingend global eindeutig; daher group_id inkludieren.
            combo_sig = tuple(
                (gid, str(selection[gid].get("combo_pair_id", "")))
                for gid in sorted(selection.keys())
                if gid in selection and selection[gid] is not None
            )

            if combo_sig not in seen_combinations:
                seen_combinations.add(combo_sig)
                yield selection
                yielded += 1

            attempts += 1

        # Warnung wenn wir nicht genug eindeutige Kombinationen finden konnten
        if yielded < n_samples:
            print(
                f"[PortfolioSampler] ⚠️ Warnung: Nur {yielded} eindeutige Kombinationen gefunden (angefordert: {n_samples})"
            )
            print(
                f"[PortfolioSampler]   Grund: Zu viele Duplikate oder zu wenige Kandidaten pro Gruppe"
            )


def _random_selection(
    index: Dict[str, List[Dict[str, Any]]], rng: np.random.Generator
) -> Dict[str, Dict[str, Any]]:
    sel: Dict[str, Dict[str, Any]] = {}
    for gid, lst in index.items():
        if not lst:
            continue
        i = int(rng.integers(0, len(lst)))
        sel[gid] = lst[i]
    return sel


def _evaluate_portfolio_batch(
    args: Tuple[List[Dict[str, Dict[str, Any]]], bytes],
) -> List[Optional[Dict[str, Any]]]:
    """OPTIMIERT: Evaluiert einen Batch von Portfolios (reduziert Process-Communication Overhead).

    Args:
        args: Tuple von (batch_selections, pickled_matrix_bytes)

    Returns:
        Liste von Result-Dicts (oder None bei Fehler)
    """
    batch_selections, matrix_bytes = args
    # Matrix einmal entpacken (statt pro Portfolio)
    # Pickle von vertrauenswürdigen internen Daten (eigene Backtest-Matrix)
    matrix = pickle.loads(matrix_bytes)  # nosec B301

    results: List[Optional[Dict[str, Any]]] = []
    for selection in batch_selections:
        try:
            res = aggregate_final_combo(selection, write_files=False)
            # Build minimal dataframe for this single portfolio
            df = build_final_combos_dataframe([res], [selection], matrix)
            if df.empty:
                results.append(None)
                continue

            # Compute metrics
            df_metrics = compute_global_metrics(
                df.rename(columns={"final_combo_pair_id": "combo_pair_id"})
            )
            df_metrics = df_metrics.rename(
                columns={"combo_pair_id": "final_combo_pair_id"}
            )
            df_metrics = compute_additional_scores(df_metrics)
            df_metrics = compute_final_score(df_metrics)

            # Return as dict
            results.append(df_metrics.iloc[0].to_dict())
        except Exception as e:
            # Silently skip (zu viel Output bei vielen Tasks)
            results.append(None)

    return results


def _evaluate_portfolio_batch_arrow(
    args: Tuple[List[Dict[str, Dict[str, Any]]], str],
) -> List[Optional[Dict[str, Any]]]:
    """QW3 OPTIMIERT: Arrow-basierte Batch-Evaluation mit Memory-Mapped Parquet.

    Nutzt Apache Arrow statt Pickle für -40% Serialisierungs-Overhead.
    Matrix wird via Memory-Mapped Parquet gelesen (Zero-Copy Deserialisierung).

    Args:
        args: Tuple von (batch_selections, parquet_path)

    Returns:
        Liste von Result-Dicts (oder None bei Fehler)
    """
    import pyarrow.parquet as pq

    batch_selections, parquet_path = args

    # QW3: Zero-copy read via Arrow Memory Mapping (nicht pickle.loads!)
    try:
        table = pq.read_table(parquet_path, memory_map=True)
        matrix = table.to_pandas()  # Lazy view, keine vollständige Kopie
    except Exception as e:
        # Fallback bei Arrow-Fehler
        print(f"[Arrow] Fehler beim Lesen von {parquet_path}: {e}")
        return [None] * len(batch_selections)

    results: List[Optional[Dict[str, Any]]] = []
    for selection in batch_selections:
        try:
            res = aggregate_final_combo(selection, write_files=False)
            df = build_final_combos_dataframe([res], [selection], matrix)
            if df.empty:
                results.append(None)
                continue

            # Compute metrics
            df_metrics = compute_global_metrics(
                df.rename(columns={"final_combo_pair_id": "combo_pair_id"})
            )
            df_metrics = df_metrics.rename(
                columns={"combo_pair_id": "final_combo_pair_id"}
            )
            df_metrics = compute_additional_scores(df_metrics)
            df_metrics = compute_final_score(df_metrics)

            results.append(df_metrics.iloc[0].to_dict())
        except Exception:
            results.append(None)

    return results


def _evaluate_single_portfolio(
    args: Tuple[Dict[str, Dict[str, Any]], pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    """Helper function for parallel evaluation of a single portfolio (Legacy-Kompatibilität).

    HINWEIS: Für bessere Performance nutze _evaluate_portfolio_batch stattdessen.

    Args:
        args: Tuple of (selection, matrix)

    Returns:
        Result dict or None if evaluation failed
    """
    selection, matrix = args
    try:
        res = aggregate_final_combo(selection, write_files=False)
        # Build minimal dataframe for this single portfolio
        df = build_final_combos_dataframe([res], [selection], matrix)
        if df.empty:
            return None

        # Compute metrics
        df_metrics = compute_global_metrics(
            df.rename(columns={"final_combo_pair_id": "combo_pair_id"})
        )
        df_metrics = df_metrics.rename(columns={"combo_pair_id": "final_combo_pair_id"})
        df_metrics = compute_additional_scores(df_metrics)
        df_metrics = compute_final_score(df_metrics)

        # Return as dict
        return df_metrics.iloc[0].to_dict()
    except Exception as e:
        # print(f"[Portfolio Eval Error] {e}")  # Zu viel Output
        return None


def monte_carlo_portfolio_search(
    index: Dict[str, List[Dict[str, Any]]],
    matrix: pd.DataFrame,
    *,
    eval_state: Optional[MonteCarloEvalState] = None,
    num_samples: int = 10000,
    n_jobs: Optional[int] = None,
    rng_seed: Optional[int] = None,
    dev_mode: bool = False,
    batch_size: int = 20,
    use_batch_mode: bool = True,
    use_arrow: bool = True,
) -> pd.DataFrame:
    """Monte Carlo Portfolio Search mit vollständiger Evaluation.

    OPTIMIERT (Dezember 2025):
    - Batch Processing: Reduziert Process-Communication Overhead
    - Pre-Serialisierung: Matrix wird einmal serialisiert, nicht pro Task
    - ProcessPoolExecutor: Bessere macOS-Kompatibilität mit 'spawn'
    - Shared Memory Cache: Equity/Trades werden über Prozesse geteilt
    - QW3 (Phase 2): Arrow/Parquet statt Pickle für -40% Serialisierungs-Overhead

    Args:
        index: Index mapping group_id -> List[combo_pair_entry]
        matrix: Combined matrix DataFrame
        num_samples: Anzahl zufälliger Portfolios zu testen (default: 10000)
        n_jobs: Anzahl paralleler Prozesse (None = cpu_count())
        rng_seed: Random seed für Reproduzierbarkeit (None = zufällig)
        dev_mode: Wenn True, nutze den Seed für reproducible Entwicklung
        batch_size: Anzahl Portfolios pro Batch (default: 20, optimal für Multi-Core)
        use_batch_mode: Wenn True, nutze Batch-Processing (empfohlen)
        use_arrow: Wenn True, nutze Arrow/Parquet statt Pickle (QW3, empfohlen)

    Returns:
        DataFrame mit evaluierten Portfolios, sortiert nach final_score
    """
    # NOTE: use_batch_mode/use_arrow werden aus Kompatibilitätsgründen beibehalten,
    # die aktuelle Implementierung nutzt jedoch einen vorberechneten NumPy-Evaluator,
    # um IPC/Serialisierung und per-Portfolio Pandas-Overhead zu eliminieren.

    # Seed-Logik: Nur wenn dev_mode aktiv
    if dev_mode and rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
        print(
            f"[Monte Carlo] DEVELOPMENT MODE: Nutze Seed {rng_seed} für reproducible Ergebnisse"
        )
    else:
        rng = np.random.default_rng()  # Zufälliger Seed für Produktion
        if rng_seed is not None:
            print(
                f"[Monte Carlo] PRODUCTION MODE: Ignoriere Seed {rng_seed} (nur für Entwicklung)"
            )

    n_jobs = n_jobs or cpu_count()

    # Prepare fast evaluator state once (dominant speedup for Step 5)
    state = eval_state or prepare_monte_carlo_eval_state(index)
    if len(state.group_ids) == 0 or len(state.daily_index_utc) == 0:
        print("[Monte Carlo] Keine gültige gemeinsame Equity-Zeitleiste. Abbruch.")
        return pd.DataFrame()

    group_sizes = [len(index.get(gid, [])) for gid in state.group_ids]
    if any(sz <= 0 for sz in group_sizes):
        print("[Monte Carlo] Mindestens eine Gruppe hat keine Kandidaten. Abbruch.")
        return pd.DataFrame()

    print(
        f"[Monte Carlo] Prepared eval state: groups={len(state.group_ids)}, "
        f"candidates_total={sum(group_sizes)}, days={len(state.daily_index_utc)}"
    )

    # Auto-tune batch size to keep per-worker peak memory bounded (rough heuristic).
    # pnl(float32) + equity(float64) ≈ 12 bytes per timepoint per portfolio (+ overhead).
    t = max(1, len(state.daily_index_utc))
    bytes_per_portfolio = 12 * t
    target_bytes = 128_000_000  # ~128 MB working set per worker
    max_batch = max(64, int(target_bytes / bytes_per_portfolio))
    if int(batch_size) > max_batch:
        print(
            f"[Monte Carlo] Batch-Size Auto-Tune: {batch_size} → {max_batch} (days={t})"
        )
        batch_size = max_batch

    print(
        f"[Monte Carlo] Starte parallele Batch-Evaluation (NumPy, batch_size={batch_size}) auf {n_jobs} Kernen..."
    )

    # Streaming unique sampling over integer-coded portfolios
    unique_only = True
    seen: Set[Tuple[int, ...]] = set()
    yielded = 0
    attempts = 0
    max_attempts = int(num_samples) * 10

    def next_batch() -> Optional[np.ndarray]:
        nonlocal yielded, attempts
        remaining = int(num_samples) - yielded
        if remaining <= 0:
            return None
        b = min(int(batch_size), remaining)
        out = np.empty((b, len(group_sizes)), dtype=np.int32)
        filled = 0
        while filled < b and attempts < max_attempts:
            row = [
                int(rng.integers(0, group_sizes[j])) for j in range(len(group_sizes))
            ]
            sig = tuple(row) if unique_only else None
            if (not unique_only) or (sig not in seen):
                if unique_only:
                    seen.add(sig)  # type: ignore[arg-type]
                out[filled, :] = row
                filled += 1
                yielded += 1
            attempts += 1
        if filled == 0:
            return None
        return out[:filled, :]

    # Parallel evaluation (true streaming; bounded in-flight futures)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    results_chunks: List[pd.DataFrame] = []

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_monte_carlo_worker,
        initargs=(state,),
    ) as executor:
        in_flight = set()
        pbar = tqdm(
            total=int(num_samples), desc="Portfolios evaluieren", unit="portfolio"
        )

        def submit_one(batch: np.ndarray) -> None:
            in_flight.add(executor.submit(_evaluate_indices_batch_fast, batch))

        try:
            while True:
                while len(in_flight) < max(1, 2 * int(n_jobs)):
                    batch = next_batch()
                    if batch is None:
                        break
                    submit_one(batch)
                if not in_flight:
                    break
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    payload = fut.result()
                    sel = payload.pop("selections")
                    n = int(sel.shape[0])
                    pbar.update(n)
                    # Build DataFrame chunk without heavy objects
                    df_chunk = pd.DataFrame(payload)
                    df_chunk["_selections"] = list(sel.astype(np.int32))
                    results_chunks.append(df_chunk)
        finally:
            pbar.close()

    if not results_chunks:
        print("[Monte Carlo] Keine erfolgreichen Evaluationen. Abbruch.")
        return pd.DataFrame()

    df = pd.concat(results_chunks, ignore_index=True)

    # Decode selections -> IDs + mapping JSON (keep this in parent process)
    group_ids = list(state.group_ids)
    combo_ids = state.combo_pair_ids_by_group
    final_ids: List[str] = []
    mapping_jsons: List[str] = []
    groups_count = len(group_ids)

    for sel in df["_selections"].tolist():
        # sel is a 1D ndarray of indices
        mapping = {group_ids[i]: combo_ids[i][int(sel[i])] for i in range(groups_count)}
        mapping_jsons.append(json.dumps(mapping))
        # Deterministic hash id (same semantics as _final_combo_id_from_selection)
        parts = [f"{gid}={mapping[gid]}" for gid in sorted(mapping.keys())]
        base = "__".join(parts)
        import hashlib

        # SHA1 nur für nicht-kryptografische ID-Generierung verwendet
        digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[
            :16
        ]  # nosec B324
        final_ids.append(f"final_{digest}")

    df.insert(0, "final_combo_pair_id", final_ids)
    df["groups_count"] = groups_count
    df["groups_mapping_json"] = mapping_jsons
    df = df.drop(columns=["_selections"])

    # Sort by final_score
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    print(f"[Monte Carlo] Erfolgreich evaluiert: {len(df)}/{num_samples} Portfolios")
    return df


def hydrate_portfolio_artifacts_for_categorical(
    df: pd.DataFrame,
    *,
    eval_state: MonteCarloEvalState,
    index: Dict[str, List[Dict[str, Any]]],
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Attach `_equity_internal` (pd.Series) and `_trades_internal` (list[dict]) to portfolios.

    This is intentionally executed AFTER Monte-Carlo ranking/deduplication to avoid
    serializing huge objects during the search phase.
    """
    if df is None or df.empty:
        return df
    if len(eval_state.group_ids) == 0 or len(eval_state.daily_index_utc) == 0:
        return df

    group_ids = list(eval_state.group_ids)
    g = len(group_ids)
    t = len(eval_state.daily_index_utc)

    # Fast lookup: group_id -> combo_pair_id -> selection-index
    cpid_to_idx: List[Dict[str, int]] = []
    for gi in range(g):
        mapping: Dict[str, int] = {}
        for j, cpid in enumerate(eval_state.combo_pair_ids_by_group[gi]):
            mapping[str(cpid)] = int(j)
        cpid_to_idx.append(mapping)

    # Parse mapping JSON → selection indices
    selections: List[np.ndarray] = []
    for mapping_json in df["groups_mapping_json"].tolist():
        mapping = json.loads(mapping_json)
        row = np.empty((g,), dtype=np.int32)
        for gi, gid in enumerate(group_ids):
            cpid = str(mapping.get(gid, ""))
            row[gi] = int(cpid_to_idx[gi].get(cpid, 0))
        selections.append(row)

    # Hydrate equity in chunks (batchwise NumPy; then wrap rows as pd.Series)
    equities_out: List[Optional[pd.Series]] = [None] * len(df)
    trades_out: List[List[Dict[str, Any]]] = [[] for _ in range(len(df))]

    chunk_size = max(1, int(chunk_size))
    for start in tqdm(
        range(0, len(df), chunk_size),
        desc="[Categorical] Hydrate equity/trades",
        unit="chunk",
    ):
        end = min(len(df), start + chunk_size)
        sel_chunk = np.stack(selections[start:end], axis=0)  # (B, G)
        b = sel_chunk.shape[0]

        pnl = np.zeros((b, t), dtype=np.float32)
        for gi in range(g):
            pnl += eval_state.equity_daily_pnl_by_group[gi][sel_chunk[:, gi]]
        equity = eval_state.start_equity + np.cumsum(pnl.astype(np.float64), axis=1)

        for i in range(b):
            equities_out[start + i] = pd.Series(
                equity[i, :], index=eval_state.daily_index_utc
            )

    # Trades hydration (list concat; keep as list[dict] to avoid DF overhead unless needed later)
    # Build fast path maps to candidate entries
    entry_by_group_and_cpid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for gid in group_ids:
        entry_by_group_and_cpid[gid] = {
            str(e.get("combo_pair_id", "")): e for e in index.get(gid, [])
        }

    for i, mapping_json in enumerate(df["groups_mapping_json"].tolist()):
        mapping = json.loads(mapping_json)
        combined: List[Dict[str, Any]] = []
        for gid in group_ids:
            cpid = str(mapping.get(gid, ""))
            entry = entry_by_group_and_cpid.get(gid, {}).get(cpid)
            if not entry:
                continue
            tr_path = str(entry.get("trades_path", ""))
            combined.extend(get_trades_cached(tr_path))
        trades_out[i] = combined

    df = df.copy()
    df["_equity_internal"] = equities_out
    df["_trades_internal"] = trades_out

    # Ensure duration_days is available for downstream adjusted metrics.
    # Without it, `_add_total_adust_metrics_to_portfolios()` falls back to n_years=1.0,
    # which makes the CSV outputs look like they only scale with groups_count.
    if "duration_days" not in df.columns:
        duration_days = np.nan
        if len(eval_state.daily_index_utc) >= 2:
            try:
                duration_days = float(
                    (
                        eval_state.daily_index_utc[-1] - eval_state.daily_index_utc[0]
                    ).total_seconds()
                    / (24.0 * 3600.0)
                )
            except Exception:
                duration_days = np.nan
        df["duration_days"] = duration_days
    return df


def aggregate_final_combo(
    selection: Dict[str, Dict[str, Any]], write_files: bool = True
) -> Dict[str, Any]:
    """
    Aggregiere Equity und Trades über alle gewählten combo_pair_id (eine pro Gruppe).
    Equity-Kombination: Summe über alle Serien (Union der Zeitstempel, ffill),
    Trades: concat aller Trades nach entry_time sortiert.

    Args:
        selection: Mapping group_id -> entry dict mit equity_path, trades_path
        write_files: Wenn False, werden Dateien übersprungen (nur in-memory Berechnung)

    Returns: {"final_id": str, "equity": Path/Series, "trades": Path/DataFrame}
    """
    final_id = _final_combo_id_from_selection(selection)
    target_dir = COMBINED_MATRIX_DIR / "final_combos" / final_id

    # ⚡ ULTRA-OPTIMIERT: Batch-Aggregation mit numpy
    series_list: List[pd.Series] = []

    for entry in selection.values():
        eq_path = str(entry["equity_path"])
        # Cache-Lookup (thread-safe helper will load & cache if missing)
        s = get_equity_cached(eq_path)

        if s is not None and not s.empty:
            series_list.append(s)

    sum_series = None
    if series_list:
        if len(series_list) == 1:
            sum_series = series_list[0]
        else:
            # Batch-concat + ffill + sum
            combined = pd.concat(series_list, axis=1).sort_index()
            combined_filled = combined.ffill()
            sum_series = combined_filled.sum(axis=1, skipna=True)

    # Equity-Start auf 100_000 normieren, ohne Profite/Verluste zu verändern
    # (reine vertikale Verschiebung der Kurve).
    if sum_series is not None:
        first_valid = sum_series.dropna().head(1)
        if not first_valid.empty:
            start_val = float(first_valid.iloc[0])
            equity_shift = start_val - 100_000.0
            sum_series = sum_series - equity_shift

    if write_files and sum_series is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        equity_path = target_dir / "equity.csv"
        df_out = sum_series.reset_index()
        df_out.columns = ["timestamp", "equity"]
        df_out.to_csv(equity_path, index=False)
    else:
        equity_path = sum_series  # Return series directly

    # Trades sammeln (OPTIMIERT: Cache-Lookup)
    all_trades: List[Dict[str, Any]] = []
    for entry in selection.values():
        tr_path = str(entry["trades_path"])
        # Thread-safe trades loader
        trades = get_trades_cached(tr_path)
        if trades:
            all_trades.extend(trades)

    if write_files and all_trades:
        target_dir.mkdir(parents=True, exist_ok=True)
        trades_path = target_dir / "trades.json"
        df = pd.DataFrame(all_trades)
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(
                df["entry_time"], utc=True, errors="coerce"
            )
            df = df.sort_values("entry_time")
        # format timestamps
        records = df.to_dict("records")
        for rec in records:
            for key in ("entry_time", "exit_time"):
                if key in rec and pd.notna(rec[key]):
                    try:
                        ts = pd.to_datetime(rec[key], utc=True)
                        rec[key] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pass
        with trades_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        trades_path = pd.DataFrame(all_trades) if all_trades else None

    return {"final_id": final_id, "equity": equity_path, "trades": trades_path}


def _process_batch_parallel(
    batch_selections: List[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Path]]:
    """Worker-Funktion für parallele Batch-Verarbeitung."""
    results = []
    for sel in batch_selections:
        res = aggregate_final_combo(sel)
        results.append(res)
    return results


def build_final_combos_dataframe(
    final_results: List[Dict[str, Path]],
    selections: List[Dict[str, Dict[str, Any]]],
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeuge DataFrame für final_combo_pair_id mit Pfaden und Metadaten.
    Extrahiert Robustness-Werte aus der Matrix für die beteiligten combo_pair_ids.
    """
    rows: List[Dict[str, Any]] = []
    for res, sel in zip(final_results, selections):
        # Speichere equity/trades nur intern für Metrik-Berechnung
        equity_obj = res["equity"]
        trades_obj = res["trades"]

        row: Dict[str, Any] = {
            "final_combo_pair_id": res["final_id"],
            "_equity_internal": equity_obj,  # Intern, nicht in CSV
            "_trades_internal": trades_obj,  # Intern, nicht in CSV
        }
        # Meta: Anzahl Gruppen, Liste von group_id -> combo_pair_id
        mapping = {gid: entry["combo_pair_id"] for gid, entry in sel.items()}
        row["groups_count"] = len(mapping)
        row["groups_mapping_json"] = json.dumps(mapping)

        # Extrahiere Robustness-Werte aus der Matrix für alle beteiligten combo_pair_ids
        combo_pair_ids = list(mapping.values())
        robustness_values = []

        for cpid in combo_pair_ids:
            # Finde die Zeile in der Matrix für diese combo_pair_id
            matrix_row = matrix[matrix["combo_pair_id"] == cpid]
            if not matrix_row.empty:
                matrix_row = matrix_row.iloc[0]

                # Extrahiere Robustness nach der gleichen Prioritätslogik
                rob_val = np.nan

                # Priorität 1: robustness_1_mean
                if "robustness_1_mean" in matrix_row.index:
                    rob_val = pd.to_numeric(
                        matrix_row["robustness_1_mean"], errors="coerce"
                    )

                # Priorität 2: Mittelwert robustness_score_1 aus Leg A und B
                if pd.isna(rob_val):
                    rob_val = get_pair_robustness_mean_from_top50(matrix_row)

                if pd.notna(rob_val):
                    robustness_values.append(rob_val)

        # Berechne Durchschnitt der Robustness-Werte
        if robustness_values:
            row["robustness_mean"] = float(np.mean(robustness_values))
        else:
            row["robustness_mean"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def compute_global_metrics(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet globale Kennzahlen pro Matrixzeile basierend auf den
    aggregierten equity.csv und trades.json.

    OPTIMIERT: Batch-Loads und vectorized operations wo möglich.

    Kennzahlen:
    - total_profit: Gesamtprofit über alle Jahre
    - total_max_dd: Maximaler Drawdown über alle Jahre
    - avg_r: Durchschnittliches R über alle Trades
    - winrate: Winrate über alle Trades
    - total_trades: Anzahl Trades
    - identical_trades_absolut: Anzahl Trades an gemeinsamen Entry-Zeitpunkten
    - identical_trades_entry: Anzahl gemeinsamer Entry-Zeitpunkte (mind. 2 Trades)
    - identical_trades_absolut_percentage: Anteil dieser gemeinsamen Entry-Zeitpunkte

    NEU: Equity-Shape Features für HF-Stil Clustering:
    - equity_returns_skew: Schiefe der monatlichen Returns
    - equity_returns_kurtosis: Kurtosis der monatlichen Returns
    - equity_returns_autocorr: Autokorrelation (Lag-1) der monatlichen Returns
    - equity_returns_volatility: Standardabweichung der monatlichen Returns

    Returns:
        Matrix mit zusätzlichen Spalten
    """
    if matrix.empty:
        return matrix

    # QW1 OPTIMIERT: Keine .copy() — modifiziere in-place statt Kopie zu erstellen

    # Initialisierung der neuen Spalten
    matrix["total_profit"] = np.nan
    matrix["total_profit_over_dd"] = np.nan
    matrix["total_max_dd"] = np.nan
    matrix["avg_r"] = np.nan
    matrix["winrate"] = np.nan
    # NEU: Sharpe/Sortino (pro Trade, auf R-Multiples – analog zu metrics.calculate_metrics)
    matrix["sharpe_trade"] = np.nan
    matrix["sortino_trade"] = np.nan
    matrix["total_trades"] = 0
    matrix["identical_trades_entry"] = 0
    matrix["identical_trades_absolut"] = 0
    matrix["identical_trades_absolut_percentage"] = np.nan

    # NEU: Equity-Shape Features für Clustering
    matrix["equity_returns_skew"] = np.nan
    matrix["equity_returns_kurtosis"] = np.nan
    matrix["equity_returns_autocorr"] = np.nan
    matrix["equity_returns_volatility"] = np.nan

    # NEU: Equity-basierte Sharpe/Sortino (HF-Standard, monatliche Returns)
    matrix["sharpe_equity"] = np.nan
    matrix["sortino_equity"] = np.nan

    # Hilfsfunktion: Sharpe- und Sortino-Ratio aus R-Multiples pro Trade
    def _sharpe_sortino_from_r(
        r_list, *, risk_free: float = 0.0, mar: float = 0.0
    ) -> Tuple[float, float]:
        """
        Berechnet Sharpe- und Sortino-Ratio auf *pro-Trade*-Basis aus R-Multiples.
        - risk_free: risikofreier Return pro Trade (auf R-Basis, i.d.R. 0)
        - mar: Minimum Acceptable Return für Sortino (auf R-Basis, i.d.R. 0)
        Hinweis: Keine Annualisierung – Kennzahlen gelten pro Trade.
        """
        # Filter: None/NaN/Inf entfernen
        arr = np.asarray([x for x in r_list if x is not None], dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            # Zu wenig Daten für sinnvolle Streuung
            return 0.0, 0.0

        # Excess-Return relativ zu risk-free (pro Trade)
        excess = arr - risk_free
        mu = excess.mean()

        # Sharpe: Stichproben-Std (ddof=1). Bei sigma=0 -> 0.0
        sigma = excess.std(ddof=1)
        sharpe = float(mu / sigma) if sigma > 0.0 else 0.0

        # Sortino: Semideviation relativ zu MAR über die gesamte Stichprobe
        downside_diff = np.minimum(excess - mar, 0.0)
        semi_dev = float(np.sqrt(np.mean(downside_diff**2)))
        sortino = float(mu / semi_dev) if semi_dev > 0.0 else 0.0

        return sharpe, sortino

    for idx, row in matrix.iterrows():
        # Verwende interne Spalten falls vorhanden (In-Memory-Modus)
        equity_path = row.get("_equity_internal", row.get("equity_path", ""))
        trades_path = row.get("_trades_internal", row.get("trades_path", ""))

        # ⚡ OPTIMIERT: Equity kann entweder Path oder Series sein
        eq_series = None
        if isinstance(equity_path, pd.Series):
            eq_series = equity_path
        elif equity_path and not isinstance(equity_path, pd.Series):
            try:
                equity_path_obj = Path(str(equity_path))
                if equity_path_obj.exists():
                    eq_df = pd.read_csv(equity_path_obj)
                    ts_col, eq_col = _detect_equity_columns(eq_df)

                    ts = pd.to_datetime(eq_df[ts_col], utc=True, errors="coerce")
                    eq_vals = pd.to_numeric(eq_df[eq_col], errors="coerce")
                    mask = ts.notna() & eq_vals.notna()

                    if mask.any():
                        eq_series = pd.Series(eq_vals[mask].values, index=ts[mask])
                        eq_series = eq_series.sort_index()
            except Exception as e:
                print(f"[Metrics] Fehler beim Lesen von {equity_path_obj}: {e}")

        if eq_series is not None and not eq_series.empty:
            # Total Profit: letzter - erster Wert
            if len(eq_series) >= 2:
                total_profit = float(eq_series.iloc[-1] - eq_series.iloc[0])
                matrix.at[idx, "total_profit"] = total_profit

            # Max Drawdown
            roll_max = eq_series.cummax()
            dd = roll_max - eq_series
            max_dd = float(dd.max())
            matrix.at[idx, "total_max_dd"] = max_dd

            # NEU: Equity-Shape Features aus monatlichen Returns
            try:
                # Monatliche Resampling
                monthly = eq_series.resample("ME").last()
                # Entferne leere Monate (falls vorhanden) - verhindert NaNs in base
                monthly = monthly.dropna()
                if monthly.empty:
                    # Keine gültigen monatlichen Werte
                    raise ValueError(
                        "keine gültigen monatlichen Equity-Werte nach Resample"
                    )

                monthly_profits = monthly.diff().dropna()

                # Bestimme Basiswert (Start-Equity). Falls dieser 0/NaN/inf ist, weiche auf einen stabilen Fallback aus
                base = float(monthly.iloc[0])
                if not np.isfinite(base) or abs(base) < 1e-12:
                    # Sicherheits-Fallback: konsistente Normalisierung (z.B. 100_000) - verhindert Division durch Null
                    # Hinweis: Falls du statisches Risiko verwendest, stelle sicher, dass alle Serien vorab
                    # dieselbe Start-Equity haben. Dieser Fallback ist nur eine robuste Notlösung.
                    base = 100_000.0

                if len(monthly) >= 3:  # Mindestens 3 Monate für sinnvolle Statistik
                    # Monatliche Returns relativ zur festen Start-Equity (fixed-base, nicht periodisch skaliert)
                    monthly_returns = (
                        monthly_profits / base
                    )  # Alternative -> monthly.pct_change(fill_method=None).dropna()

                    if len(monthly_returns) >= 2:
                        # Skewness (Schiefe)
                        skew_val = (
                            float(monthly_returns.skew())
                            if len(monthly_returns) >= 3
                            else np.nan
                        )
                        matrix.at[idx, "equity_returns_skew"] = skew_val

                        # Kurtosis (Wölbung)
                        kurt_val = (
                            float(monthly_returns.kurtosis())
                            if len(monthly_returns) >= 4
                            else np.nan
                        )
                        matrix.at[idx, "equity_returns_kurtosis"] = kurt_val

                        # Autokorrelation (Lag-1)
                        if len(monthly_returns) >= 3:
                            autocorr_val = float(monthly_returns.autocorr(lag=1))
                            if np.isfinite(autocorr_val):
                                matrix.at[idx, "equity_returns_autocorr"] = autocorr_val

                        # Volatilität (Standardabweichung)
                        vol_val = float(monthly_returns.std())
                        if np.isfinite(vol_val):
                            matrix.at[idx, "equity_returns_volatility"] = vol_val

                        # HF-STANDARD: Equity-basierte Sharpe/Sortino auf monatlichen Returns
                        # Sharpe Ratio (annualisiert: √12)
                        mean_return = float(monthly_returns.mean())
                        std_return = float(monthly_returns.std())
                        if std_return > 0.0:
                            sharpe_equity = (mean_return / std_return) * np.sqrt(12)
                            matrix.at[idx, "sharpe_equity"] = sharpe_equity

                        # Sortino Ratio (annualisiert: √12, nur negative Returns)
                        downside_returns = monthly_returns[monthly_returns < 0.0]
                        if len(downside_returns) > 0:
                            downside_std = float(downside_returns.std())
                            if downside_std > 0.0:
                                sortino_equity = (mean_return / downside_std) * np.sqrt(
                                    12
                                )
                                matrix.at[idx, "sortino_equity"] = sortino_equity
                        elif mean_return > 0.0:
                            # Sonderfall: Keine negativen Returns, aber positive mean
                            # Sortino ist technisch unendlich, setze auf sehr hohen Wert
                            matrix.at[idx, "sortino_equity"] = 999.99
            except Exception as e:
                print(
                    f"[Metrics] Warnung: Fehler bei Equity-Shape Features für idx {idx}: {e}"
                )

        # ⚡ OPTIMIERT: Trades kann entweder Path oder DataFrame sein
        trades_df = None
        if isinstance(trades_path, pd.DataFrame):
            trades_df = trades_path
        elif trades_path and not isinstance(trades_path, pd.DataFrame):
            try:
                trades_path_obj = Path(str(trades_path))
                if trades_path_obj.exists():
                    with trades_path_obj.open("r", encoding="utf-8") as f:
                        trades = json.load(f)

                    if isinstance(trades, list) and trades:
                        trades_df = pd.DataFrame(trades)
            except Exception as e:
                print(f"[Metrics] Fehler beim Lesen von {trades_path_obj}: {e}")

        if trades_df is not None and not trades_df.empty:
            # Anzahl Trades
            matrix.at[idx, "total_trades"] = len(trades_df)

            # R-Multiple
            if "r_multiple" in trades_df.columns:
                r_vals = pd.to_numeric(trades_df["r_multiple"], errors="coerce")
                r_mean = float(r_vals.mean()) if not r_vals.isna().all() else np.nan
                matrix.at[idx, "avg_r"] = r_mean

                # Sharpe/Sortino pro Trade auf Basis der R-Multiples
                sharpe, sortino = _sharpe_sortino_from_r(
                    r_vals.tolist(), risk_free=0.0, mar=0.0
                )
                matrix.at[idx, "sharpe_trade"] = sharpe
                matrix.at[idx, "sortino_trade"] = sortino

            # Winrate
            if "result" in trades_df.columns:
                results = pd.to_numeric(trades_df["result"], errors="coerce")
                wins = (results > 0).sum()
                total = results.notna().sum()
                winrate = float(wins / total * 100) if total > 0 else np.nan
                matrix.at[idx, "winrate"] = winrate

            # Identische Trades (Entry-Zeitpunkte) über alle Legs einer Kombination:
            # 1) identical_trades_absolut: Anzahl aller Trades, die sich einen Entry-Timestamp
            #    mit mindestens einem weiteren Trade teilen
            # 2) identical_trades_entry : Anzahl dieser Entry-Timestamps (mindestens 2 Trades)
            # 3) identical_trades_absolut_percentage: Anteil dieser Entry-Timestamps an allen Trades
            if "entry_time" in trades_df.columns:
                et = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
                et = et.dropna()
                if not et.empty:
                    counts = et.value_counts()
                    # Jeder Timestamp mit count >= 2 gilt als "identischer Trade-Zeitpunkt"
                    shared_mask = counts >= 2
                    identical_ts = int(shared_mask.sum())
                    # Anzahl aller Trades, die an solchen Zeitpunkten stattfinden
                    identical_trades_total = (
                        int(counts[shared_mask].sum()) if identical_ts > 0 else 0
                    )
                    # total_ts = Anzahl aller Trades (mit gültigem entry_time)
                    total_ts = int(et.size)
                    matrix.at[idx, "identical_trades_absolut"] = identical_trades_total
                    matrix.at[idx, "identical_trades_entry"] = identical_ts
                    if total_ts > 0:
                        matrix.at[idx, "identical_trades_absolut_percentage"] = float(
                            identical_trades_total / float(total_ts)
                        )

        # Profit over DD (global) auf Basis der zuvor gesetzten total_profit/total_max_dd
        total_profit = pd.to_numeric(matrix.at[idx, "total_profit"], errors="coerce")
        total_max_dd = pd.to_numeric(matrix.at[idx, "total_max_dd"], errors="coerce")
        if pd.notna(total_profit) and pd.notna(total_max_dd):
            pod = _safe_profit_over_dd(total_profit, total_max_dd)
            matrix.at[idx, "total_profit_over_dd"] = float(pod)

    # print(f"[Metrics] Globale Metriken berechnet")

    return matrix


def compute_additional_scores(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet zusätzliche Scores:

    1. Stability Score auf monatlicher Basis (aus Equity)
    2. Robustness Score Mittelwert (aus existing data)
    3. Composite Score basierend auf jährlichen Metriken:
       comp_score = (avg_r + winrate/100 + (profit_over_dd / (1 + profit_over_dd))) / 3

    Returns:
        Matrix mit zusätzlichen Spalten
    """
    if matrix.empty:
        return matrix

    # QW1 OPTIMIERT: Keine .copy() — modifiziere in-place

    # Initialisierung (nur wenn Spalten noch nicht existieren, um vorhandene Werte zu erhalten)
    if "stability_score_monthly" not in matrix.columns:
        matrix["stability_score_monthly"] = np.nan
    if "robustness_mean" not in matrix.columns:
        matrix["robustness_mean"] = np.nan
    if "comp_score" not in matrix.columns:
        matrix["comp_score"] = np.nan
    if "_composite_rep_score" not in matrix.columns:
        matrix["_composite_rep_score"] = np.nan

    for idx, row in matrix.iterrows():
        # 1. Stability Score aus Equity (monatlich) - basierend auf __compute_yearly_stability
        # Verwende interne Spalten falls vorhanden (In-Memory-Modus)
        equity_path = row.get("_equity_internal", row.get("equity_path", ""))

        # ⚡ OPTIMIERT: Equity kann entweder Path oder Series sein
        eq_series = None
        if isinstance(equity_path, pd.Series):
            eq_series = equity_path
        elif equity_path and not isinstance(equity_path, pd.Series):
            try:
                equity_path_obj = Path(str(equity_path))
                if equity_path_obj.exists():
                    eq_df = pd.read_csv(equity_path_obj)
                    ts_col, eq_col = _detect_equity_columns(eq_df)

                    ts = pd.to_datetime(eq_df[ts_col], utc=True, errors="coerce")
                    eq_vals = pd.to_numeric(eq_df[eq_col], errors="coerce")
                    mask = ts.notna() & eq_vals.notna()

                    if mask.any():
                        eq_series = pd.Series(eq_vals[mask].values, index=ts[mask])
                        eq_series = eq_series.sort_index()
            except Exception as e:
                print(f"[Scores] Fehler beim Lesen von {equity_path_obj}: {e}")

        if eq_series is not None and not eq_series.empty:
            # Monatliche Profits berechnen (analog zu yearly)
            monthly = eq_series.resample("ME").last()
            if len(monthly) >= 2:
                # Profit pro Monat = Differenz zwischen Ende und Start des Monats
                monthly_profits = monthly.diff().dropna()

                if len(monthly_profits) >= 2:
                    # Dauer pro Monat in Tagen (approximativ)
                    # Wir verwenden die tatsächliche Anzahl Tage zwischen den Timestamps
                    monthly_dates = monthly_profits.index

                    profits = []
                    weights = []

                    for i, date in enumerate(monthly_dates):
                        profit = float(monthly_profits.iloc[i])

                        # Anzahl Tage in diesem Monat berechnen
                        year = date.year
                        month = date.month
                        if month == 12:
                            next_month_days = (
                                pd.Timestamp(year + 1, 1, 1)
                                - pd.Timestamp(year, month, 1)
                            ).days
                        else:
                            next_month_days = (
                                pd.Timestamp(year, month + 1, 1)
                                - pd.Timestamp(year, month, 1)
                            ).days

                        duration = float(next_month_days)

                        profits.append(profit)
                        weights.append(duration)

                    # WMAPE-basierte Stability-Berechnung (wie in __compute_yearly_stability)
                    P_total = float(sum(profits))
                    D_total = float(sum(weights))

                    if D_total > 0:
                        mu = P_total / D_total  # Erwarteter täglicher Profit
                        S_min = float(max(100.0, 0.02 * abs(P_total)))

                        # Compute WMAPE
                        wmape_acc = 0.0
                        for P_j, D_j in zip(profits, weights):
                            E_j = mu * D_j  # Erwarteter Profit für diesen Monat
                            denom = max(abs(E_j), S_min)
                            r_j = abs(P_j - E_j) / denom if denom > 0 else 0.0
                            w_j = D_j / D_total
                            wmape_acc += w_j * r_j

                        wmape = float(wmape_acc)
                        stability = (
                            float(1.0 / (1.0 + wmape)) if np.isfinite(wmape) else 0.0
                        )
                        matrix.at[idx, "stability_score_monthly"] = stability
                    else:
                        matrix.at[idx, "stability_score_monthly"] = 1.0

        # 2. Robustness Mean - bevorzugt aus robustness_1_mean (refined), sonst robustness_score_1
        # Behalte vorhandenen Wert bei (z.B. aus build_final_combos_dataframe), falls bereits gesetzt
        existing_robustness = (
            matrix.at[idx, "robustness_mean"]
            if "robustness_mean" in matrix.columns
            else np.nan
        )

        # Nur neu berechnen, wenn noch kein Wert vorhanden ist
        if pd.isna(existing_robustness):
            robustness_value = np.nan

            # Priorität 1: robustness_1_mean aus refined Top-10
            if "robustness_1_mean" in row.index:
                robustness_value = pd.to_numeric(
                    row["robustness_1_mean"], errors="coerce"
                )

            # Priorität 2: Mittelwert robustness_score_1 aus Leg A und B
            if pd.isna(robustness_value):
                robustness_value = get_pair_robustness_mean_from_top50(row)

            if pd.notna(robustness_value):
                matrix.at[idx, "robustness_mean"] = float(robustness_value)

        # 3. Composite Score (nutzt total_profit_over_dd aus compute_global_metrics)
        # mit trade-count adjusted Metriken
        avg_r_raw = pd.to_numeric(row.get("avg_r", np.nan), errors="coerce")
        winrate_raw = pd.to_numeric(row.get("winrate", np.nan), errors="coerce")
        pod_global_raw = pd.to_numeric(
            row.get("total_profit_over_dd", np.nan), errors="coerce"
        )
        n_trades = pd.to_numeric(row.get("total_trades", 0), errors="coerce")

        if (
            pd.notna(avg_r_raw)
            and pd.notna(winrate_raw)
            and pd.notna(pod_global_raw)
            and pd.notna(n_trades)
        ):
            # Calculate n_years from equity data for this row
            combo_pair_id = row.get("final_combo_pair_id", "")
            n_years = 1.0  # fallback

            # Try to get equity data to determine n_years
            try:
                # Attempt to read equity from the row's combo data
                # This requires the equity to be available, which it typically is in the matrix
                if "yearly_metrics" in row.index and pd.notna(row["yearly_metrics"]):
                    yearly_data = row["yearly_metrics"]
                    if isinstance(yearly_data, dict):
                        n_years = float(len(yearly_data))
                elif combo_pair_id:
                    # Fallback: try to infer from group mappings or equity series
                    # This is a best-effort approach
                    equity_dir = COMBINED_DIR / "equity" / combo_pair_id
                    equity_path = equity_dir / "equity.csv"
                    if equity_path.exists():
                        eq_df = pd.read_csv(equity_path)
                        ts_col, _ = _detect_equity_columns(eq_df)
                        ts = pd.to_datetime(eq_df[ts_col], utc=True, errors="coerce")
                        if ts.notna().any():
                            years_in_data = ts[ts.notna()].dt.year.unique()
                            n_years = (
                                float(len(years_in_data))
                                if len(years_in_data) > 0
                                else 1.0
                            )
            except Exception:
                pass  # Keep fallback n_years=1.0

            # Portfolio breadth scaling (groups_count).
            # Some pipelines only carry groups_mapping_json; infer len(mapping) as fallback.
            n_categories = 1.0
            n_categories_raw = pd.to_numeric(
                row.get("groups_count", np.nan), errors="coerce"
            )
            if pd.notna(n_categories_raw):
                n_categories = float(n_categories_raw)
            else:
                mapping_json = row.get("groups_mapping_json", None)
                if isinstance(mapping_json, str) and mapping_json:
                    try:
                        mapping_obj = json.loads(mapping_json)
                        if isinstance(mapping_obj, dict) and len(mapping_obj) > 0:
                            n_categories = float(len(mapping_obj))
                    except Exception:
                        pass

            if not np.isfinite(n_categories) or n_categories <= 0.0:
                n_categories = 1.0

            # Apply adjustments (total metrics)
            avg_r_adjusted = shrinkage_adjusted(
                average_r=float(avg_r_raw),
                n_trades=int(n_trades),
                n_years=n_years,
                n_categories=n_categories,
            )

            # Winrate: Wilson Score Lower Bound adjustment (conservative CI)
            winrate_decimal = float(winrate_raw) / 100.0
            winrate_adjusted = wilson_score_lower_bound(
                winrate=winrate_decimal, n_trades=int(n_trades)
            )

            pod_adjusted = risk_adjusted(
                profit_over_drawdown=float(max(pod_global_raw, 0.0)),
                n_trades=int(n_trades),
                n_years=n_years,
                n_categories=n_categories,
            )

            # Normalize adjusted PoD for score
            pod_term = pod_adjusted / (1.0 + pod_adjusted) if pod_adjusted >= 0 else 0.0

            comp_score = (avg_r_adjusted + winrate_adjusted + pod_term) / 3.0
            matrix.at[idx, "comp_score"] = float(comp_score)

            # Store trade-count adjusted metrics as separate columns.
            # Naming uses "*_adust_*" (requested), winrate in % (0-100).
            matrix.at[idx, "winrate_adust"] = float(winrate_adjusted * 100.0)
            matrix.at[idx, "avg_r_adust"] = float(avg_r_adjusted)
            matrix.at[idx, "profit_over_dd_adust"] = float(pod_adjusted)

            # DEBUG: Log first few updated scores
            if idx < 3:
                print(
                    f"  [Scores] Portfolio {idx}: "
                    f"wr_raw={winrate_raw:.1f}% → adust={winrate_adjusted*100:.1f}%, "
                    f"avg_r_raw={avg_r_raw:.3f} → adust={avg_r_adjusted:.3f}, "
                    f"pod_raw={pod_global_raw:.3f} → adust={pod_adjusted:.3f}, "
                    f"comp_score={comp_score:.4f}"
                )

    # 4. Calculate persistent _composite_rep_score for cluster representative selection
    # Formula: 0.5 * comp_score_combined + 0.25 * stability_score_combined + 0.25 * robustness_mean
    # This allows reuse throughout the pipeline without recalculation
    matrix["_composite_rep_score"] = matrix.apply(
        _calculate_composite_score_for_representatives, axis=1
    )

    # print(f"[Scores] Zusätzliche Scores berechnet")

    return matrix


def compute_final_score(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet den finalen Score:

    final_score = 0.5 * comp_score
                + 0.25 * stability_score
                + 0.25 * robustness_mean

    Returns:
        Matrix mit final_score Spalte, sortiert nach final_score (absteigend)
    """
    if matrix.empty:
        return matrix

    # QW1 OPTIMIERT: Keine .copy() — modifiziere in-place

    comp = pd.to_numeric(matrix.get("comp_score", np.nan), errors="coerce").fillna(0.0)
    stab = pd.to_numeric(
        matrix.get("stability_score_monthly", np.nan), errors="coerce"
    ).fillna(0.0)
    rob = pd.to_numeric(matrix.get("robustness_mean", np.nan), errors="coerce").fillna(
        0.0
    )

    matrix["final_score"] = 0.5 * comp + 0.25 * stab + 0.25 * rob

    # Nach final_score sortieren
    matrix = matrix.sort_values("final_score", ascending=False).reset_index(drop=True)

    # print(f"[Final] Finaler Score berechnet und sortiert")

    return matrix


def _compute_yearly_metrics_from_equity_and_trades(
    equity_path: Path, trades_path: Path
) -> Dict[str, Dict[str, float]]:
    """
    Berechnet jährliche Metriken aus equity.csv und trades.json Dateien.

    Args:
        equity_path: Path zur equity.csv
        trades_path: Path zur trades.json

    Returns:
        Dict mit Jahr -> Metriken Mapping
        Format: {"2025": {"net_pnl": 100.0, "max_dd": 50.0, "profit_over_dd": 2.0,
                         "winrate": 60.0, "avg_r": 0.5, "trades": 10}, ...}
    """
    yearly_metrics: Dict[str, Dict[str, float]] = {}

    if not equity_path.exists():
        return yearly_metrics

    try:
        # Equity laden
        df = pd.read_csv(equity_path)
        ts_col, eq_col = _detect_equity_columns(df)

        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        eq_vals = pd.to_numeric(df[eq_col], errors="coerce")
        mask = ts.notna() & eq_vals.notna()

        if not mask.any():
            return yearly_metrics

        equity_series = pd.Series(eq_vals[mask].values, index=ts[mask]).sort_index()

        # Trades laden (optional)
        trades_df = None
        if trades_path.exists():
            try:
                with trades_path.open("r", encoding="utf-8") as f:
                    trades = json.load(f)

                if isinstance(trades, list) and trades:
                    trades_df = pd.DataFrame(trades)
                    # Entry-Time konvertieren
                    if "entry_time" in trades_df.columns:
                        trades_df["entry_time"] = pd.to_datetime(
                            trades_df["entry_time"], utc=True, errors="coerce"
                        )
            except Exception as e:
                print(f"[YearlyMetrics] Warnung: Konnte Trades nicht laden: {e}")

        # Nach Jahren gruppieren (chronologisch aufsteigend)
        years = sorted(equity_series.index.year.unique())
        prev_year_last_equity: Optional[float] = None

        for year in years:
            year_str = str(year)
            year_mask = equity_series.index.year == year
            year_equity = equity_series[year_mask]

            if year_equity.empty:
                continue

            year_first_equity = float(year_equity.iloc[0])
            year_last_equity = float(year_equity.iloc[-1])

            # Wenn nur ein Punkt im Jahr vorhanden ist, können wir keine
            # intra-year-Metriken berechnen, merken uns aber den letzten
            # Equity-Wert als Basis für das Folgejahr.
            if len(year_equity) < 2:
                prev_year_last_equity = year_last_equity
                continue

            # Net P&L:
            # - Erstes Jahr: equity_last - equity_first (wie bisher)
            # - Folgejahre: equity_last - equity_last des Vorjahres
            if prev_year_last_equity is None:
                net_pnl = float(year_last_equity - year_first_equity)
            else:
                net_pnl = float(year_last_equity - prev_year_last_equity)

            # Max Drawdown:
            # - Erstes Jahr: klassisch auf Basis der Jahres-Equity
            # - Folgejahre: rollierendes Maximum mindestens auf Höhe
            #   des letzten Equity-Werts des Vorjahres
            roll_max = year_equity.cummax()
            if prev_year_last_equity is not None:
                roll_max = roll_max.clip(lower=prev_year_last_equity)
            dd = roll_max - year_equity
            max_dd = float(dd.max())

            # Profit over DD
            profit_over_dd = _safe_profit_over_dd(net_pnl, max_dd)

            # Initialisiere Metriken
            yearly_metrics[year_str] = {
                "net_pnl": net_pnl,
                "profit_over_dd": profit_over_dd,
                "winrate": np.nan,
                "avg_r": np.nan,
                "max_dd": max_dd,
                "trades": 0,
            }

            # Trades-basierte Metriken berechnen
            if (
                trades_df is not None
                and not trades_df.empty
                and "entry_time" in trades_df.columns
            ):
                # Filter Trades für dieses Jahr
                year_trades = trades_df[trades_df["entry_time"].dt.year == year]

                if not year_trades.empty:
                    # Anzahl Trades
                    yearly_metrics[year_str]["trades"] = int(len(year_trades))

                    # Winrate
                    if "result" in year_trades.columns:
                        results = pd.to_numeric(year_trades["result"], errors="coerce")
                        wins = (results > 0).sum()
                        total = results.notna().sum()
                        if total > 0:
                            winrate = float(wins / total * 100)
                            yearly_metrics[year_str]["winrate"] = winrate

                    # Average R-Multiple
                    if "r_multiple" in year_trades.columns:
                        r_vals = pd.to_numeric(
                            year_trades["r_multiple"], errors="coerce"
                        )
                        if not r_vals.isna().all():
                            avg_r = float(r_vals.mean())
                            yearly_metrics[year_str]["avg_r"] = avg_r

            # Letzten Equity-Wert für das Folgejahr merken
            prev_year_last_equity = year_last_equity

    except Exception as e:
        print(f"[YearlyMetrics] Fehler beim Berechnen für {equity_path}: {e}")

    return yearly_metrics


def _expand_final_combos_for_display(
    matrix: pd.DataFrame, groups: List[WalkforwardGroup]
) -> pd.DataFrame:
    """
    Erweitert die finale Top-10-Matrix so, dass jede final_combo_pair_id
    in separate Zeilen für jedes Leg (group_id) aufgeteilt wird.

    Analog zu _expand_pairs_for_display aus walkforward_analyzer.py:
    - Erste Spalte: combo_pair_id (final_combo_pair_id)
    - Zweite Spalte: group_id (entspricht dem spezifischen Leg)
    - Pro Leg eine eigene Zeile mit allen relevanten Daten

    Args:
        matrix: DataFrame mit final_combo_pair_id und groups_mapping_json
        groups: Liste der WalkforwardGroup-Objekte

    Returns:
        Erweitertes DataFrame mit einer Zeile pro Leg
    """
    if matrix.empty:
        return matrix

    rows: List[Dict[str, Any]] = []

    # Gruppenindex erstellen für schnellen Zugriff
    group_by_id = {g.group_id: g for g in groups}

    for _, final_row in matrix.iterrows():
        final_combo_pair_id = str(
            final_row.get("final_combo_pair_id", final_row.get("combo_pair_id", ""))
        )
        groups_mapping_json = final_row.get("groups_mapping_json", "{}")

        # Parse groups_mapping
        try:
            groups_mapping = (
                json.loads(groups_mapping_json)
                if isinstance(groups_mapping_json, str)
                else groups_mapping_json
            )
        except Exception:
            groups_mapping = {}

        if not groups_mapping:
            print(f"[Expand] Warnung: Keine groups_mapping für {final_combo_pair_id}")
            continue

        # Gemeinsame Paar-Scores (gelten für alle Legs)
        pair_scores = {
            "combo_pair_id": final_combo_pair_id,
            "category": final_row.get("category", "N/A"),  # Kategorie hinzufügen
            "top11_categories": final_row.get(
                "top11_categories", ""
            ),  # Top-11-Kategorien-Zugehörigkeit
            "final_score": _round_value(final_row.get("final_score", np.nan), 5),
            "comp_score": _round_value(final_row.get("comp_score", np.nan), 5),
            "stability_score_monthly": _round_value(
                final_row.get("stability_score_monthly", np.nan), 5
            ),
            "robustness_mean": _round_value(
                final_row.get("robustness_mean", np.nan), 5
            ),
            "total_profit": _round_value(final_row.get("total_profit", np.nan), 2),
            "total_max_dd": _round_value(final_row.get("total_max_dd", np.nan), 2),
            "total_profit_over_dd": _round_value(
                final_row.get("total_profit_over_dd", np.nan), 2
            ),
            "profit_over_dd_adust": _round_value(
                final_row.get("profit_over_dd_adust", np.nan), 2
            ),
            "avg_r": _round_value(final_row.get("avg_r", np.nan), 4),
            "avg_r_adust": _round_value(final_row.get("avg_r_adust", np.nan), 4),
            "sharpe_equity": _round_value(final_row.get("sharpe_equity", np.nan), 5),
            "sortino_equity": _round_value(final_row.get("sortino_equity", np.nan), 5),
            "sharpe_trade": _round_value(final_row.get("sharpe_trade", np.nan), 5),
            "sortino_trade": _round_value(final_row.get("sortino_trade", np.nan), 5),
            "winrate": _round_value(final_row.get("winrate", np.nan), 2),
            "winrate_adust": _round_value(final_row.get("winrate_adust", np.nan), 2),
            "total_trades": (
                int(final_row.get("total_trades", 0))
                if pd.notna(final_row.get("total_trades"))
                else 0
            ),
            "identical_trades_entry": (
                int(final_row.get("identical_trades_entry", 0))
                if pd.notna(final_row.get("identical_trades_entry"))
                else 0
            ),
            "identical_trades_absolut": (
                int(final_row.get("identical_trades_absolut", 0))
                if pd.notna(final_row.get("identical_trades_absolut"))
                else 0
            ),
            "identical_trades_absolut_percentage": _round_value(
                final_row.get("identical_trades_absolut_percentage", np.nan), 5
            ),
        }

        # Pro Leg (group_id) eine Zeile erstellen
        for group_id, combo_pair_id in groups_mapping.items():
            # Lade die Top-10-Daten dieser Gruppe
            group = group_by_id.get(group_id)
            if not group:
                print(f"[Expand] Warnung: Gruppe {group_id} nicht gefunden")
                continue

            # Bevorzugt die verfeinerte Top-50, fällt zurück auf Basis-Top-50,
            # danach auf Legacy-Top-50 falls vorhanden.
            candidates = [
                group.output_dir / "top_50_walkforward_combos_refined.csv",
                group.output_dir / "top_50_walkforward_combos.csv",
            ]
            top_csv = next((p for p in candidates if p.exists()), None)

            if not top_csv:
                print(f"[Expand] Warnung: Keine Top-50-Datei für {group_id}")
                continue

            try:
                top50_df = pd.read_csv(top_csv, skip_blank_lines=True)

                # Finde alle Zeilen für diese spezifische combo_pair_id (Leg A und B)
                combo_data = top50_df[top50_df["combo_pair_id"] == combo_pair_id]

                if combo_data.empty:
                    print(
                        f"[Expand] Warnung: combo_pair_id {combo_pair_id} nicht in {group_id} gefunden"
                    )
                    continue

                year_prefix_pattern = re.compile(r"^(\d{4})_")

                # Erzeuge eine Zeile pro Leg (z.B. A und B)
                for _, combo_row in combo_data.iterrows():
                    leg_row = dict(pair_scores)
                    leg_row["group_id"] = group_id

                    # combo_leg explizit übernehmen, falls vorhanden
                    if "combo_leg" in combo_row.index:
                        leg_row["combo_leg"] = combo_row["combo_leg"]

                    # Übertrage alle Spalten aus der Gruppen-Top-50
                    # (außer den bereits gesetzten Paar-Scores und Jahresspalten)
                    for col in combo_row.index:
                        # Überspringe Jahresspalten (werden später neu berechnet)
                        if year_prefix_pattern.match(str(col)):
                            continue
                        if col in ("combo_leg",):
                            continue
                        if col not in leg_row:
                            leg_row[col] = combo_row[col]

                    rows.append(leg_row)

            except Exception as e:
                print(f"[Expand] Fehler beim Verarbeiten von {group_id}: {e}")
                import traceback

                traceback.print_exc()

        # Leerzeile als Trennung zwischen finalen Kombinationen
        if rows:
            blank_row = {col: pd.NA for col in rows[-1].keys()}
            rows.append(blank_row)

    if not rows:
        return pd.DataFrame()

    expanded_df = pd.DataFrame(rows)

    # Berechne neue jährliche Metriken aus den final_combos equity.csv Dateien
    print(
        "\n[YearlyMetrics] Berechne neue jährliche Metriken aus final_combos equity.csv..."
    )

    # Gruppiere nach final_combo_pair_id um Duplikate zu vermeiden
    unique_combos = expanded_df["combo_pair_id"].dropna().unique()

    # Dict zum Speichern: combo_pair_id -> yearly_metrics
    combo_yearly_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for combo_pair_id in unique_combos:
        # Paths zu equity.csv und trades.json dieser final_combo
        combo_dir = COMBINED_MATRIX_DIR / "final_combos" / combo_pair_id
        equity_path = combo_dir / "equity.csv"
        trades_path = combo_dir / "trades.json"

        if not equity_path.exists():
            print(
                f"[YearlyMetrics] Warnung: equity.csv nicht gefunden für {combo_pair_id}"
            )
            continue

        # Berechne Metriken (inkl. Trades-Daten)
        yearly_metrics = _compute_yearly_metrics_from_equity_and_trades(
            equity_path, trades_path
        )

        if yearly_metrics:
            combo_yearly_metrics[combo_pair_id] = yearly_metrics
            years_str = ", ".join(sorted(yearly_metrics.keys(), reverse=True))
            print(f"[YearlyMetrics] ✓ {combo_pair_id}: {years_str}")

    # Füge jährliche Metriken als Spalten hinzu
    if combo_yearly_metrics:
        # Sammle alle Jahre
        all_years = set()
        for metrics in combo_yearly_metrics.values():
            all_years.update(metrics.keys())

        # Sortiere Jahre absteigend
        sorted_years = sorted(all_years, reverse=True)

        # Füge Spalten hinzu:
        # YYYY_year, YYYY_net_pnl, YYYY_max_dd, YYYY_profit_over_dd,
        # YYYY_winrate, YYYY_avg_r, YYYY_trades
        for year in sorted_years:
            expanded_df[f"{year}_year"] = pd.NA
            expanded_df[f"{year}_net_pnl"] = pd.NA
            expanded_df[f"{year}_max_dd"] = pd.NA
            expanded_df[f"{year}_profit_over_dd"] = pd.NA
            expanded_df[f"{year}_winrate"] = pd.NA
            expanded_df[f"{year}_avg_r"] = pd.NA
            expanded_df[f"{year}_trades"] = pd.NA

        # Fülle Werte für jede Zeile basierend auf combo_pair_id
        for idx, row in expanded_df.iterrows():
            combo_pair_id = row.get("combo_pair_id")
            if pd.isna(combo_pair_id) or combo_pair_id not in combo_yearly_metrics:
                continue

            metrics = combo_yearly_metrics[combo_pair_id]

            for year in sorted_years:
                if year in metrics:
                    # Jahres-Spalte explizit setzen (Jahr als Zahl)
                    expanded_df.at[idx, f"{year}_year"] = int(year)
                    expanded_df.at[idx, f"{year}_net_pnl"] = _round_value(
                        metrics[year]["net_pnl"], 2
                    )
                    expanded_df.at[idx, f"{year}_max_dd"] = _round_value(
                        metrics[year]["max_dd"], 2
                    )
                    expanded_df.at[idx, f"{year}_profit_over_dd"] = _round_value(
                        metrics[year]["profit_over_dd"], 2
                    )
                    expanded_df.at[idx, f"{year}_winrate"] = _round_value(
                        metrics[year]["winrate"], 2
                    )
                    expanded_df.at[idx, f"{year}_avg_r"] = _round_value(
                        metrics[year]["avg_r"], 4
                    )
                    expanded_df.at[idx, f"{year}_trades"] = (
                        int(metrics[year]["trades"])
                        if pd.notna(metrics[year]["trades"])
                        else pd.NA
                    )

        print(
            f"[YearlyMetrics] ✓ Jährliche Metriken (6 pro Jahr) für {len(sorted_years)} Jahre hinzugefügt: {', '.join(sorted_years)}"
        )

    # Spalten-Reihenfolge definieren (wie in walkforward_analyzer.py)
    # 1. Snapshot Meta
    snapshot_meta_cols = [
        "symbol",
        "timeframe",
        "direction",
        "strategy_name",
        "szenario",
    ]

    # 2. Primary Meta
    primary_meta = [
        "combo_pair_id",
        "category",  # Kategorie-Zuordnung
        "top11_categories",  # Top-11-Kategorien-Zugehörigkeit
        "group_id",
        "combo_id",
        "source_walkforward",
    ]

    # 3. Secondary Pair Meta (Scores)
    secondary_pair_meta = [
        "final_score",
        "comp_score",
        "stability_score_monthly",
        "robustness_mean",
        "total_profit",
        "total_profit_over_dd",
        "profit_over_dd_adust",
        "winrate",
        "winrate_adust",
        "sharpe_equity",
        "sortino_equity",
        "sharpe_trade",
        "sortino_trade",
        "avg_r",
        "avg_r_adust",
        "total_max_dd",
        "total_trades",
        "identical_trades_entry",
        "identical_trades_absolut",
        "identical_trades_absolut_percentage",
    ]

    # 4. Per-Leg Metriken
    # Hinweis: robustness_score_1_jittered_80 (falls vorhanden) enthält den
    # pro-Leg gejitterten Robustness-1-Wert aus den verfeinerten Top-10-Listen
    # des walkforward_analyzer und wird vor robustness_score_1 einsortiert.
    # robustness_score_1_jittered_80 wird nur angezeigt, wenn auch robustness_1_mean vorhanden ist.
    per_leg_metric_display_base = [
        "Net Profit",
        "Drawdown",
        "profit_over_dd",
        "Commission",
        "Avg R-Multiple",
        "Winrate (%)",
        "trades",
        "Sharpe (trade)",
        "Sortino (trade)",
        "robustness_1_mean",
        "robustness_score_1_jittered_80",
        "robustness_score_1",
        "robustness_score_2",
        "tp_sl_stress_score",
        "stability_score",
        "p_mean_r_gt_0",
        "p_net_profit_gt_0",
        "same_trades_entry",
        "same_trades_absolut",
        "same_trades_absolut_percentage",
    ]

    # Bedingte Anzeige: robustness_score_1_jittered_80 nur wenn robustness_1_mean vorhanden
    per_leg_metric_display = per_leg_metric_display_base.copy()
    if "robustness_1_mean" not in expanded_df.columns:
        per_leg_metric_display = [
            m for m in per_leg_metric_display if m != "robustness_score_1_jittered_80"
        ]

    # 5. Session/HTF Settings
    session_htf_cols = [
        "session_filter",
        "htf_tf",
        "htf_filter",
        "extra_htf_tf",
        "extra_htf_filter",
        "time_period",
        "total_stars_combined",
        "yearly_stars_per_leg",
    ]

    # 6. Parameter identifizieren (alles was nicht Meta, Metrik oder Jahresspalte ist)
    year_prefix = re.compile(r"^(\d{4})_")
    known_cols = set(
        snapshot_meta_cols
        + primary_meta
        + secondary_pair_meta
        + per_leg_metric_display
        + session_htf_cols
    )

    param_cols_raw = []
    yearly_cols = []

    for col in expanded_df.columns:
        if col in known_cols:
            continue
        # Jahresspalten identifizieren (Format: YYYY_metric)
        if year_prefix.match(str(col)):
            yearly_cols.append(col)
        else:
            # Alles andere sind Parameter - aber nur wenn mindestens ein Wert vorhanden ist
            # Prüfe ob die Spalte komplett leer ist (nur NA/NaN/None/leere Strings)
            col_values = expanded_df[col]
            has_value = False
            for val in col_values:
                if pd.notna(val) and str(val).strip() != "":
                    has_value = True
                    break
            if has_value:
                param_cols_raw.append(col)

    # Sortiere Parameter so, dass zusammengehörige direkt nacheinander kommen
    # z_score_long -> z_score_short
    # htf_ema -> extra_htf_ema
    param_cols = []
    skip_next = set()

    for col in param_cols_raw:
        if col in skip_next:
            continue

        param_cols.append(col)

        # Wenn z_score_long, füge z_score_short direkt danach ein
        if col == "z_score_long" and "z_score_short" in param_cols_raw:
            param_cols.append("z_score_short")
            skip_next.add("z_score_short")

        # Wenn htf_ema, füge extra_htf_ema direkt danach ein
        if col == "htf_ema" and "extra_htf_ema" in param_cols_raw:
            param_cols.append("extra_htf_ema")
            skip_next.add("extra_htf_ema")

    # Jahresspalten sortieren: neueste Jahre zuerst
    # Gruppiere nach Jahr und sortiere innerhalb des Jahres nach Metrik-Reihenfolge
    year_groups = {}
    for col in yearly_cols:
        match = year_prefix.match(str(col))
        if match:
            year = int(match.group(1))
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(col)

    # Sortiere Jahre absteigend und behalte die Metrik-Reihenfolge innerhalb jedes Jahres
    # Jahr-Spalte zuerst, danach die Kennzahlen
    metric_order = [
        "year",
        "net_pnl",
        "profit_over_dd",
        "winrate",
        "avg_r",
        "max_dd",
        "trades",
    ]
    sorted_yearly_cols = []
    for year in sorted(year_groups.keys(), reverse=True):
        year_cols = year_groups[year]
        # Sortiere nach Metrik-Reihenfolge
        sorted_year_cols = []
        for metric in metric_order:
            for col in year_cols:
                if col.endswith(f"_{metric}"):
                    sorted_year_cols.append(col)
        # Füge übrige Spalten hinzu (falls welche übrig sind)
        for col in year_cols:
            if col not in sorted_year_cols:
                sorted_year_cols.append(col)
        sorted_yearly_cols.extend(sorted_year_cols)

    # 7. Finale Spaltenreihenfolge (wie in walkforward_analyzer.py)
    ordered_cols = (
        snapshot_meta_cols
        + primary_meta
        + secondary_pair_meta
        + per_leg_metric_display
        + param_cols  # PARAMETER VOR JAHRESWERTEN!
        + session_htf_cols
        + sorted_yearly_cols
    )

    # Nur existierende Spalten verwenden
    ordered_cols = [c for c in ordered_cols if c in expanded_df.columns]

    expanded_df = expanded_df[ordered_cols]

    # Stelle sicher, dass szenario als Integer (ohne Nachkommastellen) ausgegeben wird
    if "szenario" in expanded_df.columns:
        expanded_df["szenario"] = pd.to_numeric(
            expanded_df["szenario"], errors="coerce"
        ).astype("Int64")

    return expanded_df


def _collapse_champions_combo_pairs_to_one_liners(
    expanded_df: pd.DataFrame,
) -> pd.DataFrame:
    """Kollabiert die Leg-basierte Champion-Ansicht zu einer Portfolio-1-Zeile pro combo_pair_id.

    Regel:
    - Es werden nur Spalten übernommen, deren Werte für *alle* Legs eines Portfolios
      exakt identisch sind.
    - Spalten, die in einer Portfolio-Gruppe komplett leer sind, werden nicht übernommen.

        Hinweis:
        - Es wird bewusst *keine* Leg-Zusammenfassungsspalte (z.B. `portfolio_legs`) erzeugt.
            Diese Übersicht soll ausschließlich „wirklich gemeinsame“ Spalten enthalten.
    """
    if expanded_df is None or expanded_df.empty:
        return pd.DataFrame()

    if "combo_pair_id" not in expanded_df.columns:
        return pd.DataFrame()

    # Trennzeilen entfernen (combo_pair_id ist dort NA)
    df = expanded_df.loc[pd.notna(expanded_df["combo_pair_id"])].copy()
    if df.empty:
        return pd.DataFrame()

    def _is_effectively_empty(v: Any) -> bool:
        if pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    def _all_values_equal(values: List[Any]) -> bool:
        first_set = False
        first = None
        for v in values:
            if not first_set:
                first = v
                first_set = True
                continue

            # Beide leer/NA
            if pd.isna(first) and pd.isna(v):
                continue
            # Einer leer, einer nicht
            if pd.isna(first) != pd.isna(v):
                return False

            try:
                eq = v == first
                if isinstance(eq, (np.ndarray, pd.Series)):
                    eq = bool(np.all(eq))
                if eq is pd.NA:
                    eq = False
            except Exception:
                eq = str(v) == str(first)

            if not bool(eq):
                return False

        return True

    out_rows: List[Dict[str, Any]] = []
    base_cols_in_order = list(df.columns)

    for combo_pair_id, sub in df.groupby("combo_pair_id", sort=True):
        row_out: Dict[str, Any] = {"combo_pair_id": combo_pair_id}

        for col in base_cols_in_order:
            if col == "combo_pair_id":
                continue

            values = list(sub[col].tolist())

            # Nur behalten, wenn nicht komplett leer und exakt identisch
            if all(_is_effectively_empty(v) for v in values):
                continue
            if not _all_values_equal(values):
                continue

            # Repräsentativen Wert wählen (prefer non-empty)
            chosen = None
            for v in values:
                if not _is_effectively_empty(v):
                    chosen = v
                    break
            if chosen is None:
                continue

            row_out[col] = chosen

        out_rows.append(row_out)

    if not out_rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(out_rows)

    # Spaltenreihenfolge: combo_pair_id, category/top11_categories falls vorhanden, dann Rest in Eingangsreihenfolge
    preferred_front = ["combo_pair_id", "category", "top11_categories"]
    ordered = []
    for c in preferred_front:
        if c in out_df.columns and c not in ordered:
            ordered.append(c)
    for c in base_cols_in_order:
        if c in out_df.columns and c not in ordered:
            ordered.append(c)

    out_df = out_df[ordered]

    # Sortierung: identisch zur Top-11-Kategorien-Reihenfolge
    if "category" in out_df.columns:
        category_order = [
            "Top Performer",
            "Capital Efficiency",
            "Defensive Low Drawdown",
            "Stable Compounder",
            "Sharpe Trader",
            "High Conviction",
            "High Turnover",
            "Cost Efficient",
            "Return Shape",
            "Independent",
            "Diversifier",
        ]
        try:
            out_df = out_df.copy()
            out_df["category"] = pd.Categorical(
                out_df["category"], categories=category_order, ordered=True
            )
            out_df = out_df.sort_values(
                by=["category", "combo_pair_id"], kind="mergesort", na_position="last"
            ).reset_index(drop=True)
        except Exception:
            # defensiv: falls Kategorie-Spalte unerwartete Typen enthält
            pass

    return out_df


def _compute_additional_categorical_metrics(
    matrix: pd.DataFrame,
    *,
    chunk_size: int = 512,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Berechnet zusätzliche Metriken für das kategorische Ranking.

    OPTIMIERT: Vollständig vectorisiert mit batch-processing und konsistentem Cache-Zugriff.
    Reduziert O(n × m) Disk I/O auf O(n) durch get_equity_cached()/get_trades_cached().
    Eliminiert iterrows()-Schleifen und nutzt NumPy/pandas vectorized operations.

    Neue Metriken:
    - worst_weekly_profit: Niedrigster wöchentlicher Profit
    - average_trade_duration_hours: Durchschnittliche Trade-Dauer in Stunden
    - commission: Gesamtkosten (Commission)
    - profit_without_commission: Profit vor Kosten
    - fee_drag: Verhältnis abs(commission) / profit_without_commission (kleiner ist besser)
    - time_in_market_hours: Gesamte Zeit in Trades (Stunden)
    - duration_days: Anzahl Tage der Testperiode
    - ulcer_index_weekly: Ulcer Index auf wöchentlicher Basis
    - yearly_pnl_dispersion: Standardabweichung der jährlichen Profits
    - max_trades_simult: Maximale Anzahl simultaner Trades
    - long_short_overlap_episodes: Anzahl Long/Short Overlap-Episoden
    - equity_curvature: mittlere absolute zweite Differenz der monatlichen Log-Equity L_t
    - equity_log_vol: Standardabweichung der monatlichen Log-Differenzen ΔL_t
    - dd_slope_stability: 1 / (1 + std(ΔDD_t | DD_t>0)) auf Basis der ursprünglichen Equity-Frequenz
    - time_in_highs: Anteil der Zeitpunkte, an denen E_t dem laufenden Maximum entspricht

    Returns:
        Matrix mit zusätzlichen Spalten
    """
    if matrix.empty:
        return matrix

    # QW1 OPTIMIERT: Keine .copy() - in-place Modifikation ist sicher
    # matrix = matrix.copy()

    n = len(matrix)
    chunk_size = max(1, int(chunk_size))

    if show_progress:
        print(
            f"[Categorical] Berechne zusätzliche Metriken für {n} Portfolios (chunk_size={chunk_size})..."
        )

    # Pre-fetch total_profit once (avoids KeyError surfacing deep in the loop and improves speed)
    total_profits_all = _as_numeric_array(matrix, "total_profit")

    # Allocate output arrays once (keeps peak memory stable even for large matrices)
    equity_out: Dict[str, np.ndarray] = {
        "worst_weekly_profit": np.full(n, np.nan),
        "ulcer_index_weekly": np.full(n, np.nan),
        "yearly_pnl_dispersion": np.full(n, np.nan),
        "duration_days": np.full(n, np.nan),
        # Equity-Shape Features (monthly returns; required for multiple categories)
        "equity_returns_skew": np.full(n, np.nan),
        "equity_returns_kurtosis": np.full(n, np.nan),
        "equity_returns_autocorr": np.full(n, np.nan),
        "equity_returns_volatility": np.full(n, np.nan),
        # Equity Sharpe/Sortino (monthly returns; informational + optional scoring extensions)
        "sharpe_equity": np.full(n, np.nan),
        "sortino_equity": np.full(n, np.nan),
        "equity_curvature": np.full(n, np.nan),
        "equity_log_vol": np.full(n, np.nan),
        "dd_slope_stability": np.full(n, np.nan),
        "time_in_highs": np.full(n, np.nan),
    }
    trades_out: Dict[str, np.ndarray] = {
        "average_trade_duration_hours": np.full(n, np.nan),
        "time_in_market_hours": np.full(n, np.nan),
        "commission": np.full(n, np.nan),
        "profit_without_commission": np.full(n, np.nan),
        "fee_drag": np.full(n, np.nan),
        "max_trades_simult": np.zeros(n, dtype=int),
        "long_short_overlap_episodes": np.zeros(n, dtype=int),
        # Trade Sharpe/Sortino (R-multiples; required for Sharpe/Turnover categories)
        "sharpe_trade": np.full(n, np.nan),
        "sortino_trade": np.full(n, np.nan),
        # Trade overlap (same entry timestamps within a portfolio)
        "identical_trades_entry": np.zeros(n, dtype=int),
        "identical_trades_absolut": np.zeros(n, dtype=int),
        "identical_trades_absolut_percentage": np.full(n, np.nan),
    }

    equity_load_stats: Dict[str, int] = {"direct": 0, "cached": 0, "missing": 0}
    trades_load_stats: Dict[str, int] = {"direct": 0, "cached": 0, "missing": 0}
    equity_errors: Dict[str, int] = {}
    trades_errors: Dict[str, int] = {}
    valid_equity_count = 0
    valid_trades_count = 0
    both_missing = 0

    chunk_iter = range(0, n, chunk_size)
    if show_progress:
        chunk_iter = tqdm(
            chunk_iter,
            total=(n + chunk_size - 1) // chunk_size,
            desc="[Categorical] Metrics",
            unit="chunk",
        )

    for start in chunk_iter:
        end = min(n, start + chunk_size)
        chunk = matrix.iloc[start:end]

        equity_series_list = _load_equity_batch(
            chunk,
            show_progress=False,
            stats=equity_load_stats,
            print_stats=False,
        )
        trades_list = _load_trades_batch(
            chunk,
            show_progress=False,
            stats=trades_load_stats,
            print_stats=False,
        )

        for eq in equity_series_list:
            if eq is not None and not eq.empty:
                valid_equity_count += 1
        for tr in trades_list:
            if isinstance(tr, pd.DataFrame):
                if not tr.empty:
                    valid_trades_count += 1
            else:
                if len(tr) > 0:
                    valid_trades_count += 1
        for eq, tr in zip(equity_series_list, trades_list):
            eq_missing = (eq is None) or (isinstance(eq, pd.Series) and eq.empty)
            tr_missing = (
                (tr is None)
                or (isinstance(tr, pd.DataFrame) and tr.empty)
                or (not isinstance(tr, pd.DataFrame) and len(tr) == 0)
            )
            if eq_missing and tr_missing:
                both_missing += 1

        eq_metrics = _compute_equity_metrics_batch(
            equity_series_list,
            show_progress=False,
            error_counts=equity_errors,
        )
        tr_metrics = _compute_trades_metrics_batch(
            trades_list,
            total_profits_all[start:end],
            show_progress=False,
            error_counts=trades_errors,
        )

        for col, arr in eq_metrics.items():
            equity_out[col][start:end] = arr
        for col, arr in tr_metrics.items():
            trades_out[col][start:end] = arr

    # Assign columns back to matrix (single pass → minimal DataFrame fragmentation)
    for col, arr in equity_out.items():
        matrix[col] = arr
    for col, arr in trades_out.items():
        matrix[col] = arr

    if show_progress:
        missing_equity_count = n - valid_equity_count
        missing_trades_count = n - valid_trades_count
        print(
            f"[Categorical] Equity: {valid_equity_count} gültig, {missing_equity_count} fehlend/leer "
            f"(direct={equity_load_stats['direct']}, cached={equity_load_stats['cached']}, missing={equity_load_stats['missing']})"
        )
        print(
            f"[Categorical] Trades: {valid_trades_count} gültig, {missing_trades_count} fehlend/leer "
            f"(direct={trades_load_stats['direct']}, cached={trades_load_stats['cached']}, missing={trades_load_stats['missing']})"
        )
        if both_missing > 0:
            print(
                f"[Categorical] ⚠️ PORTFOLIOS MIT FEHLENDEM DATEN: {both_missing} haben KEINE Equity/Trades"
            )

        if equity_errors:
            top = sorted(equity_errors.items(), key=lambda x: (-x[1], x[0]))[:3]
            print(
                f"[Categorical] Warnung: Equity-Metriken Fehler bei {sum(equity_errors.values())} Portfolios (Top: {top})"
            )
        if trades_errors:
            top = sorted(trades_errors.items(), key=lambda x: (-x[1], x[0]))[:3]
            print(
                f"[Categorical] Warnung: Trades-Metriken Fehler bei {sum(trades_errors.values())} Portfolios (Top: {top})"
            )

    return matrix


def _load_equity_batch(
    matrix: pd.DataFrame,
    *,
    show_progress: bool = True,
    stats: Optional[Dict[str, int]] = None,
    print_stats: bool = True,
) -> List[Optional[pd.Series]]:
    """Lade Equity-Series für alle Portfolios (batch).

    OPTIMIERT: Nutzt bereits geladene Series direkt OHNE Cache-Pollution.
    Nur Path-basierte Objekte werden via get_equity_cached() geladen.

    Returns:
        Liste von pd.Series (oder None bei fehlendem/ungültigem Entry).
    """
    series_list = []
    debug_stats = (
        stats if stats is not None else {"direct": 0, "cached": 0, "missing": 0}
    )
    iterator = range(len(matrix))
    if show_progress:
        iterator = tqdm(iterator, desc="[Categorical] Equity laden", unit="portfolio")

    for idx in iterator:
        row = matrix.iloc[idx]

        # Priorität 1: _equity_internal (kann Series oder Path sein)
        eq_obj = row.get("_equity_internal")
        if isinstance(eq_obj, pd.Series):
            # Already loaded → nutze direkt OHNE Cache
            series_list.append(eq_obj)
            debug_stats["direct"] += 1
            continue
        elif eq_obj and not isinstance(eq_obj, pd.Series) and str(eq_obj).strip():
            path = Path(str(eq_obj))
            if path.exists():
                series_list.append(get_equity_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        # Priorität 2: equity_path Spalte
        eq_path = row.get("equity_path")
        if eq_path and pd.notna(eq_path) and str(eq_path).strip():
            path = Path(str(eq_path))
            if path.exists():
                series_list.append(get_equity_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        # Priorität 3: Konstruiere Pfad aus final_combo_pair_id
        final_id = row.get("final_combo_pair_id")
        if final_id and pd.notna(final_id):
            path = COMBINED_MATRIX_DIR / "final_combos" / str(final_id) / "equity.csv"
            if path.exists():
                series_list.append(get_equity_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        series_list.append(None)
        debug_stats["missing"] += 1

    if print_stats:
        print(
            f"[Categorical] Equity: {debug_stats['direct']} direct, {debug_stats['cached']} cached, "
            f"{debug_stats['missing']} missing"
        )

    return series_list


def _load_trades_batch(
    matrix: pd.DataFrame,
    *,
    show_progress: bool = True,
    stats: Optional[Dict[str, int]] = None,
    print_stats: bool = True,
) -> List[pd.DataFrame | List[Dict[str, Any]]]:
    """Lade Trades für alle Portfolios (batch).

    OPTIMIERT: Nutzt bereits geladene DataFrames direkt OHNE Cache-Pollution.
    Nur Path-basierte Objekte werden via get_trades_cached() geladen.

    Returns:
        Liste von Trade-Listen (Liste von Dicts) für jedes Portfolio.
    """
    trades_list = []
    debug_stats = (
        stats if stats is not None else {"direct": 0, "cached": 0, "missing": 0}
    )
    iterator = range(len(matrix))
    if show_progress:
        iterator = tqdm(iterator, desc="[Categorical] Trades laden", unit="portfolio")

    for idx in iterator:
        row = matrix.iloc[idx]

        # Priorität 1: _trades_internal (kann DataFrame oder Path sein)
        tr_obj = row.get("_trades_internal")
        if isinstance(tr_obj, pd.DataFrame):
            # Already loaded → nutze DataFrame direkt (vermeidet teure to_dict → DataFrame Roundtrips)
            trades_list.append(tr_obj)
            debug_stats["direct"] += 1
            continue
        if isinstance(tr_obj, list):
            # Already loaded → nutze Trade-Liste direkt
            trades_list.append(tr_obj)
            debug_stats["direct"] += 1
            continue
        elif tr_obj and not isinstance(tr_obj, pd.DataFrame) and str(tr_obj).strip():
            path = Path(str(tr_obj))
            if path.exists():
                trades_list.append(get_trades_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        # Priorität 2: trades_path Spalte
        tr_path = row.get("trades_path")
        if tr_path and pd.notna(tr_path) and str(tr_path).strip():
            path = Path(str(tr_path))
            if path.exists():
                trades_list.append(get_trades_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        # Priorität 3: Konstruiere Pfad aus final_combo_pair_id
        final_id = row.get("final_combo_pair_id")
        if final_id and pd.notna(final_id):
            path = COMBINED_MATRIX_DIR / "final_combos" / str(final_id) / "trades.json"
            if path.exists():
                trades_list.append(get_trades_cached(str(path)))
                debug_stats["cached"] += 1
                continue

        trades_list.append([])
        debug_stats["missing"] += 1

    if print_stats:
        print(
            f"[Categorical] Trades: {debug_stats['direct']} direct, {debug_stats['cached']} cached, "
            f"{debug_stats['missing']} missing"
        )

    return trades_list


def _compute_equity_metrics_batch(
    equity_series_list: List[Optional[pd.Series]],
    *,
    show_progress: bool = True,
    error_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Berechnet alle Equity-basierten Metriken für eine Liste von Equity-Serien (batch).

    OPTIMIERT: Vollständig vectorized, keine Schleifen über Portfolios falls möglich.

    Returns:
        Dict mit Metrik-Namen → numpy arrays (Länge = len(equity_series_list))
    """
    n = len(equity_series_list)
    results = {
        "worst_weekly_profit": np.full(n, np.nan),
        "ulcer_index_weekly": np.full(n, np.nan),
        "yearly_pnl_dispersion": np.full(n, np.nan),
        "duration_days": np.full(n, np.nan),
        # Equity-Shape Features (monthly returns)
        "equity_returns_skew": np.full(n, np.nan),
        "equity_returns_kurtosis": np.full(n, np.nan),
        "equity_returns_autocorr": np.full(n, np.nan),
        "equity_returns_volatility": np.full(n, np.nan),
        # Equity Sharpe/Sortino (monthly returns)
        "sharpe_equity": np.full(n, np.nan),
        "sortino_equity": np.full(n, np.nan),
        "equity_curvature": np.full(n, np.nan),
        "equity_log_vol": np.full(n, np.nan),
        "dd_slope_stability": np.full(n, np.nan),
        "time_in_highs": np.full(n, np.nan),
    }

    iterator = enumerate(equity_series_list)
    if show_progress:
        iterator = tqdm(
            iterator, total=n, desc="[Categorical] Equity-Metriken", unit="portfolio"
        )

    for i, eq_series in iterator:
        if eq_series is None or eq_series.empty or len(eq_series) < 2:
            continue

        try:
            # Worst Weekly Profit
            weekly = eq_series.resample("W").last()
            if len(weekly) >= 2:
                weekly_profits = weekly.diff().dropna()
                if len(weekly_profits) > 0:
                    results["worst_weekly_profit"][i] = float(weekly_profits.min())

            # Ulcer Index (weekly)
            if len(weekly) >= 2:
                roll_max = weekly.cummax()
                dd_pct = ((weekly - roll_max) / roll_max * 100).fillna(0)
                ulcer = np.sqrt((dd_pct**2).mean())
                if np.isfinite(ulcer):
                    results["ulcer_index_weekly"][i] = float(ulcer)

            # Yearly PnL Dispersion
            yearly = eq_series.resample("YE").last()
            if len(yearly) >= 2:
                yearly_profits = yearly.diff().dropna()
                if len(yearly_profits) >= 2:
                    results["yearly_pnl_dispersion"][i] = float(yearly_profits.std())

            # Duration in days
            duration = (eq_series.index[-1] - eq_series.index[0]).total_seconds() / (
                24 * 3600
            )
            results["duration_days"][i] = float(duration)

            # Monthly series for multiple metrics (shape + sharpe/sortino)
            monthly = eq_series.resample("ME").last().dropna()
            if len(monthly) >= 3:
                # Monthly returns based on fixed base (start equity) for stability
                monthly_profits = monthly.diff().dropna()
                base = float(monthly.iloc[0]) if not monthly.empty else np.nan
                if not np.isfinite(base) or abs(base) < 1e-12:
                    base = 100_000.0

                if len(monthly_profits) >= 2:
                    monthly_returns = (monthly_profits / base).astype(float)
                    # Volatility
                    vol_val = (
                        float(monthly_returns.std(ddof=1))
                        if len(monthly_returns) >= 2
                        else np.nan
                    )
                    if np.isfinite(vol_val):
                        results["equity_returns_volatility"][i] = vol_val
                    # Skew/Kurtosis
                    if len(monthly_returns) >= 3:
                        sk = float(monthly_returns.skew())
                        if np.isfinite(sk):
                            results["equity_returns_skew"][i] = sk
                        ac = float(monthly_returns.autocorr(lag=1))
                        if np.isfinite(ac):
                            results["equity_returns_autocorr"][i] = ac
                    if len(monthly_returns) >= 4:
                        ku = float(monthly_returns.kurtosis())
                        if np.isfinite(ku):
                            results["equity_returns_kurtosis"][i] = ku

                    # Equity Sharpe/Sortino (not annualized)
                    mu = float(monthly_returns.mean())
                    sigma = float(monthly_returns.std(ddof=1))
                    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0:
                        results["sharpe_equity"][i] = mu / sigma
                    downside = np.minimum(
                        monthly_returns.to_numpy(dtype=float) - 0.0, 0.0
                    )
                    semi = (
                        float(np.sqrt(np.mean(downside**2)))
                        if downside.size > 0
                        else np.nan
                    )
                    if np.isfinite(mu) and np.isfinite(semi) and semi > 0.0:
                        results["sortino_equity"][i] = mu / semi

                # Log-Equity
                if (monthly > 0).all():
                    L = np.log(monthly.values.astype(float))
                else:
                    L = np.log1p(np.maximum(monthly.values.astype(float), 0.0))

                # Erste Differenz ΔL_t und Volatilität
                dL = np.diff(L)
                if len(dL) >= 1:
                    log_vol = float(np.std(dL))
                    if np.isfinite(log_vol):
                        results["equity_log_vol"][i] = log_vol

                # Zweite Differenz Δ²L_t und Curvature
                if len(L) >= 3:
                    d2L = np.diff(L, n=2)
                    curvature = float(np.mean(np.abs(d2L)))
                    if np.isfinite(curvature):
                        results["equity_curvature"][i] = curvature

            # Drawdown Slope Stability (auf originaler Frequenz)
            roll_max_full = eq_series.cummax()
            dd_full = (roll_max_full - eq_series).values
            dd_phase = dd_full[dd_full > 0]
            if len(dd_phase) >= 2:
                delta_dd = np.diff(dd_phase)
                dd_std = float(np.std(delta_dd)) if len(delta_dd) >= 1 else 0.0
                if np.isfinite(dd_std):
                    results["dd_slope_stability"][i] = float(1.0 / (1.0 + dd_std))

            # Time in Highs
            running_max = eq_series.cummax().values
            highs_mask = eq_series.values == running_max
            if highs_mask.size > 0:
                results["time_in_highs"][i] = float(np.mean(highs_mask))

        except Exception as exc:
            if error_counts is not None:
                key = type(exc).__name__
                error_counts[key] = error_counts.get(key, 0) + 1
            continue

    return results


def _compute_trades_metrics_batch(
    trades_list: List[pd.DataFrame | List[Dict[str, Any]]],
    total_profits: np.ndarray,
    *,
    show_progress: bool = True,
    error_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Berechnet alle Trades-basierten Metriken für eine Liste von Trades (batch).

    OPTIMIERT: Vectorized wo möglich, mit event-basierter Overlap-Berechnung.

    Args:
        trades_list: Liste von Trade-Listen (jedes Element ist eine Liste von Trade-Dicts)
        total_profits: Array mit total_profit für jedes Portfolio (für fee_drag Berechnung)

    Returns:
        Dict mit Metrik-Namen → numpy arrays
    """
    n = len(trades_list)
    results = {
        "average_trade_duration_hours": np.full(n, np.nan),
        "time_in_market_hours": np.full(n, np.nan),
        "commission": np.full(n, np.nan),
        "profit_without_commission": np.full(n, np.nan),
        "fee_drag": np.full(n, np.nan),
        "max_trades_simult": np.zeros(n, dtype=int),
        "long_short_overlap_episodes": np.zeros(n, dtype=int),
        "sharpe_trade": np.full(n, np.nan),
        "sortino_trade": np.full(n, np.nan),
        "identical_trades_entry": np.zeros(n, dtype=int),
        "identical_trades_absolut": np.zeros(n, dtype=int),
        "identical_trades_absolut_percentage": np.full(n, np.nan),
    }

    iterator = enumerate(trades_list)
    if show_progress:
        iterator = tqdm(
            iterator, total=n, desc="[Categorical] Trades-Metriken", unit="portfolio"
        )

    for i, trades in iterator:
        if trades is None:
            continue
        if isinstance(trades, pd.DataFrame):
            if trades.empty:
                continue
            trades_df = trades
        else:
            if not trades:
                continue
            trades_df = pd.DataFrame(trades)

        if trades_df.empty:
            continue

        try:
            # Trade durations (vectorized)
            if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
                entry_times = pd.to_datetime(
                    trades_df["entry_time"], utc=True, errors="coerce"
                )
                exit_times = pd.to_datetime(
                    trades_df["exit_time"], utc=True, errors="coerce"
                )
                mask = entry_times.notna() & exit_times.notna()
                if mask.any():
                    durations = (
                        exit_times[mask] - entry_times[mask]
                    ).dt.total_seconds() / 3600
                    results["average_trade_duration_hours"][i] = float(durations.mean())
                    results["time_in_market_hours"][i] = float(durations.sum())

            # Commission (robust to various field names)
            commission_total = 0.0
            if "commission" in trades_df.columns:
                commission_total = (
                    pd.to_numeric(trades_df["commission"], errors="coerce")
                    .fillna(0)
                    .sum()
                )
            elif "total_fee" in trades_df.columns:
                commission_total = (
                    pd.to_numeric(trades_df["total_fee"], errors="coerce")
                    .fillna(0)
                    .sum()
                )
            elif "total_commission" in trades_df.columns:
                commission_total = (
                    pd.to_numeric(trades_df["total_commission"], errors="coerce")
                    .fillna(0)
                    .sum()
                )
            else:
                ef = pd.to_numeric(
                    trades_df.get("entry_fee", 0), errors="coerce"
                ).fillna(0)
                xf = pd.to_numeric(
                    trades_df.get("exit_fee", 0), errors="coerce"
                ).fillna(0)
                commission_total = float(ef.sum() + xf.sum())

            results["commission"][i] = float(commission_total)

            # Profit without commission & fee drag
            total_profit = total_profits[i]
            if pd.notna(total_profit):
                profit_without_comm = total_profit + abs(commission_total)
                results["profit_without_commission"][i] = float(profit_without_comm)

                if commission_total == 0.0:
                    results["fee_drag"][i] = 0.0
                elif profit_without_comm != 0:
                    results["fee_drag"][i] = float(
                        abs(commission_total) / profit_without_comm
                    )

            # OPTIMIERT: Vectorized Trade Overlap Berechnung (event-basiert)
            overlap_metrics = _compute_trade_overlap_vectorized(trades_df)
            results["max_trades_simult"][i] = overlap_metrics["max_simult"]
            results["long_short_overlap_episodes"][i] = overlap_metrics[
                "overlap_episodes"
            ]

            # Sharpe/Sortino on R-multiples (per trade; not annualized)
            if "r_multiple" in trades_df.columns:
                r_vals = pd.to_numeric(
                    trades_df["r_multiple"], errors="coerce"
                ).dropna()
                if len(r_vals) >= 2:
                    mu = float(r_vals.mean())
                    sigma = float(r_vals.std(ddof=1))
                    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0:
                        results["sharpe_trade"][i] = mu / sigma
                    downside = np.minimum(r_vals.to_numpy(dtype=float) - 0.0, 0.0)
                    semi = (
                        float(np.sqrt(np.mean(downside**2)))
                        if downside.size > 0
                        else np.nan
                    )
                    if np.isfinite(mu) and np.isfinite(semi) and semi > 0.0:
                        results["sortino_trade"][i] = mu / semi

            # Identical trades by entry_time (within a portfolio)
            if "entry_time" in trades_df.columns:
                et = pd.to_datetime(
                    trades_df["entry_time"], utc=True, errors="coerce"
                ).dropna()
                if not et.empty:
                    counts = et.value_counts()
                    shared = counts[counts >= 2]
                    if not shared.empty:
                        identical_entry = int(shared.shape[0])
                        identical_abs = int(shared.sum())
                        results["identical_trades_entry"][i] = identical_entry
                        results["identical_trades_absolut"][i] = identical_abs
                        total_trades = int(len(trades_df))
                        if total_trades > 0:
                            results["identical_trades_absolut_percentage"][i] = float(
                                identical_abs / total_trades
                            )

        except Exception as exc:
            if error_counts is not None:
                key = type(exc).__name__
                error_counts[key] = error_counts.get(key, 0) + 1
            continue

    return results


def _compute_trade_overlap_vectorized(trades_df: pd.DataFrame) -> Dict[str, int]:
    """Berechnet max_trades_simult und long_short_overlap_episodes (OPTIMIERT: event-basiert).

    VORHER: O(n²) mit iterrows()-Schleife
    NACHHER: O(n log n) mit sortiertem Event-Array

    Args:
        trades_df: DataFrame mit Trades (entry_time, exit_time, direction)

    Returns:
        Dict mit max_simult und overlap_episodes
    """
    result = {"max_simult": 0, "overlap_episodes": 0}

    if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
        return result

    # Vectorized datetime conversion
    entry_times = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    exit_times = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

    # Filter valid trades
    valid_mask = entry_times.notna() & exit_times.notna() & (exit_times > entry_times)
    if not valid_mask.any():
        return result

    entry_times = entry_times[valid_mask]
    exit_times = exit_times[valid_mask]

    # Build event list: (timestamp, delta, direction_flag)
    # direction_flag: 0=any, 1=long, 2=short
    events = []

    if "direction" in trades_df.columns:
        directions = (
            trades_df.loc[valid_mask, "direction"].fillna("").astype(str).str.lower()
        )
        for et, xt, d in zip(entry_times, exit_times, directions):
            events.append((et, 1, 0))  # Entry (any)
            events.append((xt, -1, 0))  # Exit (any)

            if d == "long":
                events.append((et, 1, 1))  # Entry (long)
                events.append((xt, -1, 1))  # Exit (long)
            elif d == "short":
                events.append((et, 1, 2))  # Entry (short)
                events.append((xt, -1, 2))  # Exit (short)
    else:
        # Keine Direction-Info vorhanden
        for et, xt in zip(entry_times, exit_times):
            events.append((et, 1, 0))
            events.append((xt, -1, 0))

    if not events:
        return result

    # Sort events: (timestamp, delta, direction) → bei gleichem timestamp: +1 vor -1
    events.sort(key=lambda x: (x[0], -x[1]))

    # Compute max simultaneity
    active = 0
    max_active = 0
    for _, delta, dir_flag in events:
        if dir_flag == 0:  # Only count "any" direction for total simultaneity
            active += delta
            max_active = max(max_active, active)

    result["max_simult"] = max_active

    # Compute long/short overlap episodes
    active_long = 0
    active_short = 0
    overlap_episodes = 0

    for _, delta, dir_flag in events:
        had_overlap_before = active_long > 0 and active_short > 0

        if dir_flag == 1:  # Long
            active_long += delta
        elif dir_flag == 2:  # Short
            active_short += delta

        had_overlap_after = active_long > 0 and active_short > 0

        # New overlap episode started
        if not had_overlap_before and had_overlap_after:
            overlap_episodes += 1

    result["overlap_episodes"] = overlap_episodes
    return result


def _as_numeric_array(
    df: pd.DataFrame, column: str, fill_value: float = np.nan
) -> np.ndarray:
    """Return column as float64 numpy array, filling missing column with fill_value.

    Args:
        df: Source DataFrame.
        column: Column name to extract.
        fill_value: Value used if the column is missing.

    Returns:
        1-D numpy array of dtype float64.
    """
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), fill_value, dtype=float)


def _compute_category_stats_fast(matrix: pd.DataFrame) -> Dict[str, float]:
    """Compute reusable category statistics once (means, quantiles).

    This replaces repeated per-category scans with a single vectorized pass,
    substantially reducing wall-clock time for the categorical ranking step.
    """
    stats: Dict[str, float] = {}

    def _mean(col: str) -> float:
        vals = _as_numeric_array(matrix, col)
        # Guard against all-NaN arrays to prevent RuntimeWarning
        if len(vals) == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanmean(vals))

    def _std(col: str) -> float:
        vals = _as_numeric_array(matrix, col)
        # Guard against all-NaN arrays to prevent RuntimeWarning
        if len(vals) == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanstd(vals))

    def _q(col: str, q: float) -> float:
        vals = _as_numeric_array(matrix, col)
        if len(vals) == 0:
            return np.nan
        # Guard against all-NaN arrays: np.nanquantile will emit a RuntimeWarning
        # in that case ("All-NaN slice encountered"). Detect and short-circuit
        # to return np.nan while emitting a debug log so we can trace the origin.
        if np.all(np.isnan(vals)):
            # Silently return NaN - this is expected for optional metrics
            return np.nan
        return float(np.nanquantile(vals, q))

    # Means / stds
    stats["mean_total_trades"] = _mean("total_trades")
    stats["mean_winrate"] = _mean("winrate")
    stats["mean_equity_returns_skew"] = _mean("equity_returns_skew")
    stats["mean_equity_returns_kurtosis"] = _mean("equity_returns_kurtosis")
    stats["mean_equity_returns_volatility"] = _mean("equity_returns_volatility")
    # Return Shape (neu)
    stats["mean_equity_curvature"] = _mean("equity_curvature")
    stats["mean_equity_log_vol"] = _mean("equity_log_vol")
    stats["mean_same_trades_pct"] = _mean("identical_trades_absolut_percentage")
    stats["mean_max_trades_simult"] = _mean("max_trades_simult")
    stats["mean_long_short_overlap"] = _mean("long_short_overlap_episodes")
    stats["mean_volatility"] = _mean("equity_returns_volatility")
    stats["mean_profit"] = _mean("total_profit")

    stats["std_equity_returns_skew"] = _std("equity_returns_skew")
    stats["std_equity_returns_kurtosis"] = _std("equity_returns_kurtosis")

    # Quantiles
    stats["p30_max_dd"] = _q("total_max_dd", 0.30)
    stats["p30_worst_weekly"] = _q("worst_weekly_profit", 0.30)
    stats["p30_volatility"] = _q("equity_returns_volatility", 0.30)
    stats["p30_yearly_dispersion"] = _q("yearly_pnl_dispersion", 0.30)
    stats["p30_avg_duration"] = _q("average_trade_duration_hours", 0.30)
    # Gate für Return Shape (neu)
    stats["p30_time_in_highs"] = _q("time_in_highs", 0.30)
    stats["p70_total_trades"] = _q("total_trades", 0.70)
    stats["p70_volatility"] = _q("equity_returns_volatility", 0.70)
    stats["p70_fee_drag"] = _q("fee_drag", 0.70)
    stats["p40_winrate"] = _q("winrate", 0.40)

    return stats


def _build_category_score_matrix(
    matrix: pd.DataFrame, stats: Dict[str, float], category_order: List[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Vectorized computation of per-category scores.

    Each column j in the returned score matrix corresponds to category_order[j].
    Invalid candidates receive ``-np.inf`` so they are naturally excluded by
    the assignment step. All computations are fully vectorized (NumPy) to avoid
    repeated pandas scans and Python-level sorting per category.
    """
    n = len(matrix)
    if n == 0:
        return np.empty((0, len(category_order))), {}

    # Pre-fetch columns once
    arr: Dict[str, np.ndarray] = {
        col: _as_numeric_array(matrix, col)
        for col in [
            "final_score",
            "comp_score",
            "stability_score_monthly",
            "total_profit_over_dd",
            "duration_days",
            "time_in_market_hours",
            "total_max_dd",
            "worst_weekly_profit",
            "equity_returns_volatility",
            "ulcer_index_weekly",
            "yearly_pnl_dispersion",
            "total_trades",
            "sharpe_trade",
            "sortino_trade",
            "avg_r",
            "winrate",
            "average_trade_duration_hours",
            "fee_drag",
            "total_profit",
            "equity_returns_skew",
            "equity_returns_kurtosis",
            "equity_returns_autocorr",
            "identical_trades_absolut_percentage",
            "max_trades_simult",
            "long_short_overlap_episodes",
            # Return Shape (neu)
            "equity_curvature",
            "equity_log_vol",
            "dd_slope_stability",
            "time_in_highs",
        ]
    }

    scores: List[np.ndarray] = []

    # Helper: mark invalid as -inf
    def _mask_invalid(base: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        out = np.full_like(base, -np.inf)
        out[valid_mask] = base[valid_mask]
        return out

    # Category 1: Top Performer (final_score)
    scores.append(arr["final_score"])

    # Category 2: Capital Efficiency
    time_in_market_ratio = np.where(
        arr["duration_days"] > 0,
        arr["time_in_market_hours"] / (arr["duration_days"] * 24.0),
        np.nan,
    )
    cap_eff_score = np.where(
        time_in_market_ratio > 0,
        arr["total_profit_over_dd"] / time_in_market_ratio,
        np.nan,
    )
    scores.append(cap_eff_score)

    # Category 3: Defensive Low Drawdown
    mask_def = (
        (arr["total_max_dd"] <= stats["p30_max_dd"])
        & (arr["worst_weekly_profit"] <= stats["p30_worst_weekly"])
        & (arr["equity_returns_volatility"] <= stats["p30_volatility"])
    )
    defensive_score = _mask_invalid(-arr["ulcer_index_weekly"], mask_def)
    scores.append(defensive_score)

    # Category 4: Stable Compounder
    mask_stable = (arr["equity_returns_volatility"] <= stats["mean_volatility"]) & (
        arr["yearly_pnl_dispersion"] <= stats["p30_yearly_dispersion"]
    )
    stable_score = _mask_invalid(arr["stability_score_monthly"], mask_stable)
    scores.append(stable_score)

    # Category 5: Sharpe Trader
    mask_sharpe = arr["total_trades"] >= stats["mean_total_trades"]
    sharpe_core = np.sqrt(
        np.maximum(arr["sharpe_trade"], 0.0) * np.maximum(arr["sortino_trade"], 0.0)
    )
    sharpe_score = _mask_invalid(sharpe_core, mask_sharpe)
    scores.append(sharpe_score)

    # Category 6: High Conviction
    mask_conviction = (arr["total_trades"] <= stats["mean_total_trades"]) & (
        arr["winrate"] >= stats["mean_winrate"]
    )
    conviction_score = _mask_invalid(arr["avg_r"], mask_conviction)
    scores.append(conviction_score)

    # Category 7: High Turnover
    fee_threshold = stats.get("p70_fee_drag", np.nan)
    mask_turnover = (
        (arr["total_trades"] >= stats["p70_total_trades"])
        & (arr["average_trade_duration_hours"] <= stats["p30_avg_duration"])
        & (arr["winrate"] >= stats["p40_winrate"])
    )
    if not np.isnan(fee_threshold):
        mask_turnover &= arr["fee_drag"] <= fee_threshold
    turnover_score = np.where(
        mask_turnover,
        arr["sharpe_trade"] * np.sqrt(np.maximum(arr["total_trades"], 0.0)),
        -np.inf,
    )
    scores.append(turnover_score)

    # Category 8: Cost Efficient (lower fee_drag better)
    mask_cost = (arr["total_trades"] >= stats["mean_total_trades"]) & (
        arr["total_profit"] >= stats["mean_profit"]
    )
    cost_score = _mask_invalid(-arr["fee_drag"], mask_cost)
    scores.append(cost_score)

    # Category 9: Return Shape (neu)
    # Gate: Zeit im Hoch >= p30 der Matrix
    p30_highs = stats.get("p30_time_in_highs", np.nan)
    mask_return_shape = (
        np.isfinite(arr["time_in_highs"])
        & np.isfinite(p30_highs)
        & (arr["time_in_highs"] >= p30_highs)
    )

    mean_curv = stats.get("mean_equity_curvature", np.nan)
    mean_logvol = stats.get("mean_equity_log_vol", np.nan)

    # Robust normalisierte Gütemaße
    curv_good = 1.0 / (
        1.0
        + np.divide(
            arr["equity_curvature"],
            (mean_curv if np.isfinite(mean_curv) else 0.0) + 1e-9,
        )
    )
    logvol_good = 1.0 / (
        1.0
        + np.divide(
            arr["equity_log_vol"],
            (mean_logvol if np.isfinite(mean_logvol) else 0.0) + 1e-9,
        )
    )
    dd_good = np.where(
        np.isfinite(arr["dd_slope_stability"]), arr["dd_slope_stability"], 0.0
    )
    highs_good = np.where(np.isfinite(arr["time_in_highs"]), arr["time_in_highs"], 0.0)

    return_shape_score = (
        0.35 * curv_good + 0.25 * logvol_good + 0.20 * dd_good + 0.20 * highs_good
    )
    return_shape_score = _mask_invalid(return_shape_score, mask_return_shape)
    scores.append(return_shape_score)

    # Category 10: Independent (lower overlap is better -> negative score)
    same_norm = arr["identical_trades_absolut_percentage"] / (
        stats["mean_same_trades_pct"] + 1e-9
    )
    max_sim_norm = arr["max_trades_simult"] / (stats["mean_max_trades_simult"] + 1e-9)
    overlap_norm = arr["long_short_overlap_episodes"] / (
        stats["mean_long_short_overlap"] + 1e-9
    )
    independent_core = (same_norm + max_sim_norm + overlap_norm) / 3.0
    independent_score = -independent_core
    scores.append(independent_score)

    # Category 11: Diversifier
    equity_shape_cols = [
        "equity_returns_skew",
        "equity_returns_kurtosis",
        "equity_returns_autocorr",
        "equity_returns_volatility",
    ]
    available_cols = [c for c in equity_shape_cols if c in matrix.columns]
    if available_cols:
        X = np.column_stack([arr[c] for c in available_cols])
        X = np.where(np.isfinite(X), X, 0.0)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        sigma[sigma == 0] = 1.0
        Z = (X - mu) / sigma
        # Referenz: globaler Top Performer (höchster final_score)
        if np.all(~np.isfinite(arr["final_score"])):
            ref_idx = 0
        else:
            ref_idx = int(np.nanargmax(arr["final_score"]))
        ref_point = Z[ref_idx]
        dists = np.linalg.norm(Z - ref_point, axis=1)
        alpha = 0.05
        diversifier_score = dists + alpha * np.where(
            np.isfinite(arr["final_score"]), arr["final_score"], 0.0
        )
    else:
        # Fallback: reuse independence-style penalties but keep monotonicity
        diversifier_score = 0.3 * np.where(
            np.isfinite(arr["final_score"]), arr["final_score"], -np.inf
        )
        diversifier_score += (
            0.3 * (1 - same_norm) + 0.2 * (1 - max_sim_norm) + 0.2 * (1 - overlap_norm)
        )
    scores.append(diversifier_score)

    score_matrix = np.column_stack(scores)

    # Ensure strictly invalid entries are -inf (e.g., division produced nan)
    score_matrix[~np.isfinite(score_matrix)] = -np.inf

    tie_breakers = {
        "final_score": arr["final_score"],
        "comp_score": arr["comp_score"],
        "stability": arr["stability_score_monthly"],
    }
    return score_matrix, tie_breakers


def _build_rank_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Build a rank matrix for each category.

    For each column (category), compute ranks where:
      - 1 = best score in this category
      - larger integers = worse ranks
      - NaN = invalid / non-candidate (e.g., original score was -inf or NaN)

    The function is vectorized over categories and reuses ``_dense_rank_desc``
    (which ranks in descending order: higher score is better).
    """
    if score_matrix.size == 0:
        return np.empty_like(score_matrix)

    n_portfolios, n_cats = score_matrix.shape
    rank_matrix = np.full((n_portfolios, n_cats), np.nan, dtype=float)

    for cat_idx in tqdm(range(n_cats), desc="[Categorical] Rank-Matrix", unit="cat"):
        col = score_matrix[:, cat_idx]
        valid = np.isfinite(col)
        if not np.any(valid):
            continue

        dense_ranks = _dense_rank_desc(col[valid])
        rank_matrix[valid, cat_idx] = dense_ranks + 1.0  # 1-based ranks (1 = best)

    return rank_matrix


def _compute_top11_category_memberships(
    champions_df: pd.DataFrame,
    matrix: pd.DataFrame,
    rank_matrix: np.ndarray,
    category_order: List[str],
    id_col: str = "final_combo_pair_id",
) -> List[str]:
    """Return a "|"-joined list of categories where each champion ranks in the top 11.

    For each category in the top 11, include the rank (1-based) in parentheses.
    Format: "Category Name (rank)"

    Args:
        champions_df: Selected champions (must contain ``id_col`` and ``category``).
        matrix: Full candidate matrix used to build the score matrix (for index mapping).
        rank_matrix: Precomputed rank matrix aligned with ``matrix`` rows.
        category_order: Categories in column order of ``score_matrix`` / ``rank_matrix``.
        id_col: Identifier column used to map champions back to matrix rows.

    Returns:
        List of strings (pipe-separated category names with ranks) aligned with ``champions_df`` rows.
    """
    if len(matrix) != len(rank_matrix):
        raise ValueError("rank_matrix must align with matrix rows")

    # Build deterministic mapping from identifier to matrix row index
    if id_col in matrix.columns:
        id_series = matrix[id_col].reset_index(drop=True)
    else:
        id_series = pd.Series(np.arange(len(matrix)))

    id_to_idx: Dict[Any, int] = {}
    for idx, val in id_series.items():
        if pd.isna(val):
            continue
        if val not in id_to_idx:
            id_to_idx[val] = idx  # keep first occurrence deterministically

    categories_arr = np.array(category_order)
    results: List[str] = []

    for _, row in champions_df.iterrows():
        identifier = row.get(id_col) if id_col in row else None
        matrix_idx = id_to_idx.get(identifier)
        if matrix_idx is None:
            results.append("")
            continue

        ranks = rank_matrix[matrix_idx]
        mask = np.isfinite(ranks) & (ranks <= 11)
        cats = categories_arr[mask]
        cat_ranks = ranks[mask]

        # Format: "Category (rank)" where rank is 1-based
        cat_with_ranks = [
            f"{cat} ({int(rank)})"
            for cat, rank in zip(cats.tolist(), cat_ranks.tolist())
        ]
        results.append(" | ".join(cat_with_ranks))

    return results


def _dense_rank_desc(values: np.ndarray) -> np.ndarray:
    """Return 0-based ordinal ranks for descending values (nan/inf → worst).

    Args:
        values: 1-D array with numeric values.

    Returns:
        Ranks where 0 is best/highest value; ties are broken deterministically by original index.
    """
    finite = np.where(np.isfinite(values), values, -np.inf)
    order = np.lexsort((np.arange(len(finite)), -finite))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(finite), dtype=float)
    return ranks


def _build_category_rankings(
    score_matrix: np.ndarray,
    tie_breakers: Dict[str, np.ndarray],
    top_k: int = 128,
) -> List[List[int]]:
    """Build per-category rankings (descending score) with deterministic tie-breaks."""
    n_portfolios, n_cats = score_matrix.shape
    rankings: List[List[int]] = []

    for cat_idx in tqdm(
        range(n_cats), desc="[Categorical] Kategorie-Rankings", unit="cat"
    ):
        col = score_matrix[:, cat_idx]
        valid_mask = np.isfinite(col)
        if not np.any(valid_mask):
            rankings.append([])
            continue

        candidates = np.where(valid_mask)[0]

        scores = col[candidates]
        tie_final = tie_breakers["final_score"][candidates]
        tie_comp = tie_breakers["comp_score"][candidates]
        tie_stab = tie_breakers["stability"][candidates]

        order = np.lexsort(
            (
                candidates,
                -np.where(np.isfinite(tie_stab), tie_stab, -np.inf),
                -np.where(np.isfinite(tie_comp), tie_comp, -np.inf),
                -np.where(np.isfinite(tie_final), tie_final, -np.inf),
                -np.where(np.isfinite(scores), scores, -np.inf),
            )
        )

        ranked = candidates[order]
        if top_k > 0:
            ranked = ranked[:top_k]

        rankings.append([int(x) for x in ranked])

    return rankings


def _compute_category_distributions(
    score_matrix: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-category mean and std (std clipped; NaN if low variance/empty)."""
    if score_matrix.size == 0:
        return np.array([]), np.array([])

    n_cats = score_matrix.shape[1]
    mu = np.full(n_cats, np.nan, dtype=float)
    sigma = np.full(n_cats, np.nan, dtype=float)

    for cat_idx in range(n_cats):
        col = score_matrix[:, cat_idx]
        valid = col[np.isfinite(col)]
        if valid.size == 0:
            continue
        mu_val = float(np.mean(valid))
        sigma_val = float(np.std(valid))
        mu[cat_idx] = mu_val
        sigma[cat_idx] = sigma_val if sigma_val > eps else np.nan

    return mu, sigma


def _compute_margin_for_portfolio(
    portfolio_idx: int,
    category_idx: int,
    rankings: List[List[int]],
    score_matrix: np.ndarray,
) -> float:
    """Return raw score margin between champion and runner-up for a category."""
    scores = score_matrix[:, category_idx]
    champion_score = scores[portfolio_idx]
    if not np.isfinite(champion_score):
        return -np.inf

    runner_score = -np.inf
    for cand in rankings[category_idx]:
        if cand != portfolio_idx:
            runner_score = scores[cand]
            break

    if not np.isfinite(runner_score):
        return np.inf
    return champion_score - runner_score


def _choose_best_category_for_portfolio(
    portfolio_idx: int,
    categories: List[int],
    rankings: List[List[int]],
    score_matrix: np.ndarray,
    tie_breakers: Dict[str, np.ndarray],
    sigma_per_category: np.ndarray,
) -> int:
    """Select category with maximum margin (Model B) for a conflicted portfolio."""
    best_cat = categories[0]
    best_key = None

    for cat_idx in categories:
        margin_raw = _compute_margin_for_portfolio(
            portfolio_idx, cat_idx, rankings, score_matrix
        )
        sigma_c = (
            sigma_per_category[cat_idx] if sigma_per_category.size > cat_idx else np.nan
        )
        if np.isfinite(sigma_c):
            margin_z = margin_raw / sigma_c if np.isfinite(margin_raw) else margin_raw
        else:
            margin_z = 0.0  # low variance or no valid std -> neutral margin

        score = score_matrix[portfolio_idx, cat_idx]
        key = (
            margin_z,
            np.where(np.isfinite(score), score, -np.inf),
            np.where(
                np.isfinite(tie_breakers["final_score"][portfolio_idx]),
                tie_breakers["final_score"][portfolio_idx],
                -np.inf,
            ),
            np.where(
                np.isfinite(tie_breakers["comp_score"][portfolio_idx]),
                tie_breakers["comp_score"][portfolio_idx],
                -np.inf,
            ),
            np.where(
                np.isfinite(tie_breakers["stability"][portfolio_idx]),
                tie_breakers["stability"][portfolio_idx],
                -np.inf,
            ),
            -cat_idx,  # deterministic preference for earlier categories
        )

        if best_key is None or key > best_key:
            best_key = key
            best_cat = cat_idx

    return best_cat


def _find_next_candidate_for_category(
    category_idx: int,
    rankings: List[List[int]],
    used_portfolios: Set[int],
) -> Optional[int]:
    """Return next available portfolio for category respecting used set."""
    for cand in rankings[category_idx]:
        if cand in used_portfolios:
            continue
        return int(cand)
    return None


def _resolve_conflicts_model_b(
    champions: List[Optional[int]],
    rankings: List[List[int]],
    score_matrix: np.ndarray,
    tie_breakers: Dict[str, np.ndarray],
    sigma_per_category: np.ndarray,
) -> List[Optional[int]]:
    """Ensure each portfolio wins at most one category using margin-based resolution."""
    while True:
        categories_by_portfolio: Dict[int, List[int]] = defaultdict(list)
        for cat_idx, p in enumerate(champions):
            if p is not None:
                categories_by_portfolio[p].append(cat_idx)

        conflicted = [p for p, cats in categories_by_portfolio.items() if len(cats) > 1]
        if not conflicted:
            break

        for portfolio_idx in conflicted:
            cats = categories_by_portfolio[portfolio_idx]
            best_cat = _choose_best_category_for_portfolio(
                portfolio_idx,
                cats,
                rankings,
                score_matrix,
                tie_breakers,
                sigma_per_category,
            )

            for cat_idx in cats:
                if cat_idx == best_cat:
                    continue

                champions[cat_idx] = None
                used_now = {c for c in champions if c is not None}
                replacement = _find_next_candidate_for_category(
                    cat_idx, rankings, used_now
                )
                champions[cat_idx] = replacement

    return champions


def _apply_category_fallbacks(
    champions: List[Optional[int]],
    rankings: List[List[int]],
    score_matrix: np.ndarray,
    tie_breakers: Dict[str, np.ndarray],
) -> List[Optional[int]]:
    """Fill categories without champions using global deterministic ordering."""
    n_portfolios, n_cats = score_matrix.shape
    global_order = np.lexsort(
        (
            np.arange(n_portfolios),
            -np.where(
                np.isfinite(tie_breakers["stability"]),
                tie_breakers["stability"],
                -np.inf,
            ),
            -np.where(
                np.isfinite(tie_breakers["comp_score"]),
                tie_breakers["comp_score"],
                -np.inf,
            ),
            -np.where(
                np.isfinite(tie_breakers["final_score"]),
                tie_breakers["final_score"],
                -np.inf,
            ),
        )
    )

    used = {c for c in champions if c is not None}

    for cat_idx in range(n_cats):
        if champions[cat_idx] is not None:
            continue

        # Prefer category-local ranking
        replacement = _find_next_candidate_for_category(cat_idx, rankings, used)
        if replacement is not None and np.isfinite(score_matrix[replacement, cat_idx]):
            champions[cat_idx] = replacement
            used.add(replacement)
            continue

        # Fallback to global deterministic order
        for cand in global_order:
            if cand in used:
                continue
            if np.isfinite(score_matrix[cand, cat_idx]):
                champions[cat_idx] = int(cand)
                used.add(int(cand))
                break

    return champions


def _select_champions_model_b(
    score_matrix: np.ndarray,
    category_order: List[str],
    matrix: pd.DataFrame,
    tie_breakers: Dict[str, np.ndarray],
    top_k: int = 128,
) -> pd.DataFrame:
    """Assign champions per category using margin-based conflict resolution (Model B)."""
    n_portfolios, n_cats = score_matrix.shape
    rankings = _build_category_rankings(score_matrix, tie_breakers, top_k=top_k)
    _, sigma_per_category = _compute_category_distributions(score_matrix)

    champions: List[Optional[int]] = []
    for cat_idx in range(n_cats):
        champions.append(rankings[cat_idx][0] if rankings[cat_idx] else None)

    champions = _resolve_conflicts_model_b(
        champions, rankings, score_matrix, tie_breakers, sigma_per_category
    )
    champions = _apply_category_fallbacks(
        champions, rankings, score_matrix, tie_breakers
    )

    selections: List[Dict[str, Any]] = []
    for cat_idx, champ_idx in enumerate(champions):
        if champ_idx is None:
            continue
        row = matrix.iloc[champ_idx].to_dict()
        row["category"] = category_order[cat_idx]
        selections.append(row)

    return pd.DataFrame(selections)


def _compute_category_champions(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Implementiert das kategorische Ranking-System mit 11 Kategorien.

    Jede Kategorie hat eigene Filter und Ranking-Kriterien.
    Ein Portfolio darf maximal einmal als Champion erscheinen (keine Duplikate).
    Die Auswahl folgt Modell B: pro Kategorie wird der interne Champion gewählt,
    Konflikte werden margin-basiert aufgelöst (größter Abstand zum Runner-Up),
    anschließend Fallback über globale Rangfolge für leere Kategorien.

    Returns:
        DataFrame mit Champions (max. 11 Zeilen, eine pro Kategorie)
    """
    if matrix.empty:
        print("[Categorical] Matrix ist leer, keine Champions")
        return pd.DataFrame()

    print("[Categorical] Starte kategorisches Ranking...")

    # Ensure all needed metrics exist
    matrix = _compute_additional_categorical_metrics(matrix)

    category_order = [
        "Top Performer",
        "Capital Efficiency",
        "Defensive Low Drawdown",
        "Stable Compounder",
        "Sharpe Trader",
        "High Conviction",
        "High Turnover",
        "Cost Efficient",
        "Return Shape",
        "Independent",
        "Diversifier",
    ]

    print(f"[Categorical] Berechne Kategorie-Statistiken...")
    stats = _compute_category_stats_fast(matrix)

    print(f"\n[Categorical] VOR Score-Matrix-Berechnung: {len(matrix)} Portfolios")
    print(f"[Categorical] NaN-Verteilung (kritische Spalten):")
    critical_cols = [
        "final_score",
        "comp_score",
        "worst_weekly_profit",
        "ulcer_index_weekly",
        "yearly_pnl_dispersion",
        "equity_returns_volatility",
    ]
    for col in critical_cols:
        if col in matrix.columns:
            nan_count = matrix[col].isna().sum()
            print(f"  - {col}: {nan_count} NaN ({nan_count/len(matrix)*100:.1f}%)")

    print(
        f"[Categorical] Erstelle Score-Matrix für {len(category_order)} Kategorien..."
    )
    score_matrix, tie_breakers = _build_category_score_matrix(
        matrix, stats, category_order
    )

    print(
        f"[Categorical] Score-Matrix: {score_matrix.shape[0]} Portfolios × {score_matrix.shape[1]} Kategorien"
    )
    # Count portfolios with at least one valid (non -inf) score
    valid_per_portfolio = np.sum(score_matrix > -np.inf, axis=1)
    fully_invalid = np.sum(valid_per_portfolio == 0)
    print(f"[Categorical]   - Vollständig ungültig: {fully_invalid}")
    print(
        f"[Categorical]   - Gültig (mind. 1 Kategorie): {len(matrix) - fully_invalid}"
    )

    print(f"[Categorical] Erstelle Rank-Matrix...")
    rank_matrix = _build_rank_matrix(score_matrix)

    print(f"[Categorical] Wähle Champions (Modell B: margin-basiert)...")
    champions_df = _select_champions_model_b(
        score_matrix, category_order, matrix, tie_breakers
    )

    if champions_df.empty:
        print("[Categorical] Keine Champions gefunden")
        return champions_df

    print(f"[Categorical] Berechne Top-11 Kategorie-Mitgliedschaften...")
    top11_categories = _compute_top11_category_memberships(
        champions_df=champions_df,
        matrix=matrix.reset_index(drop=True),
        rank_matrix=rank_matrix,
        category_order=category_order,
    )
    insert_pos = (
        champions_df.columns.get_loc("category") + 1
        if "category" in champions_df.columns
        else len(champions_df.columns)
    )
    champions_df.insert(insert_pos, "top11_categories", top11_categories)

    print(
        f"[Categorical] ✓ {len(champions_df)} Champions aus {len(category_order)} Kategorien ausgewählt"
    )
    return champions_df


def create_categorical_champion_list(
    matrix: pd.DataFrame, output_path: Path, groups: List[WalkforwardGroup]
) -> None:
    """
    Erstellt die kategorische Champion-Liste (ersetzt die bisherige Top-10-Liste).

    Format identisch zu top_50_walkforward_combos.csv aus walkforward_analyzer.py:
    - Jede combo_pair_id hat multiple Zeilen (eine pro Leg/group_id)
    - Erste Spalte: combo_pair_id (final_combo_pair_id)
    - Zweite Spalte: group_id (identifiziert das Leg)
    - Zusätzlich: category-Spalte identifiziert die Champion-Kategorie

    Args:
        matrix: Matrix mit bereits ausgewählten Champions (bereits von _compute_category_champions gefiltert)
                Muss die Spalte 'category' enthalten
        output_path: Pfad zur Output-CSV
        groups: Liste der WalkforwardGroup-Objekte
    """
    if matrix.empty:
        print("[Champions] Matrix ist leer, keine Champion-Liste erstellt")
        return

    # Die Champions wurden bereits in main() durch _compute_category_champions ausgewählt
    # Hier verwenden wir die bereits gefilterte Matrix direkt (keine erneute Filterung!)
    champions = matrix.copy()

    # Sicherstellen, dass category-Spalte vorhanden ist
    if "category" not in champions.columns:
        print("[Champions] Warnung: category-Spalte fehlt in der Matrix")
        champions["category"] = "N/A"

    if champions.empty:
        print("[Champions] Keine Champions gefunden")
        return

    # Expandiere in Leg-basiertes Format
    expanded_df = _expand_final_combos_for_display(champions, groups)

    if expanded_df.empty:
        print("[Champions] Warnung: Expandiertes DataFrame ist leer")
        return

    # Unerwünschte Score-Spalten für die finale Liste entfernen
    cols_to_drop = [
        "comp_score_final",
        "comp_score_combined",
        "stability_score_combined",
        "_equity_internal",
        "_trades_internal",
        # Temporäre Scoring-Spalten
        "time_in_market_ratio",
        "score_capital_efficiency",
        "sharpe_score",
        "score_high_turnover",
        "skew_norm",
        "kurtosis_norm",
        "autocorr_penalty",
        "volatility_penalty",
        "score_return_shape",
        "same_trades_norm",
        "max_simult_norm",
        "overlap_norm",
        "score_independent",
        "score_diversifier",
    ]
    existing_drop = [c for c in cols_to_drop if c in expanded_df.columns]
    if existing_drop:
        expanded_df = expanded_df.drop(columns=existing_drop)

    # Speichern
    expanded_df.to_csv(output_path, index=False)
    print(f"[Champions] Kategorische Champion-Liste gespeichert: {output_path}")
    print(f"[Champions] Format: {len(expanded_df)} Zeilen (inkl. Trennzeilen)")

    # Zusätzlicher Export: Portfolio-1-Zeiler (nur identische Spalten über alle Legs)
    try:
        one_liner_df = _collapse_champions_combo_pairs_to_one_liners(expanded_df)
        if not one_liner_df.empty:
            one_liner_df.to_csv(
                CATEGORICAL_CHAMPIONS_PORTFOLIOS_ONE_LINER_CSV, index=False
            )
            print(
                f"[Champions] Portfolio-1-Zeiler gespeichert: {CATEGORICAL_CHAMPIONS_PORTFOLIOS_ONE_LINER_CSV}"
            )
    except Exception as e:
        print(f"[Champions] Warnung: Konnte Portfolio-1-Zeiler nicht erstellen: {e}")

    # Übersicht ausgeben (pro finaler Kombination)
    print("\n" + "=" * 80)
    print("KATEGORISCHE CHAMPION-LISTE")
    print("=" * 80)

    # Gruppiere nach combo_pair_id für Übersichtsausgabe
    combo_groups = {}
    for _, row in expanded_df.iterrows():
        cpid = row.get("combo_pair_id")
        if pd.notna(cpid):
            if cpid not in combo_groups:
                combo_groups[cpid] = []
            combo_groups[cpid].append(row)

    for i, (combo_id, combo_rows) in enumerate(combo_groups.items(), 1):
        first_row = combo_rows[0]
        category = first_row.get("category", "N/A")

        print(f"\n{i}. {combo_id} [{category}]")
        print(f"   Final Score: {first_row.get('final_score', np.nan):.5f}")
        print(f"   Comp Score:  {first_row.get('comp_score', np.nan):.5f}")
        print(f"   Stability:   {first_row.get('stability_score_monthly', np.nan):.5f}")
        print(f"   Robustness:  {first_row.get('robustness_mean', np.nan):.5f}")
        print(f"   Profit:      {first_row.get('total_profit', np.nan):.2f}")
        print(f"   Max DD:      {first_row.get('total_max_dd', np.nan):.2f}")
        print(f"   Sharpe (Eq): {first_row.get('sharpe_equity', np.nan):.4f}")
        print(f"   Sortino (Eq):{first_row.get('sortino_equity', np.nan):.4f}")
        print(f"   Sharpe (Tr): {first_row.get('sharpe_trade', np.nan):.4f}")
        print(f"   Sortino (Tr):{first_row.get('sortino_trade', np.nan):.4f}")
        print(f"   Avg R:       {first_row.get('avg_r', np.nan):.4f}")
        print(f"   Winrate:     {first_row.get('winrate', np.nan):.2f}%")
        print(f"   Trades:      {first_row.get('total_trades', 0):.0f}")
        print(
            f"   Legs: {', '.join([str(r.get('group_id', 'N/A')) for r in combo_rows])}"
        )

    print("\n" + "=" * 80 + "\n")


def create_top10_list(
    matrix: pd.DataFrame, output_path: Path, groups: List[WalkforwardGroup]
) -> None:
    """
    DEPRECATED: Diese Funktion wird durch create_categorical_champion_list ersetzt.
    Wird noch aufgerufen für Legacy-Kompatibilität, leitet aber an neue Funktion weiter.

    Args:
        matrix: Sortierte Matrix mit final_score
        output_path: Pfad zur Output-CSV
        groups: Liste der WalkforwardGroup-Objekte
    """
    print("[Top10] DEPRECATED: Verwende kategorisches Ranking statt Top-10")
    create_categorical_champion_list(matrix, output_path, groups)


def persist_final_combo_artifacts(
    df: pd.DataFrame,
    *,
    index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    """Persistiert equity.csv und trades.json für die übergebenen finalen Kombinationen.

    Hintergrund:
    - Im Monte-Carlo-Pfad wird Equity intern auf einem gemeinsamen Daily-Grid berechnet.
      Dieses Grid kann (absichtlich) eine konservative Schnittmenge verwenden und dadurch
      Artefakte wie "Equity endet zu früh" erzeugen, obwohl Trades später weiterlaufen.
    - Für die finalen Artefakte (v.a. Champions) wollen wir jedoch eine vollständige
      Equity-Kurve basierend auf den Original-Equity-Serien (Union der Timestamps).

    Wenn ``index`` übergeben wird (group_id -> Kandidaten mit equity_path/trades_path),
    rekonstruieren wir deshalb die Artefakte per ``aggregate_final_combo`` und schreiben
    daraus equity/trades. Falls Rekonstruktion nicht möglich ist, fällt die Funktion
    auf den Legacy-Persist-Mechanismus (aus ``_equity_internal`` / ``_trades_internal``)
    zurück.

    Erwartete Spalten (Legacy-Fallback):
    - final_combo_pair_id
    - _equity_internal (pd.Series oder Pfad)
    - _trades_internal (pd.DataFrame | list[dict] | Pfad)
    - groups_mapping_json (für Rekonstruktion; empfohlen)
    """
    if df is None or df.empty:
        return

    entry_lookup: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    if index:
        entry_lookup = {
            str(gid): {str(e.get("combo_pair_id", "")): e for e in entries}
            for gid, entries in index.items()
        }

    for _, row in df.iterrows():
        final_id = str(row.get("final_combo_pair_id", "")).strip()
        if not final_id:
            continue

        # Preferred path: rebuild from selection paths (full union timeline)
        if (
            entry_lookup is not None
            and "groups_mapping_json" in row
            and pd.notna(row.get("groups_mapping_json"))
        ):
            try:
                mapping = json.loads(str(row.get("groups_mapping_json")))
                if isinstance(mapping, dict) and mapping:
                    selection: Dict[str, Dict[str, Any]] = {}
                    for gid, cpid in mapping.items():
                        entry = entry_lookup.get(str(gid), {}).get(str(cpid))
                        if entry is not None:
                            selection[str(gid)] = entry

                    if selection and len(selection) == len(mapping):
                        # aggregate_final_combo writes to COMBINED_MATRIX_DIR/final_combos/<final_id>/
                        # using the same deterministic hashing semantics.
                        aggregate_final_combo(selection, write_files=True)
                        continue
            except Exception as e:
                print(
                    f"[Persist] Warnung: Rekonstruktion via index fehlgeschlagen für {final_id}: {e}"
                )

        # Fallback: persist internal objects as-is
        target_dir = COMBINED_MATRIX_DIR / "final_combos" / final_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Equity schreiben
        eq_obj = row.get("_equity_internal")
        try:
            if isinstance(eq_obj, pd.Series) and not eq_obj.empty:
                df_out = eq_obj.sort_index().reset_index()
                df_out.columns = ["timestamp", "equity"]
                (target_dir / "equity.csv").write_text(
                    df_out.to_csv(index=False), encoding="utf-8"
                )
            elif isinstance(eq_obj, (str, Path)):
                p = Path(eq_obj)
                if p.exists():
                    shutil.copy2(p, target_dir / "equity.csv")
        except Exception as e:
            print(
                f"[Persist] Warnung: Konnte equity für {final_id} nicht schreiben: {e}"
            )

        # Trades schreiben
        tr_obj = row.get("_trades_internal")
        try:
            if isinstance(tr_obj, pd.DataFrame) and not tr_obj.empty:
                df_tr = tr_obj.copy()
                if "entry_time" in df_tr.columns:
                    df_tr["entry_time"] = pd.to_datetime(
                        df_tr["entry_time"], utc=True, errors="coerce"
                    )
                    df_tr = df_tr.sort_values("entry_time")
                records = df_tr.to_dict("records")
                # Timestamps formatieren
                for rec in records:
                    for key in ("entry_time", "exit_time"):
                        if key in rec and pd.notna(rec[key]):
                            try:
                                ts = pd.to_datetime(rec[key], utc=True)
                                rec[key] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                            except Exception:
                                pass
                with (target_dir / "trades.json").open("w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            elif isinstance(tr_obj, list) and tr_obj:
                # Already a list[dict] (e.g. hydrated without DataFrame to reduce overhead)
                records = tr_obj
                # Best-effort sort by entry_time (if present)
                try:

                    def _key(rec: Dict[str, Any]) -> str:
                        v = rec.get("entry_time")
                        return str(v) if v is not None else ""

                    records = sorted(records, key=_key)
                except Exception:
                    pass
                with (target_dir / "trades.json").open("w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            elif isinstance(tr_obj, (str, Path)):
                p = Path(tr_obj)
                if p.exists():
                    shutil.copy2(p, target_dir / "trades.json")
        except Exception as e:
            print(
                f"[Persist] Warnung: Konnte trades für {final_id} nicht schreiben: {e}"
            )


def main():
    # OPTIMIERT: Setze 'spawn' als start method für bessere macOS-Kompatibilität
    # Dies verhindert Copy-on-Write Issues und verbessert Parallelisierung
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Bereits gesetzt, ignorieren
        pass

    parser = argparse.ArgumentParser(
        description="Kombinierte Walkforward-Matrix-Analyse über alle Symbol x Timeframe x Richtung Kombinationen"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(WALKFORWARD_ROOT),
        help="Root-Verzeichnis der Walkforward-Ergebnisse",
    )
    parser.add_argument(
        "--skip-analyzer",
        action="store_true",
        help="Überspringe die Ausführung von walkforward_analyzer (verwende vorhandene Ergebnisse)",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Überspringe die Aggregation von Equity/Trades (verwende vorhandene Dateien)",
    )
    parser.add_argument(
        "--max-final-combos",
        type=int,
        default=0,
        help="Maximale Anzahl finaler Kombinationen (0 = unbegrenzt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch-Größe für finale Kombinationen (Anzahl Selections pro Batch).",
    )
    # New search pipeline controls
    parser.add_argument(
        "--disable-pareto",
        action="store_true",
        help="Pareto-Filter pro group_id deaktivieren",
    )
    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        help="Clustering pro group_id deaktivieren",
    )
    parser.add_argument(
        "--target-cluster-size",
        type=int,
        default=4,
        help="Zielgröße pro Cluster (je kleiner, desto mehr Cluster)",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=3,
        help="Maximale Anzahl Combos pro group_id nach Clustering (K)",
    )
    parser.add_argument(
        "--min-diversifier-score",
        type=float,
        default=0.0,
        help="Mindestscore für Diversifier-Kandidaten",
    )
    # Monte Carlo Search
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=10000,
        help="Anzahl Portfolios für Monte Carlo Search (Default: 10000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Anzahl paralleler Prozesse für Monte Carlo Search (None = cpu_count())",
    )
    parser.add_argument(
        "--development-seed",
        type=int,
        default=None,
        help="Seed für reproducible Entwicklung (None = no seed, random)",
    )
    parser.add_argument(
        "--enable-dev-mode",
        action="store_true",
        help="Aktiviere Development Mode (nutze Seed für reproducible Ergebnisse)",
    )
    parser.add_argument(
        "--monte-carlo-window",
        type=str,
        choices=["union", "intersection"],
        default="union",
        help=(
            "Bewertungsfenster für Monte Carlo (Daily EOD UTC). "
            "union=voller Zeitraum (Default, empfohlen für Kapitalallokation), "
            "intersection=konservative Schnittmenge (legacy)."
        ),
    )
    parser.add_argument(
        "--mc-grid",
        type=str,
        choices=["events", "daily"],
        default="events",
        help=(
            "Zeitgitter für Monte Carlo Bewertung. events=Union der Equity/Exit-Zeitpunkte (Default, korrekt), "
            "daily=tägliche EOD-Snapshots (schneller, kann intraday DD verfälschen)."
        ),
    )
    parser.add_argument(
        "--mc-trade-equity-tolerance-seconds",
        type=float,
        default=60.0,
        help=(
            "Toleranz (Sekunden) für das Matching: Exit-Time der Trades muss in equity.csv vorhanden sein. "
            "Default 60s."
        ),
    )
    parser.add_argument(
        "--mc-skip-trade-equity-mismatch",
        action="store_true",
        help=(
            "NICHT empfohlen: Bei Trade↔Equity-Mismatch wird nicht abgebrochen, sondern der Kandidat invalidiert "
            "(final_score=-inf). Default: Abbruch für korrekte Daten."
        ),
    )

    args = parser.parse_args()
    root = Path(args.root)

    print("\n" + "=" * 80)
    print("KOMBINIERTE WALKFORWARD MATRIX ANALYZER")
    print("=" * 80 + "\n")

    # Schritt 1: Walkforward-Ordner gruppieren
    print("[Schritt 1/8] Walkforward-Ergebnisse gruppieren...")
    groups = discover_walkforward_groups(root)

    if not groups:
        print("Keine Walkforward-Gruppen gefunden. Abbruch.")
        return

    # Schritt 2: Pro Kombination walkforward_analyzer ausführen
    if not args.skip_analyzer:
        print("\n[Schritt 2/8] Einzel-Analysen ausführen (walkforward_analyzer)...")
        for group in groups:
            run_analyzer_for_group(group)
    else:
        print("\n[Schritt 2/8] Übersprungen (--skip-analyzer)")

    # Schritt 3: Equity und Trades für jede combo_pair_id aggregieren
    if not args.skip_aggregate:
        print("\n[Schritt 3/8] Daten-Aggregation (Equity & Trades)...")
        for group in groups:
            aggregate_all_combo_pairs_for_group(group)
    else:
        print("\n[Schritt 3/8] Übersprungen (--skip-aggregate)")

    # Schritt 4: Kombinierte Matrix aufbauen
    print("\n[Schritt 4/8] Matrix-Aufbau...")
    matrix = build_combined_matrix(groups)

    if matrix.empty:
        print("Matrix ist leer. Abbruch.")
        return

    # Matrix zwischenspeichern
    matrix_raw_path = COMBINED_MATRIX_DIR / "matrix_raw.csv"
    matrix.to_csv(matrix_raw_path, index=False)
    print(f"[Matrix] Rohe Matrix gespeichert: {matrix_raw_path}")

    # Schritt 5: Metriken + Scores je Matrixzeile berechnen
    print("\n[Schritt 5/8] Metrik-Berechnung (Global Scores)...")
    matrix = compute_global_metrics(matrix)
    matrix = compute_additional_scores(matrix)
    matrix = compute_final_score(matrix)

    # Schritt 6: Vorverarbeitung (Pareto, Clustering, Pruning)
    print("\n[Schritt 6/8] Vorverarbeitung (Pareto, Clustering, Pruning)...")

    # 6a: Pareto
    if not args.disable_pareto:
        print("  > Pareto-Filter pro group_id…")
        matrix = apply_pareto_filter_per_group(matrix)
        # Persist matrix after Pareto
        matrix_after_pareto_path = COMBINED_MATRIX_DIR / "matrix_after_pareto.csv"
        matrix.to_csv(matrix_after_pareto_path, index=False)
        print(f"  [Matrix] Nach Pareto gespeichert: {matrix_after_pareto_path}")
    else:
        print("  > Pareto-Filter deaktiviert (--disable-pareto)")

    # 6b: Clustering
    if not args.disable_clustering:
        print("  > Clustering pro group_id…")
        matrix = apply_clustering_per_group(
            matrix, target_cluster_size=max(2, args.target_cluster_size)
        )
    else:
        print("  > Clustering deaktiviert (--disable-clustering)")

    # Persist matrix after clustering (final preprocessed stage)
    matrix_after_clustering_path = COMBINED_MATRIX_DIR / "matrix_after_clustering.csv"
    matrix.to_csv(matrix_after_clustering_path, index=False)
    print(f"  [Matrix] Nach Clustering gespeichert: {matrix_after_clustering_path}")

    # 6c: Top-K Pruning
    print(f"  > Prune zu Top-K diverse pro group_id (K={args.max_per_group})…")
    matrix = prune_to_topK_diverse_per_group(
        matrix,
        K=max(1, args.max_per_group),
        min_diversifier_score=args.min_diversifier_score,
    )

    # Persist matrix after pruning
    matrix_after_pruning_path = COMBINED_MATRIX_DIR / "matrix_after_topK_pruning.csv"
    matrix.to_csv(matrix_after_pruning_path, index=False)
    print(f"  [Matrix] Nach Top-K Pruning gespeichert: {matrix_after_pruning_path}")

    # Schritt 7: Monte Carlo Portfolio Search
    print(
        "\n[Schritt 7/8] Monte Carlo Portfolio Search (parallele vollständige Evaluation)…"
    )
    index = build_final_combo_index(matrix)
    if not index:
        print("Keine Einträge für finale Kombinationen gefunden. Abbruch.")
        return

    # Prepare Monte-Carlo evaluator state (kompakt, spawn-safe, ohne riesige IPC-Objekte)
    # WICHTIG: Standard ist UNION-Fenster + strict end-coverage, um vollständige Korrektheit
    # der Bewertung (bis zum Ende) zu erzwingen.
    eval_state = prepare_monte_carlo_eval_state(
        index,
        window_mode=args.monte_carlo_window,
        grid=args.mc_grid,
        trade_equity_tolerance_seconds=float(args.mc_trade_equity_tolerance_seconds),
        fail_on_trade_equity_mismatch=(not args.mc_skip_trade_equity_mismatch),
    )

    # ✅ Berechne adaptive Samples: min(N_total, default * num_groups)
    adaptive_samples = compute_adaptive_level1_samples(
        index=index, default_level1_samples=args.monte_carlo_samples
    )
    print(f"[Monte Carlo] Adaptive Samples: {adaptive_samples} (N_groups={len(index)})")

    # Monte Carlo Portfolio Search: Vollständige parallele Evaluation
    t0 = time.time()
    n_jobs = args.n_jobs if args.n_jobs else cpu_count()

    # Seed-Logik für Entwicklung
    dev_seed = (
        args.development_seed if args.development_seed is not None else DEVELOPMENT_SEED
    )

    # OPTIMIERT: Nutze Batch-Processing und Shared Memory
    exact_seeds = monte_carlo_portfolio_search(
        index=index,
        matrix=matrix,
        eval_state=eval_state,
        num_samples=adaptive_samples,
        n_jobs=n_jobs,
        rng_seed=dev_seed,
        dev_mode=args.enable_dev_mode,
        batch_size=1000,  # 1000 Portfolios pro Batch (optimaler Wert für Multi-Core)
        use_batch_mode=True,  # Aktiviere Batch Processing
    )

    if exact_seeds.empty:
        print("[FinalCombos] Keine Seeds nach Monte Carlo Search gefunden. Abbruch.")
        return
    print(
        f"[Monte Carlo] Exakt evaluierte Portfolios: {len(exact_seeds)} (t={time.time()-t0:.1f}s)"
    )

    # ✅ Monte Carlo Search ist der finale Optimierungsschritt
    # Dedupliziere und bereite alle evaluierten Portfolios vor
    global_top_df = exact_seeds.copy()

    # Count duplicates before dedup
    n_before_dedup = len(global_top_df)
    unique_combo_ids = global_top_df["final_combo_pair_id"].nunique()

    # Deduplicate and keep best
    global_top_df = (
        global_top_df.sort_values("final_score", ascending=False)
        .drop_duplicates(subset=["final_combo_pair_id"], keep="first")
        .reset_index(drop=True)
    )
    total_evaluated = len(global_top_df)

    # Log deduplication statistics
    n_duplicates = n_before_dedup - total_evaluated
    if n_duplicates > 0:
        print(f"[Monte Carlo] NACHTRÄGL. DEDUPLIZIERUNG (unique_only war deaktiviert):")
        print(
            f"[Monte Carlo]   {n_before_dedup} Evaluationen → {total_evaluated} eindeutige Kombinationen"
        )
        print(
            f"[Monte Carlo]   Removed: {n_duplicates} Duplikate ({n_duplicates/n_before_dedup*100:.1f}%)"
        )
    else:
        print(
            f"[Monte Carlo] ✓ KEINE DUPLIKATE (unique_only=True hat Duplikate bei Sampling eliminiert)"
        )
        print(f"[Monte Carlo]   Evaluierte eindeutige Kombinationen: {total_evaluated}")
    print()

    # Hydrate Equity/Trades erst NACH Dedup (verhindert riesige Serialisierung während der Suche)
    global_top_df = hydrate_portfolio_artifacts_for_categorical(
        global_top_df,
        eval_state=eval_state,
        index=index,
        chunk_size=256,
    )

    # Add trade-count adjusted total metrics (portfolio level) for CSV outputs
    global_top_df = _add_total_adust_metrics_to_portfolios(global_top_df)

    # Schritt 8: Kategorisches Ranking & Finalisierung
    print("\n[Schritt 8/8] Kategorisches Ranking & Finalisierung...")
    champions_df = _compute_category_champions(global_top_df)

    if champions_df.empty:
        print("[Champions] Keine Champions gefunden. Abbruch.")
        return

    # Persistiere Artefakte (equity.csv, trades.json) für Champions
    # Rekonstruiert (wenn möglich) aus den Original-Pfaden via ``index``, um eine
    # vollständige Equity-Kurve (Union der Timestamps) zu speichern.
    persist_final_combo_artifacts(champions_df, index=index)

    # Entferne interne Spalten vor dem Speichern der CSV
    cols_to_drop = ["_equity_internal", "_trades_internal"]
    champions_df_clean = champions_df.drop(
        columns=[c for c in cols_to_drop if c in champions_df.columns]
    )

    # Finale Ergebnisse speichern
    matrix_full_path = COMBINED_MATRIX_DIR / "final_combos_categorical_champions.csv"
    champions_df_clean.to_csv(matrix_full_path, index=False)
    print(f"[Matrix] Finale kategorische Champions gespeichert: {matrix_full_path}")

    # Champions formatiert ausgeben
    print("\n[Kategorische Champions - Finale Kombinationen]")
    create_categorical_champion_list(
        champions_df.rename(columns={"final_combo_pair_id": "combo_pair_id"}),
        TOP10_FINAL_COMBOS_CSV,
        groups,
    )

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)
    print(f"\nErgebnisse:")
    print(f"  - Kategorische Champions: {TOP10_FINAL_COMBOS_CSV}")
    print(
        f"  - Champions (1-Zeiler):   {CATEGORICAL_CHAMPIONS_PORTFOLIOS_ONE_LINER_CSV}"
    )
    print(f"  - Champions-Matrix:       {matrix_full_path}")
    print(f"  - Gruppen-Ergebnisse:     {COMBINED_MATRIX_DIR}")
    print(f"  - Anzahl Champions:       {len(champions_df_clean)}")
    print(f"  - Evaluierte Portfolios:  {total_evaluated:,}")
    print()


if __name__ == "__main__":
    main()
