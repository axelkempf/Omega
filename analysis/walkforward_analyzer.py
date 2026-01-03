"""
Walkforward Analysis Tool

Analyzes walk-forward optimization results and generates combined scores.

Security Note:
- _find_final_scores_files() uses comprehensive loop detection when following
  symlinks to prevent infinite loops from circular references while still
  discovering all final_scores files in the directory tree.

Performance Notes:
- Regex patterns are compiled once at module level to avoid O(n) recompilation.
- Trades and equity data are cached to prevent redundant file I/O.
- LRU caches have bounded sizes to prevent memory bloat.
- DataFrame operations are vectorized where possible to replace .iterrows().
- Duplicate star-rating logic is unified via parameterized function.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.metric_adjustments import (
    risk_adjusted,
    shrinkage_adjusted,
    wilson_score_lower_bound,
)
from backtest_engine.optimizer.final_param_selector import (
    __compute_yearly_stability,
    _merge_scores_with_yearly,
    _parameter_columns,
)
from backtest_engine.runner import (
    _compute_backtest_robust_metrics,
    run_backtest_and_return_portfolio,
)

# Module-level regex compilation (avoids O(n) recompilation in loops)
_YEAR_PREFIX_PATTERN = re.compile(r"^(\d{4})\s+(.+)$")
_YEAR_SUFFIX_PATTERN = re.compile(r"^(.+?)[_\s-](\d{4})$")
_RUN_ID_SCENARIO_PATTERN = re.compile(r"_z(\d+)$")


WALKFORWARD_ROOT = Path("var/results/analysis")
COMBINED_DIR = WALKFORWARD_ROOT / "combined"
BASE_COMBINED_CSV = COMBINED_DIR / "combined_base.csv"
TOP100_CSV = COMBINED_DIR / "top_100_walkforward_combos.csv"
TOP50_CSV = COMBINED_DIR / "top_50_walkforward_combos.csv"
TOP50_REFINED_CSV = COMBINED_DIR / "top_50_walkforward_combos_refined.csv"

# alternative Artefakte für Backfill-Auswertungen
BACKFILL_COMBINED_FILENAME = "05_final_scores_combined_backfill.csv"
BACKFILL_SNAPSHOT_NAME = "frozen_snapshot_backfill.json"

# Hard-gate thresholds (can be tuned if needed)
ROBUSTNESS_MIN = 0.8
STABILITY_MIN = 0.5
TP_SL_STRESS_MIN = 0.9

# Star-rating thresholds
STAR_WINRATE_MIN = 40.0
STAR_AVG_R_MIN = 0.15
STAR_PROFIT_OVER_DD_MIN = 2.0

# Robustness settings for refined Top-50 re-evaluation
REFINED_ROBUST_JITTER_REPEATS = 80
REFINED_ROBUST_JITTER_FRAC = 0.05


@dataclass
class YearlyColumns:
    years: List[str]
    winrate: Dict[str, str]
    avg_r: Dict[str, str]
    profit_over_dd: Dict[str, str]
    net_pnl: Dict[str, str]
    max_dd: Dict[str, str]
    trades: Dict[str, str]


def _find_final_scores_files(root: Path) -> List[Path]:
    """
    Find all 05_final_scores_combined.csv files under root.

    Uses comprehensive loop detection to safely follow symlinks without
    infinite recursion. Tracks all visited real paths (resolved symlinks)
    to prevent circular references.
    """
    files: List[Path] = []
    visited_dirs: set[Path] = set()  # Track all visited real paths
    final_dirs_seen: set[Path] = set()  # Track processed final_selection dirs

    for dirpath, _, _ in os.walk(root, followlinks=True):
        # Resolve to real path for loop detection
        real_path = Path(dirpath).resolve()

        # Skip if we've already visited this directory (prevents infinite loops)
        if real_path in visited_dirs:
            continue
        visited_dirs.add(real_path)

        # Only process directories containing "final_selection"
        if "final_selection" not in dirpath:
            continue

        final_dir = Path(dirpath)
        # Avoid processing the same final_selection directory twice
        if final_dir in final_dirs_seen:
            continue
        final_dirs_seen.add(final_dir)

        # Priorität: Backfill-Combined, falls vorhanden
        backfill_path = final_dir / BACKFILL_COMBINED_FILENAME
        if backfill_path.exists():
            files.append(backfill_path)
            continue

        combined_path = final_dir / "05_final_scores_combined.csv"
        if combined_path.exists():
            files.append(combined_path)
            continue
        rebuilt = _rebuild_combined_if_missing(final_dir)
        if rebuilt:
            files.append(rebuilt)
    files.sort()
    return files


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        path.write_text("", encoding="utf-8")
    else:
        df.to_csv(path, index=False)


def _sanitize_str_value(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _detect_equity_columns(df: pd.DataFrame) -> Tuple[str, str]:
    ts_candidates = [
        "timestamp",
        "time",
        "date",
        "datetime",
        "Date",
        "Timestamp",
    ]
    eq_candidates = [
        "equity",
        "Equity",
        "equity_value",
        "value",
    ]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    eq_col = next((c for c in eq_candidates if c in df.columns), None)
    if ts_col is None or eq_col is None:
        raise ValueError(
            f"Keine Equity-Spalten gefunden. Verfügbare Spalten: {list(df.columns)}"
        )
    return ts_col, eq_col


def _load_equity_series_for_combo(
    source_walkforward: str,
    combo_id: str,
    cache: Dict[Tuple[str, str], Optional[pd.Series]],
) -> Optional[pd.Series]:
    key = (source_walkforward, combo_id)
    if key in cache:
        return cache[key]
    src = _sanitize_str_value(source_walkforward)
    cid = _sanitize_str_value(combo_id)
    if not src or not cid:
        cache[key] = None
        return None
    eq_path = (
        WALKFORWARD_ROOT
        / src
        / "final_selection"
        / "equity_curves"
        / cid
        / "equity.csv"
    )
    if not eq_path.exists():
        cache[key] = None
        return None
    try:
        df = pd.read_csv(eq_path)
        ts_col, eq_col = _detect_equity_columns(df)
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        eq_vals = pd.to_numeric(df[eq_col], errors="coerce")
        mask = ts.notna() & eq_vals.notna()
        if not mask.any():
            cache[key] = None
            return None
        tmp = pd.DataFrame({"ts": ts[mask], "eq": eq_vals[mask]})
        tmp = tmp.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        idx = tmp["ts"].dt.tz_convert("UTC")
        series = pd.Series(tmp["eq"].values, index=idx)
        series.name = f"{src}__{cid}"
        cache[key] = series
        return series
    except Exception as exc:
        print(f"Warnung: Konnte Equity für {cid} ({src}) nicht laden: {exc}")
        cache[key] = None
        return None


def _load_trades_for_combo(
    source_walkforward: str,
    combo_id: str,
) -> Optional[pd.DataFrame]:
    src = _sanitize_str_value(source_walkforward)
    cid = _sanitize_str_value(combo_id)
    if not src or not cid:
        return None
    trades_path = (
        WALKFORWARD_ROOT / src / "final_selection" / "trades" / cid / "trades.json"
    )
    if not trades_path.exists():
        return None
    try:
        with trades_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        if not data:
            return None
        df = pd.DataFrame(data)
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(
                df["entry_time"], utc=True, errors="coerce"
            )
        return df
    except Exception as exc:
        print(f"Warnung: Konnte Trades für {cid} ({src}) nicht laden: {exc}")
        return None


def _combine_equity_series(
    series_a: Optional[pd.Series],
    series_b: Optional[pd.Series],
) -> Optional[pd.Series]:
    valid = [s for s in (series_a, series_b) if s is not None and not s.empty]
    if not valid:
        return None
    wide = pd.concat(valid, axis=1)
    wide.columns = [f"leg_{i}" for i in range(len(valid))]
    wide = wide.sort_index()
    init_vals: List[float] = []
    for s in valid:
        first_val = s.dropna().head(1)
        init_vals.append(float(first_val.iloc[0]) if not first_val.empty else np.nan)
    wide = wide.ffill()
    for col, init in zip(wide.columns, init_vals):
        if pd.notna(init):
            wide[col] = wide[col].fillna(init)
    # Jede Leg wird mit gleichem Kapital-Anteil gehandelt, daher Equity mitteln
    weight = 1.0 / len(valid)
    combined = wide.sum(axis=1, min_count=1) * weight
    combined.name = "equity_combined"
    return combined


def _drawdowns_by_year_from_equity(
    equity: Optional[pd.Series],
    years: Sequence[str],
) -> Dict[str, float]:
    if equity is None or equity.empty:
        return {}
    eq_vals = pd.to_numeric(equity, errors="coerce")
    idx = pd.to_datetime(equity.index, utc=True, errors="coerce")
    mask = idx.notna() & eq_vals.notna()
    if not mask.any():
        return {}
    eq_clean = pd.Series(eq_vals[mask].values, index=idx[mask]).sort_index()
    result: Dict[str, float] = {}
    for year in years:
        year_str = str(year).strip()
        if not year_str.isdigit():
            continue
        year_int = int(year_str)
        year_mask = eq_clean.index.year == year_int
        if not year_mask.any():
            continue
        values = eq_clean.loc[year_mask]
        if values.empty:
            continue
        roll_max = values.cummax()
        dd = (roll_max - values).max()
        if pd.notna(dd):
            result[year_str] = float(dd)
    return result


def _extract_year_segments_from_instrumentation(
    final_dir: Path,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    instr_path = final_dir / "instrumentation.json"
    if not instr_path.exists():
        return []
    try:
        payload = json.loads(instr_path.read_text())
    except Exception as exc:
        print(f"Warnung: Konnte {instr_path} nicht lesen: {exc}")
        return []
    stages = payload.get("stages") or []
    for stage in stages:
        details = stage.get("details") or {}
        segs = details.get("yearly_segments")
        if not segs:
            continue
        out: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
        for seg in segs:
            label = str(seg.get("label", "")).strip()
            if not label:
                continue
            start_raw = seg.get("start")
            end_raw = seg.get("end")
            try:
                start_ts = pd.to_datetime(start_raw) if start_raw else pd.NaT
            except Exception:
                start_ts = pd.NaT
            try:
                end_ts = pd.to_datetime(end_raw) if end_raw else pd.NaT
            except Exception:
                end_ts = pd.NaT
            out.append((label, start_ts, end_ts))
        if out:
            return out
    return []


def _infer_year_segments_from_detailed(
    df_scores_detailed: pd.DataFrame,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    segments: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if df_scores_detailed is None or df_scores_detailed.empty:
        return segments
    seen: set[str] = set()
    for col in df_scores_detailed.columns:
        name = str(col).strip().lower()
        if not name.startswith("year"):
            continue
        try:
            series = pd.to_numeric(df_scores_detailed[col], errors="coerce")
        except Exception:
            continue
        first_valid = series.dropna().head(1)
        if first_valid.empty:
            continue
        year_val = first_valid.iloc[0]
        try:
            label = str(int(float(year_val)))
        except Exception:
            label = str(year_val).strip()
        if not label or label in seen:
            continue
        segments.append((label, pd.NaT, pd.NaT))
        seen.add(label)
    return segments


def _rebuild_combined_if_missing(final_dir: Path) -> Optional[Path]:
    combined_path = final_dir / "05_final_scores_combined.csv"
    if combined_path.exists():
        return combined_path
    scores_path = final_dir / "05_final_scores.csv"
    detailed_path = final_dir / "05_final_scores_detailed.csv"
    if not scores_path.exists() or not detailed_path.exists():
        return None
    try:
        df_scores = pd.read_csv(scores_path)
        df_scores_detailed = pd.read_csv(detailed_path)
        # Spalten-Namen defensiv trimmen, damit combo_id etc. erkannt werden
        df_scores.columns = [str(c).strip() for c in df_scores.columns]
        df_scores_detailed.columns = [
            str(c).strip() for c in df_scores_detailed.columns
        ]
    except Exception as exc:
        print(f"Warnung: Konnte final_scores CSVs in {final_dir} nicht laden: {exc}")
        return None

    year_segments = _extract_year_segments_from_instrumentation(final_dir)
    if not year_segments:
        year_segments = _infer_year_segments_from_detailed(df_scores_detailed)

    try:
        df_combined = _merge_scores_with_yearly(
            df_scores, df_scores_detailed, year_segments
        )
    except Exception as exc:
        print(
            f"Warnung: Zusammenführen der final_scores in {final_dir} fehlgeschlagen: {exc}"
        )
        return None

    try:
        _safe_to_csv(df_combined, combined_path)
        print(f"05_final_scores_combined.csv rekonstruiert unter: {combined_path}")
        return combined_path
    except Exception as exc:
        print(
            f"Warnung: Konnte rekonstruiertes combined unter {combined_path} nicht speichern: {exc}"
        )
        return None


def _load_and_combine(files: Sequence[Path]) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    ref_cols: List[str] = []

    for idx, path in enumerate(files):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warnung: Konnte {path} nicht lesen: {exc}")
            continue
        rel_root = WALKFORWARD_ROOT
        try:
            rel = path.parent.parent.relative_to(rel_root)
        except Exception:
            rel = path.parent.parent
        df = df.copy()
        # Spalten-Namen normalisieren (String + Whitespace außen entfernen),
        # damit wir später keine scheinbar unterschiedlichen, aber eigentlich
        # gleichen Spalten (z.B. 'atr_mult ' vs. 'atr_mult') bekommen.
        df.columns = [str(c).strip() for c in df.columns]
        df["source_walkforward"] = str(rel)

        # Referenz-Spaltenreihenfolge als geordnete Vereinigung
        # aller CSVs aufbauen (Reihenfolge = erste Auftretensreihenfolge).
        for col in df.columns:
            if col not in ref_cols:
                ref_cols.append(col)

        frames.append(df)

    if not frames:
        return pd.DataFrame(), ref_cols
    # Spalten-Union herstellen und Frames in einem Schritt reindizieren, um
    # viele einzelne Spalten-Inserts (Fragmentierung) zu vermeiden.
    all_cols: List[str] = sorted({str(c) for df in frames for c in df.columns})
    aligned: List[pd.DataFrame] = [df.reindex(columns=all_cols) for df in frames]
    combined = pd.concat(aligned, axis=0, ignore_index=True)
    # Sicherstellen, dass es eine combo_id gibt
    if "combo_id" not in combined.columns:
        combined["combo_id"] = [f"combo_{i}" for i in range(len(combined))]
    combined["combo_id"] = combined["combo_id"].astype(str)
    # Eindeutige ID pro Zeile (combo + Walkforward-Kontext)
    if "source_walkforward" in combined.columns:
        combined["combo_key"] = (
            combined["combo_id"].astype(str)
            + "|"
            + combined["source_walkforward"].astype(str)
        )
    else:
        combined["combo_key"] = combined["combo_id"].astype(str)
    # Falls es immer noch Duplikate gibt, Index anhängen
    if combined["combo_key"].duplicated().any():
        combined["combo_key"] = [
            f"{ck}#{i}" for i, ck in enumerate(combined["combo_key"].astype(str))
        ]
    return combined, ref_cols


def _detect_yearly_columns(df: pd.DataFrame) -> YearlyColumns:
    years: set[str] = set()
    winrate: Dict[str, str] = {}
    avg_r: Dict[str, str] = {}
    profit_over_dd: Dict[str, str] = {}
    net_pnl: Dict[str, str] = {}
    max_dd: Dict[str, str] = {}
    trades: Dict[str, str] = {}

    # Pattern 1: "YYYY <Metric>" - use pre-compiled regex (OPTIMIZATION)
    for col in df.columns:
        name = str(col).strip()
        m = _YEAR_PREFIX_PATTERN.match(name)
        if not m:
            continue
        year, metric = m.group(1), m.group(2).strip().lower()
        if not year.isdigit():
            continue
        if "winrate" in metric:
            winrate[year] = name
        elif metric.startswith("avg r"):
            avg_r[year] = name
        elif metric.startswith("net profit"):
            net_pnl[year] = name
        elif "drawdown" in metric:
            max_dd[year] = name
        elif "total_trades" in metric or "trades" in metric:
            trades[year] = name
        years.add(year)

    # Pattern 2: "<metric>_<YYYY>" - use pre-compiled regex (OPTIMIZATION)
    for col in df.columns:
        name = str(col).strip()
        lower = name.lower()
        m = _YEAR_SUFFIX_PATTERN.match(lower)
        if not m:
            continue
        base, year = m.group(1).strip(), m.group(2)
        if not year.isdigit():
            continue
        if "winrate" in base and year not in winrate:
            winrate[year] = col
        elif base in ("avg_r", "avg-r", "avgr") and year not in avg_r:
            avg_r[year] = col
        elif base in ("net_pnl", "netpnl") and year not in net_pnl:
            net_pnl[year] = col
        elif base in ("max_dd", "drawdown") and year not in max_dd:
            max_dd[year] = col
        elif "trades" in base and year not in trades:
            trades[year] = col
        elif "profit_over_dd" in base and year not in profit_over_dd:
            profit_over_dd[year] = col
        years.add(year)

    year_list = sorted({y for y in years if y.isdigit()}, key=int)
    return YearlyColumns(
        years=year_list,
        winrate=winrate,
        avg_r=avg_r,
        profit_over_dd=profit_over_dd,
        net_pnl=net_pnl,
        max_dd=max_dd,
        trades=trades,
    )


def _apply_hard_gates(df: pd.DataFrame, yearly: YearlyColumns) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    # Robustness-Filter
    if "robustness_score_1" in out.columns:
        out = out[
            pd.to_numeric(out["robustness_score_1"], errors="coerce") >= ROBUSTNESS_MIN
        ]
    # Vorläufiger Stability-Filter (falls vorhanden)
    if "stability_score" in out.columns:
        out = out[
            pd.to_numeric(out["stability_score"], errors="coerce") >= STABILITY_MIN
        ]
    # TP/SL Stress Score Filter
    if "tp_sl_stress_score" in out.columns:
        out = out[
            pd.to_numeric(out["tp_sl_stress_score"], errors="coerce")
            >= TP_SL_STRESS_MIN
        ]
    if out.empty or not yearly.years:
        return out
    # Entferne Zeilen mit mindestens zwei negativen Jahres-Net-PnLs
    neg_counts = np.zeros(len(out), dtype=int)
    for year in yearly.years:
        col = yearly.net_pnl.get(year)
        if not col or col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        neg_counts += (vals < 0.0).to_numpy(dtype=int)
    # Erlaube bis zu ein negatives Jahr, verwerfe ab zwei
    mask_keep = neg_counts < 2
    out = out.loc[mask_keep].reset_index(drop=True)
    return out


def _safe_profit_over_dd(net_pnl: float, max_dd: float) -> float:
    max_dd = float(max_dd) if pd.notna(max_dd) else 0.0
    net_pnl = float(net_pnl) if pd.notna(net_pnl) else 0.0
    denom = abs(max_dd)
    # Drawdowns mit Betrag < 1 auf 1 normalisieren,
    # damit kein/kleiner Drawdown nicht bestraft wird
    if denom < 1.0:
        denom = 1.0
    return net_pnl / denom


def _round_value(value: Any, decimals: int) -> Any:
    """
    Safely round numeric values while preserving NaN/NA.
    """
    if pd.isna(value):
        return value
    try:
        return round(float(value), int(decimals))
    except Exception:
        return value


def _reorder_with_parameter_block(
    df: pd.DataFrame,
    meta_cols: Sequence[str],
    param_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Reorder columns so that:
      - meta_cols appear first (in the given order),
      - all parameter columns (per _parameter_columns) form one contiguous block,
      - all remaining metric/derived columns follow afterwards.

    OPTIMIZATION: Uses pre-compiled regex pattern to avoid O(n) recompilation.
    """
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    meta: List[Any] = [c for c in meta_cols if c in cols]

    # Jahres-Metriken erkennen (Gruppe 1 = Jahr) - use pre-compiled pattern
    # Nicht-jährliche Metriken (wie in 05_final_scores_combined)
    metric_cols_known = {
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
        "score",
        "stability",
        "stability_score",
        "wmape",
        "risk_adjusted",
        "profit_component",
        "cost_robustness",
        "robustness_penalty",
        "robustness_score",
        "robustness_score_stress",
        "p_mean_r_gt_0",
        "p_net_profit_gt_0",
        # zusätzliche Robustness/Stresstest-Metriken aus den kombinierten CSVs
        "robustness_score_1",
        "cost_shock_score",
        "timing_jitter_score",
        "trade_dropout_score",
        "tp_sl_stress_score",
    }

    # Kandidaten anhand _parameter_columns (alles, was dort als Parameter gilt)
    try:
        raw_param_candidates = set(_parameter_columns(df))
    except Exception:
        raw_param_candidates = set()

    # Metriken (nicht-jährlich, nicht Meta), Reihenfolge nach param_order/cols
    metrics: List[Any] = []
    seen_metrics: set[Any] = set()
    order_source: Sequence[Any] = param_order if param_order is not None else cols
    for c in order_source:
        if (
            c in cols
            and c not in meta
            and str(c) in metric_cols_known
            and not _YEAR_PREFIX_PATTERN.match(str(c))
            and c not in seen_metrics
        ):
            metrics.append(c)
            seen_metrics.add(c)

    # Parameter-Kandidaten: alles, was _parameter_columns liefert,
    # aber keine Meta-Spalte, keine bekannte Metrik und keine Jahres-Spalte ist.
    param_candidates_all = {
        c
        for c in raw_param_candidates
        if (
            c not in meta
            and str(c) not in metric_cols_known
            and not _YEAR_PREFIX_PATTERN.match(str(c))
        )
    }

    # Parameter in Referenz-Reihenfolge
    param_cols: List[Any] = []
    if param_order is not None:
        for c in param_order:
            if c in cols and c in param_candidates_all:
                param_cols.append(c)

    # Zusätzliche Parameter, die nicht in param_order stehen
    extra_params = [
        c for c in cols if c in param_candidates_all and c not in param_cols
    ]
    param_cols.extend(extra_params)

    taken = set(meta) | set(metrics) | set(param_cols)
    rest: List[Any] = [c for c in cols if c not in taken]

    # rest in Jahres-Metriken und sonstige Spalten aufteilen
    yearly_cols: List[Any] = [c for c in rest if _YEAR_PREFIX_PATTERN.match(str(c))]
    other_rest: List[Any] = [c for c in rest if c not in yearly_cols]

    # Jahres-Metriken nach Jahr sortieren: neuestes Jahr zuerst,
    # innerhalb eines Jahres Reihenfolge wie in param_order (falls vorhanden) bzw. cols.
    order_source: Sequence[Any] = param_order if param_order is not None else cols
    order_index = {c: i for i, c in enumerate(order_source)}

    def _year_sort_key(col: Any) -> Tuple[int, int]:
        m = _YEAR_PREFIX_PATTERN.match(str(col))
        year = int(m.group(1)) if m else -(10**9)
        # Negative Sortierung für "neueste zuerst"
        return (-year, order_index.get(col, 10**9))

    yearly_sorted = sorted(yearly_cols, key=_year_sort_key)

    return df.loc[:, meta + metrics + param_cols + other_rest + yearly_sorted]


def _add_star_ratings(
    df: pd.DataFrame, yearly: YearlyColumns
) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty or not yearly.years:
        return df.copy(), []
    out = df.copy()
    years = yearly.years
    star_cols: List[str] = []
    for year in years:
        win_col = yearly.winrate.get(year)
        avg_col = yearly.avg_r.get(year)
        # Für profit_over_dd pro Jahr ggf. NetProfit/Drawdown ableiten
        pod_col = None
        if yearly.profit_over_dd.get(year):
            pod_col = yearly.profit_over_dd[year]
        else:
            # on-the-fly berechnen und als temporäre Serie nutzen
            net_col = yearly.net_pnl.get(year)
            dd_col = yearly.max_dd.get(year)
            if net_col and dd_col and net_col in out.columns and dd_col in out.columns:
                net_vals = pd.to_numeric(out[net_col], errors="coerce")
                dd_vals = pd.to_numeric(out[dd_col], errors="coerce")
                pod_vals = [
                    _safe_profit_over_dd(n, d) for n, d in zip(net_vals, dd_vals)
                ]
                col_name = f"__tmp_profit_over_dd_{year}"
                out[col_name] = pod_vals
                pod_col = col_name
        star_col = f"star_{year}"
        star_cols.append(star_col)
        wr = (
            pd.to_numeric(out[win_col], errors="coerce")
            if win_col in out.columns
            else 0.0
        )
        avg_r = (
            pd.to_numeric(out[avg_col], errors="coerce")
            if avg_col in out.columns
            else 0.0
        )
        pod = (
            pd.to_numeric(out[pod_col], errors="coerce")
            if pod_col and pod_col in out.columns
            else 0.0
        )
        cond = (
            (wr > STAR_WINRATE_MIN)
            & (avg_r > STAR_AVG_R_MIN)
            & (pod > STAR_PROFIT_OVER_DD_MIN)
        )
        out[star_col] = cond.astype(int)
    # Gesamtsterne
    if star_cols:
        out["total_stars"] = out[star_cols].sum(axis=1)
        out = out[out["total_stars"] >= 1].reset_index(drop=True)
    # Temporäre Spalten entfernen
    tmp_cols = [c for c in out.columns if str(c).startswith("__tmp_profit_over_dd_")]
    if tmp_cols:
        out = out.drop(columns=tmp_cols)
    return out, years


def _build_star_matrix(
    df: pd.DataFrame, years: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        key = str(row["combo_key"])
        stars: Dict[str, int] = {}
        for year in years:
            col = f"star_{year}"
            val = int(row.get(col, 0) or 0)
            stars[year] = val
        matrix[key] = stars
    return matrix


def _generate_pairs(df: pd.DataFrame, years: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    matrix = _build_star_matrix(df, years)
    keys = list(matrix.keys())
    pairs_meta: List[Dict[str, str]] = []
    for i, j in itertools.combinations(range(len(keys)), 2):
        k1, k2 = keys[i], keys[j]
        stars1 = matrix.get(k1, {})
        stars2 = matrix.get(k2, {})
        valid = True
        for year in years:
            s1 = int(stars1.get(year, 0) or 0)
            s2 = int(stars2.get(year, 0) or 0)
            if s1 == 0 and s2 == 0:
                valid = False
                break
        if not valid:
            continue
        pairs_meta.append(
            {
                "combo_key_1": k1,
                "combo_key_2": k2,
            }
        )
    if not pairs_meta:
        return pd.DataFrame()
    pairs = pd.DataFrame(pairs_meta)
    # Original combo_ids und Walkforward-Metadaten anhängen
    idx = df.set_index("combo_key")
    for suffix in ("1", "2"):
        key_col = f"combo_key_{suffix}"
        pairs[f"combo_id_{suffix}"] = idx.loc[pairs[key_col], "combo_id"].values
        if "source_walkforward" in idx.columns:
            pairs[f"source_walkforward_{suffix}"] = idx.loc[
                pairs[key_col], "source_walkforward"
            ].values
    # Paar-ID: nur aus den combo_ids der beiden Legs ableiten,
    # ohne source_walkforward im Identifier zu kodieren.
    pairs["combo_pair_id"] = [
        f"{a}__{b}"
        for a, b in zip(
            pairs["combo_id_1"].astype(str), pairs["combo_id_2"].astype(str)
        )
    ]
    return pairs


def _compute_pair_yearly_metrics(
    pairs: pd.DataFrame,
    singles: pd.DataFrame,
    yearly: YearlyColumns,
) -> pd.DataFrame:
    """
    Compute combined yearly metrics for pairs.

    OPTIMIZATION: Pre-compute lookup dictionaries and use vectorized
    operations instead of per-suffix iteration.
    """
    if pairs is None or pairs.empty or singles is None or singles.empty:
        return pd.DataFrame()
    out = pairs.copy()
    base = singles.set_index("combo_key")
    years = yearly.years

    # OPTIMIZATION: Pre-build lookup dicts for all required columns
    lookup_cols = {
        "win": yearly.winrate,
        "avg": yearly.avg_r,
        "pnl": yearly.net_pnl,
        "dd": yearly.max_dd,
        "trades": yearly.trades,
    }

    for year in years:
        # Prepare all column mappings for this year
        col_map = {}
        for key, col_dict in lookup_cols.items():
            col_map[key] = col_dict.get(year)

        # OPTIMIZATION: For each leg suffix, do bulk lookups instead of per-column .loc
        for suffix, leg_label in (("1", "A"), ("2", "B")):
            key_col = f"combo_key_{suffix}"
            combo_keys = out[key_col].values

            # Create result dict for this leg/year
            leg_data = {}

            # Bulk lookup for all metrics
            for metric_key, col_name in col_map.items():
                if col_name and col_name in base.columns:
                    try:
                        vals = pd.to_numeric(
                            base.loc[combo_keys, col_name], errors="coerce"
                        ).values
                        leg_data[metric_key] = vals
                    except Exception:
                        leg_data[metric_key] = np.full(len(combo_keys), np.nan)
                else:
                    leg_data[metric_key] = np.full(len(combo_keys), np.nan)

            # Assign all metrics at once (vectorized)
            prefix = f"{year}_{leg_label}"
            out[f"{prefix}_winrate"] = leg_data.get("win", np.nan)
            out[f"{prefix}_avg_r"] = leg_data.get("avg", np.nan)
            out[f"{prefix}_net_pnl"] = leg_data.get("pnl", np.nan)
            out[f"{prefix}_max_dd"] = leg_data.get("dd", np.nan)
            out[f"{prefix}_trades"] = leg_data.get("trades", np.nan)

        # Kombinierte Kennzahlen - vectorized
        wa = pd.to_numeric(out[f"{year}_A_winrate"], errors="coerce")
        wb = pd.to_numeric(out[f"{year}_B_winrate"], errors="coerce")
        ta = pd.to_numeric(out[f"{year}_A_trades"], errors="coerce")
        tb = pd.to_numeric(out[f"{year}_B_trades"], errors="coerce")
        denom_trades = ta + tb
        win_comb = (wa * ta + wb * tb) / denom_trades.replace(0, np.nan)
        avg_a = out[f"{year}_A_avg_r"]
        avg_b = out[f"{year}_B_avg_r"]
        avg_comb = (avg_a * ta + avg_b * tb) / denom_trades.replace(0, np.nan)
        pnl_a = out[f"{year}_A_net_pnl"]
        pnl_b = out[f"{year}_B_net_pnl"]
        pnl_comb = (pnl_a + pnl_b) / 2.0
        dd_a = out[f"{year}_A_max_dd"]
        dd_b = out[f"{year}_B_max_dd"]
        dd_comb = (dd_a + dd_b) / 2.0
        trades_comb = denom_trades

        out[f"winrate_combined_{year}"] = win_comb
        out[f"avg_r_combined_{year}"] = avg_comb
        out[f"net_pnl_combined_{year}"] = pnl_comb
        out[f"max_dd_combined_{year}"] = dd_comb
        out[f"trades_combined_{year}"] = trades_comb

        # profit_over_dd_combined_YYYY
        # OPTIMIZATION: Vectorized profit_over_dd computation instead of Python loop
        # Convert to numpy for faster computation
        pnl_arr = np.asarray(pnl_comb, dtype=float)
        dd_arr = np.asarray(dd_comb, dtype=float)

        # Compute denominator: abs(dd), but minimum 1.0
        denom = np.abs(dd_arr)
        denom = np.where(denom < 1.0, 1.0, denom)

        # Handle NaN values
        pod_comb = np.where(np.isfinite(denom), pnl_arr / denom, np.nan)
        out[f"profit_over_dd_combined_{year}"] = pod_comb
    out = _override_combined_drawdowns_with_equity(out, years)
    return out


def _override_combined_drawdowns_with_equity(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
) -> pd.DataFrame:
    """
    Override combined drawdowns with equity-calculated values.

    OPTIMIZATION: Load all equity series once, align them with a lightweight
    numpy forward-fill, and batch-apply drawdown/PoD updates to avoid
    per-row DataFrame construction.
    """
    if df_pairs is None or df_pairs.empty or not years:
        return df_pairs
    out = df_pairs.copy()
    year_ints = [int(y) for y in years if str(y).isdigit()]
    if not year_ints:
        return out

    n_rows = len(out)
    # Start with existing values and override only when equity data is available
    dd_arrays: Dict[str, np.ndarray] = {}
    pod_arrays: Dict[str, np.ndarray] = {}
    pnl_arrays: Dict[str, np.ndarray] = {}
    for y_int in year_ints:
        y_str = str(y_int)
        dd_col = f"max_dd_combined_{y_str}"
        pod_col = f"profit_over_dd_combined_{y_str}"
        pnl_col = f"net_pnl_combined_{y_str}"
        dd_arrays[y_str] = (
            out[dd_col].to_numpy(copy=True)
            if dd_col in out.columns
            else np.full(n_rows, np.nan, dtype=float)
        )
        pod_arrays[y_str] = (
            out[pod_col].to_numpy(copy=True)
            if pod_col in out.columns
            else np.full(n_rows, np.nan, dtype=float)
        )
        pnl_arrays[y_str] = pd.to_numeric(
            out.get(pnl_col, np.nan), errors="coerce"
        ).to_numpy()

    # Collect all unique (source, combo) pairs once and cache equity reads
    equity_cache: Dict[Tuple[str, str], Optional[pd.Series]] = {}
    combos_to_load: set[Tuple[str, str]] = set()
    for src_col, cid_col in (
        ("source_walkforward_1", "combo_id_1"),
        ("source_walkforward_2", "combo_id_2"),
    ):
        if src_col not in out.columns or cid_col not in out.columns:
            continue
        for src_raw, cid_raw in zip(out[src_col], out[cid_col]):
            src = _sanitize_str_value(src_raw)
            cid = _sanitize_str_value(cid_raw)
            if src and cid:
                combos_to_load.add((src, cid))

    equity_arrays: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for src, cid in combos_to_load:
        series = _load_equity_series_for_combo(src, cid, equity_cache)
        if series is None or series.empty:
            continue
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        ts = pd.to_datetime(series.index, utc=True, errors="coerce")
        mask = ts.notna() & np.isfinite(vals)
        if not mask.any():
            continue
        ts_int = ts.view("int64")[mask]
        equity_arrays[(src, cid)] = (ts_int, vals[mask])

    if not equity_arrays:
        return out

    src1_list = [_sanitize_str_value(s) for s in out.get("source_walkforward_1", [])]
    cid1_list = [_sanitize_str_value(s) for s in out.get("combo_id_1", [])]
    src2_list = [_sanitize_str_value(s) for s in out.get("source_walkforward_2", [])]
    cid2_list = [_sanitize_str_value(s) for s in out.get("combo_id_2", [])]

    def _ffill_to_union(
        ts_int: np.ndarray, values: np.ndarray, union: np.ndarray
    ) -> np.ndarray:
        """Forward-fill values onto a common union index using numpy only."""
        filled = np.full(union.shape[0], np.nan, dtype=float)
        if ts_int.size == 0:
            return filled
        positions = np.searchsorted(union, ts_int)
        filled[positions] = values
        mask_valid = np.isfinite(filled)
        if not mask_valid.any():
            return filled
        first_valid = int(np.flatnonzero(mask_valid)[0])
        mask_valid[: first_valid + 1] = True
        last_idx = np.where(mask_valid, np.arange(filled.size), first_valid)
        np.maximum.accumulate(last_idx, out=last_idx)
        return filled[last_idx]

    for row_idx in range(n_rows):
        key1 = (
            (src1_list[row_idx], cid1_list[row_idx])
            if row_idx < len(src1_list)
            else ("", "")
        )
        key2 = (
            (src2_list[row_idx], cid2_list[row_idx])
            if row_idx < len(src2_list)
            else ("", "")
        )

        series_data: List[Tuple[np.ndarray, np.ndarray]] = []
        if key1 in equity_arrays:
            series_data.append(equity_arrays[key1])
        if key2 in equity_arrays:
            series_data.append(equity_arrays[key2])
        if not series_data:
            continue

        if len(series_data) == 1:
            union_index = series_data[0][0]
        else:
            union_index = np.union1d(series_data[0][0], series_data[1][0])
        if union_index.size == 0:
            continue

        filled_series = [
            _ffill_to_union(ts_int, vals, union_index) for ts_int, vals in series_data
        ]
        combined_equity = np.sum(filled_series, axis=0) / float(len(filled_series))

        years_for_union = pd.to_datetime(union_index).year
        for y_int in year_ints:
            year_mask = years_for_union == y_int
            if not year_mask.any():
                continue
            vals_year = combined_equity[year_mask]
            if vals_year.size == 0:
                continue
            # Numeric operations only; avoids per-year pandas objects
            cummax = np.maximum.accumulate(vals_year)
            dd_val = float(np.max(cummax - vals_year))
            if math.isnan(dd_val):
                continue
            y_str = str(y_int)
            dd_arrays[y_str][row_idx] = dd_val
            pnl_val = pnl_arrays.get(y_str, np.array([]))
            if pnl_val.size > row_idx:
                pnl = pnl_val[row_idx]
                if math.isnan(pnl):
                    continue
                denom = abs(dd_val)
                if denom < 1.0:
                    denom = 1.0
                pod_arrays[y_str][row_idx] = pnl / denom

    for y_str, arr in dd_arrays.items():
        out[f"max_dd_combined_{y_str}"] = arr
    for y_str, arr in pod_arrays.items():
        out[f"profit_over_dd_combined_{y_str}"] = arr

    return out


def _add_combined_star_ratings(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
) -> pd.DataFrame:
    if df_pairs is None or df_pairs.empty or not years:
        return df_pairs
    out = df_pairs.copy()
    star_cols: List[str] = []
    for year in years:
        win_col = f"winrate_combined_{year}"
        avg_col = f"avg_r_combined_{year}"
        pod_col = f"profit_over_dd_combined_{year}"
        wr = pd.to_numeric(out.get(win_col, np.nan), errors="coerce")
        avg_r = pd.to_numeric(out.get(avg_col, np.nan), errors="coerce")
        pod = pd.to_numeric(out.get(pod_col, np.nan), errors="coerce")
        star_col = f"star_combined_{year}"
        cond = (
            (wr > STAR_WINRATE_MIN)
            & (avg_r > STAR_AVG_R_MIN)
            & (pod > STAR_PROFIT_OVER_DD_MIN)
        )
        out[star_col] = cond.astype(int)
        star_cols.append(star_col)
    if star_cols:
        out["total_stars_combined"] = out[star_cols].sum(axis=1)
        max_stars = len(years)
        # Mindestens max_stars - 1
        out = out[out["total_stars_combined"] >= max_stars - 1].reset_index(drop=True)
    # Filter auf negative kombinierte Jahres-Net-PnLs
    if out.empty:
        return out
    mask_keep = np.ones(len(out), dtype=bool)
    for year in years:
        col = f"net_pnl_combined_{year}"
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        mask_keep &= ~(vals < 0.0)
    out = out.loc[mask_keep].reset_index(drop=True)
    return out


def _add_yearly_composite_scores(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
) -> pd.DataFrame:
    """
    Add yearly composite scores with trade-count adjusted metrics.

    Applies shrinkage adjustments to Average R and Profit over Drawdown,
    and Wilson Score Lower Bound to Winrate based on the number of trades per year.

    OPTIMIZATION: Combine multiple mask/fillna/where operations into
    single vectorized operations per year.
    """
    if df_pairs is None or df_pairs.empty or not years:
        return df_pairs
    out = df_pairs.copy()
    comp_cols: List[str] = []

    for year in years:
        win_col = f"winrate_combined_{year}"
        avg_col = f"avg_r_combined_{year}"
        pod_col = f"profit_over_dd_combined_{year}"
        trades_col = f"trades_combined_{year}"

        # Extract raw metrics
        wr_raw = pd.to_numeric(out.get(win_col, 0.0), errors="coerce").fillna(0.0)
        avg_r_raw = pd.to_numeric(out.get(avg_col, 0.0), errors="coerce").fillna(0.0)
        pod_raw = pd.to_numeric(out.get(pod_col, 0.0), errors="coerce").fillna(0.0)
        n_trades = pd.to_numeric(out.get(trades_col, 0.0), errors="coerce").fillna(0.0)

        # Apply adjustments (yearly: n_years=1.0)
        # Winrate: convert to decimal, Wilson lower bound adjust, keep as decimal
        wr_decimal = wr_raw.values / 100.0
        wr_adjusted = wilson_score_lower_bound(
            winrate=wr_decimal, n_trades=n_trades.values
        )

        # Average R: shrinkage adjustment
        avg_r_adjusted = shrinkage_adjusted(
            average_r=avg_r_raw.values,
            n_trades=n_trades.values,
            n_years=1.0,  # yearly metrics
        )

        # Profit over Drawdown: risk adjustment (clip negatives first)
        pod_raw_clipped = np.where(pod_raw.values < 0.0, 0.0, pod_raw.values)
        pod_adjusted = risk_adjusted(
            profit_over_drawdown=pod_raw_clipped,
            n_trades=n_trades.values,
            n_years=1.0,  # yearly metrics
        )

        # Normalize adjusted PoD for score
        pod_term = pod_adjusted / (1.0 + pod_adjusted)
        pod_term = np.where(np.isfinite(pod_term), pod_term, 0.0)

        # Composite score with adjusted metrics
        comp = (wr_adjusted + avg_r_adjusted + pod_term) * 0.33
        col = f"comp_score_{year}"
        out[col] = comp
        comp_cols.append(col)

        # Store trade-count adjusted yearly metrics as separate columns.
        # Naming uses year-prefix and "*_adust" (requested), winrate in % (0-100).
        out[f"{year}_winrate_adust"] = wr_adjusted * 100.0
        out[f"{year}_avg_r_adust"] = avg_r_adjusted
        out[f"{year}_profit_over_dd_adust"] = pod_adjusted

        # DEBUG: Log first few scores to validate adjustments
        if year == years[0] and len(out) <= 3:  # First year, first few rows
            for idx in range(min(3, len(out))):
                print(
                    f"  [WFA] Year {year}, Row {idx}: "
                    f"wr_raw={wr_raw.iloc[idx]:.1f}% → adj={wr_adjusted[idx]*100:.1f}%, "
                    f"avg_r_raw={avg_r_raw.iloc[idx]:.3f} → adj={avg_r_adjusted[idx]:.3f}, "
                    f"pod_raw={pod_raw.iloc[idx]:.3f} → adj={pod_adjusted[idx]:.3f}, "
                    f"comp_score={comp[idx]:.4f}"
                )

    if comp_cols:
        out["comp_score_combined"] = out[comp_cols].mean(axis=1)
    else:
        out["comp_score_combined"] = 0.0
    return out


def _add_stability_scores_for_pairs(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
) -> pd.DataFrame:
    if df_pairs is None or df_pairs.empty or not years:
        return df_pairs
    # df_scores für Paare: combo_id + comp_score_combined
    scores = pd.DataFrame(
        {
            "combo_id": df_pairs["combo_pair_id"].astype(str),
            "comp_score_combined": pd.to_numeric(
                df_pairs.get("comp_score_combined", 0.0), errors="coerce"
            ).fillna(0.0),
        }
    )
    # df_scores_detailed als MultiIndex-DataFrame mit yearly Net Profit
    cols: List[Tuple[str, str]] = [("meta", "combo_id")]
    for year in years:
        cols.append((year, "Net Profit"))
    multi_cols = pd.MultiIndex.from_tuples(cols, names=["scope", "metric"])
    data: List[List[float]] = []
    for _, row in df_pairs.iterrows():
        combo_id = str(row["combo_pair_id"])
        vals: List[float] = [combo_id]
        for year in years:
            col = f"net_pnl_combined_{year}"
            vals.append(
                float(pd.to_numeric(row.get(col, np.nan), errors="coerce") or 0.0)
            )
        data.append(vals)
    df_detailed = pd.DataFrame(data, columns=multi_cols)
    try:
        df_stab = __compute_yearly_stability(scores, df_detailed)
    except Exception as exc:
        print(f"Warnung: Stability-Berechnung fehlgeschlagen: {exc}")
        df_stab = pd.DataFrame(
            {
                "combo_id": scores["combo_id"],
                "stability_score": np.ones(len(scores), dtype=float),
            }
        )
    if df_stab.empty:
        df_pairs = df_pairs.copy()
        df_pairs["stability_score"] = 1.0
        return df_pairs
    df_stab = df_stab[["combo_id", "stability_score"]].copy()
    df_stab.rename(columns={"combo_id": "combo_pair_id"}, inplace=True)
    out = df_pairs.merge(df_stab, on="combo_pair_id", how="left")
    out["stability_score"] = pd.to_numeric(
        out.get("stability_score", 1.0), errors="coerce"
    ).fillna(1.0)
    return out


def _add_final_score_and_sort(df_pairs: pd.DataFrame) -> pd.DataFrame:
    if df_pairs is None or df_pairs.empty:
        return df_pairs
    out = df_pairs.copy()
    comp = pd.to_numeric(out.get("comp_score_combined", 0.0), errors="coerce").fillna(
        0.0
    )
    stab = pd.to_numeric(out.get("stability_score", 1.0), errors="coerce").fillna(1.0)
    # Gewichtung: 70% comp_score_combined, 30% stability_score
    out["comp_score_final"] = 0.7 * comp + 0.3 * stab
    out = out.sort_values("comp_score_final", ascending=False).reset_index(drop=True)
    return out


def _winsorize_series(
    series: pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Clip values of a Series to the given lower/upper quantiles.

    Keeps NaNs untouched and returns a copy to avoid side effects.
    """
    if series is None:
        return pd.Series(dtype=float)
    arr = pd.to_numeric(series, errors="coerce")
    mask = arr.notna()
    if not mask.any():
        return arr
    lo = float(np.nanquantile(arr[mask], lower))
    hi = float(np.nanquantile(arr[mask], upper))
    clipped = arr.clip(lower=lo, upper=hi)
    return clipped


def _build_year_profile_features(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
    *,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build per-year performance profile features for clustering.

        For each year y the following features are produced:
            - winrate fraction (0..1)
            - average R
            - net_pnl (signed log1p-transform)
            - max_drawdown (signed log1p-transform)
    """
    if df_pairs is None or df_pairs.empty or not years:
        return pd.DataFrame(), []

    features: Dict[str, pd.Series] = {}
    feature_cols: List[str] = []

    for year in years:
        win_col = f"winrate_combined_{year}"
        avg_col = f"avg_r_combined_{year}"
        pnl_col = f"net_pnl_combined_{year}"
        dd_col = f"max_dd_combined_{year}"

        winrate_raw = pd.to_numeric(df_pairs.get(win_col, np.nan), errors="coerce")
        avg_r_raw = pd.to_numeric(df_pairs.get(avg_col, np.nan), errors="coerce")
        pnl_raw = pd.to_numeric(df_pairs.get(pnl_col, np.nan), errors="coerce")
        dd_raw = pd.to_numeric(df_pairs.get(dd_col, np.nan), errors="coerce")

        win_feat = (winrate_raw / 100.0).astype(float)
        avg_feat = avg_r_raw.astype(float)
        # Signed log1p-Transform für stabilere Skalen, dann Winsorisierung auf transformierten Werten
        pnl_arr = pnl_raw.to_numpy(dtype=float)
        dd_arr = dd_raw.to_numpy(dtype=float)
        pnl_log = np.sign(pnl_arr) * np.log1p(np.abs(pnl_arr))
        dd_log = np.sign(dd_arr) * np.log1p(np.abs(dd_arr))
        # Winsorisiere auf den transformierten Werten
        pnl_feat = _winsorize_series(
            pd.Series(pnl_log), lower=winsor_lower, upper=winsor_upper
        ).to_numpy()
        dd_feat = _winsorize_series(
            pd.Series(dd_log), lower=winsor_lower, upper=winsor_upper
        ).to_numpy()

        cols = [
            f"{year}_wr",
            f"{year}_avg_r",
            f"{year}_net_pnl",
            f"{year}_max_dd",
        ]
        feature_cols.extend(cols)

        features[cols[0]] = win_feat
        features[cols[1]] = avg_feat
        features[cols[2]] = pnl_feat
        features[cols[3]] = dd_feat

    feature_df = pd.DataFrame(features, index=df_pairs.index)
    return feature_df, feature_cols


def _robust_scale_features(
    feature_df: pd.DataFrame,
    *,
    epsilon: float = 1e-9,
    min_mad: float = 1e-3,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Robust-scale (median/MAD) each feature column with type-aware handling.

    Edgecases:
      - Sehr kleine MAD (< min_mad) werden „weich“ behandelt: nur zentrieren,
        kein aggressives Hochskalieren (denom=1.0). epsilon bleibt Absicherung
        gegen Division durch 0.

    Returns scaled DataFrame plus medians and MADs for diagnostics.
    """
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    scaled_cols: Dict[str, pd.Series] = {}
    medians = pd.Series(index=feature_df.columns, dtype=float)
    mads = pd.Series(index=feature_df.columns, dtype=float)

    for col in feature_df.columns:
        clipped = pd.to_numeric(feature_df[col], errors="coerce")

        med_val = np.nanmedian(clipped)
        med = float(med_val) if np.isfinite(med_val) else 0.0
        mad_val = np.nanmedian(np.abs(clipped - med))
        mad = float(mad_val) if np.isfinite(mad_val) else 0.0
        medians[col] = med
        mads[col] = mad

        if mad < min_mad or not np.isfinite(mad):
            scaled = clipped - med
        else:
            denom = mad if mad > epsilon else 1.0
            scaled = (clipped - med) / (denom + epsilon)
        scaled_cols[col] = scaled

    scaled_df = pd.DataFrame(scaled_cols, index=feature_df.index)
    return scaled_df, medians, mads


def _cluster_hdbscan(
    features: pd.DataFrame,
    *,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    allow_single_cluster: bool = False,
    allow_noise: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run HDBSCAN on scaled features. Returns (labels, probabilities, outlier_scores).
    If hdbscan is unavailable or input is insufficient, returns (None, None, None).
    """
    if features is None or features.empty or len(features) < 2:
        return None, None, None
    try:
        import hdbscan  # type: ignore
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised in environments without hdbscan
        print(f"[Cluster] HDBSCAN nicht verfügbar: {exc}. Überspringe Clustering.")
        return None, None, None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=min_samples,
        metric=metric,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method="eom",
        prediction_data=False,
        algorithm="best",
    )
    labels = clusterer.fit_predict(features)
    if labels is None:
        return None, None, None
    probs = getattr(clusterer, "probabilities_", None)
    outlier_scores = getattr(clusterer, "outlier_scores_", None)

    if not allow_noise and labels is not None:
        # Reassign noise to nearest cluster by probability if requested
        if probs is not None and len(probs) == len(labels):
            mask_noise = labels == -1
            if mask_noise.any():
                # Deterministic tie-breaker: pick max probability cluster label
                core_labels = clusterer.labels_
                for idx, is_noise in enumerate(mask_noise):
                    if not is_noise:
                        continue
                    # fallback to most probable cluster if available
                    if core_labels is not None and core_labels.size == len(labels):
                        labels[idx] = int(core_labels[probs.argmax()])
        labels = np.where(labels == -1, 0, labels)
    return labels, probs, outlier_scores


def _cluster_metric_tolerance(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
    interval: float = 0.10,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """
    Alternative Clustering-Methodik basierend auf metrischen Intervallen.

    Optimierte Version mit NumPy-Vektorisierung für maximale Performance.

     Algorithmus:
     1. Sortiere DataFrame nach comp_score_final absteigend
     2. Nimm besten Kandidaten und definiere für JEDES JAHR y ±interval Intervalle für:
         - avg_r_combined_y, winrate_combined_y
     3. Alle Kandidaten, die für JEDES Jahr in ALLE Intervalle fallen, bilden Cluster 1
    4. Entferne Cluster 1 aus DataFrame, wiederhole mit verbleibendem DataFrame
    5. Stoppe wenn kein Cluster mit min_cluster_size mehr möglich
    6. Kandidaten ohne Cluster (-1) werden automatisch durchgelassen

    Returns:
        labels: np.ndarray mit Cluster-IDs (-1 für Noise/Singles, 0, 1, 2, ...)
    """
    if df_pairs is None or df_pairs.empty:
        return np.array([])

    if not years or len(years) == 0:
        print("[Cluster Interval] Keine Jahre verfügbar, überspringe Clustering.")
        return np.full(len(df_pairs), -1, dtype=int)

    n_candidates = len(df_pairs)
    n_years = len(years)

    # Initialisiere Labels mit -1 (Noise/Singles)
    labels = np.full(n_candidates, -1, dtype=int)

    # Sortiere nach comp_score_final absteigend
    comp_scores = pd.to_numeric(
        df_pairs.get("comp_score_final", 0.0), errors="coerce"
    ).fillna(0.0)
    sorted_indices = comp_scores.argsort()[::-1]  # Absteigende Sortierung

    # Extrahiere alle Metriken als NumPy Arrays für schnellen Zugriff
    metric_names = ["avg_r", "winrate"]
    n_metrics = len(metric_names)
    metrics_array = np.full((n_candidates, n_years, n_metrics), np.nan, dtype=float)
    for year_idx, year in enumerate(years):
        for metric_idx, metric in enumerate(metric_names):
            col_name = f"{metric}_combined_{year}"
            if col_name in df_pairs.columns:
                values = pd.to_numeric(df_pairs[col_name], errors="coerce").values
                metrics_array[:, year_idx, metric_idx] = values

    # Maske für valide Kandidaten (alle Metriken über alle Jahre sind finite)
    valid_mask = np.all(np.isfinite(metrics_array), axis=(1, 2))

    # Tracking der verbleibenden Kandidaten
    remaining_mask = np.ones(n_candidates, dtype=bool)
    cluster_id = 0

    while True:
        # Kombiniere: noch verfügbar UND valide
        available_mask = remaining_mask & valid_mask
        available_count = np.sum(available_mask)

        if available_count < min_cluster_size:
            break

        # Finde besten verfügbaren Kandidaten (nach Score sortiert)
        anchor_idx = None
        for idx in sorted_indices:
            if available_mask[idx]:
                anchor_idx = idx
                break

        if anchor_idx is None:
            break

        # Anchor-Metriken: Shape (n_years, n_metrics)
        anchor_metrics = metrics_array[anchor_idx]

        # Berechne Intervall-Grenzen für alle Jahre und Metriken
        # Shape: (n_years, n_metrics, 2) - last dim: [min, max]
        val_min = anchor_metrics * (1 - interval)
        val_max = anchor_metrics * (1 + interval)

        # Handle negative Werte (min/max swap)
        bounds_min = np.minimum(val_min, val_max)
        bounds_max = np.maximum(val_min, val_max)

        # Vektorisierter Intervall-Check für ALLE verfügbaren Kandidaten
        # Prüfe ob jeder Kandidat in den Intervallen liegt
        # Shape: (n_candidates, n_years, n_metrics) -> boolean
        in_interval = (metrics_array >= bounds_min[np.newaxis, :, :]) & (
            metrics_array <= bounds_max[np.newaxis, :, :]
        )

        # Kandidat muss für ALLE Jahre und ALLE Metriken im Intervall sein
        # Shape: (n_candidates,)
        passes_all_checks = np.all(in_interval, axis=(1, 2))

        # Kombiniere mit available_mask
        cluster_mask = available_mask & passes_all_checks
        cluster_indices = np.where(cluster_mask)[0]

        # Cluster nur bilden wenn min_cluster_size erreicht
        if len(cluster_indices) >= min_cluster_size:
            # Weise Cluster-ID zu
            labels[cluster_indices] = cluster_id
            remaining_mask[cluster_indices] = False

            # Debug-Ausgabe
            if cluster_id % 10 == 0 or cluster_id < 5:  # Reduziere Output
                combo_id = df_pairs.iloc[anchor_idx].get("combo_pair_id", "?")
                score = comp_scores.iloc[anchor_idx]
                print(
                    f"[Cluster Interval] Cluster {cluster_id}: {len(cluster_indices)} Kandidaten "
                    f"(Anchor: {combo_id}, Score: {score:.4f})"
                )

            cluster_id += 1
        else:
            # Anchor bleibt Noise, markiere als nicht mehr verfügbar
            remaining_mask[anchor_idx] = False

    # Statistik ausgeben
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
    print(f"[Cluster Interval] Label-Verteilung: {label_dist}")
    print(
        f"[Cluster Interval] {cluster_id} Cluster gebildet auf Basis von {n_years} Jahren"
    )

    return labels


def _compute_robust_stress_score(
    row: pd.Series,
    singles_lookup: Optional[pd.DataFrame],
) -> float:
    """
    Compute robust_stress_score for a pair row using per-leg metrics.

    robustness_score_1_mean: mean of leg robustness_score_1
    tp_sl_stress_score_mean: mean of leg tp_sl_stress_score
    robust_stress_score = mean(robustness_score_1_mean, tp_sl_stress_score_mean)

    If values are missing, fall back gracefully and may return NaN.
    """

    def _leg_metric(combo_key: str, metric: str) -> float:
        if not combo_key:
            return math.nan
        if singles_lookup is not None and metric in singles_lookup.columns:
            try:
                val = singles_lookup.loc[combo_key, metric]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                return float(val)
            except Exception:
                return math.nan
        # fallback: check if aggregated column exists on row (e.g., robustness_1_mean)
        col_alt = f"{metric}_mean"
        if col_alt in row:
            try:
                return float(row.get(col_alt))
            except Exception:
                return math.nan
        return math.nan

    robustness_vals: List[float] = []
    tp_sl_vals: List[float] = []
    for suffix in ("1", "2"):
        key = str(row.get(f"combo_key_{suffix}", "") or "").strip()
        if key:
            robustness_vals.append(_leg_metric(key, "robustness_score_1"))
            tp_sl_vals.append(_leg_metric(key, "tp_sl_stress_score"))

    def _nanmean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return math.nan
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return math.nan
        with np.errstate(all="ignore"):
            result = np.nanmean(arr[finite_mask])
            return float(result) if np.isfinite(result) else math.nan

    robustness_mean = _nanmean(robustness_vals)
    tp_sl_mean = _nanmean(tp_sl_vals)

    components: List[float] = [
        v for v in (robustness_mean, tp_sl_mean) if np.isfinite(v)
    ]
    if not components:
        return math.nan
    return float(np.nanmean(components))


def _log_cluster_diagnostics(
    df_pairs: pd.DataFrame,
    labels: np.ndarray,
    label_series: pd.Series,
    selected_indices: List[int],
) -> None:
    """
    Log diagnostic information and export to CSV: 10 random clusters with ALL their candidates
    (including yearly metrics) and 10 random noise candidates (including yearly metrics).
    """
    print("\n" + "=" * 80)
    print(
        "[Cluster Diagnostics] 10 Cluster mit allen Kandidaten + 10 Noise mit jährlichen Metriken"
    )
    print("=" * 80)

    clusters_data: List[Dict[str, Any]] = []
    noise_data: List[Dict[str, Any]] = []

    # Extrahiere Jahre aus den Spalten
    years = []
    for col in df_pairs.columns:
        if col.startswith("winrate_combined_"):
            year = col.replace("winrate_combined_", "")
            if year not in years:
                years.append(year)
    years = sorted(years)

    # 10 zufällige Cluster auswählen
    unique_clusters = [l for l in label_series.unique() if l != -1]
    if unique_clusters:
        np.random.seed(42)
        sample_clusters = np.random.choice(
            unique_clusters, size=min(10, len(unique_clusters)), replace=False
        )
        print(
            f"\n[Clusters] 10 exemplarische Cluster (von {len(unique_clusters)} gesamt) mit allen Kandidaten:"
        )

        for cluster_label in sorted(sample_clusters):
            cluster_indices = label_series[label_series == cluster_label].index
            cluster_size = len(cluster_indices)

            print(f"\n  Cluster {int(cluster_label)}: {cluster_size} Kandidaten")

            # ALLE Kandidaten in diesem Cluster
            for idx in sorted(cluster_indices):
                row = df_pairs.loc[idx]
                combo_pair_id = str(row.get("combo_pair_id", "?"))
                comp_score = float(row.get("comp_score_final", np.nan))
                is_rep = idx in selected_indices
                rep_marker = " [REP]" if is_rep else ""

                print(f"    ├─ {combo_pair_id} (score={comp_score:.4f}){rep_marker}")

                # Sammle für CSV mit jährlichen Metriken
                entry = {
                    "cluster_id": int(cluster_label),
                    "cluster_size": cluster_size,
                    "combo_pair_id": combo_pair_id,
                    "comp_score_final": comp_score,
                    "is_representative": is_rep,
                    "source_walkforward_1": row.get("source_walkforward_1", ""),
                    "source_walkforward_2": row.get("source_walkforward_2", ""),
                }

                # Jährliche Metriken hinzufügen
                for year in years:
                    wr_col = f"winrate_combined_{year}"
                    avg_col = f"avg_r_combined_{year}"
                    pnl_col = f"net_pnl_combined_{year}"
                    dd_col = f"max_dd_combined_{year}"

                    entry[f"{year}_winrate"] = row.get(wr_col, np.nan)
                    entry[f"{year}_avg_r"] = row.get(avg_col, np.nan)
                    entry[f"{year}_net_pnl"] = row.get(pnl_col, np.nan)
                    entry[f"{year}_max_dd"] = row.get(dd_col, np.nan)

                clusters_data.append(entry)

    # 10 zufällige Noise-Kandidaten
    noise_indices = label_series[label_series == -1].index
    if len(noise_indices) > 0:
        np.random.seed(42)
        sample_noise = np.random.choice(
            noise_indices, size=min(10, len(noise_indices)), replace=False
        )
        print(
            f"\n[Noise] 10 exemplarische Noise-Kandidaten (von {len(noise_indices)} gesamt):"
        )

        for noise_idx in sorted(sample_noise):
            row = df_pairs.loc[noise_idx]
            combo_pair_id = str(row.get("combo_pair_id", "?"))
            comp_score = float(row.get("comp_score_final", np.nan))

            print(f"  ├─ {combo_pair_id} (score={comp_score:.4f})")

            # Sammle für CSV mit jährlichen Metriken
            entry = {
                "combo_pair_id": combo_pair_id,
                "comp_score_final": comp_score,
                "source_walkforward_1": row.get("source_walkforward_1", ""),
                "source_walkforward_2": row.get("source_walkforward_2", ""),
            }

            # Jährliche Metriken hinzufügen
            for year in years:
                wr_col = f"winrate_combined_{year}"
                avg_col = f"avg_r_combined_{year}"
                pnl_col = f"net_pnl_combined_{year}"
                dd_col = f"max_dd_combined_{year}"

                entry[f"{year}_winrate"] = row.get(wr_col, np.nan)
                entry[f"{year}_avg_r"] = row.get(avg_col, np.nan)
                entry[f"{year}_net_pnl"] = row.get(pnl_col, np.nan)
                entry[f"{year}_max_dd"] = row.get(dd_col, np.nan)

            noise_data.append(entry)
    else:
        print(f"\n[Noise] Keine Noise-Kandidaten (alle Daten sind geclustert)")

    # Speichere CSVs
    try:
        clusters_csv = COMBINED_DIR / "cluster_diagnostics_clusters_sample.csv"
        if clusters_data:
            df_clusters = pd.DataFrame(clusters_data)
            df_clusters.to_csv(clusters_csv, index=False)
            print(f"\n[CSV Export] Cluster-Sample gespeichert: {clusters_csv}")
            print(f"             ({len(clusters_data)} Zeilen aus 10 Clustern)")

        noise_csv = COMBINED_DIR / "cluster_diagnostics_noise_sample.csv"
        if noise_data:
            df_noise = pd.DataFrame(noise_data)
            df_noise.to_csv(noise_csv, index=False)
            print(f"[CSV Export] Noise-Sample gespeichert: {noise_csv}")
            print(f"             ({len(noise_data)} Zeilen)")
    except Exception as exc:
        print(f"[CSV Export] Fehler beim Speichern der Diagnostik-CSVs: {exc}")

    print("\n" + "=" * 80 + "\n")


def _select_cluster_representatives(
    df_pairs: pd.DataFrame,
    labels: Optional[np.ndarray],
    *,
    singles_lookup: Optional[pd.DataFrame],
    ignore_noise: bool = True,
) -> pd.DataFrame:
    """
    Select cluster representatives:
      - Metric champion: highest comp_score_final per cluster
      - Robust-stress champion: highest robust_stress_score (≠ metric champion)
    """
    if df_pairs is None or df_pairs.empty or labels is None:
        return df_pairs
    if len(labels) != len(df_pairs):
        print("[Cluster] Label-Länge passt nicht zu df_pairs. Überspringe Clustering.")
        return df_pairs

    reps: List[int] = []
    label_series = pd.Series(labels, index=df_pairs.index)
    unique_labels = label_series.unique()
    cluster_labels = [l for l in unique_labels if not (ignore_noise and l == -1)]

    # Precompute robust_stress_score per row
    robust_scores = pd.Series(index=df_pairs.index, dtype=float)
    for idx, row in df_pairs.iterrows():
        robust_scores.at[idx] = _compute_robust_stress_score(row, singles_lookup)

    for lbl in cluster_labels:
        cluster_idx = label_series[label_series == lbl].index
        if cluster_idx.empty:
            continue
        cluster_df = df_pairs.loc[cluster_idx]
        cluster_df = cluster_df.assign(
            _comp=pd.to_numeric(
                cluster_df.get("comp_score_final", 0.0), errors="coerce"
            ).fillna(0.0),
            _robust=robust_scores.loc[cluster_idx].values,
        )
        cluster_df = cluster_df.sort_values(
            ["_comp", "combo_pair_id"], ascending=[False, True]
        )
        metric_champion_idx = cluster_df.index[0]
        reps.append(metric_champion_idx)

        candidates = cluster_df.drop(index=metric_champion_idx)
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            by=["_robust", "_comp", "combo_pair_id"],
            ascending=[False, False, True],
        )
        best_robust_idx = candidates.index[0]
        best_score = candidates.loc[best_robust_idx, "_robust"]
        if np.isfinite(best_score):
            reps.append(best_robust_idx)

    reps_unique = []
    seen = set()
    for idx in reps:
        if idx not in seen:
            reps_unique.append(idx)
            seen.add(idx)

    if not reps_unique:
        print("[Cluster] Keine Repräsentanten gefunden, nutze Baseline-Ergebnis.")
        return df_pairs

    selected = df_pairs.loc[reps_unique]
    # Reihenfolge beibehalten wie im ursprünglichen Ranking
    selected = selected.loc[df_pairs.index.intersection(selected.index)]

    # Diagnostic: zeige 10 zufällige Cluster mit Repräsentanten
    _log_cluster_diagnostics(df_pairs, labels, label_series, reps_unique)

    return selected.reset_index(drop=True)


def _expand_pairs_for_display(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
    base_df: Optional[pd.DataFrame] = None,
    include_leg_robustness: bool = False,
) -> pd.DataFrame:
    """
    Erzeuge eine Anzeige-Variante der Paar-Toplisten für den CSV-Export
    (Top 100 / Top 50), in der:

      - jede combo_pair_id genau zwei Leg-Zeilen hat (A und B),
      - beide Legs klar derselben combo_pair_id zugeordnet sind,
      - jedes Leg seine eigenen Jahresmetriken in einem konsistenten
        Schema ohne A/B-Präfix erhält (z.B. „2022_winrate“, nicht
        „2022_A_winrate“),
      - Paar-Metriken (kombinierte Scores / Sterne) klar von den
        Leg-Metriken getrennt sind und der Paar-ID zugeordnet bleiben.

    Damit entsteht pro Paar ein „Block“ aus:
      - gemeinsamen Paar-Kennzahlen und
      - zwei vollständig ausgewiesenen Leg-Faktentabellen (Zeile A/B).
    """
    if df_pairs is None or df_pairs.empty:
        return df_pairs

    # Nur numerische Jahreswerte übernehmen, neuestes Jahr zuerst (wie combined_base.csv)
    years_sorted_int = sorted(
        {int(y) for y in years if str(y).isdigit()},
        reverse=True,
    )
    years_str = [str(y) for y in years_sorted_int]

    # Parameter- und per-Leg-Metrik-Spalten aus combined_base bzw. base_df ermitteln
    param_cols: List[str] = []
    param_map: Optional[pd.DataFrame] = None
    metric_cols: List[str] = []
    metric_map: Optional[pd.DataFrame] = None
    trades_sum_map: Optional[pd.Series] = None

    base_source: Optional[pd.DataFrame] = None
    if base_df is not None and not base_df.empty:
        base_source = base_df
    else:
        try:
            if BASE_COMBINED_CSV.exists():
                base_source = pd.read_csv(BASE_COMBINED_CSV)
        except Exception as exc:
            print(f"Warnung: Konnte {BASE_COMBINED_CSV} nicht lesen: {exc}")
            base_source = None

    if base_source is not None and not base_source.empty:
        base_norm = base_source.copy()
        base_norm.columns = [str(c).strip() for c in base_norm.columns]
        if "combo_key" in base_norm.columns:
            # Gleiche Heuristik wie in _reorder_with_parameter_block verwenden,
            # damit die Parameter-Spalten exakt dem Block in combined_base.csv entsprechen.
            year_prefix = re.compile(r"^(\d{4})\s")
            metric_cols_known = {
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
                "score",
                "stability",
                "stability_score",
                "wmape",
                "risk_adjusted",
                "profit_component",
                "cost_robustness",
                "robustness_penalty",
                "robustness_score",
                "robustness_score_stress",
                "p_mean_r_gt_0",
                "p_net_profit_gt_0",
                "robustness_score_1",
                "cost_shock_score",
                "timing_jitter_score",
                "trade_dropout_score",
                "tp_sl_stress_score",
            }
            try:
                raw_param_candidates = set(_parameter_columns(base_norm))
            except Exception:
                raw_param_candidates = set()
            param_candidates_all = {
                c
                for c in raw_param_candidates
                if (
                    c not in ("combo_key", "source_walkforward")
                    and str(c) not in metric_cols_known
                    and not year_prefix.match(str(c))
                )
            }
            param_cols = [c for c in base_norm.columns if c in param_candidates_all]
            if param_cols:
                try:
                    param_map = base_norm.set_index("combo_key")[param_cols]
                except Exception as exc:
                    print(
                        "Warnung: Konnte Parameter-Mapping aus combined_base.csv nicht "
                        f"aufbauen: {exc}"
                    )
                    param_cols = []
                    param_map = None

            # Per-Leg-Metriken wie in combined_base.csv für die Anzeige
            metric_display_candidates: List[str] = [
                "Net Profit",
                "Drawdown",
                "profit_over_dd",
                "Commission",
                "Avg R-Multiple",
                "Winrate (%)",
                "Sharpe (trade)",
                "Sortino (trade)",
                "robustness_score_1",
                "cost_shock_score",
                "timing_jitter_score",
                "trade_dropout_score",
                "tp_sl_stress_score",
                "stability_score",
                "p_mean_r_gt_0",
                "p_net_profit_gt_0",
            ]
            metric_cols = [
                c for c in metric_display_candidates if c in base_norm.columns
            ]
            if metric_cols:
                try:
                    metric_map = base_norm.set_index("combo_key")[metric_cols]
                except Exception as exc:
                    print(
                        "Warnung: Konnte Metrik-Mapping aus combined_base.csv nicht "
                        f"aufbauen: {exc}"
                    )
                    metric_cols = []
                    metric_map = None

            # Total Trades pro Leg über alle Jahre summieren
            try:
                yearly_cols_info = _detect_yearly_columns(base_norm)
                trade_cols = [
                    c
                    for c in yearly_cols_info.trades.values()
                    if c in base_norm.columns
                ]
                if trade_cols:
                    trades_sum_series = (
                        base_norm[trade_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0.0)
                        .sum(axis=1)
                    )
                    trades_sum_map = pd.Series(
                        trades_sum_series.values,
                        index=base_norm["combo_key"].astype(str),
                    )
            except Exception as exc:
                print(f"Warnung: Konnte Trades-Summe nicht berechnen: {exc}")
                trades_sum_map = None

    # Paar-Meta- und Score-Spalten (so weit vorhanden)
    pair_meta_cols: List[str] = ["combo_pair_id"]
    # Score-/Meta-Spalten des Paars in der Reihenfolge,
    # in der auch sortiert wird: verfeinerter Score zuerst,
    # ansonsten der ursprüngliche finale Score.
    # Hinweis: comp_score_combined wird in der Anzeige erst
    # nach den per-Leg-Metriken platziert.
    pair_score_candidates = [
        "comp_score_final_refined",
        "comp_score_final",
        "robust_stress_score",
        "robustness_1_mean",
        "comp_score_combined",
    ]
    pair_meta_cols.extend([c for c in pair_score_candidates if c in df_pairs.columns])

    # Leg-spezifische Meta-Spalten in der Anzeige
    leg_meta_cols: List[str] = [
        "combo_leg",
        "combo_id",
        "source_walkforward",
    ]

    yearly_cols: List[str] = []
    for year in years_str:
        yearly_cols.append(f"{year}_year")
        yearly_cols.append(f"{year}_net_pnl")
        yearly_cols.append(f"{year}_winrate")
        yearly_cols.append(f"{year}_winrate_adust")
        yearly_cols.append(f"{year}_avg_r")
        yearly_cols.append(f"{year}_avg_r_adust")
        yearly_cols.append(f"{year}_max_dd")
        yearly_cols.append(f"{year}_trades")
        yearly_cols.append(f"{year}_profit_over_dd")
        yearly_cols.append(f"{year}_profit_over_dd_adust")

    # Anzeige-Reihenfolge: zuerst Snapshot-Meta (aus frozen_snapshot),
    # dann Paar-ID und Leg-Meta und übrige Paar-Meta-/Score-Spalten.
    snapshot_meta_cols: List[str] = [
        "symbol",
        "timeframe",
        "direction",
        "strategy_name",
        "szenario",
    ]
    primary_meta: List[str] = ["combo_pair_id"] + leg_meta_cols
    secondary_pair_meta: List[str] = [
        c for c in pair_meta_cols if c not in ("combo_pair_id", "comp_score_combined")
    ]
    # Per-Leg-Metriken analog zu combined_base.csv
    # Hinweis: robustness_score_1_jittered_80 enthält – falls vorhanden –
    # den pro-Leg gejitterten Robustness-1-Wert aus den verfeinerten
    # Backtests. Er wird vor dem ursprünglichen robustness_score_1
    # einsortiert und nur angezeigt, wenn include_leg_robustness=True.
    per_leg_metric_display_base: List[str] = [
        "Net Profit",
        "Drawdown",
        "profit_over_dd",
        "Commission",
        "Avg R-Multiple",
        "Winrate (%)",
        "trades",
        "Sharpe (trade)",
        "Sortino (trade)",
        "robustness_score_1_jittered_80",
        "robustness_1_num_samples",
        "robustness_score_1",
        "cost_shock_score",
        "timing_jitter_score",
        "trade_dropout_score",
        "tp_sl_stress_score",
        "stability_score",
        "p_mean_r_gt_0",
        "p_net_profit_gt_0",
        "same_trades_entry",
        "same_trades_absolut",
        "same_trades_absolut_percentage",
    ]

    # Bedingte Anzeige: robustness_score_1_jittered_80 nur wenn include_leg_robustness=True
    per_leg_metric_display = per_leg_metric_display_base.copy()
    if not include_leg_robustness:
        per_leg_metric_display = [
            m for m in per_leg_metric_display if m != "robustness_score_1_jittered_80"
        ]
        per_leg_metric_display = [
            m for m in per_leg_metric_display if m != "robustness_1_num_samples"
        ]
    # comp_score_combined und stability_score_combined sollen in der Anzeige
    # direkt nach den per-Leg-Metriken (also nach p_net_profit_gt_0) erscheinen.
    extra_scores_after_metrics: List[str] = []
    if "comp_score_combined" in pair_meta_cols:
        extra_scores_after_metrics.append("comp_score_combined")
        extra_scores_after_metrics.append("stability_score_combined")

    # Session-/HTF-Block zwischen Parametern und Jahreswerten
    session_htf_cols: List[str] = [
        "session_filter",
        "htf_tf",
        "htf_filter",
        "extra_htf_tf",
        "extra_htf_filter",
        "time_period",
        "total_stars_combined",
        "yearly_stars_per_leg",
    ]

    display_cols: List[str] = (
        snapshot_meta_cols
        + primary_meta
        + secondary_pair_meta
        + per_leg_metric_display
        + extra_scores_after_metrics
        + param_cols
        + session_htf_cols
        + yearly_cols
    )

    rows: List[Dict[str, Any]] = []

    # OPTIMIZATION: Pre-compute same_trades metrics for all pairs to avoid redundant file I/O
    # Cache trades and same_trades computations
    trades_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
    same_trades_cache: Dict[Tuple[str, str, str, str], Tuple[Any, Any, Any]] = {}

    # Pre-load and cache all unique source+combo pairs
    unique_pairs_to_load: set[Tuple[str, str]] = set()
    for _, row in df_pairs.iterrows():
        for suffix in ("1", "2"):
            src = str(row.get(f"source_walkforward_{suffix}", "")).strip()
            cid = str(row.get(f"combo_id_{suffix}", "")).strip()
            if src and cid:
                unique_pairs_to_load.add((src, cid))

    # Load all trades at once (OPTIMIZATION: bulk cache instead of lazy loading per pair)
    for src, cid in unique_pairs_to_load:
        trades_cache[(src, cid)] = _load_trades_for_combo(src, cid)

    for _, row in df_pairs.iterrows():
        # Gemeinsame Trades pro Paar (Entry-Zeitpunkte)
        # same_trades_absolut: Anzahl Trades an gemeinsamen Entry-Zeitpunkten
        # same_trades_entry: Anzahl gemeinsamer Entry-Zeitpunkte (mind. 2 Trades)
        # same_trades_absolut_percentage: Anteil dieser gemeinsamen Trades relativ zur Anzahl aller Trades
        same_trades_entry = pd.NA
        same_trades_absolut = pd.NA
        same_trades_absolut_percentage = pd.NA
        src1 = str(row.get("source_walkforward_1", "") or "").strip()
        src2 = str(row.get("source_walkforward_2", "") or "").strip()
        cid1 = str(row.get("combo_id_1", "") or "").strip()
        cid2 = str(row.get("combo_id_2", "") or "").strip()
        if src1 and src2 and cid1 and cid2:
            try:
                # OPTIMIZATION: Use pre-cached trades instead of loading each time
                trades_a = trades_cache.get((src1, cid1))
                trades_b = trades_cache.get((src2, cid2))
                if trades_a is not None and trades_b is not None:
                    # Kombiniere alle Trades beider Legs und berechne gemeinsame Entry-Zeitpunkte
                    all_trades = []
                    if not trades_a.empty:
                        all_trades.append(trades_a)
                    if not trades_b.empty:
                        all_trades.append(trades_b)
                    if all_trades:
                        trades_df = pd.concat(all_trades, ignore_index=True)
                        if "entry_time" in trades_df.columns:
                            et = pd.to_datetime(
                                trades_df["entry_time"], utc=True, errors="coerce"
                            )
                            et = et.dropna()
                            if not et.empty:
                                counts = et.value_counts()
                                shared_mask = counts >= 2
                                # Anzahl gemeinsamer Entry-Zeitpunkte (mind. 2 Trades)
                                identical_ts = int(shared_mask.sum())
                                # Anzahl aller Trades an diesen Zeitpunkten
                                identical_trades_total = (
                                    int(counts[shared_mask].sum())
                                    if identical_ts > 0
                                    else 0
                                )
                                total_ts = int(et.size)
                                same_trades_entry = identical_ts
                                same_trades_absolut = identical_trades_total
                                if total_ts > 0:
                                    same_trades_absolut_percentage = _round_value(
                                        float(identical_trades_total) / float(total_ts),
                                        5,
                                    )
            except Exception as exc:
                print(
                    f"Warnung: Konnte same_trades für Paar {cid1}__{cid2} nicht berechnen: {exc}"
                )

        # Basis-Paarinformationen (werden für beide Legs identisch übernommen)
        base_pair: Dict[str, Any] = {col: row.get(col, pd.NA) for col in pair_meta_cols}
        # Composite-/Stability-/Robustness-Scores auf 5 Nachkommastellen runden
        for score_col in (
            "comp_score_combined",
            "stability_score",
            "robustness_1_mean",
            "robust_stress_score",
            "comp_score_final",
            "comp_score_final_refined",
        ):
            if score_col in base_pair:
                base_pair[score_col] = _round_value(base_pair[score_col], 5)
        # Kombinierte Stability explizit als eigene Spalte für die Anzeige bereitstellen
        if "stability_score" in row:
            base_pair["stability_score_combined"] = _round_value(
                row.get("stability_score", pd.NA), 5
            )

        for leg_label, suffix in (("A", "1"), ("B", "2")):
            leg_row: Dict[str, Any] = dict(base_pair)
            leg_row["combo_leg"] = leg_label
            leg_row["combo_id"] = row.get(f"combo_id_{suffix}", pd.NA)
            leg_row["source_walkforward"] = row.get(
                f"source_walkforward_{suffix}", pd.NA
            )
            # total_stars_combined gehört jetzt in den Session/HTF-Block
            leg_row["total_stars_combined"] = row.get("total_stars_combined", pd.NA)

            # Same trades Metriken pro Leg (identisch für beide Legs, da Paar-Metriken)
            leg_row["same_trades_entry"] = same_trades_entry
            leg_row["same_trades_absolut"] = same_trades_absolut
            leg_row["same_trades_absolut_percentage"] = same_trades_absolut_percentage

            # Snapshot-Metadaten pro Run (symbol, timeframe, direction,
            # strategy_name, szenario) aus dem frozen_snapshot lesen
            run_id_val = str(leg_row.get("source_walkforward", "") or "").strip()
            if run_id_val:
                snap_meta = _get_snapshot_meta_for_run(run_id_val)
            else:
                snap_meta = {}
            for col in (
                "symbol",
                "timeframe",
                "direction",
                "strategy_name",
                "szenario",
                "session_filter",
                "htf_tf",
                "htf_filter",
                "extra_htf_tf",
                "extra_htf_filter",
                "time_period",
            ):
                if col in snap_meta:
                    leg_row[col] = snap_meta.get(col, pd.NA)
                elif col not in leg_row:
                    leg_row[col] = pd.NA

            # Leg-spezifischen gejitterten Robustness-1-Score aus den Paar-Spalten übernehmen
            # und in der Anzeige-Spalte robustness_score_1_jittered_80 bereitstellen
            leg_score_col = f"robustness_1_leg_{leg_label}"
            if include_leg_robustness and leg_score_col in row:
                leg_row["robustness_score_1_jittered_80"] = _round_value(
                    row.get(leg_score_col, pd.NA), 5
                )
            else:
                leg_row["robustness_score_1_jittered_80"] = pd.NA

            # Leg-spezifische Sample-Anzahl (Observability) aus den Paar-Spalten übernehmen.
            # In der Anzeige bewusst ohne A/B-Suffix, da pro Paar ohnehin je eine Zeile
            # für Leg A und Leg B erzeugt wird.
            leg_samples_col = f"robustness_1_num_samples_leg_{leg_label}"
            if include_leg_robustness and leg_samples_col in row:
                raw_samples = row.get(leg_samples_col, pd.NA)
                if pd.isna(raw_samples):
                    leg_row["robustness_1_num_samples"] = pd.NA
                else:
                    try:
                        leg_row["robustness_1_num_samples"] = int(raw_samples)
                    except Exception:
                        leg_row["robustness_1_num_samples"] = pd.NA
            else:
                leg_row["robustness_1_num_samples"] = pd.NA

            # Parameterwerte pro Leg aus combined_base übernehmen
            if param_map is not None and param_cols:
                combo_key_val = str(row.get(f"combo_key_{suffix}", "")).strip()
                if combo_key_val and combo_key_val in param_map.index:
                    base_params = param_map.loc[combo_key_val]
                    for p in param_cols:
                        leg_row[p] = base_params.get(p, pd.NA)
                else:
                    for p in param_cols:
                        leg_row[p] = pd.NA
            # Per-Leg-Metriken aus combined_base übernehmen
            if metric_map is not None and metric_cols:
                combo_key_val = str(row.get(f"combo_key_{suffix}", "")).strip()
                if combo_key_val and combo_key_val in metric_map.index:
                    base_metrics = metric_map.loc[combo_key_val]
                    for m in metric_cols:
                        val = base_metrics.get(m, pd.NA)
                        if m == "profit_over_dd":
                            leg_row[m] = _round_value(val, 2)
                        elif m in (
                            "robustness_score_1",
                            "cost_shock_score",
                            "timing_jitter_score",
                            "trade_dropout_score",
                            "stability_score",
                            "tp_sl_stress_score",
                        ):
                            leg_row[m] = _round_value(val, 5)
                        else:
                            leg_row[m] = val
                else:
                    for m in metric_cols:
                        leg_row[m] = pd.NA
            # Trades gesamt pro Leg (Summe über alle Jahres-Trades)
            leg_trades_total = pd.NA
            combo_key_val_for_trades = str(row.get(f"combo_key_{suffix}", "")).strip()
            if trades_sum_map is not None and combo_key_val_for_trades:
                try:
                    if combo_key_val_for_trades in trades_sum_map.index:
                        leg_trades_total = trades_sum_map.get(
                            combo_key_val_for_trades, pd.NA
                        )
                except Exception:
                    leg_trades_total = pd.NA
            leg_row["trades"] = leg_trades_total

            # Jahresmetriken: für beide Legs die gleichen kombinierten Werte anzeigen
            for year in years_str:
                # Jahr-Spalte
                leg_row[f"{year}_year"] = int(year)

                win = _round_value(row.get(f"winrate_combined_{year}", pd.NA), 2)
                avg_r = _round_value(row.get(f"avg_r_combined_{year}", pd.NA), 4)
                net_pnl = _round_value(row.get(f"net_pnl_combined_{year}", pd.NA), 2)
                max_dd = _round_value(row.get(f"max_dd_combined_{year}", pd.NA), 2)
                trades = row.get(f"trades_combined_{year}", pd.NA)
                pod = _round_value(row.get(f"profit_over_dd_combined_{year}", pd.NA), 2)

                leg_row[f"{year}_net_pnl"] = net_pnl
                leg_row[f"{year}_winrate"] = win
                leg_row[f"{year}_winrate_adust"] = _round_value(
                    row.get(f"{year}_winrate_adust", pd.NA), 2
                )
                leg_row[f"{year}_avg_r"] = avg_r
                leg_row[f"{year}_avg_r_adust"] = _round_value(
                    row.get(f"{year}_avg_r_adust", pd.NA), 4
                )
                leg_row[f"{year}_max_dd"] = max_dd
                leg_row[f"{year}_trades"] = trades
                leg_row[f"{year}_profit_over_dd"] = pod
                leg_row[f"{year}_profit_over_dd_adust"] = _round_value(
                    row.get(f"{year}_profit_over_dd_adust", pd.NA), 2
                )

            # Jahre sammeln, in denen dieses Leg einen Stern erhält
            star_years: List[str] = []
            for year in years_str:
                prefix = f"{year}_{leg_label}"
                win = pd.to_numeric(
                    row.get(f"{prefix}_winrate", np.nan), errors="coerce"
                )
                avg_r = pd.to_numeric(
                    row.get(f"{prefix}_avg_r", np.nan), errors="coerce"
                )
                net = pd.to_numeric(
                    row.get(f"{prefix}_net_pnl", np.nan), errors="coerce"
                )
                dd = pd.to_numeric(row.get(f"{prefix}_max_dd", np.nan), errors="coerce")
                pod_val = (
                    _safe_profit_over_dd(net, dd)
                    if pd.notna(net) and pd.notna(dd)
                    else np.nan
                )
                cond = (
                    pd.notna(win)
                    and pd.notna(avg_r)
                    and pd.notna(pod_val)
                    and win > STAR_WINRATE_MIN
                    and avg_r > STAR_AVG_R_MIN
                    and pod_val > STAR_PROFIT_OVER_DD_MIN
                )
                if cond:
                    star_years.append(year)
            leg_row["yearly_stars_per_leg"] = (
                ", ".join(star_years) if star_years else pd.NA
            )

            rows.append(leg_row)

        # Leerzeile als optische Trennung zwischen Paar-Blöcken
        blank_row: Dict[str, Any] = {col: pd.NA for col in display_cols}
        rows.append(blank_row)

    df_display = pd.DataFrame(rows)
    # Sicherstellen, dass alle erwarteten Spalten vorhanden sind
    for col in display_cols:
        if col not in df_display.columns:
            df_display[col] = pd.NA
    return df_display.loc[:, display_cols]


def _reorder_pair_output_for_export(
    df_pairs: pd.DataFrame,
    years: Sequence[str],
) -> pd.DataFrame:
    """
    Reorder columns for pair-based CSVs (Top 100/Top 50) so that they follow
    the same high-level Struktur wie combined_base.csv:
      - Meta-Daten zuerst,
      - dann nicht-jährliche Metriken,
      - danach sämtliche jahresbezogenen Spalten, sortiert von neuestem zu ältestem Jahr.
    """
    if df_pairs is None or df_pairs.empty:
        return df_pairs
    cols = list(df_pairs.columns)

    # Meta-Spalten der Paar-Kombinationen
    meta_candidates = [
        "combo_key_1",
        "combo_key_2",
        "combo_id_1",
        "source_walkforward_1",
        "combo_id_2",
        "source_walkforward_2",
        "combo_pair_id",
    ]
    meta: List[Any] = [c for c in meta_candidates if c in cols]

    # Nicht-jährliche Metriken (Aggregationen über alle Jahre)
    metric_candidates = [
        "comp_score_final_refined",
        "total_stars_combined",
        "comp_score_combined",
        "stability_score",
        "comp_score_final",
        "robust_stress_score",
        "robustness_1_mean",
    ]
    metrics: List[Any] = [c for c in metric_candidates if c in cols]

    taken = set(meta) | set(metrics)

    # Jahres-Spalten: alles, was eine Jahreszahl aus 'years' im Namen trägt
    years_sorted = sorted({int(y) for y in years if str(y).isdigit()}, reverse=True)
    yearly: List[Any] = []
    for y in years_sorted:
        y_str = str(y)
        for c in cols:
            if c in taken or c in yearly:
                continue
            if y_str in str(c):
                yearly.append(c)

    taken.update(yearly)

    # Übrige Spalten, die weder Meta noch Aggregatsmetriken noch jahresbezogen sind
    rest: List[Any] = [c for c in cols if c not in taken]

    ordered_cols: List[Any] = meta + metrics + rest + yearly
    return df_pairs.loc[:, ordered_cols]


@lru_cache(maxsize=64)  # OPTIMIZATION: Bounded cache to prevent memory bloat
def _load_snapshot_base_config(run_id: str) -> Dict[str, Any]:
    """Load base_config for a given walkforward run (run_YYYYMMDD_HHMMSS)."""
    baseline_dir = WALKFORWARD_ROOT / run_id / "baseline"
    backfill = baseline_dir / BACKFILL_SNAPSHOT_NAME
    snap_path = backfill if backfill.exists() else baseline_dir / "frozen_snapshot.json"
    if not snap_path.exists():
        raise FileNotFoundError(f"Baseline-Snapshot nicht gefunden: {snap_path}")
    print(f"[WF-Analyzer] Lade frozen_snapshot für Run '{run_id}' von: {snap_path}")
    with snap_path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    base_cfg = blob.get("base_config") or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Ungültige base_config in {snap_path}")
    upgraded = _upgrade_base_config(base_cfg, run_id=run_id)
    return upgraded


@lru_cache(maxsize=1)
def _standard_mean_rev_reporting() -> Optional[Dict[str, Any]]:
    """
    Lädt das aktuelle Standard-Reporting aus configs/backtest/mean_reversion_z_score.json.
    Wird als Referenz benutzt, um fehlende oder abweichende Reporting-Einträge zu ergänzen.
    """
    cfg_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "backtest"
        / "mean_reversion_z_score.json"
    )
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            default_cfg = json.load(fh)
        rep = default_cfg.get("reporting")
        if isinstance(rep, dict):
            return rep
        print(f"[WF-Analyzer] Warnung: Kein Reporting in {cfg_path} gefunden.")
    except Exception as exc:
        print(f"[WF-Analyzer] Warnung: Konnte Standard-Reporting nicht laden: {exc}")
    return None


def _infer_meta_from_run_id(
    run_id: str,
) -> Tuple[Optional[str], Optional[int], Optional[bool]]:
    """
    Leite Richtung, Szenario und Position-Manager-Fallback aus dem Run-Ordnernamen ab.
    Regeln:
      - Enthält der Name 'long' -> direction_filter = long
      - Enthält der Name 'short' -> direction_filter = short
      - Endet der Name auf '_Z<zahl>' -> enabled_scenarios = [zahl]
      - Szenario 3 erzwingt use_position_manager=True, andere Szenarien -> False

    OPTIMIZATION: Uses pre-compiled regex pattern.
    """
    name = Path(run_id).name
    name_lower = name.lower()

    direction: Optional[str] = None
    if "long" in name_lower:
        direction = "long"
    elif "short" in name_lower:
        direction = "short"

    scenario: Optional[int] = None
    match = _RUN_ID_SCENARIO_PATTERN.search(name_lower)
    if match:
        try:
            scenario = int(match.group(1))
        except Exception:
            scenario = None

    use_pm: Optional[bool] = None
    if scenario is not None:
        use_pm = scenario == 3

    return direction, scenario, use_pm


def _upgrade_base_config(base_cfg: Dict[str, Any], *, run_id: str) -> Dict[str, Any]:
    """
    Ergänzt fehlende Felder in base_config:
      - Reporting auf Standard setzen (mean_reversion_z_score.json), falls abweichend/fehlend.
      - direction_filter, enabled_scenarios, use_position_manager aus Run-Namen ableiten,
        falls nicht im Snapshot gesetzt.
    """
    cfg = deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    if not isinstance(cfg, dict):
        print(f"[WF-Analyzer] Warnung: base_config ist kein Dict für Run {run_id}")
        return {}

    # Basis-Diagnose
    has_rep = "reporting" in cfg
    strat_snapshot = cfg.get("strategy") or {}
    params_snapshot = strat_snapshot.get("parameters") or {}
    print(
        f"[WF-Analyzer] Upgrade-Check für Run '{run_id}': "
        f"has_reporting={has_rep}, "
        f"direction_filter={params_snapshot.get('direction_filter')!r}, "
        f"enabled_scenarios={params_snapshot.get('enabled_scenarios')!r}, "
        f"use_position_manager={params_snapshot.get('use_position_manager')!r}"
    )

    # Reporting standardisieren
    standard_reporting = _standard_mean_rev_reporting()
    if standard_reporting is not None:
        current_reporting = cfg.get("reporting")
        if current_reporting != standard_reporting:
            cfg["reporting"] = deepcopy(standard_reporting)
            print(
                f"[WF-Analyzer] Reporting für Run '{run_id}' auf Standard gesetzt "
                "(mean_reversion_z_score.json)."
            )

    strat_cfg = cfg.get("strategy") or {}
    if not isinstance(strat_cfg, dict):
        strat_cfg = {}
    params = strat_cfg.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    direction_hint, scenario_hint, pm_hint = _infer_meta_from_run_id(run_id)

    if "direction_filter" not in params or params.get("direction_filter") in (None, ""):
        if direction_hint:
            params["direction_filter"] = direction_hint
            print(
                f"[WF-Analyzer] direction_filter aus Run '{run_id}' abgeleitet: "
                f"{direction_hint}"
            )

    enabled = params.get("enabled_scenarios")
    if (
        not isinstance(enabled, (list, tuple)) or not enabled
    ) and scenario_hint is not None:
        params["enabled_scenarios"] = [int(scenario_hint)]
        print(
            f"[WF-Analyzer] enabled_scenarios aus Run '{run_id}' abgeleitet: "
            f"[{scenario_hint}]"
        )

    if (
        "use_position_manager" not in params
        or params.get("use_position_manager") is None
    ):
        if pm_hint is not None:
            params["use_position_manager"] = bool(pm_hint)
            print(
                f"[WF-Analyzer] use_position_manager aus Szenario-Hinweis für Run "
                f"'{run_id}' gesetzt: {bool(pm_hint)}"
            )

    strat_cfg["parameters"] = params
    cfg["strategy"] = strat_cfg
    return cfg


def _infer_direction_from_source(base_cfg: Dict[str, Any]) -> Optional[str]:
    """
    Lese die Handelsrichtung direkt aus dem frozen_snapshot:
    erwartet wird ein Feld 'direction_filter' in strategy.parameters.

    Rückgabe:
      - 'long' oder 'short', falls gesetzt
      - None, falls nichts erkannt wird.
    """
    try:
        strat_cfg = base_cfg.get("strategy") or {}
        if not isinstance(strat_cfg, dict):
            return None
        params = strat_cfg.get("parameters") or {}
        if not isinstance(params, dict):
            return None
        raw = params.get("direction_filter")
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if s in ("long", "short"):
            return s
    except Exception:
        return None
    return None


def _infer_scenario_from_snapshot(base_cfg: Dict[str, Any]) -> Optional[int]:
    """
    Lese das Primär-Szenario direkt aus dem frozen_snapshot:
    erwartet wird ein Feld 'enabled_scenarios' (Liste von Ints) in
    strategy.parameters. Es wird das erste Element verwendet.

    Rückgabe:
      - Szenario-Nummer (z.B. 2, 3, 4, ...) oder None.
    """
    try:
        strat_cfg = base_cfg.get("strategy") or {}
        if not isinstance(strat_cfg, dict):
            return None
        params = strat_cfg.get("parameters") or {}
        if not isinstance(params, dict):
            return None
        enabled_scenarios = params.get("enabled_scenarios")
        if isinstance(enabled_scenarios, (list, tuple)) and enabled_scenarios:
            return int(enabled_scenarios[0])
    except Exception:
        return None
    return None


def _infer_use_position_manager(base_cfg: Dict[str, Any]) -> Optional[bool]:
    """
    Lese das Flag 'use_position_manager' direkt aus dem frozen_snapshot.

    Rückgabe:
      - True/False, falls gesetzt
      - None, falls nichts erkannt wird.
    """
    try:
        strat_cfg = base_cfg.get("strategy") or {}
        if not isinstance(strat_cfg, dict):
            return None
        params = strat_cfg.get("parameters") or {}
        if not isinstance(params, dict):
            return None
        raw = params.get("use_position_manager")
        if isinstance(raw, bool):
            return raw
        if raw is not None:
            return bool(raw)
    except Exception:
        return None
    return None


@lru_cache(maxsize=64)  # OPTIMIZATION: Bounded cache to prevent memory bloat
def _get_snapshot_meta_for_run(run_id: str) -> Dict[str, Any]:
    """
    Liefert Meta-Informationen (symbol, timeframe, direction,
    strategy_name, szenario) für einen Walkforward-Run basierend auf
    dem frozen_snapshot.

    Rückgabe-Keys (für die Anzeige-Spalten):
      - 'symbol'
      - 'timeframe'
      - 'direction'
      - 'strategy_name'
      - 'szenario'
    """
    meta: Dict[str, Any] = {
        "symbol": pd.NA,
        "timeframe": pd.NA,
        "direction": pd.NA,
        "strategy_name": pd.NA,
        "szenario": pd.NA,
        "session_filter": pd.NA,
        "htf_tf": pd.NA,
        "htf_filter": pd.NA,
        "extra_htf_tf": pd.NA,
        "extra_htf_filter": pd.NA,
        "time_period": pd.NA,
    }
    run_id = str(run_id or "").strip()
    if not run_id:
        return meta
    try:
        base_cfg = _load_snapshot_base_config(run_id)
    except Exception as exc:
        print(
            f"[WF-Analyzer] Konnte frozen_snapshot für Run '{run_id}' nicht laden: {exc}"
        )
        return meta

    try:
        symbol = (
            base_cfg.get("symbol")
            or ((base_cfg.get("rates") or {}).get("pairs") or [None])[0]
        )
        if symbol:
            meta["symbol"] = str(symbol)
    except Exception:
        pass

    try:
        tf_cfg = base_cfg.get("timeframes") or {}
        primary_tf = tf_cfg.get("primary") or (base_cfg.get("rates") or {}).get(
            "timeframe"
        )
        if primary_tf:
            meta["timeframe"] = str(primary_tf)
    except Exception:
        pass

    try:
        strat_cfg = base_cfg.get("strategy") or {}
        module = str(strat_cfg.get("module") or "")
        cls_name = str(strat_cfg.get("class") or "")
        strategy_name = ""
        if cls_name:
            strategy_name = cls_name
        elif module:
            strategy_name = module.split(".")[-1]
        if strategy_name:
            meta["strategy_name"] = strategy_name
    except Exception:
        pass

    # Session-Filter: nur Zeitfenster als klar getrennte Strings
    try:
        sess = base_cfg.get("session_filter")
        if isinstance(sess, dict):
            fixed = sess.get("fixed_times") or []
            parts: List[str] = []
            for item in fixed:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                start = str(item[0]).strip()
                end = str(item[1]).strip()
                if not start and not end:
                    continue
                parts.append(f"{start}-{end}")
            if parts:
                # Beispiel: "23:00-05:00; 15:00-19:00"
                meta["session_filter"] = "; ".join(parts)
        elif sess is not None:
            # Fallback: einfache String-Repräsentation, falls Struktur unerwartet ist
            meta["session_filter"] = str(sess)
    except Exception:
        pass

    # HTF-/Extra-HTF-Settings aus den Strategy-Parametern
    try:
        strat_cfg = base_cfg.get("strategy") or {}
        params = strat_cfg.get("parameters") or {}
        if isinstance(params, dict):
            htf_tf_raw = params.get("htf_tf")
            htf_filter_raw = params.get("htf_filter")
            extra_htf_tf_raw = params.get("extra_htf_tf")
            extra_htf_filter_raw = params.get("extra_htf_filter")

            def _norm_filter(val: Any) -> Optional[str]:
                if val is None:
                    return None
                s = str(val).strip()
                if not s:
                    return None
                if s.lower() in ("none", "null"):
                    return None
                return s

            htf_filter_norm = _norm_filter(htf_filter_raw)
            extra_htf_filter_norm = _norm_filter(extra_htf_filter_raw)

            if htf_filter_norm is not None:
                meta["htf_filter"] = htf_filter_norm
                if htf_tf_raw is not None:
                    meta["htf_tf"] = str(htf_tf_raw)
            # Wenn Filter leer/None ist, bleibt sowohl htf_tf als auch Filter auf NA

            if extra_htf_filter_norm is not None:
                meta["extra_htf_filter"] = extra_htf_filter_norm
                if extra_htf_tf_raw is not None:
                    meta["extra_htf_tf"] = str(extra_htf_tf_raw)
            # Gleiches Verhalten für extra_htf_*: bei fehlendem Filter bleiben beide NA
    except Exception:
        pass

    # Zeitperiode
    try:
        start_date = base_cfg.get("start_date")
        end_date = base_cfg.get("end_date")
        if start_date and end_date:
            meta["time_period"] = f"{start_date} bis {end_date}"
    except Exception:
        pass

    try:
        direction = _infer_direction_from_source(base_cfg)
        if direction:
            meta["direction"] = str(direction)
    except Exception:
        pass

    try:
        scenario = _infer_scenario_from_snapshot(base_cfg)
        if scenario is not None:
            meta["szenario"] = int(scenario)
    except Exception:
        pass

    return meta


def _build_param_map_from_combined(
    base_df: Optional[pd.DataFrame],
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Erzeuge ein Mapping combo_key -> Parameter-Spalten basierend auf combined_base.

    Gibt zurück:
      - Liste der Parameter-Spaltennamen
      - DataFrame mit Index combo_key und genau diesen Spalten (oder None)
    """
    param_cols: List[str] = []
    param_map: Optional[pd.DataFrame] = None

    base_source: Optional[pd.DataFrame] = None
    if base_df is not None and not base_df.empty:
        base_source = base_df
    else:
        try:
            if BASE_COMBINED_CSV.exists():
                base_source = pd.read_csv(BASE_COMBINED_CSV)
        except Exception as exc:
            print(f"Warnung: Konnte {BASE_COMBINED_CSV} nicht lesen: {exc}")
            base_source = None

    if base_source is None or base_source.empty:
        return param_cols, param_map

    base_norm = base_source.copy()
    base_norm.columns = [str(c).strip() for c in base_norm.columns]
    if "combo_key" not in base_norm.columns:
        return param_cols, param_map

    year_prefix = re.compile(r"^(\d{4})\s")
    metric_cols_known = {
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
        "score",
        "stability",
        "stability_score",
        "wmape",
        "risk_adjusted",
        "profit_component",
        "cost_robustness",
        "robustness_penalty",
        "robustness_score",
        "robustness_score_stress",
        "p_mean_r_gt_0",
        "p_net_profit_gt_0",
        # zusätzliche Robustness/Stresstest-Metriken aus den kombinierten CSVs
        "robustness_score_1",
        "cost_shock_score",
        "timing_jitter_score",
        "trade_dropout_score",
    }
    try:
        raw_param_candidates = set(_parameter_columns(base_norm))
    except Exception:
        raw_param_candidates = set()
    param_candidates_all = {
        c
        for c in raw_param_candidates
        if (
            c not in ("combo_key", "source_walkforward")
            and str(c) not in metric_cols_known
            and not year_prefix.match(str(c))
        )
    }
    param_cols = [c for c in base_norm.columns if c in param_candidates_all]
    if not param_cols:
        return param_cols, None
    try:
        param_map = base_norm.set_index("combo_key")[param_cols]
    except Exception as exc:
        print(
            "Warnung: Konnte Parameter-Mapping aus combined_base.csv nicht "
            f"aufbauen: {exc}"
        )
        param_cols = []
        param_map = None
    return param_cols, param_map


def _build_config_for_combo(
    combo_key: str,
    source_walkforward: str,
    param_map: Optional[pd.DataFrame],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Rekonstruiert eine Backtest-Config für eine einzelne Walkforward-Kombination.

    - Lädt base_config aus dem passenden frozen_snapshot.json.
    - Überschreibt Strategie-Parameter gemäß combined_base (param_map).
    - Setzt Richtung, Szenario und Position Manager basierend auf dem frozen_snapshot:
        * Richtung: aus direction_filter in strategy.parameters.
        * Szenario: erstes Element von enabled_scenarios (falls vorhanden).
        * Position Manager: aus use_position_manager in strategy.parameters.
    - Erzwingt die verfeinerten Robustness-Settings (jitter_frac/jitter_repeats).

    Gibt (config, meta) zurück, wobei meta Informationen zu Richtung/Szenario/PM enthält.
    """
    base_cfg = _load_snapshot_base_config(source_walkforward)
    cfg = deepcopy(base_cfg)

    # Sicherstellen, dass Strategy/Parameters-Struktur vorhanden ist
    strat_cfg = cfg.get("strategy") or {}
    if not isinstance(strat_cfg, dict):
        strat_cfg = {}

    # Strategy-Modulnamen normalisieren:
    # Auf manchen Servern werden z.B. Kopien des Strategie-Ordners mit
    # Suffixen wie "_4" angelegt (mean_reversion_z_score_4), was dazu
    # führt, dass das Modul in dieser Umgebung nicht importiert werden kann.
    # Hier wird speziell mean_reversion_z_score_X wieder auf
    # mean_reversion_z_score gemappt.
    module_path = str(strat_cfg.get("module") or "")
    if module_path:
        core = module_path
        prefix = ""
        if core.startswith("strategies."):
            prefix = "strategies."
            core = core[len(prefix) :]
        first, *rest_parts = core.split(".", 1)
        rest = rest_parts[0] if rest_parts else ""
        if re.match(r"^mean_reversion_z_score_\d+$", first):
            first = "mean_reversion_z_score"
            core = first + (f".{rest}" if rest else "")
            strat_cfg["module"] = prefix + core if prefix else core

    params = strat_cfg.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    # Parameter aus combined_base anwenden (falls vorhanden)
    if param_map is not None and combo_key in param_map.index:
        row = param_map.loc[combo_key]
        try:
            items_iter = row.items()
        except AttributeError:
            items_iter = []
        for name, value in items_iter:
            if pd.isna(value):
                continue
            params[str(name)] = value

    # Richtung direkt aus dem frozen_snapshot (direction_filter) ableiten
    direction = _infer_direction_from_source(base_cfg)
    if direction:
        params["direction_filter"] = direction

    # Szenario direkt aus dem frozen_snapshot ableiten
    scenario_val: Optional[int] = _infer_scenario_from_snapshot(base_cfg)

    # use_position_manager direkt aus dem frozen_snapshot lesen
    upm_snapshot = _infer_use_position_manager(base_cfg)
    if upm_snapshot is None:
        # Fallback: vorhandenen Wert aus params respektieren, sonst False
        upm_snapshot = bool(params.get("use_position_manager", False))
    use_position_manager = bool(upm_snapshot)
    params["use_position_manager"] = use_position_manager

    strat_cfg["parameters"] = params
    cfg["strategy"] = strat_cfg

    # Robustness-Settings für verfeinerte Runs erzwingen
    # WICHTIG: Hard-Override statt setdefault(), da frozen_snapshot.json
    # ggf. alte Werte (z.B. robust_jitter_repeats=0) enthält, die sonst
    # über _resolve_robust_setting() Vorrang bekommen würden.
    rep = cfg.get("reporting") or {}
    if not isinstance(rep, dict):
        rep = {}
    rep["enable_backtest_robust_metrics"] = True
    # Nur Robustness 1 für die Walkforward-Re-Evals berechnen, um Laufzeit zu sparen
    rep["robust_metrics_mode"] = "r1_only"
    rep["jitter_frac"] = REFINED_ROBUST_JITTER_FRAC
    rep["robust_jitter_frac"] = REFINED_ROBUST_JITTER_FRAC
    rep["robust_jitter_repeats"] = REFINED_ROBUST_JITTER_REPEATS
    rep["jitter_repeats"] = REFINED_ROBUST_JITTER_REPEATS
    cfg["reporting"] = rep
    cfg["enable_backtest_robust_metrics"] = True
    cfg["robust_jitter_frac"] = REFINED_ROBUST_JITTER_FRAC
    cfg["robust_jitter_repeats"] = REFINED_ROBUST_JITTER_REPEATS

    # Auch auf Strategie-Parameter spiegeln (für _resolve_robust_setting)
    # Hard-Override für konsistente Übersteuerung
    params["jitter_frac"] = REFINED_ROBUST_JITTER_FRAC
    params["robust_jitter_frac"] = REFINED_ROBUST_JITTER_FRAC
    params["robust_jitter_repeats"] = REFINED_ROBUST_JITTER_REPEATS
    params["jitter_repeats"] = REFINED_ROBUST_JITTER_REPEATS
    strat_cfg["parameters"] = params
    cfg["strategy"] = strat_cfg

    meta = {
        "direction": direction,
        "scenario": scenario_val,
        "use_position_manager": use_position_manager,
    }
    return cfg, meta


def _compute_true_robustness_1(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Führt einen Backtest für die gegebene Config aus und berechnet
    die exakten Robustness-Metriken via _compute_backtest_robust_metrics.

    Rückgabe:
      - robustness_1 (float)
      - cost_shock_score (float)
      - timing_jitter_score (float)
      - trade_dropout_score (float)
      - p_mean_gt (float)
      - stability_score (float)
    """
    strat_cfg = config.get("strategy") or {}
    params = strat_cfg.get("parameters") or {}
    direction = params.get("direction_filter")
    enabled_scenarios = params.get("enabled_scenarios")
    try:
        jitter_frac = float(
            config.get("robust_jitter_frac", params.get("robust_jitter_frac", np.nan))
        )
    except Exception:
        jitter_frac = np.nan
    try:
        jitter_repeats = int(
            config.get("robust_jitter_repeats", params.get("robust_jitter_repeats", -1))
        )
    except Exception:
        jitter_repeats = -1
    print(
        "[WF-Analyzer] Starte Robustness-Backtest "
        f"(direction={direction}, enabled_scenarios={enabled_scenarios}, "
        f"jitter_frac={jitter_frac}, jitter_repeats={jitter_repeats})"
    )
    portfolio, _ = run_backtest_and_return_portfolio(config)
    metrics = _compute_backtest_robust_metrics(config, portfolio)
    print(
        "[WF-Analyzer] Robustness-Backtest fertig: "
        f"robustness_1={metrics.get('robustness_1')}, "
        f"robustness_1_num_samples={metrics.get('robustness_1_num_samples')}, "
        f"data_jitter_score={metrics.get('data_jitter_score')}, "
        f"data_jitter_num_samples={metrics.get('data_jitter_num_samples')}, "
        f"cost_shock_score={metrics.get('cost_shock_score')}, "
        f"timing_jitter_score={metrics.get('timing_jitter_score')}, "
        f"trade_dropout_score={metrics.get('trade_dropout_score')}, "
        f"p_mean_gt={metrics.get('p_mean_gt')}, "
        f"stability_score={metrics.get('stability_score')}, "
        f"ulcer_index={metrics.get('ulcer_index')}, "
        f"ulcer_index_score={metrics.get('ulcer_index_score')}"
    )
    return {
        "robustness_1": float(metrics.get("robustness_1", 0.0) or 0.0),
        "robustness_1_num_samples": int(
            metrics.get("robustness_1_num_samples", 0) or 0
        ),
        "data_jitter_score": float(metrics.get("data_jitter_score", 0.0) or 0.0),
        "data_jitter_num_samples": int(metrics.get("data_jitter_num_samples", 0) or 0),
        "cost_shock_score": float(metrics.get("cost_shock_score", 0.0) or 0.0),
        "timing_jitter_score": float(metrics.get("timing_jitter_score", 0.0) or 0.0),
        "trade_dropout_score": float(metrics.get("trade_dropout_score", 0.0) or 0.0),
        "p_mean_gt": float(metrics.get("p_mean_gt", 0.0) or 0.0),
        "stability_score": float(metrics.get("stability_score", 0.0) or 0.0),
        "ulcer_index": float(metrics.get("ulcer_index", np.nan) or np.nan),
        "ulcer_index_score": float(metrics.get("ulcer_index_score", 0.0) or 0.0),
    }


def _compute_true_robustness_1_task(
    payload: Tuple[str, str, Dict[str, Any]],
) -> Tuple[str, str, Dict[str, float]]:
    """
    Helper für parallele Ausführung: nimmt (combo_key, source_walkforward, config),
    berechnet Robustness-Metriken und gibt (combo_key, source_walkforward, metrics) zurück.

    Wichtig: Nach der Berechnung werden globale Caches explizit geleert, um
    Memory-Leaks in Worker-Prozessen zu vermeiden (diese Caches werden nicht
    an den Hauptprozess zurückgegeben und akkumulieren sonst im Worker-Speicher).
    """
    combo_key, source_walkforward, cfg = payload
    try:
        metrics = _compute_true_robustness_1(cfg)
        return combo_key, source_walkforward, metrics
    finally:
        # Explizite Cache-Bereinigung nach jedem Task um Memory-Leaks zu verhindern
        try:
            from backtest_engine.runner import clear_alignment_cache

            clear_alignment_cache(keep_last=0)
        except Exception:
            pass
        try:
            from backtest_engine.core.indicator_cache import clear_indicator_cache_pool

            clear_indicator_cache_pool()
        except Exception:
            pass


def _refine_top50_with_robustness(
    final_pairs: pd.DataFrame,
    combined_for_export: pd.DataFrame,
    top_n: int = 50,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Re-evaluiert die Top-N Paar-Kombinationen mit einem harten Robustness-Setup:

      - jitter_repeats = 80
      - jitter_frac = 0.05

    Für jedes Paar werden beide Legs separat mit diesen Einstellungen
    backgetestet, Robustness 1 berechnet, der Mittelwert der beiden Legs
    gebildet und anschließend ein verfeinerter Gesamt-Score:

        comp_score_final_refined = (comp_score_final + mean(robustness_1_leg_A/B)) / 2

    Die Ausgabe enthält zusätzliche Spalten:
      - robustness_1_leg_A/B
      - robustness_1_mean
      - comp_score_final_refined
      - robust_direction_A/B
      - robust_scenario_A/B
      - robust_use_position_manager_A/B
    """
    if final_pairs is None or final_pairs.empty:
        return final_pairs

    base = final_pairs.copy()
    base = base.sort_values("comp_score_final", ascending=False).reset_index(drop=True)
    top = base.head(top_n).copy()
    if top.empty:
        return top

    # Parameter-Mapping aus combined_base aufbauen
    param_cols, param_map = _build_param_map_from_combined(combined_for_export)
    if param_map is None or not param_cols:
        print("Warnung: Konnte kein Parameter-Mapping für refined Top-50 aufbauen.")

    # Ergebnis-Spalten vorbereiten
    top["robustness_1_leg_A"] = np.nan
    top["robustness_1_leg_B"] = np.nan
    top["robustness_1_mean"] = np.nan
    top["comp_score_final_refined"] = np.nan

    top["robust_direction_A"] = pd.NA
    top["robust_direction_B"] = pd.NA
    top["robust_scenario_A"] = pd.NA
    top["robust_scenario_B"] = pd.NA
    top["robust_use_position_manager_A"] = pd.NA
    top["robust_use_position_manager_B"] = pd.NA

    # Einmal pro einzigartiger combo_key+run Robustness berechnen (optional parallel)
    work_items: List[Tuple[str, str, Dict[str, Any]]] = []
    meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_keys: set[Tuple[str, str]] = set()

    for _, row in top.iterrows():
        for _, suffix in (("A", "1"), ("B", "2")):
            combo_key_col = f"combo_key_{suffix}"
            src_col = f"source_walkforward_{suffix}"
            combo_key = str(row.get(combo_key_col, "")).strip()
            src = str(row.get(src_col, "")).strip()
            if not combo_key or not src:
                continue
            key = (combo_key, src)
            if key in seen_keys:
                continue
            try:
                cfg, meta = _build_config_for_combo(combo_key, src, param_map)
            except Exception as exc:
                print(
                    f"Warnung: Konnte Config für combo_key={combo_key} "
                    f"(run={src}) nicht aufbauen: {exc}"
                )
                continue
            seen_keys.add(key)
            meta_cache[key] = meta
            work_items.append((combo_key, src, cfg))

    robust_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    total_tasks = len(work_items)
    if total_tasks:
        if max_workers is None:
            cpu_workers = os.cpu_count() or 1
            worker_count = max(1, min(cpu_workers, total_tasks))
        else:
            worker_count = max(1, min(int(max_workers), total_tasks))

        if worker_count <= 1:
            for payload in work_items:
                try:
                    ck, src, metrics = _compute_true_robustness_1_task(payload)
                    robust_cache[(ck, src)] = metrics
                except Exception as exc:
                    print(
                        f"Warnung: Robustness-Berechnung für combo_key={payload[0]} "
                        f"(run={payload[1]}) fehlgeschlagen: {exc}"
                    )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(_compute_true_robustness_1_task, payload): (
                        payload[0],
                        payload[1],
                    )
                    for payload in work_items
                }
                for fut in as_completed(future_map):
                    ck, src = future_map[fut]
                    try:
                        ck_res, src_res, metrics = fut.result()
                        robust_cache[(ck_res, src_res)] = metrics
                    except Exception as exc:
                        print(
                            f"Warnung: Robustness-Task combo_key={ck} (run={src}) fehlgeschlagen: {exc}"
                        )

    # Ergebnisse je Paar/Leg einfügen (aus Cache)
    for idx, row in top.iterrows():
        robustness_vals: Dict[str, float] = {}
        num_samples_vals: Dict[str, int] = {}
        meta_vals: Dict[str, Dict[str, Any]] = {}

        for leg_label, suffix in (("A", "1"), ("B", "2")):
            combo_key_col = f"combo_key_{suffix}"
            src_col = f"source_walkforward_{suffix}"
            combo_key = str(row.get(combo_key_col, "")).strip()
            src = str(row.get(src_col, "")).strip()
            if not combo_key or not src:
                continue
            key = (combo_key, src)
            metrics = robust_cache.get(key)
            if metrics:
                robustness_vals[leg_label] = metrics.get("robustness_1", 0.0)
                num_samples_vals[leg_label] = int(
                    metrics.get("robustness_1_num_samples", 0) or 0
                )
            meta = meta_cache.get(key)
            if meta:
                meta_vals[leg_label] = meta

        if "A" in robustness_vals:
            top.at[idx, "robustness_1_leg_A"] = robustness_vals["A"]
        if "B" in robustness_vals:
            top.at[idx, "robustness_1_leg_B"] = robustness_vals["B"]
        # Sample counts für Observability
        if "A" in num_samples_vals:
            top.at[idx, "robustness_1_num_samples_leg_A"] = num_samples_vals["A"]
        if "B" in num_samples_vals:
            top.at[idx, "robustness_1_num_samples_leg_B"] = num_samples_vals["B"]

        # Mittelwert, nur über vorhandene Legs
        if robustness_vals:
            mean_val = float(sum(robustness_vals.values()) / len(robustness_vals))
            top.at[idx, "robustness_1_mean"] = mean_val
            orig = float(row.get("comp_score_final", 0.0) or 0.0)
            top.at[idx, "comp_score_final_refined"] = (orig + mean_val) / 2.0

        # Meta-Spalten setzen
        for leg_label, suffix in (("A", "1"), ("B", "2")):
            meta = meta_vals.get(leg_label)
            if not meta:
                continue
            top.at[idx, f"robust_direction_{leg_label}"] = meta.get("direction", None)
            top.at[idx, f"robust_scenario_{leg_label}"] = meta.get("scenario", None)
            top.at[idx, f"robust_use_position_manager_{leg_label}"] = bool(
                meta.get("use_position_manager", False)
            )

    # Nach verfeinertem Score sortieren, Fallback auf ursprünglichen Score
    comp_ref = pd.to_numeric(
        top.get("comp_score_final_refined", np.nan), errors="coerce"
    )
    comp_orig = pd.to_numeric(
        top.get("comp_score_final", np.nan), errors="coerce"
    ).fillna(0.0)
    sort_key = comp_ref.where(comp_ref.notna(), comp_orig)
    top = top.assign(_sort_key=sort_key)
    top = top.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
    top = top.reset_index(drop=True)
    return top


def run_walkforward_analysis(
    *,
    root: Path = WALKFORWARD_ROOT,
    save_base_combined: bool = True,
    refine_top50: bool = False,
    robust_workers: Optional[int] = None,
    enable_clustering: bool = False,
    cluster_method: str = "hdbscan",
    cluster_min_size: int = 5,
    cluster_min_samples: Optional[int] = 1,
    cluster_metric: str = "euclidean",
    cluster_allow_noise: bool = True,
    cluster_interval: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = _find_final_scores_files(root)
    if not files:
        print(f"Keine 05_final_scores_combined.csv Dateien unter '{root}' gefunden.")
        return pd.DataFrame(), pd.DataFrame()
    print(f"Gefundene Walkforward-Files: {len(files)}")
    combined, param_order = _load_and_combine(files)
    if combined.empty:
        print("Kombinierter DataFrame ist leer.")
        return combined, pd.DataFrame()
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    # Spalten so anordnen, dass Parameter als Block zusammenhängen
    combined_for_export = _reorder_with_parameter_block(
        combined,
        meta_cols=("combo_id", "source_walkforward", "combo_key"),
        param_order=param_order,
    )
    if save_base_combined:
        combined_for_export.to_csv(BASE_COMBINED_CSV, index=False)
        print(f"Basis-CSV gespeichert unter: {BASE_COMBINED_CSV}")

    yearly = _detect_yearly_columns(combined)
    if not yearly.years:
        print("Konnte keine jährlichen Spalten erkennen. Abbruch.")
        return combined, pd.DataFrame()
    # Hardgates
    after_gates = _apply_hard_gates(combined, yearly)
    if after_gates.empty:
        print("Keine Kandidaten nach Hardgates übrig.")
        return combined, pd.DataFrame()
    # Sterne pro Einzel-Kombination
    singles_with_stars, years = _add_star_ratings(after_gates, yearly)
    if singles_with_stars.empty:
        print("Keine Kandidaten nach Stern-Filter übrig.")
        return combined, pd.DataFrame()
    singles_lookup = (
        singles_with_stars.set_index("combo_key")
        if not singles_with_stars.empty
        else None
    )
    # Paare bilden
    pairs = _generate_pairs(singles_with_stars, years)
    if pairs.empty:
        print("Keine gültigen Paar-Kombinationen gefunden.")
        return combined, pd.DataFrame()
    # Kombinierte Jahreskennzahlen
    pairs_metrics = _compute_pair_yearly_metrics(pairs, singles_with_stars, yearly)
    if pairs_metrics.empty:
        print("Fehler beim Berechnen kombinierter Jahresmetriken.")
        return combined, pd.DataFrame()
    # Sterne für kombinierte Kombinationen + Filter
    pairs_stars = _add_combined_star_ratings(pairs_metrics, years)
    if pairs_stars.empty:
        print("Keine Paar-Kandidaten nach kombinierter Stern-Logik übrig.")
        return combined, pd.DataFrame()
    # Composite Scores
    pairs_comp = _add_yearly_composite_scores(pairs_stars, years)
    # Stability Score über __compute_yearly_stability
    pairs_stab = _add_stability_scores_for_pairs(pairs_comp, years)
    # Finaler Score + Sortierung
    final_pairs = _add_final_score_and_sort(pairs_stab)
    if final_pairs.empty:
        print("Finale Kombinationen-Übersicht ist leer.")
        return combined, final_pairs

    # Optional: Diversifizierendes Clustering vor Top-Listen
    final_for_export = final_pairs
    if enable_clustering:
        if cluster_method.lower() == "interval":
            # Alternative Methodik: Metrische Toleranz-Intervalle
            print(
                f"[Cluster] Verwende Methode: Interval (±{cluster_interval*100:.0f}%)"
            )
            labels = _cluster_metric_tolerance(
                final_pairs,
                years,
                interval=cluster_interval,
                min_cluster_size=cluster_min_size,
            )
            if labels is None or len(labels) == 0:
                print("[Cluster] Keine Clusterlabels erhalten. Überspringe Clustering.")
            else:
                final_for_export = _select_cluster_representatives(
                    final_pairs,
                    labels,
                    singles_lookup=singles_lookup,
                    ignore_noise=not bool(cluster_allow_noise),
                )
                print(
                    f"[Cluster] Repräsentanten ausgewählt: "
                    f"{len(final_for_export)}/{len(final_pairs)} Kandidaten"
                )

        elif cluster_method.lower() == "hdbscan":
            # Bestehende Methodik: HDBSCAN
            print(f"[Cluster] Verwende Methode: HDBSCAN")
            feat_df, feat_cols = _build_year_profile_features(final_pairs, years)
            if feat_df.empty or not feat_cols:
                print(
                    "[Cluster] Keine Feature-Matrix aufgebaut. Überspringe Clustering."
                )
            else:
                scaled_feats, medians, mads = _robust_scale_features(feat_df)
                if scaled_feats.empty:
                    print(
                        "[Cluster] Skaliertes Feature-Set ist leer. Überspringe Clustering."
                    )
                else:
                    np.random.seed(42)
                    labels, probs, outlier = _cluster_hdbscan(
                        scaled_feats,
                        min_cluster_size=cluster_min_size,
                        min_samples=cluster_min_samples,
                        metric=cluster_metric,
                        allow_single_cluster=False,
                        allow_noise=True,
                    )
                    if labels is None:
                        print(
                            "[Cluster] Keine Clusterlabels erhalten. Überspringe Clustering."
                        )
                    else:
                        cluster_series = pd.Series(labels)
                        cluster_counts = cluster_series.value_counts().sort_index()
                        print(
                            f"[Cluster] HDBSCAN Label-Verteilung: "
                            f"{cluster_counts.to_dict()}"
                        )
                        final_for_export = _select_cluster_representatives(
                            final_pairs,
                            labels,
                            singles_lookup=singles_lookup,
                            ignore_noise=not bool(cluster_allow_noise),
                        )
                        print(
                            f"[Cluster] Repräsentanten ausgewählt: "
                            f"{len(final_for_export)}/{len(final_pairs)} Kandidaten"
                        )
        else:
            print(
                f"[Cluster] Unbekannte Methode '{cluster_method}'. Verfügbar: hdbscan, interval"
            )

    # robust_stress_score für alle Kandidaten berechnen (für Rankings/Toplisten)
    if final_for_export is not None and not final_for_export.empty:
        final_for_export = final_for_export.copy()
        robust_series = pd.Series(
            (
                _compute_robust_stress_score(row, singles_lookup)
                for _, row in final_for_export.iterrows()
            ),
            index=final_for_export.index,
            dtype=float,
        )
        final_for_export["robust_stress_score"] = robust_series

    def _top_unique_by_metric(
        df: pd.DataFrame,
        metric: str,
        n: int,
        *,
        exclude_keys: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty or metric not in df.columns:
            return pd.DataFrame(columns=df.columns)
        ranked = df.copy()
        ranked["__metric_sort"] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=["__metric_sort"])
        if exclude_keys:
            exclude_set = {str(k) for k in exclude_keys if pd.notna(k)}
            ranked = ranked[~ranked["combo_pair_id"].astype(str).isin(exclude_set)]
        ranked = ranked.sort_values("__metric_sort", ascending=False)
        ranked = ranked.drop_duplicates(subset="combo_pair_id", keep="first")
        result = ranked.head(n).drop(columns=["__metric_sort"], errors="ignore")
        return result

    # Top-Listen exportieren (Baseline-Variante)
    final_pairs_for_export = _reorder_pair_output_for_export(final_for_export, years)
    comp_top100 = _top_unique_by_metric(final_pairs_for_export, "comp_score_final", 50)
    comp_keys_100 = set(comp_top100.get("combo_pair_id", []))
    robust_top100 = _top_unique_by_metric(
        final_pairs_for_export, "robust_stress_score", 50, exclude_keys=comp_keys_100
    )
    top100_pairs = pd.concat([comp_top100, robust_top100], ignore_index=True)

    comp_top50 = _top_unique_by_metric(final_pairs_for_export, "comp_score_final", 25)
    comp_keys_50 = set(comp_top50.get("combo_pair_id", []))
    robust_top50 = _top_unique_by_metric(
        final_pairs_for_export, "robust_stress_score", 25, exclude_keys=comp_keys_50
    )
    top50_pairs = pd.concat([comp_top50, robust_top50], ignore_index=True)

    # Anzeige-Variante mit 2 Zeilen pro Paar + Leerzeile
    # (ohne Leg-Robustness-Spalte in den Basis-Toplisten)
    top100_display = _expand_pairs_for_display(
        top100_pairs, years, base_df=combined_for_export, include_leg_robustness=False
    )
    top50_display = _expand_pairs_for_display(
        top50_pairs, years, base_df=combined_for_export, include_leg_robustness=False
    )

    top100_display.to_csv(TOP100_CSV, index=False)
    top50_display.to_csv(TOP50_CSV, index=False)
    print(f"Top 100 gespeichert unter: {TOP100_CSV}")
    print(f"Top 50 gespeichert unter: {TOP50_CSV}")

    # Optionale, verfeinerte Top-50-Liste mit erneuter Robustness-Bewertung
    if refine_top50:
        refined_raw = _refine_top50_with_robustness(
            final_for_export,
            combined_for_export,
            top_n=50,
            max_workers=robust_workers,
        )
        if refined_raw is not None and not refined_raw.empty:
            refined_for_export = _reorder_pair_output_for_export(refined_raw, years)
            refined_display = _expand_pairs_for_display(
                refined_for_export,
                years,
                base_df=combined_for_export,
                include_leg_robustness=True,
            )
            refined_display.to_csv(TOP50_REFINED_CSV, index=False)
            print(f"Top 50 (refined) gespeichert unter: {TOP50_REFINED_CSV}")

    return combined, final_for_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analysiert Walkforward-Ergebnisse und erzeugt finale kombinierte Kombinations-Scores."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(WALKFORWARD_ROOT),
        help="Root-Verzeichnis der Walkforward-Ergebnisse (Standard: var/results/analysis)",
    )
    parser.add_argument(
        "--no-save-base",
        action="store_true",
        help="combined_base.csv nicht speichern.",
    )
    parser.add_argument(
        "--refine-top50",
        action="store_true",
        help=(
            "Top-50-Kombinationen mit Backtest-Robustness (jitter_repeats=80, "
            "jitter_frac=0.05) neu bewerten und top_50_walkforward_combos_refined.csv schreiben."
        ),
    )
    parser.add_argument(
        "--robust-workers",
        type=int,
        default=None,
        help="Anzahl paralleler Prozesse für Robustness-Runs (Default: 1).",
    )
    parser.add_argument(
        "--cluster-toplists",
        action="store_true",
        help="Aktiviere diversifizierendes Clustering vor den Top-Listen.",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "interval"],
        help="Clustering-Methode: 'hdbscan' (HDBSCAN-basiert) oder 'interval' (metrische Intervall-Cluster). Default: hdbscan",
    )
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=5,
        help="Minimale Cluster-Größe (Default: 5 für HDBSCAN, 3 für Interval).",
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=1,
        help="min_samples für HDBSCAN (Default: 1).",
    )
    parser.add_argument(
        "--cluster-metric",
        type=str,
        default="euclidean",
        help="Distanzmetrik für HDBSCAN (Default: euclidean).",
    )
    parser.add_argument(
        "--cluster-allow-noise",
        action="store_true",
        help="Noise-Cluster (-1) in der Repräsentanten-Auswahl berücksichtigen (Default: ignorieren).",
    )
    parser.add_argument(
        "--cluster-interval",
        type=float,
        default=0.20,
        help="Intervall für 'interval' Methode in Prozent (Default: 0.20 = ±20%%).",
    )
    args = parser.parse_args()
    root = Path(args.root)
    save_base = not args.no_save_base
    do_refine = bool(getattr(args, "refine_top50", False))
    robust_workers = getattr(args, "robust_workers", None)
    run_walkforward_analysis(
        root=root,
        save_base_combined=save_base,
        refine_top50=do_refine,
        robust_workers=robust_workers,
        enable_clustering=bool(getattr(args, "cluster_toplists", False)),
        cluster_method=str(getattr(args, "cluster_method", "hdbscan")),
        cluster_min_size=int(getattr(args, "cluster_min_size", 5)),
        cluster_min_samples=getattr(args, "cluster_min_samples", 1),
        cluster_metric=str(getattr(args, "cluster_metric", "euclidean")),
        cluster_allow_noise=bool(getattr(args, "cluster_allow_noise", False)),
        cluster_interval=float(getattr(args, "cluster_interval", 0.20)),
    )


if __name__ == "__main__":
    main()
