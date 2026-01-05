# hf_engine/backtester/optimizer/robust_zone_analyzer.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Non-interactive plotting for artifact images
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_engine.optimizer.instrumentation import StageRecorder

# ---------- Public API ---------------------------------------------------------


def run_robust_zone_analysis(
    walkforward_root: str,
    param_grid: Dict[str, Dict[str, Any]],
    *,
    analyze_alpha: float = 0.10,
    analyze_min_coverage: float = 0.10,
    analyze_min_sharpe_trade: float = 0.50,
    metrics_weights: Optional[Dict[str, float]] = None,
    enable_plots: bool = True,
    plots_dirname: str = "figures",
    recorder: Optional[StageRecorder] = None,
) -> Tuple[Path, StageRecorder]:
    """
    Main entry: runs Steps 1..5 and exports artifacts for full traceability.
    Returns a tuple with the report path and the populated StageRecorder.
    """
    root = Path(walkforward_root)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir: Optional[Path] = (out_dir / plots_dirname) if enable_plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    if recorder is None:
        rec = StageRecorder(
            scope="robust_zone_analysis",
            metadata={
                "walkforward_root": str(root),
                "analysis_dir": str(out_dir),
                "plots_dir": str(plots_dir) if plots_dir is not None else None,
            },
        )
    else:
        rec = recorder
        rec.add_metadata(
            walkforward_root=str(root),
            analysis_dir=str(out_dir),
            plots_dir=str(plots_dir) if plots_dir is not None else None,
        )
    rec.add_metadata(
        analyze_alpha=analyze_alpha,
        analyze_min_coverage=analyze_min_coverage,
        analyze_min_sharpe_trade=analyze_min_sharpe_trade,
        plots_enabled=bool(enable_plots),
    )

    walkforward_options = _load_walkforward_options(root)
    dynamic_trade_threshold = _compute_min_total_trades_threshold(walkforward_options)
    rec.add_metadata(
        walkforward_options=walkforward_options,
        dynamic_trade_threshold=dynamic_trade_threshold,
    )

    # ---- Load OOS data (preferred: consolidated); else per-window CSVs
    with rec.stage("load_oos_dataset") as stage:
        df_raw, source_info = _load_oos_dataset(root)
        stage.add_details(
            rows=int(len(df_raw)),
            columns=int(len(df_raw.columns)) if not df_raw.empty else 0,
            source_used=source_info.get("used"),
            file_count=len(source_info.get("files", [])),
        )

    # STEP 1: Clean & round
    with rec.stage("step1_clean_and_round") as stage:
        df_clean, step1_meta = _step1_clean_and_round(df_raw, param_grid)
        clean_path = out_dir / "01_cleaned.parquet"
        summary_path = out_dir / "01_clean_summary.json"
        _save_df(df_clean, clean_path)
        _save_json(step1_meta, summary_path)
        stage.add_details(
            step1_meta,
            rows_out=int(len(df_clean)),
            columns=int(len(df_clean.columns)) if not df_clean.empty else 0,
            artifact_clean=str(clean_path),
            artifact_summary=str(summary_path),
        )

    # STEP 2: Hard Gates (exactly as specified)
    with rec.stage("step2_apply_hard_gates") as stage:
        df_filt, step2_meta = _step2_apply_hard_gates(
            df_clean, min_total_trades_threshold=dynamic_trade_threshold
        )
        filt_path = out_dir / "02_filtered.parquet"
        gates_path = out_dir / "02_gates_summary.json"
        _save_df(df_filt, filt_path)
        _save_json(step2_meta, gates_path)
        _plot_histograms_step2(df_filt, plots_dir)
        stage.add_details(
            step2_meta,
            rows_out=int(len(df_filt)),
            artifact_filtered=str(filt_path),
            artifact_summary=str(gates_path),
            plots_dir=str(plots_dir) if plots_dir is not None else None,
        )

    # Quick exit: if nothing left, fallback -> full intervals
    if df_filt.empty:
        with rec.stage("fallback_full_intervals") as stage:
            zones_fallback = _fallback_full_intervals(
                param_grid, reason="No rows after gates"
            )
            fallback_path = out_dir / "04_zones_robust.json"
            _save_json(zones_fallback, fallback_path)
            stage.add_details(
                reason="No rows after gates",
                parameters=len(zones_fallback),
                artifact_zones=str(fallback_path),
            )
        with rec.stage("render_report") as stage:
            report = _render_report(
                out_dir,
                source_info,
                analyze_alpha,
                analyze_min_coverage,
                analyze_min_sharpe_trade,
                metrics_weights,
                zones_fallback,
                step1_meta,
                step2_meta,
                step3=None,
                step4=None,
                plots_dir=plots_dir,
            )
            stage.add_details(report_path=str(report))

        rec.add_metadata(report_path=str(report))
        instrumentation_path = out_dir / "instrumentation.json"
        rec.add_metadata(instrumentation_path=str(instrumentation_path))
        _save_json(rec.to_dict(), instrumentation_path)
        return report, rec

    # STEP 3: Per-parameter, per-metric performant zones (window-aware)
    metrics_list = _default_metrics_list()
    with rec.stage("step3_param_metric_zones") as stage:
        zones_by_param_metric, step3_meta = _step3_build_param_metric_zones(
            df_filt,
            param_grid,
            metrics_list,
            alpha=analyze_alpha,
            min_coverage=analyze_min_coverage,
            plot_dir=plots_dir,
        )
        zones_path = out_dir / "03_param_metric_zones.json"
        meta_path = out_dir / "03_zones_meta.json"
        _save_json(zones_by_param_metric, zones_path)
        _save_json(step3_meta, meta_path)
        metric_counts = {
            p: {m: len(lst or []) for m, lst in (by_metric or {}).items()}
            for p, by_metric in zones_by_param_metric.items()
        }
        stage.add_details(
            step3_meta,
            params_with_zones=sum(
                1 for _, bm in zones_by_param_metric.items() if any(bm.values())
            ),
            metric_zone_counts=metric_counts,
            artifact_zones=str(zones_path),
            artifact_meta=str(meta_path),
        )

    # STEP 4: Overlay zones across metrics with importance weights
    weights = metrics_weights or _default_metric_weights()
    with rec.stage("step4_overlay_robust_zones") as stage:
        zones_robust, step4_meta = _step4_overlay_to_robust_zones(
            zones_by_param_metric,
            param_grid,
            weights,
            total_windows=_infer_total_windows(df_filt),
            min_coverage=analyze_min_coverage,
            artifacts_dir=out_dir,
            plot_dir=plots_dir,
        )
        robust_path = out_dir / "04_zones_robust.json"
        overlay_meta_path = out_dir / "04_overlay_meta.json"
        _save_json(zones_robust, robust_path)
        _save_json(step4_meta, overlay_meta_path)
        grid_path = _robust_zones_grid(zones_robust, param_grid, out_dir)
        stage.add_details(
            step4_meta,
            params=len(zones_robust),
            robust_zone_counts={p: len(zs or []) for p, zs in zones_robust.items()},
            artifact_robust=str(robust_path),
            artifact_overlay_meta=str(overlay_meta_path),
            artifact_grid=str(grid_path),
        )

    # STEP 5: Report
    with rec.stage("render_report") as stage:
        report = _render_report(
            out_dir,
            source_info,
            analyze_alpha,
            analyze_min_coverage,
            analyze_min_sharpe_trade,
            weights,
            zones_robust,
            step1_meta,
            step2_meta,
            step3_meta,
            step4_meta,
            plots_dir=plots_dir,
        )
        stage.add_details(report_path=str(report))

    rec.add_metadata(report_path=str(report))
    instrumentation_path = out_dir / "instrumentation.json"
    rec.add_metadata(instrumentation_path=str(instrumentation_path))
    _save_json(rec.to_dict(), instrumentation_path)
    return report, rec


# ---------- Step 0: Load -------------------------------------------------------


def _load_oos_dataset(root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_top = root / "all_top_out_of_sample.csv"
    src: Dict[str, Any] = {"used": None, "files": []}
    if all_top.exists():
        df = pd.read_csv(all_top)
        src["used"] = str(all_top)
        src["files"].append(str(all_top))
    else:
        # Fallback: gather per window
        parts = []
        for p in root.glob("window_*/*top_out_of_sample_results.csv"):
            try:
                parts.append(pd.read_csv(p))
                src["files"].append(str(p))
            except Exception:
                pass
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        src["used"] = "windows_top_csv_glob"
    return df, src


def _load_walkforward_options(root: Path) -> Dict[str, Any]:
    """
    Returns the persisted walkforward options (train/test days, thresholds, ...).
    Falls back to an empty dict when the manifest is unavailable or malformed.
    """
    manifest_path = root / "walkforward_run_config.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return dict(manifest.get("walkforward_options") or {})
    except Exception:
        return {}


def _compute_min_total_trades_threshold(options: Dict[str, Any]) -> Optional[float]:
    """
    Scales the minimum trade count with the IS/OOS window ratio.
    Returns None when required inputs are missing or invalid.
    """
    try:
        min_trades = float(options["min_trades"])
        train_days = float(options["train_days"])
        test_days = float(options["test_days"])
    except (KeyError, TypeError, ValueError):
        return None
    if train_days <= 0 or test_days <= 0:
        return None
    threshold = min_trades * (test_days / train_days)
    if not math.isfinite(threshold):
        return None
    return max(0.0, threshold)


# ---------- Step 1: Clean & round ---------------------------------------------


def _step1_clean_and_round(
    df: pd.DataFrame, param_grid: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "rows_out": 0,
        "dropped_cols": [],
        "duplicates_removed": 0,
    }
    if df is None or df.empty:
        return pd.DataFrame(), meta

    df = df.copy()

    # Drop fully-empty columns/rows
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols, errors="ignore")
        meta["dropped_cols"] = empty_cols
    df = df.dropna(how="all")

    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Rounding: parameters only (align to step if provided)
    param_cols = list(param_grid.keys())
    float_params: List[str] = []
    for p in param_cols:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")
            spec = param_grid[p]
            if spec["type"] == "int":
                step_int = int(spec.get("step", 1) or 1)
                low_int, high_int = int(spec["low"]), int(spec["high"])
                df[p] = (
                    ((np.round((df[p] - low_int) / step_int) * step_int) + low_int)
                    .clip(low_int, high_int)
                    .astype("Int64")
                )
            elif spec["type"] == "float":
                step_val = spec.get("step")
                low_f, high_f = float(spec["low"]), float(spec["high"])
                if step_val:
                    step_f = float(step_val)
                    df[p] = (
                        ((np.round((df[p] - low_f) / step_f) * step_f) + low_f)
                        .clip(low_f, high_f)
                        .round(6)
                    )
                else:
                    df[p] = df[p].round(6)
                float_params.append(p)
            elif spec["type"] == "categorical":
                pass

    # Zusätzliche Entschärfung: -0.0 -> 0.0 und winzige Reste auf 0 ziehen
    for p in float_params:
        if p in df.columns:
            s = df[p]
            # kleine Epsilon-Reste auf 0.0 setzen
            s = s.mask(s.abs() < 1e-12, 0.0)
            # sicherstellen, dass typische -0.0 nicht durchrutschen
            df[p] = s.round(6)

    # Remove duplicates: "keine doppelten Kombis pro Window"
    subset = (["window_id"] if "window_id" in df.columns else []) + [
        c for c in param_cols if c in df.columns
    ]
    if subset:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        meta["duplicates_removed"] = int(before - len(df))

    # Coerce key metrics
    for col in [
        "Net Profit",
        "Drawdown",
        "Commission",
        "total_trades",
        "Sharpe (trade)",
        "Avg R-Multiple",
        "Winrate (%)",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final drop of rows with essential NaNs in metrics
    essential = [
        c
        for c in ["Net Profit", "Drawdown", "Commission", "total_trades"]
        if c in df.columns
    ]
    if essential:
        df = df.dropna(subset=essential)

    meta["rows_out"] = int(len(df))
    return df, meta


# ---------- Step 2: Hard Gates -------------------------------------------------


def _step2_apply_hard_gates(
    df: pd.DataFrame, *, min_total_trades_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "rows_out": 0,
        "drop_reason_counts": {},
    }
    if df.empty:
        return df, meta
    df = df.copy()

    reasons: List[Tuple[str, int]] = []

    def _format_threshold(value: float) -> str:
        if not isfinite(value):
            return "nan"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _mark(mask: "pd.Series[bool]", name: str) -> pd.DataFrame:
        reasons.append((name, int((~mask).sum())))
        return df.loc[mask].copy()

    if min_total_trades_threshold is None or not isfinite(min_total_trades_threshold):
        trade_threshold = 10.0
    else:
        trade_threshold = max(0.0, float(min_total_trades_threshold))
    meta["thresholds"] = {"min_total_trades": trade_threshold}
    trade_gate_name = f"trades_ge_{_format_threshold(trade_threshold)}"

    # Profit > 0
    if "Net Profit" in df.columns:
        mask = df["Net Profit"] > 0.0
        df = _mark(mask, "profit_gt_0")

    # Profit/DD >= 1 (guard dd<=0)
    if "Drawdown" in df.columns and "Net Profit" in df.columns:
        dd = df["Drawdown"].replace({0.0: np.nan})
        ratio = df["Net Profit"] / dd
        mask = ratio >= 1.0
        df = _mark(mask.fillna(False), "profit_over_dd_ge_1")

    # Commission/Profit < 0.7
    if "Commission" in df.columns and "Net Profit" in df.columns:
        # net profit already > 0 by first gate
        mask = (df["Commission"] / df["Net Profit"]) < 0.70
        df = _mark(mask.fillna(False), "commission_over_profit_lt_0p70")

    # total_trades >= threshold (scaled from walkforward settings when available)
    if "total_trades" in df.columns:
        trades = pd.to_numeric(df["total_trades"], errors="coerce").fillna(0)
        mask = trades >= trade_threshold
        df = _mark(mask, trade_gate_name)

    meta["rows_out"] = int(len(df))
    meta["drop_reason_counts"] = {k: v for k, v in reasons}
    return df, meta


def _plot_histograms_step2(df: pd.DataFrame, plot_dir: Optional[Path]) -> None:
    if df.empty or plot_dir is None:
        return
    for col in ["Net Profit", "Drawdown", "Commission", "Sharpe (trade)"]:
        if col in df.columns:
            try:
                fig = plt.figure(figsize=(6, 4), dpi=120)
                df[col].dropna().hist(bins=30)
                plt.title(f"Distribution: {col}")
                plt.tight_layout()
                fig.savefig(
                    plot_dir
                    / f"02_hist_{col.replace(' ','_').replace('(','').replace(')','')}.png"
                )
                plt.close(fig)
            except Exception:
                pass


# ---------- Step 3: Param–Metric zones (window-aware) -------------------------


def _default_metrics_list() -> List[Tuple[str, str]]:
    # (metric_name, direction), direction: "high" = larger is better, "low" = smaller is better
    return [
        ("Sharpe (trade)", "high"),
        ("Net Profit", "high"),
        ("Avg R-Multiple", "high"),
        ("Winrate (%)", "high"),
        ("Drawdown", "low"),
    ]


@dataclass
class Zone:
    low: float
    high: float
    mean: float
    std: float
    window_coverage: float  # proportion of windows that support this zone
    windows_supported: int


def _infer_total_windows(df: pd.DataFrame) -> int:
    return int(df["window_id"].nunique()) if "window_id" in df.columns else 0


def _fd_bins(x: np.ndarray) -> int:
    # Freedman–Diaconis rule for robust bin width; fallbacks applied
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return max(1, n)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return min(30, max(5, int(math.sqrt(n))))
    h = 2 * iqr * (n ** (-1 / 3))
    if h <= 0:
        return min(30, max(5, int(math.sqrt(n))))
    k = int(np.ceil((x.max() - x.min()) / h))
    return min(max(k, 5), 60)


def _window_select_top(
    df_w: pd.DataFrame, metric: str, direction: str, alpha: float
) -> pd.DataFrame:
    if df_w.empty or metric not in df_w.columns:
        return df_w.iloc[0:0]
    s = pd.to_numeric(df_w[metric], errors="coerce")
    if direction == "high":
        thr = s.quantile(1.0 - alpha)
        return df_w.loc[s >= thr]
    else:
        thr = s.quantile(alpha)
        return df_w.loc[s <= thr]


def _zones_from_hist(
    vals: np.ndarray, bin_count: int, min_coverage: float, total_windows: int
) -> List[Tuple[float, float, int]]:
    """
    Identify contiguous high-density regions; return [(low,high,windows_supported_est)].
    We approximate window support by unique windows present inside each region.
    """
    if len(vals) == 0:
        return []
    hist, edges = np.histogram(vals, bins=bin_count)
    max_h = hist.max() if hist.size else 0
    if max_h == 0:
        return []

    # threshold: 15% of max bin height -> "salient" density
    thr = max(1, int(round(max_h * 0.15)))
    regions = []
    start = None
    for i, h in enumerate(hist):
        if h >= thr and start is None:
            start = i
        if (h < thr or i == len(hist) - 1) and start is not None:
            end = i if h < thr else i  # inclusive
            low, high = edges[start], edges[end + 1]
            regions.append((low, high))
            start = None
    # Filter by coverage threshold later (with real window mapping)
    return [(a, b, 0) for (a, b) in regions]


def _step3_build_param_metric_zones(
    df: pd.DataFrame,
    param_grid: Dict[str, Dict[str, Any]],
    metrics_list: List[Tuple[str, str]],
    *,
    alpha: float,
    min_coverage: float,
    plot_dir: Optional[Path],
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Any]]:
    param_cols = [p for p in param_grid.keys() if p in df.columns]
    total_windows = _infer_total_windows(df)
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    meta = {
        "total_windows": total_windows,
        "alpha": alpha,
        "min_coverage": min_coverage,
    }

    for p in param_cols:
        results[p] = {}
        for metric, direction in metrics_list:
            if metric not in df.columns:
                continue

            # Per-window selection of top alpha-quantile for the metric
            selected_rows = []
            for wid, grp in (
                df.groupby("window_id") if "window_id" in df.columns else [(-1, df)]
            ):
                top = _window_select_top(grp, metric, direction, alpha)
                if not top.empty:
                    # use median param value per window to avoid overweight
                    med = pd.to_numeric(top[p], errors="coerce").median()
                    if np.isfinite(med):
                        selected_rows.append({"window_id": wid, p: float(med)})
            if not selected_rows:
                continue

            sel_df = pd.DataFrame(selected_rows)
            vals = np.asarray(pd.to_numeric(sel_df[p], errors="coerce").dropna().values)
            if vals.size == 0:
                continue

            bin_count = _fd_bins(vals)
            raw_regions = _zones_from_hist(vals, bin_count, min_coverage, total_windows)

            zones_for_metric: List[Dict[str, Any]] = []
            # compute coverage by counting windows whose median lies within region
            for low, high, _ in raw_regions:
                mask = (sel_df[p] >= low) & (sel_df[p] <= high)
                windows_supported = (
                    int(sel_df.loc[mask, "window_id"].nunique())
                    if "window_id" in sel_df.columns
                    else (mask.sum() > 0)
                )
                coverage = (
                    (windows_supported / total_windows) if total_windows > 0 else 0.0
                )
                # keep region only if coverage >= threshold
                if coverage >= min_coverage:
                    vals_in = np.asarray(sel_df.loc[mask, p].astype(float).values)
                    zones_for_metric.append(
                        {
                            "metric": metric,
                            "direction": direction,
                            "low": float(np.min(vals_in)),
                            "high": float(np.max(vals_in)),
                            "mean": float(np.mean(vals_in)),
                            "std": (
                                float(np.std(vals_in, ddof=1))
                                if len(vals_in) > 1
                                else 0.0
                            ),
                            "window_coverage": round(float(coverage), 6),
                            "windows_supported": int(windows_supported),
                            "n_samples": int(len(vals_in)),
                        }
                    )

            # Artifact: histogram + shaded regions
            if zones_for_metric and plot_dir is not None:
                _plot_param_metric_histogram(
                    np.asarray(sel_df[p].values),
                    zones_for_metric,
                    plot_dir / f"03_hist_{p}__{metric}.png",
                )

            results[p][metric] = zones_for_metric

    return results, meta


def _plot_param_metric_histogram(
    x: Any, zones: List[Dict[str, Any]], out_path: Path
) -> None:
    try:
        fig = plt.figure(figsize=(6, 4), dpi=130)
        plt.hist(x, bins=_fd_bins(np.asarray(x)), alpha=0.7)
        for z in zones:
            plt.axvspan(z["low"], z["high"], alpha=0.25)
        ttl = f"Param distribution with performant zones"
        plt.title(ttl)
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass


# ---------- Step 4: Overlay across metrics ------------------------------------


def _default_metric_weights() -> Dict[str, float]:
    # Hedge-fund bias: risk-adjusted first, then profitability, then efficiency
    return {
        "Sharpe (trade)": 0.35,
        "Net Profit": 0.30,
        "Avg R-Multiple": 0.20,
        "Winrate (%)": 0.10,
        "Drawdown": 0.05,  # contributes inversely; handled via membership only
    }


def _collect_boundaries(
    zones_by_metric: Dict[str, List[Dict[str, Any]]],
) -> List[float]:
    b = set()
    for lst in zones_by_metric.values():
        for z in lst or []:
            b.add(float(z["low"]))
            b.add(float(z["high"]))
    bb = sorted(b)
    # guard against degenerate bounds
    return bb


def _overlay_intervals_for_param(
    zones_by_metric: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float],
    *,
    min_weight_sum: float = 0.50,
) -> List[Tuple[float, float, float]]:
    """
    Weighted overlay: split by all boundaries; keep segments where sum(weights of covering metrics) >= threshold.
    Return [(low, high, weight_sum)] merged.
    """
    boundaries = _collect_boundaries(zones_by_metric)
    if len(boundaries) < 2:
        return []

    segments = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        mid = (a + b) / 2.0
        wsum = 0.0
        for metric, zones in zones_by_metric.items():
            if not zones:
                continue
            for z in zones:
                if z["low"] <= mid <= z["high"]:
                    wsum += float(weights.get(metric, 0.0))
                    break
        if wsum >= min_weight_sum:
            segments.append((a, b, wsum))

    # merge adjacent segments with similar weight_sum
    merged: List[Tuple[float, float, float]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            la, lb, lw = merged[-1]
            a, b, w = seg
            if abs(w - lw) < 1e-9 and abs(a - lb) < 1e-12:
                merged[-1] = (la, b, w)
            else:
                merged.append(seg)
    return merged


def _coverage_for_zone(
    df: pd.DataFrame, param: str, low: float, high: float
) -> Tuple[int, float]:
    if df.empty:
        return 0, 0.0
    if "window_id" not in df.columns:
        return 0, 0.0
    sup = 0
    for wid, grp in df.groupby("window_id"):
        has_support = (
            pd.to_numeric(grp[param], errors="coerce").between(low, high)
        ).any()
        if has_support:
            sup += 1
    tot = df["window_id"].nunique()
    return sup, (sup / tot) if tot > 0 else 0.0


def _step4_overlay_to_robust_zones(
    zones_by_param_metric: Dict[str, Dict[str, List[Dict[str, Any]]]],
    param_grid: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    *,
    total_windows: int,
    min_coverage: float,
    artifacts_dir: Path,
    plot_dir: Optional[Path],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    robust: Dict[str, List[Dict[str, Any]]] = {}
    meta = {"weights": weights, "min_weight_sum": 0.50, "min_coverage": min_coverage}

    # For coverage measurement we need the filtered dataset again: store a minimal shadow file in step2 if available.
    # To avoid coupling, recompute coverage using the step2 artifact:
    df_step2 = None
    try:
        df_step2 = pd.read_parquet(artifacts_dir / "02_filtered.parquet")
    except Exception:
        df_step2 = pd.DataFrame()

    for p, by_metric in zones_by_param_metric.items():
        segs = _overlay_intervals_for_param(by_metric, weights, min_weight_sum=0.50)
        zones = []
        for low, high, wsum in segs:
            ws, cov = _coverage_for_zone(df_step2, p, low, high)
            if cov >= min_coverage:
                zones.append(
                    {
                        "low": float(low),
                        "high": float(high),
                        "weight_sum": float(round(wsum, 6)),
                        "windows_supported": int(ws),
                        "window_coverage": float(round(cov, 6)),
                    }
                )
        # If no zones -> fallback to full param interval
        if not zones:
            spec = param_grid.get(p, {})
            low = float(spec.get("low", np.nan))
            high = float(spec.get("high", np.nan))
            if not np.isfinite(low) or not np.isfinite(high):
                low, high = _robust_min_max(df_step2, p)
            zones = [
                {
                    "low": float(low),
                    "high": float(high),
                    "weight_sum": 0.0,
                    "windows_supported": 0,
                    "window_coverage": 0.0,
                    "fallback": True,
                }
            ]
        robust[p] = zones

        # Artifact: overlay plot
        if plot_dir is not None:
            _plot_overlay(
                p, by_metric, zones, plot_dir / f"04_overlay_{p}.png", weights
            )

    return robust, meta


def _robust_min_max(df: pd.DataFrame, param: str) -> Tuple[float, float]:
    s = pd.to_numeric(df[param], errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return (0.0, 0.0)
    return (float(s.min()), float(s.max()))


def _plot_overlay(
    param: str,
    by_metric: Dict[str, List[Dict[str, Any]]],
    zones: List[Dict[str, Any]],
    out_path: Path,
    weights: Dict[str, float],
) -> None:
    try:
        fig = plt.figure(figsize=(7, 2.8), dpi=140)
        # plot metric zones as colored bands (stacked vertically)
        y = 0
        yticks: List[float] = []
        ylabels: List[str] = []
        for m, lst in by_metric.items():
            for z in lst or []:
                plt.plot([z["low"], z["high"]], [y, y], linewidth=6)
            yticks.append(y)
            ylabels.append(f"{m} (w={weights.get(m,0):.2f})")
            y += 1
        # robust zones on top
        for z in zones:
            plt.plot([z["low"], z["high"]], [y + 0.2, y + 0.2], linewidth=10)
        yticks.append(y + 0.2)
        ylabels.append("ROBUST")
        plt.yticks(yticks, ylabels)
        plt.title(f"Overlay zones for param: {param}")
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass


# ---------- Step 5: Report -----------------------------------------------------


def _render_report(
    out_dir: Path,
    source_info: Dict[str, Any],
    alpha: float,
    min_cov: float,
    min_sharpe: float,
    weights: Optional[Dict[str, float]],
    robust_zones: Dict[str, Any],
    step1: Optional[Dict[str, Any]],
    step2: Optional[Dict[str, Any]],
    step3: Optional[Dict[str, Any]],
    step4: Optional[Dict[str, Any]],
    plots_dir: Optional[Path] = None,
) -> Path:
    md = []
    md.append(f"# Robust Zone Analysis\n")
    md.append(f"- Data source: `{source_info.get('used')}`")
    if source_info.get("files"):
        md.append(f"- Files: {len(source_info['files'])}")
    md.append(f"- alpha (top-quantile per window): {alpha}")
    md.append(f"- min_coverage (windows): {min_cov}")
    md.append(f"- min_sharpe_trade (informational): {min_sharpe}")
    if weights:
        md.append(f"- metric weights: {json.dumps(weights)}")

    if step1:
        md.append(f"\n## Step 1 Summary\n```json\n{json.dumps(step1, indent=2)}\n```")
    if step2:
        md.append(
            f"\n## Step 2 Summary (Hard Gates)\n```json\n{json.dumps(step2, indent=2)}\n```"
        )
    if step3:
        md.append(f"\n## Step 3 Meta\n```json\n{json.dumps(step3, indent=2)}\n```")
    if step4:
        md.append(f"\n## Step 4 Meta\n```json\n{json.dumps(step4, indent=2)}\n```")

    md.append("\n## Robust Zones (final)")
    md.append("```json")
    md.append(json.dumps(robust_zones, indent=2))
    md.append("```")

    # link artifacts
    md.append("\n## Artifacts\n")
    if plots_dir is not None:
        rel = os.path.relpath(plots_dir, out_dir)
        for p in sorted(robust_zones.keys()):
            md.append(f"- Histograms & raw zones: `{rel}/03_hist_{p}__*.png`")
            md.append(f"- Overlay: `{rel}/04_overlay_{p}.png`")
    else:
        md.append("- (Plots disabled)")

    report = out_dir / "ANALYSIS_REPORT.md"
    report.write_text("\n".join(md), encoding="utf-8")
    return report


# ---------- Helpers ------------------------------------------------------------


def _save_df(df: pd.DataFrame, path: Path) -> None:
    """
    Schreibe Artefakte immer als Parquet **und** als CSV (gleicher Basename).
    - Parquet: schneller & kompakt
    - CSV: sofort in jedem Editor/Viewer lesbar
    """
    # 1) Parquet (Best Effort)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        print(f"⚠️ Parquet-Export fehlgeschlagen ({path.name}): {e}")

    # 2) CSV (immer)
    try:
        csv_path = path.with_suffix(".csv") if path.suffix else Path(str(path) + ".csv")
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"❌ CSV-Export fehlgeschlagen ({csv_path.name}): {e}")


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _robust_zones_grid(
    zones_robust: Dict[str, List[Dict[str, Any]]],
    param_grid: Dict[str, Dict[str, Any]],
    out_dir: Path,
    *,
    top_k: int = 2,
    tau_c: float = 0.60,  # coverage gate
    tau_b: float = 0.40,  # max relative width
    delta: float = 0.05,  # merge-gap threshold (relative to full range)
    max_overlap: float = 0.10,  # allowed overlap fraction during greedy pick
    alpha: float = 0.50,  # weight for weight_sum
    beta: float = 0.35,  # weight for window_coverage
    gamma: float = 0.15,  # weight for (1 - rel_width)
) -> Path:
    """Selects and exports the mathematically best zones (1–3 each) as:
    - 05_zones_best.json (rich, with reasons)
    - 05_zones_best_grid.json (minimal grid for machines)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step2_path = out_dir / "02_filtered.parquet"
    total_windows = None
    if step2_path.exists():
        try:
            df2 = pd.read_parquet(step2_path)
            if "window_id" in df2.columns:
                total_windows = int(df2["window_id"].nunique())
        except Exception:
            total_windows = None
    meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "method": {
            "coverage_gate": tau_c,
            "rel_width_gate": tau_b,
            "merge_gap_rel": delta,
            "max_overlap": max_overlap,
            "score_weights": {
                "alpha_weight_sum": alpha,
                "beta_coverage": beta,
                "gamma_width_reward": gamma,
            },
            "top_k": top_k,
            "total_windows": total_windows,
        },
    }

    def _param_range(p: str) -> Tuple[float, float]:
        spec = param_grid.get(p, {})
        lo = spec.get("low", None)
        hi = spec.get("high", None)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return float(lo), float(hi)
        zs = zones_robust.get(p, [])
        if not zs:
            return float("nan"), float("nan")
        lo = min(z["low"] for z in zs)
        hi = max(z["high"] for z in zs)
        return float(lo), float(hi)

    def _merge_nearby(zlist: List[Dict[str, Any]], R: float) -> List[Dict[str, Any]]:
        if not zlist:
            return []
        zlist = sorted(zlist, key=lambda z: (z["low"], z["high"]))
        merged: List[Dict[str, Any]] = [zlist[0].copy()]
        for z in zlist[1:]:
            prev = merged[-1]
            gap = max(0.0, z["low"] - prev["high"])
            if gap <= delta * max(R, 1e-12):
                prev["high"] = max(prev["high"], z["high"])
                for k in ("weight_sum", "window_coverage", "windows_supported"):
                    if k in z:
                        prev[k] = max(prev.get(k, 0.0), z.get(k, 0.0))
                prev.setdefault("notes", []).append("merged")
            else:
                merged.append(z.copy())
        return merged

    def _rel_width(z: Dict[str, Any], R: float) -> float:
        if not (isinstance(R, (int, float)) and np.isfinite(R) and R > 0):
            return 1.0
        return max(0.0, float(z["high"] - z["low"]) / R)

    def _score(z: Dict[str, Any], R: float) -> float:
        w = float(z.get("weight_sum", 0.0))
        c = float(z.get("window_coverage", 0.0))
        rw = 1.0 - _rel_width(z, R)
        return alpha * w + beta * c + gamma * rw

    def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        inter = max(0.0, min(a["high"], b["high"]) - max(a["low"], b["low"]))
        w_small = min(a["high"] - a["low"], b["high"] - b["low"])
        if w_small <= 0:
            return 0.0
        return float(inter / w_small)

    best: Dict[str, Any] = {}
    rejected: Dict[str, List[Dict[str, Any]]] = {}
    for p, zones in zones_robust.items():
        p_lo, p_hi = _param_range(p)
        R = float(p_hi - p_lo) if (np.isfinite(p_lo) and np.isfinite(p_hi)) else None
        if not (isinstance(R, (int, float)) and np.isfinite(R) and R > 0):
            if zones:
                R = max(z["high"] for z in zones) - min(z["low"] for z in zones)
            else:
                R = 1.0
        kept: List[Dict[str, Any]] = []
        rej: List[Dict[str, Any]] = []
        W = None
        if total_windows and total_windows > 0:
            W = total_windows
        for z in zones:
            z = z.copy()
            c = float(z.get("window_coverage", 0.0))
            ws = int(z.get("windows_supported", 0))
            if W is None and c > 0:
                est_W = int(round(ws / c)) if c > 0 else 0
                if est_W > 0:
                    W = est_W
            if c < tau_c:
                z["__reject_reason"] = "coverage_lt_tau_c"
                rej.append(z)
                continue
            if W:
                min_ws = max(5, int(math.ceil(0.4 * W)))
                if ws < min_ws:
                    z["__reject_reason"] = f"windows_supported_lt_{min_ws}"
                    rej.append(z)
                    continue
            rel_w = _rel_width(z, R)
            if (rel_w >= tau_b) and (c < 0.70):
                z["__reject_reason"] = "rel_width_gt_tau_b"
                z["rel_width"] = rel_w
                rej.append(z)
                continue
            z["rel_width"] = rel_w
            kept.append(z)
        if not kept and zones:
            rel_order = [
                ("tau_b", 0.50),
                ("tau_c", 0.55),
                ("tau_b", 0.60),
                ("tau_c", 0.50),
            ]
            kept = []
            for name, newv in rel_order:
                if name == "tau_b":
                    _tau_b = newv
                    _tau_c = tau_c
                else:
                    _tau_b = tau_b
                    _tau_c = newv
                tmp = []
                for z in zones:
                    c = float(z.get("window_coverage", 0.0))
                    ws = int(z.get("windows_supported", 0))
                    rel_w = _rel_width(z, R)
                    cond = (c >= _tau_c) and (rel_w <= _tau_b)
                    if W:
                        min_ws = max(5, int(math.ceil(0.4 * W)))
                        cond = cond and (ws >= min_ws)
                    if cond:
                        t = z.copy()
                        t["rel_width"] = rel_w
                        tmp.append(t)
                if tmp:
                    kept = tmp
                    relaxed_list: List[Dict[str, float]] = meta.setdefault("relaxed", [])  # type: ignore[assignment]
                    relaxed_list.append({name: newv})
                    break
        kept = _merge_nearby(kept, R)
        for z in kept:
            z["score"] = float(_score(z, R))
        kept.sort(key=lambda z: z["score"], reverse=True)
        selected: List[Dict[str, Any]] = []
        for z in kept:
            ok = True
            for s in selected:
                if _overlap(z, s) > max_overlap:
                    ok = False
                    break
            if ok:
                selected.append(z)
                if len(selected) >= max(1, top_k):
                    break
        best[p] = [
            {
                "low": float(z["low"]),
                "high": float(z["high"]),
                "score": float(z["score"]),
                "weight_sum": float(z.get("weight_sum", 0.0)),
                "window_coverage": float(z.get("window_coverage", 0.0)),
                "windows_supported": int(z.get("windows_supported", 0)),
                "rel_width": float(z.get("rel_width", float("nan"))),
            }
            for z in selected
        ]
        rejected[p] = [
            {
                "low": float(z.get("low", float("nan"))),
                "high": float(z.get("high", float("nan"))),
                "weight_sum": float(z.get("weight_sum", 0.0)),
                "window_coverage": float(z.get("window_coverage", 0.0)),
                "windows_supported": int(z.get("windows_supported", 0)),
                "rel_width": float(z.get("rel_width", float("nan"))),
                "reason": str(z.get("__reject_reason", "gate_or_overlap")),
            }
            for z in rej
        ]
    payload = {"metadata": meta, "zones_best": best, "rejected": rejected}
    path = out_dir / "05_zones_best.json"
    _save_json(payload, path)
    zones_out: Dict[str, List[Dict[str, Any]]] = {}
    for p, lst in best.items():
        spec = param_grid.get(p, {})
        if spec.get("type") == "categorical":
            zones_out[p] = [
                {
                    "choices": list(spec.get("choices", [])),
                    "note": "categorical_unfiltered",
                }
            ]
            continue
        step = spec.get("step", None)
        if step is None:
            if spec.get("type") == "int":
                step = 1
            elif spec.get("type") == "float":
                step = float(spec.get("step_guess", 0.1))
            else:
                step = float(0.1)
        zones_out[p] = [
            {
                "min": float(z["low"]),
                "max": float(z["high"]),
                "step": step,
                "score": float(z["score"]),
                "window_coverage": float(z.get("window_coverage", 0.0)),
                "windows_supported": int(z.get("windows_supported", 0)),
            }
            for z in lst
        ]
    grid_payload = {"metadata": meta, "zones": zones_out}
    grid_path = out_dir / "05_zones_best_grid.json"
    _save_json(grid_payload, grid_path)
    return path


def _fallback_full_intervals(
    param_grid: Dict[str, Dict[str, Any]], reason: str
) -> Dict[str, Any]:
    out = {}
    for p, spec in param_grid.items():
        if spec["type"] in ("int", "float"):
            out[p] = [
                {
                    "low": float(spec["low"]),
                    "high": float(spec["high"]),
                    "fallback": True,
                    "reason": reason,
                }
            ]
        elif spec["type"] == "categorical":
            out[p] = [
                {
                    "choices": list(spec.get("choices", [])),
                    "fallback": True,
                    "reason": reason,
                }
            ]
        else:
            out[p] = [{"fallback": True, "reason": reason}]
    return out
