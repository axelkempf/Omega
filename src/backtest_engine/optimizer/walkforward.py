import gc
import getpass
import hashlib
import json
import math
import numbers
import os
import platform
import random
import socket

# --- Windows/Console: sichere UTF-8-Ausgabe auch in Joblib-Worker -----------------
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from backtest_engine.optimizer._settings import get_selection_config
from backtest_engine.optimizer.final_param_selector import (
    run_final_parameter_selection,
)
from backtest_engine.optimizer.instrumentation import (
    StageRecorder,
    _format_stage_summary,
    _to_jsonable,
)
from backtest_engine.optimizer.optuna_optimizer import (
    optimize_strategy_with_optuna_pareto,
)
from backtest_engine.optimizer.robust_zone_analyzer import run_robust_zone_analysis
from backtest_engine.optimizer.walkforward_utils import (
    estimate_n_trials,
    update_master_index,
)
from backtest_engine.rating.p_values import bootstrap_p_value_mean_gt_zero
from backtest_engine.rating.strategy_rating import rate_strategy_performance
from backtest_engine.report.metrics import calculate_metrics
from backtest_engine.runner import run_backtest_and_return_portfolio
from hf_engine.infra.config.paths import PARQUET_DIR, WALKFORWARD_RESULTS_DIR
from hf_engine.infra.logging.log_manager import log_optuna_report
from hf_engine.infra.monitoring.telegram_bot import (
    send_walkforward_telegram_message,
)


def _ensure_utf8_stdout():
    """
    Verhindert UnicodeEncodeError bei Emojis/Unicode auf Windows (cp1252).
    - setzt PYTHONIOENCODING=utf-8 (wirkt auf Joblib-Child-Prozesse)
    - reconfiguriert stdout auf utf-8, falls möglich
    """
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        # Bei Python 3.7+ verfügbar:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_ensure_utf8_stdout()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _safe_slice_df_by_time(df, start, end):
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    # Preloaded market data follows the repo schema: a 'UTC time' column.
    # Ensure slicing is timezone-safe and does not silently return empty windows.

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    # 1) DatetimeIndex?
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return df.loc[(idx >= start_ts) & (idx <= end_ts)]

    # 2) Gängige Spaltennamen (inkl. Standard 'UTC time')
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in ("utc time", "utc_time", "time", "timestamp", "datetime"):
        col = col_map.get(cand)
        if col is None:
            continue
        s = pd.to_datetime(df[col], errors="coerce", utc=True)
        mask = (s >= start_ts) & (s <= end_ts)
        return df.loc[mask]

    # 3) Fail closed (lieber strikt leeren Slice als unsliced zurückgeben)
    return df.iloc[0:0]


# ====== NEU: Hilfsfunktionen für Rundung & Export ======
NUM_ROUND_TRIALS = {
    "profit": 2,
    "avg_r": 4,
    "winrate": 2,
    "drawdown": 2,
    # DEAKTIVIERT: Robustness Score
    # "robustness_score": 3,
    "fees_mean": 2,
    "fees_sum": 2,
}

NUM_ROUND_TEST = {
    # Test-Result Keys (nach Umbenennung unten)
    "Net Profit": 2,
    "Avg R-Multiple": 4,
    "Winrate (%)": 2,
    "Drawdown": 2,
    # DEAKTIVIERT: Robustness Score
    # "robustness_score": 3,
    "Sharpe (trade)": 3,
    "Sortino (trade)": 3,
    "Sharpe (daily)": 3,
    "Sortino (daily)": 3,
    "Drawdown (%)": 2,
    "p_mean_r_gt_0": 4,
    "p_net_profit_gt_0": 4,
    "winrate_ci_low": 2,
    "winrate_ci_high": 2,
    "Commission": 2,
}

# Zusätzliche Rundungen/Typen für die Top-20-Exports
NUM_ROUND_TOP = {
    "Net Profit": 2,
    "Avg R-Multiple": 4,
    "Winrate (%)": 2,
    "Drawdown": 2,
    "Sharpe (trade)": 3,
    "Sortino (trade)": 3,
    "Drawdown (%)": 2,
    "p_mean_r_gt_0": 4,
    "p_net_profit_gt_0": 4,
    "Commission": 2,
}
INT_COLS_TOP = {"window_id", "total_trades", "active_days"}

INT_COLS_COMMON = {"number", "total_trades", "window_id", "base_seed"}


def _int_safe(x, default: int = 0) -> int:
    """Robuster Int-Cast: NaN/None/±inf -> default (0) statt ValueError."""
    try:
        f = float(x)
        return default if (pd.isna(f) or not np.isfinite(f)) else int(f)
    except Exception:
        try:
            return int(x)
        except Exception:
            return default


def _apply_rounding_to_df(
    df: pd.DataFrame, round_map: Dict[str, int], int_cols: Optional[set] = None
) -> pd.DataFrame:
    out = df.copy()
    for col, nd in round_map.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(nd)
    if int_cols:
        for col in int_cols:
            if col in out.columns:
                # erst auf 0 Nachkommastellen runden, dann als nullable Int casten
                vals = pd.to_numeric(out[col], errors="coerce").round(0)
                out[col] = vals.astype("Int64")
    # Inf/NaN robust entschärfen (nur Darstellung)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _expand_and_round_score_column(
    df: pd.DataFrame, decimals: Optional[int] = None, keep_original_list: bool = False
) -> pd.DataFrame:
    """
    Zerlegt die 'score' Liste (Profit, AvgR, Winrate, Drawdown) in einzelne Spalten
    und rundet diese stabil. Optional bleibt die Original-Liste bestehen (gerundet).
    """
    if "score" not in df.columns:
        return df
    out = df.copy()
    sp_profit, sp_avgr, sp_winrate, sp_dd = [], [], [], []
    rounded_score_list = []
    for v in out["score"].tolist():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            p, a, w, d = v
            # Nur runden, wenn explizit gewünscht; ansonsten rohe Float-Darstellung
            if not pd.isna(p):
                p = float(p) if decimals is None else round(float(p), decimals)
            else:
                p = None
            if not pd.isna(a):
                a = float(a) if decimals is None else round(float(a), decimals)
            else:
                a = None
            if not pd.isna(w):
                w = float(w) if decimals is None else round(float(w), decimals)
            else:
                w = None
            if not pd.isna(d):
                d = float(d) if decimals is None else round(float(d), decimals)
            else:
                d = None
            sp_profit.append(p)
            sp_avgr.append(a)
            sp_winrate.append(w)
            sp_dd.append(d)
            if keep_original_list:
                rounded_score_list.append([p, a, w, d])
        else:
            sp_profit.append(np.nan)
            sp_avgr.append(np.nan)
            sp_winrate.append(np.nan)
            sp_dd.append(np.nan)
            if keep_original_list:
                rounded_score_list.append([np.nan, np.nan, np.nan, np.nan])
    out["score_profit"] = sp_profit
    out["score_avg_r"] = sp_avgr
    out["score_winrate"] = sp_winrate
    out["score_drawdown"] = sp_dd
    if keep_original_list:
        out["score"] = rounded_score_list
    else:
        out = out.drop(columns=["score"])
    return out


def _round_floats(d: dict, ndigits: int = 6) -> dict:
    """Konvertiert Float-Werte stabil zu float ohne erzwungene Rundung (ndigits wird ignoriert)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(float(v), ndigits)
        else:
            out[k] = v
    return out


def _sha256_of(obj: Any) -> str:
    """SHA-256 eines Objekts auf Basis einer stabilen JSON-Darstellung."""
    try:
        payload = json.dumps(
            _to_jsonable(obj), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except Exception:
        payload = repr(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _try_find_known_configs(config_template_path: str) -> Dict[str, Any]:
    """
    Versucht, häufige Zusatz-Konfigs im Umfeld zu finden (execution_costs.yaml, symbol_specs.yaml).
    Best Effort: durchsucht Template-Ordner, CWD und ./config.
    """
    names = ["execution_costs.yaml", "symbol_specs.yaml"]
    roots = {
        Path(config_template_path).parent,
        Path.cwd(),
        Path.cwd() / "config",
        Path.cwd() / "configs",
    }
    found = {}
    for nm in names:
        for r in list(roots):
            p = (r / nm).resolve()
            if p.exists() and p.is_file():
                try:
                    with open(p, "rb") as f:
                        h = hashlib.sha256(f.read()).hexdigest()
                    found[nm] = {"path": str(p), "sha256": h}
                    break
                except Exception:
                    found[nm] = {"path": str(p), "sha256": None}
                    break
    return found


def _safe_file_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _extract_strategy_name(base_config: Dict[str, Any]) -> Optional[str]:
    strategy_cfg = base_config.get("strategy")
    if isinstance(strategy_cfg, dict):
        return strategy_cfg.get("class") or strategy_cfg.get("module")
    if isinstance(strategy_cfg, str):
        return strategy_cfg
    return None


def _send_walkforward_telegram_reminder(
    *,
    strategy_name: Optional[str],
    symbol: Optional[str],
    walkforward_root: Path,
    export_results_path: str,
    total_runtime_sec: Optional[float],
) -> None:
    """
    Sends a concise Telegram reminder once the walkforward run completes.
    """
    try:
        root_path = Path(walkforward_root).resolve()
    except Exception:
        root_path = Path(walkforward_root)
    try:
        ratings_path = Path(export_results_path).resolve()
    except Exception:
        ratings_path = Path(export_results_path)

    runtime_min = None
    if total_runtime_sec is not None:
        runtime_min = total_runtime_sec / 60.0

    lines = [
        "Walkforward abgeschlossen ✅",
        f"Strategie: {strategy_name or '-'}",
        f"Symbol: {symbol or '-'}",
        f"Ordner: {root_path}",
        f"Ratings: {ratings_path}",
    ]
    if runtime_min is not None:
        lines.append(f"Dauer: {runtime_min:.1f} min")

    message = "\n".join(lines)
    try:
        send_walkforward_telegram_message(message, parse_mode=None)
    except Exception as exc:
        print(f"⚠️ Telegram-Reminder konnte nicht gesendet werden: {exc}")


def _summarize_optuna_study(study: Any) -> Dict[str, Any]:
    trials = list(getattr(study, "trials", []) or [])
    summary: Dict[str, Any] = {
        "trial_count_total": len(trials),
    }
    state_counts: Dict[str, int] = defaultdict(int)
    durations: List[float] = []
    for trial in trials:
        state = getattr(trial, "state", None)
        state_name = (
            getattr(state, "name", str(state)) if state is not None else "unknown"
        )
        state_counts[state_name] += 1
        dur = getattr(trial, "duration", None)
        if dur is not None:
            try:
                durations.append(float(dur.total_seconds()))
            except Exception:
                pass

    if durations:
        total = sum(durations)
        summary.update(
            {
                "trial_duration_total_sec": round(total, 4),
                "trial_duration_avg_sec": round(total / len(durations), 4),
                "trial_duration_max_sec": round(max(durations), 4),
            }
        )
    summary["trial_states"] = dict(state_counts)

    try:
        directions = getattr(study, "directions", None)
        if directions:
            summary["directions"] = list(directions)
    except Exception:
        pass

    try:
        best_trials = list(getattr(study, "best_trials", []) or [])
        slim_best = []
        for bt in best_trials[:3]:
            slim_best.append(
                {
                    "number": bt.number,
                    "values": _to_jsonable(
                        bt.values if hasattr(bt, "values") else [bt.value]
                    ),
                    "params": _to_jsonable(bt.params),
                }
            )
        if slim_best:
            summary["best_trials"] = slim_best
    except Exception:
        pass


    return summary


def _freeze_baseline_snapshot(
    walkforward_root: str,
    *,
    base_config: Dict[str, Any],
    config_template_path: str,
    param_grid: Dict[str, Any],
    rating_thresholds: Optional[Dict[str, Any]],
    walkforward_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Sichert eine Baseline-Snapshot der Kernkonfiguration für spätere Vergleiche."""

    baseline_dir = Path(walkforward_root) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    current_snapshot = {
        "created_at": datetime.utcnow().isoformat(),
        "config_template_path": str(Path(config_template_path).resolve()),
        "base_config": _to_jsonable(base_config),
        "param_grid": _to_jsonable(param_grid),
        "rating_thresholds": _to_jsonable(rating_thresholds or {}),
        "walkforward_options": _to_jsonable(walkforward_options),
    }
    current_snapshot["hashes"] = {
        "base_config": _sha256_of(base_config),
        "param_grid": _sha256_of(param_grid),
        "rating_thresholds": _sha256_of(rating_thresholds or {}),
        "walkforward_options": _sha256_of(walkforward_options),
    }

    frozen_path = baseline_dir / "frozen_snapshot.json"
    current_path = baseline_dir / "current_snapshot.json"
    manifest_path = baseline_dir / "baseline_manifest.json"

    _write_json(current_path, current_snapshot)

    status = "aligned"
    diff: Dict[str, Any] = {}

    if not frozen_path.exists():
        _write_json(frozen_path, current_snapshot)
        status = "frozen"
    else:
        try:
            with open(frozen_path, "r", encoding="utf-8") as fh:
                frozen_snapshot = json.load(fh)
        except Exception as exc:
            status = "error_loading_baseline"
            diff["error"] = str(exc)
            frozen_snapshot = {}

        frozen_hashes = (
            frozen_snapshot.get("hashes", {})
            if isinstance(frozen_snapshot, dict)
            else {}
        )
        current_hashes = current_snapshot["hashes"]
        drift_keys = [
            key
            for key, value in current_hashes.items()
            if frozen_hashes.get(key) != value
        ]
        if drift_keys:
            status = "drift_detected"
            diff["changed_keys"] = drift_keys
            diff["baseline_hashes"] = frozen_hashes
            diff["current_hashes"] = current_hashes

    manifest_payload = {
        "status": status,
        "frozen_snapshot": str(frozen_path.resolve()),
        "current_snapshot": str(current_path.resolve()),
        "last_compared_at": datetime.utcnow().isoformat(),
    }
    if diff:
        manifest_payload["diff"] = _to_jsonable(diff)
    _write_json(manifest_path, manifest_payload)

    return manifest_payload


def _load_window_instrumentation(window_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(window_dir) / "instrumentation.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _aggregate_window_stage_totals(
    window_payloads: List[Dict[str, Any]],
) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for payload in window_payloads:
        stages = payload.get("stages", []) if isinstance(payload, dict) else []
        for stage in stages:
            try:
                name = stage.get("name")
                duration = float(stage.get("duration_sec", 0.0) or 0.0)
                totals[name] += duration
            except Exception:
                continue
    return {k: round(v, 6) for k, v in totals.items()}


def _export_walkforward_run_config(
    out_dir: str,
    *,
    base_config: Dict[str, Any],
    config_template_path: str,
    param_grid: Dict[str, Any],
    rating_thresholds: Optional[Dict[str, Any]],
    train_days: int,
    test_days: int,
    buffer_days: int,
    roll_interval_days: int,
    preload_mode: str,
    min_trades: int,
    min_days_active: int,
    kfold_splits: int,
    robustness_jitter_frac: float,
    robustness_repeats: int,
    n_jobs: Optional[int],
    seed: int,
    analyze_after: bool,
    analyze_alpha: float,
    analyze_min_coverage: float,
    analyze_min_sharpe_trade: float,
    window_ranges: List[Dict[str, Any]],
    effective_trials: int,
    export_results_path: str,
    run_started_ts: float,
    run_finished_ts: float,
    instrumentation_overview: Optional[Dict[str, Any]] = None,
):
    """
    Schreibt eine vollständige, reproduzierbare Run-Dokumentation nach out_dir/walkforward_run_config.json.
    """
    started = datetime.fromtimestamp(run_started_ts).isoformat()
    finished = datetime.fromtimestamp(run_finished_ts).isoformat()
    duration_sec = round(run_finished_ts - run_started_ts, 3)

    # Auswahl-Settings (falls vorhanden)
    try:
        from backtest_engine.optimizer._settings import get_selection_config

        selection_cfg = get_selection_config({})
    except Exception:
        selection_cfg = {}

    # bekannte Zusatz-Konfigs (best effort)
    known_cfgs = _try_find_known_configs(config_template_path)

    # Fenster in ISO-Strings projizieren (keine datetime-Objekte speichern)
    def _isoize(win: Dict[str, Any]) -> Dict[str, Any]:
        dd = {}
        for k, v in win.items():
            if isinstance(v, datetime):
                dd[k] = v.isoformat()
            else:
                dd[k] = v
        return dd

    manifest = {
        "run_meta": {
            "created_at": finished,
            "started_at": started,
            "duration_seconds": duration_sec,
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "n_jobs": n_jobs,
            "cwd": str(Path.cwd()),
            "seed": seed,
        },
        "paths": {
            "walkforward_root": str(Path(out_dir).resolve()),
            "export_results_path": str(Path(export_results_path).resolve()),
            "config_template_path": str(Path(config_template_path).resolve()),
            **({} if not known_cfgs else {"aux_configs": known_cfgs}),
        },
        "base_config": {
            "sha256": _sha256_of(base_config),
            "symbol": base_config.get("symbol"),
            "timeframes": base_config.get("timeframes"),
            "mode": base_config.get("mode"),
            "risk_per_trade": base_config.get("risk_per_trade"),
            "initial_balance": base_config.get("initial_balance"),
            "start_date": base_config.get("start_date"),
            "end_date": base_config.get("end_date"),
            "warmup_bars": base_config.get("warmup_bars"),
            "strategy": base_config.get("strategy"),
            "session_filter": base_config.get("session_filter"),
        },
        "param_grid": {
            "sha256": _sha256_of(param_grid),
            "definition": _to_jsonable(param_grid),
            "effective_trials_per_window": int(effective_trials),
        },
        "rating_thresholds": _to_jsonable(rating_thresholds or {}),
        "walkforward_options": {
            "train_days": int(train_days),
            "test_days": int(test_days),
            "buffer_days": int(buffer_days),
            "roll_interval_days": int(roll_interval_days),
            "preload_mode": str(preload_mode),
            "min_trades": int(min_trades),
            "min_days_active": int(min_days_active),
            "kfold_splits": int(kfold_splits),
            "robustness_jitter_frac": float(robustness_jitter_frac),
            "robustness_repeats": int(robustness_repeats),
            "analyze_after": bool(analyze_after),
            "analyze_alpha": float(analyze_alpha),
            "analyze_min_coverage": float(analyze_min_coverage),
            "analyze_min_sharpe_trade": float(analyze_min_sharpe_trade),
        },
        "windowing": {
            "overall_start": base_config.get("start_date"),
            "overall_end": base_config.get("end_date"),
            "n_windows": len(window_ranges),
            "windows": [_isoize(w) for w in window_ranges],
        },
        "selection_settings": _to_jsonable(selection_cfg),
    }

    if instrumentation_overview:
        manifest["instrumentation"] = _to_jsonable(instrumentation_overview)

    out_path = Path(out_dir) / "walkforward_run_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(manifest), f, indent=2, ensure_ascii=False)
    print(f"📝 Run-Konfiguration exportiert: {out_path}")


# === NEW: IS-Filter & Top-N-Selektion ==================================
def _select_is_topN_candidates(
    trials_df: pd.DataFrame, sel_cfg: Dict[str, Any], *, fallback_topN: int = 30
) -> List[Dict[str, Any]]:
    """
    Nimmt alle Optuna-TRIALS (IS), appliziert harte Gates und erstellt ein Ranking,
    um die Top-N Parameterkombinationen für die OOS-Evaluierung zu wählen.
    Erwartete Spalten im trials_df: profit, avg_r, winrate, drawdown, (Robustness Score deaktiviert), total_trades, params (als user_attrs).
    """
    # DEAKTIVIERT: Robustness Score
    # Hinweis: Robustness temporär deaktiviert; Reaktivierung durch Entfernen der Kommentare.
    if trials_df is None or trials_df.empty:
        return []

    # Konfiguration (Defaults + _settings-Override)
    is_filter = (sel_cfg or {}).get("is_filter", {})
    gates = {
        "profit_min": 0.0,
        "avg_r_min": 0.0,
        "winrate_min": 0.0,  # in %
        "drawdown_max": float("inf"),
        # DEAKTIVIERT: Robustness Score
        # "robustness_min": 0.0,
        "min_trades": 0,
    }
    gates.update(is_filter.get("gates", {}))
    topN = int(is_filter.get("topN", fallback_topN))

    df = trials_df.copy()
    # Fehlende Spalten robust anlegen
    for c in (
        "profit",
        "avg_r",
        "winrate",
        "drawdown",
        # DEAKTIVIERT: Robustness Score
        # "robustness_score",
        "total_trades",
    ):
        if c not in df.columns:
            df[c] = np.nan
    if "params" not in df.columns:
        # params in all_trials liegen in einzelnen Spalten; wir rekonstruieren ein flaches Dict pro Zeile
        # Heuristik: Alle Spalten, die nicht in bekannten Metriken/Meta sind, als Parameter interpretieren.
        known = set(
            [
                "number",
                "profit",
                "avg_r",
                "winrate",
                "drawdown",
                # DEAKTIVIERT: Robustness Score
                # "robustness_score",
                "total_trades",
                "sharpe",
                "window_id",
                "train_window",
                "test_window",
                "score",
                "score_profit",
                "score_avg_r",
                "score_winrate",
                "score_drawdown",
            ]
        )
        param_cols = [c for c in df.columns if c not in known]
        df["params"] = df[param_cols].apply(
            lambda r: {k: r[k] for k in param_cols if pd.notna(r[k])}, axis=1
        )

    # Harte Gates
    mask = (
        (
            pd.to_numeric(df["profit"], errors="coerce").fillna(-1e9)
            > gates["profit_min"]
        )
        & (
            pd.to_numeric(df["avg_r"], errors="coerce").fillna(-1e9)
            >= gates["avg_r_min"]
        )
        & (
            pd.to_numeric(df["winrate"], errors="coerce").fillna(-1e9)
            >= gates["winrate_min"]
        )
        & (
            pd.to_numeric(df["drawdown"], errors="coerce").fillna(1e9)
            <= gates["drawdown_max"]
        )
        # DEAKTIVIERT: Robustness Score
        # & (
        #     pd.to_numeric(df["robustness_score"], errors="coerce").fillna(0.0)
        #     >= gates["robustness_min"]
        # )
        # & (
        #     pd.to_numeric(df["total_trades"], errors="coerce").fillna(0)
        #     >= gates["min_trades"]
        # )
    )
    df = df.loc[mask].copy()
    if df.empty:
        return []

    # Ranking-Score (nicht über-optimieren): robust, multi-kriteriell
    # Höher ist besser: Profit, AvgR, Winrate, Robustheit; kleiner ist besser: Drawdown
    def _safe(col: str, default: float):
        """
        Robust:
        - Falls Spalte fehlt -> konstante Serie mit Default.
        - Für 'sharpe' werden gängige Alias-Namen versucht.
        """
        alias_map = {
            "sharpe": [
                "sharpe",
                "sharpe_trade",
                "Sharpe (trade)",
                "Sharpe (daily)",
                "sharpe_daily",
            ],
            "sortino": [
                "sortino",
                "sortino_trade",
                "Sortino (trade)",
                "Sortino (daily)",
                "sortino_daily",
            ],
        }
        # direkte Spalte vorhanden?
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        # Alias prüfen (z.B. 'sharpe')
        for cand in alias_map.get(col, []):
            if cand in df.columns:
                return pd.to_numeric(df[cand], errors="coerce").fillna(default)
        # Fallback: konstante Default-Serie
        return pd.Series(default, index=df.index, dtype=float)

    # Composite wie im Analyzer: Profit↓, AvgR↓, Winrate↓ (hoch gut), DD↑ (niedrig gut), Robustness↓ (hoch gut)
    pr = lambda s, asc=False: s.rank(pct=True, ascending=asc)
    df["_comp"] = (
        0.35 * pr(_safe("profit", 0.0), asc=False)
        + 0.25 * pr(_safe("avg_r", 0.0), asc=False)
        + 0.15 * pr(_safe("winrate", 0.0), asc=False)
        + 0.15 * pr(_safe("sharpe", 0.0), asc=False)
        + 0.07 * pr(_safe("drawdown", 1e9), asc=True)
        # DEAKTIVIERT: Robustness Score
        # + 0.03 * pr(_safe("robustness_score", 0.0), asc=False)
    )
    df.sort_values(by="_comp", ascending=False, inplace=True)

    # Top-N extrahieren
    pick = df.head(topN)
    # Kandidatenliste als [{params:..., robustness_score:...}, ...]
    out = []
    for _, row in pick.iterrows():
        params = row.get("params", {})
        if not isinstance(params, dict) or not params:
            # Fallback: Rekonstruieren aus Spalten wie oben
            params = {}
        out.append(
            {
                "params": _round_floats(dict(params), 6),
                # DEAKTIVIERT: Robustness Score
                # "robustness_score": float(
                #     pd.to_numeric(row.get("robustness_score"), errors="coerce") or 0.0
                # ),
            }
        )
    return out


# === NEW: Signifikanz-Tools ==============================================
def _wilson_interval(successes: int, n: int, conf: float = 0.95):
    if n <= 0:
        return (0.0, 0.0)
    from math import sqrt

    # z für 95 % ~ 1.96
    z = 1.959963984540054
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = p_hat + z * z / (2 * n)
    half = z * ((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) ** 0.5
    low = (center - half) / denom
    high = (center + half) / denom
    return (max(0.0, low) * 100.0, min(1.0, high) * 100.0)


def _compute_significance_from_trades(trades_df):
    """
    Erwartete Spalten: bevorzugt 'r_multiple' und 'result' (EUR).
    Gibt p-Werte und Winrate-CI zurück; robust gegen leere/kleine/NaN-haltige Daten.
    """
    import numpy as np

    if trades_df is None or len(trades_df) == 0:
        return {}

    # Spalten robust detektieren
    r_col = None
    pnl_col = None
    for c in trades_df.columns:
        lc = str(c).lower()
        if r_col is None and lc in ("r_multiple", "r", "r-multiple"):
            r_col = c
        if pnl_col is None and lc in ("result", "pnl", "profit_eur", "result_eur"):
            pnl_col = c
        if r_col and pnl_col:
            break

    out = {}

    # Helper: sichere NumPy-Arrays (float64, finite) extrahieren
    def _clean_numeric(series):
        if series is None:
            return np.array([], dtype=np.float64)
        arr = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        return arr[np.isfinite(arr)]

    # p(mean r > 0)
    if r_col is not None:
        r_vals = _clean_numeric(trades_df[r_col])
        out["p_mean_r_gt_0"] = (
            round(bootstrap_p_value_mean_gt_zero(r_vals, n_boot=1000, seed=1234), 4)
            if r_vals.size >= 2
            else 1.0
        )

    # p(net profit > 0) über Trade-Results
    if pnl_col is not None:
        # Default: net-of-fees if fee columns exist (align with "after fees" metrics).
        pnl_series = trades_df[pnl_col]
        try:
            if "total_fee" in trades_df.columns:
                pnl_series = pd.to_numeric(pnl_series, errors="coerce") - pd.to_numeric(
                    trades_df["total_fee"], errors="coerce"
                )
            elif "entry_fee" in trades_df.columns and "exit_fee" in trades_df.columns:
                pnl_series = (
                    pd.to_numeric(pnl_series, errors="coerce")
                    - pd.to_numeric(trades_df["entry_fee"], errors="coerce")
                    - pd.to_numeric(trades_df["exit_fee"], errors="coerce")
                )
        except Exception:
            pnl_series = trades_df[pnl_col]
        pnl_vals = _clean_numeric(pnl_series)
        out["p_net_profit_gt_0"] = (
            round(bootstrap_p_value_mean_gt_zero(pnl_vals, n_boot=1000, seed=1234), 4)
            if pnl_vals.size >= 2
            else 1.0
        )

    # Winrate-CI (Wilson 95 %) – nur wenn mindestens 1 Trade
    if pnl_col is not None:
        pnl_vals = _clean_numeric(trades_df[pnl_col])
        if pnl_vals.size >= 1:
            wins = int((pnl_vals > 0).sum())
            total = int(pnl_vals.size)
            low, high = _wilson_interval(wins, total, conf=0.95)
            out["winrate_ci_low"] = round(low, 2)
            out["winrate_ci_high"] = round(high, 2)

    return out


# ========================================================================


def run_walkforward_window(
    window_info: Dict[str, Any],
    base_config: Dict[str, Any],
    param_grid: Dict[str, Any],
    n_trials: int,
    rating_thresholds: Dict[str, Any],
    walkforward_root: str,
    data_preload: Dict[str, Any],
    *,
    min_trades: int = 10,
    min_days_active: int = 5,
    kfold_splits: int = 3,
    robustness_jitter_frac: float = 0.08,
    robustness_repeats: int = 2,
    preload_mode: str = "window",
    export_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Optimiert ein Walkforward-Window (Train/Test) und exportiert Reports.
    Inklusive CV, Robustness-Score und erweitertem Instrumentation-Logging.
    """
    window_id = window_info["window_id"]
    train_start = window_info["train_start"]
    train_end = window_info["train_end"]
    test_start = window_info["test_start"]
    test_end = window_info["test_end"]

    train_window = f"{train_start.date()} → {train_end.date()}"
    test_window_str = f"{test_start.date()} → {test_end.date()}"

    window_folder = Path(walkforward_root) / f"window_{window_id:02d}"
    window_folder.mkdir(parents=True, exist_ok=True)
    instrumentation_path = window_folder / "instrumentation.json"

    stage_recorder = StageRecorder(
        scope=f"window_{window_id}",
        metadata={
            "window_id": window_id,
            "train_window": train_window,
            "test_window": test_window_str,
        },
    )
    stage_recorder.add_metadata(n_trials_requested=int(n_trials))

    print(f"\n🚀 Starte Window {window_id}: {train_start.date()} → {test_end.date()}")

    # --- TRAIN SETUP ----------------------------------------------------
    train_config = deepcopy(base_config)
    train_config["start_date"] = train_start.strftime("%Y-%m-%d")
    train_config["end_date"] = train_end.strftime("%Y-%m-%d")

    window_preload_train: Dict[str, Any] = {}
    window_preload_test: Dict[str, Any] = {}
    warmup_days = 0

    with stage_recorder.stage("prepare_window") as stage:
        stage.add_details(
            preload_mode=str(preload_mode),
            data_preload_keys=len(data_preload),
            train_span_days=(train_end - train_start).days,
            test_span_days=(test_end - test_start).days,
        )
        if data_preload:
            tf_primary = base_config.get("timeframes", {}).get("primary", "M15")
            bars_per_day = {
                "M1": 1440,
                "M5": 288,
                "M15": 96,
                "M30": 48,
                "H1": 24,
                "H4": 6,
                "D1": 1,
            }.get(tf_primary, 96)
            warmup_days = int(
                base_config.get("warmup_bars", 500) // max(1, bars_per_day)
            )
            stage.add_details(primary_tf=tf_primary, warmup_days=warmup_days)

            s_train = train_start - timedelta(days=max(0, warmup_days))
            e_train = train_end
            s_test = test_start - timedelta(days=max(0, warmup_days))
            e_test = test_end

            for key, df in data_preload.items():
                window_preload_train[key] = _safe_slice_df_by_time(df, s_train, e_train)
                window_preload_test[key] = _safe_slice_df_by_time(df, s_test, e_test)
            stage.add_details(preload_tables=len(window_preload_train))
        else:
            stage.add_details(
                primary_tf=base_config.get("timeframes", {}).get("primary")
            )

    stage_recorder.add_metadata(warmup_days=warmup_days)

    # --- OPTUNA TRAINING ------------------------------------------------
    with stage_recorder.stage("optuna_training") as stage:
        study = optimize_strategy_with_optuna_pareto(
            config_template=train_config,
            param_space=param_grid,
            n_trials=n_trials,
            preloaded_data=window_preload_train,
            use_pruner=True,
            pruner_warmup_folds=1,
            visualize=False,
            kfold_splits=kfold_splits,
            robustness_jitter_frac=robustness_jitter_frac,
            robustness_repeats=robustness_repeats,
            min_trades_threshold=max(2, min_trades // kfold_splits),
        )
        stage.add_details(_summarize_optuna_study(study))

    # --- TRIAL VERARBEITUNG --------------------------------------------
    with stage_recorder.stage("process_trials") as stage:
        all_trials: List[Dict[str, Any]] = []
        for t in study.trials:
            if getattr(t, "user_attrs", {}).get("invalid", False):
                continue
            entry = _round_floats(dict(t.params), 6)
            entry["number"] = t.number
            if hasattr(t, "values"):
                entry["profit"] = t.values[0]
                entry["avg_r"] = t.values[1]
                entry["winrate"] = t.values[2]
                entry["drawdown"] = t.values[3]
            elif hasattr(t, "value"):
                entry["score"] = t.value
            if hasattr(t, "user_attrs"):
                entry.update(t.user_attrs)
            entry["window_id"] = window_id
            entry["train_window"] = train_window
            entry["test_window"] = test_window_str
            all_trials.append(entry)

        trials_df = pd.DataFrame(all_trials)
        trials_df = _expand_and_round_score_column(
            trials_df, decimals=None, keep_original_list=True
        )
        trials_df = _apply_rounding_to_df(
            trials_df, NUM_ROUND_TRIALS, int_cols=INT_COLS_COMMON
        )
        trials_csv_path = window_folder / "all_trials_sensitivity.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        stage.add_details(
            trials_valid=len(all_trials),
            trials_csv=str(trials_csv_path),
            trials_csv_size_bytes=_safe_file_size(trials_csv_path),
        )

    # --- KANDIDATEN SELEKTION ------------------------------------------
    test_candidates: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    with stage_recorder.stage("candidate_selection") as stage:
        try:
            sel_cfg = get_selection_config({})
        except Exception:
            sel_cfg = {}
            stage.mark_error("selection_config_load_failed")

        try:
            sel_min_trades = int(
                sel_cfg.get("is_filter", {}).get("gates", {}).get("min_trades", 0)
            )
            sel_cfg.setdefault("is_filter", {}).setdefault("gates", {})[
                "min_trades"
            ] = max(sel_min_trades, max(0, int(min_trades) // 2))
        except Exception:
            pass

        test_candidates = _select_is_topN_candidates(
            trials_df, sel_cfg, fallback_topN=30
        )
        candidate_count = len(test_candidates)
        stage.add_details(candidate_count=candidate_count)

        best_params = test_candidates[0]["params"] if test_candidates else None
        if not isinstance(best_params, dict) or not best_params:
            metric_priority = [
                ("profit", False),
                ("total_trades", False),
                ("avg_r", False),
                ("winrate", False),
                ("drawdown", True),
                ("sharpe", False),
                ("calmar", False),
                ("sortino", False),
                ("profit_factor", False),
            ]
            by = [c for c, _ in metric_priority if c in trials_df.columns]
            ascending = [asc for c, asc in metric_priority if c in trials_df.columns]
            if by:
                fb = trials_df.sort_values(by=by, ascending=ascending).head(1)
            else:
                fb = trials_df.head(1)

            cols_exclude = {
                "number",
                "trial_number",
                "trial_id",
                "window_id",
                "train_window",
                "test_window",
                "score",
                "score_profit",
                "score_avg_r",
                "score_winrate",
                "score_drawdown",
                "base_seed",
                "profit",
                "total_trades",
                "avg_r",
                "winrate",
                "drawdown",
                "max_dd",
                "sharpe",
                "sortino",
                "calmar",
                "profit_factor",
                # DEAKTIVIERT: Robustness Score
                # "robustness_score",
                "kfold_score",
                "oos_score",
            }
            if not fb.empty:
                row = fb.iloc[0]
                if "params" in fb.columns and isinstance(row.get("params", None), dict):
                    best_params = dict(row["params"])
                    stage.add_details(best_params_source="fallback_params_field")
                else:
                    best_params = {
                        k: row[k]
                        for k in fb.columns
                        if (k not in cols_exclude) and pd.notna(row.get(k))
                    }
                    stage.add_details(best_params_source="fallback_row_values")
            else:
                best_params = {}
                stage.mark_error("fallback_best_params_empty")
        else:
            stage.add_details(best_params_source="candidate_list")

    if best_params is None:
        best_params = {}

    # --- TEST (OOS) -----------------------------------------------------
    summary: Dict[str, Any] = {}
    rating: Dict[str, Any] = {}
    trades_df = None
    with stage_recorder.stage("oos_backtest") as stage:
        test_config = deepcopy(base_config)
        test_config["start_date"] = test_start.strftime("%Y-%m-%d")
        test_config["end_date"] = test_end.strftime("%Y-%m-%d")
        if "strategy" not in test_config:
            test_config["strategy"] = {}
        if "parameters" not in test_config["strategy"] or not isinstance(
            test_config["strategy"]["parameters"], dict
        ):
            test_config["strategy"]["parameters"] = {}
        test_config["strategy"]["parameters"].update(best_params or {})

        portfolio, extra = run_backtest_and_return_portfolio(
            test_config,
            preloaded_data=window_preload_test,
        )
        try:
            if isinstance(extra, dict) and "trades" in extra:
                trades_df = (
                    pd.DataFrame(extra["trades"])
                    if not isinstance(extra["trades"], pd.DataFrame)
                    else extra["trades"]
                )
            elif hasattr(portfolio, "trades_to_dataframe"):
                trades_df = portfolio.trades_to_dataframe()
            elif hasattr(portfolio, "trades"):
                trades_df = pd.DataFrame(portfolio.trades)
        except Exception:
            trades_df = None

        summary = calculate_metrics(portfolio)
        summary.update(_compute_significance_from_trades(trades_df))

        data_issue_keys = []
        for k, v in list(summary.items()):
            if isinstance(v, numbers.Real):
                if not np.isfinite(v):
                    data_issue_keys.append(k)
        if data_issue_keys:
            summary["data_issue"] = True
            summary["data_issue_keys"] = data_issue_keys
        total_trades_val = int(summary.get("total_trades", 0) or 0)
        active_days_val = int(summary.get("active_days", 0) or 0)
        summary["valid_result"] = (
            (total_trades_val >= min_trades)
            and (active_days_val >= min_days_active)
            and not summary.get("data_issue", False)
        )

        summary.update(
            {
                "Net Profit": summary.pop("net_profit_after_fees_eur", 0),
                "Commission": summary.pop("fees_total_eur", 0.0),
                "Avg R-Multiple": summary.pop("avg_r_multiple", 0),
                "Winrate (%)": summary.pop("winrate_percent", 0),
                "Drawdown": summary.pop("drawdown_eur", 0),
                "Sharpe (trade)": summary.pop("sharpe_trade", 0.0),
                "Sortino (trade)": summary.pop("sortino_trade", 0.0),
                "Drawdown (%)": summary.pop("drawdown_percent", 0.0),
            }
        )

        rating = rate_strategy_performance(summary, thresholds=rating_thresholds)
        stage.add_details(
            total_trades=total_trades_val,
            active_days=active_days_val,
            valid_result=summary.get("valid_result"),
            net_profit=summary.get("Net Profit"),
            avg_r=summary.get("Avg R-Multiple"),
            rating_score=rating.get("Score"),
        )

    rounded_params = _round_floats(dict(best_params), 6)
    test_result = {
        **summary,
        "window_id": window_id,
        "train_window": train_window,
        "test_window": test_window_str,
        "params": rounded_params,
        **rating,
    }

    # --- TOP-KANDIDATEN OOS-RETEST --------------------------------------
    with stage_recorder.stage("top_candidate_retests") as stage:
        top_eval_rows: List[Dict[str, Any]] = []
        if test_candidates:
            cache: Dict[Any, Any] = {}
            cache_hits = 0
            cache_misses = 0
            for item in test_candidates:
                cand_params = item["params"]
                # DEAKTIVIERT: Robustness Score
                # cand_robust = float(item.get("robustness_score", 0.0))
                key = (tuple(sorted(cand_params.items())), window_id)
                if key in cache:
                    cache_hits += 1
                    summ_c, trades_c = cache[key]
                else:
                    cache_misses += 1
                    tc = deepcopy(base_config)
                    tc["start_date"] = test_start.strftime("%Y-%m-%d")
                    tc["end_date"] = test_end.strftime("%Y-%m-%d")
                    tc["strategy"]["parameters"].update(cand_params)

                    port_c, extra_c = run_backtest_and_return_portfolio(
                        tc,
                        preloaded_data=window_preload_test,
                    )
                    summ_c = calculate_metrics(port_c)
                    trades_c = None
                    try:
                        if isinstance(extra_c, dict) and "trades" in extra_c:
                            trades_c = pd.DataFrame(extra_c["trades"])
                        elif hasattr(port_c, "trades_to_dataframe"):
                            trades_c = port_c.trades_to_dataframe()
                    except Exception:
                        trades_c = None
                    cache[key] = (summ_c, trades_c)

                sig_c = (
                    _compute_significance_from_trades(trades_c)
                    if trades_c is not None
                    else {}
                )
                if "p_mean_r_gt_0" not in sig_c:
                    sig_c["p_mean_r_gt_0"] = 1.0
                if "p_net_profit_gt_0" not in sig_c:
                    sig_c["p_net_profit_gt_0"] = 1.0

                param_keys = (
                    set(param_grid.keys())
                    if isinstance(param_grid, dict)
                    else set(param_grid)
                )
                flat_params = _round_floats(
                    {k: v for k, v in dict(cand_params).items() if k in param_keys}, 6
                )

                row = {
                    "window_id": window_id,
                    "Net Profit": float(
                        summ_c.get("net_profit_after_fees_eur", 0.0) or 0.0
                    ),
                    "Commission": float(summ_c.get("fees_total_eur", 0.0) or 0.0),
                    "Avg R-Multiple": float(summ_c.get("avg_r_multiple", 0.0) or 0.0),
                    "Winrate (%)": float(summ_c.get("winrate_percent", 0.0) or 0.0),
                    "Drawdown": float(summ_c.get("drawdown_eur", 0.0) or 0.0),
                    "Drawdown (%)": float(summ_c.get("drawdown_percent", 0.0) or 0.0),
                    "initial_drawdown_eur": float(
                        summ_c.get("initial_drawdown_eur", 0.0) or 0.0
                    ),
                    "total_trades": _int_safe(summ_c.get("total_trades")),
                    "active_days": float(summ_c.get("active_days", 0.0) or 0.0),
                    "Sharpe (trade)": float(summ_c.get("sharpe_trade", 0.0) or 0.0),
                    "Sortino (trade)": float(summ_c.get("sortino_trade", 0.0) or 0.0),
                    # DEAKTIVIERT: Robustness Score
                    # "robustness_score": cand_robust,
                    **sig_c,
                    **flat_params,
                    "train_window": train_window,
                    "test_window": test_window_str,
                }
                top_eval_rows.append(row)

            top_eval_rows.sort(
                key=lambda r: (r.get("Sharpe (trade)", 0.0), r["Net Profit"]),
                reverse=True,
            )

            stage.add_details(
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hit_ratio=round(
                    cache_hits / max(1, cache_hits + cache_misses), 4
                ),
                top_rows=len(top_eval_rows),
            )

            if export_artifacts and top_eval_rows:
                df_top = pd.DataFrame(top_eval_rows)
                if "window_id" not in df_top.columns:
                    df_top["window_id"] = int(window_id)
                df_top = _apply_rounding_to_df(
                    df_top, NUM_ROUND_TOP, int_cols=INT_COLS_TOP
                )
                top_path = window_folder / "top_out_of_sample_results.csv"
                df_top.to_csv(top_path, index=False)
                stage.add_details(
                    top_csv=str(top_path),
                    top_csv_size_bytes=_safe_file_size(top_path),
                )
        else:
            stage.add_details(message="no_test_candidates")

    print(
        f"✅ Window {window_id} fertig – Score: {rating['Score']} | Ø R: {test_result['Avg R-Multiple']}"
    )

    window_metrics = stage_recorder.to_dict()
    summary_line = _format_stage_summary(window_metrics)
    if summary_line:
        print(f"📊 Window {window_id} Stagezeiten → {summary_line}")
    try:
        _write_json(instrumentation_path, window_metrics)
        print(f"📝 Instrumentation gespeichert: {instrumentation_path}")
    except Exception as exc:
        print(f"⚠️ Instrumentation-Export fehlgeschlagen: {exc}")

    test_result["runtime_seconds"] = round(
        float(window_metrics.get("total_duration_sec", 0.0) or 0.0), 4
    )
    test_result["instrumentation_path"] = str(instrumentation_path)

    # Joblib-Worker können mehrere Windows nacheinander ausführen.
    # Explizites GC hilft, kurzlebige große Objekte (DataFrames, Candles, Arrays)
    # schneller freizugeben, bevor der nächste Window-Run startet.
    gc.collect()

    return test_result


def walkforward_optimization(
    config_template_path: str,
    param_grid: Dict[str, Any],
    train_days: int = 90,
    test_days: int = 30,
    buffer_days: int = 3,
    roll_interval_days: int = 60,
    rating_thresholds: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    walkforward_root: str = "walkforward/",
    export_results_path: str = "walkforward/ratings_summary.json",
    seed: int = 42,
    *,
    preload_mode: str = "window",
    min_trades: int = 10,
    min_days_active: int = 5,
    kfold_splits: int = 3,
    robustness_jitter_frac: float = 0.08,
    robustness_repeats: int = 2,
    n_jobs: Optional[int] = None,
    final_mode: str = "smart",  # "smart" or "grid"
    export_artifacts: bool = True,
    analyze_after: bool = True,  # <— NEU
    analyze_alpha: float = 0.10,  # <— optional Tuning
    analyze_min_coverage: float = 0.10,  # <— optional Tuning
    analyze_min_sharpe_trade: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Vollständige Walkforward-Optimierung mit CV, Robustheitsprüfung und umfassendem Instrumentation-Logging.
    """
    preload_mode = str(preload_mode).strip().lower()
    global_time_start = time.time()
    set_global_seed(seed)

    walkforward_root_path = Path(walkforward_root)
    walkforward_root_path.mkdir(parents=True, exist_ok=True)

    pipeline_recorder = StageRecorder(
        scope="walkforward_pipeline",
        metadata={
            "seed": seed,
            "n_trials_requested": n_trials,
            "walkforward_root": str(walkforward_root_path.resolve()),
        },
    )

    with pipeline_recorder.stage("load_config") as stage:
        with open(config_template_path, "r") as f:
            base_config = json.load(f)
        stage.add_details(
            config_template=str(Path(config_template_path).resolve()),
            config_sha=_sha256_of(base_config),
            symbol=base_config.get("symbol"),
            start_date=base_config.get("start_date"),
            end_date=base_config.get("end_date"),
        )

    symbol = base_config.get("symbol")
    pipeline_recorder.add_metadata(symbol=symbol)
    tfs = base_config.get("timeframes", {"primary": "M1", "additional": []})
    all_tfs = [tfs.get("primary")] + tfs.get("additional", [])

    data_preload: Dict[Any, Any] = {}
    with pipeline_recorder.stage("data_preload") as stage:
        stage.add_details(preload_mode=str(preload_mode), timeframes=all_tfs)
        if preload_mode == "full":
            data_dir = PARQUET_DIR / symbol

            def _find_parquet(
                base: Path, sym: str, tf: str, side: str
            ) -> Optional[Path]:
                """Find parquet file, preferring uppercase BID/ASK."""
                upper_path = base / f"{sym}_{tf}_{side.upper()}.parquet"
                if upper_path.exists():
                    return upper_path
                lower_path = base / f"{sym}_{tf}_{side.lower()}.parquet"
                if lower_path.exists():
                    return lower_path
                return None

            for tf in all_tfs:
                if not tf:
                    continue
                bid_path = _find_parquet(data_dir, symbol, tf, "bid")
                ask_path = _find_parquet(data_dir, symbol, tf, "ask")
                try:
                    if bid_path:
                        data_preload[(tf, "bid")] = pd.read_parquet(bid_path)
                    if ask_path:
                        data_preload[(tf, "ask")] = pd.read_parquet(ask_path)
                except Exception as e:
                    stage.mark_error(f"preload_failed_{tf}")
                    print(f"⚠️ Preload fehlgeschlagen für {symbol} {tf}: {e}")
            stage.add_details(preload_tables=len(data_preload))
        elif preload_mode == "window":
            stage.add_details(message="per_window_preload")
            data_preload = {}
        else:
            stage.mark_error(f"unsupported_mode:{preload_mode}")
            print(
                f"ℹ️ preload_mode='{preload_mode}' nicht implementiert – verwende Fallback ohne Preload."
            )
            data_preload = {}

    window_ranges: List[Dict[str, Any]] = []
    with pipeline_recorder.stage("generate_windows") as stage:
        all_start = datetime.strptime(base_config["start_date"], "%Y-%m-%d")
        all_end = datetime.strptime(base_config["end_date"], "%Y-%m-%d")
        stage.add_details(
            overall_start=str(all_start.date()), overall_end=str(all_end.date())
        )

        window_id = 1
        current = all_start
        while True:
            train_start = current
            train_end = train_start + timedelta(days=train_days)
            buffer_start = train_end
            buffer_end = buffer_start + timedelta(days=buffer_days)
            test_start = buffer_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > all_end:
                break

            window_ranges.append(
                {
                    "window_id": window_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "buffer_start": buffer_start,
                    "buffer_end": buffer_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

            current += timedelta(days=roll_interval_days)
            window_id += 1

        stage.add_details(
            window_count=len(window_ranges), roll_interval_days=roll_interval_days
        )

    pipeline_recorder.add_metadata(window_count=len(window_ranges))

    effective_trials = n_trials
    n_jobs_local = n_jobs if n_jobs is not None else None
    with pipeline_recorder.stage("determine_trials") as stage:
        try:
            min_trials, opt_trials, _ = estimate_n_trials(param_grid)
            effective_trials = max(int(n_trials or 0), int(min_trials))
            if os.cpu_count() and os.cpu_count() >= 16:
                effective_trials = max(effective_trials, int(opt_trials))
            stage.add_details(min_trials=min_trials, opt_trials=opt_trials)
        except Exception as exc:
            stage.mark_error(f"estimate_trials_failed: {exc}")
            effective_trials = n_trials

        if n_jobs_local is None:
            try:
                cpu = os.cpu_count() or 4
                n_jobs_local = max(1, cpu - 1)
            except Exception:
                n_jobs_local = 3
        stage.add_details(effective_trials=effective_trials, n_jobs=n_jobs_local)

    n_jobs = int(n_jobs_local or 1)
    pipeline_recorder.add_metadata(
        effective_trials_per_window=effective_trials, n_jobs=n_jobs
    )

    walkforward_options = {
        "train_days": int(train_days),
        "test_days": int(test_days),
        "buffer_days": int(buffer_days),
        "roll_interval_days": int(roll_interval_days),
        "preload_mode": str(preload_mode),
        "min_trades": int(min_trades),
        "min_days_active": int(min_days_active),
        "kfold_splits": int(kfold_splits),
        "robustness_jitter_frac": float(robustness_jitter_frac),
        "robustness_repeats": int(robustness_repeats),
        "n_trials_requested": int(n_trials),
        "effective_trials": int(effective_trials),
    }
    baseline_manifest: Dict[str, Any] = {}
    with pipeline_recorder.stage("baseline_snapshot") as stage:
        baseline_manifest = _freeze_baseline_snapshot(
            walkforward_root,
            base_config=base_config,
            config_template_path=config_template_path,
            param_grid=param_grid,
            rating_thresholds=rating_thresholds,
            walkforward_options=walkforward_options,
        )
        stage.add_details(baseline_manifest)

    pipeline_recorder.add_metadata(
        export_results_path=str(Path(export_results_path).resolve())
    )

    with pipeline_recorder.stage("prepare_execution") as stage:
        try:
            tmp = (walkforward_root_path / "joblib_tmp").resolve()
            tmp.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(tmp))
            stage.add_details(joblib_tmp=str(tmp))
        except Exception as exc:
            stage.mark_error(f"joblib_tmp_failed: {exc}")

        if preload_mode == "full" and n_jobs > 1:
            print(
                "⚠️ preload_mode='full' + parallele Prozesse duplizieren große DataFrames pro Worker → hoher RAM-Verbrauch. "
                "Empfehlung: preload_mode='window' auf Systemen ≤ 16 GB RAM."
            )
        print(
            f"\n🚀 Starte Walkforward mit {n_jobs} parallelen Jobs... (Trials/Fenster: {effective_trials})"
        )

    with pipeline_recorder.stage("execute_windows") as stage:
        backend = (
            os.getenv("WALKFORWARD_PARALLEL_BACKEND", "loky").strip().lower() or "loky"
        )
        try:
            batch_size = int(os.getenv("WALKFORWARD_JOBLIB_BATCH_SIZE", "1") or 1)
        except Exception:
            batch_size = 1
        try:
            recycle_every = int(
                os.getenv("WALKFORWARD_RECYCLE_WORKERS_EVERY", "0") or 0
            )
        except Exception:
            recycle_every = 0

        stage.add_details(
            parallel_backend=backend,
            joblib_batch_size=batch_size,
            recycle_workers_every=recycle_every,
        )

        def _window_kwargs(window_info_local: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "window_info": window_info_local,
                "base_config": base_config,
                "param_grid": param_grid,
                "n_trials": effective_trials,
                "rating_thresholds": rating_thresholds,
                "walkforward_root": walkforward_root,
                "data_preload": data_preload,
                "min_trades": min_trades,
                "min_days_active": min_days_active,
                "kfold_splits": kfold_splits,
                "robustness_jitter_frac": robustness_jitter_frac,
                "robustness_repeats": robustness_repeats,
                "preload_mode": preload_mode,
                "export_artifacts": export_artifacts,
            }

        if recycle_every and recycle_every > 0:
            window_test_results = []
            for i in range(0, len(window_ranges), recycle_every):
                batch = window_ranges[i : i + recycle_every]
                batch_results = Parallel(
                    n_jobs=n_jobs, backend=backend, batch_size=batch_size
                )(delayed(run_walkforward_window)(**_window_kwargs(w)) for w in batch)
                window_test_results.extend(batch_results)
                # Worker-Recycling: beendet Child-Prozesse, damit resident memory wieder frei wird.
                try:
                    get_reusable_executor().shutdown(wait=True)
                except Exception as exc:
                    stage.mark_error(f"executor_shutdown_failed: {exc}")
                gc.collect()
        else:
            window_test_results = Parallel(
                n_jobs=n_jobs, backend=backend, batch_size=batch_size
            )(
                delayed(run_walkforward_window)(**_window_kwargs(window_info))
                for window_info in window_ranges
            )

        stage.add_details(window_count=len(window_test_results))

    window_instrumentation: List[Dict[str, Any]] = []
    with pipeline_recorder.stage("collect_window_metrics") as stage:
        for window in window_ranges:
            window_dir = walkforward_root_path / f"window_{window['window_id']:02d}"
            payload = _load_window_instrumentation(window_dir)
            if payload:
                window_instrumentation.append(payload)
        stage.add_details(
            collected=len(window_instrumentation), expected=len(window_ranges)
        )

    pipeline_recorder.add_metadata(collected_window_metrics=len(window_instrumentation))

    with pipeline_recorder.stage("export_ratings") as stage:
        export_path = Path(export_results_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f_out:
            json.dump(
                _to_jsonable(window_test_results), f_out, indent=2, ensure_ascii=False
            )
        stage.add_details(ratings_path=str(export_path.resolve()))
        print(f"✅ Ratings gespeichert: {export_path}")

    if export_artifacts:
        with pipeline_recorder.stage("aggregate_top_artifacts") as stage:
            try:
                all_top = []
                for w in window_ranges:
                    p = (
                        Path(walkforward_root)
                        / f"window_{w['window_id']:02d}"
                        / "top_out_of_sample_results.csv"
                    )
                    if p.exists():
                        try:
                            all_top.append(pd.read_csv(p))
                        except Exception:
                            pass
                if all_top:
                    all_top_df = pd.concat(all_top, ignore_index=True)
                    _drop_cols = [
                        c
                        for c in all_top_df.columns
                        if c in ("base_seed", "fees_total_sum", "fees_total_mean")
                        or (
                            isinstance(c, str)
                            and c.startswith("fees_")
                            and c != "Commission"
                        )
                    ]
                    if _drop_cols:
                        all_top_df = all_top_df.drop(
                            columns=_drop_cols, errors="ignore"
                        )
                    all_top_df = _apply_rounding_to_df(
                        all_top_df, NUM_ROUND_TOP, int_cols=INT_COLS_TOP
                    )
                    out_top_path = Path(walkforward_root) / "all_top_out_of_sample.csv"
                    all_top_df.to_csv(out_top_path, index=False)
                    stage.add_details(top_csv=str(out_top_path))
                    print(
                        f"✅ Top-Out-Of-Sample je Window (Test, Profit>0) konsolidiert: {out_top_path}"
                    )
            except Exception as ex:
                stage.mark_error(str(ex))
                print(f"⚠️ Sammel-Top-Export fehlgeschlagen: {ex}")

    with pipeline_recorder.stage("cleanup") as stage:
        try:
            get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            stage.mark_error(str(exc))
        gc.collect()

    total_runtime_sec = time.time() - global_time_start
    pipeline_recorder.add_metadata(total_runtime_sec=round(total_runtime_sec, 3))
    print(f"⏱️ Gesamtzeit Walkforward-Optimierung: {total_runtime_sec:.1f} Sekunden.")

    pipeline_summary = pipeline_recorder.to_dict()
    stage_summary_line = _format_stage_summary(pipeline_summary)
    if stage_summary_line:
        print(f"📊 Pipeline Stagezeiten → {stage_summary_line}")

    window_stage_totals = _aggregate_window_stage_totals(window_instrumentation)
    slowest_windows = []
    try:
        sorted_windows = sorted(
            window_instrumentation,
            key=lambda x: float(x.get("total_duration_sec", 0.0) or 0.0),
            reverse=True,
        )
        for entry in sorted_windows[:3]:
            meta = entry.get("metadata", {})
            slowest_windows.append(
                {
                    "window_id": meta.get("window_id"),
                    "duration_sec": entry.get("total_duration_sec"),
                    "train_window": meta.get("train_window"),
                    "test_window": meta.get("test_window"),
                }
            )
    except Exception:
        slowest_windows = []

    instrumentation_summary = {
        "pipeline": pipeline_summary,
        "windows": window_instrumentation,
        "window_stage_totals": window_stage_totals,
        "slowest_windows": slowest_windows,
        "baseline": baseline_manifest,
        "total_runtime_sec": round(total_runtime_sec, 3),
    }

    instrumentation_file = walkforward_root_path / "instrumentation_summary.json"
    try:
        _write_json(instrumentation_file, instrumentation_summary)
        print(f"📝 Instrumentation-Summary gespeichert: {instrumentation_file}")
    except Exception as exc:
        print(f"⚠️ Instrumentation-Summary konnte nicht gespeichert werden: {exc}")

    with pipeline_recorder.stage("export_manifest") as stage:
        try:
            _export_walkforward_run_config(
                out_dir=walkforward_root,
                base_config=base_config,
                config_template_path=config_template_path,
                param_grid=param_grid,
                rating_thresholds=rating_thresholds,
                train_days=train_days,
                test_days=test_days,
                buffer_days=buffer_days,
                roll_interval_days=roll_interval_days,
                preload_mode=preload_mode,
                min_trades=min_trades,
                min_days_active=min_days_active,
                kfold_splits=kfold_splits,
                robustness_jitter_frac=robustness_jitter_frac,
                robustness_repeats=robustness_repeats,
                n_jobs=n_jobs,
                seed=seed,
                analyze_after=analyze_after,
                analyze_alpha=analyze_alpha,
                analyze_min_coverage=analyze_min_coverage,
                analyze_min_sharpe_trade=analyze_min_sharpe_trade,
                window_ranges=window_ranges,
                effective_trials=effective_trials,
                export_results_path=export_results_path,
                run_started_ts=global_time_start,
                run_finished_ts=time.time(),
                instrumentation_overview=instrumentation_summary,
            )
            stage.add_details(
                manifest_path=str(
                    (Path(walkforward_root) / "walkforward_run_config.json").resolve()
                )
            )
        except Exception as exc:
            stage.mark_error(str(exc))
            print(f"⚠️ Run-Dokumentation konnte nicht geschrieben werden: {exc}")

    with pipeline_recorder.stage("robust_zone_analysis") as stage:
        if analyze_after:
            try:
                print("🧪 Starte automatische Robust-Zonen-Analyse...")
                report_path, analysis_recorder = run_robust_zone_analysis(
                    walkforward_root=walkforward_root,
                    param_grid=param_grid,
                    analyze_alpha=analyze_alpha,
                    analyze_min_coverage=analyze_min_coverage,
                    analyze_min_sharpe_trade=analyze_min_sharpe_trade,
                )
                instrumentation_path = (
                    Path(walkforward_root) / "analysis" / "instrumentation.json"
                )
                stage.add_details(
                    report_path=str(report_path),
                    instrumentation_path=str(instrumentation_path),
                    stage_summary=_format_stage_summary(analysis_recorder.to_dict()),
                )
                print(f"✅ Analyse abgeschlossen. Report: {report_path}")
            except Exception as exc:
                stage.mark_error(str(exc))
                print(f"⚠️ Analyse konnte nicht durchgeführt werden: {exc}")

    with pipeline_recorder.stage("final_parameter_selection") as stage:
        try:
            print(
                "🔎 Starte finale Parameter-Selektion (Grid über robuste Zonen, Halbjahrs-Checks, Robustness & Score)..."
            )
            final_report, final_recorder = run_final_parameter_selection(
                walkforward_root=walkforward_root,
                base_config=base_config,
                config_template_path=config_template_path,
                param_grid=param_grid,
                preload_mode=preload_mode,
                n_jobs=n_jobs,
                search_mode=final_mode,
                smart_n_trials=effective_trials,
            )
            instrumentation_path = (
                Path(walkforward_root) / "final_selection" / "instrumentation.json"
            )
            stage.add_details(
                final_report=str(final_report),
                instrumentation_path=str(instrumentation_path),
                stage_summary=_format_stage_summary(final_recorder.to_dict()),
                n_jobs_final=n_jobs,
            )
            print(
                f"✅ Finale Parameter-Selektion abgeschlossen. Report: {final_report}"
            )
        except Exception as exc:
            stage.mark_error(str(exc))
            print(f"⚠️ Finale Parameter-Selektion fehlgeschlagen: {exc}")

    _send_walkforward_telegram_reminder(
        strategy_name=_extract_strategy_name(base_config),
        symbol=symbol,
        walkforward_root=walkforward_root_path,
        export_results_path=export_results_path,
        total_runtime_sec=total_runtime_sec,
    )

    return window_test_results
