import json
import os
import random
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import NSGAIISampler, TPESampler

from backtest_engine.report.metrics import calculate_metrics
from backtest_engine.runner import (
    _get_or_build_alignment,
    load_data,
    prepare_time_window,
    run_backtest_and_return_portfolio,
)


# =========================
# Visualisierung (unverändert)
# =========================
def visualize_pareto(study: optuna.Study) -> None:
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_pareto_front,
    )

    print("📊 Öffne Pareto-Visualisierung...")
    plot_pareto_front(
        study, target_names=["Profit", "Avg R", "Winrate", "Drawdown"]
    ).show()
    plot_parallel_coordinate(study).show()
    plot_optimization_history(study).show()


# =========================
# Helfer
# =========================
_DATE_FMT = "%Y-%m-%d"
_THREADS_SET = False


def _configure_optuna_experimental_warnings() -> None:
    """Reduziert Optuna-Warnungsrauschen bei Multi-Processing.

    Optuna markiert `constraints_func` (für NSGA-II Constraints) als experimentell und
    gibt pro Prozess/Worker ein `ExperimentalWarning` aus. Im Walkforward-Setup wird
    Optuna typischerweise in mehreren Workern initialisiert, wodurch das Warning sehr
    laut wird, ohne den Lauf tatsächlich zu beeinflussen.

    Setze `OPTUNA_SHOW_EXPERIMENTAL_WARNINGS=1`, um die Warnungen wieder zu sehen.
    """

    if os.getenv("OPTUNA_SHOW_EXPERIMENTAL_WARNINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    try:
        from optuna._experimental import ExperimentalWarning
    except Exception:
        return

    warnings.filterwarnings(
        "ignore",
        category=ExperimentalWarning,
        message=r".*constraints_func.*",
    )


def _set_thread_env_once():
    global _THREADS_SET
    if _THREADS_SET:
        return
    # Begrenze NumPy/BLAS Threads, um Multi-Trials nicht zu überparallellisieren
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _THREADS_SET = True


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, _DATE_FMT)


def _fmt_date(dt: datetime) -> str:
    return dt.strftime(_DATE_FMT)


# Decimal-Genauigkeit hoch genug wählen, um Step-Snapping exakt zu halten
getcontext().prec = 28


def _snap_to_step(value: float, low: float, step: float) -> float:
    """
    Snapt value exakt auf eine diskrete Stufe 'low + k*step' (k ∈ Z),
    rundet dabei sauber (ROUND_HALF_UP). Gibt float zurück (für Downstream).
    """
    d_val = Decimal(str(value))
    d_low = Decimal(str(low))
    d_step = Decimal(str(step))
    k = ((d_val - d_low) / d_step).to_integral_value(rounding=ROUND_HALF_UP)
    return float(d_low + k * d_step)


def _split_train_period(
    start_s: str, end_s: str, k: int, min_days: int = 5
) -> List[Tuple[str, str]]:
    """Zeitbasierte, aufeinanderfolgende Splits innerhalb des Trainingsfensters.
    Erzwingt mind. 'min_days' pro Fold (falls möglich)."""
    if k <= 1:
        return [(start_s, end_s)]
    start = _parse_date(start_s)
    end = _parse_date(end_s)
    total_days = (end - start).days
    if total_days <= 0:
        return [(start_s, end_s)]

    # Obergrenze k, sodass min_days eingehalten werden kann
    max_k = max(1, total_days // max(1, min_days))
    k = int(max(1, min(k, max_k)))

    seg = (end - start) / k
    out: List[Tuple[str, str]] = []
    for i in range(k):
        s = start + seg * i
        e = start + seg * (i + 1)
        s = datetime(s.year, s.month, s.day)
        e = datetime(e.year, e.month, e.day)
        if e <= s:
            e = s + timedelta(days=min_days)
        out.append((_fmt_date(s), _fmt_date(e)))
    out[-1] = (out[-1][0], _fmt_date(end))
    return out


def _split_base_config(
    config_template: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Zerlegt die geladene Config in:
      - base_fixed: alles außer 'start_date', 'end_date', strategy.parameters
      - base_params: dict mit den (ggf. leeren) Strategy-Parametern
    """
    base_fixed = {
        k: v
        for k, v in config_template.items()
        if k not in ("start_date", "end_date", "strategy")
    }
    # Strategy-Block separat verwalten
    strat_block = config_template.get("strategy", {})
    strat_other = {k: v for k, v in strat_block.items() if k not in ("parameters",)}
    base_params = deepcopy(strat_block.get("parameters", {}))

    fixed = dict(base_fixed)
    fixed["strategy"] = dict(strat_other)  # ohne parameters
    # Preserve default strategy parameters so trials inherit non-optimized values.
    fixed["strategy"]["parameters"] = deepcopy(base_params)
    return fixed, base_params


def _build_trial_config(
    base_fixed: Dict[str, Any], params: Dict[str, Any], start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Baut eine flache Trial-Config ohne deepcopy-Orgie.
    """
    cfg = deepcopy(base_fixed)  # klein & schnell
    cfg["start_date"] = start_date
    cfg["end_date"] = end_date
    # Strategy zusammenbauen
    if "strategy" not in cfg:
        cfg["strategy"] = {}
    strat = cfg["strategy"]
    # existierende keys (class/module/...) bleiben erhalten
    current_params = dict(strat.get("parameters", {}))
    current_params.update(params or {})
    strat["parameters"] = current_params
    # Phase-A: Trial-Runs ohne Diagnose/Entry-Logging
    # (Kein Over-Engineering: nur hart ausschalten, wenn Feld existiert/nützlich)
    cfg["enable_entry_logging"] = False
    ts_align = cfg.get("timestamp_alignment", {})
    if not isinstance(ts_align, dict):
        ts_align = {}
    ts_align["diagnostics"] = False
    cfg["timestamp_alignment"] = ts_align
    return cfg


def _evaluate_config(
    conf: Dict[str, Any],
    preloaded_data: Optional[Dict[str, Any]] = None,
    _prealigned_cache: Optional[Dict[Tuple[str, str, str, str], Tuple]] = None,
) -> Dict[str, Any]:
    """Führt Backtest aus und gibt Metrik-Dict zurück (mit sicheren Defaults)."""
    prealigned = None
    if _prealigned_cache is not None:
        # Key: (symbol, primary_tf, start_date, end_date)
        sym = conf.get("symbol") or (
            conf.get("multi_symbols") and list(conf["multi_symbols"].keys())[0]
        )
        primary_tf = conf.get("timeframes", {}).get("primary", "M15")
        key = (sym, primary_tf, conf["start_date"], conf["end_date"])
        prealigned = _prealigned_cache.get(key)
        if prealigned is None:
            # Daten laden (nur für dieses Fenster), Alignment 1× berechnen und cachen
            start_dt, end_dt, extended_start, _ = prepare_time_window(
                conf
            )  # falls hier verfügbar; sonst importieren
            symbol_map, bid_candles, ask_candles, tick_data = load_data(
                conf,
                conf.get("mode", "candle"),
                extended_start,
                end_dt,
                preloaded_data=preloaded_data,
            )
            bid_aligned, ask_aligned, multi_candle_data_aligned = (
                _get_or_build_alignment(
                    symbol_map=symbol_map,
                    primary_tf=primary_tf,
                    config=conf,
                    start_dt=start_dt,
                )
            )
            prealigned = (bid_aligned, ask_aligned, multi_candle_data_aligned)
            _prealigned_cache[key] = prealigned

    portfolio, entry_df = run_backtest_and_return_portfolio(
        conf, preloaded_data=preloaded_data, prealigned=prealigned
    )
    return calculate_metrics(portfolio)


def _jitter_value(
    val, space: Dict[str, Any], frac: float, rnd: Optional[callable] = None
):
    """Jitter für float/int Parameter innerhalb der Bounds."""
    if rnd is None:
        rnd = random.random
    typ = space["type"]
    if typ == "categorical":
        return val  # kein Jitter
    low, high = space["low"], space["high"]
    if typ == "float":
        # relative Schwankung, fallback auf Range-basiert
        delta = abs(val) * frac
        if delta == 0:
            delta = (high - low) * (frac * 0.5)
        new_val = val + (delta if rnd() < 0.5 else -delta)
        new_val = max(low, min(high, new_val))
        step = space.get("step")
        if step:
            # exakt auf Step snappen (Decimal), anschließend clampen
            new_val = _snap_to_step(new_val, low, step)
            new_val = max(low, min(high, new_val))
        return new_val
    if typ == "int":
        delta = max(1, round(abs(val) * frac))
        new_val = val + (delta if rnd() < 0.5 else -delta)
        step = space.get("step", 1)
        # auf Schritt runden & clampen
        new_val = max(low, min(high, int(round((new_val - low) / step) * step + low)))
        return new_val
    raise ValueError(f"Unknown param type for jitter: {typ}")


def _aggregate_folds(fold_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not fold_metrics:
        return {
            "profit": 0.0,
            "avg_r": 0.0,
            "winrate": 0.0,
            "drawdown": 1e9,
            "total_trades": 0,
            "profit_sum": 0.0,
            "avg_r_sum": 0.0,
            "drawdown_mean": 1e9,
            "fees_mean": 0.0,
            "fees_sum": 0.0,
        }
    n = len(fold_metrics)
    profit_list = [m.get("net_profit_after_fees_eur", 0.0) for m in fold_metrics]
    avg_r_list = [m.get("avg_r_multiple", 0.0) for m in fold_metrics]
    winrate_list = [m.get("winrate_percent", 0.0) for m in fold_metrics]
    dd_list = [m.get("drawdown_eur", 0.0) for m in fold_metrics]
    fees_list = [m.get("fees_total_eur", 0.0) for m in fold_metrics]

    def _trades(m):
        return int(m.get("total_trades", m.get("trades", 0)) or 0)

    trades = sum(_trades(m) for m in fold_metrics)

    profit_mean = sum(profit_list) / n
    avg_r_mean = sum(avg_r_list) / n
    winrate_mean = sum(winrate_list) / n
    drawdown_worst = max(dd_list)
    drawdown_mean = sum(dd_list) / n
    fees_mean = sum(fees_list) / n

    return {
        "profit": profit_mean,
        "avg_r": avg_r_mean,
        "winrate": winrate_mean,
        "drawdown": drawdown_worst,
        "total_trades": trades,
        # NEU: übersichtlich gekennzeichnete Aggregationen
        "profit_sum": sum(profit_list),
        "avg_r_sum": sum(avg_r_list),
        "drawdown_mean": drawdown_mean,
        "fees_mean": fees_mean,
        "fees_sum": sum(fees_list),
    }


def _robustness_penalty(
    base: Dict[str, float], samples: List[Dict[str, float]]
) -> float:
    """
    Penalty 0..0.5 basierend auf durchschnittlichen prozentualen Einbrüchen.
    Bewertet Profit, AvgR, Winrate negativ; Drawdown positiv (steigt).
    """
    if not samples:
        return 0.0
    drops = []
    for s in samples:
        # Schutz gegen 0-Division
        def pct_drop(b, x, invert=False):
            if invert:  # Drawdown: je größer desto schlechter → Zuwachs ist "Drop"
                b = max(b, 1e-9)
                x = max(x, 1e-9)
                return max(0.0, (x - b) / b)
            else:
                b = max(b, 1e-9)
                return max(0.0, (b - x) / b)

        d1 = pct_drop(base["profit"], s["profit"])
        d2 = pct_drop(base["avg_r"], s["avg_r"])
        d3 = pct_drop(base["winrate"], s["winrate"])
        d4 = pct_drop(base["drawdown"], s["drawdown"], invert=True)
        drops.append((d1 + d2 + d3 + d4) / 4.0)
    pen = sum(drops) / len(drops)
    return float(max(0.0, min(0.5, pen)))


# =========================
# Hauptoptimierer (erweitert)
# =========================
def optimize_strategy_with_optuna_pareto(
    config_template: Union[str, Dict[str, Any]],
    param_space: Dict[str, Dict[str, Any]],
    n_trials: int = 50,
    seed: int = 42,
    visualize: bool = False,
    preloaded_data: Optional[Dict[str, Any]] = None,
    pruner_warmup_folds: int = 1,
    use_pruner: bool = True,
    *,
    kfold_splits: int = 1,
    robustness_jitter_frac: float = 0.05,
    robustness_repeats: int = 0,
    min_trades_threshold: int = 2,
) -> optuna.Study:
    """
    Multi-Objective Pareto-Optimierung mit optionaler zeitbasierter K-Fold-CV
    und integrierter Robustheitsprüfung (Parameter-Jitter).

    Ziele: Profit, AvgR, Winrate = maximize | Drawdown = minimize
    """
    # Lade ggf. JSON-Konfig
    if isinstance(config_template, str):
        with open(config_template, "r") as f:
            config_template = json.load(f)

    # Threads begrenzen
    _set_thread_env_once()
    _configure_optuna_experimental_warnings()
    # Study mit optionalem Pruner
    directions = ["maximize", "maximize", "maximize", "minimize"]
    # Optuna unterstützt trial.report/Pruning NICHT für Multi-Objective.
    # Deshalb Pruner hier nur für Single-Objective aktivieren (Safety-Switch).
    is_multi_objective = len(directions) > 1
    pruner = None
    if use_pruner and not is_multi_objective:
        pruner = MedianPruner(n_warmup_steps=max(1, int(pruner_warmup_folds)))

    def _constraints_func(ft):
        inv = (
            1.0
            if ft.user_attrs.get("invalid", False)
            or ft.user_attrs.get("pruned_like", False)
            else 0.0
        )
        need = int(min_trades_threshold) - int(ft.user_attrs.get("trades_sum", 0))
        min_tr_violation = float(max(0, need))
        dd_worst = float(ft.user_attrs.get("drawdown_worst", 0.0))
        dd_cap = float(os.getenv("DD_CAP", "1e9"))
        dd_violation = float(max(0.0, dd_worst - dd_cap))
        return (inv, min_tr_violation, dd_violation)

    study = optuna.create_study(
        directions=directions,
        sampler=NSGAIISampler(seed=seed, constraints_func=_constraints_func),
        study_name=f"opt_{int(time.time())}",
    )

    # Vorbereiten CV-Splits
    train_start_s = config_template.get("start_date")
    train_end_s = config_template.get("end_date")
    cv_splits = _split_train_period(
        train_start_s, train_end_s, max(1, int(kfold_splits))
    )

    # Base-Fixed Teil der Config einmal erstellen (spart deepcopy im Loop)
    base_fixed, _ = _split_base_config(config_template)

    EARLY_MIN_TRADES_FOLD0 = max(2, int(min_trades_threshold))  # konservativ
    EARLY_MAX_NEG_DD_FOLD0 = float(
        os.getenv("EARLY_MAX_NEG_DD_FOLD0", "1e9")
    )  # optionaler Guard

    def objective(trial: optuna.Trial):
        # reproduzierbarer RNG & Seeds pro Trial (wichtig bei Parallelisierung)
        base_seed = seed + trial.number
        random.seed(base_seed)
        np.random.seed(base_seed)

        # Parameter vorschlagen
        params = {}
        for param, space in param_space.items():
            typ = space["type"]
            if typ == "float":
                step = space.get("step")
                is_log = bool(space.get("log", False))
                if step is not None and is_log:
                    raise ValueError(
                        f"Param '{param}': 'step' und 'log' gleichzeitig gesetzt – wähle eines."
                    )
                value = trial.suggest_float(
                    param, space["low"], space["high"], step=step, log=is_log
                )
                # Exaktes Einrasten auf Step (nur wenn 'step' definiert und nicht log)
                if step is not None and not is_log:
                    value = _snap_to_step(value, space["low"], step)
            elif typ == "int":
                value = trial.suggest_int(
                    param, space["low"], space["high"], step=space.get("step", 1)
                )
            elif typ == "categorical":
                value = trial.suggest_categorical(param, space["choices"])
            else:
                raise ValueError(f"Unknown param type: {space['type']}")
            params[param] = value

        # Meta-Infos für spätere Auswertung
        trial.set_user_attr("base_seed", int(base_seed))

        # === CV-Auswertung
        fold_metrics = []
        prealigned_cache: Dict[Tuple[str, str, str, str], Tuple] = {}
        try:
            for fold_idx, (s, e) in enumerate(cv_splits):
                conf_fold = _build_trial_config(
                    base_fixed=base_fixed, params=params, start_date=s, end_date=e
                )

                summary = _evaluate_config(
                    conf_fold, preloaded_data, _prealigned_cache=prealigned_cache
                )
                fold_metrics.append(summary)

                # ---------- EARLY EXIT nach Fold-0 ----------
                if fold_idx == 0:
                    t = int(summary.get("total_trades", 0) or 0)
                    avg_r = float(summary.get("avg_r_multiple", 0.0) or 0.0)
                    net = float(summary.get("net_profit_after_fees_eur", 0.0) or 0.0)
                    # Metrik-Gates: zu wenig Stichprobe ODER klar negative Qualität
                    if (t < EARLY_MIN_TRADES_FOLD0) or (net < 0.0 and avg_r < 0.0):
                        trial.set_user_attr("pruned_like", True)
                        trial.set_user_attr(
                            "prune_reason", "fold0_min_trades_or_dual_negative"
                        )
                        # Optional: skaliere die Strafe für stabilere Sortierung
                        penalty = 1e6 + float(max(0, EARLY_MIN_TRADES_FOLD0 - t)) * 1e4
                        return [0.0, 0.0, 0.0, penalty]

                # Optionales Pruning NUR bei Single-Objective:
                # trial.report ist in Multi-Objective nicht erlaubt.
                if pruner is not None:
                    partial_score = float(summary.get("avg_r_multiple", 0.0))
                    trial.report(partial_score, step=fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        except optuna.TrialPruned:
            # Bei Pruning neutrale Rückgabe (wird intern behandelt)
            raise
        except MemoryError as e:
            # OOM darf NICHT still in eine Penalty umgewandelt werden – sonst sieht
            # der Nutzer nur „genullte“ Ergebnisse.
            try:
                trial.set_user_attr("error", "memory_error")
            except Exception:
                pass
            print(f"❌ Trial abgebrochen (MemoryError): {e}")
            raise
        except Exception as e:
            try:
                trial.set_user_attr("error", str(e)[:500])
            except Exception:
                pass
            print(f"❌ Trial fehlgeschlagen (CV): {e}")
            return [0.0, 0.0, 0.0, 1e6]

        agg = _aggregate_folds(fold_metrics)

        # Mindestanforderung
        if int(agg.get("total_trades", 0) or 0) < min_trades_threshold:
            trial.set_user_attr("invalid", True)
            return [0.0, 0.0, 0.0, 1e6]

        # DEAKTIVIERT: Robustness Score
        # Hinweis: Robustness temporär deaktiviert; Reaktivierung durch Entfernen der Kommentare.
        # robust_samples = []
        # if robustness_repeats > 0 and robustness_jitter_frac > 0:
        #     trial_rng = random.Random(base_seed)  # stabil je Trial
        #     for _ in range(int(robustness_repeats)):
        #         jittered = {}
        #         for p, space in param_space.items():
        #             jittered[p] = _jitter_value(
        #                 params[p], space, robustness_jitter_frac, rnd=trial_rng.random
        #             )
        #
        #         conf_jit = _build_trial_config(
        #             base_fixed=base_fixed,
        #             params=jittered,
        #             start_date=train_start_s,
        #             end_date=train_end_s,
        #         )
        #
        #         try:
        #             summary_j = _evaluate_config(
        #                 conf_jit, preloaded_data, _prealigned_cache=prealigned_cache
        #             )
        #
        #             trades_j = int(
        #                 summary_j.get("trades", summary_j.get("total_trades", 0))
        #             )
        #             if trades_j < min_trades_threshold:
        #                 # zu wenige Trades → harte Penalty für Robustness
        #                 robust_samples.append(
        #                     {
        #                         "profit": 0.0,
        #                         "avg_r": 0.0,
        #                         "winrate": 0.0,
        #                         "drawdown": 1e9,
        #                     }
        #                 )
        #             else:
        #                 # Robustness-Jitter bewertet ebenfalls den Profit NACH Fees
        #                 robust_samples.append(
        #                     {
        #                         "profit": float(
        #                             summary_j.get("net_profit_after_fees_eur", 0.0)
        #                         ),
        #                         "avg_r": float(summary_j.get("avg_r_multiple", 0.0)),
        #                         "winrate": float(summary_j.get("winrate_percent", 0.0)),
        #                         "drawdown": float(summary_j.get("drawdown_eur", 1e9)),
        #                     }
        #                 )
        #
        #         except Exception:
        #             # fehlgeschlagener Robustness-Lauf zählt als maximaler Penalty
        #             robust_samples.append(
        #                 {"profit": 0.0, "avg_r": 0.0, "winrate": 0.0, "drawdown": 1e9}
        #             )

        base_scores = {
            "profit": agg["profit"],
            "avg_r": agg["avg_r"],
            "winrate": agg["winrate"],
            "drawdown": agg["drawdown"],
        }
        # DEAKTIVIERT: Robustness Score
        # penalty = _robustness_penalty(base_scores, robust_samples)
        # robustness_score = float(max(0.0, 1.0 - penalty))  # 1.0 = sehr robust

        profit = base_scores["profit"]
        avg_r = base_scores["avg_r"]
        winrate = base_scores["winrate"]
        drawdown = base_scores["drawdown"]
        # Für konsistente Darstellung downstream runden wir die im Study gespeicherten User-Attribute
        r2 = lambda x: float(round(float(x), 2))
        r3 = lambda x: float(round(float(x), 3))

        # User-Attribute für spätere Auswahl/Reporting
        # trial.set_user_attr("cv_profit_mean", float(agg.get("profit", 0.0)))
        # trial.set_user_attr("cv_avg_r_mean", float(agg.get("avg_r", 0.0)))
        # trial.set_user_attr("cv_winrate_mean", float(agg.get("winrate", 0.0)))
        # trial.set_user_attr("cv_drawdown_worst", float(agg.get("drawdown", 0.0)))
        # trial.set_user_attr("cv_profit_sum", float(agg.get("profit_sum", 0.0)))
        # trial.set_user_attr("cv_avg_r_sum", float(agg.get("avg_r_sum", 0.0)))
        # trial.set_user_attr("cv_drawdown_mean", float(agg.get("drawdown_mean", 0.0)))
        trial.set_user_attr("fees_total_sum", agg.get("fees_sum", 0.0))
        trial.set_user_attr("total_trades", agg.get("total_trades", 0.0))
        # DEAKTIVIERT: Robustness Score
        # trial.set_user_attr("robustness_score", robustness_score)

        return [r2(profit), r3(avg_r), r2(winrate), r2(drawdown)]

    # Optimize (mit optionalem Pruning)
    study.optimize(objective, n_trials=n_trials)

    if visualize and len(study.directions) <= 3:
        visualize_pareto(study)

    return study
