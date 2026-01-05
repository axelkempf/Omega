# mypy: disable-error-code="no-untyped-def,no-untyped-call,arg-type,no-any-return,assignment,return-value,misc,union-attr,operator,unused-ignore,unreachable,no-redef,comparison-overlap,call-overload,type-arg"
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest_engine.data.data_handler import (
    reset_candle_build_caches,
    trim_candle_build_caches,
)
from backtest_engine.optimizer.instrumentation import (
    StageRecorder,
    _format_stage_summary,
)
from backtest_engine.rating.cost_shock_score import (
    COST_SHOCK_FACTORS,
    apply_cost_shock_inplace,
    compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.data_jitter_score import (
    _stable_data_jitter_seed,
    build_jittered_preloaded_data,
    compute_data_jitter_score,
    precompute_atr_cache,
)
from backtest_engine.rating.p_values import compute_p_values
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_and_wmape_from_yearly_profits,
)
from backtest_engine.rating.timing_jitter_score import (
    apply_timing_jitter_month_shift_inplace,
    compute_timing_jitter_score,
    get_timing_jitter_backward_shift_months,
)
from backtest_engine.rating.tp_sl_stress_score import (
    PrimaryCandleArrays,
    align_primary_candles,
    compute_tp_sl_stress_score,
)
from backtest_engine.rating.trade_dropout_score import (
    compute_multi_run_trade_dropout_score,
    simulate_trade_dropout_metrics_multi,
)
from backtest_engine.rating.ulcer_index_score import compute_ulcer_index_and_score
from backtest_engine.report.metrics import calculate_metrics
from backtest_engine.runner import (
    clear_alignment_cache,
    run_backtest_and_return_portfolio,
)
from hf_engine.infra.config.paths import PARQUET_DIR

_SEED_MODULUS = 2**32


def _stable_int_seed(base_seed: int, *parts: object) -> int:
    """
    Stable 32-bit seed derived from (base_seed, parts...).

    Used to make stochastic scoring reproducible across processes and independent of
    scheduling/order (e.g., with joblib).
    """
    try:
        base_seed_int = int(base_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"base_seed must be an int-convertible value, got {base_seed!r}"
        ) from exc
    payload = "|".join([str(base_seed_int), *(str(p) for p in parts)]).encode(
        "utf-8", errors="surrogatepass"
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def _set_execution_random_seed(cfg: Dict[str, Any], seed: int) -> None:
    """
    Ensure `execution.random_seed` is set on the config dict in-place.

    This helper normalizes the presence of an ``execution`` sub-dictionary on
    ``cfg`` and writes ``random_seed`` into it. The resulting
    ``execution.random_seed`` value is later consumed by the backtest runner's
    seeding logic (see ``_maybe_seed_deterministic_rng`` in
    :mod:`backtest_engine.runner`) to initialize all relevant random number
    generators deterministically.

    Parameters
    ----------
    cfg:
        Mutable configuration dictionary that will be modified in-place.
    seed:
        Integer seed value to store under ``execution.random_seed``.
    """
    exec_cfg = cfg.get("execution")
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
        cfg["execution"] = exec_cfg
    exec_cfg["random_seed"] = int(seed)


class _Progress:
    """Simple single-line progress printer with ETA."""

    def __init__(self, total: int, label: str = "", every: Optional[int] = None):
        self.total = max(1, int(total or 1))
        self.label = label
        self.n = 0
        self.t0 = time.time()
        try:
            env_every = int(os.getenv("FINALSEL_PROGRESS_EVERY", "").strip() or "0")
        except Exception:
            env_every = 0
        if every is None or every <= 0:
            every = max(1, self.total // 100)
        self.every = max(1, env_every or every)

    def _fmt_eta(self, sec: float) -> str:
        if not math.isfinite(sec) or sec < 0:
            return "--:--:--"
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(self, k: int = 1) -> None:
        self.n = min(self.total, self.n + k)
        if (self.n % self.every != 0) and (self.n < self.total):
            return
        elapsed = max(1e-9, time.time() - self.t0)
        rate = self.n / elapsed
        remaining = (self.total - self.n) / max(rate, 1e-9)
        pct = 100.0 * self.n / self.total
        msg = (
            f"\r⏳ {self.label}: {self.n}/{self.total} "
            f"({pct:5.1f}%)  ETA~{self._fmt_eta(remaining)}"
        )
        print(msg, end="", flush=True)
        if self.n >= self.total:
            print("", flush=True)

    def done(self) -> None:
        if self.n < self.total:
            self.n = self.total
            self.update(0)


def run_final_parameter_selection(
    *,
    walkforward_root: str,
    base_config: Dict[str, Any],
    config_template_path: str,
    param_grid: Dict[str, Any],
    search_mode: str = "grid",  # "smart" | "grid"
    preload_mode: str = "window",
    jitter_frac: float = 0.05,
    jitter_repeats: int = 10,
    segment_positive_ratio: float = 0.75,
    max_grid_candidates: Optional[int] = None,
    n_jobs: Optional[int] = None,
    # --- SMART search knobs (ported from _2) ---
    smart_n_trials: int = 400,
    smart_topk_reval: int = 100,
    smart_seed: int = 123,
    smart_explore_prob: float = 0.50,
    smart_trust_every: int = 10,
    recorder: Optional[StageRecorder] = None,
    # --- Data jitter stress-test knobs ---
    data_jitter_repeats: int = 10,
    data_jitter_atr_length: int = 14,
    data_jitter_sigma: float = 0.10,
    data_jitter_fraq: float = 0.15,
) -> Tuple[Path, StageRecorder]:
    """Execute the hedge-fund grade final selection pipeline."""

    root = Path(walkforward_root)
    out_dir = root / "final_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_grid_candidates is None:
        try:
            max_grid_candidates = int(os.getenv("MAX_FINAL_GRID", "5000"))
        except Exception:
            max_grid_candidates = 5000
    try:
        max_grid_candidates = int(max_grid_candidates)
    except Exception:
        max_grid_candidates = 5000

    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 4
            n_jobs = max(1, cpu - 1)
        except Exception:
            n_jobs = 3

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "config_template_path": str(Path(config_template_path).resolve()),
        "base_period": {
            "start": base_config.get("start_date"),
            "end": base_config.get("end_date"),
        },
        "params_in": sorted(param_grid.keys()),
        "preload_mode": preload_mode,
        "jitter_frac": float(jitter_frac),
        "jitter_repeats": int(jitter_repeats),
        "segment_positive_ratio": float(segment_positive_ratio),
        "max_grid_candidates": int(max_grid_candidates),
        "n_jobs": int(n_jobs),
    }

    # Optional: Beschränkung der Jitter-Parameter auf die Schnittmenge
    # aus param_grid und dem im frozen_snapshot hinterlegten
    # reporting.param_jitter_include_by_scenario für das gehandelte Szenario.
    jitter_param_filter = _infer_jitter_param_filter_from_snapshot(
        walkforward_root=str(root),
        base_config=base_config,
        param_grid=param_grid,
    )
    if jitter_param_filter is not None:
        try:
            manifest["jitter_param_filter"] = sorted(jitter_param_filter)
        except Exception:
            manifest["jitter_param_filter"] = list(jitter_param_filter)

    if recorder is None:
        rec = StageRecorder(
            scope="final_parameter_selection",
            metadata={
                "walkforward_root": str(root.resolve()),
                "final_selection_dir": str(out_dir.resolve()),
                "config_template_path": str(Path(config_template_path).resolve()),
            },
        )
    else:
        rec = recorder
        rec.add_metadata(
            walkforward_root=str(root.resolve()),
            final_selection_dir=str(out_dir.resolve()),
            config_template_path=str(Path(config_template_path).resolve()),
        )

    rec.add_metadata(
        search_mode=str(search_mode),
        preload_mode=str(preload_mode),
        jitter_frac=float(jitter_frac),
        jitter_repeats=int(jitter_repeats),
        segment_positive_ratio=float(segment_positive_ratio),
        max_grid_candidates=int(max_grid_candidates),
        n_jobs=int(n_jobs),
    )

    zones_path = root / "analysis" / "05_zones_best_grid.json"
    if not zones_path.exists():
        raise FileNotFoundError(f"Expected robust zones grid at: {zones_path}")
    zones_blob = zones_path.read_bytes()
    manifest["zones_grid_sha256"] = hashlib.sha256(zones_blob).hexdigest()
    zones_grid = json.loads(zones_blob.decode("utf-8"))

    instrumentation_path = out_dir / "instrumentation.json"

    def _as_bool(val: Any, default: bool = False) -> bool:
        try:
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            return default

    # "Dev mode" for deterministic score computation (global root seed).
    # This is primarily used to make stress-tests reproducible under joblib and
    # stochastic execution costs (e.g. random slippage).
    rep_cfg = base_config.get("reporting", {}) or {}
    deterministic_scores = _as_bool(rep_cfg.get("dev_mode", False), default=False)
    try:
        deterministic_seed = int(rep_cfg.get("dev_seed", smart_seed))
    except Exception:
        deterministic_seed = int(smart_seed)
    manifest["dev_mode"] = bool(deterministic_scores)
    if deterministic_scores:
        manifest["dev_seed"] = int(deterministic_seed)
    try:
        rec.add_metadata(
            dev_mode=bool(deterministic_scores),
            dev_seed=int(deterministic_seed) if deterministic_scores else None,
        )
    except Exception:
        pass

    # Optional: Equity-Kurven pro Kandidat im Basis-Run persistieren.
    save_equity_curves = _as_bool(os.getenv("FINALSEL_SAVE_EQUITY"), True)
    try:
        cfg_flag = (base_config.get("final_selection", {}) or {}).get(
            "save_equity_curves", None
        )
        if cfg_flag is not None:
            save_equity_curves = _as_bool(cfg_flag, save_equity_curves)
    except Exception:
        pass
    equity_out_dir = out_dir / "equity_curves"

    # Optional: Trades pro Kandidat im Basis-Run persistieren.
    save_trades = _as_bool(os.getenv("FINALSEL_SAVE_TRADES"), True)
    try:
        trades_flag = (base_config.get("final_selection", {}) or {}).get(
            "save_trades", None
        )
        if trades_flag is not None:
            save_trades = _as_bool(trades_flag, save_trades)
    except Exception:
        pass
    trades_out_dir = out_dir / "trades"

    def _finalize_report(*, reason: Optional[str] = None) -> Tuple[Path, StageRecorder]:
        # Step 5 is now the final step. No holdout/clustering stages.
        with rec.stage("step5_reporting") as stage:
            stage.add_details(reason=reason or "Final scores computed in Step 5")
            report = _render_report(
                out_dir,
                manifest,
                reason=reason or "Final scores computed in Step 5",
                shortlist=None,
            )
            stage.add_details(
                artifact_report=str(report),
                shortlist_rows=0,
            )
            try:
                instrumentation_payload = rec.to_dict()
                instrumentation_path.write_text(
                    _json_dumps(instrumentation_payload, indent=2), encoding="utf-8"
                )
                stage.add_details(
                    instrumentation_path=str(instrumentation_path.resolve()),
                    instrumentation_stage_summary=_format_stage_summary(
                        instrumentation_payload
                    ),
                )
                rec.add_metadata(
                    instrumentation_path=str(instrumentation_path.resolve())
                )
            except Exception as exc:
                stage.mark_error(f"instrumentation_export_failed: {exc}")
        return report, rec

    print("📦 [Step 1/5] Kandidaten-Erzeugung …")
    if (search_mode or "").lower().strip() == "smart":
        with rec.stage("step1_smart_search") as stage:
            stage.add_details(
                zones_path=str(zones_path.resolve()),
                smart_n_trials=int(smart_n_trials),
                smart_topk_reval=int(smart_topk_reval),
                smart_seed=int(smart_seed),
                smart_explore_prob=float(smart_explore_prob),
                smart_trust_every=int(smart_trust_every),
                n_jobs=int(n_jobs),
            )
            trials_df, best_full_df = _smart_parameter_search(
                zones_grid=zones_grid,
                param_grid=param_grid,
                base_config=base_config,
                out_dir=out_dir,
                n_jobs=n_jobs,
                preload_mode=preload_mode,
                n_trials=int(smart_n_trials),
                topk_reeval=int(smart_topk_reval),
                seed=int(smart_seed),
                explore_prob=float(smart_explore_prob),
                trust_every=int(smart_trust_every),
            )
            # Artefakte: Trials und Vollperioden-Ergebnisse (alle Survivors)
            _to_csv(trials_df, out_dir / "00_smart_trials.csv")
            _to_csv(best_full_df, out_dir / "00_smart_best.csv")
            df_candidates = _ensure_columns_order(best_full_df.copy())
            _to_csv(df_candidates, out_dir / "01_candidates_raw.csv")
            stage.add_details(
                smart_trials=int(len(trials_df)),
                smart_best_rows=int(len(best_full_df)),
                artifact_smart_trials=str((out_dir / "00_smart_trials.csv")),
                artifact_smart_best=str((out_dir / "00_smart_best.csv")),
                artifact_candidates=str((out_dir / "01_candidates_raw.csv")),
            )
    else:
        with rec.stage("step1_grid_build") as stage:
            stage.add_details(
                zones_path=str(zones_path.resolve()),
                max_grid_candidates=int(max_grid_candidates),
                n_jobs=int(n_jobs),
            )
            candidates = _build_grid_from_zones(
                zones_grid, param_grid, max_grid_candidates
            )
            manifest["grid_candidates"] = len(candidates)
            stage.add_details(candidates_total=len(candidates))
            (
                df_round1,
                round1_selected_ids,
                cache,
                semiannual_segments,
            ) = _sh_phase1(
                candidates,
                base_config,
                preload_mode,
                out_dir,
                n_jobs=n_jobs,
            )
            round1_path = out_dir / "01_sh_round1.csv"
            _to_csv(df_round1, round1_path)
            manifest["sh_round1_total"] = int(len(df_round1))
            manifest["sh_round1_selected"] = int(len(round1_selected_ids))
            stage.add_details(
                sh_round1_total=int(len(df_round1)),
                sh_round1_selected=int(len(round1_selected_ids)),
                artifact_sh_round1=str(round1_path),
            )

            if not round1_selected_ids:
                df_candidates = pd.DataFrame(columns=df_round1.columns)
                _to_csv(
                    pd.DataFrame(columns=df_round1.columns),
                    out_dir / "01_sh_round2.csv",
                )
                _to_csv(
                    pd.DataFrame(columns=df_round1.columns), out_dir / "01_sh_final.csv"
                )
                _to_csv(
                    pd.DataFrame(columns=df_round1.columns),
                    out_dir / "01_candidates_raw.csv",
                )
                stage.add_details(reason="no_candidates_after_round1")
            else:
                df_round2, round2_selected_ids = _sh_phase2(
                    round1_selected_ids,
                    candidates,
                    base_config,
                    preload_mode,
                    out_dir,
                    cache,
                    semiannual_segments,
                    n_jobs=n_jobs,
                )
                round2_path = out_dir / "01_sh_round2.csv"
                _to_csv(df_round2, round2_path)
                manifest["sh_round2_total"] = int(len(df_round2))
                manifest["sh_round2_selected"] = int(len(round2_selected_ids))
                stage.add_details(
                    sh_round2_total=int(len(df_round2)),
                    sh_round2_selected=int(len(round2_selected_ids)),
                    artifact_sh_round2=str(round2_path),
                )

                if not round2_selected_ids:
                    df_candidates = pd.DataFrame(columns=df_round2.columns)
                    _to_csv(
                        pd.DataFrame(columns=df_round2.columns),
                        out_dir / "01_sh_final.csv",
                    )
                    _to_csv(
                        pd.DataFrame(columns=df_round2.columns),
                        out_dir / "01_candidates_raw.csv",
                    )
                    stage.add_details(reason="no_candidates_after_round2")
                else:
                    df_candidates = _sh_phase3(
                        round2_selected_ids,
                        candidates,
                        base_config,
                        preload_mode,
                        n_jobs=n_jobs,
                    )
                    df_candidates = _ensure_columns_order(df_candidates)
                    candidates_path = out_dir / "01_sh_final.csv"
                    _to_csv(df_candidates, candidates_path)
                    _to_csv(df_candidates, out_dir / "01_candidates_raw.csv")
                    manifest["candidates_after_sh"] = int(len(df_candidates))
                    stage.add_details(
                        evaluated_rows=int(len(df_candidates)),
                        artifact_candidates=str(candidates_path),
                    )

    if df_candidates.empty:
        _materialize_empties(out_dir, start_step=2)
        return _finalize_report(reason="No candidates produced")

    print("🚧 [Step 2/5] Hard Gates anwenden …")
    with rec.stage("step2_hard_gates") as stage:
        df_after_gates = _apply_hard_gates(df_candidates)
        gates_path = out_dir / "02_candidates_after_gates.csv"
        _to_csv(df_after_gates, gates_path)
        manifest["candidates_after_gates"] = len(df_after_gates)
        stage.add_details(
            rows_in=int(len(df_candidates)),
            rows_out=int(len(df_after_gates)),
            artifact_candidates_after_gates=str(gates_path),
        )
        if df_after_gates.empty:
            stage.add_details(reason="no_candidates_after_gates")

    if df_after_gates.empty:
        _materialize_empties(out_dir, start_step=3)
        return _finalize_report(reason="No candidates after hard gates")

    print("🗓️  [Step 3/5] Halbjahres-Segment-Backtests …")
    with rec.stage("step3_segments") as stage:
        segments = _semiannual_segments(
            base_config.get("start_date"), base_config.get("end_date")
        )
        manifest["segments"] = segments
        stage.add_details(segment_count=len(segments))
        if segments:
            seg_rows, _ = _evaluate_segments(
                df_after_gates,
                base_config,
                preload_mode,
                segments,
                out_dir,
                n_jobs=n_jobs,
            )
            df_segments = pd.DataFrame(seg_rows)
        else:
            df_segments = pd.DataFrame(
                columns=[
                    "combo_id",
                    "segment_id",
                    "segment_start",
                    "segment_end",
                    "Net Profit",
                ]
            )
        segments_path = out_dir / "03_segments_results.csv"
        _to_csv(df_segments, segments_path)
        stage.add_details(
            artifact_segments=str(segments_path),
            segment_rows=int(len(df_segments)),
            n_jobs=int(n_jobs),
        )

    print("🧹 [Step 4/5] Segment-Stabilität prüfen …")
    with rec.stage("step4_segment_filter") as stage:
        kept_ids = _segment_filter_pass(
            df_segments, threshold=float(segment_positive_ratio)
        )
        if df_segments.empty:
            df_segment_pass = df_after_gates.copy()
            kept_ids = set(df_segment_pass["combo_id"].tolist())
        else:
            df_segment_pass = df_after_gates[
                df_after_gates["combo_id"].isin(kept_ids)
            ].copy()
        segment_pass_path = out_dir / "04_candidates_segment_pass.csv"
        _to_csv(df_segment_pass, segment_pass_path)
        manifest["candidates_after_segments"] = len(df_segment_pass)
        stage.add_details(
            candidates_in=int(len(df_after_gates)),
            candidates_kept=int(len(df_segment_pass)),
            kept_combo_ids=len(kept_ids),
            artifact_segment_pass=str(segment_pass_path),
        )
        if df_segment_pass.empty:
            stage.add_details(reason="no_candidates_after_segments")

    if df_segment_pass.empty:
        _materialize_empties(out_dir, start_step=5)
        return _finalize_report(reason="No candidates passed segment stability")

    print("🧪 [Step 5/5] Robustness-Stresstests & Final Scoring …")
    with rec.stage("step5_stresstests") as stage:

        df_scores = _score_candidates(
            df_segment_pass,
            base_config,
            preload_mode,
            jitter_frac=jitter_frac,
            jitter_repeats=jitter_repeats,
            dropout_frac=0.10,
            n_jobs=n_jobs,
            deterministic=deterministic_scores,
            deterministic_seed=deterministic_seed,
            equity_out_dir=equity_out_dir,
            save_equity_curves=save_equity_curves,
            trades_out_dir=trades_out_dir,
            save_trades=save_trades,
            jitter_param_filter=jitter_param_filter,
            data_jitter_repeats=data_jitter_repeats,
            data_jitter_atr_length=data_jitter_atr_length,
            data_jitter_sigma=data_jitter_sigma,
            data_jitter_fraq=data_jitter_fraq,
        )
        scores_path = out_dir / "05_final_scores.csv"
        year_segments = _yearly_segments(
            base_config.get("start_date"), base_config.get("end_date")
        )
        # Reorder yearly segments to descending (latest -> oldest)
        try:
            year_segments = sorted(
                list(year_segments), key=lambda t: int(str(t[0])), reverse=True
            )
        except Exception:
            year_segments = list(reversed(list(year_segments)))
        detailed_scores_path = out_dir / "05_final_scores_detailed.csv"
        df_scores_detailed = _yearly_breakdown(
            df_scores,
            base_config,
            preload_mode,
            year_segments,
            n_jobs=n_jobs,
        )
        # Remove additional score columns ONLY from the detailed CSV (keep elsewhere).
        # Keep original column order otherwise.
        if isinstance(df_scores_detailed.columns, pd.MultiIndex):
            drop_cols = []
            for k in [
                ("parameters", "robustness_score_1"),
                ("parameters", "data_jitter_score"),
                ("parameters", "cost_shock_score"),
                ("parameters", "timing_jitter_score"),
                ("parameters", "trade_dropout_score"),
                ("parameters", "ulcer_index"),
                ("parameters", "ulcer_index_score"),
            ]:
                if k in df_scores_detailed.columns:
                    drop_cols.append(k)
            if drop_cols:
                df_scores_detailed = df_scores_detailed.drop(columns=drop_cols)
            # Write with a single header row equal to the 2nd level of the MultiIndex
            flat_header = [str(col[1]) for col in df_scores_detailed.columns]
            _to_csv(df_scores_detailed, detailed_scores_path, header=flat_header)
        else:
            # Non-MultiIndex fallback: just drop columns if present
            for c in [
                "robustness_score_1",
                "data_jitter_score",
                "cost_shock_score",
                "timing_jitter_score",
                "trade_dropout_score",
                "ulcer_index",
                "ulcer_index_score",
            ]:
                if c in df_scores_detailed.columns:
                    df_scores_detailed = df_scores_detailed.drop(columns=[c])
            _to_csv(df_scores_detailed, detailed_scores_path)

        # Compute stability scores per combo based on yearly breakdown
        try:
            df_stability = __compute_yearly_stability(
                df_scores, df_scores_detailed, year_segments=year_segments
            )
        except Exception:
            df_stability = pd.DataFrame(
                {
                    "combo_id": df_scores.get("combo_id", pd.Series(dtype=str)).astype(
                        str
                    ),
                    "stability_score": 1.0,
                }
            )
        if not df_stability.empty:
            # Join without overwriting existing columns
            add_cols = [c for c in df_stability.columns if c not in ("combo_id",)]
            df_scores = df_scores.merge(
                df_stability[["combo_id"] + add_cols], on="combo_id", how="left"
            )

        # New: Replace overall score logic per specification.
        # - Use metrics: profit_over_dd, Avg R-Multiple, Winrate, Sortino(trade),
        #   robustness_score_1, cost_shock_score, timing_jitter_score, trade_dropout_score,
        #   tp_sl_stress_score, p_mean_r_gt_0, stability_score
        # - Normalizations:
        #   Winrate -> winrate / 100
        #   Avg R, Profit/DD, Sortino -> x / (1 + x)
        #   p_mean_r_gt_0 -> 1 / (1 + x)
        #   robustness_score_1 / stress scores / stability_score kept as-is
        def _to_num(s: pd.Series) -> pd.Series:
            return pd.to_numeric(s, errors="coerce")

        def _safe_x_over_1px(x: pd.Series) -> pd.Series:
            # x / (1 + x), robust to NaN/Inf and x == -1 → 0.0 fallback
            x = _to_num(x)
            denom = 1.0 + x
            out = x / denom
            out = out.mask(~np.isfinite(out), 0.0)
            out = out.fillna(0.0)
            return out

        # Build normalized components with robust fallbacks
        comp: List[pd.Series] = []
        if "profit_over_dd" in df_scores.columns:
            comp.append(_safe_x_over_1px(df_scores["profit_over_dd"]))
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        if "Avg R-Multiple" in df_scores.columns:
            comp.append(_safe_x_over_1px(df_scores["Avg R-Multiple"]))
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        if "Winrate (%)" in df_scores.columns:
            wr = _to_num(df_scores["Winrate (%)"]).fillna(0.0) / 100.0
            wr = wr.mask(~np.isfinite(wr), 0.0).fillna(0.0)
            comp.append(wr)
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        if "Sortino (trade)" in df_scores.columns:
            comp.append(_safe_x_over_1px(df_scores["Sortino (trade)"]))
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        # Robustness/stress scores are already normalized (take as-is)
        for col in (
            "robustness_score_1",
            "data_jitter_score",
            "cost_shock_score",
            "timing_jitter_score",
            "trade_dropout_score",
        ):
            if col in df_scores.columns:
                s = (
                    _to_num(df_scores[col])
                    .mask(~np.isfinite(_to_num(df_scores[col])), 0.0)
                    .fillna(0.0)
                )
                comp.append(s)
            else:
                comp.append(pd.Series(0.0, index=df_scores.index))
        # TP/SL-Stress-Score ist bereits 0..1 skaliert (take as-is)
        if "tp_sl_stress_score" in df_scores.columns:
            ts = (
                _to_num(df_scores["tp_sl_stress_score"])
                .mask(~np.isfinite(_to_num(df_scores["tp_sl_stress_score"])), 0.0)
                .fillna(0.0)
            )
            comp.append(ts)
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        # p-mean (mean R multiple > 0) p-value → 1 / (1 + x)
        if "p_mean_r_gt_0" in df_scores.columns:
            p = _to_num(df_scores["p_mean_r_gt_0"]).fillna(1.0)
            pm = 1.0 / (1.0 + p)
            pm = pm.mask(~np.isfinite(pm), 0.0).fillna(0.0)
            comp.append(pm)
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))
        # stability_score kept as-is (post-merge)
        if "stability_score" in df_scores.columns:
            st = (
                _to_num(df_scores["stability_score"])
                .mask(~np.isfinite(_to_num(df_scores["stability_score"])), 0.0)
                .fillna(0.0)
            )
            comp.append(st)
        else:
            comp.append(pd.Series(0.0, index=df_scores.index))

        # Aggregate: arithmetic mean over all components
        if comp:
            stacked = pd.concat(comp, axis=1)
            df_scores["score"] = stacked.mean(axis=1).astype(float)
        else:
            df_scores["score"] = 0.0

        # Maintain determinism: sort by new score descending
        df_scores = df_scores.sort_values("score", ascending=False).reset_index(
            drop=True
        )

        # Column order adjustments for 05_final_scores.csv:
        # - Drop wmape
        # - Ensure Sortino(trade) sits right after Sharpe(trade)
        # - Place data_jitter_score right after robustness_score_1
        # - Place ulcer columns in the score block after stability_score
        # - Place tp_sl_stress_score and stability_score right after trade_dropout_score
        def _reorder_final_cols(df_in: pd.DataFrame) -> pd.DataFrame:
            cols = list(df_in.columns)
            # Drop wmape if present
            if "wmape" in cols:
                cols.remove("wmape")
            # Ensure Sortino after Sharpe
            if "Sharpe (trade)" in cols and "Sortino (trade)" in cols:
                cols.remove("Sortino (trade)")
                sharpe_idx = cols.index("Sharpe (trade)")
                cols.insert(sharpe_idx + 1, "Sortino (trade)")
            # Ensure data_jitter_score right after robustness_score_1
            if "data_jitter_score" in cols and "robustness_score_1" in cols:
                cols.remove("data_jitter_score")
                r1_idx = cols.index("robustness_score_1")
                cols.insert(r1_idx + 1, "data_jitter_score")
            # Ensure tp_sl_stress_score and stability_score after trade_dropout_score
            if "tp_sl_stress_score" in cols:
                cols.remove("tp_sl_stress_score")
            if "stability_score" in cols:
                cols.remove("stability_score")
            anchor: Optional[str] = None
            for candidate in (
                "trade_dropout_score",
                "timing_jitter_score",
                "cost_shock_score",
                "data_jitter_score",
            ):
                if candidate in cols:
                    anchor = candidate
                    break
            if anchor is not None:
                idx = cols.index(anchor)
                if "tp_sl_stress_score" in df_in.columns:
                    cols.insert(idx + 1, "tp_sl_stress_score")
                    idx += 1
                if "stability_score" in df_in.columns:
                    cols.insert(idx + 1, "stability_score")
            else:
                if "tp_sl_stress_score" in df_in.columns:
                    cols.append("tp_sl_stress_score")
                if "stability_score" in df_in.columns:
                    cols.append("stability_score")
            # Place ulcer columns after stability_score (or at the end of score block)
            ulcer_cols = ["ulcer_index", "ulcer_index_score"]
            for uc in ulcer_cols:
                if uc in cols:
                    cols.remove(uc)
            # Find stability_score or last score anchor to insert ulcer columns after
            ulcer_anchor = None
            for cand in (
                "stability_score",
                "tp_sl_stress_score",
                "trade_dropout_score",
            ):
                if cand in cols:
                    ulcer_anchor = cand
                    break
            if ulcer_anchor is not None:
                ulcer_idx = cols.index(ulcer_anchor)
                for i, uc in enumerate(ulcer_cols):
                    if uc in df_in.columns:
                        cols.insert(ulcer_idx + 1 + i, uc)
            else:
                for uc in ulcer_cols:
                    if uc in df_in.columns:
                        cols.append(uc)
            return df_in.loc[:, cols]

        df_scores = _reorder_final_cols(df_scores)

        # Export final scores (after new scoring + ordering). wmape removed.
        combined_path = out_dir / "05_final_scores_combined.csv"
        df_scores_combined = _merge_scores_with_yearly(
            df_scores, df_scores_detailed, year_segments
        )
        _to_csv(df_scores, scores_path)
        _to_csv(df_scores_combined, combined_path)
        manifest["scored_candidates"] = len(df_scores)
        stage.add_details(
            candidates_in=int(len(df_segment_pass)),
            scored_candidates=int(len(df_scores)),
            artifact_scores=str(scores_path),
            artifact_scores_detailed=str(detailed_scores_path),
            artifact_scores_combined=str(combined_path),
            jitter_frac=float(jitter_frac),
            jitter_repeats=int(jitter_repeats),
            n_jobs=int(n_jobs),
            yearly_segments=[
                {
                    "label": label,
                    "start": _format_segment_boundary(
                        seg_start, base_config.get("start_date")
                    ),
                    "end": _format_segment_boundary(
                        seg_end, base_config.get("end_date")
                    ),
                }
                for label, seg_start, seg_end in year_segments
            ],
        )
        if df_scores.empty:
            stage.add_details(reason="no_scores")

    # Step 5 is final. Report and exit.
    if df_scores.empty:
        return _finalize_report(reason="Scoring produced no results")
    print("✅ Final scores computed in Step 5.")
    report, recorder_obj = _finalize_report()

    try:
        clear_alignment_cache(keep_last=0)
    except Exception:
        pass
    try:
        reset_candle_build_caches()
    except Exception:
        pass

    return report, recorder_obj


_WORKER_PRELOADED: Dict[Tuple[str, str], pd.DataFrame] = {}
_WORKER_SYMBOL: Optional[str] = None
_WORKER_TFS: Optional[Dict[str, Any]] = None
_WORKER_PRELOAD_MODE: str = "full"
_WORKER_PATHS: Dict[Tuple[str, str], Path] = {}
# Cache for primary‑TF candle arrays per (symbol, timeframe) in each worker.
_WORKER_CANDLE_CACHE: Dict[Tuple[str, str], PrimaryCandleArrays] = {}


def _limit_blas_threads_once() -> None:
    for env in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env, "1")


def _ensure_preloaded_in_worker(base_config: Dict[str, Any], mode: str) -> None:
    global _WORKER_PRELOADED, _WORKER_SYMBOL, _WORKER_TFS, _WORKER_PRELOAD_MODE, _WORKER_PATHS, _WORKER_CANDLE_CACHE
    _limit_blas_threads_once()
    if _WORKER_PRELOADED and _WORKER_SYMBOL == base_config.get("symbol"):
        return
    _WORKER_PRELOADED = {}
    _WORKER_SYMBOL = base_config.get("symbol")
    _WORKER_TFS = base_config.get("timeframes", {"primary": "M15", "additional": []})
    _WORKER_PRELOAD_MODE = mode or "full"
    _WORKER_PATHS = {}
    _WORKER_CANDLE_CACHE = {}

    symbol = _WORKER_SYMBOL
    tfs = _WORKER_TFS or {}
    all_tfs = [tfs.get("primary")] + list(tfs.get("additional", []))
    base_dir = PARQUET_DIR / str(symbol)

    def _find_parquet_file(base: Path, sym: str, tf: str, side: str) -> Optional[Path]:
        """Find parquet file, preferring uppercase BID/ASK, falling back to lowercase."""
        # Priority 1: Uppercase (BID/ASK)
        upper_path = base / f"{sym}_{tf}_{side.upper()}.parquet"
        if upper_path.exists():
            return upper_path
        # Priority 2: Lowercase (bid/ask)
        lower_path = base / f"{sym}_{tf}_{side.lower()}.parquet"
        if lower_path.exists():
            return lower_path
        return None

    for tf in filter(None, all_tfs):
        for side in ("bid", "ask"):
            path = _find_parquet_file(base_dir, symbol, tf, side)
            if path is None:
                continue
            if _WORKER_PRELOAD_MODE == "full":
                try:
                    _WORKER_PRELOADED[(tf, side)] = pd.read_parquet(path)
                except Exception:
                    pass
            else:
                _WORKER_PATHS[(tf, side)] = path


def _load_or_get_preloaded_data(
    base_cfg: Dict[str, Any],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Gibt preloaded data zurück, unabhängig vom preload_mode.

    Bei 'full' mode: nutzt _WORKER_PRELOADED direkt
    Bei 'window' mode: lädt die kompletten Daten aus _WORKER_PATHS

    Diese Funktion wird für Data-Jitter-Tests benötigt, die die kompletten
    Daten benötigen (nicht nur das Window des aktuellen Backtests).
    """
    if _WORKER_PRELOADED:
        return _WORKER_PRELOADED

    if not _WORKER_PATHS:
        return {}

    # Window mode: lade die Daten komplett (für Data Jitter Tests)
    result: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, path in _WORKER_PATHS.items():
        try:
            df = pd.read_parquet(path)
            result[key] = df
        except Exception:
            continue
    return result


def _with_worker_preload(cfg: Dict[str, Any]):
    if _WORKER_PRELOAD_MODE != "window" or not _WORKER_PATHS:
        return run_backtest_and_return_portfolio(cfg, preloaded_data=_WORKER_PRELOADED)

    start = pd.to_datetime(cfg.get("start_date"))
    end = pd.to_datetime(cfg.get("end_date"))
    pre: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, path in _WORKER_PATHS.items():
        try:
            df = pd.read_parquet(
                path,
                engine="pyarrow",
                filters=[
                    ("UTC time", ">=", start.to_pydatetime()),
                    ("UTC time", "<=", end.to_pydatetime()),
                ],
            )
        except Exception:
            df = pd.read_parquet(path)
            if "time" in df.columns:
                t = pd.to_datetime(df["time"], errors="coerce")
                df = df.loc[(t >= start) & (t <= end)]
        pre[key] = df
    try:
        return run_backtest_and_return_portfolio(cfg, preloaded_data=pre)
    finally:
        pre.clear()
        gc.collect()
        try:
            trim_candle_build_caches()
        except Exception:
            pass


def _get_primary_candle_arrays(
    base_config: Dict[str, Any],
) -> Optional[PrimaryCandleArrays]:
    """
    Build (and cache per worker) fast lookup arrays for primary‑TF bid/ask candles.
    Uses the same preloaded Parquet sources as the backtest engine (_WORKER_PRELOADED / _WORKER_PATHS).
    """
    symbol = str(base_config.get("symbol") or "").strip()
    tfs = base_config.get("timeframes") or {}
    primary_tf = str(tfs.get("primary") or "").strip()
    try:
        print(f"[tp_sl-stress] debug: symbol={symbol!r} primary_tf={primary_tf!r}")
    except Exception:
        pass
    if not symbol or not primary_tf:
        try:
            print("[tp_sl-stress] debug: missing symbol or primary_tf -> arrays=None")
        except Exception:
            pass
        return None

    key = (symbol, primary_tf)
    cached = _WORKER_CANDLE_CACHE.get(key)
    if cached is not None:
        return cached

    bid_df: Optional[pd.DataFrame] = _WORKER_PRELOADED.get((primary_tf, "bid"))
    ask_df: Optional[pd.DataFrame] = _WORKER_PRELOADED.get((primary_tf, "ask"))

    if bid_df is None or ask_df is None:
        try:
            print(
                f"[tp_sl-stress] debug: WORKER_PATHS keys={list(_WORKER_PATHS.keys())}"
            )
            if (primary_tf, "bid") not in _WORKER_PATHS or (
                primary_tf,
                "ask",
            ) not in _WORKER_PATHS:
                print("[tp_sl-stress] no worker paths for primary_tf -> None")
        except Exception:
            pass
        bid_path = _WORKER_PATHS.get((primary_tf, "bid"))
        ask_path = _WORKER_PATHS.get((primary_tf, "ask"))
        try:
            if bid_df is None and bid_path is not None and bid_path.exists():
                bid_df = pd.read_parquet(bid_path)
        except Exception:
            bid_df = None
        try:
            if ask_df is None and ask_path is not None and ask_path.exists():
                ask_df = pd.read_parquet(ask_path)
        except Exception:
            ask_df = None

    if bid_df is None or ask_df is None:
        try:
            print(
                f"[tp_sl-stress] debug: no bid/ask DataFrames for symbol={symbol} tf={primary_tf} -> arrays=None"
            )
        except Exception:
            pass
        return None

    arrays = align_primary_candles(bid_df, ask_df)
    if arrays is None:
        try:
            print(
                f"[tp_sl-stress] debug: align_primary_candles returned None for symbol={symbol} tf={primary_tf}"
            )
        except Exception:
            pass
        return None

    _WORKER_CANDLE_CACHE[key] = arrays
    try:
        print(
            f"🔎 [tp_sl_stress] Built primary candle arrays "
            f"symbol={symbol} tf={primary_tf} n={len(arrays.get('times_ns', []))}"
        )
    except Exception:
        pass
    return arrays


def _json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, default=_to_builtin, **kwargs)


def _to_builtin(obj: Any):
    from datetime import datetime as _dt

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, _dt):
        return obj.isoformat()
    return str(obj)


def _cartesian(values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    from itertools import product

    keys = list(values.keys())
    out: List[Dict[str, Any]] = []
    for tup in product(*[values[k] for k in keys]):
        out.append({k: tup[i] for i, k in enumerate(keys)})
    return out


def _cartesian_with_progress(
    values: Dict[str, List[Any]], label: str = "Grid build"
) -> List[Dict[str, Any]]:
    """Cartesian product mit Progress/ETA – identisch zu _cartesian, nur mit Anzeige."""
    from itertools import product

    keys = list(values.keys())
    # Gesamtanzahl möglichst korrekt schätzen
    total = 1
    for k in keys:
        total *= max(1, len(values.get(k, [])))
    prog = _Progress(total, label=label)
    out: List[Dict[str, Any]] = []
    for tup in product(*[values[k] for k in keys]):
        out.append({k: tup[i] for i, k in enumerate(keys)})
        prog.update(1)
    prog.done()
    return out


def _combo_id(params: Dict[str, Any]) -> str:
    kv = sorted(
        [(k, params[k]) for k in params if not k.startswith("_")], key=lambda x: x[0]
    )
    blob = _json_dumps(kv, separators=(",", ":"), ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _build_grid_from_zones(
    zones_grid: Dict[str, Any],
    param_grid: Dict[str, Any],
    max_grid_candidates: Optional[int],
) -> List[Dict[str, Any]]:
    zones = zones_grid.get("zones", {})
    values: Dict[str, List[Any]] = {}
    for name, spec in param_grid.items():
        if spec.get("type") == "categorical":
            if name in zones and zones[name] and "choices" in zones[name][0]:
                values[name] = list(zones[name][0]["choices"])
            else:
                values[name] = list(spec.get("choices", []))
        else:
            lo = float(spec.get("low"))
            hi = float(spec.get("high"))
            step = float(spec.get("step", 1.0))
            candidates: List[float] = []
            for entry in zones.get(name, []):
                lo = float(entry.get("min", entry.get("low", lo)))
                hi = float(entry.get("max", entry.get("high", hi)))
                step = float(entry.get("step", step))
                step = step if step > 0 else 1.0
                span = hi - lo
                # add small epsilon so exact multiples survive floating point drift
                k = int(math.floor((span + 1e-9) / step)) + 1
                seq = [lo + i * step for i in range(max(k, 1))]
                if seq and seq[-1] < hi - 1e-9:
                    seq.append(hi)
                candidates.extend(seq)
            if not candidates:
                span = hi - lo
                # ensure the upper bound is included even if floats wobble
                k = int(math.floor((span + 1e-9) / step)) + 1
                candidates = [lo + i * step for i in range(max(k, 1))]
                if candidates and candidates[-1] < hi - 1e-9:
                    candidates.append(hi)
            vals = sorted(set(float(round(v, 6)) for v in candidates))
            if spec.get("type") == "int":
                vals = [int(round(v)) for v in vals]
            values[name] = vals

    total = 1
    for seq in values.values():
        total *= max(1, len(seq))

    if max_grid_candidates is None or max_grid_candidates <= 0:
        try:
            max_grid_candidates = int(os.getenv("MAX_FINAL_GRID", "5000"))
        except Exception:
            max_grid_candidates = 5000

    try:
        max_grid_candidates = int(max_grid_candidates)
    except Exception:
        max_grid_candidates = 5000

    if total <= max_grid_candidates:
        # volle Kartesische Menge – mit Progress anzeigen
        combos = _cartesian_with_progress(values, label="Grid build")
    else:
        limit = max(1, min(max_grid_candidates, total))
        names = list(values.keys())

        if not names:
            combos = _cartesian(values)
        else:
            lengths = [len(values[name]) for name in names]

            if any(length == 0 for length in lengths):
                combos = []
            else:

                def _counts_better(
                    candidate: Tuple[int, ...],
                    existing: Tuple[int, ...],
                    dims_lengths: Sequence[int],
                ) -> bool:
                    def _max_shortfall(counts: Tuple[int, ...]) -> float:
                        if not counts:
                            return 0.0
                        return max(
                            (length - count) / max(length, 1)
                            for count, length in zip(counts, dims_lengths)
                        )

                    cand_shortfall = _max_shortfall(candidate)
                    exist_shortfall = _max_shortfall(existing)
                    if cand_shortfall < exist_shortfall - 1e-9:
                        return True
                    if cand_shortfall > exist_shortfall + 1e-9:
                        return False
                    return sum(candidate) > sum(existing)

                dp: Dict[int, Tuple[int, ...]] = {1: tuple()}
                for dim_idx, length in enumerate(lengths):
                    max_c = min(length, limit)
                    next_dp: Dict[int, Tuple[int, ...]] = {}
                    prefix_lengths = lengths[: dim_idx + 1]
                    for prod_val, counts_prefix in dp.items():
                        for count in range(1, max_c + 1):
                            new_prod = prod_val * count
                            if new_prod > limit:
                                break
                            candidate = counts_prefix + (count,)
                            existing = next_dp.get(new_prod)
                            if existing is None or _counts_better(
                                candidate, existing, prefix_lengths
                            ):
                                next_dp[new_prod] = candidate
                    dp = next_dp

                best_prod = max(dp.keys()) if dp else 1
                best_counts = dp.get(best_prod, tuple(1 for _ in names))

                def _even_sample(seq: List[Any], target: int) -> List[Any]:
                    n = len(seq)
                    if n == 0:
                        return []
                    if target >= n:
                        return list(seq)
                    if target <= 1:
                        return [seq[0]]
                    span = n - 1
                    indices: List[int] = []
                    for i in range(target):
                        pos = i * span / (target - 1)
                        idx = int(math.floor(pos + 1e-9))
                        min_idx = indices[-1] + 1 if indices else 0
                        max_idx = n - (target - i)
                        if idx < min_idx:
                            idx = min_idx
                        if idx > max_idx:
                            idx = max_idx
                        indices.append(idx)
                    return [seq[idx] for idx in indices]

                sampled: Dict[str, List[Any]] = {}
                for name, count in zip(names, best_counts):
                    seq = values[name]
                    sampled[name] = _even_sample(seq, int(count))

                combos = _cartesian_with_progress(sampled, label="Grid build (sampled)")
                if len(combos) > limit:
                    combos = combos[:limit]

    for combo in combos:
        combo["_combo_id"] = _combo_id(combo)
    return combos


def _evaluate_candidates_full_period(
    candidates: Sequence[Dict[str, Any]],
    base_config: Dict[str, Any],
    preload_mode: str,
    n_jobs: int,
) -> pd.DataFrame:
    def _one(combo: Dict[str, Any], base_cfg, preload_mode_inner):
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        cfg = _inject_params(base_cfg, combo)
        try:
            portfolio, extra = _with_worker_preload(cfg)
            trades_df = _extract_trades(portfolio, extra)
            summary = calculate_metrics(portfolio)
        except Exception:
            summary = {}
            trades_df = None
        row = _metrics_row(combo, summary, trades_df)
        row["combo_id"] = combo["_combo_id"]
        return row

    prog = _Progress(len(candidates), label="Full-period backtests")
    rows: List[Dict[str, Any]] = []
    for chunk in _parallel_iter(
        candidates,
        func=_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
    ):
        rows.extend(chunk)
        prog.update(len(chunk))
    prog.done()
    return pd.DataFrame(rows)


def _sh_phase1(
    candidates: Sequence[Dict[str, Any]],
    base_config: Dict[str, Any],
    preload_mode: str,
    out_dir: Path,
    n_jobs: int,
):
    start_s = base_config.get("start_date")
    end_s = base_config.get("end_date")
    segments = _semiannual_segments(start_s, end_s)
    if not segments:
        df_full = _evaluate_candidates_full_period(
            candidates, base_config, preload_mode, n_jobs
        )
        df_full = _ensure_columns_order(df_full)
        df_full["sh_round1_score"] = 1.0
        df_full["selected_for_round2"] = True
        cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        selected = (
            df_full["combo_id"].astype(str).tolist() if "combo_id" in df_full else []
        )
        return df_full, selected, cache, []

    # === Early-Stop Backstep-Logic over at most 3 semiannual segments ===
    phase_segments = segments[-3:] if len(segments) >= 3 else segments
    combos = list(candidates)

    # cache behält weiterhin (combo_id, seg_start, seg_end) -> row,
    # damit Phase 2 ohne doppelte Läufe auskommt.
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def _sh_eval_with_backsteps_one(
        combo: Dict[str, Any], base_cfg, preload_mode_inner
    ):
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        combo_id = str(combo.get("_combo_id"))
        # Reihenfolge: jüngstes Halbjahr zuerst, dann max. zwei Rücksprünge
        ordered = list(reversed(phase_segments))
        tested_rows: List[Dict[str, Any]] = []
        chosen: Optional[Dict[str, Any]] = None
        backsteps = 0
        for seg in ordered:
            # Stoppen, wenn bereits ein Segment mit >=10 Trades gefunden wurde
            if chosen is not None:
                break
            cfg = _inject_params(base_cfg, combo, start=seg[0], end=seg[1])
            try:
                portfolio, extra = _with_worker_preload(cfg)
                trades_df = _extract_trades(portfolio, extra)
                summary = calculate_metrics(portfolio)
            except Exception:
                summary = {}
                trades_df = None
            row = _metrics_row(combo, summary, trades_df)
            row["combo_id"] = combo_id
            row["segment_id"] = ordered.index(seg) + 1
            row["segment_start"], row["segment_end"] = seg
            tested_rows.append(row)
            cache[(combo_id, seg[0], seg[1])] = dict(row)

            trades = float(row.get("total_trades", 0) or 0)
            if trades < 10:
                backsteps += 1
                if backsteps >= 3:
                    break
                continue

            profit = float(row.get("Net Profit", 0.0) or 0.0)
            if profit <= 0.0:
                # trades >=10 aber Profit <=0 -> kein Durchkommen; nicht weiter rückspringen
                break
            dd = float(row.get("Drawdown", 0.0) or 0.0)
            ratio = (
                (profit / dd)
                if dd > 0
                else (float("inf") if profit > 0 else float("nan"))
            )
            if (not math.isfinite(ratio) and ratio != float("inf")) or ratio < 2.0:
                break
            commission = float(row.get("Commission", 0.0) or 0.0)
            if commission >= 0.5 * profit:
                break
            # Geschafft: dieses Halbjahr bringt den Kandidaten durch
            chosen = dict(row)
            # Early-Stop: keine weiteren Backtests für diese Kombo
            break

        return chosen, tested_rows

    # Parallel über Kandidaten evaluieren (jeder Kandidat macht max. 3 Läufe seriell)
    survivors: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    # Expliziter Fortschritt über Kandidaten (jeder Kandidat hat intern bis zu 3 Backsteps)
    progress_every = max(1, len(combos) // 200)
    round1_prog = _Progress(len(combos), label="SH Round 1", every=progress_every)
    for chunk in _parallel_iter(
        combos,
        func=_sh_eval_with_backsteps_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
        batch_size=8,
    ):
        for chosen, tested in chunk:
            all_rows.extend(tested)
            if chosen is not None:
                survivors.append(chosen)
            round1_prog.update()
    round1_prog.done()

    df_segments = pd.DataFrame(all_rows)  # für Artefakte/Kompatibilität
    # 📄 Neu: Audit-Artefakt mit allen tatsächlich gelaufenen Segment-Backtests
    try:
        seg_path = out_dir / "01_sh_round1_segments.csv"
        _to_csv(df_segments, seg_path)
    except Exception:
        # Artefakt-Schreiben ist "best effort" und soll die Pipeline nicht stoppen
        pass
    if not survivors:
        return pd.DataFrame(columns=df_segments.columns), [], cache, segments

    df_round1 = pd.DataFrame(survivors)
    df_round1["combo_id"] = df_round1["combo_id"].astype(str)
    df_round1["sh_profit_over_dd"] = _safe_ratio_series(
        df_round1["Net Profit"], df_round1["Drawdown"]
    )
    df_round1["sh_profit_over_comm"] = _safe_ratio_series(
        df_round1["Net Profit"], df_round1["Commission"]
    )
    df_round1["sh_round1_score"] = _composite_score(
        df_round1,
        [
            "sh_profit_over_dd",
            "Avg R-Multiple",
            "sh_profit_over_comm",
            "Sortino (trade)",
            "Winrate (%)",
        ],
    )
    df_round1 = df_round1.sort_values("sh_round1_score", ascending=False).reset_index(
        drop=True
    )
    top_k = max(1, math.ceil(len(df_round1) * 0.50))
    selected_ids = df_round1.head(top_k)["combo_id"].tolist()
    df_round1["selected_for_round2"] = df_round1["combo_id"].isin(selected_ids)
    return df_round1, selected_ids, cache, segments


def _sh_phase2(
    selected_ids: Sequence[str],
    candidates: Sequence[Dict[str, Any]],
    base_config: Dict[str, Any],
    preload_mode: str,
    out_dir: Path,
    cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    segments: Sequence[Tuple[str, str]],
    n_jobs: int,
):
    if not selected_ids:
        return pd.DataFrame(), []
    if not segments:
        df = pd.DataFrame({"combo_id": list(selected_ids)})
        df["selected_for_full_period"] = True
        df["sh_round2_score"] = 1.0
        return df, list(selected_ids)

    combo_map = _combo_lookup(candidates)
    combos = [combo_map[cid] for cid in selected_ids if cid in combo_map]
    target_segments = segments[-3:] if len(segments) >= 3 else segments
    df_segments = _sh_evaluate_segments(
        combos,
        target_segments,
        base_config,
        preload_mode,
        n_jobs,
        cache=cache,
        label="SH Round 2",
    )

    if df_segments.empty:
        return pd.DataFrame(columns=["combo_id"]), []

    frames: List[pd.DataFrame] = []
    for idx, seg in enumerate(target_segments, start=1):
        mask = (df_segments["segment_start"] == seg[0]) & (
            df_segments["segment_end"] == seg[1]
        )
        seg_df = df_segments.loc[mask].copy()
        if seg_df.empty:
            continue
        seg_df["sh_profit_over_dd"] = _safe_ratio_series(
            seg_df["Net Profit"], seg_df["Drawdown"]
        )
        seg_df["sh_profit_over_comm"] = _safe_ratio_series(
            seg_df["Net Profit"], seg_df["Commission"]
        )
        seg_df[f"segment_{idx}_score"] = _composite_score(
            seg_df,
            [
                "sh_profit_over_dd",
                "Avg R-Multiple",
                "sh_profit_over_comm",
                "Sortino (trade)",
                "Winrate (%)",
            ],
        )
        frames.append(seg_df[["combo_id", f"segment_{idx}_score"]])

    if not frames:
        return pd.DataFrame(columns=["combo_id"]), []

    scores_df = frames[0].set_index("combo_id")
    for frame in frames[1:]:
        scores_df = scores_df.join(frame.set_index("combo_id"), how="outer")

    scores_df["sh_round2_score"] = scores_df.mean(axis=1, skipna=True)
    scores_df = scores_df.reset_index()

    data_rows: List[Dict[str, Any]] = []
    for _, row in scores_df.iterrows():
        entry: Dict[str, Any] = {"combo_id": row["combo_id"]}
        for col in scores_df.columns:
            if col == "combo_id":
                continue
            entry[col] = row[col]
        combo = combo_map.get(row["combo_id"])
        if combo:
            for k, v in combo.items():
                if not k.startswith("_"):
                    entry[k] = v
        data_rows.append(entry)

    df_round2 = pd.DataFrame(data_rows)
    if df_round2.empty:
        return df_round2, []
    df_round2 = df_round2.sort_values("sh_round2_score", ascending=False).reset_index(
        drop=True
    )
    top_k = max(1, math.ceil(len(df_round2) * 0.50))
    selected_ids_round2 = df_round2.head(top_k)["combo_id"].tolist()
    df_round2["selected_for_full_period"] = df_round2["combo_id"].isin(
        selected_ids_round2
    )
    return df_round2, selected_ids_round2


def _sh_phase3(
    selected_ids: Sequence[str],
    candidates: Sequence[Dict[str, Any]],
    base_config: Dict[str, Any],
    preload_mode: str,
    n_jobs: int,
):
    combo_map = _combo_lookup(candidates)
    combos = [combo_map[cid] for cid in selected_ids if cid in combo_map]
    if not combos:
        return pd.DataFrame()
    df = _evaluate_candidates_full_period(combos, base_config, preload_mode, n_jobs)
    return df


def _parallel_iter(
    items: Sequence[Any],
    *,
    func,
    base_config: Dict[str, Any],
    preload_mode: str,
    n_jobs: int,
    batch_size: int = 8,
) -> Iterable[List[Any]]:
    if not items:
        return []
    # Sequenzieller Fast-Path für Single-Core/Single-Process Ausführung
    backend_env = (
        os.getenv("FINALSEL_PARALLEL_BACKEND", "loky").strip().lower() or "loky"
    )
    if (n_jobs is None or int(n_jobs) <= 1) or backend_env in {
        "sequential",
        "none",
        "off",
    }:
        chunk: List[Any] = []
        for item in items:
            res = func(item, base_config, preload_mode)
            chunk.append(res)
            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        return
    tmp = (
        Path(base_config.get("walkforward_root", "."))
        / "final_selection"
        / "joblib_tmp"
    )
    tmp.mkdir(parents=True, exist_ok=True)

    # Use process-based parallelism by default for real multi-core execution.
    # Can be overridden via FINALSEL_PARALLEL_BACKEND (e.g., "threading" or "loky").
    backend = backend_env

    # Prefer streaming results so progress updates stay live while work continues.
    try:
        gen = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            pre_dispatch="1*n_jobs",
            batch_size=1,
            temp_folder=str(tmp),
            return_as="generator",  # type: ignore[arg-type]
        )(delayed(func)(item, base_config, preload_mode) for item in items)

        chunk: List[Any] = []
        for res in gen:
            chunk.append(res)
            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        return
    except TypeError:
        # Older joblib without generator support; fall back to rolling windows.
        pass

    from itertools import islice

    try:
        window_env = int(os.getenv("FINALSEL_STREAM_WINDOW", "").strip() or "0")
    except Exception:
        window_env = 0
    window = window_env if window_env > 0 else max(n_jobs * 2, batch_size * 8)

    it = iter(items)
    while True:
        batch_items = list(islice(it, window))
        if not batch_items:
            break
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            pre_dispatch="1*n_jobs",
            batch_size=1,
            temp_folder=str(tmp),
        )(delayed(func)(item, base_config, preload_mode) for item in batch_items)

        chunk: List[Any] = []
        for res in results:
            chunk.append(res)
            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _inject_params(
    base_config: Dict[str, Any],
    params: Dict[str, Any],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    if start:
        cfg["start_date"] = start
    if end:
        cfg["end_date"] = end
    if "strategy" not in cfg:
        cfg["strategy"] = {}
    strat = cfg["strategy"]
    if "parameters" not in strat or not isinstance(strat["parameters"], dict):
        strat["parameters"] = {}
    # Ensure all injected params are plain Python types (avoid NumPy scalars)
    for k, v in params.items():
        if k.startswith("_"):
            continue
        try:
            # Convert NumPy scalar types cleanly
            import numpy as _np  # local import to avoid top-level dependency

            if isinstance(v, _np.generic):
                v = v.item()
        except Exception:
            pass
        # Defensive: datetime.timedelta doesn't accept numpy integers for minutes/etc.
        # Coerce common numeric-like values to builtins without changing semantics.
        try:
            # Avoid bools being cast to int inadvertently
            if isinstance(v, bool):
                strat["parameters"][k] = v
            elif isinstance(v, (int, float)):
                strat["parameters"][k] = v
            else:
                strat["parameters"][k] = v
        except Exception:
            strat["parameters"][k] = v
    return cfg


def _extract_trades(portfolio, extra) -> Optional[pd.DataFrame]:
    try:
        if isinstance(extra, dict) and "trades" in extra:
            return pd.DataFrame(extra["trades"])
        if hasattr(portfolio, "trades_to_dataframe"):
            return portfolio.trades_to_dataframe()
    except Exception:
        return None
    return None


def _metrics_row(
    params: Dict[str, Any], summary: Dict[str, Any], trades_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    row = {
        "Net Profit": float(summary.get("net_profit_after_fees_eur", 0.0) or 0.0),
        "Commission": float(summary.get("fees_total_eur", 0.0) or 0.0),
        "Avg R-Multiple": float(summary.get("avg_r_multiple", 0.0) or 0.0),
        "Winrate (%)": float(summary.get("winrate_percent", 0.0) or 0.0),
        "Drawdown": float(summary.get("drawdown_eur", 0.0) or 0.0),
        "Sharpe (trade)": float(summary.get("sharpe_trade", 0.0) or 0.0),
        "Sortino (trade)": float(summary.get("sortino_trade", 0.0) or 0.0),
        "total_trades": int(summary.get("total_trades", 0) or 0),
        "active_days": float(summary.get("active_days", 0.0) or 0.0),
    }
    for k, v in params.items():
        if not k.startswith("_"):
            row[k] = v
    dd = row["Drawdown"] if row["Drawdown"] != 0 else np.nan
    row["profit_over_dd"] = (row["Net Profit"] / dd) if np.isfinite(dd) else np.nan
    row["comm_over_profit"] = (
        (row["Commission"] / row["Net Profit"])
        if row["Net Profit"] not in (0, np.nan)
        else np.nan
    )
    return row


def _to_csv(
    df: pd.DataFrame, path: Path, header: Optional[Sequence[str]] = None
) -> None:
    if df is None or df.empty:
        path.write_text("", encoding="utf-8")
    else:
        # If a custom header is provided (e.g., for flattening MultiIndex to one line),
        # forward it to pandas to_csv; otherwise use default behavior.
        if header is not None:
            df.to_csv(path, index=False, header=list(header))
        else:
            df.to_csv(path, index=False)


def _combo_lookup(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for combo in candidates:
        combo_id = combo.get("_combo_id")
        if combo_id:
            lookup[str(combo_id)] = combo
    return lookup


def _safe_ratio_series(num: pd.Series, denom: pd.Series) -> pd.Series:
    num_f = pd.to_numeric(num, errors="coerce")
    denom_f = pd.to_numeric(denom, errors="coerce")
    ratios: List[float] = []
    for n, d in zip(num_f.fillna(0.0), denom_f.fillna(0.0)):
        if d == 0:
            if n > 0:
                ratios.append(float("inf"))
            elif n < 0:
                ratios.append(float("-inf"))
            else:
                ratios.append(float("nan"))
        else:
            ratios.append(n / d)
    return pd.Series(ratios, index=num.index)


def _normalized_rank(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return pd.Series([], dtype=float)
    pos_inf = np.isposinf(arr)
    neg_inf = np.isneginf(arr)
    finite_mask = ~(pos_inf | neg_inf | np.isnan(arr))
    max_val = arr[finite_mask].max() if finite_mask.any() else 0.0
    min_val = arr[finite_mask].min() if finite_mask.any() else 0.0
    bump = max(1.0, abs(max_val)) + 1.0
    arr[pos_inf] = max_val + bump
    arr[neg_inf] = min_val - bump
    arr[~np.isfinite(arr)] = np.nan
    ranked = pd.Series(arr, index=series.index).rank(method="average", pct=True)
    return ranked.fillna(0.0)


def _composite_score(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    ranks: List[pd.Series] = []
    for col in columns:
        if col not in df.columns:
            continue
        ranks.append(_normalized_rank(df[col]))
    if not ranks:
        return pd.Series(0.0, index=df.index)
    total = sum(ranks)
    return (total / len(ranks)).fillna(0.0)


def _sh_evaluate_segments(
    combos: Sequence[Dict[str, Any]],
    segments: Sequence[Tuple[str, str]],
    base_config: Dict[str, Any],
    preload_mode: str,
    n_jobs: int,
    *,
    cache: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None,
    label: str = "Segments",
) -> pd.DataFrame:
    if not combos or not segments:
        return pd.DataFrame()

    tasks: List[Tuple[Dict[str, Any], int, Tuple[str, str]]] = []
    rows: List[Dict[str, Any]] = []
    for combo in combos:
        combo_id = combo.get("_combo_id")
        if not combo_id:
            continue
        for seg_idx, seg in enumerate(segments, start=1):
            key = (str(combo_id), seg[0], seg[1])
            if cache is not None and key in cache:
                rows.append(dict(cache[key]))
                continue
            tasks.append((combo, seg_idx, seg))

    prog = _Progress(len(tasks), label=label) if tasks else None

    def _one(task, base_cfg, preload_mode_inner):
        combo, seg_idx, seg = task
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        cfg = _inject_params(base_cfg, combo, start=seg[0], end=seg[1])
        try:
            portfolio, extra = _with_worker_preload(cfg)
            trades_df = _extract_trades(portfolio, extra)
            summary = calculate_metrics(portfolio)
        except Exception:
            summary = {}
            trades_df = None
        row = _metrics_row(combo, summary, trades_df)
        row["combo_id"] = str(combo.get("_combo_id"))
        row["segment_id"] = seg_idx
        row["segment_start"], row["segment_end"] = seg
        return row

    if tasks:
        for chunk in _parallel_iter(
            tasks,
            func=_one,
            base_config=base_config,
            preload_mode=preload_mode,
            n_jobs=n_jobs,
        ):
            rows.extend(chunk)
            if cache is not None:
                for row in chunk:
                    key = (
                        str(row["combo_id"]),
                        row["segment_start"],
                        row["segment_end"],
                    )
                    cache[key] = dict(row)
            if prog is not None:
                prog.update(len(chunk))
    if prog is not None:
        prog.done()

    return pd.DataFrame(rows)


def _apply_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    x = df.copy()
    profit = pd.to_numeric(x["Net Profit"], errors="coerce")
    x = x.loc[profit > 0.0]
    dd = pd.to_numeric(x["Drawdown"], errors="coerce").replace({0.0: np.nan})
    ratio = pd.to_numeric(x["Net Profit"], errors="coerce") / dd
    x = x.loc[ratio >= 3.0]
    cp = pd.to_numeric(x["Commission"], errors="coerce") / pd.to_numeric(
        x["Net Profit"], errors="coerce"
    ).replace({0.0: np.nan})
    x = x.loc[cp < 0.50]
    trades = pd.to_numeric(x["total_trades"], errors="coerce").fillna(0)
    x = x.loc[trades >= 10]
    return x


def _semiannual_segments(start_s: str, end_s: str) -> List[Tuple[str, str]]:
    if not start_s or not end_s:
        return []
    start = datetime.strptime(start_s, "%Y-%m-%d")
    end = datetime.strptime(end_s, "%Y-%m-%d")
    if end <= start:
        return []
    segments: List[Tuple[datetime, datetime]] = []
    for year in range(start.year, end.year + 1):
        h1 = (datetime(year, 1, 1), datetime(year, 6, 29))
        h2 = (datetime(year, 7, 2), datetime(year, 12, 31))
        if start <= h1[0] and end >= h1[1]:
            segments.append(h1)
        if start <= h2[0] and end >= h2[1]:
            segments.append(h2)
    segments.sort(key=lambda tpl: tpl[0])
    return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in segments]


def _evaluate_segments(
    df_candidates: pd.DataFrame,
    base_config: Dict[str, Any],
    preload_mode: str,
    segments: Sequence[Tuple[str, str]],
    out_dir: Path,
    n_jobs: int,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    metric_cols = {
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
        "segment_id",
        "segment_start",
        "segment_end",
    }
    param_cols = [
        c for c in df_candidates.columns if c not in metric_cols | {"combo_id"}
    ]
    combos = df_candidates[["combo_id"] + param_cols].drop_duplicates("combo_id")

    tasks: List[Tuple[pd.Series, int, Tuple[str, str]]] = []
    for _, combo_row in combos.iterrows():
        for idx, seg in enumerate(segments, start=1):
            tasks.append((combo_row, idx, seg))

    if not tasks:
        return [], []

    def _one(task, base_cfg, preload_mode_inner):
        combo_row, seg_idx, seg = task
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        params = {k: combo_row[k] for k in combo_row.index if k not in ("combo_id",)}
        cfg = _inject_params(base_cfg, params, start=seg[0], end=seg[1])
        portfolio, extra = _with_worker_preload(cfg)
        trades_df = _extract_trades(portfolio, extra)
        summary = calculate_metrics(portfolio)
        row = _metrics_row(params, summary, trades_df)
        row["combo_id"] = combo_row["combo_id"]
        row["segment_id"] = seg_idx
        row["segment_start"] = seg[0]
        row["segment_end"] = seg[1]
        return row

    rows: List[Dict[str, Any]] = []
    prog = _Progress(len(tasks), label="Segments")
    for chunk in _parallel_iter(
        tasks,
        func=_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
    ):
        rows.extend(chunk)
        prog.update(len(chunk))
    prog.done()
    return rows, []


def _segment_filter_pass(df_segments: pd.DataFrame, *, threshold: float) -> set[str]:
    if df_segments is None or df_segments.empty:
        return set()
    seg = df_segments.copy()
    seg["Net Profit"] = pd.to_numeric(seg["Net Profit"], errors="coerce")
    grouped = seg.groupby("combo_id")["Net Profit"]
    counts = grouped.count()
    positive = grouped.apply(lambda s: (s >= 0.0).sum())
    ratio = positive / counts.replace({0: np.nan})
    ratio = ratio.fillna(0.0)
    return set(ratio[ratio >= threshold].index.astype(str))


def _parameter_columns(df: pd.DataFrame) -> List[str]:
    metric_cols = {
        "combo_id",
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
        # Centralized additional metrics (Step-5)
        "robustness_score_1",
        "data_jitter_score",
        "cost_shock_score",
        "timing_jitter_score",
        "trade_dropout_score",
        "tp_sl_stress_score",
        "p_mean_r_gt_0",
        "p_net_profit_gt_0",
    }
    return [c for c in df.columns if c not in metric_cols]


_YEARLY_METRIC_COLUMNS = [
    "Net Profit",
    "Winrate (%)",
    "Avg R-Multiple",
    "Drawdown",
    "total_trades",
]


def _yearly_segments(
    start: Any, end: Any
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    if getattr(start_ts, "tzinfo", None) is not None:
        start_ts = start_ts.tz_convert(None)
    if getattr(end_ts, "tzinfo", None) is not None:
        end_ts = end_ts.tz_convert(None)
    if start_ts > end_ts:
        return []
    segments: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for year in range(start_ts.year, end_ts.year + 1):
        seg_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        seg_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        if seg_start <= seg_end:
            segments.append((str(year), seg_start, seg_end))
    return segments


def _format_segment_boundary(ts: pd.Timestamp, reference: Any) -> str:
    if ts is pd.NaT:
        return ""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    ref = reference if isinstance(reference, str) else ""
    if "T" in ref or ":" in ref:
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    return ts.strftime("%Y-%m-%d")


def _yearly_breakdown(
    df_scores: pd.DataFrame,
    base_config: Dict[str, Any],
    preload_mode: str,
    year_segments: Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    if df_scores is None or df_scores.empty or not year_segments:
        return pd.DataFrame()

    param_cols = _parameter_columns(df_scores)
    meta_fields = [
        "combo_id",
        "primary",
    ]

    def _one(row, base_cfg, preload_mode_inner):
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        entry: Dict[Tuple[str, str], Any] = {}
        combo_id = getattr(row, "combo_id", "")
        entry[("meta", "combo_id")] = str(combo_id)
        # Add primary timeframe in meta as requested
        try:
            tf_primary = (
                (base_cfg.get("timeframes") or {}).get("primary")
                if isinstance(base_cfg.get("timeframes"), dict)
                else None
            )
        except Exception:
            tf_primary = None
        entry[("meta", "primary")] = str(tf_primary or "")
        params: Dict[str, Any] = {}
        for col in param_cols:
            value = getattr(row, col, np.nan)
            entry[("parameters", col)] = value
            if pd.notna(value):
                params[col] = value
        for idx, (label, seg_start, seg_end) in enumerate(year_segments):
            start_str = _format_segment_boundary(seg_start, base_cfg.get("start_date"))
            end_str = _format_segment_boundary(seg_end, base_cfg.get("end_date"))
            cfg = _inject_params(base_cfg, params, start=start_str, end=end_str)
            portfolio = {}
            extra = None
            try:
                portfolio, extra = _with_worker_preload(cfg)
            except Exception:
                portfolio, extra = {}, None
            try:
                summary = calculate_metrics(portfolio)
            except Exception:
                summary = {}
            trades_df = _extract_trades(portfolio, extra)
            metrics_row = _metrics_row({}, summary, trades_df)
            # Year marker column should appear before metrics of the same year
            entry[(label, "year")] = str(label)
            for metric in _YEARLY_METRIC_COLUMNS:
                entry[(label, metric)] = metrics_row.get(metric, np.nan)
        return entry

    rows: List[Dict[Tuple[str, str], Any]] = []
    for chunk in _parallel_iter(
        list(df_scores.itertuples(index=False)),
        func=_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
    ):
        rows.extend(chunk)

    meta_columns = [("meta", "combo_id"), ("meta", "primary")]
    param_columns = [("parameters", col) for col in param_cols]
    year_columns: List[Tuple[str, str]] = []
    for _, (label, _, _) in enumerate(year_segments):
        # Year marker first, then metrics for that year
        year_columns.append((label, "year"))
        for metric in _YEARLY_METRIC_COLUMNS:
            year_columns.append((label, metric))

    column_order = meta_columns + param_columns + year_columns
    columns = pd.MultiIndex.from_tuples(column_order, names=["scope", "metric"])

    if not rows:
        return pd.DataFrame(columns=columns)

    data: List[List[Any]] = []
    for entry in rows:
        data.append([entry.get(col, np.nan) for col in column_order])

    df = pd.DataFrame(data, columns=columns)
    return df


def _merge_scores_with_yearly(
    df_scores: pd.DataFrame,
    df_scores_detailed: pd.DataFrame,
    year_segments: Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """
    Combine the flattened final scores with yearly breakdown metrics, preserving the
    ordering of df_scores and using combo_id as join key.
    """
    if df_scores is None or df_scores.empty:
        return pd.DataFrame()
    if df_scores_detailed is None or df_scores_detailed.empty:
        return df_scores.copy()

    scores = df_scores.copy()
    if "combo_id" not in scores.columns:
        return scores

    def _fmt_label(label: Any, metric: Any) -> str:
        label_str = str(label).strip()
        metric_str = str(metric).strip()
        if not label_str:
            return metric_str
        if not metric_str:
            return label_str
        return f"{label_str} {metric_str}".strip()

    if isinstance(df_scores_detailed.columns, pd.MultiIndex):
        combo_key = ("meta", "combo_id")
        year_cols = [
            col
            for col in df_scores_detailed.columns
            if len(col) == 2 and col[0] not in ("meta", "parameters")
        ]
        if not year_cols:
            return scores
        df_year = df_scores_detailed.loc[:, year_cols].copy()
        df_year.columns = [_fmt_label(col[0], col[1]) for col in year_cols]
        if combo_key in df_scores_detailed.columns:
            combo_values = df_scores_detailed[combo_key].astype(str).to_numpy()
        else:
            base_vals = scores["combo_id"].astype(str).to_numpy()
            if len(base_vals) == 0:
                base_vals = np.array([""], dtype=str)
            combo_values = np.resize(base_vals, len(df_year))
        df_year.insert(0, "combo_id", combo_values)
    else:
        columns = list(df_scores_detailed.columns)
        if not columns:
            return scores
        first_year_idx: Optional[int] = None
        for idx, name in enumerate(columns):
            if str(name).strip().lower() == "year":
                first_year_idx = idx
                break
        if first_year_idx is None and year_segments:
            year_labels = [
                str(label).strip() for label, *_ in year_segments if str(label).strip()
            ]
            for idx, name in enumerate(columns):
                text = str(name).strip()
                if any(text.startswith(lbl) for lbl in year_labels):
                    first_year_idx = idx
                    break
        if first_year_idx is None:
            return scores
        year_cols = columns[first_year_idx:]
        df_year = df_scores_detailed.loc[:, year_cols].copy()
        expected_cols: List[str] = []
        for label, *_ in year_segments:
            expected_cols.append(_fmt_label(label, "year"))
            for metric in _YEARLY_METRIC_COLUMNS:
                expected_cols.append(_fmt_label(label, metric))
        if len(expected_cols) == len(year_cols):
            df_year.columns = expected_cols
        else:
            df_year.columns = [str(col) for col in year_cols]
        if "combo_id" in df_scores_detailed.columns:
            combo_values = df_scores_detailed["combo_id"].astype(str).to_numpy()
        else:
            base_vals = scores["combo_id"].astype(str).to_numpy()
            if len(base_vals) == 0:
                base_vals = np.array([""], dtype=str)
            combo_values = np.resize(base_vals, len(df_year))
        df_year.insert(0, "combo_id", combo_values)

    df_year = df_year.drop_duplicates(subset=["combo_id"]).reset_index(drop=True)
    scores = scores.copy()
    scores["combo_id"] = scores["combo_id"].astype(str)
    df_year["combo_id"] = df_year["combo_id"].astype(str)
    merged = scores.merge(df_year, on="combo_id", how="left")
    return merged


def __compute_yearly_stability(
    df_scores: pd.DataFrame,
    df_scores_detailed: pd.DataFrame,
    *,
    year_segments: Optional[Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]]] = None,
) -> pd.DataFrame:
    """
    Compute stability_score per combo_id based on yearly breakdown.

    The score is 1 / (1 + wmape), with wmape defined as a duration-weighted
    mean absolute percentage error relative to an expected daily rate.

    - Does not modify df_scores_detailed
    - Returns a DataFrame with columns: combo_id, stability_score, and wmape (debug)
    """
    # Defensive: empty inputs → default stability of 1.0
    if df_scores is None or df_scores.empty:
        return pd.DataFrame(columns=["combo_id", "stability_score", "wmape"])

    # Determine yearly labels and accessors for Net Profit per year
    years: list[str] = []
    profit_cols: dict[str, Any] = {}

    if isinstance(getattr(df_scores_detailed, "columns", None), pd.MultiIndex):
        for col in df_scores_detailed.columns:
            if not isinstance(col, tuple) or len(col) != 2:
                continue
            scope, metric = col
            if scope in ("meta", "parameters"):
                continue
            if str(metric) == "Net Profit":
                years.append(str(scope))
                profit_cols[str(scope)] = col
        years = sorted(set(y for y in years if y.isdigit()), key=int)
        combo_col = (
            ("meta", "combo_id")
            if ("meta", "combo_id") in df_scores_detailed.columns
            else None
        )
    else:
        # Fallback: no MultiIndex -> cannot resolve yearly profits reliably
        # Return default stability=1.0 for all combos
        combos = df_scores.get("combo_id", pd.Series(dtype=str)).astype(str)
        return pd.DataFrame(
            {
                "combo_id": combos,
                "stability_score": np.ones(len(combos), dtype=float),
                "wmape": np.zeros(len(combos), dtype=float),
            }
        )

    # If no yearly profit columns found, assign perfect stability
    if not years or not profit_cols:
        combos = df_scores.get("combo_id", pd.Series(dtype=str)).astype(str)
        return pd.DataFrame(
            {
                "combo_id": combos,
                "stability_score": np.ones(len(combos), dtype=float),
                "wmape": np.zeros(len(combos), dtype=float),
            }
        )

    durations_by_year: dict[int, float] = {}
    try:
        for label, seg_start, seg_end in year_segments or []:
            try:
                y = int(str(label))
            except Exception:
                continue
            durations_by_year[y] = float((seg_end.date() - seg_start.date()).days + 1)
    except Exception:
        durations_by_year = {}

    # Prepare output vectors
    if combo_col is not None and combo_col in df_scores_detailed.columns:
        combo_ids = df_scores_detailed[combo_col].astype(str).tolist()
    else:
        combo_ids = df_scores.get("combo_id", pd.Series(dtype=str)).astype(str).tolist()

    stability_scores: list[float] = []
    wmapes: list[float] = []

    for idx in range(len(df_scores_detailed)):
        # Robust access if df_scores_detailed filtered differs
        row = df_scores_detailed.iloc[idx]
        # Collect yearly profits and compute via centralized helper
        profits_by_year: dict[int, float] = {}
        for y in years:
            col = profit_cols.get(y)
            p_y = float(pd.to_numeric(row.get(col, np.nan), errors="coerce") or 0.0)
            try:
                y_int = int(y)
            except Exception:
                continue
            profits_by_year[y_int] = p_y

        if not profits_by_year:
            stability_scores.append(1.0)
            wmapes.append(0.0)
            continue
        score, wmape = compute_stability_score_and_wmape_from_yearly_profits(
            profits_by_year,
            durations_by_year=(durations_by_year or None),
        )
        stability_scores.append(float(score))
        wmapes.append(float(wmape))

    out = pd.DataFrame(
        {
            "combo_id": pd.Series(combo_ids, dtype=str)[: len(stability_scores)],
            "stability_score": stability_scores,
            "wmape": wmapes,
        }
    )
    # Ensure uniqueness by combo_id (in case of duplicates)
    out = out.drop_duplicates(subset=["combo_id"]).reset_index(drop=True)
    return out


def _compute_tp_sl_stress(
    trades_df: Optional[pd.DataFrame],
    base_config: Dict[str, Any],
) -> float:
    """
    Compute tp_sl_stress_score per combo based on per-trade TP/SL robustness.

    Nur Trades mit reason == 'take_profit' werden betrachtet. Für jeden dieser
    Trades wird ein per-Trade-Score (1 - penalty) berechnet:

      - TP/SL werden um den Spread verschoben
          Long:  TP_var = TP + spread, SL_var = SL + spread
          Short: TP_var = TP - spread, SL_var = SL - spread
      - Ab entry_time wird auf der Primary-TF-Candle-Timeline vorwärts simuliert,
        bis zum ersten Treffer (TP_var oder SL_var).
      - Straflogik:
          * TP immer noch auf derselben Kerze wie Original-Exit:
                penalty = 0.0
          * TP später, und in dieser neuen TP-Kerze wird der SL NICHT getroffen:
                penalty = min(0.1 * delay_bars, 0.5)
          * TP später, und in derselben neuen TP-Kerze wird auch der SL getroffen:
                penalty = 1.0
          * SL wird vor TP erreicht (TP nie vorher erreicht):
                penalty = 1.0

    tp_sl_stress_score ist der Mittelwert aller per-Trade-Scores.
    Falls keine geeigneten Trades/Candles vorhanden sind, wird 1.0 zurückgegeben.
    """
    arrays = _get_primary_candle_arrays(base_config)
    return compute_tp_sl_stress_score(trades_df, arrays)


def _score_candidates(
    df_pass: pd.DataFrame,
    base_config: Dict[str, Any],
    preload_mode: str,
    *,
    jitter_frac: float,
    jitter_repeats: int,
    dropout_frac: float,
    n_jobs: int,
    deterministic: bool = False,
    deterministic_seed: int = 123,
    equity_out_dir: Optional[Path] = None,
    save_equity_curves: bool = False,
    trades_out_dir: Optional[Path] = None,
    save_trades: bool = False,
    jitter_param_filter: Optional[set[str]] = None,
    data_jitter_repeats: int = 10,
    data_jitter_atr_length: int = 14,
    data_jitter_sigma: float = 0.10,
    data_jitter_fraq: float = 0.15,
) -> pd.DataFrame:
    """
    Alte Step-5-Logik (aus _2):
      • Timing-Jitter um die Parameter → Robustness-Penalty (0..0.5), Robustness-Score = 1 - Penalty
      • Bootstrap-p-Values: mean(R) > 0 und Net Profit > 0
      • Komposit-Score mit den historischen Gewichtungen
      • KEINE Segment-Stabilität, KEIN Cost-Shock, KEIN Dropout
    """
    eq_save_enabled = bool(save_equity_curves) and equity_out_dir is not None
    equity_dir_resolved: Optional[Path] = None
    if eq_save_enabled:
        try:
            equity_dir_resolved = Path(equity_out_dir).resolve()
        except Exception:
            equity_dir_resolved = Path(equity_out_dir)

    trades_save_enabled = bool(save_trades) and trades_out_dir is not None
    trades_dir_resolved: Optional[Path] = None
    if trades_save_enabled:
        try:
            trades_dir_resolved = Path(trades_out_dir).resolve()
        except Exception:
            trades_dir_resolved = Path(trades_out_dir)

    def _save_equity_curve(portfolio, combo_id: str) -> None:
        if not eq_save_enabled or not combo_id:
            return
        try:
            curve = (
                portfolio.get_equity_curve()
                if hasattr(portfolio, "get_equity_curve")
                else []
            )
            if not curve:
                return
            rows = []
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
                return
            dest_dir = (equity_dir_resolved or Path(equity_out_dir)) / str(combo_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest_dir / "equity.csv", index=False)
        except Exception:
            # I/O oder Konvertierungsfehler sollen die Stresstests nicht blockieren
            pass

    def _save_trades(trades_df: Optional[pd.DataFrame], combo_id: str) -> None:
        if not trades_save_enabled or not combo_id:
            return
        try:
            if trades_df is None or trades_df.empty:
                return
        except Exception:
            return
        try:
            df = trades_df.copy()
            # Nur abgeschlossene Trades erwarten wir hier typischerweise bereits,
            # aber defensiv prüfen wir das Schema nicht weiter.
            # Timestamps in Strings konvertieren, damit JSON serialisierbar bleibt.
            for col in ("entry_time", "exit_time"):
                if col in df.columns:
                    df[col] = (
                        pd.to_datetime(df[col], errors="coerce")
                        .dt.tz_convert("UTC")
                        .astype("datetime64[ns, UTC]")
                    )
                    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            records = df.to_dict(orient="records")
            if not records:
                return
            dest_dir = (trades_dir_resolved or Path(trades_out_dir)) / str(combo_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "trades.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            # I/O-Fehler sollen die Stresstests nicht blockieren
            pass

    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 4
            n_jobs = max(1, cpu - 1)
        except Exception:
            n_jobs = 3

    # Param-Spalten so bestimmen wie früher (_2)
    param_cols = [
        c
        for c in df_pass.columns
        if c
        not in (
            "combo_id",
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
        )
    ]

    # Optional: nur eine Teilmenge der Parameter jittertbar machen
    # (Schnittmenge aus param_grid und reporting.param_jitter_include_by_scenario
    #  für das gehandelte Szenario, siehe _infer_jitter_param_filter_from_snapshot).
    allowed_for_jitter: Optional[set[str]] = None
    if jitter_param_filter:
        try:
            allowed_for_jitter = {
                str(p) for p in jitter_param_filter if str(p) in param_cols
            }
        except Exception:
            allowed_for_jitter = None
        if allowed_for_jitter is not None and not allowed_for_jitter:
            # Leere Schnittmenge → Fallback: alle Param-Spalten jitterbar
            allowed_for_jitter = None

    def _one(row, base_cfg, preload_mode_inner):
        _ensure_preloaded_in_worker(base_cfg, preload_mode_inner)
        params = {k: getattr(row, k) for k in param_cols if hasattr(row, k)}
        combo_id = getattr(row, "combo_id", "")

        # Deterministic seeds: one global root seed, then per combo and per sub-run derivation.
        if bool(deterministic):
            combo_seed = _stable_int_seed(
                int(deterministic_seed), "final_selection", str(combo_id)
            )
            rng_param = np.random.default_rng(
                _stable_int_seed(combo_seed, "param_jitter_rng")
            )
            rng_timing = np.random.default_rng(
                _stable_int_seed(combo_seed, "timing_jitter_rng")
            )
        else:
            try:
                seed_int = int(str(combo_id)[:8], 16)
            except Exception:
                seed_int = 12345
            combo_seed = int(1234567 + seed_int)
            rng_param = np.random.default_rng(combo_seed)
            rng_timing = rng_param

        # Basis-Run
        cfg_base = _inject_params(base_cfg, params)
        if bool(deterministic):
            _set_execution_random_seed(
                cfg_base, _stable_int_seed(combo_seed, "bt_base")
            )
        portfolio, extra = _with_worker_preload(cfg_base)
        trades_df = _extract_trades(portfolio, extra)
        summ = calculate_metrics(portfolio)
        if eq_save_enabled:
            _save_equity_curve(portfolio, combo_id)
        if trades_save_enabled:
            _save_trades(trades_df, combo_id)
        base = {
            "profit": float(summ.get("net_profit_after_fees_eur", 0.0) or 0.0),
            "avg_r": float(summ.get("avg_r_multiple", 0.0) or 0.0),
            "winrate": float(summ.get("winrate_percent", 0.0) or 0.0),
            "drawdown": float(summ.get("drawdown_eur", 0.0) or 0.0),
        }
        # Robustness-1: Parameter-Jitter (Samples sammeln, Berechnung zentral im Modul)
        param_jitter_samples: list[dict[str, float]] = []
        jitter_attempts = int(jitter_repeats)
        for i_repeat in range(jitter_attempts):
            try:
                jittered = {}
                for p in param_cols:
                    v = getattr(row, p)
                    # Falls ein Filter gesetzt ist: nur diese Parameter jitteren,
                    # alle übrigen unverändert übernehmen.
                    if allowed_for_jitter is not None and p not in allowed_for_jitter:
                        jittered[p] = v
                        continue
                    if isinstance(v, (int, np.integer)):
                        base_val = int(v)
                        span = max(1, int(round(abs(base_val) * float(jitter_frac))))
                        lo = base_val - span
                        hi = base_val + span
                        if lo > hi:
                            jittered[p] = base_val
                        else:
                            jittered[p] = int(rng_param.integers(lo, hi + 1))
                    elif isinstance(v, (float, np.floating)):
                        base_val = float(v)
                        span = abs(base_val) * float(jitter_frac)
                        if span == 0.0:
                            span = float(jitter_frac)
                        lo = base_val - span
                        hi = base_val + span
                        if hi <= lo:
                            jittered[p] = base_val
                        else:
                            jittered[p] = float(rng_param.uniform(lo, hi))
                    else:
                        jittered[p] = v
                cfg_j = _inject_params(base_cfg, jittered)
                if bool(deterministic):
                    _set_execution_random_seed(
                        cfg_j, _stable_int_seed(combo_seed, "bt_r1", int(i_repeat))
                    )
                portfolio_j, extra_j = _with_worker_preload(cfg_j)
                summ_j = calculate_metrics(portfolio_j)
                param_jitter_samples.append(
                    {
                        "profit": float(
                            summ_j.get("net_profit_after_fees_eur", 0.0) or 0.0
                        ),
                        "avg_r": float(summ_j.get("avg_r_multiple", 0.0) or 0.0),
                        "winrate": float(summ_j.get("winrate_percent", 0.0) or 0.0),
                        "drawdown": float(summ_j.get("drawdown_eur", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue
        robust_score = float(
            compute_robustness_score_1(base, param_jitter_samples, penalty_cap=0.5)
        )

        # Data Jitter Score: ATR-based OHLC jitter stress-test
        data_jitter_score = 0.0
        try:
            if int(data_jitter_repeats) > 0:
                # Lade/hole preloaded data (funktioniert auch bei window mode)
                preloaded_for_jitter = _load_or_get_preloaded_data(base_cfg)
                if preloaded_for_jitter:
                    # Einmalig ATR-Cache berechnen (cached per worker, hier für diesen combo)
                    atr_cache = precompute_atr_cache(
                        preloaded_for_jitter, period=int(data_jitter_atr_length)
                    )
                    data_jitter_samples: List[Dict[str, float]] = []
                    for dj_repeat in range(int(data_jitter_repeats)):
                        try:
                            dj_seed = (
                                _stable_data_jitter_seed(combo_seed, dj_repeat)
                                if bool(deterministic)
                                else int(np.random.default_rng().integers(0, 2**31))
                            )
                            jittered_preloaded = build_jittered_preloaded_data(
                                preloaded_for_jitter,
                                atr_cache=atr_cache,
                                sigma_atr=float(data_jitter_sigma),
                                seed=dj_seed,
                                min_price=1e-9,
                                fraq=float(data_jitter_fraq),
                            )
                            # Run backtest with jittered data
                            cfg_dj = _inject_params(base_cfg, params)
                            if bool(deterministic):
                                _set_execution_random_seed(
                                    cfg_dj,
                                    _stable_int_seed(
                                        combo_seed, "bt_dj", int(dj_repeat)
                                    ),
                                )
                            portfolio_dj, _ = run_backtest_and_return_portfolio(
                                cfg_dj, preloaded_data=jittered_preloaded
                            )
                            summ_dj = calculate_metrics(portfolio_dj)
                            data_jitter_samples.append(
                                {
                                    "profit": float(
                                        summ_dj.get("net_profit_after_fees_eur", 0.0)
                                        or 0.0
                                    ),
                                    "avg_r": float(
                                        summ_dj.get("avg_r_multiple", 0.0) or 0.0
                                    ),
                                    "winrate": float(
                                        summ_dj.get("winrate_percent", 0.0) or 0.0
                                    ),
                                    "drawdown": float(
                                        summ_dj.get("drawdown_eur", 0.0) or 0.0
                                    ),
                                }
                            )
                        except Exception:
                            continue
                    if data_jitter_samples:
                        data_jitter_score = float(
                            compute_data_jitter_score(
                                base, data_jitter_samples, penalty_cap=0.5
                            )
                        )
        except Exception:
            data_jitter_score = 0.0

        # Base-Metriken (profit/drawdown/sharpe) für Stress-Scores (ohne erneuten Run)
        base_metrics = {
            "profit": float(summ.get("net_profit_after_fees_eur", 0.0) or 0.0),
            "drawdown": float(summ.get("drawdown_eur", 0.0) or 0.0),
            "profit_over_dd": float(summ.get("profit_over_dd", 0.0) or 0.0),
            "sharpe": float(summ.get("sharpe_trade", 0.0) or 0.0),
            "sortino": float(summ.get("sortino_trade", 0.0) or 0.0),
            "commission": float(summ.get("fees_total_eur", 0.0) or 0.0),
            "trades": int(summ.get("total_trades", 0) or 0),
            "comm_over_profit": float(summ.get("comm_over_profit", np.nan) or np.nan),
        }
        if not math.isfinite(base_metrics["profit_over_dd"]):
            dd_safe = (
                base_metrics["drawdown"] if base_metrics["drawdown"] != 0 else np.nan
            )
            base_metrics["profit_over_dd"] = (
                base_metrics["profit"] / dd_safe if math.isfinite(dd_safe) else np.nan
            )
        if not math.isfinite(base_metrics["comm_over_profit"]):
            base_metrics["comm_over_profit"] = (
                base_metrics["commission"] / base_metrics["profit"]
                if base_metrics["profit"] not in (0, np.nan)
                else np.nan
            )

        # Tail/Drawdown metric (Ulcer Index only)
        ulcer_index = math.nan
        ulcer_index_score = 0.0

        try:
            rep_metrics = base_cfg.get("reporting", {}) or {}
        except Exception:
            rep_metrics = {}
        try:
            ulcer_cap = float(rep_metrics.get("ulcer_cap", 10.0))
        except Exception:
            ulcer_cap = 10.0

        try:
            equity_curve = (
                portfolio.get_equity_curve()
                if hasattr(portfolio, "get_equity_curve")
                else []
            )
        except Exception:
            equity_curve = []
        try:
            ulcer_index, ulcer_index_score = compute_ulcer_index_and_score(
                equity_curve, ulcer_cap=ulcer_cap
            )
        except Exception:
            ulcer_index, ulcer_index_score = math.nan, 0.0

        # Independent stress scores (each capped with penalty_cap=0.5)
        base_stress = {
            "profit": float(base_metrics.get("profit", 0.0) or 0.0),
            "drawdown": float(base_metrics.get("drawdown", 0.0) or 0.0),
            "sharpe": float(base_metrics.get("sharpe", 0.0) or 0.0),
        }

        # Cost shock (multi-factor, deterministic)
        cost_shocked_metrics: List[Dict[str, float]] = []
        for cs_factor in COST_SHOCK_FACTORS:
            try:
                cost_metrics, _ = _run_with_metrics(
                    base_cfg,
                    params,
                    preload_mode_inner,
                    seed=(
                        _stable_int_seed(combo_seed, f"bt_cost_shock_{cs_factor}")
                        if bool(deterministic)
                        else None
                    ),
                    mutate=lambda cfg, f=cs_factor: apply_cost_shock_inplace(
                        cfg, factor=f
                    ),
                )
            except Exception:
                cost_metrics = {
                    "profit": 0.0,
                    "drawdown": float(base_stress.get("drawdown", 0.0) or 0.0) * 2,
                    "sharpe": 0.0,
                }
            cost_shocked_metrics.append(cost_metrics)

        cost_shock_score = float(
            compute_multi_factor_cost_shock_score(
                base_stress, cost_shocked_metrics, penalty_cap=0.5
            )
        )

        # Time jitter (BACKWARD month shifts): 3 fixed intervals derived from the
        # overall backtest window length: /10, /5, /20 (min 1 month). We do NOT
        # shift forward to avoid "looking into the future".
        timing_jitter_metrics: List[Dict[str, float]] = []

        # Optional debug output for validating timing-jitter window mutation.
        # Enable via config:
        #   reporting: { debug_timing_jitter: true }
        # or via env var:
        #   TIMING_JITTER_DEBUG=1
        try:
            rep_dbg = base_cfg.get("reporting", {}) or {}
            debug_timing_jitter = bool(rep_dbg.get("debug_timing_jitter", False))
        except Exception:
            debug_timing_jitter = False
        try:
            env_flag = str(os.getenv("TIMING_JITTER_DEBUG", "") or "").strip().lower()
            if env_flag in {"1", "true", "yes", "y", "on"}:
                debug_timing_jitter = True
        except Exception:
            pass

        shift_months_list = get_timing_jitter_backward_shift_months(
            start_date=str(base_cfg.get("start_date") or ""),
            end_date=str(base_cfg.get("end_date") or ""),
            divisors=(10, 5, 20),
            min_months=1,
        )

        if debug_timing_jitter:
            try:
                print(
                    "[timing-jitter][debug] combo_id={} base_window={}..{} shifts_months={}".format(
                        combo_id,
                        str(base_cfg.get("start_date") or ""),
                        str(base_cfg.get("end_date") or ""),
                        list(map(int, shift_months_list)) if shift_months_list else [],
                    )
                )
            except Exception:
                pass

        for i_tj, shift_months in enumerate(shift_months_list):

            def _mutate_timing_jitter(cfg: Dict[str, Any], *, months: int) -> None:
                before_start = str(cfg.get("start_date") or "")
                before_end = str(cfg.get("end_date") or "")
                apply_timing_jitter_month_shift_inplace(
                    cfg, shift_months_backward=int(months)
                )
                if debug_timing_jitter:
                    try:
                        print(
                            "[timing-jitter][debug] combo_id={} shift_months={} {}..{} -> {}..{}".format(
                                combo_id,
                                int(months),
                                before_start,
                                before_end,
                                str(cfg.get("start_date") or ""),
                                str(cfg.get("end_date") or ""),
                            )
                        )
                    except Exception:
                        pass

            met, _ = _run_with_metrics(
                base_cfg,
                params,
                preload_mode_inner,
                seed=(
                    _stable_int_seed(
                        combo_seed, "bt_timing_months", int(i_tj), int(shift_months)
                    )
                    if bool(deterministic)
                    else None
                ),
                mutate=lambda cfg, m=shift_months: _mutate_timing_jitter(
                    cfg, months=int(m)
                ),
            )
            timing_jitter_metrics.append(met)

        timing_jitter_score = float(
            compute_timing_jitter_score(
                base_stress, timing_jitter_metrics, penalty_cap=0.5
            )
            if timing_jitter_metrics
            else 0.0
        )

        # Trade dropout (multi-run, averaged)
        try:
            rep_dbg = base_cfg.get("reporting", {}) or {}
            debug_trade_dropout = bool(rep_dbg.get("debug_trade_dropout", False))
        except Exception:
            debug_trade_dropout = False
        try:
            dropout_runs = int(rep_dbg.get("robust_dropout_runs", 1) or 1)
        except Exception:
            dropout_runs = 1

        dropout_seed_base: Optional[int]
        if bool(deterministic):
            dropout_seed_base = _stable_int_seed(combo_seed, "trade_dropout_runs")
        else:
            dropout_seed_base = None

        dropout_metrics_list = simulate_trade_dropout_metrics_multi(
            trades_df,
            dropout_frac=float(dropout_frac),
            base_metrics=base_stress,
            n_runs=dropout_runs,
            seed=dropout_seed_base,
            debug=debug_trade_dropout,
        )
        trade_dropout_score = float(
            compute_multi_run_trade_dropout_score(
                base_stress, dropout_metrics_list, penalty_cap=0.5
            )
        )

        # p-values (centralized; default net-of-fees to match "Net Profit after fees")
        try:
            rep_cfg = base_cfg.get("reporting", {}) or {}
            p_net_of_fees = bool(rep_cfg.get("p_values_net_of_fees", True))
        except Exception:
            p_net_of_fees = True
        if bool(deterministic):
            seed_r = int(_stable_int_seed(combo_seed, "p_values_r"))
            seed_pnl = int(_stable_int_seed(combo_seed, "p_values_pnl"))
        else:
            seed_r = 123
            seed_pnl = 456
        pvals = compute_p_values(
            trades_df,
            r_col="r_multiple",
            pnl_col="result",
            net_of_fees_pnl=p_net_of_fees,
            n_boot=2000,
            seed_r=int(seed_r),
            seed_pnl=int(seed_pnl),
        )
        p_r = float(pvals.get("p_mean_r_gt_0", 1.0) or 1.0)
        p_pnl = float(pvals.get("p_net_profit_gt_0", 1.0) or 1.0)

        # TP/SL-Stress-Score (per-Trade Exit-Robustheit)
        tp_sl_score = _compute_tp_sl_stress(trades_df, base_cfg)
        try:
            print(f"🔎 [tp_sl_stress] combo_id={combo_id} " f"score={tp_sl_score:.4f}")
        except Exception:
            pass

        # Base net profit and drawdown used in output metrics
        net = float(base_metrics.get("profit", 0.0) or 0.0)
        dd = float(base_metrics.get("drawdown", 0.0) or 0.0)

        out = {
            "combo_id": getattr(row, "combo_id", ""),
            # Score is computed later after stability_score is known (Step 5 final aggregation)
            "score": np.nan,
            "Net Profit": net,
            "Drawdown": dd,
            "profit_over_dd": (net / dd) if dd > 0 else 0.0,
            "Commission": float(summ.get("fees_total_eur", 0.0) or 0.0),
            "Avg R-Multiple": float(summ.get("avg_r_multiple", 0.0) or 0.0),
            "Winrate (%)": float(summ.get("winrate_percent", 0.0) or 0.0),
            "Sharpe (trade)": float(summ.get("sharpe_trade", 0.0) or 0.0),
            "Sortino (trade)": float(summ.get("sortino_trade", 0.0) or 0.0),
            "robustness_score_1": robust_score,
            "data_jitter_score": data_jitter_score,
            "cost_shock_score": cost_shock_score,
            "timing_jitter_score": timing_jitter_score,
            "trade_dropout_score": trade_dropout_score,
            "tp_sl_stress_score": float(tp_sl_score),
            "p_mean_r_gt_0": float(round(p_r, 4)),
            "p_net_profit_gt_0": float(round(p_pnl, 4)),
        }
        for k in param_cols:
            out[k] = getattr(row, k, np.nan)
        out["ulcer_index"] = float(ulcer_index)
        out["ulcer_index_score"] = float(ulcer_index_score)
        return out

    items = list(df_pass.itertuples(index=False))
    rows: List[Dict[str, Any]] = []
    prog = _Progress(len(items), label="Stresstests & Scoring")
    for chunk in _parallel_iter(
        items,
        func=_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
    ):
        rows.extend(chunk)
        prog.update(len(chunk))
    prog.done()
    df = pd.DataFrame(rows)
    # Do not sort by score here; final scoring is done after stability merge
    return df.reset_index(drop=True)


def _infer_jitter_param_filter_from_snapshot(
    *,
    walkforward_root: str,
    base_config: Dict[str, Any],
    param_grid: Dict[str, Any],
) -> Optional[set[str]]:
    """
    Ermittelt die Menge an Parametern, die für Robustness‑1 gejittert werden sollen,
    basierend auf dem frozen_snapshot einer Walkforward‑Runs:

      - Lies base_config aus baseline/frozen_snapshot.json (falls vorhanden),
        ansonsten verwende das übergebene base_config.
      - Bestimme das gehandelte Szenario über strategy.parameters.enabled_scenarios.
      - Schneide reporting.param_jitter_include_by_scenario[scenarioX] mit den
        Schlüsseln des param_grid.

    Rückgabe:
      - Menge an Parameternamen, die gejittert werden dürfen, oder
      - None, falls kein Filter angewendet werden soll.
    """
    try:
        root = Path(walkforward_root)
    except Exception:
        root = Path(".")

    cfg = base_config
    try:
        manifest_path = root / "baseline" / "baseline_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            snap_str = str(manifest.get("frozen_snapshot") or "").strip()
            frozen_path = (
                Path(snap_str)
                if snap_str
                else root / "baseline" / "frozen_snapshot.json"
            )
        else:
            frozen_path = root / "baseline" / "frozen_snapshot.json"
        if frozen_path.exists():
            try:
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                frozen_base = frozen.get("base_config")
                if isinstance(frozen_base, dict):
                    cfg = frozen_base
            except Exception:
                pass
    except Exception:
        # Fallback: verwende das übergebene base_config unverändert
        cfg = base_config

    try:
        strat_cfg = cfg.get("strategy") or {}
        params = strat_cfg.get("parameters") or {}
        enabled = params.get("enabled_scenarios")
        scenario_key: Optional[str] = None
        if isinstance(enabled, (list, tuple)):
            for v in enabled:
                try:
                    iv = int(v)
                    scenario_key = f"scenario{iv}"
                    break
                except Exception:
                    continue

        reporting_cfg = cfg.get("reporting") or {}
        by_scenario = reporting_cfg.get("param_jitter_include_by_scenario") or {}
        if not isinstance(by_scenario, dict):
            return None

        allowed: Optional[set[str]] = None
        if scenario_key and scenario_key in by_scenario:
            vals = by_scenario.get(scenario_key)
            if isinstance(vals, (list, tuple, set)):
                allowed = {str(x) for x in vals}

        if not allowed:
            return None

        grid_names = {str(k) for k in param_grid.keys()}
        names = allowed & grid_names
        if not names:
            return None
        try:
            desc = ", ".join(sorted(names))
        except Exception:
            desc = ", ".join(sorted(str(x) for x in names))
        try:
            print(
                f"🔎 [FinalSel] Robustness‑1 Jitter‑Filter aktiv "
                f"(scenario={scenario_key or 'n/a'}): {desc}"
            )
        except Exception:
            pass
        return names
    except Exception:
        return None


def _run_with_metrics(
    base_config: Dict[str, Any],
    params: Dict[str, Any],
    preload_mode: str,
    *,
    seed: Optional[int] = None,
    mutate: Optional[Any] = None,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    cfg = _inject_params(base_config, params)
    if mutate is not None:
        mutate(cfg)
    if seed is not None:
        _set_execution_random_seed(cfg, int(seed))
    portfolio, extra = _with_worker_preload(cfg)
    trades_df = _extract_trades(portfolio, extra)
    summary = calculate_metrics(portfolio)
    metrics = {
        "profit": float(summary.get("net_profit_after_fees_eur", 0.0) or 0.0),
        "drawdown": float(summary.get("drawdown_eur", 0.0) or 0.0),
        "profit_over_dd": float(summary.get("profit_over_dd", 0.0) or 0.0),
        "sharpe": float(summary.get("sharpe_trade", 0.0) or 0.0),
        "sortino": float(summary.get("sortino_trade", 0.0) or 0.0),
        "commission": float(summary.get("fees_total_eur", 0.0) or 0.0),
        "trades": int(summary.get("total_trades", 0) or 0),
        "comm_over_profit": float(summary.get("comm_over_profit", np.nan) or np.nan),
    }
    if not math.isfinite(metrics["profit_over_dd"]):
        dd = metrics["drawdown"] if metrics["drawdown"] != 0 else np.nan
        metrics["profit_over_dd"] = (
            metrics["profit"] / dd if math.isfinite(dd) else np.nan
        )
    if not math.isfinite(metrics["comm_over_profit"]):
        metrics["comm_over_profit"] = (
            metrics["commission"] / metrics["profit"]
            if metrics["profit"] not in (0, np.nan)
            else np.nan
        )
    return metrics, trades_df


def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    metrics = [
        ("Net Profit", True),
        ("profit_over_dd", True),
        ("Sharpe (trade)", True),
        ("Drawdown", False),
        ("comm_over_profit", False),
        ("total_trades", True),
    ]
    values = df.copy()
    for col, _ in metrics:
        values[col] = pd.to_numeric(values[col], errors="coerce")

    dominated = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        if dominated[i]:
            continue
        for j in range(len(values)):
            if i == j or dominated[j]:
                continue
            if _dominates(values.iloc[j], values.iloc[i], metrics):
                dominated[i] = True
                break
    return values.loc[~dominated].reset_index(drop=True)


def _dominates(row_a: pd.Series, row_b: pd.Series, metrics) -> bool:
    better_or_equal = True
    strictly_better = False
    for col, maximize in metrics:
        a = row_a.get(col)
        b = row_b.get(col)
        if not math.isfinite(a):
            a = -math.inf if maximize else math.inf
        if not math.isfinite(b):
            b = -math.inf if maximize else math.inf
        if maximize:
            if a < b:
                better_or_equal = False
                break
            if a > b:
                strictly_better = True
        else:
            if a > b:
                better_or_equal = False
                break
            if a < b:
                strictly_better = True
    return better_or_equal and strictly_better


def _cluster_candidates(df: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["combo_id", "cluster_id"])
    data = df[["combo_id"] + param_cols].copy()
    encoded = []
    for col in param_cols:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=float)
        else:
            arr = pd.Categorical(series).codes.astype(float)
        if np.std(arr) > 0:
            arr = (arr - np.mean(arr)) / np.std(arr)
        else:
            arr = arr - np.mean(arr)
        encoded.append(arr)
    if not encoded:
        encoded = [np.zeros(len(data))]
    matrix = np.vstack(encoded).T
    n = len(data)
    if n == 1:
        return pd.DataFrame({"combo_id": data["combo_id"], "cluster_id": [0]})
    k = min(5, max(2, math.ceil(n / 10)))
    dist = _distance_matrix(matrix)
    medoids = _init_medoids(dist, k)
    for _ in range(10):
        labels = _assign_clusters(dist, medoids)
        new_medoids = []
        for cluster_id in range(len(medoids)):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                new_medoids.append(medoids[cluster_id])
                continue
            sub = dist[np.ix_(members, members)]
            medoid_local = members[np.argmin(sub.sum(axis=1))]
            new_medoids.append(int(medoid_local))
        new_medoids = sorted(set(new_medoids))
        if medoids == new_medoids:
            break
        medoids = new_medoids
    labels = _assign_clusters(dist, medoids)
    return pd.DataFrame({"combo_id": data["combo_id"], "cluster_id": labels})


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _init_medoids(dist: np.ndarray, k: int) -> List[int]:
    n = dist.shape[0]
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    while len(medoids) < k and len(medoids) < n:
        remaining = [i for i in range(n) if i not in medoids]
        if not remaining:
            break
        dists = dist[np.ix_(remaining, medoids)]
        next_idx = remaining[int(np.argmax(dists.min(axis=1)))]
        medoids.append(next_idx)
    return sorted(medoids)


def _assign_clusters(dist: np.ndarray, medoids: Sequence[int]) -> np.ndarray:
    medoids = list(medoids)
    assign = np.zeros(dist.shape[0], dtype=int)
    for idx in range(dist.shape[0]):
        distances = [dist[idx, m] for m in medoids]
        assign[idx] = int(np.argmin(distances))
    return assign


def _select_medoids(df: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if df.empty or clusters.empty:
        return pd.DataFrame(columns=df.columns)
    merged = df.merge(clusters, on="combo_id", how="inner")
    medoids: List[pd.Series] = []
    for _, group in merged.groupby("cluster_id"):
        idx = group["score"].idxmax()
        medoids.append(group.loc[idx])
    return pd.DataFrame(medoids).reset_index(drop=True)


def _recommend_heavy_n_jobs(default_parent: Optional[int] = None) -> int:
    try:
        env = int(os.getenv("FINALSEL_HEAVY_NJOBS", "").strip() or "0")
        if env > 0:
            return max(1, env)
    except Exception:
        pass
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        total_gb = 8.0
    if total_gb <= 8.5:
        return 1
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2)


def _evaluate_holdout(
    df_medoids: pd.DataFrame,
    base_config: Dict[str, Any],
    preload_mode: str,
    holdout_cfg: Dict[str, Any],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    if df_medoids is None or df_medoids.empty:
        return pd.DataFrame(
            columns=df_medoids.columns if df_medoids is not None else []
        )
    start = holdout_cfg.get("start")
    end = holdout_cfg.get("end")
    if not start or not end:
        return pd.DataFrame(columns=df_medoids.columns)

    param_cols = _parameter_columns(df_medoids)
    rows: List[Dict[str, Any]] = []

    def _one(row, base_cfg, preload_mode_inner):
        params = {k: getattr(row, k) for k in param_cols if hasattr(row, k)}
        cfg = _inject_params(base_cfg, params, start=start, end=end)
        portfolio, extra = _with_worker_preload(cfg)
        trades_df = _extract_trades(portfolio, extra)
        summary = calculate_metrics(portfolio)
        metrics = _metrics_row(params, summary, trades_df)
        metrics["combo_id"] = row.combo_id
        return metrics

    for chunk in _parallel_iter(
        list(df_medoids.itertuples(index=False)),
        func=_one,
        base_config=base_config,
        preload_mode=preload_mode,
        n_jobs=n_jobs,
    ):
        rows.extend(chunk)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["Net Profit"] > 0.0]
    dd = pd.to_numeric(df["Drawdown"], errors="coerce").replace({0.0: np.nan})
    ratio = pd.to_numeric(df["Net Profit"], errors="coerce") / dd
    df = df.loc[ratio >= 1.0]
    return df.reset_index(drop=True)


def _materialize_empties(out_dir: Path, *, start_step: int) -> None:
    names = [
        "01_candidates_raw.csv",
        "02_candidates_after_gates.csv",
        "03_segments_results.csv",
        "04_candidates_segment_pass.csv",
        "05_final_scores.csv",
        "05_final_scores_detailed.csv",
    ]
    for name in names[start_step - 1 :]:
        (out_dir / name).write_text("", encoding="utf-8")


def _ensure_columns_order(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    if "combo_id" in cols:
        cols.insert(0, cols.pop(cols.index("combo_id")))
        df = df.loc[:, cols]
    return df


# --------------------------------------------------------------------
# SMART SEARCH — Helpers (ported from *_2 version)
# --------------------------------------------------------------------
def _stage_windows(base_config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """2M, 6M, volle Periode.

    Hinweis: Die frühere Prüfung des letzten vollen Halbjahres ("H")
    ist entfernt. Nach der 6M-Stage folgt direkt die Vollperiode.
    """
    start_all = datetime.strptime(base_config["start_date"], "%Y-%m-%d")
    end_all = datetime.strptime(base_config["end_date"], "%Y-%m-%d")
    stages: List[Tuple[str, str, str]] = []
    s2 = max(start_all, end_all - timedelta(days=60))
    if (end_all - s2).days >= 30:
        stages.append(("2m", s2.strftime("%Y-%m-%d"), end_all.strftime("%Y-%m-%d")))
    s6 = max(start_all, end_all - timedelta(days=180))
    if (end_all - s6).days >= 120:
        stages.append(("6m", s6.strftime("%Y-%m-%d"), end_all.strftime("%Y-%m-%d")))
    stages.append(("full", base_config["start_date"], base_config["end_date"]))
    return stages  # :contentReference[oaicite:7]{index=7}


def _lhs_seed_samples(
    space: Dict[str, List[Any]], n: int, rng: np.random.Generator
) -> List[Dict[str, Any]]:
    """Einfache LHS für diskrete Räume."""
    params = list(space.keys())
    m = n
    index_grids: Dict[str, np.ndarray] = {}
    for p in params:
        L = len(space[p])
        if L == 0:
            index_grids[p] = np.zeros(m, dtype=int)
            continue
        u = rng.random(m)
        idx = np.clip(
            np.round(((np.arange(m) + u) / max(1, m)) * (L - 1)).astype(int), 0, L - 1
        )
        rng.shuffle(idx)
        index_grids[p] = idx
    seeds: List[Dict[str, Any]] = []
    for i in range(m):
        d: Dict[str, Any] = {}
        for p in params:
            choices = space[p]
            if choices:
                j = int(index_grids[p][i % len(index_grids[p])])
                d[p] = choices[j]
        seeds.append(d)
    return seeds  # :contentReference[oaicite:8]{index=8}


def _objective_from_summary(summ: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Robuster Stage-Score: Profit/DD, mit Abzügen bei Net<=0, Trades<5, Comm/Profit>=0.6"""
    net = float(summ.get("net_profit_after_fees_eur", 0.0) or 0.0)
    dd = float(summ.get("drawdown_eur", 0.0) or 0.0)
    trades = int(summ.get("total_trades", 0) or 0)
    comm = float(summ.get("fees_total_eur", 0.0) or 0.0)
    ratio = (net / dd) if dd > 0 else -1e9
    comm_ratio = (comm / net) if net > 0 else 1e9
    score = ratio
    if net <= 0:
        score -= 5.0
    if trades < 5:
        score -= 1.0
    if np.isfinite(comm_ratio) and comm_ratio >= 0.6:
        score -= 1.0
    return float(score), {
        "Net Profit": net,
        "Drawdown": dd,
        "Commission": comm,
        "total_trades": float(trades),
        "profit_over_dd": (net / dd) if dd > 0 else np.nan,
        "comm_over_profit": (comm / net) if net != 0 else np.nan,
    }  # :contentReference[oaicite:9]{index=9}


def _evaluate_on_window(
    params: Dict[str, Any], base_config: Dict[str, Any], start: str, end: str
):
    cfg = _inject_params(base_config, params, start=start, end=end)
    portfolio, extra = _with_worker_preload(cfg)
    summ = calculate_metrics(portfolio)
    return portfolio, extra, summ  # :contentReference[oaicite:10]{index=10}


def _random_2m_window_within(
    base_config: Dict[str, Any], rng: np.random.Generator
) -> Tuple[str, str]:
    start_all = datetime.strptime(base_config["start_date"], "%Y-%m-%d")
    end_all = datetime.strptime(base_config["end_date"], "%Y-%m-%d")
    full_days = (end_all - start_all).days
    if full_days <= 60:
        return (base_config["start_date"], base_config["end_date"])
    max_offset = max(0, full_days - 60)
    offset = int(rng.integers(0, max_offset + 1))
    s = start_all + timedelta(days=offset)
    e = s + timedelta(days=60)
    if e > end_all:
        s = s - (e - end_all)
        e = end_all
    return (
        s.strftime("%Y-%m-%d"),
        e.strftime("%Y-%m-%d"),
    )  # :contentReference[oaicite:11]{index=11}


def _random_6m_window_within(
    base_config: Dict[str, Any], rng: np.random.Generator
) -> Tuple[str, str]:
    start_all = datetime.strptime(base_config["start_date"], "%Y-%m-%d")
    end_all = datetime.strptime(base_config["end_date"], "%Y-%m-%d")
    full_days = (end_all - start_all).days
    if full_days <= 180:
        return (base_config["start_date"], base_config["end_date"])
    max_offset = max(0, full_days - 180)
    offset = int(rng.integers(0, max_offset + 1))
    s = start_all + timedelta(days=offset)
    e = s + timedelta(days=180)
    if e > end_all:
        s = s - (e - end_all)
        e = end_all
    return (
        s.strftime("%Y-%m-%d"),
        e.strftime("%Y-%m-%d"),
    )  # :contentReference[oaicite:12]{index=12}


def _neighbor_sample(
    anchor: Dict[str, Any], space: Dict[str, List[Any]], rng: np.random.Generator
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p, vals in space.items():
        if not vals:
            out[p] = None
            continue
        try:
            idx = vals.index(anchor[p])
        except Exception:
            out[p] = rng.choice(vals)
            continue
        step = rng.choice([-1, 1]) if len(vals) > 1 else 0
        j = int(np.clip(idx + step, 0, len(vals) - 1))
        if rng.random() < 0.25 and len(vals) > 2:
            j = int(np.clip(j + (-1 if step > 0 else 1), 0, len(vals) - 1))
        out[p] = vals[j]
    if all(anchor.get(k) == out.get(k) for k in space.keys()):
        rp = rng.choice(list(space.keys()))
        if len(space[rp]) > 1:
            cand = [v for v in space[rp] if v != out[rp]]
            out[rp] = rng.choice(cand)
    return out  # :contentReference[oaicite:13]{index=13}


def _random_sample(
    space: Dict[str, List[Any]], rng: np.random.Generator
) -> Dict[str, Any]:
    return {
        p: (rng.choice(v) if len(v) > 0 else None) for p, v in space.items()
    }  # :contentReference[oaicite:14]{index=14}


def _discretized_space_from_zones(
    zones_grid: Dict[str, Any], param_grid: Dict[str, Any]
) -> Tuple[Dict[str, List[Any]], Dict[str, Optional[float]]]:
    z = zones_grid.get("zones", {})
    space: Dict[str, List[Any]] = {}
    step_hint: Dict[str, Optional[float]] = {}
    for p, spec in param_grid.items():
        if spec.get("type") == "categorical":
            vals = (
                list(z.get(p, [{}]))[0].get("choices", spec.get("choices", []))
                if (p in z and z[p])
                else list(spec.get("choices", []))
            )
            space[p] = list(vals)
            step_hint[p] = None
        else:
            choices: List[float] = []
            st_used: Optional[float] = None
            for entry in z.get(p, []):
                lo = float(entry.get("min", entry.get("low", spec.get("low"))))
                hi = float(entry.get("max", entry.get("high", spec.get("high"))))
                st = float(entry.get("step", spec.get("step", 1.0))) or 1.0
                k = int(math.floor((hi - lo) / st)) + 1
                seq = [lo + i * st for i in range(max(k, 1))]
                choices.extend(seq)
                st_used = st if st_used is None else min(st_used, st)
            if not choices:
                lo = float(spec.get("low"))
                hi = float(spec.get("high"))
                st = float(spec.get("step", 1.0))
                k = int(math.floor((hi - lo) / st)) + 1
                choices = [lo + i * st for i in range(max(k, 1))]
                st_used = st
            vals = sorted(set(float(round(v, 6)) for v in choices))
            if spec.get("type") == "int":
                vals = [int(round(v)) for v in vals]
            space[p] = vals
            step_hint[p] = float(st_used) if st_used is not None else None
    return space, step_hint  # :contentReference[oaicite:15]{index=15}


def _smart_parameter_search(
    *,
    zones_grid: Dict[str, Any],
    param_grid: Dict[str, Any],
    base_config: Dict[str, Any],
    out_dir: Path,
    n_jobs: Optional[int],
    preload_mode: str,
    n_trials: int,
    topk_reeval: int,
    seed: int,
    explore_prob: float,
    trust_every: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Smart-Suche: Seeds (LHS) → 2M/6M mit Pruning → Exploit/Explore + Trust-Region → Vollperiode für alle Survivors.

    Änderungen:
    - 2M-Stage nutzt dynamische Mindest-Trades je Timeframe
      (M5→25, M15→20, M30→15, H1/H4/D1→10).
    - Prüfung des letzten vollen Halbjahres wurde entfernt.
    - Nach 6M erfolgt kein Top-K-Reeval; stattdessen wird die Vollperiode
      für alle Kandidaten gefahren, die die Filter bestehen.
    """
    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 4
            n_jobs = max(1, cpu - 1)
        except Exception:
            n_jobs = 3
    rng = np.random.default_rng(int(seed))
    space, _ = _discretized_space_from_zones(zones_grid, param_grid)
    stages = _stage_windows(base_config)
    n_seed = min(max(10, int(round(0.2 * n_trials))), n_trials)
    seeds = _lhs_seed_samples(space, n_seed, rng)
    seen: set = set()
    trials_rows: List[Dict[str, Any]] = []
    archive: List[Tuple[float, Dict[str, Any]]] = []

    # Dynamische Mindest-Trades je Timeframe für 2M-Stage
    tf = (
        ((base_config.get("timeframes") or {}).get("primary", "D1"))
        if isinstance(base_config.get("timeframes"), dict)
        else "D1"
    )
    tf = str(tf).upper()
    _min_trades_map = {"M5": 25, "M15": 20, "M30": 15, "H1": 10, "H4": 10, "D1": 10}
    min_trades_2m = int(_min_trades_map.get(tf, 10))

    def register_trial(
        trial_idx: int,
        stage_label: str,
        stage_start: str,
        stage_end: str,
        params: Dict[str, Any],
        summ: Dict[str, Any],
        score: float,
    ):
        row = {
            "trial": trial_idx,
            "stage": stage_label,
            "stage_start": stage_start,
            "stage_end": stage_end,
            "stage_score": float(score),
        }
        tmp = {k: v for k, v in params.items()}
        tmp["_combo_id"] = _combo_id(tmp)
        row["combo_id"] = tmp["_combo_id"]
        _, met_vals = _objective_from_summary(summ)
        row.update(tmp)
        row.update(met_vals)
        trials_rows.append(row)
        return tmp["_combo_id"]

    prog = _Progress(n_trials, label="SmartSearch Trials")
    trial_counter = 0
    # 1) Seeds
    for s in seeds:
        if trial_counter >= n_trials:
            break
        cand = s
        cid = _combo_id(cand)
        if cid in seen:
            continue
        seen.add(cid)
        last_score = -1e12
        pruned = False
        for lab, s0, s1 in stages[:-1]:
            if lab == "2m":
                agg_profit = 0.0
                total_trades = 0
                draws = 0
                while total_trades < min_trades_2m and draws < 30:
                    r0, r1 = _random_2m_window_within(base_config, rng)
                    _, _, summ = _evaluate_on_window(cand, base_config, r0, r1)
                    score, _ = _objective_from_summary(summ)
                    register_trial(trial_counter, "2m.rand", r0, r1, cand, summ, score)
                    agg_profit += float(
                        summ.get("net_profit_after_fees_eur", 0.0) or 0.0
                    )
                    total_trades += int(summ.get("total_trades", 0) or 0)
                    draws += 1
                last_score = float(agg_profit)
                if agg_profit <= 0.0:
                    pruned = True
                    break
                continue
            if lab == "6m":
                agg_profit_6m = 0.0
                max_dd_6m = 0.0
                total_trades_6m = 0
                draws_6m = 0
                while total_trades_6m < 30 and draws_6m < 80:
                    r0, r1 = _random_6m_window_within(base_config, rng)
                    _, _, summ6 = _evaluate_on_window(cand, base_config, r0, r1)
                    score6, _ = _objective_from_summary(summ6)
                    register_trial(
                        trial_counter, "6m.rand", r0, r1, cand, summ6, score6
                    )
                    agg_profit_6m += float(
                        summ6.get("net_profit_after_fees_eur", 0.0) or 0.0
                    )
                    dd6 = float(summ6.get("drawdown_eur", 0.0) or 0.0)
                    if np.isfinite(dd6):
                        max_dd_6m = float(max(max_dd_6m, dd6))
                    total_trades_6m += int(summ6.get("total_trades", 0) or 0)
                    draws_6m += 1
                if max_dd_6m <= 0.0:
                    pruned = True
                    break
                ratio_6m = float(agg_profit_6m / max_dd_6m)
                last_score = ratio_6m
                if ratio_6m <= 1.5:
                    pruned = True
                    break
                continue
            # keine weitere Stage (H) mehr
        if not pruned:
            archive.append((last_score, cand))
        trial_counter += 1
        prog.update(1)

    # 2) Iterative Exploration/Exploitation
    while trial_counter < n_trials:
        pick_neighbor = (
            trust_every > 0
            and trial_counter > 0
            and trial_counter % trust_every == 0
            and len(archive) > 0
        )
        if pick_neighbor:
            best_params = max(archive, key=lambda t: t[0])[1]
            cand = _neighbor_sample(best_params, space, rng)
        else:
            if rng.random() < float(explore_prob) or len(archive) == 0:
                cand = _random_sample(space, rng)
            else:
                k = max(1, int(math.ceil(0.2 * len(archive))))
                top = sorted(archive, key=lambda t: t[0], reverse=True)[:k]
                anchor = top[rng.integers(0, len(top))][1]
                cand = _neighbor_sample(anchor, space, rng)
        cid = _combo_id(cand)
        if cid in seen:
            continue
        seen.add(cid)
        last_score = -1e12
        pruned = False
        for lab, s0, s1 in stages[:-1]:
            if lab == "2m":
                agg_profit = 0.0
                total_trades = 0
                draws = 0
                while total_trades < min_trades_2m and draws < 30:
                    r0, r1 = _random_2m_window_within(base_config, rng)
                    _, _, summ = _evaluate_on_window(cand, base_config, r0, r1)
                    score, _ = _objective_from_summary(summ)
                    register_trial(trial_counter, "2m.rand", r0, r1, cand, summ, score)
                    agg_profit += float(
                        summ.get("net_profit_after_fees_eur", 0.0) or 0.0
                    )
                    total_trades += int(summ.get("total_trades", 0) or 0)
                    draws += 1
                last_score = float(agg_profit)
                if agg_profit <= 0.0:
                    pruned = True
                    break
                continue
            if lab == "6m":
                agg_profit_6m = 0.0
                max_dd_6m = 0.0
                total_trades_6m = 0
                draws_6m = 0
                while total_trades_6m < 30 and draws_6m < 80:
                    r0, r1 = _random_6m_window_within(base_config, rng)
                    _, _, summ6 = _evaluate_on_window(cand, base_config, r0, r1)
                    score6, _ = _objective_from_summary(summ6)
                    register_trial(
                        trial_counter, "6m.rand", r0, r1, cand, summ6, score6
                    )
                    agg_profit_6m += float(
                        summ6.get("net_profit_after_fees_eur", 0.0) or 0.0
                    )
                    dd6 = float(summ6.get("drawdown_eur", 0.0) or 0.0)
                    if np.isfinite(dd6):
                        max_dd_6m = float(max(max_dd_6m, dd6))
                    total_trades_6m += int(summ6.get("total_trades", 0) or 0)
                    draws_6m += 1
                if max_dd_6m <= 0.0:
                    pruned = True
                    break
                ratio_6m = float(agg_profit_6m / max_dd_6m)
                last_score = ratio_6m
                if ratio_6m <= 1.5:
                    pruned = True
                    break
                continue
            # keine weitere Stage (H) mehr
        if not pruned:
            archive.append((last_score, cand))
        trial_counter += 1
        prog.update(1)
    prog.done()

    # 3) Vollperiode für ALLE Survivors (keine Top-K-Filterung)
    survivors_params: List[Dict[str, Any]] = []
    seen_survivors: set[str] = set()
    for _, params in archive:
        cid = _combo_id(params)
        if cid in seen_survivors:
            continue
        seen_survivors.add(cid)
        d = dict(params)
        d["_combo_id"] = cid
        survivors_params.append(d)

    if survivors_params:
        print(
            f"🔁 Vollperioden-Test für {len(survivors_params)} Survivors … (n_jobs={n_jobs or 1})"
        )
        best_full_df = _evaluate_candidates_full_period(
            survivors_params,
            base_config,
            preload_mode=preload_mode or "window",
            n_jobs=n_jobs,
        )
    else:
        best_full_df = pd.DataFrame()
    trials_df = pd.DataFrame(trials_rows)
    return (
        trials_df.sort_values(["trial", "stage"]).reset_index(drop=True),
        best_full_df,
    )  # :contentReference[oaicite:16]{index=16}


def _render_report(
    out_dir: Path,
    manifest: Dict[str, Any],
    *,
    reason: Optional[str] = None,
    shortlist: Optional[pd.DataFrame] = None,
) -> Path:
    md: List[str] = ["# Final Parameter Selection\n"]
    if reason:
        md.append(f"> Early exit reason: **{reason}**\n")
    md.append("## Manifest\n")
    md.append("```json\n")
    md.append(_json_dumps(manifest, indent=2))
    md.append("\n```\n")
    md.append("## Artifacts\n")
    for name in [
        "01_candidates_raw.csv",
        "02_candidates_after_gates.csv",
        "03_segments_results.csv",
        "04_candidates_segment_pass.csv",
        "05_final_scores.csv",
        "05_final_scores_detailed.csv",
    ]:
        md.append(f"- {name}\n")
    if shortlist is not None and not shortlist.empty:
        md.append("\n## Shortlist\n")
        md.append(shortlist.to_markdown(index=False))
        md.append("\n")
    report = out_dir / "FINAL_REPORT.md"
    report.write_text("\n".join(md), encoding="utf-8")
    return report
