# hf_engine/strategies/mean_reversion_m5/backtest_strategy.py
"""Mean Reversion Z-Score backtest strategy implementation."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
from backtest_engine.data.news_filter import NewsFilter
from strategies.mean_reversion_z_score.backtest.position_manager import (  # noqa: E402
    BacktestPositionManager,
)

from hf_engine.adapter.broker.broker_utils import get_pip_size

if TYPE_CHECKING:
    from backtest_engine.simulation.portfolio import Portfolio


class MeanReversionZScoreStrategy:
    """
    IndicatorCache-konforme Mean-Reversion-Strategie (Z-Score / Kalman-Z).

    Nutzt ausschließlich vektorisierte Reihen aus dem IndicatorCache, der vom
    EventLoop bereitgestellt und über SymbolDataSlice zugänglich ist.
    """

    def __init__(self, symbol: str, timeframe: str, **kwargs: Any) -> None:
        """
        Initialize the backtest strategy with parameters.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe for the strategy.
            **kwargs: Additional strategy parameters.
        """
        self.symbol: str = symbol
        self.timeframe: str = timeframe

        # Parameter
        self.pip_size: float = get_pip_size(symbol)
        self.atr_length: int = int(kwargs.get("atr_length", 14))
        self.atr_mult: float = float(kwargs.get("atr_mult", 2.0))
        self.b_b_length: int = int(kwargs.get("b_b_length", 20))
        self.std_factor: float = float(kwargs.get("std_factor", 2.0))
        self.window_length: int = int(kwargs.get("window_length", 100))
        self.z_score_long: float = float(kwargs.get("z_score_long", -2.5))
        self.z_score_short: float = float(kwargs.get("z_score_short", 2.5))
        self.ema_length: int = int(kwargs.get("ema_length", 30))
        self.kalman_r: float = float(kwargs.get("kalman_r", 0.01))
        self.kalman_q: float = float(kwargs.get("kalman_q", 1.0))
        self.tp_min_distance: float = float(kwargs.get("tp_min_distance", 1.0))
        # --- Szenario-/Richtungs- und Positionsmanager-Steuerung (Backtest) ---
        # Globaler Richtungsfilter: "long" | "short" | "both"
        self.direction_filter: str = (
            str(kwargs.get("direction_filter", "both")).strip().lower()
        )
        # Szenario-Whitelist (1–6); leer/None => alle Szenarien aktiv
        raw_enabled = kwargs.get("enabled_scenarios")
        enabled: Set[int] = set()
        if isinstance(raw_enabled, (list, tuple, set)):
            for v in raw_enabled:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if 1 <= iv <= 6:
                    enabled.add(iv)
        if not enabled:
            enabled = {1, 2, 3, 4, 5, 6}
        self.enabled_scenarios: Set[int] = enabled

        # Abgeleitete Labels (z. B. "long_2", "short_5") aus Szenario+Richtung
        dirs: List[str]
        if self.direction_filter in ("long", "short"):
            dirs = [self.direction_filter]
        else:
            dirs = ["long", "short"]
        self.allowed_labels: Set[str] = {
            f"{d}_{n}" for n in self.enabled_scenarios for d in dirs
        }

        # Positionsmanager im Backtest per Config schaltbar
        self.use_position_manager: bool = bool(kwargs.get("use_position_manager", True))
        self.position_manager: BacktestPositionManager | None = (
            BacktestPositionManager(
                max_holding_minutes=kwargs.get("max_holding_minutes", 30),
            )
            if self.use_position_manager
            else None
        )

        # =========================
        # HTF-EMA-Filter: 2 Ebenen
        # =========================
        # --- Filter A (immer D1) -> rückwärtskompatibel zu bisherigen Keys ---
        self.htfA_tf = str(kwargs.get("htf_tf", "D1")).upper()  # bestehender Key
        self.htfA_ema = int(kwargs.get("htf_ema", 100))  # bestehender Key
        self.htfA_filter = str(
            kwargs.get("htf_filter", "both")
        ).lower()  # "above"/"below"/"both"/"none"

        # --- Filter B (zusätzlich, abhängig vom Primary-TF) ---
        # extra_htf_tf = "AUTO" | "H1" | "H4" | "NONE"
        raw_extra_tf = str(kwargs.get("extra_htf_tf", "AUTO")).upper()

        def _auto_extra_htf(primary: str) -> str:
            if primary == "M5":
                return "H1"
            if primary in ("M15", "M30"):
                return "H4"
            return "NONE"

        self.htfB_tf = (
            _auto_extra_htf(self.timeframe) if raw_extra_tf == "AUTO" else raw_extra_tf
        )
        self.htfB_ema = int(kwargs.get("extra_htf_ema", 50))
        self.htfB_filter = str(kwargs.get("extra_htf_filter", "both")).lower()

        # News-Filter (UTC-Timestamps werden in der Engine bereits geführt)
        default_csv = (
            Path(__file__).parents[4] / "data" / "news" / "news_calender_history.csv"
        )
        self.news_filter: NewsFilter = NewsFilter(default_csv)

        # Portfolio wird vom StrategyWrapper gesetzt
        self.portfolio: Portfolio | None = None

        # --- GARCH Parameter (für Szenario 4) ---
        self.garch_alpha: float = float(kwargs.get("garch_alpha", 0.05))
        self.garch_beta: float = float(kwargs.get("garch_beta", 0.90))
        self.garch_omega: float | None = kwargs.get("garch_omega", None)
        self.garch_use_log_returns: bool = bool(
            kwargs.get("garch_use_log_returns", True)
        )
        self.garch_scale: float = float(kwargs.get("garch_scale", 100.0))
        self.garch_min_periods: int = int(kwargs.get("garch_min_periods", 50))
        self.garch_sigma_floor: float = float(kwargs.get("garch_sigma_floor", 1e-6))

        # Optional: Lokales Fenster für Kalman+GARCH‑Z (Szenario 4)
        # Wenn > 0 gesetzt, wird Z nur aus den letzten N Bars am Index berechnet,
        # um Pfadabhängigkeit (Warmup) zu vermeiden.
        self.local_z_lookback: int = int(kwargs.get("local_z_lookback", 0))

        # Intraday Volatilitäts-Cluster (Szenario 5)
        self.intraday_vol_feature: str = str(
            kwargs.get("intraday_vol_feature", "garch_forecast")
        ).lower()
        self.intraday_vol_cluster_window: int = int(
            kwargs.get(
                "intraday_vol_cluster_window",
                self._default_cluster_window(self.timeframe),
            )
        )
        self.intraday_vol_cluster_k: int = int(kwargs.get("intraday_vol_cluster_k", 3))
        self.intraday_vol_log_transform: bool = bool(
            kwargs.get("intraday_vol_log_transform", True)
        )
        self.intraday_vol_min_points: int = int(
            kwargs.get("intraday_vol_min_points", 60)
        )
        self.intraday_vol_garch_lookback: int = int(
            kwargs.get("intraday_vol_garch_lookback", 500)
        )
        if self.intraday_vol_garch_lookback <= 0:
            self.intraday_vol_garch_lookback = 500
        self.cluster_hysteresis_bars: int = int(
            kwargs.get("cluster_hysteresis_bars", 0)
        )
        allowed_default: List[str] = ["low", "mid"]
        self.intraday_vol_allowed: List[str] = kwargs.get(
            "intraday_vol_allowed", allowed_default
        )
        # Cache last computed volatility cluster state so long/short reuse the same work per bar
        self._vol_cluster_cache: Dict[str, Any] = {}
        # Cache für lokalen Z‑Score (einfacher Per‑Bar Cache)
        self._local_z_cache: Dict[str, Any] = {}

        # ------------- Szenario 6 (Multi-TF Overlay auf Basis Szenario 2) -------------
        scenario6_mode = str(kwargs.get("scenario6_mode", "all")).strip().lower()
        self.scenario6_mode: str = (
            scenario6_mode if scenario6_mode in ("all", "any") else "all"
        )

        raw_tfs = kwargs.get("scenario6_timeframes", [])
        if isinstance(raw_tfs, str):
            raw_tfs = [raw_tfs]
        self.scenario6_timeframes: List[str] = []
        for tf in raw_tfs or []:
            tf_norm = str(tf or "").strip().upper()
            if not tf_norm:
                continue
            if tf_norm == "PRIMARY":
                tf_norm = str(self.timeframe).upper()
            if tf_norm not in self.scenario6_timeframes:
                self.scenario6_timeframes.append(tf_norm)

        raw_params = kwargs.get("scenario6_params", {}) or {}
        scenario6_params: Dict[str, Any] = {}
        for tf, overrides in raw_params.items():
            tf_key = str(tf or "").strip().upper()
            if not tf_key:
                continue
            if tf_key == "PRIMARY":
                tf_key = str(self.timeframe).upper()
            scenario6_params[tf_key] = overrides
        # Optional: Flattened Key Overrides (e.g. scenario6_M30_long_window_length)
        # This allows the optimizer to tune per‑TF parameters without passing a nested dict.
        # Pattern: scenario6_<TF>_(long|short)_(window_length|b_b_length|std_factor|z_score|z_score_long|z_score_short|kalman_r|kalman_q)
        flat_pattern = re.compile(
            r"^scenario6_([A-Za-z0-9]+)_(long|short)_(window_length|b_b_length|std_factor|z_score|z_score_long|z_score_short|kalman_r|kalman_q)$",
            re.IGNORECASE,
        )
        flat_overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for key, val in kwargs.items():
            m = flat_pattern.match(str(key))
            if not m:
                continue
            tf_raw, dir_raw, p_raw = m.groups()
            tf_key = self._scenario6_normalize_tf(tf_raw)
            dir_key = "long" if str(dir_raw).lower() == "long" else "short"
            p_key = p_raw.lower()
            # normalize param keys used internally
            if p_key in ("z_score",):
                # symmetric override; handled later by _scenario6_params_for
                norm_key = "z_score"
            elif p_key in ("z_score_long", "z_score_short"):
                norm_key = p_key
            elif p_key in (
                "window_length",
                "b_b_length",
                "std_factor",
                "kalman_r",
                "kalman_q",
            ):
                norm_key = p_key
            else:
                continue
            flat_overrides.setdefault(tf_key, {}).setdefault(dir_key, {})[
                norm_key
            ] = val
        # Merge flattened overrides into scenario6_params (takes precedence)
        if flat_overrides:
            for tf_key, by_dir in flat_overrides.items():
                base = scenario6_params.get(tf_key, {})
                if not isinstance(base, dict):
                    base = {}
                for dir_key, ov in by_dir.items():
                    cur = (
                        dict(base.get(dir_key, {}))
                        if isinstance(base.get(dir_key, {}), dict)
                        else {}
                    )
                    cur.update(ov)
                    base[dir_key] = cur
                scenario6_params[tf_key] = base

        self.scenario6_params = scenario6_params

    def _get_local_z(self, symbol_slice, idx: int) -> Optional[float]:
        if not (self.local_z_lookback and self.local_z_lookback > 0):
            return None
        key = (
            self.timeframe,
            idx,
            float(self.kalman_r),
            float(self.kalman_q),
            float(self.garch_alpha),
            float(self.garch_beta),
            float(self.garch_omega if self.garch_omega is not None else -1.0),
            bool(self.garch_use_log_returns),
            float(self.garch_scale),
            int(self.garch_min_periods),
            float(self.garch_sigma_floor),
            int(self.local_z_lookback),
        )
        cache = self._local_z_cache
        if cache.get("key") == key:
            return cache.get("val")
        try:
            z_now = symbol_slice.indicators.kalman_garch_zscore_local(
                self.timeframe,
                "bid",
                idx,
                lookback=int(self.local_z_lookback),
                R=self.kalman_r,
                Q=self.kalman_q,
                alpha=self.garch_alpha,
                beta=self.garch_beta,
                omega=self.garch_omega,
                use_log_returns=self.garch_use_log_returns,
                scale=self.garch_scale,
                min_periods=self.garch_min_periods,
                sigma_floor=self.garch_sigma_floor,
            )
        except Exception:
            z_now = None
        self._local_z_cache = {"key": key, "val": z_now}
        return z_now

    # --- Szenario-/Richtungs-/Label-Helfer ---------------------------------
    def _is_scenario_enabled(self, n: int) -> bool:
        try:
            return int(n) in self.enabled_scenarios
        except Exception:
            return False

    def _is_direction_allowed(self, direction: str) -> bool:
        d = str(direction or "").strip().lower()
        if d not in ("long", "short"):
            return True
        if self.direction_filter in ("long", "short"):
            return self.direction_filter == d
        return True

    def _make_label(self, n: int, direction: str) -> str:
        d = str(direction or "").strip().lower()
        if d not in ("long", "short"):
            d = d or "unknown"
        try:
            n_int = int(n)
        except Exception:
            n_int = n
        return f"{d}_{n_int}"

    def _is_label_allowed(self, label: str) -> bool:
        return str(label or "") in self.allowed_labels

    # --- API erwartet von StrategyWrapper ---------------------------------
    def get_primary_timeframe(self) -> str:
        return self.timeframe

    def on_data(self, slice_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Wird pro Bar vom StrategyWrapper aufgerufen.
        Greift auf den IndicatorCache über den SymbolDataSlice zu.
        """
        symbol_slice = slice_map.get(self.symbol)
        if symbol_slice is None:
            return None

        # aktuelle Kerzen
        bid_candle = symbol_slice.latest(self.timeframe, price_type="bid")
        ask_candle = symbol_slice.latest(self.timeframe, price_type="ask")
        if bid_candle is None or ask_candle is None:
            return None
        ts = getattr(bid_candle, "timestamp", None)

        # News-Sperre
        if ts is None or not self.news_filter.is_trading_allowed(
            ts, self.symbol, mode="close", tf=self.timeframe
        ):
            return None

        # nur eine Position gleichzeitig pro Symbol
        if self.portfolio is not None:
            if len(self.portfolio.get_open_positions(self.symbol)) >= 1:
                return None

        # Szenario-/Richtungslogik: Reihenfolge 1→6 wie bisher, aber
        # durch enabled_scenarios + direction_filter + Label-Filter steuerbar.
        checks = [
            (1, "short", self._evaluate_short_1),
            (1, "long", self._evaluate_long_1),
            (2, "short", self._evaluate_short_2),
            (2, "long", self._evaluate_long_2),
            (3, "short", self._evaluate_short_3),
            (3, "long", self._evaluate_long_3),
            (4, "short", self._evaluate_short_4),
            (4, "long", self._evaluate_long_4),
            (5, "long", self._evaluate_long_5),
            (5, "short", self._evaluate_short_5),
            (6, "long", self._evaluate_long_6),
            (6, "short", self._evaluate_short_6),
        ]

        for scen, direction, fn in checks:
            if not self._is_scenario_enabled(scen):
                continue
            if not self._is_direction_allowed(direction):
                continue
            sig = fn(symbol_slice, bid_candle, ask_candle)
            if not sig:
                continue
            label = self._make_label(scen, direction)
            if not self._is_label_allowed(label):
                continue
            sig["scenario"] = label
            meta = sig.setdefault("meta", {})
            meta.setdefault("scenario", label)
            return sig

        return None

    # --- Helfer: bequemer Zugriff auf IndicatorCache -----------------------
    def _idx(self, symbol_slice) -> int:
        # Der EventLoop verschiebt den Index pro Bar auf demselben SymbolDataSlice.
        return int(getattr(symbol_slice, "index", -1))

    def _safe_at(self, series, idx) -> Optional[float]:
        # IndicatorCache stellt .at(series, index) bereit (robust gg. NaN/Bounds).
        try:
            from backtest_engine.core.indicator_cache import (  # import for typing/reference
                IndicatorCache,
            )
        except Exception:
            pass
        try:
            from backtest_engine.core.indicator_cache import IndicatorCache as _IC

            return _IC.at(self, series, idx)  # type: ignore
        except Exception:
            # Fallback, falls direkte Nutzung gewünscht
            if idx < 0 or idx >= len(series):
                return None
            v = series.iloc[idx]
            return float(v) if np.isfinite(v) else None

    # ------------- Intraday-Volatilitätscluster (Szenario 5) -------------

    @staticmethod
    def _default_cluster_window(timeframe: str) -> int:
        tf = (timeframe or "").upper()
        if tf == "M5":
            return 300
        if tf in ("M15",):
            return 200
        if tf in ("M30", "H1"):
            return 240
        return 200

    def _vol_cluster_guard(
        self, symbol_slice, idx: int, direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "feature": self.intraday_vol_feature,
            "window": self.intraday_vol_cluster_window,
            "k": self.intraday_vol_cluster_k,
            "min_points": self.intraday_vol_min_points,
            "log_transform": self.intraday_vol_log_transform,
            "hysteresis_bars": self.cluster_hysteresis_bars,
            "direction": direction,
        }
        if idx < 0:
            meta["status"] = "invalid_index"
            return False, meta
        feature = self.intraday_vol_feature
        window = max(self.intraday_vol_cluster_window, 1)
        k = max(self.intraday_vol_cluster_k, 1)
        min_points = max(self.intraday_vol_min_points, 1)
        state = self._vol_cluster_state(
            symbol_slice,
            idx,
            feature,
            window,
            k,
            min_points,
            self.intraday_vol_log_transform,
        )

        meta["sample_size"] = int(state.get("sample_size", 0))
        status = state.get("status")
        if status and status != "ok":
            meta["status"] = status
            return False, meta

        labels = state["labels"]
        mapping = state["mapping"]
        current_idx = int(labels[-1])
        current_label = mapping.get(current_idx, "unknown")
        meta["label"] = current_label
        meta["sigma"] = state.get("sigma")
        if state.get("centers") is not None:
            meta["centers"] = state["centers"]
        if state.get("centers_log") is not None:
            meta["centers_log"] = state["centers_log"]

        allowed = {str(label).strip().lower() for label in self.intraday_vol_allowed}
        meta["allowed_labels"] = sorted(list(allowed))

        hysteresis_ok = True
        h = max(int(self.cluster_hysteresis_bars), 0)
        if h > 1:
            if len(labels) < h:
                hysteresis_ok = False
            else:
                recent = labels[-h:]
                hysteresis_ok = all(int(lbl) == current_idx for lbl in recent)
        meta["hysteresis_ok"] = hysteresis_ok

        allowed_now = current_label.lower() in allowed
        meta["allowed_now"] = allowed_now

        if not hysteresis_ok:
            meta["status"] = "hysteresis_block"
            return False, meta
        if not allowed_now:
            meta["status"] = "label_block"
            return False, meta

        meta["status"] = "ok"
        return True, meta

    def _vol_cluster_state(
        self,
        symbol_slice,
        idx: int,
        feature: str,
        window: int,
        k: int,
        min_points: int,
        log_transform: bool,
    ) -> Dict[str, Any]:
        cache_key = (
            idx,
            feature,
            window,
            k,
            min_points,
            log_transform,
            self.timeframe,
            float(self.garch_alpha),
            float(self.garch_beta),
            float(self.garch_omega if self.garch_omega is not None else -1.0),
            bool(self.garch_use_log_returns),
            float(self.garch_scale),
            int(self.garch_min_periods),
            float(self.garch_sigma_floor),
            int(self.atr_length),
            int(self.intraday_vol_garch_lookback),
        )
        cache = self._vol_cluster_cache
        if cache.get("key") == cache_key:
            return cache["state"]

        try:
            import pandas as pd
        except Exception:
            state = {"status": "series_unavailable", "sample_size": 0}
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        series = symbol_slice.indicators.vol_cluster_series(
            self.timeframe,
            "bid",
            idx=idx,
            feature=feature,
            atr_length=self.atr_length,
            garch_lookback=self.intraday_vol_garch_lookback,
            garch_alpha=self.garch_alpha,
            garch_beta=self.garch_beta,
            garch_omega=self.garch_omega,
            garch_use_log_returns=self.garch_use_log_returns,
            garch_scale=self.garch_scale,
            garch_min_periods=self.garch_min_periods,
            garch_sigma_floor=self.garch_sigma_floor,
        )
        if series is None:
            state = {"status": "series_unavailable", "sample_size": 0}
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        try:
            ser_full = series if isinstance(series, pd.Series) else pd.Series(series)
            # idx ist der Bar‑Index (0‑basiert). Für das Cluster‑Fenster soll
            # die aktuelle Bar inklusive sein, daher clampen wir idx in
            # [0, len(ser_full)-1] und schneiden bis idx_int + 1.
            if len(ser_full) == 0:
                ser = ser_full
            else:
                idx_int = min(max(int(idx), 0), len(ser_full) - 1)
                cutoff = idx_int + 1
                ser = ser_full.iloc[:cutoff]
        except Exception:
            state = {"status": "series_unavailable", "sample_size": 0}
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        ser = ser.dropna()
        sample_size = int(len(ser))
        if sample_size < min_points:
            state = {"status": "insufficient_points", "sample_size": sample_size}
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        tail = ser.tail(window)
        if len(tail) < k:
            state = {
                "status": "insufficient_unique",
                "sample_size": sample_size,
            }
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        values = tail.to_numpy(dtype=float)
        if log_transform:
            values = np.log(np.clip(values, 1e-12, None))

        clusters = self._kmeans_1d(values, k)
        if clusters is None:
            state = {"status": "clustering_failed", "sample_size": sample_size}
            self._vol_cluster_cache = {"key": cache_key, "state": state}
            return state

        centers, labels = clusters
        order = np.argsort(centers)
        label_names = ["low", "mid", "high", "very_high", "extreme"]
        mapping: Dict[int, str] = {}
        for rank, idx_c in enumerate(order):
            if rank < len(label_names):
                mapping[idx_c] = label_names[rank]
            else:
                mapping[idx_c] = f"cluster_{rank}"

        if log_transform:
            centers_sorted = [float(np.exp(centers[i])) for i in order]
            centers_log = [float(centers[i]) for i in order]
        else:
            centers_sorted = [float(centers[i]) for i in order]
            centers_log = None

        sigma = float(tail.iloc[-1]) if len(tail) else float("nan")
        state = {
            "status": "ok",
            "sample_size": sample_size,
            "labels": labels.copy(),
            "mapping": mapping,
            "centers": centers_sorted,
            "centers_log": centers_log,
            "sigma": sigma,
        }
        self._vol_cluster_cache = {"key": cache_key, "state": state}
        return state

    @staticmethod
    def _kmeans_1d(
        values: np.ndarray, k: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if values.size == 0:
            return None
        unique_vals = np.unique(values)
        k = max(1, min(k, unique_vals.size))
        if k == 1:
            center = np.array([float(np.mean(values))], dtype=float)
            labels = np.zeros(values.shape[0], dtype=int)
            return center, labels

        quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
        init = np.quantile(values, quantiles)
        centers = init.astype(float)
        for _ in range(30):
            distances = np.abs(values[:, None] - centers[None, :])
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers[j] = float(values[mask].mean())
            if np.allclose(new_centers, centers, atol=1e-6, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        else:
            distances = np.abs(values[:, None] - centers[None, :])
            labels = distances.argmin(axis=1)
        return centers, labels

    # ------------- HTF-Bias Helpers -------------

    def _htf_bias_series(self, symbol_slice, tf: str, ema_len: int, idx: int):
        """
        Liefert +1 wenn HTF close > HTF EMA (bullish), -1 wenn close < EMA (bearish), 0 sonst/NaN.
        Nutzt aligned HTF-Daten aus dem IndicatorCache → Look-Ahead-safe.
        """
        if tf in (None, "", "NONE"):
            # neutraler Bias (immer true)
            return None  # bedeutet: nicht anwenden

        ind = symbol_slice.indicators
        close_series = ind.get_closes(tf, "bid")
        ema = getattr(ind, "ema_stepwise")(tf, "bid", ema_len)
        c_now = self._safe_at(close_series, idx)
        e_now = self._safe_at(ema, idx)
        if c_now is None or e_now is None or not np.isfinite(e_now):
            return None
        if c_now > e_now:
            return +1
        if c_now < e_now:
            return -1
        return 0  # exakt auf EMA → neutral

    def _bias_label(self, bias: Optional[int]) -> str:
        """Hilfsfunktion für Tagging: 'above' | 'below' | 'neutral' | 'na'."""
        if bias is None:
            return "na"
        if bias > 0:
            return "above"
        if bias < 0:
            return "below"
        return "neutral"

    def _passes_filter(self, bias: Optional[int], mode: str) -> bool:
        """
        mode: "above" | "below" | "both" | "none"
        Richtungsunabhängig:
          - "above":   require bias > 0
          - "below":   require bias < 0
          - "both":    allow both (nur Tagging)
          - "none":    kein Filter
        """
        if mode == "none" or mode == "both":
            return True
        if bias is None:
            return False
        if mode == "above":
            return bias > 0
        if mode == "below":
            return bias < 0
        return True

    def _passes_htf_for_long(self, symbol_slice, idx: int):
        """Gibt (ok: bool, labels: dict) zurück, inkl. Bias-Labels fürs Tagging."""
        biasA = self._htf_bias_series(symbol_slice, self.htfA_tf, self.htfA_ema, idx)
        biasB = self._htf_bias_series(symbol_slice, self.htfB_tf, self.htfB_ema, idx)

        okA = self._passes_filter(biasA, self.htfA_filter)
        okB = self._passes_filter(biasB, self.htfB_filter)

        labels = {"biasA": self._bias_label(biasA), "biasB": self._bias_label(biasB)}

        return (okA and okB), labels

    def _passes_htf_for_short(self, symbol_slice, idx: int):
        biasA = self._htf_bias_series(symbol_slice, self.htfA_tf, self.htfA_ema, idx)
        biasB = self._htf_bias_series(symbol_slice, self.htfB_tf, self.htfB_ema, idx)

        okA = self._passes_filter(biasA, self.htfA_filter)
        okB = self._passes_filter(biasB, self.htfB_filter)

        labels = {"biasA": self._bias_label(biasA), "biasB": self._bias_label(biasB)}

        return (okA and okB), labels

    def _htf_meta_snapshot(self, symbol_slice, idx: int) -> dict:
        """Erfasst HTF-EMA-Werte (stepwise) + Bias-Labels für A/B."""
        ind = symbol_slice.indicators
        out = {"A": None, "B": None}
        # A
        try:
            emaA_series = getattr(ind, "ema_stepwise")(
                self.htfA_tf, "bid", self.htfA_ema
            )
            emaA = self._safe_at(emaA_series, idx)
            closeA_series = ind.get_closes(self.htfA_tf, "bid")
            closeA = self._safe_at(closeA_series, idx)
        except Exception:
            emaA = None
            closeA = None
        biasA_val = self._htf_bias_series(
            symbol_slice, self.htfA_tf, self.htfA_ema, idx
        )
        out["A"] = {
            "tf": self.htfA_tf,
            "ema_len": self.htfA_ema,
            "ema": emaA,
            "close_price": closeA,
            "bias_label": self._bias_label(biasA_val),
        }
        # B
        try:
            emaB_series = (
                getattr(ind, "ema_stepwise")(self.htfB_tf, "bid", self.htfB_ema)
                if self.htfB_tf not in (None, "", "NONE")
                else None
            )
            emaB = self._safe_at(emaB_series, idx) if emaB_series is not None else None
            closeB_series = ind.get_closes(self.htfB_tf, "bid")
            closeB = self._safe_at(closeB_series, idx)
        except Exception:
            emaB = None
            closeB = None
        biasB_val = self._htf_bias_series(
            symbol_slice, self.htfB_tf, self.htfB_ema, idx
        )
        out["B"] = {
            "tf": self.htfB_tf,
            "ema_len": self.htfB_ema,
            "ema": emaB,
            "close_price": closeB,
            "bias_label": self._bias_label(biasB_val),
        }
        return out

    # --- Szenario 6 Helper -------------------------------------------------

    def _scenario6_normalize_tf(self, tf: str) -> str:
        tf_norm = str(tf or "").strip().upper()
        if not tf_norm:
            return ""
        if tf_norm == "PRIMARY":
            return str(self.timeframe).upper()
        return tf_norm

    def _scenario6_params_for(self, tf: str, direction: str) -> Dict[str, Any]:
        """Build per-timeframe parameter set for Scenario 6 checks."""
        tf_norm = self._scenario6_normalize_tf(tf)
        dir_key = "long" if direction == "long" else "short"
        overrides = self.scenario6_params.get(tf_norm, {})

        if (
            isinstance(overrides, dict)
            and dir_key in overrides
            and isinstance(overrides[dir_key], dict)
        ):
            dir_overrides = overrides[dir_key]
        else:
            dir_overrides = overrides if isinstance(overrides, dict) else {}

        def _cast_int(key: str, default: int) -> int:
            val = dir_overrides.get(key, default)
            try:
                return int(val)
            except Exception:
                return int(default)

        def _cast_float(key: str, default: float) -> float:
            val = dir_overrides.get(key, default)
            try:
                return float(val)
            except Exception:
                return float(default)

        params = {
            "window_length": _cast_int("window_length", self.window_length),
            "b_b_length": _cast_int("b_b_length", self.b_b_length),
            "std_factor": _cast_float("std_factor", self.std_factor),
            "kalman_r": _cast_float("kalman_r", self.kalman_r),
            "kalman_q": _cast_float("kalman_q", self.kalman_q),
            "z_score_long": _cast_float(
                "z_score_long", dir_overrides.get("z_score", self.z_score_long)
            ),
            "z_score_short": _cast_float(
                "z_score_short", dir_overrides.get("z_score", self.z_score_short)
            ),
        }
        # Allow direct overrides for symmetric z_score usage
        if "z_score" in dir_overrides:
            params["z_score_long"] = params["z_score_short"] = _cast_float(
                "z_score", dir_overrides["z_score"]
            )
        return params

    def _scenario6_check_tf(
        self, symbol_slice, tf: str, idx: int, direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        tf_norm = self._scenario6_normalize_tf(tf)
        meta: Dict[str, Any] = {"tf": tf_norm, "direction": direction}

        if not tf_norm:
            meta["status"] = "invalid_timeframe"
            return False, meta

        ind = symbol_slice.indicators
        closes = ind.get_closes(tf_norm, "bid")
        price_now = self._safe_at(closes, idx)
        if price_now is None or not np.isfinite(price_now):
            meta["status"] = "no_price"
            return False, meta

        params = self._scenario6_params_for(tf_norm, direction)
        meta["params"] = {k: params[k] for k in params}

        try:
            z_series = ind.kalman_zscore_stepwise(
                tf_norm,
                "bid",
                window=int(params["window_length"]),
                R=float(params["kalman_r"]),
                Q=float(params["kalman_q"]),
            )
            z_now = self._safe_at(z_series, idx)
        except Exception:
            z_now = None
        if z_now is None or not np.isfinite(z_now):
            meta["status"] = "no_zscore"
            return False, meta

        meta["z"] = float(z_now)

        try:
            upper, mid, lower = ind.bollinger_stepwise(
                tf_norm,
                "bid",
                period=int(params["b_b_length"]),
                std_factor=float(params["std_factor"]),
            )
        except Exception:
            upper = mid = lower = None

        threshold = (
            float(params["z_score_long"])
            if direction == "long"
            else float(params["z_score_short"])
        )

        if direction == "long":
            lower_now = self._safe_at(lower, idx) if lower is not None else None
            meta["lower"] = float(lower_now) if lower_now is not None else None
            if z_now > threshold:
                meta["status"] = "z_above_threshold"
                meta["threshold"] = threshold
                return False, meta
            if lower_now is None or not np.isfinite(lower_now):
                meta["status"] = "no_lower_band"
                return False, meta
            if price_now > lower_now:
                meta["status"] = "price_above_lower"
                meta["threshold"] = threshold
                meta["price"] = float(price_now)
                return False, meta
        else:
            upper_now = self._safe_at(upper, idx) if upper is not None else None
            meta["upper"] = float(upper_now) if upper_now is not None else None
            if z_now < threshold:
                meta["status"] = "z_below_threshold"
                meta["threshold"] = threshold
                return False, meta
            if upper_now is None or not np.isfinite(upper_now):
                meta["status"] = "no_upper_band"
                return False, meta
            if price_now < upper_now:
                meta["status"] = "price_below_upper"
                meta["threshold"] = threshold
                meta["price"] = float(price_now)
                return False, meta

        meta["price"] = float(price_now)
        meta["status"] = "ok"
        return True, meta

    def _scenario6_evaluate_chain(
        self, symbol_slice, idx: int, direction: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []
        successes = 0
        for tf in self.scenario6_timeframes:
            ok, meta = self._scenario6_check_tf(symbol_slice, tf, idx, direction)
            meta["ok"] = bool(ok)
            results.append(meta)
            if ok:
                successes += 1

        if not results:
            return False, results

        if self.scenario6_mode == "any":
            return successes > 0, results
        return successes == len(results), results

    # --- Szenario 1: Z-Score relativ zu EMA --------------------------------
    def _evaluate_long_1(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators  # vom Engine-Lauf gesetzt und wiederverwendet.
        idx = self._idx(symbol_slice)

        # Z-Score (mean_source='ema')
        z = ind.zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            mean_source="ema",
            ema_period=self.ema_length,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now > self.z_score_long:
            return None

        # ATR, EMA
        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        ema = ind.ema(self.timeframe, "bid", self.ema_length)
        ema_now = self._safe_at(ema, idx)

        if atr_now is None or ema_now is None:
            return None

        sl = float(bid_candle.low) - self.atr_mult * atr_now
        tp = float(ema_now)
        return {
            "direction": "long",
            "entry": float(ask_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "zscore_ema_long",
            "tags": ["zscore", "ema", "scenario1"],
        }

    def _evaluate_short_1(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        z = ind.zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            mean_source="ema",
            ema_period=self.ema_length,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now < self.z_score_short:
            return None

        atr = ind.atr(
            self.timeframe, "ask", self.atr_length
        )  # Short: Exit/SL am Ask relevant
        atr_now = self._safe_at(atr, idx)
        ema = ind.ema(self.timeframe, "bid", self.ema_length)
        ema_now = self._safe_at(ema, idx)
        if atr_now is None or ema_now is None:
            return None

        sl = float(bid_candle.high) + self.atr_mult * atr_now
        tp = float(ema_now)
        return {
            "direction": "short",
            "entry": float(bid_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "zscore_ema_short",
            "tags": ["zscore", "ema", "scenario1"],
        }

    # --- Szenario 2: Kalman-Z + Bollinger ---------------------------------
    def _evaluate_long_2(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_long(symbol_slice, idx)
        if not ok_htf:
            return None

        z = ind.kalman_zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            R=self.kalman_r,
            Q=self.kalman_q,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now > self.z_score_long:
            return None

        upper, mid, lower = ind.bollinger(
            self.timeframe, "bid", period=self.b_b_length, std_factor=self.std_factor
        )
        lower_now = self._safe_at(lower, idx)
        mid_now = self._safe_at(mid, idx)
        if lower_now is None or mid_now is None:
            return None
        if float(bid_candle.close) > lower_now:
            return None

        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        if atr_now is None:
            return None

        sl = float(bid_candle.low) - self.atr_mult * atr_now
        tp = float(mid_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "boll": {"upper": None, "mid": mid_now, "lower": lower_now},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "long",
            "entry": float(ask_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_z_boll_long",
            "tags": [
                "kalman",
                "bollinger",
                "scenario2",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        # scenario & tags zusätzlich auch in meta spiegeln (failsafe für Wrapper)
        ret["meta"]["scenario"] = "long_2"
        ret["meta"]["tags"] = ret["tags"]
        return ret

    def _evaluate_short_2(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_short(symbol_slice, idx)
        if not ok_htf:
            return None

        z = ind.kalman_zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            R=self.kalman_r,
            Q=self.kalman_q,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now < self.z_score_short:
            return None

        upper, mid, lower = ind.bollinger(
            self.timeframe, "bid", period=self.b_b_length, std_factor=self.std_factor
        )
        upper_now = self._safe_at(upper, idx)
        mid_now = self._safe_at(mid, idx)
        if upper_now is None or mid_now is None:
            return None
        if float(bid_candle.close) < upper_now:
            return None

        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        if atr_now is None:
            return None

        sl = float(bid_candle.high) + self.atr_mult * atr_now
        tp = float(mid_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "boll": {"upper": upper_now, "mid": mid_now, "lower": None},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "short",
            "entry": float(bid_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_z_boll_short",
            "tags": [
                "kalman",
                "bollinger",
                "scenario2",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        # scenario & tags zusätzlich auch in meta spiegeln (failsafe für Wrapper)
        ret["meta"]["scenario"] = "short_2"
        ret["meta"]["tags"] = ret["tags"]
        return ret

    # --- Szenario 6: Multi-TF Overlay (Szenario 2 Basis) -------------------
    def _evaluate_long_6(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:

        idx = self._idx(symbol_slice)
        ok_htf, _ = self._passes_htf_for_long(symbol_slice, idx)
        if not ok_htf:
            return None

        chain_ok, chain_meta = self._scenario6_evaluate_chain(symbol_slice, idx, "long")
        if not chain_ok:
            return None

        base = self._evaluate_long_2(symbol_slice, bid_candle, ask_candle)
        if not base:
            return None

        ret = copy.deepcopy(base)
        ret["reason"] = "kalman_z_boll_long_multi_tf"
        tags = [t for t in ret.get("tags", []) if t not in {"scenario2", "scenario5"}]
        tags.extend(["scenario6", "multi_tf"])
        for meta in chain_meta:
            tf_tag = str(meta.get("tf", "")).lower()
            if tf_tag:
                tags.append(f"tf_{tf_tag}_{'ok' if meta.get('ok') else 'fail'}")
        # deduplicate while preserving order
        seen = set()
        dedup_tags = []
        for tag in tags:
            if tag not in seen:
                dedup_tags.append(tag)
                seen.add(tag)
        ret["tags"] = dedup_tags

        meta = ret.setdefault("meta", {})
        meta["scenario"] = "long_6"
        meta["tags"] = ret["tags"]
        meta["scenario6"] = {
            "mode": self.scenario6_mode,
            "timeframes": chain_meta,
        }
        return ret

    def _evaluate_short_6(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:

        idx = self._idx(symbol_slice)
        ok_htf, _ = self._passes_htf_for_short(symbol_slice, idx)
        if not ok_htf:
            return None

        chain_ok, chain_meta = self._scenario6_evaluate_chain(
            symbol_slice, idx, "short"
        )
        if not chain_ok:
            return None

        base = self._evaluate_short_2(symbol_slice, bid_candle, ask_candle)
        if not base:
            return None

        ret = copy.deepcopy(base)
        ret["reason"] = "kalman_z_boll_short_multi_tf"
        tags = [t for t in ret.get("tags", []) if t not in {"scenario2", "scenario5"}]
        tags.extend(["scenario6", "multi_tf"])
        for meta in chain_meta:
            tf_tag = str(meta.get("tf", "")).lower()
            if tf_tag:
                tags.append(f"tf_{tf_tag}_{'ok' if meta.get('ok') else 'fail'}")
        seen = set()
        dedup_tags = []
        for tag in tags:
            if tag not in seen:
                dedup_tags.append(tag)
                seen.add(tag)
        ret["tags"] = dedup_tags

        meta = ret.setdefault("meta", {})
        meta["scenario"] = "short_6"
        meta["tags"] = ret["tags"]
        meta["scenario6"] = {
            "mode": self.scenario6_mode,
            "timeframes": chain_meta,
        }
        return ret

    # --- Szenario 5: Szenario 2 + Intraday-Vol-Cluster --------------------
    def _evaluate_long_5(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        base = self._evaluate_long_2(symbol_slice, bid_candle, ask_candle)
        if not base:
            return None

        idx = self._idx(symbol_slice)
        allowed, cluster_meta = self._vol_cluster_guard(symbol_slice, idx, "long")
        if not allowed:
            return None

        ret = copy.deepcopy(base)
        ret["reason"] = "kalman_z_boll_long_vol_cluster"
        tags = [t for t in ret.get("tags", []) if t != "scenario2"]
        if "vol_cluster" not in tags:
            tags.append("vol_cluster")
        if "scenario5" not in tags:
            tags.append("scenario5")
        ret["tags"] = tags
        meta = ret.setdefault("meta", {})
        meta["scenario"] = "long_5"
        meta["tags"] = tags
        meta["vol_cluster"] = cluster_meta
        return ret

    def _evaluate_short_5(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        base = self._evaluate_short_2(symbol_slice, bid_candle, ask_candle)
        if not base:
            return None

        idx = self._idx(symbol_slice)
        allowed, cluster_meta = self._vol_cluster_guard(symbol_slice, idx, "short")
        if not allowed:
            return None

        ret = copy.deepcopy(base)
        ret["reason"] = "kalman_z_boll_short_vol_cluster"
        tags = [t for t in ret.get("tags", []) if t != "scenario2"]
        if "vol_cluster" not in tags:
            tags.append("vol_cluster")
        if "scenario5" not in tags:
            tags.append("scenario5")
        ret["tags"] = tags
        meta = ret.setdefault("meta", {})
        meta["scenario"] = "short_5"
        meta["tags"] = tags
        meta["vol_cluster"] = cluster_meta
        return ret

    # --- Szenario 3: Kalman-Z + Bollinger + EMA Take Profit ---------------------------------
    def _evaluate_long_3(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_long(symbol_slice, idx)
        if not ok_htf:
            return None

        z = ind.kalman_zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            R=self.kalman_r,
            Q=self.kalman_q,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now > self.z_score_long:
            return None

        upper, mid, lower = ind.bollinger(
            self.timeframe, "bid", period=self.b_b_length, std_factor=self.std_factor
        )
        lower_now = self._safe_at(lower, idx)
        ema = ind.ema(self.timeframe, "bid", self.ema_length)
        ema_now = self._safe_at(ema, idx)
        if lower_now is None or ema_now is None:
            return None
        if float(bid_candle.close) > lower_now:
            return None
        if float(ema_now) <= float(ask_candle.close) + self.tp_min_distance:
            return None

        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        if atr_now is None:
            return None

        sl = float(bid_candle.low) - self.atr_mult * atr_now
        tp = float(ema_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "ema": ema_now,
            "boll": {"upper": None, "mid": None, "lower": lower_now},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "long",
            "entry": float(ask_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_z_boll_long",
            "tags": [
                "kalman",
                "bollinger",
                "scenario3",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        # scenario & tags zusätzlich auch in meta spiegeln (failsafe für Wrapper)
        ret["meta"]["scenario"] = "long_3"
        ret["meta"]["tags"] = ret["tags"]
        return ret

    def _evaluate_short_3(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_short(symbol_slice, idx)
        if not ok_htf:
            return None

        z = ind.kalman_zscore(
            self.timeframe,
            "bid",
            window=self.window_length,
            R=self.kalman_r,
            Q=self.kalman_q,
        )
        z_now = self._safe_at(z, idx)
        if z_now is None or not np.isfinite(z_now) or z_now < self.z_score_short:
            return None

        upper, mid, lower = ind.bollinger(
            self.timeframe, "bid", period=self.b_b_length, std_factor=self.std_factor
        )
        upper_now = self._safe_at(upper, idx)
        ema = ind.ema(self.timeframe, "bid", self.ema_length)
        ema_now = self._safe_at(ema, idx)
        if upper_now is None or ema_now is None:
            return None
        if float(bid_candle.close) < upper_now:
            return None
        if float(ema_now) >= float(bid_candle.close) - self.tp_min_distance:
            return None

        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        if atr_now is None:
            return None

        sl = float(bid_candle.high) + self.atr_mult * atr_now
        tp = float(ema_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "ema": ema_now,
            "boll": {"upper": upper_now, "mid": None, "lower": None},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "short",
            "entry": float(bid_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_z_boll_short",
            "tags": [
                "kalman",
                "bollinger",
                "scenario3",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        # scenario & tags zusätzlich auch in meta spiegeln (failsafe für Wrapper)
        ret["meta"]["scenario"] = "short_3"
        ret["meta"]["tags"] = ret["tags"]
        return ret

    # --- Szenario 4: Kalman + GARCH-Z + Bollinger (identisch zu Szenario 2, nur anderer Z) ---
    def _evaluate_long_4(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_long(symbol_slice, idx)
        if not ok_htf:
            return None

        if self.local_z_lookback and self.local_z_lookback > 0:
            # Günstige Checks zuerst: Bollinger + Preisrelation, dann teuren lokalen Z berechnen
            upper, mid, lower = ind.bollinger(
                self.timeframe,
                "bid",
                period=self.b_b_length,
                std_factor=self.std_factor,
            )
            lower_now = self._safe_at(lower, idx)
            mid_now = self._safe_at(mid, idx)
            if lower_now is None or mid_now is None:
                return None
            if float(bid_candle.close) > lower_now:
                return None

            atr = ind.atr(self.timeframe, "bid", self.atr_length)
            atr_now = self._safe_at(atr, idx)
            if atr_now is None:
                return None
            z_now = self._get_local_z(symbol_slice, idx)
            if z_now is None or not np.isfinite(z_now) or z_now > self.z_score_long:
                return None
        else:
            # Günstige Checks zuerst (Band + Preis), dann Z-Serie lesen
            upper, mid, lower = ind.bollinger(
                self.timeframe,
                "bid",
                period=self.b_b_length,
                std_factor=self.std_factor,
            )
            lower_now = self._safe_at(lower, idx)
            mid_now = self._safe_at(mid, idx)
            if lower_now is None or mid_now is None:
                return None
            if float(bid_candle.close) > lower_now:
                return None

            atr = ind.atr(self.timeframe, "bid", self.atr_length)
            atr_now = self._safe_at(atr, idx)
            if atr_now is None:
                return None

            z = ind.kalman_garch_zscore(
                self.timeframe,
                "bid",
                R=self.kalman_r,
                Q=self.kalman_q,
                alpha=self.garch_alpha,
                beta=self.garch_beta,
                omega=self.garch_omega,
                use_log_returns=self.garch_use_log_returns,
                scale=self.garch_scale,
                min_periods=self.garch_min_periods,
                sigma_floor=self.garch_sigma_floor,
            )
            z_now = self._safe_at(z, idx)
            if z_now is None or not np.isfinite(z_now) or z_now > self.z_score_long:
                return None

        sl = float(bid_candle.low) - self.atr_mult * atr_now
        tp = float(mid_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "boll": {"upper": None, "mid": mid_now, "lower": lower_now},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "long",
            "entry": float(ask_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_garch_z_boll_long",
            "tags": [
                "kalman",
                "garch",
                "bollinger",
                "scenario4",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        ret["meta"]["scenario"] = "long_4"
        ret["meta"]["tags"] = ret["tags"]
        return ret

    def _evaluate_short_4(
        self, symbol_slice, bid_candle, ask_candle
    ) -> Optional[Dict[str, Any]]:
        ind = symbol_slice.indicators
        idx = self._idx(symbol_slice)

        # ✅ HTF-Filter: richtungsunabhängig; "both" lässt beide zu, wir taggen sie
        ok_htf, htf_labels = self._passes_htf_for_short(symbol_slice, idx)
        if not ok_htf:
            return None

        if self.local_z_lookback and self.local_z_lookback > 0:
            # Günstige Checks zuerst: Bollinger + Preisrelation, dann teuren lokalen Z berechnen
            upper, mid, lower = ind.bollinger(
                self.timeframe,
                "bid",
                period=self.b_b_length,
                std_factor=self.std_factor,
            )
            upper_now = self._safe_at(upper, idx)
            mid_now = self._safe_at(mid, idx)
            if upper_now is None or mid_now is None:
                return None
            if float(bid_candle.close) < upper_now:
                return None

            atr = ind.atr(self.timeframe, "bid", self.atr_length)
            atr_now = self._safe_at(atr, idx)
            if atr_now is None:
                return None
            z_now = self._get_local_z(symbol_slice, idx)
            if z_now is None or not np.isfinite(z_now) or z_now < self.z_score_short:
                return None
        else:
            # Günstige Checks zuerst (Band + Preis), dann Z-Serie lesen
            upper, mid, lower = ind.bollinger(
                self.timeframe,
                "bid",
                period=self.b_b_length,
                std_factor=self.std_factor,
            )
            upper_now = self._safe_at(upper, idx)
            mid_now = self._safe_at(mid, idx)
            if upper_now is None or mid_now is None:
                return None
            if float(bid_candle.close) < upper_now:
                return None

            atr = ind.atr(self.timeframe, "bid", self.atr_length)
            atr_now = self._safe_at(atr, idx)
            if atr_now is None:
                return None

            z = ind.kalman_garch_zscore(
                self.timeframe,
                "bid",
                R=self.kalman_r,
                Q=self.kalman_q,
                alpha=self.garch_alpha,
                beta=self.garch_beta,
                omega=self.garch_omega,
                use_log_returns=self.garch_use_log_returns,
                scale=self.garch_scale,
                min_periods=self.garch_min_periods,
                sigma_floor=self.garch_sigma_floor,
            )
            z_now = self._safe_at(z, idx)
            if z_now is None or not np.isfinite(z_now) or z_now < self.z_score_short:
                return None

        atr = ind.atr(self.timeframe, "bid", self.atr_length)
        atr_now = self._safe_at(atr, idx)
        if atr_now is None:
            return None

        sl = float(bid_candle.high) + self.atr_mult * atr_now
        tp = float(mid_now)
        htf_meta = self._htf_meta_snapshot(symbol_slice, idx)
        spread = float(ask_candle.close) - float(bid_candle.close)
        meta = {
            "z": z_now,
            "atr": atr_now,
            "boll": {"upper": upper_now, "mid": mid_now, "lower": None},
            "htf": htf_meta,
            "prices": {
                "bid_close": float(bid_candle.close),
                "ask_close": float(ask_candle.close),
                "spread": spread,
            },
        }
        ret = {
            "direction": "short",
            "entry": float(bid_candle.close),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "symbol": self.symbol,
            "type": "market",
            "reason": "kalman_garch_z_boll_short",
            "tags": [
                "kalman",
                "garch",
                "bollinger",
                "scenario4",
                f"htf_{self.htfA_tf}_ema{self.htfA_ema}",
                f"biasA_{htf_labels['biasA']}",
                f"biasB_{htf_labels['biasB']}",
            ],
            "meta": meta,
        }
        ret["meta"]["scenario"] = "short_4"
        ret["meta"]["tags"] = ret["tags"]
        return ret


if __name__ == "__main__":
    import json

    from backtest_engine.runner import run_backtest

    from hf_engine.infra.config.paths import BACKTEST_CONFIG_DIR

    config_path = BACKTEST_CONFIG_DIR / "mean_reversion_z_score.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    run_backtest(config)
