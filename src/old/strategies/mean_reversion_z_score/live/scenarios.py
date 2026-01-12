# hf_engine/strategies/mean_reversion_z_score/live/scenarios.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hf_engine.adapter.broker.broker_utils import (
    calculate_atr,
    calculate_atr_series,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_garch_volatility,
    calculate_kalman_garch_zscore,
    calculate_kalman_zscore,
    calculate_vol_cluster_state,
    calculate_zscore,
    get_pip_size,
)
from hf_engine.infra.logging.log_service import log_service
from strategies._base.base_scenarios import BaseSzenario
from strategies.mean_reversion_z_score.live.utils import (
    signal_long_market,
    signal_short_market,
)


class SzenarioEvaluator(BaseSzenario):
    """
    Bewertet Marktszenarien und gibt bei Eintreffen ein standardisiertes Signal-Dict zurück.
    Berechnet Indikatoren getrennt für Long/Short und cached sie pro (symbol, timeframe, last_ts).
    """

    def __init__(self, config: Dict[str, Any], data_provider: Any) -> None:
        self.config = config
        self.data = data_provider
        # Cache: key=(symbol,timeframe) -> {"last_ts": int, "sig_long": tuple, "sig_short": tuple,
        #                                   "ind_long": {...}, "ind_short": {...}}
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Optional: Cache für Intraday‑Volatilitäts‑Cluster (Szenario 5)
        self._vol_cluster_cache: Dict[str, Any] = {}

    def name(self) -> str:
        return str(self.config.get("strategy_name", "Mean_Reversion_Z_Score"))

    # ---- Direction-Mode (globaler Richtungsfilter) ------------------------
    def _dir_mode(self) -> str:
        """
        Globaler Richtungsfilter aus der CONFIG:
          - 'long'  : nur Long-Setups evaluieren
          - 'short' : nur Short-Setups evaluieren
          - 'both'  : Long & Short (Default, falls nicht gesetzt)
        """
        v = str(self.config.get("direction_filter", "both")).strip().lower()
        if v in ("long", "short"):
            return v
        return "both"

    def _is_allowed(self, direction: str) -> bool:
        mode = self._dir_mode()
        if mode == "both":
            return True
        d = (direction or "").lower()
        return mode == d

    def _get_daily_trend_ok(
        self, symbol: str, current_close: float, ema_period: int, relation: str
    ) -> bool:
        """
        Trend-Filter: Daily EMA(N) vs. Schlusskurs der zuletzt GESCHLOSSENEN D1‑Kerze.
        relation ∈ {"above","below","any"}:
          - "above":  nur wenn D1‑Close > EMA(N)
          - "below":  nur wenn D1‑Close < EMA(N)
          - "any":    Filter deaktiviert (immer True)

        Hinweis: 'current_close' wird hier NICHT verwendet; maßgeblich ist der
        Schlusskurs der HTF‑Kerze (D1), konsistent mit dem Backtest.
        """
        # Weiterleitung auf generische HTF-Prüfung (nutzt den HTF‑Close intern)
        return self._get_trend_ok(symbol, "D1", ema_period, relation)

    def _get_trend_ok(
        self,
        symbol: str,
        timeframe: str,
        ema_period: int,
        relation: str,
    ) -> bool:
        """
        Generische Trend-Filter-Prüfung für ein gewünschtes HTF (D1/H4/H1 ...):
          - Berechnet EMA(N) auf HTF‑Closes
          - Vergleicht den SCHLUSSKURS der zuletzt GESCHLOSSENEN HTF‑Kerze
            mit dem EMA(N) derselben HTF (wie im Backtest)
          - relation ∈ {"above","below","any","both","none"}
            • "above": HTF‑Close > EMA(N)
            • "below": HTF‑Close < EMA(N)
            • "any"/"both"/"none": Filter deaktiviert (immer True)

        Hinweis: Der Parameter 'current_close' wird ignoriert; maßgeblich ist
        ausschließlich der letzte geschlossene Close der angegebenen HTF.
        """
        try:
            candles = self.data.get_ohlc_series(symbol, timeframe, 500)
            df_htf = self._df_from_candles(candles)
        except Exception:
            return False
        closes = df_htf["close"].astype(float)
        if closes is None or len(closes) == 0:
            return False
        # Letzter geschlossener HTF‑Close
        htf_close = float(closes.iloc[-1])
        # EMA auf HTF berechnen und letzten Wert nehmen
        emaN = calculate_ema(closes.tolist(), int(max(1, ema_period)))
        if emaN is None or len(emaN) == 0:
            return False
        ema_val = float(emaN[-1])
        rel = (str(relation) or "any").lower()
        if rel in ("any", "both", "none"):
            return True
        if rel == "above":
            return htf_close > ema_val
        if rel == "below":
            return htf_close < ema_val
        # unbekannter Wert -> vorsichtig False
        return False

    def _multi_trend_ok(self, symbol: str, params: Dict[str, Any]) -> bool:
        """
        Prüft Daily/H4/H1 Trendfilter gemeinsam. Welche Filter aktiv sind, wird
        über ihre relation-Parameter gesteuert ("any"/"both"/"none" → deaktiviert).
        Wieviele Filter erfüllt sein müssen, wird über 'trend_min_filters_required'
        in params (oder global) bestimmt. Standard: alle aktiven müssen bestehen.

        Wichtig: Für jeden aktiven HTF‑Filter wird der Schlusskurs der zuletzt
        geschlossenen Kerze der jeweiligen HTF mit dem EMA derselben HTF
        verglichen (nicht der aktuelle LTF‑Preis).
        """
        # Daily
        d_rel = str(params.get("daily_trend_relation", "any"))
        d_ema = int(
            params.get(
                "daily_trend_ema_period", self.config.get("daily_trend_ema_period", 50)
            )
        )
        # H4
        h4_rel = str(params.get("h4_trend_relation", "any"))
        h4_ema = int(
            params.get(
                "h4_trend_ema_period", self.config.get("h4_trend_ema_period", 50)
            )
        )
        # H1
        h1_rel = str(params.get("h1_trend_relation", "any"))
        h1_ema = int(
            params.get(
                "h1_trend_ema_period", self.config.get("h1_trend_ema_period", 50)
            )
        )

        checks = []  # True/False je aktiver Filter
        # Daily aktiv?
        if d_rel.lower() not in ("any", "both", "none"):
            checks.append(self._get_trend_ok(symbol, "D1", d_ema, d_rel))
        # H4 aktiv?
        if h4_rel.lower() not in ("any", "both", "none"):
            checks.append(self._get_trend_ok(symbol, "H4", h4_ema, h4_rel))
        # H1 aktiv?
        if h1_rel.lower() not in ("any", "both", "none"):
            checks.append(self._get_trend_ok(symbol, "H1", h1_ema, h1_rel))

        active = len(checks)
        passed = sum(1 for v in checks if v)

        # Mindestanzahl der erfüllten Filter – Standard: alle aktiven (AND)
        min_req = params.get("trend_min_filters_required", None)
        if isinstance(min_req, (int, float)):
            min_req = int(min_req)
        else:
            min_req = active  # alle aktiven müssen bestehen

        # Wenn keine Filter aktiv sind → automatisch ok
        if active == 0:
            return True
        return passed >= min_req

    def _htf_meta_snapshot(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Liefert HTF-Metadaten (Close, EMA, Bias) für Daily/H4/H1 – analog zur Backtest-Ausgabe.
        """

        def _one(tf: str, ema_period: int, relation: str) -> Dict[str, Any]:
            rel = (relation or "any").lower()
            out = {
                "tf": tf,
                "ema_len": int(ema_period),
                "relation": rel,
                "active": rel not in ("any", "both", "none"),
                "close_price": None,
                "ema": None,
                "bias_label": "na",
            }
            if not out["active"]:
                return out
            try:
                candles = self.data.get_ohlc_series(symbol, tf, 500)
                df_htf = self._df_from_candles(candles)
                closes = df_htf["close"].astype(float)
                if closes is None or len(closes) == 0:
                    return out
                close_val = float(closes.iloc[-1])
                ema_vals = calculate_ema(closes.tolist(), int(max(1, ema_period)))
                ema_val = float(ema_vals[-1]) if ema_vals else None
                out["close_price"] = close_val if np.isfinite(close_val) else None
                out["ema"] = (
                    ema_val if (ema_val is not None and np.isfinite(ema_val)) else None
                )
                if out["ema"] is not None and out["close_price"] is not None:
                    if out["close_price"] > out["ema"]:
                        out["bias_label"] = "above"
                    elif out["close_price"] < out["ema"]:
                        out["bias_label"] = "below"
                    else:
                        out["bias_label"] = "neutral"
            except Exception:
                # belasse Defaults
                pass
            return out

        d_rel = str(params.get("daily_trend_relation", "any"))
        d_ema = int(params.get("daily_trend_ema_period", 50))
        h4_rel = str(params.get("h4_trend_relation", "any"))
        h4_ema = int(params.get("h4_trend_ema_period", 50))
        h1_rel = str(params.get("h1_trend_relation", "any"))
        h1_ema = int(params.get("h1_trend_ema_period", 50))

        return {
            "daily": _one("D1", d_ema, d_rel),
            "h4": _one("H4", h4_ema, h4_rel),
            "h1": _one("H1", h1_ema, h1_rel),
        }

    # ---------- Parametrisierung: Resolver & Signatur ----------------------
    def _dir_norm(self, d: str) -> str:
        d = (d or "").lower()
        if d in ("long", "buy"):
            return "long"
        if d in ("short", "sell"):
            return "short"
        return d

    def _defaults_for_direction(self, direction: str) -> Dict[str, Any]:
        """
        Baseline aus globaler CONFIG (kompatibel zu existierenden Keys).
        """
        d = self._dir_norm(direction)
        if d == "long":
            base = {
                "ema_period": self.config.get("ema_period_long", 13),
                "atr_length": self.config.get("atr_length_long", 14),
                "atr_mult": self.config.get("atr_mult_long", 1.5),
                "bb_length": self.config.get("bb_length_long", 20),
                "bb_std": self.config.get("bb_std_long", 2.0),
                "zscore_length": self.config.get("zscore_length_long", 11),
                "kalman_window": self.config.get("kalman_window_long", 20),
                "kalman_r": self.config.get("kalman_r_long", 0.01),
                "kalman_q": self.config.get("kalman_q_long", 0.01),
                "z_score": float(self.config.get("z_score_long", -2.0)),
                # Szenario 3
                "tp_min_distance": float(self.config.get("tp_min_distance_long", 0.0)),
                # GARCH (Szenario 4)
                "garch_alpha": float(self.config.get("garch_alpha_long", 0.05)),
                "garch_beta": float(self.config.get("garch_beta_long", 0.90)),
                "garch_omega": self.config.get("garch_omega_long", None),
                "garch_use_log_returns": bool(
                    self.config.get("garch_use_log_returns_long", True)
                ),
                "garch_scale": float(self.config.get("garch_scale_long", 100.0)),
                "garch_min_periods": int(self.config.get("garch_min_periods_long", 50)),
                "garch_sigma_floor": float(
                    self.config.get("garch_sigma_floor_long", 1e-6)
                ),
                "daily_trend_ema_period": int(
                    self.config.get("daily_trend_ema_period", 50)
                ),
                # "above" | "below" | "any"
                "daily_trend_relation": str(
                    self.config.get("daily_trend_relation_long", "above")
                ),
                # H4/H1 Trendfilter (optional; relation "any"/"both"/"none" deaktiviert)
                "h4_trend_ema_period": int(self.config.get("h4_trend_ema_period", 50)),
                "h4_trend_relation": str(
                    self.config.get("h4_trend_relation_long", "above")
                ),
                "h1_trend_ema_period": int(self.config.get("h1_trend_ema_period", 50)),
                "h1_trend_relation": str(
                    self.config.get("h1_trend_relation_long", "above")
                ),
            }
            # Optional: Mindestanzahl aktiver Filter
            _min_req = self.config.get("trend_min_filters_required", None)
            if isinstance(_min_req, (int, float)):
                base["trend_min_filters_required"] = int(_min_req)
            return base
        else:
            base = {
                "ema_period": self.config.get("ema_period_short", 13),
                "atr_length": self.config.get("atr_length_short", 14),
                "atr_mult": self.config.get("atr_mult_short", 1.5),
                "bb_length": self.config.get("bb_length_short", 20),
                "bb_std": self.config.get("bb_std_short", 2.0),
                "zscore_length": self.config.get("zscore_length_short", 11),
                "kalman_window": self.config.get("kalman_window_short", 20),
                "kalman_r": self.config.get("kalman_r_short", 0.01),
                "kalman_q": self.config.get("kalman_q_short", 0.01),
                "z_score": float(self.config.get("z_score_short", 2.0)),
                # Szenario 3
                "tp_min_distance": float(self.config.get("tp_min_distance_short", 0.0)),
                # GARCH (Szenario 4)
                "garch_alpha": float(self.config.get("garch_alpha_short", 0.05)),
                "garch_beta": float(self.config.get("garch_beta_short", 0.90)),
                "garch_omega": self.config.get("garch_omega_short", None),
                "garch_use_log_returns": bool(
                    self.config.get("garch_use_log_returns_short", True)
                ),
                "garch_scale": float(self.config.get("garch_scale_short", 100.0)),
                "garch_min_periods": int(
                    self.config.get("garch_min_periods_short", 50)
                ),
                "garch_sigma_floor": float(
                    self.config.get("garch_sigma_floor_short", 1e-6)
                ),
                "daily_trend_ema_period": int(
                    self.config.get("daily_trend_ema_period", 50)
                ),
                # "above" | "below" | "any"
                "daily_trend_relation": str(
                    self.config.get("daily_trend_relation_short", "below")
                ),
                # H4/H1 Trendfilter (optional; relation "any"/"both"/"none" deaktiviert)
                "h4_trend_ema_period": int(self.config.get("h4_trend_ema_period", 50)),
                "h4_trend_relation": str(
                    self.config.get("h4_trend_relation_short", "below")
                ),
                "h1_trend_ema_period": int(self.config.get("h1_trend_ema_period", 50)),
                "h1_trend_relation": str(
                    self.config.get("h1_trend_relation_short", "below")
                ),
            }
            _min_req = self.config.get("trend_min_filters_required", None)
            if isinstance(_min_req, (int, float)):
                base["trend_min_filters_required"] = int(_min_req)
            return base

    def _resolve_params(
        self, symbol: str, timeframe: str, direction: str
    ) -> Dict[str, Any]:
        """
        Merged Parameter gemäss Fallback-Hierarchie:
        global -> */* -> symbol/* -> */tf -> symbol/tf
        """
        d = self._dir_norm(direction)
        base = dict(self._defaults_for_direction(d))
        ov = self.config.get("param_overrides", {}) or {}

        # Neu: Normalisierung
        sym = (symbol or "").upper().strip()
        tf = (timeframe or "").upper().strip()

        def get_ov(sym_key: str, tf_key: str) -> Dict[str, Any]:
            try:
                block = (ov.get(sym_key, {}) or {}).get(tf_key, {}) or {}
            except Exception:
                return {}
            if d == "long":
                return dict(block.get("long") or block.get("buy") or {})
            else:
                return dict(block.get("short") or block.get("sell") or {})

        merged = {
            **base,
            **get_ov("*", "*"),
            **get_ov(sym, "*"),
            **get_ov("*", tf),
            **get_ov(sym, tf),
        }
        return merged

    def _params_signature(self, p: Dict[str, Any]) -> tuple:
        """Deterministische Signatur für Cache-Invalidierung."""
        num_keys = (
            "ema_period",
            "atr_length",
            "atr_mult",
            "bb_length",
            "bb_std",
            "zscore_length",
            "kalman_window",
            "kalman_r",
            "kalman_q",
            "z_score",
            "tp_min_distance",
            # garch
            "garch_alpha",
            "garch_beta",
            "garch_scale",
            "garch_min_periods",
            "garch_sigma_floor",
            "daily_trend_ema_period",
            "h4_trend_ema_period",
            "h1_trend_ema_period",
        )
        sig = []
        for k in num_keys:
            if k in p:
                sig.append((k, float(p[k])))
        # String-Parameter separat (nicht in float gießen)
        if "daily_trend_relation" in p:
            sig.append(("daily_trend_relation", str(p["daily_trend_relation"]).lower()))
        if "h4_trend_relation" in p:
            sig.append(("h4_trend_relation", str(p["h4_trend_relation"]).lower()))
        if "h1_trend_relation" in p:
            sig.append(("h1_trend_relation", str(p["h1_trend_relation"]).lower()))
        if (
            "trend_min_filters_required" in p
            and p.get("trend_min_filters_required") is not None
        ):
            try:
                sig.append(
                    (
                        "trend_min_filters_required",
                        float(p["trend_min_filters_required"]),
                    )
                )
            except Exception:
                pass
        if "garch_use_log_returns" in p:
            sig.append(("garch_use_log_returns", bool(p["garch_use_log_returns"])))
        if "garch_omega" in p and p.get("garch_omega") is not None:
            try:
                sig.append(("garch_omega", float(p["garch_omega"])))
            except Exception:
                pass
        return tuple(sig)

    # ---- internals ---------------------------------------------------------

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
        self, ind: Dict[str, Any], direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Filtert anhand des Intraday‑Volatilitätsclusters.
        Implementiert dieselbe Logik wie die Backtest‑Variante (_vol_cluster_guard).
        """
        cfg = self.config or {}
        tf = str(cfg.get("timeframe") or "").upper()
        params = ind.get("_params", {}) or {}

        # Cluster-Parameter vorrangig aus params (Setup/Richtung), Fallback: Root-Config
        feature = str(
            params.get(
                "intraday_vol_feature",
                cfg.get("intraday_vol_feature", "garch_forecast"),
            )
        ).lower()
        window = int(
            params.get(
                "intraday_vol_cluster_window",
                cfg.get(
                    "intraday_vol_cluster_window", self._default_cluster_window(tf)
                ),
            )
        )
        k = int(
            params.get(
                "intraday_vol_cluster_k",
                cfg.get("intraday_vol_cluster_k", 3),
            )
        )
        min_points = int(
            params.get(
                "intraday_vol_min_points",
                cfg.get("intraday_vol_min_points", 60),
            )
        )
        log_transform = bool(
            params.get(
                "intraday_vol_log_transform",
                cfg.get("intraday_vol_log_transform", True),
            )
        )
        hysteresis_bars = int(
            params.get(
                "cluster_hysteresis_bars",
                cfg.get("cluster_hysteresis_bars", 0),
            )
        )
        allowed_default = ["low", "mid"]
        intraday_allowed = (
            params.get(
                "intraday_vol_allowed",
                cfg.get("intraday_vol_allowed", allowed_default),
            )
            or allowed_default
        )

        meta: Dict[str, Any] = {
            "feature": feature,
            "window": window,
            "k": k,
            "min_points": min_points,
            "log_transform": log_transform,
            "hysteresis_bars": hysteresis_bars,
            "direction": direction,
        }

        # Feature-Serie bestimmen (ATR-Punkte oder GARCH-Sigma)
        atr_len = int(params.get("atr_length", 14))
        try:
            if feature == "atr_points":
                df = ind.get("df")
                if not isinstance(df, pd.DataFrame):
                    state = {"status": "series_unavailable", "sample_size": 0}
                else:
                    series_raw = calculate_atr_series(df, atr_len)
                    # Nur abgeschlossene Bars: aktuelle Bar ausklammern
                    ser = series_raw.iloc[:-1] if len(series_raw) > 1 else series_raw
                    state = calculate_vol_cluster_state(
                        ser,
                        window=max(1, window),
                        k=max(1, k),
                        min_points=max(1, min_points),
                        log_transform=log_transform,
                    )
            else:
                series_raw = ind.get("garch_sigma_ret")
                if series_raw is None:
                    state = {"status": "series_unavailable", "sample_size": 0}
                else:
                    # GARCH-Sigma-Serie vollständig verwenden; Warmup/NaNs werden
                    # in calculate_vol_cluster_state behandelt. So entspricht die
                    # effektive sample_size der Backtest-Logik.
                    state = calculate_vol_cluster_state(
                        series_raw,
                        window=max(1, window),
                        k=max(1, k),
                        min_points=max(1, min_points),
                        log_transform=log_transform,
                    )
        except Exception:
            state = {"status": "series_unavailable", "sample_size": 0}

        meta["sample_size"] = int(state.get("sample_size", 0))
        status = state.get("status")
        if status and status != "ok":
            meta["status"] = status
            return False, meta

        labels = state.get("labels")
        mapping = state.get("mapping", {})
        if labels is None or len(labels) == 0:
            meta["status"] = "no_labels"
            return False, meta

        current_idx = int(labels[-1])
        current_label = str(mapping.get(current_idx, "unknown"))
        meta["label"] = current_label
        meta["sigma"] = state.get("sigma")
        if state.get("centers") is not None:
            meta["centers"] = state["centers"]
        if state.get("centers_log") is not None:
            meta["centers_log"] = state["centers_log"]

        allowed = {str(label).strip().lower() for label in intraday_allowed}
        meta["allowed_labels"] = sorted(list(allowed))

        hysteresis_ok = True
        h = max(int(hysteresis_bars), 0)
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

    @staticmethod
    def _df_from_candles(candles) -> pd.DataFrame:
        """
        Erzwingt saubere Struktur für Indikatoren.
        Erwartete Spalten: time, open, high, low, close, [volume]
        """
        df = pd.DataFrame(candles or [])
        if df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
            raise ValueError("Ungültige Kerzeneingabe – erforderliche Spalten fehlen.")
        return df

    @staticmethod
    def _last_val(x):
        """Robuste Extraktion des letzten Wertes aus Liste/np.array/Series/Scalar."""
        if x is None:
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            return float(x[-1]) if len(x) else None
        if isinstance(x, pd.Series):
            return float(x.iloc[-1]) if len(x) else None
        try:
            return float(x)
        except Exception:
            return None

    # --- Indicator-Builders (Long/Short getrennt)

    def _build_indicators_with_params(
        self, df: pd.DataFrame, p: Dict[str, Any]
    ) -> Dict[str, Any]:
        closes = df["close"].astype(float)
        ema = calculate_ema(closes.tolist(), int(p["ema_period"]))

        atr_raw = calculate_atr(df, int(p["atr_length"]))
        if isinstance(atr_raw, pd.Series):
            atr_val = float(atr_raw.iloc[-1])
        else:
            atr_val = float(atr_raw) if atr_raw is not None else float("nan")

        upper, mid, lower = calculate_bollinger_bands(
            closes, period=int(p["bb_length"]), std_factor=float(p["bb_std"])
        )
        zscore = calculate_zscore(pd.Series(closes), int(p["zscore_length"]))
        kalman_z = calculate_kalman_zscore(
            closes,
            window=int(p["kalman_window"]),
            R=float(p["kalman_r"]),
            Q=float(p["kalman_q"]),
        )
        # GARCH-basierte Größen (für Szenario 4)
        try:
            sigma_ret = calculate_garch_volatility(
                closes,
                alpha=float(p.get("garch_alpha", 0.05)),
                beta=float(p.get("garch_beta", 0.90)),
                omega=p.get("garch_omega", None),
                use_log_returns=bool(p.get("garch_use_log_returns", True)),
                scale=float(p.get("garch_scale", 100.0)),
                min_periods=int(p.get("garch_min_periods", 50)),
                sigma_floor=float(p.get("garch_sigma_floor", 1e-6)),
            )
        except Exception:
            sigma_ret = None
        try:
            kalman_garch_z = calculate_kalman_garch_zscore(
                closes,
                R=float(p.get("kalman_r", 0.01)),
                Q=float(p.get("kalman_q", 0.01)),
                alpha=float(p.get("garch_alpha", 0.05)),
                beta=float(p.get("garch_beta", 0.90)),
                omega=p.get("garch_omega", None),
                use_log_returns=bool(p.get("garch_use_log_returns", True)),
                scale=float(p.get("garch_scale", 100.0)),
                min_periods=int(p.get("garch_min_periods", 50)),
                sigma_floor=float(p.get("garch_sigma_floor", 1e-6)),
            )
        except Exception:
            kalman_garch_z = None
        return {
            "df": df,
            "closes": closes,
            "ema": ema,
            "atr": atr_val,
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "zscore": zscore,
            "kalman_z": kalman_z,
            "kalman_garch_z": kalman_garch_z,
            "garch_sigma_ret": sigma_ret,
            "_params": p,
        }

    # ---- public API --------------------------------------------------------

    def _snapshot_indicators(self, ind: Dict[str, Any]) -> Dict[str, Any]:
        """Extract compact, last-value indicator snapshot for logging/metadata."""
        snap = {}
        try:
            # EMA may be list/np/Series
            ema = ind.get("ema")
            snap["ema"] = self._last_val(ema)
        except Exception:
            pass
        try:
            snap["atr"] = float(ind.get("atr")) if ind.get("atr") is not None else None
        except Exception:
            pass
        try:
            bb_u = ind.get("bb_upper")
            bb_m = ind.get("bb_mid")
            bb_l = ind.get("bb_lower")
            snap["bb_upper"] = self._last_val(bb_u)
            snap["bb_mid"] = self._last_val(bb_m)
            snap["bb_lower"] = self._last_val(bb_l)
        except Exception:
            pass
        try:
            snap["zscore"] = self._last_val(ind.get("zscore"))
        except Exception:
            pass
        try:
            snap["kalman_z"] = self._last_val(ind.get("kalman_z"))
        except Exception:
            pass
        try:
            snap["kalman_garch_z"] = self._last_val(ind.get("kalman_garch_z"))
        except Exception:
            pass
        try:
            snap["garch_sigma_ret"] = self._last_val(ind.get("garch_sigma_ret"))
        except Exception:
            pass
        # Also attach parameters that influenced indicators
        try:
            params = dict(ind.get("_params") or {})
            # Only include simple types that help interpretation
            keep = (
                "ema_period",
                "atr_length",
                "atr_mult",
                "bb_length",
                "bb_std",
                "zscore_length",
                "kalman_window",
                "kalman_r",
                "kalman_q",
                "z_score",
                "tp_min_distance",
                "garch_alpha",
                "garch_beta",
                "garch_omega",
                "garch_use_log_returns",
                "garch_scale",
                "garch_min_periods",
                "garch_sigma_floor",
                "daily_trend_ema_period",
                "daily_trend_relation",
                "h4_trend_ema_period",
                "h4_trend_relation",
                "h1_trend_ema_period",
                "h1_trend_relation",
                "trend_min_filters_required",
            )
            snap["params"] = {k: params.get(k) for k in keep if k in params}
        except Exception:
            pass
        return snap

    def evaluate_all(self, symbol: str, timeframe: str) -> Optional[dict]:
        """
        Ruft die Daten vom Provider ab, berechnet Indikatoren (mit Cache) und evaluiert die Szenarien.
        Gibt beim ersten Treffer ein Signal-Dict zurück, sonst None.
        """
        try:
            candles = self.data.get_ohlc_series(symbol, timeframe, 500)
        except Exception as e:
            log_service.log_system(
                f"[ZScore SzenarioEval] ❌ Datenabruf fehlgeschlagen {symbol}/{timeframe}: {e}",
                level="ERROR",
            )
            return None

        if not candles or len(candles) < 50:
            return None

        try:
            df = self._df_from_candles(candles)
        except Exception as e:
            log_service.log_system(f"[ZScore SzenarioEval] ❌ {e}", level="ERROR")
            return None

        last_ts = (
            int(pd.to_datetime(df["time"].iloc[-1]).value)
            if "time" in df.columns
            else len(df)  # Fallback
        )

        key = (symbol, timeframe)
        cached = self._cache.get(key)
        # Parameter je Richtung auflösen
        params_long = self._resolve_params(symbol, timeframe, "long")
        params_short = self._resolve_params(symbol, timeframe, "short")
        sig_long = self._params_signature(params_long)
        sig_short = self._params_signature(params_short)

        need_long = True
        need_short = True
        if cached and cached.get("last_ts") == last_ts:
            if cached.get("sig_long") == sig_long:
                ind_long = cached["ind_long"]
                need_long = False
            if cached.get("sig_short") == sig_short:
                ind_short = cached["ind_short"]
                need_short = False

        if need_long:
            ind_long = self._build_indicators_with_params(df, params_long)
            ind_long["symbol"] = symbol
        if need_short:
            ind_short = self._build_indicators_with_params(df, params_short)
            ind_short["symbol"] = symbol

        self._cache[key] = {
            "last_ts": last_ts,
            "sig_long": sig_long,
            "ind_long": ind_long,
            "sig_short": sig_short,
            "ind_short": ind_short,
        }

        # Szenario-Kandidaten inkl. Namen vorbereiten. Die Namen werden
        # für das Whitelist-Pre‑Filtering verwendet, damit nur erlaubte
        # Szenarien überhaupt evaluiert werden.
        candidates = [
            ("szenario_2_long", "long", self.szenario_2_long, ind_long),
            ("szenario_2_short", "short", self.szenario_2_short, ind_short),
            ("szenario_3_long", "long", self.szenario_3_long, ind_long),
            ("szenario_3_short", "short", self.szenario_3_short, ind_short),
            ("szenario_4_long", "long", self.szenario_4_long, ind_long),
            ("szenario_4_short", "short", self.szenario_4_short, ind_short),
            # Szenario 5 (Kalman-Z + Bollinger + Intraday-Vol-Cluster)
            ("szenario_5_long", "long", self.szenario_5_long, ind_long),
            ("szenario_5_short", "short", self.szenario_5_short, ind_short),
            # Szenario 6 (Multi‑TF Overlay auf Basis Szenario 2)
            ("szenario_6_long", "long", self.szenario_6_long, ind_long),
            ("szenario_6_short", "short", self.szenario_6_short, ind_short),
        ]

        # Whitelist-Filter VOR Evaluation: nur gewünschte Szenarien zulassen.
        allowed = self.config.get("allowed_scenarios")
        if allowed:
            try:
                allowed_set = set(allowed)
            except Exception:
                # Fallback: robustes Casten
                allowed_set = set(list(allowed))
            candidates = [c for c in candidates if c[0] in allowed_set]

        # Nur erlaubte Richtungen evaluieren (vollständige Entkopplung)
        candidates = [c for c in candidates if self._is_allowed(c[1])]

        for scen_name, direction, fn, ind in candidates:
            try:
                result = fn(ind)
                if result:
                    # Attach indicator snapshot for metadata/logging
                    try:
                        indicators = self._snapshot_indicators(ind)
                        meta = result.get("meta")
                        if isinstance(meta, dict):
                            # Szenario‑6‑Kette (Multi‑TF‑Overlay) sichtbar machen
                            scen6 = meta.get("scenario6")
                            if scen6:
                                indicators = dict(indicators or {})
                                indicators["scenario6"] = scen6
                            # HTF‑Snapshot (Trend‑Bias über D1/H4/H1) in den
                            # Indicator‑Snapshot spiegeln – entspricht Backtest‑Meta.
                            htf_meta = meta.get("htf")
                            if htf_meta is not None:
                                indicators = dict(indicators or {})
                                indicators["htf"] = htf_meta
                            # Szenario‑5‑Metadaten (Intraday‑Vol‑Cluster) ebenfalls
                            # im Indicator‑Snapshot ablegen, damit sie im Trade‑Log
                            # unter metadata.indicators verfügbar sind.
                            vol_cluster_meta = meta.get("vol_cluster")
                            if vol_cluster_meta is not None:
                                indicators = dict(indicators or {})
                                indicators["vol_cluster"] = vol_cluster_meta
                        result["indicators"] = indicators
                        # Also expose direction used for indicator resolution
                        result.setdefault(
                            "direction", "buy" if direction == "long" else "sell"
                        )
                        # Ensure scenario name is set (defensive)
                        result.setdefault("scenario", scen_name)
                    except Exception:
                        pass
                    return result
            except Exception as e:
                log_service.log_system(
                    f"[ZScore SzenarioEval] Fehler in {fn.__name__}: {e}", level="ERROR"
                )

        return None

    # ---- Szenarien ---------------------------------------------------------

    def szenario_1_long(self, ind) -> Optional[dict]:
        df, ema, atr, z = ind["df"], ind["ema"], ind["atr"], ind["zscore"]
        thr = float(ind["_params"]["z_score"])  # negative Schwelle
        if (
            z is None
            or len(z) == 0
            or not np.isfinite(z.iloc[-1])
            or z.iloc[-1] > thr
            or not np.isfinite(atr)
        ):
            return None
        c = df.iloc[-1]
        atr_mult = float(ind["_params"].get("atr_mult", 1.5))
        sl = float(c["low"]) - atr_mult * float(atr)
        tp = self._last_val(ema)
        if tp is None:
            return None
        return signal_long_market(sl=sl, tp=tp) | {"scenario": "szenario_1_long"}

    def szenario_1_short(self, ind) -> Optional[dict]:
        df, ema, atr, z = ind["df"], ind["ema"], ind["atr"], ind["zscore"]
        thr = float(ind["_params"]["z_score"])  # positive Schwelle
        if (
            z is None
            or len(z) == 0
            or not np.isfinite(z.iloc[-1])
            or z.iloc[-1] < thr
            or not np.isfinite(atr)
        ):
            return None
        c = df.iloc[-1]
        atr_mult = float(ind["_params"].get("atr_mult", 1.5))
        sl = float(c["high"]) + atr_mult * float(atr)
        tp = self._last_val(ema)
        if tp is None:
            return None
        return signal_short_market(sl=sl, tp=tp) | {"scenario": "szenario_1_short"}

    def szenario_2_long(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kz = ind["kalman_z"]
        lower, mid = ind["bb_lower"], ind["bb_mid"]
        thr = float(ind["_params"]["z_score"])
        # Trendfilter (Daily/H4/H1) – kombinierbar über params
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, ind["_params"]):
            return None
        if (
            kz is None
            or len(kz) == 0
            or not np.isfinite(kz.iloc[-1])
            or kz.iloc[-1] > thr
        ):
            return None
        if lower is None or mid is None or len(lower) == 0 or len(mid) == 0:
            return None
        # Close muss unterhalb der unteren BB sein
        if float(c["close"]) > float(lower.iloc[-1]):
            return None
        atr_mult = float(ind["_params"].get("atr_mult", 1.5))
        atr = float(ind["atr"])
        if not np.isfinite(atr):
            return None
        sl = float(c["low"]) - atr_mult * atr
        tp = float(mid.iloc[-1])
        meta = {"htf": self._htf_meta_snapshot(symbol, ind["_params"])}
        return signal_long_market(sl=sl, tp=tp) | {
            "scenario": "szenario_2_long",
            "meta": meta,
        }

    def szenario_2_short(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kz = ind["kalman_z"]
        upper, mid = ind["bb_upper"], ind["bb_mid"]
        thr = float(ind["_params"]["z_score"])
        # Trendfilter (Daily/H4/H1)
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, ind["_params"]):
            return None
        if (
            kz is None
            or len(kz) == 0
            or not np.isfinite(kz.iloc[-1])
            or kz.iloc[-1] < thr
        ):
            return None
        if upper is None or mid is None or len(upper) == 0 or len(mid) == 0:
            return None
        # Close muss oberhalb der oberen BB sein
        if float(c["close"]) < float(upper.iloc[-1]):
            return None
        atr_mult = float(ind["_params"].get("atr_mult", 1.5))
        atr = float(ind["atr"])
        if not np.isfinite(atr):
            return None
        sl = float(c["high"]) + atr_mult * atr
        tp = float(mid.iloc[-1])
        meta = {"htf": self._htf_meta_snapshot(symbol, ind["_params"])}
        return signal_short_market(sl=sl, tp=tp) | {
            "scenario": "szenario_2_short",
            "meta": meta,
        }

    # ---- Szenario 3: Kalman-Z + Bollinger + EMA(TP) -----------------------
    def szenario_3_long(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kz = ind["kalman_z"]
        lower, mid = ind["bb_lower"], ind["bb_mid"]
        params = ind["_params"]
        thr = float(params["z_score"])  # negative Schwelle
        # Trendfilter (Daily/H4/H1)
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, params):
            return None
        if (
            kz is None
            or len(kz) == 0
            or not np.isfinite(kz.iloc[-1])
            or kz.iloc[-1] > thr
        ):
            return None
        if lower is None or len(lower) == 0:
            return None
        if float(c["close"]) > float(lower.iloc[-1]):
            return None
        ema_list = ind.get("ema")
        ema_val = self._last_val(ema_list)
        if ema_val is None:
            return None
        # Interpret 'tp_min_distance' as pips and convert to price distance via pip_size
        pip_size = get_pip_size(symbol)
        tp_min_dist = float(params.get("tp_min_distance", 0.0)) * float(pip_size)
        if (ema_val - float(c["close"])) < tp_min_dist:
            return None
        atr = float(ind.get("atr", np.nan))
        if not np.isfinite(atr):
            return None
        atr_mult = float(params.get("atr_mult", 1.5))
        sl = float(c["low"]) - atr_mult * atr
        tp = float(ema_val)
        return signal_long_market(sl=sl, tp=tp) | {"scenario": "szenario_3_long"}

    def szenario_3_short(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kz = ind["kalman_z"]
        upper, mid = ind["bb_upper"], ind["bb_mid"]
        params = ind["_params"]
        thr = float(params["z_score"])  # positive Schwelle
        # Trendfilter (Daily/H4/H1)
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, params):
            return None
        if (
            kz is None
            or len(kz) == 0
            or not np.isfinite(kz.iloc[-1])
            or kz.iloc[-1] < thr
        ):
            return None
        if upper is None or len(upper) == 0:
            return None
        if float(c["close"]) < float(upper.iloc[-1]):
            return None
        ema_list = ind.get("ema")
        ema_val = self._last_val(ema_list)
        if ema_val is None:
            return None
        # Interpret 'tp_min_distance' as pips and convert to price distance via pip_size
        pip_size = get_pip_size(symbol)
        tp_min_dist = float(params.get("tp_min_distance", 0.0)) * float(pip_size)
        if (float(c["close"]) - ema_val) < tp_min_dist:
            return None
        atr = float(ind.get("atr", np.nan))
        if not np.isfinite(atr):
            return None
        atr_mult = float(params.get("atr_mult", 1.5))
        sl = float(c["high"]) + atr_mult * atr
        tp = float(ema_val)
        return signal_short_market(sl=sl, tp=tp) | {"scenario": "szenario_3_short"}

    # ---- Szenario 4: Kalman + GARCH-Z + Bollinger ------------------------
    def szenario_4_long(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kgz = ind.get("kalman_garch_z")
        lower, mid = ind.get("bb_lower"), ind.get("bb_mid")
        params = ind["_params"]
        thr = float(params["z_score"])  # negative Schwelle
        # Trendfilter (Daily/H4/H1)
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, params):
            return None
        if (
            kgz is None
            or len(kgz) == 0
            or not np.isfinite(kgz.iloc[-1])
            or kgz.iloc[-1] > thr
        ):
            return None
        if lower is None or mid is None or len(lower) == 0 or len(mid) == 0:
            return None
        if float(c["close"]) > float(lower.iloc[-1]):
            return None
        atr = float(ind.get("atr", np.nan))
        if not np.isfinite(atr):
            return None
        atr_mult = float(params.get("atr_mult", 1.5))
        sl = float(c["low"]) - atr_mult * atr
        tp = float(mid.iloc[-1])
        return signal_long_market(sl=sl, tp=tp) | {"scenario": "szenario_4_long"}

    def szenario_4_short(self, ind) -> Optional[dict]:
        df = ind["df"]
        c = df.iloc[-1]
        kgz = ind.get("kalman_garch_z")
        upper, mid = ind.get("bb_upper"), ind.get("bb_mid")
        params = ind["_params"]
        thr = float(params["z_score"])  # positive Schwelle
        # Trendfilter (Daily/H4/H1)
        symbol = ind.get("symbol")
        if not self._multi_trend_ok(symbol, params):
            return None
        if (
            kgz is None
            or len(kgz) == 0
            or not np.isfinite(kgz.iloc[-1])
            or kgz.iloc[-1] < thr
        ):
            return None
        if upper is None or mid is None or len(upper) == 0 or len(mid) == 0:
            return None
        if float(c["close"]) < float(upper.iloc[-1]):
            return None
        atr = float(ind.get("atr", np.nan))
        if not np.isfinite(atr):
            return None
        atr_mult = float(params.get("atr_mult", 1.5))
        sl = float(c["high"]) + atr_mult * atr
        tp = float(mid.iloc[-1])
        return signal_short_market(sl=sl, tp=tp) | {"scenario": "szenario_4_short"}

    # ---- Szenario 5: Szenario 2 + Intraday-Vol-Cluster --------------------
    def szenario_5_long(self, ind) -> Optional[dict]:
        """
        Entspricht Szenario 2 (Kalman-Z + Bollinger), erweitert um
        Intraday-Volatilitätscluster-Filter wie im Backtest.
        """
        base = self.szenario_2_long(ind)
        if not base:
            return None

        allowed, cluster_meta = self._vol_cluster_guard(ind, "long")
        if not allowed:
            return None

        base["scenario"] = "szenario_5_long"
        meta = base.get("meta") if isinstance(base.get("meta"), dict) else {}
        meta["vol_cluster"] = cluster_meta
        # HTF-Metadaten sicherstellen (falls Basis keine hatte)
        if "htf" not in meta:
            symbol = ind.get("symbol")
            params = ind.get("_params", {})
            meta["htf"] = self._htf_meta_snapshot(symbol, params)
        base["meta"] = meta
        return base

    def szenario_5_short(self, ind) -> Optional[dict]:
        """
        Entspricht Szenario 2 (Kalman-Z + Bollinger), erweitert um
        Intraday-Volatilitätscluster-Filter wie im Backtest.
        """
        base = self.szenario_2_short(ind)
        if not base:
            return None

        allowed, cluster_meta = self._vol_cluster_guard(ind, "short")
        if not allowed:
            return None

        base["scenario"] = "szenario_5_short"
        meta = base.get("meta") if isinstance(base.get("meta"), dict) else {}
        meta["vol_cluster"] = cluster_meta
        if "htf" not in meta:
            symbol = ind.get("symbol")
            params = ind.get("_params", {})
            meta["htf"] = self._htf_meta_snapshot(symbol, params)
        base["meta"] = meta
        return base

    # ---- Szenario 6: Multi‑TF Overlay (baut auf Szenario 2 auf) ------------
    # Die Logik spiegelt die Backtest‑Variante: für jedes in scenario6_timeframes
    # angegebene TF werden Kalman‑Z und Bollinger‑Bänder geprüft. Die TF‑spezifischen
    # Parameter werden aus scenario6_params bezogen. Je nach scenario6_mode müssen
    # entweder alle ("all") oder mindestens eines ("any") bestehen.

    @staticmethod
    def _scenario6_normalize_tf(tf: str) -> str:
        tf_norm = str(tf or "").strip().upper()
        return tf_norm

    def _scenario6_params_for(self, tf: str, direction: str) -> Dict[str, Any]:
        tf_norm = self._scenario6_normalize_tf(tf)
        dir_key = "long" if str(direction).lower() in ("long", "buy") else "short"

        scen6: Dict[str, Any] = self.config.get("scenario6_params", {}) or {}
        overrides = scen6.get(tf_norm, {}) if isinstance(scen6, dict) else {}

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

        # Baseline aus global/Direction‑Defaults (Kompatibilität zu Live‑Parametern)
        base = self._defaults_for_direction(dir_key)
        params = {
            # Backtest‑Keynamen bewusst beibehalten, um 1:1 Kompatibilität zu sichern
            "window_length": _cast_int(
                "window_length",
                int(base.get("kalman_window", base.get("zscore_length", 100))),
            ),
            "b_b_length": _cast_int("b_b_length", int(base.get("bb_length", 20))),
            "std_factor": _cast_float("std_factor", float(base.get("bb_std", 2.0))),
            "kalman_r": _cast_float("kalman_r", float(base.get("kalman_r", 0.01))),
            "kalman_q": _cast_float("kalman_q", float(base.get("kalman_q", 0.01))),
        }
        # z‑Schwellen – Symmetrisch via z_score ODER getrennt via z_score_long/short
        if "z_score" in dir_overrides:
            zs = _cast_float(
                "z_score",
                float(base.get("z_score", -2.0 if dir_key == "long" else 2.0)),
            )
            params["z_score_long"] = zs
            params["z_score_short"] = zs
        else:
            params["z_score_long"] = _cast_float(
                "z_score_long", float(base.get("z_score", -2.0))
            )
            params["z_score_short"] = _cast_float(
                "z_score_short", float(base.get("z_score", 2.0))
            )
        return params

    def _scenario6_check_tf(
        self, symbol: str, tf: str, direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        tf_norm = self._scenario6_normalize_tf(tf)
        meta: Dict[str, Any] = {"tf": tf_norm, "direction": direction}
        if not tf_norm:
            meta["status"] = "invalid_timeframe"
            return False, meta

        try:
            candles = self.data.get_ohlc_series(symbol, tf_norm, 500)
            df = self._df_from_candles(candles)
        except Exception:
            meta["status"] = "no_data"
            return False, meta

        closes = df["close"].astype(float)
        price_now = float(closes.iloc[-1]) if len(closes) else float("nan")
        if not np.isfinite(price_now):
            meta["status"] = "no_price"
            return False, meta

        params = self._scenario6_params_for(tf_norm, direction)
        meta["params"] = {k: params[k] for k in params}

        # Kalman‑Z je TF
        try:
            z_series = calculate_kalman_zscore(
                closes,
                window=int(params["window_length"]),
                R=float(params["kalman_r"]),
                Q=float(params["kalman_q"]),
            )
            z_now = (
                float(z_series.iloc[-1])
                if hasattr(z_series, "iloc")
                else float(z_series[-1])
            )
        except Exception:
            z_now = float("nan")
        if not np.isfinite(z_now):
            meta["status"] = "no_zscore"
            return False, meta
        meta["z"] = z_now

        # Bollinger je TF
        try:
            upper, mid, lower = calculate_bollinger_bands(
                closes,
                period=int(params["b_b_length"]),
                std_factor=float(params["std_factor"]),
            )
            upper_now = (
                float(upper.iloc[-1]) if hasattr(upper, "iloc") else float(upper[-1])
            )
            lower_now = (
                float(lower.iloc[-1]) if hasattr(lower, "iloc") else float(lower[-1])
            )
        except Exception:
            upper_now = lower_now = float("nan")

        threshold = (
            float(params["z_score_long"])
            if str(direction).lower() in ("long", "buy")
            else float(params["z_score_short"])
        )

        if str(direction).lower() in ("long", "buy"):
            meta["lower"] = lower_now if np.isfinite(lower_now) else None
            if z_now > threshold:
                meta["status"] = "z_above_threshold"
                meta["threshold"] = threshold
                return False, meta
            if not np.isfinite(lower_now):
                meta["status"] = "no_lower_band"
                return False, meta
            if price_now > lower_now:
                meta["status"] = "price_above_lower"
                meta["price"] = price_now
                return False, meta
        else:
            meta["upper"] = upper_now if np.isfinite(upper_now) else None
            if z_now < threshold:
                meta["status"] = "z_below_threshold"
                meta["threshold"] = threshold
                return False, meta
            if not np.isfinite(upper_now):
                meta["status"] = "no_upper_band"
                return False, meta
            if price_now < upper_now:
                meta["status"] = "price_below_upper"
                meta["price"] = price_now
                return False, meta

        meta["price"] = price_now
        meta["status"] = "ok"
        return True, meta

    def _scenario6_evaluate_chain(
        self, symbol: str, direction: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        raw_tfs = self.config.get("scenario6_timeframes", []) or []
        if isinstance(raw_tfs, str):
            raw_tfs = [raw_tfs]
        tfs = [
            self._scenario6_normalize_tf(tf) for tf in raw_tfs if str(tf or "").strip()
        ]
        mode = str(self.config.get("scenario6_mode", "all")).strip().lower()
        mode = mode if mode in ("all", "any") else "all"

        results: List[Dict[str, Any]] = []
        successes = 0
        for tf in tfs:
            ok, meta = self._scenario6_check_tf(symbol, tf, direction)
            meta["ok"] = bool(ok)
            results.append(meta)
            if ok:
                successes += 1

        if not results:
            return False, []
        if mode == "any":
            return successes > 0, results
        return successes == len(results), results

    def szenario_6_long(self, ind) -> Optional[dict]:
        symbol = ind.get("symbol")
        ok, chain = self._scenario6_evaluate_chain(symbol, "long")
        if not ok:
            return None
        # Basissignal aus Szenario 2 (identische Entry/SL/TP‑Logik)
        base = self.szenario_2_long(ind)
        if not base:
            return None
        base["scenario"] = "szenario_6_long"
        # meta: scenario6 chain anhängen, falls genutzt wird
        meta = base.setdefault("meta", {})
        meta["scenario6"] = {
            "mode": self.config.get("scenario6_mode", "all"),
            "chain": chain,
        }
        return base

    def szenario_6_short(self, ind) -> Optional[dict]:
        symbol = ind.get("symbol")
        ok, chain = self._scenario6_evaluate_chain(symbol, "short")
        if not ok:
            return None
        base = self.szenario_2_short(ind)
        if not base:
            return None
        base["scenario"] = "szenario_6_short"
        meta = base.setdefault("meta", {})
        meta["scenario6"] = {
            "mode": self.config.get("scenario6_mode", "all"),
            "chain": chain,
        }
        return base
