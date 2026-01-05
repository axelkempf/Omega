# -*- coding: utf-8 -*-
# mypy: disable-error-code="no-any-return,call-overload"
# NOTE: This module uses a Dict[Tuple[Any, ...], Any] cache by design for
# maximum flexibility and performance. The no-any-return errors are expected
# and suppressed at module level. Cache consumers should validate types.
# The call-overload errors are false positives from pandas-stubs not supporting
# List[int] indexing with .iloc correctly.
from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import Series


class IndicatorCache:
    """
    High-performance Indikator-Cache für aligned Multi-TF-Daten.

    Erwartete Struktur von multi_candle_data (bereits auf das Primary-Raster aligned):
        {
          "<TF>": {
             "bid": List[Candle|None],   # Länge == Anzahl Primary Bars
             "ask": List[Candle|None]
          },
          ...
        }

    Handling von None:
      - Bars ohne gültige Candle (carry_forward stale oder strict) werden als NaN abgebildet.
      - EMA/RSI/MACD/Bollinger/ATR rechnen vektorisiert über Pandas; NaNs werden korrekt propagiert.
    """

    def __init__(self, multi_candle_data: Dict[str, Dict[str, List[Any]]]) -> None:
        self._data = multi_candle_data
        # Cache-Strukturen
        self._df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (tf, side) -> DF
        self._ind_cache: Dict[Tuple[Any, ...], Any] = (
            {}
        )  # (name, tf, side, params...) -> pd.Series/tuple

        # DataFrames früh erstellen (einmalig, schnellster Zugriff später)
        for tf, sides in self._data.items():
            for side in ("bid", "ask"):
                if side in sides:
                    self._ensure_df(tf, side)

    # ---------- Low-level: OHLCV-Frames ----------

    def _ensure_df(self, tf: str, side: str) -> pd.DataFrame:
        key = (tf, side)
        df = self._df_cache.get(key)
        if df is not None:
            return df

        candles = self._data.get(tf, {}).get(side, [])
        n = len(candles)

        # Vor-alloziere Arrays; fülle mit NaN für fehlende Bars
        opens = np.full(n, np.nan, dtype="float64")
        highs = np.full(n, np.nan, dtype="float64")
        lows = np.full(n, np.nan, dtype="float64")
        closes = np.full(n, np.nan, dtype="float64")
        vols = np.full(n, np.nan, dtype="float64")

        # Schneller Tight-Loop
        for i, c in enumerate(candles):
            if c is None:
                continue
            # Candle kann Dict oder Objekt sein (dein Code unterstützt beides in utils)
            try:
                opens[i] = c["open"] if isinstance(c, dict) else float(c.open)
                highs[i] = c["high"] if isinstance(c, dict) else float(c.high)
                lows[i] = c["low"] if isinstance(c, dict) else float(c.low)
                closes[i] = c["close"] if isinstance(c, dict) else float(c.close)
                vols[i] = c["volume"] if isinstance(c, dict) else float(c.volume)
            except Exception:
                # Safety: wenn einzelne Felder fehlen → NaN (Bar wird automatisch ignoriert)
                pass

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": vols,
            }
        )
        self._df_cache[key] = df
        return df

    def _get_np_closes(self, tf: str, price_type: str) -> np.ndarray:
        """Liefert Close-Werte als float64 NumPy-Array (gecached)."""
        key = ("_np_closes", tf, price_type)
        arr = self._ind_cache.get(key)
        if arr is not None:
            return arr
        closes = self.get_closes(tf, price_type)
        try:
            out = closes.to_numpy(dtype="float64")
        except Exception:
            out = np.asarray(closes, dtype="float64")
        self._ind_cache[key] = out
        return out

    def get_df(self, tf: str, price_type: str = "bid") -> pd.DataFrame:
        """Gibt das OHLCV-DataFrame (aligned, mit NaNs) zurück (kopierfrei)."""
        return self._ensure_df(tf, price_type)

    def get_closes(self, tf: str, price_type: str = "bid") -> pd.Series:
        """Schnellzugriff auf die Close-Series."""
        return self._ensure_df(tf, price_type)["close"]

    # ---------- Helpers ----------

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        if period <= 0:
            raise ValueError("EMA period must be > 0")
        # TradingView/MT5-ähnlich: ewm ohne adjust, NaNs bleiben NaNs
        return series.ewm(span=period, adjust=False).mean()

    # ---------- Public Indicator APIs (vektorisiert + gecached) ----------

    def ema(self, tf: str, price_type: str, period: int) -> pd.Series:
        key = ("ema", tf, price_type, int(period))
        s = self._ind_cache.get(key)
        if s is not None:
            return s
        closes = self.get_closes(tf, price_type)
        s = self._ema(closes, period)
        self._ind_cache[key] = s
        return s

    def _stepwise_indices(self, tf: str, price_type: str) -> List[int]:
        candles = self._data.get(tf, {}).get(price_type, [])
        if not candles:
            return []
        new_idx: List[int] = []
        prev = None
        for i, c in enumerate(candles):
            if c is None:
                continue
            if (prev is None) or (c is not prev):
                new_idx.append(i)
                prev = c
        return new_idx

    def ema_stepwise(self, tf: str, price_type: str, period: int) -> pd.Series:
        """
        EMA nur bei *neuer HTF-Bar* fortschreiben und dann auf das Primary-Raster vorwärts füllen.
        Verhindert das künstliche „Ziehen“ der EMA durch carry_forward-Wiederholungen.
        """
        key = ("ema_stepwise", tf, price_type, int(period))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        # Aligned Candles (Candle|None) und zugehörige Close-Serie (NaN bei None)
        closes = self.get_closes(tf, price_type)
        if closes.empty:
            s = pd.Series([], dtype="float64")
            self._ind_cache[key] = s
            return s

        new_idx = self._stepwise_indices(tf, price_type)
        if not new_idx:
            # nichts Verwertbares -> komplett NaN
            out = pd.Series(np.nan, index=closes.index)
            self._ind_cache[key] = out
            return out

        # Reduzierte Serie: genau ein Close pro HTF-Bar
        reduced = closes.iloc[new_idx]
        # EMA auf der reduzierten Serie (ein echter "D1-Schritt" je Tag)
        reduced_ema = self._ema(reduced, period)

        # Auf vollständige Länge bringen: nur an new_idx schreiben und vorwärts füllen
        full = pd.Series(np.nan, index=closes.index, dtype="float64")
        full.iloc[new_idx] = reduced_ema.values
        full = full.ffill()

        self._ind_cache[key] = full
        return full

    def sma(self, tf: str, price_type: str, period: int) -> pd.Series:
        key = ("sma", tf, price_type, int(period))
        s = self._ind_cache.get(key)
        if s is not None:
            return s
        closes = self.get_closes(tf, price_type)
        s = closes.rolling(window=int(period), min_periods=int(period)).mean()
        self._ind_cache[key] = s
        return s

    def rsi(self, tf: str, price_type: str, period: int = 14) -> pd.Series:
        key = ("rsi", tf, price_type, int(period))
        r = self._ind_cache.get(key)
        if r is not None:
            return r
        closes = self.get_closes(tf, price_type)
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        r = 100 - (100 / (1 + rs))
        self._ind_cache[key] = r
        return r

    def macd(
        self,
        tf: str,
        price_type: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        key = (
            "macd",
            tf,
            price_type,
            int(fast_period),
            int(slow_period),
            int(signal_period),
        )
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        self._ind_cache[key] = (macd_line, signal_line)
        return self._ind_cache[key]

    def roc(self, tf: str, price_type: str, period: int = 14) -> pd.Series:
        """
        Rate of Change in %, vollständig als Serie:
          ROC_t = ((Close_t - Close_{t-period}) / Close_{t-period}) * 100
        """
        key = ("roc", tf, price_type, int(period))
        s = self._ind_cache.get(key)
        if s is not None:
            return s
        closes = self.get_closes(tf, price_type)
        prev = closes.shift(int(period))
        s = (closes - prev) / prev * 100.0
        self._ind_cache[key] = s
        return s

    def dmi(
        self, tf: str, price_type: str, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Directional Movement Index:
          gibt (+DI, -DI, ADX) als drei Serien zurück (Wilder/EWMA-Glättung).
        """
        key = ("dmi", tf, price_type, int(period))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        df = self.get_df(tf, price_type)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1, skipna=True)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100.0
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        self._ind_cache[key] = (plus_di, minus_di, adx)
        return self._ind_cache[key]

    def bollinger(
        self,
        tf: str,
        price_type: str,
        period: int = 20,
        std_factor: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        key = ("bb", tf, price_type, int(period), float(std_factor))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached
        closes = self.get_closes(tf, price_type)
        mid = closes.rolling(window=period, min_periods=period).mean()
        std = closes.rolling(window=period, min_periods=period).std()
        upper = mid + std_factor * std
        lower = mid - std_factor * std
        self._ind_cache[key] = (upper, mid, lower)
        return self._ind_cache[key]

    def bollinger_stepwise(
        self,
        tf: str,
        price_type: str,
        period: int = 20,
        std_factor: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        key = ("bb_stepwise", tf, price_type, int(period), float(std_factor))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        if closes.empty:
            empty = (
                pd.Series([], dtype="float64"),
                pd.Series([], dtype="float64"),
                pd.Series([], dtype="float64"),
            )
            self._ind_cache[key] = empty
            return empty

        new_idx = self._stepwise_indices(tf, price_type)
        if not new_idx:
            nan_series: pd.Series[float] = pd.Series(
                np.nan, index=closes.index, dtype="float64"
            )
            empty_tuple: Tuple[pd.Series[Any], pd.Series[Any], pd.Series[Any]] = (
                nan_series.copy(),
                nan_series.copy(),
                nan_series.copy(),
            )
            self._ind_cache[key] = empty_tuple
            return empty_tuple

        reduced = closes.iloc[new_idx]
        mid_reduced = reduced.rolling(window=period, min_periods=period).mean()
        std_reduced = reduced.rolling(window=period, min_periods=period).std()
        upper_reduced = mid_reduced + std_factor * std_reduced
        lower_reduced = mid_reduced - std_factor * std_reduced

        def _expand(series: pd.Series) -> pd.Series:
            full = pd.Series(np.nan, index=closes.index, dtype="float64")
            full.iloc[new_idx] = series.to_numpy()
            return full.ffill()

        result = (
            _expand(upper_reduced),
            _expand(mid_reduced),
            _expand(lower_reduced),
        )
        self._ind_cache[key] = result
        return result

    def atr(self, tf: str, price_type: str, period: int = 14) -> pd.Series:
        """
        Wilder-ATR (Bloomberg/TradingView-kompatibel) als komplette Serie.
        price_type steuert die Quelle (bid/ask).
        Definition:
          ATR_0 = SMA(TR[0:period]),
          ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
        """
        key = ("atr", tf, price_type, int(period))
        s = self._ind_cache.get(key)
        if s is not None:
            return s
        df = self.get_df(tf, price_type)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1, skipna=True)

        n = len(tr)
        out = np.full(n, np.nan, dtype="float64")
        tr_vals = tr.to_numpy(dtype=float)

        if period <= 0 or n == 0:
            s = pd.Series(out, index=df.index)
            self._ind_cache[key] = s
            return s

        first_valid = int(np.argmax(~np.isnan(tr_vals)))
        if (
            first_valid + period <= n
            and np.isfinite(tr_vals[first_valid : first_valid + period]).all()
        ):
            # SMA-Startwert
            atr_prev = float(np.mean(tr_vals[first_valid : first_valid + period]))
            out[first_valid + period - 1] = atr_prev
            # Wilder-Glättung
            for i in range(first_valid + period, n):
                if np.isnan(tr_vals[i]):
                    out[i] = out[i - 1]  # carry forward bei Lücken
                    continue
                atr_prev = (atr_prev * (period - 1) + tr_vals[i]) / period
                out[i] = atr_prev

        s = pd.Series(out, index=df.index)
        self._ind_cache[key] = s
        return s

    def choppiness(self, tf: str, price_type: str, period: int = 14) -> pd.Series:
        key = ("chop", tf, price_type, int(period))
        s = self._ind_cache.get(key)
        if s is not None:
            return s
        df = self.get_df(tf, price_type)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr_sum = tr.rolling(window=period, min_periods=period).sum()
        high_max = high.rolling(window=period, min_periods=period).max()
        low_min = low.rolling(window=period, min_periods=period).min()
        rng = high_max - low_min
        rng = rng.replace(0.0, np.nan)

        chop = 100 * np.log10(atr_sum / rng) / np.log10(period)
        self._ind_cache[key] = chop
        return chop

    @staticmethod
    def _kalman_mean_from_series(series: pd.Series, R: float, Q: float) -> pd.Series:
        n = len(series)
        xhat = np.full(n, np.nan, dtype="float64")
        P = np.full(n, np.nan, dtype="float64")
        if n == 0:
            return pd.Series(
                xhat, index=series.index if hasattr(series, "index") else None
            )

        mask = pd.notna(series).to_numpy()
        if not mask.any():
            return pd.Series(xhat, index=series.index)

        first_idx = int(np.argmax(mask))
        if not mask[first_idx]:
            return pd.Series(xhat, index=series.index)

        xhat[first_idx] = float(series.iloc[first_idx])
        P[first_idx] = 1.0

        for k in range(first_idx + 1, n):
            meas = series.iloc[k]
            xhat_minus = xhat[k - 1]
            P_minus = (P[k - 1] if pd.notna(P[k - 1]) else 1.0) + Q
            if pd.notna(meas):
                K = P_minus / (P_minus + R)
                xhat[k] = xhat_minus + K * (float(meas) - xhat_minus)
                P[k] = (1.0 - K) * P_minus
            else:
                xhat[k] = np.nan
                P[k] = np.nan

        return pd.Series(xhat, index=series.index)

    def kalman_mean(
        self, tf: str, price_type: str, R: float = 0.01, Q: float = 1.0
    ) -> pd.Series:
        """
        Einfache 1D-Kalman-Filter-Schätzung des gleitenden Mittelwerts über die Close-Serie.
        NaNs werden sauber propagiert: Lücken bleiben NaN bis wieder Messungen vorliegen.
        """
        key = ("kalman_mean", tf, price_type, float(R), float(Q))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        km = self._kalman_mean_from_series(closes, R, Q)
        self._ind_cache[key] = km
        return km

    def kalman_zscore(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        R: float = 0.01,
        Q: float = 1.0,
    ) -> pd.Series:
        """
        Z-Score der Residuen (Close - KalmanMean), Normierung via rolling Std über 'window'.
        """
        key = ("kalman_z", tf, price_type, int(window), float(R), float(Q))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        km = self.kalman_mean(tf, price_type, R=R, Q=Q)
        resid = closes - km
        std = resid.rolling(window=window, min_periods=window).std()
        z = resid / std
        # konsistente NaN-Handhabung
        z[std.isna()] = np.nan
        self._ind_cache[key] = z
        return z

    def kalman_zscore_stepwise(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        R: float = 0.01,
        Q: float = 1.0,
    ) -> pd.Series:
        key = ("kalman_z_stepwise", tf, price_type, int(window), float(R), float(Q))
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        if closes.empty:
            empty = pd.Series([], dtype="float64")
            self._ind_cache[key] = empty
            return empty

        new_idx = self._stepwise_indices(tf, price_type)
        if not new_idx:
            nan_series = pd.Series(np.nan, index=closes.index, dtype="float64")
            self._ind_cache[key] = nan_series
            return nan_series

        reduced = closes.iloc[new_idx]
        km_reduced = self._kalman_mean_from_series(reduced, R, Q)
        resid_reduced = reduced - km_reduced
        std_reduced = resid_reduced.rolling(window=window, min_periods=window).std()
        z_reduced = resid_reduced / std_reduced
        z_reduced[std_reduced.isna()] = np.nan

        z_full = pd.Series(np.nan, index=closes.index, dtype="float64")
        z_full.iloc[new_idx] = z_reduced.to_numpy()
        z_full = z_full.ffill()

        self._ind_cache[key] = z_full
        return z_full

    def zscore(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        mean_source: str = "rolling",
        ema_period: Optional[int] = None,
    ) -> pd.Series:
        """
        Flexibler Z-Score:
          - mean_source="rolling": (x - SMA(window)) / STD(window)
          - mean_source="ema":     (x - EMA(ema_period)) / STD(window)
        """
        key = (
            "zscore",
            tf,
            price_type,
            int(window),
            str(mean_source),
            int(ema_period or -1),
        )
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type)
        if mean_source == "rolling":
            mean = closes.rolling(window=window, min_periods=window).mean()
        elif mean_source == "ema":
            if not ema_period:
                raise ValueError(
                    "ema_period muss gesetzt sein, wenn mean_source='ema' ist."
                )
            mean = self.ema(tf, price_type, ema_period)
        else:
            raise ValueError("mean_source muss 'rolling' oder 'ema' sein.")
        std = (closes - mean).rolling(window=window, min_periods=window).std()
        z = (closes - mean) / std
        z[std.isna()] = np.nan
        self._ind_cache[key] = z
        return z

    # ---------- Returns & GARCH Volatilität (leichtgewichtig, ohne externe Libs) ----------

    @staticmethod
    def _returns(series: pd.Series[Any], use_log: bool = True) -> pd.Series[Any]:
        """Berechnet 1-Lag Returns (log oder einfache), NaN-sicher."""
        s = series.astype("float64")
        if use_log:
            return pd.Series(np.log(s / s.shift(1)), index=series.index)
        return s.pct_change()

    def garch_volatility(
        self,
        tf: str,
        price_type: str,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: Optional[float] = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> pd.Series:
        """
        Leichtgewichtiges rekursives GARCH(1,1) auf Returns (in % skaliert).
          r_t = scale * return_t
          var_t = omega + alpha * eps_{t-1}^2 + beta * var_{t-1}
        Rückgabe: sigma_t in Return-Einheiten (d. h. gleiche Einheit wie return_t), NICHT in Preisen.
        """
        key = (
            "garch_vol",
            tf,
            price_type,
            float(alpha),
            float(beta),
            float(omega if omega is not None else -1.0),
            bool(use_log_returns),
            float(scale),
            int(min_periods),
            float(sigma_floor),
        )
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        if alpha + beta >= 1.0:
            # Skip/invalid → konsistent NaN zurückgeben, Cache befüllen
            closes = self.get_closes(tf, price_type)
            s = pd.Series(np.nan, index=closes.index, dtype="float64")
            self._ind_cache[key] = s
            # Optional sichtbar machen:
            # print(f"⚠️ GARCH skip: alpha+beta={alpha+beta:.4f} ≥ 1 (tf={tf}, {price_type})")
            return s

        closes = self.get_closes(tf, price_type)
        rets = self._returns(closes, use_log=use_log_returns)
        r = (rets * scale).astype("float64")  # z.B. *100 → Prozent-Returns

        n = len(r)
        if n == 0:
            out = pd.Series([], dtype="float64")
            self._ind_cache[key] = out
            return out

        r_vals = r.to_numpy()
        out_var = np.full(n, np.nan, dtype="float64")

        # Startindex (erste gültige Return-Observation)
        first_idx = int(np.argmax(pd.notna(r).to_numpy()))
        if not pd.notna(r.iloc[first_idx]):
            s = pd.Series(out_var, index=r.index)
            self._ind_cache[key] = s
            return s

        # Long-Run-Varianz für Initialisierung + ggf. omega
        lr_var = float(
            np.nanvar(r_vals[max(0, first_idx - 1000) : first_idx + 1])
        )  # robust
        if not np.isfinite(lr_var) or lr_var <= 0:
            lr_var = 1e-6
        _omega = (
            float(omega)
            if omega is not None
            else lr_var * max(1e-6, (1.0 - alpha - beta))
        )
        _omega = max(_omega, 0.0)

        # zentrierte Schocks (Mean ~ 0 annehmen, sonst geringe Schätzung über r.mean)
        mu = float(np.nanmean(r_vals[max(0, first_idx - 2000) : first_idx + 1]))
        eps_prev2 = (r_vals[first_idx] - mu) ** 2
        var_prev = max(lr_var, eps_prev2, sigma_floor**2)
        out_var[first_idx] = var_prev

        for i in range(first_idx + 1, n):
            ri = r_vals[i]
            if not np.isfinite(ri):
                out_var[i] = out_var[i - 1]
                continue
            # Rekursion
            var_t = _omega + alpha * eps_prev2 + beta * var_prev
            var_t = max(var_t, (sigma_floor**2))
            out_var[i] = var_t
            # Update Speicher
            eps_prev2 = (ri - mu) ** 2
            var_prev = var_t

        sigma = np.sqrt(out_var) / scale  # zurückskalieren in Return-Einheiten
        # Mindestanzahl an Beobachtungen bevor wir der Schätzung trauen:
        valid = pd.Series(
            np.arange(n) >= (first_idx + max(0, int(min_periods))), index=r.index
        )
        sigma_series = pd.Series(sigma, index=r.index).where(valid, other=np.nan)
        self._ind_cache[key] = sigma_series
        return sigma_series

    def kalman_garch_zscore(
        self,
        tf: str,
        price_type: str,
        R: float = 0.01,
        Q: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: Optional[float] = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> pd.Series:
        """
        Z-Score = (Close − KalmanMean) / (Preis-skalierte GARCH-Sigma)
        Näherung: sigma_price ≈ |Close| * sigma_return  (für kleine Returns).
        """
        key = (
            "kalman_garch_z",
            tf,
            price_type,
            float(R),
            float(Q),
            float(alpha),
            float(beta),
            float(omega if omega is not None else -1.0),
            bool(use_log_returns),
            float(scale),
            int(min_periods),
            float(sigma_floor),
        )
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        closes = self.get_closes(tf, price_type).astype("float64")
        km = self.kalman_mean(tf, price_type, R=R, Q=Q).astype("float64")
        resid = closes - km
        sigma_ret = self.garch_volatility(
            tf,
            price_type,
            alpha=alpha,
            beta=beta,
            omega=omega,
            use_log_returns=use_log_returns,
            scale=scale,
            min_periods=min_periods,
            sigma_floor=sigma_floor,
        )
        sigma_price = (closes.abs() * sigma_ret).replace(0.0, np.nan)
        z = resid / sigma_price
        z[(sigma_price.isna()) | (km.isna())] = np.nan
        self._ind_cache[key] = z
        return z

    def garch_volatility_local(
        self,
        tf: str,
        price_type: str,
        idx: int,
        lookback: int = 400,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: Optional[float] = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> pd.Series:
        """
        Lokale GARCH(1,1)-Volatilität auf Returns über ein capped Lookback-Fenster.

        Rückgabe: pd.Series (sigma der Returns) mit dem Index des Ausschnitts.
        """
        key = (
            "garch_vol_local",
            tf,
            price_type,
            int(idx),
            int(lookback),
            float(alpha),
            float(beta),
            float(omega if omega is not None else -1.0),
            bool(use_log_returns),
            float(scale),
            int(min_periods),
            float(sigma_floor),
        )
        cached = self._ind_cache.get(key)
        if cached is not None:
            return cached

        # Stabilitäts-Check
        try:
            a = float(alpha)
            b = float(beta)
        except Exception:
            a = alpha
            b = beta
        if (a + b) >= 1.0:
            out = pd.Series([], dtype="float64")
            self._ind_cache[key] = out
            return out

        closes_full = self.get_closes(tf, price_type)
        n_total = len(closes_full)
        if idx is None or idx < 0 or n_total == 0:
            out = pd.Series([], dtype="float64")
            self._ind_cache[key] = out
            return out

        # idx referenziert die aktuelle Kerze (0‑basiert). Für ein inklusives
        # Fenster bis *einschließlich* idx clampen wir idx in [0, n_total-1]
        # und setzen end_pos = idx_int + 1 (Python‑Slicing: end exklusiv).
        idx_int = min(max(int(idx), 0), n_total - 1)
        end_pos = idx_int + 1
        start_pos = max(0, end_pos - max(1, int(lookback)))
        window = closes_full.iloc[start_pos:end_pos]
        if window.empty:
            out = pd.Series([], dtype="float64")
            self._ind_cache[key] = out
            return out

        prices = window.to_numpy(dtype="float64")
        n = prices.size
        r_np = np.full(n, np.nan, dtype="float64")
        if n > 1:
            if use_log_returns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_np[1:] = np.log(prices[1:] / prices[:-1]) * float(scale)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_np[1:] = ((prices[1:] / prices[:-1]) - 1.0) * float(scale)

        valid_mask = np.isfinite(r_np)
        if not valid_mask.any():
            out = pd.Series(np.nan, index=window.index, dtype="float64")
            self._ind_cache[key] = out
            return out

        first_idx = int(np.argmax(valid_mask))
        out_var = np.full(n, np.nan, dtype="float64")

        lr_slice = r_np[max(0, first_idx - 1000) : first_idx + 1]
        lr_var = float(np.nanvar(lr_slice)) if lr_slice.size > 0 else 1e-6
        if not np.isfinite(lr_var) or lr_var <= 0.0:
            lr_var = 1e-6

        _omega = (
            float(omega)
            if omega is not None
            else lr_var * max(1e-6, (1.0 - float(alpha) - float(beta)))
        )
        _omega = max(_omega, 0.0)

        mu_slice = r_np[max(0, first_idx - 2000) : first_idx + 1]
        mu = float(np.nanmean(mu_slice)) if mu_slice.size > 0 else 0.0
        eps_prev2 = (r_np[first_idx] - mu) ** 2
        sigma_floor_sq = float(sigma_floor) ** 2
        var_prev = max(lr_var, eps_prev2, sigma_floor_sq)
        out_var[first_idx] = var_prev

        for i in range(first_idx + 1, n):
            ri = r_np[i]
            if not np.isfinite(ri):
                out_var[i] = out_var[i - 1]
                continue
            var_t = _omega + float(alpha) * eps_prev2 + float(beta) * var_prev
            var_t = max(var_t, sigma_floor_sq)
            out_var[i] = var_t
            eps_prev2 = (ri - mu) ** 2
            var_prev = var_t

        sigma = np.sqrt(out_var) / float(scale)
        valid_len = np.arange(n) >= (first_idx + max(0, int(min_periods)))
        sigma_series = pd.Series(sigma, index=window.index)
        sigma_series = sigma_series.where(valid_len, other=np.nan)
        self._ind_cache[key] = sigma_series
        return sigma_series

    def vol_cluster_series(
        self,
        tf: str,
        price_type: str,
        idx: int,
        feature: str,
        atr_length: int,
        garch_lookback: int,
        garch_alpha: float = 0.05,
        garch_beta: float = 0.90,
        garch_omega: Optional[float] = None,
        garch_use_log_returns: bool = True,
        garch_scale: float = 100.0,
        garch_min_periods: int = 50,
        garch_sigma_floor: float = 1e-6,
    ) -> Optional[pd.Series]:
        """
        Liefert die Feature-Serie für Volatilitäts-Cluster.
        Aktuell: ATR (in Punkten) oder lokale GARCH-Volatilität.
        """
        feat = str(feature or "").strip().lower()
        if feat == "atr_points":
            return self.atr(tf, price_type, atr_length)
        return self.garch_volatility_local(
            tf,
            price_type,
            idx=idx,
            lookback=garch_lookback,
            alpha=garch_alpha,
            beta=garch_beta,
            omega=garch_omega,
            use_log_returns=garch_use_log_returns,
            scale=garch_scale,
            min_periods=garch_min_periods,
            sigma_floor=garch_sigma_floor,
        )

    def kalman_garch_zscore_local(
        self,
        tf: str,
        price_type: str,
        idx: int,
        lookback: int = 400,
        R: float = 0.01,
        Q: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: Optional[float] = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> Optional[float]:
        """
        Lokale Variante des Kalman+GARCH-Z‑Scores am gegebenen Index.

        Berechnet Z nur aus den letzten `lookback` Bars bis einschließlich `idx`.
        Dadurch hängt das Ergebnis nur von einem festen lokalen Fenster ab und ist
        robust gegenüber unterschiedlichen Warmup‑Fenstern.

        Rückgabe: float(Z) oder None, wenn nicht genügend Daten vorhanden sind.
        """
        # Parameter-Validierung
        try:
            a = float(alpha)
            b = float(beta)
        except Exception:
            a = alpha
            b = beta
        if (a + b) >= 1.0:
            return None

        try:
            closes_full_np = self._get_np_closes(tf, price_type)
        except Exception:
            return None

        n = len(closes_full_np)
        if idx is None or idx < 0 or idx >= n:
            return None

        start = max(0, int(idx) - int(lookback) + 1)
        closes_np = closes_full_np[start : idx + 1]
        if closes_np.size < 2:
            return None

        # Kalman‑Mittelwert auf dem lokalen Segment
        # _kalman_mean_from_series erwartet eine pd.Series
        try:
            import pandas as pd  # noqa: F401

            closes = pd.Series(closes_np)
        except Exception:
            return None
        km_seg = self._kalman_mean_from_series(closes, R, Q)
        km_last = km_seg.iloc[-1]
        if not pd.notna(km_last):
            return None
        close_last = float(closes_np[-1])
        resid_last = float(close_last - km_last)

        # GARCH(1,1) Volatilität der Returns (lokal)
        # Schnelle NumPy-Returns (identisch zum Segment-Verhalten: erstes Element NaN)
        s = closes_np
        r_np = np.empty_like(s, dtype="float64")
        r_np[0] = np.nan
        if use_log_returns:
            with np.errstate(divide="ignore", invalid="ignore"):
                r_np[1:] = np.log(s[1:] / s[:-1]) * float(scale)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                r_np[1:] = (s[1:] / s[:-1] - 1.0) * float(scale)

        # Index des ersten gültigen Returns im Segment
        valid_mask = np.isfinite(r_np)
        if not valid_mask.any():
            return None
        first_idx = int(np.argmax(valid_mask))

        seg_len = int(r_np.size)
        # Früher Abbruch: Mindest‑Periode kann nicht erreicht werden
        if (seg_len - 1) - first_idx < int(min_periods):
            return None
        out_var = np.full(seg_len, np.nan, dtype="float64")

        # Long‑Run‑Varianz (nur auf dem Segment geschätzt) und omega Default
        lr_slice = r_np[max(0, first_idx - 1000) : first_idx + 1]
        lr_var = float(np.nanvar(lr_slice)) if lr_slice.size > 0 else 1e-6
        if not np.isfinite(lr_var) or lr_var <= 0.0:
            lr_var = 1e-6
        _omega = (
            float(omega)
            if omega is not None
            else lr_var * max(1e-6, (1.0 - float(alpha) - float(beta)))
        )
        _omega = max(_omega, 0.0)

        mu = float(np.nanmean(lr_slice)) if lr_slice.size > 0 else 0.0
        eps_prev2 = (r_np[first_idx] - mu) ** 2
        var_prev = max(lr_var, eps_prev2, float(sigma_floor) ** 2)
        out_var[first_idx] = var_prev

        for i in range(first_idx + 1, seg_len):
            ri = r_np[i]
            if not np.isfinite(ri):
                out_var[i] = out_var[i - 1]
                continue
            var_t = _omega + float(alpha) * eps_prev2 + float(beta) * var_prev
            var_t = max(var_t, float(sigma_floor) ** 2)
            out_var[i] = var_t
            eps_prev2 = (ri - mu) ** 2
            var_prev = var_t

        sigma_ret = np.sqrt(out_var) / float(scale)
        # Mindestens min_periods Beobachtungen nach first_idx erforderlich
        valid = np.arange(seg_len) >= (first_idx + max(0, int(min_periods)))
        sigma_ret_last = float(sigma_ret[-1]) if valid[-1] else float("nan")
        if not np.isfinite(sigma_ret_last) or sigma_ret_last <= 0.0:
            return None

        sigma_price_last = abs(float(close_last)) * sigma_ret_last
        if not np.isfinite(sigma_price_last) or sigma_price_last == 0.0:
            return None

        z_now = resid_last / sigma_price_last
        return float(z_now) if np.isfinite(z_now) else None

    # ---------- Convenience single-value Getter ----------

    def at(self, series: pd.Series, index: int) -> Optional[float]:
        """Schneller Wertzugriff (mit Bounds-Check)."""
        if index < 0 or index >= len(series):
            return None
        val = series.iloc[index]
        try:
            return float(val) if pd.notna(val) else None
        except Exception:
            return None


# -------- Globaler Cache-Pool (wiederverwendet IndicatorCache-Instanzen) -------
#
# WICHTIG: WeakValueDictionary entfernt Einträge automatisch, sobald die
# IndicatorCache-Instanz nicht mehr referenziert wird.
# Das verhindert ein unbounded Wachstum der Key-Map (klassische Leak-Quelle
# bei handgerollten weakref.ref-Pools).
_INDCACHE_POOL: (
    "weakref.WeakValueDictionary[Tuple[Tuple[str, str, int, int], ...], IndicatorCache]"
) = weakref.WeakValueDictionary()


def indicator_cache_pool_size() -> int:
    """Gibt die aktuelle Größe des globalen IndicatorCache-Pools zurück."""

    return len(_INDCACHE_POOL)


def clear_indicator_cache_pool() -> None:
    """Leert den globalen IndicatorCache-Pool (Debug/Memory-Recovery)."""

    _INDCACHE_POOL.clear()


def _signature_from_multi(
    multi_candle_data: Dict[str, Dict[str, List[Any]]],
) -> Tuple[Tuple[str, str, int, int], ...]:
    """
    Erzeugt eine stabile, kleine Signatur:
      [(tf, side, id(list), len(list)), ...] sortiert nach (tf, side).
    Objekt-IDs + Längen genügen, da Listen-Objekte für ein Fenster konstant bleiben.
    """
    sig: List[Tuple[str, str, int, int]] = []
    for tf in sorted(multi_candle_data.keys()):
        sides = multi_candle_data[tf]
        for side in ("bid", "ask"):
            seq = sides.get(side, [])
            sig.append((tf, side, id(seq), len(seq)))
    return tuple(sig)


def get_cached_indicator_cache(
    multi_candle_data: Dict[str, Dict[str, List[Any]]],
) -> "IndicatorCache":
    """
    Liefert eine wiederverwendete IndicatorCache-Instanz für identische Datenpakete.
    """
    sig = _signature_from_multi(multi_candle_data)
    inst = _INDCACHE_POOL.get(sig)
    if inst is not None:
        return inst
    inst = IndicatorCache(multi_candle_data)
    _INDCACHE_POOL[sig] = inst
    return inst
