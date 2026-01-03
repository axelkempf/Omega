# core/broker/broker_utils.py

"""
Broker-unabhängige Hilfsfunktionen für Strategien.
Alle Funktionen verwenden ausschließlich das BrokerInterface zur Kommunikation.
Ermöglicht saubere Trennung und Nutzung in MT5, Backtests oder anderen Brokern.

Überarbeitung:
- Konsistentere Rückgaben (None statt gemischter Exceptions/None; gezielte ValueError nur bei klarer Fehlbedienung)
- Robustere NaN/Zero-Division-Guards bei Indikatoren
- Bugfix: calculate_macd gab 3x None zurück, Signatur erwartet 2 Werte
- Bugfix/Implementierung: median_atr_ema jetzt echte ATR‑Berechnung + korrekte Guards
- Kleinere Performance-/Lesbarkeitsverbesserungen
- Unbenutzte Importe entfernt
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.infra.logging.log_service import log_service
from hf_engine.infra.monitoring.telegram_bot import send_telegram_message

# ---------------------------------------------------------------------------
# Generische Utilities -------------------------------------------------------
# ---------------------------------------------------------------------------


def get_pip_size(symbol: str) -> float:
    """Gibt die Pip-Größe für das Symbol zurück (z. B. 0.01 für JPY-Paare)."""
    if not symbol:
        return 0.0001
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def get_range_high_low(
    data_provider,
    symbol: str,
    from_time: datetime,
    to_time: datetime,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Berechnet den High/Low-Range eines Zeitintervalls.

    Funktioniert mit MT5DataProvider (liefert DataFrame) und RemoteDataProvider (liefert JSON).
    """
    try:
        bars = data_provider.get_rates_range(
            symbol, timeframe="M1", start=from_time, end=to_time
        )

        # Auto-Konvertierung bei RemoteDataProvider
        if isinstance(bars, list):
            df = pd.DataFrame(bars)
        else:
            df = bars  # MT5DataProvider gibt bereits ein DataFrame zurück

        if (
            df is None
            or df.empty
            or "high" not in df.columns
            or "low" not in df.columns
        ):
            log_service.log_system(
                f"[AsiaRange Fehler] Kein gültiger Datenbereich für {symbol} ({from_time}–{to_time})"
            )
            send_telegram_message(f"⚠️ Keine Asia-Range-Daten für {symbol}")
            return None, None

        hi = pd.to_numeric(df["high"], errors="coerce").max()
        lo = pd.to_numeric(df["low"], errors="coerce").min()
        if pd.isna(hi) or pd.isna(lo):
            log_service.log_system(
                f"[AsiaRange Fehler] Ungültige High/Low-Werte für {symbol} ({from_time}–{to_time})"
            )
            return None, None
        return float(hi), float(lo)

    except Exception as e:
        log_service.log_exception(f"[AsiaRange Fehler] {symbol}:", e)
        return None, None


def _col_as_float(candles, col: str) -> pd.Series:
    """
    Liefert die angefragte Spalte als float‑Series, egal ob candles ein
    DataFrame (mit Spalten 'open','high','low','close') oder eine Series
    aus dicts/Objs ist.
    """
    if isinstance(candles, pd.DataFrame):
        if col not in candles.columns:
            raise KeyError(f"Spalte '{col}' fehlt im DataFrame.")
        return pd.to_numeric(candles[col], errors="coerce")

    # Series von dict/Objekten:
    def _get(attr: str, x):
        try:
            return x[attr] if isinstance(x, dict) else getattr(x, attr)
        except Exception:
            return np.nan

    return pd.to_numeric(candles.apply(lambda x: _get(col, x)), errors="coerce")


# ---------------------------------------------------------------------------
# Technische Indikatoren -----------------------------------------------------
# ---------------------------------------------------------------------------


def _is_valid_series(series: pd.Series, min_len: int) -> bool:
    return (
        isinstance(series, pd.Series)
        and len(series) >= min_len
        and series.notna().any()
    )


def calculate_sma(series: pd.Series, period: int) -> Optional[float]:
    """Berechnet den Simple Moving Average für die letzten *period* Werte."""
    if not _is_valid_series(series, period):
        return None
    window = series.tail(period).astype(float)
    if window.isna().all():
        return None
    return float(window.mean())


def calculate_rsi(series: pd.Series, period: int) -> Optional[pd.Series]:
    if not _is_valid_series(series, period + 1):
        return None

    s = series.astype(float)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # EWM (Wilder-ähnlich)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Division-by-zero guard
    avg_loss_safe = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(prices: List[float], period: int) -> List[Optional[float]]:
    """
    Berechnet den EMA mit SMA als Startwert, um mit MT5 / TradingView übereinzustimmen.
    Rückgabe-Liste enthält für die ersten period-1 Elemente None, an Index period-1 den initialen SMA.
    """
    if prices is None or len(prices) < period:
        raise ValueError("Nicht genug Datenpunkte für EMA")

    ema_values: List[Optional[float]] = [None] * (period - 1)
    sma = sum(prices[:period]) / period
    ema_values.append(float(sma))

    alpha = 2 / (period + 1)
    for price in prices[period:]:
        prev_ema = ema_values[-1]
        # prev_ema kann hier nicht None sein
        new_ema = alpha * float(price) + (1 - alpha) * float(prev_ema)  # type: ignore[arg-type]
        ema_values.append(new_ema)

    return ema_values


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Berechnet den MACD und die Signallinie aus einer Preisreihe.
    Standardwerte: fast=12, slow=26, signal=9 (wie bei TradingView).

    Rückgabe:
        macd_line, signal_line (je pd.Series) oder (None, None), wenn zu wenige Daten.
    """
    if not _is_valid_series(series, slow_period + signal_period):
        return None, None

    s = series.astype(float)
    ema_fast = s.ewm(span=fast_period, adjust=False).mean()
    ema_slow = s.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line


def calculate_atr(candles: pd.Series | pd.DataFrame, period: int) -> Optional[float]:
    """
    Wilder-ATR (Bloomberg/TradingView-kompatibel):
      1) TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
      2) ATR_0 = SMA(TR[0:period])
      3) ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
    Rückgabe: letzter ATR-Wert.
    """
    if candles is None or len(candles) < period:
        return None

    highs = _col_as_float(candles, "high")
    lows = _col_as_float(candles, "low")
    closes = _col_as_float(candles, "close")
    prev_closes = closes.shift(1)

    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    tr_vals = tr.to_numpy(dtype=float)
    # ersten gültigen Index finden
    first_valid = int(np.argmax(~np.isnan(tr_vals)))
    if not np.isfinite(tr_vals[first_valid : first_valid + period]).all():
        # Falls in der Startspanne nicht genug gültige TRs vorhanden sind
        # (sollte praktisch kaum auftreten), gib None zurück.
        return None

    # SMA-Start
    atr_prev = float(np.mean(tr_vals[first_valid : first_valid + period]))
    idx = first_valid + period
    # Wilder-Glättung
    for i in range(idx, len(tr_vals)):
        if np.isnan(tr_vals[i]):
            continue
        atr_prev = (atr_prev * (period - 1) + tr_vals[i]) / period

    return float(atr_prev) if np.isfinite(atr_prev) else None


def calculate_atr_series(candles: pd.Series | pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder-ATR als vollständige Serie.

    Logik:
      1) TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
      2) ATR_0 = SMA(TR[0:period])
      3) ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period

    Rückgabe:
      pd.Series gleicher Länge wie candles.index mit NaN vor dem
      Startfenster; ab dem ersten vollständigen Fenster kontinuierliche ATR-Werte.
    """
    if candles is None or len(candles) < period:
        # leere Serie mit sinnvollem Index, falls möglich
        if isinstance(candles, pd.DataFrame):
            return pd.Series(np.nan, index=candles.index, dtype="float64")
        if isinstance(candles, pd.Series):
            return pd.Series(np.nan, index=candles.index, dtype="float64")
        return pd.Series([], dtype="float64")

    highs = _col_as_float(candles, "high")
    lows = _col_as_float(candles, "low")
    closes = _col_as_float(candles, "close")
    prev_closes = closes.shift(1)

    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    tr_vals = tr.to_numpy(dtype="float64")
    n = len(tr_vals)
    if n == 0:
        return pd.Series([], index=tr.index, dtype="float64")

    # ersten gültigen Index finden
    first_valid = int(np.argmax(~np.isnan(tr_vals)))
    if (
        first_valid + period > n
        or not np.isfinite(tr_vals[first_valid : first_valid + period]).all()
    ):
        # Zu wenig gültige TR-Werte im Startfenster -> Serie mit NaNs
        return pd.Series(np.nan, index=tr.index, dtype="float64")

    atr_vals = np.full(n, np.nan, dtype="float64")

    # SMA-Start
    atr_prev = float(np.mean(tr_vals[first_valid : first_valid + period]))
    idx = first_valid + period
    # ATR_0 gehört auf Index idx-1 (kompatibel zu calculate_atr-Definition)
    if idx - 1 < n:
        atr_vals[idx - 1] = atr_prev

    # Wilder-Glättung
    for i in range(idx, n):
        if np.isnan(tr_vals[i]):
            atr_vals[i] = atr_prev
            continue
        atr_prev = (atr_prev * (period - 1) + tr_vals[i]) / period
        atr_vals[i] = atr_prev

    return pd.Series(atr_vals, index=tr.index, dtype="float64")


def calculate_choppiness_index(
    candles: pd.Series | pd.DataFrame, period: int = 14
) -> Optional[float]:
    if candles is None or len(candles) < period + 1:
        return None

    highs = _col_as_float(candles, "high")
    lows = _col_as_float(candles, "low")
    closes = _col_as_float(candles, "close")
    prev_closes = closes.shift(1)

    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_sum = tr.rolling(window=period).sum()
    high_max = highs.rolling(window=period).max()
    low_min = lows.rolling(window=period).min()

    range_ = (high_max - low_min).replace(0.0, np.nan)
    chop = 100 * np.log10(atr_sum / range_) / np.log10(period)
    last = chop.iloc[-1]
    return float(last) if pd.notna(last) else None


def _kmeans_1d(values: np.ndarray, k: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Einfache 1D-K-Means-Implementierung (ohne externe Abhängigkeiten).
    Rückgabe: (cluster_centers, labels) oder None bei leeren Daten.
    """
    if values.size == 0:
        return None
    unique_vals = np.unique(values)
    k = max(1, min(k, unique_vals.size))
    if k == 1:
        center = np.array([float(np.mean(values))], dtype="float64")
        labels = np.zeros(values.shape[0], dtype=int)
        return center, labels

    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    init = np.quantile(values, quantiles)
    centers = init.astype("float64")
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


def calculate_vol_cluster_state(
    series: pd.Series | np.ndarray | List[float],
    window: int,
    k: int,
    min_points: int = 60,
    log_transform: bool = True,
) -> Dict[str, Any]:
    """
    Berechnet den Zustand eines Intraday‑Volatilitäts‑Clusters.

    Parameter:
      - series: Zeitreihe des Volatilitätsmaßes (z.B. ATR-Punkte oder GARCH-Sigma)
      - window: Größe des Auswertefensters (letzte N Punkte)
      - k: Anzahl Cluster
      - min_points: Mindestanzahl an Punkten insgesamt
      - log_transform: ob vor dem Clustering log-transformiert wird

    Rückgabe:
      {
        "status": "ok" | <Fehler-String>,
        "sample_size": int,
        "labels": np.ndarray[int] oder None,
        "mapping": {cluster_index -> label_name},
        "centers": List[float],
        "centers_log": List[float] | None,
        "sigma": float  # letzter Wert im Fenster
      }
    """
    try:
        if isinstance(series, pd.Series):
            ser = series.copy()
        else:
            ser = pd.Series(series)
    except Exception:
        return {"status": "series_unavailable", "sample_size": 0}

    ser = ser.dropna()
    sample_size = int(len(ser))
    if sample_size < min_points:
        return {"status": "insufficient_points", "sample_size": sample_size}

    tail = ser.tail(int(window))
    if len(tail) < int(k):
        return {"status": "insufficient_unique", "sample_size": sample_size}

    values = tail.to_numpy(dtype="float64")
    if log_transform:
        values = np.log(np.clip(values, 1e-12, None))

    clusters = _kmeans_1d(values, int(k))
    if clusters is None:
        return {"status": "clustering_failed", "sample_size": sample_size}

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
    return {
        "status": "ok",
        "sample_size": sample_size,
        "labels": labels.copy(),
        "mapping": mapping,
        "centers": centers_sorted,
        "centers_log": centers_log,
        "sigma": sigma,
    }


def calculate_bollinger_bands(
    series: pd.Series, period: int = 20, std_factor: float = 2.0
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Berechnet Bollinger-Bänder als vollständige Series.

    Rückgabe:
        upper_band, middle_band (SMA), lower_band
        oder (None, None, None), wenn zu wenige Daten.
    """
    if not _is_valid_series(series, period):
        return None, None, None

    s = series.astype(float)
    sma = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()

    upper_band = sma + std_factor * std
    lower_band = sma - std_factor * std

    return upper_band, sma, lower_band


def calculate_roc(series: pd.Series, period: int = 14) -> Optional[pd.Series]:
    """Berechnet den ROC als vollständige Serie (nicht nur letzten Wert)."""
    if not _is_valid_series(series, period + 1):
        return None
    s = series.astype(float)
    previous_price = s.shift(period)
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = ((s - previous_price) / previous_price) * 100
    return roc


def calculate_donchian_channel(
    series_high: pd.Series, series_low: pd.Series, period: int = 20
) -> Optional[Tuple[float, float, float]]:
    if (
        not isinstance(series_high, pd.Series)
        or not isinstance(series_low, pd.Series)
        or len(series_high) < period
        or len(series_low) < period
    ):
        return None

    uh = series_high.astype(float).rolling(window=period).max()
    lo = series_low.astype(float).rolling(window=period).min()
    mid = (uh + lo) / 2

    return float(uh.iloc[-1]), float(mid.iloc[-1]), float(lo.iloc[-1])


def calculate_dmi(
    candles: pd.Series | pd.DataFrame, period: int = 14
) -> Optional[Tuple[float, float, float]]:
    if candles is None or len(candles) < period + 1:
        return None

    highs = _col_as_float(candles, "high")
    lows = _col_as_float(candles, "low")
    closes = _col_as_float(candles, "close")

    prev_highs = highs.shift(1)
    prev_lows = lows.shift(1)
    prev_closes = closes.shift(1)

    plus_dm_raw = highs - prev_highs
    minus_dm_raw = prev_lows - lows

    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where(
        (minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0
    )

    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr_safe = atr.replace(0.0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (abs(plus_di - minus_di) / denom) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    last_plus = plus_di.iloc[-1]
    last_minus = minus_di.iloc[-1]
    last_adx = adx.iloc[-1]

    if pd.isna(last_plus) or pd.isna(last_minus) or pd.isna(last_adx):
        return None
    return float(last_plus), float(last_minus), float(last_adx)


def calculate_zscore(
    series: pd.Series, window: int = 100, mean_series: Optional[pd.Series] = None
) -> Optional[pd.Series]:
    if not _is_valid_series(series, window):
        return None

    s = series.astype(float)
    if mean_series is None:
        mean_series = s.rolling(window=window).mean()
    std = s.rolling(window=window).std()

    valid = s.notna() & mean_series.notna() & std.notna()
    result = (s - mean_series) / std
    result = result.where(valid, np.nan)
    return result


def kalman_filter(series: pd.Series, R: float = 0.01, Q: float = 1.0) -> pd.Series:
    """
    Einfache 1D-Kalman-Filter-Implementierung für gleitenden Mittelwert.
    R: Messfehler (Noise)
    Q: Prozessrauschen (Smoothness)
    """
    if not _is_valid_series(series, 1):
        return pd.Series([], dtype=float)

    s = pd.to_numeric(series, errors="coerce").ffill()
    n = len(s)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = s.iloc[0]
    P[0] = 1.0
    for k in range(1, n):
        xhatminus = xhat[k - 1]
        Pminus = P[k - 1] + Q
        # Schutz gegen Division durch 0 oder numerische Instabilität
        denominator = Pminus + R
        if denominator <= 0 or not np.isfinite(denominator):
            K = 0.0
        else:
            K = Pminus / denominator
        xhat[k] = xhatminus + K * (s.iloc[k] - xhatminus)
        P[k] = (1 - K) * Pminus
    return pd.Series(xhat, index=s.index)


def calculate_kalman_zscore(
    series: pd.Series,
    window: int = 100,
    R: float = 0.01,
    Q: float = 1.0,
) -> pd.Series:
    """
    Berechnet den Z-Score mit Kalman-geglättetem Mittelwert.
    """
    kalman_mean = kalman_filter(series, R=R, Q=Q)
    residuen = series.astype(float) - kalman_mean
    rolling_std = residuen.rolling(window=window, min_periods=window).std()
    zscore = residuen / rolling_std
    zscore[rolling_std.isna()] = np.nan
    return zscore


def _returns(series: pd.Series, use_log: bool = True) -> pd.Series:
    """Berechnet 1-Lag Returns (log oder einfach), NaN-sicher."""
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(s / s.shift(1))
    return s.pct_change()


def calculate_garch_volatility(
    series: pd.Series,
    alpha: float = 0.05,
    beta: float = 0.90,
    omega: Optional[float] = None,
    use_log_returns: bool = True,
    scale: float = 100.0,
    min_periods: int = 50,
    sigma_floor: float = 1e-6,
) -> pd.Series:
    """
    Rekursives GARCH(1,1) auf Returns (in scale-Einheiten, z. B. Prozent) – leichtgewichtig, ohne externe Libs.

    Rückgabe: sigma_t (StDev der Returns) als Series – gleiche Länge/Index wie input.
    Ungültige Bereiche (vor min_periods) werden mit NaN gefüllt.
    """
    try:
        closes = pd.to_numeric(series, errors="coerce").astype("float64")
    except Exception:
        return pd.Series([], dtype="float64")

    n = len(closes)
    if n == 0:
        return pd.Series([], dtype="float64")

    if alpha + beta >= 1.0:
        return pd.Series(np.nan, index=closes.index, dtype="float64")

    r = (_returns(closes, use_log=use_log_returns) * scale).astype("float64")
    r_vals = r.to_numpy()
    out_var = np.full(n, np.nan, dtype="float64")

    # erster gültiger Return
    valid_mask = pd.notna(r).to_numpy()
    if not valid_mask.any():
        return pd.Series(out_var, index=closes.index)
    first_idx = int(np.argmax(valid_mask))

    # Long-Run-Varianz + omega default
    lr_slice = r_vals[max(0, first_idx - 1000) : first_idx + 1]
    lr_var = float(np.nanvar(lr_slice)) if lr_slice.size > 0 else 1e-6
    if not np.isfinite(lr_var) or lr_var <= 0.0:
        lr_var = 1e-6
    _omega = (
        float(omega) if omega is not None else lr_var * max(1e-6, (1.0 - alpha - beta))
    )
    _omega = max(_omega, 0.0)

    mu = float(np.nanmean(lr_slice)) if lr_slice.size > 0 else 0.0
    eps_prev2 = (r_vals[first_idx] - mu) ** 2
    var_prev = max(lr_var, eps_prev2, sigma_floor**2)
    out_var[first_idx] = var_prev

    for i in range(first_idx + 1, n):
        ri = r_vals[i]
        if not np.isfinite(ri):
            out_var[i] = out_var[i - 1]
            continue
        var_t = _omega + alpha * eps_prev2 + beta * var_prev
        var_t = max(var_t, sigma_floor**2)
        out_var[i] = var_t
        eps_prev2 = (ri - mu) ** 2
        var_prev = var_t

    sigma = np.sqrt(out_var) / scale  # zurück in Return-Einheiten
    valid = np.arange(n) >= (first_idx + max(0, int(min_periods)))
    sigma_series = pd.Series(sigma, index=closes.index).where(valid, other=np.nan)
    return sigma_series


def calculate_kalman_garch_zscore(
    series: pd.Series,
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
    Z-Score = (Close − KalmanMean) / (|Close| * sigma_return)
    mit sigma_return aus GARCH(1,1) auf Returns.
    """
    closes = pd.to_numeric(series, errors="coerce").astype("float64")
    km = kalman_filter(closes, R=R, Q=Q).astype("float64")
    resid = closes - km
    sigma_ret = calculate_garch_volatility(
        closes,
        alpha=alpha,
        beta=beta,
        omega=omega,
        use_log_returns=use_log_returns,
        scale=scale,
        min_periods=min_periods,
        sigma_floor=sigma_floor,
    )
    sigma_price = (closes.abs() * sigma_ret).replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = resid / sigma_price
    z[(sigma_price.isna()) | (km.isna())] = np.nan
    return z


# ---------------------------------------------------------------------------
# Spread & News‑Filter -------------------------------------------------------
# ---------------------------------------------------------------------------


def get_spread_pips(broker: BrokerInterface, symbol: str) -> Optional[float]:
    """Gibt aktuellen Spread (Punkte/Pips je nach Brokerdefinition) zurück."""
    try:
        spread_points = broker.get_symbol_spread(symbol)
    except Exception as e:
        log_service.log_exception(f"[Spread Fehler] {symbol}:", e)
        return None
    return float(spread_points) if spread_points is not None else None


def is_high_impact_news_window(now: datetime, symbol: str, block_minutes: int) -> bool:
    """Stub‑Funktion → True, falls grade Hochimpact‑News‑Fenster aktiv ist."""
    # TODO: News‑API integrieren. Fürs Erste immer False.
    return False


# ---------------------------------------------------------------------------
# ATR‑Median für H1 Vola‑Warnung --------------------------------------------
# ---------------------------------------------------------------------------


def median_atr_ema(df: pd.DataFrame, period: int, lookback: int) -> Optional[float]:
    """
    Median der ATR‑EMA‑Werte über *lookback* H1‑Kerzen.
    Erwartete Spalten: 'high', 'low', 'close'.
    """
    if (
        df is None
        or df.empty
        or any(col not in df.columns for col in ("high", "low", "close"))
        or len(df) < max(lookback, period) + 1
    ):
        return None

    highs = pd.to_numeric(df["high"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")
    closes = pd.to_numeric(df["close"], errors="coerce")
    prev_closes = closes.shift(1)

    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_ema = tr.ewm(span=period, adjust=False).mean()
    window = atr_ema.tail(lookback)
    if window.isna().all():
        return None
    return float(window.median())


# ---------------------------------------------------------------------------
# Candlestick Patterns ------------------------------------------------------
# ---------------------------------------------------------------------------


def _body_size(c: Dict) -> Optional[float]:
    try:
        return abs(float(c["close"]) - float(c["open"]))
    except Exception:
        return None


def get_candle_direction(candle: Dict) -> str:
    """
    Gibt die Richtung der Candle zurück:
    - 'bullish' wenn close > open
    - 'bearish' wenn close < open
    - 'neutral' wenn close == open
    """
    try:
        close = float(candle["close"])  # type: ignore[index]
        open_ = float(candle["open"])  # type: ignore[index]
    except Exception:
        return "neutral"

    if close > open_:
        return "bullish"
    elif close < open_:
        return "bearish"
    else:
        return "neutral"


def candle_size_ratio(c1: Dict, c2: Dict) -> float:
    """Verhältnis der Körpergrößen zweier Candles (c1 / c2)."""
    b1 = _body_size(c1)
    b2 = _body_size(c2)
    if b1 is None or b2 is None:
        return float("nan")
    if b2 == 0:
        return float("inf")
    return b1 / b2


def is_morning_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Klassisches Morning-Star Muster (bullisches Reversal)."""
    try:
        return (
            float(c1["close"]) < float(c1["open"])
            and _body_size(c2) is not None
            and _body_size(c1) is not None
            and abs(float(c2["close"]) - float(c2["open"]))
            < abs(float(c1["close"]) - float(c1["open"])) * 0.5
            and float(c3["close"]) > float(c3["open"])
            and float(c3["close"]) > ((float(c1["open"]) + float(c1["close"])) / 2)
        )
    except Exception:
        return False


def is_evening_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Klassisches Evening-Star Muster (bärisches Reversal)."""
    try:
        return (
            float(c1["close"]) > float(c1["open"])
            and _body_size(c2) is not None
            and _body_size(c1) is not None
            and abs(float(c2["close"]) - float(c2["open"]))
            < abs(float(c1["close"]) - float(c1["open"])) * 0.5
            and float(c3["close"]) < float(c3["open"])
            and float(c3["close"]) < ((float(c1["open"]) + float(c1["close"])) / 2)
        )
    except Exception:
        return False
