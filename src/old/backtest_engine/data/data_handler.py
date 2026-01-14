import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from backtest_engine.data.candle import Candle
from backtest_engine.data.market_hours import (
    is_valid_trading_time,
    is_valid_trading_time_vectorized,
)

from hf_engine.infra.config.paths import PARQUET_DIR, RAW_DATA_DIR

_PARQUET_BUILD_CACHE: OrderedDict[str, Any] = OrderedDict()
_DF_BUILD_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_MAX_PARQUET = int(os.getenv("HF_CACHE_MAX_PARQUET_BUILDS", "16"))
_CACHE_MAX_DF = int(os.getenv("HF_CACHE_MAX_DF_BUILDS", "32"))


def _evict_lru(cache: OrderedDict[str, Any], max_len: int) -> None:
    try:
        while len(cache) > max_len > 0:
            cache.popitem(last=False)
    except Exception:
        pass


def trim_candle_build_caches(
    keep_parquet: Optional[int] = None, keep_df: Optional[int] = None
) -> None:
    """
    LRU-Trim der Candle-Build-Caches (speicherschonend statt full reset).
    """
    global _PARQUET_BUILD_CACHE, _DF_BUILD_CACHE
    _evict_lru(_PARQUET_BUILD_CACHE, keep_parquet or _CACHE_MAX_PARQUET)
    _evict_lru(_DF_BUILD_CACHE, keep_df or _CACHE_MAX_DF)


def _float32_enabled() -> bool:
    """ENV-Schalter: HF_FLOAT32=1 aktiviert Downcast für OHLC/Volume."""
    try:
        return os.getenv("HF_FLOAT32", "0").strip() == "1"
    except Exception:
        return False


def _floor_to_tf_vec(ts: pd.Series, tf: str) -> pd.Series:
    """
    Vektorisierte Timeframe-Flooring: nutzt .dt.floor statt per-row apply.
    Fällt bei unbekanntem TF no-op zurück.
    """
    try:
        if tf.startswith("M"):
            return ts.dt.floor(f"{int(tf[1:])}min")
        if tf.startswith("H"):
            return ts.dt.floor(f"{int(tf[1:])}h")
        if tf.startswith("D"):
            return ts.dt.floor(f"{int(tf[1:])}D")
    except Exception:
        pass
    return ts


def _apply_market_hours_fast(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Schneller Market-Hours-Filter:
    - nutzt vektorisierten Sydney-Session-Filter (DST-aware),
    - fällt bei Inkompatibilitäten sicher auf Zeilen-basiert zurück.
    """
    s = df[col]
    try:
        # Vektorisiert & DST-korrekt: filtert Samstag >=07:00 SYD, Sonntag komplett,
        # Montag <07:00 SYD sowie behält Di–Fr komplett.
        mask = is_valid_trading_time_vectorized(s)
        return df[mask]
    except Exception:
        # Fallback (langsamer): row-wise
        return df[s.apply(is_valid_trading_time)]


class CSVDataHandler:
    """
    Lädt Candle-Daten aus CSV- oder Parquet-Dateien.
    Optional kann ein bereits geladenes DataFrame-Objekt verwendet werden.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M1",
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
        normalize_to_timeframe: bool = False,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.preloaded_data = preloaded_data

        self.csv_dir = RAW_DATA_DIR / "csv" / symbol
        self.parquet_dir = PARQUET_DIR / symbol

        self.bid_file = self._resolve_csv_path(symbol, timeframe, "bid")
        self.ask_file = self._resolve_csv_path(symbol, timeframe, "ask")

        # Robust parquet path resolution: prefer BID/ASK, fallback to bid/ask
        self.bid_parquet = self._resolve_parquet_path(symbol, timeframe, "bid")
        self.ask_parquet = self._resolve_parquet_path(symbol, timeframe, "ask")

        self.normalize_to_timeframe = normalize_to_timeframe

    def _resolve_csv_path(self, symbol: str, timeframe: str, side: str) -> Path:
        """Resolve csv path with preference for uppercase BID/ASK."""
        # Priority 1: Uppercase (BID/ASK)
        upper_path = self.csv_dir / f"{symbol}_{timeframe}_{side.upper()}.csv"
        if upper_path.exists():
            return upper_path
        # Priority 2: Lowercase (bid/ask) - fallback
        lower_path = self.csv_dir / f"{symbol}_{timeframe}_{side.lower()}.csv"
        return lower_path  # Return even if not exists, for consistency

    def _resolve_parquet_path(self, symbol: str, timeframe: str, side: str) -> Path:
        """Resolve parquet path with preference for uppercase BID/ASK."""
        # Priority 1: Uppercase (BID/ASK)
        upper_path = self.parquet_dir / f"{symbol}_{timeframe}_{side.upper()}.parquet"
        if upper_path.exists():
            return upper_path
        # Priority 2: Lowercase (bid/ask) - fallback
        lower_path = self.parquet_dir / f"{symbol}_{timeframe}_{side.lower()}.parquet"
        return lower_path  # Return even if not exists, for consistency

    def _load_file(
        self,
        path: Path,
        candle_type: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> List[Candle]:
        if not path.exists():
            return []

        # CSV lesen (optional mit Float32/Int32)
        usecols = ["UTC time", "Open", "High", "Low", "Close", "Volume"]
        if _float32_enabled():
            # Dtype-Hinweise beim CSV-Read reduzieren Peak-RAM bereits beim Parsen
            dtypes = {
                "Open": "float32",
                "High": "float32",
                "Low": "float32",
                "Close": "float32",
                "Volume": "int32",
            }
            df = pd.read_csv(
                path, usecols=usecols, parse_dates=["UTC time"], dtype=dtypes
            )
        else:
            df = pd.read_csv(path, usecols=usecols, parse_dates=["UTC time"])

        # Timestamps konsistent UTC (tz-aware)
        if df["UTC time"].dt.tz is None:
            df["UTC time"] = df["UTC time"].dt.tz_localize("UTC")
        else:
            df["UTC time"] = df["UTC time"].dt.tz_convert("UTC")

        # Optionales Downcast auch im CSV-Pfad (falls dtype-Hinweise oben nicht gegriffen haben)
        if _float32_enabled():
            try:
                for c in ("Open", "High", "Low", "Close"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], downcast="float")
                if "Volume" in df.columns:
                    df["Volume"] = pd.to_numeric(df["Volume"], downcast="integer")
            except Exception:
                pass

        # Zeitfenster schneiden (vor Filter/Normalize für weniger Daten)
        if start_dt is not None:
            df = df[df["UTC time"] >= start_dt]
        if end_dt is not None:
            df = df[df["UTC time"] <= end_dt]

        # Market-Hours schnell (mit Fallback)
        if not self.timeframe.upper().startswith("D"):
            df = _apply_market_hours_fast(df, "UTC time")

        # --- Daily-Cleanup: Wochenenden & 0-Volumen-Flat-Platzhalter entfernen ---
        if self.timeframe.upper().startswith("D"):
            # Sicherheit: UTC tz-aware (sollte bereits so sein)
            df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True)

            # a) Sa/So (5=Samstag, 6=Sonntag) verwerfen
            wd = df["UTC time"].dt.weekday
            not_weekend = ~wd.isin([5, 6])

            # b) 0-Volumen-Flat (Open==High==Low==Close) & Volume==0 verwerfen
            flat_zero = (
                (df["Volume"] == 0)
                & df["Open"].eq(df["High"])
                & df["High"].eq(df["Low"])
                & df["Low"].eq(df["Close"])
            )
            df = df[not_weekend & ~flat_zero]

        # Optional: TF-Normalisierung – vektorisiert
        if self.normalize_to_timeframe:
            df["UTC time"] = _floor_to_tf_vec(df["UTC time"], self.timeframe)

        # Candle-Liste effizient bauen
        CandleCls = Candle
        ts = df["UTC time"].values
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        v = df["Volume"].values

        return [
            CandleCls(
                timestamp=ts[i],
                open=float(o[i]),
                high=float(h[i]),
                low=float(l[i]),
                close=float(c[i]),
                volume=float(v[i]),
                candle_type=candle_type,
            )
            for i in range(len(df))
        ]

    def _load_parquet(
        self,
        path: Path,
        candle_type: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> List[Candle]:
        """
        Lädt Candles (Parquet oder preloaded) und baut eine stabil gecachte Candle-Liste
        je (path, candle_type, start, end, normalize_flag).
        """
        from typing import Optional as _Opt
        from typing import Tuple

        global _PARQUET_BUILD_CACHE

        # 1) DataFrame beziehen (preloaded oder Parquet)
        # WICHTIG: Wenn preloaded_data vorhanden ist, darf ein bestehender
        # _PARQUET_BUILD_CACHE-Eintrag NICHT greifen (sonst werden jittered/preloaded
        # Daten stillschweigend durch Base-Parquet-Candles ersetzt).
        if self.preloaded_data and (self.timeframe, candle_type) in self.preloaded_data:
            base_df = self.preloaded_data[(self.timeframe, candle_type)]
            # --- Build-Cache für PRELOADED-DFS (stabil auf Basis des Basis-DF-Objekts) ---
            cache_key_df = (
                id(base_df),
                candle_type,
                start_dt.isoformat() if start_dt else None,
                end_dt.isoformat() if end_dt else None,
                bool(self.normalize_to_timeframe),
            )
            cached_df_build = _DF_BUILD_CACHE.get(cache_key_df)
            if cached_df_build is not None:
                return cached_df_build

            # Fenster IMMER schneiden (neues df-Objekt)
            df = base_df
            if start_dt is not None:
                df = df[df["UTC time"] >= start_dt]
            if end_dt is not None:
                df = df[df["UTC time"] <= end_dt]
        else:
            cache_key_parquet: Tuple[str, str, _Opt[str], _Opt[str], bool] = (
                str(path),
                candle_type,
                start_dt.isoformat() if start_dt else None,
                end_dt.isoformat() if end_dt else None,
                bool(self.normalize_to_timeframe),
            )
            cached = _PARQUET_BUILD_CACHE.get(cache_key_parquet)
            if cached is not None:
                return cached

            base_df = None
            df = pd.read_parquet(
                path, columns=["UTC time", "Open", "High", "Low", "Close", "Volume"]
            )
            # Optional: Float32/Int32 senkt Peak-RAM; neutral bzgl. Geschwindigkeit
            if _float32_enabled():
                df = _maybe_downcast_ohlc(df)

        # 2) UTC (tz-aware) erzwingen
        df["UTC time"] = pd.to_datetime(df["UTC time"])
        if df["UTC time"].dt.tz is None:
            df["UTC time"] = df["UTC time"].dt.tz_localize("UTC")
        else:
            df["UTC time"] = df["UTC time"].dt.tz_convert("UTC")

        # 3) Für Parquet-DF: Fenster jetzt schneiden (preloaded bereits geschnitten)
        if not (
            self.preloaded_data and (self.timeframe, candle_type) in self.preloaded_data
        ):
            if start_dt is not None:
                df = df[df["UTC time"] >= start_dt]
            if end_dt is not None:
                df = df[df["UTC time"] <= end_dt]

        # 4) Market-Hours schnell (mit Fallback)
        if not self.timeframe.upper().startswith("D"):
            df = _apply_market_hours_fast(df, "UTC time")

        # 4.5) Daily-Cleanup: Wochenenden & 0-Volumen-Flat-Platzhalter entfernen
        if self.timeframe.upper().startswith("D"):
            # Sicherheit: UTC tz-aware (sollte bereits so sein)
            df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True)

            # a) Sa/So (5=Samstag, 6=Sonntag) verwerfen
            wd = df["UTC time"].dt.weekday
            not_weekend = ~wd.isin([5, 6])

            # b) 0-Volumen-Flat (Open==High==Low==Close) & Volume==0 verwerfen
            flat_zero = (
                (df["Volume"] == 0)
                & df["Open"].eq(df["High"])
                & df["High"].eq(df["Low"])
                & df["Low"].eq(df["Close"])
            )
            df = df[not_weekend & ~flat_zero]

        # 5) Optional: TF-Normalisierung – vektorisiert
        ts_col = df["UTC time"]
        if self.normalize_to_timeframe:
            ts_col = _floor_to_tf_vec(ts_col, self.timeframe)

        # 6) Vektorisiertes Candle-Build
        CandleCls = Candle
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        v = df["Volume"].values

        candles: List[Candle] = [
            CandleCls(
                timestamp=ts_col.iloc[i],
                open=float(o[i]),
                high=float(h[i]),
                low=float(l[i]),
                close=float(c[i]),
                volume=float(v[i]),
                candle_type=candle_type,
            )
            for i in range(len(df))
        ]

        # --- Ergebnis cachen (Parquet- oder Preloaded-Pfad) ---
        if base_df is None:
            _PARQUET_BUILD_CACHE[cache_key_parquet] = candles
            _PARQUET_BUILD_CACHE.move_to_end(cache_key_parquet, last=True)
            _evict_lru(_PARQUET_BUILD_CACHE, _CACHE_MAX_PARQUET)
        else:
            _DF_BUILD_CACHE[cache_key_df] = candles
            _DF_BUILD_CACHE.move_to_end(cache_key_df, last=True)
            _evict_lru(_DF_BUILD_CACHE, _CACHE_MAX_DF)
        return candles

    def load_candles(
        self, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
    ) -> Dict[str, List[Candle]]:
        """
        Lädt Bid- und Ask-Candles (bevorzugt Parquet, sonst CSV).
        """
        if os.path.exists(self.bid_parquet):
            bid = self._load_parquet(self.bid_parquet, "bid", start_dt, end_dt)
        else:
            bid = self._load_file(self.bid_file, "bid", start_dt, end_dt)

        if os.path.exists(self.ask_parquet):
            ask = self._load_parquet(self.ask_parquet, "ask", start_dt, end_dt)
        else:
            ask = self._load_file(self.ask_file, "ask", start_dt, end_dt)

        return {"bid": bid, "ask": ask}


# === RAM-Hygiene ==============================================================
def _maybe_downcast_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcastet OHLC→float32 und Volume→int32 (sofern vorhanden) zur RAM-Reduktion.
    Wirkt nur auf DataFrame-Ebene; Candles bleiben Python-Floats (keine Logik-Änderung).
    Aktivierung per ENV: HF_FLOAT32=1
    """
    try:
        for c in ("Open", "High", "Low", "Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], downcast="float")
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], downcast="integer")
    except Exception:
        # defensiv: niemals den Load-Pfad hart abbrechen
        pass
    return df


def reset_candle_build_caches() -> None:
    """
    Drop internal DataFrame→Candle build caches to free RAM between heavy phases.
    """
    _PARQUET_BUILD_CACHE.clear()
    _DF_BUILD_CACHE.clear()
