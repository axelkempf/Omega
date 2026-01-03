# core/data/mt5_data_provider.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

from hf_engine.adapter.data.data_provider_interface import DataProviderInterface
from hf_engine.infra.config.environment import BROKER_TIMEZONE, TIMEZONE
from hf_engine.infra.config.symbol_mapper import SymbolMapper
from hf_engine.infra.config.time_utils import from_utc_to_broker, to_utc
from hf_engine.infra.logging.log_service import log_service


class MT5DataProvider(DataProviderInterface):
    """
    Datenprovider für MT5.
    - Alle externen Zeiten (Eingaben und MT5-Outputs) werden konsistent als UTC behandelt/geliefert.
    - Broker-Zeit wird ausschließlich für MT5-Abfragen verwendet.
    """

    # Zentrale, einmalige Timeframe-Map
    TIMEFRAME_MAP: Dict[str, int] = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        data_path: Optional[str] = None,
        symbol_mapper: Optional[SymbolMapper] = None,
    ) -> None:
        self.symbol_mapper = symbol_mapper or SymbolMapper({})

        try:
            if terminal_path:
                initialized = mt5.initialize(
                    path=terminal_path,
                    login=login,
                    password=password,
                    server=server,
                    portable=True,
                    data_path=data_path,
                )
            else:
                initialized = mt5.initialize()
        except Exception as exc:
            # Harte Initialisierungsfehler (z. B. DLL/Terminal nicht auffindbar)
            raise RuntimeError(
                f"MT5 konnte nicht initialisiert werden (Exception): {exc!r}"
            )

        if not initialized:
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 konnte nicht initialisiert werden: {code} – {msg}")

        log_service.log_system("[MT5DataProvider] ✅ Erfolgreich initialisiert")

    # --- interne Hilfen -----------------------------------------------------

    @staticmethod
    def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Erwartet MT5 'time' in Sekunden seit Epoch.
        Interpretiert diese Zeit als Broker-/Datafeed-Zeit und konvertiert
        sie nach **UTC** (tz-aware), anschließend als Index setzen.
        """
        if df is None or df.empty:
            return df
        df = df.copy()
        # MT5 liefert epoch-seconds in 'time'. Diese als Brokerzeit interpretieren
        # und in die System-Referenz (UTC) konvertieren.
        dt = pd.to_datetime(df["time"], unit="s", utc=False)
        dt = dt.dt.tz_localize(BROKER_TIMEZONE).dt.tz_convert(TIMEZONE)
        df["time"] = dt
        df.set_index("time", inplace=True)
        return df

    @classmethod
    def _resolve_timeframe(cls, timeframe: str) -> int:
        tf = cls.TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        return tf

    @staticmethod
    def _validate_range(start: datetime, end: datetime) -> None:
        if start is None or end is None:
            raise ValueError("Start und Endzeit dürfen nicht None sein.")
        if to_utc(start) >= to_utc(end):
            raise ValueError(f"Start ({start}) muss vor Endzeit ({end}) liegen.")

    def _to_broker(self, dt: datetime) -> datetime:
        """
        Erwartet dt in beliebiger Form (naive → UTC angenommen, oder tz-aware).
        Gibt **Brokerzeit** zurück (tz-aware).
        """
        return from_utc_to_broker(to_utc(dt))

    @staticmethod
    def _ensure_non_empty(data, context: str) -> None:
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            raise RuntimeError(f"Keine Daten erhalten: {context}")

    # --- Public API ---------------------------------------------------------

    def close(self) -> None:
        """Sauberes Beenden der MT5-Session."""
        try:
            mt5.shutdown()
            log_service.log_system("[MT5DataProvider] 🔌 Session beendet")
        except Exception as exc:
            # Nicht kritisch, aber nützlich fürs Debugging
            log_service.log_system(
                f"[MT5DataProvider] ⚠️ shutdown() Fehler: {exc!r}", level="WARNING"
            )

    # Kontextmanager-Unterstützung (optional)
    def __enter__(self) -> "MT5DataProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------- Historical bars (range) ----------------

    def get_rates_range(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """
        Liefert OHLCV-Balken im angegebenen Zeitintervall als DataFrame mit **UTC-Index**.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Ungültiges Symbol.")

        self._validate_range(start, end)
        tf = self._resolve_timeframe(timeframe)

        broker_symbol = self.symbol_mapper.to_datafeed(symbol)
        start_b = self._to_broker(start)
        end_b = self._to_broker(end)

        rates = mt5.copy_rates_range(broker_symbol, tf, start_b, end_b)
        self._ensure_non_empty(
            rates, f"copy_rates_range({broker_symbol}, {timeframe}, {start}, {end})"
        )

        df = pd.DataFrame(rates)
        df = self._ensure_utc_index(df)  # ← **UTC-aware Index**
        return df

    # ---------------- Historical bars (from position) ----------------

    def get_rates_from_pos(
        self, symbol: str, timeframe: str, start_pos: int, count: int
    ) -> pd.DataFrame:
        """
        Liefert OHLCV-Balken ab Position `start_pos` (0 = aktuell offen), Anzahl `count`,
        als DataFrame mit **UTC-Index**.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Ungültiges Symbol.")
        if count <= 0:
            raise ValueError("count muss > 0 sein.")
        if start_pos < 0:
            raise ValueError("start_pos muss >= 0 sein.")

        tf = self._resolve_timeframe(timeframe)
        broker_symbol = self.symbol_mapper.to_datafeed(symbol)

        data = mt5.copy_rates_from_pos(broker_symbol, tf, start_pos, count)
        self._ensure_non_empty(
            data,
            f"copy_rates_from_pos({broker_symbol}, {timeframe}, {start_pos}, {count})",
        )

        df = pd.DataFrame(data)
        df = self._ensure_utc_index(df)
        return df

    # ---------------- Tick data (range) ----------------

    def get_tick_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """
        Liefert Ticks im Zeitraum als DataFrame mit **UTC-Index**.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Ungültiges Symbol.")
        self._validate_range(start, end)

        broker_symbol = self.symbol_mapper.to_datafeed(symbol)
        start_b = self._to_broker(start)
        end_b = self._to_broker(end)

        ticks = mt5.copy_ticks_range(broker_symbol, start_b, end_b, mt5.COPY_TICKS_ALL)
        self._ensure_non_empty(
            ticks, f"copy_ticks_range({broker_symbol}, {start}, {end})"
        )

        df = pd.DataFrame(ticks)
        df = self._ensure_utc_index(df)
        return df

    # ---------------- Single closed candle ----------------

    def get_ohlc_for_closed_candle(
        self, symbol: str, timeframe: str, offset: int = 1
    ) -> Optional[dict]:
        """
        Gibt die OHLC-Daten einer einzelnen **abgeschlossenen** Kerze zurück (UTC-Zeit als ISO8601).
        offset: 1 = letzte abgeschlossene Kerze, 2 = vorletzte, ...
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Ungültiges Symbol.")
        if offset <= 0:
            raise ValueError("offset muss >= 1 sein.")

        tf = self._resolve_timeframe(timeframe)
        broker_symbol = self.symbol_mapper.to_datafeed(symbol)

        # +1 laden, damit der Index für offset sicher vorhanden ist
        rates = mt5.copy_rates_from_pos(broker_symbol, tf, 0, offset + 1)
        self._ensure_non_empty(
            rates, f"copy_rates_from_pos({broker_symbol}, {timeframe}, 0, {offset + 1})"
        )
        if len(rates) <= offset:
            raise RuntimeError("Nicht genügend Kerzen geladen")

        df = pd.DataFrame(rates)
        df = self._ensure_utc_index(df)

        # -1 = aktuell offen; -(offset+1) = gewünschte abgeschlossene Kerze
        rec = df.iloc[-(offset + 1)].to_dict()
        # 'time' ist aktuell der Index → hole aus Index (UTC) und gebe ISO8601 zurück
        # Achtung: wenn df.reset_index() nicht erfolgte, sitzt 'time' nicht in rec.
        # Daher 'time' aus dem Index extrahieren:
        rec["time"] = df.index[-(offset + 1)].isoformat()
        return rec

    # ---------------- Series of closed candles ----------------

    def get_ohlc_series(self, symbol: str, timeframe: str, count: int) -> List[dict]:
        """
        Gibt eine Liste **abgeschlossener** Kerzen als Dicts mit OHLC und UTC-Zeit (ISO8601) zurück.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Ungültiges Symbol.")
        if count <= 0:
            raise ValueError("count muss > 0 sein.")

        tf = self._resolve_timeframe(timeframe)
        broker_symbol = self.symbol_mapper.to_datafeed(symbol)

        # start_pos=1 → letzte abgeschlossene Kerze, count Stück zurück
        rates = mt5.copy_rates_from_pos(broker_symbol, tf, 1, count)
        self._ensure_non_empty(
            rates, f"copy_rates_from_pos({broker_symbol}, {timeframe}, 1, {count})"
        )
        if len(rates) < count:
            log_service.log_system(
                f"[MT5DataProvider] ❌ Nur {len(rates)} Kerzen geladen für {broker_symbol} @ {timeframe}",
                level="ERROR",
            )
            raise RuntimeError("Nicht genügend Kerzen geladen")

        df = pd.DataFrame(rates)
        df = self._ensure_utc_index(df)

        # Nur benötigte Spalten und Zeit als ISO8601 aus dem Index
        out = []
        for idx, row in df.iterrows():
            out.append(
                {
                    "time": idx.isoformat(),  # UTC, tz-aware
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    # optional: "tick_volume": int(row.get("tick_volume", 0)),
                }
            )
        return out
