# remote_data_provider.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hf_engine.infra.logging.log_service import log_service

logger = log_service.logger


class RemoteDataError(RuntimeError):
    """Fehler beim Abruf vom Remote‑Datenservice."""


class RemoteDataProvider:
    """
    Robuster HTTP‑Client für Marktdaten (OHLC, Tickdaten, Zeitbereiche).
    Minimal gehalten, mit Timeouts, Retries, Eingabevalidierung und sauberem Fehlerbild.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        *,
        scheme: str = "http",
        timeout: float = 5.0,
        retries_total: int = 3,
        backoff_factor: float = 0.2,
        status_forcelist: tuple[int, ...] = (500, 502, 503, 504),
        user_agent: str = "RemoteDataProvider/1.0",
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        if scheme not in {"http", "https"}:
            raise ValueError("scheme muss 'http' oder 'https' sein")
        if port <= 0:
            raise ValueError("port muss > 0 sein")
        if timeout <= 0:
            raise ValueError("timeout muss > 0 sein")

        self.base_url = f"{scheme}://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

        retries = Retry(
            total=retries_total,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Standard‑Header
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept": "application/json"}
        )
        if extra_headers:
            self.session.headers.update(extra_headers)

    # --- Public API ---------------------------------------------------------

    def get_ohlc_series(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        self._require_non_empty(symbol, "symbol")
        self._require_non_empty(timeframe, "timeframe")
        self._require_positive(count, "count")
        payload = {"symbol": symbol, "timeframe": timeframe, "count": count}
        return self._post("/ohlc", payload)

    def get_ohlc_for_closed_candle(
        self, symbol: str, timeframe: str, offset: int = 1
    ) -> Any:
        self._require_non_empty(symbol, "symbol")
        self._require_non_empty(timeframe, "timeframe")
        self._require_non_negative(offset, "offset")
        payload = {"symbol": symbol, "timeframe": timeframe, "offset": offset}
        return self._post("/ohlc_closed", payload)

    def get_rates_range(
        self,
        symbol: str,
        timeframe: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> list[dict]:
        self._require_non_empty(symbol, "symbol")
        self._require_non_empty(timeframe, "timeframe")
        start_s = self._to_iso(start, "start")
        end_s = self._to_iso(end, "end")
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start_s,
            "end": end_s,
        }
        return self._post("/rates_range", payload)

    def get_rates_from_pos(
        self, symbol: str, timeframe: str, start_pos: int, count: int
    ) -> list[dict]:
        self._require_non_empty(symbol, "symbol")
        self._require_non_empty(timeframe, "timeframe")
        self._require_non_negative(start_pos, "start_pos")
        self._require_positive(count, "count")
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_pos": start_pos,
            "count": count,
        }
        return self._post("/rates_from_pos", payload)

    def get_tick_data(
        self, symbol: str, start: Union[str, datetime], end: Union[str, datetime]
    ) -> list[dict]:
        self._require_non_empty(symbol, "symbol")
        start_s = self._to_iso(start, "start")
        end_s = self._to_iso(end, "end")
        payload = {"symbol": symbol, "start": start_s, "end": end_s}
        return self._post("/tick_data", payload)

    # --- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Schließt die zugrunde liegende HTTP‑Session."""
        try:
            self.session.close()
        except Exception:  # pragma: no cover
            logger.debug("Fehler beim Schließen der Session", exc_info=True)

    def __enter__(self) -> "RemoteDataProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- Internals ----------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> Any:
        url = f"{self.base_url}{endpoint}"
        try:
            logger.debug("POST %s payload=%s", url, payload)
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            # HTTP‑Fehler (>=400) in Exception wandeln
            resp.raise_for_status()

            # JSON parse & Struktur prüfen
            try:
                body = resp.json()
            except ValueError as je:
                snippet = (resp.text or "")[:500]
                raise RemoteDataError(
                    f"Ungültige JSON‑Antwort von {endpoint}: {je}. BodySnippet={snippet!r}"
                ) from je

            if "data" not in body:
                raise RemoteDataError(
                    f"Antwort von {endpoint} enthält kein 'data'‑Feld. Keys={list(body.keys())}"
                )

            logger.debug("OK %s", url)
            return body["data"]

        except requests.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            text_snippet = ""
            try:
                text_snippet = (he.response.text or "")[:500]  # max 500 Zeichen
            except Exception:
                pass
            msg = f"HTTP {status} bei {endpoint}. BodySnippet={text_snippet!r}"
            logger.warning(msg)
            raise RemoteDataError(msg) from he

        except requests.RequestException as re:
            msg = f"Netzwerkfehler bei {endpoint}: {re}"
            logger.warning(msg)
            raise RemoteDataError(msg) from re

    @staticmethod
    def _to_iso(value: Union[str, datetime], field: str) -> str:
        if isinstance(value, datetime):
            # Keine zusätzliche TZ‑Annahme – Upstream‑Service definiert Erwartung.
            return value.isoformat()
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError(f"'{field}' muss datetime oder nicht‑leerer str sein")

    @staticmethod
    def _require_non_empty(s: Optional[str], name: str) -> None:
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"'{name}' muss ein nicht‑leerer String sein")

    @staticmethod
    def _require_positive(n: int, name: str) -> None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"'{name}' muss eine positive ganze Zahl sein")

    @staticmethod
    def _require_non_negative(n: int, name: str) -> None:
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"'{name}' muss eine ganze Zahl ≥ 0 sein")
