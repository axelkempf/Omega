# mt5_feed_server.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

from hf_engine.adapter.data.mt5_data_provider import MT5DataProvider
from hf_engine.infra.config.symbol_mapper import SymbolMapper

# =========================
# Logging (strukturiert)
# =========================
try:
    from hf_engine.infra.logging.log_service import log_service

    logger = log_service.logger
    _USING_LOG_SERVICE = True
except Exception:
    import logging

    _USING_LOG_SERVICE = False
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    # basicConfig nur setzen, wenn noch keine Handler existieren
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger = logging.getLogger("HFEngine.mt5_feed_server")

# Obergrenze für große Abfragen (DoS-/Fehlbedienungsschutz)
MAX_BARS = int(os.environ.get("DATAFEED_MAX_BARS", "10000"))


# =========================
# Utilities
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_utc(dt: datetime) -> datetime:
    """Macht Datetime tz-aware in UTC. Naive Werte werden als UTC interpretiert."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ok(data: Any) -> Dict[str, Any]:
    return {"data": data, "ts": utc_now_iso()}


def clamp_count(n: int) -> int:
    if n <= 0:
        raise HTTPException(status_code=400, detail="count muss > 0 sein.")
    if n > MAX_BARS:
        logger.warning("count %s > MAX_BARS %s, wird begrenzt.", n, MAX_BARS)
        return MAX_BARS
    return n


# =========================
# Optional: API-Key Schutz
# (aktiv, wenn ENV DATAFEED_API_KEY gesetzt ist)
# =========================
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("DATAFEED_API_KEY")
    if not expected:
        return  # Schutz deaktiviert
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =========================
# Lifespan (kein globaler State)
# =========================
def lifespan(app: FastAPI):
    @asynccontextmanager
    async def _lifespan_cm():
        config_path = os.environ.get("DATAFEED_CONFIG")
        if not config_path or not os.path.exists(config_path):
            raise RuntimeError("DATAFEED_CONFIG nicht gefunden oder ungültig")

        with open(config_path, "r", encoding="utf-8") as f:
            conf = json.load(f)

        mapper = SymbolMapper(
            broker_map=conf.get("symbol_map", {}),
            data_map=conf.get("data_map", conf.get("symbol_map", {})),
        )

        provider = MT5DataProvider(
            terminal_path=conf.get("terminal_path"),
            login=conf.get("account_id"),
            password=conf.get("password"),
            server=conf.get("server"),
            data_path=conf.get("data_path"),
            symbol_mapper=mapper,
        )

        app.state.provider = provider
        app.state.account_id = conf.get("account_id")
        logger.info("✅ Datafeed gestartet für Account %s", app.state.account_id)

        try:
            yield
        finally:
            # Falls der Provider einen Close/Shutdown besitzt, hier aufrufen:
            # getattr(app.state.provider, "close", lambda: None)()
            app.state.provider = None
            logger.info("🛑 Datafeed wird beendet...")

    # Für Tests: sync UND async nutzbar
    return _SyncOrAsyncCM(_lifespan_cm())


class _SyncOrAsyncCM:
    """
    Wrappt einen async Context-Manager und bietet __enter__/__exit__,
    sodass Tests `with lifespan(app):` synchron nutzen können.
    """

    def __init__(self, async_cm):
        self._acm = async_cm

    async def __aenter__(self):
        return await self._acm.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        return await self._acm.__aexit__(exc_type, exc, tb)

    def __enter__(self):
        return _run_async(self.__aenter__())

    def __exit__(self, exc_type, exc, tb):
        return _run_async(self.__aexit__(exc_type, exc, tb))


def _run_async(coro):
    # kleines Hilfsmittel für Tests ohne laufenden Loop
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        # Falls bereits ein Loop läuft (z. B. in Testframeworks),
        # synchron blockierend ausführen.
        return loop.run_until_complete(coro)


# =========================
# FastAPI App + Middleware
# =========================
app = FastAPI(lifespan=lifespan)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "REQ %s %s | id=%s | status=%s | %.2fms",
                request.method,
                request.url.path,
                req_id,
                getattr(request, "state", {}).__dict__.get("status", "?"),
                duration_ms,
            )
        response.headers["X-Request-ID"] = req_id
        return response


app.add_middleware(RequestLoggingMiddleware)


# =========================
# Einheitliche Fehler-Handler
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    logger.warning("HTTP %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": exc.status_code, "message": str(exc.detail)},
            "ts": utc_now_iso(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {"code": 500, "message": "Internal Server Error"},
            "ts": utc_now_iso(),
        },
    )


# =========================
# Pydantic Request-Modelle
# =========================
class OHLCRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    count: int = Field(..., gt=0)


class OHLCClosedRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    offset: int = Field(default=1, ge=0)


class RatesRangeRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    start: datetime
    end: datetime

    @validator("start", "end", pre=True)
    def _coerce_dt(cls, v):
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v)
            except Exception:
                raise ValueError("Ungültiges Datumsformat. Erwartet ISO-8601.")
        return ensure_utc(v)

    @validator("end")
    def _validate_range(cls, v, values):
        start = values.get("start")
        if start and v <= start:
            raise ValueError("end muss nach start liegen.")
        return v


class RatesFromPosRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    start_pos: int
    count: int = Field(..., gt=0)


class TickDataRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    start: datetime
    end: datetime

    @validator("start", "end", pre=True)
    def _coerce_dt(cls, v):
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v)
            except Exception:
                raise ValueError("Ungültiges Datumsformat. Erwartet ISO-8601.")
        return ensure_utc(v)

    @validator("end")
    def _validate_range(cls, v, values):
        start = values.get("start")
        if start and v <= start:
            raise ValueError("end muss nach start liegen.")
        return v


# =========================
# Helper: Provider holen
# =========================
def _get_provider(request: Request) -> MT5DataProvider:
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="MT5-Provider nicht initialisiert")
    return provider


# =========================
# ROUTES
# =========================
@app.post("/ohlc", dependencies=[Depends(require_api_key)])
def get_ohlc(req: OHLCRequest, request: Request):
    provider = _get_provider(request)
    try:
        tf = req.timeframe.upper()
        count = clamp_count(req.count)
        data = provider.get_ohlc_series(req.symbol, tf, count)
        return ok(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fehler beim Laden von OHLC-Daten")
        raise HTTPException(
            status_code=500, detail=f"Fehler beim Laden von OHLC-Daten: {e}"
        )


@app.post("/ohlc_closed", dependencies=[Depends(require_api_key)])
def get_ohlc_closed(req: OHLCClosedRequest, request: Request):
    provider = _get_provider(request)
    tf = req.timeframe.upper()
    data = provider.get_ohlc_for_closed_candle(req.symbol, tf, req.offset)
    return ok(data)


@app.post("/rates_range", dependencies=[Depends(require_api_key)])
def get_rates_range(req: RatesRangeRequest, request: Request):
    provider = _get_provider(request)
    tf = req.timeframe.upper()
    df = provider.get_rates_range(
        symbol=req.symbol, timeframe=tf, start=req.start, end=req.end
    )
    return ok(df.to_dict(orient="records"))


@app.post("/rates_from_pos", dependencies=[Depends(require_api_key)])
def get_rates_from_pos(req: RatesFromPosRequest, request: Request):
    provider = _get_provider(request)
    tf = req.timeframe.upper()
    count = clamp_count(req.count)
    df = provider.get_rates_from_pos(req.symbol, tf, req.start_pos, count)
    return ok(df.to_dict(orient="records"))


@app.post("/tick_data", dependencies=[Depends(require_api_key)])
def get_tick_data(req: TickDataRequest, request: Request):
    provider = _get_provider(request)
    df = provider.get_tick_data(req.symbol, req.start, req.end)
    return ok(df.to_dict(orient="records"))


@app.get("/health")
def healthcheck(request: Request):
    # Health offen lassen (keine API-Key-Pflicht)
    return ok(
        {
            "status": "ok",
            "account_id": getattr(request.app.state, "account_id", None),
        }
    )
