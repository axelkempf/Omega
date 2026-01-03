# ui_engine/main.py
"""
FastAPI UI Engine for Trading System Process Management.

Provides REST API and WebSocket endpoints for controlling trading strategies,
monitoring process health, and streaming logs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from hf_engine.infra.config.paths import TMP_DIR
from ui_engine.config import CONFIG_DIR, LOG_DIR
from ui_engine.controller import (
    check_datafeed_health,
    get_resource_usage,
    restart_unresponsive_strategies,
    start_datafeed_server,
)
from ui_engine.datafeeds.factory import get_datafeed_manager
from ui_engine.models import (
    DatafeedActionResponse,
    DatafeedHealth,
    ResourceUsage,
    StrategyResponse,
)
from ui_engine.registry.strategy_alias import resolve_alias
from ui_engine.strategies.factory import get_strategy_manager
from ui_engine.utils import read_log_tail


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Lifespan Start")
    threading.Thread(target=restart_unresponsive_strategies, daemon=True).start()
    yield
    print("🛑 Lifespan Ende")


from hf_engine.infra.config.branding import APP_DISPLAY_NAME, APP_VERSION

app = FastAPI(title=APP_DISPLAY_NAME, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> dict[str, str]:
    return {"message": "Willkommen beim Strategie Controller 👋"}


@app.post("/start/{name}", response_model=StrategyResponse)
def start(name: str) -> StrategyResponse:
    if name == "datafeed":
        manager = get_datafeed_manager(name)
    else:
        manager = get_strategy_manager(name)
    if not manager.start():
        return StrategyResponse(
            name=name, status="Konfig fehlerhaft oder Start fehlgeschlagen."
        )

    return StrategyResponse(name=name, status="Running")


@app.post("/stop/{name}", response_model=StrategyResponse)
def stop(name: str) -> StrategyResponse:
    if name == "datafeed":
        manager = get_datafeed_manager(name)
    else:
        manager = get_strategy_manager(name)
    if not manager.stop():
        raise HTTPException(
            status_code=400, detail=f"{name} konnte nicht gestoppt werden."
        )
    return StrategyResponse(name=name, status="Stopped")


@app.get("/status/{name}", response_model=StrategyResponse)
def status(name: str) -> StrategyResponse:
    if name == "datafeed":
        manager = get_datafeed_manager(name)
    else:
        manager = get_strategy_manager(name)
    return StrategyResponse(name=name, status=manager.status())


@app.post("/datafeed/start", response_model=DatafeedActionResponse)
def start_datafeed() -> DatafeedActionResponse:
    if not start_datafeed_server():
        raise HTTPException(status_code=500, detail="Datafeed-Start fehlgeschlagen.")
    return DatafeedActionResponse(status="started")


@app.post("/datafeed/stop", response_model=DatafeedActionResponse)
def stop_datafeed() -> DatafeedActionResponse:
    shutdown_path = TMP_DIR / "stop_15582434.signal"
    try:
        shutdown_path.write_text("stop")
        return DatafeedActionResponse(status="stopping")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Stoppen: {e}")


@app.get("/datafeed/health", response_model=DatafeedHealth)
def datafeed_status() -> DatafeedHealth:
    return DatafeedHealth(status="ok" if check_datafeed_health() else "offline")


@app.get("/logs/{account_id}", response_class=PlainTextResponse)
def get_log(account_id: str, lines: int = 100) -> str:
    resolved = resolve_alias(account_id)
    log_path = LOG_DIR / f"{resolved}.log"
    if not log_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Logfile für {resolved} nicht gefunden."
        )
    content = read_log_tail(str(log_path), lines=lines)
    return content


@app.websocket("/ws/logs/{account_id}")
async def log_stream(websocket: WebSocket, account_id: str) -> None:
    await websocket.accept()
    resolved = resolve_alias(account_id)
    log_path = LOG_DIR / f"{resolved}.log"

    if not log_path.exists():
        await websocket.send_text(f"⚠️ Logdatei {resolved}.log nicht gefunden.")
        await websocket.close(code=1008)
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # springe ans Dateiende

            while True:
                line = f.readline()
                if line:
                    await websocket.send_text(line.strip())
                else:
                    await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"📡 WebSocket getrennt für {account_id}")
    except asyncio.CancelledError:
        pass  # oder nur loggen, aber keinen Trace auslösen
    except Exception as e:
        print(f"❌ Unerwarteter Fehler im Logstream: {e}")


@app.post("/restart/{name}", response_model=StrategyResponse)
def restart(name: str) -> StrategyResponse:
    if name == "datafeed":
        manager = get_datafeed_manager(name)
    else:
        manager = get_strategy_manager(name)

    if not manager.stop():
        raise HTTPException(
            status_code=400, detail=f"{name} konnte nicht gestoppt werden."
        )

    time.sleep(6)  # kurze Pause, damit Shutdown greift

    if not manager.start():
        raise HTTPException(
            status_code=500, detail=f"{name} konnte nicht neu gestartet werden."
        )

    return StrategyResponse(name=name, status="Restarted")


@app.get("/resource/{name}", response_model=ResourceUsage)
def resource_usage(name: str) -> ResourceUsage:
    data = get_resource_usage(name)
    return ResourceUsage(
        status=data.get("status", "unknown"),
        cpu_percent=data.get("cpu_percent"),
        memory_mb=data.get("memory_mb"),
        threads=data.get("threads"),
        start_time=data.get("start_time"),
    )
