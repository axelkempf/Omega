# health_server.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from hf_engine.core.controlling.multi_strategy_controller import controller
from hf_engine.infra.config.environment import ENV

# ---------------------------
# Helpers
# ---------------------------


def _env_get(key: str, default: Any = None) -> Any:
    """ENV access that works for both dict-like and attr-like ENV objects."""
    try:
        if isinstance(ENV, dict):
            return ENV.get(key, default)
        # fallback to attribute
        return getattr(ENV, key, default)
    except Exception:
        return default


def _utc_iso(ts: Optional[float]) -> Optional[str]:
    """Return RFC 3339/ISO 8601 UTC string with 'Z' suffix or None."""
    if ts is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        # Use 'Z' suffix for UTC
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


# ---------------------------
# API Models
# ---------------------------


class StrategyStatus(BaseModel):
    name: str = Field(..., description="Strategiename")
    last_heartbeat: Optional[str] = Field(
        None, description="Letzter Heartbeat in UTC (RFC3339)"
    )
    healthy: Optional[bool] = Field(
        None, description="True, wenn Heartbeat innerhalb des Toleranzfensters"
    )
    details: Optional[str] = Field(
        None, description="Optionale Zusatzinfos/Fehlermeldungen"
    )


class HealthResponse(BaseModel):
    env: Any
    status: str = Field(..., description="online | degraded | error")
    strategies: List[StrategyStatus] = Field(default_factory=list)
    server_time_utc: str = Field(..., description="Serverzeit in UTC (RFC3339)")


# ---------------------------
# Dependencies
# ---------------------------


def get_controller():
    """Dependency to allow easy mocking in tests."""
    return controller


def verify_health_key(x_health_key: Optional[str] = Header(default=None)) -> None:
    """
    Optional: einfacher Header-Check, wenn HEALTH_API_KEY in ENV gesetzt ist.
    Wenn kein Key konfiguriert ist, wird keine Auth erzwungen.
    """
    expected = _env_get("HEALTH_API_KEY", None)
    if expected:
        if not x_health_key or x_health_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid health key",
            )


# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(title="Health Server", version="1.0.0")


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
)
async def healthcheck(
    _auth: None = Depends(verify_health_key),
    ctrl=Depends(get_controller),
):
    """
    Liefert Health-Status des Systems.
    - Zeiten sind UTC (RFC3339 mit 'Z').
    - 'status' wird dynamisch bestimmt: 'online', 'degraded' oder 'error'.
    - Optionaler simpler Header-Schutz via X-Health-Key, wenn ENV.HEALTH_API_KEY gesetzt ist.
    """
    server_time = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

    # Toleranzfenster für Heartbeat (sek)
    hb_nominal = float(_env_get("HEARTBEAT_SECS", 60))
    hb_grace = float(_env_get("HEARTBEAT_GRACE_SECS", 2 * hb_nominal))
    hb_deadline = hb_nominal + hb_grace
    now_ts = _now_ts()

    strategies: List[StrategyStatus] = []
    overall_status = "online"

    try:
        runners = getattr(ctrl, "runners", []) or []
        last_map = getattr(ctrl, "last_heartbeat", {}) or {}

        for r in runners:
            try:
                name = r.strategy.name()
            except Exception as e:
                strategies.append(
                    StrategyStatus(
                        name="unknown",
                        last_heartbeat=None,
                        healthy=False,
                        details=f"strategy.name() error: {e}",
                    )
                )
                overall_status = "degraded"
                continue

            ts = last_map.get(r, None)
            last_iso = _utc_iso(ts)
            healthy = None
            detail = None
            if ts is not None:
                try:
                    age = max(0.0, now_ts - float(ts))
                    healthy = age <= hb_deadline
                    if not healthy:
                        detail = f"stale heartbeat: age={age:.1f}s > deadline={hb_deadline:.1f}s"
                except Exception as e:
                    healthy = False
                    detail = f"heartbeat parse error: {e}"
                    overall_status = "degraded"
            else:
                healthy = False
                detail = "no heartbeat"
                overall_status = "degraded"

            strategies.append(
                StrategyStatus(
                    name=name,
                    last_heartbeat=last_iso,
                    healthy=healthy,
                    details=detail,
                )
            )

        # Wenn es Runner gibt und mindestens einer unhealthy ist -> degraded
        if runners and any(s.healthy is False for s in strategies):
            overall_status = "degraded"

    except Exception as e:
        # Harte Fehler im Controller führen zu 'error'
        return HealthResponse(
            env=ENV,
            status="error",
            strategies=[
                StrategyStatus(name="controller", details=f"controller error: {e}")
            ],
            server_time_utc=server_time,
        )

    return HealthResponse(
        env=ENV,
        status=overall_status,
        strategies=strategies,
        server_time_utc=server_time,
    )
