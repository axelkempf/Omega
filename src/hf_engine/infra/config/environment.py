"""
environment.py
--------------
Zentrale Umgebungs-/Konfigurationseinstellungen für die Engine.

- Lädt .env (falls vorhanden) und liest relevante Variablen.
- Validiert Pflichtparameter abhängig von Feature-Flags.
- Stellt konsistente, explizite Zeitzonenobjekte bereit.

Wichtige .env-Variablen (Auszug):
  ENVIRONMENT=dev|staging|prod                 (Default: dev)
  LOG_LEVEL=CRITICAL|ERROR|WARNING|INFO|DEBUG  (Default: INFO)

  # Telegram (nur falls TELEGRAM_ENABLED=true)
  TELEGRAM_ENABLED=true|false                  (Default: false)
  TELEGRAM_TOKEN=...                           (required wenn enabled)
  TELEGRAM_CHAT_ID=...                         (required wenn enabled)
  TELEGRAM_TOKEN_WATCHDOG=...                  (optional: separater Bot für Watchdogs)
  TELEGRAM_CHAT_ID_WATCHDOG=...                (optional: separater Chat für Watchdogs)
  TELEGRAM_TOKEN_WALKFORWARD=...               (optional: separater Bot für Walkforward)
  TELEGRAM_CHAT_ID_WALKFORWARD=...             (optional: separater Chat für Walkforward)

  # MT5 (optional – nur prüfen, wenn MT5_ENABLED=true)
  MT5_ENABLED=true|false                       (Default: false)
  MT5_LOGIN=...
  MT5_PASSWORD=...
  MT5_SERVER=...

  # Zeitzonen
  SYSTEM_TIMEZONE=UTC                          (Default: UTC)
  BROKER_TIMEZONE=Etc/GMT-3                    (Default: Etc/GMT-3; keine DST)
"""

from __future__ import annotations

import os
from datetime import (
    timezone as _dt_timezone,  # (bewusst nicht genutzt, nur für type hints)
)
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# --- Load .env -----------------------------------------------------------------
load_dotenv()


# --- Helpers -------------------------------------------------------------------
def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_choice(name: str, choices: set[str], default: str) -> str:
    raw = os.getenv(name, default).strip().upper()
    return raw if raw in choices else default


def _require(name: str, value: Optional[str]) -> None:
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")


# --- General -------------------------------------------------------------------
ENV: str = os.getenv("ENVIRONMENT", "dev").strip().lower()
LOG_LEVEL: str = _get_choice(
    "LOG_LEVEL",
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"},
    "INFO",
)

# --- Feature Flags --------------------------------------------------------------
TELEGRAM_ENABLED: bool = _get_bool("TELEGRAM_ENABLED", default=False)
MT5_ENABLED: bool = _get_bool("MT5_ENABLED", default=False)

# --- Telegram ------------------------------------------------------------------
TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN_WATCHDOG: Optional[str] = os.getenv("TELEGRAM_TOKEN_WATCHDOG")
TELEGRAM_CHAT_ID_WATCHDOG: Optional[str] = os.getenv("TELEGRAM_CHAT_ID_WATCHDOG")
TELEGRAM_TOKEN_WALKFORWARD: Optional[str] = os.getenv("TELEGRAM_TOKEN_WALKFORWARD")
TELEGRAM_CHAT_ID_WALKFORWARD: Optional[str] = os.getenv("TELEGRAM_CHAT_ID_WALKFORWARD")

if TELEGRAM_ENABLED:
    _require("TELEGRAM_TOKEN", TELEGRAM_TOKEN)
    _require("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)


# --- MT5 -----------------------------------------------------------------------
MT5_LOGIN: Optional[str] = os.getenv("MT5_LOGIN")
MT5_PASSWORD: Optional[str] = os.getenv("MT5_PASSWORD")
MT5_SERVER: Optional[str] = os.getenv("MT5_SERVER")

if MT5_ENABLED:
    _require("MT5_LOGIN", MT5_LOGIN)
    _require("MT5_PASSWORD", MT5_PASSWORD)
    _require("MT5_SERVER", MT5_SERVER)

# --- Timezones -----------------------------------------------------------------
# Systemweite Referenzzeitzone (Engine intern), Default: UTC
TIMEZONE: ZoneInfo = ZoneInfo(os.getenv("SYSTEM_TIMEZONE", "UTC"))

# Broker-/Datafeed-Zeitzone. Default ist 'Etc/GMT-3' (feste UTC+3, keine DST).
# Tipp: Für broker-/rechenzentrumsabhängige Zonen mit DST lieber eine IANA-Zone verwenden,
# z.B. 'Europe/Athens' oder 'Europe/Moscow'.
BROKER_TIMEZONE: ZoneInfo = ZoneInfo(os.getenv("BROKER_TIMEZONE", "Etc/GMT-3"))

# Backward compatible aliases (falls im Code bereits genutzt)
VANTAGE_TIMEZONE: ZoneInfo = BROKER_TIMEZONE

# --- Public API ----------------------------------------------------------------
__all__ = [
    "ENV",
    "LOG_LEVEL",
    "TELEGRAM_ENABLED",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_TOKEN_WATCHDOG",
    "TELEGRAM_CHAT_ID_WATCHDOG",
    "TELEGRAM_TOKEN_WALKFORWARD",
    "TELEGRAM_CHAT_ID_WALKFORWARD",
    "MT5_ENABLED",
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "TIMEZONE",
    "BROKER_TIMEZONE",
    "VANTAGE_TIMEZONE",
]
