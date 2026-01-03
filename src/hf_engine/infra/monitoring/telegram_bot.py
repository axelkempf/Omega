# telegram_bot.py
"""
Telegram Bot für Benachrichtigungen (Erfolge, Fehler, Warnungen, Status).
Robust gegen Netzwerkfehler, mit Retry/Backoff, konfigurierbarem Timeout,
sicherem Escaping (MarkdownV2/HTML) und automatischem Chunking >4096 Zeichen.
"""

from __future__ import annotations

import html
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Tuple

import requests

from hf_engine.infra.config.environment import (
    TELEGRAM_CHAT_ID,
    TELEGRAM_CHAT_ID_WALKFORWARD,
    TELEGRAM_CHAT_ID_WATCHDOG,
    TELEGRAM_TOKEN,
    TELEGRAM_TOKEN_WALKFORWARD,
    TELEGRAM_TOKEN_WATCHDOG,
)
from hf_engine.infra.logging.log_service import log_service

# --- Konstanten / Defaults ---
_TELEGRAM_API_BASE = "https://api.telegram.org"
_MAX_MSG_LEN = 4096  # Telegram-Limit für Textnachrichten
_DEFAULT_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "10"))
_DEFAULT_MAX_RETRIES = int(os.getenv("TELEGRAM_MAX_RETRIES", "3"))
_DEFAULT_BACKOFF_BASE = float(os.getenv("TELEGRAM_BACKOFF_BASE", "0.5"))  # Sekunden
_DEFAULT_BACKOFF_MAX = float(os.getenv("TELEGRAM_BACKOFF_MAX", "5"))  # Sekunden
_DEFAULT_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "MarkdownV2")  # Sicherer Default
_FIRE_AND_FORGET = os.getenv("TELEGRAM_FIRE_AND_FORGET", "0") == "1"

# Kleiner Threadpool, falls fire_and_forget genutzt wird (blockiert Hauptthread nicht)
_executor: Optional[ThreadPoolExecutor] = (
    ThreadPoolExecutor(max_workers=2, thread_name_prefix="tg-bot")
    if _FIRE_AND_FORGET
    else None
)


# --- Hilfsfunktionen: Escaping & Chunking ---

_MD_V2_SPECIALS = r"_*[]()~`>#+-=|{}.!"  # Telegram MarkdownV2: muss escaped werden


def _escape_markdown_v2(text: str) -> str:
    """Escape für Telegram MarkdownV2 gemäß Spezifikation."""
    # Backslash zuerst escapen, dann alle Specials
    text = text.replace("\\", "\\\\")
    for ch in _MD_V2_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


def _sanitize_text(text: str, parse_mode: Optional[str]) -> str:
    if not parse_mode:
        return text
    pm = parse_mode.upper()
    if pm in ("MARKDOWNV2", "MARKDOWN"):
        # Für Markdown nehmen wir das strengere MarkdownV2-Escaping
        return _escape_markdown_v2(text)
    if pm == "HTML":
        # HTML sicher escapen; wer formatiert senden will, kann parse_mode=None setzen
        return html.escape(text, quote=False)
    return text


def _chunks(text: str, limit: int = _MAX_MSG_LEN) -> Iterable[str]:
    """Zerlegt lange Nachrichten in sinnvolle Blöcke <= limit."""
    if len(text) <= limit:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        # Versuche, sauber an einem Zeilenumbruch zu trennen
        split_pos = text.rfind("\n", start, end)
        if split_pos == -1 or split_pos <= start + int(0.5 * limit):
            # Falls kein guter Zeilenumbruch, hart trennen
            split_pos = end
        yield text[start:split_pos]
        start = split_pos


def _backoff_sleep(attempt: int) -> None:
    # Exponentialer Backoff mit leichter Zufallsjitter-Beimischung
    base = _DEFAULT_BACKOFF_BASE * (2 ** (attempt - 1))
    sleep_for = min(_DEFAULT_BACKOFF_MAX, base * (0.8 + 0.4 * random.random()))
    time.sleep(sleep_for)


# --- HTTP-Client ---


class _TelegramClient:
    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[str],
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def _endpoint(self, method: str) -> str:
        return f"{_TELEGRAM_API_BASE}/bot{self.token}/{method}"

    def _post(self, url: str, data: dict) -> Tuple[int, str]:
        resp = self.session.post(url, data=data, timeout=self.timeout)
        return resp.status_code, resp.text

    def send_message(
        self,
        message: str,
        parse_mode: Optional[str] = _DEFAULT_PARSE_MODE,
        disable_notification: bool = False,
        message_thread_id: Optional[int] = None,
    ) -> bool:
        # Frühzeitige Validierung
        if not self.token or not self.chat_id:
            log_service.log_system(
                "[Telegram Warnung] Token oder Chat-ID fehlen – Nachricht nicht gesendet."
            )
            return False

        if not isinstance(message, str) or not message.strip():
            log_service.log_system(
                "[Telegram Warnung] Leere oder ungültige Nachricht – übersprungen."
            )
            return False

        # Sanitizing & Chunking
        sanitized = _sanitize_text(message, parse_mode)
        parts: List[str] = list(_chunks(sanitized, _MAX_MSG_LEN))

        all_ok = True
        for idx, part in enumerate(parts, start=1):
            url = self._endpoint("sendMessage")
            payload = {
                "chat_id": self.chat_id,
                "text": part,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if disable_notification:
                payload["disable_notification"] = True
            if message_thread_id is not None:
                payload["message_thread_id"] = message_thread_id

            attempt = 0
            while True:
                attempt += 1
                try:
                    status, body = self._post(url, payload)

                    if status == 200:
                        break  # OK für diesen Part

                    # 429: Rate Limit – Retry-After respektieren, aber capped
                    if status == 429:
                        retry_after = 1.0
                        # Telegram sendet häufig {"parameters":{"retry_after": X}}
                        # Da wir body nicht parsen müssen, backoffen wir konservativ
                        _backoff_sleep(attempt)
                        if attempt <= self.max_retries:
                            continue

                    # Sonstige HTTP-Fehler
                    log_service.log_system(
                        f"[Telegram Fehler] HTTP {status} beim Senden (Teil {idx}/{len(parts)}): {body}"
                    )
                    all_ok = False
                    break

                except requests.Timeout:
                    log_service.log_system(
                        f"[Telegram Fehler] Timeout beim Senden (Teil {idx}/{len(parts)}), Versuch {attempt}."
                    )
                    if attempt <= self.max_retries:
                        _backoff_sleep(attempt)
                        continue
                    all_ok = False
                    break

                except requests.RequestException as e:
                    log_service.log_system(
                        f"[Telegram Fehler] Netzwerkfehler (Teil {idx}/{len(parts)}), Versuch {attempt}: {e}"
                    )
                    if attempt <= self.max_retries:
                        _backoff_sleep(attempt)
                        continue
                    all_ok = False
                    break

                except Exception as e:  # Unerwartete Fehler sauber protokollieren
                    log_service.log_system(
                        f"[Telegram Fehler] Unerwarteter Fehler (Teil {idx}/{len(parts)}): {e}"
                    )
                    all_ok = False
                    break

        return all_ok


# Singleton-Client auf Basis der Environment-Config
_client = _TelegramClient(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


def send_telegram_message(
    message: str,
    parse_mode: str = _DEFAULT_PARSE_MODE,
    *,
    disable_notification: bool = False,
    message_thread_id: Optional[int] = None,
    fire_and_forget: Optional[bool] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """
    Sendet eine Nachricht an den konfigurierten Telegram-Chat.

    Args:
        message: Inhalt der Nachricht.
        parse_mode: "MarkdownV2" (empfohlen), "Markdown", "HTML" oder None.
        disable_notification: Stilles Senden ohne Push.
        message_thread_id: Für Foren/Topics innerhalb von Gruppen.
        fire_and_forget: Wenn True, asynchron im kleinen Threadpool.
                         Default: aus ENV ("TELEGRAM_FIRE_AND_FORGET").
        timeout: Optionaler Timeout-Override in Sekunden.
        max_retries: Optionaler Retry-Override.

    Optional:
        token: Override für Bot-Token (z.B. separater Watchdog-Bot).
        chat_id: Override für Ziel-Chat (z.B. separater Watchdog-Channel).

    Returns:
        bool: True bei Erfolg (alle Chunks gesendet), sonst False.
              Bei fire_and_forget wird immer True zurückgegeben (Dispatch-Erfolg).
    """
    # Ziel-Credentials bestimmen (optional mit Override, z.B. Watchdog-Bot)
    base_token = _client.token
    base_chat_id = _client.chat_id

    effective_token = token if token is not None else base_token
    effective_chat_id = chat_id if chat_id is not None else base_chat_id

    # Optional Client-Klon mit Overrides erzeugen, ohne das Singleton zu verändern
    client = _client
    if (
        timeout is not None
        or max_retries is not None
        or token is not None
        or chat_id is not None
    ):
        client = _TelegramClient(
            token=effective_token,
            chat_id=effective_chat_id,
            timeout=timeout if timeout is not None else _client.timeout,
            max_retries=max_retries if max_retries is not None else _client.max_retries,
            session=_client.session,
        )

    faf = _FIRE_AND_FORGET if fire_and_forget is None else fire_and_forget

    if faf and _executor is not None:
        # Asynchron ausführen, Rückgabewert signalisiert nur erfolgreichen Dispatch
        try:
            _executor.submit(
                client.send_message,
                message,
                parse_mode,
                disable_notification,
                message_thread_id,
            )
            return True
        except Exception as e:
            log_service.log_system(
                f"[Telegram Fehler] Asynchroner Dispatch fehlgeschlagen: {e}"
            )
            return False

    return client.send_message(
        message=message,
        parse_mode=parse_mode,
        disable_notification=disable_notification,
        message_thread_id=message_thread_id,
    )


def send_watchdog_telegram_message(
    message: str,
    parse_mode: str = _DEFAULT_PARSE_MODE,
    *,
    disable_notification: bool = False,
    message_thread_id: Optional[int] = None,
    fire_and_forget: Optional[bool] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> bool:
    """
    Convenience-Wrapper für Watchdog-Benachrichtigungen.

    Nutzt TELEGRAM_TOKEN_WATCHDOG / TELEGRAM_CHAT_ID_WATCHDOG,
    fällt bei fehlender Konfiguration auf den Standard-Bot zurück.
    """
    if not TELEGRAM_TOKEN_WATCHDOG or not TELEGRAM_CHAT_ID_WATCHDOG:
        log_service.log_system(
            "[Telegram Watchdog] Watchdog-Credentials fehlen – nutze Standard-Bot.",
            level="WARNING",
        )
        return send_telegram_message(
            message=message,
            parse_mode=parse_mode,
            disable_notification=disable_notification,
            message_thread_id=message_thread_id,
            fire_and_forget=fire_and_forget,
            timeout=timeout,
            max_retries=max_retries,
        )

    return send_telegram_message(
        message=message,
        parse_mode=parse_mode,
        disable_notification=disable_notification,
        message_thread_id=message_thread_id,
        fire_and_forget=fire_and_forget,
        timeout=timeout,
        max_retries=max_retries,
        token=TELEGRAM_TOKEN_WATCHDOG,
        chat_id=TELEGRAM_CHAT_ID_WATCHDOG,
    )


def send_walkforward_telegram_message(
    message: str,
    parse_mode: str = _DEFAULT_PARSE_MODE,
    *,
    disable_notification: bool = False,
    message_thread_id: Optional[int] = None,
    fire_and_forget: Optional[bool] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> bool:
    """
    Convenience-Wrapper für Walkforward-Benachrichtigungen.

    Nutzt TELEGRAM_TOKEN_WALKFORWARD / TELEGRAM_CHAT_ID_WALKFORWARD,
    fällt bei fehlender Konfiguration auf den Standard-Bot zurück.
    """
    if not TELEGRAM_TOKEN_WALKFORWARD or not TELEGRAM_CHAT_ID_WALKFORWARD:
        log_service.log_system(
            "[Telegram Walkforward] Walkforward-Credentials fehlen – nutze Standard-Bot.",
            level="WARNING",
        )
        return send_telegram_message(
            message=message,
            parse_mode=parse_mode,
            disable_notification=disable_notification,
            message_thread_id=message_thread_id,
            fire_and_forget=fire_and_forget,
            timeout=timeout,
            max_retries=max_retries,
        )

    return send_telegram_message(
        message=message,
        parse_mode=parse_mode,
        disable_notification=disable_notification,
        message_thread_id=message_thread_id,
        fire_and_forget=fire_and_forget,
        timeout=timeout,
        max_retries=max_retries,
        token=TELEGRAM_TOKEN_WALKFORWARD,
        chat_id=TELEGRAM_CHAT_ID_WALKFORWARD,
    )
