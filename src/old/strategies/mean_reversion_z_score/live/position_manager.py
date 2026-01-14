# hf_engine/strategies/mean_reversion_z_score/live/position_manager.py
"""Position manager for mean_reversion_z_score live strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from strategies._base.base_position_manager import BasePositionManager

from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.adapter.broker.broker_utils import get_pip_size
from hf_engine.adapter.data.mt5_data_provider import MT5DataProvider
from hf_engine.core.execution.execution_tracker import ExecutionTracker
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.log_service import log_service

if TYPE_CHECKING:
    from strategies._base.base_strategy import TradeSetup


class StrategyPositionManager(BasePositionManager):
    """
    Positionsmanager im reinen Step-Modus.

    Erwartet, dass 'monitor_step()' True zurückgibt, wenn das Monitoring beendet werden kann.
    Beendet Überwachung, sobald die Position (Ticket) nicht mehr existiert.
    """

    def __init__(
        self,
        setup: TradeSetup,
        broker: BrokerInterface,
        data_provider: MT5DataProvider,
    ) -> None:
        """
        Initialize the position manager.

        Args:
            setup: Trade setup containing position details.
            broker: Broker interface for trade operations.
            data_provider: Market data provider.
        """
        super().__init__(setup, broker, data_provider)

        md: dict = getattr(setup, "metadata", {}) or {}
        # Szenario-Name (z. B. "szenario_3_long") aus Metadaten
        self.scenario: str | None = md.get("scenario")
        max_holding_minutes = getattr(
            setup, "max_holding_minutes", md.get("max_holding_minutes")
        )
        self.direction: str | None = getattr(setup, "direction", md.get("direction"))
        self.ticket_id: int | str | None = getattr(
            setup, "ticket_id", md.get("ticket_id")
        )
        self.sl: float | None = getattr(setup, "sl", md.get("sl"))
        self.tp: float | None = getattr(setup, "tp", md.get("tp"))
        self.entry: float | None = getattr(setup, "entry", md.get("entry"))

        self._done: bool = False
        self._cancel: bool = False
        # Startzeit: bevorzugt die echte Entry‑Zeit (Resume‑Fälle),
        # ansonsten Zeitpunkt der Manager-Erstellung – nicht mehr den Kerzen‑Timestamp.
        self._started_at: datetime = self._parse_entry_time(
            getattr(setup, "entry_time", None) or md.get("entry_time")
        )
        # Max-Haltedauer robust ermitteln (0 = deaktiviert)
        try:
            _mh = int(max_holding_minutes or 0)
        except Exception:
            _mh = 0
        self._max_hold: timedelta = timedelta(minutes=_mh)
        # Nur für Szenario 3 (long/short) aktivieren
        scen = str(self.scenario or "").lower()
        self._use_max_hold: bool = scen in ("szenario_3_long", "szenario_3_short")
        self.pip_size: float | None = get_pip_size(self.symbol)

    def _parse_entry_time(self, raw: datetime | str | None) -> datetime:
        """
        Parse entry time from various formats to timezone-aware datetime.

        Args:
            raw: Entry time as datetime, ISO string, or None.

        Returns:
            Timezone-aware datetime, defaults to current UTC time if parsing fails.
        """
        if raw is None:
            return now_utc()
        try:
            if isinstance(raw, datetime):
                return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            s = str(raw).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return now_utc()

    # optionale Hooks
    def stop_monitoring(self) -> None:
        """Stop monitoring the position."""
        self._cancel = True

    def cancel(self) -> None:
        """Cancel the position monitoring."""
        self._cancel = True

    def monitor_step(self) -> bool:
        """
        Execute one monitoring step for the position.

        Returns:
            True if monitoring is complete (position closed or cancelled),
            False to continue monitoring.
        """
        if self._done or self._cancel:
            if self._cancel:
                log_service.log_system(
                    f"[Monitor] ⏹️ Monitoring abgebrochen für {self.symbol} ({self.ticket_id})"
                )
            return True

        # 1) Max-Haltedauer nur für Szenario 3 (long/short) anwenden
        if (
            self._use_max_hold
            and self._max_hold.total_seconds() > 0
            and (now_utc() - self._started_at >= self._max_hold)
        ):
            try:
                log_service.log_system(
                    f"[Monitor] ⏱️ Max-Haltedauer erreicht ({self._max_hold}) für {self.symbol} / {self.ticket_id} (Scenario={self.scenario}) – schließe Position"
                )
            except Exception:
                pass
            # versuche sauberes Close über Broker
            pos = self.broker.get_position_by_ticket(self.ticket_id)
            if pos:
                try:
                    self.broker.close_position_full(self.ticket_id)
                except Exception as e:
                    log_service.log_system(
                        f"[DummyMon] ⚠️ Close-Versuch scheiterte: {e}", level="WARNING"
                    )
            self._track_close_and_finish()
            return True

        # Robust gegen Broker-Fehler
        try:
            pos = self.broker.get_position_by_ticket(self.ticket_id)
        except Exception as e:
            log_service.log_system(
                f"[Monitor] ❌ Brokerfehler bei Positionsabfrage {self.symbol}/{self.ticket_id}: {e}",
                level="ERROR",
            )
            return False  # nächster Step erneut versuchen

        # Nur wenn Position NICHT mehr existiert, Tracking abschließen
        try:
            if pos:
                return False  # weiter monitoren
        except Exception:
            # konservativ: bei unlesbarem Objekt im nächsten Schritt erneut prüfen
            return False

        self._track_close_and_finish()
        return True

    # Internals
    def _track_close_and_finish(self) -> None:
        """Track position close in execution tracker and log the trade."""
        try:
            tracker = ExecutionTracker()
            exit_price = self.broker.get_symbol_price(
                self.symbol, "sell" if self.direction == "buy" else "buy"
            )
            # Verwende denselben Composite-Tracker-Key (Suffix) wie beim Open:
            # ExecutionEngine hat metadata['setup'] als 'strategy'-Teil des Keys verwendet.
            strategy_key = (getattr(self.setup, "metadata", {}) or {}).get(
                "setup", self.strategy_name
            )
            tracker.mark_trade_closed(
                self.symbol,
                strategy_key,
                exit_price=exit_price,
                direction=self.direction,
            )
            log_service.log_system(
                f"[Tracker] 📉 Position geschlossen: {self.symbol} / {self.ticket_id}"
            )

            # Trade-Rekonstruktion und Logging
            try:
                pip_size = self.pip_size or get_pip_size(self.symbol)
                trade = self.broker.reconstruct_trade_from_deal_ticket(
                    int(self.ticket_id), float(pip_size)
                )
            except Exception as e:
                trade = None
                log_service.log_system(
                    f"[Monitor] ❌ Trade-Rekonstruktion fehlgeschlagen {self.symbol}/{self.ticket_id}: {e}",
                    level="ERROR",
                )

            if trade:
                trade["strategy"] = self.strategy_name
                trade["sl"] = self.sl
                trade["tp"] = self.tp
                # Flags anhand Exit-Typ
                exit_type = (trade.get("exit_type") or "").strip()
                trade["sl_hit"] = 1 if exit_type == "StopLoss" else 0
                trade["tp_hit"] = 1 if exit_type == "TakeProfit" else 0
                # Vollständige Metadaten inkl. Indikatorwerte vom Entry
                trade["metadata"] = getattr(self.setup, "metadata", {}) or {}

                log_service.log_trade(trade)
            else:
                log_service.log_system(
                    f"[Monitor] ❌ Trade konnte nicht geloggt werden für {self.symbol}",
                    level="ERROR",
                )
        except Exception as e:
            log_service.log_system(
                f"[Monitor] ❌ Close-Tracking Fehler {self.symbol}/{self.ticket_id}: {e}",
                level="ERROR",
            )
        finally:
            self._done = True
