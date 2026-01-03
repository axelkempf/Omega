# broker_interface.py
"""
Abstraktes Interface für Broker‑Anbindungen (z. B. MetaTrader5, OANDA, Backtesting).

Ziele:
- Klare Entkopplung des Trading‑Kerns von der konkreten Broker‑API
- Konsistente Typen und Signaturen
- Einheitliche Erwartungen an Datums-/Zeitangaben (UTC, tz‑aware)

Hinweis zu Zeitstempeln:
Alle in/ausgehenden Zeitstempel sind als **UTC** und **tz‑aware** (datetime mit tzinfo) zu verstehen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """
    Ergebnis einer Order‑Aktion.

    Wichtiger Hinweis / Backwards‑Compat:
    - In einer früheren Version wurde `order_id` fälschlich mit der Magic‑Number belegt.
      Um Kompatibilität zu erhalten, bleibt `order_id` bestehen. Zusätzlich existiert
      jetzt `magic_number`, sodass Implementierungen beide Felder sinnvoll setzen können.
    """

    success: bool
    message: str = ""
    # historische Reihenfolge beibehalten (Kompatibilität für Positionals)
    magic_number: Optional[int] = None
    timestamp: Optional[datetime] = None  # UTC, tz‑aware
    order: Optional[int] = None  # Broker‑Order/Ticket‑ID (optional)
    # neues, sauberes Feld (optional zu befüllen von Implementierungen)
    order_id: Optional[int] = None  # bevorzugtes Feld für Broker‑Order/Ticket‑ID

    def __repr__(self) -> str:
        return (
            f"OrderResult(success={self.success}, message='{self.message}', "
            f"order_id={self.order_id}, magic_number={self.magic_number})"
        )


@dataclass
class Position:
    position_id: int
    symbol: str
    entry: float
    direction: Direction
    sl: float
    tp: float
    volume: float
    comment: str
    open_time: datetime  # UTC, tz‑aware

    def __repr__(self) -> str:
        return f"<Position {self.symbol} {self.direction} @ {self.entry} Vol={self.volume}>"


@dataclass
class TradeResult:
    symbol: str
    entry_price: float
    exit_price: float
    direction: Direction
    volume: float
    entry_time: datetime  # UTC, tz‑aware
    exit_time: datetime  # UTC, tz‑aware
    profit: float
    comment: str
    fees: float = 0.0
    swap: float = 0.0
    commission: float = 0.0


class BrokerInterface(ABC):
    """
    Abstrakte Basisklasse für Broker‑Anbindungen. Definiert die notwendigen Methoden,
    die jede konkrete Broker‑Implementierung bereitstellen muss.
    """

    # --- Lifecycle / Connection -------------------------------------------------

    @abstractmethod
    def _connect_mt5(self) -> bool:
        """Initialisiert die Verbindung zum Broker/API. Gibt True bei Erfolg zurück."""
        pass

    @abstractmethod
    def ensure_connection(self) -> bool:
        """
        Überprüft die Verbindung und versucht bei Bedarf einen Reconnect.
        Gibt True zurück, wenn die Verbindung funktionsfähig ist.
        """
        pass

    @abstractmethod
    def __del__(self) -> None:
        """Sorgt für sauberes Aufräumen (falls nötig)."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Beendet die Verbindung zum Broker/API und räumt Ressourcen auf."""
        pass

    @abstractmethod
    def set_magic_number(self, new_magic: int) -> None:
        """Setzt/aktualisiert die Magic‑Number für nachfolgende Aufträge."""
        pass

    @abstractmethod
    def _validate_account(self) -> None:
        """Validiert Konto‑Status/Berechtigungen und wirft bei Problemen eine Exception."""
        pass

    # --- Account / Symbol Info --------------------------------------------------

    @abstractmethod
    def get_account_equity(self) -> float:
        """Aktuelle Kontoequity."""
        pass

    @abstractmethod
    def get_account_currency(self) -> str:
        """Kontowährung (z. B. 'USD', 'EUR')."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Symbol‑Metainformationen. Muss mindestens 'digits' enthalten.
        Beispiel: {'digits': 5, 'lot_step': 0.01, ...}
        """
        pass

    @abstractmethod
    def get_symbol_tick(self, symbol: str) -> Dict[str, float]:
        """Aktueller Tick mit mindestens 'bid' und 'ask' als float."""
        pass

    @abstractmethod
    def get_symbol_price(self, symbol: str, direction: Direction) -> Optional[float]:
        """
        Preis für Kauf/Verkauf eines Symbols.
        direction: BUY → ask, SELL → bid (je nach Implementierung).
        """
        pass

    @abstractmethod
    def get_symbol_spread(self, symbol: str) -> float:
        """Spread zwischen ask und bid (in Preis‑Einheiten)."""
        pass

    @abstractmethod
    def get_min_lot_size(self, symbol: str) -> float:
        """Minimale Lotgröße für das Symbol."""
        pass

    # --- Risk / Analytics -------------------------------------------------------

    @abstractmethod
    def calculate_risk_amount(
        self, symbol: str, entry_price: float, sl_price: float, volume: float
    ) -> float:
        """Berechnet die riskierte Geldmenge für den geplanten Trade."""
        pass

    @abstractmethod
    def get_current_r_multiple(
        self, symbol: str, initial_sl: float, ticket_id: Optional[int] = None
    ) -> float:
        """Aktueller R‑Multiple eines Trades (floating)."""
        pass

    @abstractmethod
    def reconstruct_trade_from_deal_ticket(
        self, ticket_id: int, pip_size: float = 0.0001
    ) -> Optional[Dict[str, Any]]:
        """Rekonstruiert Trade‑Daten aus einem Deal‑Ticket als Dict, oder None."""
        pass

    @abstractmethod
    def get_realized_r_multiple(self, magic_number: int) -> float:
        """Bereits realisierter R‑Multiple (aggregiert) für die Strategie/Magic."""
        pass

    @abstractmethod
    def get_floating_r_multiple(self, strategy: str) -> float:
        """Aktueller (unrealisierter) R‑Multiple über alle offenen Trades der Strategie."""
        pass

    @abstractmethod
    def get_total_r_multiple(self, magic_number: int, strategy: str) -> float:
        """Gesamter R‑Multiple (realisiert + floating) für die Strategie."""
        pass

    # --- Orders / Positions -----------------------------------------------------

    @abstractmethod
    def place_pending_order(
        self,
        symbol: str,
        direction: Direction,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        comment: str = "",
        order_type: OrderType = OrderType.STOP,
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        """
        Platziert einen Pending‑Order.
        order_type: STOP (BUY_STOP/SELL_STOP) oder LIMIT (BUY_LIMIT/SELL_LIMIT)
        """
        pass

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        direction: Direction,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        volume: float = 1.0,
        comment: str = "Market Order",
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        """Platziert eine sofortige Market‑Order."""
        pass

    @abstractmethod
    def cancel_all_pending_orders(self, symbol: str) -> None:
        """Löscht alle offenen Pending‑Orders für das Symbol."""
        pass

    @abstractmethod
    def cancel_pending_order_by_ticket(
        self, ticket_id: int, magic_number: Optional[int] = None
    ) -> bool:
        """Löscht eine Pending‑Order per Ticket‑ID (optional gefiltert per Magic‑Number)."""
        pass

    @abstractmethod
    def get_pending_order_by_ticket(
        self, ticket_id: int, magic_number: Optional[int] = None
    ) -> Optional[object]:
        """
        Gibt die eigene Pending‑Order zur Ticket‑ID zurück, sofern vorhanden und (falls angegeben)
        zur Magic‑Number passend. Rückgabetyp ist implementationsspezifisch.
        """
        pass

    @abstractmethod
    def get_own_pending_orders(
        self, symbol: str, magic_number: Optional[int] = None
    ) -> Optional[object]:
        """
        Gibt die eigene(n) Pending‑Order(s) für ein Symbol zurück (Implementierungsspezifik).
        Hinweis: Einige Broker erlauben mehrere gleichzeitige Pending‑Orders pro Symbol.
        """
        pass

    @abstractmethod
    def get_own_position(
        self, symbol: str, magic_number: Optional[int] = None
    ) -> Optional[Position]:
        """Eigene offene Position für ein Symbol (falls vorhanden)."""
        pass

    @abstractmethod
    def get_all_own_positions(
        self, magic_number: Optional[int] = None
    ) -> List[Position]:
        """Alle aktuell offenen Positionen (optional gefiltert per Magic‑Number)."""
        pass

    @abstractmethod
    def get_position_by_ticket(self, ticket: int) -> Optional[Position]:
        """Einzelne Position per exakter Ticket‑ID, oder None."""
        pass

    @abstractmethod
    def position_direction(self, pos: Position) -> Direction:
        """Handelsrichtung der Position."""
        pass

    @abstractmethod
    def pending_order_direction(self, order: object) -> Direction:
        """Handelsrichtung der Pending‑Order."""
        pass

    @abstractmethod
    def modify_sl(self, position_id: int, new_sl: float) -> OrderResult:
        """Setzt/aktualisiert den Stop‑Loss einer bestehenden Position."""
        pass

    @abstractmethod
    def modify_tp(self, position_id: int, new_tp: float) -> OrderResult:
        """Aktualisiert den Take‑Profit einer Position."""
        pass

    @abstractmethod
    def close_position_partial(self, position_id: int, volume: float) -> OrderResult:
        """Schließt einen Teil der Position (Teilgewinnmitnahme)."""
        pass

    @abstractmethod
    def close_position_full(self, position_id: int) -> OrderResult:
        """Schließt die gesamte Position."""
        pass
