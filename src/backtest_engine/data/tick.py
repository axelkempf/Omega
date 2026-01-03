from dataclasses import dataclass
from datetime import datetime


@dataclass
class Tick:
    """
    Repräsentiert einen einzelnen Tick (Bid/Ask-Quote) im Backtest oder Livehandel.

    Attributes:
        timestamp: Zeitpunkt des Ticks (UTC, datetime).
        bid: Bid-Preis.
        ask: Ask-Preis.
        volume: (Optional) Tick-Volumen, falls vorhanden (default 0.0).
    """

    timestamp: datetime
    bid: float
    ask: float
    volume: float = 0.0
