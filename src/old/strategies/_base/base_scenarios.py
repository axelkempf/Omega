from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypedDict


class SignalDict(TypedDict, total=False):
    """
    Einheitliches Rückgabeformat eines Szenario-Evaluators.
    Minimale Felder: direction, entry, sl, tp, order_type.
    Optional: scenario (Name/String-ID), beliebige Metadaten.
    """

    direction: str  # "buy" | "sell"
    entry: float
    sl: float
    tp: float
    order_type: str  # z.B. "market", "limit", "stop"
    scenario: str


class BaseSzenario(ABC):
    """
    Schlanker Vertrag für beliebige Szenario-Implementierungen.
    - Keine starren Methodennamen (szenario_1_*, …)
    - Nur name() und evaluate_all() sind verpflichtend.
    - Optionale, NICHT-abstrakte Hooks für Cache & Datenbedarf,
      die Subklassen bei Bedarf überschreiben können.
    """

    # ---- Pflicht-API -----------------------------------------------------

    @abstractmethod
    def name(self) -> str:
        """Name der Strategie/Szenario-Gruppe (z. B. 'Mean_Reversion_Z_Score')."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_all(self, symbol: str, timeframe: str) -> Optional[SignalDict]:
        """
        Prüft alle internen Szenarien/Regeln und liefert beim ersten Treffer ein SignalDict.
        Kehrt None zurück, wenn kein Szenario zutrifft.
        MUSS ausfallsicher sein (Fehler intern abfangen oder None liefern).
        """
        raise NotImplementedError

    # ---- Optionale Hooks (Default-Implementierung) -----------------------

    def required_history(self) -> int:
        """
        Gibt an, wie viele Kerzen/Datapoints minimal benötigt werden (Faustwert).
        Subklassen können das für Vorab-Checks nutzen.
        """
        return 200

    def cache_key(
        self, symbol: str, timeframe: str, last_ts: int
    ) -> Tuple[str, str, int]:
        """
        Einheitlicher Schlüssel für Indikator-/Daten-Caches.
        Subklassen können das Schema bei Bedarf anpassen/erweitern.
        """
        return (symbol, timeframe, last_ts)

    def reset_cache(self) -> None:
        """
        Optionaler Hook zum invalidieren interner Caches (z. B. bei Config-Änderungen).
        """
        return

    def validate_market_gates(self, context: Dict[str, Any]) -> bool:
        """
        Optionaler Hook für Markt-Filter (Spread-/ATR-/Session-Gates etc.).
        Rückgabe True erlaubt Signale, False blockiert.
        Der Kontext kann z. B. letzte Kerze, ATR, Spread, Uhrzeit enthalten.
        """
        return True
