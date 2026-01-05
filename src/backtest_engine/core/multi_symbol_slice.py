from typing import Any, Dict, Iterator, KeysView, Optional


class MultiSymbolSlice:
    """
    Stellt für einen Timestamp synchronisierte Bid-/Ask-Candles für mehrere Symbole bereit.

    Args:
        candle_lookups: Dict[symbol][price_type][timestamp] -> Candle
        timestamp: Der aktuelle Zeitstempel
        primary_tf: Haupt-Timeframe (z.B. 'M15')
    """

    def __init__(
        self,
        candle_lookups: Dict[str, Dict[str, Dict[Any, Any]]],
        timestamp: Any,
        primary_tf: str,
    ):
        self.candle_lookups = candle_lookups
        self.timestamp = timestamp
        self.primary_tf = primary_tf
        self._index: int = 0

    @property
    def slices(self) -> KeysView[str]:
        """Alias for keys() for backward compatibility."""
        return self.candle_lookups.keys()

    def set_index(self, index: int) -> None:
        """Speichert den aktuellen Index (für Index-basierte Iteration)."""
        self._index = index

    def get(self, symbol: str, price_type: str = "bid") -> Optional[Any]:
        """
        Holt die Candle für das gegebene Symbol, Preis-Typ und Timestamp.

        Args:
            symbol: Symbolname (z.B. 'EURUSD')
            price_type: 'bid' oder 'ask'

        Returns:
            Die passende Candle oder None.
        """
        return (
            self.candle_lookups.get(symbol, {})
            .get(price_type, {})
            .get(self.timestamp, None)
        )

    def __getitem__(self, symbol: str) -> "SliceView":
        """
        Ermöglicht dict-ähnlichen Zugriff: slice[symbol]
        Gibt eine SliceView zurück.
        """
        return SliceView(self.get(symbol))

    def set_timestamp(self, timestamp: Any) -> None:
        self.timestamp = timestamp

    def keys(self) -> KeysView[str]:
        """
        Gibt alle Symbol-Keys zurück.
        """
        return self.candle_lookups.keys()

    def __iter__(self) -> Iterator[str]:
        """
        Ermöglicht Iteration über alle Symbol-Keys.
        """
        return iter(self.candle_lookups.keys())

    def __len__(self) -> int:
        """
        Gibt die Anzahl der Symbole zurück.
        """
        return len(self.candle_lookups)


class SliceView:
    """
    Kapselt Candle und bietet Zugriff auf latest().
    """

    def __init__(self, candle: Optional[Any]) -> None:
        self.candle = candle

    def latest(self, tf: Optional[str] = None, price_type: str = "bid") -> Optional[Any]:
        """
        Gibt die aktuelle Candle zurück.
        """
        return self.candle
