from typing import Any, Dict, Optional


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

    def __getitem__(self, symbol: str):
        """
        Ermöglicht dict-ähnlichen Zugriff: slice[symbol]
        Gibt eine SliceView zurück.
        """

        class SliceView:
            """
            Kapselt Candle und bietet Zugriff auf latest().
            """

            def __init__(self, candle):
                self.candle = candle

            def latest(self, tf: str = None, price_type: str = "bid"):
                """
                Gibt die aktuelle Candle zurück.
                """
                return self.candle

        return SliceView(self.get(symbol))

    def set_timestamp(self, timestamp: Any) -> None:
        self.timestamp = timestamp

    def keys(self):
        """
        Gibt alle Symbol-Keys zurück.
        """
        return self.candle_lookups.keys()

    def __iter__(self):
        """
        Ermöglicht Iteration über alle Symbol-Keys.
        """
        return iter(self.candle_lookups.keys())

    def __len__(self):
        """
        Gibt die Anzahl der Symbole zurück.
        """
        return len(self.candle_lookups)
