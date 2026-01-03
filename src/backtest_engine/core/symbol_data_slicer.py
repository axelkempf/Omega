from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class CandleSet:
    """
    Verwaltet mehrere Candle-Serien (bid/ask) für verschiedene Timeframes eines Symbols.
    """

    def __init__(
        self, tf_bid_candles: Dict[str, List[Any]], tf_ask_candles: Dict[str, List[Any]]
    ):
        self.tf_bid_candles = tf_bid_candles
        self.tf_ask_candles = tf_ask_candles

    def get_latest(
        self, timeframe: str, index: int, price_type: str = "bid"
    ) -> Optional[Any]:
        """
        Holt die Candle am Index für das gewünschte Timeframe und Preisart.

        Args:
            timeframe: Timeframe-String (z.B. 'M15').
            index: Aktueller Backtest-Index.
            price_type: "bid" oder "ask".

        Returns:
            Die Candle oder None.
        """
        data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
        return (
            data[timeframe][index]
            if timeframe in data and index < len(data[timeframe])
            else None
        )

    def get_all(self, timeframe: str, price_type: str = "bid") -> List[Any]:
        """
        Gibt alle Candles eines Timeframes zurück.

        Args:
            timeframe: Timeframe-String.
            price_type: "bid" oder "ask".

        Returns:
            Liste von Candles.
        """
        data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
        return data.get(timeframe, [])


class SymbolDataSlice:
    """
    Bietet bequemen Zugriff auf alle Bid/Ask-Candle-Daten für einen Index im Backtest
    über mehrere Timeframes.

    Args:
        multi_candle_data: Dict[TF][{"bid": [...], "ask": [...]}]
        index: Backtest-Index (int)
    """

    def __init__(
        self,
        multi_candle_data: Dict[str, Dict[str, List[Any]]],
        index: int,
        indicator_cache: Any = None,
    ):
        self.index = index
        self.tf_bid_candles: Dict[str, List[Any]] = {
            tf: candles["bid"]
            for tf, candles in multi_candle_data.items()
            if "bid" in candles
        }
        self.tf_ask_candles: Dict[str, List[Any]] = {
            tf: candles["ask"]
            for tf, candles in multi_candle_data.items()
            if "ask" in candles
        }
        self.indicators = indicator_cache
        self._history_cache: Dict[Tuple[str, str, int], deque] = {}
        self._last_index: int = index

    def set_index(self, index: int) -> None:
        """
        Setzt den aktuellen Index. Bei sequentieller Erhöhung (+1) werden
        alle vorhandenen History-Deques um die neue Candle erweitert.
        Bei Sprüngen wird der Cache sicherheitshalber geleert.
        """
        step = index - self.index
        if step == 1 and self._history_cache:
            for (tf, price_type, length), dq in self._history_cache.items():
                series = (
                    self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
                )
                seq = series.get(tf, [])
                if index < len(seq):
                    dq.append(seq[index])
        elif step != 0:
            # Sprung (z.B. Warmup-Cut) → Cache verwerfen
            self._history_cache.clear()
        self.index = index

    def get(self, timeframe: str, price_type: str = "bid") -> List[Any]:
        """
        Gibt alle Candles eines Timeframes (bid/ask) zurück.
        """
        data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
        return data.get(timeframe, [])

    def series_ref(self, timeframe: str, price_type: str = "bid") -> List[Any]:
        """
        Referenz auf die *originale* Candle-Liste (keine Kopie).
        Achtung: read-only verwenden.
        """
        data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
        return data.get(timeframe, [])

    def latest(self, timeframe: str, price_type: str = "bid") -> Optional[Any]:
        """
        Gibt die aktuelle Candle für das gegebene Timeframe und Preisart zurück.
        """
        data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
        if timeframe not in data or self.index >= len(data[timeframe]):
            return None
        return data[timeframe][self.index]

    def history(
        self, timeframe: str, price_type: str = "bid", length: int = 20
    ) -> List[Any]:
        """
        Gibt die letzten `length` Candles als Liste zurück – ohne Kopierorgien.
        Erstaufruf initialisiert einen deque(maxlen=length), set_index(+1) hängt an.
        """
        key = (timeframe, price_type, int(length))
        dq = self._history_cache.get(key)
        if dq is None:
            data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
            series = data.get(timeframe, [])
            if not series:
                return []
            start = max(0, self.index - length + 1)
            end = min(self.index + 1, len(series))
            dq = deque(series[start:end], maxlen=int(length))
            self._history_cache[key] = dq
        # Bisherige API: Liste zurückgeben (Kopie) – für Abwärtskompatibilität
        return list(dq)

    def history_view(
        self, timeframe: str, price_type: str = "bid", length: int = 20
    ) -> deque:
        """
        Gibt den *internen deque-Cache* zurück (keine Kopie).
        Read-only nutzen! set_index(+1) pflegt den deque automatisch.
        """
        key = (timeframe, price_type, int(length))
        dq = self._history_cache.get(key)
        if dq is None:
            data = self.tf_bid_candles if price_type == "bid" else self.tf_ask_candles
            series = data.get(timeframe, [])
            if not series:
                return deque(maxlen=int(length))
            start = max(0, self.index - length + 1)
            end = min(self.index + 1, len(series))
            dq = deque(series[start:end], maxlen=int(length))
            self._history_cache[key] = dq
        return dq

    def __getitem__(self, key: str) -> List[Any]:
        """
        Dict-Kompatibilität: slice["M15"] gibt alle Candles für M15 (bid).
        """
        return self.get(key)
