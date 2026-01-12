from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

try:
    # Daten-Loader aus deinem Backtester nutzen
    from backtest_engine.data.data_handler import CSVDataHandler
except Exception:
    CSVDataHandler = None  # erlaubt Import in reinen Testumgebungen


@dataclass(frozen=True)
class FxPath:
    path: List[str]  # e.g. ["AUD->EUR"] or ["AUD->USD", "USD->EUR"]


class RateProvider:
    """Abstract FX rate provider (mid rates)."""

    def get(self, pair: str, t=None) -> float:
        """Return mid rate for 'EURUSD' (= USD per EUR)."""
        raise NotImplementedError

    def fx_convert(
        self,
        amount: float,
        from_ccy: str,
        to_ccy: str,
        t=None,
        via: Optional[str] = "USD",
    ) -> Tuple[float, FxPath]:
        """
        Convert amount from from_ccy → to_ccy:
          1) direct pair (TO per FROM):  amount * get(FROM+TO)
          2) inverse pair (FROM per TO): amount / get(TO+FROM)
          3) triangulate via 'via':      amount * get(FROM+via) * get(via+TO)
        Raises ValueError if no path is found.
        """
        if from_ccy == to_ccy:
            return float(amount), FxPath([f"{from_ccy}->{to_ccy}"])

        direct = f"{from_ccy}{to_ccy}"
        inverse = f"{to_ccy}{from_ccy}"
        try:
            r = self.get(direct, t)
            if r is not None:
                return float(amount) * float(r), FxPath([f"{from_ccy}->{to_ccy}"])
        except Exception:
            pass

        try:
            r_inv = self.get(inverse, t)
            if r_inv is not None and r_inv != 0.0:
                return float(amount) / float(r_inv), FxPath([f"{from_ccy}<-{to_ccy}"])
        except Exception:
            pass

        if not via:
            raise ValueError(f"No FX path {from_ccy}->{to_ccy} and no 'via' provided.")
        if via in (from_ccy, to_ccy):
            # simple guard; still try
            pass

        # Triangulate via hub 'via'
        path = []
        amt1, p1 = self.fx_convert(amount, from_ccy, via, t=t, via=None)
        path += p1.path
        amt2, p2 = self.fx_convert(amt1, via, to_ccy, t=t, via=None)
        path += p2.path
        return amt2, FxPath(path)


class StaticRateProvider(RateProvider):
    """
    Deterministic static provider for backtests & unit tests.
    rates: dict like {"EURUSD": 1.10, "EURAUD": 1.65, "GBPAUD": 1.90, ...}
    """

    def __init__(self, rates: Dict[str, float], strict: bool = True):
        self._rates = {k.upper(): float(v) for k, v in (rates or {}).items()}
        self._strict = bool(strict)

    def get(self, pair: str, t=None) -> float:
        key = pair.upper()
        if key in self._rates:
            return self._rates[key]
        if self._strict:
            raise ValueError(f"StaticRateProvider: missing rate for pair '{pair}'.")
        return None


class TimeSeriesRateProvider(RateProvider):
    """
    Historischer FX-Provider auf Candle-Basis mit as-of Lookup (<= t) und Stale-Guard.
    - Lädt pro Paar Bid/Ask, bildet Mid (standard: Close), speichert (timestamps, mids).
    - Gibt nur *abgeschlossene* Bars zurück: bar_timestamp <= t.
    - Optional: stale_limit steuert, wie "alt" die letzte Bar relativ zu t sein darf.
    """

    def __init__(
        self,
        pairs: List[str],
        timeframe: str = "M1",
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        use_price: str = "close",  # "close" oder "open"
        stale_limit_bars: Optional[int] = None,
        strict: bool = True,
    ):
        if CSVDataHandler is None:
            raise RuntimeError(
                "CSVDataHandler nicht verfügbar – TimeSeriesRateProvider benötigt Datenzugriff."
            )

        self._series: Dict[str, Tuple[List[datetime], List[float]]] = {}
        self._tf = timeframe.upper()
        self._use_close = use_price.lower() == "close"
        self._strict = bool(strict)
        self._stale_bars = stale_limit_bars
        self._bar_delta = self._tf_to_timedelta(self._tf)

        for raw in pairs or []:
            pair = raw.upper()
            dh = CSVDataHandler(
                symbol=pair, timeframe=self._tf, normalize_to_timeframe=True
            )
            candles = dh.load_candles(start_dt=start_dt, end_dt=end_dt)
            bid = candles.get("bid", [])
            ask = candles.get("ask", [])
            if not bid or not ask:
                if self._strict:
                    raise ValueError(
                        f"Keine Candle-Daten für Paar {pair} ({self._tf})."
                    )
                else:
                    continue
            # schnelles Dict für Schnittmenge Bid/Ask
            ask_map = {c.timestamp: c for c in ask}
            ts: List[datetime] = []
            mids: List[float] = []
            for b in bid:
                a = ask_map.get(b.timestamp)
                if a is None:
                    continue
                price_b = b.close if self._use_close else b.open
                price_a = a.close if self._use_close else a.open
                mid = 0.5 * (float(price_b) + float(price_a))
                # Timestamps als echte aware UTC-Datetimes abspeichern
                t = b.timestamp
                if getattr(t, "tzinfo", None) is None:
                    t = t.replace(tzinfo=timezone.utc)
                ts.append(t)
                mids.append(mid)
            if ts:
                self._series[pair] = (ts, mids)
            elif self._strict:
                raise ValueError(
                    f"Kein gemeinsamer Bid/Ask-Schnitt für {pair} ({self._tf})."
                )

    def _tf_to_timedelta(self, tf: str) -> timedelta:
        tf = tf.upper()
        if tf.startswith("M"):
            return timedelta(minutes=int(tf[1:]))
        if tf.startswith("H"):
            return timedelta(hours=int(tf[1:]))
        if tf.startswith("D"):
            return timedelta(days=int(tf[1:]))
        raise ValueError(f"Unbekannter TF: {tf}")

    def get(self, pair: str, t=None) -> float:
        key = (pair or "").upper()
        seq = self._series.get(key)
        if seq is None:
            if self._strict:
                raise ValueError(
                    f"TimeSeriesRateProvider: fehlende Serie für '{pair}'."
                )
            return None
        ts, mids = seq
        if not ts:
            if self._strict:
                raise ValueError(f"TimeSeriesRateProvider: leere Serie für '{pair}'.")
            return None
        if t is None:
            return mids[-1]
        # sicherstellen: t ist aware UTC
        if getattr(t, "tzinfo", None) is None:
            t = t.replace(tzinfo=timezone.utc)
        # as-of lookup (<= t)
        i = bisect_right(ts, t) - 1
        if i < 0:
            if self._strict:
                raise ValueError(
                    f"TimeSeriesRateProvider: keine Bar <= {t} für '{pair}'."
                )
            return None
        # Stale-Guard (wie viele Bars "darf" die letzte bekannte Bar zurückliegen?)
        if self._stale_bars is not None:
            age = t - ts[i]
            max_age = self._bar_delta * max(1, int(self._stale_bars))
            if age > max_age:
                if self._strict:
                    raise ValueError(
                        f"TimeSeriesRateProvider: letzte Bar für '{pair}' ist zu alt "
                        f"(age={age}, max={max_age}, t={t}, last={ts[i]})."
                    )
                return None
        return mids[i]


class CompositeRateProvider(RateProvider):
    """
    Kaskadiert mehrere Provider (z.B. [TimeSeries, Static]) und nimmt die erste verfügbare Rate.
    """

    def __init__(self, providers: List[RateProvider]):
        self._providers = list(providers or [])

    def get(self, pair: str, t=None) -> float:
        last_err = None
        for p in self._providers:
            try:
                r = p.get(pair, t)
                if r is not None:
                    return r
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return None
