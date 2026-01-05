from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from pandas import DataFrame, Series
else:  # pragma: no cover
    DataFrame = Any  # type: ignore[assignment]
    Series = Any  # type: ignore[assignment]


# NOTE: This module intentionally avoids importing heavy runtime dependencies.
# Type annotations rely on postponed evaluation (Python 3.12+) and TYPE_CHECKING.


@runtime_checkable
class OHLCProtocol(Protocol):
    timestamp: Any
    open: float
    high: float
    low: float
    close: float


@runtime_checkable
class OHLCVProtocol(OHLCProtocol, Protocol):
    volume: float


CandleLike = OHLCProtocol | Mapping[str, Any]


@runtime_checkable
class TradeSignalProtocol(Protocol):
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    timestamp: datetime
    type: str

    reason: str | None
    tags: Sequence[str]
    scenario: str | None
    meta: Mapping[str, Any]


@runtime_checkable
class IndicatorCacheProtocol(Protocol):
    def get_df(self, tf: str, price_type: str = "bid") -> DataFrame: ...

    def get_closes(self, tf: str, price_type: str = "bid") -> Series: ...

    def ema(self, tf: str, price_type: str, period: int) -> Series: ...

    def ema_stepwise(self, tf: str, price_type: str, period: int) -> Series: ...

    def atr(self, tf: str, price_type: str, period: int = 14) -> Series: ...

    def bollinger(
        self,
        tf: str,
        price_type: str,
        period: int = 20,
        std_factor: float = 2.0,
    ) -> tuple[Series, Series, Series]: ...

    def bollinger_stepwise(
        self,
        tf: str,
        price_type: str,
        period: int = 20,
        std_factor: float = 2.0,
    ) -> tuple[Series, Series, Series]: ...

    def zscore(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        mean_source: str = "rolling",
        ema_period: int | None = None,
    ) -> Series: ...

    def kalman_zscore(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        R: float = 0.01,
        Q: float = 1.0,
    ) -> Series: ...

    def kalman_zscore_stepwise(
        self,
        tf: str,
        price_type: str,
        window: int = 100,
        R: float = 0.01,
        Q: float = 1.0,
    ) -> Series: ...

    def kalman_garch_zscore(
        self,
        tf: str,
        price_type: str,
        R: float = 0.01,
        Q: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: float | None = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> Series: ...

    def vol_cluster_series(
        self,
        tf: str,
        price_type: str,
        idx: int,
        feature: str,
        atr_length: int,
        garch_lookback: int,
        garch_alpha: float = 0.05,
        garch_beta: float = 0.90,
        garch_omega: float | None = None,
        garch_use_log_returns: bool = True,
        garch_scale: float = 100.0,
        garch_min_periods: int = 50,
        garch_sigma_floor: float = 1e-6,
    ) -> Series | None: ...

    def kalman_garch_zscore_local(
        self,
        tf: str,
        price_type: str,
        idx: int,
        lookback: int = 400,
        R: float = 0.01,
        Q: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.90,
        omega: float | None = None,
        use_log_returns: bool = True,
        scale: float = 100.0,
        min_periods: int = 50,
        sigma_floor: float = 1e-6,
    ) -> float | None: ...


@runtime_checkable
class SymbolDataSliceProtocol(Protocol):
    index: int
    indicators: IndicatorCacheProtocol | None

    def set_index(self, index: int) -> None: ...

    def latest(self, timeframe: str, price_type: str = "bid") -> CandleLike | None: ...

    def history(
        self, timeframe: str, price_type: str = "bid", length: int = 20
    ) -> list[CandleLike]: ...


@runtime_checkable
class MultiSymbolSliceViewProtocol(Protocol):
    def latest(
        self, tf: str | None = None, price_type: str = "bid"
    ) -> CandleLike | None: ...


@runtime_checkable
class MultiSymbolSliceProtocol(Protocol):
    def get(self, symbol: str, price_type: str = "bid") -> CandleLike | None: ...

    def __getitem__(self, symbol: str) -> MultiSymbolSliceViewProtocol: ...

    def set_timestamp(self, timestamp: Any) -> None: ...

    def keys(self) -> Sequence[str]: ...


@runtime_checkable
class PortfolioProtocol(Protocol):
    def get_open_positions(self, symbol: str | None = None) -> Sequence[Any]: ...

    def update(self, current_time: datetime) -> None: ...


@runtime_checkable
class ExecutionSimulatorProtocol(Protocol):
    active_positions: Sequence[Any]

    def process_signal(self, signal: TradeSignalProtocol) -> None: ...

    def evaluate_exits(
        self, bid_candle: CandleLike, ask_candle: CandleLike | None = None
    ) -> None: ...


@runtime_checkable
class StrategyEvaluatorProtocol(Protocol):
    def evaluate(
        self, index: int, slice_map: Mapping[str, SymbolDataSliceProtocol]
    ) -> TradeSignalProtocol | Sequence[TradeSignalProtocol] | None: ...


@runtime_checkable
class CrossSymbolStrategyEvaluatorProtocol(Protocol):
    def evaluate(
        self, index: int, multi_slice: MultiSymbolSliceProtocol
    ) -> TradeSignalProtocol | Sequence[TradeSignalProtocol] | None: ...


@runtime_checkable
class BrokerProtocol(Protocol):
    def get_margin_info(self, symbol: str) -> dict[str, Any]: ...


@runtime_checkable
class DataProviderProtocol(Protocol):
    def get_candles(
        self, symbol: str, timeframe: str, count: int
    ) -> list[dict[str, Any]]: ...
