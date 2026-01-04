from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Mapping, Sequence, TypeAlias, TypedDict

from backtest_engine.data.candle import Candle
from backtest_engine.data.tick import Tick

# --- Core scalar types ---

Symbol: TypeAlias = str
PriceType: TypeAlias = Literal["bid", "ask"]
Direction: TypeAlias = Literal["long", "short"]
OrderType: TypeAlias = Literal["market", "limit", "stop"]
PositionStatus: TypeAlias = Literal["open", "pending", "closed"]
Timeframe: TypeAlias = str
Timestamp: TypeAlias = datetime

# Timestamp wird in der Engine teils als datetime/pandas.Timestamp geführt.
# Für Mapping-Keys lassen wir bewusst auch primitive Formen zu (z.B. beim Serialisieren).
TimestampKey: TypeAlias = datetime | str | int | float


# --- JSON-serialisierbare Meta-Strukturen (für Reports/FFI) ---

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]


# --- Candle representations (keeps current "dict OR object" flexibility) ---


class CandleDict(TypedDict, total=False):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    candle_type: PriceType


CandleLike: TypeAlias = Candle | CandleDict
AlignedCandle: TypeAlias = CandleLike | None


# --- Multi-Candle-Datenformen (raw vs aligned) ---

# Eine Candle-Serie (unterschiedliche Repräsentationen werden im Code unterstützt).
CandleSeries: TypeAlias = Sequence[Candle]
CandleLikeSeries: TypeAlias = Sequence[CandleLike]
AlignedCandleSeries: TypeAlias = Sequence[AlignedCandle]

SideToRawCandles: TypeAlias = Mapping[PriceType, CandleSeries]
SideToCandleLike: TypeAlias = Mapping[PriceType, CandleLikeSeries]
SideToAlignedCandles: TypeAlias = Mapping[PriceType, AlignedCandleSeries]

# Dict[TF][side] -> candles (un-aligned / ohne None)
RawMultiCandleData: TypeAlias = Mapping[Timeframe, SideToRawCandles]

# Dict[TF][side] -> candles (Candle oder dict, ohne None)
CandleLikeMultiCandleData: TypeAlias = Mapping[Timeframe, SideToCandleLike]

# Dict[TF][side] -> aligned candles (len == primary bars, None möglich)
AlignedMultiCandleData: TypeAlias = Mapping[Timeframe, SideToAlignedCandles]

# Backcompat: bisheriger Aliasname (aligned).
MultiCandleData: TypeAlias = AlignedMultiCandleData


# Dict[symbol][side][timestamp] -> Candle (multi-symbol engine)
CandleLookups: TypeAlias = Mapping[Symbol, Mapping[PriceType, Mapping[TimestampKey, Candle]]]

# Gemeinsame, synchronisierte Zeitachse für Cross-Symbol Loops
CommonTimestamps: TypeAlias = Sequence[TimestampKey]


# --- Tick & Signal Interfaces (Strategy <-> Core) ---


class TickDict(TypedDict, total=False):
    timestamp: datetime
    bid: float
    ask: float
    volume: float


TickLike: TypeAlias = Tick | TickDict

# Mapping {symbol: [Tick, ...]} (Tick-Backtests)
TickDataMap: TypeAlias = Mapping[Symbol, Sequence[TickLike]]


RawSignalDirection: TypeAlias = Literal["buy", "sell", "long", "short"]


class TradeSignalMeta(TypedDict, total=False):
    # StrategyWrapper schreibt diese Felder standardisiert in meta.
    decision_time: datetime
    timestamp_semantics: Literal["open", "close"]
    bar_open: datetime
    tf: Timeframe
    scenario: str
    tags: Sequence[str]
    reason: str


class TradeSignalDict(TypedDict, total=False):
    """Raw Signal-Shape wie von Strategien (dict) geliefert.

    Diese Struktur repräsentiert das *Input-Interface* in Richtung
    `StrategyWrapper._build_signal()`.
    """

    direction: RawSignalDirection
    entry: float
    sl: float
    tp: float

    symbol: Symbol
    type: OrderType

    reason: str
    tags: Sequence[str]
    scenario: str

    # 'meta' ist der Standard; 'metadata' wird als Backcompat akzeptiert.
    meta: Mapping[str, Any]
    metadata: Mapping[str, Any]


# --- Portfolio/Trade Export Shapes ---


class FeeLogEntry(TypedDict, total=False):
    time: datetime
    kind: Literal["entry", "exit", "other"]
    symbol: Symbol | None
    size: float | None
    fee: float


class PortfolioPositionExport(TypedDict, total=False):
    # Output von PortfolioPosition.to_dict()
    entry_time: str | None
    exit_time: str | None
    direction: Direction
    entry_price: float | None
    exit_price: float | None
    initial_stop_loss: float | None
    stop_loss: float | None
    take_profit: float | None
    size: float
    result: float | None
    reason: str | None
    order_type: OrderType
    status: PositionStatus
    r_multiple: float | None
    meta: Mapping[str, Any]


class TradeRow(TypedDict, total=False):
    # Rows in Portfolio.trades_to_dataframe() ("Trades"-Export)
    entry_time: datetime
    exit_time: datetime | None
    direction: Direction
    symbol: Symbol
    entry_price: float
    exit_price: float | None
    initial_stop_loss: float
    stop_loss: float
    take_profit: float
    size: float
    result: float | None
    entry_fee: float
    exit_fee: float
    total_fee: float
    reason: str | None
    order_type: OrderType
    status: PositionStatus
    r_multiple: float
    meta: JSONValue


PortfolioSummaryDict: TypeAlias = TypedDict(
    "PortfolioSummaryDict",
    {
        # Core summary
        "Initial Balance": float,
        "Final Balance": float,
        "Equity": float,
        "Max Drawdown": float,
        "Drawdown Initial Balance": float,
        "Total Fees": float,
        "Total Lots": float,
        "Total Trades": int,
        "Expired Orders": int,
        "Partial Closed Orders": int,
        "Orders closed at Break Even": int,
        "Avg R-Multiple": float,
        "Winrate": float,
        "Wins": int,
        "Losses": int,

        # Optional: Backtest-Robustheitsmetriken (nur wenn aktiviert)
        "Robustness 1": float,
        "Robustness 1 Num Samples": int,
        "Cost Shock Score": float,
        "Timing Jitter Score": float,
        "Trade Dropout Score": float,
        "Ulcer Index": float,
        "Ulcer Index Score": float,
        "Data Jitter Score": float,
        "Data Jitter Num Samples": int,
        "p_mean_gt": float,
        "Stability Score": float,
        "TP/SL Stress Score": float,
    },
    total=False,
)
