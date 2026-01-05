"""Apache Arrow Schema-Definitionen für FFI-Grenzen.

Dieses Modul definiert zentrale Arrow-Schemas für die Serialisierung
von Datenstrukturen über FFI-Grenzen (Python ↔ Rust ↔ Julia).

Phase 2 Task: P2-06
Status: Implementiert (2026-01-05)

Verwendung:
    from shared.arrow_schemas import (
        OHLCV_SCHEMA,
        TRADE_SIGNAL_SCHEMA,
        create_ohlcv_batch,
    )

Schemas sind optimiert für:
- Zero-Copy Transfer (NumPy ↔ Arrow ↔ Rust ndarray)
- Schema-Validierung an FFI-Grenzen
- Interoperabilität mit Arrow.jl (Julia)

Referenz: docs/adr/ADR-0002-serialization-format.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

# Lazy import für pyarrow (optional dependency)
# Falls pyarrow nicht verfügbar: Fallback zu dict-basierter Serialisierung
try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

if TYPE_CHECKING:
    import pyarrow as pa
else:
    # Zur Laufzeit, wenn pyarrow nicht verfügbar ist
    if not PYARROW_AVAILABLE:
        pa = None  # type: ignore[assignment]

# =============================================================================
# Type Mapping Reference (Python → Arrow → Rust → Julia)
# =============================================================================
#
# Python Type     | Arrow Type        | Rust Type        | Julia Type
# ----------------|-------------------|------------------|------------------
# float           | float64           | f64              | Float64
# int             | int64             | i64              | Int64
# bool            | bool              | bool             | Bool
# str             | utf8              | String           | String
# datetime (UTC)  | timestamp[us,UTC] | DateTime<Utc>    | ZonedDateTime
# bytes           | binary            | Vec<u8>          | Vector{UInt8}
# None            | null              | Option<T>        | Union{T, Nothing}
# List[T]         | list<T>           | Vec<T>           | Vector{T}
# Dict[K,V]       | map<K,V>          | HashMap<K,V>     | Dict{K,V}
#
# =============================================================================


def _ensure_pyarrow() -> None:
    """Raise ImportError if pyarrow is not available."""
    if not PYARROW_AVAILABLE:
        raise ImportError(
            "pyarrow is required for Arrow serialization. "
            "Install with: pip install pyarrow>=14.0"
        )


# =============================================================================
# OHLCV Schema (Candle Data)
# =============================================================================

def get_ohlcv_schema() -> Any:
    """Schema für OHLCV (Candlestick) Daten.

    Verwendung:
        - indicator_cache.py: Cache-Daten
        - event_engine.py: Candle-Input
        - execution_simulator.py: Price-Lookup

    Arrow Schema:
        timestamp: timestamp[us, tz=UTC]  - Bar-Zeitstempel
        open:      float64                - Eröffnungspreis
        high:      float64                - Höchstpreis
        low:       float64                - Tiefstpreis
        close:     float64                - Schlusskurs
        volume:    float64                - Volumen
        valid:     bool                   - Validity Mask (false = None/NaN)

    Metadata:
        symbol:     Symbol-Name (z.B. "EURUSD")
        timeframe:  Timeframe-String (z.B. "M1", "H1")
        price_type: "bid" oder "ask"
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("open", pa.float64(), nullable=False),
            pa.field("high", pa.float64(), nullable=False),
            pa.field("low", pa.float64(), nullable=False),
            pa.field("close", pa.float64(), nullable=False),
            pa.field("volume", pa.float64(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )


OHLCV_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    OHLCV_SCHEMA = get_ohlcv_schema()


# =============================================================================
# Trade Signal Schema
# =============================================================================

def get_trade_signal_schema() -> Any:
    """Schema für Trade-Signale.

    Verwendung:
        - strategy.evaluate() Output
        - execution_simulator.process_signal() Input

    Arrow Schema:
        timestamp:  timestamp[us, tz=UTC]  - Signal-Zeitstempel
        direction:  utf8 (dict-encoded)    - "long" | "short"
        entry:      float64                - Entry-Preis
        sl:         float64                - Stop-Loss
        tp:         float64                - Take-Profit
        size:       float64                - Position Size (Lots)
        symbol:     utf8 (dict-encoded)    - Symbol-Name
        order_type: utf8 (dict-encoded)    - "market" | "limit" | "stop"
        reason:     utf8                   - Signal-Grund (nullable)
        scenario:   utf8                   - Scenario-Tag (nullable)
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("direction", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("entry", pa.float64(), nullable=False),
            pa.field("sl", pa.float64(), nullable=False),
            pa.field("tp", pa.float64(), nullable=False),
            pa.field("size", pa.float64(), nullable=False),
            pa.field("symbol", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("order_type", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("reason", pa.utf8(), nullable=True),
            pa.field("scenario", pa.utf8(), nullable=True),
        ]
    )


TRADE_SIGNAL_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    TRADE_SIGNAL_SCHEMA = get_trade_signal_schema()


# =============================================================================
# Position Schema
# =============================================================================

def get_position_schema() -> Any:
    """Schema für Portfolio-Positionen.

    Verwendung:
        - Portfolio.positions Export
        - execution_simulator.active_positions

    Arrow Schema:
        entry_time:     timestamp[us, tz=UTC]  - Entry-Zeitstempel
        exit_time:      timestamp[us, tz=UTC]  - Exit-Zeitstempel (nullable)
        direction:      utf8 (dict-encoded)    - "long" | "short"
        symbol:         utf8 (dict-encoded)    - Symbol-Name
        entry_price:    float64                - Entry-Preis
        exit_price:     float64                - Exit-Preis (nullable, NaN wenn offen)
        initial_sl:     float64                - Initialer Stop-Loss
        current_sl:     float64                - Aktueller Stop-Loss
        tp:             float64                - Take-Profit
        size:           float64                - Position Size
        result:         float64                - P/L (nullable, NaN wenn offen)
        r_multiple:     float64                - R-Multiple
        status:         utf8 (dict-encoded)    - "open" | "pending" | "closed"
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("entry_time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("exit_time", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("direction", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("symbol", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("entry_price", pa.float64(), nullable=False),
            pa.field("exit_price", pa.float64(), nullable=True),
            pa.field("initial_sl", pa.float64(), nullable=False),
            pa.field("current_sl", pa.float64(), nullable=False),
            pa.field("tp", pa.float64(), nullable=False),
            pa.field("size", pa.float64(), nullable=False),
            pa.field("result", pa.float64(), nullable=True),
            pa.field("r_multiple", pa.float64(), nullable=True),
            pa.field("status", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
        ]
    )


POSITION_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    POSITION_SCHEMA = get_position_schema()


# =============================================================================
# Indicator Series Schema
# =============================================================================

def get_indicator_schema() -> Any:
    """Schema für Indikator-Serien.

    Verwendung:
        - indicator_cache.py Output
        - strategy.evaluate() Input

    Arrow Schema:
        timestamp: timestamp[us, tz=UTC]  - Bar-Zeitstempel (aligned)
        value:     float64                - Indikator-Wert
        valid:     bool                   - Validity Mask (false = NaN)

    Metadata:
        indicator_name: Name des Indikators (z.B. "ema_20")
        timeframe:      Timeframe-String
        params:         JSON-encoded Parameter-Dict
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )


INDICATOR_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    INDICATOR_SCHEMA = get_indicator_schema()


# =============================================================================
# Rating Score Schema
# =============================================================================

def get_rating_score_schema() -> Any:
    """Schema für Rating-Scores.

    Verwendung:
        - rating/ Module Output
        - optimizer Fitness-Bewertung

    Arrow Schema:
        metric_name:  utf8 (dict-encoded)  - Name der Metrik
        raw_value:    float64              - Roher Metrik-Wert
        score:        float64              - Normalisierter Score [0, 1]
        weight:       float64              - Gewichtung
        weighted:     float64              - Gewichteter Score
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("metric_name", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
            pa.field("raw_value", pa.float64(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("weight", pa.float64(), nullable=False),
            pa.field("weighted", pa.float64(), nullable=False),
        ]
    )


RATING_SCORE_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    RATING_SCORE_SCHEMA = get_rating_score_schema()


# =============================================================================
# Portfolio Equity Curve Schema
# =============================================================================

def get_equity_curve_schema() -> Any:
    """Schema für Equity-Kurven.

    Verwendung:
        - Portfolio.equity_curve Export
        - Drawdown-Berechnung
        - Visualisierung

    Arrow Schema:
        timestamp:  timestamp[us, tz=UTC]  - Zeitstempel
        equity:     float64                - Equity-Wert
        balance:    float64                - Cash-Balance
        drawdown:   float64                - Aktueller Drawdown (negativ)
        high_water: float64                - High-Water-Mark
    """
    _ensure_pyarrow()
    return pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("equity", pa.float64(), nullable=False),
            pa.field("balance", pa.float64(), nullable=False),
            pa.field("drawdown", pa.float64(), nullable=False),
            pa.field("high_water", pa.float64(), nullable=False),
        ]
    )


EQUITY_CURVE_SCHEMA: Any = None
if PYARROW_AVAILABLE:
    EQUITY_CURVE_SCHEMA = get_equity_curve_schema()


# =============================================================================
# Factory Functions für RecordBatches
# =============================================================================

def create_ohlcv_batch(
    timestamps: np.ndarray | Sequence[datetime],
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    valid_mask: np.ndarray | None = None,
    symbol: str = "",
    timeframe: str = "",
    price_type: str = "bid",
) -> Any:
    """Erstellt einen Arrow RecordBatch aus OHLCV-Daten.

    Args:
        timestamps: Array von Zeitstempeln (datetime oder int64 epoch micros)
        opens: Open-Preise (float64)
        highs: High-Preise (float64)
        lows: Low-Preise (float64)
        closes: Close-Preise (float64)
        volumes: Volumen (float64)
        valid_mask: Validity-Maske (True = gültig), default: alle gültig
        symbol: Symbol-Name für Metadata
        timeframe: Timeframe-String für Metadata
        price_type: "bid" oder "ask" für Metadata

    Returns:
        pa.RecordBatch mit OHLCV_SCHEMA

    Example:
        >>> batch = create_ohlcv_batch(
        ...     timestamps=np.array([dt1, dt2, dt3]),
        ...     opens=np.array([1.1000, 1.1010, 1.1005]),
        ...     highs=np.array([1.1020, 1.1030, 1.1015]),
        ...     lows=np.array([1.0990, 1.1000, 1.0995]),
        ...     closes=np.array([1.1010, 1.1005, 1.1010]),
        ...     volumes=np.array([100.0, 150.0, 120.0]),
        ...     symbol="EURUSD",
        ...     timeframe="M1",
        ... )
    """
    _ensure_pyarrow()

    n = len(opens)

    # Timestamp-Konvertierung
    if isinstance(timestamps, np.ndarray) and timestamps.dtype == np.dtype("datetime64[us]"):
        ts_array = pa.array(timestamps, type=pa.timestamp("us", tz="UTC"))
    else:
        # datetime objects → timestamp
        ts_array = pa.array(
            [
                int(t.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
                if isinstance(t, datetime)
                else int(t)
                for t in timestamps
            ],
            type=pa.timestamp("us", tz="UTC"),
        )

    # Validity Mask
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)

    # Schema mit Metadata
    schema = get_ohlcv_schema().with_metadata(
        {
            b"symbol": symbol.encode("utf-8"),
            b"timeframe": timeframe.encode("utf-8"),
            b"price_type": price_type.encode("utf-8"),
        }
    )

    return pa.RecordBatch.from_arrays(
        [
            ts_array,
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array(volumes, type=pa.float64()),
            pa.array(valid_mask, type=pa.bool_()),
        ],
        schema=schema,
    )


def create_indicator_batch(
    timestamps: np.ndarray | Sequence[datetime],
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    indicator_name: str = "",
    timeframe: str = "",
    params: dict[str, Any] | None = None,
) -> Any:
    """Erstellt einen Arrow RecordBatch aus Indikator-Werten.

    Args:
        timestamps: Array von Zeitstempeln
        values: Indikator-Werte (float64)
        valid_mask: Validity-Maske (True = gültig), default: nicht-NaN
        indicator_name: Name des Indikators für Metadata
        timeframe: Timeframe-String für Metadata
        params: Indikator-Parameter für Metadata (wird JSON-encoded)

    Returns:
        pa.RecordBatch mit INDICATOR_SCHEMA
    """
    _ensure_pyarrow()
    import json

    n = len(values)

    # Timestamp-Konvertierung
    if isinstance(timestamps, np.ndarray) and timestamps.dtype == np.dtype("datetime64[us]"):
        ts_array = pa.array(timestamps, type=pa.timestamp("us", tz="UTC"))
    else:
        ts_array = pa.array(
            [
                int(t.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
                if isinstance(t, datetime)
                else int(t)
                for t in timestamps
            ],
            type=pa.timestamp("us", tz="UTC"),
        )

    # Validity Mask (default: nicht-NaN)
    if valid_mask is None:
        valid_mask = ~np.isnan(values)

    # Schema mit Metadata
    schema = get_indicator_schema().with_metadata(
        {
            b"indicator_name": indicator_name.encode("utf-8"),
            b"timeframe": timeframe.encode("utf-8"),
            b"params": json.dumps(params or {}).encode("utf-8"),
        }
    )

    return pa.RecordBatch.from_arrays(
        [
            ts_array,
            pa.array(values, type=pa.float64()),
            pa.array(valid_mask, type=pa.bool_()),
        ],
        schema=schema,
    )


# =============================================================================
# Zero-Copy Utilities
# =============================================================================

def numpy_to_arrow_buffer(array: np.ndarray) -> Any:
    """Zero-Copy Konvertierung von NumPy Array zu Arrow Buffer.

    Args:
        array: NumPy array (contiguous, C-order)

    Returns:
        pa.Buffer wrapping the NumPy array data

    Note:
        Der NumPy Array muss für die Lebensdauer des Buffers gültig bleiben!
    """
    _ensure_pyarrow()
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return pa.py_buffer(array)


def arrow_to_numpy_zero_copy(array: Any) -> np.ndarray:
    """Zero-Copy Konvertierung von Arrow Array zu NumPy.

    Args:
        array: Arrow array (primitive type, no nulls)

    Returns:
        NumPy array sharing memory with Arrow

    Raises:
        ValueError: Wenn Array Nulls enthält (Zero-Copy nicht möglich)
    """
    _ensure_pyarrow()
    if array.null_count > 0:
        raise ValueError(
            f"Cannot zero-copy convert array with {array.null_count} nulls. "
            "Use array.to_numpy(zero_copy_only=False) instead."
        )
    return array.to_numpy(zero_copy_only=True)


# =============================================================================
# Schema Registry (für dynamische Schema-Lookup)
# =============================================================================

SCHEMA_REGISTRY: dict[str, Any] = {
    "ohlcv": OHLCV_SCHEMA,
    "trade_signal": TRADE_SIGNAL_SCHEMA,
    "position": POSITION_SCHEMA,
    "indicator": INDICATOR_SCHEMA,
    "rating_score": RATING_SCORE_SCHEMA,
    "equity_curve": EQUITY_CURVE_SCHEMA,
}


def get_schema(name: str) -> Any:
    """Gibt Schema aus Registry zurück.

    Args:
        name: Schema-Name (z.B. "ohlcv", "trade_signal")

    Returns:
        pa.Schema

    Raises:
        KeyError: Wenn Schema nicht registriert
        ImportError: Wenn pyarrow nicht verfügbar
    """
    _ensure_pyarrow()
    schema = SCHEMA_REGISTRY.get(name)
    if schema is None:
        if name not in SCHEMA_REGISTRY:
            raise KeyError(f"Unknown schema: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
        # Schema existiert aber ist None (pyarrow war beim Import nicht verfügbar)
        raise ImportError("pyarrow is required for Arrow schemas")
    return schema


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Availability Flag
    "PYARROW_AVAILABLE",
    # Schema Constants
    "OHLCV_SCHEMA",
    "TRADE_SIGNAL_SCHEMA",
    "POSITION_SCHEMA",
    "INDICATOR_SCHEMA",
    "RATING_SCORE_SCHEMA",
    "EQUITY_CURVE_SCHEMA",
    # Schema Factory Functions
    "get_ohlcv_schema",
    "get_trade_signal_schema",
    "get_position_schema",
    "get_indicator_schema",
    "get_rating_score_schema",
    "get_equity_curve_schema",
    # RecordBatch Factory Functions
    "create_ohlcv_batch",
    "create_indicator_batch",
    # Zero-Copy Utilities
    "numpy_to_arrow_buffer",
    "arrow_to_numpy_zero_copy",
    # Registry
    "SCHEMA_REGISTRY",
    "get_schema",
]
