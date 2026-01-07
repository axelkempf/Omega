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
        SCHEMA_REGISTRY,
        get_schema_fingerprint,
    )

Schemas sind optimiert für:
- Zero-Copy Transfer (NumPy ↔ Arrow ↔ Rust ndarray)
- Schema-Validierung an FFI-Grenzen
- Interoperabilität mit Arrow.jl (Julia)
- Schema-Versionierung für Drift-Detection (CI-Guardrail)

Referenz: docs/adr/ADR-0002-serialization-format.md
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Final, Sequence

import numpy as np

# =============================================================================
# Schema Registry Version
# =============================================================================
# Increment MINOR on backward-compatible changes (new nullable fields)
# Increment MAJOR on breaking changes (removed fields, type changes)
# v2.0.0: Changed dictionary index type from int8 to int32 for scalability
#         (int8 limited to 127 unique values, int32 supports ~2 billion)
SCHEMA_REGISTRY_VERSION: Final[str] = "2.0.0"

# pyarrow ist Core-Dependency (pyproject.toml: pyarrow>=14.0)
# Import sollte immer erfolgreich sein. PYARROW_AVAILABLE bleibt für
# Abwärtskompatibilität mit bestehendem Code erhalten.
try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    # Sollte nicht auftreten da pyarrow Core-Dependency ist.
    # Fallback nur für Edge-Cases (z.B. kaputte Installation).
    PYARROW_AVAILABLE = False
    pa = None

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


def _datetime_to_utc_micros(t: datetime | int) -> int:
    """Convert datetime to UTC microseconds since epoch.

    Policy:
        - Naive datetime: Interpreted as UTC (no conversion)
        - Aware datetime: Converted to UTC via astimezone()
        - Integer: Assumed to be microseconds, passed through

    Args:
        t: datetime object or integer microseconds

    Returns:
        Microseconds since Unix epoch (UTC)

    Raises:
        ValueError: If datetime conversion fails
    """
    if isinstance(t, int):
        return t

    if not isinstance(t, datetime):
        raise ValueError(f"Expected datetime or int, got {type(t).__name__}")

    if t.tzinfo is None:
        # Naive datetime: interpret as UTC
        utc_dt = t.replace(tzinfo=timezone.utc)
    else:
        # Aware datetime: convert to UTC
        utc_dt = t.astimezone(timezone.utc)

    return int(utc_dt.timestamp() * 1_000_000)


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
            pa.field(
                "direction",
                pa.dictionary(pa.int32(), pa.utf8()),
                nullable=False,
            ),
            pa.field("entry", pa.float64(), nullable=False),
            pa.field("sl", pa.float64(), nullable=False),
            pa.field("tp", pa.float64(), nullable=False),
            pa.field("size", pa.float64(), nullable=False),
            pa.field("symbol", pa.dictionary(pa.int32(), pa.utf8()), nullable=False),
            pa.field(
                "order_type",
                pa.dictionary(pa.int32(), pa.utf8()),
                nullable=False,
            ),
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
        exit_price:     float64                - Exit-Preis (nullable, NaN wenn
                                               offen)
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
            pa.field("direction", pa.dictionary(pa.int32(), pa.utf8()), nullable=False),
            pa.field("symbol", pa.dictionary(pa.int32(), pa.utf8()), nullable=False),
            pa.field("entry_price", pa.float64(), nullable=False),
            pa.field("exit_price", pa.float64(), nullable=True),
            pa.field("initial_sl", pa.float64(), nullable=False),
            pa.field("current_sl", pa.float64(), nullable=False),
            pa.field("tp", pa.float64(), nullable=False),
            pa.field("size", pa.float64(), nullable=False),
            pa.field("result", pa.float64(), nullable=True),
            pa.field("r_multiple", pa.float64(), nullable=True),
            pa.field("status", pa.dictionary(pa.int32(), pa.utf8()), nullable=False),
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
            pa.field(
                "metric_name", pa.dictionary(pa.int32(), pa.utf8()), nullable=False
            ),
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
    if isinstance(timestamps, np.ndarray) and timestamps.dtype == np.dtype(
        "datetime64[us]"
    ):
        ts_array = pa.array(timestamps, type=pa.timestamp("us", tz="UTC"))
    else:
        # datetime objects → timestamp (microseconds since epoch UTC)
        ts_array = pa.array(
            [_datetime_to_utc_micros(t) for t in timestamps],
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

    # Timestamp-Konvertierung
    if isinstance(timestamps, np.ndarray) and timestamps.dtype == np.dtype(
        "datetime64[us]"
    ):
        ts_array = pa.array(timestamps, type=pa.timestamp("us", tz="UTC"))
    else:
        ts_array = pa.array(
            [_datetime_to_utc_micros(t) for t in timestamps],
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


def arrow_to_numpy_zero_copy(array: Any) -> np.ndarray[Any, np.dtype[Any]]:
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
    result: np.ndarray[Any, np.dtype[Any]] = array.to_numpy(zero_copy_only=True)
    return result


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
            available = list(SCHEMA_REGISTRY.keys())
            raise KeyError(f"Unknown schema: {name}. Available: {available}")
        # Schema existiert aber ist None
        # (pyarrow war beim Import nicht verfügbar)
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
    "SCHEMA_REGISTRY_VERSION",
    "get_schema",
    # Schema Versioning
    "get_schema_fingerprint",
    "get_all_schema_fingerprints",
    "validate_schema_registry",
]


# =============================================================================
# Schema Fingerprinting (CI Guardrail for Drift Detection)
# =============================================================================


def get_schema_fingerprint(schema: Any) -> str:
    """Generate a deterministic fingerprint for an Arrow schema.

    Used by CI to detect schema drift between Python/Rust/Julia.

    Args:
        schema: pyarrow.Schema object

    Returns:
        SHA-256 hash (hex) of normalized schema representation

    Example:
        >>> fingerprint = get_schema_fingerprint(OHLCV_SCHEMA)
        >>> print(fingerprint[:16])  # First 16 chars
        'a3b9c7d8e1f2a4b5'
    """
    _ensure_pyarrow()

    # Normalize schema to deterministic string representation
    # Include field names, types, nullability - exclude metadata
    schema_repr = []
    for field in schema:
        field_info = {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        schema_repr.append(field_info)

    # Sort for determinism
    normalized = json.dumps(schema_repr, sort_keys=True, ensure_ascii=True)

    # Generate hash
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_all_schema_fingerprints() -> dict[str, str]:
    """Get fingerprints for all registered schemas.

    Returns:
        Dict mapping schema name to fingerprint

    Example:
        >>> fps = get_all_schema_fingerprints()
        >>> print(fps["ohlcv"][:16])
    """
    _ensure_pyarrow()

    result = {}
    for name, schema in SCHEMA_REGISTRY.items():
        if schema is not None:
            result[name] = get_schema_fingerprint(schema)
    return result


def validate_schema_registry(expected_fingerprints: dict[str, str]) -> list[str]:
    """Validate current schemas against expected fingerprints.

    Used by CI to detect schema drift.

    Args:
        expected_fingerprints: Dict of {schema_name: expected_fingerprint}

    Returns:
        List of error messages (empty if all valid)

    Example:
        >>> expected = {"ohlcv": "a3b9c7d8..."}
        >>> errors = validate_schema_registry(expected)
        >>> if errors:
        ...     raise AssertionError("\\n".join(errors))
    """
    _ensure_pyarrow()

    errors = []
    current = get_all_schema_fingerprints()

    # Check for missing schemas
    for name in expected_fingerprints:
        if name not in current:
            errors.append(f"Missing schema: {name}")

    # Check for fingerprint mismatches
    for name, expected_fp in expected_fingerprints.items():
        if name in current and current[name] != expected_fp:
            errors.append(
                f"Schema drift detected for '{name}': "
                f"expected {expected_fp[:16]}..., got {current[name][:16]}..."
            )

    # Check for new schemas not in expected
    for name in current:
        if name not in expected_fingerprints:
            errors.append(
                f"New schema '{name}' not in expected fingerprints. "
                f"Fingerprint: {current[name]}"
            )

    return errors
