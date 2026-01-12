# FFI Interface Specification: DataHandler

**Modul:** `src/backtest_engine/data/data_handler.py`  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Wave:** 4  
**Status:** 📋 Draft (2026-01-11)

---

## Executive Summary

`CSVDataHandler` ist verantwortlich für das Laden von OHLCV-Marktdaten aus CSV- und Parquet-Dateien. Das Modul ist ein Kandidat für Rust-Migration aufgrund:

- I/O-intensiver Operationen (File Parsing)
- Parallelisierungspotenzial (Bid + Ask separat)
- Hoher Memory-Footprint bei großen Dateien
- Vektorisierbare Filter-Operationen (Market Hours, Weekends)

---

## Data Structures

### Input: DataLoadRequest

```python
# @ffi_boundary: Input
class DataLoadRequest(TypedDict):
    """Request für Rust Data Loader."""
    
    path: str
    """Absoluter Pfad zur Datei (Parquet oder CSV)."""
    
    candle_type: Literal["bid", "ask"]
    """Typ der Candles."""
    
    start_us: int | None
    """
    Startzeit als Epoch Microseconds (UTC).
    None = keine untere Grenze.
    """
    
    end_us: int | None
    """
    Endzeit als Epoch Microseconds (UTC).
    None = keine obere Grenze.
    """
    
    timeframe: str
    """
    Timeframe-String: "M1", "M5", "M15", "M30", "H1", "H4", "D1".
    Verwendet für:
    - Timeframe-Normalisierung (floor)
    - Daily-spezifische Filter
    """
    
    normalize_to_tf: bool
    """
    Wenn True: Floor Timestamps auf Timeframe-Grenzen.
    M5 -> Minute wird auf 0, 5, 10, ... gefloort.
    """
    
    filter_market_hours: bool
    """
    Wenn True: Filtert nach Sydney Trading Session.
    Geschlossen: Sa >=07:00 Sydney, So komplett, Mo <07:00 Sydney.
    Sollte False sein für Daily-Timeframes.
    """
    
    filter_weekend_flat: bool
    """
    Wenn True: Entfernt Wochenenden (Sa/So) und 0-Volumen Flat-Bars.
    Flat-Bar: Volume==0 AND Open==High==Low==Close.
    Primär für Daily-Timeframes.
    """
```

### Output: Arrow RecordBatch

```python
# @ffi_boundary: Output
# Arrow Schema für OHLCV Daten

import pyarrow as pa

DATA_LOAD_RESULT_SCHEMA = pa.schema([
    # Timestamp als Epoch Microseconds UTC
    # i64 statt timestamp für einfacheres FFI-Handling
    pa.field("timestamp_us", pa.int64(), nullable=False),
    
    # OHLCV Werte
    pa.field("open", pa.float64(), nullable=False),
    pa.field("high", pa.float64(), nullable=False),
    pa.field("low", pa.float64(), nullable=False),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("volume", pa.float64(), nullable=False),
    
    # Candle Type für Kontext
    pa.field("candle_type", pa.dictionary(pa.int8(), pa.utf8()), nullable=False),
])
```

**Arrow IPC Serialization:**

```
┌────────────────────────────────────────────────────────┐
│  Arrow IPC Format (Little-Endian)                       │
├────────────────────────────────────────────────────────┤
│  Schema Message (DATA_LOAD_RESULT_SCHEMA)               │
│  RecordBatch Message:                                   │
│    ├─ timestamp_us: Int64Array    [n rows]              │
│    ├─ open:         Float64Array  [n rows]              │
│    ├─ high:         Float64Array  [n rows]              │
│    ├─ low:          Float64Array  [n rows]              │
│    ├─ close:        Float64Array  [n rows]              │
│    ├─ volume:       Float64Array  [n rows]              │
│    └─ candle_type:  DictArray     [n rows]              │
└────────────────────────────────────────────────────────┘
```

---

## Rust API Signatures

### Primary: DataHandlerRust

```rust
#[pyclass]
pub struct DataHandlerRust;

#[pymethods]
impl DataHandlerRust {
    /// Erstellt neue Instanz (stateless).
    #[new]
    pub fn new() -> Self;
    
    /// Lädt Parquet-Datei und gibt Arrow IPC Bytes zurück.
    ///
    /// # Arguments
    /// * `path` - Absoluter Pfad zur Parquet-Datei
    /// * `candle_type` - "bid" oder "ask"
    /// * `start_us` - Epoch Microseconds (UTC), None = unbegrenzt
    /// * `end_us` - Epoch Microseconds (UTC), None = unbegrenzt
    /// * `timeframe` - "M1", "H1", "D1" etc.
    /// * `normalize_to_tf` - Floor Timestamps auf TF-Grenzen
    /// * `filter_market_hours` - Sydney Session Filter
    /// * `filter_weekend_flat` - Weekend + Flat-Bar Filter (Daily)
    ///
    /// # Returns
    /// Arrow IPC serialized RecordBatch als Vec<u8>
    ///
    /// # Errors
    /// * `DataError::FileNotFound` - Datei existiert nicht
    /// * `DataError::ParseError` - Parsing fehlgeschlagen
    /// * `DataError::InvalidTimeframe` - Unbekannter Timeframe
    pub fn load_parquet_arrow(
        &self,
        path: &str,
        candle_type: &str,
        start_us: Option<i64>,
        end_us: Option<i64>,
        timeframe: &str,
        normalize_to_tf: bool,
        filter_market_hours: bool,
        filter_weekend_flat: bool,
    ) -> PyResult<Vec<u8>>;
    
    /// Lädt CSV-Datei und gibt Arrow IPC Bytes zurück.
    /// Parameter identisch zu load_parquet_arrow.
    pub fn load_csv_arrow(
        &self,
        path: &str,
        candle_type: &str,
        start_us: Option<i64>,
        end_us: Option<i64>,
        timeframe: &str,
        normalize_to_tf: bool,
        filter_market_hours: bool,
        filter_weekend_flat: bool,
    ) -> PyResult<Vec<u8>>;
    
    /// Paralleles Laden von Bid + Ask (Parquet).
    ///
    /// Nutzt rayon::join für echte Parallelität.
    /// Schneller als zwei sequentielle Calls.
    ///
    /// # Returns
    /// Tuple (bid_arrow_bytes, ask_arrow_bytes)
    pub fn parallel_load_bid_ask(
        &self,
        bid_path: &str,
        ask_path: &str,
        start_us: Option<i64>,
        end_us: Option<i64>,
        timeframe: &str,
        normalize_to_tf: bool,
        filter_market_hours: bool,
    ) -> PyResult<(Vec<u8>, Vec<u8>)>;
    
    /// Paralleles Laden mehrerer Symbole.
    ///
    /// # Arguments
    /// * `requests` - Vec von (bid_path, ask_path, symbol_name)
    ///
    /// # Returns
    /// Vec von (symbol_name, bid_arrow_bytes, ask_arrow_bytes)
    pub fn load_multi_symbol_parallel(
        &self,
        requests: Vec<(String, String, String)>,
        start_us: Option<i64>,
        end_us: Option<i64>,
        timeframe: &str,
        normalize_to_tf: bool,
        filter_market_hours: bool,
    ) -> PyResult<Vec<(String, Vec<u8>, Vec<u8>)>>;
}
```

---

## Error Codes

```python
# Erweiterung von src/shared/error_codes.py

class ErrorCode(IntEnum):
    # ... bestehende Codes ...
    
    # Data Handler Errors (2100-2199)
    DATA_FILE_NOT_FOUND = 2100
    DATA_PARSE_ERROR = 2101
    DATA_INVALID_SCHEMA = 2102
    DATA_INVALID_TIMEFRAME = 2103
    DATA_EMPTY_RESULT = 2104
    DATA_TIMESTAMP_ERROR = 2105
    DATA_ARROW_SERIALIZATION_ERROR = 2106
```

```rust
// src/rust_modules/omega_rust/src/data/error.rs

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Invalid schema: expected {expected}, got {actual}")]
    InvalidSchema { expected: String, actual: String },
    
    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),
    
    #[error("Empty result after filtering")]
    EmptyResult,
    
    #[error("Timestamp error: {0}")]
    TimestampError(String),
    
    #[error("Arrow serialization error: {0}")]
    ArrowSerializationError(String),
}

impl From<DataError> for PyErr {
    fn from(err: DataError) -> PyErr {
        use pyo3::exceptions::*;
        match err {
            DataError::FileNotFound(_) => PyFileNotFoundError::new_err(err.to_string()),
            DataError::ParseError(_) => PyValueError::new_err(err.to_string()),
            DataError::InvalidSchema { .. } => PyValueError::new_err(err.to_string()),
            DataError::InvalidTimeframe(_) => PyValueError::new_err(err.to_string()),
            DataError::EmptyResult => PyValueError::new_err(err.to_string()),
            DataError::TimestampError(_) => PyValueError::new_err(err.to_string()),
            DataError::ArrowSerializationError(_) => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
```

---

## Python Integration

### Feature Flag

```python
# Environment Variable
OMEGA_USE_RUST_DATA_HANDLER = os.getenv("OMEGA_USE_RUST_DATA_HANDLER", "auto")

# Values:
# "0"    - Immer Python verwenden
# "1"    - Immer Rust verwenden (Fehler wenn nicht verfügbar)
# "auto" - Rust wenn verfügbar, sonst Python (Default)
```

### Helper Functions

```python
# src/shared/arrow_schemas.py (Erweiterung)

def arrow_ipc_to_candle_list(
    ipc_bytes: bytes,
    candle_type: str,
) -> List[Candle]:
    """
    Konvertiert Arrow IPC Bytes zu Candle-Liste.
    
    @ffi_boundary: Input (von Rust)
    
    Args:
        ipc_bytes: Arrow IPC serialized RecordBatch
        candle_type: "bid" oder "ask"
    
    Returns:
        Liste von Candle-Objekten
    """
    import pyarrow as pa
    from backtest_engine.data.candle import Candle
    
    reader = pa.ipc.open_stream(ipc_bytes)
    batch = reader.read_all()
    
    timestamps = batch.column("timestamp_us").to_pylist()
    opens = batch.column("open").to_pylist()
    highs = batch.column("high").to_pylist()
    lows = batch.column("low").to_pylist()
    closes = batch.column("close").to_pylist()
    volumes = batch.column("volume").to_pylist()
    
    return [
        Candle(
            timestamp=datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc),
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            candle_type=candle_type,
        )
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes)
    ]
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Backtest Runner                                                            │
│  load_data(config) → CSVDataHandler.load_candles()                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CSVDataHandler.load_candles()                                              │
│  ├─ Check: _use_rust_data_handler()                                         │
│  │   ├─ OMEGA_USE_RUST_DATA_HANDLER == "0" → Python                         │
│  │   ├─ OMEGA_USE_RUST_DATA_HANDLER == "1" → Rust                           │
│  │   └─ "auto" → Rust if available, else Python                             │
│  │                                                                          │
│  ├─ [Python Path] ─────────────────────────────────────────────────┐        │
│  │   ├─ _load_parquet() or _load_file()                            │        │
│  │   ├─ pd.read_parquet() / pd.read_csv()                          │        │
│  │   ├─ UTC Conversion                                             │        │
│  │   ├─ Market Hours Filter (_apply_market_hours_fast)             │        │
│  │   ├─ Weekend/Flat Filter                                        │        │
│  │   ├─ TF Normalization (_floor_to_tf_vec)                        │        │
│  │   └─ Build Candle List (List Comprehension)                     │        │
│  │                                                                 │        │
│  └─ [Rust Path] ───────────────────────────────────────────────────┤        │
│      ├─ DataHandlerRust.parallel_load_bid_ask()                    │        │
│      │   ├─ rayon::join(load_bid, load_ask)                        │        │
│      │   ├─ polars::read_parquet()                                 │        │
│      │   ├─ filter_market_hours() (chrono-tz)                      │        │
│      │   ├─ filter_weekend_flat()                                  │        │
│      │   ├─ floor_to_timeframe()                                   │        │
│      │   └─ Arrow IPC Serialize                                    │        │
│      │                                                             │        │
│      └─ arrow_ipc_to_candle_list() ────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Return: {"bid": List[Candle], "ask": List[Candle]}                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Python Baseline (1M rows)

| Operation | Time | Memory |
|-----------|------|--------|
| pd.read_parquet | 150ms | 180MB |
| UTC Conversion | 20ms | - |
| Market Hours Filter | 80ms | 50MB |
| Weekend/Flat Filter | 30ms | - |
| TF Normalization | 25ms | - |
| Candle List Build | 95ms | 120MB |
| **Total** | **400ms** | **350MB peak** |

### Rust Target (1M rows)

| Operation | Time | Memory |
|-----------|------|--------|
| polars::read_parquet | 30ms | 40MB |
| All Filters (vectorized) | 15ms | - |
| Arrow IPC Serialize | 10ms | 20MB |
| Python Candle Build | 25ms | 60MB |
| **Total** | **80ms** | **120MB peak** |

### Parallel Speedup

| Scenario | Sequential | Parallel | Speedup |
|----------|------------|----------|---------|
| Bid + Ask (2 files) | 160ms | 85ms | 1.9x |
| 5 Symbols (10 files) | 800ms | 180ms | 4.4x |
| 10 Symbols (20 files) | 1600ms | 250ms | 6.4x |

---

## Edge Cases

### 1. Leere Dateien

```python
# Input: Parquet mit 0 Zeilen
# Expected: Leere Candle-Liste, kein Error
result = handler.load_parquet_arrow(empty_file, ...)
assert len(result) == 0  # Leerer Arrow Batch
```

### 2. Fehlende Spalten

```python
# Input: Parquet ohne "Volume" Spalte
# Expected: DataError::InvalidSchema
with pytest.raises(ValueError, match="Invalid schema"):
    handler.load_parquet_arrow(missing_column_file, ...)
```

### 3. DST-Übergänge

```python
# Sydney DST: 2. Sonntag im Oktober → UTC+11
# Input: Timestamps um DST-Wechsel
# Expected: Korrekte Market Hours trotz DST
# Test mit Daten vom 2026-10-04 (DST-Ende in Sydney)
```

### 4. Timezone-Naive Timestamps

```python
# Input: Parquet mit timezone-naive Timestamps
# Expected: Behandlung als UTC
# Kein Fehler, aber Warning loggen
```

---

## Migration Strategy

### Phase 1: Parallel Implementation

1. Rust-Module implementieren ohne Python-Code zu ändern
2. Separate Test-Suite für Rust-Funktionen
3. Parity-Tests: Rust vs Python

### Phase 2: Feature-Flag Integration

1. Feature-Flag `OMEGA_USE_RUST_DATA_HANDLER` hinzufügen
2. CSVDataHandler entscheidet basierend auf Flag
3. Beide Pfade bleiben funktional

### Phase 3: Performance Validation

1. Benchmark-Suite mit pytest-benchmark
2. CI-Integration mit Performance-Thresholds
3. Memory-Profiling

### Phase 4: Gradual Rollout

1. Default: `auto` (Rust wenn verfügbar)
2. Monitoring: Logs für Pfad-Nutzung
3. Nach Stabilisierung: Default auf `1` setzen

### Rollback

```bash
# Sofortiger Rollback via ENV
export OMEGA_USE_RUST_DATA_HANDLER=0

# Oder in .env
OMEGA_USE_RUST_DATA_HANDLER=0
```

---

## References

- [Wave 4 Migration Plan](../WAVE_4_DATA_HANDLER_RUST_MIGRATION_PLAN.md)
- [ADR-0002: Serialization Format](../adr/ADR-0002-serialization-format.md)
- [ADR-0003: Error Handling](../adr/ADR-0003-error-handling.md)
- [Arrow Schemas](../../src/shared/arrow_schemas.py)
- [Error Codes](../../src/shared/error_codes.py)
- [Python Data Handler](../../src/backtest_engine/data/data_handler.py)
