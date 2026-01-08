# FFI-Spezifikation: MultiSymbolSlice

**Modul:** `src/backtest_engine/core/multi_symbol_slice.py`  
**Task-ID:** P6-01  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Status:** ✅ Spezifiziert (2026-01-06, aktualisiert 2026-01-06)

---

## 1. Übersicht

MultiSymbolSlice verwaltet synchronisierte Candle-Daten für mehrere Symbole über verschiedene Zeitrahmen. Es ist die zentrale Datenstruktur für Multi-Symbol-Backtests und Cross-Symbol-Strategien.

### 1.1 Verantwortlichkeiten

- Synchronisierte Timestamp-Iteration über mehrere Symbole
- Effiziente Slice-Erstellung für einen Zeitpunkt
- Memory-effizientes Caching von Multi-Symbol-Daten
- Support für verschiedene Timeframes pro Symbol

---

## 2. Python Interface

### 2.1 Klassen-Signatur

```python
@dataclass
class MultiSymbolSlice:
    """Synchronisierter Snapshot aller Symbol-Daten zu einem Zeitpunkt."""
    
    timestamp: datetime
    symbols: dict[str, SymbolSnapshot]
    
    def get_price(self, symbol: str, price_type: Literal["bid", "ask", "mid"]) -> float | None:
        """Hole aktuellen Preis für Symbol."""
        ...
    
    def get_candle(self, symbol: str, timeframe: str) -> CandleData | None:
        """Hole Candle für Symbol und Timeframe."""
        ...
    
    def get_spread(self, symbol: str) -> float | None:
        """Berechne Bid-Ask-Spread für Symbol."""
        ...

@dataclass
class SymbolSnapshot:
    """Snapshot eines einzelnen Symbols."""
    
    symbol: str
    timestamp: datetime
    bid: CandleData | None
    ask: CandleData | None
    indicators: dict[str, float] | None
```

### 2.2 Iterator-Interface

```python
class MultiSymbolDataIterator:
    """Iteriert über synchronisierte Multi-Symbol-Daten."""
    
    def __init__(
        self,
        symbol_data: dict[str, SymbolData],
        start_time: datetime,
        end_time: datetime,
        primary_timeframe: str = "M1",
    ) -> None: ...
    
    def __iter__(self) -> Iterator[MultiSymbolSlice]: ...
    
    def __next__(self) -> MultiSymbolSlice: ...
    
    def peek(self, n: int = 1) -> list[MultiSymbolSlice]:
        """Vorschau auf nächste n Slices ohne Iteration."""
        ...
    
    def skip(self, n: int) -> None:
        """Überspringe n Slices."""
        ...
```

---

## 3. Arrow Schema

### 3.1 MultiSymbolSlice Schema

```python
MULTI_SYMBOL_SLICE_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("symbol", pa.utf8(), nullable=False),
    pa.field("bid_open", pa.float64(), nullable=True),
    pa.field("bid_high", pa.float64(), nullable=True),
    pa.field("bid_low", pa.float64(), nullable=True),
    pa.field("bid_close", pa.float64(), nullable=True),
    pa.field("bid_volume", pa.float64(), nullable=True),
    pa.field("ask_open", pa.float64(), nullable=True),
    pa.field("ask_high", pa.float64(), nullable=True),
    pa.field("ask_low", pa.float64(), nullable=True),
    pa.field("ask_close", pa.float64(), nullable=True),
    pa.field("ask_volume", pa.float64(), nullable=True),
    pa.field("valid_bid", pa.bool_(), nullable=False),
    pa.field("valid_ask", pa.bool_(), nullable=False),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "multi_symbol_slice",
})
```

### 3.2 Batch-Format für FFI

Die Daten werden als "Symbol-Major" Layout übertragen:
- Äußere Dimension: Symbole
- Innere Dimension: Zeitpunkte

```
┌─────────────────────────────────────────┐
│ Batch Layout (Arrow IPC)                │
├─────────────────────────────────────────┤
│ [Symbol 1: EURUSD]                      │
│   timestamp[], bid_open[], bid_close[]  │
│ [Symbol 2: GBPUSD]                      │
│   timestamp[], bid_open[], bid_close[]  │
│ [Symbol 3: USDJPY]                      │
│   timestamp[], bid_open[], bid_close[]  │
└─────────────────────────────────────────┘
```

---

## 4. Rust Interface

### 4.1 Core Types

```rust
use arrow::datatypes::*;
use chrono::{DateTime, Utc};

/// Synchronized multi-symbol data slice
#[derive(Clone, Debug)]
pub struct MultiSymbolSlice {
    pub timestamp: DateTime<Utc>,
    pub symbols: HashMap<String, SymbolSnapshot>,
}

#[derive(Clone, Debug)]
pub struct SymbolSnapshot {
    pub symbol: String,
    pub bid: Option<CandleData>,
    pub ask: Option<CandleData>,
    pub indicators: Option<HashMap<String, f64>>,
}

#[derive(Clone, Debug)]
pub struct CandleData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

### 4.2 PyO3 Interface

```rust
#[pyclass]
pub struct MultiSymbolSliceRust {
    inner: MultiSymbolSlice,
}

#[pymethods]
impl MultiSymbolSliceRust {
    /// Get price for symbol
    pub fn get_price(&self, symbol: &str, price_type: &str) -> Option<f64>;
    
    /// Get candle for symbol
    pub fn get_candle(&self, symbol: &str) -> Option<CandleDataPy>;
    
    /// Get spread for symbol
    pub fn get_spread(&self, symbol: &str) -> Option<f64>;
    
    /// Export as Arrow IPC bytes
    pub fn to_arrow(&self) -> PyResult<Vec<u8>>;
    
    /// Create from Arrow IPC bytes
    #[staticmethod]
    pub fn from_arrow(data: &[u8]) -> PyResult<Self>;
}

#[pyclass]
pub struct MultiSymbolDataIteratorRust {
    inner: InnerIterator,
}

#[pymethods]
impl MultiSymbolDataIteratorRust {
    #[new]
    pub fn new(
        symbol_data_ipc: &[u8],  // Arrow IPC with all symbol data
        start_time: i64,         // Unix timestamp microseconds
        end_time: i64,
        primary_timeframe: &str,
    ) -> PyResult<Self>;
    
    /// Python __next__ implementation
    pub fn __next__(&mut self) -> PyResult<Option<MultiSymbolSliceRust>>;
    
    /// Peek ahead n slices
    pub fn peek(&self, n: usize) -> PyResult<Vec<MultiSymbolSliceRust>>;
    
    /// Skip n slices
    pub fn skip(&mut self, n: usize) -> PyResult<()>;
    
    /// Get remaining count
    pub fn remaining(&self) -> usize;
}
```

---

## 5. Nullability-Konvention

| Feld | Nullable | Semantik |
|------|----------|----------|
| `timestamp` | Nein | Immer vorhanden |
| `symbol` | Nein | Immer vorhanden |
| `bid_*` | Ja | None = kein Bid-Daten für Zeitpunkt |
| `ask_*` | Ja | None = kein Ask-Daten für Zeitpunkt |
| `valid_bid` | Nein | false = bid_* enthält NaN/Invalid |
| `valid_ask` | Nein | false = ask_* enthält NaN/Invalid |
| `indicators` | Ja | None = keine Indikatoren berechnet |

### 5.1 Missing-Data Handling

```python
# Python: None für fehlende Daten
slice.get_price("EURUSD", "bid")  # -> None wenn nicht vorhanden

# Rust: Option<f64>
if let Some(price) = slice.get_price("EURUSD", "bid") {
    // price available
}
```

---

## 6. Error Handling

### 6.1 Error-Codes

| Code | Name | Beschreibung |
|------|------|--------------|
| `3001` | `SYMBOL_NOT_FOUND` | Symbol nicht in Slice vorhanden |
| `3002` | `TIMEFRAME_NOT_FOUND` | Timeframe nicht für Symbol geladen |
| `3003` | `NO_DATA_FOR_TIMESTAMP` | Keine Daten für angefragten Zeitpunkt |
| `3004` | `INVALID_PRICE_TYPE` | Ungültiger price_type Parameter |
| `3005` | `ITERATOR_EXHAUSTED` | Iterator hat Ende erreicht |
| `3006` | `DESERIALIZATION_ERROR` | Arrow IPC Parsing fehlgeschlagen |

### 6.2 Exception Mapping

```python
# Python exceptions (in shared/exceptions.py)
class SymbolNotFoundError(OmegaError):
    error_code = ErrorCode.SYMBOL_NOT_FOUND

class NoDataForTimestampError(OmegaError):
    error_code = ErrorCode.NO_DATA_FOR_TIMESTAMP
```

---

## 7. Performance-Charakteristika

### 7.1 Baselines (Python)

| Operation | 10 Symbole | 50 Symbole | 100 Symbole |
|-----------|------------|------------|-------------|
| Slice Creation | 0.8ms | 3.5ms | 7.2ms |
| Iterator Step | 0.2ms | 0.9ms | 1.8ms |
| get_price() | 0.02ms | 0.02ms | 0.02ms |
| Full Iteration (1M steps) | 200s | 900s | 1800s |

### 7.2 Rust Targets

| Operation | Target | Speedup |
|-----------|--------|---------|
| Slice Creation | <0.1ms | 8x |
| Iterator Step | <0.05ms | 4x |
| get_price() | <0.001ms | 20x |
| Full Iteration (1M steps) | <50s | 4x |

---

## 8. Migration Strategy

### 8.1 Empfohlener Ansatz: Hybrid Iterator

1. **Phase 1:** Rust-basierter Iterator über vorgeladene Arrow-Daten
2. **Phase 2:** Python MultiSymbolSlice wrapper um Rust-Slice
3. **Phase 3:** Full Rust wenn EventEngine migriert ist

### 8.2 Data Flow

```
Python                          Rust
──────                          ────
SymbolData[] ──Arrow IPC──▶ MultiSymbolDataIteratorRust
                                    │
                                    ▼
                            Iterator<MultiSymbolSliceRust>
                                    │
                            ◀──Arrow IPC──┘
MultiSymbolSlice (Python wrapper)
```

---

## 9. Referenzen

- Arrow Schema Registry: `src/shared/arrow_schemas.py`
- Error Codes: `src/shared/error_codes.py`
- Nullability Convention: `docs/ffi/nullability-convention.md`
- Data Flow Diagrams: `docs/ffi/data-flow-diagrams.md`
