# FFI-Spezifikation: SymbolDataSlicer

**Modul:** `src/backtest_engine/core/symbol_data_slicer.py`  
**Task-ID:** P6-02  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Status:** ✅ Spezifiziert (2026-01-06, aktualisiert 2026-01-06)

---

## 1. Übersicht

SymbolDataSlicer extrahiert Zeitfenster-Slices aus vorgeladenen Symbol-Daten. Er ist der Hauptzugang zu historischen Candle-Daten während des Backtests.

### 1.1 Verantwortlichkeiten

- Effiziente Zeitfenster-Extraktion aus DataFrames
- Alignment verschiedener Timeframes
- Caching von häufig genutzten Slices
- Memory-effizientes Handling großer Datensätze

---

## 2. Python Interface

```python
class SymbolDataSlicer:
    """Sliced Zugriff auf Symbol-Daten."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        price_type: Literal["bid", "ask"],
    ) -> None: ...
    
    def get_slice(
        self,
        start: datetime,
        end: datetime,
        include_current: bool = True,
    ) -> pd.DataFrame:
        """Hole Daten-Slice für Zeitfenster."""
        ...
    
    def get_candle_at(self, timestamp: datetime) -> CandleData | None:
        """Hole einzelne Candle für Zeitpunkt."""
        ...
    
    def get_lookback(
        self,
        current_time: datetime,
        periods: int,
    ) -> pd.DataFrame:
        """Hole die letzten n Perioden ab current_time."""
        ...
```

---

## 3. Arrow Schema

```python
SYMBOL_SLICE_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("open", pa.float64(), nullable=False),
    pa.field("high", pa.float64(), nullable=False),
    pa.field("low", pa.float64(), nullable=False),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("volume", pa.float64(), nullable=True),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "symbol_slice",
})
```

---

## 4. Rust Interface

```rust
#[pyclass]
pub struct SymbolDataSlicerRust {
    data: Arc<arrow::array::RecordBatch>,
    symbol: String,
    timeframe: String,
    price_type: PriceType,
    index: TimestampIndex,
}

#[pymethods]
impl SymbolDataSlicerRust {
    #[new]
    pub fn new(data_ipc: &[u8], symbol: &str, timeframe: &str, price_type: &str) -> PyResult<Self>;
    
    /// Get slice for time range, returns Arrow IPC
    pub fn get_slice(&self, start: i64, end: i64, include_current: bool) -> PyResult<Vec<u8>>;
    
    /// Get single candle
    pub fn get_candle_at(&self, timestamp: i64) -> PyResult<Option<CandleDataPy>>;
    
    /// Get lookback periods
    pub fn get_lookback(&self, current_time: i64, periods: usize) -> PyResult<Vec<u8>>;
}
```

---

## 5. Error Handling

| Code | Name | Beschreibung |
| ---- | ---- | ------------ |
| `2001` | `SLICE_OUT_OF_BOUNDS` | Angeforderter Zeitraum außerhalb Daten |
| `2002` | `NO_DATA_AT_TIMESTAMP` | Keine Candle für exakten Zeitpunkt |
| `2003` | `INSUFFICIENT_LOOKBACK` | Nicht genug Historie für lookback |

---

## 6. Performance-Targets

| Operation | Python Baseline | Rust Target | Speedup |
| --------- | --------------- | ----------- | ------- |
| get_slice (1K rows) | 2.5ms | 0.3ms | 8x |
| get_candle_at | 0.15ms | 0.01ms | 15x |
| get_lookback (200) | 0.8ms | 0.1ms | 8x |

---

## 7. Referenzen

- Arrow-Schemas: `src/shared/arrow_schemas.py`
- Data-Flow: `docs/ffi/data-flow-diagrams.md`
