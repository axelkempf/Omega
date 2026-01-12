# Wave 4: Data Handler Rust Migration Plan

**Version:** 1.0.0  
**Erstellt:** 2026-01-11  
**Status:** 📋 Draft  
**Verantwortlich:** Axel Kempf

---

## Executive Summary

Dieser Plan beschreibt die inkrementelle Migration des `CSVDataHandler`-Moduls (`src/backtest_engine/data/data_handler.py`) nach Rust. Die Migration folgt den etablierten Patterns aus Wave 0-3 und nutzt die bestehende FFI-Infrastruktur (Arrow IPC, PyO3, Error Codes).

### Ziele

| Ziel | Metrik | Target |
|------|--------|--------|
| **Parquet-Load Performance** | Ladezeit 1M Zeilen | ≤50ms (vs. 200ms Python) |
| **CSV-Load Performance** | Ladezeit 1M Zeilen | ≤500ms (vs. 3-5s Python) |
| **Peak-RAM** | Memory beim Laden | -50% vs. Python |
| **Paralleles Laden** | Multi-Symbol Load | Bid+Ask parallel (2x Speedup) |

### Nicht-Ziele

- Migration des LRU-Caching (bleibt Python)
- Migration der `TickDataHandler`-Klasse (Phase 2)
- Änderungen am Dateiformat oder Schema

---

## Architektur-Übersicht

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Python Layer (Orchestration + Caching)                                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CSVDataHandler (Facade)                                           │  │
│  │  ├─ __init__(): Symbol/TF/Paths initialisieren                     │  │
│  │  ├─ load_candles() → Rust oder Python (Feature-Flag)               │  │
│  │  └─ LRU-Cache (_PARQUET_BUILD_CACHE, _DF_BUILD_CACHE)              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                          │                                               │
│                          ▼ Feature Flag: OMEGA_USE_RUST_DATA_HANDLER    │
│                          │                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  FFI Bridge (Arrow IPC)                                            │  │
│  │  ├─ Python → Rust: Path, start_dt, end_dt, flags                   │  │
│  │  └─ Rust → Python: Arrow RecordBatch (OHLCV)                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Rust Layer (omega_rust::data)                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DataHandlerRust                                                   │  │
│  │  ├─ load_parquet_arrow() → RecordBatch                             │  │
│  │  ├─ load_csv_arrow() → RecordBatch                                 │  │
│  │  ├─ filter_market_hours_vec() → RecordBatch                        │  │
│  │  ├─ filter_weekend_flat() → RecordBatch                            │  │
│  │  └─ parallel_load_bid_ask() → (RecordBatch, RecordBatch)           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Dependencies                                                      │  │
│  │  ├─ polars (CSV/Parquet I/O)                                       │  │
│  │  ├─ arrow2 (Arrow IPC Serialization)                               │  │
│  │  ├─ chrono + chrono-tz (Datetime + DST)                            │  │
│  │  └─ rayon (Parallelism)                                            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Phasen-Übersicht

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     WAVE 4: DATA HANDLER RUST MIGRATION                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Phase 4.0: Vorbereitung (Woche 1)                                           ║
║  ├─ FFI-Interface-Spezifikation erstellen                                    ║
║  ├─ Arrow Schema für OHLCV erweitern (Validity Mask)                         ║
║  ├─ Performance-Baseline dokumentieren                                       ║
║  └─ Runbook-Template anpassen                                                ║
║                                                                              ║
║  Phase 4.1: Rust Core Implementation (Woche 2-3)                             ║
║  ├─ omega_rust::data Modul erstellen                                         ║
║  ├─ Parquet-Loader implementieren (polars)                                   ║
║  ├─ CSV-Loader implementieren (polars)                                       ║
║  └─ Unit-Tests in Rust (cargo test)                                          ║
║                                                                              ║
║  Phase 4.2: Filtering & Normalization (Woche 3-4)                            ║
║  ├─ Market-Hours-Filter (DST-aware Sydney Session)                           ║
║  ├─ Weekend + Flat-Bar Filter                                                ║
║  ├─ Timeframe-Normalisierung (floor)                                         ║
║  └─ UTC Timestamp Handling                                                   ║
║                                                                              ║
║  Phase 4.3: Python FFI Bridge (Woche 4-5)                                    ║
║  ├─ PyO3 Bindings für DataHandlerRust                                        ║
║  ├─ Arrow IPC Serialization                                                  ║
║  ├─ Feature-Flag Integration (OMEGA_USE_RUST_DATA_HANDLER)                   ║
║  └─ CSVDataHandler-Integration (Hybrid-Pfad)                                 ║
║                                                                              ║
║  Phase 4.4: Parity & Performance Tests (Woche 5-6)                           ║
║  ├─ Numerische Parity-Tests (Python vs Rust)                                 ║
║  ├─ Edge-Case-Tests (leere Dateien, fehlende Spalten)                        ║
║  ├─ Performance-Benchmarks (pytest-benchmark)                                │
║  └─ Integration mit bestehenden Backtests                                    ║
║                                                                              ║
║  Phase 4.5: Parallelisierung & Optimization (Woche 6-7)                      ║
║  ├─ Parallel Bid+Ask Loading (rayon)                                         ║
║  ├─ Multi-Symbol Parallel Loading                                            ║
║  ├─ Memory-Profiling und Optimierung                                         ║
║  └─ SIMD-Optimierungen (optional)                                            ║
║                                                                              ║
║  Phase 4.6: Dokumentation & Rollout (Woche 7-8)                              ║
║  ├─ Runbook finalisieren                                                     ║
║  ├─ ADR für Data Handler Migration                                           ║
║  ├─ CI/CD Pipeline anpassen                                                  ║
║  └─ Feature-Flag Default auf "auto" setzen                                   ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║  Meilensteine:                                                               ║
║  [W4-M1] Woche 1:  FFI-Spec + Arrow Schema fertig                            ║
║  [W4-M2] Woche 3:  Rust Core (load_parquet/csv) funktional                   ║
║  [W4-M3] Woche 5:  Python Integration + Parity Tests PASS                    ║
║  [W4-M4] Woche 7:  Performance Targets erreicht                              ║
║  [W4-M5] Woche 8:  Production-Ready (Rollout)                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Phase 4.0: Vorbereitung

### Task W4-P0-01: FFI-Interface-Spezifikation

**Datei:** `docs/ffi/data_handler.md`

```python
# @ffi_boundary: Input
class DataLoadRequest:
    """Request für Rust Data Loader."""
    path: str                      # Absoluter Pfad zur Datei
    candle_type: Literal["bid", "ask"]
    start_dt: Optional[int]        # Epoch microseconds (UTC) oder None
    end_dt: Optional[int]          # Epoch microseconds (UTC) oder None
    timeframe: str                 # "M1", "H1", "D1" etc.
    normalize_to_tf: bool          # Timeframe-Flooring aktivieren
    filter_market_hours: bool      # Sydney Session Filter
    filter_weekend_flat: bool      # Wochenenden + Flat-Bars filtern

# @ffi_boundary: Output
# Arrow RecordBatch mit OHLCV_SCHEMA
# Siehe: src/shared/arrow_schemas.py → OHLCV_SCHEMA
```

**Akzeptanzkriterien:**
- [ ] FFI-Spec Dokument in `docs/ffi/data_handler.md`
- [ ] Input/Output-Typen vollständig spezifiziert
- [ ] Error Codes definiert (DATA_FILE_NOT_FOUND, DATA_PARSE_ERROR, etc.)

### Task W4-P0-02: Arrow Schema erweitern

**Datei:** `src/shared/arrow_schemas.py`

```python
# Erweiterung für Data Handler
DATA_LOAD_RESULT_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    # NEU: Validity für filtered/normalized rows
    ("valid", pa.bool_()),
])
```

**Akzeptanzkriterien:**
- [ ] Schema in `arrow_schemas.py` hinzugefügt
- [ ] Schema-Fingerprint in Registry
- [ ] CI-Test für Schema-Drift

### Task W4-P0-03: Performance-Baseline

**Datei:** `reports/performance_baselines/data_handler_baseline.json`

| Operation | Dataset | Python (ms) | Target Rust (ms) |
|-----------|---------|-------------|------------------|
| Parquet Load | 100k rows | 50 | ≤15 |
| Parquet Load | 1M rows | 200 | ≤50 |
| CSV Load | 100k rows | 500 | ≤100 |
| CSV Load | 1M rows | 3500 | ≤500 |
| Market Hours Filter | 1M rows | 150 | ≤30 |
| Full Pipeline | 1M rows | 400 | ≤80 |

**Akzeptanzkriterien:**
- [ ] Baseline-Script: `tools/benchmark_data_handler.py`
- [ ] JSON-Output in `reports/performance_baselines/`
- [ ] 3 Runs pro Operation (Median)

---

## Phase 4.1: Rust Core Implementation

### Task W4-P1-01: Modul-Struktur erstellen

**Pfad:** `src/rust_modules/omega_rust/src/data/`

```
src/data/
├─ mod.rs           # Modul-Exports
├─ loader.rs        # DataHandlerRust Hauptklasse
├─ parquet.rs       # Parquet-spezifische Logik
├─ csv.rs           # CSV-spezifische Logik
├─ filters.rs       # Market Hours, Weekend, Flat-Bar Filter
├─ normalize.rs     # Timeframe Normalization
└─ types.rs         # Interne Rust-Typen
```

**Cargo.toml Erweiterungen:**

```toml
[dependencies]
polars = { version = "0.36", features = ["parquet", "csv", "lazy", "dtype-datetime"] }
chrono = "0.4"
chrono-tz = "0.8"
rayon = "1.8"
arrow2 = { version = "0.18", features = ["io_ipc"] }
```

### Task W4-P1-02: Parquet Loader

**Datei:** `src/rust_modules/omega_rust/src/data/parquet.rs`

```rust
use polars::prelude::*;
use std::path::Path;

pub fn load_parquet(
    path: &Path,
    columns: &[&str],  // ["UTC time", "Open", "High", "Low", "Close", "Volume"]
    start_us: Option<i64>,
    end_us: Option<i64>,
) -> Result<DataFrame, DataError> {
    let df = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
        .select(columns.iter().map(|c| col(*c)).collect::<Vec<_>>())
        .filter(time_range_filter(start_us, end_us))
        .collect()?;
    
    Ok(df)
}
```

**Akzeptanzkriterien:**
- [ ] Parquet-Laden mit Spalten-Selektion
- [ ] Zeitfenster-Filterung (predicate pushdown)
- [ ] UTC-Timestamp-Konvertierung
- [ ] Unit-Tests: `cargo test parquet`

### Task W4-P1-03: CSV Loader

**Datei:** `src/rust_modules/omega_rust/src/data/csv.rs`

```rust
pub fn load_csv(
    path: &Path,
    columns: &[&str],
    start_us: Option<i64>,
    end_us: Option<i64>,
    dtypes: Option<&Schema>,
) -> Result<DataFrame, DataError> {
    let df = CsvReader::from_path(path)?
        .with_columns(Some(columns.to_vec()))
        .with_dtypes(dtypes.cloned())
        .with_try_parse_dates(true)
        .finish()?;
    
    // Post-load filtering (CSV hat kein predicate pushdown)
    filter_time_range(df, start_us, end_us)
}
```

**Akzeptanzkriterien:**
- [ ] CSV-Laden mit dtype-Hints (Float32 optional)
- [ ] Datetime-Parsing (UTC)
- [ ] Unit-Tests: `cargo test csv`

---

## Phase 4.2: Filtering & Normalization

### Task W4-P2-01: Market Hours Filter (DST-aware)

**Datei:** `src/rust_modules/omega_rust/src/data/filters.rs`

```rust
use chrono_tz::Australia::Sydney;

/// Filtert Timestamps basierend auf Sydney Trading Session.
/// Market ist GESCHLOSSEN wenn:
/// - Samstag >= 07:00 Sydney Zeit
/// - Sonntag komplett
/// - Montag < 07:00 Sydney Zeit
pub fn filter_market_hours(
    timestamps: &ChunkedArray<Int64Type>,  // Epoch microseconds
) -> BooleanChunked {
    timestamps.apply(|ts| {
        let utc = Utc.timestamp_micros(ts).unwrap();
        let sydney = utc.with_timezone(&Sydney);
        is_valid_trading_time_sydney(&sydney)
    })
}

fn is_valid_trading_time_sydney(dt: &DateTime<Tz>) -> bool {
    let weekday = dt.weekday();
    let hour = dt.hour();
    
    match weekday {
        Weekday::Sat => hour < 7,      // Vor Sydney-Close
        Weekday::Sun => false,          // Komplett geschlossen
        Weekday::Mon => hour >= 7,      // Nach Sydney-Open
        _ => true,                      // Di-Fr: immer offen
    }
}
```

**Akzeptanzkriterien:**
- [ ] DST-korrekte Sydney-Session-Logik
- [ ] Parity mit Python `is_valid_trading_time_vectorized()`
- [ ] Property-Based Tests mit verschiedenen DST-Übergängen

### Task W4-P2-02: Weekend & Flat-Bar Filter

**Datei:** `src/rust_modules/omega_rust/src/data/filters.rs`

```rust
/// Filtert Wochenenden und 0-Volumen Flat-Bars (Daily TF).
pub fn filter_weekend_flat(
    df: &DataFrame,
    is_daily: bool,
) -> Result<DataFrame, DataError> {
    if !is_daily {
        return Ok(df.clone());
    }
    
    let weekday = df.column("UTC time")?.datetime()?.weekday();
    let is_weekend = weekday.is_in(&[5, 6]);  // Sa=5, So=6
    
    let is_flat_zero = df.column("Volume")?.equal(0)?
        & df.column("Open")?.equal(df.column("High")?)?
        & df.column("High")?.equal(df.column("Low")?)?
        & df.column("Low")?.equal(df.column("Close")?)?;
    
    df.filter(&(!is_weekend & !is_flat_zero))
}
```

### Task W4-P2-03: Timeframe Normalization

**Datei:** `src/rust_modules/omega_rust/src/data/normalize.rs`

```rust
/// Floor Timestamps auf Timeframe-Grenzen.
pub fn floor_to_timeframe(
    timestamps: &ChunkedArray<Int64Type>,
    timeframe: &str,  // "M1", "M5", "H1", "H4", "D1"
) -> Result<ChunkedArray<Int64Type>, DataError> {
    let duration_us = parse_timeframe_duration(timeframe)?;
    
    Ok(timestamps.apply(|ts| {
        (ts / duration_us) * duration_us
    }))
}

fn parse_timeframe_duration(tf: &str) -> Result<i64, DataError> {
    match tf {
        "M1" => Ok(60_000_000),
        "M5" => Ok(5 * 60_000_000),
        "M15" => Ok(15 * 60_000_000),
        "M30" => Ok(30 * 60_000_000),
        "H1" => Ok(60 * 60_000_000),
        "H4" => Ok(4 * 60 * 60_000_000),
        "D1" => Ok(24 * 60 * 60_000_000),
        _ => Err(DataError::InvalidTimeframe(tf.to_string())),
    }
}
```

---

## Phase 4.3: Python FFI Bridge

### Task W4-P3-01: PyO3 Bindings

**Datei:** `src/rust_modules/omega_rust/src/data/mod.rs`

```rust
use pyo3::prelude::*;
use arrow2::io::ipc;

#[pyclass]
pub struct DataHandlerRust {
    // Keine internen State - stateless für einfaches FFI
}

#[pymethods]
impl DataHandlerRust {
    #[new]
    pub fn new() -> Self {
        Self {}
    }
    
    /// Lädt Parquet-Datei und gibt Arrow IPC Bytes zurück.
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
    ) -> PyResult<Vec<u8>> {
        let df = parquet::load_parquet(
            Path::new(path),
            &OHLCV_COLUMNS,
            start_us,
            end_us,
        )?;
        
        let df = if filter_market_hours {
            filters::apply_market_hours_filter(&df)?
        } else {
            df
        };
        
        let df = filters::filter_weekend_flat(&df, timeframe.starts_with("D"))?;
        
        let df = if normalize_to_tf {
            normalize::apply_tf_floor(&df, timeframe)?
        } else {
            df
        };
        
        // Serialize to Arrow IPC
        let batch = df_to_arrow_batch(&df)?;
        let buffer = ipc::write::to_bytes(&batch)?;
        
        Ok(buffer)
    }
    
    /// Paralleles Laden von Bid + Ask.
    pub fn parallel_load_bid_ask(
        &self,
        bid_path: &str,
        ask_path: &str,
        start_us: Option<i64>,
        end_us: Option<i64>,
        timeframe: &str,
        normalize_to_tf: bool,
        filter_market_hours: bool,
    ) -> PyResult<(Vec<u8>, Vec<u8>)> {
        use rayon::prelude::*;
        
        let (bid_result, ask_result) = rayon::join(
            || self.load_parquet_arrow(bid_path, "bid", start_us, end_us, timeframe, normalize_to_tf, filter_market_hours, true),
            || self.load_parquet_arrow(ask_path, "ask", start_us, end_us, timeframe, normalize_to_tf, filter_market_hours, true),
        );
        
        Ok((bid_result?, ask_result?))
    }
}
```

### Task W4-P3-02: Python Integration

**Datei:** `src/backtest_engine/data/data_handler.py` (Erweiterung)

```python
import os
from typing import Optional

# Feature Flag
_RUST_DATA_HANDLER_MODE = os.getenv("OMEGA_USE_RUST_DATA_HANDLER", "auto").lower()

def _use_rust_data_handler() -> bool:
    """Bestimmt ob Rust Data Handler verwendet werden soll."""
    if _RUST_DATA_HANDLER_MODE == "0":
        return False
    if _RUST_DATA_HANDLER_MODE == "1":
        return True
    # auto: Verwende Rust wenn verfügbar
    try:
        from omega_rust import DataHandlerRust
        return True
    except ImportError:
        return False

# Lazy-Loaded Rust Handler
_rust_handler: Optional["DataHandlerRust"] = None

def _get_rust_handler() -> "DataHandlerRust":
    global _rust_handler
    if _rust_handler is None:
        from omega_rust import DataHandlerRust
        _rust_handler = DataHandlerRust()
    return _rust_handler


class CSVDataHandler:
    # ... bestehender Code ...
    
    def load_candles(
        self, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
    ) -> Dict[str, List[Candle]]:
        """Lädt Bid- und Ask-Candles (Rust oder Python)."""
        
        if _use_rust_data_handler() and os.path.exists(self.bid_parquet):
            return self._load_candles_rust(start_dt, end_dt)
        
        # Fallback: Python-Implementierung
        return self._load_candles_python(start_dt, end_dt)
    
    def _load_candles_rust(
        self, start_dt: Optional[datetime], end_dt: Optional[datetime]
    ) -> Dict[str, List[Candle]]:
        """Rust-basiertes paralleles Laden."""
        from shared.arrow_schemas import arrow_to_candle_list
        
        handler = _get_rust_handler()
        
        start_us = int(start_dt.timestamp() * 1_000_000) if start_dt else None
        end_us = int(end_dt.timestamp() * 1_000_000) if end_dt else None
        
        # Paralleles Laden via Rust
        bid_bytes, ask_bytes = handler.parallel_load_bid_ask(
            str(self.bid_parquet),
            str(self.ask_parquet),
            start_us,
            end_us,
            self.timeframe,
            self.normalize_to_timeframe,
            not self.timeframe.upper().startswith("D"),  # filter_market_hours
        )
        
        # Arrow IPC → Candle-Liste
        bid_candles = arrow_to_candle_list(bid_bytes, "bid")
        ask_candles = arrow_to_candle_list(ask_bytes, "ask")
        
        return {"bid": bid_candles, "ask": ask_candles}
    
    def _load_candles_python(
        self, start_dt: Optional[datetime], end_dt: Optional[datetime]
    ) -> Dict[str, List[Candle]]:
        """Bestehende Python-Implementierung (unverändert)."""
        # ... bestehender Code aus load_candles() ...
```

---

## Phase 4.4: Parity & Performance Tests

### Task W4-P4-01: Numerische Parity Tests

**Datei:** `tests/test_data_handler_parity.py`

```python
import pytest
import numpy as np
from datetime import datetime, timezone

from backtest_engine.data.data_handler import CSVDataHandler

@pytest.fixture
def sample_parquet_path(tmp_path):
    """Erstellt Test-Parquet-Datei."""
    # ... Setup ...

class TestDataHandlerParity:
    """Vergleicht Rust vs Python Implementierung."""
    
    def test_parquet_load_parity(self, sample_parquet_path):
        """Rust und Python liefern identische Ergebnisse."""
        # Python-Pfad
        with patch.dict(os.environ, {"OMEGA_USE_RUST_DATA_HANDLER": "0"}):
            handler_py = CSVDataHandler(symbol="EURUSD", timeframe="H1")
            candles_py = handler_py.load_candles()
        
        # Rust-Pfad
        with patch.dict(os.environ, {"OMEGA_USE_RUST_DATA_HANDLER": "1"}):
            handler_rust = CSVDataHandler(symbol="EURUSD", timeframe="H1")
            candles_rust = handler_rust.load_candles()
        
        # Vergleich
        assert len(candles_py["bid"]) == len(candles_rust["bid"])
        for py_c, rust_c in zip(candles_py["bid"], candles_rust["bid"]):
            assert py_c.timestamp == rust_c.timestamp
            assert np.isclose(py_c.open, rust_c.open, rtol=1e-10)
            assert np.isclose(py_c.high, rust_c.high, rtol=1e-10)
            assert np.isclose(py_c.low, rust_c.low, rtol=1e-10)
            assert np.isclose(py_c.close, rust_c.close, rtol=1e-10)
    
    @pytest.mark.parametrize("tf", ["M1", "M5", "H1", "H4", "D1"])
    def test_market_hours_filter_parity(self, sample_data, tf):
        """Market Hours Filter: Python == Rust."""
        # ... Test-Implementation ...
    
    def test_dst_transition_handling(self):
        """DST-Übergänge korrekt behandelt."""
        # Test mit Daten um DST-Wechsel (März/Oktober)
        # ... Test-Implementation ...
```

### Task W4-P4-02: Performance Benchmarks

**Datei:** `tools/benchmark_data_handler.py`

```python
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

class TestDataHandlerBenchmarks:
    
    @pytest.mark.benchmark(group="parquet-load")
    def test_parquet_load_1m_rows_python(self, benchmark, large_parquet_file):
        """Baseline: Python Parquet Load."""
        with patch.dict(os.environ, {"OMEGA_USE_RUST_DATA_HANDLER": "0"}):
            handler = CSVDataHandler(symbol="EURUSD", timeframe="M1")
            benchmark(handler.load_candles)
    
    @pytest.mark.benchmark(group="parquet-load")
    def test_parquet_load_1m_rows_rust(self, benchmark, large_parquet_file):
        """Target: Rust Parquet Load."""
        with patch.dict(os.environ, {"OMEGA_USE_RUST_DATA_HANDLER": "1"}):
            handler = CSVDataHandler(symbol="EURUSD", timeframe="M1")
            benchmark(handler.load_candles)
    
    @pytest.mark.benchmark(group="parallel-load")
    def test_parallel_bid_ask_load(self, benchmark, large_parquet_files):
        """Rust Parallel Load Performance."""
        handler = CSVDataHandler(symbol="EURUSD", timeframe="M1")
        benchmark(handler._load_candles_rust, None, None)
```

---

## Phase 4.5: Parallelisierung & Optimization

### Task W4-P5-01: Multi-Symbol Parallel Loading

**Datei:** `src/rust_modules/omega_rust/src/data/mod.rs` (Erweiterung)

```rust
/// Lädt mehrere Symbole parallel.
pub fn load_multi_symbol_parallel(
    &self,
    requests: Vec<(String, String, String)>,  // [(bid_path, ask_path, symbol), ...]
    start_us: Option<i64>,
    end_us: Option<i64>,
    timeframe: &str,
) -> PyResult<Vec<(String, Vec<u8>, Vec<u8>)>> {
    use rayon::prelude::*;
    
    requests.par_iter()
        .map(|(bid_path, ask_path, symbol)| {
            let (bid, ask) = self.parallel_load_bid_ask(bid_path, ask_path, start_us, end_us, timeframe, false, true)?;
            Ok((symbol.clone(), bid, ask))
        })
        .collect()
}
```

### Task W4-P5-02: Memory Profiling

**Akzeptanzkriterien:**
- [ ] Peak-RAM bei 1M Zeilen: ≤150 MB (vs. ~400 MB Python)
- [ ] Keine Memory-Leaks bei wiederholtem Laden
- [ ] Profiling-Report mit `valgrind` oder `heaptrack`

---

## Phase 4.6: Dokumentation & Rollout

### Task W4-P6-01: Runbook erstellen

**Datei:** `docs/runbooks/data_handler_migration.md`

Struktur analog zu `docs/runbooks/indicator_cache_migration.md`:
- Voraussetzungen
- Aktivierung via Feature-Flag
- Rollback-Prozedur
- Troubleshooting
- Performance-Monitoring

### Task W4-P6-02: ADR erstellen

**Datei:** `docs/adr/ADR-0006-data-handler-rust-migration.md`

```markdown
---
title: "ADR-0006: Data Handler Rust Migration"
status: Proposed
date: 2026-01-XX
---

## Kontext
Der CSVDataHandler lädt OHLCV-Marktdaten aus CSV/Parquet-Dateien.
Bei Multi-Symbol-Optimierungen mit großen Datensätzen ist das Laden ein Bottleneck.

## Entscheidung
Migration der Parsing-Logik nach Rust mit:
- polars für I/O
- rayon für Parallelisierung
- Arrow IPC für Zero-Copy Transfer

## Konsequenzen
+ 5-10x schnelleres Laden großer Dateien
+ Paralleles Multi-Symbol-Laden
+ 50% weniger Peak-RAM
- Zusätzliche Build-Komplexität (Rust)
- DST-Handling muss synchron gehalten werden
```

### Task W4-P6-03: CI/CD Anpassungen

**Datei:** `.github/workflows/ci.yml` (Erweiterung)

```yaml
jobs:
  test-data-handler-parity:
    runs-on: ubuntu-latest
    steps:
      - name: Run Data Handler Parity Tests
        run: |
          pytest tests/test_data_handler_parity.py -v
        env:
          OMEGA_USE_RUST_DATA_HANDLER: "1"
  
  benchmark-data-handler:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Run Data Handler Benchmarks
        run: |
          pytest tools/benchmark_data_handler.py --benchmark-json=benchmark.json
      - name: Compare with Baseline
        run: |
          python tools/compare_benchmark.py benchmark.json reports/performance_baselines/data_handler_baseline.json
```

---

## Risiko-Matrix

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| DST-Handling Divergenz | Mittel | Hoch | Property-Based Tests für DST-Übergänge; Shared Timezone-DB |
| Performance-Regression | Niedrig | Mittel | CI-Benchmarks mit Threshold-Gates |
| Arrow Schema Drift | Niedrig | Hoch | Schema-Fingerprint-Tests in CI |
| Memory Leaks in Rust | Niedrig | Hoch | Valgrind-Tests in CI; MIRI für unsafe Code |
| Polars API Breaking Changes | Mittel | Mittel | Cargo.lock pinning; Renovate-Bot |

---

## Erfolgsmetriken

| Metrik | Baseline (Python) | Target (Rust) | Messmethode |
|--------|-------------------|---------------|-------------|
| Parquet Load 1M rows | 200 ms | ≤50 ms | pytest-benchmark |
| CSV Load 1M rows | 3500 ms | ≤500 ms | pytest-benchmark |
| Peak RAM 1M rows | 400 MB | ≤150 MB | tracemalloc / heaptrack |
| Parallel Speedup (Bid+Ask) | 1.0x (sequential) | 1.8-2.0x | pytest-benchmark |
| Parity Tests | N/A | 100% PASS | CI Gate |

---

## Abhängigkeiten zu bestehenden Waves

| Wave | Modul | Abhängigkeit |
|------|-------|--------------|
| Wave 1 | IndicatorCache | Keine direkte Abhängigkeit |
| Wave 2 | Portfolio | Konsumiert Candle-Daten |
| Wave 3 | EventEngine | Konsumiert Candle-Daten |
| Wave 0 | Slippage/Fee | Keine Abhängigkeit |

**Empfehlung:** Wave 4 kann parallel zu Wave 3 entwickelt werden.

---

## Rollout-Strategie

### Phase 1: Alpha (Woche 6-7)
- Feature-Flag: `OMEGA_USE_RUST_DATA_HANDLER=0` (Default)
- Manuelle Aktivierung für Tester
- Monitoring: Parity-Abweichungen loggen

### Phase 2: Beta (Woche 7-8)
- Feature-Flag: `OMEGA_USE_RUST_DATA_HANDLER=auto` (Default)
- Automatische Aktivierung wenn Rust verfügbar
- Fallback auf Python bei Fehlern

### Phase 3: GA (Woche 8+)
- Feature-Flag: `OMEGA_USE_RUST_DATA_HANDLER=1` (Default)
- Python-Pfad bleibt als Fallback
- Deprecation-Warning für Python-only Usage

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0.0 | 2026-01-11 | Initiale Version |

---

## Referenzen

- [ADR-0001: Migration Strategy](adr/ADR-0001-migration-strategy.md)
- [ADR-0002: Serialization Format (Arrow IPC)](adr/ADR-0002-serialization-format.md)
- [ADR-0003: Error Handling](adr/ADR-0003-error-handling.md)
- [Wave 1 Indicator Cache Report](../reports/rust_indicator_analysis_report.md)
- [FFI Interface Specs](ffi/README.md)
- [Runbook Template](runbooks/MIGRATION_RUNBOOK_TEMPLATE.md)
