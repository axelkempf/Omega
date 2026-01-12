# Wave 1.1: FFI-Overhead-Reduktion Implementation Plan

**Document Version:** 1.0  
**Created:** 2026-01-09  
**Updated:** 2026-01-09  
**Status:** 📋 PLANNED  
**Scope:** FFI-Overhead-Optimierung für IndicatorCache

---

## Executive Summary

Dieser Plan beschreibt die Reduktion des FFI-Overheads zwischen Python und Rust für das IndicatorCache-Modul als **Wave 1.1** – ein Performance-Optimierungs-Follow-up zu Wave 1 (IndicatorCache Rust Migration).

Die Wave 1 Benchmarks zeigen, dass **komplexe Indikatoren massiv profitieren** (Kalman Mean: 528x, ATR: 79x), aber **einfache Indikatoren durch FFI-Overhead regredierten** (Bollinger: 0.6x, MACD: 0.7x, Z-Score: 0.2x).

Wave 1.1 adressiert diese Regression durch drei komplementäre Optimierungen:

| Option | Beschreibung | Erwarteter Impact | Aufwand |
|--------|--------------|-------------------|---------|
| **Option 5** | pandas-Wrapper eliminieren (NumPy-Array-API) | Eliminiert ~50-200μs pro Aufruf | 1-2 Tage |
| **Option 1** | Batch-Indikator-Berechnung (Single FFI-Call) | Eliminiert N-1 FFI-Calls | 2-3 Tage |
| **Option 4** | Apache Arrow IPC für Zero-Copy Transfer | True Zero-Copy, Julia-kompatibel | 1-2 Wochen |

### Warum Wave 1.1?

| Problem | Messwert (Wave 1) | Ursache | Lösung |
|---------|-------------------|---------|--------|
| Bollinger Bands 0.6x langsamer | 8.77ms vs 5.34ms Python | `_series_from_rust_array()` Overhead | Option 5 |
| MACD 0.7x langsamer | 9.24ms vs 6.09ms Python | `_series_from_rust_array()` Overhead | Option 5 |
| Z-Score 0.2x langsamer | 34.98ms vs 6.35ms Python | Mehrfache FFI-Calls + Series Wrapper | Option 1 + 5 |
| Strategien rufen N Indikatoren einzeln auf | N FFI-Round-Trips | Keine Batch-API | Option 1 |
| NumPy-Kopien bei FFI | Potenzielle Memory-Kopie | Kein Arrow IPC | Option 4 |

### Performance-Targets

| Metrik | Wave 1 Ist | Wave 1.1 Ziel | Verbesserung |
|--------|------------|---------------|--------------|
| Bollinger Speedup | 0.6x | ≥3x | +400% |
| MACD Speedup | 0.7x | ≥3x | +330% |
| Z-Score Speedup | 0.2x | ≥3x | +1400% |
| FFI-Calls pro Strategie-Tick | N (5-15) | 1 (Batch) | -90% |
| Memory-Kopien | 1+ pro Call | 0 (Zero-Copy) | -100% |

---

## Inhaltsverzeichnis

1. [Voraussetzungen & Status](#1-voraussetzungen--status)
2. [Architektur-Übersicht](#2-architektur-übersicht)
3. [Phase 1: NumPy-Array-API (Option 5)](#3-phase-1-numpy-array-api-option-5)
4. [Phase 2: Batch-Indikator-API (Option 1)](#4-phase-2-batch-indikator-api-option-1)
5. [Phase 3: Apache Arrow IPC (Option 4)](#5-phase-3-apache-arrow-ipc-option-4)
6. [Test-Strategie](#6-test-strategie)
7. [Validierung & Akzeptanzkriterien](#7-validierung--akzeptanzkriterien)
8. [Rollback-Plan](#8-rollback-plan)
9. [Lessons Learned aus Wave 1](#9-lessons-learned-aus-wave-1)
10. [Checklisten](#10-checklisten)

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure-Readiness (aus Wave 1 etabliert)

| Komponente | Status | Evidenz |
|------------|--------|---------|
| Rust IndicatorCache | ✅ | `src/rust_modules/omega_rust/src/indicators/` |
| PyO3/Maturin | ✅ | Version 0.27, abi3-py312 |
| Feature-Flag System | ✅ | `OMEGA_USE_RUST_INDICATOR_CACHE` |
| 24 Rust-Indikatoren | ✅ | Vollständig integriert |
| Parity Tests | ✅ | `tests/test_indicator_cache_rust.py` (17 Tests) |
| Benchmark-Tool | ✅ | `tools/benchmark_indicator_cache.py` |
| Arrow (optional) | ⚠️ | In Cargo.toml, aber **nicht aktiviert** |

### 1.2 Wave 1 Benchmark Baseline

**Test-Konfiguration:** 100.000 Bars, Direct Calls ohne Caching

| Indikator | Python (ms) | Rust (ms) | Speedup | Problem |
|-----------|-------------|-----------|---------|---------|
| `ema(20)` | 6.61 | 1.90 | **3.5x** | ✅ OK |
| `atr(14)` | 164.85 | 2.09 | **79.0x** | ✅ OK |
| `kalman_mean` | 631.29 | 1.20 | **528.2x** | ✅ OK |
| `bollinger(20)` | 5.34 | 8.77 | **0.6x** | ⚠️ **Regression** |
| `macd(12,26,9)` | 6.09 | 9.24 | **0.7x** | ⚠️ **Regression** |
| `zscore(100)` | 6.35 | 34.98 | **0.2x** | ⚠️ **Regression** |

**Ursachen-Analyse:**

```
Rust-Berechnung:     ~1-5ms (schnell)
PyO3 FFI-Call:       ~10-50μs (akzeptabel)
GIL-Handling:        ~5-20μs (akzeptabel)
_series_from_rust_array(): ~50-200μs (Problem!)
pandas.Series():     ~100-300μs (Problem!)
───────────────────────────────────────────
Gesamter Overhead:   ~200-600μs pro Aufruf
```

**Fazit:** Der pandas-Wrapper dominiert bei einfachen Indikatoren die Gesamtzeit.

### 1.3 Aktueller Datenfluss (zu optimieren)

```
Python Strategy
    │
    │  ind.ema(tf, "bid", 14)
    ▼
IndicatorCache.ema()
    │
    │  self._rust_available_for("ema")  ← hasattr() Check (~1μs)
    ▼
IndicatorCacheRust.ema()               ← PyO3 FFI Call (~20μs)
    │
    │  Rust Computation                 ← Fast (~1-5ms)
    ▼
result.to_pyarray(py)                  ← NumPy Array Return (~5μs)
    │
    ▼
_series_from_rust_array()              ← pandas.Series() (~200μs) ⚠️
    │
    ▼
pd.Series zurück an Strategy
```

---

## 2. Architektur-Übersicht

### 2.1 Ziel-Architektur nach Wave 1.1

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STRATEGY LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │   Phase 1: NumPy-Array-API (Option 5)                                      │ │
│  │                                                                            │ │
│  │   # Alt (mit pandas-Overhead):                                             │ │
│  │   z = ind.zscore(tf, "bid", window=100)     # → pd.Series                  │ │
│  │   z_now = self._safe_at(z, idx)             # → Index-Lookup               │ │
│  │                                                                            │ │
│  │   # Neu (ohne pandas-Overhead):                                            │ │
│  │   z = ind.zscore_array(tf, "bid", window=100)  # → np.ndarray ✅           │ │
│  │   z_now = z[idx] if idx < len(z) else np.nan   # → Direct Index ✅         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │   Phase 2: Batch-Indikator-API (Option 1)                                  │ │
│  │                                                                            │ │
│  │   # Alt (N FFI-Calls pro Tick):                                            │ │
│  │   ema = ind.ema(tf, "bid", 14)              # FFI Call 1                   │ │
│  │   atr = ind.atr(tf, "bid", 14)              # FFI Call 2                   │ │
│  │   zscore = ind.zscore(tf, "bid", 100)       # FFI Call 3                   │ │
│  │   ...                                        # FFI Call N                   │ │
│  │                                                                            │ │
│  │   # Neu (1 FFI-Call für alle):                                             │ │
│  │   results = ind.compute_batch([                                            │ │
│  │       {"name": "ema", "period": 14},                                       │ │
│  │       {"name": "atr", "period": 14},                                       │ │
│  │       {"name": "zscore", "window": 100},                                   │ │
│  │   ], tf, "bid")  # → dict[str, np.ndarray] ✅                              │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              │ FFI Boundary (PyO3 + Arrow IPC)                   │
│                              ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │   Phase 3: Apache Arrow IPC (Option 4)                                     │ │
│  │                                                                            │ │
│  │   # Zero-Copy OHLCV Transfer (Python → Rust)                               │ │
│  │   arrow_batch = pa.RecordBatch.from_pandas(ohlcv_df)                       │ │
│  │   rust_cache.register_ohlcv_arrow(arrow_batch)  # Zero-Copy ✅             │ │
│  │                                                                            │ │
│  │   # Zero-Copy Indicator Return (Rust → Python)                             │ │
│  │   arrow_result = rust_cache.ema_arrow(...)      # Zero-Copy ✅             │ │
│  │   np_array = arrow_result.to_numpy()            # Zero-Copy View ✅        │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Datei-Struktur nach Wave 1.1

```
src/
├── rust_modules/
│   └── omega_rust/
│       ├── src/
│       │   ├── lib.rs                       # Arrow-Feature aktivieren
│       │   ├── indicators/
│       │   │   ├── mod.rs
│       │   │   ├── cache.rs                 # Bestehendes Modul
│       │   │   ├── py_bindings.rs           # ERWEITERT: Batch-API + Arrow
│       │   │   ├── batch.rs                 # NEU: Batch-Berechnung
│       │   │   └── arrow_interop.rs         # NEU: Arrow IPC Integration
│       │   └── arrow/                       # NEU: Arrow Utilities
│       │       ├── mod.rs
│       │       └── schemas.rs               # Schema-Definitionen
│       └── Cargo.toml                       # Arrow-Feature aktivieren
│
├── backtest_engine/
│   └── core/
│       └── indicator_cache.py               # ERWEITERT: *_array() + compute_batch()
│
├── shared/
│   └── arrow_schemas.py                     # ERWEITERT: Schema-Syncing
│
└── strategies/
    ├── _template/
    │   └── strategy.py                      # ERWEITERT: Batch-Pattern Beispiel
    └── mean_reversion_z_score/
        └── strategy.py                      # MIGRATION: NumPy-Array-API

tests/
├── test_indicator_cache_array_api.py        # NEU: Array-API Tests
├── test_indicator_cache_batch.py            # NEU: Batch-API Tests
├── test_indicator_cache_arrow.py            # NEU: Arrow IPC Tests
└── benchmarks/
    └── test_bench_wave_1_1.py               # NEU: Overhead-Benchmark
```

---

## 3. Phase 1: NumPy-Array-API (Option 5)

**Ziel:** pandas-Wrapper eliminieren, direkter NumPy-Array-Return

**Aufwand:** 1-2 Tage

### 3.1 Änderungen in `indicator_cache.py`

```python
# src/backtest_engine/core/indicator_cache.py

class IndicatorCache:
    """Extended with direct NumPy array access methods."""
    
    # ========== NEUE ARRAY-API (ohne pandas-Overhead) ==========
    
    def ema_array(
        self,
        tf: str,
        price_type: str,
        period: int,
    ) -> np.ndarray:
        """Return EMA as NumPy array (no pandas overhead).
        
        Use this for performance-critical code paths where 
        pd.Series index alignment is not needed.
        """
        if self._rust_available_for("ema"):
            try:
                return self._rust_cache.ema(
                    "BACKTEST", tf, price_type.upper(), period
                )
            except Exception:
                pass  # Fallback to Python
        # Python fallback (still returns np.ndarray)
        series = self.ema(tf, price_type, period)
        return series.to_numpy()
    
    def atr_array(
        self,
        tf: str,
        price_type: str,
        period: int,
    ) -> np.ndarray:
        """Return ATR as NumPy array (no pandas overhead)."""
        if self._rust_available_for("atr"):
            try:
                return self._rust_cache.atr(
                    "BACKTEST", tf, price_type.upper(), period
                )
            except Exception:
                pass
        return self.atr(tf, price_type, period).to_numpy()
    
    def zscore_array(
        self,
        tf: str,
        price_type: str,
        window: int,
        mean_source: str = "rolling",
        ema_period: int | None = None,
    ) -> np.ndarray:
        """Return Z-Score as NumPy array (no pandas overhead)."""
        if self._rust_available_for("zscore") and mean_source == "rolling":
            try:
                return self._rust_cache.zscore(
                    "BACKTEST", tf, price_type.upper(), window
                )
            except Exception:
                pass
        return self.zscore(tf, price_type, window, mean_source, ema_period).to_numpy()
    
    def bollinger_array(
        self,
        tf: str,
        price_type: str,
        period: int,
        std_dev: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Bollinger Bands as NumPy arrays (upper, middle, lower)."""
        if self._rust_available_for("bollinger"):
            try:
                return self._rust_cache.bollinger(
                    "BACKTEST", tf, price_type.upper(), period, std_dev
                )
            except Exception:
                pass
        upper, middle, lower = self.bollinger(tf, price_type, period, std_dev)
        return upper.to_numpy(), middle.to_numpy(), lower.to_numpy()
    
    def macd_array(
        self,
        tf: str,
        price_type: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return MACD as NumPy arrays (macd_line, signal_line, histogram)."""
        if self._rust_available_for("macd"):
            try:
                return self._rust_cache.macd(
                    "BACKTEST", tf, price_type.upper(), fast, slow, signal
                )
            except Exception:
                pass
        macd_line, signal_line, hist = self.macd(tf, price_type, fast, slow, signal)
        return macd_line.to_numpy(), signal_line.to_numpy(), hist.to_numpy()
    
    # ... weitere *_array() Methoden für alle 24 Indikatoren
```

### 3.2 Strategie-Migration (Beispiel)

**Vorher:**
```python
# src/strategies/mean_reversion_z_score/strategy.py

def _evaluate_long_1(self, symbol_slice, bid_candle, ask_candle):
    ind = symbol_slice.indicators
    idx = self._idx(symbol_slice)
    
    # Aktuell: pandas.Series (mit Overhead)
    z = ind.zscore(self.timeframe, "bid", window=self.window_length, mean_source="ema")
    z_now = self._safe_at(z, idx)
    
    atr = ind.atr(self.timeframe, "bid", self.atr_length)
    atr_now = self._safe_at(atr, idx)
```

**Nachher:**
```python
# src/strategies/mean_reversion_z_score/strategy.py

def _evaluate_long_1(self, symbol_slice, bid_candle, ask_candle):
    ind = symbol_slice.indicators
    idx = self._idx(symbol_slice)
    
    # Neu: NumPy array (ohne Overhead)
    z = ind.zscore_array(self.timeframe, "bid", window=self.window_length)
    z_now = z[idx] if idx < len(z) else np.nan
    
    atr = ind.atr_array(self.timeframe, "bid", self.atr_length)
    atr_now = atr[idx] if idx < len(atr) else np.nan
```

### 3.3 Rust-Änderungen (minimal)

Die Rust-Seite benötigt **keine Änderungen** – sie gibt bereits `numpy::PyArray1<f64>` zurück.

Die Optimierung liegt rein auf der Python-Seite durch Eliminierung von `_series_from_rust_array()`.

### 3.4 Erwartete Performance-Verbesserung

| Indikator | Wave 1 | Nach Phase 1 | Verbesserung |
|-----------|--------|--------------|--------------|
| `bollinger` | 0.6x | ~2.5x | +320% |
| `macd` | 0.7x | ~2.8x | +300% |
| `zscore` | 0.2x | ~1.5x | +650% |
| `ema` | 3.5x | ~4.0x | +14% |

---

## 4. Phase 2: Batch-Indikator-API (Option 1)

**Ziel:** Alle Indikatoren einer Strategie in einem FFI-Call berechnen

**Aufwand:** 2-3 Tage

### 4.1 Python-API

```python
# src/backtest_engine/core/indicator_cache.py

class IndicatorCache:
    """Extended with batch computation API."""
    
    def compute_batch(
        self,
        indicators: list[dict],
        tf: str,
        price_type: str,
    ) -> dict[str, np.ndarray]:
        """Compute multiple indicators in a single FFI call.
        
        Args:
            indicators: List of indicator specifications, e.g.:
                [
                    {"name": "ema", "period": 14},
                    {"name": "atr", "period": 14},
                    {"name": "zscore", "window": 100},
                    {"name": "bollinger", "period": 20, "std_dev": 2.0},
                ]
            tf: Timeframe (e.g., "M5", "H1")
            price_type: Price type ("bid" or "ask")
        
        Returns:
            Dictionary mapping indicator names to NumPy arrays.
            For multi-output indicators (bollinger, macd), keys are suffixed:
            "bollinger_upper", "bollinger_middle", "bollinger_lower"
        
        Example:
            results = ind.compute_batch([
                {"name": "ema", "period": 14},
                {"name": "atr", "period": 14},
            ], "M5", "bid")
            
            ema_values = results["ema"]
            atr_values = results["atr"]
        """
        if self._use_rust and self._rust_cache is not None:
            try:
                return self._rust_cache.compute_batch(
                    "BACKTEST",
                    tf,
                    price_type.upper(),
                    indicators,
                )
            except Exception as e:
                logger.warning(f"Rust batch computation failed: {e}, falling back to Python")
        
        # Python fallback: compute individually
        results = {}
        for spec in indicators:
            name = spec["name"]
            if name == "ema":
                results[name] = self.ema_array(tf, price_type, spec["period"])
            elif name == "atr":
                results[name] = self.atr_array(tf, price_type, spec["period"])
            elif name == "zscore":
                results[name] = self.zscore_array(tf, price_type, spec["window"])
            elif name == "bollinger":
                upper, middle, lower = self.bollinger_array(
                    tf, price_type, spec["period"], spec.get("std_dev", 2.0)
                )
                results["bollinger_upper"] = upper
                results["bollinger_middle"] = middle
                results["bollinger_lower"] = lower
            elif name == "macd":
                macd, signal, hist = self.macd_array(
                    tf, price_type,
                    spec.get("fast", 12),
                    spec.get("slow", 26),
                    spec.get("signal", 9),
                )
                results["macd_line"] = macd
                results["macd_signal"] = signal
                results["macd_histogram"] = hist
            # ... weitere Indikatoren
        return results
```

### 4.2 Rust-Implementation

```rust
// src/rust_modules/omega_rust/src/indicators/batch.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, ToPyArray};
use std::collections::HashMap;

/// Batch indicator computation for minimal FFI overhead.
pub fn compute_indicator_batch<'py>(
    py: Python<'py>,
    cache: &mut IndicatorCache,
    symbol: &str,
    timeframe: &str,
    price_type: &str,
    indicators: &Bound<'py, PyList>,
) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
    let mut results = HashMap::new();
    
    for item in indicators.iter() {
        let spec: &Bound<PyDict> = item.downcast()?;
        let name: String = spec.get_item("name")?.unwrap().extract()?;
        
        match name.as_str() {
            "ema" => {
                let period: usize = spec.get_item("period")?.unwrap().extract()?;
                let result = cache.ema(symbol, timeframe, price_type, period)?;
                results.insert(name, result.to_pyarray(py).into());
            }
            "atr" => {
                let period: usize = spec.get_item("period")?.unwrap().extract()?;
                let result = cache.atr(symbol, timeframe, price_type, period)?;
                results.insert(name, result.to_pyarray(py).into());
            }
            "zscore" => {
                let window: usize = spec.get_item("window")?.unwrap().extract()?;
                let result = cache.zscore(symbol, timeframe, price_type, window)?;
                results.insert(name, result.to_pyarray(py).into());
            }
            "bollinger" => {
                let period: usize = spec.get_item("period")?.unwrap().extract()?;
                let std_dev: f64 = spec
                    .get_item("std_dev")?
                    .map(|v| v.extract().unwrap_or(2.0))
                    .unwrap_or(2.0);
                let (upper, middle, lower) = cache.bollinger(
                    symbol, timeframe, price_type, period, std_dev
                )?;
                results.insert("bollinger_upper".to_string(), upper.to_pyarray(py).into());
                results.insert("bollinger_middle".to_string(), middle.to_pyarray(py).into());
                results.insert("bollinger_lower".to_string(), lower.to_pyarray(py).into());
            }
            "macd" => {
                let fast: usize = spec.get_item("fast")?.map(|v| v.extract().unwrap_or(12)).unwrap_or(12);
                let slow: usize = spec.get_item("slow")?.map(|v| v.extract().unwrap_or(26)).unwrap_or(26);
                let signal: usize = spec.get_item("signal")?.map(|v| v.extract().unwrap_or(9)).unwrap_or(9);
                let (macd_line, signal_line, histogram) = cache.macd(
                    symbol, timeframe, price_type, fast, slow, signal
                )?;
                results.insert("macd_line".to_string(), macd_line.to_pyarray(py).into());
                results.insert("macd_signal".to_string(), signal_line.to_pyarray(py).into());
                results.insert("macd_histogram".to_string(), histogram.to_pyarray(py).into());
            }
            // ... weitere Indikatoren
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown indicator: {}", name)
                ));
            }
        }
    }
    
    Ok(results)
}
```

### 4.3 PyO3 Binding

```rust
// src/rust_modules/omega_rust/src/indicators/py_bindings.rs

#[pymethods]
impl PyIndicatorCache {
    // ... bestehende Methoden ...
    
    /// Compute multiple indicators in a single call.
    /// 
    /// This reduces FFI overhead by computing all indicators
    /// in one Python → Rust → Python round-trip.
    pub fn compute_batch<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        indicators: &Bound<'py, PyList>,
    ) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
        let mut cache = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to acquire lock: {}", e)
            )
        })?;
        
        batch::compute_indicator_batch(py, &mut cache, symbol, timeframe, price_type, indicators)
    }
}
```

### 4.4 Strategie-Verwendung (Batch-Pattern)

```python
# src/strategies/mean_reversion_z_score/strategy.py

class MeanReversionZScoreStrategy(BaseStrategy):
    
    def _precompute_indicators(self, symbol_slice):
        """Precompute all needed indicators in one FFI call."""
        ind = symbol_slice.indicators
        
        # Alle Indikatoren in einem Call berechnen
        self._indicators = ind.compute_batch([
            {"name": "zscore", "window": self.window_length},
            {"name": "atr", "period": self.atr_length},
            {"name": "ema", "period": self.ema_length},
            {"name": "bollinger", "period": 20, "std_dev": 2.0},
        ], self.timeframe, "bid")
    
    def _evaluate_long_1(self, symbol_slice, bid_candle, ask_candle):
        idx = self._idx(symbol_slice)
        
        # Direkter Array-Zugriff (kein FFI-Call mehr!)
        z_now = self._indicators["zscore"][idx]
        atr_now = self._indicators["atr"][idx]
        ema_now = self._indicators["ema"][idx]
```

### 4.5 Erwartete Performance-Verbesserung

| Szenario | FFI-Calls (vorher) | FFI-Calls (nachher) | Reduktion |
|----------|-------------------|---------------------|-----------|
| Strategie mit 5 Indikatoren | 5 | 1 | -80% |
| Strategie mit 10 Indikatoren | 10 | 1 | -90% |
| Optimizer mit 1000 Iterationen | 5000 | 1000 | -80% |

**Time Savings:**
- FFI-Overhead pro Call: ~50-100μs
- 5 Indikatoren × 50μs = 250μs → 50μs (5x schneller)
- Pro 100.000 Ticks: 25s → 5s Ersparnis

---

## 5. Phase 3: Apache Arrow IPC (Option 4)

**Ziel:** True Zero-Copy Datentransfer zwischen Python und Rust

**Aufwand:** 1-2 Wochen

### 5.1 Cargo.toml Änderungen

```toml
# src/rust_modules/omega_rust/Cargo.toml

[dependencies]
# ... bestehende Dependencies ...

# Arrow für Zero-Copy FFI
arrow = { version = "55", features = ["ffi"] }
arrow-array = "55"
arrow-schema = "55"

[features]
default = ["arrow"]  # Arrow standardmäßig aktivieren
arrow = ["dep:arrow", "dep:arrow-array", "dep:arrow-schema"]
```

### 5.2 Arrow Schema Definition (Rust-Seite)

```rust
// src/rust_modules/omega_rust/src/arrow/schemas.rs

use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// OHLCV Arrow Schema für Zero-Copy Transfer
pub fn ohlcv_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, true),
    ]))
}

/// Indicator Result Schema
pub fn indicator_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),  // nullable für NaN
    ]))
}

/// Multi-Output Indicator Schema (z.B. Bollinger, MACD)
pub fn multi_indicator_schema(output_names: &[&str]) -> Arc<Schema> {
    let fields: Vec<Field> = std::iter::once(
        Field::new("timestamp", DataType::Int64, false)
    )
    .chain(output_names.iter().map(|name| {
        Field::new(*name, DataType::Float64, true)
    }))
    .collect();
    
    Arc::new(Schema::new(fields))
}
```

### 5.3 Arrow IPC Integration (Rust-Seite)

```rust
// src/rust_modules/omega_rust/src/indicators/arrow_interop.rs

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Float64Array, Int64Array, RecordBatch};
use pyo3::prelude::*;
use pyo3::ffi::c_void;

/// Receive OHLCV data via Arrow FFI (Zero-Copy)
pub fn register_ohlcv_from_arrow(
    cache: &mut IndicatorCache,
    symbol: &str,
    timeframe: &str,
    price_type: &str,
    arrow_ptr: *const c_void,
    schema_ptr: *const c_void,
) -> PyResult<()> {
    unsafe {
        // Import Arrow array from Python (Zero-Copy!)
        let ffi_array = &*(arrow_ptr as *const FFI_ArrowArray);
        let ffi_schema = &*(schema_ptr as *const FFI_ArrowSchema);
        
        let array = arrow::ffi::import_array_from_c(
            ffi_array.clone(),
            ffi_schema.clone(),
        )?;
        
        let record_batch = RecordBatch::from(array);
        
        // Extract columns
        let timestamps = record_batch
            .column_by_name("timestamp")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        
        let opens = extract_f64_column(&record_batch, "open")?;
        let highs = extract_f64_column(&record_batch, "high")?;
        let lows = extract_f64_column(&record_batch, "low")?;
        let closes = extract_f64_column(&record_batch, "close")?;
        let volumes = extract_f64_column(&record_batch, "volume")?;
        
        // Store in cache
        cache.register_ohlcv_data(
            symbol, timeframe, price_type,
            timestamps.values().to_vec(),
            opens, highs, lows, closes, volumes,
        )?;
    }
    
    Ok(())
}

/// Export indicator result via Arrow FFI (Zero-Copy)
pub fn export_indicator_as_arrow<'py>(
    py: Python<'py>,
    timestamps: &[i64],
    values: &[f64],
) -> PyResult<(usize, usize)> {
    let timestamp_array = Int64Array::from(timestamps.to_vec());
    let value_array = Float64Array::from(values.to_vec());
    
    let batch = RecordBatch::try_new(
        super::schemas::indicator_schema(),
        vec![
            Arc::new(timestamp_array),
            Arc::new(value_array),
        ],
    )?;
    
    // Export to Arrow FFI pointers
    let (ffi_array, ffi_schema) = arrow::ffi::export_array_to_c(batch.into())?;
    
    // Return pointers as integers for Python
    Ok((
        Box::into_raw(Box::new(ffi_array)) as usize,
        Box::into_raw(Box::new(ffi_schema)) as usize,
    ))
}
```

### 5.4 Python Arrow Integration

```python
# src/backtest_engine/core/indicator_cache.py

import pyarrow as pa
from typing import Optional

class IndicatorCache:
    """Extended with Arrow IPC support."""
    
    def _init_rust_cache_arrow(self) -> None:
        """Initialize Rust cache with Arrow Zero-Copy transfer."""
        if not self._use_rust or self._rust_cache is None:
            return
        
        for (tf, side), df in self._dataframes.items():
            # Convert pandas DataFrame to Arrow RecordBatch
            arrow_table = pa.Table.from_pandas(df[["open", "high", "low", "close", "volume"]])
            
            # Add timestamp column
            timestamps = pa.array(df.index.astype("int64") // 10**9, type=pa.int64())
            arrow_table = arrow_table.append_column("timestamp", timestamps)
            
            # Get FFI pointers (Zero-Copy!)
            batch = arrow_table.to_batches()[0]
            array_ptr = batch._export_to_c()
            schema_ptr = batch.schema._export_to_c()
            
            # Register in Rust (Zero-Copy!)
            self._rust_cache.register_ohlcv_arrow(
                "BACKTEST",
                tf,
                side.upper(),
                array_ptr,
                schema_ptr,
            )
    
    def ema_arrow(
        self,
        tf: str,
        price_type: str,
        period: int,
    ) -> pa.Array:
        """Return EMA as Arrow array (True Zero-Copy)."""
        if not self._use_rust:
            raise RuntimeError("Arrow API requires Rust backend")
        
        array_ptr, schema_ptr = self._rust_cache.ema_arrow(
            "BACKTEST", tf, price_type.upper(), period
        )
        
        # Import Arrow array from Rust (Zero-Copy!)
        return pa.Array._import_from_c(array_ptr, schema_ptr)
```

### 5.5 Schema-Synchronisation

```python
# src/shared/arrow_schemas.py

import pyarrow as pa

# Synchron mit Rust-Definitionen in omega_rust/src/arrow/schemas.rs

OHLCV_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
])

INDICATOR_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("value", pa.float64()),
])

def bollinger_schema() -> pa.Schema:
    return pa.schema([
        ("timestamp", pa.int64()),
        ("upper", pa.float64()),
        ("middle", pa.float64()),
        ("lower", pa.float64()),
    ])

def macd_schema() -> pa.Schema:
    return pa.schema([
        ("timestamp", pa.int64()),
        ("macd_line", pa.float64()),
        ("signal_line", pa.float64()),
        ("histogram", pa.float64()),
    ])
```

### 5.6 Erwartete Performance-Verbesserung

| Operation | NumPy (Wave 1) | Arrow (Wave 1.1) | Verbesserung |
|-----------|----------------|------------------|--------------|
| OHLCV Transfer (100K Bars) | ~50ms (copy) | ~0.1ms (zero-copy) | **500x** |
| Indicator Return | ~5μs (copy) | ~0.1μs (zero-copy) | **50x** |
| Memory Usage | 2x (Python + Rust) | 1x (shared) | **-50%** |

---

## 6. Test-Strategie

### 6.1 Test-Pyramide

```
                    ┌─────────────────┐
                    │   Benchmark     │ ← Performance Regression Gate
                    │     Tests       │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Integration   │ ← Rust↔Python Parität
                    │     Tests       │
                    └────────┬────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                      │
    ┌─────┴─────┐                          ┌─────┴─────┐
    │   Array   │                          │   Batch   │
    │   API     │                          │   API     │
    │   Tests   │                          │   Tests   │
    └───────────┘                          └───────────┘
          │                                      │
          └──────────────────┬───────────────────┘
                             │
                    ┌────────┴────────┐
                    │     Arrow       │ ← Zero-Copy Validation
                    │   IPC Tests     │
                    └─────────────────┘
```

### 6.2 Test-Dateien

<!-- docs-lint:planned - Test files to be created during Wave 1.1 implementation -->
| Datei | Phase | Beschreibung |
|-------|-------|--------------|
| `tests/test_indicator_cache_array_api.py` | 1 | NumPy-Array-API Parität | <!-- docs-lint:planned -->
| `tests/test_indicator_cache_batch.py` | 2 | Batch-API Funktionalität | <!-- docs-lint:planned -->
| `tests/test_indicator_cache_arrow.py` | 3 | Arrow IPC Zero-Copy | <!-- docs-lint:planned -->
| `tests/benchmarks/test_bench_wave_1_1.py` | Alle | Performance Regression | <!-- docs-lint:planned -->

### 6.3 Benchmark-Suite

```python
# tests/benchmarks/test_bench_wave_1_1.py

import pytest
import numpy as np
from time import perf_counter

@pytest.fixture
def large_indicator_cache():
    """100K bars synthetic data."""
    from backtest_engine.core.indicator_cache import get_cached_indicator_cache
    # ... setup code ...
    return cache

class TestWave11Benchmarks:
    """Performance benchmarks for Wave 1.1 optimizations."""
    
    def test_phase1_array_api_speedup(self, large_indicator_cache):
        """Verify pandas wrapper elimination improves performance."""
        cache = large_indicator_cache
        
        # Measure pandas API
        start = perf_counter()
        for _ in range(100):
            _ = cache.bollinger("M5", "bid", 20)
        pandas_time = perf_counter() - start
        
        # Measure array API
        start = perf_counter()
        for _ in range(100):
            _ = cache.bollinger_array("M5", "bid", 20)
        array_time = perf_counter() - start
        
        speedup = pandas_time / array_time
        assert speedup >= 2.0, f"Expected ≥2x speedup, got {speedup:.2f}x"
    
    def test_phase2_batch_api_speedup(self, large_indicator_cache):
        """Verify batch API reduces FFI overhead."""
        cache = large_indicator_cache
        
        indicators = [
            {"name": "ema", "period": 14},
            {"name": "atr", "period": 14},
            {"name": "zscore", "window": 100},
            {"name": "bollinger", "period": 20},
            {"name": "macd"},
        ]
        
        # Measure individual calls
        start = perf_counter()
        for _ in range(100):
            cache.ema_array("M5", "bid", 14)
            cache.atr_array("M5", "bid", 14)
            cache.zscore_array("M5", "bid", 100)
            cache.bollinger_array("M5", "bid", 20)
            cache.macd_array("M5", "bid")
        individual_time = perf_counter() - start
        
        # Measure batch call
        start = perf_counter()
        for _ in range(100):
            cache.compute_batch(indicators, "M5", "bid")
        batch_time = perf_counter() - start
        
        speedup = individual_time / batch_time
        assert speedup >= 3.0, f"Expected ≥3x speedup with batch, got {speedup:.2f}x"
    
    def test_phase3_arrow_zero_copy(self, large_indicator_cache):
        """Verify Arrow IPC is true zero-copy."""
        cache = large_indicator_cache
        
        # This test verifies memory is shared, not copied
        import pyarrow as pa
        
        arrow_result = cache.ema_arrow("M5", "bid", 14)
        
        # Check that no copy was made
        assert arrow_result.buffers()[1].address != 0  # Data buffer exists
        # Additional zero-copy verification would check memory addresses
```

---

## 7. Validierung & Akzeptanzkriterien

### 7.1 Phase 1 Kriterien (NumPy-Array-API)

| Kriterium | Schwellwert | Status |
|-----------|-------------|--------|
| Bollinger Speedup | ≥2x | ⬜ |
| MACD Speedup | ≥2x | ⬜ |
| Z-Score Speedup | ≥3x | ⬜ |
| Numerische Parität | ≤1e-10 | ⬜ |
| Alle bestehenden Tests grün | 100% | ⬜ |

### 7.2 Phase 2 Kriterien (Batch-API)

| Kriterium | Schwellwert | Status |
|-----------|-------------|--------|
| FFI-Call-Reduktion | ≥80% | ⬜ |
| Batch vs Individual Speedup | ≥3x (5 Indikatoren) | ⬜ |
| Strategien kompilieren | 100% | ⬜ |
| Numerische Parität | ≤1e-10 | ⬜ |

### 7.3 Phase 3 Kriterien (Arrow IPC)

| Kriterium | Schwellwert | Status |
|-----------|-------------|--------|
| OHLCV Transfer Zero-Copy | Verifiziert | ⬜ |
| Indicator Return Zero-Copy | Verifiziert | ⬜ |
| Memory Usage Reduktion | ≥30% | ⬜ |
| Julia Kompatibilität | Basis-Test | ⬜ |

### 7.4 Gesamte Wave 1.1 Kriterien

| Kriterium | Schwellwert | Status |
|-----------|-------------|--------|
| Kein Indikator mit Regression (<1x) | 0 Regressionen | ⬜ |
| Durchschnittlicher Speedup | ≥5x | ⬜ |
| Backtest-Ergebnis-Parität | Identisch | ⬜ |
| CI/CD grün | 100% | ⬜ |

---

## 8. Rollback-Plan

### 8.1 Feature-Flag-basierter Rollback

```bash
# Phase 1 Rollback: Array-API deaktivieren
# (Strategien nutzen weiterhin pandas-API)
# Keine Env-Variable nötig - alte API bleibt verfügbar

# Phase 2 Rollback: Batch-API nicht nutzen
# Strategien rufen Indikatoren einzeln auf
# Keine Code-Änderung nötig

# Phase 3 Rollback: Arrow deaktivieren
export OMEGA_USE_ARROW_IPC=false
# oder in Cargo.toml:
# default = []  # Arrow-Feature deaktivieren
```

### 8.2 Rollback-Trigger

| Trigger | Phase | Aktion |
|---------|-------|--------|
| Numerische Differenz >1e-10 | Alle | Sofort-Rollback |
| Performance-Regression >10% | Alle | Analyse → ggf. Rollback |
| Memory Leak | 3 | Arrow-Feature deaktivieren |
| Segmentation Fault | 3 | Arrow-Feature deaktivieren |
| Backtest-Ergebnis-Differenz | Alle | Sofort-Rollback |

### 8.3 Rollback-Kompatibilität

**Garantien:**
- Alle neuen APIs sind **additiv** (alte APIs bleiben)
- Strategien mit `*_array()` kompilieren auch ohne Rust
- `compute_batch()` fällt auf Einzel-Calls zurück
- Arrow IPC ist optional (Feature-Flag)

---

## 9. Lessons Learned aus Wave 1

### 9.1 FFI-Overhead-Dominanz

**Erkenntnis:** Bei einfachen Indikatoren (Bollinger, MACD) dominiert der FFI-Overhead:
- Rust-Berechnung: ~1-3ms
- `_series_from_rust_array()`: ~200μs
- pandas.Series(): ~100-300μs
- **Gesamter Wrapper-Overhead: ~400-600μs**

**Lösung Wave 1.1:** pandas-Wrapper eliminieren, Batch-API

### 9.2 Caching-Effektivität

**Erkenntnis:** Python-seitiges Caching ist sehr effektiv:
- First Call: ~5-50ms
- Cached Call: ~3-10μs
- **Caching speedup: 1000x+**

**Implikation:** Der FFI-Overhead ist nur beim First Call relevant.
Wave 1.1 fokussiert auf First-Call-Optimierung.

### 9.3 Komplexe Indikatoren profitieren maximal

**Erkenntnis:** Je komplexer der Indikator, desto mehr Speedup:
- Kalman Mean: 528x
- Kalman GARCH ZScore: 182x
- ATR: 79x
- GARCH Volatility: 67x

**Implikation:** Batch-API sollte komplexe Indikatoren priorisieren.

### 9.4 NumPy-Interop ist fast Zero-Copy

**Erkenntnis:** `result.to_pyarray(py)` in PyO3 ist bereits effizient:
- Nutzt NumPy C-API
- Erstellt View auf Rust-Speicher

**Implikation:** Arrow IPC bietet nur marginalen Vorteil für Indicator-Return.
Hauptvorteil: OHLCV-Transfer + Julia-Kompatibilität.

---

## 10. Checklisten

### 10.1 Pre-Implementation Checklist

- [ ] Wave 1 Benchmark-Baseline dokumentiert
- [ ] FFI-Overhead-Ursachen analysiert und verstanden
- [ ] Rust-Build funktioniert (`cargo build`)
- [ ] pytest-Suite grün
- [ ] Strategie-Inventar für Migration erstellt

### 10.2 Phase 1 Checklist (NumPy-Array-API)

#### Implementation
- [ ] `*_array()` Methoden für alle 24 Indikatoren hinzufügen
- [ ] Docstrings und Type Hints vollständig
- [ ] Python-Fallback für alle Methoden

#### Tests
- [ ] `test_indicator_cache_array_api.py` erstellt
- [ ] Numerische Parität für alle Indikatoren verifiziert
- [ ] Benchmark zeigt ≥2x Speedup für Bollinger/MACD

#### Migration
- [ ] `mean_reversion_z_score` auf Array-API migriert
- [ ] Template-Strategie mit Array-API Beispiel aktualisiert

### 10.3 Phase 2 Checklist (Batch-API)

#### Rust-Implementation
- [ ] `batch.rs` erstellt
- [ ] `compute_indicator_batch()` implementiert
- [ ] PyO3 Binding hinzugefügt
- [ ] `cargo test` grün
- [ ] `cargo clippy` grün

#### Python-Integration
- [ ] `compute_batch()` Methode hinzugefügt
- [ ] Python-Fallback implementiert
- [ ] Docstrings und Type Hints vollständig

#### Tests
- [ ] `test_indicator_cache_batch.py` erstellt
- [ ] Alle 24 Indikatoren in Batch-API unterstützt
- [ ] Benchmark zeigt ≥3x Speedup für 5 Indikatoren

### 10.4 Phase 3 Checklist (Arrow IPC)

#### Rust-Implementation
- [ ] Arrow-Feature in Cargo.toml aktiviert
- [ ] `arrow/` Modul erstellt
- [ ] Schema-Definitionen synchron mit Python
- [ ] `register_ohlcv_arrow()` implementiert
- [ ] `*_arrow()` Export-Methoden implementiert
- [ ] `cargo test --features arrow` grün

#### Python-Integration
- [ ] `_init_rust_cache_arrow()` implementiert
- [ ] `*_arrow()` Methoden hinzugefügt
- [ ] Schema-Synchronisation validiert

#### Tests
- [ ] `test_indicator_cache_arrow.py` erstellt
- [ ] Zero-Copy verifiziert (Memory-Adressen)
- [ ] Julia-Basis-Kompatibilität getestet

### 10.5 Post-Implementation Checklist

- [ ] Alle Benchmarks dokumentiert in `reports/`
- [ ] CHANGELOG.md Eintrag erstellt
- [ ] architecture.md aktualisiert
- [ ] Strategie-Migration-Guide erstellt
- [ ] Performance-Verbesserungen quantifiziert

### 10.6 Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Developer | | | ⬜ |
| Phase 1 Tests | | | ⬜ |
| Phase 2 Tests | | | ⬜ |
| Phase 3 Tests | | | ⬜ |
| Benchmark Validation | | | ⬜ |
| Tech Lead | | | ⬜ |

---

## 11. Zeitplan

| Phase | Aufwand | Abhängigkeiten | Deadline |
|-------|---------|----------------|----------|
| Phase 1: NumPy-Array-API | 1-2 Tage | Wave 1 abgeschlossen | +2 Tage |
| Phase 2: Batch-API | 2-3 Tage | Phase 1 | +5 Tage |
| Phase 3: Arrow IPC | 1-2 Wochen | Phase 2 | +2 Wochen |
| **Gesamt** | **2-3 Wochen** | | |

---

## 12. References

- [Wave 1: IndicatorCache Rust Migration](./WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md)
- [ADR-0002: Serialization Format (Arrow)](./adr/ADR-0002-serialization-format.md)
- [FFI Boundaries Instructions](../.github/instructions/ffi-boundaries.instructions.md)
- [Rust Indicator Analysis Report](../reports/rust_indicator_analysis_report.md)
- [Performance Baseline: IndicatorCache](../reports/performance_baselines/p0-01_indicator_cache.json)

---

## Änderungshistorie

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-09 | 1.0 | Initiale Version | AI Agent |

---

*Document Status: 📋 READY FOR IMPLEMENTATION*
