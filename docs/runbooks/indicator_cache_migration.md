# Migration Runbook: IndicatorCache

**Python-Pfad:** `src/backtest_engine/core/indicator_cache.py`  
**Zielsprache:** Rust  
**FFI-Integration:** PyO3/Maturin  
**Priorität:** High  
**Geschätzter Aufwand:** L (Large)  
**Status:** 🔴 Nicht begonnen

---

## Executive Summary

`IndicatorCache` ist der zentrale Cache für technische Indikatoren (EMA, SMA, RSI, MACD, Bollinger, ATR, DMI, ROC).
Das Modul wird in jedem Backtest-Tick aufgerufen und ist ein Performance-Hotspot.
Die Migration zu Rust soll einen **5-10x Speedup** für Indikator-Berechnungen erreichen, insbesondere für:
- `atr()` (aktuell ~1s für 50k Bars)
- `ema_stepwise()` / `bollinger_stepwise()` (aktuell ~50-90ms)
- `dmi()` (aktuell ~65ms)

---

## Vorbedingungen

### Typ-Sicherheit
- [x] Modul ist mypy --strict compliant
- [x] Alle öffentlichen Funktionen haben vollständige Type Hints
- [x] TypedDict/Protocol-Definitionen in `src/backtest_engine/core/types.py`

### Interface-Dokumentation
- [x] FFI-Spezifikation in `docs/ffi/indicator_cache.md`
- [x] Arrow-Schemas definiert in `src/shared/arrow_schemas.py`
- [x] Nullability-Konvention dokumentiert

### Test-Infrastruktur
- [x] Benchmark-Suite in `tests/benchmarks/test_bench_indicator_cache.py`
- [x] Property-Based Tests in `tests/property/test_property_indicators.py`
- [x] Golden-File Tests in `tests/golden/test_golden_backtest.py`
- [x] Test-Coverage ≥ 85%

### Performance-Baselines
- [x] Baseline in `reports/performance_baselines/p0-01_indicator_cache.json`
- [x] Improvement-Targets definiert (siehe unten)

---

## Performance-Baseline

**Quelle:** `reports/performance_baselines/p0-01_indicator_cache.json`  
**Test-Parameter:** 50.000 Bars, 3 Wiederholungen

| Operation | First Call (ms) | Cached Call (µs) | Peak Memory (MB) | Target Speedup |
|-----------|-----------------|------------------|------------------|----------------|
| `__init__` | 187.4 | - | 6.01 | 2x |
| `ema` | 1.25 | 3 | 1.21 | 10x |
| `ema_stepwise` | 51.1 | 6 | 3.92 | 20x |
| `sma` | 1.35 | 3 | 1.21 | 10x |
| `rsi` | 6.88 | 5 | 3.22 | 10x |
| `macd` | 2.70 | 6 | 2.41 | 10x |
| `roc` | 1.11 | 3 | 1.20 | 10x |
| `dmi` | 65.2 | 6 | 6.57 | 20x |
| `bollinger` | 3.69 | 4 | 2.05 | 10x |
| `bollinger_stepwise` | 88.5 | 10 | 5.90 | 20x |
| `atr` | 954.4 | 4 | 4.16 | **50x** |

**Primäre Optimierungs-Ziele:** `atr`, `ema_stepwise`, `bollinger_stepwise`, `dmi`

---

## Architektur-Übersicht

### Aktueller Python-Flow

```
                                     ┌─────────────────────────┐
                                     │  AlignedMultiCandleData │
                                     │  (Python Dict)          │
                                     └───────────┬─────────────┘
                                                 │
                                     ┌───────────▼─────────────┐
                                     │   IndicatorCache        │
                                     │   - _df_cache (Dict)    │
                                     │   - _indicator_cache    │
                                     └───────────┬─────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          │                      │                      │
               ┌──────────▼───────┐   ┌──────────▼───────┐   ┌──────────▼───────┐
               │     ema()        │   │     rsi()        │   │     atr()        │
               │  (pd.ewm)        │   │  (ta.rsi)        │   │  (loop + ewm)    │
               └──────────────────┘   └──────────────────┘   └──────────────────┘
```

### Geplanter Rust-Flow

```
                                     ┌─────────────────────────┐
                                     │  Arrow RecordBatch      │
                                     │  (Zero-Copy from numpy) │
                                     └───────────┬─────────────┘
                                                 │
                                     ┌───────────▼─────────────┐
                                     │   RustIndicatorCache    │
                                     │   (PyO3 Extension)      │
                                     │   - cache: HashMap      │
                                     │   - parallel: Rayon     │
                                     └───────────┬─────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          │                      │                      │
               ┌──────────▼───────┐   ┌──────────▼───────┐   ┌──────────▼───────┐
               │  ema_rust()      │   │  rsi_rust()      │   │  atr_rust()      │
               │  (SIMD/ndarray)  │   │  (SIMD/ndarray)  │   │  (SIMD/ndarray)  │
               └──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## Migration Steps

### Step 1: Rust Modul Setup

```bash
# Bereits vorhanden in src/rust_modules/omega_rust/
cd src/rust_modules/omega_rust

# Neues Indikator-Modul erstellen
mkdir -p src/indicators
touch src/indicators/mod.rs
touch src/indicators/ema.rs
touch src/indicators/rsi.rs
touch src/indicators/macd.rs
touch src/indicators/bollinger.rs
touch src/indicators/atr.rs
touch src/indicators/dmi.rs
```

- [x] Cargo.toml Dependencies (ndarray, rayon, numpy)
- [x] lib.rs Module-Deklaration
- [x] indicators/mod.rs Exports

### Step 2: Interface Implementation

**Input-Typen (Arrow → Rust):**

```rust
// src/rust_modules/omega_rust/src/indicators/types.rs

use ndarray::Array1;

/// OHLCV-Daten als separate Arrays (columnar)
pub struct OhlcvData {
    pub open: Array1<f64>,
    pub high: Array1<f64>,
    pub low: Array1<f64>,
    pub close: Array1<f64>,
    pub volume: Array1<f64>,
    /// Validity mask für None-Candles
    pub valid: Array1<bool>,
}

/// Cache-Key für Indikator-Lookup
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct CacheKey {
    pub timeframe: String,
    pub price_type: String,  // "bid" | "ask"
    pub indicator: String,
    pub params: String,      // JSON-serialisierte Parameter
}
```

**Output-Typen (Rust → Python):**

```rust
/// Indikator-Ergebnis als NumPy-kompatibles Array
pub type IndicatorResult = Array1<f64>;

/// Mehrere Outputs (z.B. Bollinger: upper, middle, lower)
pub struct BollingerResult {
    pub upper: Array1<f64>,
    pub middle: Array1<f64>,
    pub lower: Array1<f64>,
}

/// MACD mit Signal und Histogram
pub struct MacdResult {
    pub macd: Array1<f64>,
    pub signal: Array1<f64>,
    pub histogram: Array1<f64>,
}
```

- [ ] Arrow Schema Validation
- [ ] Nullability Handling (NaN propagation)
- [ ] Type Conversion Tests

### Step 3: Core-Logik portieren

#### EMA (Exponential Moving Average)

```rust
// src/rust_modules/omega_rust/src/indicators/ema.rs

use ndarray::Array1;

/// Berechnet EMA mit Span (wie pandas.ewm)
/// 
/// alpha = 2 / (span + 1)
/// ema[0] = close[0]
/// ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
pub fn ema(close: &Array1<f64>, span: usize) -> Array1<f64> {
    let n = close.len();
    let mut result = Array1::zeros(n);
    
    if n == 0 {
        return result;
    }
    
    let alpha = 2.0 / (span as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    
    result[0] = close[0];
    
    for i in 1..n {
        result[i] = alpha * close[i] + one_minus_alpha * result[i - 1];
    }
    
    result
}
```

- [ ] EMA Implementation
- [ ] SMA Implementation
- [ ] RSI Implementation
- [ ] MACD Implementation (EMA-basiert)
- [ ] Bollinger Bands Implementation
- [ ] ATR Implementation (True Range + EMA)
- [ ] DMI Implementation (DI+, DI-, ADX)
- [ ] ROC Implementation

### Step 4: FFI-Bindings

```rust
// src/rust_modules/omega_rust/src/lib.rs

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod indicators;
use indicators::ema::ema;

#[pyfunction]
fn calc_ema(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    span: usize
) -> PyResult<Py<PyArray1<f64>>> {
    let close_arr = close.as_array().to_owned();
    let result = ema(&close_arr, span);
    Ok(PyArray1::from_array(py, &result).into())
}

#[pymodule]
fn omega_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_ema, m)?)?;
    // ... weitere Funktionen
    Ok(())
}
```

- [ ] PyO3 Bindings für alle Indikatoren
- [ ] Error Handling (OmegaError → PyErr)
- [ ] GIL Release für lange Berechnungen
- [ ] NumPy Array Interop

### Step 5: Python-Wrapper

```python
# src/backtest_engine/core/indicator_cache.py

from typing import Optional
import numpy as np
from numpy.typing import NDArray

# Feature Flag
USE_RUST_INDICATORS = True

def _try_import_rust():
    """Lazy Import für Rust-Extension."""
    try:
        from omega_rust import calc_ema, calc_rsi, calc_atr
        return calc_ema, calc_rsi, calc_atr
    except ImportError:
        return None, None, None

_rust_ema, _rust_rsi, _rust_atr = _try_import_rust()


class IndicatorCache:
    """Indikator-Cache mit optionalem Rust-Backend."""
    
    def ema(
        self,
        tf: str,
        pt: str,
        source: str,
        period: int,
        i: Optional[int] = None
    ) -> NDArray[np.float64]:
        cache_key = (tf, pt, "ema", source, period)
        
        if cache_key in self._indicator_cache:
            result = self._indicator_cache[cache_key]
        else:
            close = self._get_series(tf, pt, source)
            
            if USE_RUST_INDICATORS and _rust_ema is not None:
                # Rust-Backend
                result = _rust_ema(close, period)
            else:
                # Python-Fallback
                result = self._ema_python(close, period)
            
            self._indicator_cache[cache_key] = result
        
        if i is not None:
            return result[i] if i < len(result) else np.nan
        return result
```

- [ ] Feature-Flag Implementation
- [ ] Lazy Import mit Fallback
- [ ] Cache-Key Kompatibilität
- [ ] Typ-Signatur-Kompatibilität

### Step 6: Testing

```bash
# Unit Tests
pytest tests/test_indicator_cache_pool_cleanup.py -v

# Property-Based Tests
pytest tests/property/test_prop_indicators.py -v --hypothesis-show-statistics

# Golden-File Tests (Determinismus)
pytest tests/golden/test_golden_backtest.py -v -k indicator

# Benchmarks
pytest tests/benchmarks/test_bench_indicator_cache.py --benchmark-json=results.json

# Rust Unit Tests
cd src/rust_modules/omega_rust
cargo test

# Rust Benchmarks
cargo bench
```

- [ ] Alle Unit-Tests passieren
- [ ] Property-Based Tests passieren
- [ ] Golden-File Tests passieren (Determinismus!)
- [ ] Benchmark zeigt ≥5x Speedup
- [ ] Rust `cargo test` passiert
- [ ] Integration mit Backtest-Engine

### Step 7: Documentation

- [ ] Docstrings in Python aktualisiert
- [ ] Rustdoc für Rust-Modul
- [ ] FFI-Dokumentation in `docs/ffi/indicator_cache.md` aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] README.md Performance-Zahlen aktualisiert

---

## Rollback-Plan

### Bei Fehler in Produktion

1. **Sofortmaßnahme:** Feature-Flag deaktivieren
   ```python
   # src/backtest_engine/core/indicator_cache.py
   USE_RUST_INDICATORS = False
   ```

2. **Fallback:** Python-Implementation wird automatisch verwendet

3. **Analyse:**
   - Logs prüfen auf `ImportError` oder `RuntimeError`
   - Edge-Case identifizieren (NaN? Inf? Leere Arrays?)
   - Issue erstellen mit Reproduktions-Script

4. **Fix:**
   - Bugfix in Rust (`src/rust_modules/omega_rust/`)
   - Property-Test erweitern für Edge-Case
   - Golden-File updaten falls legitime Änderung

### Bei Performance-Regression

1. Benchmark-History prüfen:
   ```bash
   python tools/benchmark_history.py compare --module indicator_cache
   ```

2. Profiling:
   ```bash
   # Rust Flamegraph
   cd src/rust_modules/omega_rust
   cargo flamegraph --bench indicator_bench
   ```

3. Bei > 10% Regression: Rollback zu Python

---

## Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Numerische Abweichungen durch Float-Rounding | Mittel | Hoch | Property-Tests mit Toleranzen; Golden-Files |
| NaN-Propagation unterschiedlich | Niedrig | Hoch | Explizite NaN-Handling-Tests |
| Memory Layout Inkompatibilität | Niedrig | Mittel | Arrow IPC für Transfer; Benchmarks |
| GIL-Contention bei Multi-Threading | Mittel | Mittel | GIL-Release in langen Berechnungen |
| Backtest-Determinismus-Bruch | Niedrig | Kritisch | Golden-File Tests; Seed-Propagation |

---

## Akzeptanzkriterien

### Funktional
- [ ] Alle 242 bestehenden Tests passieren
- [ ] Keine Regression in Backtest-Determinismus (Golden-Files)
- [ ] Output-Format identisch mit Python-Version (dtype, shape)
- [ ] NaN-Handling identisch

### Performance
- [ ] EMA: ≥10x Speedup
- [ ] RSI: ≥10x Speedup
- [ ] ATR: ≥50x Speedup
- [ ] Memory-Usage ≤ Python-Baseline
- [ ] Keine Memory-Leaks (miri clean)

### Qualität
- [ ] Code Review bestanden
- [ ] mypy --strict für Python-Wrapper
- [ ] clippy --pedantic für Rust (0 Warnings)
- [ ] Dokumentation vollständig
- [ ] Benchmark-History etabliert

---

## Referenzen

- FFI-Spezifikation: `docs/ffi/indicator_cache.md`
- Performance-Baseline: `reports/performance_baselines/p0-01_indicator_cache.json`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
- Rust-Modul: `src/rust_modules/omega_rust/src/indicators/`
- ADR-0001: Migration Strategy
- ADR-0002: Serialization Format
- ADR-0003: Error Handling
- ADR-0004: Build System

---

## Changelog

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-05 | 1.0 | Initiale Version des Runbooks | Omega Team |
