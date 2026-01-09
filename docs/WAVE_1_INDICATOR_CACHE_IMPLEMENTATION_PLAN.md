# Wave 1: IndicatorCache Rust Migration Implementation Plan

**Document Version:** 1.0  
**Created:** 2026-01-09  
**Updated:** 2026-01-09  
**Status:** 🔴 NOT STARTED  
**Module:** `src/backtest_engine/core/indicator_cache.py`

---

## Executive Summary

Dieser Plan beschreibt die vollständige Migration des `IndicatorCache`-Moduls zu Rust (PyO3/maturin) als **Wave 1** der Rust/Julia-Migrationsstrategie. Das Modul ist ein High-Performance Indikator-Cache für aligned Multi-Timeframe OHLCV-Daten und wird in jedem Backtest-Tick aufgerufen – ein klar identifizierter Performance-Hotspot.

**Julia ist NICHT Teil dieser Wave** – der Fokus liegt ausschließlich auf der Rust-Migration.

### Warum IndicatorCache als Wave 1?

| Eigenschaft | Bewertung | Begründung |
|-------------|-----------|------------|
| **Numerisch Intensiv** | ✅ Kritisch | EMA, RSI, MACD, Bollinger, ATR, DMI – alle berechnen aufwendige Float-Operationen |
| **Aufruffrequenz** | ✅ Sehr Hoch | Wird bei jedem Backtest-Tick aufgerufen |
| **SIMD-Potenzial** | ✅ Hervorragend | Vektorisierte Indikatoren ideal für Rust + SIMD |
| **Cachability** | ✅ Hoch | Deterministische Berechnungen, Cache-freundlich |
| **Isolierte Logik** | ✅ Gut | Klare Input/Output-Grenzen (OHLCV → Indicator-Serien) |
| **Testbarkeit** | ✅ Gut | Property-Based Tests, Benchmarks, Golden-Files vorhanden |
| **Risiko** | ⚠️ Mittel | NaN-Propagation und Float-Determinismus kritisch |
| **Geschätzter Aufwand** | ⚠️ 8-10 Tage | Umfangreiche Indikator-Bibliothek |

### Performance-Targets (aus `p0-01_indicator_cache.json`)

| Operation | Python Baseline (First Call) | Rust Target | Target Speedup |
|-----------|------------------------------|-------------|----------------|
| `atr` | 954ms | ≤19ms | **50x** |
| `ema_stepwise` | 51ms | ≤2.5ms | **20x** |
| `bollinger_stepwise` | 88ms | ≤4.4ms | **20x** |
| `dmi` | 65ms | ≤3.3ms | **20x** |
| `ema` | 1.25ms | ≤0.125ms | 10x |
| `rsi` | 6.9ms | ≤0.69ms | 10x |
| `macd` | 2.7ms | ≤0.27ms | 10x |
| `bollinger` | 3.7ms | ≤0.37ms | 10x |

---

## Inhaltsverzeichnis

1. [Voraussetzungen & Status](#1-voraussetzungen--status)
2. [Architektur-Übersicht](#2-architektur-übersicht)
3. [Implementierungs-Phasen](#3-implementierungs-phasen)
4. [Rust-Implementation](#4-rust-implementation)
5. [Python-Integration](#5-python-integration)
6. [Test-Strategie](#6-test-strategie)
7. [Validierung & Akzeptanzkriterien](#7-validierung--akzeptanzkriterien)
8. [Rollback-Plan](#8-rollback-plan)
9. [Lessons Learned aus Wave 0 & 2](#9-lessons-learned-aus-wave-0--2)
10. [Checklisten](#10-checklisten)

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure-Readiness (aus Wave 0 & 2 etabliert)

| Komponente | Status | Evidenz |
|------------|--------|---------|
| Rust Build System | ✅ | `src/rust_modules/omega_rust/Cargo.toml` |
| PyO3/Maturin | ✅ | Version 0.27 konfiguriert |
| Error Handling | ✅ | `src/rust_modules/omega_rust/src/error.rs` |
| FFI-Spezifikation | ✅ | `docs/ffi/indicator_cache.md` |
| Migration Runbook | ✅ | `docs/runbooks/indicator_cache_migration.md` |
| mypy strict | ✅ | `backtest_engine.core.*` strict-compliant |
| Benchmarks | ✅ | `tests/benchmarks/test_bench_indicator_cache.py` |
| Performance Baseline | ✅ | `reports/performance_baselines/p0-01_indicator_cache.json` |
| Arrow Schemas | ✅ | `src/shared/arrow_schemas.py` (INDICATOR_SCHEMA) |

**Referenz:** `docs/MIGRATION_READINESS_VALIDATION.md` – Status: ✅ APPROVED FOR PILOT

### 1.2 Python-Modul Baseline

**Datei:** `src/backtest_engine/core/indicator_cache.py` (~1136 LOC)

Die aktuelle Python-Implementation enthält:

**Core-Klasse `IndicatorCache`:**
- `__init__()`: Initialisierung mit `multi_candle_data`, DataFrame-Erstellung
- `_ensure_df()`: Lazy DataFrame construction (OHLCV)
- `get_df()`: OHLCV-DataFrame Accessor
- `get_closes()`: Close-Serie Accessor

**Indikator-APIs (vektorisiert + gecached):**
- `ema()` / `ema_stepwise()`: Exponential Moving Average
- `sma()`: Simple Moving Average
- `rsi()`: Relative Strength Index (Wilder)
- `macd()`: MACD Line + Signal
- `roc()`: Rate of Change
- `dmi()`: Directional Movement Index (+DI, -DI, ADX)
- `bollinger()` / `bollinger_stepwise()`: Bollinger Bands
- `atr()`: Average True Range (Wilder)
- `choppiness()`: Choppiness Index
- `zscore()`: Z-Score (rolling/EMA)
- `kalman_mean()` / `kalman_zscore()` / `kalman_zscore_stepwise()`: Kalman-Filter
- `kalman_garch_zscore()`: Kalman-GARCH Z-Score

**Kritische Invarianten:**
- NaN-Propagation bei fehlenden Bars
- Deterministische Floating-Point-Berechnungen
- Cache-Key-basiertes Caching (Tuple-Keys)
- HTF-Bar Stepwise-Semantik (verhindert carry_forward Drift)

### 1.3 Performance-Baseline (aus `p0-01_indicator_cache.json`)

**Test-Parameter:** 50.000 Bars, 3 Wiederholungen

```json
{
  "meta": {
    "num_bars": 50000,
    "repetitions": 3,
    "generated_at": "2026-01-03T21:37:37Z"
  },
  "init_seconds": 0.187437,
  "init_peak_mb": 6.01048,
  "operations": {
    "atr": { "first_call_seconds": 0.954385, "cached_call_seconds": 4e-06 },
    "ema_stepwise": { "first_call_seconds": 0.051055, "cached_call_seconds": 6e-06 },
    "bollinger_stepwise": { "first_call_seconds": 0.088489, "cached_call_seconds": 1e-05 },
    "dmi": { "first_call_seconds": 0.065167, "cached_call_seconds": 6e-06 },
    "ema": { "first_call_seconds": 0.001253, "cached_call_seconds": 3e-06 },
    "rsi": { "first_call_seconds": 0.006878, "cached_call_seconds": 5e-06 },
    "macd": { "first_call_seconds": 0.0027, "cached_call_seconds": 6e-06 },
    "bollinger": { "first_call_seconds": 0.003689, "cached_call_seconds": 4e-06 }
  }
}
```

**Profiling-Hotspots (Top 5):**
1. `_ensure_df()` (DataFrame-Erstellung): 212ms
2. `atr()` (Wilder-Loop): 80ms
3. `bollinger_stepwise()`: 38ms
4. `ema_stepwise()`: 28ms
5. `dmi()`: 20ms

---

## 2. Architektur-Übersicht

### 2.1 Ziel-Architektur

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKTEST ENGINE                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │     Python API Layer (src/backtest_engine/core/indicator_cache.py)         │ │
│  │                                                                            │ │
│  │  class IndicatorCache:                                                     │ │
│  │      def __init__(self, multi_candle_data: AlignedMultiCandleData):        │ │
│  │          if USE_RUST_INDICATOR_CACHE:                                      │ │
│  │              self._rust = IndicatorCacheRust(...)   ◄── Rust               │ │
│  │          else:                                                             │ │
│  │              self._rust = None                      ◄── Pure Python        │ │
│  │                                                                            │ │
│  │      def atr(self, tf, price_type, period) -> pd.Series:                   │ │
│  │          if self._rust:                                                    │ │
│  │              return self._rust.atr(tf, price_type, period)                 │ │
│  │          else:                                                             │ │
│  │              return self._atr_python(tf, price_type, period)               │ │
│  │                                                                            │ │
│  │      # ... weitere Indikatoren mit Rust/Python-Delegation                  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              │ FFI Boundary (PyO3 + NumPy Interop)               │
│                              ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │       Rust Layer (src/rust_modules/omega_rust/src/indicators/)             │ │
│  │                                                                            │ │
│  │  #[pyclass]                                                                │ │
│  │  pub struct IndicatorCacheRust {                                           │ │
│  │      ohlcv_data: HashMap<(String, String), OhlcvData>,  // (tf, side)      │ │
│  │      ind_cache: HashMap<CacheKey, IndicatorResult>,     // (name, params)  │ │
│  │  }                                                                         │ │
│  │                                                                            │ │
│  │  #[pymethods]                                                              │ │
│  │  impl IndicatorCacheRust {                                                 │ │
│  │      fn atr(&self, tf: &str, pt: &str, period: usize) -> PyResult<..>;     │ │
│  │      fn ema(&self, tf: &str, pt: &str, period: usize) -> PyResult<..>;     │ │
│  │      fn rsi(&self, tf: &str, pt: &str, period: usize) -> PyResult<..>;     │ │
│  │      fn dmi(&self, tf: &str, pt: &str, period: usize) -> PyResult<..>;     │ │
│  │      // ... SIMD-optimierte Varianten                                      │ │
│  │  }                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Feature-Flag-System (analog zu Wave 0 & 2)

```python
# src/backtest_engine/core/indicator_cache.py

import os
from typing import Any, Optional

_RUST_AVAILABLE: bool = False
_RUST_MODULE: Any = None

def _check_rust_indicator_cache_available() -> bool:
    """Check if Rust IndicatorCache module is available and functional."""
    global _RUST_MODULE
    try:
        import omega_rust
        if hasattr(omega_rust, "IndicatorCacheRust"):
            _RUST_MODULE = omega_rust
            return True
    except ImportError:
        pass
    return False

def _should_use_rust_indicator_cache() -> bool:
    """Determine if Rust implementation should be used."""
    env_val = os.environ.get("OMEGA_USE_RUST_INDICATOR_CACHE", "auto").lower()
    if env_val == "false":
        return False
    if env_val == "true":
        return _RUST_AVAILABLE
    # auto: use Rust if available
    return _RUST_AVAILABLE

# Initialize on module load
_RUST_AVAILABLE = _check_rust_indicator_cache_available()
USE_RUST_INDICATOR_CACHE = _should_use_rust_indicator_cache()
```

### 2.3 Datei-Struktur nach Migration

```
src/
├── rust_modules/
│   └── omega_rust/
│       ├── src/
│       │   ├── lib.rs                    # Modul-Registration erweitern
│       │   ├── error.rs                  # Bestehendes Error-Handling
│       │   ├── costs/                    # Wave 0: Slippage & Fee
│       │   ├── portfolio/                # Wave 2: Portfolio
│       │   └── indicators/               # WAVE 1: Indicator-Module
│       │       ├── mod.rs                # NEU: Module exports
│       │       ├── types.rs              # NEU: OhlcvData, CacheKey
│       │       ├── cache.rs              # NEU: IndicatorCacheRust class
│       │       ├── ema.rs                # NEU: EMA + EMA Stepwise
│       │       ├── sma.rs                # NEU: SMA
│       │       ├── rsi.rs                # NEU: RSI (Wilder)
│       │       ├── macd.rs               # NEU: MACD
│       │       ├── bollinger.rs          # NEU: Bollinger + Stepwise
│       │       ├── atr.rs                # NEU: ATR (Wilder) ← Kritisch!
│       │       ├── dmi.rs                # NEU: DMI (+DI, -DI, ADX)
│       │       ├── roc.rs                # NEU: Rate of Change
│       │       ├── zscore.rs             # NEU: Z-Score Varianten
│       │       ├── kalman.rs             # NEU: Kalman-Filter
│       │       └── choppiness.rs         # NEU: Choppiness Index
│       └── Cargo.toml                    # ndarray, rayon Dependencies
│
├── backtest_engine/
│   └── core/
│       └── indicator_cache.py            # Erweitert mit Rust-Integration
│
└── shared/
    └── arrow_schemas.py                  # OHLCV_SCHEMA, INDICATOR_SCHEMA

tests/
├── golden/
│   ├── test_golden_indicator_cache.py    # NEU: Golden-Tests für Indikatoren
│   └── reference/
│       └── indicators/
│           └── indicator_cache_v1.json   # NEU: Golden-Reference
├── benchmarks/
│   └── test_bench_indicator_cache.py     # Bestehend, erweitern für Rust
├── property/
│   └── test_prop_indicators.py           # Bestehend, Property-Based Tests
└── integration/
    └── test_indicator_cache_rust_parity.py  # NEU: Rust↔Python Parität
```

---

## 3. Implementierungs-Phasen

### Phase 1: Rust-Modul Setup (Tag 1-2, ~8h)

#### 3.1.1 Verzeichnisstruktur erstellen

```bash
# Erweitern des bestehenden indicators/ Verzeichnisses
mkdir -p src/rust_modules/omega_rust/src/indicators

# Core-Dateien
touch src/rust_modules/omega_rust/src/indicators/mod.rs
touch src/rust_modules/omega_rust/src/indicators/types.rs
touch src/rust_modules/omega_rust/src/indicators/cache.rs

# Indikator-Implementierungen
touch src/rust_modules/omega_rust/src/indicators/ema.rs
touch src/rust_modules/omega_rust/src/indicators/sma.rs
touch src/rust_modules/omega_rust/src/indicators/rsi.rs
touch src/rust_modules/omega_rust/src/indicators/macd.rs
touch src/rust_modules/omega_rust/src/indicators/bollinger.rs
touch src/rust_modules/omega_rust/src/indicators/atr.rs
touch src/rust_modules/omega_rust/src/indicators/dmi.rs
touch src/rust_modules/omega_rust/src/indicators/roc.rs
touch src/rust_modules/omega_rust/src/indicators/zscore.rs
touch src/rust_modules/omega_rust/src/indicators/kalman.rs
touch src/rust_modules/omega_rust/src/indicators/choppiness.rs
```

#### 3.1.2 Cargo.toml aktualisieren

```toml
# Hinzufügen zu [dependencies]
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"           # Parallel Iteration
numpy = "0.22"           # NumPy Interop für PyO3

# Optional für SIMD (später)
# packed_simd_2 = "0.3"  # Experimentell
```

#### 3.1.3 Module registrieren in lib.rs

```rust
pub mod indicators;  // NEU

use indicators::IndicatorCacheRust;

#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bestehende Module (Wave 0, Wave 2)...
    
    // NEU: IndicatorCache Class
    m.add_class::<IndicatorCacheRust>()?;
    
    Ok(())
}
```

### Phase 2: Core Rust Structures (Tag 2-3, ~12h)

#### 3.2.1 Type Definitions

**Datei:** `src/rust_modules/omega_rust/src/indicators/types.rs`

```rust
use ndarray::Array1;
use std::collections::HashMap;

/// OHLCV-Daten als columnar Arrays (analog zur FFI-Spec)
pub struct OhlcvData {
    pub open: Array1<f64>,
    pub high: Array1<f64>,
    pub low: Array1<f64>,
    pub close: Array1<f64>,
    pub volume: Array1<f64>,
    /// Validity mask für None-Candles (true = valid)
    pub valid: Array1<bool>,
    pub n_bars: usize,
}

/// Cache-Key für Indikator-Lookup (Hashable)
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct CacheKey {
    pub indicator: String,
    pub timeframe: String,
    pub price_type: String,
    pub params: String,  // JSON-serialisierte Parameter
}

/// Indikator-Ergebnis Varianten
pub enum IndicatorResult {
    Single(Array1<f64>),
    Pair(Array1<f64>, Array1<f64>),               // MACD
    Triple(Array1<f64>, Array1<f64>, Array1<f64>), // Bollinger, DMI
}
```

#### 3.2.2 Cache Implementation

**Datei:** `src/rust_modules/omega_rust/src/indicators/cache.rs`

Kernaufgaben:
- Lazy OHLCV-DataFrame Erstellung aus Python-Input
- Cache-Management mit `HashMap<CacheKey, IndicatorResult>`
- NumPy Array Output via `PyArray1`

### Phase 3: Indikator-Implementation (Tag 3-6, ~24h)

#### Priorität 1: ATR (50x Speedup Target)

**Kritisch:** Die ATR-Implementierung (Wilder) ist ein Hotspot mit ~1s Laufzeit.

```rust
// src/rust_modules/omega_rust/src/indicators/atr.rs

/// Wilder ATR (Bloomberg/TradingView-kompatibel)
/// ATR_0 = SMA(TR[0:period])
/// ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
pub fn atr(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> Array1<f64> {
    // True Range + Wilder Smoothing
    // SIMD-optimierbar für True Range Berechnung
}
```

#### Priorität 2: Stepwise-Varianten (20x Speedup Target)

- `ema_stepwise`: Identifiziere HTF-Bar-Indizes, berechne EMA nur dort, forward-fill
- `bollinger_stepwise`: Analog für Bollinger Bands

#### Priorität 3: DMI (20x Speedup Target)

```rust
// src/rust_modules/omega_rust/src/indicators/dmi.rs

/// Directional Movement Index
/// Returns: (+DI, -DI, ADX)
pub fn dmi(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // Wilder Smoothing für DI und ADX
}
```

#### Priorität 4: Standard-Indikatoren (10x Speedup Target)

- `ema`: Einfaches EWM
- `sma`: Rolling Mean
- `rsi`: Wilder RSI
- `macd`: EMA-basiert
- `bollinger`: Rolling Mean + Std

#### Priorität 5: Zusatz-Indikatoren

- `zscore`, `kalman_mean`, `kalman_zscore`, `choppiness`
- Diese können nach den Core-Indikatoren migriert werden

### Phase 4: Python-Integration (Tag 7, ~8h)

#### 3.4.1 Feature-Flag + Delegation

Änderungen an `src/backtest_engine/core/indicator_cache.py`:

1. Feature-Flag `OMEGA_USE_RUST_INDICATOR_CACHE`
2. Wrapper-Pattern für alle public methods
3. Identisches Return-Format (pd.Series, Tuple[pd.Series, ...])
4. 100% Abwärtskompatibilität

#### 3.4.2 Data Conversion

**Input:** `multi_candle_data: Dict[str, Dict[str, List[Candle|None]]]`  
**Output:** `OhlcvData` Rust struct via NumPy arrays

```python
def _prepare_ohlcv_for_rust(
    self,
    tf: str,
    price_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare OHLCV data for Rust (open, high, low, close, volume, valid)."""
    df = self.get_df(tf, price_type)
    valid = ~df.isna().any(axis=1)
    return (
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
        valid.to_numpy(dtype=bool),
    )
```

### Phase 5: Testing & Validierung (Tag 8-9, ~12h)

Siehe Abschnitt 6 (Test-Strategie)

### Phase 6: Performance-Validierung (Tag 10, ~6h)

1. Benchmark-Suite gegen Python-Baseline
2. Speedup-Verifikation (≥ Target)
3. Memory-Profiling
4. Regression-Tests

---

## 4. Rust-Implementation Details

### 4.1 Zusammenfassung der Rust-Dateien

| Datei | Beschreibung | LOC (geschätzt) |
|-------|--------------|-----------------|
| `indicators/mod.rs` | Module exports | ~50 |
| `indicators/types.rs` | Type definitions | ~80 |
| `indicators/cache.rs` | IndicatorCacheRust class | ~300 |
| `indicators/ema.rs` | EMA + Stepwise | ~150 |
| `indicators/sma.rs` | SMA | ~50 |
| `indicators/rsi.rs` | RSI (Wilder) | ~100 |
| `indicators/macd.rs` | MACD | ~80 |
| `indicators/bollinger.rs` | Bollinger + Stepwise | ~200 |
| `indicators/atr.rs` | ATR (Wilder) – **Kritisch** | ~150 |
| `indicators/dmi.rs` | DMI (+DI, -DI, ADX) | ~200 |
| `indicators/roc.rs` | Rate of Change | ~50 |
| `indicators/zscore.rs` | Z-Score Varianten | ~150 |
| `indicators/kalman.rs` | Kalman-Filter | ~200 |
| `indicators/choppiness.rs` | Choppiness Index | ~80 |

**Gesamt:** ~1840 LOC Rust

### 4.2 Dependencies

```toml
# Cargo.toml [dependencies]
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"
numpy = "0.22"
```

### 4.3 Error Handling

Nutzung des bestehenden Error-Handling aus `src/rust_modules/omega_rust/src/error.rs`:

```rust
// Neue Varianten hinzufügen:
pub enum OmegaError {
    // ... bestehende Varianten ...
    
    /// Invalid indicator period
    #[error("[{code}] Invalid period: {value}", code = ErrorCode::InvalidParameter.as_i32())]
    InvalidPeriod { value: usize },
    
    /// Invalid timeframe
    #[error("[{code}] Unknown timeframe: {tf}", code = ErrorCode::InvalidParameter.as_i32())]
    UnknownTimeframe { tf: String },
    
    /// Empty data
    #[error("[{code}] Empty OHLCV data for {tf}/{pt}", code = ErrorCode::InvalidState.as_i32())]
    EmptyOhlcvData { tf: String, pt: String },
}
```

### 4.4 NaN-Handling-Strategie

**Kritisch für Parität:**

```rust
/// NaN Propagation Rules (FFI-Spec Compliance)
/// 
/// 1. NaN in Input → NaN in Output (keine Filterung)
/// 2. Warmup-Periode → NaN (erste `period-1` Werte)
/// 3. Division by Zero → NaN (z.B. RSI bei avg_loss=0)
/// 4. EWM: NaN → Carry-Forward des letzten gültigen Werts
/// 5. Stepwise: NaN → Forward-Fill nach reduced Berechnung

fn handle_nan(value: f64, fallback: f64) -> f64 {
    if value.is_nan() || value.is_infinite() {
        fallback
    } else {
        value
    }
}
```

---

## 5. Python-Integration Details

### 5.1 Environment Variables

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| `OMEGA_USE_RUST_INDICATOR_CACHE` | `"auto"` | `"true"` / `"false"` / `"auto"` |
| `OMEGA_REQUIRE_RUST_FFI` | `"0"` | `"1"` = Fehler wenn Rust nicht verfügbar |

### 5.2 Import-Pfade

```python
# Primärer Import (nutzt automatisch Rust wenn verfügbar)
from backtest_engine.core.indicator_cache import IndicatorCache

# Direkter Rust-Import (für Tests/Benchmarks)
from omega_rust import IndicatorCacheRust
```

### 5.3 Cache-Key Kompatibilität

**Python Cache-Key Format:**
```python
key = ("ema", tf, price_type, int(period))
```

**Rust Cache-Key Format:**
```rust
CacheKey {
    indicator: "ema".to_string(),
    timeframe: tf.to_string(),
    price_type: price_type.to_string(),
    params: format!("{}", period),
}
```

Die Rust-Implementierung verwendet einen identischen Key-Raum, sodass Python-Fallback und Rust-Backend kompatibel sind.

---

## 6. Test-Strategie

### 6.1 Definition von "Identisch" (Semantische Parität)

**Akzeptanz-Toleranzen:**

| Metrik | Toleranz | Begründung |
|--------|----------|------------|
| Numerische Differenz | ≤ 1e-12 | IEEE 754 double precision, strenger als Wave 0/2 wegen Indikator-Sensitivität |
| NaN-Position | Exakt identisch | NaN-Propagation ist kritisch für Backtest-Determinismus |
| Array-Länge | Exakt identisch | Output-Shape muss 1:1 matchen |
| dtype | `float64` | Konsistent für alle Outputs |

**Seed-Handling:**
- IndicatorCache hat **keine Randomness** → deterministisch per Design
- Gleiche Inputs müssen immer gleiche Outputs produzieren

### 6.2 Test-Pyramide

```
                    ┌─────────────────┐
                    │   Golden File   │ ← Determinismus-Gate (höchste Priorität)
                    │     Tests       │   (tests/golden/test_golden_indicator_cache.py)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Integration   │ ← Rust↔Python Parität
                    │     Tests       │   (tests/integration/test_indicator_cache_rust_parity.py)
                    └────────┬────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                      │
    ┌─────┴─────┐                          ┌─────┴─────┐
    │   Rust    │                          │  Property │
    │   Unit    │                          │   Based   │
    │   Tests   │                          │   Tests   │
    └───────────┘                          └───────────┘
```

### 6.3 Test-Dateien

| Datei | Typ | Beschreibung | CI Gate |
|-------|-----|--------------|---------|
| `tests/golden/test_golden_indicator_cache.py` | Golden | Hash-basierte Determinismus-Prüfung | ✅ Blocking |
| `tests/integration/test_indicator_cache_rust_parity.py` | Integration | Rust↔Python Parität für alle Indikatoren | ✅ Blocking |
| `tests/property/test_prop_indicators.py` | Property | Bestehend, erweitern für Edge Cases | ✅ Blocking |
| `tests/benchmarks/test_bench_indicator_cache.py` | Benchmark | Bestehend + Rust-Varianten | ✅ Regression |
| `src/rust_modules/omega_rust/src/indicators/*.rs` | Rust Unit | `#[cfg(test)]` Module | ✅ cargo test |

### 6.4 Konkrete Test-Szenarien

#### Golden-File Tests

```python
# tests/golden/test_golden_indicator_cache.py

def test_indicator_golden_determinism():
    """
    Validiert dass Rust-Backend identische Ergebnisse liefert.
    
    Golden-Reference: tests/golden/reference/indicators/indicator_cache_v1.json
    """
    data = load_fixture("aligned_multi_tf_50k.json")
    cache_python = IndicatorCache(data)  # Force Python
    cache_rust = IndicatorCache(data)    # Force Rust
    
    for indicator in ["ema", "rsi", "atr", "dmi", "bollinger"]:
        result_py = getattr(cache_python, indicator)("M5", "bid", 14)
        result_rs = getattr(cache_rust, indicator)("M5", "bid", 14)
        
        # Exakte Übereinstimmung
        np.testing.assert_allclose(
            result_py.to_numpy(),
            result_rs.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        
        # Hash-Vergleich für Golden-File
        hash_py = sha256(result_py.to_numpy().tobytes()).hexdigest()
        hash_rs = sha256(result_rs.to_numpy().tobytes()).hexdigest()
        assert hash_py == hash_rs
```

#### Parity Tests

```python
# tests/integration/test_indicator_cache_rust_parity.py

@pytest.mark.parametrize("indicator,params", [
    ("ema", {"period": 14}),
    ("ema", {"period": 50}),
    ("ema", {"period": 200}),
    ("ema_stepwise", {"period": 20}),
    ("sma", {"period": 20}),
    ("rsi", {"period": 14}),
    ("macd", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    ("bollinger", {"period": 20, "std_factor": 2.0}),
    ("bollinger_stepwise", {"period": 20, "std_factor": 2.0}),
    ("atr", {"period": 14}),
    ("dmi", {"period": 14}),
])
def test_rust_python_parity(indicator, params):
    """Vergleicht Rust vs Python für jeden Indikator."""
    # Test mit verschiedenen Datengrößen: 100, 1000, 10000, 50000
```

#### FFI Contract Tests

```python
# tests/test_ffi_contracts.py (erweitern)

def test_indicator_cache_ffi_contract():
    """Validiert FFI-Kontrakt aus docs/ffi/indicator_cache.md"""
    # Input: AlignedMultiCandleData
    # Output: pd.Series[float64]
    # NaN-Handling: korrekte Propagation
```

### 6.5 Golden-File Format

**Datei:** `tests/golden/reference/indicators/indicator_cache_v1.json`

```json
{
  "metadata": {
    "version": "1.0",
    "created": "2026-01-09T...",
    "num_bars": 50000,
    "tolerance": 1e-12,
    "description": "Golden-Reference für IndicatorCache Migration Wave 1"
  },
  "test_data_hash": "sha256...",
  "indicators": {
    "ema_14": {
      "hash": "sha256...",
      "nan_count": 13,
      "mean": 1.10234,
      "std": 0.00512
    },
    "rsi_14": {
      "hash": "sha256...",
      "nan_count": 14,
      "mean": 49.87,
      "std": 15.32
    },
    "atr_14": {
      "hash": "sha256...",
      "nan_count": 14,
      "mean": 0.00123,
      "std": 0.00045
    }
  }
}
```

---

## 7. Validierung & Akzeptanzkriterien

### 7.1 Funktionale Kriterien

| ID | Kriterium | Toleranz | Validierung |
|----|-----------|----------|-------------|
| F1 | `ema()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F2 | `ema_stepwise()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F3 | `sma()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F4 | `rsi()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F5 | `macd()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F6 | `bollinger()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F7 | `bollinger_stepwise()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F8 | `atr()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F9 | `dmi()` identisch | ≤1e-12 | Numerischer Diff + Hash |
| F10 | NaN-Positionen identisch | Exakt | Array-Element-Vergleich |
| F11 | Backtest-Ergebnisse identisch | 0% Abweichung | Full Backtest Comparison |
| F12 | Golden-File Tests pass | 100% | CI Gate |

### 7.2 Performance-Kriterien

| Operation | Python Baseline | Rust Target | Min Speedup | Status |
|-----------|-----------------|-------------|-------------|--------|
| `atr` (50k bars) | 954ms | ≤19ms | **50x** | ⏳ |
| `ema_stepwise` | 51ms | ≤2.5ms | **20x** | ⏳ |
| `bollinger_stepwise` | 88ms | ≤4.4ms | **20x** | ⏳ |
| `dmi` | 65ms | ≤3.3ms | **20x** | ⏳ |
| `ema` | 1.25ms | ≤0.125ms | 10x | ⏳ |
| `rsi` | 6.9ms | ≤0.69ms | 10x | ⏳ |
| `macd` | 2.7ms | ≤0.27ms | 10x | ⏳ |
| `bollinger` | 3.7ms | ≤0.37ms | 10x | ⏳ |
| Memory (peak) | 6MB | ≤6MB | ≥1.0x | ⏳ |

**Referenz:** `reports/performance_baselines/p0-01_indicator_cache.json`

### 7.3 Qualitäts-Kriterien

- [ ] **Q1:** `cargo clippy --all-targets -- -D warnings` = 0 Warnungen
- [ ] **Q2:** `cargo test` = alle Tests bestanden
- [ ] **Q3:** `mypy --strict` = keine Fehler für modifizierte Python-Dateien
- [ ] **Q4:** `miri` = keine UB-Findings (Memory Safety)
- [ ] **Q5:** Docstrings für alle öffentlichen Funktionen
- [ ] **Q6:** CHANGELOG.md Eintrag erstellt

---

## 8. Rollback-Plan

### 8.1 Sofort-Rollback (< 1 Minute)

```bash
# Option 1: Feature-Flag deaktivieren
export OMEGA_USE_RUST_INDICATOR_CACHE=false

# Option 2: In Code (falls notwendig)
# src/backtest_engine/core/indicator_cache.py
USE_RUST_INDICATOR_CACHE = False
```

### 8.2 Rollback-Trigger

| Trigger | Schwellwert | Aktion |
|---------|-------------|--------|
| Golden-File Hash Mismatch | Jeder | Sofort-Rollback |
| Numerische Differenz | > 1e-12 | Sofort-Rollback |
| NaN-Position unterschiedlich | Jeder | Sofort-Rollback |
| Backtest-Ergebnis abweichend | Jeder | Sofort-Rollback |
| Performance-Regression | > 10% langsamer | Analyse → ggf. Rollback |
| Memory Leak | Jeder | Sofort-Rollback |
| Panic/Crash | Jeder | Sofort-Rollback |

### 8.3 Post-Rollback

1. Issue erstellen mit Reproduktionsschritten
2. Root-Cause-Analysis durchführen
3. Fix entwickeln und neue Tests hinzufügen
4. Property-Test erweitern für Edge-Case
5. Re-Deployment nach Validierung

### 8.4 Fallback-Semantik

**Wichtig:** Der Python-Fallback muss **immer** funktionsfähig bleiben.

```python
class IndicatorCache:
    def atr(self, tf: str, price_type: str, period: int = 14) -> pd.Series:
        if self._rust and USE_RUST_INDICATOR_CACHE:
            try:
                return self._rust.atr(tf, price_type, period)
            except Exception as e:
                # Log warning, fallback to Python
                logger.warning(f"Rust atr() failed, using Python: {e}")
        
        return self._atr_python(tf, price_type, period)
```

---

## 9. Lessons Learned aus Wave 0 & 2

### 9.1 Erfolgreich angewandte Patterns

| Pattern | Beschreibung | Anwendung in Wave 1 |
|---------|--------------|---------------------|
| Feature-Flag System | `OMEGA_USE_RUST_*` Environment Variable | ✅ Übernehmen |
| Golden-File Tests | Hash-basierte Determinismus-Prüfung | ✅ Übernehmen |
| Hybrid API | Python-Interface mit Rust-Backend | ✅ Übernehmen |
| Error Enum | `OmegaError` mit Python-Mapping | ✅ Übernehmen |
| FFI-Spec First | Dokumentierte Schnittstellen vor Code | ✅ Übernehmen |

### 9.2 Gelöste Probleme aus Wave 0 & 2

#### Problem 1: Namespace Conflict (`logging` module)
- **Wave 0 Lösung:** Verzeichnis umbenannt zu `bt_logging`
- **Wave 1 Relevanz:** ✅ Bereits gelöst, keine Aktion nötig

#### Problem 2: PYTHONPATH Configuration
- **Wave 0 Lösung:** Beide Pfade (root + src) in PYTHONPATH
- **Wave 1 Relevanz:** ✅ Bereits gelöst, Dokumentation vorhanden

#### Problem 3: FFI-Overhead bei Single Calls
- **Wave 2 Erkenntnis:** ~5µs Overhead pro FFI-Call
- **Wave 1 Mitigation:** 
  - Indikatoren werden gecached → nur erster Call langsam
  - Batch-Init: Alle OHLCVs auf einmal an Rust übergeben
  - Cached Calls bleiben in Python (O(1) HashMap Lookup)

#### Problem 4: Datetime-Konvertierung
- **Wave 2 Lösung:** `i64` Unix timestamps in Microseconds
- **Wave 1 Relevanz:** ⚠️ IndicatorCache hat keine Timestamps → nicht relevant

### 9.3 Neue Herausforderungen für Wave 1

| Herausforderung | Mitigation |
|-----------------|------------|
| **NaN-Propagation** | Explizite Tests für jede NaN-Position; Golden-File Vergleich |
| **Float-Determinismus** | IEEE 754 strict; keine SIMD ohne Validierung; `#[repr(C)]` für Arrays |
| **Stepwise-Semantik** | Identische Index-Berechnung; Tests mit HTF-Candle-Fixtures |
| **Cache-Key-Hashing** | Konsistente Serialisierung; Tests für Cache-Hit/Miss |
| **SIMD-Stabilität** | Optional/später; erst nach validierter Scalar-Implementierung |
| **Memory-Management** | `ndarray` mit Rust Ownership; keine Memory Leaks |

### 9.4 Performance-Optimierung Insights

**Aus Wave 0:**
- Batch-First Design erreichte 14.4x Speedup
- FFI-Overhead amortisiert sich ab ~10 Operationen

**Aus Wave 2:**
- State-basierte Module profitieren weniger von FFI
- Aggregierte Operationen (get_summary) zeigen besseren Speedup

**Wave 1 Strategie:**
1. **Init-Phase:** Alle OHLCVs einmalig an Rust übergeben
2. **Compute-Phase:** Indikator-Berechnungen in Rust (hier liegt der Speedup!)
3. **Cache-Phase:** Ergebnisse in Rust-seitigem HashMap
4. **Return-Phase:** NumPy array zurück an Python (Zero-Copy wenn möglich)

### 9.5 Schema-Drift-Prävention

**Referenz:** `reports/schema_fingerprints.json`

Wave 1 muss die folgenden Schemas respektieren:
- `OHLCV_SCHEMA`: Input-Format für Candle-Daten
- `INDICATOR_SCHEMA`: Output-Format für Indikator-Ergebnisse

**CI-Gate:** Schema-Drift-Detection ist aktiv und blockierend.

---

## 10. Checklisten

### 10.1 Pre-Implementation Checklist

- [x] FFI-Spezifikation finalisiert (`docs/ffi/indicator_cache.md`)
- [x] Migration Runbook vorhanden (`docs/runbooks/indicator_cache_migration.md`)
- [x] Benchmarks vorhanden (`tests/benchmarks/test_bench_indicator_cache.py`)
- [x] Performance-Baseline dokumentiert (`reports/performance_baselines/p0-01_indicator_cache.json`)
- [x] Property-Based Tests vorhanden (`tests/property/test_prop_indicators.py`)
- [x] Rust Build-System funktioniert (Wave 0 & 2 validiert)
- [x] Migration Readiness ✅ (`docs/MIGRATION_READINESS_VALIDATION.md`)
- [ ] Golden-Tests vorbereitet (`tests/golden/test_golden_indicator_cache.py`)
- [ ] Lokale Entwicklungsumgebung verifiziert (Rust 1.75+, ndarray, numpy)

### 10.2 Implementation Checklist

#### Phase 1: Setup
- [ ] Verzeichnisstruktur erstellen (`src/rust_modules/omega_rust/src/indicators/`)
- [ ] Cargo.toml Dependencies hinzufügen (`ndarray`, `rayon`, `numpy`)
- [ ] `mod.rs` erstellen und in `lib.rs` registrieren

#### Phase 2: Core Structures
- [ ] `types.rs` implementieren (OhlcvData, CacheKey, IndicatorResult)
- [ ] `cache.rs` implementieren (IndicatorCacheRust class)
- [ ] PyO3 Bindings für Constructor und OHLCV-Init

#### Phase 3: Indikator-Implementation
- [ ] `atr.rs` implementieren (**Priorität 1: 50x Target**)
- [ ] `ema.rs` + `ema_stepwise` implementieren (**Priorität 2: 20x Target**)
- [ ] `bollinger.rs` + `bollinger_stepwise` implementieren (**Priorität 2: 20x Target**)
- [ ] `dmi.rs` implementieren (**Priorität 2: 20x Target**)
- [ ] `sma.rs` implementieren
- [ ] `rsi.rs` implementieren
- [ ] `macd.rs` implementieren
- [ ] `roc.rs` implementieren
- [ ] `zscore.rs` implementieren
- [ ] `kalman.rs` implementieren
- [ ] `choppiness.rs` implementieren
- [ ] `cargo test` bestanden
- [ ] `cargo clippy` bestanden (0 Warnings)

#### Phase 4: Python-Integration
- [ ] `indicator_cache.py` erweitern mit Feature-Flag
- [ ] Wrapper-Methoden für alle Indikatoren
- [ ] `get_rust_status()` Funktion hinzufügen
- [ ] mypy --strict validiert

#### Phase 5: Testing
- [ ] Golden-Tests erstellt und bestanden
- [ ] Integration-Tests (Rust↔Python Parität) erstellt und bestanden
- [ ] Property-Based Tests erweitert und bestanden
- [ ] Benchmark-Suite erweitert für Rust
- [ ] Rust-Unit-Tests bestanden

#### Phase 6: Performance-Validierung
- [ ] ATR ≥ 50x Speedup
- [ ] ema_stepwise, bollinger_stepwise, dmi ≥ 20x Speedup
- [ ] Standard-Indikatoren ≥ 10x Speedup
- [ ] Memory ≤ Python-Baseline
- [ ] Keine Memory-Leaks (miri)

### 10.3 Post-Implementation Checklist

- [ ] Dokumentation aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] README.md Performance-Zahlen aktualisiert
- [ ] Code-Review abgeschlossen
- [ ] Sign-off Matrix ausgefüllt

### 10.4 Sign-off Matrix

| Rolle | Name | Datum | Status |
|-------|------|-------|--------|
| Developer | - | - | ⏳ Pending |
| FFI-Spec Review | - | - | ⏳ Pending |
| Golden Tests | pytest | - | ⏳ Pending |
| Integration Tests | pytest | - | ⏳ Pending |
| Benchmarks | pytest-benchmark | - | ⏳ Pending |
| Performance Validation | perf_indicator_cache.py | - | ⏳ Pending |
| Security Review | miri + clippy | - | ⏳ Pending |
| Tech Lead | - | - | ⏳ Pending |

---

## 11. Zeitplan

| Tag | Phase | Aufgaben |
|-----|-------|----------|
| 1-2 | Setup | Rust-Modul Setup, Dependencies, Type Definitions |
| 2-3 | Core Structures | OhlcvData, CacheKey, IndicatorCacheRust Skeleton |
| 3-6 | Indikator-Implementation | ATR (Prio 1), Stepwise-Varianten (Prio 2), DMI (Prio 2), Standard-Indikatoren |
| 7 | Python-Integration | Feature-Flag, Wrapper, Conversion |
| 8-9 | Testing | Golden-Tests, Parity-Tests, Property-Tests |
| 10 | Performance + Buffer | Benchmark-Validierung, Fixes, Dokumentation |

**Geschätzter Aufwand:** 8-10 Arbeitstage

---

## 12. References

- [FFI Specification: IndicatorCache](./ffi/indicator_cache.md)
- [Migration Runbook: IndicatorCache](./runbooks/indicator_cache_migration.md)
- [Performance Baseline](../reports/performance_baselines/p0-01_indicator_cache.json)
- [Benchmark Suite](../tests/benchmarks/test_bench_indicator_cache.py)
- [Migration Readiness Validation](./MIGRATION_READINESS_VALIDATION.md)
- [Wave 0: Slippage & Fee Implementation Plan](./WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md)
- [Wave 2: Portfolio Implementation Plan](./WAVE_2_PORTFOLIO_IMPLEMENTATION_PLAN.md)
- [ADR-0001: Migration Strategy](./adr/ADR-0001-migration-strategy.md)
- [ADR-0003: Error Handling](./adr/ADR-0003-error-handling.md)

---

## Änderungshistorie

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-09 | 1.0 | Initiale Version des Implementationsplans | AI Agent |

---

*Document Status: 🔴 READY FOR IMPLEMENTATION*
