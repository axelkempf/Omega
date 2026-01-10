# Wave 3: Event Engine Rust Migration Implementation Plan

**Document Version:** 1.0  
**Created:** 2026-01-10  
**Updated:** 2026-01-10  
**Status:** 🔵 READY FOR IMPLEMENTATION  
**Module:** `src/backtest_engine/core/event_engine.py`

---

## Executive Summary

Dieser Plan beschreibt die vollständige Implementierung der Migration der Event Engine zu Rust als **Wave 3**. Die Event Engine ist die zentrale Hauptschleife des Backtest-Systems und orchestriert den gesamten Datenfluss zwischen Candle-Events, Strategie-Signalen und Execution. Die Migration folgt den etablierten Patterns und **Lessons Learned** aus Wave 0 (Slippage & Fee), Wave 1 (IndicatorCache) und Wave 2 (Portfolio).

### Warum Event Engine als Wave 3?

| Eigenschaft | Bewertung | Begründung |
|-------------|-----------|------------|
| **Abhängigkeiten** | ✅ Erfüllt | Wave 0-2 abgeschlossen (Fees, Portfolio, IndicatorCache) |
| **Performance-Impact** | ✅ Kritisch | Hauptschleife mit ~20k Iterationen pro Backtest |
| **Komplexität** | ⚠️ Hoch | Koordiniert Strategy, Executor, Portfolio, IndicatorCache |
| **Callback-Handling** | ⚠️ Komplex | Python-Strategien müssen per FFI aufgerufen werden |
| **Testbarkeit** | ✅ Gut | Golden-Tests, Determinismus-Validierung vorhanden |
| **Risiko** | ⚠️ Mittel-Hoch | Determinismus kritisch, aber gut testbar |
| **Geschätzter Aufwand** | ⚠️ 8-12 Tage | XL-Modul mit vielen Interaktionen |

### Prerequisites (alle erfüllt)

- ✅ **Wave 0:** Slippage & Fee (Rust Integration, FFI Pattern etabliert)
- ✅ **Wave 1:** IndicatorCache (Rust-Backend für Indikatoren)
- ✅ **Wave 2:** Portfolio (State-Management in Rust)
- ✅ **FFI-Spezifikation:** `docs/ffi/event_engine.md`
- ✅ **Migration Runbook:** `docs/runbooks/event_engine_migration.md`
- ✅ **Performance-Baseline:** `reports/performance_baselines/p0-01_event_engine.json`

---

## Inhaltsverzeichnis

1. [Voraussetzungen & Status](#1-voraussetzungen--status)
2. [Lessons Learned aus Wave 0-2](#2-lessons-learned-aus-wave-0-2)
3. [Architektur-Übersicht](#3-architektur-übersicht)
4. [Implementierungs-Phasen](#4-implementierungs-phasen)
5. [Rust-Implementation](#5-rust-implementation)
6. [Python-Integration](#6-python-integration)
7. [Test-Strategie](#7-test-strategie)
8. [Validierung & Akzeptanzkriterien](#8-validierung--akzeptanzkriterien)
9. [Rollback-Plan](#9-rollback-plan)
10. [Checklisten](#10-checklisten)

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure-Readiness (aus Wave 0-2 etabliert)

| Komponente | Status | Evidenz |
|------------|--------|---------|
| Rust Build System | ✅ | `src/rust_modules/omega_rust/Cargo.toml` |
| PyO3/Maturin | ✅ | Version 0.27 konfiguriert |
| Error Handling | ✅ | `src/rust_modules/omega_rust/src/error.rs` |
| FFI-Spezifikation | ✅ | `docs/ffi/event_engine.md` |
| Migration Runbook | ✅ | `docs/runbooks/event_engine_migration.md` |
| mypy strict | ✅ | `backtest_engine.core.*` strict-compliant |
| Golden-Tests | ✅ | `tests/golden/test_golden_backtest.py` |
| Benchmarks | ✅ | `tests/benchmarks/test_bench_event_engine.py` |
| Performance Baseline | ✅ | `reports/performance_baselines/p0-01_event_engine.json` |

### 1.2 Python-Modul Baseline

**Datei:** `src/backtest_engine/core/event_engine.py`

Die aktuelle Python-Implementation (~230 LOC) enthält:

**Klassen:**
- `EventEngine`: Single-Symbol Event Loop
- `CrossSymbolEventEngine`: Multi-Symbol Event Loop

**Core-Methoden (EventEngine.run()):**
1. Warmup-Index Berechnung (`original_start_dt`)
2. IndicatorCache Initialisierung (`get_cached_indicator_cache()`)
3. Bar-by-Bar Iteration
4. Strategy Evaluation → Signale
5. Execution (process_signal, evaluate_exits)
6. Position Management
7. Portfolio Update
8. Progress Callback

### 1.3 Performance-Baseline (aus `p0-01_event_engine.json`)

```json
{
  "meta": { "num_bars": 20000 },
  "first_run_seconds": 0.436,
  "second_run_seconds": 0.337,
  "profile_top20": {
    "event_engine.py:run": "0.032s (9.5% Loop-Overhead)",
    "indicator_cache.py:__init__": "0.131s (39% Init)",
    "strategy_wrapper.py:evaluate": "0.126s (37% Strategy)",
    "portfolio.py:update": "0.026s (7.7%)"
  }
}
```

**Bottleneck-Analyse:**
| Komponente | Zeit | Anteil | Rust-Migration |
|------------|------|--------|----------------|
| Loop-Overhead (Python) | ~32ms | 9.5% | ✅ Wave 3 |
| IndicatorCache Init | ~131ms | 39% | ✅ Wave 1 (erledigt) |
| Strategy Evaluate | ~126ms | 37% | ⏸️ Bleibt Python (Callbacks) |
| Portfolio Update | ~26ms | 7.7% | ✅ Wave 2 (erledigt) |

**Erwartete Speedups nach Wave 3:**
| Operation | Python Baseline | Rust Target | Erwarteter Speedup |
|-----------|-----------------|-------------|-------------------|
| Event Loop Dispatch | 32ms (20k bars) | 5ms | **6x** |
| State Transitions | Inkludiert | 1ms | 10x |
| Full Backtest (20k bars) | 337ms | 200ms* | **1.7x** |

*) Strategy-Callbacks bleiben Python-bound, limitiert Gesamtspeedup

---

## 2. Lessons Learned aus Wave 0-2

### 2.1 Kritische Learnings aus Wave 0 (Slippage & Fee)

| Learning | Anwendung für Wave 3 |
|----------|---------------------|
| **Namespace Conflicts** | Proaktiv Python stdlib-Konflikte scannen (`event`, `signal`) |
| **PYTHONPATH Config** | Dokumentieren: `PYTHONPATH=/Omega:/Omega/src` |
| **Batch-First Design** | Event-Batches statt einzelne Callbacks wo möglich |
| **RNG Determinismus** | Keine Rust-RNG in Event-Engine (wird in Strategy verwendet) |
| **FFI-Overhead** | Callback-Overhead ~5μs, Batch ab 10 Events amortisiert |

### 2.2 Kritische Learnings aus Wave 1 (IndicatorCache)

| Learning | Anwendung für Wave 3 |
|----------|---------------------|
| **Integration ≠ Implementierung** | End-to-End Test: Rust-Backend muss im Backtest tatsächlich aktiv sein |
| **Feature-Flag ohne Effekt** | CI-Test mit aktiviertem Flag + Backend-Verifikation |
| **API-Drift** | Gemeinsames Protocol für Python/Rust Backend |
| **Methoden-Parität** | Parity-Test für alle Strategy-verwendeten Methoden |
| **Multi-Engine-Varianten** | `CrossSymbolEventEngine` ebenfalls integrieren |
| **Dokumentations-Drift** | Reality-Check Sektion mit automatisierter Validierung |

### 2.3 Kritische Learnings aus Wave 2 (Portfolio)

| Learning | Anwendung für Wave 3 |
|----------|---------------------|
| **State-Management** | Event-Engine ist stateful, aber orchestriert externe State |
| **FFI-Crossings minimieren** | Aggregierte Operationen: Batch-Event-Processing |
| **Batch-API** | `process_batch()` Pattern für Event-Sequenzen |
| **Type-Conversion** | Explizite Rust↔Python Konverter für Candle, Signal, Slice |

### 2.4 Design-Entscheidungen basierend auf Learnings

**Entscheidung 1: Hybrid-Architektur (Strategy B)**
- Rust-Loop für Event-Dispatch
- Python-Callbacks für Strategy-Evaluation
- Begründung: Bestandene Strategien bleiben kompatibel

**Entscheidung 2: Batch-Processing für Exits**
- `evaluate_exits_batch()` statt pro-Bar Calls
- Begründung: FFI-Overhead amortisieren (Wave 0 Learning)

**Entscheidung 3: Feature-Flag mit CI-Verifikation**
- `OMEGA_USE_RUST_EVENT_ENGINE` mit Default `auto`
- CI-Test prüft aktives Backend (Wave 1 Learning)

**Entscheidung 4: End-to-End Golden-Test**
- Voller Backtest-Vergleich Python vs Rust
- Bit-genaue Reproduzierbarkeit (Determinismus)

---

## 3. Architektur-Übersicht

### 3.1 Ziel-Architektur (Hybrid: Rust Loop + Python Callbacks)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKTEST ENGINE                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │     Python API Layer (src/backtest_engine/core/event_engine.py)            │ │
│  │                                                                            │ │
│  │  class EventEngine:                                                        │ │
│  │      def __init__(self, ...):                                              │ │
│  │          if USE_RUST_EVENT_ENGINE:                                         │ │
│  │              self._rust = EventEngineRust(...)     ◄── Rust Backend        │ │
│  │          else:                                                             │ │
│  │              self._rust = None                     ◄── Pure Python         │ │
│  │                                                                            │ │
│  │      def run(self) -> None:                                                │ │
│  │          if self._rust:                                                    │ │
│  │              self._rust.run(                                               │ │
│  │                  strategy_callback=self._wrap_strategy,                    │ │
│  │                  progress_callback=self.on_progress                        │ │
│  │              )                                                             │ │
│  │          else:                                                             │ │
│  │              self._run_python()                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              │ FFI Boundary (PyO3)                               │
│                              ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │        Rust Layer (src/rust_modules/omega_rust/src/event/)                 │ │
│  │                                                                            │ │
│  │  #[pyclass]                                                                │ │
│  │  pub struct EventEngineRust {                                              │ │
│  │      candle_data: CandleStore,        // Zero-Copy Arrow Data              │ │
│  │      current_index: usize,            // Aktueller Bar-Index               │ │
│  │      start_index: usize,              // Warmup-Ende Index                 │ │
│  │      total_bars: usize,               // Gesamtzahl Bars                   │ │
│  │  }                                                                         │ │
│  │                                                                            │ │
│  │  #[pymethods]                                                              │ │
│  │  impl EventEngineRust {                                                    │ │
│  │      #[new]                                                                │ │
│  │      fn new(bid: Vec<Candle>, ask: Vec<Candle>, ...) -> Self;              │ │
│  │                                                                            │ │
│  │      fn run(                                                               │ │
│  │          &mut self,                                                        │ │
│  │          py: Python,                                                       │ │
│  │          strategy_callback: PyObject,      // Python Strategy.evaluate     │ │
│  │          executor: PyObject,               // Python ExecutionSimulator    │ │
│  │          portfolio: PyObject,              // Python/Rust Portfolio        │ │
│  │          progress_callback: Option<PyObject>,                              │ │
│  │      ) -> PyResult<()>;                                                    │ │
│  │  }                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Integration mit anderen Rust-Modulen:                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                          │
│  │ PortfolioRust│  │IndicatorCache│  │  Costs       │                          │
│  │   (Wave 2)   │  │   (Wave 1)   │  │  (Wave 0)    │                          │
│  └──────────────┘  └──────────────┘  └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Event Loop State Machine

```
┌──────────────────┐
│  INITIALIZE      │
│  - Parse candles │
│  - Find warmup   │
│  - Init cache    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  EVENT LOOP (Rust): for i in start_index..total_bars                        │
│                                                                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │ 1. PREPARE    │───▶│ 2. STRATEGY   │───▶│ 3. PROCESS    │                │
│  │ slice.set_idx │    │ callback(py)  │    │ signals       │                │
│  │ (Rust)        │    │ (Python)      │    │ (Rust/Python) │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
│         │                    │                    │                          │
│         │                    │                    ▼                          │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │ 6. PROGRESS   │◄───│ 5. PORTFOLIO  │◄───│ 4. EXITS      │                │
│  │ callback(py)  │    │ update()      │    │ eval_exits()  │                │
│  │ (Python)      │    │ (Rust Wave 2) │    │ (Rust/Python) │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
│         │                                                                    │
│  ◄──────┴──────────────────────────────────────────────── (next i)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  FINALIZE        │
│  - Cleanup       │
│  - Return stats  │
└──────────────────┘
```

### 3.3 Feature-Flag-System

```python
# src/backtest_engine/core/event_engine.py

import os
from typing import Any, Optional

_RUST_AVAILABLE: bool = False
_RUST_MODULE: Any = None

def _check_rust_event_engine_available() -> bool:
    """Check if Rust EventEngine module is available and functional."""
    global _RUST_MODULE
    try:
        import omega_rust
        if hasattr(omega_rust, "EventEngineRust"):
            _RUST_MODULE = omega_rust
            return True
    except ImportError:
        pass
    return False

def _should_use_rust_event_engine() -> bool:
    """Determine if Rust implementation should be used."""
    env_val = os.environ.get("OMEGA_USE_RUST_EVENT_ENGINE", "auto").lower()
    if env_val == "false" or env_val == "0":
        return False
    if env_val == "true" or env_val == "1":
        return _RUST_AVAILABLE
    # auto: use Rust if available
    return _RUST_AVAILABLE

# Initialize on module load
_RUST_AVAILABLE = _check_rust_event_engine_available()

def get_active_backend() -> str:
    """Returns 'rust' or 'python' - for CI verification (Wave 1 Learning)."""
    return "rust" if _should_use_rust_event_engine() else "python"
```

### 3.4 Datei-Struktur nach Migration

```
src/
├── rust_modules/
│   └── omega_rust/
│       ├── src/
│       │   ├── lib.rs                    # Modul-Registration erweitern
│       │   ├── error.rs                  # Bestehendes Error-Handling
│       │   ├── costs/                    # Wave 0: Slippage & Fee
│       │   ├── indicators/               # Wave 1: IndicatorCache
│       │   ├── portfolio/                # Wave 2: Portfolio
│       │   └── event/                    # NEU: Event Engine
│       │       ├── mod.rs                # NEU: Module exports
│       │       ├── types.rs              # NEU: CandleEvent, SignalResult
│       │       ├── queue.rs              # NEU: EventQueue (VecDeque)
│       │       ├── engine.rs             # NEU: EventEngineRust struct
│       │       └── callbacks.rs          # NEU: Python Callback Bridge
│       └── Cargo.toml                    # parking_lot für Mutex
│
├── backtest_engine/
│   └── core/
│       └── event_engine.py               # Erweitert mit Rust-Integration
│
tests/
├── golden/
│   └── test_golden_backtest.py           # Bestehendes (Determinismus)
├── benchmarks/
│   └── test_bench_event_engine.py        # Bestehendes + Rust-Varianten
└── integration/
    ├── test_event_engine_rust.py         # NEU: Rust-spezifische Tests
    └── test_event_engine_backend_verify.py  # NEU: Backend-Verifikation (Wave 1 Learning)
```

---

## 4. Implementierungs-Phasen

### Phase 1: Rust-Modul Setup (Tag 1, ~4h)

#### 4.1.1 Verzeichnisstruktur erstellen

```bash
mkdir -p src/rust_modules/omega_rust/src/event
touch src/rust_modules/omega_rust/src/event/mod.rs
touch src/rust_modules/omega_rust/src/event/types.rs
touch src/rust_modules/omega_rust/src/event/queue.rs
touch src/rust_modules/omega_rust/src/event/engine.rs
touch src/rust_modules/omega_rust/src/event/callbacks.rs
```

#### 4.1.2 Cargo.toml aktualisieren

```toml
# Hinzufügen zu [dependencies]
parking_lot = "0.12"       # Schnelle Mutex für State
crossbeam-channel = "0.5"  # Optional: Channel für Event-Queue
```

#### 4.1.3 Module registrieren in lib.rs

```rust
pub mod event;  // NEU

use event::EventEngineRust;

#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bestehende Funktionen...
    
    // NEU: Event Engine Class
    m.add_class::<EventEngineRust>()?;
    
    Ok(())
}
```

### Phase 2: Core Rust Structures (Tag 1-2, ~8h)

#### 4.2.1 Type Definitions

**Datei:** `src/rust_modules/omega_rust/src/event/types.rs` <!-- docs-lint:planned -->

```rust
use pyo3::prelude::*;

/// Repräsentiert eine einzelne Candle
#[derive(Clone, Debug)]
pub struct CandleData {
    pub timestamp: i64,      // Unix timestamp (microseconds)
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl CandleData {
    pub fn from_pyobject(py: Python<'_>, obj: &PyAny) -> PyResult<Self> {
        Ok(Self {
            timestamp: obj.getattr("timestamp")?.extract()?,
            open: obj.getattr("open")?.extract()?,
            high: obj.getattr("high")?.extract()?,
            low: obj.getattr("low")?.extract()?,
            close: obj.getattr("close")?.extract()?,
            volume: obj.getattr("volume")?.extract()?,
        })
    }
}

/// Signal von Strategy-Callback
#[derive(Clone, Debug)]
pub enum SignalDirection {
    Long,
    Short,
    None,
}

#[derive(Clone, Debug)]
pub struct TradeSignalRust {
    pub direction: SignalDirection,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub symbol: String,
    pub reason: Option<String>,
}

impl TradeSignalRust {
    pub fn from_pyobject(py: Python<'_>, obj: &PyAny) -> PyResult<Option<Self>> {
        if obj.is_none() {
            return Ok(None);
        }
        
        let direction_str: String = obj.getattr("direction")?.extract()?;
        let direction = match direction_str.as_str() {
            "long" | "buy" => SignalDirection::Long,
            "short" | "sell" => SignalDirection::Short,
            _ => SignalDirection::None,
        };
        
        Ok(Some(Self {
            direction,
            entry_price: obj.getattr("entry_price")?.extract()?,
            stop_loss: obj.getattr("stop_loss")?.extract()?,
            take_profit: obj.getattr("take_profit")?.extract()?,
            symbol: obj.getattr("symbol")?.extract()?,
            reason: obj.getattr("reason").ok().and_then(|r| r.extract().ok()),
        }))
    }
}

/// Batch-Result für Performance-Monitoring
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct EventEngineStats {
    #[pyo3(get)]
    pub bars_processed: usize,
    #[pyo3(get)]
    pub signals_generated: usize,
    #[pyo3(get)]
    pub exits_triggered: usize,
    #[pyo3(get)]
    pub total_time_ms: f64,
    #[pyo3(get)]
    pub callback_time_ms: f64,
}
```

### Phase 3: Event Engine Implementation (Tag 3-5, ~16h)

#### 4.3.1 EventEngineRust Struct

**Datei:** `src/rust_modules/omega_rust/src/event/engine.rs` <!-- docs-lint:planned -->

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Instant;

use super::types::{CandleData, EventEngineStats, TradeSignalRust};
use crate::error::{OmegaError, Result};

/// Rust implementation of EventEngine for high-performance backtesting
#[pyclass]
pub struct EventEngineRust {
    bid_candles: Vec<CandleData>,
    ask_candles: Vec<CandleData>,
    symbol: String,
    start_index: usize,
    current_index: usize,
    total_bars: usize,
    stats: EventEngineStats,
}

#[pymethods]
impl EventEngineRust {
    #[new]
    #[pyo3(signature = (bid_candles, ask_candles, symbol, original_start_dt))]
    pub fn new(
        py: Python<'_>,
        bid_candles: &PyList,
        ask_candles: &PyList,
        symbol: String,
        original_start_dt: i64,
    ) -> PyResult<Self> {
        // Parse Candles from Python objects
        let bid: Vec<CandleData> = bid_candles
            .iter()
            .map(|c| CandleData::from_pyobject(py, c))
            .collect::<PyResult<Vec<_>>>()?;
            
        let ask: Vec<CandleData> = ask_candles
            .iter()
            .map(|c| CandleData::from_pyobject(py, c))
            .collect::<PyResult<Vec<_>>>()?;
        
        if bid.len() != ask.len() {
            return Err(PyErr::from(OmegaError::InvalidParameter {
                reason: "bid_candles and ask_candles must have same length".to_string(),
            }));
        }
        
        // Find start index (after warmup)
        let start_index = bid
            .iter()
            .position(|c| c.timestamp >= original_start_dt)
            .ok_or_else(|| OmegaError::InvalidParameter {
                reason: "No candle found >= original_start_dt".to_string(),
            })?;
        
        let total_bars = bid.len();
        
        Ok(Self {
            bid_candles: bid,
            ask_candles: ask,
            symbol,
            start_index,
            current_index: start_index,
            total_bars,
            stats: EventEngineStats::default(),
        })
    }
    
    /// Main event loop with Python callbacks
    #[pyo3(signature = (strategy, executor, portfolio, slice_map, on_progress=None))]
    pub fn run(
        &mut self,
        py: Python<'_>,
        strategy: PyObject,          // StrategyWrapper with evaluate()
        executor: PyObject,          // ExecutionSimulator
        portfolio: PyObject,         // Portfolio (Python or Rust)
        slice_map: PyObject,         // Dict[symbol, SymbolDataSlice]
        on_progress: Option<PyObject>,
    ) -> PyResult<EventEngineStats> {
        let loop_start = Instant::now();
        let mut callback_time = 0.0_f64;
        
        for i in self.start_index..self.total_bars {
            self.current_index = i;
            let bid_candle = &self.bid_candles[i];
            let ask_candle = &self.ask_candles[i];
            let timestamp = bid_candle.timestamp;
            
            // === 1. UPDATE SLICE INDEX ===
            // Get symbol_slice from slice_map and update index
            let slice = slice_map.call_method1(py, "get", (&self.symbol,))?;
            slice.call_method1(py, "set_index", (i,))?;
            
            // === 2. STRATEGY EVALUATION (Python Callback) ===
            let callback_start = Instant::now();
            let signals_result = strategy.call_method1(py, "evaluate", (i, &slice_map))?;
            callback_time += callback_start.elapsed().as_secs_f64() * 1000.0;
            
            // Process signals if any
            if !signals_result.is_none(py) {
                let signals = if signals_result.downcast::<PyList>(py).is_ok() {
                    signals_result.extract::<Vec<PyObject>>(py)?
                } else {
                    vec![signals_result]
                };
                
                for signal in signals {
                    executor.call_method1(py, "process_signal", (&signal,))?;
                    self.stats.signals_generated += 1;
                }
            }
            
            // === 3. EVALUATE EXITS ===
            let active_positions = executor.getattr(py, "active_positions")?;
            let has_positions: bool = !active_positions.call_method0(py, "__len__")?.extract::<usize>(py)?.eq(&0);
            
            if has_positions {
                // Convert Rust candles to Python for executor
                let bid_py = self.candle_to_pydict(py, bid_candle)?;
                let ask_py = self.candle_to_pydict(py, ask_candle)?;
                
                executor.call_method1(py, "evaluate_exits", (&bid_py, &ask_py))?;
            }
            
            // === 4. POSITION MANAGEMENT ===
            // This remains in Python due to complex strategy-specific logic
            // (Trailing stops, break-even, etc.)
            
            // === 5. PORTFOLIO UPDATE ===
            portfolio.call_method1(py, "update", (timestamp,))?;
            
            // === 6. PROGRESS CALLBACK ===
            if let Some(ref callback) = on_progress {
                let current = i - self.start_index + 1;
                let total = self.total_bars - self.start_index;
                callback.call1(py, (current, total))?;
            }
            
            self.stats.bars_processed += 1;
        }
        
        self.stats.total_time_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
        self.stats.callback_time_ms = callback_time;
        
        Ok(self.stats.clone())
    }
    
    /// Convert Rust CandleData to Python dict (for executor compatibility)
    fn candle_to_pydict(&self, py: Python<'_>, candle: &CandleData) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("timestamp", candle.timestamp)?;
        dict.set_item("open", candle.open)?;
        dict.set_item("high", candle.high)?;
        dict.set_item("low", candle.low)?;
        dict.set_item("close", candle.close)?;
        dict.set_item("volume", candle.volume)?;
        Ok(dict.into())
    }
    
    /// Get current statistics
    #[getter]
    pub fn stats(&self) -> EventEngineStats {
        self.stats.clone()
    }
    
    /// Get current bar index (for debugging)
    #[getter]
    pub fn current_index(&self) -> usize {
        self.current_index
    }
}
```

### Phase 4: Python Integration (Tag 6-7, ~12h)

#### 4.4.1 EventEngine Wrapper erweitern

**Datei:** `src/backtest_engine/core/event_engine.py` (aktualisiert)

```python
from typing import Any, Callable, Dict, List, Optional
import os

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.indicator_cache import get_cached_indicator_cache
from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.candle import Candle
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper

# === Feature Flag System (Wave 1 Learning) ===
_RUST_AVAILABLE: bool = False
_RUST_MODULE: Any = None

def _check_rust_event_engine_available() -> bool:
    global _RUST_MODULE
    try:
        import omega_rust
        if hasattr(omega_rust, "EventEngineRust"):
            _RUST_MODULE = omega_rust
            return True
    except ImportError:
        pass
    return False

def _should_use_rust() -> bool:
    env_val = os.environ.get("OMEGA_USE_RUST_EVENT_ENGINE", "auto").lower()
    if env_val in ("false", "0"):
        return False
    if env_val in ("true", "1"):
        return _RUST_AVAILABLE
    return _RUST_AVAILABLE

_RUST_AVAILABLE = _check_rust_event_engine_available()

def get_active_backend() -> str:
    """CI verification helper (Wave 1 Learning)."""
    return "rust" if _should_use_rust() else "python"


class EventEngine:
    """
    Event Engine für Single-Symbol Backtests.
    
    Supports both Python and Rust backends via feature flag.
    """

    def __init__(
        self,
        bid_candles: List[Candle],
        ask_candles: List[Candle],
        strategy: StrategyWrapper,
        executor: ExecutionSimulator,
        portfolio: Portfolio,
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        symbol: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        original_start_dt: Optional[Any] = None,
    ):
        self.bid_candles = bid_candles
        self.ask_candles = ask_candles
        self.strategy = strategy
        self.executor = executor
        self.portfolio = portfolio
        self.multi_candle_data = multi_candle_data
        self.symbol = symbol
        self.on_progress = on_progress
        self.original_start_dt = original_start_dt
        
        # Rust backend initialization
        self._rust_engine: Any = None
        self._use_rust = _should_use_rust()
        
        if self._use_rust and _RUST_MODULE is not None:
            try:
                self._init_rust_engine()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Rust EventEngine init failed, falling back to Python: {e}"
                )
                self._use_rust = False

    def _init_rust_engine(self) -> None:
        """Initialize Rust backend."""
        if self.original_start_dt is None:
            raise ValueError("original_start_dt required for Rust backend")
        
        # Convert datetime to microseconds timestamp
        start_ts = int(self.original_start_dt.timestamp() * 1_000_000)
        
        self._rust_engine = _RUST_MODULE.EventEngineRust(
            bid_candles=self.bid_candles,
            ask_candles=self.ask_candles,
            symbol=self.symbol,
            original_start_dt=start_ts,
        )

    def run(self) -> None:
        """Main event loop - delegates to Rust or Python backend."""
        if self._use_rust and self._rust_engine is not None:
            self._run_rust()
        else:
            self._run_python()

    def _run_rust(self) -> None:
        """Rust-accelerated event loop."""
        # Prepare indicator cache and slice
        ind_cache = get_cached_indicator_cache(self.multi_candle_data)
        
        start_index = next(
            (i for i, c in enumerate(self.bid_candles) 
             if c.timestamp >= self.original_start_dt),
            None,
        )
        if start_index is None:
            raise ValueError("No start index found")
        
        symbol_slice = SymbolDataSlice(
            multi_candle_data=self.multi_candle_data,
            index=start_index,
            indicator_cache=ind_cache,
        )
        slice_map = {self.symbol: symbol_slice}
        
        # Run Rust event loop with Python callbacks
        stats = self._rust_engine.run(
            strategy=self.strategy,
            executor=self.executor,
            portfolio=self.portfolio,
            slice_map=slice_map,
            on_progress=self.on_progress,
        )
        
        # Log performance stats (optional)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Rust EventEngine stats: {stats.bars_processed} bars, "
            f"{stats.total_time_ms:.1f}ms total, "
            f"{stats.callback_time_ms:.1f}ms in callbacks"
        )

    def _run_python(self) -> None:
        """Original Python implementation (fallback)."""
        # [Original Python code remains unchanged]
        total = len(self.bid_candles)

        if self.original_start_dt is None:
            raise ValueError("original_start_dt muss gesetzt werden!")

        start_index = next(
            (i for i, c in enumerate(self.bid_candles)
             if c.timestamp >= self.original_start_dt),
            None,
        )
        if start_index is None:
            raise ValueError("Kein Startindex gefunden!")

        ind_cache = get_cached_indicator_cache(self.multi_candle_data)
        symbol_slice = SymbolDataSlice(
            multi_candle_data=self.multi_candle_data,
            index=start_index,
            indicator_cache=ind_cache,
        )
        slice_map = {self.symbol: symbol_slice}

        for i in range(start_index, total):
            bid_candle = self.bid_candles[i]
            ask_candle = self.ask_candles[i]
            timestamp = bid_candle.timestamp
            symbol_slice.set_index(i)

            # ENTRY
            signals = self.strategy.evaluate(i, slice_map)
            if signals:
                if not isinstance(signals, list):
                    signals = [signals]
                for signal in signals:
                    self.executor.process_signal(signal)

            # EXITS
            if self.executor.active_positions:
                self.executor.evaluate_exits(bid_candle, ask_candle)

            # POSITION MANAGEMENT
            if self.executor.active_positions:
                strategy_instance = getattr(self.strategy.strategy, "strategy", None)
                pm = getattr(strategy_instance, "position_manager", None)
                if pm:
                    if not getattr(pm, "portfolio", None):
                        pm.attach_portfolio(self.portfolio)
                    open_pos = self.portfolio.get_open_positions(self.symbol)
                    all_pos = self.executor.active_positions
                    pm.manage_positions(
                        open_positions=open_pos,
                        symbol_slice=symbol_slice,
                        bid_candle=bid_candle,
                        ask_candle=ask_candle,
                        all_positions=all_pos,
                    )

            # PORTFOLIO UPDATE
            self.portfolio.update(timestamp)

            # PROGRESS
            if callable(self.on_progress):
                self.on_progress((i - start_index) + 1, (total - start_index))
```

### Phase 5: Testing (Tag 8-9, ~12h)

#### 4.5.1 Test-Strategie

<!-- docs-lint:planned - Test files to be created during Wave 3 implementation -->
| Test-Typ | Datei | Zweck |
|----------|-------|-------|
| Unit Tests (Rust) | `omega_rust/tests/` | Rust-interne Logik |
| Integration Tests | `tests/integration/test_event_engine_rust.py` | Rust↔Python Integration | <!-- docs-lint:planned -->
| Backend Verification | `tests/integration/test_event_engine_backend_verify.py` | CI: Rust aktiv prüfen | <!-- docs-lint:planned -->
| Golden Tests | `tests/golden/test_golden_backtest.py` | Determinismus-Validierung |
| Benchmarks | `tests/benchmarks/test_bench_event_engine.py` | Performance-Regression |
| Parity Tests | `tests/test_event_engine_parity.py` | Python vs Rust Ergebnisse | <!-- docs-lint:planned -->

#### 4.5.2 Backend Verification Test (Wave 1 Learning)

```python
# tests/integration/test_event_engine_backend_verify.py

import os
import pytest

def test_rust_backend_active_when_enabled():
    """Verify Rust backend is actually used when flag is set (Wave 1 Learning)."""
    os.environ["OMEGA_USE_RUST_EVENT_ENGINE"] = "1"
    
    # Force reimport
    import importlib
    from backtest_engine.core import event_engine
    importlib.reload(event_engine)
    
    assert event_engine.get_active_backend() == "rust", \
        "Rust backend should be active when OMEGA_USE_RUST_EVENT_ENGINE=1"

def test_python_fallback_when_disabled():
    """Verify Python fallback works when flag is disabled."""
    os.environ["OMEGA_USE_RUST_EVENT_ENGINE"] = "0"
    
    import importlib
    from backtest_engine.core import event_engine
    importlib.reload(event_engine)
    
    assert event_engine.get_active_backend() == "python", \
        "Python backend should be active when OMEGA_USE_RUST_EVENT_ENGINE=0"
```

#### 4.5.3 Determinismus-Test (Kritisch!)

```python
# tests/golden/test_golden_backtest_rust.py

import pytest
import hashlib
import json

def test_rust_determinism_matches_golden():
    """
    KRITISCH: Rust-Backtest muss bit-genau identisch zu Golden-File sein.
    """
    # Run backtest with Rust backend
    os.environ["OMEGA_USE_RUST_EVENT_ENGINE"] = "1"
    
    result_rust = run_backtest_fixture()
    
    # Run same backtest with Python backend
    os.environ["OMEGA_USE_RUST_EVENT_ENGINE"] = "0"
    
    result_python = run_backtest_fixture()
    
    # Compare critical metrics
    assert result_rust["final_balance"] == result_python["final_balance"], \
        f"Balance mismatch: Rust={result_rust['final_balance']}, Python={result_python['final_balance']}"
    
    assert result_rust["total_trades"] == result_python["total_trades"], \
        f"Trades mismatch: Rust={result_rust['total_trades']}, Python={result_python['total_trades']}"
    
    assert result_rust["winrate"] == pytest.approx(result_python["winrate"], abs=1e-6), \
        f"Winrate mismatch"
```

### Phase 6: Documentation & Finalization (Tag 10, ~4h)

- [ ] Docstrings aktualisieren
- [ ] Rustdoc für Event-Modul
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisieren
- [ ] Benchmark-Ergebnisse dokumentieren

---

## 5. Rust-Implementation Details

### 5.1 Module-Struktur

```rust
// src/rust_modules/omega_rust/src/event/mod.rs

mod types;
mod queue;
mod engine;
mod callbacks;

pub use engine::EventEngineRust;
pub use types::{EventEngineStats, CandleData};
```

### 5.2 Cargo.toml Dependencies

```toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
parking_lot = "0.12"
# Optional für future batch processing:
# rayon = "1.10"
```

### 5.3 Error Codes

Neue Error-Codes in `src/rust_modules/omega_rust/src/error.rs`:

```rust
#[derive(Debug, Clone)]
pub enum EventEngineError {
    InvalidCandleData { reason: String },
    WarmupIndexNotFound,
    CallbackFailed { callback: String, error: String },
    StateSyncError { reason: String },
}

impl From<EventEngineError> for OmegaError {
    fn from(e: EventEngineError) -> Self {
        match e {
            EventEngineError::InvalidCandleData { reason } => 
                OmegaError::InvalidParameter { reason },
            EventEngineError::WarmupIndexNotFound => 
                OmegaError::InvalidParameter { 
                    reason: "No candle found >= original_start_dt".to_string() 
                },
            EventEngineError::CallbackFailed { callback, error } =>
                OmegaError::CalculationError { 
                    reason: format!("Callback '{}' failed: {}", callback, error) 
                },
            EventEngineError::StateSyncError { reason } =>
                OmegaError::StateError { reason },
        }
    }
}
```

---

## 6. Python-Integration Details

### 6.1 Callback-Wrapper für GIL-Management

```python
class _StrategyCallbackWrapper:
    """
    Wrapper to safely call Python strategy from Rust.
    Handles GIL acquisition and error propagation.
    """
    
    def __init__(self, strategy: StrategyWrapper):
        self._strategy = strategy
    
    def evaluate(self, index: int, slice_map: dict) -> Any:
        """Called from Rust with GIL held."""
        try:
            return self._strategy.evaluate(index, slice_map)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Strategy callback failed: {e}")
            raise
```

### 6.2 Type Conversion Helpers

```python
def _candle_to_dict(candle: Candle) -> dict:
    """Convert Candle object to dict for Rust interop."""
    return {
        "timestamp": int(candle.timestamp.timestamp() * 1_000_000),
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": float(candle.volume),
    }
```

---

## 7. Test-Strategie

### 7.1 Test-Matrix

| Test-Kategorie | Anzahl Tests | Kritikalität | CI-Blocker |
|----------------|-------------|--------------|------------|
| Rust Unit Tests | ~20 | Mittel | Ja |
| Python Integration | ~15 | Hoch | Ja |
| Backend Verification | 3 | Kritisch | Ja |
| Golden/Determinismus | 5 | **Kritisch** | **Ja** |
| Benchmarks | ~10 | Mittel | Nein |
| Parity Tests | ~10 | Hoch | Ja |

### 7.2 CI-Pipeline Integration

```yaml
# .github/workflows/ci.yml (Ergänzung)

  test-event-engine-rust:
    runs-on: ubuntu-latest
    steps:
      - name: Test Rust backend active
        run: |
          export OMEGA_USE_RUST_EVENT_ENGINE=1
          pytest tests/integration/test_event_engine_backend_verify.py -v
      
      - name: Test determinism
        run: |
          pytest tests/golden/test_golden_backtest.py -v
      
      - name: Test parity
        run: |
          pytest tests/test_event_engine_parity.py -v
```

---

## 8. Validierung & Akzeptanzkriterien

### 8.1 Funktionale Kriterien

| Kriterium | Ziel | Validierung |
|-----------|------|-------------|
| Determinismus | 100% identisch | Golden-File Hash Match |
| Alle bestehenden Tests | Grün | pytest CI |
| Strategy Kompatibilität | Alle Strategien | E2E Tests |
| Multi-Symbol Support | CrossSymbolEventEngine | Integration Tests |

### 8.2 Performance-Kriterien

| Metrik | Python Baseline | Rust Target | Akzeptanz |
|--------|-----------------|-------------|-----------|
| Loop Overhead | 32ms (20k bars) | <10ms | ≥3x Speedup |
| Full Backtest | 337ms | <250ms | ≥1.3x Speedup |
| Memory Usage | ~3MB | ≤3MB | Nicht schlechter |
| Callback Overhead | N/A | <5% total | Max 5% Overhead |

### 8.3 Quality Gates

- [ ] `cargo test` bestanden (0 failures)
- [ ] `cargo clippy --all-targets` (0 warnings)
- [ ] `pytest -q` bestanden (alle Tests)
- [ ] mypy --strict für Python-Wrapper
- [ ] Golden-File Tests: 100% Match
- [ ] Backend Verification Tests: Grün in CI

---

## 9. Rollback-Plan

### 9.1 Sofort-Rollback (< 1 Minute)

```bash
# Option 1: Feature-Flag deaktivieren
export OMEGA_USE_RUST_EVENT_ENGINE=false

# Option 2: In Code
# src/backtest_engine/core/event_engine.py
# _RUST_AVAILABLE = False  # Force Python
```

### 9.2 Rollback-Trigger

| Trigger | Schwellwert | Aktion |
|---------|-------------|--------|
| Determinismus-Bruch | Jeder | **Sofort-Rollback** |
| Golden-File Mismatch | Jeder | **Sofort-Rollback** |
| Performance-Regression | > 10% langsamer | Analyse → ggf. Rollback |
| Runtime Error in Prod | Jeder | Sofort-Rollback |
| Callback-Fehler | > 1% | Analyse → ggf. Rollback |

### 9.3 Post-Rollback Prozess

1. Issue erstellen mit Reproduktionsschritten
2. Golden-File-Diff analysieren
3. Root-Cause identifizieren
4. Fix entwickeln + Regression-Test
5. Re-Deployment nach Validierung

---

## 10. Checklisten

### 10.1 Pre-Implementation Checklist

- [x] FFI-Spezifikation finalisiert (`docs/ffi/event_engine.md`)
- [x] Migration Runbook vorhanden (`docs/runbooks/event_engine_migration.md`)
- [x] Benchmarks vorhanden (`tests/benchmarks/test_bench_event_engine.py`)
- [x] Performance-Baseline dokumentiert (`reports/performance_baselines/p0-01_event_engine.json`)
- [x] Golden-Tests vorhanden (`tests/golden/test_golden_backtest.py`)
- [x] Wave 0-2 abgeschlossen (Dependencies)
- [x] Lessons Learned dokumentiert (dieses Dokument)
- [ ] Lokale Entwicklungsumgebung verifiziert (Rust 1.75+)

### 10.2 Implementation Checklist

#### Phase 1: Setup
- [ ] Verzeichnisstruktur erstellen (`src/rust_modules/omega_rust/src/event/`) <!-- docs-lint:planned -->
- [ ] Cargo.toml Dependencies hinzufügen
- [ ] `mod.rs` erstellen und in `lib.rs` registrieren

#### Phase 2: Core Structures
- [ ] `types.rs` implementieren (CandleData, TradeSignalRust, EventEngineStats)
- [ ] `queue.rs` implementieren (optional für future batching)
- [ ] PyO3 Bindings für Type Conversions

#### Phase 3: Engine Implementation
- [ ] `engine.rs` implementieren (EventEngineRust)
- [ ] `callbacks.rs` implementieren (Python Callback Bridge)
- [ ] GIL-Management für Callbacks
- [ ] `cargo test` bestanden
- [ ] `cargo clippy` bestanden

#### Phase 4: Python Integration
- [ ] `event_engine.py` erweitern
- [ ] Feature-Flag implementieren (`OMEGA_USE_RUST_EVENT_ENGINE`)
- [ ] `get_active_backend()` Funktion (Wave 1 Learning)
- [ ] Callback-Wrapper implementieren
- [ ] Fallback zu Python bei Fehler

#### Phase 5: Testing
- [ ] Backend Verification Tests (Wave 1 Learning)
- [ ] Determinismus/Golden-File Tests
- [ ] Parity Tests (Rust vs Python)
- [ ] Integration Tests
- [ ] Benchmark Tests

#### Phase 6: Finalization
- [ ] Dokumentation aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] Code-Review abgeschlossen

### 10.3 Post-Implementation Checklist

- [ ] Performance-Vergleich dokumentiert
- [ ] Lessons Learned für Wave 4 notiert
- [ ] Sign-off Matrix ausgefüllt
- [ ] PR merged und tagged

### 10.4 Sign-off Matrix

| Rolle | Name | Datum | Status |
|-------|------|-------|--------|
| Developer | - | - | ⏳ Pending |
| FFI-Spec Review | - | - | ⏳ Pending |
| Unit Tests | pytest | - | ⏳ Pending |
| Golden Tests | pytest | - | ⏳ Pending |
| Determinism Validation | - | - | ⏳ Pending |
| Performance Validation | - | - | ⏳ Pending |
| Security Review | cargo clippy | - | ⏳ Pending |
| Tech Lead | axelkempf | - | ⏳ Pending |

---

## 11. Zeitplan

| Tag | Phase | Aufgaben |
|-----|-------|----------|
| 1 | Setup | Rust-Modul Setup, Dependencies, Type Definitions |
| 2-3 | Core Structures | CandleData, TradeSignalRust, Type Conversions |
| 3-5 | Engine Implementation | EventEngineRust, Callback Bridge, GIL Management |
| 6-7 | Python Integration | Feature-Flag, Wrapper, Fallback |
| 8-9 | Testing | Golden Tests, Parity Tests, Backend Verification |
| 10-12 | Finalization + Buffer | Benchmarks, Docs, Fixes, Review |

**Geschätzter Aufwand:** 8-12 Arbeitstage

---

## 12. Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Determinismus-Bruch | Mittel | **Kritisch** | Golden-File Tests, Mehrfach-Ausführung |
| Callback-Overhead zu hoch | Hoch | Hoch | Batch-API Future, GIL-Management |
| State-Sync-Probleme | Mittel | Hoch | Klare State-Ownership, Tests |
| Memory-Leaks bei Callbacks | Niedrig | Mittel | PyO3 RAII, Valgrind |
| Strategy-Inkompatibilität | Niedrig | Mittel | Adapter-Pattern, Feature-Flag |
| CrossSymbol nicht integriert | Mittel | Mittel | Phase 4.1 explizit |

---

## 13. Future Optimizations (Post-Wave 3)

### 13.1 Batch Processing (Wave 3.1)

Nach erfolgreicher Integration kann ein Batch-API implementiert werden:

```rust
/// Batch-verarbeite mehrere Events für reduzierten FFI-Overhead
pub fn process_batch(
    &mut self,
    py: Python<'_>,
    start_idx: usize,
    end_idx: usize,
    strategy: PyObject,
) -> PyResult<Vec<TradeSignalRust>> { ... }
```

### 13.2 SIMD-Optimierung (Wave 3.2)

Für Exit-Checks könnte SIMD verwendet werden:

```rust
// Parallel exit check für alle Positionen
pub fn evaluate_exits_simd(
    &self,
    positions: &[Position],
    bid: f64,
    ask: f64,
) -> Vec<ExitResult> { ... }
```

### 13.3 Full Rust Strategy (Wave 4+)

Langfristig könnten Strategien auch in Rust implementiert werden:

```rust
trait RustStrategy {
    fn evaluate(&self, slice: &DataSlice) -> Option<TradeSignal>;
}
```

---

## 14. References

- [FFI Specification: EventEngine](./ffi/event_engine.md)
- [Migration Runbook: EventEngine](./runbooks/event_engine_migration.md)
- [Performance Baseline](../reports/performance_baselines/p0-01_event_engine.json)
- [Benchmark Suite](../tests/benchmarks/test_bench_event_engine.py)
- [Golden-Tests](../tests/golden/test_golden_backtest.py)
- [Wave 0: Slippage & Fee](./WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md)
- [Wave 1: IndicatorCache](./WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md)
- [Wave 2: Portfolio](./WAVE_2_PORTFOLIO_IMPLEMENTATION_PLAN.md)
- [ADR-0001: Migration Strategy](./adr/ADR-0001-migration-strategy.md)
- [ADR-0003: Error Handling](./adr/ADR-0003-error-handling.md)

---

## Änderungshistorie

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-10 | 1.0 | Initiale Version basierend auf Wave 0-2 Learnings | AI Agent (Claude Opus 4.5) |

---

*Document Status: 🔵 READY FOR IMPLEMENTATION*
