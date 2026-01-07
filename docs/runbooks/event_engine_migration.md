# Migration Runbook: EventEngine

**Python-Pfad:** `src/backtest_engine/core/event_engine.py`  
**Zielsprache:** Rust  
**FFI-Integration:** PyO3/Maturin  
**Priorität:** Medium  
**Geschätzter Aufwand:** XL (Extra Large)  
**Status:** 🔴 Nicht begonnen

---

## Executive Summary

`EventEngine` ist die zentrale Event-Loop für das Backtest-System. 
Sie orchestriert den Datenfluss zwischen Candle-Events, Strategie-Signalen und Execution.
Die Migration zu Rust soll einen **3-5x Speedup** für die Event-Dispatch-Logik erreichen, 
wobei die Strategie-Logik in Python bleibt (Callback-basiert).

**Achtung:** Dies ist ein hochsensibles Core-Modul. Die Migration erfordert besondere Sorgfalt 
bei der Interface-Definition und dem Determinismus-Nachweis.

---

## Vorbedingungen

### Typ-Sicherheit
- [x] Modul ist mypy --strict compliant
- [x] Alle öffentlichen Funktionen haben vollständige Type Hints
- [x] TypedDict/Protocol-Definitionen in `src/backtest_engine/core/types.py`

### Interface-Dokumentation
- [x] FFI-Spezifikation in `docs/ffi/event_engine.md`
- [x] Arrow-Schemas definiert in `src/shared/arrow_schemas.py`
- [x] Nullability-Konvention dokumentiert

### Test-Infrastruktur
- [x] Benchmark-Suite in `tests/benchmarks/test_bench_event_engine.py`
- [x] Property-Based Tests in `tests/property/test_property_indicators.py`
- [x] Golden-File Tests in `tests/golden/test_golden_backtest.py`
- [x] Test-Coverage ≥ 85%

### Performance-Baselines
- [x] Baseline in `reports/performance_baselines/p0-01_event_engine.json`
- [x] Improvement-Target definiert (siehe unten)

---

## Performance-Baseline

**Quelle:** `reports/performance_baselines/p0-01_event_engine.json`  
**Test-Parameter:** Variabel (siehe FFI-Spezifikation)

| Operation | Typische Zeit | Target Speedup | Priorität |
|-----------|---------------|----------------|-----------|
| Event Dispatch | ~0.34s (50k events) | 5x | High |
| State Management | Inkludiert | 3x | Medium |
| Callback Invocation | Python-bound | 1x (no change) | Low |

**Hinweis:** Die größten Gewinne kommen aus der Event-Loop-Optimierung, 
nicht aus der Callback-Invocation (bleibt Python).

---

## Architektur-Übersicht

### Aktueller Python-Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EventEngine (Python)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │ DataLoader  │───▶│ Event Queue │───▶│ Strategy Callback (Python)  │ │
│  │ (Candles)   │    │ (List)      │    │ - on_candle()               │ │
│  └─────────────┘    └─────────────┘    │ - check_exit()              │ │
│                            │           └─────────────────────────────┘ │
│                            ▼                                            │
│                     ┌─────────────────┐    ┌─────────────────────────┐ │
│                     │ IndicatorCache  │───▶│ ExecutionSimulator      │ │
│                     │ (Modul 1)       │    │ (Modul 2)               │ │
│                     └─────────────────┘    └─────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Geplanter Hybrid-Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      EventEngine (Rust + Python)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    RustEventLoop (Rust Core)                       │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │ │
│  │  │ Arrow IPC   │───▶│ Event Queue │───▶│ Batch Dispatcher    │   │ │
│  │  │ (Zero-Copy) │    │ (VecDeque)  │    │ (Rayon Parallel)    │   │ │
│  │  └─────────────┘    └─────────────┘    └──────────┬──────────┘   │ │
│  │                                                    │              │ │
│  └────────────────────────────────────────────────────┼──────────────┘ │
│                                                       │                 │
│                           ┌───────────────────────────▼───────────────┐ │
│                           │        Python Callback Bridge              │ │
│                           │  - on_candle() (per Bar)                   │ │
│                           │  - on_batch() (optional, für Vectorisierung)│ │
│                           └───────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Migration Strategies

Die FFI-Spezifikation (`docs/ffi/event_engine.md`) definiert drei mögliche Strategien:

### Strategy A: Full Rust Core (Recommended)

**Vorteile:**
- Maximaler Performance-Gewinn
- Klare FFI-Boundary

**Nachteile:**
- Komplexere Callback-Bridge
- Strategie-Entwicklung erfordert Python-Wrapper

### Strategy B: Hybrid (Rust Loop + Python Callbacks)

**Vorteile:**
- Strategie-Code bleibt unverändert
- Inkrementelle Migration

**Nachteile:**
- FFI-Overhead pro Event
- Geringerer Speedup

### Strategy C: Batch Processing

**Vorteile:**
- Reduzierter FFI-Overhead
- Vectorisierung möglich

**Nachteile:**
- Strategie muss Batch-API unterstützen
- Komplexere Implementierung

**Empfehlung:** Starte mit **Strategy B** (Hybrid), evaluiere Performance, 
dann optional Upgrade zu **Strategy A** oder **C**.

---

## Migration Steps

### Step 1: Rust Modul Setup

```bash
cd src/rust_modules/omega_rust

# Neues Event-Modul erstellen
mkdir -p src/event
touch src/event/mod.rs
touch src/event/engine.rs
touch src/event/queue.rs
touch src/event/types.rs
```

- [ ] Cargo.toml Dependencies (parking_lot für Mutex)
- [ ] lib.rs Module-Deklaration
- [ ] event/mod.rs Exports

### Step 2: Interface Implementation

**Event-Typen (Rust):**

```rust
// src/rust_modules/omega_rust/src/event/types.rs

use std::collections::HashMap;

/// Candle-Event für FFI
#[derive(Clone, Debug)]
pub struct CandleEvent {
    pub timestamp: i64,  // Epoch microseconds
    pub timeframe: String,
    pub price_type: String,  // "bid" | "ask"
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Signal von Strategie
#[derive(Clone, Debug)]
pub enum SignalType {
    Long,
    Short,
    Close,
    None,
}

#[derive(Clone, Debug)]
pub struct StrategySignal {
    pub signal_type: SignalType,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub metadata: HashMap<String, String>,
}

/// Execution-Result
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub executed: bool,
    pub fill_price: f64,
    pub slippage: f64,
    pub commission: f64,
}
```

- [ ] Arrow Schema Mapping
- [ ] Python TypedDict ↔ Rust Struct Conversion
- [ ] Nullability Handling

### Step 3: Core-Logik portieren

**Event Queue (Rust):**

```rust
// src/rust_modules/omega_rust/src/event/queue.rs

use std::collections::VecDeque;
use super::types::CandleEvent;

pub struct EventQueue {
    events: VecDeque<CandleEvent>,
    capacity: usize,
}

impl EventQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, event: CandleEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
    
    pub fn pop(&mut self) -> Option<CandleEvent> {
        self.events.pop_front()
    }
    
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
```

**Event Engine (Rust Core):**

```rust
// src/rust_modules/omega_rust/src/event/engine.rs

use pyo3::prelude::*;
use super::queue::EventQueue;
use super::types::{CandleEvent, StrategySignal};

#[pyclass]
pub struct RustEventEngine {
    queue: EventQueue,
    current_bar_index: usize,
}

#[pymethods]
impl RustEventEngine {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: EventQueue::new(capacity),
            current_bar_index: 0,
        }
    }
    
    /// Lädt Events aus Arrow RecordBatch
    pub fn load_events(&mut self, arrow_buffer: &[u8]) -> PyResult<usize> {
        // Arrow IPC deserialisieren
        // Events in Queue laden
        todo!()
    }
    
    /// Führt Event-Loop aus mit Python-Callback
    pub fn run(
        &mut self,
        py: Python<'_>,
        strategy_callback: PyObject,
    ) -> PyResult<Vec<StrategySignal>> {
        let mut signals = Vec::new();
        
        while let Some(event) = self.queue.pop() {
            // GIL freigeben für Python-Callback
            let signal: StrategySignal = py.allow_threads(|| {
                // Callback in Python aufrufen
                Python::with_gil(|py| {
                    let result = strategy_callback
                        .call1(py, (/* event args */))
                        .expect("Callback failed");
                    // Result parsen
                    todo!()
                })
            });
            
            signals.push(signal);
            self.current_bar_index += 1;
        }
        
        Ok(signals)
    }
}
```

- [ ] Event Queue Implementation
- [ ] Event Engine Core
- [ ] Callback Bridge
- [ ] State Management
- [ ] Error Propagation

### Step 4: FFI-Bindings

```rust
// src/rust_modules/omega_rust/src/lib.rs

mod event;

use pyo3::prelude::*;
use event::engine::RustEventEngine;

#[pymodule]
fn omega_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustEventEngine>()?;
    Ok(())
}
```

- [ ] PyO3 Class Export
- [ ] Callback Protocol
- [ ] Error Handling

### Step 5: Python-Wrapper

```python
# src/backtest_engine/core/event_engine.py

from typing import Protocol, Callable, Optional
import numpy as np

# Feature Flag
USE_RUST_EVENT_ENGINE = False  # Konservativ: erst nach Validierung

class StrategyCallback(Protocol):
    """Protocol für Strategie-Callbacks."""
    def on_candle(self, candle: dict, indicators: dict) -> dict:
        ...


class EventEngine:
    """Event Engine mit optionalem Rust-Backend."""
    
    def __init__(self, use_rust: Optional[bool] = None):
        self._use_rust = use_rust if use_rust is not None else USE_RUST_EVENT_ENGINE
        self._rust_engine = None
        
        if self._use_rust:
            try:
                from omega_rust import RustEventEngine
                self._rust_engine = RustEventEngine(capacity=100_000)
            except ImportError:
                self._use_rust = False
    
    def run(
        self,
        candles: np.ndarray,
        strategy: StrategyCallback,
    ) -> list[dict]:
        """Führt Backtest-Loop aus."""
        if self._use_rust and self._rust_engine is not None:
            return self._run_rust(candles, strategy)
        return self._run_python(candles, strategy)
    
    def _run_rust(self, candles, strategy):
        """Rust-Backend."""
        # Arrow IPC serialisieren
        arrow_buffer = self._to_arrow(candles)
        self._rust_engine.load_events(arrow_buffer)
        
        # Wrapper für Callback
        def callback_wrapper(*args):
            return strategy.on_candle(*args)
        
        return self._rust_engine.run(callback_wrapper)
    
    def _run_python(self, candles, strategy):
        """Python-Fallback."""
        results = []
        for i, candle in enumerate(candles):
            result = strategy.on_candle(candle, {})
            results.append(result)
        return results
```

- [ ] Feature-Flag Implementation
- [ ] Callback Wrapper
- [ ] Arrow Serialization
- [ ] Error Handling

### Step 6: Testing

```bash
# Tests (bestehende)
pytest tests/integration/test_event_bus_integration.py -v

# Property-Based Tests
pytest tests/property/ -v

# Golden-File Tests (Determinismus - KRITISCH!)
pytest tests/golden/test_golden_backtest.py -v

# Benchmarks
pytest tests/benchmarks/test_bench_event_engine.py --benchmark-json=results.json

# Rust Tests
cd src/rust_modules/omega_rust
cargo test

# Weitere Integration-Checks
pytest tests/integration/test_resume_semantics.py -v
```

- [ ] Alle Unit-Tests passieren
- [ ] Property-Based Tests passieren
- [ ] **Golden-File Tests passieren (KRITISCH für Determinismus)**
- [ ] Benchmark zeigt ≥3x Speedup
- [ ] Rust `cargo test` passiert
- [ ] Integration mit Backtest-Engine
- [ ] Integration mit IndicatorCache
- [ ] Integration mit ExecutionSimulator

### Step 7: Documentation

- [ ] Docstrings in Python aktualisiert
- [ ] Rustdoc für Rust-Modul
- [ ] FFI-Dokumentation in `docs/ffi/event_engine.md` aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] Strategie-Entwickler-Guide aktualisiert

---

## Rollback-Plan

### Bei Fehler in Produktion

1. **Sofortmaßnahme:** Feature-Flag deaktivieren
   ```python
   # src/backtest_engine/core/event_engine.py
   USE_RUST_EVENT_ENGINE = False
   ```

2. **Fallback:** Python-Implementation wird automatisch verwendet

3. **Analyse:**
   - Golden-File-Diff erstellen
   - Determinismus-Bruch lokalisieren
   - Issue mit Reproduktions-Script

4. **Fix:**
   - Bugfix in Rust
   - Golden-File validieren
   - Regression-Test hinzufügen

### Bei Determinismus-Bruch

**KRITISCH:** Event Engine muss 100% deterministisch sein!

1. Sofort Rollback zu Python
2. Golden-File-Diff analysieren
3. Root-Cause identifizieren:
   - Float-Rounding?
   - Event-Reihenfolge?
   - Seed-Propagation?
4. Fix verifizieren mit mehrfacher Ausführung

### Bei Performance-Regression

1. Benchmark-History prüfen
2. Rust Profiling mit `perf` oder `flamegraph`
3. Callback-Overhead analysieren
4. Bei >10% Regression: Rollback

---

## Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Determinismus-Bruch | Mittel | **Kritisch** | Golden-File Tests; Mehrfach-Ausführung |
| Callback-Overhead zu hoch | Hoch | Hoch | Batch-API; GIL-Management |
| State-Sync-Probleme | Mittel | Hoch | Klare State-Ownership; Tests |
| Memory-Leaks bei Callbacks | Niedrig | Mittel | PyO3 RAII; miri; Valgrind |
| Strategy-Inkompatibilität | Niedrig | Mittel | Adapter-Pattern; Feature-Flag |

---

## Akzeptanzkriterien

### Funktional
- [ ] Alle bestehenden Backtest-Tests passieren
- [ ] **Backtest-Determinismus 100% erhalten** (Golden-Files)
- [ ] Callback-Interface kompatibel mit allen Strategien
- [ ] State-Management identisch

### Performance
- [ ] Event-Dispatch: ≥3x Speedup
- [ ] Callback-Overhead: <5% zusätzlich
- [ ] Memory-Usage ≤ Python-Baseline
- [ ] Keine Memory-Leaks

### Qualität
- [ ] Code Review bestanden (2+ Reviewer)
- [ ] mypy --strict für Python-Wrapper
- [ ] clippy --pedantic für Rust (0 Warnings)
- [ ] Dokumentation vollständig
- [ ] Strategie-Entwickler-Guide aktualisiert

---

## Dependencies zu anderen Modulen

```
EventEngine (dieses Modul)
    │
    ├── IndicatorCache (P5-06)
    │   └── Muss VOR EventEngine migriert werden
    │
    ├── ExecutionSimulator (noch nicht geplant)
    │   └── Kann NACH EventEngine migriert werden
    │
    └── Portfolio (noch nicht geplant)
        └── Kann parallel migriert werden
```

**Empfohlene Migrations-Reihenfolge:**
1. IndicatorCache (unabhängig, klare Grenzen)
2. EventEngine (Core-Loop)
3. ExecutionSimulator (Trade-Matching)
4. Portfolio (State-Management)

---

## Referenzen

- FFI-Spezifikation: `docs/ffi/event_engine.md`
- Performance-Baseline: `reports/performance_baselines/p0-01_event_engine.json`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
- Rust-Modul: `src/rust_modules/omega_rust/src/event/`
- ADR-0001: Migration Strategy
- ADR-0002: Serialization Format
- ADR-0003: Error Handling
- ADR-0004: Build System

---

## Changelog

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-05 | 1.0 | Initiale Version des Runbooks | Omega Team |
