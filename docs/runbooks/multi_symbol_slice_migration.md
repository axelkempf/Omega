# Migrations-Runbook: MultiSymbolSlice (Rust)

**Modul:** `src/backtest_engine/core/multi_symbol_slice.py`  
**Target-Sprache:** Rust  
**Priorität:** Wave 6 (Performance Cleanup)  
**Aufwand:** L (Large)  
**Status:** 🟢 READY FOR MIGRATION

---

## 1. Übersicht

MultiSymbolSlice ist die Datenstruktur für synchronisierte Multi-Symbol-Candle-Daten. Sie ermöglicht Cross-Symbol-Strategien und Multi-Asset-Backtests.

### 1.1 Aktuelle Architektur

```
┌─────────────────┐    ┌────────────────────┐    ┌────────────────┐
│   SymbolData    │───▶│  MultiSymbolSlice  │───▶│  EventEngine   │
│   (per Symbol)  │    │  Iterator          │    │  (Dispatch)    │
└─────────────────┘    └────────────────────┘    └────────────────┘
         │
         ▼
    ┌─────────────┐
    │ DataHandler │
    │ (Parquet)   │
    └─────────────┘
```

### 1.2 Warum Migration?

| Metrik | Python Baseline | Rust Target | Speedup |
|--------|-----------------|-------------|---------|
| Iterator Step (50 Symbols) | 0.9ms | <0.05ms | 18x |
| Full Iteration (1M steps) | 900s | <50s | 18x |
| Memory per Slice | 2.5KB | <0.5KB | 5x |

### 1.3 Abhängigkeiten

- **Upstream:** DataHandler (Parquet Loading)
- **Downstream:** EventEngine (Iteration), IndicatorCache (Data Access)
- **FFI:** Arrow IPC für Batch-Data-Transfer

---

## 2. Vorbereitungs-Checkliste

### 2.1 Type Safety ✅

- [x] Core types in `core/types.py` definiert
- [x] Mypy --strict für Modul aktiviert
- [x] Protocol-Klassen für Iterator-Interface
- [x] TypedDict für Snapshot-Daten

### 2.2 Test Coverage ✅

- [x] Unit Tests vorhanden
- [x] Iterator-Edge-Cases getestet
- [x] Multi-Symbol-Synchronisation getestet
- [x] Coverage ≥ 85%

### 2.3 Performance Baseline ✅

- [x] Benchmark-Suite vorhanden
- [x] Baselines dokumentiert
- [x] Scaling-Verhalten (10/50/100 Symbole) gemessen

### 2.4 FFI-Dokumentation ✅

- [x] Interface-Spec: `docs/ffi/multi_symbol_slice.md`
- [x] Arrow-Schema definiert
- [x] Nullability dokumentiert
- [x] Error-Codes definiert

---

## 3. Rust-Architektur

### 3.1 Modul-Struktur

```
src/rust_modules/omega_rust/src/
├── data/
│   ├── mod.rs               # Modul-Exports
│   ├── multi_symbol.rs      # MultiSymbolSlice, Iterator
│   ├── candle.rs            # CandleData Type
│   └── sync.rs              # Timestamp-Synchronisation
└── lib.rs
```

### 3.2 Core Types

```rust
use std::collections::HashMap;
use chrono::{DateTime, Utc};

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

pub struct MultiSymbolDataIterator {
    data: Arc<MultiSymbolData>,
    current_index: usize,
    timestamps: Vec<DateTime<Utc>>,
}

impl Iterator for MultiSymbolDataIterator {
    type Item = MultiSymbolSlice;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Efficient slice creation
    }
}
```

### 3.3 Memory-Optimierung

```rust
// Zero-Copy Slice via Arc
pub struct MultiSymbolSliceRef<'a> {
    timestamp: DateTime<Utc>,
    data: &'a MultiSymbolData,
    symbol_indices: &'a [usize],
}

// Reuse allocations across iterations
pub struct MultiSymbolDataIterator {
    // Pre-allocated buffer for current slice
    buffer: MultiSymbolSlice,
    // ...
}
```

---

## 4. Migration Steps

### Phase 1: Type Hardening (1-2 Tage)

1. Finalisiere TypedDict-Schemas in `core/types.py`
2. Aktiviere mypy --strict für multi_symbol_slice.py
3. Behebe Type-Errors (falls vorhanden)
4. Dokumentiere carve-outs (falls nötig)

### Phase 2: Rust Scaffold (1-2 Tage)

1. Erstelle `src/rust_modules/omega_rust/src/data/` Verzeichnis
2. Definiere Core Types in Rust
3. Implementiere Arrow Schema Parsing
4. PyO3 Bindings für MultiSymbolSliceRust

### Phase 3: Iterator Implementation (3-4 Tage)

1. Implementiere `MultiSymbolDataIteratorRust`
2. Timestamp-Synchronisation logik
3. Effiziente Slice-Erstellung
4. Memory-Pool für Allocation-Reuse

### Phase 4: Zero-Copy Optimierung (2-3 Tage)

1. Arrow IPC Deserialisierung ohne Kopie
2. Slice-References statt Kopien
3. SIMD für Daten-Lookup
4. Cache-freundliches Memory-Layout

### Phase 5: Integration (2-3 Tage)

1. Python wrapper in `shared/ffi_wrapper.py`
2. Feature flag: `OMEGA_USE_RUST_MULTI_SYMBOL`
3. Fallback-Mechanismus
4. Logging und Metrics

### Phase 6: Testing (3-4 Tage)

1. Port Unit-Tests zu Rust
2. Property-Based Tests:
   - Iterator-Determinismus
   - Slice-Korrektheit
   - Memory-Safety
3. Golden-File Tests für Multi-Symbol-Backtests
4. Performance-Benchmarks

### Phase 7: Rollout (2-3 Tage)

1. Staging-Tests mit realen Backtests
2. Vergleich Python vs Rust Ergebnisse
3. Gradual Rollout
4. Monitoring

---

## 5. Rollback-Plan

### 5.1 Rollback-Trigger

- Daten-Divergenz zwischen Python und Rust
- Iterator produziert unterschiedliche Sequenz
- Memory-Leak oder Crash
- Performance-Regression

### 5.2 Rollback-Prozedur

```bash
export OMEGA_USE_RUST_MULTI_SYMBOL=false
pytest tests/test_shared_protocols_runtime.py -v -k multi_symbol
pytest tests/benchmarks/test_bench_multi_symbol_slice.py -v
```

### 5.3 Validierung

1. Golden-File Tests müssen passen
2. Multi-Symbol-Backtest-Ergebnisse identisch
3. Memory-Usage normal

---

## 6. Akzeptanzkriterien

### 6.1 Funktional

- [ ] Iterator produziert identische Sequenz wie Python
- [ ] Alle Symbole korrekt synchronisiert
- [ ] Missing-Data korrekt gehandelt
- [ ] Edge-Cases (Start/Ende, Gaps) funktionieren

### 6.2 Performance

- [ ] Iterator Step: <0.05ms (50 Symbole)
- [ ] Full Iteration (1M): <50s
- [ ] Memory per Slice: <0.5KB

### 6.3 Memory-Safety

- [ ] Keine Memory-Leaks (Valgrind/ASAN)
- [ ] Keine Use-After-Free
- [ ] Thread-safe für parallele Backtests

---

## 7. Sign-Off

| Rolle | Name | Datum | Signatur |
| ----- | ---- | ----- | -------- |
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

---

## 8. Referenzen

- FFI-Spec: [multi_symbol_slice.md](../ffi/multi_symbol_slice.md)
- Arrow-Schemas: [arrow_schemas.py](../../src/shared/arrow_schemas.py)
- Data-Flow: [data-flow-diagrams.md](../ffi/data-flow-diagrams.md)
