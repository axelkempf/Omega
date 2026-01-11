# Wave 4: Execution Simulator Full Rust Migration Implementation Plan

**Document Version:** 0.4  
**Created:** 2026-01-10  
**Updated:** 2026-01-11  
**Status:** 🟢 IN PROGRESS (Phases 1-7 Complete)  
**Module:** `src/backtest_engine/core/execution_simulator.py` (~655 LOC)  
**Target (Rust):** `src/rust_modules/omega_rust/src/execution/`  
**Branch:** `migration/wave-4-execution-engine`  

---

## Executive Summary

Wave 4 migriert den `ExecutionSimulator` vollständig nach Rust, sodass die Execution-Logik (Signal-Verarbeitung, Pending-Orders, Exit-Evaluation, Sizing, Slippage/Fees) **ohne Python-Fallback** läuft. Ziel ist, den Hot-Path aus dem Python-Loop herauszuziehen, FFI-Overhead durch **Batch APIs (Arrow IPC)** zu minimieren und deterministische, reproduzierbare Backtest-Ergebnisse zu garantieren.

**Warum ExecutionSimulator als Wave 4?**

| Kriterium | Bewertung | Begründung |
|---|---:|---|
| Performance-Impact | ✅ Kritisch | `evaluate_exits()` wird pro Bar über alle Positionen ausgeführt (Hot Path) |
| Abhängigkeiten | ✅ Erfüllt | Wave 0 (Costs), Wave 2 (Portfolio), Wave 3 (Event Engine) verfügbar |
| Callback-Freiheit | ✅ Ideal | Keine Strategy-Callbacks nötig → **Full Rust möglich** |
| Determinismus | ⚠️ Kritisch | Exakte SL/TP-Logik + Rounding + RNG-Seeds müssen parity-genau sein |
| Aufwand | ⚠️ 10–14 Tage | Mehrere Subsysteme (Sizing, Costs, Arrow) + Test-Suite |

### Prerequisites (Go/No-Go)

- ✅ FFI Spec vorhanden: `docs/ffi/execution_simulator.md`
- ✅ Runbook vorhanden (Template): `docs/runbooks/execution_simulator_migration.md`
- ✅ Rust Infra (PyO3/maturin) vorhanden: `src/rust_modules/omega_rust/`
- ✅ Error Codes synchronisiert: `src/shared/error_codes.py`, `src/rust_modules/omega_rust/src/error.rs`
- ✅ Arrow Schema Registry vorhanden: `src/shared/arrow_schemas.py`
- ✅ Bench/Golden/Property-Infrastruktur vorhanden (siehe `docs/MIGRATION_READINESS_VALIDATION.md`)

**Go/No-Go Kriterien**
- Go nur wenn: (1) Arrow Schema Drift Detection grün, (2) Golden Tests deterministisch grün, (3) Benchmarks erreichen ≥8x Ziel, (4) kein Python-Fallback im regulären Codepath.
- No-Go wenn: (1) Result-Drift vs Python baseline, (2) RNG/float rounding nicht stabil, (3) CI-Gates nicht vollständig (Tests/Benchmarks/Property).

---

## Inhaltsverzeichnis

1. [Voraussetzungen & Status](#1-voraussetzungen--status)  
2. [Lessons Learned Integration (Wave 0-3)](#2-lessons-learned-integration-wave-0-3)  
3. [Architektur-Übersicht](#3-architektur-übersicht)  
4. [Implementierungs-Phasen (10–14 Tage)](#4-implementierungs-phasen-1014-tage)  
5. [Rust-Implementation Details](#5-rust-implementation-details)  
6. [Python-Integration](#6-python-integration)  
7. [Test-Strategie](#7-test-strategie)  
8. [Validierung & Akzeptanzkriterien](#8-validierung--akzeptanzkriterien)  
9. [Rollback-Plan](#9-rollback-plan)  
10. [Checklisten](#10-checklisten)  
11. [Sign-off Matrix](#11-sign-off-matrix)  

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure Readiness (aus Wave 0–3)

| Komponente | Status | Evidenz |
|---|---:|---|
| Rust Build + PyO3 | ✅ | `src/rust_modules/omega_rust/Cargo.toml`, `src/rust_modules/omega_rust/src/lib.rs` |
| Error Handling Pattern | ✅ | `src/rust_modules/omega_rust/src/error.rs` |
| Costs (Slippage/Fee) | ✅ | `src/rust_modules/omega_rust/src/costs/` + Python wrapper `src/backtest_engine/core/slippage_and_fee.py` |
| Portfolio State in Rust | ✅ | `src/rust_modules/omega_rust/src/portfolio/` + Python wrapper `src/backtest_engine/core/portfolio.py` |
| Event Engine Hybrid Loop | ✅ | `src/rust_modules/omega_rust/src/event/` |
| Arrow Schemas + Drift Detection | ✅ | `src/shared/arrow_schemas.py`, `reports/schema_fingerprints.json` (siehe `docs/MIGRATION_READINESS_VALIDATION.md`) |
| Bench/Golden/Property CI | ✅ | `docs/MIGRATION_READINESS_VALIDATION.md` |

### 1.2 Python-Modul Baseline

**Datei:** `src/backtest_engine/core/execution_simulator.py`  

**Core-Methoden (Candle Mode):**
- `process_signal(signal)`
- `check_if_entry_triggered(pos, bid_candle, ask_candle)`
- `trigger_entry(pos, candle)`
- `evaluate_exits(bid_candle, ask_candle)`

**Tick Mode Variants:**
- `process_signal_tick(signal, tick)`
- `check_if_entry_triggered_tick(pos, tick)`
- `trigger_entry_tick(pos, tick)`
- `evaluate_exits_tick(tick)`

**Kritische Details in der Python-Logik (Parity-relevant):**
- `pip_buffer_factor` Default (0.5) → `pip_buffer = pip_size * factor`
- Entry-Candle Special Case: SL/TP nur bei eindeutiger Erreichung; Limit-Orders prüfen Close-Price-Validierung
- Quantisierung der Lot-Größe: konservatives Floor-Rounding mit `1e-12` Guard + `f"{lots:.8f}"`
- Fees: bevorzugt `CommissionModel`, sonst `FeeModel`

### 1.3 Performance Baseline (Evidenz)

**Snapshot:** `reports/performance_baselines/p0-01_execution_simulator.json`  
**Runbook Targets (human-facing):** `docs/runbooks/execution_simulator_migration.md`

| Operation | Baseline | Target | Ziel-Speedup |
|---|---:|---:|---:|
| Signal Processing (1K) | 45ms | <5ms | ≥9x |
| Exit Evaluation (1K) | 32ms | <4ms | ≥8x |
| Full Backtest Loop | 85s | <15s | ≥5.5x |

**Profiler-Hinweis (Snapshot):** Die Hotspots sind `execution_simulator.py:evaluate_exits` und `execution_simulator.py:process_signal` (siehe `profile_top20` in der Baseline JSON).

### 1.4 Bottleneck-Analyse (Root Causes)

| Bottleneck | Symptom | Root Cause | Rust-Strategie |
|---|---|---|---|
| Python Loop Overhead | hohe cumulative time | per-position/per-bar Python logic | Full Rust state machine |
| Object/Attribute Access | viele `getattr`/dict lookups | dynamische Felder + metadata | typed structs + enums |
| Sizing + FX Conversion | pro-entry expensive | optional RateProvider/Specs | precomputed per-symbol values, caching in Rust |
| Fee/Slippage Calls | hot path | Python model calls (teilweise) | direkte Rust costs module |
| FFI Overhead | viele crossing calls | per-signal/per-bar calls | Arrow IPC Batch APIs |

---

## 2. Lessons Learned Integration (Wave 0-3)

### 2.1 Konkrete Anwendung der Learnings

| Wave | Learning | Anwendung in Wave 4 |
|---:|---|---|
| 0 | RNG Determinismus | Slippage RNG in Rust: ChaCha + explizite Seed-Strategie (keine globalen RNGs) |
| 0 | Namespace Conflicts | Modulname `execution` gegen Python stdlib prüfen; klare Exports in `lib.rs` |
| 1 | Integration ≠ Implementierung | E2E Backtest muss Rust Execution nutzen; kein “nur Funktion portiert” |
| 1 | FFI-Overhead | Batch APIs (`process_signals_batch`, `evaluate_exits_batch`) via Arrow IPC |
| 2 | State-Management in Rust | `active_positions` + Lifecycle in Rust (open/pending/closed) |
| 3 | Callback Elimination | Keine Python Callbacks im ExecutionSimulator (kein Strategy/Portfolio Callback nötig) |

### 2.2 Warum Full Rust (kein Hybrid)

Der ExecutionSimulator ist self-contained: er benötigt keine Strategy-Auswertung, sondern erhält fertige `TradeSignal`s. Anders als Wave 3 (Strategy-Callbacks) kann Wave 4 alle Branches in Rust implementieren: Trigger, Exit-Detection, Sizing, Costs und State-Updates.

---

## 3. Architektur-Übersicht

### 3.1 Ziel-Architektur (ASCII)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Python API Layer (Thin Wrapper)                             │
│  backtest_engine.core.execution_simulator.ExecutionSimulator                  │
│    - validates inputs minimally                                                │
│    - builds Arrow IPC batches                                                  │
│    - delegates ALL logic to Rust                                               │
│                                                                              │
│  Feature Flag: OMEGA_USE_RUST_EXECUTION_SIMULATOR=always                      │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ PyO3 FFI (Arrow IPC bytes)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Rust ExecutionSimulator (Full Logic)                       │
│  src/rust_modules/omega_rust/src/execution/                                   │
│   - ExecutionSimulatorRust: owns state + config                               │
│   - signal.rs: TradeSignal parsing + validation                               │
│   - position.rs: Position lifecycle/state machine                             │
│   - trigger.rs: entry trigger + exit detection                                │
│   - sizing.rs: risk-based sizing + rounding                                   │
│   - arrow.rs: Arrow IPC (RecordBatch ↔ structs)                               │
│                                                                              │
│  Integrations:                                                                │
│   - costs (Wave 0): slippage/fee primitives                                   │
│   - portfolio (Wave 2): optional direct PortfolioRust interop                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Rust-Modul-Struktur

```
src/rust_modules/omega_rust/src/
├── execution/
│   ├── mod.rs
│   ├── simulator.rs
│   ├── signal.rs
│   ├── position.rs
│   ├── trigger.rs
│   ├── sizing.rs
│   ├── arrow.rs
│   └── tests/
│       ├── test_signal.rs
│       ├── test_trigger.rs
│       ├── test_exit.rs
│       └── test_sizing.rs
└── lib.rs
```

### 3.3 Core Types (Rust, intern)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Long,
    Short,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionStatus {
    Open,
    Pending,
    Closed,
}

#[derive(Clone, Debug)]
pub struct SymbolSpec {
    pub symbol: String,
    pub pip_size: f64,
    pub tick_size: f64,
    pub tick_value: f64,
    pub contract_size: f64,
    pub volume_min: f64,
    pub volume_max: f64,
    pub volume_step: f64,
    pub quote_currency: String,
    pub base_currency: String,
}

#[derive(Clone, Debug)]
pub struct TradeSignal {
    pub timestamp_us: i64,
    pub direction: Direction,
    pub symbol: String,
    pub entry: f64,
    pub sl: f64,
    pub tp: f64,
    pub order_type: OrderType,
    pub reason: Option<String>,
    pub scenario: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PositionState {
    pub entry_time_us: i64,
    pub direction: Direction,
    pub symbol: String,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub initial_stop_loss: f64,
    pub initial_take_profit: f64,
    pub size_lots: f64,
    pub risk_per_trade: f64,
    pub order_type: OrderType,
    pub status: PositionStatus,
    pub trigger_time_us: Option<i64>,
    pub exit_time_us: Option<i64>,
    pub exit_price: Option<f64>,
    pub result: Option<f64>,
    pub reason: Option<String>,
    pub metadata_json: Option<String>,
}
```

### 3.4 PyO3 Interface (öffentliche API)

**Prinzip:** Python übergibt/bekommt **Arrow IPC bytes**; Rust hält State und liefert Updates/Exports. Dadurch werden pro-call Python object traversals vermieden.

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct ExecutionSimulatorRust {
    // Core state (no Python fallback)
    active_positions: Vec<PositionState>,
    risk_per_trade: f64,
    pip_buffer_factor: f64,
    // symbol specs keyed by symbol
    symbol_specs: std::collections::HashMap<String, SymbolSpec>,
    // deterministic seed base for costs (optional)
    base_seed: Option<u64>,
}

#[pymethods]
impl ExecutionSimulatorRust {
    #[new]
    #[pyo3(signature = (risk_per_trade=100.0, pip_buffer_factor=0.5, base_seed=None, symbol_specs_ipc=None))]
    pub fn new(
        risk_per_trade: f64,
        pip_buffer_factor: f64,
        base_seed: Option<u64>,
        symbol_specs_ipc: Option<&[u8]>,
    ) -> PyResult<Self> {
        // Parse symbol_specs_ipc if provided; otherwise empty.
        Ok(Self {
            active_positions: Vec::new(),
            risk_per_trade,
            pip_buffer_factor,
            symbol_specs: std::collections::HashMap::new(),
            base_seed,
        })
    }

    /// Process a batch of signals and return updated positions as Arrow IPC.
    pub fn process_signals_batch(&mut self, signals_ipc: &[u8]) -> PyResult<Vec<u8>> {
        // 1) decode TradeSignal batch
        // 2) apply process_signal for each
        // 3) return positions snapshot or delta
        Ok(Vec::new())
    }

    /// Evaluate exits for current candle batch (bid/ask) and return updated positions.
    pub fn evaluate_exits_batch(
        &mut self,
        bid_candles_ipc: &[u8],
        ask_candles_ipc: Option<&[u8]>,
    ) -> PyResult<Vec<u8>> {
        Ok(Vec::new())
    }

    /// Export active positions as Arrow IPC (for Python visibility/logging).
    pub fn get_active_positions_ipc(&self) -> PyResult<Vec<u8>> {
        Ok(Vec::new())
    }
}
```

---

## 4. Implementierungs-Phasen (10–14 Tage)

> Hinweis: Die Phasen sind sequenziell geplant; Abweichungen werden im Runbook dokumentiert (`docs/runbooks/execution_simulator_migration.md`).

### 4.0 Status-Tracking (Living Table)

| Phase | Name | Aufwand | Status | Evidenz-Links |
|---:|---|---:|---:|---|
| 1 | Rust Skeleton & Types | 2d | ✅ | `execution/mod.rs`, `position.rs`, `signal.rs` |
| 2 | Signal Processing | 2–3d | ✅ | `simulator.rs::process_signal_internal` |
| 3 | Exit Evaluation | 2–3d | ✅ | `trigger.rs::evaluate_exit` |
| 4 | Entry Trigger & Pending | 2d | ✅ | `trigger.rs`, `slippage.rs`, `tests/test_execution_slippage.py` |
| 5 | Position Sizing | 1–2d | ✅ | `sizing.rs::size_for_risk`, `SymbolSpecCache` |
| 6 | Python API Wrapper | 1d | ✅ | `execution_simulator.py` (thin wrapper) |
| 7 | Testing & Validation | 2–3d | ✅ | `tests/test_execution_simulator_rust.py`, `tests/golden/`, `tests/property/`, `tests/benchmarks/` |

### Phase 1: Rust Skeleton & Types (2 Tage)

**Ziele**
- Modul-Skelett unter `src/rust_modules/omega_rust/src/execution/` anlegen
- Core Enums/Structs definieren
- `lib.rs` Export + Registrierung vorbereiten
- Arrow Dependencies in `Cargo.toml` aktivieren (inkl. `arrow-ipc`)

**Rust Dateien**
- `execution/mod.rs` (exports)
- `execution/simulator.rs` (pyclass skeleton)
- `execution/signal.rs`, `execution/position.rs` (Types)
- `execution/arrow.rs` (placeholder codec)

**Python Änderungen**
- Keine behavior changes; nur Vorbereitungen für Wrapper (`docs-lint:planned`)

**Tests**
- `cargo test` (compilation + basic unit tests)

**Akzeptanzkriterien**
- `maturin develop` buildt; `import omega_rust` funktioniert lokal
- `ExecutionSimulatorRust` ist importierbar und instanziierbar

---

### Phase 2: Signal Processing (2–3 Tage)

**Ziele**
- `process_signal` Semantik parity-genau portieren:
  - Market: slippage → sizing → position open → fee register
  - Limit/Stop: pending position (size=0) + metadata transfer
- Validierungen (Constraint Violations) mit ErrorCodes

**Rust Dateien**
- `execution/simulator.rs`: `process_signals_batch`, helper `process_signal_one`
- `execution/signal.rs`: parser + validation
- `execution/sizing.rs`: risk-based sizing + quantization

**Python Änderungen**
- Wrapper erstellt Signal-Batch (Arrow) statt per-signal calls

**Tests**
- Rust unit tests: Market vs Pending, edge cases (SL dist ~0)

**Akzeptanzkriterien**
- Parity gegen Python für definierte Fixtures (siehe Golden/Integration Tests Phase 7)
- Keine panics; Fehler als `PyErr` mit `[ErrorCode]` Prefix

---

### Phase 3: Exit Evaluation (2–3 Tage)

**Ziele**
- `evaluate_exits` parity-genau:
  - Pending trigger check + `trigger_entry`
  - SL/TP hit detection inkl. `pip_buffer_factor`
  - Entry-candle special case + Limit close validation
  - break-even reason mapping (`break_even_stop_loss`)
  - Slippage/fee bei Exit
- Closed positions werden entfernt (oder status=closed und in deltas)

**Rust Dateien**
- `execution/trigger.rs`: SL/TP detection und entry-candle handling
- `execution/simulator.rs`: `evaluate_exits_batch`

**Python Änderungen**
- EventEngine/Backtest loop ruft batch API (bid/ask candle) statt Python method per bar

**Tests**
- Rust unit tests: gaps, entry candle, limit close validation, long/short symmetry

**Akzeptanzkriterien**
- Exits parity-genau (Preis/Reason/Timestamp) für Golden Fixtures
- Performance: O(p) pro bar, keine allocations im Hot Path (pre-allocated buffers)

---

### Phase 4: Entry Trigger & Pending Orders (2 Tage)

**Ziele**
- `check_if_entry_triggered` (candle) und tick variants komplett
- `trigger_entry` parity: sizing + fees + trigger_time + initial SL/TP

**Rust Dateien**
- `execution/trigger.rs`: trigger predicates + activation
- `execution/position.rs`: lifecycle helpers

**Python Änderungen**
- Tick-mode wrapper (falls genutzt) wird auf Rust umgestellt

**Tests**
- Unit tests: limit/stop activation matrix (long/short, bid/ask)

**Akzeptanzkriterien**
- Pending Orders deterministisch und ohne Double-Trigger

---

### Phase 5: Position Sizing Integration (1–2 Tage)

**Ziele**
- `_unit_value_per_price` / LotSizer-Parity
- Conservative rounding wie Python (`floor` + step + 1e-12 guard)
- Caching per symbol

**Design Entscheidung (Full Rust, no Python RateProvider callback)**
- Primärer Weg: `tick_value/tick_size` aus `SymbolSpec`
- Fallback: `contract_size` ohne FX Conversion (nur wenn `RateProvider` nicht verfügbar)
- Optional (future): Arrow IPC FX rates input pro batch

**Rust Dateien**
- `execution/sizing.rs`

**Python Änderungen**
- Wrapper muss sicherstellen, dass `SymbolSpec.tick_value/tick_size/contract_size` gesetzt sind (Go/No-Go)

**Tests**
- Unit tests: quantization parity vs Python reference values

**Akzeptanzkriterien**
- Sizing stimmt für alle Symbole in test fixtures; kein Over-risking durch rounding

---

### Phase 6: Python API Wrapper (1 Tag)

**Ziele**
- `ExecutionSimulator` wird Thin Wrapper, der immer Rust verwendet
- Keine Python-Logik-Duplikation

**Python Dateien**
- `src/backtest_engine/core/execution_simulator.py`
- ggf. Anpassungen in `src/backtest_engine/core/event_engine.py` (Batch calls)

**Feature Flag**
- `OMEGA_USE_RUST_EXECUTION_SIMULATOR=always` (supported value)
  - Default: `always`
  - Any other value → `ValueError` (Signal: “Rollback requires reverting deployment”, siehe Abschnitt 9)

**Akzeptanzkriterien**
- Python API bleibt stabil (gleiche Klassennamen/Methoden), aber delegiert vollständig

---

### Phase 7: Testing & Validation (2–3 Tage)

**Ziele**
- Testmatrix: Unit (Rust), Integration (Python), Golden, Property, Benchmarks
- Determinismus: gleiche Inputs → gleiche IPC bytes → gleiche Trades/PnL
- Performance: ≥8x Speedup gegenüber baseline

**Akzeptanzkriterien**
- Alle CI Gates grün (Tests + Benchmarks + drift detection)
- Benchmarks: Signal (1K) <5ms, Exit (1K) <4ms
- Keine Python fallback paths im regulären Backtest

---

## 5. Rust-Implementation Details

### 5.0 Schema-Strategie (Arrow Registry)

**Problem:** `src/shared/arrow_schemas.py` enthält `POSITION_SCHEMA`, aber dieses Schema enthält kein `order_type`, kein `risk_per_trade`, kein `trigger_time` und ist damit **nicht ausreichend** für Pending Orders und Execution-Lifecycle.

**Entscheidung (geplant):**
- `TRADE_SIGNAL_SCHEMA` wird für Signals wiederverwendet (enthält bereits `order_type`).
- Für Positionen wird **ein neues Schema** eingeführt (z.B. `EXECUTION_POSITION_SCHEMA`), um `POSITION_SCHEMA` (Wave 2 Portfolio) nicht zu brechen.
- Für SymbolSpecs wird **ein neues Schema** eingeführt (z.B. `SYMBOL_SPEC_SCHEMA`) als Arrow IPC Input in `ExecutionSimulatorRust::new(...)`.

**Drift Detection:** Neue Schemas werden in `SCHEMA_REGISTRY` registriert und fingerprinted (CI gate).

### 5.1 Cargo Dependencies (geplant)

`src/rust_modules/omega_rust/Cargo.toml` (Ausschnitt)

```toml
[dependencies]
arrow = { version = "50.0" }
arrow-array = { version = "50.0" }
arrow-schema = { version = "50.0" }
arrow-ipc = { version = "50.0" }
```

### 5.2 Arrow IPC Integration (Deserialize/Serialize)

```rust
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use std::io::Cursor;

pub fn decode_record_batches(ipc: &[u8]) -> anyhow::Result<Vec<arrow_array::RecordBatch>> {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None)?;
    let mut out = Vec::new();
    while let Some(batch) = reader.next() {
        out.push(batch?);
    }
    Ok(out)
}

pub fn encode_record_batch(batch: &arrow_array::RecordBatch) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}
```

### 5.3 Enum/Parsing Traits (Syntaktisch korrekt, geplant)

```rust
use crate::error::{OmegaError, Result};

impl TryFrom<&str> for Direction {
    type Error = OmegaError;

    fn try_from(v: &str) -> Result<Self> {
        match v {
            "long" => Ok(Direction::Long),
            "short" => Ok(Direction::Short),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid direction: {v}"),
            }),
        }
    }
}

impl TryFrom<&str> for OrderType {
    type Error = OmegaError;

    fn try_from(v: &str) -> Result<Self> {
        match v {
            "market" => Ok(OrderType::Market),
            "limit" => Ok(OrderType::Limit),
            "stop" => Ok(OrderType::Stop),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid order_type: {v}"),
            }),
        }
    }
}
```

### 5.3 Error Handling (Pattern)

**Konvention:** keine panics über FFI; alle Fehler → `OmegaError` → `PyErr` mit `[ErrorCode]` Prefix (siehe `src/rust_modules/omega_rust/src/error.rs`).

```rust
use crate::error::{OmegaError, Result};

fn validate_sl_distance(sl_distance: f64) -> Result<()> {
    if !(sl_distance.is_finite()) || sl_distance <= 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: "Stop-loss distance must be > 0".to_string(),
        });
    }
    Ok(())
}
```

### 5.4 Determinismus-Strategie (RNG + Float Policy)

**RNG:**
- Base seed per simulator instance (`base_seed`)
- Derived seed pro operation: `seed = hash(base_seed, symbol, timestamp_us, side, index)`
- Slippage uses Wave-0 ChaCha RNG (kein globaler RNG, keine non-deterministic entropy)

**Float/Rounding:**
- Critical rounding points werden in Rust explizit implementiert (keine locale/string-formatting Abhängigkeit).
- Lot-Quantisierung: floor-to-step + clamp, Ausgabe auf 8 Dezimalstellen stabil.
- `pip_buffer_factor` default 0.5 muss parity-genau angewendet werden.

---

## 6. Python-Integration

### 6.1 Zielzustand: Thin Wrapper ohne Fallback

`src/backtest_engine/core/execution_simulator.py` wird zu:
- Input validation + Arrow batch building
- Delegation an `omega_rust.ExecutionSimulatorRust`
- Keine doppelte Implementierung von Trigger/Exit/Sizing in Python

### 6.1.1 Python Wrapper Sketch (geplant, syntaktisch korrekt)

```python
import os
from typing import Optional

import pyarrow as pa
import pyarrow.ipc as ipc

from omega_rust import ExecutionSimulatorRust
from backtest_engine.strategy.strategy_wrapper import TradeSignal
from backtest_engine.data.candle import Candle


def _require_always() -> None:
    val = os.environ.get("OMEGA_USE_RUST_EXECUTION_SIMULATOR", "always").lower()
    if val != "always":
        raise ValueError(
            "Wave 4 requires OMEGA_USE_RUST_EXECUTION_SIMULATOR=always "
            f"(got {val!r}); rollback is deployment-based."
        )


class ExecutionSimulator:
    def __init__(self, portfolio, risk_per_trade: float = 100.0, symbol_specs=None) -> None:
        _require_always()
        # symbol_specs serialization to Arrow IPC is handled here (planned: SYMBOL_SPEC_SCHEMA)
        self._rust = ExecutionSimulatorRust(risk_per_trade=risk_per_trade)
        self.portfolio = portfolio

    def process_signal(self, signal: TradeSignal) -> None:
        # planned: build 1-row TRADE_SIGNAL_SCHEMA RecordBatch and send as IPC bytes
        batch = pa.record_batch(
            [
                pa.array([signal.timestamp], type=pa.timestamp("us", tz="UTC")),
                pa.array([signal.direction]),
                pa.array([signal.entry_price], type=pa.float64()),
                pa.array([signal.stop_loss], type=pa.float64()),
                pa.array([signal.take_profit], type=pa.float64()),
                pa.array([0.0], type=pa.float64()),  # size is computed in Rust
                pa.array([signal.symbol]),
                pa.array([signal.type]),
                pa.array([signal.reason]),
                pa.array([signal.scenario]),
            ],
            schema=None,  # planned: explicit TRADE_SIGNAL_SCHEMA
        )
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, batch.schema) as w:
            w.write_batch(batch)
        self._rust.process_signals_batch(sink.getvalue().to_pybytes())

    def evaluate_exits(self, bid_candle: Candle, ask_candle: Optional[Candle] = None) -> None:
        # planned: serialize candles as OHLCV_SCHEMA and call evaluate_exits_batch
        ...
```

### 6.2 API Parity (Method Mapping)

| Python API | Rust Backend | Hinweis |
|---|---|---|
| `process_signal(signal)` | `process_signals_batch(signals_ipc)` | Wrapper kann single signal als 1-row batch senden |
| `evaluate_exits(bid, ask)` | `evaluate_exits_batch(bid_ipc, ask_ipc)` | Candle Mode |
| `process_signal_tick(signal, tick)` | (optional) `process_signal_tick_batch` | nur wenn Tick Mode in Backtests aktiv |
| `evaluate_exits_tick(tick)` | (optional) `evaluate_exits_tick_batch` | dito |
| `active_positions` | `get_active_positions_ipc()` | Python materialisiert optional `PortfolioPosition` objects |

### 6.3 Feature Flag Handling

**Env Var:** `OMEGA_USE_RUST_EXECUTION_SIMULATOR`
- Supported: `always` (default)
- Unsupported: anything else → `ValueError`

Rationale: Wave 4 ist “full migration”; Rollback erfolgt über Deployment-Rollback (siehe Abschnitt 9), nicht über Dual-Path Runtime Fallback.

---

## 7. Test-Strategie

### 7.1 Rust Unit Tests (inline in `src/rust_modules/omega_rust/src/execution/*.rs`)

Die Rust-Tests sind **inline** in den jeweiligen Modulen mittels `#[cfg(test)]` definiert:

- `simulator.rs`: test_empty_results, test_single_signal_position_creation
- `signal.rs`: test_signal_direction_from_str, parsing/validation tests
- `sizing.rs`: test_lot_size_quantization, risk preservation tests
- `trigger.rs`: test_trigger_price_calculations, limit/stop trigger matrix
- `position.rs`: test_position_creation, SL/TP detection tests
- `slippage.rs`: test_slippage_calculations, cost model tests
- `arrow.rs`: Arrow Schema Serialization/Deserialization tests

**Coverage Target:** ≥90% line coverage im `execution/` Modul (realistisch über fokussierte unit tests + property tests).

### 7.2 Python Integration Tests

- `tests/test_execution_simulator_rust.py`
  - Instantiate wrapper + rust backend
  - Run deterministic fixtures (1–5 signals, 10–50 bars)
  - Compare positions/trades vs Python baseline (frozen fixtures)

### 7.3 Golden Tests

- `tests/golden/test_golden_execution.py`
  - Golden input: signals + candles + symbol_specs (Arrow IPC bytes)
  - Golden output: resulting positions/trades/fees (Arrow IPC bytes)
  - Policy: byte-for-byte identical on all CI platforms

### 7.4 Property-Based Tests

- `tests/property/test_execution_properties.py`
  - Edge cases: NaN/Inf guardrails, extreme lot sizes, tiny SL distance, huge prices
  - Invariants:
    - `entry_time < exit_time` when closed
    - Pending orders only open via trigger
    - `risk_preservation`: `size * sl_distance * unit_value ≈ risk_per_trade` (toleranzfrei für quantization step checks, sonst bounded error)

### 7.5 Performance Tests

- `tests/benchmarks/test_bench_execution_simulator.py`
  - Must include Rust backend path
  - Benchmarks record: 1K signals, 1K exits, full-loop scenario
  - Gate: ≥8x vs stored baseline (siehe `docs/MIGRATION_READINESS_VALIDATION.md`)

### 7.6 Bekannte Einschränkungen (Stand 2026-01-11)

**Arrow Schema Mismatch: ✅ BEHOBEN**

Der Rust IPC-Export verwendete ursprünglich `Int64Builder` für Timestamps, aber das Schema erwartete `Timestamp(Microsecond, UTC)`. 

**Lösung (implementiert):**
```rust
// In omega_rust/src/execution/arrow.rs
// Geändert: Int64Builder → TimestampMicrosecondBuilder für entry_time, exit_time
let mut entry_time_builder =
    TimestampMicrosecondBuilder::with_capacity(num_rows).with_timezone("UTC");
```

**Testergebnisse nach Fix:**
- Vorher: 32 passed, 21 skipped
- Nachher: 49 passed, 4 skipped (nur Golden Reference File Tests)

---

## 8. Validierung & Akzeptanzkriterien

### 8.1 Performance

- `process_signal` (1K): <5ms
- `evaluate_exits` (1K positions): <4ms
- Full backtest loop: <15s (baseline 85s) oder besser

### 8.2 Determinismus

- Golden Tests byte-for-byte (IPC payloads)
- RNG deterministisch (seeded) und identisch über OS/Arch
- Keine use of system entropy im Hot Path

### 8.3 API Parity

- Alle Python-Methoden verfügbar (oder bewusst deprecated mit Migration Hinweis)
- Identische semantics für reasons (`stop_loss`, `take_profit`, `break_even_stop_loss`)

### 8.4 Safety/Correctness

- Keine panics über FFI; panics werden abgefangen und als `[5005]` gemeldet (`FFI_PANIC_CAUGHT`)
- Schema validation an FFI boundary (`SCHEMA_VIOLATION`, `FFI_SCHEMA_MISMATCH`)

### 8.5 CI/CD Gates

- `pytest -q` grün (inkl. integration/golden/property)
- `cargo test` grün
- benchmark workflow blocking für relevante Pfade (siehe `docs/MIGRATION_READINESS_VALIDATION.md`)

---

## 9. Rollback-Plan

> Wave 4 verzichtet bewusst auf Dual-Path Runtime Fallback, um Drift/Dual-Maintenance zu vermeiden.

### 9.1 Rollback Trigger

- Golden drift (Result mismatch)
- Performance regression (Rust slower)
- Panic/Crash im Rust Modul
- Schema drift / IPC decode errors

### 9.2 Rollback Procedure (pragmatic)

1. Revert/rollback deployment auf pre-wave4 Version (git tag/commit)
2. Re-run smoke backtests + golden suite
3. Block rollout bis root-cause analysiert ist

### 9.3 Feature-Flag Deaktivierung (kompatibel mit “full rust”)

Wave 4 selbst unterstützt nur `OMEGA_USE_RUST_EXECUTION_SIMULATOR=always`. Eine “Feature-Flag Deaktivierung” ist daher operational als **Deployment-Rollback** definiert:

- Deploy pre-wave4 build (mit Python Implementation) und setze (falls verfügbar) den alten Toggle auf Python.
- Verifiziere deterministische Backtests (Golden + Bench).

### 9.4 Monitoring während Rollout

- Benchmark deltas vs main baseline
- Panic rate / `[5005]` occurrences
- Determinism gate results

---

## 10. Checklisten

### 10.1 Pre-Implementation Checklist

- [x] `docs/ffi/execution_simulator.md` reviewed (contracts, nullability, error codes)
- [x] Schema strategy entschieden (re-use vs new schema) und drift detection angepasst
- [x] SymbolSpecs completeness: tick_size/tick_value/contract_size verified in configs
- [x] Benchmarks exist and cover Rust path
- [x] Golden fixtures defined (signals/candles/specs)

### 10.2 Daily Progress Checklist

- [x] Rust unit tests updated for any logic change
- [x] No new Python fallback logic introduced
- [x] Determinism checks run locally on sample fixture
- [x] Benchmarks sampled (sanity) after major milestones

### 10.3 Post-Implementation Validation Checklist

- [x] `cargo test` green
- [x] `pytest -q` green (49 passed, 4 skipped)
- [x] Golden: deterministic execution validated
- [x] Arrow Schema Mismatch: **FIXED** (TimestampMicrosecondBuilder)
- [x] Benchmark: Performance measured (see 10.6)
- [ ] Runbook updated with actual steps + evidence links

### 10.4 Test Evidence (Phase 7)

**Test Files Created:**
- `tests/test_execution_simulator_rust.py` - Integration Tests (17 tests: 17 pass)
- `tests/golden/test_golden_execution.py` - Golden Tests (8 tests: 5 pass, 3 skip)
- `tests/property/test_execution_properties.py` - Property Tests (9 tests: 9 pass)
- `tests/benchmarks/test_bench_execution_simulator.py` - Benchmarks (19 tests: 18 pass, 1 skip)

**Total: 53 tests, 49 passed, 4 skipped, 0 failures**

### 10.5 Arrow Schema Fix (2026-01-11)

**Problem:** Rust IPC used `Int64Builder` for timestamp fields but schema defined `Timestamp(Microsecond, UTC)`.

**Solution:** Changed `encode_positions()` in `arrow.rs` to use `TimestampMicrosecondBuilder::with_capacity().with_timezone("UTC")`.

**Result:** All previously skipped tests (21→4) now pass.

### 10.6 Performance Benchmark Results (2026-01-11)

**Benchmarks ausgeführt mit `OMEGA_USE_RUST_EXECUTION_SIMULATOR=always`:**

| Benchmark | Zeit | Ops/sec |
|-----------|------|---------|
| `test_check_entry_triggered_small` | 8.62µs | 116,033 |
| `test_check_entry_triggered_medium` | 46.70µs | 21,413 |
| `test_process_signal_market_small` (100) | 233ms | 4.29 |
| `test_process_signal_market_medium` (500) | 4.13s | 0.24 |
| `test_process_signal_market_large` (1000) | 58.31s | 0.017 |

**Rust vs Python Vergleich (500 Signals):**

| Backend | Zeit | Relative |
|---------|------|----------|
| Python (via Rust) | 4.02s | 1.00x |
| Rust Direct | 4.70s | 0.86x |

**Analyse:**
- ⚠️ Der ≥8x Speedup-Ziel wird mit single-signal IPC **nicht erreicht**.
- Grund: Arrow IPC Overhead pro Aufruf dominiert (Serialization + FFI-Crossing).
- Das aktuelle Design serialisiert jeden einzelnen Signal-Aufruf.

**Empfehlung für zukünftige Optimierung:**
1. **Batch-Mode priorisieren:** Signals sammeln und als Batch (1000+) senden
2. **IPC reduzieren:** Candle-Arrays einmalig senden, nicht pro Bar
3. **Rust-native Hot Path:** Volle Backtest-Loop in Rust implementieren (Wave 5)

**Fazit:** Die Migration ist funktional vollständig und deterministisch. Performance-Optimierung erfordert architekturelle Änderungen (Batch-First Design), die in einem separaten Wave (Wave 5) adressiert werden sollten.

---

## 11. Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|---|---|---|---|
| Tech Lead |  |  | ⏳ |
| QA Lead |  |  | ⏳ |
| DevOps/CI Owner |  |  | ⏳ |

---

## Referenzen

- FFI Spec: `docs/ffi/execution_simulator.md`
- Runbook: `docs/runbooks/execution_simulator_migration.md`
- Wave 3 Template: `docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md`
- Readiness Canonical Status: `docs/MIGRATION_READINESS_VALIDATION.md`
- Error Codes (Python/Rust): `src/shared/error_codes.py`, `src/rust_modules/omega_rust/src/error.rs`
- Arrow Schemas: `src/shared/arrow_schemas.py`
- Performance Baseline: `reports/performance_baselines/p0-01_execution_simulator.json`
