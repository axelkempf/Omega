---
module: execution_simulator
phase: 6
prerequisites:
    - docs/MIGRATION_READINESS_VALIDATION.md
    - docs/ffi/execution_simulator.md
    - tests/benchmarks/
    - tests/golden/
    - tests/property/
rollback_procedure: docs/runbooks/rollback_generic.md
---

## Migrations-Runbook: ExecutionSimulator (Rust)

**Modul:** `src/backtest_engine/core/execution_simulator.py`  
**Target-Sprache:** Rust  
**Priorität:** Wave 3 (Core Loop)  
**Aufwand:** L (Large)  
**Status:** TEMPLATE (Readiness/Go-No-Go: `docs/MIGRATION_READINESS_VALIDATION.md`)

---

## 1. Übersicht

Der ExecutionSimulator ist die zentrale Komponente für Trade-Matching und Portfolio-State-Management im Backtest. Er verarbeitet TradeSignals, evaluiert Entry/Exit-Trigger und verwaltet offene Positionen.

### 1.1 Aktuelle Architektur

```
┌─────────────────┐    ┌──────────────────────┐    ┌────────────────┐
│   EventEngine   │───▶│  ExecutionSimulator  │───▶│   Portfolio    │
│  (TradeSignals) │    │  - process_signal()  │    │   (State)      │
└─────────────────┘    │  - evaluate_exit()   │    └────────────────┘
                       │  - entry_trigger()   │
                       └──────────────────────┘
```

### 1.2 Warum Migration?

| Metrik | Python Baseline | Rust Target | Speedup |
|--------|-----------------|-------------|---------|
| Signal Processing (1K) | 45ms | <5ms | 9x |
| Exit Evaluation (1K) | 32ms | <4ms | 8x |
| Full Backtest Loop | 85s | <15s | 5.5x |

### 1.3 Abhängigkeiten

- **Upstream:** EventEngine (Signals), IndicatorCache (Prices)
- **Downstream:** Portfolio (State Updates), TradeLog (Persistence)
- **FFI:** Arrow IPC für Position/Signal Serialization

---

## 2. Vorbereitungs-Checkliste

### 2.1 Type Safety

- [x] Modul ist mypy --strict compliant
- [x] Alle öffentlichen APIs haben Type Hints
- [x] TypedDict-Schemas in `core/types.py` definiert
- [x] Protocol-Klassen in `shared/protocols.py`

### 2.2 Test Coverage

- [x] Unit Tests: `tests/test_execution_simulator*.py`
- [x] Integration Tests: `tests/integration/test_backtest_e2e.py` (docs-lint:planned)
- [x] Property-Based Tests: `tests/property/test_execution_properties.py` (docs-lint:planned)
- [x] Golden-File Tests: `tests/golden/test_golden_backtest.py`
- [x] Coverage ≥ 87%

### 2.3 Performance Baseline

- [x] Benchmark: `tests/benchmarks/test_bench_execution_simulator.py`
- [x] Baseline gespeichert: `reports/performance_baselines/execution_simulator.json`
- [x] Improvement-Target definiert: 8x Speedup

### 2.4 FFI-Dokumentation

- [x] Interface-Spec: `docs/ffi/execution_simulator.md`
- [x] Arrow-Schema: PortfolioPosition, TradeSignal (in `shared/arrow_schemas.py`)
- [x] Nullability: Dokumentiert in `docs/ffi/nullability-convention.md`
- [x] Error-Codes: Definiert in `shared/error_codes.py`

---

## 3. Rust-Architektur

### 3.1 Modul-Struktur

```
src/rust_modules/omega_rust/src/
├── execution/
│   ├── mod.rs           # Modul-Exports
│   ├── simulator.rs     # ExecutionSimulator Struct
│   ├── position.rs      # PortfolioPosition State
│   ├── signal.rs        # TradeSignal Processing
│   ├── trigger.rs       # Entry/Exit Trigger Logic
│   └── slippage.rs      # Slippage & Fee Models
└── lib.rs               # PyO3 Entry-Point
```

### 3.2 Core Types (Rust)

```rust
/// Portfolio Position State
#[derive(Clone, Debug)]
pub struct PortfolioPosition {
    pub symbol: String,
    pub magic_number: i64,
    pub direction: Direction,  // Long = 1, Short = -1
    pub entry_price: f64,
    pub entry_time: i64,       // Unix timestamp (microseconds)
    pub size: f64,
    pub take_profit: Option<f64>,
    pub stop_loss: Option<f64>,
    pub trailing_stop: Option<TrailingStop>,
}

/// Trade Signal from Strategy
#[derive(Clone, Debug)]
pub struct TradeSignal {
    pub symbol: String,
    pub timestamp: i64,
    pub signal_type: SignalType,  // Entry, Exit, Modify
    pub direction: Option<Direction>,
    pub price: f64,
    pub size: Option<f64>,
    pub take_profit: Option<f64>,
    pub stop_loss: Option<f64>,
    pub metadata: HashMap<String, String>,
}

/// Exit Evaluation Result
#[derive(Clone, Debug)]
pub enum ExitResult {
    NoExit,
    TakeProfit { price: f64, pnl: f64 },
    StopLoss { price: f64, pnl: f64 },
    TrailingStop { price: f64, pnl: f64 },
    SignalExit { price: f64, pnl: f64 },
}
```

### 3.3 PyO3 Interface

```rust
#[pyclass]
pub struct ExecutionSimulatorRust {
    positions: HashMap<String, PortfolioPosition>,
    config: ExecutionConfig,
    slippage_model: SlippageModel,
    fee_model: FeeModel,
}

#[pymethods]
impl ExecutionSimulatorRust {
    #[new]
    pub fn new(config: ExecutionConfigPy) -> PyResult<Self>;
    
    /// Process a batch of signals, returns Arrow IPC bytes
    pub fn process_signals_batch(&mut self, signals_ipc: &[u8]) -> PyResult<Vec<u8>>;
    
    /// Evaluate exits for all positions, returns exit results as Arrow IPC
    pub fn evaluate_exits(&self, current_candle: &CandleData) -> PyResult<Vec<u8>>;
    
    /// Get current positions as Arrow IPC
    pub fn get_positions_arrow(&self) -> PyResult<Vec<u8>>;
    
    /// Apply position updates from Arrow IPC
    pub fn apply_updates(&mut self, updates_ipc: &[u8]) -> PyResult<()>;
}
```

---

## 4. Migration Steps

### Phase 1: Rust Scaffold (1-2 Tage)

1. Create `src/rust_modules/omega_rust/src/execution/` directory (docs-lint:planned)
2. Define core types in `position.rs`, `signal.rs`
3. Implement basic `ExecutionSimulatorRust` struct
4. Add PyO3 bindings and module registration
5. Verify build with `maturin develop`

### Phase 2: Core Logic (3-5 Tage)

1. Implement `process_signal()` matching Python behavior
2. Implement `evaluate_exit()` with all exit types
3. Add slippage and fee calculation models
4. Port entry trigger logic

### Phase 3: Arrow Integration (2-3 Tage)

1. Define Arrow schemas in Rust (`arrow-rs`)
2. Implement IPC serialization for Position/Signal
3. Add zero-copy utilities for batch processing
4. Test round-trip serialization

### Phase 4: Hybrid Mode (2-3 Tage)

1. Create Python wrapper in `shared/ffi_wrapper.py`
2. Add feature flag: `OMEGA_USE_RUST_EXECUTION`
3. Implement fallback to Python on error
4. Add metrics logging for both paths

### Phase 5: Testing (3-4 Tage)

1. Port all unit tests to Rust (`cargo test`)
2. Add Rust benchmarks with Criterion
3. Run Python integration tests against Rust backend
4. Verify determinism with Golden-File tests
5. Run Property-Based tests with both backends

### Phase 6: Benchmarking (1-2 Tage)

1. Run `pytest tests/benchmarks/test_bench_execution_simulator.py`
2. Compare Python vs Rust performance
3. Document results in `reports/performance_baselines/`
4. Verify 8x speedup target

### Phase 7: Rollout (2-3 Tage)

1. Enable Rust backend in staging
2. Run full backtests, compare results
3. Monitor error rates and performance
4. Gradual rollout to production backtests

---

## 5. Rollback-Plan

### 5.1 Rollback-Trigger

- Rust module produces different results than Python (determinism failure)
- Performance regression (Rust slower than Python)
- Crash or panic in Rust code
- Memory leak or resource exhaustion

### 5.2 Rollback-Prozedur

```bash
# Immediate rollback
export OMEGA_USE_RUST_EXECUTION=false

# Or in Python
from omega.config import set_feature_flag
set_feature_flag("use_rust_execution", False)
```

### 5.3 Rollback-Validierung

1. Run full test suite: `pytest tests/ -v`
2. Run Golden-File tests: `pytest tests/golden/ -v`
3. Verify backtest results match pre-migration baseline

---

## 6. Akzeptanzkriterien

### 6.1 Funktional

- [ ] Alle Unit Tests pass (Python + Rust)
- [ ] Integration Tests pass
- [ ] Golden-File Tests pass (determinism)
- [ ] Property-Based Tests pass

### 6.2 Performance

- [ ] Signal Processing: <5ms für 1K Signals
- [ ] Exit Evaluation: <4ms für 1K Positions
- [ ] Full Backtest: <15s (5x+ Speedup)

### 6.3 Determinismus

- [ ] Rust produces identical results to Python
- [ ] Same seed → same trades → same PnL
- [ ] No floating-point divergence

### 6.4 Safety

- [ ] No panics reach Python (all caught at FFI boundary)
- [ ] No memory leaks (verified with Valgrind/ASAN)
- [ ] Thread-safe for concurrent backtests

---

## 7. Sign-Off

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

---

## 8. Referenzen

- FFI-Spec: [execution_simulator.md](../ffi/execution_simulator.md)
- Arrow-Schemas: [arrow_schemas.py](../../src/shared/arrow_schemas.py)
- Error-Codes: [error_codes.py](../../src/shared/error_codes.py)
- ADR-0002: [Serialization Format](../adr/ADR-0002-serialization-format.md)
- ADR-0003: [Error Handling](../adr/ADR-0003-error-handling.md)
