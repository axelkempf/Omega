# Migrations-Runbook: Rating Modules (Rust)

**Module:** `src/backtest_engine/rating/*.py`  
**Target-Sprache:** Rust  
**Priorität:** Wave 1 (Pilot-Module)  
**Aufwand:** M (Medium)  
**Status:** 🟢 READY FOR MIGRATION

---

## 1. Übersicht

Die Rating-Module berechnen Qualitäts- und Robustheits-Scores für Backtest-Ergebnisse. Sie sind ideale Pilot-Kandidaten: reine Mathematik, keine Abhängigkeiten, gut testbar.

### 1.1 Module in Scope

| Modul | Funktion | Komplexität |
|-------|----------|-------------|
| `strategy_rating.py` | Haupt-Rating Aggregation | Niedrig |
| `robustness_score_1.py` | Parameter-Jitter Robustness | Niedrig |
| `stability_score.py` | Yearly Profit Stability | Niedrig |
| `cost_shock_score.py` | Cost Sensitivity Analysis | Niedrig |
| `trade_dropout_score.py` | Trade Dropout Analysis | Niedrig |
| `stress_penalty.py` | Drawdown/Stress Penalty | Niedrig |

### 1.2 Warum Migration?

| Metrik | Python Baseline | Rust Target | Speedup |
|--------|-----------------|-------------|---------|
| Robustness Score (50 jitters) | 125ms | <15ms | 8x |
| Cost Shock Score (batch) | 85ms | <10ms | 8x |
| Full Rating Pipeline | 450ms | <50ms | 9x |
| Trade Dropout (1000 trades) | 95ms | <12ms | 8x |

### 1.3 Abhängigkeiten

- **Upstream:** Backtest Results (TradesDF, Metrics)
- **Downstream:** Optimizer (Score Selection), Reports
- **FFI:** Arrow IPC für Batch-Metrics-Transfer

---

## 2. Vorbereitungs-Checkliste

### 2.1 Type Safety ✅

- [x] Alle Module mypy --strict compliant
- [x] Score-Typen in `core/types.py` definiert
- [x] MetricsDict TypedDict vorhanden
- [x] Keine `# type: ignore` ohne Begründung

### 2.2 Test Coverage ✅

- [x] Unit Tests: `tests/test_rating_*.py`
- [x] Property-Based Tests: `tests/property/test_property_scoring.py`
- [x] Edge-Case Tests (empty trades, single trade, extreme values)
- [x] Coverage ≥ 90%

### 2.3 Performance Baseline ✅

- [x] Benchmark: `tests/benchmarks/test_bench_rating.py`
- [x] Baseline gespeichert: `reports/performance_baselines/rating_modules.json`
- [x] Improvement-Target: 8x Speedup

### 2.4 FFI-Dokumentation ✅

- [x] Interface-Spec: `docs/ffi/rating_modules.md`
- [x] Arrow-Schema: MetricsDict, RatingScore
- [x] Nullability: Dokumentiert
- [x] Error-Codes: RatingError-Kategorie definiert

---

## 3. Rust-Architektur

### 3.1 Modul-Struktur

```
src/rust_modules/omega_rust/src/
├── rating/
│   ├── mod.rs              # Modul-Exports
│   ├── robustness.rs       # Robustness Score (Jitter)
│   ├── stability.rs        # Stability Score (Yearly)
│   ├── cost_shock.rs       # Cost Shock Analysis
│   ├── trade_dropout.rs    # Trade Dropout Score
│   ├── stress_penalty.rs   # Stress/Drawdown Penalty
│   └── aggregator.rs       # Rating Aggregation
└── lib.rs
```

### 3.2 Core Types (Rust)

```rust
/// Input metrics for rating calculation
#[derive(Clone, Debug)]
pub struct MetricsInput {
    pub total_trades: i64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub avg_trade_duration: f64,
    pub yearly_profits: Vec<f64>,
    pub cost_per_trade: f64,
    // ... additional metrics
}

/// Rating score output
#[derive(Clone, Debug)]
pub struct RatingScore {
    pub robustness_score: f64,     // [0, 1]
    pub stability_score: f64,      // [0, 1]
    pub cost_shock_score: f64,     // [0, 1]
    pub trade_dropout_score: f64,  // [0, 1]
    pub stress_penalty: f64,       // [0, 1], lower = better
    pub final_rating: f64,         // Weighted aggregate
}

/// Jitter parameters for robustness testing
#[derive(Clone, Debug)]
pub struct JitterConfig {
    pub param_variation: f64,  // e.g., 0.1 = ±10%
    pub n_iterations: usize,   // e.g., 50
    pub seed: Option<u64>,
}
```

### 3.3 Batch API (FFI-Overhead Minimierung)

```rust
#[pyfunction]
/// Calculate ratings for a batch of backtest results
/// 
/// Args:
///     metrics_ipc: Arrow IPC bytes containing Vec<MetricsInput>
///     config: RatingConfig with weights and thresholds
///
/// Returns:
///     Arrow IPC bytes containing Vec<RatingScore>
pub fn calculate_ratings_batch(
    metrics_ipc: &[u8],
    config: RatingConfigPy,
) -> PyResult<Vec<u8>>;

#[pyfunction]
/// Calculate robustness score with parameter jitter
/// 
/// Uses SIMD for parallel jitter evaluation
pub fn robustness_score_batch(
    base_metrics: &[u8],    // Arrow IPC: Vec<MetricsInput>
    jitter_config: JitterConfigPy,
) -> PyResult<Vec<f64>>;    // Scores for each input

#[pyfunction]
/// Calculate cost shock scores for multiple cost scenarios
pub fn cost_shock_score_batch(
    base_metrics: &[u8],           // Arrow IPC
    cost_multipliers: Vec<f64>,    // e.g., [1.0, 1.1, 1.2, 1.5]
) -> PyResult<Vec<u8>>;            // Arrow IPC with results
```

---

## 4. Migration Steps

### Phase 1: Rust Scaffold (0.5-1 Tag)

1. Create `src/rust_modules/omega_rust/src/rating/` directory
2. Define core types in `mod.rs`
3. Add PyO3 module registration
4. Verify build with `maturin develop`

### Phase 2: Individual Scores (2-3 Tage)

1. Implement `robustness_score()` in `robustness.rs`
   - Port jitter logic
   - Add SIMD optimization for batch processing
   
2. Implement `stability_score()` in `stability.rs`
   - Port yearly profit analysis
   - Handle edge cases (< 2 years)

3. Implement `cost_shock_score()` in `cost_shock.rs`
   - Port cost sensitivity calculation
   - Batch support for multiple cost scenarios

4. Implement `trade_dropout_score()` in `trade_dropout.rs`
   - Port dropout simulation
   - Optimize for large trade counts

5. Implement `stress_penalty()` in `stress_penalty.rs`
   - Port drawdown penalty logic

### Phase 3: Aggregation (1 Tag)

1. Implement `RatingAggregator` in `aggregator.rs`
2. Add configurable weights
3. Add final rating calculation

### Phase 4: Arrow Integration (1-2 Tage)

1. Define Arrow schemas for MetricsInput, RatingScore
2. Implement IPC serialization/deserialization
3. Add batch API functions
4. Test zero-copy performance

### Phase 5: Hybrid Mode (1 Tag)

1. Create Python wrapper in `shared/rating_ffi.py`
2. Add feature flag: `OMEGA_USE_RUST_RATING`
3. Implement fallback to Python
4. Add logging for path selection

### Phase 6: Testing (2-3 Tage)

1. Port all unit tests to Rust (`cargo test`)
2. Add Criterion benchmarks
3. Run Property-Based tests:
   - Score bounds [0, 1]
   - Determinism (same input → same output)
   - Monotonicity (better metrics → better scores)
4. Compare Python vs Rust outputs for 1000+ random inputs

### Phase 7: Benchmarking & Rollout (1-2 Tage)

1. Run `pytest tests/benchmarks/test_bench_rating.py`
2. Verify 8x+ speedup
3. Enable in staging, compare results
4. Gradual rollout to optimizer pipeline

---

## 5. Property-Validierung

Die folgenden Properties müssen für Python und Rust identisch sein:

### 5.1 Score Bounds

```python
@given(metrics=valid_metrics())
def test_scores_in_bounds(metrics):
    score = calculate_rating(metrics)
    assert 0 <= score.robustness_score <= 1
    assert 0 <= score.stability_score <= 1
    assert 0 <= score.cost_shock_score <= 1
    assert 0 <= score.trade_dropout_score <= 1
    assert 0 <= score.stress_penalty <= 1
```

### 5.2 Determinism

```python
@given(metrics=valid_metrics(), seed=st.integers(0, 2**32-1))
def test_determinism(metrics, seed):
    result1 = calculate_rating(metrics, seed=seed)
    result2 = calculate_rating(metrics, seed=seed)
    assert result1 == result2
```

### 5.3 Monotonicity (Higher Profit Factor → Higher Rating)

```python
@given(
    metrics=valid_metrics(),
    pf_delta=st.floats(0.01, 0.5)
)
def test_monotonicity_profit_factor(metrics, pf_delta):
    metrics_better = copy(metrics)
    metrics_better.profit_factor += pf_delta
    
    score_base = calculate_rating(metrics)
    score_better = calculate_rating(metrics_better)
    
    assert score_better.final_rating >= score_base.final_rating
```

---

## 6. Rollback-Plan

### 6.1 Rollback-Trigger

- Score divergence > 0.001 zwischen Python und Rust
- Performance regression
- Crash oder Panic im Rust-Code
- Unerwartete NaN/Inf Werte

### 6.2 Rollback-Prozedur

```bash
# Immediate rollback
export OMEGA_USE_RUST_RATING=false

# Verify
pytest tests/test_rating_*.py -v
```

### 6.3 Rollback-Validierung

1. Run full rating test suite
2. Compare optimizer results with pre-migration baseline
3. Verify no score divergence in optimizer logs

---

## 7. Akzeptanzkriterien

### 7.1 Funktional

- [ ] Alle Unit Tests pass (Python + Rust)
- [ ] Property-Based Tests pass
- [ ] Score-Divergenz < 0.0001 (float precision)
- [ ] Edge-Cases handled (empty, single trade, extreme values)

### 7.2 Performance

- [ ] Robustness Score (50 jitters): <15ms
- [ ] Full Rating Pipeline: <50ms
- [ ] Batch Mode: Linear scaling with input size

### 7.3 Batch-Effizienz

- [ ] FFI-Overhead < 10% der Rechenzeit
- [ ] Arrow IPC round-trip < 1ms für 1K metrics
- [ ] Memory-effizient für große Batches

---

## 8. Sign-Off

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

---

## 9. Referenzen

- FFI-Spec: [rating_modules.md](../ffi/rating_modules.md)
- Arrow-Schemas: [arrow_schemas.py](../../src/shared/arrow_schemas.py)
- Property-Tests: [test_prop_scoring.py](../../tests/property/test_prop_scoring.py)
- Benchmarks: [test_bench_rating.py](../../tests/benchmarks/test_bench_rating.py)
