---
module: slippage_fee
phase: 1
prerequisites:
  - Type-Hints vollständig (mypy --strict)
  - ≥90% Test-Coverage (einfache Logik)
  - Performance-Baseline dokumentiert
  - FFI-Spec finalisiert
rollback_procedure: docs/runbooks/rollback_generic.md
---

# Migration Runbook: Slippage & Fee

**Status:** 🔴 Nicht begonnen (Readiness/Go-No-Go: `docs/MIGRATION_READINESS_VALIDATION.md`)

## 1. Modul-Übersicht

| Attribut | Wert |
| -------- | ---- |
| Quell-Module | `src/backtest_engine/core/slippage_and_fee.py` |
| Ziel-Sprache | Rust (PyO3 + SIMD) |
| Priorität | P1 - Idealer Pilot-Kandidat |
| Geschätzter Aufwand | 2-3 Tage |

---

## 2. Warum Pilot-Kandidat?

✅ **Pure Functions** - Keine State-Abhängigkeiten  
✅ **Einfache Logik** - Mathematische Formeln  
✅ **Batch-fähig** - SIMD-Optimierung möglich  
✅ **Gut testbar** - Deterministisches Verhalten  
✅ **Niedrige Risiko** - Isoliert von anderen Modulen  

---

## 3. Migration Steps

### Phase 1: Setup (Day 1)

```bash
# Rust-Modul erstellen
mkdir -p src/rust_modules/omega_rust/src/costs
touch src/rust_modules/omega_rust/src/costs/slippage.rs
touch src/rust_modules/omega_rust/src/costs/fee.rs
touch src/rust_modules/omega_rust/src/costs/mod.rs
```

### Phase 2: Implementation (Day 1-2)

```rust
// src/rust_modules/omega_rust/src/costs/slippage.rs

use std::simd::f64x4;

/// Calculate slippage for a batch using SIMD
pub fn calculate_slippage_batch_simd(
    prices: &[f64],
    spreads: &[f64],
    model: SlippageModel,
) -> Vec<f64> {
    // Process 4 values at a time with SIMD
    prices.chunks_exact(4)
        .zip(spreads.chunks_exact(4))
        .flat_map(|(p, s)| {
            let prices_simd = f64x4::from_slice(p);
            let spreads_simd = f64x4::from_slice(s);
            
            let result = match model {
                SlippageModel::Fixed => spreads_simd / f64x4::splat(2.0),
                // ...
            };
            
            result.to_array()
        })
        .collect()
}
```

### Phase 3: Testing (Day 2-3)

```bash
# Unit Tests
pytest tests/test_deterministic_dev_mode_scores.py -v
pytest tests/test_rating_cost_shock_and_timing_jitter.py -v

# Performance (Proxy via ExecutionSimulator Benchmarks)
pytest tests/benchmarks/test_bench_execution_simulator.py --benchmark-only --benchmark-json=reports/performance_baselines/execution_simulator.json
```

---

## 4. Validierung

### 4.1 Numerical Equivalence

```python
def test_slippage_equivalence():
    """Rust and Python must produce identical results."""
    test_cases = [
        (100.0, 0.0001, "fixed"),
        (1.3456, 0.00012, "proportional"),
        # ... viele Fälle
    ]
    
    for price, spread, model in test_cases:
        py_result = slippage_python(price, spread, model)
        rs_result = slippage_rust(price, spread, model)
        assert abs(py_result - rs_result) < 1e-10
```

### 4.2 Performance Check

> **Status: PLANNED** - Performance-Werte sind geschätzte Ziele; tatsächliche Messungen erfolgen nach Rust-Implementierung.

| Operation | Python | Rust | Target | Status |
| --------- | ------ | ---- | ------ | ------ |
| Slippage (single) | TBD | - | 0.001ms | ⏳ PLANNED |
| Slippage (1K batch) | TBD | - | 0.5ms | ⏳ PLANNED |
| Fee (single) | TBD | - | 0.002ms | ⏳ PLANNED |
| Fee (1K batch) | TBD | - | 1ms | ⏳ PLANNED |

---

## 5. Config Integration

### 5.1 YAML Config laden

```rust
// Fee-Config aus YAML laden
pub fn load_fee_config(yaml_path: &str) -> Result<FeeConfig, Error> {
    let content = std::fs::read_to_string(yaml_path)?;
    let config: FeeConfig = serde_yaml::from_str(&content)?;
    Ok(config)
}
```

### 5.2 Config-Pfad

```python
# Python ruft Rust mit Config-Pfad auf
fee_result = calculate_fee_rust(
    prices, sizes, symbols,
    config_path="configs/execution_costs.yaml"
)
```

---

## 6. Rollback-Trigger

| Kriterium | Schwellwert | Aktion |
| --------- | ----------- | ------ |
| Numerical Diff | >1e-8 | Rollback |
| Performance-Regression | >5% | Rollback |
| Config-Parse Error | Jeder | Rollback |

---

## 7. Abnahme-Checkliste

- [ ] Alle Equivalence-Tests grün
- [ ] SIMD-Optimierung verifiziert
- [ ] Performance-Target erreicht (≥20x batch)
- [ ] Config-Loading funktioniert
- [ ] Code-Review abgeschlossen
- [ ] Dokumentation aktualisiert

---

## 8. Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

