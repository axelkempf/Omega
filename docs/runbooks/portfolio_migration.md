---
module: portfolio
phase: 2
prerequisites:
  - Type-Hints vollständig (mypy --strict)
  - ≥80% Test-Coverage
  - Performance-Baseline dokumentiert
  - FFI-Spec finalisiert
rollback_procedure: docs/runbooks/rollback_generic.md
---

# Migration Runbook: Portfolio

## 1. Modul-Übersicht

| Attribut | Wert |
| -------- | ---- |
| Quell-Modul | `src/backtest_engine/core/portfolio.py` |
| Ziel-Sprache | Rust (PyO3) |
| Priorität | P2 - Core State Management |
| Geschätzter Aufwand | 4-5 Tage |

---

## 2. Voraussetzungen

### 2.1 Type Safety

```bash
mypy --strict src/backtest_engine/core/portfolio.py
# Erwartetes Ergebnis: Success: no issues found
```

### 2.2 Test-Coverage

```bash
pytest tests/test_portfolio_summary_extra_metrics.py -v --cov=src/backtest_engine/core/portfolio --cov-report=term-missing
# Erwartete Coverage: ≥80%
```

### 2.3 Performance-Baseline

```bash
pytest tests/benchmarks/test_bench_portfolio.py --benchmark-only --benchmark-json=reports/performance_baselines/portfolio.json
```

---

## 3. Migration Steps

### Phase 1: State-Design (Day 1)

```rust
// Core State Structures
pub struct PortfolioState {
    balance: f64,
    equity: f64,
    margin_used: f64,
    positions: HashMap<String, Position>,
    // ...
}
```

### Phase 2: Implementation (Day 2-4)

1. **Position Management**
   - `open_position()` - neue Position öffnen
   - `close_position()` - Position schließen
   - `modify_position()` - SL/TP ändern

2. **State Tracking**
   - `update_equity()` - Equity bei Preisänderung
   - `calculate_margin()` - Margin-Berechnung
   - `get_state()` - State als Arrow IPC

3. **Metrics Calculation**
   - `calculate_metrics()` - Performance-Metriken
   - `get_equity_curve()` - Equity-History

### Phase 3: Testing (Day 4-5)

```bash
# Funktionale Regression (bestehende Tests)
pytest tests/test_portfolio_summary_extra_metrics.py -v

# Equivalence / Property-Based Tests
# Status: PLANNED (erst nach Rust-Implementation)

# Performance-Baseline (pytest-benchmark)
pytest tests/benchmarks/test_bench_portfolio.py --benchmark-only --benchmark-json=reports/performance_baselines/portfolio.json

# Optional: Regression-Report gegen gespeicherte Historie
python tools/benchmark_history.py report reports/performance_baselines/portfolio.json
```

---

## 4. Validierung

### 4.1 Decimal Precision

Portfolio arbeitet mit monetären Werten → Precision-Tests:

```python
def test_portfolio_precision():
    """Ensure no floating-point precision loss."""
    portfolio = PortfolioRust(initial_balance=100000.0)
    
    # Many small trades
    for _ in range(10000):
        portfolio.open_position(...)
        portfolio.close_position(...)
    
    # Balance should be exact
    assert portfolio.get_state().balance == expected_exact_balance
```

### 4.2 Performance Check

| Operation | Python | Rust | Target | Status |
| --------- | ------ | ---- | ------ | ------ |
| open_position | 0.5ms | - | 0.05ms | ⏳ |
| close_position | 0.4ms | - | 0.04ms | ⏳ |
| get_state | 0.2ms | - | 0.02ms | ⏳ |
| calculate_metrics | 15ms | - | 2ms | ⏳ |

---

## 5. Kritische Invarianten

### 5.1 Balance-Invariante

```python
# Nach jeder Operation:
assert portfolio.balance + portfolio.unrealized_pnl == portfolio.equity
```

### 5.2 Position-Invariante

```python
# Alle Positionen müssen valide sein:
for pos in portfolio.positions.values():
    assert pos.size > 0
    assert pos.entry_price > 0
    assert pos.entry_time <= current_time
```

---

## 6. Rollback-Trigger

| Kriterium | Schwellwert | Aktion |
| --------- | ----------- | ------ |
| Balance-Invariante verletzt | Jeder Verstoß | Rollback |
| Position-State korrupt | Jeder Fall | Rollback |
| Performance-Regression | >10% | Rollback |
| Precision-Loss | >0.01 USD | Rollback |

---

## 7. Rollback-Prozedur

```bash
# 1. Feature-Flag deaktivieren
export USE_RUST_PORTFOLIO=false

# 2. State-Recovery (falls nötig)
python -m src.backtest_engine.core.portfolio --recover-state

# 3. Issue erstellen
```

---

## 8. Abnahme-Checkliste

- [ ] Alle Unit-Tests grün
- [ ] Property-Based Tests bestehen
- [ ] Precision-Tests bestehen
- [ ] Performance-Target erreicht (≥7x speedup)
- [ ] Invarianten in CI validiert
- [ ] Code-Review abgeschlossen
- [ ] Dokumentation aktualisiert
