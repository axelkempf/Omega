---
module: symbol_data_slicer
phase: 2
prerequisites:
  - Type-Hints vollständig (mypy --strict)
  - ≥80% Test-Coverage
  - Performance-Baseline dokumentiert
  - FFI-Spec finalisiert
rollback_procedure: docs/runbooks/rollback_generic.md
---

# Migration Runbook: SymbolDataSlicer

## 1. Modul-Übersicht

| Attribut | Wert |
| -------- | ---- |
| Quell-Modul | `src/backtest_engine/core/symbol_data_slicer.py` |
| Ziel-Sprache | Rust (PyO3) |
| Priorität | P2 - Performance-kritisch |
| Geschätzter Aufwand | 3-4 Tage |

---

## 2. Voraussetzungen

### 2.1 Type Safety

```bash
# Type-Check ausführen
cd /Users/axelkempf/Omega
mypy --strict src/backtest_engine/core/symbol_data_slicer.py

# Erwartetes Ergebnis: Success: no issues found
```

### 2.2 Test-Coverage

```bash
pytest tests/test_symbol_data_slicer.py --cov=src/backtest_engine/core/symbol_data_slicer --cov-report=term-missing

# Erwartete Coverage: ≥80%
```

### 2.3 Performance-Baseline

```bash
pytest tests/test_symbol_data_slicer.py -k benchmark --benchmark-only --benchmark-json=reports/performance_baselines/symbol_data_slicer.json
```

---

## 3. Migration Steps

### Phase 1: Setup (Day 1)

```bash
# 1. Rust-Modul erstellen
mkdir -p src/rust_modules/omega_rust/src/data
touch src/rust_modules/omega_rust/src/data/slicer.rs

# 2. Arrow-Schema validieren
python -c "from src.shared.arrow_schemas import SYMBOL_SLICE_SCHEMA; print(SYMBOL_SLICE_SCHEMA)"
```

### Phase 2: Implementation (Day 2-3)

1. **Core-Funktionen in Rust implementieren**
   - `SymbolDataSlicerRust::new()`
   - `SymbolDataSlicerRust::get_slice()`
   - `SymbolDataSlicerRust::get_candle_at()`
   - `SymbolDataSlicerRust::get_lookback()`

2. **Python-Wrapper erstellen**
   - Feature-Flag `USE_RUST_SLICER` in Config
   - Fallback zu Python bei Rust-Fehler

### Phase 3: Testing (Day 3-4)

```bash
# Golden-File Tests
pytest tests/test_symbol_data_slicer.py -k rust --golden

# Performance Comparison
pytest tests/test_symbol_data_slicer.py -k benchmark --benchmark-compare=reports/performance_baselines/symbol_data_slicer.json
```

---

## 4. Validierung

### 4.1 Correctness Check

```bash
# Bit-für-Bit Vergleich Python vs Rust
pytest tests/test_symbol_data_slicer.py -k equivalence -v
```

### 4.2 Performance Check

| Operation | Python | Rust | Target | Status |
| --------- | ------ | ---- | ------ | ------ |
| get_slice (1K) | 2.5ms | - | 0.3ms | ⏳ |
| get_candle_at | 0.15ms | - | 0.01ms | ⏳ |
| get_lookback (200) | 0.8ms | - | 0.1ms | ⏳ |

---

## 5. Rollback-Trigger

| Kriterium | Schwellwert | Aktion |
| --------- | ----------- | ------ |
| Correctness-Tests fehlschlagen | >0 Fehler | Rollback |
| Performance-Regression | >10% langsamer | Rollback |
| Memory-Leak Detection | Jedes Leak | Rollback |
| Production Error Rate | >0.1% | Rollback |

---

## 6. Rollback-Prozedur

```bash
# 1. Feature-Flag deaktivieren
export USE_RUST_SLICER=false

# 2. Service neu starten
# (nur für Live-Engine relevant)

# 3. Issue erstellen mit Logs
```

---

## 7. Abnahme-Checkliste

- [ ] Alle Unit-Tests grün (Python + Rust)
- [ ] Golden-File Tests bestehen
- [ ] Performance-Target erreicht (≥8x speedup)
- [ ] Memory-Safety verifiziert (keine Leaks)
- [ ] Code-Review abgeschlossen
- [ ] Dokumentation aktualisiert
- [ ] Feature-Flag dokumentiert
