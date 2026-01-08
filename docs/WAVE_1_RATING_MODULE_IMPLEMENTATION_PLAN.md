# Wave 1: Rating Module Migration Implementation Plan

**Document Version:** 1.0  
**Created:** 2026-01-08  
**Updated:** 2026-01-08
**Status:** 📋 READY FOR IMPLEMENTATION  
**Modules:** `src/backtest_engine/rating/*.py`

---

## Executive Summary

Dieser Plan beschreibt die vollständige Implementierung der Migration der Rating-Module zu Rust als **Wave 1**. Das Ziel ist die Migration aller 6 Rating-Score-Module zu Rust mit vollständiger numerischer Parität zu den Python-Implementierungen.

### Warum Rating Module als Wave 1?

| Eigenschaft | Bewertung | Begründung |
|-------------|-----------|------------|
| **Pure Functions** | ✅ Ideal | Keine State-Abhängigkeiten, rein mathematisch |
| **Isolierte Logik** | ✅ Ideal | Keine externen Abhängigkeiten (außer NumPy für Python) |
| **Testbarkeit** | ✅ Ideal | Golden-Tests, Property-Tests, Determinismus nachgewiesen |
| **Batch-Potenzial** | ✅ Hoch | Optimizer-Szenarien mit vielen Evaluierungen |
| **Risiko** | ✅ Niedrig | Fehler isoliert, Feature-Flag ermöglicht Rollback |
| **Aufwand** | ⚡ Mittel | 5-7 Tage geschätzt |

### Module in Scope

| Modul | Funktion | Komplexität | Python LOC |
|-------|----------|-------------|------------|
| `strategy_rating.py` | Deployment-Entscheidung | ⭐ Niedrig | ~57 |
| `robustness_score_1.py` | Parameter-Jitter Robustness | ⭐⭐ Niedrig | ~83 |
| `stability_score.py` | Yearly Profit Stability | ⭐⭐ Niedrig | ~88 |
| `cost_shock_score.py` | Cost Sensitivity Analysis | ⭐ Niedrig | ~92 |
| `trade_dropout_score.py` | Trade Dropout Simulation | ⭐⭐⭐ Mittel | ~335 |
| `stress_penalty.py` | Basis-Penalty-Logik | ⭐ Niedrig | ~84 |

**Gesamt:** ~739 LOC Python → ~600-800 LOC Rust (geschätzt)

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
9. [Checklisten](#9-checklisten)
10. [Lessons Learned aus Wave 0](#10-lessons-learned-aus-wave-0)

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure-Readiness (✅ ERFÜLLT)

| Komponente | Status | Evidenz |
|------------|--------|---------|
| Rust Build System | ✅ | `src/rust_modules/omega_rust/Cargo.toml` |
| PyO3/Maturin | ✅ | Version 0.27 konfiguriert, Wave 0 erfolgreich |
| Error Handling | ✅ | `src/rust_modules/omega_rust/src/error.rs` |
| Golden-Tests | ✅ | `tests/golden/test_golden_rating.py` |
| Property-Tests | ✅ | `tests/property/test_prop_scoring.py` |
| Benchmarks | ✅ | `tests/benchmarks/test_bench_rating.py` |
| FFI-Spezifikation | ✅ | `docs/ffi/rating_modules.md` |
| Migration Runbook | ✅ | `docs/runbooks/rating_modules_migration.md` |
| mypy strict | ✅ | `backtest_engine.rating.*` strict-compliant |
| Wave 0 Pilot | ✅ | Slippage & Fee erfolgreich migriert |

### 1.2 Python-Modul Baseline

**Verzeichnis:** `src/backtest_engine/rating/`

Die aktuellen Python-Implementierungen (~739 LOC) enthalten:

| Modul | Haupt-Funktion(en) |
|-------|-------------------|
| `strategy_rating.py` | `rate_strategy_performance()` |
| `robustness_score_1.py` | `compute_robustness_score_1()` |
| `stability_score.py` | `compute_stability_score_and_wmape_from_yearly_profits()`, `compute_stability_score_from_yearly_profits()` |
| `stress_penalty.py` | `compute_penalty_profit_drawdown_sharpe()`, `score_from_penalty()` |
| `cost_shock_score.py` | `compute_cost_shock_score()`, `compute_multi_factor_cost_shock_score()`, `apply_cost_shock_inplace()` |
| `trade_dropout_score.py` | `simulate_trade_dropout_metrics()`, `simulate_trade_dropout_metrics_multi()`, `compute_trade_dropout_score()`, `compute_multi_run_trade_dropout_score()` |

### 1.3 Golden-File Referenz

**Datei:** `tests/golden/reference/rating/rating_modules_v1.json`

- **Outputs Hash:** `ebab73b47d1822759bbae18bd49bde6581751a63e1df978d0571534fd9afc682`
- **Seed:** 42
- **Toleranz:** 1e-10

### 1.4 Performance Baseline

**Datei:** `reports/performance_baselines/p0-01_rating.json`

| Operation | Python Baseline | Rust Target | Speedup-Ziel |
|-----------|-----------------|-------------|--------------|
| robustness_1 | ~1.3ms | <150µs | 8x |
| stability | ~80µs | <10µs | 8x |
| cost_shock | ~590µs | <75µs | 8x |
| trade_dropout | ~646µs | <80µs | 8x |
| ulcer_index | ~22.7ms | <3ms | 8x |
| strategy_rating | ~17µs | <2µs | 8x |
| tp_sl_stress | ~47.3ms | <6ms | 8x |

---

## 2. Architektur-Übersicht

### 2.1 Ziel-Architektur

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTEST ENGINE - RATING MODULE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │   Python API Layer (src/backtest_engine/rating/*.py)                   │ │
│  │                                                                        │ │
│  │  def compute_robustness_score_1(...) -> float:                         │ │
│  │      if USE_RUST_RATING:                                               │ │
│  │          return omega_rust.compute_robustness_score_1(...)  ◄── Rust   │ │
│  │      else:                                                             │ │
│  │          return _compute_robustness_score_1_python(...)     ◄── Python │ │
│  │                                                                        │ │
│  │  def compute_stability_score(...) -> float:                            │ │
│  │      if USE_RUST_RATING:                                               │ │
│  │          return omega_rust.compute_stability_score(...)     ◄── Rust   │ │
│  │      else:                                                             │ │
│  │          return _compute_stability_score_python(...)        ◄── Python │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              │ FFI Boundary (PyO3)                           │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │          Rust Layer (src/rust_modules/omega_rust/src/rating/)          │ │
│  │                                                                        │ │
│  │  pub fn compute_robustness_score_1(                                    │ │
│  │      base_metrics: HashMap<String, f64>,                               │ │
│  │      jitter_metrics: Vec<HashMap<String, f64>>,                        │ │
│  │      penalty_cap: f64,                                                 │ │
│  │  ) -> PyResult<f64>                                                    │ │
│  │                                                                        │ │
│  │  pub fn compute_stability_score(                                       │ │
│  │      profits_by_year: HashMap<i32, f64>,                               │ │
│  │      durations_by_year: Option<HashMap<i32, f64>>,                     │ │
│  │  ) -> PyResult<(f64, f64)>                                             │ │
│  │                                                                        │ │
│  │  pub fn compute_robustness_batch(...)       → Vec<f64>   ◄── Optimiert │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Feature-Flag-System

```python
# src/backtest_engine/rating/__init__.py

import os

def _check_rust_rating_available() -> bool:
    """Check if Rust rating functions are available."""
    try:
        from omega._rust import (
            compute_robustness_score_1,
            compute_stability_score,
            compute_penalty_profit_drawdown_sharpe,
            score_from_penalty,
            compute_cost_shock_score,
            compute_multi_factor_cost_shock_score,
            compute_trade_dropout_score,
            simulate_trade_dropout_metrics,
            rate_strategy_performance,
        )
        return True
    except ImportError:
        return False

def _is_rust_enabled() -> bool:
    """Determine if Rust rating should be used."""
    env_val = os.getenv("OMEGA_USE_RUST_RATING", "auto").lower()
    if env_val == "false":
        return False
    if env_val == "true":
        return _check_rust_rating_available()
    # "auto" - use if available
    return _check_rust_rating_available()

USE_RUST_RATING = _is_rust_enabled()

def get_rust_rating_status() -> dict:
    """Return status of Rust rating module for diagnostics."""
    available = _check_rust_rating_available()
    env_val = os.getenv("OMEGA_USE_RUST_RATING", "auto").lower()
    
    reason = "Available and enabled" if USE_RUST_RATING else (
        "Explicitly disabled" if env_val == "false" else
        "Module not available"
    )
    
    return {
        "available": available,
        "enabled": USE_RUST_RATING,
        "reason": reason,
        "env_var": env_val,
    }
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
│       │   ├── indicators/               # Bestehendes Modul
│       │   └── rating/                   # NEU: Rating-Module
│       │       ├── mod.rs                # NEU: Module exports
│       │       ├── common.rs             # NEU: Gemeinsame Helpers
│       │       ├── robustness.rs         # NEU: Robustness Score
│       │       ├── stability.rs          # NEU: Stability Score
│       │       ├── stress_penalty.rs     # NEU: Stress/Penalty Logik
│       │       ├── cost_shock.rs         # NEU: Cost Shock Score
│       │       ├── trade_dropout.rs      # NEU: Trade Dropout
│       │       └── strategy_rating.rs    # NEU: Strategy Rating
│       └── Cargo.toml                    # Dependencies ggf. erweitern
│
├── backtest_engine/
│   └── rating/
│       ├── __init__.py                   # NEU: Feature-Flag + Exports
│       ├── robustness_score_1.py         # Erweitert mit Rust-Integration
│       ├── stability_score.py            # Erweitert mit Rust-Integration
│       ├── stress_penalty.py             # Erweitert mit Rust-Integration
│       ├── cost_shock_score.py           # Erweitert mit Rust-Integration
│       ├── trade_dropout_score.py        # Erweitert mit Rust-Integration
│       └── strategy_rating.py            # Erweitert mit Rust-Integration
│
tests/
├── golden/
│   └── test_golden_rating.py             # Bestehendes, validiert Rust-Parität
├── property/
│   └── test_prop_scoring.py              # Bestehendes, erweitert für Rust
└── integration/
    └── test_rust_rating_parity.py        # NEU: Rust-spezifische Parity Tests
```

---

## 3. Implementierungs-Phasen

### Phase 1: Rust-Modul Setup (Tag 1, ~4h)

#### 3.1.1 Verzeichnisstruktur erstellen

```bash
mkdir -p src/rust_modules/omega_rust/src/rating
touch src/rust_modules/omega_rust/src/rating/mod.rs
touch src/rust_modules/omega_rust/src/rating/common.rs
touch src/rust_modules/omega_rust/src/rating/robustness.rs
touch src/rust_modules/omega_rust/src/rating/stability.rs
touch src/rust_modules/omega_rust/src/rating/stress_penalty.rs
touch src/rust_modules/omega_rust/src/rating/cost_shock.rs
touch src/rust_modules/omega_rust/src/rating/trade_dropout.rs
touch src/rust_modules/omega_rust/src/rating/strategy_rating.rs
```

#### 3.1.2 Module registrieren in lib.rs

```rust
pub mod rating;  // NEU

use rating::{
    compute_robustness_score_1,
    compute_robustness_score_1_batch,
    compute_stability_score,
    compute_stability_score_and_wmape,
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
    compute_cost_shock_score,
    compute_multi_factor_cost_shock_score,
    simulate_trade_dropout_metrics,
    compute_trade_dropout_score,
    compute_multi_run_trade_dropout_score,
    rate_strategy_performance,
};

#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bestehende Funktionen...
    
    // NEU: Rating Functions
    m.add_function(wrap_pyfunction!(compute_robustness_score_1, m)?)?;
    m.add_function(wrap_pyfunction!(compute_robustness_score_1_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stability_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stability_score_and_wmape, m)?)?;
    m.add_function(wrap_pyfunction!(compute_penalty_profit_drawdown_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(score_from_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cost_shock_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multi_factor_cost_shock_score, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_trade_dropout_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_trade_dropout_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multi_run_trade_dropout_score, m)?)?;
    m.add_function(wrap_pyfunction!(rate_strategy_performance, m)?)?;
    
    Ok(())
}
```

### Phase 2: Core Rating Implementation (Tag 1-3, ~16h)

#### 3.2.1 Common Helpers (`common.rs`)

Gemeinsame Funktionen für alle Rating-Module:
- `to_finite(x, default)` - NaN/Inf handling
- `pct_drop(base, x, invert)` - Relative Verschlechterung
- `clamp(value, min, max)` - Value clamping

#### 3.2.2 Stress Penalty (`stress_penalty.rs`)

Basis-Logik für alle Stress-basierten Scores:
- `compute_penalty_profit_drawdown_sharpe()` - Penalty-Berechnung
- `score_from_penalty()` - Penalty → Score Konvertierung

#### 3.2.3 Robustness Score (`robustness.rs`)

- `compute_robustness_score_1()` - Single evaluation
- `compute_robustness_score_1_batch()` - Batch für Optimizer

Kritische Design-Entscheidungen:
- HashMap<String, f64> für MetricsDict
- Penalty wird auf [0, penalty_cap] geclippt
- NaN/Inf werden als 0.0 behandelt
- Leere jitter_metrics → return 1.0 - penalty_cap

#### 3.2.4 Stability Score (`stability.rs`)

- `compute_stability_score_and_wmape()` - Score + WMAPE
- `compute_stability_score()` - Convenience wrapper

Kritische Design-Entscheidungen:
- HashMap<i32, f64> für profits_by_year
- Leap year handling: `_days_in_year()`
- S_min = max(100.0, 0.02 * |P_total|)

#### 3.2.5 Cost Shock Score (`cost_shock.rs`)

- `compute_cost_shock_score()` - Single shock
- `compute_multi_factor_cost_shock_score()` - Multiple shocks

Delegiert an `stress_penalty` Logik.

#### 3.2.6 Trade Dropout Score (`trade_dropout.rs`)

- `simulate_trade_dropout_metrics()` - Dropout simulation
- `compute_trade_dropout_score()` - Score berechnung
- `compute_multi_run_trade_dropout_score()` - Multi-run aggregation
- `_drawdown_from_results()` - Max drawdown helper
- `_sharpe_from_r_multiples()` - Sharpe helper

Kritische Design-Entscheidungen:
- Deterministische RNG via ChaCha8 (wie Wave 0)
- Fee-handling: net-of-fee wenn vorhanden
- Chronologische Sortierung vor Dropout

#### 3.2.7 Strategy Rating (`strategy_rating.rs`)

- `rate_strategy_performance()` - Threshold-basierte Bewertung

Einfachstes Modul - 5 Threshold-Checks.

### Phase 3: Python-Integration (Tag 3-4, ~8h)

#### 3.3.1 Feature-Flag System

Neues `__init__.py` mit:
- `USE_RUST_RATING` global flag
- `get_rust_rating_status()` für Diagnostik
- Bedingte Imports

#### 3.3.2 Module erweitern

Jedes Python-Modul erhält:
- `_<function>_python()` - Original-Implementation
- `_<function>_rust()` - Rust-Wrapper
- `<function>()` - Dispatch basierend auf Feature-Flag

#### 3.3.3 Abwärtskompatibilität

Die API bleibt **100% abwärtskompatibel**:
```python
# Bestehender Code funktioniert unverändert:
score = compute_robustness_score_1(base_metrics, jitter_metrics)

# Neue optionale Batch-Features:
scores = compute_robustness_score_1_batch(metrics_list)
```

### Phase 4: Testing & Validierung (Tag 4-6, ~12h)

#### 3.4.1 Test-Strategie

```
                    ┌─────────────────┐
                    │   Golden File   │ ← Determinismus-Gate
                    │     Tests       │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Property      │ ← Invarianten
                    │     Tests       │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Integration   │ ← Rust↔Python Parität
                    │     Tests       │
                    └────────┬────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                      │
    ┌─────┴─────┐                          ┌─────┴─────┐
    │   Rust    │                          │  Python   │
    │   Unit    │                          │   Unit    │
    │   Tests   │                          │   Tests   │
    └───────────┘                          └───────────┘
```

#### 3.4.2 Test-Dateien

| Datei | Typ | Gate |
|-------|-----|------|
| `tests/golden/test_golden_rating.py` | Golden | ✅ CI |
| `tests/property/test_prop_scoring.py` | Property | ✅ CI |
| `tests/integration/test_rust_rating_parity.py` | Integration | ✅ CI (wenn Rust gebaut) |
| `src/rust_modules/omega_rust/src/rating/*.rs` | Rust Unit | ✅ cargo test |

### Phase 5: Benchmarking & Rollout (Tag 6-7, ~8h)

#### 3.5.1 Benchmark-Validierung

```bash
# Rust-only Benchmarks
cd src/rust_modules/omega_rust
cargo bench

# Python-Integration Benchmarks
pytest tests/benchmarks/test_bench_rating.py -v --benchmark-json=results.json
```

#### 3.5.2 Performance-Validierung

| Metrik | Python Baseline | Rust Target | Akzeptanz |
|--------|-----------------|-------------|-----------|
| robustness_1 (50 jitter) | ~1.3ms | <150µs | ✅ wenn < 200µs |
| stability (5 Jahre) | ~80µs | <10µs | ✅ wenn < 15µs |
| cost_shock (3 factors) | ~590µs | <75µs | ✅ wenn < 100µs |
| trade_dropout (500 trades) | ~646µs | <80µs | ✅ wenn < 100µs |

---

## 4. Rust-Implementation Details

### 4.1 Zusammenfassung der Rust-Dateien

| Datei | Beschreibung | LOC (geschätzt) |
|-------|--------------|-----------------|
| `src/rust_modules/omega_rust/src/rating/mod.rs` | Module exports | ~30 |
| `src/rust_modules/omega_rust/src/rating/common.rs` | Helpers | ~50 |
| `src/rust_modules/omega_rust/src/rating/robustness.rs` | Robustness + Tests | ~150 |
| `src/rust_modules/omega_rust/src/rating/stability.rs` | Stability + Tests | ~120 |
| `src/rust_modules/omega_rust/src/rating/stress_penalty.rs` | Penalty + Tests | ~100 |
| `src/rust_modules/omega_rust/src/rating/cost_shock.rs` | Cost Shock + Tests | ~80 |
| `src/rust_modules/omega_rust/src/rating/trade_dropout.rs` | Dropout + Tests | ~200 |
| `src/rust_modules/omega_rust/src/rating/strategy_rating.rs` | Rating + Tests | ~80 |
| `src/rust_modules/omega_rust/src/lib.rs` | Module registration | ~20 |

**Gesamt:** ~830 LOC Rust

### 4.2 Dependencies

Keine zusätzlichen Dependencies erforderlich - Wave 0 hat bereits:
- `rand` / `rand_chacha` für RNG (trade_dropout)
- `pyo3` für Python-Bindings
- Alle anderen Berechnungen nutzen nur Rust stdlib

### 4.3 Error Handling

Alle Rust-Funktionen nutzen das bestehende Error-Handling aus `src/rust_modules/omega_rust/src/error.rs`:
- `OmegaError::InvalidParameter` für ungültige Eingaben
- `OmegaError::CalculationError` für Berechnungsfehler
- Automatische Konvertierung zu Python `ValueError`/`RuntimeError`

### 4.4 Type Mappings

| Python Type | Rust Type | Notes |
|-------------|-----------|-------|
| `Mapping[str, float]` | `HashMap<String, f64>` | MetricsDict |
| `Mapping[int, float]` | `HashMap<i32, f64>` | YearlyProfits |
| `Sequence[Mapping[str, float]]` | `Vec<HashMap<String, f64>>` | JitterMetrics |
| `float` | `f64` | Alle Scores |
| `Optional[int]` | `Option<u64>` | Seeds |

---

## 5. Python-Integration Details

### 5.1 Environment Variables

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| `OMEGA_USE_RUST_RATING` | `"auto"` | `"true"` / `"false"` / `"auto"` |
| `OMEGA_REQUIRE_RUST_FFI` | `"0"` | `"1"` = Fehler wenn Rust nicht verfügbar |

### 5.2 Import-Pfade

```python
# Primärer Import (nutzt automatisch Rust wenn verfügbar)
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import compute_stability_score_from_yearly_profits

# Direkter Rust-Import (für Tests/Benchmarks)
from omega._rust import compute_robustness_score_1 as compute_robustness_score_1_rust

# Status-Check
from backtest_engine.rating import get_rust_rating_status
```

### 5.3 Migration Pattern (pro Modul)

```python
# Beispiel: robustness_score_1.py

from __future__ import annotations
import math
from typing import Mapping, Sequence
import numpy as np

# Lazy import for Rust module
_rust_module = None

def _get_rust_module():
    global _rust_module
    if _rust_module is None:
        try:
            import omega_rust
            _rust_module = omega_rust
        except ImportError:
            _rust_module = False
    return _rust_module if _rust_module else None

# Feature flag check
def _use_rust() -> bool:
    import os
    env_val = os.getenv("OMEGA_USE_RUST_RATING", "auto").lower()
    if env_val == "false":
        return False
    rust = _get_rust_module()
    return rust is not None and hasattr(rust, 'compute_robustness_score_1')

# Original Python implementation (renamed)
def _compute_robustness_score_1_python(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    # ... original implementation ...

# Rust wrapper
def _compute_robustness_score_1_rust(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    rust = _get_rust_module()
    return rust.compute_robustness_score_1(
        dict(base_metrics),
        [dict(m) for m in jitter_metrics],
        penalty_cap,
    )

# Public API (dispatch)
def compute_robustness_score_1(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Robustness-1 score (parameter jitter).
    
    Uses Rust implementation if available for better performance.
    """
    if _use_rust():
        return _compute_robustness_score_1_rust(
            base_metrics, jitter_metrics, penalty_cap=penalty_cap
        )
    return _compute_robustness_score_1_python(
        base_metrics, jitter_metrics, penalty_cap=penalty_cap
    )
```

---

## 6. Test-Strategie

### 6.1 Golden-File Tests

**Datei:** `tests/golden/test_golden_rating.py`

Validiert:
- Alle Score-Outputs sind identisch mit Reference
- Hash-Vergleich für deterministische Outputs
- Toleranz: 1e-10

### 6.2 Property-Based Tests

**Datei:** `tests/property/test_prop_scoring.py`

Invarianten:
1. Alle Scores in [0, 1]
2. Determinismus (gleicher Input → gleicher Output)
3. Score = 1 bei identischen Jitter-Metriken
4. Monotonie (schlechtere Inputs → niedrigere Scores)
5. NaN/Inf Handling

### 6.3 Parity Tests

**Datei:** `tests/integration/test_rust_rating_parity.py` (NEU)

```python
@pytest.mark.rust_integration
class TestRobustnessRustParity:
    """Tests für Rust↔Python Robustness Score Parity."""
    
    def test_robustness_parity_basic(self, rust_available) -> None:
        """Robustness Score muss zwischen Python und Rust identisch sein."""
        if not rust_available:
            pytest.skip("Rust-Modul nicht verfügbar")
            
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}
        jitter = [
            {"profit": 9000.0, "avg_r": 1.3, "winrate": 0.55, "drawdown": 2200.0},
            {"profit": 9500.0, "avg_r": 1.4, "winrate": 0.58, "drawdown": 2100.0},
        ]
        
        # Python result
        os.environ["OMEGA_USE_RUST_RATING"] = "false"
        python_result = compute_robustness_score_1(base, jitter)
        
        # Rust result
        os.environ["OMEGA_USE_RUST_RATING"] = "true"
        rust_result = compute_robustness_score_1(base, jitter)
        
        assert abs(python_result - rust_result) < 1e-10, (
            f"Parity Error!\nPython: {python_result}\nRust: {rust_result}"
        )
```

### 6.4 Rust Unit Tests

Jedes Rust-Modul enthält `#[cfg(test)]` Module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_robustness_score_bounds() {
        // Score muss in [0, 1] liegen
    }
    
    #[test]
    fn test_robustness_deterministic() {
        // Gleicher Input → gleicher Output
    }
    
    #[test]
    fn test_robustness_empty_jitter() {
        // Leere Jitter → 1.0 - penalty_cap
    }
}
```

---

## 7. Validierung & Akzeptanzkriterien

### 7.1 Funktionale Kriterien

- [ ] **F1:** Alle 6 Rating-Module implementiert und registriert
- [ ] **F2:** Golden-File Tests bestanden (Hash-Match)
- [ ] **F3:** Property-Based Tests bestanden
- [ ] **F4:** Numerische Parität < 1e-10 zwischen Python und Rust
- [ ] **F5:** Edge-Cases korrekt behandelt (empty, NaN, negative)
- [ ] **F6:** Backtest-Workflow unverändert lauffähig

### 7.2 Performance-Kriterien

| Operation | Python Baseline | Rust Target | Status |
|-----------|-----------------|-------------|--------|
| robustness_1 (50 jitter) | ~1.3ms | <150µs | ⏳ |
| stability (5 Jahre) | ~80µs | <10µs | ⏳ |
| cost_shock (3 factors) | ~590µs | <75µs | ⏳ |
| trade_dropout (500 trades) | ~646µs | <80µs | ⏳ |
| Full Rating Pipeline | ~450ms | <60ms | ⏳ |

**Ziel-Speedup:** 8x (wie in FFI-Spec definiert)

### 7.3 Qualitäts-Kriterien

- [ ] **Q1:** `cargo clippy --all-targets -- -D warnings` = 0 Warnungen
- [ ] **Q2:** `cargo test` = alle Tests bestanden
- [ ] **Q3:** `mypy --strict` = keine Fehler für modifizierte Python-Dateien
- [ ] **Q4:** Docstrings für alle öffentlichen Funktionen
- [ ] **Q5:** CHANGELOG.md Eintrag erstellt
- [ ] **Q6:** architecture.md aktualisiert

### 7.4 Akzeptanz-Toleranzen

| Metrik | Toleranz | Grund |
|--------|----------|-------|
| Numerische Differenz | ≤ 1e-10 | IEEE 754 double precision (strenger als Wave 0) |
| Hash-Differenz | 0 | Binäre Identität für Golden Files |
| Performance | ≥ 8x (Target) | Migrations-Ziel laut FFI-Spec |

---

## 8. Rollback-Plan

### 8.1 Sofort-Rollback (< 1 Minute)

```bash
# Option 1: Feature-Flag deaktivieren
export OMEGA_USE_RUST_RATING=false

# Option 2: In Code (falls notwendig)
# In src/backtest_engine/rating/__init__.py:
USE_RUST_RATING = False
```

### 8.2 Rollback-Trigger

| Trigger | Schwellwert | Aktion |
|---------|-------------|--------|
| Golden-File Hash Mismatch | Jeder | Sofort-Rollback |
| Numerische Differenz | > 1e-10 | Sofort-Rollback |
| Performance-Regression | > 5% langsamer als Python | Analyse → ggf. Rollback |
| Runtime Error | Jeder in Production | Sofort-Rollback |
| NaN/Inf in Output | Jeder | Sofort-Rollback |

### 8.3 Post-Rollback

1. Issue erstellen mit Reproduktionsschritten
2. Root-Cause-Analysis durchführen
3. Fix entwickeln und neue Tests hinzufügen
4. Re-Deployment nach vollständiger Validierung

### 8.4 Rollback-Validierung

```bash
# Nach Rollback verifizieren
export OMEGA_USE_RUST_RATING=false
pytest tests/golden/test_golden_rating.py -v
pytest tests/property/test_prop_scoring.py -v
pytest tests/benchmarks/test_bench_rating.py -v
```

---

## 9. Checklisten

### 9.1 Pre-Implementation Checklist

- [x] FFI-Spezifikation finalisiert (`docs/ffi/rating_modules.md`)
- [x] Golden-Tests vorhanden (`tests/golden/test_golden_rating.py`)
- [x] Property-Tests vorhanden (`tests/property/test_prop_scoring.py`)
- [x] Benchmarks vorhanden (`tests/benchmarks/test_bench_rating.py`)
- [x] Performance Baseline dokumentiert (`reports/performance_baselines/p0-01_rating.json`)
- [x] Rust Build-System funktioniert (Wave 0 erfolgreich)
- [x] Migration Readiness ✅ (`docs/MIGRATION_READINESS_VALIDATION.md`)
- [x] Wave 0 Lessons Learned dokumentiert

### 9.2 Implementation Checklist

#### Phase 1: Setup ⏳
- [ ] Verzeichnisstruktur erstellen (`src/rust_modules/omega_rust/src/rating/`)
- [ ] `mod.rs` erstellen
- [ ] `lib.rs` Module registrieren

#### Phase 2: Rust-Code ⏳
- [ ] `common.rs` implementieren (Helpers)
- [ ] `stress_penalty.rs` implementieren
- [ ] `robustness.rs` implementieren
- [ ] `stability.rs` implementieren
- [ ] `cost_shock.rs` implementieren
- [ ] `trade_dropout.rs` implementieren (inkl. ChaCha8 RNG)
- [ ] `strategy_rating.rs` implementieren
- [ ] `cargo test` bestanden
- [ ] `cargo clippy` bestanden

#### Phase 3: Python-Integration ⏳
- [ ] `__init__.py` mit Feature-Flag erstellen
- [ ] `robustness_score_1.py` erweitern
- [ ] `stability_score.py` erweitern
- [ ] `stress_penalty.py` erweitern
- [ ] `cost_shock_score.py` erweitern
- [ ] `trade_dropout_score.py` erweitern
- [ ] `strategy_rating.py` erweitern
- [ ] mypy strict compliance

#### Phase 4: Testing ⏳
- [ ] Golden-Tests bestanden (Python mode)
- [ ] Golden-Tests bestanden (Rust mode)
- [ ] Property-Tests bestanden
- [ ] Parity-Tests erstellt und bestanden
- [ ] Rust-Unit-Tests bestanden
- [ ] Backtest-Workflow validiert

#### Phase 5: Benchmarking ⏳
- [ ] Rust Benchmarks erstellt
- [ ] Performance-Ziele erreicht (8x Speedup)
- [ ] Benchmark-Ergebnisse dokumentiert

### 9.3 Post-Implementation Checklist

- [ ] Dokumentation aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] Code-Review abgeschlossen
- [ ] Performance-Benchmark dokumentiert
- [ ] Sign-off Matrix ausgefüllt

### 9.4 Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Developer | | | ⏳ |
| Integration Tests | pytest | | ⏳ |
| Backtest Validation | runner.py | | ⏳ |
| Tech Lead | axelkempf | | ⏳ |

---

## 10. Lessons Learned aus Wave 0

### 10.1 Kritische Issues aus Wave 0 (beachten!)

#### Issue 1: Namespace Conflict (`logging` module)
- **Problem:** Python's `logging` module was shadowed by `backtest_engine/logging/`
- **Symptom:** `AttributeError: module 'logging' has no attribute 'getLogger'`
- **Resolution:** Renamed directory to `bt_logging`
- **Prävention für Wave 1:** 
  - Keine Module mit Namen von Python stdlib
  - `rating` ist sicher (kein stdlib Modul)

#### Issue 2: PYTHONPATH Configuration
- **Problem:** `ModuleNotFoundError: No module named 'configs'`
- **Resolution:** Required both project root AND src in PYTHONPATH
- **Prävention für Wave 1:** 
  - Tests mit korrektem PYTHONPATH laufen lassen
  - CI bereits korrekt konfiguriert

#### Issue 3: RNG Unterschiede
- **Problem:** Python `random.Random` vs Rust `ChaCha8Rng` produzieren unterschiedliche Sequenzen
- **Observed:** <0.27 pips Varianz pro Trade bei Slippage
- **Assessment:** Akzeptabel für Wave 0 (innerhalb Toleranz)
- **Relevanz für Wave 1:** 
  - `trade_dropout_score.py` nutzt `np.random.Generator` (NumPy)
  - Rust wird `ChaCha8Rng` nutzen
  - **Erwartung:** Leichte Unterschiede bei dropout_metrics (verschiedene Trades werden gedroppt)
  - **Lösung:** Score-Vergleich statt Metrics-Vergleich; Scores sollten innerhalb Toleranz sein

### 10.2 Erfolgreiche Patterns aus Wave 0 (wiederverwenden!)

#### Pattern 1: Feature-Flag Design
```python
USE_RUST_SLIPPAGE_FEE = (
    os.getenv("OMEGA_USE_RUST_SLIPPAGE_FEE", "auto") != "false" 
    and _check_rust_available()
)
```
→ Übernehmen für `USE_RUST_RATING`

#### Pattern 2: Batch-First Design
- Single-Operations haben hohen FFI-Overhead (~5µs)
- Batch-Operations amortisieren Overhead
→ `compute_robustness_score_1_batch()` etc. bereitstellen

#### Pattern 3: Determinismus via ChaCha8
- ChaCha8Rng ist plattformübergreifend deterministisch
- Seed-Management über `Option<u64>`
→ Für `trade_dropout` übernehmen

#### Pattern 4: PyO3 Signature mit Defaults
```rust
#[pyfunction]
#[pyo3(signature = (price, direction, pip_size, fixed_pips, random_pips, seed=None))]
pub fn calculate_slippage(...) -> PyResult<f64>
```
→ Für alle Rating-Funktionen übernehmen

### 10.3 Performance-Erkenntnisse

| Erkenntnis | Implikation für Wave 1 |
|------------|------------------------|
| FFI-Overhead ~5µs pro Call | Batch-APIs bereitstellen |
| 14.4x Speedup bei Batch (1K) | Rating-Batch sollte ähnlich sein |
| SIMD-Potenzial nicht voll ausgeschöpft | Für große Batches (>100) optimieren |

### 10.4 Empfehlungen für Wave 1

1. **Batch-First Design:** Prioritize batch operations for maximum speedup
2. **FFI Threshold:** Consider batch size >10 before switching to Rust
3. **RNG Strategy:** Document RNG differences clearly (Python NumPy vs Rust ChaCha8)
4. **Test Strategy:** Test Scores, nicht raw metrics (RNG-unabhängig)
5. **Golden Files:** Bei RNG-basierten Tests → Golden für Rust neu generieren

---

## 11. Timeline & Ressourcen

### 11.1 Geschätzter Zeitplan

| Phase | Dauer | Beschreibung |
|-------|-------|--------------|
| Phase 1: Setup | 0.5 Tage | Verzeichnisse, mod.rs, lib.rs |
| Phase 2: Rust-Implementation | 2-3 Tage | Alle 6 Module + Unit Tests |
| Phase 3: Python-Integration | 1 Tag | Feature-Flags, Wrappers |
| Phase 4: Testing | 2 Tage | Golden, Property, Parity, Backtest |
| Phase 5: Benchmarking | 1 Tag | Performance-Validierung, Doku |

**Gesamt:** 5-7 Tage

### 11.2 Risiken & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| RNG-Divergenz bei trade_dropout | Mittel | Niedrig | Score-basierte Tests (nicht raw metrics) |
| Performance unter 8x | Niedrig | Mittel | Batch-APIs, SIMD-Optimierung |
| Numerische Präzisionsprobleme | Niedrig | Hoch | Extensive Edge-Case Tests |
| FFI-Overhead dominiert | Niedrig | Mittel | Batch-Threshold implementieren |

---

## 12. Referenzen

- [ADR-0001: Migration Strategy](./adr/ADR-0001-migration-strategy.md)
- [ADR-0003: Error Handling](./adr/ADR-0003-error-handling.md)
- [FFI Specification: Rating Modules](./ffi/rating_modules.md)
- [Migration Runbook: Rating Modules](./runbooks/rating_modules_migration.md)
- [Migration Readiness Validation](./MIGRATION_READINESS_VALIDATION.md)
- [Wave 0 Implementation Plan](./WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md)
- [Golden-File Reference](../tests/golden/reference/rating/rating_modules_v1.json)
- [Performance Baseline](../reports/performance_baselines/p0-01_rating.json)
- [Property Tests](../tests/property/test_prop_scoring.py)
- [Benchmarks](../tests/benchmarks/test_bench_rating.py)

---

## Änderungshistorie

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-08 | 1.0 | Initiale Version | AI Agent |

---

*Document Status: 📋 READY FOR IMPLEMENTATION*
