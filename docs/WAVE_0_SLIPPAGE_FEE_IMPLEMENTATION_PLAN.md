# Wave 0: Slippage & Fee Migration Implementation Plan

**Document Version:** 1.0  
**Created:** 2026-01-08  
**Status:** 🟢 READY FOR IMPLEMENTATION  
**Pilot Module:** `src/backtest_engine/core/slippage_and_fee.py`

---

## Executive Summary

Dieser Plan beschreibt die vollständige Implementierung der Migration des Slippage & Fee Moduls zu Rust als **Wave 0 Pilotprojekt**. Das Ziel ist die Validierung der gesamten Migrations-Toolchain bei gleichzeitiger Sicherstellung identischer Backtest-Ergebnisse.

### Warum Slippage & Fee als Pilot?

| Eigenschaft | Bewertung | Begründung |
|-------------|-----------|------------|
| **Pure Functions** | ✅ Ideal | Keine State-Abhängigkeiten, rein mathematisch |
| **Isolierte Logik** | ✅ Ideal | Keine Abhängigkeiten zu anderen Modulen |
| **Testbarkeit** | ✅ Ideal | Deterministisches Verhalten, Golden-Tests vorhanden |
| **SIMD-Potenzial** | ✅ Hoch | Batch-Berechnungen für Optimizer-Szenarien |
| **Risiko** | ✅ Niedrig | Fehler isoliert, einfacher Rollback |
| **Aufwand** | ✅ Gering | 2-3 Tage geschätzt |

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

---

## 1. Voraussetzungen & Status

### 1.1 Infrastructure-Readiness (✅ ERFÜLLT)

| Komponente | Status | Evidenz |
|------------|--------|---------|
| Rust Build System | ✅ | `src/rust_modules/omega_rust/Cargo.toml` |
| PyO3/Maturin | ✅ | Version 0.27 konfiguriert |
| Error Handling | ✅ | `src/rust_modules/omega_rust/src/error.rs` |
| Golden-Tests | ✅ | `tests/golden/test_golden_slippage_fee.py` |
| FFI-Spezifikation | ✅ | `docs/ffi/slippage_fee.md` |
| Migration Runbook | ✅ | `docs/runbooks/slippage_fee_migration.md` |
| mypy strict | ✅ | `backtest_engine.core.*` strict-compliant |

### 1.2 Python-Modul Baseline

**Datei:** `src/backtest_engine/core/slippage_and_fee.py`

Die aktuelle Python-Implementation (~77 LOC) enthält:

- `SlippageModel`: Berechnet Ausführungspreis mit fixem + zufälligem Slippage
- `FeeModel`: Berechnet Handelsgebühren basierend auf Notional

### 1.3 Golden-File Referenz

**Datei:** `tests/golden/reference/slippage_fee/slippage_fee_v1.json`

- **Slippage Hash:** `da570884f652a2e9604ccb851ae4d0a650d4a274902cc874eb1ab22366a7adcd`
- **Fee Hash:** `c00ad9a2d1363bc538de442432672f74aaf803950e51988027ca0106dd477289`
- **Seed:** 42
- **Toleranz:** 1e-8

---

## 2. Architektur-Übersicht

### 2.1 Ziel-Architektur

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTEST ENGINE                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │   Python API Layer (src/backtest_engine/core/slippage_and_fee.py)     │ │
│  │                                                                        │ │
│  │  class SlippageModel:                                                  │ │
│  │      def apply(...) -> float:                                          │ │
│  │          if USE_RUST:                                                  │ │
│  │              return omega_rust.calculate_slippage(...)  ◄── Rust       │ │
│  │          else:                                                         │ │
│  │              return self._python_apply(...)             ◄── Fallback   │ │
│  │                                                                        │ │
│  │  class FeeModel:                                                       │ │
│  │      def calculate(...) -> float:                                      │ │
│  │          if USE_RUST:                                                  │ │
│  │              return omega_rust.calculate_fee(...)       ◄── Rust       │ │
│  │          else:                                                         │ │
│  │              return self._python_calculate(...)         ◄── Fallback   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              │ FFI Boundary (PyO3)                           │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │          Rust Layer (src/rust_modules/omega_rust/src/costs/)           │ │
│  │                                                                        │ │
│  │  pub fn calculate_slippage(                                            │ │
│  │      price: f64, direction: i8, pip_size: f64,                         │ │
│  │      fixed_pips: f64, random_pips: f64, seed: Option<u64>              │ │
│  │  ) -> PyResult<f64>                                                    │ │
│  │                                                                        │ │
│  │  pub fn calculate_fee(                                                 │ │
│  │      volume_lots: f64, price: f64, contract_size: f64,                 │ │
│  │      per_million: f64, min_fee: f64                                    │ │
│  │  ) -> PyResult<f64>                                                    │ │
│  │                                                                        │ │
│  │  pub fn calculate_slippage_batch(...)  → Vec<f64>   ◄── SIMD optimiert │ │
│  │  pub fn calculate_fee_batch(...)       → Vec<f64>   ◄── SIMD optimiert │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Feature-Flag-System

```python
# Automatische Detection mit Override-Möglichkeit
import os

def _check_rust_available() -> bool:
    try:
        from omega._rust import calculate_slippage, calculate_fee
        return True
    except ImportError:
        return False

USE_RUST_SLIPPAGE_FEE = (
    os.getenv("OMEGA_USE_RUST_SLIPPAGE_FEE", "auto") != "false" 
    and _check_rust_available()
)
```

### 2.3 Datei-Struktur nach Migration

```
src/
├── rust_modules/
│   └── omega_rust/
│       ├── src/
│       │   ├── lib.rs                    # Modul-Registration erweitern
│       │   ├── error.rs                  # Bestehendes Error-Handling
│       │   ├── indicators/               # Bestehendes Modul
│       │   └── costs/                    # NEU: Kosten-Module
│       │       ├── mod.rs                # NEU: Module exports
│       │       ├── slippage.rs           # NEU: Slippage-Implementierung
│       │       └── fee.rs                # NEU: Fee-Implementierung
│       └── Cargo.toml                    # rand-Dependency hinzufügen
│
├── backtest_engine/
│   └── core/
│       └── slippage_and_fee.py           # Erweitert mit Rust-Integration
│
└── shared/
    └── arrow_schemas.py                  # Optional: EXECUTION_COSTS_SCHEMA

tests/
├── golden/
│   └── test_golden_slippage_fee.py       # Bestehendes, validiert Rust-Parität
└── integration/
    └── test_slippage_fee_rust.py         # NEU: Rust-spezifische Tests
```

---

## 3. Implementierungs-Phasen

### Phase 1: Rust-Modul Setup (Tag 1, ~4h)

#### 3.1.1 Verzeichnisstruktur erstellen

```bash
mkdir -p src/rust_modules/omega_rust/src/costs
touch src/rust_modules/omega_rust/src/costs/mod.rs
touch src/rust_modules/omega_rust/src/costs/slippage.rs
touch src/rust_modules/omega_rust/src/costs/fee.rs
```

#### 3.1.2 Cargo.toml aktualisieren

```toml
# Hinzufügen zu [dependencies]
rand = "0.8"           # Für deterministische Random-Slippage
rand_chacha = "0.3"    # ChaCha RNG für Reproduzierbarkeit
```

#### 3.1.3 Module registrieren in lib.rs

```rust
pub mod costs;  // NEU

use costs::{calculate_fee, calculate_slippage, calculate_slippage_batch, calculate_fee_batch};

#[pymodule]
fn omega_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bestehende Funktionen...
    
    // NEU: Cost Functions
    m.add_function(wrap_pyfunction!(calculate_slippage, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fee, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_slippage_batch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fee_batch, m)?)?;
    
    Ok(())
}
```

### Phase 2: Core-Implementation (Tag 1-2, ~8h)

#### 3.2.1 Slippage-Implementation

**Datei:** `src/rust_modules/omega_rust/src/costs/slippage.rs`

Kernfunktionen:
- `calculate_slippage()` - Single-Trade Berechnung
- `calculate_slippage_batch()` - Batch-Berechnung für Optimizer
- Deterministische Random-Komponente via ChaCha8 RNG

Kritische Design-Entscheidungen:
- Direction als i8 (1=long, -1=short) statt String für Effizienz
- ChaCha8 RNG garantiert Plattform-übergreifende Reproduzierbarkeit
- Seed als Option<u64> für optionale Determinismus-Kontrolle

#### 3.2.2 Fee-Implementation

**Datei:** `src/rust_modules/omega_rust/src/costs/fee.rs`

Kernfunktionen:
- `calculate_fee()` - Single-Trade Berechnung
- `calculate_fee_batch()` - Batch-Berechnung

Formel: `fee = max(notional / 1_000_000 * per_million, min_fee)`
wobei `notional = volume_lots * contract_size * price`

#### 3.2.3 Module Export

**Datei:** `src/rust_modules/omega_rust/src/costs/mod.rs`

Re-exportiert alle öffentlichen Funktionen aus `slippage.rs` und `fee.rs`.

### Phase 3: Python-Integration (Tag 2, ~4h)

#### 3.3.1 Erweiterte slippage_and_fee.py

Änderungen:
- Feature-Flag `USE_RUST_SLIPPAGE_FEE` hinzufügen
- `_apply_rust()` und `_apply_python()` Methoden trennen
- Neue `apply_batch()` und `calculate_batch()` Methoden
- Optionaler `seed` Parameter für Determinismus

#### 3.3.2 Abwärtskompatibilität

Die API bleibt **100% abwärtskompatibel**:
```python
# Bestehender Code funktioniert unverändert:
model = SlippageModel(fixed_pips=0.5, random_pips=1.0)
adjusted = model.apply(1.10000, "long", 0.0001)

# Neue optionale Features:
adjusted = model.apply(1.10000, "long", 0.0001, seed=42)  # Determinismus
adjusted_batch = model.apply_batch(prices, directions, seed=42)  # Batch
```

### Phase 4: Testing & Validierung (Tag 2-3, ~6h)

#### 3.4.1 Test-Strategie

```
                    ┌─────────────────┐
                    │   Golden File   │ ← Determinismus-Gate
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
| `tests/golden/test_golden_slippage_fee.py` | Golden | ✅ CI |
| `tests/integration/test_slippage_fee_rust.py` | Integration | ✅ CI (wenn Rust gebaut) |
| `src/rust_modules/omega_rust/src/costs/*.rs` | Rust Unit | ✅ cargo test |

---

## 4. Rust-Implementation Details

### 4.1 Zusammenfassung der Rust-Dateien

| Datei | Beschreibung | LOC (geschätzt) |
|-------|--------------|-----------------|
| `src/costs/mod.rs` | Module exports | ~20 |
| `src/costs/slippage.rs` | Slippage-Berechnung + Tests | ~150 |
| `src/costs/fee.rs` | Fee-Berechnung + Tests | ~120 |
| `src/lib.rs` | Module registration (Erweiterung) | ~10 |

**Gesamt:** ~300 LOC Rust

### 4.2 Dependencies

```toml
# Hinzufügen zu Cargo.toml [dependencies]
rand = "0.8"           # RNG für Random-Slippage
rand_chacha = "0.3"    # ChaCha8 RNG für Determinismus
```

### 4.3 Error Handling

Alle Rust-Funktionen nutzen das bestehende Error-Handling aus `src/error.rs`:
- `OmegaError::InvalidParameter` für ungültige Eingaben
- Automatische Konvertierung zu Python `ValueError`/`RuntimeError`

---

## 5. Python-Integration Details

### 5.1 Environment Variables

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| `OMEGA_USE_RUST_SLIPPAGE_FEE` | `"auto"` | `"true"` / `"false"` / `"auto"` |
| `OMEGA_REQUIRE_RUST_FFI` | `"0"` | `"1"` = Fehler wenn Rust nicht verfügbar |

### 5.2 Import-Pfade

```python
# Primärer Import (nutzt automatisch Rust wenn verfügbar)
from backtest_engine.core.slippage_and_fee import SlippageModel, FeeModel

# Direkter Rust-Import (für Tests/Benchmarks)
from omega._rust import calculate_slippage, calculate_fee
```

---

## 6. Validierung & Akzeptanzkriterien

### 6.1 Funktionale Kriterien

- [ ] **F1:** `SlippageModel.apply()` liefert identische Ergebnisse wie Python-Baseline
- [ ] **F2:** `FeeModel.calculate()` liefert identische Ergebnisse wie Python-Baseline
- [ ] **F3:** Golden-File Hash bleibt unverändert
- [ ] **F4:** Alle bestehenden Backtest-Ergebnisse bleiben identisch
- [ ] **F5:** Direction-Awareness korrekt (long ↑, short ↓)
- [ ] **F6:** Minimum-Fee wird korrekt angewendet

### 6.2 Performance-Kriterien

| Operation | Python Baseline | Rust Target | Status |
|-----------|-----------------|-------------|--------|
| Slippage (single) | ~0.02ms | <0.001ms | ⏳ PLANNED |
| Slippage (batch 1K) | ~15ms | <0.5ms | ⏳ PLANNED |
| Fee (single) | ~0.03ms | <0.002ms | ⏳ PLANNED |
| Fee (batch 1K) | ~25ms | <1ms | ⏳ PLANNED |

### 6.3 Qualitäts-Kriterien

- [ ] **Q1:** `cargo clippy --all-targets -- -D warnings` = 0 Warnungen
- [ ] **Q2:** `cargo test` = alle Tests bestanden
- [ ] **Q3:** `mypy --strict` = keine Fehler für modifizierte Python-Dateien
- [ ] **Q4:** Docstrings für alle öffentlichen Funktionen
- [ ] **Q5:** CHANGELOG.md Eintrag erstellt

### 6.4 Akzeptanz-Toleranzen

| Metrik | Toleranz | Grund |
|--------|----------|-------|
| Numerische Differenz | ≤ 1e-8 | IEEE 754 double precision |
| Hash-Differenz | 0 | Binäre Identität für Golden Files |
| Performance | ≥ 20x (Batch) | Migrations-Ziel |

---

## 7. Rollback-Plan

### 7.1 Sofort-Rollback (< 1 Minute)

```bash
# Option 1: Feature-Flag deaktivieren
export OMEGA_USE_RUST_SLIPPAGE_FEE=false

# Option 2: In Code (falls notwendig)
# src/backtest_engine/core/slippage_and_fee.py
USE_RUST_SLIPPAGE_FEE = False
```

### 7.2 Rollback-Trigger

| Trigger | Schwellwert | Aktion |
|---------|-------------|--------|
| Golden-File Hash Mismatch | Jeder | Sofort-Rollback |
| Numerische Differenz | > 1e-8 | Sofort-Rollback |
| Performance-Regression | > 5% langsamer | Analyse → ggf. Rollback |
| Runtime Error | Jeder in Production | Sofort-Rollback |

### 7.3 Post-Rollback

1. Issue erstellen mit Reproduktionsschritten
2. Root-Cause-Analysis durchführen
3. Fix entwickeln und neue Tests hinzufügen
4. Re-Deployment nach Validierung

---

## 8. Checklisten

### 8.1 Pre-Implementation Checklist

- [x] FFI-Spezifikation finalisiert (`docs/ffi/slippage_fee.md`)
- [x] Golden-Tests vorhanden (`tests/golden/test_golden_slippage_fee.py`)
- [x] Rust Build-System funktioniert (`cargo build` erfolgreich)
- [x] Migration Readiness ✅ (`docs/MIGRATION_READINESS_VALIDATION.md`)
- [ ] Lokale Entwicklungsumgebung eingerichtet (Rust 1.75+)

### 8.2 Implementation Checklist

#### Phase 1: Setup
- [ ] Verzeichnisstruktur erstellen (`src/costs/`)
- [ ] Cargo.toml Dependencies hinzufügen
- [ ] `mod.rs` erstellen

#### Phase 2: Rust-Code
- [ ] `slippage.rs` implementieren
- [ ] `fee.rs` implementieren
- [ ] `lib.rs` Module registrieren
- [ ] `cargo test` bestanden
- [ ] `cargo clippy` bestanden

#### Phase 3: Python-Integration
- [ ] `slippage_and_fee.py` erweitern
- [ ] Feature-Flag implementieren
- [ ] Batch-Methoden hinzufügen
- [ ] mypy --strict bestanden

#### Phase 4: Testing
- [ ] Golden-Tests bestanden
- [ ] Integration-Tests erstellt und bestanden
- [ ] Rust-Unit-Tests bestanden
- [ ] Manueller Backtest-Vergleich

### 8.3 Post-Implementation Checklist

- [ ] Dokumentation aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert
- [ ] Code-Review abgeschlossen
- [ ] Performance-Benchmark dokumentiert
- [ ] Sign-off Matrix ausgefüllt

### 8.4 Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Developer | | | ⏳ |
| Tech Lead | | | ⏳ |
| QA | | | ⏳ |

---

## 9. References

- [ADR-0001: Migration Strategy](./adr/ADR-0001-migration-strategy.md)
- [ADR-0003: Error Handling](./adr/ADR-0003-error-handling.md)
- [FFI Specification: Slippage & Fee](./ffi/slippage_fee.md)
- [Migration Runbook: Slippage & Fee](./runbooks/slippage_fee_migration.md)
- [Migration Readiness Validation](./MIGRATION_READINESS_VALIDATION.md)
- [Golden-File Reference](../tests/golden/reference/slippage_fee/slippage_fee_v1.json)

---

## Änderungshistorie

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-08 | 1.0 | Initiale Version | AI Agent |

---

*Document Status: ✅ READY FOR IMPLEMENTATION*
