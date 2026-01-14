---
description: 'Omega V2 Backtest-Core development guidelines for Python Orchestrator + Rust Core architecture'
applyTo: 'rust_core/**/*.rs,python/bt/**/*.py'
---

# Omega V2 Backtest Development Instructions

> Normative Entwicklungsrichtlinien für den Omega V2 Backtest-Core (Rust) und Python Wrapper.
> Diese Instruktionen gelten **ausschließlich** für V2 Backtest-Entwicklung, nicht für Live-Trading/MT5.

---

## Architektur-Übersicht

### Grundprinzip: Python Orchestrator + Rust Core

```
┌─────────────────────────────────────────────────────────────────┐
│                     PYTHON LAYER (bt)                            │
│                   Dünner Orchestrator                            │
├─────────────────────────────────────────────────────────────────┤
│  • Config laden (JSON)                                           │
│  • Parquet-Pfade bestimmen                                       │
│  • EINMALIGER FFI-CALL → Rust-Engine starten                    │
│  • Ergebnis empfangen (JSON)                                     │
│  • Report generieren / Speichern                                 │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
                                       │  run_backtest(config_json) → result_json
                                       │  (SINGLE FFI BOUNDARY)
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RUST LAYER (rust_core)                      │
│                     Gesamter Backtest-Kern                       │
├─────────────────────────────────────────────────────────────────┤
│  Data Loader → Indicator Engine → Event Loop → Result Builder   │
└─────────────────────────────────────────────────────────────────┘
```

### Single FFI Boundary (Nicht verhandelbar)

- **EIN** Entry-Point: `run_backtest(config_json: &str) -> String`
- **KEINE** Rückflüsse nach Python während Backtest
- **KEINE** PyO3-Objekte im Rust Core (nur Serde-Serialisierung)
- **KEIN** Multi-Call FFI wie in V1

---

## Rust Core Struktur

### Workspace Layout

```
rust_core/
├── Cargo.toml                    # Workspace-Definition
├── Cargo.lock                    # MUSS versioniert werden
├── rust-toolchain.toml           # Rust-Version Pinning (Edition 2024)
│
└── crates/
    ├── types/                    # Gemeinsame Datentypen
    ├── data/                     # Parquet-Laden, Alignment
    ├── indicators/               # Indikator-Engine + Registry
    ├── execution/                # Order-Ausführung, Fill-Logik
    ├── portfolio/                # Portfolio, Equity, Stops
    ├── trade_mgmt/               # Trade-/Position-Management
    ├── strategy/                 # Strategy Trait + Implementierungen
    ├── backtest/                 # Event Loop, Runner
    ├── metrics/                  # Metrik-Berechnungen
    └── ffi/                      # PyO3 FFI Layer
```

### Crate-Abhängigkeiten (Einweg, keine Zyklen)

```
ffi
 └── backtest
      ├── strategy
      │    └── indicators
      │         └── types
      ├── execution
      │    └── types
      ├── portfolio
      │    ├── execution
      │    └── types
      ├── trade_mgmt
      │    └── types
      ├── metrics
      │    └── types
      └── data
           └── types
```

---

## Entwicklungsrichtlinien

### Rust Code Standards

```rust
// Edition 2024, stricte Clippy-Regeln
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

// Serde für alle Config/Result Structs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    // ...
}

// Result-Pattern für Fehlerbehandlung
pub type CoreResult<T> = Result<T, CoreError>;

// NIEMALS panic! über FFI-Grenze
// Alle Fehler als Result<T, E> zurückgeben
```

### Determinismus-Anforderungen

```rust
// DEV-Mode: Deterministischer RNG mit Seed
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

fn create_rng(config: &BacktestConfig) -> ChaCha8Rng {
    match config.run_mode {
        RunMode::Dev => ChaCha8Rng::seed_from_u64(config.rng_seed),
        RunMode::Prod => ChaCha8Rng::from_entropy(),
    }
}

// Kein Logging im Hot-Path das Determinismus beeinflusst
// tracing nur für strukturierte Logs, nicht für Timing
```

### Error Contract (Hybrid)

```rust
// Setup-/Input-Fehler → Python Exception via PyO3
#[pyfunction]
fn run_backtest(config_json: &str) -> PyResult<String> {
    let config: BacktestConfig = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;
    
    // Runtime-Fehler → JSON Error Result
    match engine::run(&config) {
        Ok(result) => Ok(serde_json::to_string(&result).unwrap()),
        Err(e) => Ok(serde_json::to_string(&ErrorResult::from(e)).unwrap()),
    }
}
```

---

## Python Wrapper (bt)

### Package Struktur

```
python/
└── bt/
    ├── __init__.py
    ├── _native.pyi              # Type Stubs für Rust-Extension
    ├── runner.py                # High-Level API
    ├── config.py                # Config-Validierung
    ├── result.py                # Result-Parsing
    └── tests/
        ├── test_runner.py
        ├── test_golden.py       # Golden-File Tests
        └── fixtures/
```

### API Design

```python
"""Omega V2 Backtest Runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bt._native import run_backtest as _run_backtest


def run(config: dict[str, Any] | Path) -> BacktestResult:
    """Run a backtest with the given configuration.
    
    Args:
        config: Either a config dict or path to JSON config file.
        
    Returns:
        BacktestResult with trades, metrics, and equity curve.
        
    Raises:
        ValueError: If config is invalid.
        BacktestError: If backtest execution fails.
    """
    if isinstance(config, Path):
        config = json.loads(config.read_text())
    
    config_json = json.dumps(config)
    result_json = _run_backtest(config_json)
    
    return BacktestResult.from_json(result_json)
```

---

## Testing & Validation

### Testpyramide

| Ebene | Fokus | Framework | Wo |
|-------|-------|-----------|-----|
| Unit Tests | Einzelne Funktionen | `cargo test` | `crates/*/src/*.rs` |
| Property Tests | Invarianten | `proptest` | `crates/*/tests/` |
| Integration | E2E Backtest | `pytest` | `python/bt/tests/` |
| Contract | Golden Files | `pytest` + Diff | `python/bt/tests/golden/` |

### Golden File Workflow

```bash
# Golden-Smoke (PR-Gate, schnell)
pytest python/bt/tests/test_golden.py -k "smoke"

# Full Golden (Nightly/Release)
pytest python/bt/tests/test_golden.py

# Golden-Update (nur mit Review!)
pytest python/bt/tests/test_golden.py --update-golden
```

### Golden File Struktur

```
python/bt/tests/golden/
├── fixtures/
│   ├── market/EURUSD/EURUSD_M1_BID.parquet
│   ├── market/EURUSD/EURUSD_M1_ASK.parquet
│   └── configs/mean_reversion_basic.json
└── expected/
    ├── mean_reversion_basic/
    │   ├── trades.json
    │   ├── equity.csv
    │   ├── metrics.json
    │   └── meta.json
    └── ...
```

### V1 ↔ V2 Parität

```python
# Paritäts-Test: V1 und V2 müssen gleiche Events produzieren
def test_v1_v2_parity_scenario_1():
    """Market-Entry Long → Take-Profit."""
    v1_result = run_v1_backtest(CONFIG)
    v2_result = run_v2_backtest(CONFIG)
    
    # Events MÜSSEN übereinstimmen
    assert v1_result.trades == v2_result.trades
    
    # PnL/Fees innerhalb Toleranz
    assert abs(v1_result.total_pnl - v2_result.total_pnl) < 0.01
```

### 6 Kanonische Szenarien (MUSS)

1. Market-Entry Long → Take-Profit
2. Market-Entry Long → Stop-Loss
3. Pending Entry (Limit/Stop) → Trigger ab `next_bar` → Exit
4. Same-Bar SL/TP Tie → SL-Priorität
5. `in_entry_candle` Spezialfall inkl. Limit-TP Regel
6. Mix aus Sessions/Warmup/HTF-Einflüssen

---

## Output Contract

### Artefakte (MVP)

| Datei | Format | Beschreibung |
|-------|--------|--------------|
| `trades.json` | JSON Array | Alle Trades mit Entry/Exit/Reason |
| `equity.csv` | CSV | Equity-Kurve pro Bar |
| `metrics.json` | JSON Object | Sharpe, Sortino, Drawdown, etc. |
| `meta.json` | JSON Object | Run-Metadaten, Timestamps |

### Normalisierung für Vergleiche

```python
# meta.json: Nur generated_at neutralisieren
def normalize_meta(meta: dict) -> dict:
    result = meta.copy()
    result.pop("generated_at", None)
    result.pop("generated_at_ns", None)
    return result

# JSON: Stabile Key-Order
json.dumps(data, sort_keys=True, indent=2)
```

---

## CI Integration

### PR-Gates (MUSS)

```yaml
# .github/workflows/omega-v2-ci.yml
jobs:
  rust-checks:
    steps:
      - run: cargo fmt --all -- --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test --all

  python-checks:
    steps:
      - run: pre-commit run --all-files
      - run: pytest python/bt/tests/ -k "not slow"

  wheel-build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-14, windows-latest]
    steps:
      - run: maturin build --release
```

### Golden-Smoke (PR-Gate)

```yaml
  golden-smoke:
    needs: [rust-checks, wheel-build]
    steps:
      - run: pytest python/bt/tests/test_golden.py -k "smoke"
```

---

## Kritische Pfade (V2-spezifisch)

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| `rust_core/crates/execution/` | Fill-Logik, Tie-Breaks | Determinismus-Tests + V1-Parität |
| `rust_core/crates/strategy/` | Signal-Generierung | 6 kanonische Szenarien |
| `rust_core/crates/ffi/` | FFI Boundary | Contract-Tests |
| `python/bt/tests/golden/expected/` | Artefakt-Stabilität | Review bei jeder Änderung |

---

## Checkliste für V2-Änderungen

### Vor Code-Änderung

- [ ] Verstanden welches Crate betroffen ist
- [ ] Abhängigkeits-Richtung geprüft (keine Zyklen)
- [ ] Determinismus-Auswirkung bewertet

### Nach Code-Änderung

- [ ] `cargo fmt && cargo clippy` ohne Fehler
- [ ] `cargo test` grün
- [ ] Golden-Smoke grün
- [ ] Bei Golden-Änderung: Explizite Begründung im PR

### Bei FFI-Änderungen

- [ ] Python Type Stubs aktualisiert (`_native.pyi`)
- [ ] Error Contract eingehalten (Setup → Exception, Runtime → JSON)
- [ ] Contract-Tests erweitert

---

## Referenzen

- [OMEGA_V2_ARCHITECTURE_PLAN.md](../../docs/OMEGA_V2_ARCHITECTURE_PLAN.md)
- [OMEGA_V2_TECH_STACK_PLAN.md](../../docs/OMEGA_V2_TECH_STACK_PLAN.md)
- [OMEGA_V2_TESTING_VALIDATION_PLAN.md](../../docs/OMEGA_V2_TESTING_VALIDATION_PLAN.md)
- [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](../../docs/OMEGA_V2_OUTPUT_CONTRACT_PLAN.md)
- [OMEGA_V2_CI_WORKFLOW_PLAN.md](../../docs/OMEGA_V2_CI_WORKFLOW_PLAN.md)
