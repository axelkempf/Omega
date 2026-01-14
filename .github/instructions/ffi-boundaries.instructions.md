---
description: 'Foreign Function Interface conventions for Python/Rust/Julia hybrid code'
applyTo: '**/*. {py,rs,jl}'
---

# FFI Boundary Conventions

Standards for cross-language communication in the Omega hybrid architecture.

---

## ⚠️ V1 vs V2 Unterscheidung

Dieses Dokument beschreibt **zwei verschiedene FFI-Modelle**:

| Aspekt | V1 (Live-Engine, Analysis) | V2 (Backtest-Core) |
|--------|---------------------------|-------------------|
| **Architektur** | Multi-Call FFI | Single FFI Boundary |
| **Sprachen** | Python + Rust + Julia | Python + Rust only |
| **Entry-Points** | Mehrere pro Modul | EIN: `run_backtest()` |
| **Data Transfer** | Arrow IPC | JSON (Config/Result) |
| **Pfade** | `src/rust_modules/`, `src/julia_modules/` | `rust_core/`, `python/bt/` |

**Für V2-Backtest-Entwicklung siehe:** [omega-v2-backtest.instructions.md](omega-v2-backtest.instructions.md)

---

## V2 Backtest FFI (Single Boundary)

### Grundprinzip

```
Python (bt)                          Rust (rust_core)
    │                                     │
    │  run_backtest(config_json)          │
    │ ──────────────────────────────────► │
    │                                     │  ← Gesamter Backtest läuft in Rust
    │                                     │  ← Kein Rückfluss während Execution
    │  ◄────────────────────────────────  │
    │  result_json                        │
    │                                     │
```

### Entry-Point (Normativ)

```rust
// rust_core/crates/ffi/src/lib.rs
use pyo3::prelude::*;

#[pyfunction]
fn run_backtest(config_json: &str) -> PyResult<String> {
    // Setup-Fehler → Python Exception
    let config = parse_config(config_json)?;
    
    // Runtime-Fehler → JSON Error Result
    match backtest::run(&config) {
        Ok(result) => Ok(serde_json::to_string(&result).unwrap()),
        Err(e) => Ok(serde_json::to_string(&ErrorResult::from(e)).unwrap()),
    }
}

#[pymodule]
fn _native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}
```

### Error Contract (V2)

| Fehlertyp | Behandlung | Beispiel |
|-----------|------------|----------|
| Config-/Input-Fehler | `PyResult::Err` → Python Exception | Invalid JSON, missing field |
| Runtime-Fehler | JSON `{"error": {...}}` | Data loading failed, strategy error |

### Serialization (V2)

- **Config**: JSON via `serde_json`
- **Result**: JSON via `serde_json`
- **Market Data**: Parquet via `arrow-rs` (intern in Rust, kein FFI)

### Was V2 NICHT hat

- ❌ Keine Arrow IPC über FFI-Grenze
- ❌ Keine PyO3-Objekte im Core
- ❌ Keine Callbacks nach Python
- ❌ Keine Multi-Call Patterns
- ❌ Kein Julia

---

## V1 Legacy FFI (Multi-Call)

> Gilt für: Live-Engine (`src/hf_engine/`), Julia-Module, bestehende Rust-Module

### Serialization (Arrow IPC - per ADR-0001)

- Use Apache Arrow for all bulk data transfer between Python ↔ Rust and Python ↔ Julia
- Define schemas in `.arrow` schema files under `src/schemas/`
- Validate schemas at compile time (Rust) and runtime (Python/Julia)

### Type Mapping (V1)

| Python | Rust | Julia | Notes |
|--------|------|-------|-------|
| `float` | `f64` | `Float64` | Always 64-bit for trading precision |
| `int` | `i64` | `Int64` | |
| `datetime` | `chrono::DateTime<Utc>` | `DateTime{UTC}` | Use UTC everywhere |
| `Optional[T]` | `Option<T>` | `Union{T, Nothing}` | |
| `list[T]` | `Vec<T>` | `Vector{T}` | Prefer Arrow Arrays for large data |
| `Decimal` | `rust_decimal::Decimal` | `Decimals. Decimal` | For monetary values |

### Error Handling Convention (V1)

- Python:  Raise domain-specific exceptions (e.g., `ValidationError`, `CalculationError`)
- Rust: Return `Result<T, OmegaError>` - never panic across FFI boundary
- Julia:  Throw typed exceptions, catch at Python boundary

### Contract Definition (V1)

- Define all FFI contracts in `src/shared/` as: 
  - Python: Pydantic models in `src/shared/protocols.py` and domain exceptions in `src/shared/exceptions.py`
  - Error codes: `src/shared/error_codes.py` (cross-language mapping)
  - Rust: Serde structs with `#[pyclass]` in `src/rust_modules/omega_rust/src/`
  - Julia: Struct definitions with `@kwdef` in `src/julia_modules/omega_julia/src/`

### Performance Requirements (V1)

- FFI calls must not exceed 1ms overhead for single-record operations
- Batch operations:  amortize FFI overhead over ≥1000 records
- Profile with `py-spy` (Python side) and `perf` (Rust side)

---

## Migration V1 → V2

Wenn V1-Code nach V2 migriert wird:

1. **Eliminiere Multi-Call Patterns** → Alles in Rust
2. **Ersetze Arrow IPC** → Daten bleiben in Rust (Parquet direkt lesen)
3. **Entferne PyO3-Objekte** → Nur Serde-Serialisierung
4. **Konsolidiere Entry-Points** → Single `run_backtest()`

### Migrationspfad

```
V1: Python → Rust (per Bar) → Python (Results)
                  ↓
V2: Python → Rust (EINMAL, gesamter Backtest) → Python (Results)
```