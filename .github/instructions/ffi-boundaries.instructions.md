---
description: 'Foreign Function Interface conventions for Python/Rust/Julia hybrid code'
applyTo: '**/*. {py,rs,jl}'
---

# FFI Boundary Conventions

Standards for cross-language communication in the Omega hybrid architecture.

## Serialization (Arrow IPC - per ADR-0001)

- Use Apache Arrow for all bulk data transfer between Python ↔ Rust and Python ↔ Julia
- Define schemas in `.arrow` schema files under `src/schemas/`
- Validate schemas at compile time (Rust) and runtime (Python/Julia)

## Type Mapping

| Python | Rust | Julia | Notes |
|--------|------|-------|-------|
| `float` | `f64` | `Float64` | Always 64-bit for trading precision |
| `int` | `i64` | `Int64` | |
| `datetime` | `chrono::DateTime<Utc>` | `DateTime{UTC}` | Use UTC everywhere |
| `Optional[T]` | `Option<T>` | `Union{T, Nothing}` | |
| `list[T]` | `Vec<T>` | `Vector{T}` | Prefer Arrow Arrays for large data |
| `Decimal` | `rust_decimal::Decimal` | `Decimals. Decimal` | For monetary values |

## Error Handling Convention

- Python:  Raise domain-specific exceptions (e.g., `ValidationError`, `CalculationError`)
- Rust: Return `Result<T, OmegaError>` - never panic across FFI boundary
- Julia:  Throw typed exceptions, catch at Python boundary

## Contract Definition

- Define all FFI contracts in `src/contracts/` as: 
  - Python:  Pydantic models
  - Rust:  Serde structs with `#[pyclass]`
  - Julia: Struct definitions with `@kwdef`

## Performance Requirements

- FFI calls must not exceed 1ms overhead for single-record operations
- Batch operations:  amortize FFI overhead over ≥1000 records
- Profile with `py-spy` (Python side) and `perf` (Rust side)