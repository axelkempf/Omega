---
description: 'Julia programming language coding conventions for hybrid Python/Julia integration'
applyTo: '**/*.jl'
---

# Julia Coding Conventions for Omega

Follow Julia community standards and ensure seamless Python interop via PythonCall.jl.

## General Principles

- Use multiple dispatch effectively - prefer methods over functions with type checks
- Leverage Julia's type system:  `AbstractArray`, parametric types, `Union{T, Nothing}`
- Write type-stable code to avoid runtime dispatch overhead
- Use `@inbounds` and `@simd` only after profiling confirms benefit

## Python Interoperability (PythonCall. jl)

- Define clear Julia-side types that map to Python dataclasses/Pydantic models
- Use `pyconvert` explicitly for type safety at FFI boundaries
- Prefer Arrow IPC for bulk data transfer (defined in ADR-0001)
- Handle Python exceptions with `try/catch` and convert to Julia exceptions

## Monte Carlo & Numerical Patterns (Omega-specific)

- Use `Random.seed!` for reproducible simulations
- Prefer in-place operations (`mul!`, `ldiv!`) for large arrays
- Use `StaticArrays. jl` for small, fixed-size vectors (< 100 elements)
- Parallelize with `Threads.@threads` or `Distributed.jl` based on workload

## Patterns to Avoid

- Don't use global non-const variables - they kill performance
- Avoid type instability in hot loops
- Don't mix Python objects in tight Julia loops - batch at boundaries
- Avoid `eval` and `@eval` in runtime code

## Testing Standards

- Use `Test` module with `@testset` blocks
- Include property-based tests with `Supposition. jl` (Julia equivalent of Hypothesis)
- Performance regression tests with `BenchmarkTools.@benchmark`