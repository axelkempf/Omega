# Migration Candidates (P0-04)

Diese Liste priorisiert Migrations-Kandidaten für Rust/Julia basierend auf:

- Performance-Baselines: `reports/performance_baselines/p0-01_*.json`
- Type-Readiness: AST-basierte Type-Coverage (Return- und Parameter-Annotationen)

Hinweis: Das ist ein Planungsartefakt. Die finale Reihenfolge muss zusätzlich
Interfaces/Serialisierung, Golden-File-Determinismus und FFI-Risiken berücksichtigen.

## Regeln

- Perf-Bucket: High (>= 1.0s), Medium (>= 0.15s), Low (< 0.15s)
- Type-Bucket: High (Return >= 80% & Params >= 90%), Medium (Return >= 50% & Params >= 80%), sonst Low
- Recommended Priority: konservativ aus Perf + Type abgeleitet

## Kandidaten

| Kandidat | Target | Perf | Impact (s) | Type | Return% | Param% | Priority | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Indicator Cache | Rust | High | 1.141822 | High | 100.0 | 100.0 | High | `reports/performance_baselines/p0-01_indicator_cache.json` |
| Multi-Symbol Slice | Rust | High | 7.235017 | Low | 22.2 | 90.0 | Medium | `reports/performance_baselines/p0-01_multi_symbol_slice.json` |
| Symbol Data Slicer | Rust | Medium | 0.731318 | High | 81.8 | 100.0 | Medium | `reports/performance_baselines/p0-01_symbol_data_slicer.json` |
| Optimizer (Final Selection / Robust Zone) | Julia | Medium | 0.797464 | Medium | 78.8 | 93.2 | Medium | `reports/performance_baselines/p0-01_optimizer.json` |
| Portfolio | Rust | Medium | 0.248001 | Medium | 57.1 | 100.0 | Medium | `reports/performance_baselines/p0-01_portfolio.json` |
| Slippage & Fee | Rust | Medium | 0.736007 | Medium | 50.0 | 100.0 | Medium | `reports/performance_baselines/p0-01_slippage_and_fee.json` |
| Event Engine | Rust | Medium | 0.336732 | Low | 0.0 | 100.0 | Low | `reports/performance_baselines/p0-01_event_engine.json` |
| Execution Simulator | Rust | Medium | 0.173885 | Low | 22.2 | 95.7 | Low | `reports/performance_baselines/p0-01_execution_simulator.json` |
| Rating Modules | Rust | Low | 0.077729 | High | 100.0 | 100.0 | Low | `reports/performance_baselines/p0-01_rating.json` |
| Walkforward (stubbed window) | Julia | Low | 0.132846 | Medium | 66.7 | 100.0 | Low | `reports/performance_baselines/p0-01_walkforward_stub.json` |
| Analysis Pipelines | Julia | Unknown | - | High | 92.5 | 99.3 | Low | `-` |

## Notes

- **Indicator Cache**: Hot-path für Indikatoren + Cache; starker Kandidat für Rust (ndarray/Arrow).
- **Multi-Symbol Slice**: Auffällig teuer in Baseline; Kandidat für Rust (Vectorisierung/Zero-copy).
- **Symbol Data Slicer**: Hohe Call-Frequenz im Core-Loop; Rust kann Branching/Indexing beschleunigen.
- **Optimizer (Final Selection / Robust Zone)**: Explorativ/Research-lastig; Julia kann Iteration beschleunigen (aber FFI/Arrow klären).
- **Portfolio**: Stateful Hot-path; Ownership/Mutability muss sauber spezifiziert werden.
- **Slippage & Fee**: Reine Mathematik; sehr gut als frühes, kleines Rust-Pilot-Modul.
- **Event Engine**: Core-Loop; sehr sensibel – Migration erst nach Interface-Spec + Tests.
- **Execution Simulator**: Trade-Matching + Exits; gute Rust-Kandidatur nach klarer I/O-Spec.
- **Rating Modules**: Viele numerische Scores; geeignet für Rust, aber erst Schema/Output-CSV beachten.
- **Walkforward (stubbed window)**: Orchestrierung; primär I/O + Pipeline – eher später, nach Stabilisierung.
- **Analysis Pipelines**: Research/Plots; Julia lohnt sich, aber Performance-Impact schwerer zu messen (noch keine Baseline).

## Reproduzieren

- JSON (machine-readable): `tools/migration_candidates.py --format json`
- Markdown (human-readable): `tools/migration_candidates.py --format md`
