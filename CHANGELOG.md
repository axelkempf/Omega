# Changelog

Alle nennenswerten Änderungen werden in dieser Datei dokumentiert.

> Hinweis: Historische Einträge sind ggf. unvollständig.

## [1.4.0] - Wave 1: Rust FFI Rating Module Migration

### Added
- **Rust-Dispatch für Rating-Module**: Automatisches Dispatch zu Rust-Implementierungen
  - `robustness_score_1.py`: Parameter-Jitter-Score → Rust
  - `stability_score.py`: Yearly-Profit-WMAPE-Score → Rust
  - `stress_penalty.py`: Profit/Drawdown/Sharpe-Penalty → Rust
  - `cost_shock_score.py`: Single/Multi-Factor Cost-Shock → Rust
  - `trade_dropout_score.py`: Single/Multi-Run Dropout-Score → Rust
- **Feature Flag `OMEGA_USE_RUST_RATING`**: Steuert Python/Rust-Wahl
  - `auto` (default): Rust wenn verfügbar, sonst Python-Fallback
  - `true`: Force Rust (Fehler wenn nicht verfügbar)
  - `false`: Force Python (Rollback)
- **Parity-Tests**: `tests/integration/test_rust_rating_parity.py` mit 12 Tests für Python↔Rust-Parität
- **Floating-Point-Toleranz**: `rel_tol=1e-14` für Cross-Language-Vergleiche

### Changed
- `tests/property/test_prop_scoring.py`: Determinismus-Test verwendet `math.isclose()` statt exakter Gleichheit

### Performance (Benchmarks)
| Funktion | Python-only | Rust-enabled | Speedup |
|----------|-------------|--------------|---------|
| `cost_shock_score` (single) | 41µs | 4µs | **10x** |
| `cost_shock_score` (3 factors) | 117µs | 8µs | **15x** |
| `cost_shock_score` (5 factors) | 219µs | 10µs | **22x** |
| `penalty_computation` (10 stress) | 56µs | 10µs | **6x** |
| `penalty_computation` (50 stress) | 125µs | 31µs | **4x** |
| `robustness_score_1` (10 repeats) | 69µs | 12µs | **6x** |
| `robustness_score_1` (50 repeats) | 205µs | 39µs | **5x** |
| `robustness_score_1` (100 repeats) | 383µs | 73µs | **5x** |
| `stability_score` (5 years) | 14µs | 8µs | **1.6x** |
| `stability_score` (10 years) | 17µs | 11µs | **1.5x** |

### Notes
- **Ulcer Index bleibt Python-only**: Komplexes Timestamp-Resampling (weekly closes) ist in Rust nicht implementiert
- **Numerische Parität**: Python und Rust liefern identische Ergebnisse (±1e-14 Toleranz)
- **Kein neuer Rust-Code nötig**: Wave 1 aktiviert nur den Dispatch zu bereits existierenden Rust-Funktionen

## [1.3.0] - Wave 0: Rust FFI Migration (Slippage & Fee)

### Added
- **Rust FFI Module `omega_rust.costs`**: Slippage- und Fee-Berechnungen in Rust
  - `calculate_slippage()` / `calculate_slippage_batch()` mit ChaCha8 RNG für Determinismus
  - `calculate_fee()` / `calculate_fee_batch()` mit Minimum-Fee-Support
  - 16 Rust Unit-Tests für Cost-Modul
- **Feature Flag `OMEGA_USE_RUST_SLIPPAGE_FEE`**: Steuert Python/Rust-Wahl
  - `auto` (default): Rust wenn verfügbar, sonst Python-Fallback
  - `true`: Force Rust (Fehler wenn nicht verfügbar)
  - `false`: Force Python (Rollback)
- **Batch-Operationen**: `SlippageModel.apply_batch()` und `FeeModel.calculate_batch()`
- **Debug-Funktion**: `get_rust_status()` für FFI-Diagnose
- **Integration-Tests**: 13 Tests für Rust↔Python-Parität (`tests/integration/test_rust_slippage_fee_parity.py`)
- **Implementierungsplan**: `docs/WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md`

### Changed
- `SlippageModel.apply()` und `FeeModel.calculate()` unterstützen jetzt optionales `seed`-Argument
- `logging/` → `bt_logging/` (Namespace-Konflikt mit stdlib behoben)

### Performance
- **Batch-Speedup**: 14.4x (Python 95.77ms → Rust 6.66ms für 1000×10 Trades)
- **Single-Call**: ~20x schneller (durch FFI-Overhead bei kleinen Batches)

### Notes
- Python und Rust nutzen unterschiedliche RNG-Algorithmen (Mersenne Twister vs ChaCha8)
- Beide Implementierungen sind intern deterministisch (gleicher Seed = gleiches Ergebnis)
- Fixed-only Slippage (ohne Random) ist binär identisch zwischen Python und Rust

## [1.2.0]

- Python-Baseline auf 3.12 angehoben; Dependencies für 3.12-kompatible Mindestversionen aktualisiert.
- CI/Pre-commit auf Python 3.12 umgestellt; mypy-Target auf 3.12 gesetzt.
- Dokumentation (README/CONTRIBUTING/AGENTS) an neues Version- und Python-Minimum angepasst.

## [1.1.0]

- Final Parameter Selection: erweitertes Scoring (Stabilität/Robustheit) inkl. Dropout-, Cost‑Shock- und TP/SL‑Stress.
- Walkforward/Optimizer: zusätzliche Modi/Instrumentation (Timing/Memory/Artefakte).
- Performance: Parallelisierung und robustere Pipeline-Ausführung in Analyse/Optimizer-Workflows.

