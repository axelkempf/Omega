# Changelog

Alle nennenswerten Änderungen werden in dieser Datei dokumentiert.

> Hinweis: Historische Einträge sind ggf. unvollständig.

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
- `src/backtest_engine/logging/` → `src/backtest_engine/bt_logging/` (Namespace-Konflikt mit stdlib behoben)

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

