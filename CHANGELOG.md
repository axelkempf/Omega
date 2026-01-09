# Changelog

Alle nennenswerten Änderungen werden in dieser Datei dokumentiert.

> Hinweis: Historische Einträge sind ggf. unvollständig.

## [1.6.0] - Wave 1: IndicatorCache Rust Integration Fix ✅ COMPLETED

### Fixed
- **Critical Bug**: Rust `IndicatorCacheRust` was implemented but never called
  - `indicator_cache.py` was not updated to delegate to Rust
  - `event_engine.py` always imported pure Python implementation
  - **Result**: Rust code existed but was never executed

### Added
- **Feature Flag System**: `OMEGA_USE_RUST_INDICATOR_CACHE` environment variable
  - `auto` (default): Use Rust if available, fallback to Python
  - `1`: Force Rust-only (no fallback)
  - `0`: Force Python-only

- **Rust Backend Integration**:
  - `_init_rust_cache()` function for lazy initialization
  - `_series_from_rust_array()` for NumPy → pandas Series conversion
  - All supported indicators now delegate to Rust with Python fallback

- **Benchmark Tool**: `tools/benchmark_indicator_cache.py`
  - Direct Python vs Rust indicator comparison
  - Supports configurable bar counts
  - Reports per-indicator and total speedup

### Performance Results (100k bars benchmark)
| Indicator | Speedup | Notes |
|-----------|---------|-------|
| `kalman_mean` | **357x** | Complex state machine |
| `kalman_zscore` | **109x** | Complex state machine |
| `atr(14)` | **83x** | Multiple passes |
| `dmi(14)` | **5.1x** | Complex calculation |
| `sma(50)` | **3.3x** | Rolling window |
| `choppiness(14)` | **3.2x** | ATR-based |
| `ema(20)` | **2.8x** | Simple EMA |
| **Overall** | **16.6x** | Total pipeline |

### Changed
- **Default changed from disabled to auto**: Rust is now used automatically when available
- **Documentation updated**: `docs/WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md` v2.1

### Tests
- 17 Rust integration tests passing
- All existing backtest tests passing

---

## [1.5.0] - Wave 2: Portfolio Rust Migration ✅ COMPLETED

### Added
- **Rust Portfolio-Modul**: `src/rust_modules/omega_rust/src/portfolio/`
  - `PositionRust` struct mit R-Multiple-Berechnung und voller Handels-State-Verwaltung
  - `PortfolioRust` class mit Position-Tracking, Fee-Management, Equity-Curve
  - `PortfolioState` struct für interne State-Verwaltung
  - `EquityPoint` und `FeeLogEntry` für Logging-Strukturen
  - Registriert in PyO3 für Python-Zugriff
  - Alle Clippy-Warnungen behoben

- **Python-Integration**:
  - Feature-Flag `OMEGA_USE_RUST_PORTFOLIO` (auto/true/false)
  - `get_rust_status()` für Backend-Diagnose
  - `_to_rust()` / `_from_rust()` Conversion-Methoden für PortfolioPosition
  - Docstring mit Modul-Dokumentation

- **Test-Suite (54 Tests gesamt)**:
  - **Golden-Tests** (`tests/golden/test_portfolio_rust_golden.py`): 13 Tests
    - R-Multiple-Berechnungen für Long/Short Win/Loss
    - Portfolio State Management Tests
    - Equity-Curve Generierung Tests
    - Rust-Status Struktur-Tests
  - **Integration-Tests** (`tests/integration/test_portfolio_rust_parity.py`): 18 Tests
    - Feature-Flag Behavior Tests
    - Position Parity Tests
    - Portfolio Parity Tests
    - Edge Case Tests (Zero-Risk, Empty Portfolio, All Losses)
    - Multi-Symbol Portfolio Tests
    - DataFrame Export Tests
  - **Benchmark-Tests** (`tests/benchmarks/test_bench_portfolio.py`): 23 Tests
    - Position Registration Benchmarks
    - Equity Update Benchmarks
    - Fee Registration Benchmarks
    - Summary Benchmarks
    - Position Query Benchmarks
    - Full Lifecycle Benchmarks
    - Throughput Baselines

- **Golden Reference Files**: `tests/golden/reference/portfolio/portfolio_v1.json`

### Changed
- **Implementierungsplan**: Status → COMPLETED
  - Phase 1: Setup ✅
  - Phase 2: Rust-Code ✅
  - Phase 3: Python-Integration ✅
  - Phase 4: Testing ✅
- **Lint-Fixes**: Ungenutzte Imports in Test-Dateien entfernt

## [1.4.0] - Wave 2: Portfolio Rust Migration Plan

### Added
- **Implementierungsplan Wave 2**: `docs/WAVE_2_PORTFOLIO_IMPLEMENTATION_PLAN.md`
  - Vollständige Migrationsstrategie für Portfolio-Modul zu Rust
  - Rust-Strukturdefinitionen für `PortfolioRust`, `PositionRust`, `PortfolioState`
  - Feature-Flag `OMEGA_USE_RUST_PORTFOLIO` (auto/true/false)
  - Golden-File Test-Strategie für Determinismus-Validierung
  - Lessons Learned aus Wave 0 integriert
  - 5-Tage Zeitplan mit detaillierten Phasen
  - Rollback-Prozedur und Trigger dokumentiert
- **Runbook-Referenz**: `docs/runbooks/portfolio_migration.md` mit Verweis auf Implementation Plan
- **Architecture.md Update**: Neue Wave 2 Dokumentation referenziert

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

