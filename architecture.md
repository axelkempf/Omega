### Projektarchitektur

Ordner-für-Ordner-Übersicht der Codebasis (ohne `results`-Ordner und ohne Auflistung einzelner `.csv`-Dateien).

### Wurzelverzeichnis

- `configs/`
  - `backtest/`
    - `_config_validator.py`
    - `mean_reversion_z_score.json`
    - `__pycache__/`
  - `live/`
    - `strategy_config_10927144.json`
    - `strategy_config_10928521.json`
    - `strategy_config_10929345.json`
    - `strategy_config_15582434.json`
  - `execution_costs.yaml`
  - `symbol_specs.yaml`
- `data/` *(Market data, git-ignored contents – tracked via README.md skeleton)*
  - `csv/` *({SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv, ASK.csv)*
  - `news/`
    - `csv_cleaner.py`
  - `parquet/` *({SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet, ASK.parquet)*
  - `raw/` *(Unprocessed broker exports)*
- `docs/`
  - `CATEGORICAL_RANKING_OPTIMIZATION.md` (Detaillierter Performance-Optimierung Report)
- `final_selection/`
  - `joblib_tmp/`
- `scripts/`
- `src/`
  - `backtest_engine/`
  - `hf_engine/`
  - `julia_modules/` *(Future: High-perf Julia extensions via PythonCall)*
  - `omega/` *(Python-Top-Level-Package/Namespace für FFI-Module wie `omega._rust`)*
  - `omega.egg-info/` *(build artifact from pyproject.toml name)*
  - `rust_modules/` *(Future: High-perf Rust extensions via PyO3/Maturin)*
  - `shared/` *(Shared Protocols & type aliases for stable boundaries)*
  - `strategies/`
  - `ui_engine/`
  - `watchdog/`
  - `engine_launcher.py`
- `tests/`
- `var/` *(Runtime state, git-ignored contents – tracked via README.md skeleton)*
  - `archive/`
  - `logs/`
    - `entry_logs/`
    - `optuna/`
    - `system/`
    - `trade_logs/`
  - `results/`
    - `analysis/`
    - `backtests/`
    - `walkforwards/`
  - `tmp/`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `prompts.md`
- `pyproject.toml`
- `pytest.log`
- `README.md`
- `SUMMARY.md`

---

### `src/rust_modules/` *(High-Performance Rust Extensions via PyO3/Maturin)*

- `omega_rust/`
  - `Cargo.toml` *(PyO3 0.20+, ndarray, rayon, serde)*
  - `pyproject.toml` *(Maturin build system)*
  - `rust-toolchain.toml` *(Rust 1.76.0 pinning)*
  - `README.md`
  - `src/`
    - `lib.rs` *(PyO3 module entry point)*
    - `error.rs` *(OmegaError with thiserror)*
    - `indicators/`
      - `mod.rs` *(Module exports)*
      - `ema_impl.rs` *(Exponential Moving Average)*
      - `rsi_impl.rs` *(Relative Strength Index)*
      - `statistics.rs` *(Rolling standard deviation)*
    - `costs/` *(PLANNED: Wave 0 Pilot)*
      - `mod.rs` *(Module exports)*
      - `slippage.rs` *(Slippage calculation with deterministic RNG)*
      - `fee.rs` *(Fee calculation per-million notional)*
  - `benches/`
    - `indicator_bench.rs` *(Criterion benchmarks)*

### `src/julia_modules/` *(High-Performance Julia Extensions via PythonCall)*

- `omega_julia/`
  - `Project.toml` *(PythonCall 0.9+, Arrow 2.7+, DataFrames 1.6+)*
  - `README.md`
  - `src/`
    - `OmegaJulia.jl` *(Main module)*
    - `monte_carlo.jl` *(Monte Carlo VaR simulations)*
    - `rolling_stats.jl` *(Rolling Sharpe/Sortino/Calmar)*
    - `bootstrap.jl` *(Block bootstrap methods)*
    - `risk_metrics.jl` *(Sharpe, Sortino, max_drawdown, etc.)*
  - `test/`
    - `runtests.jl` *(Test suite)*

---

### `src/shared/`

- `__init__.py`
- `protocols.py` *(runtime-checkable Protocols for stable boundaries / future FFI)*

### `src/backtest_engine/`

- `__init__.py`
- `batch_runner.py`
- `run_all.py`
- `runner.py`
- `__pycache__/`
- `analysis/`
  - `__init__.py`
  - `backfill_walkforward_equity_curves.py`
  - `combine_equity_curves.py`
  - `combined_walkforward_matrix_analyzer.py`
  - `final_combo_equity_plotter.py`
  - `metric_adjustments.py` (Trade-count basierte Metrik-Adjustierungen)
  - `walkforward_analyzer.py`
- `core/`
  - `__init__.py`
  - `event_engine.py`
  - `execution_simulator.py`
  - `indicator_cache.py`
  - `multi_strategy_controller.py`
  - `multi_symbol_slice.py`
  - `multi_tick_controller.py`
  - `portfolio.py`
  - `slippage_and_fee.py`
  - `symbol_data_slicer.py`
  - `tick_event_engine.py`
  - `__pycache__/`
- `data/`
  - `candle.py`
  - `convert_csv_candles_to_parquet.py`
  - `csv_converter.py`
  - `data_handler.py`
  - `market_hours.py`
  - `merge_csv.py`
  - `news_filter.py`
  - `tick_data_handler.py`
  - `tick.py`
  - `trading_holidays.py`
  - `__pycache__/`
- `deployment/`
  - `__init__.py`
  - `deployment_selector.py`
- `logging/`
  - `__init__.py`
  - `entry_log.py`
  - `entry_tag_analysis.ipynb`
  - `trade_logger.py`
  - `__pycache__/`
- `optimizer/`
  - `__init__.py`
  - `_settings.py`
  - `final_param_selector.py`
  - `grid_searcher.py`
  - `instrumentation.py`
  - `optuna_optimizer.py`
  - `robust_zone_analyzer.py`
  - `symbol_grid.py`
  - `walkforward_plot.py`
  - `walkforward_utils.py`
  - `walkforward.py`
  - `__pycache__/`
- `rating/`
  - `__init__.py`
  - `strategy_rating.py`
- `report/`
  - `__init__.py`
  - `exporter.py`
  - `metrics.py`
  - `overlay_plot.py`
  - `result_saver.py`
  - `visualizer.py`
  - `__pycache__/`
- `sizing/`
  - `__init__.py`
  - `commission.py`
  - `lot_sizer.py`
  - `rate_provider.py`
  - `symbol_specs_registry.py`
  - `__pycache__/`
- `strategy/`
  - `__init__.py`
  - `session_filter.py`
  - `session_time_utils.py`
  - `strategy_wrapper.py`
  - `validators.py`
  - `__pycache__/`

---

### `src/hf_engine/`

- `__init__.py`
- `__pycache__/`
- `adapter/`
  - `__init__.py`
  - `__pycache__/`
  - `broker/`
    - `__init__.py`
    - `broker_connection_fsm.py`
    - `broker_interface.py`
    - `broker_utils.py`
    - `mt5_adapter.py`
    - `__pycache__/`
  - `data/`
    - `data_provider_interface.py`
    - `mt5_data_provider.py`
    - `remote_data_provider.py`
  - `fastapi/`
    - `__init__.py`
    - `mt5_feed_server.py`
- `core/`
  - `__init__.py`
  - `controlling/`
    - `__init__.py`
    - `event_bus.py`
    - `multi_strategy_controller.py`
    - `position_monitor_controller.py`
    - `session_runner.py`
    - `strategy_runner.py`
  - `execution/`
    - `__init__.py`
    - `execution_engine.py`
    - `execution_result.py`
    - `execution_tracker.py`
    - `session_state.py`
    - `sl_tp_utils.py`
  - `risk/`
    - `__init__.py`
    - `lot_size_calculator.py`
    - `news_filter.py`
    - `risk_manager.py`
- `infra/`
  - `__init__.py`
  - `__pycache__/`
  - `config/`
    - `__init__.py`
    - `environment.py`
    - `paths.py`
    - `symbol_mapper.py`
    - `time_utils.py`
    - `__pycache__/`
  - `logging/`
    - `__init__.py`
    - `error_handler.py`
    - `log_manager.py`
    - `log_service.py`
    - `log_sqlite_viewer.py`
    - `__pycache__/`
  - `metrics/`
    - `__init__.py`
    - `performance_metrics.py`
  - `monitoring/`
    - `__init__.py`
    - `health_server.py`
    - `telegram_bot.py`
    - `__pycache__/`

---

### `src/strategies/`

- `__init__.py`
- `__pycache__/`
- `_base/`
  - `__init__.py`
  - `base_position_manager.py`
  - `base_scenarios.py`
  - `base_strategy.py`
- `_template/`
  - `__init__.py`
  - `strategy_template.py`
- `mean_reversion_z_score/`
  - `__init__.py`
  - `__pycache__/`
  - `backtest/`
    - `__init__.py`
    - `__pycache__/`
    - `backtest_strategy.py`
    - `position_manager.py`
    - `walkforward_backtest.py`
  - `live/`
    - `__init__.py`
    - `master_config.py`
    - `portfolio_runtime.py`
    - `portfolio_strategy.py`
    - `position_manager.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`

---

### `src/ui_engine/`

- `__init__.py`
- `config.py`
- `controller.py`
- `main.py`
- `models.py`
- `utils.py`
- `datafeeds/`
  - `__init__.py`
  - `base.py`
  - `dxfeed_manager.py`
  - `factory.py`
  - `mt5_manager.py`
- `registry/`
  - `__init__.py`
  - `strategy_alias.py`
- `strategies/`
  - `__init__.py`
  - `base.py`
  - `factory.py`
  - `mt5_manager.py`

---

### Weitere `src`-Verzeichnisse

- `src/engine_launcher.py`
- *(Build-Artefakt)* `*.egg-info/` (wird bei Installation/Build lokal erzeugt und nicht versioniert)

---

### `var/` (ohne `results/`)

- `archive/`
- `logs/`
  - `entry_logs/`
  - `optuna/`
  - `system/`
    - `engine_logs.db`
    - `engine.log`
  - `trade_logs/`
- `runtime/`
  - `trade_store.db`
- `tmp/`
  - `main_run_after.log`
  - `run_mc.py`
  - `timing_script.py`

---

### `analysis/` Ordner

Der `analysis/` Ordner enthält Post-Processing-Tools für Walkforward-Analysen. Alle Tools verwenden `var/results/analysis/` als zentrales Verzeichnis:

- **`walkforward_analyzer.py`**: Hauptanalyse-Tool, kombiniert Walkforward-Runs, berechnet Metriken und erstellt Snapshots.
- **`backfill_walkforward_equity_curves.py`**: Generiert Backfill-Equity-Kurven für historische Validierung.
- **`combined_walkforward_matrix_analyzer.py`**: Erstellt kombinierte Portfolio-Matrizen aus mehreren Strategien und berechnet kategoriale Champions.
- **`final_combo_equity_plotter.py`**: Erzeugt Equity-Plots und KPI-Reports für finale Kombinationen.
- **`metric_adjustments.py`**: Trade-count basierte Metrik-Adjustierungen (Shrinkage und Bayesian Methoden).

**Datenfluss:**
1. Walkforward-Optimizer schreibt Ergebnisse (Snapshots, Equity-Kurven, Trades) nach `var/results/analysis/`
2. `walkforward_analyzer.py` konsolidiert Daten → `var/results/analysis/combined/`
3. `combined_walkforward_matrix_analyzer.py` erstellt Portfolio-Kombinationen → `var/results/analysis/combined_matrix/`
4. `final_combo_equity_plotter.py` erstellt finale Plots → `var/results/analysis/combined_matrix/final_combos/plots/`

---

### Metrik-Adjustierung (Trade-Count basiert)

**Modul:** `analysis/metric_adjustments.py`

Alle Score-Berechnungen in den Analysis-Modulen verwenden trade-count adjustierte Metriken, um statistischen Overfitting bei niedrigen Trade-Zahlen zu vermeiden. Dies implementiert institutionelle Best Practices für robuste Performance-Bewertung.

**Kernfunktionen:**

1. **`shrinkage_adjusted(average_r, n_trades, n_years)`** — Average R-Multiple Adjustierung
   - Formel: `average_r * (N / (N + konst.))`
   - `konst. = n_years * TRADES_PER_YEAR_REFERENCE` (default: 15)
   - Zieht Average R zu Null bei wenigen Trades

2. **`risk_adjusted(profit_over_drawdown, n_trades, n_years)`** — Profit over Drawdown Adjustierung
   - Formel: `profit_over_drawdown * sqrt(N / (N + konst.))`
   - Stärkere Penalisierung als Average R (Wurzel-Skalierung)

3. **`bayesian_shrinkage(winrate, n_trades, all_winrates)`** — Winrate Adjustierung
   - Formel: `(wins + alpha) / (n + alpha + beta)`
   - Beta-Verteilung Prior basierend auf allen verfügbaren Winrates
   - `alpha` und `beta` aus empirischer Winrate-Verteilung berechnet

**Konfiguration:**
- `TRADES_PER_YEAR_REFERENCE = 15` (anpassbar in `metric_adjustments.py`)

**Verwendung:**

- **Yearly Metrics** (`n_years=1.0`): In `walkforward_analyzer.py` für jährliche Score-Berechnungen
- **Total Metrics** (`n_years=Backtest-Zeitraum`): In `combined_walkforward_matrix_analyzer.py` und `backfill_walkforward_equity_curves.py` für globale Scores

**CSV-Ausgabe-Spalten (trade-count adjusted):**

| Modul | Yearly-Spalten | Total-Spalten |
|------|---------------|--------------|
| `walkforward_analyzer.py` | `{YYYY}_winrate_adust` (%), `{YYYY}_avg_r_adust`, `{YYYY}_profit_over_dd_adust` | — |
| `combined_walkforward_matrix_analyzer.py` | — | `winrate_adust` (%), `avg_r_adust`, `profit_over_dd_adust` |

*Hinweis:* Winrate-Spalten sind in Prozent (0–100), konsistent mit den rohen Winrate-Spalten.

**Rationale:**
- Niedrige Trade-Zahlen → hohe statistische Unsicherheit → stärkere Shrinkage
- Verhindert Selektion von "Lucky Trades" mit hohen Scores bei wenigen Ausführungen
- Wilson Score Lower Bound gibt konservative Untergrenze für Winrate bei kleinen Samples

---

### `docs/` Ordner

Der `docs/` Ordner enthält technische Dokumentation und Migrationspläne:

- **`CATEGORICAL_RANKING_OPTIMIZATION.md`**: Detaillierter Performance-Optimierung Report
- **`PYTHON_312_MIGRATION_PLAN.md`**: Konvertierungsplan für die Migration von Python 3.10 auf Python 3.12
- **`RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`**: Vorbereitungsplan für die Migration ausgewählter Module zu Rust und Julia
- **`MIGRATION_READINESS_VALIDATION.md`**: Kanonischer Status-Report für Migration-Readiness (Go/No-Go)
- **`WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md`**: Vollständiger Implementierungsplan für Wave 0 Pilot (Slippage & Fee → Rust)
- **`rust-toolchain-requirements.md`**: Rust-Toolchain-Anforderungen (1.76.0+, PyO3, Maturin)
- **`julia-environment-requirements.md`**: Julia-Umgebungsanforderungen (1.10+, PythonCall)
- **`adr/`**: Architecture Decision Records (ADRs) für wichtige technische Entscheidungen
  - **`ADR-0001-migration-strategy.md`**: Rust und Julia Migrations-Strategie
  - **`ADR-0002-serialization-format.md`**: Arrow IPC für Zero-Copy FFI-Transfer
  - **`ADR-0003-error-handling.md`**: Hybrid Error-Handling (Python-Exceptions ↔ Result-Types)
  - **`ADR-0004-build-system-architecture.md`**: Build-System für Multi-Language Stack
- **`ffi/`**: Foreign Function Interface Spezifikationen
  - **`README.md`**: FFI-Übersicht und Konventionen
  - **`indicator_cache.md`**: IndicatorCache → Rust Interface
  - **`event_engine.md`**: EventEngine → Rust Interface
  - **`execution_simulator.md`**: ExecutionSimulator → Rust Interface
  - **`rating_modules.md`**: Rating-Module Interfaces
  - **`nullability-convention.md`**: Nullability-Regeln für FFI
- **`runbooks/`**: Migrations-Runbooks für die praktische Umsetzung
  - **`MIGRATION_RUNBOOK_TEMPLATE.md`**: Standard-Template für Modul-Migrationen
  - **`indicator_cache_migration.md`**: Runbook für IndicatorCache → Rust
  - **`event_engine_migration.md`**: Runbook für EventEngine → Rust
  - **`performance_baseline_documentation.md`**: Baseline-Dokumentation aller Kandidaten
  - **`ready_for_migration_checklist.md`**: Go/No-Go Checkliste (Template; kanonischer Status: `docs/MIGRATION_READINESS_VALIDATION.md`)

---

### Hybrid-Architektur (Python + Rust + Julia)

Das Projekt verwendet eine mehrschichtige Hybrid-Architektur, bei der performance-kritische Module optional in Rust oder Julia implementiert werden können.

#### Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Layer (Orchestrierung)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ FastAPI UI  │  │  Strategies │  │  Backtest Runner        │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                 │
│  ┌──────┴────────────────┴─────────────────────┴──────────────┐ │
│  │              Shared Protocols & Arrow Schemas              │ │
│  │                  (src/shared/protocols.py)                 │ │
│  └────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────────┘
                                │ Arrow IPC (Zero-Copy)
                ┌───────────────┴───────────────┐
                │                               │
┌───────────────▼───────────────┐ ┌─────────────▼─────────────────┐
│      Rust Layer (Hot-Paths)    │ │    Julia Layer (Research)     │
│  ┌─────────────────────────┐  │ │  ┌─────────────────────────┐  │
│  │    omega_rust (PyO3)    │  │ │  │  omega_julia (PyCall)   │  │
│  │  • IndicatorCache       │  │ │  │  • Monte Carlo VaR      │  │
│  │  • EventEngine          │  │ │  │  • Rolling Statistics   │  │
│  │  • ExecutionSimulator   │  │ │  │  • Bootstrap Methods    │  │
│  │  • Rating Functions     │  │ │  │  • Optimizer Extensions │  │
│  └─────────────────────────┘  │ │  └─────────────────────────┘  │
└───────────────────────────────┘ └───────────────────────────────┘
```

#### Datenfluss (FFI-Boundaries)

```
Python DataFrame
       │
       ▼ (pyarrow.Table → bytes)
┌──────────────────────┐
│  Arrow IPC Buffer    │  ← Zero-Copy Serialization
│  (Binary Format)     │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│  Rust   │ │  Julia  │
│ (arrow) │ │ (Arrow) │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
  Compute     Compute
     │           │
     └─────┬─────┘
           │
           ▼ (Result → Arrow → Python)
   Python Result
```

#### Module-zu-Sprache-Zuordnung

| Modul | Python | Rust | Julia | Rationale |
| --- | --- | --- | --- | --- |
| IndicatorCache | ✅ | 🎯 (Target) | - | Hot-Loop, 50x Speedup Target |
| EventEngine | ✅ | 🎯 (Target) | - | Core-Loop, 100x Speedup Target |
| ExecutionSimulator | ✅ | 🎯 (Target) | - | Trade-Matching, 50x Target |
| Rating/Scoring | ✅ | 🎯 (Target) | - | Numerische Berechnungen |
| Portfolio | ✅ | 🎯 (Target) | - | State-Management |
| Slippage & Fee | ✅ | 🎯 (Pilot) | - | Ideales Pilotmodul |
| Monte Carlo | ✅ | - | 🎯 (Target) | Research, Rapid Prototyping |
| Optimizer | ✅ | - | 🎯 (Target) | Orchestrierung, Optuna-Wrapper |
| Walkforward | ✅ | - | 🎯 (Target) | Research-Workflow |
| Strategies | ✅ | - | - | Bleibt Python (User-Code) |
| FastAPI/UI | ✅ | - | - | Bleibt Python |

**Legende:**

- ✅ = Aktuell implementiert/genutzt
- 🎯 = Migrations-Ziel (gemäß Runbooks)
- `-` = Nicht geplant für diese Sprache

#### Feature-Flag-System (geplant)

```python
# src/omega/config.py (Konzept)
import os

def _check_rust_available() -> bool:
    try:
        import omega_rust
        return True
    except ImportError:
        return False

def _check_julia_available() -> bool:
    try:
        from juliacall import Main
        return True
    except ImportError:
        return False

# Auto-Detection mit Override-Möglichkeit
USE_RUST_INDICATORS = os.getenv("OMEGA_USE_RUST", "auto") != "false" and _check_rust_available()
USE_JULIA_MONTE_CARLO = os.getenv("OMEGA_USE_JULIA", "auto") != "false" and _check_julia_available()
```

#### Build-System Integration

Das Build-System unterstützt alle drei Sprachen:

```
pyproject.toml          ← Python (pip, maturin)
├── src/rust_modules/
│   └── omega_rust/
│       ├── Cargo.toml  ← Rust (cargo, maturin)
│       └── pyproject.toml
└── src/julia_modules/
    └── omega_julia/
        └── Project.toml ← Julia (Pkg)
```

**Build-Kommandos:**

| Sprache | Development | Test | Release |
| --- | --- | --- | --- |
| Python | `pip install -e .[dev]` | `pytest` | `python -m build` |
| Rust | `maturin develop` | `cargo test` | `maturin build --release` |
| Julia | `Pkg.instantiate()` | `Pkg.test()` | (via Python wheel) |
| Alle | `make all` | `make test-all` | `make release` |

Weitere Details in `Makefile`, `justfile` und den CI-Workflows unter `.github/workflows/`.

---

### `reports/` Ordner

Der `reports/` Ordner enthält automatisch generierte Analyse-Berichte:

- **`migration_candidates/`**: Identifizierte Module für Rust/Julia-Migration
  - `p0-04_candidates.json` — Priorisierte Kandidatenliste
- **`migration_test_coverage/`**: Test-Coverage-Analyse für Kandidaten
  - `p0-05_candidate_coverage.json` — Coverage pro Modul
- **`mypy_baseline/`**: Type-Safety-Katalog
  - `p1-01_ignore_errors_catalog.json` — Module mit `ignore_errors`
- **`performance_baselines/`**: Benchmark-Baselines für Performance-Vergleich
  - `p0-01_*.json` — Baselines pro Modul (Candle + Tick Modus)
- **`type_coverage/`**: Type-Hint-Coverage-Analyse
