### Projektarchitektur

Ordner-für-Ordner-Übersicht der Codebasis (ohne `results`-Ordner und ohne Auflistung einzelner `.csv`-Dateien).

### Wurzelverzeichnis

- `configs/`
  - `backtest/`
    - `_config_validator.py`
    - `ema_rejection_trend_follow_backtest.json`
    - `ema_rsi_trend_follow_backtest.json`
    - `mean_reversion_bollinger_bands_macd.json`
    - `mean_reversion_z_score.json`
    - `statistical_arbitrage_backtest.json`
    - `trading_the_flow_backtest.json`
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
      - `ema.rs` *(Exponential Moving Average)*
      - `rsi.rs` *(Relative Strength Index)*
      - `statistics.rs` *(Rolling standard deviation)*
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
- `ema_rejection_trend_follow/`
  - `__init__.py`
  - `backtest/`
    - `__init__.py`
    - `backtes_utils.py`
    - `backtest_strategy.py`
    - `position_manager.py`
    - `walkforward_backtest.py`
  - `live/`
    - `__init__.py`
    - `config_H1.py`
    - `config_H4.py`
    - `doji_thresholds.json`
    - `pip_thresholds.json`
    - `position_manager.py`
    - `scenarios_interface.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`
- `ema_rsi_trend_follow/`
  - `__init__.py`
  - `backtest/`
    - `__init__.py`
    - `backtes_utils.py`
    - `backtest_strategy.py`
    - `position_manager.py`
    - `walkforward_backtest.py`
  - `live/`
    - `__init__.py`
    - `config_H1.py`
    - `config_M15.py`
    - `doji_thresholds.json`
    - `position_manager.py`
    - `scenarios_H1.py`
    - `scenarios_interface.py`
    - `scenarios_M15.py`
    - `strategy.py`
    - `utils.py`
- `macd_trend_follow/`
  - `__init__.py`
  - `live/`
    - `__init__.py`
    - `config_H1.py`
    - `config_H4.py`
    - `doji_thresholds.json`
    - `pip_thresholds.json`
    - `position_manager.py`
    - `scenarios_interface.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`
- `mean_reversion_bollinger_bands_plus_macd/`
  - `__init__.py`
  - `backtest/`
    - `__init__.py`
    - `backtest_strategy.py`
    - `backtest_utils.py`
    - `grid_search_backtest.py`
    - `position_manager.py`
    - `walkforward_backtest.py`
  - `live/`
    - `__init__.py`
    - `config_H1.py`
    - `config_H4.py`
    - `config_M15.py`
    - `config_M30.py`
    - `config_M5.py`
    - `doji_thresholds.json`
    - `pip_thresholds.json`
    - `position_manager.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`
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
- `pre_session_momentum/`
  - `__init__.py`
  - `live/`
    - `__init__.py`
    - `config_pre_london.py`
    - `config_pre_new_york.py`
    - `doji_thresholds.json`
    - `position_manager.py`
    - `scenarios_interface.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`
- `session_momentum/`
  - `__init__.py`
  - `live/`
    - `__init__.py`
    - `config_asia.py`
    - `config_london.py`
    - `config_new_york.py`
    - `doji_thresholds.json`
    - `position_manager.py`
    - `scenarios_interface.py`
    - `scenarios.py`
    - `strategy.py`
    - `utils.py`
- `statistical_arbitrage/`
  - `__init__.py`
  - `backtest/`
    - `__init__.py`
    - `backtes_utils.py`
    - `backtest_strategy.py`
    - `position_manager.py`
- `trading_the_flow/`
  - `__init__.py`
  - `backtest/`
    - `__init__.py`
    - `backtest_strategy.py`
    - `backtest_utils.py`
    - `position_manager.py`
    - `walkforward_backtest.py`
  - `live/`
    - `__init__.py`
    - `config_asia.py`
    - `config_london.py`
    - `config_new_york.py`
    - `doji_thresholds.json`
    - `position_manager.py`
    - `scenarios_interface.py`
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
- `src/omega.egg-info/` *(build artifact from current distribution name)*
  - `dependency_links.txt`
  - `PKG-INFO`
  - `requires.txt`
  - `SOURCES.txt`
  - `top_level.txt`
- `src/watchdog/`

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
- **`adr/`**: Architecture Decision Records (ADRs) für wichtige technische Entscheidungen
  - **`ADR-0001-migration-strategy.md`**: Rust und Julia Migrations-Strategie

