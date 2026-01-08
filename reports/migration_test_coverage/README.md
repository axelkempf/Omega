# Migration Candidate Test Coverage (P0-05)

Dieser Report dokumentiert die Test-Coverage der P0-04 Migrations-Kandidaten.

Konventionen:

- Candidate-Warnschwelle: 80% (gewichtete Statements)
- File-Warnschwelle: 70%

## Summary

| Kandidat | Priority | Target | Coverage% | Statements | Gemessen | Fehlend |
|---|---:|---|---:|---:|---:|---:|
| Indicator Cache | High | Rust | 14.3⚠ | 610 | 1 | 0 |
| Optimizer (Final Selection / Robust Zone) | Medium | Julia | 4.2⚠ | 4342 | 11 | 0 |
| Symbol Data Slicer | Medium | Rust | 21.4⚠ | 70 | 1 | 0 |
| Portfolio | Medium | Rust | 30.2⚠ | 265 | 1 | 0 |
| Multi-Symbol Slice | Medium | Rust | 39.1⚠ | 23 | 1 | 0 |
| Slippage & Fee | Medium | Rust | 60.9⚠ | 23 | 1 | 0 |
| Walkforward (stubbed window) | Low | Julia | 0.0⚠ | 846 | 1 | 0 |
| Execution Simulator | Low | Rust | 8.1⚠ | 332 | 1 | 0 |
| Event Engine | Low | Rust | 13.9⚠ | 101 | 1 | 0 |
| Analysis Pipelines | Low | Julia | 29.9⚠ | 7139 | 7 | 0 |
| Rating Modules | Low | Rust | 71.2⚠ | 925 | 12 | 0 |

## Gap-Analyse

Fokus: Kandidaten mit niedriger Coverage oder fehlenden Coverage-Daten.

### Indicator Cache

- Coverage: 14.3%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/indicator_cache.py`: 14.3% (610 stmts)

### Optimizer (Final Selection / Robust Zone)

- Coverage: 4.2%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/optimizer/_settings.py`: 0.0% (12 stmts)
  - `src/backtest_engine/optimizer/grid_searcher.py`: 0.0% (54 stmts)
  - `src/backtest_engine/optimizer/robust_zone_analyzer.py`: 0.0% (650 stmts)
  - `src/backtest_engine/optimizer/symbol_grid.py`: 0.0% (12 stmts)
  - `src/backtest_engine/optimizer/walkforward.py`: 0.0% (846 stmts)
  - `src/backtest_engine/optimizer/walkforward_plot.py`: 0.0% (25 stmts)
  - `src/backtest_engine/optimizer/walkforward_utils.py`: 0.0% (152 stmts)
  - `src/backtest_engine/optimizer/final_param_selector.py`: 5.0% (2155 stmts)
  - `src/backtest_engine/optimizer/optuna_optimizer.py`: 13.5% (281 stmts)
  - `src/backtest_engine/optimizer/instrumentation.py`: 22.6% (155 stmts)

### Symbol Data Slicer

- Coverage: 21.4%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/symbol_data_slicer.py`: 21.4% (70 stmts)

### Portfolio

- Coverage: 30.2%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/portfolio.py`: 30.2% (265 stmts)

### Multi-Symbol Slice

- Coverage: 39.1%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/multi_symbol_slice.py`: 39.1% (23 stmts)

### Slippage & Fee

- Coverage: 60.9%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/slippage_and_fee.py`: 60.9% (23 stmts)

### Walkforward (stubbed window)

- Coverage: 0.0%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/optimizer/walkforward.py`: 0.0% (846 stmts)

### Execution Simulator

- Coverage: 8.1%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/execution_simulator.py`: 8.1% (332 stmts)

### Event Engine

- Coverage: 13.9%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/core/event_engine.py`: 13.9% (101 stmts)

### Analysis Pipelines

- Coverage: 29.9%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/analysis/combine_equity_curves.py`: 0.0% (621 stmts)
  - `src/backtest_engine/analysis/final_combo_equity_plotter.py`: 0.0% (598 stmts)
  - `src/backtest_engine/analysis/walkforward_analyzer.py`: 15.8% (1969 stmts)
  - `src/backtest_engine/analysis/backfill_walkforward_equity_curves.py`: 35.1% (686 stmts)
  - `src/backtest_engine/analysis/combined_walkforward_matrix_analyzer.py`: 46.5% (3129 stmts)

### Rating Modules

- Coverage: 71.2%
- Low-Coverage Dateien (< 70%):
  - `src/backtest_engine/rating/ulcer_index_score.py`: 19.4% (67 stmts)
  - `src/backtest_engine/rating/tp_sl_stress_score.py`: 65.8% (243 stmts)
  - `src/backtest_engine/rating/trade_dropout_score.py`: 67.9% (156 stmts)
- Note: `strategy_rating.py` entfernt (Funktionalität inline in walkforward.py)

## Reproduzieren

- JSON: `tools/migration_test_coverage.py --format json`
- Markdown: `tools/migration_test_coverage.py --format md`

Hinweis: Standardmäßig werden Integrationstests via Marker ausgeschlossen: `-m not integration`.
