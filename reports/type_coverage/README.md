# Type Coverage Baseline (P0-02)

Stand: 2026-01-03 (macOS, Python 3.12.12)

Dieses Dokument erfasst die Baseline der Type-Hint-Abdeckung im Repo als Startpunkt
für den Rust/Julia-Vorbereitungsplan (P0-02).

## Ergebnisse (AST-basierte Messung)

Quelle: `tools/type_coverage.py` (AST-Analyse; zählt nur Public-Funktionen,
private Funktionen/Methode mit Prefix `_` werden übersprungen; `__dunder__` bleibt drin).

Gesamt (unter `src/`):

- Dateien analysiert: **161**
- Return Type Coverage: **83.1%** (595/716)
- Parameter Type Coverage: **95.1%** (1257/1322)

### Breakdown nach Top-Level Paket

(Definition: `src/<group>/...`)

- `backtest_engine`: **79.1%** Return Types (253/320), **97.2%** Parameter (684/704)
- `hf_engine`: **85.1%** Return Types (234/275), **93.5%** Parameter (458/490)
- `ui_engine`: **84.1%** Return Types (37/44), **100.0%** Parameter (30/30)
- `strategies`: **97.2%** Return Types (70/72), **86.3%** Parameter (82/95)
- `engine_launcher.py`: **20.0%** Return Types (1/5), **100.0%** Parameter (3/3)

## Hotspots (erste Kandidaten für Type-Hardening)

Aus der Top-10-Liste „Files needing most attention“ der aktuellen Ausführung:

- `src/backtest_engine/core/event_engine.py` (0.0%)
- `src/backtest_engine/core/multi_tick_controller.py` (0.0%)
- `src/backtest_engine/core/tick_event_engine.py` (0.0%)
- `src/backtest_engine/core/multi_strategy_controller.py` (0.0%)
- `src/backtest_engine/bt_logging/trade_logger.py` (0.0%)
- `src/hf_engine/core/controlling/strategy_runner.py` (0.0%)
- `src/ui_engine/strategies/factory.py` (0.0%)
- `src/ui_engine/datafeeds/factory.py` (0.0%)
- `src/strategies/mean_reversion_z_score/live/portfolio_strategy.py` (0.0%)
- `src/engine_launcher.py` (20.0%)

Hinweis: Einige dieser Dateien liegen in Bereichen, die laut `pyproject.toml` bereits
als „strict“ konfiguriert sind (siehe unten). Das ist ein gutes Signal, dass wir
Mypy-Strict-Overlays und tatsächliche Annotation-Coverage einmal gegeneinander
validieren sollten (z.B. via `mypy`-Run), bevor wir die Strict-Rollout-Roadmap
finalisieren.

## Mypy-Konfiguration (Auszug) und `ignore_errors`

Quelle: `pyproject.toml`

Global:

- `ignore_missing_imports = true` (reduziert Friktion, kann aber fehlende Stubs maskieren)

Strict overrides (bereits als „strict“ konfiguriert):

- `strategies._base.*`
- `strategies.mean_reversion_z_score.*`
- `backtest_engine.core.types`
- `backtest_engine.optimizer._settings`
- `backtest_engine.rating` / `backtest_engine.rating.*`
- `backtest_engine.config` / `backtest_engine.config.*`
- `shared` / `shared.*`

Relaxed overrides (`ignore_errors = true`):

- `hf_engine.adapter.*`, `hf_engine.core.*`, `hf_engine.infra.*`
- `backtest_engine.analysis.*`, `backtest_engine.data.*`, `backtest_engine.deployment.*`, `backtest_engine.logging.*`,
  `backtest_engine.report.*`, `backtest_engine.sizing.*`, `backtest_engine.strategy.*`, `backtest_engine.runner`,
  `backtest_engine.run_all`, `backtest_engine.batch_runner`
- `backtest_engine.core.*` (außer `backtest_engine.core.types` als carve-out)
- `backtest_engine.optimizer.*` (außer `backtest_engine.optimizer._settings` als carve-out)

Hinweis: `ui_engine` wird aktuell **nicht** via `ignore_errors` maskiert und ist damit als „realer“ Typing-Kandidat gut sichtbar.

Diese Relaxed-Overlays sind die expliziten Kandidaten für P1-01/P1-02 (Katalog + Ranking)
und später für die schrittweise Strict-Migration.

## Reproduzierbarkeit

Erzeugung der Baseline:

- Summary + JSON-Export: `tools/type_coverage.py`
- JSON Output: `var/reports/type_coverage.json`

Wichtig: `var/` ist Runtime-State (typisch gitignored). Dieses Markdown-Dokument ist die
„commitbare“ Baseline-Zusammenfassung; die JSON-Datei kann lokal/CI jederzeit neu erzeugt
werden.
