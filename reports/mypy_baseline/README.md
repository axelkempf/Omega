# Mypy Baseline (Phase 1: Type Safety Hardening)

Dieses Verzeichnis enthält Planungsartefakte für **Phase 1** des Rust/Julia-Vorbereitungsplans
(`docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`).

Ziel: eine **evidence-based** Priorisierung für die schrittweise Strict-Migration (Mypy) der
aktuell via `ignore_errors=true` „abgedeckten“ Bereiche, ohne Breaking Changes.

## P1-01: Katalog `ignore_errors`-Packages

Baseline (mit `tools/mypy_phase1.ini`, also *ohne* die `ignore_errors`-Overrides aus
`pyproject.toml` zu nutzen):

| Package | Dateien | Errors | Notes | Errors/Datei |
|---|---:|---:|---:|---:|
| `backtest_engine` | 77 | 449 | 96 | 5.83 |
| `hf_engine` | 46 | 74 | 19 | 1.61 |
| `ui_engine` | 17 | 3 | 0 | 0.18 |

JSON-Quelle: `reports/mypy_baseline/p1-01_ignore_errors_catalog.json`

**Interpretation (kurz):**

- `backtest_engine` ist der dominante Typing-Hotspot (ca. 86% der Errors im Katalog).
- `ui_engine` ist bereits weitgehend typisiert (nur `controller.py` fällt auf).
- `hf_engine` bleibt gemäß Plan **pure Python** und wird in Phase 1 nicht in strict migriert
  (Typing nur opportunistisch/defensiv an klaren Boundaries).

## P1-02: Prioritäts-Ranking (Strict-Migration)

Heuristik: **FFI-Relevanz** (hoch) + **Abhängigkeitslage** + **Error-Density** (Effort-Indikator).

### Tier A (frühe Wins, FFI-nah, geringe Friktion)

1. `backtest_engine.rating` (12 Dateien, 10 Errors, 0.83 Errors/Datei)
   - Gute Chance auf schnelle Strict-Enablement-Erfolge.
   - Hohe Relevanz für spätere Rust-Module (deterministische Scores, klarer I/O).
2. `backtest_engine.data` (10 Dateien, 28 Errors, 2.80 Errors/Datei)
   - Fundament für FFI-taugliche Schemas (Candle/Tick, Timeframe, etc.).
3. `ui_engine.controller.py` (1 Datei, 3 Errors)
   - Klein, isoliert, reduziert Rauschen im System.

### Tier B (FFI-kritisch, aber „schwer“)

4. `backtest_engine.optimizer` (11 Dateien, 68 Errors, 6.18 Errors/Datei)
   - Priorität hoch (Julia/Rust Kandidat), aber viele Stellen zu härten.
5. `backtest_engine.core` (12 Dateien, 75 Errors, 6.25 Errors/Datei)
   - Core-Loop/Hot-path, sehr sensibel. Strict erst nach stabilen Typ-Boundaries.

### Tier C (nicht primär FFI, hoher Aufwand oder nachgelagert)

- `backtest_engine.analysis` (7 Dateien, 191 Errors, 27.29 Errors/Datei)
- `backtest_engine.runner.py` (1 Datei, 44 Errors, 44.0 Errors/Datei)

Diese Bereiche sind wichtig, aber für die FFI-Vorbereitung nicht der schnellste Hebel.

## Reproduzieren

```bash
python tools/mypy_phase1_catalog.py
```

(Optional) Anderer Output-Pfad:

```bash
python tools/mypy_phase1_catalog.py --out-json reports/mypy_baseline/p1-01_ignore_errors_catalog.json
```

## Umsetzung im Code (Strict-Carve-outs)

Aktuell sind folgende Module **explizit strict** (während `backtest_engine.*` insgesamt noch im
`ignore_errors`-Bucket bleibt):

- `backtest_engine.core.types`
- `backtest_engine.rating` / `backtest_engine.rating.*`
