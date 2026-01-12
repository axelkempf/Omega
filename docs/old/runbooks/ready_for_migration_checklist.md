---
module: migration_readiness
phase: 5
prerequisites:
  - "docs/MIGRATION_READINESS_VALIDATION.md ist aktuell und widerspruchsfrei"
  - "Hard-gated CI (mind. pytest + strict mypy für migrationskritische Module) ist grün"
rollback_procedure: docs/runbooks/rollback_generic.md
---

## Ready-for-Migration Checkliste (Template)

**Version:** 1.0  
**Erstellt:** 2026-01-05  
**Phase:** P5-09 (Migrations-Vorbereitung Abschluss)  
**Status:** ⚠️ Template (nicht die kanonische Statusquelle)

---

## Übersicht

Diese Checkliste ist ein **Arbeits-Template** für die operativ reproduzierbare Vorbereitung.

Wichtig: Der **kanonische READY/NOT READY-Status** wird **nicht** in dieser Checkliste festgelegt,
sondern ausschließlich in `docs/MIGRATION_READINESS_VALIDATION.md`.

**Zertifizierung (Definition):** Ein Modul gilt als "Ready for Migration" nur dann, wenn die
zugrundeliegenden Checks als **hard CI gates** laufen (kein `continue-on-error`, kein `|| true`) und
lokal reproduzierbar sind.

---

## Phase 0: Foundation

### P0-01: Performance-Baseline

- [ ] Baselines für alle Kandidaten-Module generiert
- [ ] Baselines in `reports/performance_baselines/` gespeichert
- [ ] Reproduzierbar via `tools/perf_baseline.py`
- [ ] Benchmark-History-Tracking eingerichtet

**Artefakte:**

- `reports/performance_baselines/p0-01_*.json`
- `tools/perf_baseline.py`
- `tools/benchmark_history.py`

### P0-02: Type Coverage Analyse

- [ ] Type Coverage für alle Module analysiert
- [ ] Baseline dokumentiert in `reports/type_coverage/`
- [ ] Module mit `ignore_errors=true` identifiziert

**Artefakte:**

- `reports/type_coverage/README.md`
- `tools/type_coverage.py`

### P0-03: ADR-Struktur

- [ ] ADR-Verzeichnis `docs/adr/` eingerichtet
- [ ] ADR-Template erstellt
- [ ] ADR-0001 (Migration Strategy) erstellt

**Artefakte:**

- `docs/adr/ADR-TEMPLATE.md`
- `docs/adr/README.md`

### P0-04: Migrations-Kandidaten

- [ ] Kandidaten identifiziert und priorisiert
- [ ] Dokumentiert in `reports/migration_candidates/`
- [ ] JSON für automatisierte Verarbeitung

**Artefakte:**

- `reports/migration_candidates/README.md`
- `reports/migration_candidates/p0-04_candidates.json`

### P0-05: Test-Coverage Analyse

- [ ] Test-Coverage für Kandidaten dokumentiert
- [ ] Gap-Analyse durchgeführt
- [ ] Dokumentiert in `reports/migration_test_coverage/`

**Artefakte:**

- `reports/migration_test_coverage/README.md`
- `reports/migration_test_coverage/p0-05_candidate_coverage.json`

---

## Phase 1: Type Safety Hardening

### P1-01 bis P1-02: Katalog und Ranking

- [ ] Module mit `ignore_errors=true` katalogisiert
- [ ] Prioritäts-Ranking erstellt
- [ ] Dokumentiert in `reports/mypy_baseline/`

**Artefakte:**

- `reports/mypy_baseline/p1-01_ignore_errors_catalog.json`
- `reports/mypy_baseline/README.md`

### P1-03 bis P1-07: Mypy-Strict Migration

- [ ] `backtest_engine.core.types` definiert
- [ ] `backtest_engine.config.models` Pydantic-Modelle
- [ ] `backtest_engine.optimizer` mypy --strict (11/11 Files)
- [ ] `backtest_engine.core` mypy --strict (12/12 Files)
- [ ] `backtest_engine.rating` mypy --strict (12/12 Files)

**Artefakte:**

- `src/backtest_engine/core/types.py`
- `src/backtest_engine/config/models.py`
- Aktualisierte `pyproject.toml` mit Strict-Overrides

### P1-08: FFI Protocols

- [ ] Protocol-Klassen für FFI-Boundaries definiert
- [ ] `@runtime_checkable` Decorator verwendet
- [ ] Runtime-Smoke-Tests erstellt

**Artefakte:**

- `src/shared/protocols.py`
- `tests/test_shared_protocols_runtime.py`

### P1-09: Type Stubs

- [ ] Type Stubs für untyped Dependencies erstellt
- [ ] joblib Stubs vollständig
- [ ] optuna Stubs vollständig

**Artefakte:**

- `stubs/joblib/__init__.pyi`
- `stubs/optuna/__init__.pyi`
- `stubs/README.md`

### P1-10: Granulare Mypy-Konfiguration

- [ ] Tiered-Ansatz implementiert
- [ ] Kein globales `ignore_errors` mehr
- [ ] Differenzierte Overrides in `pyproject.toml`

**Artefakte:**

- `pyproject.toml` (aktualisiert)
- `reports/phase1_p1-09_p1-10_report.md`

---

## Phase 2: Interface-Definition

### P2-01 bis P2-04: FFI-Spezifikationen

- [ ] `indicator_cache.py` spezifiziert
- [ ] `event_engine.py` spezifiziert
- [ ] `execution_simulator.py` spezifiziert
- [ ] Rating-Module spezifiziert

**Artefakte:**

- `docs/ffi/indicator_cache.md`
- `docs/ffi/event_engine.md`
- `docs/ffi/execution_simulator.md`
- `docs/ffi/rating_modules.md`

### P2-05: Serialisierungsformat

- [ ] Format evaluiert (Arrow IPC gewählt)
- [ ] ADR-0002 erstellt
- [ ] Typ-Mapping dokumentiert

**Artefakte:**

- `docs/adr/ADR-0002-serialization-format.md`

### P2-06: Arrow-Schemas

- [ ] 6 Schemas definiert
- [ ] Zero-Copy Utilities implementiert
- [ ] Tests für Schema-Validierung

**Artefakte:**

- `src/shared/arrow_schemas.py`

### P2-07: Fehlerbehandlungs-Konvention

- [ ] ADR-0003 erstellt
- [ ] Hybrid-Ansatz dokumentiert
- [ ] ErrorCode Enum definiert

**Artefakte:**

- `docs/adr/ADR-0003-error-handling.md`
- `src/shared/error_codes.py`
- `src/shared/exceptions.py`

### P2-08 bis P2-10: Dokumentation

- [ ] FFI-Index erstellt
- [ ] Nullability-Konvention dokumentiert
- [ ] Data-Flow-Diagramme erstellt

**Artefakte:**

- `docs/ffi/README.md`
- `docs/ffi/nullability-convention.md`
- `docs/ffi/data-flow-diagrams.md`

---

## Phase 3: Test-Infrastruktur

### P3-01: pytest-benchmark Setup

- [ ] pytest-benchmark installiert
- [ ] Konfiguration in `pyproject.toml`
- [ ] JSON-Export konfiguriert

### P3-02 bis P3-04: Benchmark-Suites

- [ ] `tests/benchmarks/test_bench_indicator_cache.py`
- [ ] `tests/benchmarks/test_bench_event_engine.py`
- [ ] `tests/benchmarks/test_bench_rating.py`

### P3-05 bis P3-07: Property-Based Tests

- [ ] Hypothesis installiert und konfiguriert
- [ ] `tests/property/test_prop_indicators.py`
- [ ] `tests/property/test_prop_scoring.py`

### P3-08 bis P3-10: Golden-File Tests

- [ ] Golden-File Framework eingerichtet
- [ ] `tests/golden/test_golden_backtest.py`
- [ ] `tests/golden/test_golden_optimizer.py`
- [ ] Reference-Dateien werden beim ersten Golden-Testlauf unter `tests/golden/reference/` erzeugt (docs-lint:planned)

### P3-11 bis P3-12: CI-Integration

- [ ] `.github/workflows/benchmarks.yml`
- [ ] Benchmark-History-Tracking (`tools/benchmark_history.py`)
- [ ] Tests für History-Tool

---

## Phase 4: Build-System

### P4-01 bis P4-02: Toolchain-Dokumentation

- [ ] `docs/rust-toolchain-requirements.md`
- [ ] `docs/julia-environment-requirements.md`

### P4-03 bis P4-05: GitHub Actions Workflows

- [ ] `.github/workflows/rust-build.yml`
- [ ] `.github/workflows/julia-tests.yml`
- [ ] `.github/workflows/cross-platform-ci.yml`
- [ ] Cache-Strategie implementiert

### P4-06 bis P4-07: FFI-Integration Templates

- [ ] PyO3/Maturin Template (`src/rust_modules/omega_rust/`)
- [ ] PythonCall Template (`src/julia_modules/omega_julia/`)

### P4-08 bis P4-10: Lokale Entwicklung

- [ ] `Makefile` (~400 Zeilen)
- [ ] `justfile` (~350 Zeilen)
- [ ] `.devcontainer/` für VS Code

### P4-12: Release-Workflow

- [ ] `.github/workflows/release.yml`

---

## Phase 5: Dokumentation & Validation

### P5-01 bis P5-04: ADRs

- [ ] ADR-0001: Migration Strategy
- [ ] ADR-0002: Serialization Format
- [ ] ADR-0003: Error Handling
- [ ] ADR-0004: Build-System Architecture

### P5-05: Migrations-Runbook Template

- [ ] Template erstellt
- [ ] Best Practices dokumentiert
- [ ] Verwendungsanleitung

**Artefakt:** `docs/runbooks/MIGRATION_RUNBOOK_TEMPLATE.md`

### P5-06: Runbook IndicatorCache

- [ ] Vollständiges Runbook
- [ ] Performance-Targets
- [ ] Rollback-Plan

**Artefakt:** `docs/runbooks/indicator_cache_migration.md`

### P5-07: Runbook EventEngine

- [ ] Vollständiges Runbook
- [ ] Migration Strategies
- [ ] Rollback-Plan

**Artefakt:** `docs/runbooks/event_engine_migration.md`

### P5-08: Performance-Baseline-Dokumentation

- [ ] Alle Baselines zusammengefasst
- [ ] Improvement-Targets definiert
- [ ] ROI-Analyse

**Artefakt:** `docs/runbooks/performance_baseline_documentation.md`

### P5-09: Ready-for-Migration Checkliste

- [ ] Checkliste erstellt (dieses Dokument)
- [ ] Alle Phasen abgedeckt

### P5-10: README.md Update

- [ ] Rust/Julia-Support dokumentiert
- [ ] Build-Anweisungen aktualisiert
- [ ] Dev-Setup erklärt

### P5-11: CONTRIBUTING.md Update

- [ ] Rust/Julia-Contributions Guidelines
- [ ] Build-Kommandos
- [ ] Testing-Anweisungen

### P5-12: architecture.md Update

- [ ] Rust-Modul-Struktur
- [ ] Julia-Modul-Struktur
- [ ] Hybrid-Architektur erklärt

---

## Finale Validierung

### Pre-Migration Checks

Vor dem Start einer Migration muss jedes Modul folgende Checks bestehen:

#### 1. Type Safety

```bash
# Mypy-Strict für das Modul
mypy src/backtest_engine/core/indicator_cache.py --strict
# Erwartet: 0 Errors
```

- [ ] Modul ist mypy --strict compliant
- [ ] Alle öffentlichen APIs haben Type Hints
- [ ] Keine `# type: ignore` ohne Begründung

#### 2. Test Coverage

```bash
# Coverage für das Modul
pytest tests/ -k indicator_cache --cov=src/backtest_engine/core/indicator_cache --cov-report=term-missing
# Erwartet: ≥ 85% Coverage
```

- [ ] Test Coverage ≥ 85%
- [ ] Property-Based Tests vorhanden
- [ ] Golden-File Tests vorhanden

#### 3. Performance Baseline

```bash
# Benchmark ausführen
pytest tests/benchmarks/test_bench_indicator_cache.py --benchmark-json=baseline.json
```

- [ ] Baseline-Datei vorhanden
- [ ] Improvement-Target definiert
- [ ] Benchmark reproduzierbar

#### 4. FFI-Dokumentation

- [ ] Interface-Spezifikation in `docs/ffi/`
- [ ] Arrow-Schema definiert
- [ ] Nullability dokumentiert
- [ ] Error-Handling definiert

#### 5. Runbook

- [ ] Migrations-Runbook erstellt
- [ ] Rollback-Plan definiert
- [ ] Akzeptanzkriterien klar

---

## Zertifizierung (Record)

### Ready-for-Migration Status pro Modul

Diese Tabelle ist ein **lokales Arbeitsprotokoll**.

- Der **kanonische Status** pro Modul wird in `docs/MIGRATION_READINESS_VALIDATION.md` geführt.
- Wenn du hier etwas abhaken willst, aktualisiere parallel den Validierungsreport mit Evidence.

| Modul | Type Safety | Test Coverage | Baseline | FFI-Spec | Runbook | Status (kanonisch) |
| --- | --- | --- | --- | --- | --- | --- |
| IndicatorCache | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| EventEngine | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| ExecutionSimulator | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| Rating Modules | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| MultiSymbolSlice | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| SymbolDataSlicer | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| Portfolio | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| Slippage & Fee | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| Optimizer | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |
| Walkforward | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | siehe Validierungsreport |

**Legende:**

- 🟢 READY: Alle Vorbedingungen erfüllt, Migration kann beginnen
- 🟡 PENDING: Teilweise erfüllt, spezifische Items fehlen
- 🔴 NOT READY: Kritische Vorbedingungen fehlen

Hinweis: Das bloße Vorhandensein von Specs/Runbooks ist **notwendige Dokumentation**, aber kein
Ersatz für hard-gated Checks. Status-Claims gehören in den Validierungsreport.

---

## Empfohlene Migrations-Reihenfolge

Basierend auf Priorität, Abhängigkeiten und Readiness:

### Wave 1: Pilot-Module (unabhängig, klare Grenzen)

1. **Slippage & Fee** - Ideales Pilot: reine Mathematik, keine Abhängigkeiten
2. **Rating Modules** - Numerische Scores, gut testbar

### Wave 2: Core Performance (Hauptgewinne)

1. **IndicatorCache** - Größter Performance-Impact
2. **SymbolDataSlicer** - Häufig aufgerufen

### Wave 3: Core Loop (sensibel)

1. **EventEngine** - Core-Loop, nach Stabilisierung von Wave 1+2
2. **ExecutionSimulator** - Trade-Matching

### Wave 4: State Management

1. **Portfolio** - Stateful, nach stabilem Core-Loop

### Wave 5: Orchestrierung

1. **Optimizer** (Julia) - Research-lastig
2. **Walkforward** (Julia) - Orchestrierung

### Wave 6: Performance Cleanup

1. **MultiSymbolSlice** - Nach Type-Hardening

---

## Sign-Off

### Vorbereitungsphase abgeschlossen

| Rolle | Name | Datum | Signatur |
| --- | --- | --- | --- |
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

### Migration freigegeben für

| Wave | Module | Freigabe-Datum | Freigegeben von |
| --- | --- | --- | --- |
| Wave 1 | Slippage & Fee, Rating | ⏳ | ⏳ |
| Wave 2 | IndicatorCache, SymbolDataSlicer | ⏳ | ⏳ |

---

## Referenzen

- Migrations-Plan: `docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`
- ADR-Verzeichnis: `docs/adr/`
- FFI-Spezifikationen: `docs/ffi/`
- Runbooks: `docs/runbooks/`
- Baselines: `reports/performance_baselines/`
- Type Coverage: `reports/type_coverage/`
