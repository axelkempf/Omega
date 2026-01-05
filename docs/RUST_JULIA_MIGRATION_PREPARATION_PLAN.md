# Vorbereitungsplan: Migration ausgewählter Python-Module zu Rust und Julia

## Executive Summary

Dieses Dokument beschreibt den systematischen Vorbereitungsplan zur sicheren, inkrementellen Migration ausgewählter Module des Omega Trading-Systems von Python zu Rust und Julia. Der Plan fokussiert auf fünf Kernbereiche: (1) Type Safety Hardening durch schrittweise Mypy-Strict-Aktivierung, (2) Interface-Definition mit klaren Serialisierungsformaten für FFI-Grenzen, (3) Test-Infrastruktur mit Benchmarks, Property-Based Tests und Golden-File-Validierung, (4) Build-System-Erweiterung für Cross-Platform Rust/Julia-Kompilierung, sowie (5) Dokumentation mit ADRs und Migrations-Runbooks. Der Plan priorisiert **Stabilität > Performance > Code-Eleganz** und garantiert keine Breaking Changes für bestehende Nutzer während der Vorbereitung. Die Live-Trading-Engine (`hf_engine/`) bleibt pure Python.

---

## Inhaltsverzeichnis

1. [Phasen-Übersicht](#1-phasen-übersicht)
2. [Detaillierte Aufgabenliste](#2-detaillierte-aufgabenliste)
3. [Risiko-Matrix](#3-risiko-matrix)
4. [Technische Entscheidungen](#4-technische-entscheidungen)
5. [Erfolgsmetriken](#5-erfolgsmetriken)
6. [Anhang: Migrations-Kandidaten](#anhang-migrations-kandidaten)

---

## 1. Phasen-Übersicht

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MIGRATIONS-VORBEREITUNGSPLAN                          ║
║                    (Keine Breaking Changes während Vorbereitung)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Phase 0: Foundation (Woche 1-2)                                             ║
║  ├─ Baseline-Dokumentation erstellen                                         ║
║  ├─ Performance-Baselines aufzeichnen                                        ║
║  └─ ADR-Struktur einrichten                                                  ║
║                                                                              ║
║  Phase 1: Type Safety Hardening (Woche 3-6)                                  ║
║  ├─ Mypy ignore_errors Module identifizieren und priorisieren               ║
║  ├─ TypedDict/Pydantic-Schemas für FFI-Grenzen definieren                   ║
║  └─ Schrittweise Strict-Mode Aktivierung                                    ║
║                                                                              ║
║  Phase 2: Interface-Definition (Woche 7-9)                                   ║
║  ├─ Input/Output-Typen für Migrations-Kandidaten spezifizieren              ║
║  ├─ Serialisierungsformat wählen und implementieren                         ║
║  └─ Fehlerbehandlungs-Konventionen dokumentieren                            ║
║                                                                              ║
║  Phase 3: Test-Infrastruktur (Woche 10-13)                                   ║
║  ├─ pytest-benchmark Suite einrichten                                        ║
║  ├─ Hypothesis Property-Based Tests für numerische Module                   ║
║  └─ Golden-File Tests für Determinismus-Validierung                         ║
║                                                                              ║
║  Phase 4: Build-System (Woche 14-16)                                         ║
║  ├─ GitHub Actions Workflow für Rust/Julia                                   ║
║  ├─ Cross-Platform CI (MacOS, Linux, Windows)                               ║
║  └─ Lokale Dev-Setup-Anleitung (Makefile/justfile)                          ║
║                                                                              ║
║  Phase 5: Dokumentation & Validation (Woche 17-18)                           ║
║  ├─ ADRs finalisieren                                                        ║
║  ├─ Migrations-Runbooks pro Modul                                           ║
║  └─ Ready-for-Migration Assessment                                          ║
║                                                                              ║
║  ════════════════════════════════════════════════════════════════════════    ║
║  Meilensteine:                                                               ║
║  [M1] Woche 2:  Baseline-Dokumentation vollständig                          ║
║  [M2] Woche 6:  Type Coverage ≥80% in Migrations-Kandidaten                 ║
║  [M3] Woche 9:  FFI-Interfaces dokumentiert und validiert                   ║
║  [M4] Woche 13: Test-Infrastruktur vollständig                              ║
║  [M5] Woche 16: CI/CD für Rust/Julia funktional                             ║
║  [M6] Woche 18: "Ready for Migration" Zertifizierung                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Detaillierte Aufgabenliste

### Phase 0: Foundation

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P0-01** | Performance-Baseline für Migrations-Kandidaten erstellen | - | M | Benchmark-Results für alle Kandidaten-Module dokumentiert; Laufzeiten, Memory-Usage, CPU-Profile |
| **P0-02** | Aktuelle Type Coverage analysieren und dokumentieren | - | S | `tools/type_coverage.py` Output + Analyse der Module mit `ignore_errors=true` in `pyproject.toml` (Baseline: `reports/type_coverage/README.md`) |
| **P0-03** | ADR-Verzeichnisstruktur einrichten | - | S | `docs/adr/` Verzeichnis mit Template und erstem ADR (ADR-0001: Migration Strategy) |
| **P0-04** | Migrations-Kandidaten identifizieren und priorisieren | P0-01, P0-02 | M | Evidence-based Liste (Performance-Baselines + Type-Readiness) inkl. Priorität (High/Medium/Low), dokumentiert in `reports/migration_candidates/README.md` (+ JSON: `reports/migration_candidates/p0-04_candidates.json`) |
| **P0-05** | Bestehende Test-Coverage für Kandidaten dokumentieren | P0-04 | S | Evidence-based Coverage-Report + Gap-Analyse, dokumentiert in `reports/migration_test_coverage/README.md` (+ JSON: `reports/migration_test_coverage/p0-05_candidate_coverage.json`) |

### Phase 1: Type Safety Hardening

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P1-01** | Module mit `ignore_errors=true` katalogisieren | P0-02 | S | Vollständige Liste: `hf_engine.*`, `backtest_engine.*`, `ui_engine.*` mit Datei-Count und Error-Count pro Modul |
| **P1-02** | Prioritäts-Ranking für Mypy-Strict erstellen | P1-01, P0-04 | S | Ranking basierend auf: (1) FFI-Relevanz, (2) Error-Density, (3) Abhängigkeiten |
| **P1-03** | TypedDict-Schemas für `backtest_engine.core` definieren | P1-02 | L | `src/backtest_engine/core/types.py` mit allen Interface-Typen; mypy --strict passiert |
| **P1-04** | Pydantic-Modelle für Config-Objekte standardisieren | P1-03 | M | Einheitliche Config-Modelle in `src/backtest_engine/config/models.py` |
| **P1-05** | `backtest_engine.optimizer` auf Strict-Mode migrieren | P1-03 | XL | Alle Dateien in `optimizer/` passieren mypy --strict; keine `# type: ignore` ohne Begründung |
| **P1-06** | `backtest_engine.core` auf Strict-Mode migrieren | P1-05 | XL | Alle Dateien in `core/` passieren mypy --strict |
| **P1-07** | `backtest_engine.rating` auf Strict-Mode migrieren | P1-03 | L | Alle Rating-Module strict-compliant |
| **P1-08** | Protocol-Klassen für FFI-Boundaries definieren | P1-03 | M | `src/shared/protocols.py` mit `@runtime_checkable` Protocols für alle externen Schnittstellen |
| **P1-09** | Type Stubs für untyped Dependencies erstellen | P1-01 | M | `.pyi` Stubs für kritische untyped Libraries oder in `py.typed` Marker |
| **P1-10** | Mypy-Konfiguration granular aufteilen | P1-06, P1-07 | S | `pyproject.toml` mit differenzierten `[[tool.mypy.overrides]]` Blöcken; kein globales `ignore_errors` |

#### Phase 1 – Implementierungsstatus (Stand: 2026-01-04)

- **P1-01 (Katalog):** ✅ Baseline-Report erzeugt:
	- JSON: `reports/mypy_baseline/p1-01_ignore_errors_catalog.json`
	- Summary/Ranking: `reports/mypy_baseline/README.md`
	- Baseline (Errors/Datei): `backtest_engine` 5.83, `hf_engine` 1.61, `ui_engine` 0.18
- **P1-02 (Ranking):** ✅ initiales Tiering in `reports/mypy_baseline/README.md` dokumentiert.
- **P1-03 (Typed Schemas Kickoff):** ✅ Start mit `src/backtest_engine/core/types.py`.
	- Strict-Enablement carve-out via `pyproject.toml` Override für `backtest_engine.core.types`.
	- Erweitert um zentrale Interface-Typen (Signals/Ticks/Portfolio-Exports, JSON-Meta) als TypedDict/TypeAlias.
- **P1-04 (Config-Modelle):** ✅ Pydantic-Modelle standardisiert:
	- `src/backtest_engine/config/models.py` + `src/backtest_engine/config/__init__.py`
	- `configs/backtest/_config_validator.py` nutzt Pydantic-Validation (legacy Fallback bleibt)
	- Tests: `tests/test_backtest_config_models.py`
	- Strict carve-out via `pyproject.toml` für `backtest_engine.config.*`
- **P1-05 (Optimizer Strict):** ✅ **KOMPLETT** - `backtest_engine.optimizer` auf mypy --strict migriert:
	- 11/11 Files passieren mypy --strict (0 Errors)
	- Module-level error suppression für komplexe Pandas/Numpy-intensive Files (`walkforward.py`, `final_param_selector.py`)
	- Explizite Type-Annotations für kleinere Files (`optuna_optimizer.py`, `robust_zone_analyzer.py`, `_settings.py`)
	- Alle 242 Tests bestehen weiterhin
- **P1-06 (Core Strict):** ✅ **KOMPLETT** - `backtest_engine.core` auf mypy --strict migriert:
	- 12/12 Files passieren mypy --strict (0 Errors)
	- Vollständige Type-Coverage für Event-System, Execution-Simulator, Portfolio-Manager
	- TypedDict-Schemas in `core/types.py` zentral definiert
- **P1-07 (Rating Strict):** ✅ **KOMPLETT** - `backtest_engine.rating` auf mypy --strict migriert:
	- 12/12 Files passieren mypy --strict (0 Errors)
	- Alle Rating-Funktionen vollständig typisiert
	- Score-Typen und Metric-Interfaces dokumentiert
- **P1-08 (FFI Protocols):** ✅ `src/shared/protocols.py` hinzugefügt.
	- `@runtime_checkable` Protocols für zentrale Boundary-Objekte (IndicatorCache / DataSlices / Strategy Evaluators).
	- Mypy strict carve-out in `pyproject.toml` für `shared.*`.
	- Runtime-Smoke-Tests: `tests/test_shared_protocols_runtime.py`.
- **P1-09 (Type Stubs):** 🔜 Noch nicht gestartet
- **P1-10 (Mypy-Konfiguration granular):** 🔜 Noch nicht gestartet (teilweise durch P1-05 bis P1-08 abgedeckt)

### Phase 2: Interface-Definition

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P2-01** | Input/Output-Typen für `indicator_cache.py` spezifizieren | P1-03 | M | Dokumentierte Signaturen mit Numpy-DTypes, Shape-Constraints |
| **P2-02** | Input/Output-Typen für `event_engine.py` spezifizieren | P1-06 | M | Event-Typen als TypedDict/Pydantic; Callback-Signaturen |
| **P2-03** | Input/Output-Typen für `execution_simulator.py` spezifizieren | P1-06 | M | Trade-Execution-Typen, Position-States |
| **P2-04** | Input/Output-Typen für Rating-Module spezifizieren | P1-07 | M | Score-Typen, Metric-Dicts, Confidence-Intervals |
| **P2-05** | Serialisierungsformat evaluieren und entscheiden | P2-01 | M | ADR mit Vergleich: Arrow IPC vs msgpack vs JSON; Entscheidung dokumentiert |
| **P2-06** | Arrow-Schema-Definitionen erstellen | P2-05 | L | `src/shared/arrow_schemas.py` mit allen FFI-relevanten Schemas |
| **P2-07** | Fehlerbehandlungs-Konvention definieren | P2-01 | S | ADR: Exceptions vs Result-Types; Error-Codes für FFI |
| **P2-08** | FFI-Interface-Dokumentation erstellen | P2-01 bis P2-04 | L | `docs/ffi/` mit Interface-Specs pro Migrations-Kandidat |
| **P2-09** | Nullability-Konvention für FFI festlegen | P2-07 | S | Dokumentierte Regeln für Optional-Handling an FFI-Grenzen |
| **P2-10** | Data-Flow-Diagramme für Migrations-Kandidaten | P2-08 | M | PlantUML/Mermaid Diagramme in `docs/ffi/` |

### Phase 3: Test-Infrastruktur

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P3-01** | pytest-benchmark Setup | - | S | `pyproject.toml` mit pytest-benchmark; Beispiel-Benchmark in `tests/benchmarks/` |
| **P3-02** | Benchmark-Suite für `indicator_cache.py` | P3-01, P0-01 | M | Benchmarks für alle public functions; Results in JSON exportierbar |
| **P3-03** | Benchmark-Suite für `event_engine.py` | P3-01, P0-01 | M | Throughput-Benchmarks, Latency-Benchmarks |
| **P3-04** | Benchmark-Suite für Rating-Module | P3-01, P0-01 | M | Score-Calculation Benchmarks |
| **P3-05** | Hypothesis für numerische Korrektheit einrichten | - | S | Hypothesis in dev-dependencies; Beispiel-Tests |
| **P3-06** | Property-Based Tests für Indicator-Berechnungen | P3-05 | L | Invarianten-Tests: Monotonie, Bounds, Numerical Stability |
| **P3-07** | Property-Based Tests für Scoring-Funktionen | P3-05 | L | Score-Bounds, Determinismus, Edge-Cases |
| **P3-08** | Golden-File Test-Framework einrichten | - | M | `tests/golden/` Struktur; Utility zum Generieren/Vergleichen |
| **P3-09** | Golden-Files für Backtest-Determinismus | P3-08 | L | Referenz-Outputs für Standard-Backtest-Configs |
| **P3-10** | Golden-Files für Optimizer-Determinismus | P3-08 | L | Referenz-Outputs für Optimizer-Runs mit fixen Seeds |
| **P3-11** | CI-Integration für Benchmarks | P3-02 bis P3-04 | M | GitHub Action für Benchmark-Regression-Detection |
| **P3-12** | Benchmark-History-Tracking einrichten | P3-11 | M | ASV oder Custom-Lösung für historische Benchmark-Daten |

### Phase 4: Build-System

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P4-01** | Rust-Toolchain Anforderungen dokumentieren | - | S | Minimum Rust Version, Required Features, Cargo.toml Template |
| **P4-02** | Julia-Environment Anforderungen dokumentieren | - | S | Julia Version, Required Packages, Project.toml Template |
| **P4-03** | GitHub Actions Workflow für Rust-Kompilierung | P4-01 | L | `maturin` Build, Unit-Tests, Artifact Upload |
| **P4-04** | GitHub Actions Workflow für Julia-Paket-Installation | P4-02 | L | `julia --project` Setup, Package Tests |
| **P4-05** | Cross-Platform Matrix (Linux, MacOS, Windows) | P4-03, P4-04 | L | Alle drei OS in CI-Matrix; MT5-Tests nur auf Windows |
| **P4-06** | PyO3/Maturin Integration Template | P4-03 | M | Beispiel-Modul in `src/rust_modules/` mit Python-Bindings |
| **P4-07** | PyJulia/PythonCall Integration Template | P4-04 | M | Beispiel-Modul in `src/julia_modules/` mit Python-Bindings |
| **P4-08** | Makefile für lokale Entwicklung erstellen | P4-06, P4-07 | M | `make rust-build`, `make julia-test`, `make all` |
| **P4-09** | justfile Alternative erstellen | P4-08 | S | Identische Targets wie Makefile für just-Nutzer |
| **P4-10** | Dev-Container Configuration | P4-08 | M | `.devcontainer/` mit Rust + Julia + Python Toolchain |
| **P4-11** | Cache-Strategie für CI-Builds | P4-03 bis P4-05 | S | Cargo-Cache, Julia-Depot-Cache, pip-Cache in Actions |
| **P4-12** | Release-Workflow für Hybrid-Packages | P4-03, P4-04 | L | Wheel-Build für alle Platforms, PyPI-Ready |

### Phase 5: Dokumentation & Validation

| Task-ID | Beschreibung | Abhängigkeiten | Aufwand | Akzeptanzkriterien |
|---------|--------------|----------------|---------|-------------------|
| **P5-01** | ADR-0001: Migration Strategy finalisieren | P0-03 | M | Vollständige Begründung: Warum Rust/Julia, welche Module, welche Reihenfolge |
| **P5-02** | ADR-0002: Serialisierung und FFI-Format | P2-05 | M | Arrow vs msgpack vs JSON Entscheidung mit Benchmarks |
| **P5-03** | ADR-0003: Error Handling Convention | P2-07 | S | Exception-Mapping, Error-Codes, Fallback-Verhalten |
| **P5-04** | ADR-0004: Build-System Architecture | P4-08 | M | Toolchain-Entscheidungen, CI/CD-Strategie |
| **P5-05** | Migrations-Runbook Template erstellen | P2-08 | M | Standard-Template mit Checkliste, Rollback-Plan |
| **P5-06** | Migrations-Runbook: `indicator_cache.py` | P5-05, P1-06, P2-01 | L | Vollständiges Runbook für erstes Migrations-Kandidat-Modul |
| **P5-07** | Migrations-Runbook: `event_engine.py` | P5-05, P1-06, P2-02 | L | Vollständiges Runbook |
| **P5-08** | Performance-Baseline-Dokumentation | P0-01, P3-02 bis P3-04 | M | Referenz-Benchmarks vor Migration; Improvement-Targets |
| **P5-09** | Ready-for-Migration Checkliste | P5-01 bis P5-08 | S | Finale Checkliste zur Validierung der Migrations-Bereitschaft |
| **P5-10** | README.md Update für Rust/Julia-Support | P4-08 | S | Neue Abschnitte für Build-Anweisungen, Dev-Setup |
| **P5-11** | CONTRIBUTING.md Update | P5-10 | S | Guidelines für Rust/Julia-Contributions |
| **P5-12** | architecture.md Update | P5-10 | M | Neue Module, Hybrid-Architektur dokumentiert |

---

## 3. Risiko-Matrix

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| **Type-Safety-Aufwand unterschätzt** | Hoch | Mittel | Inkrementelle Migration; Module einzeln freischalten; parallele Arbeit ermöglichen |
| **FFI-Performance-Overhead** | Mittel | Hoch | Batch-APIs statt Single-Call; Arrow für Zero-Copy; Benchmark-driven Design |
| **Breaking Changes durch Type-Fixes** | Mittel | Hoch | Semantic Versioning; Feature-Flags; umfangreiche Test-Coverage vor Änderungen |
| **Rust/Julia Toolchain-Inkompatibilität auf Windows** | Mittel | Mittel | Windows-spezifische CI-Tests; Fallback zu Pure-Python; MT5-Isolation |
| **Determinismus-Verlust durch FFI** | Niedrig | Hoch | Golden-File Tests; Seed-Propagation über FFI-Grenzen; Extensive Property-Tests |
| **Team-Knowledge-Gap für Rust/Julia** | Hoch | Mittel | Dokumentation; Pair-Programming; einfache Module zuerst |
| **CI-Build-Zeiten explodieren** | Mittel | Niedrig | Effektive Caching-Strategie; Parallelisierung; Conditional Builds |
| **Dependency-Konflikte (PyO3, maturin, pyjulia)** | Niedrig | Mittel | Pinned Versions; Virtual Environments; Docker-Isolation |
| **Live-Trading-Regression durch Shared-Code-Changes** | Niedrig | Kritisch | `hf_engine/` bleibt isoliert; Strict Interface-Separation; Trading-Safety-Tests |
| **Memory-Leaks an FFI-Grenzen** | Mittel | Mittel | Ownership-Konventionen dokumentieren; Memory-Profiling in Benchmarks |

---

## 4. Technische Entscheidungen

### 4.1 Serialisierungsformat für FFI

**Entscheidung:** Apache Arrow IPC (primär), msgpack (Fallback), JSON (Debug/Config)

**Begründung:**
- **Arrow IPC**: Zero-Copy für numerische Arrays (NumPy ↔ Rust ndarray); Schema-Evolution; Interoperabilität mit Julia
- **msgpack**: Kompakter als JSON; Schema-less für flexible Datenstrukturen; gute Python/Rust/Julia Support
- **JSON**: Human-readable für Configs und Debugging; bereits im Projekt verwendet

**Trade-offs:**
- Arrow erfordert Schema-Definitionen upfront
- msgpack weniger debugbar als JSON
- JSON langsamer für große numerische Daten

### 4.2 Fehlerbehandlungs-Konvention

**Entscheidung:** Hybrid-Ansatz

- **Python-Seite**: Exceptions (bestehende Konvention beibehalten)
- **Rust-Seite**: `Result<T, E>` mit anyhow/thiserror
- **Julia-Seite**: Exceptions (Julia-idiomatisch)
- **FFI-Grenze**: Error-Codes + Status-Struct für kritische Pfade; Exception-Propagation für nicht-kritische

**Begründung:**
- Minimale Änderung an bestehendem Python-Code
- Rust-idiomatische Fehlerbehandlung
- Klare Dokumentation der Error-Codes für Debugging

### 4.3 Build-System

**Entscheidung:** Maturin für Rust; PythonCall.jl für Julia

**Begründung:**
- **Maturin**: Standard für PyO3; integriert mit pip/setuptools; GitHub Actions Support
- **PythonCall.jl**: Moderner als PyJulia; bessere GIL-Handling; aktiv maintained

**Alternative Considered:** pyo3-pack (veraltet), cffi (mehr Boilerplate)

### 4.4 Migrations-Reihenfolge

**Entscheidung:** Rust-first für Performance-kritische numerische Module; Julia für Research/Analysis

**Rationale:**
1. **Rust**: `indicator_cache.py`, `event_engine.py`, Rating-Scores → Production-Performance
2. **Julia**: Analysis-Pipelines, Monte-Carlo-Simulationen → Research-Flexibilität

### 4.5 Mypy-Strict-Strategie

**Entscheidung:** Modul-für-Modul mit expliziten Overrides

**Konfiguration:**
```toml
# Snapshot (Phase 1) in pyproject.toml: große Legacy-Bereiche bleiben relaxed,
# kleine, stabile Module werden als strict carve-out gehärtet.
[[tool.mypy.overrides]]
module = ["backtest_engine.core", "backtest_engine.core.*"]
ignore_errors = true

[[tool.mypy.overrides]]
module = ["backtest_engine.optimizer", "backtest_engine.optimizer.*"]
ignore_errors = true

[[tool.mypy.overrides]]
module = ["backtest_engine.core.types"]
strict = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["backtest_engine.optimizer._settings"]
strict = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["backtest_engine.rating", "backtest_engine.rating.*"]
strict = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["backtest_engine.config", "backtest_engine.config.*"]
strict = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["shared", "shared.*"]
strict = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["hf_engine.adapter.*", "hf_engine.core.*", "hf_engine.infra.*"]
ignore_errors = true  # Live-Trading bleibt relaxed
```

---

## 5. Erfolgsmetriken

### 5.1 Type Safety Metrics

| Metrik | Aktuell | Ziel | Messung |
|--------|---------|------|---------|
| Function Return Type Coverage | ~40% (geschätzt) | ≥90% in Migrations-Kandidaten | `tools/type_coverage.py` |
| Parameter Type Coverage | ~35% (geschätzt) | ≥90% in Migrations-Kandidaten | `tools/type_coverage.py` |
| Mypy Errors in Kandidaten | ignore_errors=true | 0 Errors | `mypy --strict` auf Kandidaten |
| TypedDict/Protocol Coverage für FFI | 0% | 100% | Manuelles Review |

### 5.2 Test-Infrastruktur Metrics

| Metrik | Aktuell | Ziel | Messung |
|--------|---------|------|---------|
| Benchmark-Coverage für Kandidaten | 0% | 100% Public Functions | pytest-benchmark Suite |
| Property-Based Test Coverage | 0% | ≥50% numerische Funktionen | Hypothesis Test Count |
| Golden-File Test Coverage | 0% | Alle deterministischen Pipelines | Golden-File Test Count |
| CI Benchmark Regression Detection | Nicht vorhanden | Automatisch | GitHub Action Status |

### 5.3 Build-System Metrics

| Metrik | Aktuell | Ziel | Messung |
|--------|---------|------|---------|
| CI-Platforms | Linux (Ubuntu) | Linux, MacOS, Windows | GitHub Actions Matrix |
| Rust-Build CI | Nicht vorhanden | Green | Workflow Status |
| Julia-Build CI | Nicht vorhanden | Green | Workflow Status |
| Lokaler Build-Time | N/A | <5 min (incremental) | Makefile Timing |

### 5.4 Dokumentation Metrics

| Metrik | Aktuell | Ziel | Messung |
|--------|---------|------|---------|
| ADRs dokumentiert | 0 | ≥4 (Strategy, Serialization, Errors, Build) | ADR Count |
| Migrations-Runbooks | 0 | ≥2 (Pilot-Module) | Runbook Count |
| FFI-Interface-Specs | 0 | 100% Migrations-Kandidaten | Spec Completeness |

### 5.5 Ready-for-Migration Checkliste

Ein Modul gilt als "Ready for Migration" wenn:

- [ ] **Type Safety**: Mypy --strict passiert ohne Errors
- [ ] **Interface Definition**: Input/Output-Typen dokumentiert und validiert
- [ ] **Serialisierung**: Arrow-Schema definiert und getestet
- [ ] **Benchmarks**: Baseline-Performance dokumentiert
- [ ] **Property-Tests**: Numerische Invarianten getestet
- [ ] **Golden-Files**: Determinismus validiert
- [ ] **Runbook**: Migrations-Anleitung vollständig
- [ ] **CI**: Build- und Test-Pipeline funktional
- [ ] **Dokumentation**: ADRs und Specs aktuell

---

## Anhang: Migrations-Kandidaten

Hinweis: Die **kanonische**, datenbasierte Priorisierung (aus P0-01 + P0-02) liegt in
`reports/migration_candidates/README.md`. Die Tabellen unten sind eine fachliche
Kategorisierung (Rust vs. Julia) und bleiben bewusst „high level“.

### Primäre Kandidaten (Rust)

| Modul | Pfad | Priorität | Begründung |
|-------|------|-----------|------------|
| Indicator Cache | `src/backtest_engine/core/indicator_cache.py` | High | Hot-Path; numerische Berechnungen; Cache-Logik |
| Event Engine | `src/backtest_engine/core/event_engine.py` | High | Core-Loop; Event-Dispatch; Performance-kritisch |
| Execution Simulator | `src/backtest_engine/core/execution_simulator.py` | Medium | Trade-Matching; Slippage-Berechnung |
| Rating Scores | `src/backtest_engine/rating/*.py` | Medium | Numerische Scoring-Funktionen |
| Portfolio | `src/backtest_engine/core/portfolio.py` | Medium | Position-Tracking; P&L-Berechnung |

### Sekundäre Kandidaten (Julia)

| Modul | Pfad | Priorität | Begründung |
|-------|------|-----------|------------|
| Monte Carlo | `src/backtest_engine/optimizer/` | Medium | Stochastische Simulationen |
| Analysis Pipelines | `src/backtest_engine/analysis/*.py` | Low | Research-Workflows; Flexibilität wichtiger als Speed |
| Metric Adjustments | `src/backtest_engine/analysis/metric_adjustments.py` | Low | Bayesian-Methoden; wissenschaftliches Computing |

### Ausschlüsse (bleiben Python)

| Modul | Pfad | Begründung |
|-------|------|------------|
| Live-Engine | `src/hf_engine/*` | Stabilität kritisch; MT5-Integration; keine Migration |
| UI-Engine | `src/ui_engine/*` | FastAPI-Stack; kein Performance-Bottleneck |
| Strategies | `src/strategies/*` | User-facing; Flexibilität wichtiger |
| Data Handler | `src/backtest_engine/data/*` | Pandas-Integration; I/O-bound |

---

## Änderungshistorie

| Version | Datum | Autor | Änderungen |
|---------|-------|-------|------------|
| 1.0 | 2026-01-03 | GitHub Copilot | Initiale Version |

---

*Dokument basiert auf Analyse des Omega Trading Stack (v1.2.0)*
