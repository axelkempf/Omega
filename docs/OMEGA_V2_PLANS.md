# Omega V2 – Plan-Übersicht & Fortschritt

> **Status**: Aktiv  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Zentrale Übersicht aller OMEGA_V2 Planungsdokumente mit Fortschritts-Tracking

---

## Fortschritts-Tracker

| Status | Bedeutung | Anzahl |
|--------|-----------|--------|
| ✅ | Existiert und vollständig | 1 |
| 🟡 | Existiert, offene Punkte | 5 |
| 🔲 | Geplant, noch nicht erstellt | 8 |

**Gesamt: 14 Pläne** | **Fortschritt: 6/14 erstellt (43%)**

---

## Existierende Pläne

### 🟡 OMEGA_V2_VISION_PLAN.md
**Zweck**: Definiert das strategische Zielbild, die Problemanalyse des V1-Systems und messbare Erfolgskriterien für die V2-Migration.

**Offene Punkte**:
- [ ] Exakte Paritätstoleranz (Entry/Exit-Events vs. PnL/Fees) spezifizieren
- [ ] 6 Szenarien der Mean-Reversion-Strategie dokumentieren
- [ ] DEV/PROD-Mode Policy für Optimizer finalisieren

---

### 🟡 OMEGA_V2_ARCHITECTURE_PLAN.md
**Zweck**: Übergeordneter Blueprint mit Crate-Struktur, Modul-Abhängigkeiten, Systemregeln und FFI-Boundary-Definition.

**Offene Punkte**:
- [ ] T5: HTF-Datenquelle (separate Parquets vs. Aggregation) entscheiden
- [ ] Logging-Strategie (Rust tracing vs. Python) finalisieren
- [ ] Parallelisierung (rayon) Policy nach Parität festlegen

---

### 🟡 OMEGA_V2_DATA_FLOW_PLAN.md
**Zweck**: Vollständige Spezifikation des Datenflusses von Config-Input über Rust-Engine bis Result-Output inkl. Validierungs-Checkpoints.

**Offene Punkte**:
- [ ] Alignment-Loss Schwelle (Warning vs. Abort) definieren
- [ ] Gap-Policy (akzeptieren/pausieren/abbrechen) festlegen
- [ ] HTF `htf_idx-1` Edge-Case bei frühen Bars klären
- [ ] Timestamp-Duplikat-Handling (abort/deduplizieren/aggregieren)

---

### 🟡 OMEGA_V2_MODULE_STRUCTURE_PLAN.md
**Zweck**: Detaillierte Ordner-, Datei- und Modul-Struktur des Rust-Workspace mit Abhängigkeits-Matrix und Test-Strategie pro Crate.

**Offene Punkte**:
- [ ] Test-Fixtures Strategie (Repo vs. generiert) entscheiden
- [ ] `alt_data.rs` / `news.rs` Modul-Details spezifizieren
- [ ] Property-Test Coverage-Ziele definieren

---

### 🟡 OMEGA_V2_OUTPUT_CONTRACT_PLAN.md
**Zweck**: Exakte Artefakt-Spezifikation für `trades.json`, `equity.csv`, `metrics.json`, `meta.json` inkl. Feldnamen, Typen, Einheiten, Zeit-Contract und Output-Pfad.

**Offene Punkte**:
- [ ] MVP-Kernmetriken-Set in `metrics.json` finalisieren (Keys + Definitionen)
- [ ] Golden-File Vergleichsregeln (Float-Toleranzen/Normalisierung) konkretisieren

---

### ✅ OMEGA_V2_CONFIG_SCHEMA_PLAN.md
**Zweck**: Normatives JSON-Schema für Backtest-Konfiguration mit Pflichtfeldern, Defaults, Ranges, Validierungsregeln und Migrations-Guide.

---

## Geplante Pläne


---

### 🔲 OMEGA_V2_EXECUTION_MODEL_PLAN.md
**Zweck**: Bid/Ask-Regeln, Fill-Algorithmen, Intrabar-Tie-Breaks, Stop/TP-Prioritäten, Limit/Market-Semantik, Netting/Hedging und Margin-Modell.

**Priorität**: 🔴 Hoch (höchstes Korrektheits-Risiko)

---

### 🔲 OMEGA_V2_METRICS_DEFINITION_PLAN.md
**Zweck**: Formale Definition aller Performance-Metriken (Sharpe, Sortino, Drawdown, etc.) inkl. Inputs, Sampling-Frequenz und Timeframe-Skalierung.

**Priorität**: 🟡 Mittel (kann parallel zu Core entwickelt werden)

---

### 🔲 OMEGA_V2_CI_WORKFLOW_PLAN.md
**Zweck**: GitHub Actions Workflow für Rust+Python CI: fmt/clippy/tests, maturin build, artifact checks, golden-file regression und Benchmarks.

**Priorität**: 🟡 Mittel (wichtig ab Phase 2)

---

### 🔲 OMEGA_V2_TECH_STACK_PLAN.md
**Zweck**: Version-Pinning (Rust toolchain, arrow/parquet, pyo3/maturin), Build-Matrix, OS-Support und Rationale für Technologie-Entscheidungen.

**Priorität**: 🟡 Mittel (vor Implementierungsstart)

---

### 🔲 OMEGA_V2_TESTING_VALIDATION_PLAN.md
**Zweck**: Testpyramide (unit/property/integration), deterministische Fixtures, Paritäts-Tests gegen V1, Fuzzing-Strategie und Lookahead-Tests.

**Priorität**: 🟡 Mittel (parallel zur Implementierung)

---

### 🔲 OMEGA_V2_AGENT_INSTRUCTION_PLAN.md
**Zweck**: Contributor-Guidelines für AI-Agenten und Entwickler: Crate-Boundaries, Code-Style, Error-Handling-Patterns, Review-Checklist und PR-DoD.

**Priorität**: 🟢 Niedrig (kann iterativ wachsen)

---

### 🔲 OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md
**Zweck**: Tracing/Telemetry-Strategie, Performance-Counter, Flamegraph-Integration und Logging-Policy ohne Python-Callbacks.

**Priorität**: 🟢 Niedrig (nach MVP)

---

### 🔲 OMEGA_V2_DATA_GOVERNANCE_PLAN.md
**Zweck**: Data-Quality-Thresholds, Gap-Policy, Alignment-Loss-Policy, Naming-Conventions und reproduzierbare Dataset-Snapshots.

**Priorität**: 🟡 Mittel (vor Production-Daten)

---

## Querverweise

Alle Pläne befinden sich in `docs/` und folgen der Namenskonvention `OMEGA_V2_<TOPIC>_PLAN.md`.

| Bereich | Primärer Plan | Ergänzende Pläne |
|---------|--------------|------------------|
| **Vision & Ziele** | VISION_PLAN | - |
| **Architektur** | ARCHITECTURE_PLAN | MODULE_STRUCTURE_PLAN |
| **Daten** | DATA_FLOW_PLAN | DATA_GOVERNANCE_PLAN |
| **Execution** | EXECUTION_MODEL_PLAN | METRICS_DEFINITION_PLAN |
| **Config/Output** | CONFIG_SCHEMA_PLAN | OUTPUT_CONTRACT_PLAN |
| **Qualität** | TESTING_VALIDATION_PLAN | CI_WORKFLOW_PLAN |
| **Entwicklung** | AGENT_INSTRUCTION_PLAN | TECH_STACK_PLAN |
| **Betrieb** | OBSERVABILITY_PROFILING_PLAN | - |

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 12.01.2026 | Initiale Version mit 4 existierenden und 10 geplanten Plänen |
| 1.1 | 12.01.2026 | Output-Contract-Plan erstellt und Tracker aktualisiert |

