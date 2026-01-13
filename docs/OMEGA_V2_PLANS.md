# Omega V2 – Plan-Übersicht & Fortschritt

> **Status**: Aktiv  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Zentrale Übersicht aller OMEGA_V2 Planungsdokumente mit Fortschritts-Tracking

---

## Fortschritts-Tracker

| Status | Bedeutung | Anzahl |
|--------|-----------|--------|
| ✅ | Existiert und vollständig | 5 |
| 🟡 | Existiert, offene Punkte | 8 |
| 🔲 | Geplant, noch nicht erstellt | 1 |

**Gesamt: 14 Pläne** | **Fortschritt: 13/14 erstellt (93%)**

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

### ✅ OMEGA_V2_DATA_GOVERNANCE_PLAN.md
**Zweck**: Normative Data-Quality-Policies (Alignment/Gaps/Duplicates), News-Governance (Parquet), sowie reproduzierbare Dataset-Snapshots (Hashes/Manifests).

---

### 🟡 OMEGA_V2_EXECUTION_MODEL_PLAN.md
**Zweck**: Bid/Ask-Regeln, Fill-Algorithmen, Intrabar-Tie-Breaks, Stop/TP-Prioritäten, Limit/Market-Semantik, Slippage/Fees und deterministische Ausführung.

**Offene Punkte**:
- [ ] Exakte Feldquelle/-name für „minimale SL-Distanz“ pro Symbol festlegen (z.B. Erweiterung `configs/symbol_specs.yaml` vs. Execution-Costs-Config)
- [ ] Finales Set an `reason`-Werten für Exits im Output-Contract bestätigen (Enum vs. freie Strings + `meta`)

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
- [ ] Golden-File Vergleichsregeln (Float-Toleranzen/Normalisierung) konkretisieren

---

### 🟡 OMEGA_V2_METRICS_DEFINITION_PLAN.md
**Zweck**: Normative Definition aller Performance-Metriken und Score-Keys (Units/Domain/Quelle/Edge-Cases) inkl. Policy für Rundung und Output-Ort (Single-Run vs. Optimizer).

**Offene Punkte**:
- [ ] Sharpe/Sortino: exakte Formeln + Annualisierung/Frequenz (Trade-R vs. Daily Equity) konkretisieren
- [ ] Optimizer-/Rating-Aggregate-Contract für Robustness-Scores spezifizieren (separates Artefakt)

---

### ✅ OMEGA_V2_CONFIG_SCHEMA_PLAN.md
**Zweck**: Normatives JSON-Schema für Backtest-Konfiguration mit Pflichtfeldern, Defaults, Ranges, Validierungsregeln und Migrations-Guide.

---

### ✅ OMEGA_V2_TECH_STACK_PLAN.md
**Zweck**: Version-Pinning (Rust Toolchain, arrow/parquet, PyO3/maturin), Build-Matrix, OS-Support sowie normierte Logging-/RNG-/Error-Contracts.


### ✅ OMEGA_V2_CI_WORKFLOW_PLAN.md
**Zweck**: GitHub Actions Workflow für Omega V2: Python/Rust Checks, Full Wheel Build-Matrix (maturin), Security Scans (CodeQL/Dependency Review), Artefakte und Release-Assets.

---

### 🟡 OMEGA_V2_TESTING_VALIDATION_PLAN.md
**Zweck**: Testpyramide (unit/property/integration/contract), deterministische Fixtures, Golden-File Regeln, Paritäts-Tests gegen V1 sowie CI-Integration (PR-Gates vs nightly).

**Offene Punkte**:
- [ ] Sharpe/Sortino: Annualisierung/Frequenz (Trade-R vs. Daily Equity) finalisieren
- [ ] MVP-Gate-Kriterien für „ready-to-merge“ (zusätzlicher Gate-Mechanismus) spezifizieren (explizit offen)

---

### ✅ OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md
**Zweck**: Normative Spezifikation der Observability-Strategie für Omega V2 (Logging/Tracing via tracing, Profiling via flamegraph/pprof, Performance-Counter, CI-Integration, Determinismus-Guardrails).

---

## Geplante Pläne
---

### 🔲 OMEGA_V2_AGENT_INSTRUCTION_PLAN.md
**Zweck**: Contributor-Guidelines für AI-Agenten und Entwickler: Crate-Boundaries, Code-Style, Error-Handling-Patterns, Review-Checklist und PR-DoD.

**Priorität**: 🟢 Niedrig (kann iterativ wachsen)

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
| **Betrieb** | OBSERVABILITY_PROFILING_PLAN | CI_WORKFLOW_PLAN |

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 12.01.2026 | Initiale Version mit 4 existierenden und 10 geplanten Plänen |
| 1.1 | 12.01.2026 | Output-Contract-Plan erstellt und Tracker aktualisiert |
| 1.2 | 12.01.2026 | Execution-Model-Plan erstellt, Querverweise/Tracker aktualisiert |
| 1.3 | 12.01.2026 | Metrics-Definition-Plan erstellt, Querverweise/Tracker aktualisiert |
| 1.4 | 12.01.2026 | Tech-Stack-Plan erstellt und Tracker aktualisiert |
| 1.5 | 13.01.2026 | CI-Workflow-Plan erstellt und Tracker aktualisiert |
| 1.6 | 13.01.2026 | Data-Governance-Plan erstellt, Tracker aktualisiert |
| 1.7 | 13.01.2026 | Observability-Profiling-Plan erstellt, Tracker aktualisiert |
| 1.8 | 13.01.2026 | Testing-Validation-Plan erstellt, Tracker aktualisiert |

