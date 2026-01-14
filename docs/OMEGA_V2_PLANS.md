# Omega V2 – Plan-Übersicht & Fortschritt

> **Status**: Aktiv  
> **Erstellt**: 12. Januar 2026  
> **Letzte Analyse**: 14. Januar 2026  
> **Zweck**: Zentrale Übersicht aller OMEGA_V2 Planungsdokumente mit Fortschritts-Tracking und konsolidierten offenen Punkten

---

## Existierende Pläne

### ✅ OMEGA_V2_VISION_PLAN.md
**Zweck**: Definiert das strategische Zielbild, die Problemanalyse des V1-Systems und messbare Erfolgskriterien für die V2-Migration.

**Vollständigkeit**: 100% - offene Punkte entschieden und dokumentiert

---

### ✅ OMEGA_V2_ARCHITECTURE_PLAN.md
**Zweck**: Übergeordneter Blueprint mit Crate-Struktur, Modul-Abhängigkeiten, Systemregeln und FFI-Boundary-Definition.

**Vollständigkeit**: 100% - Architektur inkl. Parallelisierung-Policy normiert

---

### ✅ OMEGA_V2_DATA_FLOW_PLAN.md
**Zweck**: Vollständige Spezifikation des Datenflusses von Config-Input über Rust-Engine bis Result-Output inkl. Validierungs-Checkpoints.

**Vollständigkeit**: 100% - Alle Data-Flow Edge-Cases normiert

---

### ✅ OMEGA_V2_DATA_GOVERNANCE_PLAN.md
**Zweck**: Normative Data-Quality-Policies (Alignment/Gaps/Duplicates), News-Governance (Parquet), sowie reproduzierbare Dataset-Snapshots (Hashes/Manifests).

**Status**: Vollständig - Alle Data-Quality-Regeln normativ festgelegt

---

### ✅ OMEGA_V2_EXECUTION_MODEL_PLAN.md
**Zweck**: Bid/Ask-Regeln, Fill-Algorithmen, Intrabar-Tie-Breaks, Stop/TP-Prioritäten, Limit/Market-Semantik, Slippage/Fees und deterministische Ausführung.

**Vollständigkeit**: 100% - Execution-Semantik und Exit-Reason Policy normiert

---

### ✅ OMEGA_V2_MODULE_STRUCTURE_PLAN.md
**Zweck**: Detaillierte Ordner-, Datei- und Modul-Struktur des Rust-Workspace mit Abhängigkeits-Matrix und Test-Strategie pro Crate.

**Vollständigkeit**: 100% - Struktur, News-Modul und Test-Policies normiert

---

### ✅ OMEGA_V2_INDICATOR_CACHE__PLAN.md
**Zweck**: Normative Spezifikation des Indikator-Cache (Multi-TF, Stepwise-Semantik, Cache-Keys, Missing-Values/NaN, V1-Parität) inkl. vollständigem Indikator-Inventar für MRZ Szenarien 1–6.

**Vollständigkeit**: 100% - Scope und Semantik normiert, MRZ-Indikator-Inventar vollständig

---

### ✅ OMEGA_V2_STRATEGIES_PLAN.md
**Zweck**: Normative Spezifikation der Strategie-Schicht (MVP: Mean Reversion Z-Score) inkl. Szenarien 1–6, Guards/Filter (Sessions/News/Cooldown), benötigter Indikatoren, Modul-Zerlegung und Paritätsanforderungen zu V1.

**Status**: Vollständig - MRZ Szenarien/Guards normiert; Position Manager explizit als separater Plan vorgesehen

---

### ✅ OMEGA_V2_TRADE_MANAGER_PLAN.md
**Zweck**: Normative Spezifikation des Trade-/Position-Management-Layers (Rules → Actions). MVP: V1-Parität für MaxHoldingTime + klare Close-Reasons (u.a. `timeout`), deterministische Stop-Update-Policy.

**Status**: Vollständig - MVP-Contract + Erweiterungspfade dokumentiert

---

### ✅ OMEGA_V2_OUTPUT_CONTRACT_PLAN.md
**Zweck**: Exakte Artefakt-Spezifikation für `trades.json`, `equity.csv`, `metrics.json`, `meta.json` inkl. Feldnamen, Typen, Einheiten, Zeit-Contract und Output-Pfad.

**Vollständigkeit**: 100% - Output-Contract inkl. Golden/Profiling-Referenz normiert

---

### ✅ OMEGA_V2_METRICS_DEFINITION_PLAN.md
**Zweck**: Normative Definition aller Performance-Metriken und Score-Keys (Units/Domain/Quelle/Edge-Cases) inkl. Policy für Rundung und Output-Ort (Single-Run vs. Optimizer).

**Vollständigkeit**: 100% - Single-Run Keys + Optimizer-Aggregate-Contract normiert

---

### ✅ OMEGA_V2_CONFIG_SCHEMA_PLAN.md
**Zweck**: Normatives JSON-Schema für Backtest-Konfiguration mit Pflichtfeldern, Defaults, Ranges, Validierungsregeln und Migrations-Guide.

**Status**: Vollständig - Schema v2 normativ definiert, Validierungsregeln mit jsonschema, Migrations-Guide enthalten

---

### ✅ OMEGA_V2_TECH_STACK_PLAN.md
**Zweck**: Version-Pinning (Rust Toolchain, arrow/parquet, PyO3/maturin), Build-Matrix, OS-Support sowie normierte Logging-/RNG-/Error-Contracts.

**Status**: Vollständig - Alle Toolchain-Entscheidungen festgelegt, Build-Matrix definiert


### ✅ OMEGA_V2_CI_WORKFLOW_PLAN.md
**Zweck**: GitHub Actions Workflow für Omega V2: Python/Rust Checks, Full Wheel Build-Matrix (maturin), Security Scans (CodeQL/Dependency Review), Artefakte und Release-Assets.

**Status**: Vollständig - 11 Akzeptanzkriterien definiert, Tag-Schema `omega-v2-v<MAJOR>.<MINOR>.<PATCH>`

---

### ✅ OMEGA_V2_TESTING_VALIDATION_PLAN.md
**Zweck**: Testpyramide (unit/property/integration/contract), deterministische Fixtures, Golden-File Regeln, Paritäts-Tests gegen V1 sowie CI-Integration (PR-Gates vs nightly).

**Vollständigkeit**: 100% - Teststrategie + Gate-Kategorisierung vollständig

---

### ✅ OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md
**Zweck**: Normative Spezifikation der Observability-Strategie für Omega V2 (Logging/Tracing via tracing, Profiling via flamegraph/pprof, Performance-Counter, CI-Integration, Determinismus-Guardrails).

**Status**: Vollständig - Logging-Strategie (`tracing` Rust + `logging` Python), Security-Allowlist, Profiling-Artefakte definiert

---

### ✅ OMEGA_V2_FORMATTING_PLAN.md
**Zweck**: Normative Formatierungsregeln für Dokumentation, Code und Kommentare: Tooling (Black, isort, flake8, mypy, cargo fmt/clippy), Kommentar-Policy (WHY not WHAT), Durchsetzung via pre-commit + CI.

**Status**: Vollständig - Tool-driven Formatierung, Hybrid Single Source of Truth (Prinzipien hier, Parameter in Config-Dateien), ADR-Pflicht bei Konflikten

---

## Agent-System & Entwicklung
---

### ✅ OMEGA_V2_AGENT_INSTRUCTION_PLAN.md
**Zweck**: Institutionelle Operating Procedure für AI-Agenten (Claude Opus/Sonnet, GPT-5.2, Codex-Max, Copilot): Model-Routing, Builder+Critic Workflow, Guardrails, Quality Gates, Task-Brief Standard.

**Status**: Vollständig - Agent-Infrastruktur in `.agent/` etabliert, MODULE_STRUCTURE_PLAN um Abschnitt 9 (Agent-System-Architektur) erweitert

**Komponenten**:
- `.agent/routing/` – Model-Routing + Profile (Opus, Sonnet, GPT-5.2, Codex-Max)
- `.agent/context/` – GUARDRAILS, V2_CONTEXT_PACK, QUALITY_GATES
- `.agent/prompts/` – Prompt-Templates (NEW_CRATE, BUGFIX, CONTRACT_CHANGE, PERFORMANCE)
- `agent_tasks/` – Task-Brief Workspace

---

## Querverweise

Alle Pläne befinden sich in `docs/` und folgen der Namenskonvention `OMEGA_V2_<TOPIC>_PLAN.md`.

| Bereich | Primärer Plan | Ergänzende Pläne |
|---------|--------------|------------------|
| **Vision & Ziele** | VISION_PLAN | - |
| **Architektur** | ARCHITECTURE_PLAN | MODULE_STRUCTURE_PLAN |
| **Strategien** | STRATEGIES_PLAN | CONFIG_SCHEMA_PLAN, EXECUTION_MODEL_PLAN, MODULE_STRUCTURE_PLAN |
| **Trade-/Position-Management** | TRADE_MANAGER_PLAN | EXECUTION_MODEL_PLAN, OUTPUT_CONTRACT_PLAN, CONFIG_SCHEMA_PLAN, MODULE_STRUCTURE_PLAN |
| **Indikatoren** | INDICATOR_CACHE__PLAN | STRATEGIES_PLAN, DATA_FLOW_PLAN, MODULE_STRUCTURE_PLAN, TESTING_VALIDATION_PLAN |
| **Daten** | DATA_FLOW_PLAN | DATA_GOVERNANCE_PLAN |
| **Execution** | EXECUTION_MODEL_PLAN | METRICS_DEFINITION_PLAN |
| **Config/Output** | CONFIG_SCHEMA_PLAN | OUTPUT_CONTRACT_PLAN |
| **Qualität** | TESTING_VALIDATION_PLAN | CI_WORKFLOW_PLAN |
| **Standards** | FORMATTING_PLAN | AGENT_INSTRUCTION_PLAN |
| **Agent-System** | AGENT_INSTRUCTION_PLAN | MODULE_STRUCTURE_PLAN (Abschnitt 9), TECH_STACK_PLAN |
| **Betrieb** | OBSERVABILITY_PROFILING_PLAN | CI_WORKFLOW_PLAN |

---

## Konsolidierte Offene Punkte (Quick Reference)

### Kritischer Pfad (Blockiert andere Arbeit)

| ID | Beschreibung | Plan | Blockiert |
|----|--------------|------|-----------|
| *(keine)* | – | – | – |

### Hohe Priorität (MVP-relevant)

| ID | Beschreibung | Plan | Empfehlung |
|----|--------------|------|------------|
| *(keine)* | – | – | – |

### Mittlere Priorität (Implementierungs-Details)

| ID | Beschreibung | Plan |
|----|--------------|------|
| *(keine)* | – | – |

### Niedrige Priorität (Post-MVP)

| ID | Beschreibung | Plan |
|----|--------------|------|
| *(keine)* | – | – |

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
| 2.0 | 14.01.2026 | **Vollständigkeits-Analyse**: Alle 13 Pläne analysiert, offene Punkte konsolidiert, 4 Punkte als gelöst markiert, IDs und Empfehlungen hinzugefügt |
| 2.1 | 14.01.2026 | Runde 1 Entscheidungen eingepflegt: A-1, D-3, E-1, E-2, O-1, O-2, ME-1/T-1, T-2; Tracker bereinigt |
| 2.2 | 14.01.2026 | Runde 2 Entscheidungen eingepflegt: A-3, M-1, M-2, M-3, ME-2, T-3; Pläne konsistent gemacht |
| 2.3 | 14.01.2026 | Formatting-Plan als geplanter Qualitäts-Gate aufgenommen |
| 2.4 | 13.01.2026 | Formatting-Plan erstellt und vollständig: Tool-driven (Black/isort/flake8/mypy + cargo fmt/clippy), pre-commit + CI Gates |
| 2.5 | 14.01.2026 | Strategies-Plan erstellt (`OMEGA_V2_STRATEGIES_PLAN.md`) und Querverweise in allen Plänen ergänzt |
| 2.6 | 13.01.2026 | Indicator-Cache-Plan erstellt (`OMEGA_V2_INDICATOR_CACHE__PLAN.md`) und Querverweise/Plan-Übersicht aktualisiert |
| 2.7 | 14.01.2026 | Trade-Manager-Plan erstellt (`OMEGA_V2_TRADE_MANAGER_PLAN.md`) und Querverweise in allen Plänen ergänzt |

