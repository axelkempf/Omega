# Omega V2 – Plan-Übersicht & Fortschritt

> **Status**: Aktiv  
> **Erstellt**: 12. Januar 2026  
> **Letzte Analyse**: 14. Januar 2026  
> **Zweck**: Zentrale Übersicht aller OMEGA_V2 Planungsdokumente mit Fortschritts-Tracking und konsolidierten offenen Punkten

---

## Fortschritts-Tracker

| Status | Bedeutung | Anzahl |
|--------|-----------|--------|
| ✅ | Existiert und vollständig | 6 |
| 🟡 | Existiert, offene Punkte | 7 |
| 🔲 | Geplant, noch nicht erstellt | 1 |

**Gesamt: 14 Pläne** | **Fortschritt: 13/14 erstellt (93%)**

---

## Vollständigkeits-Analyse (14.01.2026)

### Zusammenfassung

Nach systematischer Analyse aller 13 existierenden Planungsdokumente wurden folgende Erkenntnisse identifiziert:


---

## Existierende Pläne

### ✅ OMEGA_V2_VISION_PLAN.md
**Zweck**: Definiert das strategische Zielbild, die Problemanalyse des V1-Systems und messbare Erfolgskriterien für die V2-Migration.

**Vollständigkeit**: 100% - offene Punkte entschieden und dokumentiert

---

### 🟡 OMEGA_V2_ARCHITECTURE_PLAN.md
**Zweck**: Übergeordneter Blueprint mit Crate-Struktur, Modul-Abhängigkeiten, Systemregeln und FFI-Boundary-Definition.

**Offene Punkte**:
- [ ] **A-1**: T5: HTF-Datenquelle entscheiden
  - *Option A*: Separate Parquets pro Timeframe (aktuelles Design)
  - *Option B*: Aggregation aus Primary-TF (Performance-Vorteil, Komplexität)
  - *Empfehlung*: Option A für MVP, Option B als Post-MVP Optimierung
- [x] ~~**A-2**: Logging-Strategie (Rust tracing vs. Python) finalisieren~~ → **GELÖST**
  - *Lösung*: OBSERVABILITY_PROFILING_PLAN definiert: `tracing` für Rust, separates `logging` für Python
- [ ] **A-3**: Parallelisierung (rayon) Policy nach Parität festlegen
  - *Kontext*: Tech-Stack-Plan erwähnt rayon als verfügbar, aber Policy undefiniert
  - *Empfehlung*: rayon nur für Data-Load Phase, nicht für Event-Loop (Determinismus)

**Vollständigkeit**: ~90% - Architektur klar, HTF-Entscheidung offen

---

### 🟡 OMEGA_V2_DATA_FLOW_PLAN.md
**Zweck**: Vollständige Spezifikation des Datenflusses von Config-Input über Rust-Engine bis Result-Output inkl. Validierungs-Checkpoints.

**Offene Punkte**:
- [x] ~~**D-1**: Alignment-Loss Schwelle (Warning vs. Abort) definieren~~ → **GELÖST**
  - *Lösung*: DATA_GOVERNANCE_PLAN: `>1% alignment loss = hard fail`
- [x] ~~**D-2**: Gap-Policy (akzeptieren/pausieren/abbrechen) festlegen~~ → **GELÖST**
  - *Lösung*: DATA_GOVERNANCE_PLAN: `drop-bars, session-aware, >5% = warning`
- [ ] **D-3**: HTF `htf_idx-1` Edge-Case bei frühen Bars klären
  - *Problem*: Was passiert wenn `htf_idx=0` und `htf_idx-1` nicht existiert?
  - *Empfehlung*: NaN/None zurückgeben, Strategie muss prüfen
- [x] ~~**D-4**: Timestamp-Duplikat-Handling~~ → **GELÖST**
  - *Lösung*: DATA_GOVERNANCE_PLAN: `keep-first wenn OHLCV identisch, hard fail wenn unterschiedlich`

**Vollständigkeit**: ~95% - Fast vollständig, ein Edge-Case offen

---

### ✅ OMEGA_V2_DATA_GOVERNANCE_PLAN.md
**Zweck**: Normative Data-Quality-Policies (Alignment/Gaps/Duplicates), News-Governance (Parquet), sowie reproduzierbare Dataset-Snapshots (Hashes/Manifests).

**Status**: Vollständig - Alle Data-Quality-Regeln normativ festgelegt

---

### 🟡 OMEGA_V2_EXECUTION_MODEL_PLAN.md
**Zweck**: Bid/Ask-Regeln, Fill-Algorithmen, Intrabar-Tie-Breaks, Stop/TP-Prioritäten, Limit/Market-Semantik, Slippage/Fees und deterministische Ausführung.

**Offene Punkte**:
- [ ] **E-1**: Exakte Feldquelle/-name für „minimale SL-Distanz" pro Symbol festlegen
  - *Option A*: Erweiterung `configs/symbol_specs.yaml` um `min_sl_distance_pips`
  - *Option B*: Neues Feld in Execution-Costs-Config
  - *Empfehlung*: Option A (Symbol-spezifisch, passt zu bestehendem Schema)
- [ ] **E-2**: Finales Set an `reason`-Werten für Exits bestätigen
  - *Aktuelle Kandidaten*: `stop_loss`, `take_profit`, `trailing_stop`, `max_holding`, `manual`, `session_close`
  - *Offen*: Enum (strikt) vs. Strings mit optionalem `meta`-Feld (flexibel)
  - *Empfehlung*: Enum mit festem Set für MVP, Erweiterung über `other + meta` für Post-MVP

**Vollständigkeit**: ~90% - Kernlogik klar, zwei Detailfragen offen

---

### 🟡 OMEGA_V2_MODULE_STRUCTURE_PLAN.md
**Zweck**: Detaillierte Ordner-, Datei- und Modul-Struktur des Rust-Workspace mit Abhängigkeits-Matrix und Test-Strategie pro Crate.

**Offene Punkte**:
- [ ] **M-1**: Test-Fixtures Strategie entscheiden
  - *Option A*: Fixtures im Repo committed (deterministisch, offline)
  - *Option B*: Fixtures bei Bedarf generiert (kleiner Repo, potentielle Drift)
  - *Status*: TESTING_VALIDATION_PLAN empfiehlt Option A, aber nicht verbindlich festgelegt
- [ ] **M-2**: `alt_data.rs` / `news.rs` Modul-Details spezifizieren
  - *Offen*: Welche News-Felder werden geladen? Nur Zeitstempel oder auch Impact/Event-Type?
  - *Abhängigkeit*: DATA_GOVERNANCE_PLAN definiert Schema, aber Rust-Struct fehlt
- [ ] **M-3**: Property-Test Coverage-Ziele definieren
  - *Kontext*: TESTING_VALIDATION_PLAN nennt `proptest`, aber keine Coverage-Ziele
  - *Empfehlung*: 80% für `types`, `data`, `execution`; 60% für andere Crates

**Vollständigkeit**: ~85% - Struktur klar, Test-Strategie-Details fehlen

---

### 🟡 OMEGA_V2_OUTPUT_CONTRACT_PLAN.md
**Zweck**: Exakte Artefakt-Spezifikation für `trades.json`, `equity.csv`, `metrics.json`, `meta.json` inkl. Feldnamen, Typen, Einheiten, Zeit-Contract und Output-Pfad.

**Offene Punkte**:
- [ ] **O-1**: Golden-File Vergleichsregeln konkretisieren
  - *Float-Toleranzen*: Nach Contract-Rundung exakter Vergleich (TESTING_VALIDATION_PLAN 6.3)
  - *Normalisierung*: `meta.json` Felder wie `generated_at` neutralisieren
  - *Fehlend*: Dokumentation welche Felder genau neutralisiert werden
- [ ] **O-2**: `profiling/` Unterordner-Contract fehlt
  - *Kontext*: OBSERVABILITY_PLAN definiert `profiling.json`, aber nicht im OUTPUT_CONTRACT referenziert
  - *Empfehlung*: Als optionales Artefakt in OUTPUT_CONTRACT aufnehmen

**Vollständigkeit**: ~92% - Kern-Artefakte vollständig, Vergleichsregeln fehlen

---

### 🟡 OMEGA_V2_METRICS_DEFINITION_PLAN.md
**Zweck**: Normative Definition aller Performance-Metriken und Score-Keys (Units/Domain/Quelle/Edge-Cases) inkl. Policy für Rundung und Output-Ort (Single-Run vs. Optimizer).

**Offene Punkte**:
- [ ] **ME-1**: Sharpe/Sortino: exakte Formeln + Annualisierung konkretisieren
  - *Frage*: Daily Equity Returns oder Trade-R (PnL pro Trade)?
  - *Frage*: `sqrt(252)` (Trading Days) oder `sqrt(365)` (Calendar Days)?
  - *Kontext*: TESTING_VALIDATION_PLAN blockiert Tests bis geklärt
- [ ] **ME-2**: Optimizer-/Rating-Aggregate-Contract spezifizieren
  - *Offen*: Separates `optimizer_metrics.json` oder Erweiterung von `metrics.json`?
  - *Felder*: `robustness_score`, `stability_score`, `parameter_sensitivity`

**Vollständigkeit**: ~80% - Single-Run Metriken klar, Sharpe/Sortino und Optimizer-Aggregate offen

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

### 🟡 OMEGA_V2_TESTING_VALIDATION_PLAN.md
**Zweck**: Testpyramide (unit/property/integration/contract), deterministische Fixtures, Golden-File Regeln, Paritäts-Tests gegen V1 sowie CI-Integration (PR-Gates vs nightly).

**Offene Punkte**:
- [ ] **T-1**: Sharpe/Sortino Annualisierung/Frequenz finalisieren
  - *Abhängigkeit*: Blockiert durch ME-1 (METRICS_DEFINITION_PLAN)
  - *Kontext*: Trade-R vs. Daily Equity basierte Berechnung
- [ ] **T-2**: MVP-Gate-Kriterien für „ready-to-merge" spezifizieren
  - *Kontext*: Explizit als offen markiert im Dokument (Abschnitt 13, Punkt 2)
  - *Kandidaten*: Determinismus-Test grün, Golden-Regression grün, Parität-Tests grün
- [ ] **T-3**: 6 kanonische Szenarien ausarbeiten
  - *Abhängigkeit*: nicht mehr blockiert (V-2 ist entschieden)
  - *Status*: Szenario-Typen + Baseline sind normativ festgelegt; konkrete Fixtures/Configs fehlen

**Vollständigkeit**: ~85% - Teststrategie klar, spezifische Gate-Kriterien fehlen

---

### ✅ OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md
**Zweck**: Normative Spezifikation der Observability-Strategie für Omega V2 (Logging/Tracing via tracing, Profiling via flamegraph/pprof, Performance-Counter, CI-Integration, Determinismus-Guardrails).

**Status**: Vollständig - Logging-Strategie (`tracing` Rust + `logging` Python), Security-Allowlist, Profiling-Artefakte definiert

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

## Konsolidierte Offene Punkte (Quick Reference)

### Kritischer Pfad (Blockiert andere Arbeit)

| ID | Beschreibung | Plan | Blockiert |
|----|--------------|------|-----------|
| **ME-1** | Sharpe/Sortino Annualisierung/Frequenz | METRICS_DEFINITION | T-1 Tests |

### Hohe Priorität (MVP-relevant)

| ID | Beschreibung | Plan | Empfehlung |
|----|--------------|------|------------|
| **A-1** | HTF-Datenquelle entscheiden | ARCHITECTURE | Separate Parquets |
| **E-1** | SL-Distanz Feldquelle | EXECUTION_MODEL | symbol_specs.yaml |
| **E-2** | Exit-Reason Enum finalisieren | EXECUTION_MODEL | Enum mit `other` Fallback |
| **T-2** | MVP-Gate-Kriterien | TESTING_VALIDATION | Determinismus + Golden + Parität |

### Mittlere Priorität (Implementierungs-Details)

| ID | Beschreibung | Plan |
|----|--------------|------|
| **D-3** | HTF idx-1 Edge-Case | DATA_FLOW |
| **M-1** | Test-Fixtures Strategie | MODULE_STRUCTURE |
| **M-2** | News-Modul Details | MODULE_STRUCTURE |
| **O-1** | Golden-File Normalisierung | OUTPUT_CONTRACT |
| **O-2** | Profiling Artefakt-Contract | OUTPUT_CONTRACT |

### Niedrige Priorität (Post-MVP)

| ID | Beschreibung | Plan |
|----|--------------|------|
| **A-3** | Rayon Parallelisierung Policy | ARCHITECTURE |
| **M-3** | Property-Test Coverage-Ziele | MODULE_STRUCTURE |
| **ME-2** | Optimizer-Aggregate Contract | METRICS_DEFINITION |

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

