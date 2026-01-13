# Omega V2 – Testing & Validation Plan

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation der Test- und Validierungsstrategie für den Omega V2 Backtest-Core (Rust + FFI + Python Wrapper): Testpyramide, deterministische Fixtures, Golden-File Regeln, Parität zu V1, Data-Governance-Validierung, Determinismus- und CI-Integration  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Erfolgskriterien (insb. DEV-Parität) |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Crate-Grenzen, Single FFI Boundary |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Data Loading Phasen, Timestamp-Contract, Fail-Fast Checkpoints |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Data-Quality-Policies (Alignment/Gaps/Duplicates), Manifest/Snapshot Identity |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Execution-Semantik (Bid/Ask, Tie-Breaks, pip_buffer, in_entry_candle, Slippage/Fees) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Artefakt-Contract (`trades.json`, `equity.csv`, `metrics.json`, `meta.json`) |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Metrik-Keys/Units/Rundung/Clamp (Artefaktstabilität) |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Config-Schema (run_mode/data_mode/rng_seed), Defaults/Normalisierung |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchain/Pinning, Error-Contract, RNG-Policy, Packaging |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Determinismus-sichere Observability, Profiling (nightly/manuell) |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI-Gates, Build-Matrix, Scheduling (Golden: nightly + release) |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für Tests und Validierung in Omega V2.

**Geltungsbereich (festgezogen):**

- Dieses Dokument gilt **ausschließlich** für den Omega V2 Backtest-Core (Rust + FFI + Python Wrapper).
- Live-Trading/MT5 und UI-Prozesssteuerung sind **out-of-scope**.

**Primäre Qualitätsziele (MVP, Priorität):**

1. **Output-Contract & Determinismus**: DEV-Runs sind reproduzierbar und Artefakte erfüllen den Output-Contract.
2. **V1↔V2 Parität (Hybrid)**: Entry/Exit-Events müssen übereinstimmen; PnL/Fees dürfen nur innerhalb enger Toleranzen abweichen.
3. **Fail-Fast Datenqualität**: Data-Governance-Regeln werden testbar und regressionssicher durchgesetzt.

**Bewusster Stil dieses Dokuments:**

- Normativ, aber mit bewusst markierten offenen Punkten, wo andere Pläne noch offene Entscheidungen enthalten.

---

## 2. Nicht-Ziele / Guardrails

- Keine Tests für MT5/Live-Adapter (Windows-only, V1-Pfad).
- Keine UI-Engine Tests als Gate für Omega V2.
- Keine „Best-Effort“ Policies im Core: Data/Execution/Schema-Verletzungen sind deterministisch **hard fail** (siehe Data Governance / Data Flow).

---

## 3. Testpyramide & Taxonomie (Normativ)

Omega V2 nutzt eine Testpyramide mit vier Ebenen:

1. **Unit Tests** (pro Crate/Modul)
2. **Property Tests** (Rust, `proptest`)
3. **Integration/E2E Tests** (Backtest-End-to-End über `run_backtest`)
4. **Contract Tests** (Output-Artefakte: Schema/Normalisierung/Golden)

### 3.1 Unit Tests (MUSS)

Unit Tests prüfen deterministische, kleine Einheiten:

- Parsing/Normalisierung (Config/Result)
- Data Validation (Schema, Monotonie, Bid/Ask Side-Order)
- Execution Regeln (Trigger/Fill/Tie-Breaks)
- Portfolio-State Transitions
- Metrics-Funktionen (Formeln, Edge-Cases)

### 3.2 Property Tests (MUSS, Rust)

**Framework:** `proptest`.

Property Tests zielen auf Invarianten und Edge-Cases, z.B.:

- Zeitreihen-Invarianten (strictly increasing, unique) werden korrekt erkannt.
- Bid/Ask Alignment ist ein Inner-Join auf Zeitstempel und ist deterministisch.
- Execution-Invarianten (z.B. „Exit kann nicht vor Entry liegen“).
- Metrics-Domain/Clamp (Scores in 0..1).

### 3.3 Integration/E2E Tests (MUSS)

E2E Tests sind „Golden“-fähig:

- Config → `run_backtest` → Artefakte/Result
- Prüfung der Artefakt-Contracts (Form/Schema)
- Prüfung der deterministischen Result-Identität (DEV)

### 3.4 Contract Tests (MUSS)

Contract Tests prüfen:

- Output-Contract-Konformität (Files, Encoding, Zeitfelder)
- Metrik-Rundung/Units/Definitions (Metrics-Plan)
- Normalisierung (für bit-identische Vergleiche)
- Golden-File Vergleichsregeln (siehe Abschnitt 6)

---

## 4. Coverage-Policy (Zielwerte, initial)

**Normativ:** Coverage-Zahlen sind **Zielwerte** (Qualitätsindikatoren), nicht zwingend PR-Blocker.

- PR-Gates sind primär: Tests grün + deterministische Contracts.
- Coverage wird berichtet und über Zeit hochgezogen.

### 4.1 Zielwerte pro Crate (Startwerte)

| Bereich/Crate | Ziel-Coverage (initial) | Begründung |
|--------------|--------------------------|------------|
| `types` | 95% | Fundament: Datenmodelle & Invarianten |
| `data` | 90% | Data Governance ist Fail-Fast Kernrisiko |
| `execution` | 90% | Fill-/Tie-Breaks sind correctness-kritisch |
| `portfolio` | 90% | State Machine + Equity-Konsistenz |
| `strategy` | 85% | Strategie-Regeln, aber oft fixture-lastig |
| `backtest` | 80% | Event Loop ist schwer granular zu testen, E2E ergänzt |
| `metrics` | 85% | Formeln/Edge-Cases müssen stabil sein |
| `ffi` | 60% | Glue-Code, Schwerpunkt auf Contract/E2E |
| `python/bt` | 70% | Orchestrator/Reporting, Contract-lastig |

**Hinweis:** Konkrete Grenzwerte/Enforcement (z.B. `llvm-cov` Gates) werden erst eingeführt, wenn Build-Matrix und Tooling stabil sind.

---

## 5. Fixtures & Testdaten (Normativ)

### 5.1 Quelle der Testdaten

**Normativ:** Fixtures werden **im Repository committed**.

- Ziel: deterministisch, offline, CI-stabil.
- Fixtures sind klein genug, um PRs nicht zu belasten.

### 5.2 Datenformat

**Normativ:** Market-Fixtures liegen als **Parquet** vor (kanonisch zu V2).

### 5.3 Größe/Komplexität

**Normativ:** Mix aus:

- **Mikro-Fixtures** (10–200 Bars): präzise Edge-Cases
- **Kleine Fixtures** (1k–5k Bars): realistische E2E-/Golden-Regression

### 5.4 Fixture-Layout (Vorschlag)

Diese Struktur ist kompatibel zum Modul-Struktur-Plan (Tests/Fixtures im Backtest-Crate):

- `rust_core/crates/backtest/tests/fixtures/`
  - `market/{SYMBOL}/{SYMBOL}_{TF}_BID.parquet`
  - `market/{SYMBOL}/{SYMBOL}_{TF}_ASK.parquet`
  - `news/news_calender_history.parquet`
  - `manifests/<fixture_name>.manifest.json`

### 5.5 News Fixtures (MVP)

**Normativ (MVP):** News werden in Tests minimal geprüft:

- Loader kann Parquet lesen
- Schema-/Normalisierungsregeln werden eingehalten

Die Strategie-Nutzung von News ist nicht Gate im MVP.

---

## 6. Golden Files & Vergleichsregeln (Normativ)

### 6.1 Golden-Artefakte

**Normativ:** Golden-Regression umfasst alle vier MVP-Artefakte:

- `trades.json`
- `equity.csv`
- `metrics.json`
- `meta.json`

### 6.2 Normalisierung vor Vergleich

**Normativ:** Golden- und Determinismus-Vergleiche erfolgen auf **normalisierten** Artefakten.

Ziel: Unterschiede aus nicht-deterministischen Feldern (z.B. Erstellzeit) werden eliminiert, ohne die fachliche Aussage zu verfälschen.

Beispiele (nicht abschließend):

- `meta.json`: Felder wie `generated_at`, `generated_at_ns` werden für Vergleichszwecke neutralisiert.
- JSON: stabile Key-Order (kanonische Serialisierung) und deterministische Reihenfolge, wo relevant.

### 6.3 Float-Vergleich

**Normativ:** Nach Contract-Rundung wird **exakt** verglichen.

- Rundung erfolgt gemäß:
  - `OMEGA_V2_METRICS_DEFINITION_PLAN.md` (2/6 Dezimalstellen)
  - `OMEGA_V2_OUTPUT_CONTRACT_PLAN.md` (Zeit/Units/Encoding)

### 6.4 Update-Policy

**Normativ:** Golden-Updates erfolgen **nur** über einen expliziten Prozess (Script/Flag) und sind reviewpflichtig.

- „Golden drift“ ist eine Breaking-Change im Qualitätsvertrag und muss begründet werden.

### 6.5 Scheduling (CI)

**Normativ:** Golden-Regression läuft **nur**:

- nightly
- release

Sie ist **kein** PR-Gate (siehe CI-Workflow-Plan).

---

## 7. V1 ↔ V2 Parität (Hybrid, Normativ)

### 7.1 Paritäts-Definition

**Normativ:** Parität ist hybrid:

- **Events/Trades müssen übereinstimmen** (Entry/Exit Zeitpunkte, Reihenfolge, Richtung, Exit-Reason).
- **PnL/Fees müssen innerhalb enger Toleranzen übereinstimmen**.

Toleranzen werden dort definiert, wo Units und Rundung normiert sind (Execution-/Metrics-/Output-Contract). Dieses Dokument definiert keine neuen Units.

### 7.2 Kanonische 6 Szenarien (MUSS)

**Normativ:** Für die MVP-Strategie „Mean Reversion Z-Score“ existieren exakt 6 kanonische Szenario-Tests, die sowohl V1 als auch V2 ausführen und vergleichen:

1. Market-Entry Long → Take-Profit
2. Market-Entry Long → Stop-Loss
3. Pending Entry (Limit/Stop) → Trigger ab `next_bar` → Exit
4. Same-Bar SL/TP Tie → SL-Priorität
5. `in_entry_candle` Spezialfall inkl. Limit-TP Regel
6. Mix aus Sessions/Warmup/HTF-Einflüssen, der die Strategie-Signalbildung deterministisch abdeckt

**Hinweis:** Die genaue Ausgestaltung (Fixture + Config) wird in den Test-Files dokumentiert; dieses Dokument normiert die Pflicht zur Existenz und Stabilität.

---

## 8. Determinismus & Reproduzierbarkeit (Normativ)

### 8.1 DEV Determinismus

**Normativ:** In `run_mode = dev` muss gelten:

- Zwei identische Runs (gleiche Config + gleiche Inputs) erzeugen **bit-identische normalisierte** Artefakte.

### 8.2 Cross-OS Determinismus

**Normativ:** DEV-Determinismus gilt **cross-OS**:

- macOS (arm64/x86_64), Linux (x86_64), Windows (x86_64)
- normalisierte Artefakte müssen identisch sein

### 8.3 Dataset Identity / Manifest Tests

**Normativ:** Fixture-Manifests werden geprüft:

- pro-file SHA-256 Hashes sind korrekt
- Manifest-Hash ist stabil
- `meta.json` referenziert die Snapshot-ID (`manifest_sha256`) gemäß Output-Contract

---

## 9. Data Governance Validation (Normativ)

### 9.1 Negative Tests für jede Hard-Fail Regel

**Normativ:** Für jede als hard fail definierte Data-Governance-Regel existiert mindestens ein negativer Test.

Dazu gehören u.a.:

- Schema-Verletzungen
- Out-of-order Zeitstempel
- Duplicate Timestamps (harmlos vs. unterschiedliche OHLCV)
- Bid/Ask Side-Order Verletzungen
- Monotonie/Uniqueness

### 9.2 Alignment-Loss Tests

**Normativ:** Die Alignment-Policy wird als Contract getestet.

- Die Schwellen/Regeln werden aus `OMEGA_V2_DATA_GOVERNANCE_PLAN.md` übernommen.

---

## 10. Execution Model Validation (Normativ)

### 10.1 Tie-Breaks & Same-Bar Sonderfälle

**Normativ:** Execution Edge-Cases werden vollständig getestet:

- SL vs TP Priorität
- `pip_buffer` Regeln
- `in_entry_candle` Sonderlogik
- Pending Trigger erst ab `next_bar`

### 10.2 Slippage Randomness (DEV) + Sanity

**Normativ:** Slippage ist im DEV-Mode deterministisch (Seed → stabile Sequenz).

Zusätzlich sind Sanity-Checks zulässig, die keine fragile Verteilungsannahmen erzwingen.

---

## 11. Metrics Validation (Normativ)

### 11.1 Formel- und Edge-Case Tests

**Normativ:** Metrics werden über bekannte Inputs geprüft:

- Formeln für Kernmetriken
- Domain/Clamp (Scores 0..1)
- Rundung (2/6 Dezimalstellen)
- Konsistenz zwischen `metrics` und `definitions`

### 11.2 Annualisierung Sharpe/Sortino (offen)

**Offen:** Der Metrics-Plan nennt Annualisierung als offenen Punkt.

**Policy in diesem Dokument:**

- Bis Annualisierung/Frequenz endgültig festgezogen ist, werden Sharpe/Sortino Tests so gestaltet, dass sie
  - Rundung/Domain/NaN-Handling absichern,
  - aber keine implizite Annualisierungs-Konvention erzwingen.

---

## 12. CI-Integration (Normativ)

### 12.1 PR-Gates vs. Nightly

**Normativ:**

- PR-Gate: schnelle Unit/Property/Integration + Contract Checks
- Nightly/Release: Golden-Regression + Cross-OS Determinismus + Benchmarks

### 12.2 Benchmarks

**Normativ:** Performance-Regression wird über `criterion` (Rust) abgedeckt und läuft nur nightly.

---

## 13. Offene Punkte

1. **Sharpe/Sortino Annualisierung**: finale Definition (Formel/Frequenz) muss im Metrics-Plan festgezogen werden.
2. **12.1 (User-Planung offen, MVP-Gate)**: Es existiert mindestens ein zusätzlicher Pflichtpunkt, der als MVP-Gate in den Testing/Validation Plan aufgenommen werden soll. Dieser Punkt ist bewusst noch nicht spezifiziert und muss in einem Follow-up konkretisiert werden.
