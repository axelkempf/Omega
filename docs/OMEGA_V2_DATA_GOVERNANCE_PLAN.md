# Omega V2 – Data Governance Plan

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation der Daten-Qualitätsregeln (Market Data + News + Support-Daten), der Fail-Fast Policies, sowie reproduzierbarer Dataset-Snapshots (Hashes/Manifests) für Omega V2  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Verantwortlichkeiten, Single FFI Boundary |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Data Loading Phasen, Timestamp-Contract, Fail-Fast Checkpoints |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Input-Config ohne Pfade, Defaults, Env-Path-Resolution |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Artefakt-Contract inkl. `meta.json` Provenance |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Bid/Ask-Semantik, Fill-Model, deterministische Ausführung |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Data Loader / News Loader Module, Zuständigkeiten |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Parquet/Arrow, Rust I/O, Fehler-Contract |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing (tracing), Profiling (flamegraph/pprof), Performance-Counter, Determinismus |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Teststrategie, Data-Governance Validierung, Golden-Files, Determinismus |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | Quality Gates, deterministische Checks in CI |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für Datenqualität in Omega V2:

- **Welche Daten gelten als „valid“?** (Schema + Zeit-Contract + Invarianten)
- **Wie werden Inkonsistenzen behandelt?** (Alignment/Gaps/Duplicates/Out-of-Order)
- **Wie wird Provenance reproduzierbar?** (Hashes + Manifests + Snapshot-Policy)
- **Wann wird validiert und wie wird berichtet?** (Fail-Fast, `meta.json`)

**Grundsatz (normativ):**

- Omega V2 arbeitet **strict**: Datenverletzungen führen zu **hard fail** (kein „best effort“ im Core).
- Remediation (z.B. „auf Intersection alignen“, „Bars droppen“) ist **nur** dann zulässig, wenn sie
  1) vollständig deterministisch ist,
  2) transparent berichtet wird (Provenance in `meta.json`), und
  3) definierte Schwellen (z.B. Alignment-Loss) nicht überschreitet.

---

## 2. Geltungsbereich

### 2.1 In Scope

1. **Market Data (BID/ASK Candles)** im Parquet-Format.
2. **News / Economic Calendar** im Parquet-Format.
3. **Support-Daten** (`configs/execution_costs.yaml`, `configs/symbol_specs.yaml`) als „Input-Facts“ (Schema/Hashing/Versionierung).
4. **Dataset Identity** (Hashes/Manifests/Snapshots) und deren Einbettung in Run-Provenance.

### 2.2 Out of Scope

- Tick-Daten Layout/Quality (V2-MVP fokussiert Candle-Mode).
- Downstream Reporting-UX (Dashboards, UI) – hier nur Contract.
- Business-Logik der Strategie (Signal-Qualität) – hier nur Datenqualität.

---

## 3. Kanonisches Layout & Pfad-Auflösung

### 3.1 Market Data – kanonisches Layout

**Normativ:** Ohne explizite Pfade in der V2-Config.

Default Root:

- `data/parquet`

Expected Layout:

- `data/parquet/{SYMBOL}/{SYMBOL}_{TF}_BID.parquet`
- `data/parquet/{SYMBOL}/{SYMBOL}_{TF}_ASK.parquet`

### 3.2 News – kanonisches Layout

**Normativ:** Omega V2 lädt News aus **Parquet**.

Default Location:

- `data/news/news_calender_history.parquet`

**Hinweis:** Das Repo enthält historisch `data/news/news_calender_history.csv`. Der Konvertierungsschritt (CSV → Parquet) ist Teil der Datenpipeline (Python Utility) und **nicht** Teil der V2-Config.

### 3.3 Support-Daten – kanonisches Layout

- `configs/execution_costs.yaml`
- `configs/symbol_specs.yaml`

### 3.4 Env-Overrides (zulässig, aber sichtbar)

Env-Overrides sind **zulässig**, aber gelten als Teil der Provenance.

- Market Data Root: `OMEGA_DATA_PARQUET_ROOT` (Default: `data/parquet`)
- Execution Costs: `OMEGA_EXECUTION_COSTS_FILE` (Default: `configs/execution_costs.yaml`)
- Symbol Specs: `OMEGA_SYMBOL_SPECS_FILE` (Default: `configs/symbol_specs.yaml`)
- News Calendar File: `OMEGA_NEWS_CALENDAR_FILE` (Default: `data/news/news_calender_history.parquet`)

**Normativ:** Wenn ein Env-Override aktiv ist, MUSS

- der effektive Pfad in `meta.json` referenziert werden und
- in den Dataset-Hashes/Manifests einfließen (siehe Abschnitt 8).

---

## 4. Zeit- und Schema-Contract (universell)

Referenz: `OMEGA_V2_DATA_FLOW_PLAN.md` (UTC/Timestamp-Contract).

### 4.1 Zeitachse

- Zeitzone ist **immer UTC**.
- Parquet nutzt eine Spalte **`UTC time`** als **Open-Time** (Candles) bzw. Event-Zeit (News).
- Rust Core nutzt `timestamp_ns` als `i64` (Nanoseconds since epoch UTC).

### 4.2 Sortierung und Eindeutigkeit

Für jede Zeitreihe gilt:

- Zeitstempel MUSS **strictly increasing** sein.
- Zeitstempel MUSS **unique** sein.

---

## 5. Market Data Governance (BID/ASK Candles)

### 5.1 Spalten & minimale Schema-Regeln

**Normativ:** Candles sind OHLCV mit Spalten:

- `UTC time` (UTC, ns)
- `Open`, `High`, `Low`, `Close` (float)
- `Volume` (int/float, nicht-negativ)

### 5.2 Candle-Sanity Checks (strict)

Für jede Candle MUSS gelten:

- `High >= max(Open, Close)`
- `Low <= min(Open, Close)`
- `High >= Low`
- `Volume >= 0`

Verletzung ⇒ **hard fail**.

### 5.3 Bid/Ask-Sanity Checks (strict)

Für jeden Zeitstempel $t$ im aligned Stream MUSS gelten:

- `Ask.High >= Bid.High` und `Ask.Low >= Bid.Low` ist **nicht** zwingend (High/Low können artefaktbedingt divergieren),
- aber **preisliche Side-Order** MUSS konsistent sein:
  - `Ask.Close >= Bid.Close`
  - `Ask.Open >= Bid.Open`

Verletzung ⇒ **hard fail**.

### 5.4 Alignment Policy (Intersection, mit Schwelle)

**Ziel:** Omega V2 verarbeitet Candle-Events auf einer gemeinsamen Zeitachse.

**Normativ:**

- BID und ASK werden auf die **Intersection** der Zeitstempel reduziert.
- Der dadurch entstehende Verlust MUSS unter einer harten Schwelle liegen.

Definitionen:

- $B$ = Menge der BID-Zeitstempel
- $A$ = Menge der ASK-Zeitstempel
- $I = B \cap A$

Alignment-Loss:

- $loss_{bid} = 1 - \frac{|I|}{|B|}$
- $loss_{ask} = 1 - \frac{|I|}{|A|}$
- $alignment\_loss = \max(loss_{bid}, loss_{ask})$

**Policy:**

- Wenn $alignment\_loss > 0.01$ ⇒ **hard fail**.
- Sonst: Reduktion auf $I$ und Fortsetzung.

**Reporting (normativ):**

- `alignment_loss` MUSS in `meta.json` erfasst werden (siehe Abschnitt 9 / Output-Contract).

### 5.5 Gap Policy (sessions-aware, drop bars)

**Definition:** Ein Gap liegt vor, wenn innerhalb einer aktiven Trading-Session erwartete Candles fehlen.

- Erwartete Schrittweite entspricht dem Timeframe (z.B. M5 ⇒ 5 Minuten).
- Sessions sind UTC und kommen aus `config.sessions` (falls gesetzt).
- Wenn `sessions == null`: Standard ist 24/5 (Wochenende ist keine Session).

**Policy (normativ):**

- Gaps innerhalb aktiver Sessions werden **nicht** gefüllt.
- Stattdessen werden die betroffenen Zeitbereiche deterministisch „ausgeschnitten“ (drop bars) und anschließend wie Alignment behandelt.
- Wenn dadurch $alignment\_loss$ die Schwelle überschreitet ⇒ **hard fail**.

### 5.6 Duplicate Timestamps (keep-first, aber nur wenn harmlos)

**Normativ:** Duplicates sind eigentlich ein Vertragsbruch. Für Kompatibilität ist folgende deterministische Remediation erlaubt:

- Deduplizieren nach Zeitstempel mit Policy **keep-first**.
- Wenn die Duplikate unterschiedliche OHLCV-Werte haben ⇒ **hard fail** (sonst wäre keep-first nicht auditierbar).

### 5.7 Out-of-order timestamps

- Sobald out-of-order entdeckt wird ⇒ **hard fail**.

---

## 6. News Governance (Economic Calendar)

### 6.1 Kanonisches Format

**Normativ:** Omega V2 lädt News **aus Parquet**.

- Datei: `data/news/news_calender_history.parquet`
- Zeitspalte: `UTC time` (Event-Zeit in UTC)

### 6.2 Minimales Schema

Das News-Parquet MUSS mindestens enthalten:

- `UTC time` (Event-Zeit, UTC)
- `Id` (int)
- `Name` (string)
- `Impact` (enum/string; normalisiert)
- `Currency` (string, 3-letter, uppercase)

### 6.3 Normalisierung & Validierung

- `Currency` MUSS uppercase und 3-stellig sein.
- `Impact` MUSS in eine stabile Menge normalisierbar sein (z.B. `LOW|MEDIUM|HIGH`).
- Zeitstempel MUSS UTC sein.

Verletzung ⇒ **hard fail**.

### 6.4 CSV → Parquet Konvertierung (Pipeline)

**Normativ:** Das V2-System konsumiert Parquet; eine Python-Utility darf CSV importieren und als Parquet schreiben.

- Konvertierung MUSS deterministisch sein (stable sort, type normalization).
- Der erzeugte Parquet-Output MUSS in die Dataset-Manifests eingehen.

---

## 7. Support-Daten Governance (Costs/Specs)

### 7.1 `configs/execution_costs.yaml`

- MUSS syntaktisch korrektes YAML sein.
- MUSS definierte Default-Keys enthalten (Contract bleibt im Execution-Model Plan).

### 7.2 `configs/symbol_specs.yaml`

- MUSS syntaktisch korrektes YAML sein.
- MUSS pro Symbol die minimalen Specs enthalten, die Execution/Validation benötigen.

### 7.3 Provenance

- Beide Dateien MUSS in `meta.json` gehasht referenziert werden (siehe Abschnitt 8/9).

---

## 8. Dataset Identity, Hashing & Snapshot-Policy

### 8.1 Ziele

- Backtests müssen reproduzierbar und auditierbar sein.
- „Welche Daten wurden wirklich verwendet?“ muss ohne implizites Wissen beantwortbar sein.

### 8.2 Hashing (per-file + Manifest)

**Normativ:**

- Jede input-relevante Datei (Market Parquet BID/ASK je TF, News Parquet, Costs/Specs YAML) erhält einen **SHA-256** Hash.
- Zusätzlich existiert ein **Manifest-Hash**, der den gesamten Snapshot eindeutig identifiziert.

### 8.3 Manifest Inhalt (minimal, normativ)

Ein Manifest MUSS mindestens enthalten:

- `manifest_version`
- `created_at` / `created_at_ns`
- `inputs` (Liste der Dateien)
  - `path`
  - `sha256`
  - `kind` (z.B. `market_parquet_bid`, `market_parquet_ask`, `news_parquet`, `execution_costs_yaml`, `symbol_specs_yaml`)
  - optionale Minimalstatistiken: `rows`, `start_time_ns`, `end_time_ns`

### 8.4 Speicherort (unter `var/`)

**Normativ:** Manifests werden unter `var/` abgelegt (nicht unter `data/`), um Repo-Checkout stabil zu halten.

Empfohlenes Layout:

- `var/reports/data_governance/manifests/<manifest_sha256>.json`

### 8.5 Immutabilität

- „Overwrite“ ist verboten: existiert ein Manifest für einen Snapshot, darf derselbe Identifier nie überschrieben werden.
- Änderungen am Datenbestand erzeugen einen **neuen** Manifest-Hash.

---

## 9. Ausführung der Governance Checks & Reporting

### 9.1 Wann wird validiert? (Fail-Fast)

**Normativ:** Governance Checks laufen als Teil des Data Loading **vor** dem Backtest-Core:

- sobald der Loader die Datenbasis für den Run bestimmt hat,
- bevor die Rust-Engine gestartet wird.

### 9.2 Wie wird berichtet? (ohne neues Artefakt im MVP)

**Normativ:** Ergebnisse werden

- in `meta.json` (Run-Ordner) als Provenance/Summary erfasst und
- als Log-Zeile (INFO) ausgegeben.

Es gibt **kein separates Warnings-Artefakt** im MVP (siehe Output-Contract).

### 9.3 `meta.json` Minimalfelder (Governance)

Die konkrete Felddefinition ist Teil des Output-Contracts; hier nur die Governance-Anforderungen:

- Manifest-Hash MUSS enthalten sein.
- Alignment-Loss MUSS enthalten sein.
- aktive effektive Pfade (inkl. Env-Overrides) SOLLEN referenziert werden.

---

## 10. Fehler- und Exit-Policy

- Alle Violations, die als „hard fail“ markiert sind, beenden den Run deterministisch.
- „Drop bars“ ist ein deterministischer Transformationsschritt und MUSS in Provenance sichtbar sein.

---

## 11. Offene Punkte

- Tick-Data Governance (Layout, Time Contract, Bid/Ask Streams)
- Session-Calendar Default für Nicht-FX Assets (z.B. Indizes/CFDs)
