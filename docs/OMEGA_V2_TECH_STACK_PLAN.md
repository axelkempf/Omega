# Omega V2 – Tech Stack Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Normative Spezifikation des Omega V2 Tech-Stacks (Toolchains, Version-Pinning, Packaging/FFI, Build-Matrix, OS-Support, Logging/RNG/Error-Contracts)  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Modul-Grenzen, Single FFI Boundary |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | Strategie-Spezifikation (MVP: Mean Reversion Z-Score), Szenarien 1–6, Guards/Filter, Indikatoren |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Timestamp-Contract, Data Loading Phasen |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Data-Quality-Policies (Alignment/Gaps/Duplicates), News=Parquet, Snapshots/Manifests |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives JSON-Config-Schema (run_mode, data_mode, rng_seed) |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Rust Workspace/Crates, FFI/Runner Layout |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell (Bid/Ask, SL/TP, Slippage/Fees, Sizing) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Output-Artefakte, Zeit/Units, Pfade |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Metrik-Keys, Units, Rundung, Edge-Cases |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing (tracing), Profiling (flamegraph/pprof), Performance-Counter, Determinismus |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Teststrategie, Golden-Files, Parität/Determinismus (DEV), CI-Integration |
| [OMEGA_V2_FORMATTING_PLAN.md](OMEGA_V2_FORMATTING_PLAN.md) | Format-/Lint-Regeln (Code/Doku/Kommentare), Durchsetzung via pre-commit + CI |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI/CD Workflow, Quality Gates, Build-Matrix, Security, Release-Assets |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für den Omega V2 Tech-Stack.

**Normativ (MVP):**

- **Rust Core** (Performance-kritisch) + **Python Orchestrator** (Wrapper/Reporting) – siehe Architektur-Plan.
- **Single FFI Boundary** via **PyO3**: genau **ein** Entry-Point `run_backtest(config_json)`.
- **Parquet I/O** in Rust via **arrow-rs** (`arrow`/`parquet`).
- **Build/Packaging** via **maturin** als **Single Wheel** (Python Wrapper + Rust Extension).
- **Observability** via `tracing`.
- **Determinismus** via `rand_chacha` + `rng_seed` (DEV) und OS-RNG (PROD).
- **FFI Error Contract (Hybrid)**: Config-/Argumentfehler → Python Exceptions; Runtime-Fehler während Run → JSON Error Result.

---

## 2. Geltungsbereich

### 2.1 In Scope

- Toolchain-/Version-Policy (Rust/Python, Lockfiles).
- Rust Crate-Auswahl für I/O, Serialisierung, Logging, RNG.
- FFI/Packaging-Entscheidungen (PyO3/maturin, Modulnamen, Error-Contract).
- OS-/Arch-Support für Backtest-Core (CI Build-Matrix).

### 2.2 Out of Scope

- Live-Trading/MT5-Integration (V1 bleibt unverändert; MT5 ist Windows-only).
- Konkrete CI-Workflow-YAML Implementierung (siehe `OMEGA_V2_CI_WORKFLOW_PLAN.md`).

---

## 3. Grundprinzipien (Normativ)

### 3.1 Reproduzierbarkeit & Determinismus

- **Toolchain ist gepinnt**: `rust-toolchain.toml` MUSS im Repo versioniert werden.
- **Dependencies sind gelockt**: `Cargo.lock` MUSS versioniert werden (siehe Modul-Struktur-Plan).
- **DEV-Mode ist deterministisch**: wenn `run_mode = "dev"`, MUSS ein stabiler RNG mit `rng_seed` verwendet werden.
- **PROD-Mode darf stochastisch sein**: wenn `run_mode = "prod"`, SOLL der Seed aus OS-RNG kommen (und nur bei explizitem `rng_seed` reproduzierbar sein).

### 3.2 Single FFI Boundary

- FFI ist **genau eine** Grenze: `run_backtest(config_json: &str) -> ...`.
- Während eines Backtest-Runs gibt es **keine** Rückflüsse nach Python (keine Callbacks, keine PyO3-Objekte im Core).

### 3.3 Fail-Fast, aber mit klarer Fehler-Semantik

- Fehler sind **hard fail** im Sinne des Output-Contracts.
- Die Repräsentation des Fehlers ist jedoch normiert (siehe Abschnitt 7):
  - **Setup-/Inputfehler** → Python Exception
  - **Run-time Fehler** (Data/Strategy/Execution während Backtest) → JSON Error Result

---

## 4. Sprachen & Toolchains

### 4.1 Rust

**Normativ:**

- **Rust-Channel**: *latest stable zum Zeitpunkt der Implementierung/Merge*.
- **Pinning**: `rust-toolchain.toml` MUSS eine **konkrete Version** enthalten (kein „floating“), wird aber regelmäßig auf die jeweils aktuelle Stable-Version aktualisiert.
- **Edition**: **2024**.

**Kompatibilität:**

- Alle Crates im Workspace müssen mit der gepinnten Toolchain + Edition 2024 bauen.

### 4.2 Python

- Python ist Orchestrator/Wrapper.
- Python-Version: **3.12+** (konsistent mit Repo-Policy).

---

## 5. Rust Core Dependencies (MVP)

### 5.1 Data I/O (Parquet/Arrow)

**Normativ:**

- Parquet-Reader: **arrow-rs**.
- Version-Pinning: **Major pin** in `Cargo.toml` + `Cargo.lock` als deterministischer Resolver.
- Richtwert aus Modul-Struktur-Plan: `arrow = "51"`, `parquet = "51"`.

**Contract:**

- Timestamp-Contract (UTC, epoch-ns, Open-Time) ist **nicht verhandelbar** und wird im Data-Flow-Plan normiert.

### 5.2 Serialization

- `serde` + `serde_json`: MUSS für Config/Result Serialisierung genutzt werden.
- `serde_yaml`: SOLL für bestehende YAML-Konfigurationen (z.B. `configs/execution_costs.yaml`, `configs/symbol_specs.yaml`) genutzt werden.

### 5.3 Logging/Tracing

- `tracing` + `tracing-subscriber`: MUSS verwendet werden.
- Policy: strukturierte Logs; kein Logging im Hot-Path, das deterministische Runs verändert.

### 5.4 RNG

- RNG: `rand` + `rand_chacha`.
- DEV: `ChaCha*`-RNG MUSS mit `rng_seed` initialisiert werden.
- PROD: Seed SOLL aus OS-RNG kommen, sofern nicht explizit gesetzt.

---

## 6. FFI, Naming & Packaging (Normativ)

### 6.1 Python Wrapper Package

- Python Wrapper Package Name: **`bt`**.

**Hinweis (bewusst):** Der Name ist generisch; Kollisionsrisiko ist im Repo-Kontext akzeptiert, da das Paket nicht als generische Third-Party-Library positioniert ist.

### 6.2 Native Extension Module (PyO3)

- Name des Extension-Moduls (Python Import Name): **`omega_bt`**.

### 6.3 Build Tooling

- Build/Packaging: **maturin**.
- Distribution: **Single Wheel** (Python Wrapper + Rust Extension im selben Wheel).

### 6.4 Entry-Point Signatur

- Public FFI-Funktion: `run_backtest(config_json: &str)`.
- Input: JSON gemäß `OMEGA_V2_CONFIG_SCHEMA_PLAN.md`.
- Output: JSON gemäß Output-Contract bzw. Error-Contract (siehe Abschnitt 7).

---

## 7. FFI Error Contract (Hybrid, Normativ)

Der FFI-Contract unterscheidet strikt zwischen **Setup** (Input/Config) und **Run-Time** (während Backtest).

### 7.1 Python Exceptions (MUSS)

Diese Fehler werden als Python Exceptions über PyO3 propagiert:

- **Config-Parsing-Fehler** (JSON nicht parsebar, Typfehler)
- **Config-Validierung** (Schema-/Constraint-Verletzungen nach Parsing)
- **Invalid Function Arguments** (z.B. leere Strings, null/invalid pointers, nicht erlaubte Werte außerhalb des Config-Kontexts)
- **System-Panics** (Rust Panic wird gefangen und als Exception gemappt; Panic-Nachricht darf keine Secrets enthalten)

### 7.2 JSON Result (MUSS)

Diese Fehler werden **als JSON-Result** zurückgegeben (und gelten dennoch als *hard fail*):

- **Trade-Execution-Fehler**
- **Market-Data-Fehler**
- **Strategy-Fehler**
- **Alle Fehler während des Backtest-Runs**

### 7.3 Normatives Error-Result Shape (Vorschlag A – festgezogen)

Das Error-Result ist ein JSON-Objekt:

- Root-Feld `ok: false`
- Feld `error` als Objekt mit mindestens:
  - `category` (z.B. `market_data|execution|strategy|runtime`)
  - `message` (human-readable, kurz)
  - `details` (optional, strukturierte Zusatzinfos)

**Warum JSON statt Exception?**

- Erlaubt konsistente Fehler-Auswertung in Batch-/Optimizer-Kontexten.
- Hält Python-Wrapper einfach: Erfolg/Fehler ist immer über JSON prüfbar.

---

## 8. Plattform-Support & Build-Matrix (MVP)

### 8.1 Zielplattformen (Wheels)

- **macOS** (x86_64, arm64)
- **Linux** (x86_64)
- **Windows** (x86_64)

### 8.2 Python ABI

- MVP: **CPython 3.12** (konsistent zur Repo-Policy).

---

## 9. Upgrade-Policy (Normativ)

### 9.1 Rust Toolchain

- Policy: wir folgen **latest stable**, aber bleiben deterministisch.
- Umsetzung:
  - `rust-toolchain.toml` ist **konkret gepinnt**.
  - Toolchain-Bumps erfolgen als explizite Änderung (PR), idealerweise zusammen mit CI-Grün.

### 9.2 Rust Dependencies

- `Cargo.toml`: Major-Pins.
- `Cargo.lock`: ist versioniert und ist die eigentliche deterministische Auflösung.

---

## 10. Nicht-Ziele / Guardrails

- Keine Implementierung von Live-Trading/MT5 im V2-Core.
- Keine Abweichung vom Timestamp-Contract (UTC, epoch-ns, Open-Time).
- Keine zusätzliche FFI-Grenze (kein Indikator-/Execution-FFI im Loop).
