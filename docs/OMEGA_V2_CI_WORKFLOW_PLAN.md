# Omega V2 – CI Workflow Plan

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation des GitHub Actions CI/CD Workflows für Omega V2 (Rust Core + Python Wrapper): Qualitäts-Gates, Build-Matrix, Security-Checks, Artefakte und Release-Assets  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite; Implementierungs- und Policy-Referenz für `.github/workflows/` (ohne Live-/MT5-Bezug)

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Erfolgskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Single FFI Boundary, Modul-Grenzen |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Data Loading Phasen, Timestamp-Contract, Fail-Fast Policies |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema (run_mode/data_mode/rng_seed) |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell (Determinismus, Fees/Slippage Semantik) |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Rust Workspace/Crates, Tests je Crate |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Artefakt-Contract (`trades.json`, `equity.csv`, `metrics.json`, `meta.json`) |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Metrik-Keys/Units/Rundung (Golden-File Stabilität) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchain/Pinning, PyO3+maturin, OS-Support, Error-Contract |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für die Omega V2 CI/CD-Pipeline.

**Normative Ziele (MUSS):**

- **Qualitäts-Gates vor Merge:** Formatierung, Linting, Tests und Wheel-Builds müssen reproduzierbar und deterministisch prüfbar sein.
- **Single Source of Truth für Policies:** CI implementiert die im Tech-Stack-Plan definierte Toolchain-/Packaging-Strategie (PyO3 + maturin) und respektiert den Config-/Output-Contract.
- **Security „Shift-Left“:** Dependency Review + CodeQL + `pip-audit` + `cargo audit` sind Bestandteil des Plans.
- **V2-only:** CI-Plan betrifft **ausschließlich** den Omega V2 Backtest-Core (Rust Workspace + Python Wrapper). Live-/MT5-Pfade sind explizit out-of-scope.

**Bewusste Entscheidungen (festgezogen):**

- Trigger: `pull_request` und `push` auf `main`.
- Docs-only Änderungen werden für diesen Workflow **geskippt**.
- Concurrency ist aktiv und bricht alte Runs pro Branch ab.
- Python-Tests laufen auf **3.12 und 3.13**.
- Wheels werden als **Full-Matrix** bereits auf PRs gebaut.
- Golden-File Regression läuft **nur nightly + release**.
- Release publiziert Wheels als **GitHub Release Assets** (kein PyPI).

---

## 2. Geltungsbereich

### 2.1 In Scope

- GitHub Actions Workflow(s) für Omega V2 unter `.github/workflows/`.
- Gates:
  - Python: `pre-commit` + `pytest` (3.12/3.13)
  - Rust: `cargo fmt --check`, `cargo clippy`, `cargo test`
  - Packaging: `maturin` Wheel-Builds (OS/Arch-Matrix)
- Security Checks:
  - GitHub Dependency Review
  - GitHub CodeQL
  - `pip-audit`
  - `cargo audit`
- Artefakte:
  - Wheels, Logs/Reports, optionale Test-Reports (Retention 7 Tage)
- Release:
  - Tag → Build Wheels → Upload als GitHub Release Assets

### 2.2 Out of Scope

- Live-Trading, MT5, UI-Engine CI (separat behandeln).
- Deployment in Produktionsumgebungen (Kubernetes/Docker/Infra).
- Golden-File Inhalt/Erzeugungstaktik im Detail (wird in `OMEGA_V2_TESTING_VALIDATION_PLAN.md` normiert; hier nur Scheduling/Integration).

---

## 3. Workflow-Namen, Dateien und Abgrenzung

### 3.1 Workflow-Datei (Normativ)

- Workflow-Dateiname: `.github/workflows/omega-v2-ci.yml`
- Optionaler separater Release-Workflow (wenn gewünscht): `.github/workflows/omega-v2-release.yml`

**Normativ:** Der V2-Workflow darf bestehende V1-Workflows nicht verändern oder „übernehmen“. V2 ist ein eigenständiger Track.

### 3.2 Pfad-Filter (Docs-only Skip)

Docs-only Änderungen werden vom V2-Workflow ignoriert.

**Normativ:** Der Workflow nutzt `paths-ignore`, z.B.:

- `docs/**`
- `**/*.md`

---

## 4. Trigger & Concurrency (Normativ)

### 4.1 Trigger

- `pull_request`
- `push` auf `main`

### 4.2 Concurrency

- `group`: Workflow-Name + Branch
- `cancel-in-progress: true`

**Warum:** Schnelles Feedback auf PRs, kein Ressourcenverbrauch durch veraltete Runs.

---

## 5. Build- und Test-Matrix (Normativ)

### 5.1 Python Test Matrix

- Python: `3.12`, `3.13`
- OS: `ubuntu-latest`

**Begründung:** PR-Feedback schnell halten, aber Forward-Compatibility durch 3.13 absichern.

### 5.2 Wheel Build Matrix (Full auf PR)

Wheels werden auf PRs bereits in voller Zielmatrix gebaut.

**Zielplattformen (siehe Tech-Stack-Plan):**

- Linux x86_64 (manylinux)
- Windows x86_64
- macOS x86_64 und arm64

**Normativer Runner-Mapping-Vorschlag (GitHub Hosted):**

- macOS arm64: `macos-14`
- macOS x86_64: `macos-13`
- Linux x86_64: `ubuntu-latest`
- Windows x86_64: `windows-latest`

**Python ABI:**

- CI: Wheels werden für **CPython 3.12 und 3.13** gebaut.
- MVP bleibt CPython 3.12 (Tech-Stack-Plan); 3.13 Wheels dienen CI-Futuresicherheit und können als „preview“ betrachtet werden.

---

## 6. Jobs (Normativ)

> Hinweis: Der V2 Rust-Core liegt gemäß Modul-Struktur-Plan unter `rust_core/`. Solange dieser Ordner noch nicht existiert, müssen Rust-/maturin-Jobs in der realen Workflow-Implementierung konditional sein (z.B. via `hashFiles('rust_core/Cargo.toml')`).

### 6.1 Job: `python_precommit`

**MUSS:** `pre-commit run -a` auf Ubuntu.

- Purpose: Code-Style Gate und schnelle statische Checks.
- Keine Netzwerk-Secrets.

### 6.2 Job: `python_tests`

**MUSS:** `pytest` auf Ubuntu im Matrix-Modus (3.12, 3.13).

- Deterministisch: keine Netz-Calls, keine MT5-Abhängigkeit.
- SOLL: `pytest -q` + ggf. JUnit-Report als Artifact.

### 6.3 Job: `rust_checks`

**MUSS:** (wenn `rust_core/` existiert)

- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test`

### 6.4 Job: `security_dependency_review`

**MUSS (PR only):**

- GitHub Dependency Review Action (blockiert riskante Dependency-Änderungen)

### 6.5 Job: `security_audit`

**MUSS:**

- Python: `pip-audit` (gegen die installierte Environment / Lock-Resolution)
- Rust: `cargo audit` (wenn `Cargo.lock` vorhanden)

### 6.6 Job: `security_codeql`

**MUSS:**

- CodeQL Scans für Python und Rust.

**Normativ:** CodeQL ist ein Gate, sofern Findings in „critical/high“ vorliegen (konkret in Workflow-Policy festlegen).

### 6.7 Job: `build_wheels`

**MUSS:** (wenn V2 Build-Artefakte existieren)

- Build via `maturin` (PyO3/maturin, siehe Tech-Stack-Plan)
- Full-Matrix (OS/Arch) bereits auf PRs
- Python: 3.12 und 3.13

**SOLL:** Upload der Wheels als Workflow-Artifacts.

### 6.8 Job: `golden_files` (nightly + release)

**MUSS (nur nightly + release):**

- Golden-File Regression gegen kleines, versioniertes Fixture-Dataset im Repo.
- Vergleich folgt Output-Contract + Metrics-Definition (Rundung/Units).

---

## 7. Caching Policy (SOLL, aber stark empfohlen)

### 7.1 Python Cache

- Cache `pip`/`uv`/`poetry` abhängig vom Setup (Repo nutzt `pyproject.toml`).
- Key: OS + Python-Version + Hash von `pyproject.toml`.

### 7.2 Rust Cache

- Cache `~/.cargo/registry`, `~/.cargo/git` und `rust_core/target`.
- Key: OS + Hash von `rust_core/Cargo.lock`.

---

## 8. Security & Permissions (Normativ)

- `permissions` default: `contents: read`.
- Erhöhung nur für Jobs, die es benötigen:
  - Dependency Review / CodeQL benötigen zusätzliche Rechte gemäß GitHub Doku.
- Keine Secrets in Logs.
- Keine „floating“ Actions (`@main`/`@master`).

---

## 9. Artefakte & Retention (Normativ)

### 9.1 Was wird hochgeladen?

**MUSS:**

- Wheels (für Debugging/Validierung) aus `build_wheels`.
- Test-Reports/Logs, sofern vorhanden.

### 9.2 Retention

- Retention: **7 Tage**.

---

## 10. Release Flow (Normativ)

### 10.1 Tag-Schema

**Normativ:** Omega V2 Releases werden über Tags ausgelöst:

- `omega-v2-v<MAJOR>.<MINOR>.<PATCH>`

Beispiel:

- `omega-v2-v0.1.0`

### 10.2 Release Artefakte

**MUSS:**

- Build Wheels für die Zielplattformen.
- Upload als GitHub Release Assets.

**Nicht-Ziel (festgezogen):** Veröffentlichung auf PyPI.

---

## 11. Akzeptanzkriterien (Definition of Done für CI-Plan)

Ein PR, der V2-Core-Code ändert, ist nur mergebar, wenn:

- `python_precommit` grün ist
- `python_tests` (3.12 + 3.13) grün ist
- `rust_checks` grün ist (sofern V2 Rust-Core vorhanden)
- `security_dependency_review` grün ist
- `security_audit` grün ist
- `security_codeql` keine blockierenden Findings hat
- `build_wheels` grün ist (Full-Matrix)

Golden-File Regression ist kein PR-Gate (nightly/release-only).
