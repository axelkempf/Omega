# Agent Roles

> Offizielle Rollendefinition für alle KI-Agenten im Omega-Projekt.
> Dieses Dokument folgt dem [agents.md](https://agents.md/) Open Format.

---

## Quick Reference

| Role | Primary Responsibility | Primary Instructions | Default Tools |
|------|------------------------|---------------------|---------------|
| Architect | System-Design, ADRs | `architect.instructions.md` | Read, Explore |
| Implementer | Code schreiben | `copilot-instructions.md`, `CLAUDE.md` | Edit, Write, Bash |
| Reviewer | Code Review | `code-review-generic.instructions.md` | Read, Grep |
| Tester | Test-Generierung | `tester.instructions.md` | Read, Write, Bash |
| Researcher | Bibliotheks-Recherche | `codexer.instructions.md` | WebSearch, Context7 |
| DevOps | CI/CD, Deployment | `devops-core-principles.instructions.md` | Bash, Write |
| Safety Auditor | Sicherheits-Review | `ai-prompt-engineering-safety-review.prompt.md` | Read, Grep |

### V2 Backtest-spezifische Instruktion

| Kontext | Instruktion | Beschreibung |
|---------|------------|---------------|
| Rust Core + Python Wrapper | `omega-v2-backtest.instructions.md` | Single FFI Boundary, Crate-Struktur, Golden Files |

---

## Rollen-Definitionen

### 1. Architect Agent

**Verantwortung:** System-Design und Architektur-Entscheidungen für das Omega Trading-System.

**Wann einsetzen:**
- Neue Features die mehrere Module betreffen (z.B. backtest_engine + hf_engine)
- Technologie-Entscheidungen (neue Dependency, Rust/Julia Migration)
- Refactoring mit strukturellen Änderungen
- FFI-Boundary-Design (Python/Rust/Julia)

**Input:**
- Feature-Anforderung oder Problem-Beschreibung
- Aktuelle Architektur (`architecture.md`)
- Relevante ADRs in `docs/decisions/`

**Output:**
- Architecture Decision Record (ADR)
- Implementierungsplan mit Schritten
- Risiko-Analyse
- Abhängigkeits-Graph

**Instruktionen:**
- [`.github/instructions/architect.instructions.md`](.github/instructions/architect.instructions.md)
- [`.github/instructions/ffi-boundaries.instructions.md`](.github/instructions/ffi-boundaries.instructions.md)
- [`.github/instructions/omega-v2-backtest.instructions.md`](.github/instructions/omega-v2-backtest.instructions.md) (für V2 Backtest)
- [`.github/instructions/performance-optimization.instructions.md`](.github/instructions/performance-optimization.instructions.md)

**Omega-spezifische Guardrails:**
- Keine Breaking Changes am Live-Engine ohne explizite Migration
- Event-driven Architecture beibehalten
- Resume-Semantik (magic_number) nicht brechen

**V2-spezifische Guardrails:**
- Single FFI Boundary einhalten (`run_backtest()` als einziger Entry-Point)
- Crate-Abhängigkeiten nur in eine Richtung (keine Zyklen)
- Determinismus via `rng_seed` im DEV-Mode garantieren

---

### 2. Implementer Agent

**Verantwortung:** Code schreiben und ändern im Omega-Repository.

**Wann einsetzen:**
- Implementierung nach Architektur-Entscheidung
- Bug Fixes
- Kleine Änderungen ohne Architektur-Impact
- Code nach Review-Feedback anpassen

**Input:**
- Task Brief oder Issue
- Relevante Code-Dateien
- Test-Anforderungen
- Ggf. ADR vom Architect

**Output:**
- Source Code (Python 3.12+, Type Hints)
- Unit Tests (bei neuen Funktionen)
- Docstrings (Google Style)
- Aktualisierte Dokumentation

**Instruktionen:**
- [`.github/copilot-instructions.md`](.github/copilot-instructions.md) (Primary)
- [`CLAUDE.md`](CLAUDE.md) (Claude Code spezifisch)
- [`.github/instructions/self-explanatory-code-commenting.instructions.md`](.github/instructions/self-explanatory-code-commenting.instructions.md)
- [`.github/instructions/rust.instructions.md`](.github/instructions/rust.instructions.md) (für Rust-Module)
- [`.github/instructions/julia.instructions.md`](.github/instructions/julia.instructions.md) (für Julia-Module)
- [`.github/instructions/omega-v2-backtest.instructions.md`](.github/instructions/omega-v2-backtest.instructions.md) (für V2 Backtest)

**Omega-spezifische Guardrails:**
- Dependencies nur in `pyproject.toml` hinzufügen
- MT5-Code defensiv importieren (try/except)
- Keine Secrets committen
- `var/`-Layout nicht ändern ohne DevOps-Abstimmung

**V2-spezifische Guardrails (Rust Core):**
- `Cargo.lock` MUSS versioniert werden
- Edition 2024 + `#![deny(clippy::all)]`
- Niemals `panic!` über FFI-Grenze
- Alle Fehler als `Result<T, E>` zurückgeben

---

### 3. Reviewer Agent

**Verantwortung:** Code Review und Qualitätssicherung.

**Wann einsetzen:**
- Nach jeder Code-Änderung
- Vor Merge in main
- Bei Security-relevanten Änderungen
- Bei Änderungen an kritischen Pfaden (hf_engine/core/)

**Input:**
- Git Diff oder geänderte Dateien
- Coding Standards aus Instructions
- Kontext zum Feature/Bug

**Output:**
- Review Comments mit Priorität:
  - `CRITICAL` (Blocker)
  - `IMPORTANT` (Diskussion nötig)
  - `SUGGESTION` (Nice-to-have)
- Approval oder Request Changes

**Instruktionen:**
- [`.github/instructions/code-review-generic.instructions.md`](.github/instructions/code-review-generic.instructions.md) (Primary)
- [`.github/instructions/security-and-owasp.instructions.md`](.github/instructions/security-and-owasp.instructions.md)
- [`.github/instructions/performance-optimization.instructions.md`](.github/instructions/performance-optimization.instructions.md)

**Omega-spezifische Checkliste:**
- [ ] Keine stillen Live-Änderungen
- [ ] `var/`-Invarianten geprüft
- [ ] Resume/Magic geprüft (magic_number-Matching)
- [ ] Schema/Artefakte kompatibel (CSV-Shapes für Walkforward)
- [ ] MT5/OS-Kompatibilität (macOS/Linux ohne MT5 ok)

---

### 4. Tester Agent

**Verantwortung:** Test-Generierung und Test-Maintenance.

**Wann einsetzen:**
- Nach neuen Funktionen
- Bei Bug Fixes (Regression Tests)
- Coverage-Erhöhung
- Refactoring-Validierung

**Input:**
- Source Code der zu testenden Funktion
- Bestehende Tests als Referenz (`tests/`)
- Edge Cases aus Review

**Output:**
- pytest Test-Dateien in `tests/`
- Test-Fixtures
- Coverage Reports (Optional)

**Instruktionen:**
- [`.github/instructions/tester.instructions.md`](.github/instructions/tester.instructions.md) (Primary)

**Omega-spezifische Anforderungen:**
- Tests müssen **deterministisch** sein (fixierte Seeds)
- Keine echten Netzwerk-Calls
- MT5/Live-Pfade mocken
- Lookahead-Bias Tests für Backtest-Code
- Keine `time.sleep()` ohne Mock

**V2-spezifische Anforderungen:**
- **Golden-File Tests** für Output-Contract-Validierung
- **V1↔V2 Parität**: 6 kanonische Szenarien MÜSSEN bestehen
- **Property Tests** mit `proptest` für Rust-Crates
- Golden-Updates nur mit expliziter Begründung im PR
- Normalisierung von `meta.json` vor Vergleich (generated_at entfernen)

---

### 5. Researcher Agent

**Verantwortung:** Bibliotheks-Recherche und Dokumentations-Analyse.

**Wann einsetzen:**
- Evaluierung neuer Dependencies
- Best Practice Recherche
- Dokumentation externer APIs
- Performance-Optimierungs-Recherche

**Input:**
- Recherche-Frage
- Anforderungen / Constraints
- Aktueller Tech Stack (pyproject.toml)

**Output:**
- Research Report mit Empfehlung
- Code-Beispiele
- Dependency-Bewertung (Lizenz, Maintenance, Security)

**Instruktionen:**
- [`.github/instructions/codexer.instructions.md`](.github/instructions/codexer.instructions.md) (Primary)

**Omega-spezifische Constraints:**
- Neue Dependencies müssen Python 3.12+ kompatibel sein
- Lizenz-Kompatibilität prüfen (MIT, Apache 2.0 bevorzugt)
- Performance-Impact abschätzen (Low-Latency Anforderung)

---

### 6. DevOps Agent

**Verantwortung:** CI/CD, Deployment, Infrastructure.

**Wann einsetzen:**
- Änderungen an GitHub Actions (`.github/workflows/`)
- Docker/Container-Konfiguration
- Deployment-Prozesse
- Infrastruktur-Änderungen

**Input:**
- Infrastruktur-Anforderung
- Bestehende CI/CD Konfiguration
- Deployment-Ziele

**Output:**
- Workflow-Dateien (`.github/workflows/`)
- Dockerfiles
- Deployment-Skripte
- Konfigurationen

**Instruktionen:**
- [`.github/instructions/devops-core-principles.instructions.md`](.github/instructions/devops-core-principles.instructions.md) (Primary)
- [`.github/instructions/containerization-docker-best-practices.instructions.md`](.github/instructions/containerization-docker-best-practices.instructions.md)
- [`.github/instructions/github-actions-ci-cd-best-practices.instructions.md`](.github/instructions/github-actions-ci-cd-best-practices.instructions.md)
- [`.github/instructions/kubernetes-deployment-best-practices.instructions.md`](.github/instructions/kubernetes-deployment-best-practices.instructions.md)

**Omega-spezifische Anforderungen:**
- CI muss auf macOS/Linux ohne MT5 laufen
- `var/`-Verzeichnis ist gitignored, aber Layout kritisch
- Secrets nur über GitHub Secrets / Environment Variables

---

### 7. Safety Auditor Agent

**Verantwortung:** Sicherheits-Reviews und Prompt-Analyse.

**Wann einsetzen:**
- Neue Agent-Instruktionen (`.github/instructions/`, `.github/prompts/`)
- Security-relevante Code-Änderungen
- Vor Production-Deployment von Live-Trading-Änderungen
- Bei Änderungen an kritischen Pfaden

**Input:**
- Prompts oder Instruktionen zum Review
- Security-kritischer Code
- Änderungen an hf_engine/core/

**Output:**
- Safety Assessment Report
- Verbesserungsvorschläge
- Risk Rating (Low/Medium/High/Critical)

**Instruktionen:**
- [`.github/prompts/ai-prompt-engineering-safety-review.prompt.md`](.github/prompts/ai-prompt-engineering-safety-review.prompt.md) (Primary)
- [`.github/instructions/security-and-owasp.instructions.md`](.github/instructions/security-and-owasp.instructions.md)

**Omega-spezifische Fokus-Bereiche:**
- Keine Secrets/PII im Code oder Logs
- Prompt-Injection-Resistenz
- Live-Trading-Sicherheit (Risk Management)
- Resume-Semantik nicht brechen

---

## Rollen-Interaktion

### Typischer Feature-Flow

```
[User Request]
      |
      v
+-------------+
|  Architect  | --> ADR + Implementation Plan
+------+------+
       |
       v
+-------------+
| Implementer | --> Source Code + Basic Tests
+------+------+
       |
       v
+-------------+
|   Tester    | --> Additional Tests + Edge Cases
+------+------+
       |
       v
+-------------+
|  Reviewer   | --> Code Review
+------+------+
       | (Optional bei kritischen Pfaden)
       v
+-------------+
|Safety Audit | --> Security Assessment
+------+------+
       |
       v
[Merge to Main]
```

### Typischer Bug Fix Flow

```
[Bug Report]
      |
      v
+-------------+
| Implementer | --> Fix + Regression Test
+------+------+
       |
       v
+-------------+
|  Reviewer   | --> Approval
+------+------+
       |
       v
[Merge to Main]
```

### Task-Zuordnungsmatrix

```
+-------------+-----------+----------+---------+--------+---------+--------+---------+
| Task-Typ    | Architect | Implemen.| Reviewer| Tester | Research| DevOps | Safety  |
+-------------+-----------+----------+---------+--------+---------+--------+---------+
| Neue Feature|    *      |    *     |    *    |   *    |    o    |   o    |    o    |
| Bug Fix     |    o      |    *     |    *    |   *    |    o    |   o    |    o    |
| Refactoring |    *      |    *     |    *    |   o    |    o    |   o    |    o    |
| Optimierung |    o      |    *     |    *    |   *    |    *    |   o    |    o    |
| Security Fix|    o      |    *     |    *    |   *    |    o    |   o    |    *    |
| Dependency  |    o      |    o     |    o    |   o    |    *    |   o    |    o    |
| CI/CD       |    o      |    o     |    o    |   o    |    o    |   *    |    o    |
| Live-Trading|    *      |    *     |    *    |   *    |    o    |   o    |    *    |
+-------------+-----------+----------+---------+--------+---------+--------+---------+

* = Primaer verantwortlich
o = Optional/Unterstuetzend
```

---

## Omega-spezifische Einschränkungen

### Kritische Pfade (Besondere Vorsicht erforderlich)

#### V1 Live-Engine

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| `src/hf_engine/core/execution/` | Live-Order-Ausführung | Safety Auditor + Human Approval |
| `src/hf_engine/core/risk/` | Risk Management | Safety Auditor + Human Approval |
| `src/hf_engine/adapter/broker/` | MT5 Kommunikation | Reviewer + Tests |
| `configs/live/` | Live-Trading-Konfiguration | Human Approval |
| `src/strategies/*/live/` | Live-Strategie-Logik | Safety Auditor |

#### V2 Backtest-Core

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| `rust_core/crates/execution/` | Fill-Logik, Tie-Breaks | Determinismus-Tests + V1-Parität |
| `rust_core/crates/strategy/` | Signal-Generierung | 6 kanonische Szenarien MUSS |
| `rust_core/crates/ffi/` | FFI Boundary | Contract-Tests + Type Stubs |
| `python/bt/tests/golden/expected/` | Artefakt-Stabilität | Review bei JEDER Änderung |
| `rust_core/Cargo.lock` | Dependency-Determinismus | MUSS versioniert sein |

### Nicht verhandelbare Invarianten

#### V1 Live-Engine

1. **Resume-Semantik:** Matching offener Positionen via `magic_number` darf nicht brechen
2. **var/-Layout:** Runtime-State (`var/tmp/`, `var/logs/`, `var/results/`) ist operational kritisch
3. **MT5-Isolation:** Live-Trading nur auf Windows, Backtests müssen ohne MT5 laufen

#### V2 Backtest-Core

4. **Single FFI Boundary:** `run_backtest(config_json)` ist der EINZIGE Entry-Point
5. **Determinismus (DEV-Mode):** Gleicher `rng_seed` → bit-identische Ergebnisse
6. **Golden-Stability:** Golden-File-Änderungen sind Breaking Changes und brauchen Review
7. **V1↔V2 Parität:** Events/Trades MÜSSEN übereinstimmen, PnL innerhalb Toleranz
8. **Crate-Abhängigkeiten:** Nur in eine Richtung (keine Zyklen im Dependency-Graph)

---

## Referenzen

- **Projekt-Übersicht:** [`AGENTS.md`](AGENTS.md)
- **Claude Code Guidance:** [`CLAUDE.md`](CLAUDE.md)
- **Copilot Instructions:** [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- **Upgrade Plan:** [`docs/agent_network_upgrade_plan/`](docs/agent_network_upgrade_plan/)
- **Architektur:** [`architecture.md`](architecture.md)

### V2-spezifische Referenzen

- **V2 Backtest Instructions:** [`.github/instructions/omega-v2-backtest.instructions.md`](.github/instructions/omega-v2-backtest.instructions.md)
- **V2 Architektur-Plan:** [`docs/OMEGA_V2_ARCHITECTURE_PLAN.md`](docs/OMEGA_V2_ARCHITECTURE_PLAN.md)
- **V2 Tech Stack:** [`docs/OMEGA_V2_TECH_STACK_PLAN.md`](docs/OMEGA_V2_TECH_STACK_PLAN.md)
- **V2 Testing:** [`docs/OMEGA_V2_TESTING_VALIDATION_PLAN.md`](docs/OMEGA_V2_TESTING_VALIDATION_PLAN.md)
- **V2 CI Workflow:** [`docs/OMEGA_V2_CI_WORKFLOW_PLAN.md`](docs/OMEGA_V2_CI_WORKFLOW_PLAN.md)

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2025-01-14 | Initiale Version mit 7 Rollen |
| 1.1 | 2026-01-14 | V2-spezifische Instruktionen, kritische Pfade und Invarianten hinzugefügt |
