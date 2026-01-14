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
- [`.github/instructions/performance-optimization.instructions.md`](.github/instructions/performance-optimization.instructions.md)

**Omega-spezifische Guardrails:**
- Keine Breaking Changes am Live-Engine ohne explizite Migration
- Event-driven Architecture beibehalten
- Resume-Semantik (magic_number) nicht brechen

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

**Omega-spezifische Guardrails:**
- Dependencies nur in `pyproject.toml` hinzufügen
- MT5-Code defensiv importieren (try/except)
- Keine Secrets committen
- `var/`-Layout nicht ändern ohne DevOps-Abstimmung

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

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| `src/hf_engine/core/execution/` | Live-Order-Ausführung | Safety Auditor + Human Approval |
| `src/hf_engine/core/risk/` | Risk Management | Safety Auditor + Human Approval |
| `src/hf_engine/adapter/broker/` | MT5 Kommunikation | Reviewer + Tests |
| `configs/live/` | Live-Trading-Konfiguration | Human Approval |
| `src/strategies/*/live/` | Live-Strategie-Logik | Safety Auditor |

### Nicht verhandelbare Invarianten

1. **Resume-Semantik:** Matching offener Positionen via `magic_number` darf nicht brechen
2. **var/-Layout:** Runtime-State (`var/tmp/`, `var/logs/`, `var/results/`) ist operational kritisch
3. **Determinismus:** Backtests müssen reproduzierbar sein (Seeds, keine Netz-Calls)
4. **MT5-Isolation:** Live-Trading nur auf Windows, Backtests müssen ohne MT5 laufen

---

## Referenzen

- **Projekt-Übersicht:** [`AGENTS.md`](AGENTS.md)
- **Claude Code Guidance:** [`CLAUDE.md`](CLAUDE.md)
- **Copilot Instructions:** [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- **Upgrade Plan:** [`docs/agent_network_upgrade_plan/`](docs/agent_network_upgrade_plan/)
- **Architektur:** [`architecture.md`](architecture.md)

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2025-01-14 | Initiale Version mit 7 Rollen |
