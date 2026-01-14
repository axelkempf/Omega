# 01 - Agent Roles Definition

> Explizite Definition aller KI-Agent-Rollen im Omega-Projekt

**Status:** 🟢 Abgeschlossen
**Priorität:** Hoch
**Komplexität:** Niedrig
**Geschätzter Aufwand:** 2-4 Stunden

---

## Objective

Definiere klare, spezialisierte Rollen für KI-Agenten, sodass:
- Jeder Agent eine eindeutige Verantwortlichkeit hat
- Agents sich gegenseitig ergänzen, nicht überlappen
- Neue Teammitglieder (menschlich oder KI) sofort verstehen, welcher Agent wofür zuständig ist

---

## Current State

### Problem

Aktuell sind Agent-Rollen **implizit** in verschiedenen Instruktionsdateien verstreut:

| Datei | Implizite Rolle |
|-------|-----------------|
| `codexer.instructions.md` | Researcher + Implementer |
| `code-review-generic.instructions.md` | Reviewer |
| `ai-prompt-engineering-safety-review.prompt.md` | Safety Auditor |
| `CLAUDE.md` / `AGENTS.md` | Generalist |

### Probleme

1. **Keine klare Zuordnung** - Welcher Agent für welche Aufgabe?
2. **Überlappende Verantwortlichkeiten** - Codexer und Copilot haben ähnliche Regeln
3. **Fehlende Rollen** - Kein dedizierter Architect, Tester, oder DevOps Agent

---

## Target State

### AGENT_ROLES.md (Zieldatei im Repo-Root)

```markdown
# Agent Roles

| Role | Primary Responsibility | Tools | Output |
|------|------------------------|-------|--------|
| Architect | System-Design, ADRs | Read, Explore | Design Docs, ADRs |
| Implementer | Code schreiben | Edit, Write, Bash | Source Code |
| Reviewer | Code Review | Read, Grep | Review Comments |
| Tester | Test-Generierung | Read, Write, Bash | Test Files |
| Researcher | Bibliotheks-Recherche | WebSearch, Context7 | Research Reports |
| DevOps | CI/CD, Deployment | Bash, Write | Configs, Pipelines |
| Safety Auditor | Sicherheits-Review | Read, Grep | Security Reports |
```

### Rollenmatrix

```
┌─────────────────────────────────────────────────────────────────┐
│                      Task-Zuordnung                              │
├─────────────┬───────────┬──────────┬────────┬─────────┬─────────┤
│ Task-Typ    │ Architect │ Implemen.│ Reviewer│ Tester │ Research│
├─────────────┼───────────┼──────────┼─────────┼────────┼─────────┤
│ Neue Feature│    ●      │    ●     │    ○    │   ○    │    ○    │
│ Bug Fix     │    ○      │    ●     │    ●    │   ●    │    ○    │
│ Refactoring │    ●      │    ●     │    ●    │   ○    │    ○    │
│ Optimierung │    ○      │    ●     │    ●    │   ●    │    ●    │
│ Security Fix│    ○      │    ●     │    ●    │   ●    │    ○    │
│ Dependency  │    ○      │    ○     │    ○    │   ○    │    ●    │
└─────────────┴───────────┴──────────┴─────────┴────────┴─────────┘

● = Primär verantwortlich
○ = Optional/Unterstützend
```

---

## Implementation Plan

### Schritt 1: AGENT_ROLES.md erstellen

Erstelle eine neue Datei `AGENT_ROLES.md` im Repo-Root:

```markdown
# Agent Roles

> Offizielle Rollendefinition für alle KI-Agenten im Omega-Projekt

## Rollen-Übersicht

### 1. Architect Agent
**Verantwortung:** System-Design und Architektur-Entscheidungen

**Wann einsetzen:**
- Neue Features die mehrere Module betreffen
- Technologie-Entscheidungen (z.B. neue Dependency)
- Refactoring mit strukturellen Änderungen

**Input:**
- Feature-Anforderung oder Problem-Beschreibung
- Aktuelle Architektur (`architecture.md`)

**Output:**
- Architecture Decision Record (ADR)
- Implementierungsplan mit Schritten
- Risiko-Analyse

**Instruktionen:** `.github/instructions/architect.instructions.md`

---

### 2. Implementer Agent
**Verantwortung:** Code schreiben und ändern

**Wann einsetzen:**
- Implementierung nach Architektur-Entscheidung
- Bug Fixes
- Kleine Änderungen ohne Architektur-Impact

**Input:**
- Task Brief oder Issue
- Relevante Code-Dateien
- Test-Anforderungen

**Output:**
- Source Code
- Unit Tests (bei neuen Funktionen)
- Docstrings

**Instruktionen:** `.github/copilot-instructions.md`, `CLAUDE.md`

---

### 3. Reviewer Agent
**Verantwortung:** Code Review und Qualitätssicherung

**Wann einsetzen:**
- Nach jeder Code-Änderung
- Vor Merge in main
- Bei Security-relevanten Änderungen

**Input:**
- Git Diff oder geänderte Dateien
- Coding Standards (`.github/instructions/code-review-generic.instructions.md`)

**Output:**
- Review Comments (CRITICAL / IMPORTANT / SUGGESTION)
- Approval oder Request Changes

**Instruktionen:** `.github/instructions/code-review-generic.instructions.md`

---

### 4. Tester Agent
**Verantwortung:** Test-Generierung und Test-Maintenance

**Wann einsetzen:**
- Nach neuen Funktionen
- Bei Bug Fixes (Regression Tests)
- Coverage-Erhöhung

**Input:**
- Source Code der zu testenden Funktion
- Bestehende Tests als Referenz
- Edge Cases aus Review

**Output:**
- pytest Test-Dateien
- Test-Fixtures
- Coverage Reports

**Instruktionen:** `.github/instructions/tester.instructions.md` (neu)

---

### 5. Researcher Agent
**Verantwortung:** Bibliotheks-Recherche und Dokumentations-Analyse

**Wann einsetzen:**
- Evaluierung neuer Dependencies
- Best Practice Recherche
- Dokumentation externer APIs

**Input:**
- Recherche-Frage
- Anforderungen / Constraints

**Output:**
- Research Report mit Empfehlung
- Code-Beispiele
- Dependency-Bewertung

**Instruktionen:** `.github/instructions/codexer.instructions.md`

---

### 6. DevOps Agent
**Verantwortung:** CI/CD, Deployment, Infrastructure

**Wann einsetzen:**
- Änderungen an GitHub Actions
- Docker/Container-Konfiguration
- Deployment-Prozesse

**Input:**
- Infrastruktur-Anforderung
- Bestehende CI/CD Konfiguration

**Output:**
- Workflow-Dateien
- Dockerfiles
- Deployment-Skripte

**Instruktionen:** `.github/instructions/devops-core-principles.instructions.md`

---

### 7. Safety Auditor Agent
**Verantwortung:** Sicherheits-Reviews und Prompt-Analyse

**Wann einsetzen:**
- Neue Agent-Instruktionen
- Security-relevante Code-Änderungen
- Vor Production-Deployment

**Input:**
- Prompts oder Instruktionen
- Security-kritischer Code

**Output:**
- Safety Assessment Report
- Verbesserungsvorschläge
- Risk Rating

**Instruktionen:** `.github/prompts/ai-prompt-engineering-safety-review.prompt.md`

---

## Rollen-Interaktion

### Typischer Feature-Flow

```
[User Request]
      │
      ▼
┌─────────────┐
│  Architect  │ ──► ADR + Implementation Plan
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Implementer │ ──► Source Code + Tests
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Tester    │ ──► Additional Tests + Coverage
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reviewer   │ ──► Approval / Request Changes
└──────┬──────┘
       │
       ▼
[Merge to Main]
```

### Typischer Bug Fix Flow

```
[Bug Report]
      │
      ▼
┌─────────────┐
│ Implementer │ ──► Fix + Regression Test
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reviewer   │ ──► Approval
└──────┬──────┘
       │
       ▼
[Merge to Main]
```
```

### Schritt 2: Rollen-spezifische Instruktionen erstellen

Erstelle fehlende Instruktionsdateien:

- [ ] `.github/instructions/architect.instructions.md`
- [ ] `.github/instructions/tester.instructions.md`

### Schritt 3: Bestehende Instruktionen aktualisieren

Aktualisiere bestehende Dateien mit Rollen-Referenz:

- [ ] `codexer.instructions.md` → Rolle: Researcher
- [ ] `code-review-generic.instructions.md` → Rolle: Reviewer
- [ ] `copilot-instructions.md` → Rolle: Implementer (Default)

### Schritt 4: AGENTS.md und CLAUDE.md aktualisieren

Füge Verweis auf `AGENT_ROLES.md` hinzu.

---

## Acceptance Criteria

- [ ] `AGENT_ROLES.md` existiert im Repo-Root
- [ ] Alle 7 Rollen sind dokumentiert mit:
  - Verantwortung
  - Wann einsetzen
  - Input/Output
  - Verweis auf Instruktionen
- [ ] Fehlende Instruktionsdateien sind erstellt
- [ ] Bestehende Instruktionen referenzieren ihre Rolle
- [ ] `AGENTS.md` verweist auf `AGENT_ROLES.md`

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Rollen-Overlap bleibt bestehen | Mittel | Niedrig | Klare Abgrenzung in Dokumentation |
| Agenten ignorieren Rollen | Mittel | Mittel | Rollen-Check in Orchestrator (Prio 3) |
| Zu rigide Struktur | Niedrig | Niedrig | "Default" Rolle für Generalisten |

---

## Nächste Schritte nach Implementierung

1. **Orchestrator** kann Rollen-basierte Task-Zuordnung nutzen
2. **Pre-Commit** kann prüfen ob richtige Rolle verwendet wurde
3. **Permissions** können pro Rolle definiert werden
