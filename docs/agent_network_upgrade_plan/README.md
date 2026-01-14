# Agent Network Upgrade Plan

> Roadmap zur Transformation des Omega KI-Agenten-Netzwerks zu einem institutionellen Standard

## Status Overview

| # | Upgrade | Status | Priorität | Komplexität |
|---|---------|--------|-----------|-------------|
| 1 | [Agent Roles](01_agent_roles.md) | 🟢 Abgeschlossen | Hoch | Niedrig |
| 2 | [Instruction Deduplication](02_instruction_deduplication.md) | 🟢 Abgeschlossen | Hoch | Niedrig |
| 3 | [Orchestrator](03_orchestrator.md) | 🟢 Abgeschlossen | Mittel | Mittel |
| 4 | [Pre-Commit Validation](04_precommit_validation.md) | 🔴 Offen | Mittel | Mittel |
| 5 | [RAG Layer](05_rag_layer.md) | 🔴 Offen | Niedrig | Hoch |
| 6 | [Agent Permissions](06_agent_permissions.md) | 🔴 Offen | Niedrig | Hoch |

**Status-Legende:**
- 🔴 Offen - Noch nicht begonnen
- 🟡 In Arbeit - Teilweise implementiert
- 🟢 Abgeschlossen - Vollständig implementiert und getestet

---

## Ziel

Das Omega-Projekt soll ein **institutioneller Standard** für KI-Agent-basierte Softwareentwicklung werden:

1. **Reproduzierbare Ergebnisse** - Jeder Agent-Aufruf liefert konsistente Outputs
2. **Skalierbare Architektur** - Neue Agents können einfach hinzugefügt werden
3. **Sicherheit** - Least Privilege, Audit-Trails, Output-Validation
4. **Wartbarkeit** - Single Source of Truth, klare Verantwortlichkeiten

---

## Implementierungsreihenfolge

### Phase 1: Foundation (Prio 1)
Diese Upgrades sind schnell umsetzbar und haben hohen Impact:

```
[01_agent_roles.md] ─────────────────────► AGENT_ROLES.md im Repo-Root
                                           (Definition aller Rollen)

[02_instruction_deduplication.md] ───────► Refactoring .github/instructions/
                                           (Konsolidierung redundanter Regeln)
```

### Phase 2: Automation (Prio 2)
Automatisierung und Qualitätssicherung:

```
[03_orchestrator.md] ────────────────────► src/agent_orchestrator/
                                           (Python-basierte Koordination)

[04_precommit_validation.md] ────────────► .pre-commit-config.yaml
                                           (Agent-Output-Checks)
```

### Phase 3: Advanced (Prio 3)
Fortgeschrittene Features für maximale Effizienz:

```
[05_rag_layer.md] ───────────────────────► src/agent_memory/
                                           (Embedding-basierte Suche)

[06_agent_permissions.md] ───────────────► .github/agent_permissions.yaml
                                           (Zugriffskontrolle)
```

---

## Aktueller Stand (Baseline)

### Vorhandene Artefakte

```
.
├── CLAUDE.md                          # Claude Code Entry Point
├── AGENTS.md                          # Standard Agent Format
├── .github/
│   ├── copilot-instructions.md        # GitHub Copilot Hauptinstruktionen
│   ├── instructions/                  # 16 spezialisierte Instruktionen
│   │   ├── codexer.instructions.md
│   │   ├── code-review-generic.instructions.md
│   │   ├── ffi-boundaries.instructions.md
│   │   └── ...
│   └── prompts/                       # 4 Task-spezifische Prompts
│       ├── ai-prompt-engineering-safety-review.prompt.md
│       └── ...
└── agent_tasks/
    └── _TEMPLATE.md                   # Task Brief Template
```

### Gap-Analyse

| Feature | Aktuell | Ziel | Gap |
|---------|---------|------|-----|
| Agent-Rollen | Implizit | Explizit definiert | 01_agent_roles.md |
| Instruktionen | Redundant | Single Source of Truth | 02_instruction_deduplication.md |
| Koordination | Manuell | Automatisiert | 03_orchestrator.md |
| Validation | Git-basiert | Pre-Commit Hooks | 04_precommit_validation.md |
| Suche | Grep/Glob | Embedding-basiert | 05_rag_layer.md |
| Permissions | Keine | Least Privilege | 06_agent_permissions.md |

---

## Erfolgskriterien

Jedes Upgrade-Dokument enthält:

1. **Objective** - Was soll erreicht werden?
2. **Current State** - Wie ist es aktuell?
3. **Target State** - Wie soll es sein?
4. **Implementation Plan** - Konkrete Schritte
5. **Acceptance Criteria** - Wann ist es fertig?
6. **Risks & Mitigations** - Was kann schiefgehen?

---

## Referenzen

- [agents.md Open Format](https://agents.md/)
- [Anthropic Claude Code Documentation](https://docs.anthropic.com/claude/docs/claude-code)
- [GitHub Copilot Custom Instructions](https://docs.github.com/en/copilot/customization)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
