# Omega Agent System

> Zentraler Hub für AI-Agent-Konfiguration, Routing und Prompts.

## Struktur

```
.agent/
├── routing/           ← Model-Routing und Stärken-Profile
├── context/           ← Shared Context (Guardrails, Quality Gates)
├── prompts/           ← Prompt-Templates (copy/paste)
└── README.md          ← Dieses Dokument
```

## Quick Start

1. **Task-Typ identifizieren** → `routing/MODEL_ROUTING.md`
2. **Modell-spezifische Hinweise** → `routing/<MODEL>.md`
3. **Guardrails prüfen** → `context/GUARDRAILS.md`
4. **Prompt-Template wählen** → `prompts/`
5. **Task-Brief erstellen** → `agent_tasks/_TEMPLATE.md`

## Modell-Übersicht

| Modell | Datei | Primary Use |
|--------|-------|-------------|
| Claude Opus 4.5 | `routing/CLAUDE_OPUS.md` | Architektur, Contracts, Deep Critic |
| Claude Sonnet 4.5 | `routing/CLAUDE_SONNET.md` | Mittlere Tasks, Code-Review, Doku |
| GPT-5.2 | `routing/GPT_5_2.md` | Research, Multi-Domain, Alternative Critic |
| GPT-5.1-Codex-Max | `routing/CODEX_MAX.md` | Bulk-Code, Refactoring, Test-Suites |
| GitHub Copilot | `.github/copilot-instructions.md` | In-Editor, Repo-aware Patches |

## Prinzipien

- **Single Source of Truth**: Shared Guardrails in `context/`, nicht dupliziert.
- **Builder + Critic**: Bei High-Risk immer zwei Modell-Pässe.
- **Determinismus**: Keine Netz-Calls, keine Systemzeit, Seeds fix.
- **Versioniert**: Alles in Git, damit Routing/Prompts reproduzierbar sind.

## Verwandte Dokumente

- [OMEGA_V2_AGENT_INSTRUCTION_PLAN.md](../docs/OMEGA_V2_AGENT_INSTRUCTION_PLAN.md)
- [AGENTS.md](../AGENTS.md)
- [.github/copilot-instructions.md](../.github/copilot-instructions.md)
