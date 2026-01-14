# Omega V2 – Agent Instruction Plan (Copilot / Claude / Codex)

> **Status**: Draft  
> **Erstellt**: 14. Januar 2026  
> **Zweck**: Institutionelle „Operating Procedure" für AI-Agenten, um Omega V2 **maximal effizient**, **sicher** und **deterministisch** umzubauen.

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Agent-System-Architektur (Abschnitt 9) |
| [.agent/README.md](../.agent/README.md) | Agent-Hub Übersicht |
| [.agent/routing/MODEL_ROUTING.md](../.agent/routing/MODEL_ROUTING.md) | Model-Routing Decision Tree |
| [.agent/context/GUARDRAILS.md](../.agent/context/GUARDRAILS.md) | Nicht verhandelbare Invarianten |
| [.agent/context/QUALITY_GATES.md](../.agent/context/QUALITY_GATES.md) | Quality Gates |
| [AGENTS.md](../AGENTS.md) | Tool-agnostischer Einstieg |
| [.github/copilot-instructions.md](../.github/copilot-instructions.md) | Copilot-spezifische Regeln |

---

## 1. Zielbild (warum dieser Plan existiert)

Omega V2 wird zu 100% agenten-getrieben implementiert. Damit das zuverlässig funktioniert, braucht ihr:

- **einheitliche Inputs** (welchen Kontext bekommt das Modell?)
- **einheitliche Outputs** (welche Artefakte müssen entstehen?)
- **harte Guardrails** (Determinismus, Trading-Safety, Secrets, CI-Gates)
- **Model-Routing** (welches Modell macht welchen Job – und wann wird gewechselt?)

Dieser Plan definiert das als **Prozess + Repo-Artefakte**.

---

## 2. Nicht verhandelbare Invarianten (Omega-Guardrails)

Diese Regeln sind aus `AGENTS.md` / `.github/copilot-instructions.md` abgeleitet und gelten in V2 unverändert. Vollständige Details: [.agent/context/GUARDRAILS.md](../.agent/context/GUARDRAILS.md)

1. **Determinismus / Reproduzierbarkeit**
   - Backtests: deterministisch, keine Netz-Calls, Seeds fix, keine Systemzeit in der Logik.
2. **Trading Safety First**
   - Keine stillen Semantik-Änderungen in Execution/Stops/Fees.
   - Jede Verhaltensänderung braucht: Config-Flag oder Migration + Tests.
3. **Runtime-State liegt in `var/`**
   - Pfade/Layouts sind operational kritisch.
4. **MT5 ist Windows-only**
   - Live-Trading-Pfade dürfen Backtests auf macOS/Linux nicht brechen.
5. **Dependency Policy**
   - Single Source of Truth: `pyproject.toml` (Python), `Cargo.toml` (Rust).
   - Neue Imports ⇒ Dependency hinzufügen.
6. **Security-by-default**
   - Keine Secrets in Code/Logs/Docs.
   - Allowlist-Redaction (siehe Observability-Plan).

---

## 3. Arbeitsmodus: „Agenten-First" als standardisierter Loop

### 3.1 Standard-Loop (jede Aufgabe)

1. **Task-Brief erstellen** (siehe `.agent/prompts/_TEMPLATE.md`)
2. **Modell wählen** (siehe `.agent/routing/MODEL_ROUTING.md`)
3. **Kontext packen** (siehe `.agent/context/V2_CONTEXT_PACK.md`)
4. **Builder-Pass** (Implementierung)
5. **Critic-Pass** (Review, mindestens bei risky Änderungen)
6. **Quality Gates** (siehe `.agent/context/QUALITY_GATES.md`)
7. **Dokumentation updaten** (wenn Interface/Config/Contract betroffen)

### 3.2 Definition of Done (DoD) für agentische Änderungen

Eine Änderung gilt als „done", wenn:

- passende Tests existieren (mindestens Unit/Regression für Kernlogik)
- deterministische Ausführung belegt ist (wiederholbarer Run)
- CI-relevante Gates lokal/CI grün sind (pre-commit / pytest / cargo test / clippy)
- Output-Contract/Config-Schema eingehalten ist
- keine Secrets geleakt werden

---

## 4. Model-Routing: welches Modell für welchen Task?

Wichtig: Es gibt kein „bestes Modell" global. Es gibt **bestes Modell pro Task-Klasse**.

Vollständige Details: [.agent/routing/MODEL_ROUTING.md](../.agent/routing/MODEL_ROUTING.md)

### 4.1 Routing-Heuristik (Decision Tree)

```
┌─ Architektur / Policy / Contracts / ADRs?
│   └─ Ja → Claude Opus 4.5 (Critic: Sonnet oder GPT-5.2)
│
├─ Bulk-Code + Tests / Refactor / Neues Modul?
│   └─ Ja → GPT-5.1-Codex-Max (Critic: Claude Sonnet)
│
├─ Repo-nahe Patches / In-Editor Debugging?
│   └─ Ja → GitHub Copilot (Critic: Claude Sonnet)
│
├─ Mittlere Tasks / Doku / Code-Review?
│   └─ Ja → Claude Sonnet 4.5 (Critic: GPT-5.2)
│
└─ Research / Multi-Domain / Alternative Perspektive?
    └─ Ja → GPT-5.2
```

### 4.2 Konkrete Task-Matrix (empfohlen)

| Task-Typ | Builder | Critic | Gates |
|----------|---------|--------|-------|
| Architektur / ADRs | Claude Opus | GPT-5.2 | Review + Doku |
| Output-/Config-Contract | Claude Opus | Sonnet | Golden/Parity |
| Execution/Fees/Stops | Claude Opus | Sonnet | Determinismus + Regression |
| Neues Rust-Crate (V2) | Codex-Max | Sonnet | cargo test + clippy |
| Bugfix | Copilot | Sonnet | pytest, Repro-Test |
| Performance | Claude Opus | Codex-Max | Profiling + Benchmark |
| Dokumentation | Claude Sonnet | Opus | Konsistenz-Check |

**Policy:** Bei jeder Änderung an „Guardrail-Modulen" (Execution/Portfolio/Output/Config/Determinismus) ist ein Critic-Pass Pflicht.

---

## 5. Repo-Artefakte: Agent-Infrastruktur

Die vollständige Agent-Infrastruktur ist in `.agent/` zentralisiert (siehe [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md), Abschnitt 9).

### 5.1 Verzeichnisstruktur

```
.agent/                        ← Zentraler Agent-Hub
├── README.md                  ← Übersicht + Quick Start
├── routing/                   ← Model-Routing + Profile
│   ├── MODEL_ROUTING.md      ← Decision Tree, Task-Matrix
│   ├── CLAUDE_OPUS.md        ← Opus 4.5 Profil
│   ├── CLAUDE_SONNET.md      ← Sonnet 4.5 Profil
│   ├── GPT_5_2.md            ← GPT-5.2 Profil
│   └── CODEX_MAX.md          ← Codex-Max Profil
├── context/                   ← Shared Context
│   ├── GUARDRAILS.md         ← Nicht verhandelbare Invarianten
│   ├── V2_CONTEXT_PACK.md    ← V2-Dokument-Links
│   └── QUALITY_GATES.md      ← Welche Checks wann
└── prompts/                   ← Prompt-Templates
    ├── _TEMPLATE.md          ← Task-Brief Vorlage
    ├── NEW_CRATE.md          ← Neues Rust-Crate
    ├── BUGFIX.md             ← Bugfix-Pattern
    ├── CONTRACT_CHANGE.md    ← High-Risk Contract
    └── PERFORMANCE.md        ← Performance-Optimierung

agent_tasks/                   ← Aktive Task-Briefs
└── _TEMPLATE.md              ← Standard-Vorlage
```

### 5.2 Modell-spezifische Profile

| Modell | Profil | Stärken |
|--------|--------|---------|
| Claude Opus 4.5 | `.agent/routing/CLAUDE_OPUS.md` | Architektur, Contracts, Deep Critic |
| Claude Sonnet 4.5 | `.agent/routing/CLAUDE_SONNET.md` | Balance, Code-Review, Doku |
| GPT-5.2 | `.agent/routing/GPT_5_2.md` | Research, Multi-Domain |
| GPT-5.1-Codex-Max | `.agent/routing/CODEX_MAX.md` | Bulk-Code, Tests, Refactoring |
| GitHub Copilot | `.github/copilot-instructions.md` | In-Editor, Repo-aware |

### 5.3 Copilot-Native Integration

GitHub Copilot lädt automatisch:
- `.github/copilot-instructions.md` (Haupt-Datei)
- `.github/instructions/*.instructions.md` (Glob-Match)

Für andere Modelle (Claude, GPT, Codex) muss Kontext manuell übergeben werden – daher die `.agent/`-Struktur.

### 5.4 Git-Strategie

- `.agent/` **committen** (zentrale Routing/Guardrails sind versioniert)
- `agent_tasks/` **committen** (für Reproduzierbarkeit) oder `.gitignore` (wenn Secrets möglich)
- Task-Briefs **niemals Secrets** enthalten

---

## 6. Der Task-Brief (das wichtigste Prompt-Artefakt)

Modelle performen am besten, wenn der Input wie ein „kleiner, ausführbarer Vertrag" ist.

Template: [.agent/prompts/_TEMPLATE.md](../.agent/prompts/_TEMPLATE.md)

### 6.1 Minimaler Inhalt (MUSS)

- **Objective** (1 Satz)
- **In Scope / Out of Scope**
- **Constraints** (Determinismus, OS, MT5, Secrets)
- **Acceptance Criteria** (prüfbar, 5–10 bullets)
- **Files (Allowlist)**
- **How to verify** (konkrete Tests/Checks)
- **Risks** (Output-Contract, Execution Semantik)
- **Builder / Critic Assignment**

### 6.2 Beispiel-Format

```markdown
## Objective
Implementiere das Crate `omega_trade_mgmt` gemäß TRADE_MANAGER_PLAN.

## In Scope
- TradeManager Engine
- Rule Trait + RuleSet
- Action-Typen
- Unit-Tests

## Out of Scope
- FFI-Binding
- Integration mit backtest

## Constraints
- [x] Deterministisch
- [x] Keine Panics (Result<T, E>)
- [x] Einweg-Abhängigkeiten

## Files
- rust_core/crates/trade_mgmt/**

## How to verify
cargo test -p omega_trade_mgmt
cargo clippy -p omega_trade_mgmt

## Builder: Codex-Max
## Critic: Claude Sonnet
```

---

## 7. Prompting-Patterns

### 7.1 Builder + Critic

- **Builder**: Implementiert strikt nach Acceptance Criteria
- **Critic**: Sucht nach:
  - Determinismus-Leaks
  - Semantik-Drift (Fees/Stops/Reasons)
  - Fehlenden Tests
  - Secret-Leaks
  - Abweichungen von Repo-Patterns

### 7.2 Fail-Fast Inputs

Wenn eine Information fehlt, muss der Agent **explizit nachfragen**:
- „Welche Exit-Preis-Semantik gilt im Candle-Mode für timeout?"
- „Welche Datei ist Single Source of Truth für das Output-Schema?"

### 7.3 Spezifische Prompt-Templates

| Task | Template |
|------|----------|
| Neues Crate | `.agent/prompts/NEW_CRATE.md` |
| Bugfix | `.agent/prompts/BUGFIX.md` |
| Contract-Änderung | `.agent/prompts/CONTRACT_CHANGE.md` |
| Performance | `.agent/prompts/PERFORMANCE.md` |

---

## 8. Sicherheits- und Qualitäts-Gates

Vollständige Details: [.agent/context/QUALITY_GATES.md](../.agent/context/QUALITY_GATES.md)

### 8.1 Security

- Secrets: nur via ENV / Secret Manager, nie hardcoded
- Logs: Allowlist-only Redaction
- Keine netzwerkbasierten Fetches in Tests

### 8.2 Python Gates

```bash
pre-commit run -a
pytest -q
```

### 8.3 Rust Gates

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

### 8.4 Contract-/Determinismus-Änderungen

- Golden/Parity Smoke (PR-Gate)
- Full Golden (Nightly/Release)

---

## 9. Nächste Schritte

1. ✅ `.agent/`-Struktur mit Routing/Context/Prompts angelegt
2. ✅ MODULE_STRUCTURE_PLAN um Agent-Architektur erweitert
3. [ ] Erste Pilot-Tasks im neuen Format durchführen
4. [ ] Issue-Template für agent_task erstellen (`.github/ISSUE_TEMPLATE/`)
5. [ ] Template feinjustieren basierend auf Pilot-Erfahrungen
