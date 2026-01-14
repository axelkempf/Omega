# Model Routing (Decision Tree)

> Welches Modell für welchen Task – pragmatisch, nicht religiös.

## Decision Tree

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

## Model-Capabilities (GitHub Copilot)

| Modell | Input | Output | Stärke |
|--------|-------|--------|--------|
| **Claude Opus 4.5** | 128K | 16K | Deep Reasoning, Contracts, Critic |
| **Claude Sonnet 4.5** | 128K | 16K | Balance, Review, Doku |
| **GPT-5.2** | 128K | 64K | Research, längere Outputs |
| **GPT-5.1-Codex-Max** | 128K | 128K | Bulk-Code, max Output |

**Output-Implikation:**
- Für große Code-Generierung (ganze Crates, Test-Suites) → **Codex-Max** (128K Output)
- Für lange Analysen/Docs → **GPT-5.2** (64K Output)
- Für präzise, fokussierte Tasks → **Opus/Sonnet** (16K reicht meist)

---

## Task-Matrix (empfohlen)

| Task-Klasse | Builder | Critic | Gates |
|-------------|---------|--------|-------|
| Output-/Config-Contract Änderungen | Claude Opus | Sonnet | Golden/Parity + Contract-Tests |
| Execution/Fees/Stops Semantik | Claude Opus | Sonnet | Determinismus + Regression |
| Neues Rust-Crate (V2) | Codex-Max | Sonnet | cargo test + clippy + fmt |
| Bugfix in bestehendem Code | Copilot | Sonnet | pytest, Repro-Test |
| Architektur-Entscheidung/ADR | Claude Opus | GPT-5.2 | Review + Doku |
| Performance-Optimierung | Claude Opus | Codex-Max | Profiling + Benchmark |
| Dokumentation/Pläne | Claude Sonnet | Opus | Konsistenz-Check |

## Builder + Critic Pairings

| Risk Level | Builder | Critic | Wann |
|------------|---------|--------|------|
| **Critical** (Execution/Contract) | Claude Opus | Claude Sonnet + GPT-5.2 | Immer beide |
| **High** (Neue Module) | Codex-Max | Claude Sonnet | Immer |
| **Medium** (Patches) | Copilot | Claude Sonnet | Bei >50 LOC |
| **Low** (Doku/Typos) | Sonnet | – | Optional |

## Switch-Trigger (wann Modell wechseln)

- **Nach 2 Iterationen ohne Fortschritt** → Modell wechseln
- **Requirements unklar** → zurück zu Spec (Claude Opus)
- **Scope explodiert** → Task aufteilen, neue Task-Briefs
- **Kosten-Optimierung** → Sonnet statt Opus wenn Tiefe nicht nötig

## High-Risk Checklist (muss vor Merge)

- [ ] Determinismus gesichert (Seed/No-Net/No-Clock)
- [ ] Output-Contract eingehalten (Schema/Units/Rundung)
- [ ] Kein Semantik-Drift in Execution (Bid/Ask, SL/TP)
- [ ] Keine Secrets in Code/Logs
- [ ] Tests vorhanden und grün
- [ ] Critic-Pass dokumentiert
