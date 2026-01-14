# GPT-5.2 – Profil & Best Practices

## Token-Limits (GitHub Copilot)

| Input | Output |
|-------|--------|
| 128K | 64K |

**Implikation:** 4× mehr Output als Claude – gut für **längere Analysen, umfangreiche Dokumentation, ausführliche Reviews**. Für maximalen Code-Output trotzdem Codex-Max (128K).

---

## Stärken

- **Breites Wissen**: Gute Coverage über viele Domains
- **Instruktions-Befolgung**: Folgt strukturierten Prompts zuverlässig
- **Research**: Gut für explorative Fragen, Alternativen-Suche
- **Alternative Perspektive**: Zweite Meinung neben Claude
- **Multi-Domain**: Wenn Task mehrere Fachbereiche berührt

## Schwächen

- **Tiefe bei spezifischen Entscheidungen**: Claude Opus ist präziser
- **Bulk-Code**: Codex-Max ist effizienter
- **Repo-Awareness**: Copilot ist besser integriert

## Wann GPT-5.2 wählen

- Alternative Perspektive bei Architektur-Review
- Research-Phase: Ansätze evaluieren
- Multi-Domain-Tasks (Trading + ML + Infra)
- Critic bei Claude-gebauten Specs

## Wann NICHT GPT-5.2

- High-Risk Execution/Contract → Opus
- Repo-nahe Patches → Copilot
- Bulk-Code → Codex-Max

## Prompt-Hinweise

- **Strukturierte Prompts**: GPT-5.2 folgt klaren Formaten gut
- **Beispiele**: Helfen bei gewünschtem Output
- **Explizite Constraints**: Klar benennen, was nicht erlaubt ist

## Beispiel-Prompt (GPT-5.2 als Critic)

```
Prüfe diesen ADR auf Vollständigkeit und Konsistenz:

<adr>

Fragen:
1. Sind alle Alternativen fair bewertet?
2. Gibt es versteckte Trade-offs?
3. Ist die Entscheidung reversibel?
4. Fehlen Stakeholder-Perspektiven?
```
