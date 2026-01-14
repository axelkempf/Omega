# Claude Opus 4.5 – Profil & Best Practices

## Stärken

- **Deep Reasoning**: Komplexe Architektur-Entscheidungen, Trade-offs abwägen
- **Contract-Design**: Output-/Config-/Execution-Contracts präzise spezifizieren
- **Critic-Rolle**: Findet Lücken, Inkonsistenzen, Edge-Cases
- **Langform-Analyse**: ADRs, Specs, Plan-Dokumente
- **Security/Safety Review**: Threat-Modeling, OWASP-Patterns

## Schwächen / Overhead

- **Kosten**: Höher als Sonnet – nicht für triviale Tasks
- **Latenz**: Länger bei sehr großen Kontexten
- **Overkill**: Für einfache Patches/Bugfixes überdimensioniert

## Wann Opus wählen

- Execution/Fees/Stops/Output-Contract (High-Risk)
- Architektur-Entscheidungen, Modul-Grenzen
- ADRs und normative Specs
- Critic-Pass bei kritischen Änderungen
- Performance-Analyse und Optimierungs-Strategie

## Wann NICHT Opus

- Einfache Bugfixes → Copilot
- Bulk-Code-Generation → Codex-Max
- Triviale Doku-Updates → Sonnet

## Prompt-Hinweise für Opus

- **Explizite Constraints**: Opus respektiert präzise Guardrails gut
- **Akzeptanzkriterien**: Klare, prüfbare Bullets
- **„Think step-by-step"**: Bei komplexen Trade-offs hilfreich
- **Critic-Modus**: Explizit bitten, Lücken/Risiken zu finden

## Beispiel-Prompt (Opus als Critic)

```
Du bist Critic für eine Änderung am Execution-Model.

Prüfe auf:
1. Determinismus-Leaks (Netz-Calls, Systemzeit, Random ohne Seed)
2. Semantik-Drift (Bid/Ask, SL/TP-Prioritäten, Fill-Order)
3. Output-Contract-Bruch (Felder, Units, Rundung)
4. Fehlende Tests
5. Secret-Leaks

Änderung:
<code>

Liefere: Risiko-Assessment + konkrete Findings.
```
