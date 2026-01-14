# Claude Sonnet 4.5 – Profil & Best Practices

## Token-Limits (GitHub Copilot)

| Input | Output |
|-------|--------|
| 128K | 16K |

**Implikation:** Gleiche Limits wie Opus – gut für Reviews, Doku, mittlere Patches. Bei längeren Outputs GPT-5.2 oder Codex-Max wählen.

---

## Stärken

- **Balance**: Gutes Reasoning bei moderaten Kosten/Latenz
- **Code-Review**: Patterns erkennen, Inkonsistenzen finden
- **Dokumentation**: Klare, strukturierte Texte
- **Mittlere Komplexität**: Patches, kleinere Module, Doku
- **Critic-Rolle (Medium-Risk)**: Solide zweite Meinung

## Schwächen

- **Tiefe bei sehr komplexen Entscheidungen**: Opus ist besser
- **Bulk-Code**: Codex-Max ist effizienter

## Wann Sonnet wählen

- Code-Review und Critic bei Medium-Risk
- Dokumentation, Pläne, Runbooks
- Mittlere Patches (10–100 LOC)
- Doku-Konsistenz-Checks
- Schnelle Iteration bei klarem Scope

## Wann NICHT Sonnet

- High-Risk Contracts/Execution → Opus
- Bulk-Code + Tests → Codex-Max
- Repo-nahe In-Editor-Arbeit → Copilot

## Prompt-Hinweise für Sonnet

- **Klarer Scope**: Sonnet performt gut bei fokussierten Tasks
- **Akzeptanzkriterien**: Konkret und prüfbar
- **Beispiele geben**: Hilft bei gewünschtem Output-Format

## Beispiel-Prompt (Sonnet als Builder)

```
Implementiere eine Funktion `validate_config(cfg: &BacktestConfig) -> Result<(), ConfigError>`.

Constraints:
- Keine Panics
- Alle Pflichtfelder prüfen (symbol, timeframe, dates)
- Ranges validieren (dates.start < dates.end)
- Rückgabe: Ok(()) oder spezifischer ConfigError

Tests:
- Valid config → Ok
- Missing symbol → Err(MissingField)
- Invalid date range → Err(InvalidRange)
```
