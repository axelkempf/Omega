# Architecture Decision Records (ADRs)

Dieses Verzeichnis enthält Architecture Decision Records (ADRs) für das Omega Trading System.

## Was sind ADRs?

ADRs dokumentieren wichtige architektonische Entscheidungen zusammen mit ihrem Kontext und ihren Konsequenzen. Sie dienen als:

- **Historische Dokumentation**: Warum wurden bestimmte Entscheidungen getroffen?
- **Onboarding-Hilfe**: Neue Team-Mitglieder verstehen die Architektur schneller
- **Entscheidungs-Log**: Vermeidung von wiederholten Diskussionen

## ADR-Index

| ADR | Titel | Status | Datum |
|-----|-------|--------|-------|
| [ADR-0001](ADR-0001-migration-strategy.md) | Rust und Julia Migrations-Strategie | Proposed | 2026-01-03 |

## Template

Neue ADRs sollten dem [ADR-Template](ADR-TEMPLATE.md) folgen.

## Workflow

1. **Proposal**: ADR mit Status "Proposed" erstellen
2. **Review**: Team-Diskussion und Feedback
3. **Decision**: Status auf "Accepted" ändern
4. **Superseded**: Bei späteren Änderungen: neuer ADR mit Verweis auf alten

## Referenzen

- [Michael Nygard's ADR Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
