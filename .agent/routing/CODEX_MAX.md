# GPT-5.1-Codex-Max – Profil & Best Practices

## Token-Limits (GitHub Copilot)

| Input | Output |
|-------|--------|
| 128K | 128K |

**Implikation:** **Maximaler Output** – ideal für **ganze Crates, komplette Test-Suites, große Refactorings**. Das ist der Go-To für Bulk-Code-Generierung.

---

## Stärken

- **Bulk-Code-Generation**: Große Mengen Code in einem Rutsch
- **Refactoring**: Konsistente Änderungen über viele Dateien
- **Test-Erzeugung**: Unit-Tests, Property-Tests, Fixtures
- **Boilerplate**: Neue Module/Crates schnell aufsetzen
- **Pattern-Anwendung**: Bekannte Patterns zuverlässig umsetzen

## Schwächen

- **Architektur-Entscheidungen**: Nicht selbständig – braucht klare Spec
- **Tiefes Reasoning**: Claude ist besser bei Trade-offs
- **Edge-Cases**: Kann übersehen werden – Critic nötig

## Wann Codex-Max wählen

- Neue Rust-Crates/Module (V2)
- Große Refactorings mit klarem Pattern
- Test-Suites generieren
- Boilerplate nach Template
- Migrations-Scripts

## Wann NICHT Codex-Max

- Architektur/Contracts → Opus
- Kleine Patches → Copilot
- Debugging → Copilot + Claude

## Prompt-Hinweise

- **Sehr klare Spec**: Codex braucht präzise Vorgaben
- **File-Liste**: Explizit angeben, welche Dateien entstehen sollen
- **Tests mitliefern**: Im selben Prompt Unit-Tests anfordern
- **Patterns referenzieren**: Auf bestehende Repo-Patterns verweisen

## Beispiel-Prompt (Codex-Max als Builder)

```
Erstelle das Crate `omega_trade_mgmt` gemäß OMEGA_V2_MODULE_STRUCTURE_PLAN.md.

Dateien:
- rust_core/crates/trade_mgmt/Cargo.toml
- rust_core/crates/trade_mgmt/src/lib.rs
- rust_core/crates/trade_mgmt/src/engine.rs
- rust_core/crates/trade_mgmt/src/rules.rs
- rust_core/crates/trade_mgmt/src/actions.rs
- rust_core/crates/trade_mgmt/src/views.rs
- rust_core/crates/trade_mgmt/src/error.rs

Abhängigkeiten:
- omega_types (path = "../types")

Exports (lib.rs):
- TradeManager, TradeManagerConfig
- Rule trait, Action enum, CloseReason enum
- PositionView, MarketView, TradeContext

Schreibe Unit-Tests für:
- Rule-Prioritäten (niedrigere gewinnt)
- ClosePosition schlägt ModifyStops

Constraints:
- Keine Panics
- Deterministische Evaluation
- Keine externen Abhängigkeiten außer omega_types
```
