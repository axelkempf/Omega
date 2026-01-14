# Prompt: Neues Rust-Crate (V2)

## Verwendung

Für: Codex-Max (Builder), Claude Sonnet (Critic)

---

## Prompt

```
Erstelle das Crate `omega_<name>` gemäß OMEGA_V2_MODULE_STRUCTURE_PLAN.md.

## Dateien

- rust_core/crates/<name>/Cargo.toml
- rust_core/crates/<name>/src/lib.rs
- rust_core/crates/<name>/src/<module>.rs
- ...

## Abhängigkeiten

- omega_types (path = "../types")
- <weitere>

## Exports (lib.rs)

- <Struct/Enum/Trait>

## Unit-Tests

- <Test-Case 1>
- <Test-Case 2>

## Constraints

- Keine Panics (Result<T, E> verwenden)
- Deterministische Berechnungen
- Keine externen Abhängigkeiten außer Workspace-Crates
- Einweg-Abhängigkeiten (keine Zyklen)

## Akzeptanzkriterien

- [ ] `cargo build -p omega_<name>` erfolgreich
- [ ] `cargo test -p omega_<name>` grün
- [ ] `cargo clippy -p omega_<name> -- -D warnings` ohne Fehler
- [ ] lib.rs exportiert nur die öffentliche API
```

---

## Critic-Prompt (Claude Sonnet)

```
Prüfe das neu erstellte Crate `omega_<name>` auf:

1. Einweg-Abhängigkeiten (keine Zyklen)
2. Minimale öffentliche API (nur nötige Exports)
3. Error-Handling (keine Panics)
4. Determinismus (keine Systemzeit/Random ohne Seed)
5. Test-Coverage für Kernlogik
6. Konsistenz mit MODULE_STRUCTURE_PLAN

Liefere: Findings + Verbesserungsvorschläge
```
