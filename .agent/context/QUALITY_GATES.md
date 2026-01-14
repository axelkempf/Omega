# Quality Gates (wann was laufen muss)

## Grundsatz

- **Kleine Änderungen**: Schnelle Gates
- **Risky Änderungen**: Zusätzliche Regression/Golden
- **Contract-Änderungen**: Immer Golden + Parity

---

## Python (V1/V2 Wrapper)

### Pflicht vor PR

```bash
pre-commit run -a
pytest -q
```

### Bei Contract-Änderungen

```bash
pytest tests/golden/ -v
```

---

## Rust (V2 Core)

### Pflicht vor PR

```bash
cd rust_core
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

### Bei Performance-kritischen Änderungen

```bash
cargo bench
```

---

## Contract-/Determinismus-Änderungen

### Golden/Parity Smoke (PR-Gate)

- Kleines Fixture-Dataset
- Vergleich nach Contract-Rundung

### Full Golden (Nightly/Release)

- Vollständiges Fixture-Dataset
- Cross-OS Determinismus
- Benchmarks

---

## Dokumentationspflicht

Wenn geändert wird... | Dann aktualisieren...
---|---
Interface/API | `docs/` + Docstrings
Config-Felder | `CONFIG_SCHEMA_PLAN`, Beispiel-Configs
Output-Schema | `OUTPUT_CONTRACT_PLAN`, Golden-Files
Modul-Struktur | `MODULE_STRUCTURE_PLAN`, `architecture.md`
Workflows | `README.md`, `AGENTS.md`

---

## Gate-Matrix nach Task-Typ

| Task-Typ | pre-commit | pytest | cargo test | Golden | Doku |
|----------|------------|--------|------------|--------|------|
| Bugfix | ✅ | ✅ | ✅ | ○ | ○ |
| Neues Modul | ✅ | ✅ | ✅ | ○ | ✅ |
| Contract-Änderung | ✅ | ✅ | ✅ | ✅ | ✅ |
| Performance | ✅ | ✅ | ✅ + bench | ○ | ○ |
| Doku-only | ✅ | ○ | ○ | ○ | ✅ |

✅ = Pflicht, ○ = Optional/Empfohlen
