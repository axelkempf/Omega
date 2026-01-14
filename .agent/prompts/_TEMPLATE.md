# Task Brief Template

> Kopieren und ausfüllen für jeden Agent-Task.

---

## Objective

<1 Satz: Was soll erreicht werden?>

## In Scope

- …

## Out of Scope

- …

## Constraints (Guardrails)

- [ ] Deterministisch (no net, no clock, seed fix)
- [ ] Keine Secrets in Code/Logs
- [ ] Dependencies nur via `pyproject.toml` / `Cargo.toml`
- [ ] MT5 nicht voraussetzen
- [ ] Output-Contract einhalten

## Relevante V2-Dokumente

- `docs/OMEGA_V2_<TOPIC>_PLAN.md`

## Files (Allowlist)

- `<path/to/file>`

## Acceptance Criteria

- [ ] …
- [ ] …
- [ ] …

## How to verify

```bash
# Tests
pytest tests/…
# oder
cargo test -p <crate>
```

## Risks

- …

## Rollback

- …

---

## Builder/Critic Assignment

- **Builder**: <Model>
- **Critic**: <Model>
