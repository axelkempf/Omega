# Contributing

Danke fürs Mitwirken.

Dieses Repository ist ein produktionsnaher Trading-Stack (Live-Execution + Backtests). Änderungen sollten **reproduzierbar**, **deterministisch** (Backtest) und **operational sicher** (Live) sein.

## Development setup

- Python: `>= 3.12`
- Empfehlung: virtuelle Umgebung + editable install

**Hinweis:** Alle Dependencies sind zentral in `pyproject.toml` definiert (Single Source of Truth).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,analysis]
```

Für volle Umgebung inkl. ML:

```bash
pip install -e .[all]
```

## Tests

```bash
pytest -q
```

## Code style

Dieses Repo nutzt `pre-commit`.

```bash
pre-commit run -a
```

## Wichtige Guardrails (bitte beachten)

- **Keine stillschweigenden Verhaltensänderungen** in Trading/Execution.
  - Neue/änderte Logik nach Möglichkeit hinter Config-Flag oder mit klarer Migration.
- **Reproduzierbarkeit**: Backtests müssen deterministisch sein (kein Lookahead/Leakage).
- **Runtime-State liegt unter `var/`** (operational kritisch):
  - Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
  - Stop-Signale: `var/tmp/stop_<account_id>.signal`
  - Logs/Results: `var/logs/`, `var/results/`
- **MT5 ist Windows-only**: Tests/Backtests dürfen MT5 nicht voraussetzen.

## Was in PRs erwartet wird

- Eine kurze Beschreibung *warum* die Änderung nötig ist.
- Tests (neu oder angepasst), wenn produktive Kernpfade betroffen sind.
- Doku-Update bei user-facing Änderungen (z.B. Config-Felder, Output-Schemata).

### PR/Change Checklist (kurz)

- [ ] **Scope klar:** Live-Execution vs. Backtest/Analyse/UI
- [ ] **Keine stillen Live-Änderungen:** Wenn Live betroffen → Config-Flag/Migration + Hinweis im PR
- [ ] **`var/`-Invarianten geprüft:** Heartbeat/Stop-Signal/Logs/Results kompatibel
- [ ] **Resume/Magic geprüft:** `magic_number`-Matching unverändert oder per Regression-Test abgesichert
- [ ] **Schema/Artefakte geprüft:** Walkforward/Optimizer-CSV-Shapes kompatibel oder Migration + Tests
- [ ] **Dependencies korrekt:** Alle Deps in `pyproject.toml`
- [ ] **MT5/OS-Kompatibilität:** macOS/Linux ohne MT5 ok; Windows-only sauber gekapselt
- [ ] **Secrets sicher:** keine Secrets committed; neue ENV-Vars als Placeholder + README/Doku
- [ ] **Qualität:** `pre-commit run -a` und `pytest -q` grün
- [ ] **Doku konsistent:** README/docs/`architecture.md` aktualisiert, falls nötig

## Wie du Hilfe bekommst

- Nutze Issues/PRs für Diskussionen und Kontext.
- Wenn das Repository privat genutzt wird: kontaktiere den Maintainer.

---

## Rust/Julia Contributions (High-Performance Extensions)

### Rust-Module (PyO3/Maturin)

Für numerische Hot-Paths werden Rust-Erweiterungen unter `src/rust_modules/omega_rust/` entwickelt.

**Toolchain-Setup:**

```bash
# Rust installieren (1.76.0+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default 1.76.0

# Komponenten hinzufügen
rustup component add rustfmt clippy
```

**Entwicklungs-Workflow:**

```bash
# Build & install (Development)
make rust-build
# oder: just rust-build

# Tests ausführen
make rust-test

# Linting
make rust-lint

# Benchmarks
make rust-bench
```

**Style Guide für Rust:**

- `rustfmt` für Formatierung (default config)
- `clippy` muss ohne Warnungen passieren
- Fehlerbehandlung via `thiserror` + konvertierbare `PyErr`
- Zero-Copy Datentransfer via Arrow IPC (siehe `docs/adr/ADR-0002-serialization-format.md`)
- Alle public APIs müssen dokumentiert sein (`///` doc comments)
- Benchmark-Vergleich mit Python-Baseline erforderlich (≥10x Speedup Target)

**Checklist für Rust PRs:**

- [ ] `cargo fmt --all -- --check` grün
- [ ] `cargo clippy --all-targets -- -D warnings` grün
- [ ] `cargo test` grün
- [ ] Python-Bindings funktionieren (`maturin develop && pytest tests/rust/`)
- [ ] Benchmark zeigt erwarteten Speedup
- [ ] FFI-Interface dokumentiert in `docs/ffi/`

### Julia-Module (PythonCall.jl)

Für Monte Carlo, Optimierungen und Research unter `src/julia_modules/omega_julia/`.

**Julia-Setup:**

```bash
# Julia installieren (1.10+)
juliaup add 1.10
juliaup default 1.10

# Umgebung initialisieren
make julia-setup
```

**Entwicklungs-Workflow:**

```bash
# Tests ausführen
make julia-test

# REPL mit Omega-Umgebung
make julia-repl
```

**Style Guide für Julia:**

- JuliaFormatter (Standard-Config)
- Funktionen mit `@inbounds` nur wenn bounds-checked im Caller
- Arrow für Datentransfer (analog zu Rust)
- Docstrings für alle exportierten Funktionen

**Checklist für Julia PRs:**

- [ ] Alle Tests grün (`julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.test()'`)
- [ ] Docstrings vorhanden
- [ ] Python-Integration funktioniert (PythonCall)
- [ ] Performance-Baseline dokumentiert

### Serialisierung & FFI

Für den Datenaustausch zwischen Python ↔ Rust/Julia:

**Empfohlenes Format:** Apache Arrow IPC (zero-copy möglich)

**Dokumentation:**

- ADR-0002: `docs/adr/ADR-0002-serialization-format.md`
- FFI-Spezifikationen: `docs/ffi/`
- Error-Handling: `docs/adr/ADR-0003-error-handling.md`

**Schema-Validierung:**

Arrow-Schemas sind in `src/shared/arrow_schemas.py` definiert. Änderungen an Schemas erfordern:

1. Update in Python (`arrow_schemas.py`)
2. Update in Rust (`src/rust_modules/omega_rust/src/schemas.rs`)
3. Update in Julia (`src/julia_modules/omega_julia/src/schemas.jl`)
4. Golden-File Tests aktualisieren

### Build-Kommandos (Zusammenfassung)

| Aufgabe | Makefile | justfile |
| --- | --- | --- |
| Rust build | `make rust-build` | `just rust-build` |
| Rust test | `make rust-test` | `just rust-test` |
| Rust lint | `make rust-lint` | `just rust-lint` |
| Rust bench | `make rust-bench` | `just rust-bench` |
| Julia setup | `make julia-setup` | `just julia-setup` |
| Julia test | `make julia-test` | `just julia-test` |
| All (Python+Rust+Julia) | `make all` | `just all` |
| Clean | `make clean` | `just clean` |

Weitere Details in `docs/rust-toolchain-requirements.md` und `docs/julia-environment-requirements.md`.
