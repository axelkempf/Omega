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
