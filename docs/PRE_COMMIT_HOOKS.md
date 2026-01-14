# Pre-commit Hooks Dokumentation

> Dokumentation aller benutzerdefinierten Pre-commit-Hooks im Omega-Projekt.
> Diese Hooks ergänzen die Standard-Linter (black, isort, flake8, mypy, bandit, pydocstyle).

---

## Übersicht

| Hook | Typ | Blocking? | Beschreibung |
|------|-----|-----------|--------------|
| `pytest-changed` | Test | ✅ Ja | Führt relevante Tests für geänderte Dateien aus |
| `breaking-change-check` | Validation | ✅ Ja | Erkennt API-Breaking-Changes |
| `trading-safety-check` | Safety | ✅ Ja | Prüft Trading-Sicherheitsinvarianten |
| `agent-output-validation` | Quality | ❌ Nein | Validiert Code-Qualität (nur Vorschläge) |
| `architecture-check` | Docs | ❌ Nein | Erinnerung bei Strukturänderungen |

---

## pytest-changed

**Datei:** `scripts/hooks/pytest_changed.py`

### Zweck

Führt pytest nur für Dateien aus, die mit den geänderten Source-Dateien zusammenhängen. Dies ermöglicht schnelleres Feedback als das Ausführen aller Tests.

### Funktionsweise

1. Erkennt alle geänderten Python-Dateien im Commit
2. Sucht zugehörige Testdateien nach folgenden Mustern:
   - `tests/test_<module>.py`
   - `tests/<path>/test_<module>.py`
   - Test-Dateien mit gleichem Präfix
3. Führt pytest für die gefundenen Testdateien aus

### Beispiel

```bash
# Geänderte Datei: src/hf_engine/core/risk_manager.py
# Gefundene Tests: tests/test_risk_manager.py, tests/hf_engine/test_risk_manager.py

$ git add src/hf_engine/core/risk_manager.py
$ git commit -m "Fix risk calculation"
# Hook führt aus: pytest tests/test_risk_manager.py -q --tb=short
```

### Bypass

Wenn Tests temporär übersprungen werden sollen:

```bash
git commit --no-verify -m "WIP: Skip tests"
```

---

## breaking-change-check

**Datei:** `scripts/hooks/breaking_change_check.py`

### Zweck

Erkennt Breaking API Changes und erfordert explizite Bestätigung. Schützt vor versehentlichen API-Änderungen, die andere Module oder Consumer beeinträchtigen könnten.

### Was wird erkannt?

| Änderungstyp | Beispiel |
|-------------|----------|
| Entfernte Funktion | `def calculate_lot_size()` gelöscht |
| Entfernte Klasse | `class TradeManager` gelöscht |
| Entfernte Methode | `class Foo: def bar()` → `bar()` gelöscht |
| Geänderte Signatur | `def foo(a, b)` → `def foo(a, b, c)` (required param) |

### Bypass

Es gibt zwei Wege, den Hook zu umgehen:

1. **Commit-Message mit `BREAKING:` Präfix:**
   ```bash
   git commit -m "BREAKING: Remove deprecated calculate_lot_size function"
   ```

2. **Explizites Skip:**
   ```bash
   git commit --no-verify -m "Refactor internals"
   ```

### Kritische Pfade

Der Hook ist besonders streng bei Änderungen in:

- `src/hf_engine/core/` - Live-Engine Core
- `src/strategies/` - Trading-Strategien
- `src/backtest_engine/` - Backtest-Engine

---

## trading-safety-check

**Datei:** `scripts/hooks/trading_safety_check.py`

### Zweck

Prüft Trading-relevanten Code auf potenzielle Sicherheitsprobleme. Dieser Hook ist der wichtigste für die Production Safety.

### Was wird geprüft?

| Pattern | Risiko | Beispiel |
|---------|--------|----------|
| Hardcoded `magic_number` | Position-Matching bricht | `magic_number = 12345` |
| Hardcoded `lot_size` | Unkontrollierte Positionsgrößen | `lot_size = 1.0` |
| Direktes `order_send()` | Umgeht Risk-Layer | `mt5.order_send(request)` |
| Bare `except:` | Verschluckt kritische Fehler | `except: pass` |
| `time.sleep()` | Blockiert Event-Loop | `time.sleep(5)` |

### Kritische Dateien

Besonders streng geprüft werden:

- `execution_engine.py`
- `risk_manager.py`
- `lot_size_calculator.py`
- `mt5_adapter.py`
- `order_manager.py`
- `position_manager.py`

### Bypass

1. **Commit-Message mit `SAFETY-REVIEWED:` Präfix:**
   ```bash
   git commit -m "SAFETY-REVIEWED: Add emergency shutdown with sleep"
   ```

2. **Inline-Kommentar für spezifische Zeilen:**
   ```python
   time.sleep(1)  # noqa: trading-safety
   ```

3. **Explizites Skip:**
   ```bash
   git commit --no-verify -m "Fix timing issue"
   ```

---

## agent-output-validation

**Datei:** `scripts/hooks/agent_output_validation.py`

### Zweck

Validiert die Qualität von (möglicherweise KI-generiertem) Code. Dieser Hook blockiert **nicht**, sondern gibt Verbesserungsvorschläge.

### Qualitätskriterien

| Metrik | Schwellwert | Beschreibung |
|--------|-------------|--------------|
| Type Hint Coverage | ≥80% | Anteil typisierter Funktionen |
| Docstring Coverage | ≥70% | Anteil dokumentierter öffentlicher Funktionen |

### Beispiel-Output

```
=== Agent Output Quality Report ===

src/hf_engine/core/new_module.py:
  Type hint coverage: 65.0% (below 80% threshold)
  Docstring coverage: 50.0% (below 70% threshold)
  Suggestions:
  - Consider adding type hints to: calculate_risk, process_order
  - Consider adding docstrings to: calculate_risk, validate_input

Overall: 1 file(s) could benefit from improvements
```

### Hinweis

Da dieser Hook non-blocking ist, kann der Commit auch bei Unterschreitung der Schwellwerte durchgeführt werden.

---

## architecture-check

**Datei:** `scripts/hooks/architecture_check.py`

### Zweck

Erinnert daran, `architecture.md` zu aktualisieren, wenn sich die `src/`-Struktur ändert. Hilft dabei, die Dokumentation konsistent mit dem Code zu halten.

### Wann wird getriggert?

- Neue Verzeichnisse unter `src/` werden erstellt
- Neue Python-Module (`__init__.py`) werden hinzugefügt

### Beispiel-Output

```
=== Architecture Documentation Reminder ===

The following new directories/modules were detected in src/:
  - src/agent_orchestrator/agents/
  - src/agent_orchestrator/workflows/

📝 REMINDER: Please consider updating architecture.md to reflect these changes.
This hook is non-blocking - just a friendly reminder!
```

### Hinweis

Da dieser Hook non-blocking ist, kann der Commit auch ohne Aktualisierung von `architecture.md` durchgeführt werden.

---

## Installation & Verwendung

### Pre-commit installieren

```bash
# In aktivierter virtueller Umgebung
pip install pre-commit
pre-commit install
```

### Alle Hooks manuell ausführen

```bash
pre-commit run -a
```

### Einzelnen Hook ausführen

```bash
pre-commit run pytest-changed --all-files
pre-commit run trading-safety-check --all-files
```

### Hooks aktualisieren

```bash
pre-commit autoupdate
```

---

## Troubleshooting

### Hook schlägt fehl, aber die Änderung ist korrekt

1. **Prüfe den Bypass-Mechanismus** (siehe jeweilige Hook-Dokumentation)
2. **Verwende `--no-verify`** für Notfälle:
   ```bash
   git commit --no-verify -m "Emergency fix"
   ```
3. **Dokumentiere im PR** warum der Hook übersprungen wurde

### Hook findet keine Testdateien

Der `pytest-changed` Hook sucht Tests nach Konvention:
- Stelle sicher, dass Tests in `tests/` liegen
- Benenne Tests als `test_<module>.py`

### False Positives bei trading-safety-check

Nutze den Inline-Kommentar für legitime Fälle:

```python
# Legitim: Konfigurationskonstante
DEFAULT_LOT_SIZE = 0.01  # noqa: trading-safety
```

---

## Referenzen

- [Pre-commit Framework](https://pre-commit.com/)
- [Omega AGENTS.md](../AGENTS.md)
- [Omega Coding Standards](.github/copilot-instructions.md)
- [04_precommit_validation.md](agent_network_upgrade_plan/04_precommit_validation.md)
