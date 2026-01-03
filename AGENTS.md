# AGENTS.md

> Ein standardisierter Kontext für AI Coding Agents, der dem [agents.md](https://agents.md/) Open Format folgt.

Dieses Repository ist ein **Python-basierter Trading-Stack** mit Live-Engine (MetaTrader 5), event-getriebenem Backtest/Optimizer und FastAPI-UI zur Prozesssteuerung.

---

## Project Overview

| Komponente | Pfad | Beschreibung |
|------------|------|--------------|
| **Live-Engine** | `src/hf_engine/` | MT5-Adapter, Risk-Layer, Execution, EventBus |
| **Backtest-Engine** | `src/backtest_engine/` | Backtests, Optimizer, Walkforward, Final-Selektion |
| **UI-Engine** | `src/ui_engine/` | FastAPI für Start/Stop/Restart, Heartbeat-Checks, Log-Streaming |
| **Strategien** | `src/strategies/` | Trading-Strategien + Template (`_template/`) |
| **Launcher** | `src/engine_launcher.py` | Zentraler Entry-Point (live/datafeed/backtest) |

**Technologien**: Python ≥3.12, pandas, numpy, FastAPI, Optuna, MetaTrader5 (Windows-only für Live)

---

## Setup Commands

### Development Install (empfohlen)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .[dev,analysis]
```

### Full Environment (inkl. ML/Torch)

```bash
pip install -e .[all]
```

### Pre-commit Hooks installieren

```bash
pip install pre-commit
pre-commit install
```

---

## Development Workflow

### Backtest ausführen

```bash
# Single Backtest via Konfig
python src/backtest_engine/runner.py configs/backtest/<name>.json

# Beispiel
python src/backtest_engine/runner.py configs/backtest/ema_rsi_trend_follow_backtest.json
```

### Walkforward Backtest

```bash
python -m src.strategies.<strategy_name>.backtest.walkforward_backtest
```

### UI starten (FastAPI)

```bash
uvicorn src.ui_engine.main:app --reload --port 8000
```

### Live-/Datafeed-Prozess starten

```bash
python src/engine_launcher.py --config configs/live/strategy_config_<account_id>.json
```

### Ergebnisse und Logs

- Ergebnisse: `var/results/`
- Logs: `var/logs/`
- Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
- Stop-Signale: `var/tmp/stop_<account_id>.signal`

---

## Testing Instructions

### Alle Tests ausführen

```bash
pytest -q
```

### Spezifischen Test ausführen

```bash
pytest tests/test_<name>.py -v
```

### Test mit Pattern

```bash
pytest -k "metric" -v
```

### Testabdeckung

```bash
pytest --cov=src --cov-report=term-missing
```

### Testrichtlinien

- Tests müssen **deterministisch** sein (fixierte Seeds, keine Netzwerk-Calls)
- MT5/Live-Pfade dürfen in Tests **nicht vorausgesetzt** werden → mocken
- Backtests testen auf **Lookahead-Bias** und **Reproduzierbarkeit**
- Neue produktive Kernpfade brauchen passende Tests

---

## Code Style

### Formatter und Linter

```bash
# Alle Pre-commit Hooks ausführen
pre-commit run -a

# Nur black
black src/ tests/ analysis/

# Nur isort
isort src/ tests/ analysis/
```

### Konfiguration

- **Black**: Default-Konfiguration
- **isort**: Profil `black` (siehe `pyproject.toml`)

### Konventionen

- PEP 8 Style Guide
- Type Hints für alle öffentlichen Funktionen
- Docstrings nach Google Style
- Decimal für monetäre Berechnungen
- Keine Magic Numbers → Named Constants oder Config

---

## Build and Deployment

### Package bauen

```bash
pip install build
python -m build
```

### CI/CD Pipeline

- Workflow: `.github/workflows/ci.yml`
- Trigger: Push auf `main`, Pull Requests
- Steps: Setup Python 3.11, Install Dependencies, Run Tests

### Deployment Checklist

- [ ] `pre-commit run -a` grün
- [ ] `pytest -q` grün
- [ ] Keine Secrets im Code
- [ ] Doku aktualisiert (README, architecture.md)
- [ ] CHANGELOG.md aktualisiert

---

## Configuration

### Konfig-Dateien

| Typ | Pfad |
|-----|------|
| Live-Strategien | `configs/live/strategy_config_<account_id>.json` |
| Backtest-Strategien | `configs/backtest/*.json` |
| Execution-Kosten | `configs/execution_costs.yaml` |
| Symbol-Specs | `configs/symbol_specs.yaml` |

### Environment Variables (.env)

```bash
# Grundlegend
ENVIRONMENT=dev|staging|prod
LOG_LEVEL=INFO

# MT5 (Windows-only)
MT5_ENABLED=true
MT5_LOGIN=<account>
MT5_PASSWORD=<password>
MT5_SERVER=<server>

# Optional: Telegram
TELEGRAM_ENABLED=false
TELEGRAM_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat_id>

# Optional: Datafeed
DATAFEED_API_KEY=<key>
DATAFEED_MAX_BARS=10000
```

### Marktdaten-Layout

```
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.csv
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.parquet
```

Schema: `UTC time`, `Open`, `High`, `Low`, `Close`, `Volume`

---

## Critical Guardrails ⚠️

### Non-negotiable Production Rules

1. **Runtime-State liegt in `var/`** (operational kritisch)
   - Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
   - Stop-Signale: `var/tmp/stop_<account_id>.signal`
   - Logs/Results: `var/logs/`, `var/results/`

2. **Resume-Semantik darf nicht brechen**
   - Matching offener Positionen via `magic_number` ist Invariante
   - Bei Änderungen: Regression-Test erforderlich

3. **MT5 ist Windows-only**
   - Live-Trading: nur auf Windows mit MetaTrader5
   - Backtests/Analyse: müssen auf macOS/Linux ohne MT5 laufen

4. **Reproduzierbarkeit**
   - Backtests deterministisch (Seeds, keine Netz-Calls)
   - Kein Lookahead-Bias, kein Data-Leakage

5. **Keine stillen Live-Änderungen**
   - Neue Trading-Logik hinter Config-Flag oder mit Migration
   - Explizite Hinweise im PR erforderlich

---

## Dependency Policy

### Single Source of Truth: pyproject.toml

Alle Dependencies sind zentral in `pyproject.toml` definiert.

| Extra      | Inhalt                                           | Verwendung                    |
|------------|--------------------------------------------------|-------------------------------|
| `dev`      | pytest, black, isort, flake8, mypy, bandit, etc. | Entwicklung und CI            |
| `analysis` | scipy, scikit-learn, hdbscan, tqdm               | Analyse-Scripts in `analysis/`|
| `ml`       | torch>=2.1                                       | Machine Learning Research     |
| `all`      | Kombiniert dev + analysis + ml                   | Vollständige Umgebung         |

### Regeln

- Neuer Import in `src/` → **in `pyproject.toml` unter `dependencies` hinzufügen**
- Neuer Import nur in `analysis/` → **in `pyproject.toml` unter `[project.optional-dependencies].analysis` hinzufügen**
- Optionale Dependencies defensiv importieren (try/except mit Fallback)

---

## Pull Request Guidelines

### Title Format

```
[<component>] Brief description
```

Beispiele:
- `[backtest] Add walkforward optimization for EMA strategy`
- `[ui] Fix log streaming endpoint`
- `[live] Add circuit breaker for network failures`

### PR Checklist

- [ ] **Scope klar:** Live-Execution vs. Backtest/Analyse/UI
- [ ] **Keine stillen Live-Änderungen:** Config-Flag/Migration + Hinweis im PR
- [ ] **`var/`-Invarianten geprüft:** Heartbeat/Stop-Signal/Logs/Results kompatibel
- [ ] **Resume/Magic geprüft:** `magic_number`-Matching unverändert oder Regression-Test
- [ ] **Schema/Artefakte geprüft:** CSV-Shapes für Walkforward/Optimizer kompatibel
- [ ] **Dependencies korrekt:** Alle Deps in `pyproject.toml`
- [ ] **MT5/OS-Kompatibilität:** macOS/Linux ohne MT5 ok
- [ ] **Secrets sicher:** Keine Tokens/Keys committed
- [ ] **Qualität:** `pre-commit run -a` und `pytest -q` grün
- [ ] **Doku konsistent:** README/architecture.md aktualisiert

---

## Project Structure Quick Reference

```
src/
├── engine_launcher.py      # Zentraler Entry-Point
├── backtest_engine/        # Backtests, Optimizer, Walkforward
├── hf_engine/              # Live-Engine (MT5 Adapter, Risk, Execution)
├── ui_engine/              # FastAPI (Start/Stop/Logs)
├── strategies/             # Trading-Strategien
│   ├── _template/          # Strategie-Template
│   ├── ema_rsi_trend_follow/
│   ├── mean_reversion_z_score/
│   └── ...
└── watchdog/               # Process Monitoring

configs/
├── backtest/               # Backtest-Konfigurationen (JSON)
├── live/                   # Live-Strategie-Konfigurationen (JSON)
├── execution_costs.yaml    # Zentrale Kosten-Defaults
└── symbol_specs.yaml       # Symbol-Spezifikationen

var/                        # Runtime-State (gitignored)
├── logs/                   # Log-Dateien
├── results/                # Backtest-Ergebnisse
├── tmp/                    # Heartbeats, Stop-Signale
└── archive/                # Archivierte Daten

analysis/                   # Post-Processing, Analyzer-Scripts
tests/                      # pytest Tests
```

---

## Troubleshooting

### MT5-Modul nicht gefunden (macOS/Linux)

**Erwartet**: MT5 ist Windows-only. Backtests/Analyse müssen ohne MT5 funktionieren.

```python
# Korrekte defensive Import-Struktur
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
```

### Tests schlagen fehl mit Timing-Issues

- Prüfe deterministische Seeds in Tests
- Keine `time.sleep()` ohne Mock
- Keine echten Netzwerk-Calls

### Backtest produziert unterschiedliche Ergebnisse

- Prüfe Lookahead-Bias (keine Zukunftsdaten)
- Prüfe Seed-Fixierung für Random-Komponenten
- Prüfe Daten-Alignment bei Multi-Timeframe

### FastAPI startet nicht

```bash
# Prüfe ob Port belegt ist
lsof -i :8000

# Alternative Port verwenden
uvicorn src.ui_engine.main:app --port 8001
```

---

## Additional Resources

- **Vollständige Instructions:** `.github/copilot-instructions.md`
- **Architekturübersicht:** `architecture.md`
- **Änderungshistorie:** `CHANGELOG.md`
- **Contributing Guide:** `CONTRIBUTING.md`
- **Technische Zusammenfassung:** `SUMMARY.md`

---

## Contact

**Maintainer:** Axel Kempf (GitHub: `axelkempf`)

Bei Fragen oder Bugs: GitHub Issue erstellen oder Maintainer kontaktieren.
