# Omega

[![CI](https://github.com/axelkempf/Omega/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/axelkempf/Omega/actions/workflows/ci.yml)
[![Rust Build](https://github.com/axelkempf/Omega/actions/workflows/rust-build.yml/badge.svg?branch=main)](https://github.com/axelkempf/Omega/actions/workflows/rust-build.yml)
[![Julia Tests](https://github.com/axelkempf/Omega/actions/workflows/julia-tests.yml/badge.svg?branch=main)](https://github.com/axelkempf/Omega/actions/workflows/julia-tests.yml)
![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![Version](https://img.shields.io/badge/version-1.5.0-blue)

Ein Python-basierter Trading-Stack mit **Live-Engine (MetaTrader 5)**, **event-getriebenem Backtest/Optimizer** und einer **FastAPI-UI** zur Prozesssteuerung und zum Monitoring.

> Hinweis: Das Projekt enthält produktionsnahe Trading-/Execution-Logik. Nutzung erfolgt auf eigenes Risiko; keine Anlageberatung.

## Was das Projekt macht

- **Live-Trading** über einen MT5-Adapter (`src/hf_engine/`) mit Risiko- und Execution-Schicht.
- **Backtesting & Optimierung** (`src/backtest_engine/`) für Candle- und Tick-Modus, Multi-Timeframe, Walkforward und robuste Final-Selektion.
- **Operations/Monitoring** über eine FastAPI-App (`src/ui_engine/`) inkl. Start/Stop/Restart, Heartbeat-Checks und Log-Streaming.

## Warum das nützlich ist

- **Einheitliche Architektur** für Research → Backtest → Produktion (weniger „Strategy Drift“).
- **Deterministische Backtests** (Threads begrenzt, striktes Timestamp-Alignment, Lookahead-Schutz für höhere TFs).
- **Konfigurationsgetrieben** (JSON/YAML in `configs/`) und reproduzierbare Artefakte unter `var/`.
- **Observability out-of-the-box**: Heartbeats, strukturierte Logs, Ressourcen-Endpunkte.

## Schnellstart (für Developer)

### Voraussetzungen

- Python **>= 3.12**
- macOS/Linux: Backtests/Analyse funktionieren ohne MT5.
- Windows: Für **Live-Trading** wird `MetaTrader5` benötigt (Dependency wird nur unter Windows installiert).

### Installation

**Hinweis:** Alle Dependencies sind zentral in `pyproject.toml` definiert (Single Source of Truth).

**1) Entwicklung (Standard)**

Installiert Runtime-Dependencies plus Dev- und Analyse-Extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev,analysis]
```

**2) Full environment (inkl. ML/Torch)**

Installiert alles inklusive ML-Dependencies:

```bash
pip install -e .[all]
```

**Optionale Extras in `pyproject.toml`:**

| Extra      | Inhalt                                           |
|------------|--------------------------------------------------|
| `dev`      | pytest, black, isort, flake8, mypy, bandit, etc. |
| `analysis` | scipy, scikit-learn, hdbscan, tqdm               |
| `ml`       | torch>=2.1                                       |
| `all`      | Kombiniert dev + analysis + ml                   |

### Backtest ausführen

Beispiel (Single Backtest via Konfig):

- Konfigurationen: `configs/backtest/*.json`
- Runner: `src/backtest_engine/runner.py`

```bash
python src/backtest_engine/runner.py configs/backtest/mean_reversion_z_score.json
```

Ergebnisse landen typischerweise unter `var/results/` (und Logs unter `var/logs/`).

### UI starten (FastAPI)

```bash
uvicorn ui_engine.main:app --reload --port 8000
```

Wichtige Endpunkte (Auszug):

- `POST /start/{name}`
- `POST /stop/{name}`
- `POST /restart/{name}`
- `GET /status/{name}`
- `GET /logs/{account_id}?lines=N`
- `WS /ws/logs/{account_id}`

### Live-/Datafeed-Prozess starten (Launcher)

Der zentrale Entry-Point ist `src/engine_launcher.py`.

```bash
python src/engine_launcher.py --config configs/live/strategy_config_<account_id>.json
```

Der Launcher schreibt Runtime-State unter `var/`:

- Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
- Stop-Signal: `var/tmp/stop_<account_id>.signal`

## Konfiguration

- Live: `configs/live/*.json`
- Backtest: `configs/backtest/*.json` (Validator: `configs/backtest/_config_validator.py`)
- Zentrale Execution-Kosten: `configs/execution_costs.yaml`
- Symbol-Spezifikationen: `configs/symbol_specs.yaml`

### Execution-Multiplikatoren (Backtest/Optimizer)

Optional unter `execution` (Defaults jeweils `1.0`):

- `slippage_multiplier`: skaliert Slippage (z.B. `fixed_pips`, `random_pips`) – inkl. zentraler Defaults aus `configs/execution_costs.yaml`
- `fee_multiplier`: skaliert Gebühren/Commission im Backtest (CommissionModel + Legacy FeeModel)

Diese Multiplikatoren werden u.a. von den Robustness-/Stresstests genutzt (z.B. Cost-Shock), damit Shocks nicht als No-Op enden, wenn keine expliziten Kosten-Sektionen im JSON stehen.

### p-Values (Bootstrap)

Im Final-Selector kann optional `reporting.p_values_net_of_fees` gesetzt werden (Default `true`):

- `true`: `p_net_profit_gt_0` verwendet per-Trade `result - total_fee` (falls Fee-Spalten vorhanden)
- `false`: `p_net_profit_gt_0` verwendet nur `result` (gross)

Hinweis: Die p-Werte sind IID-Bootstrap-Tail-Probabilities (keine Multiple-Testing-Korrektur nach Parameter-Suche).

### .env (optional)

Beim Start wird `.env` automatisch geladen (`python-dotenv`). Relevante Variablen (Auszug, siehe `src/hf_engine/infra/config/environment.py`):

- `ENVIRONMENT=dev|staging|prod`
- `LOG_LEVEL=CRITICAL|ERROR|WARNING|INFO|DEBUG|NOTSET`
- Telegram (optional): `TELEGRAM_ENABLED`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, …
- MT5 (optional): `MT5_ENABLED`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
- Zeitzonen: `SYSTEM_TIMEZONE`, `BROKER_TIMEZONE`

Datafeed optional absicherbar:

- `DATAFEED_API_KEY` (Header `X-API-Key`)
- `DATAFEED_MAX_BARS` (DoS-/Fehlbedienungsschutz)

## Datenlayout

Backtests erwarten Marktdaten unter `data/` (CSV oder Parquet). Namenskonvention (siehe `src/backtest_engine/data/data_handler.py`):

- CSV raw:
  - `data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv`
  - `data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.csv`
- Parquet:
  - `data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet`
  - `data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.parquet`

Schema:

- `UTC time`, `Open`, `High`, `Low`, `Close`, `Volume`

## Projektstruktur (Orientierung)

- `src/engine_launcher.py` – Launcher (Live/Datafeed), Heartbeat/Shutdown
- `src/hf_engine/` – Live-Engine (Adapter, Risk, Execution, Logging)
- `src/backtest_engine/` – Backtests/Optimizer/Walkforward
  - `analysis/` – Post-Processing/Analyzer (Walkforward-Matrix, Backfill, Equity-Plots)
- `src/ui_engine/` – FastAPI UI (Start/Stop/Logs/Resources)
- `src/strategies/` – Strategien + Template
- `configs/` – Live-/Backtest-Konfigurationen + zentrale YAMLs
- `var/` – Runtime-State (gitignored): Logs/Results/tmp

## Rust/Julia High-Performance Extensions

Für performance-kritische Module wird eine Hybrid-Architektur verwendet:

### Rust-Module (PyO3/Maturin) – Migrationsstatus

| Wave | Modul | Status | Feature-Flag | Plan |
|------|-------|--------|--------------|------|
| Wave 0 | Slippage & Fee Costs | ✅ Complete | `OMEGA_USE_RUST_SLIPPAGE_FEE` | [Plan](docs/WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md) |
| Wave 1 | IndicatorCache | ✅ Complete (474x Speedup) | `OMEGA_USE_RUST_INDICATOR_CACHE` | [Plan](docs/WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md) |
| Wave 2 | Portfolio (Position, State) | ✅ Complete | `OMEGA_USE_RUST_PORTFOLIO` | [Plan](docs/WAVE_2_PORTFOLIO_IMPLEMENTATION_PLAN.md) |
| Wave 3 | Event Engine | 📋 Planned | `OMEGA_USE_RUST_EVENT_ENGINE` | [Plan](docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md) |

**Feature-Flag-Werte:** `auto` (default), `true`, `false`

- `auto`: Rust-Backend wird verwendet wenn verfügbar, sonst Python-Fallback
- `true`: Nur Rust-Backend (Error wenn nicht verfügbar)
- `false`: Nur Python-Implementation

**Backend-Status prüfen:**

```python
from src.backtest_engine.core.portfolio import get_rust_status
print(get_rust_status())
# {'available': True, 'enabled': True, 'flag': 'auto', 'error': None}
```

**Voraussetzungen (optional für lokale Entwicklung):**

```bash
# Rust Toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default 1.76.0
```

**Build (wenn Rust-Module aktiv):**

```bash
# Via Makefile
make rust-build

# Via justfile
just rust-build

# Oder direkt mit Maturin
cd src/rust_modules/omega_rust && maturin develop --release
```

**Status:** `src/rust_modules/omega_rust/` enthält Template-Struktur. Migration ausgewählter Module gemäß `docs/runbooks/`.

### Julia-Module (PythonCall.jl)

Monte Carlo, Optimierungen und Research-Analysen können in Julia ausgeführt werden.

**Voraussetzungen (optional):**

```bash
# Julia Installation (macOS)
brew install julia

# Oder via juliaup (empfohlen für Versionsmanagement)
curl -fsSL https://install.julialang.org | sh
juliaup add 1.10
juliaup default 1.10
```

**Julia-Umgebung initialisieren:**

```bash
# Via Makefile
make julia-setup

# Oder manuell
julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.instantiate()'
```

**Status:** `src/julia_modules/omega_julia/` enthält Template-Struktur. Aktiviert für Research-Workflows gemäß `docs/runbooks/`.

### Feature-Flags

Die Python-Fallbacks bleiben vollständig funktional. Rust/Julia-Module werden nur geladen, wenn verfügbar:

```python
# Automatische Detection (geplant)
from omega.config import USE_RUST_INDICATORS, USE_JULIA_MONTE_CARLO

# Defaults: False (Pure Python)
# Wenn Module gebaut und installiert → automatisch True
```

Weitere Details: `docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md` und `docs/runbooks/`.

---

## Hilfe & Doku

- Architekturübersicht: `architecture.md`
- Technische Zusammenfassung: `SUMMARY.md`
- Änderungen/Versionen: `CHANGELOG.md`
- Performance-/Baseline-Report: `reports/performance_baselines/README.md`
- Copilot Agents & Prompts: `AGENTS.md` und `prompts.md`
- **Rust/Julia Migration:** `docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`
- **FFI-Spezifikationen:** `docs/ffi/`
- **Migrations-Runbooks:** `docs/runbooks/`
- Tests als lebende Spezifikation: `tests/`

Fragen/Bugs:

- Bitte ein GitHub Issue erstellen (oder – wenn privat – den Maintainer direkt kontaktieren).

## Maintainer & Contribution

Maintainer: **Axel Kempf** (GitHub: `axelkempf`).

Beiträge sind willkommen – bitte lies zuerst `CONTRIBUTING.md`.

## Lizenz

Es gibt derzeit **keine öffentliche Lizenzdatei** in diesem Repository. Nutzung/Weitergabe nur nach Rücksprache.
