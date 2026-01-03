## AI Coding Agent Instructions (institutional baseline)

Dieses Repository ist ein Python-basierter Trading-Stack (MT5 Live-Adapter, event-getriebener Backtest/Optimizer, FastAPI UI). Dieses Dokument ist die verbindliche, projektspezifische “Operating Procedure” für KI-Agenten und Contributors.

Zusätzlicher, tool-agnostischer Einstiegspunkt: `AGENT.md` im Repo-Root (Kurzfassung + Verweise). Für GitHub Copilot bleibt diese Datei hier die primäre Quelle, da sie automatisch geladen wird.
---

## Priority Guidelines

Bei der Code-Generierung für dieses Repository:

1. **Version Compatibility**: Strikt Python ≥3.12 und exakte Dependency-Versionen aus `pyproject.toml` einhalten
2. **Context Files**: Patterns aus `.github/instructions/` priorisieren
3. **Codebase Patterns**: Bei fehlender Guidance → bestehende Muster im Code suchen und folgen
4. **Architectural Consistency**: Layered, event-driven Architektur und Komponenten-Grenzen wahren
5. **Trading Safety First**: Finanzielle Sicherheit, Risk Management und Datenintegrität priorisieren
6. **Code Quality**: Wartbarkeit, Performance, Sicherheit und Testbarkeit im Fokus

---

## Technology Stack

### Python Version
- **Required**: Python ≥3.12 (spezifiziert in `pyproject.toml`)
- Moderne Type Hints inkl. `|` Union-Syntax, `TypedDict`, `Literal`, `Final`
- `from __future__ import annotations` für Forward References
- `dataclasses` für Datencontainer

### Core Dependencies (aus pyproject.toml)
```
pandas>=2.1          numpy>=1.26         fastapi>=0.109
pydantic>=2.5        optuna>=3.4         joblib>=1.3
matplotlib>=3.8      uvicorn>=0.23       python-dotenv>=1.0
PyYAML>=6.0          requests>=2.31      psutil>=5.9
MetaTrader5>=5.0 (Windows-only)
```

### Development Dependencies
```
pytest>=7.4          httpx>=0.28
black>=24.8.0        isort>=5.13.2
```

### Optional ML/Research Dependencies
```
torch>=2.1           scipy>=1.12
scikit-learn>=1.4    hdbscan>=0.8.33
```

---
## Specialized Instructions

Relevante projektweite Leitlinien (Auswahl):

- [Code Review Generic](instructions/code-review-generic.instructions.md)
- [Security & OWASP](instructions/security-and-owasp.instructions.md)
- [Performance Optimization](instructions/performance-optimization.instructions.md)
- [DevOps Core Principles](instructions/devops-core-principles.instructions.md)
- [Self-Explanatory Code Commenting](instructions/self-explanatory-code-commenting.instructions.md)
- [Markdown Standards](instructions/markdown.instructions.md)
- [Containerization (Docker) Best Practices](instructions/containerization-docker-best-practices.instructions.md)
- [GitHub Actions CI/CD Best Practices](instructions/github-actions-ci-cd-best-practices.instructions.md)
- [Kubernetes Deployment Best Practices](instructions/kubernetes-deployment-best-practices.instructions.md)
### Zielbild (DoD für PRs/Änderungen)
- **Safety first:** Keine stillschweigenden Verhaltensänderungen in Trading/Execution. Neue Logik hinter Config-Flag oder mit klarer Migration.
- **Reproduzierbarkeit:** Backtests/Analysen müssen deterministisch und ohne Lookahead/Leakage sein.
- **Observability:** Änderungen an Logging/Artifacts/Paths sind rückwärtskompatibel oder haben Migration + Doku.
- **Tests & Doku:** Änderungen an produktiven Kernpfaden benötigen passende Tests; Änderungen an Interfaces benötigen Doku-Update.

### Orientierungs-Index (zuerst lesen)
- `src/engine_launcher.py` — zentraler Launcher (live/datafeed/backtest), Heartbeat, Shutdown.
- `src/hf_engine/` — Live-Engine (Adapter, Risk, Execution, EventBus).
- `src/backtest_engine/` — Backtests/Optimierung (Runner + Optimizer-Pipeline).
- `src/ui_engine/` — FastAPI Process Controller (Ops/Monitoring).
- `src/strategies/` — Strategien + Templates (`src/strategies/_template/`).

### Nicht verhandelbare Invarianten (Production Guardrails)
- **Runtime state liegt in `var/`:**
  - Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
  - Stop-Signale: `var/tmp/stop_<account_id>.signal`
  - Logs/Results: `var/logs/`, `var/results/`
- **Resume-Semantik:** Matching offener Positionen via `magic_number` darf nicht brechen (Regression-Test nötig).
- **Path-Helper:** Änderungen an `hf_engine.infra.config.paths` und `var/`-Layout sind operational kritisch.
- **MT5:** `MetaTrader5` ist Windows-only; Live-Runs hängen davon ab, Backtests sollen ohne MT5 auf macOS/Linux laufen.

### Environment / Secrets
- `.env` wird via `python-dotenv` geladen (siehe `src/hf_engine/infra/config/environment.py` und README).
- **Nie Secrets committen** (Tokens, API-Keys, Zugangsdaten). Wenn neue ENV-Vars erforderlich werden: Placeholder in `.env` (lokal) + Doku/README aktualisieren.

### Konfigurations- und Import-Pattern
- Config-driven: JSON in `configs/` (live/backtest). Schlüssel: `module`, `class`, `init_args`, `magic_number`, `data_provider_only`, `data_provider_host`/`port`.
- Strategie-Import: Config referenziert `strategies.<name>.strategy` + Klassenname; neue Strategien an Template spiegeln.
- Datafeed: `data_provider_only: true` startet Datenfeed-Server (Health über UI).

### Dependency-Policy (drift-frei)
Dieses Repo hat **eine Single Source of Truth** für Dependencies: `pyproject.toml`.

| Extra      | Inhalt                                           | Verwendung                    |
|------------|--------------------------------------------------|-------------------------------|
| `dev`      | pytest, black, isort, flake8, mypy, bandit, etc. | Entwicklung und CI            |
| `analysis` | scipy, scikit-learn, hdbscan, tqdm               | Analyse-Scripts in `analysis/`|
| `ml`       | torch>=2.1                                       | Machine Learning Research     |
| `all`      | Kombiniert dev + analysis + ml                   | Vollständige Umgebung         |

Regeln:
- Neuer Import in `src/` → **in `pyproject.toml` unter `dependencies` hinzufügen**
- Neuer Import nur in `analysis/` → **in `pyproject.toml` unter `[project.optional-dependencies].analysis` hinzufügen**
- Optionale Dependencies defensiv importieren (try/except mit Fallback)

### Tests (realistisch, aber strikt)
- Ziel ist nicht “100% Coverage um jeden Preis”, sondern **nachweisbare Risiko-Reduktion**:
  - Unit-Tests für pure Funktionen/Scoring/Utilities
  - Regression-Tests für Invarianten (Heartbeat/Resume/Config parsing)
  - Deterministische Tests (fixierte Seeds, keine Netz-Calls)
- MT5-/Live-Pfade dürfen in Tests nicht vorausgesetzt werden; mocken/abkapseln.

### Dokumentationspflicht
- Bei Änderungen an User-facing Interfaces/Workflows: `README.md` und relevante `docs/` aktualisieren.
- Bei Änderungen an Config-Feldern, Dateistrukturen, Output-CSV-Schemas: Doku + Beispiel-Konfigs anpassen.
- Bei neuen **relevanten Dateien/Modulen** (insb. in `src/`, `src/strategies/`, `src/hf_engine/`, `src/backtest_engine/`, `src/ui_engine/`, `analysis/`, `configs/`) oder bei strukturellen Änderungen an diesen Verzeichnissen: **`architecture.md` aktualisieren**, sodass die Ordner- und Datei-Hierarchie konsistent bleibt (weiterhin ohne Auflistung einzelner `.csv`-Dateien und ohne Inhalte aus `var/results/`).

### Gemeinsame Workflows (Commands)
- Dev install: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev,analysis]`
- Full env (inkl. ML): `pip install -e .[all]`
- Run UI: `uvicorn ui_engine.main:app --reload --port 8000`
- Live/Datafeed: `python src/engine_launcher.py --config configs/live/strategy_config_<id>.json`
- Single backtest: `python src/backtest_engine/runner.py configs/backtest/<name>.json`
- Format (pre-commit): `pre-commit run -a`
- Formatter/Imports: `black` + `isort` (isort-Profil: black; siehe `.pre-commit-config.yaml` und `pyproject.toml`).
- Tests: `pytest -q`

### Optimizer-Subsystem (wichtig für Kompatibilität)
Multi-stage Framework unter `src/backtest_engine/optimizer/`:
- `grid_searcher.py` — kombinatorische Suche / Sampling
- `optuna_optimizer.py` — Bayesian Opt + Pruning
- `walkforward.py` — Walk-forward Validation
- `final_param_selector.py` — robuste Final-Selektion (Dropout, Cost-Shock, TP/SL-Stress)
- `robust_zone_analyzer.py` — stabile Zonen via Clustering
- `instrumentation.py` — `StageRecorder` (Timing, Memory, Artefakte)

**Achtung:** `final_param_selector.py`/`walkforward.py` erwarten konkrete Output-CSV-Shapes, die von `analysis/walkforward_analyzer.py` konsumiert werden. Schema-Änderungen nur mit Migration + Tests.

### Data file naming conventions (gitignored)
Market data folgt festen Namen (siehe `src/backtest_engine/data/data_handler.py`):

**CSV raw**:
```
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.csv
```

**Parquet optimized**:
```
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.parquet
```

**Schema**:
- Columns: `UTC time`, `Open`, `High`, `Low`, `Close`, `Volume`
- `UTC time` timezone-aware (UTC) oder wird beim Load lokalisiert
- `BID`/`ASK` bevorzugt uppercase (fallback vorhanden)

**Timeframes**: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`

### UI Engine (FastAPI) – Architektur
`src/ui_engine/` stellt REST/WebSocket APIs für Prozessmanagement bereit:

**Wichtige Endpoints**:
- `POST /start/{name}` — Start by Alias
- `POST /stop/{name}` — Graceful Stop via Stop-Signal
- `POST /restart/{name}` — Stop + Delay + Start
- `GET /status/{name}` — State + Heartbeat freshness
- `GET /logs/{account_id}?lines=N` — Tail log
- `GET /resource/{name}` — CPU/RAM via psutil
- `WS /ws/logs/{account_id}` — Live log streaming

**Lifecycle**:
1. UI: `controller.start_strategy(alias)` → Config → spawn `engine_launcher.py`
2. Engine: Heartbeat in `var/tmp/heartbeat_<account_id>.txt` (≈ alle 30s)
3. UI: Watchdog prüft Heartbeats, restarts bei Stalls
4. Stop: `var/tmp/stop_<account_id>.signal` → Engine exit gracefully

Wenn etwas unklar ist (z.B. minimaler Strategy-Skeleton oder Beispiel-Backtest-Config), sag explizit, welche Komponente erweitert werden soll.
