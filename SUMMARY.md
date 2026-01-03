# Kempf Capital Algorithmus – Executive Summary

Diese Datei fasst das Repository strukturiert und objektiv zusammen. Sie beschreibt Komponenten, Datenflüsse, Kernlogik, öffentliche Schnittstellen und Konfigurationen, ohne Bewertung.

## Überblick

- Zweck: Vollständiger Handels‑Stack bestehend aus Live‑Engine (MT5), Backtest‑/Optimierungs‑Pipeline sowie UI/REST‑Steuerung und Datafeed‑Service.
- Einstiegspunkte: `src/engine_launcher.py` (Live/Datafeed Prozesse), `src/ui_engine/main.py` (UI‑API), `src/backtest_engine/runner.py` (Backtests).
- Paketstruktur (Top‑Level):
  - Live‑Engine: `src/hf_engine/*`
  - Backtests/Optimierung: `src/backtest_engine/*`
  - UI/Prozesssteuerung: `src/ui_engine/*`
  - Strategien: `src/strategies/*`
  - Konfiguration: `configs/*`
  - Daten/Ergebnisse (zur Laufzeit): `var/*` (automatisch angelegt, git‑ignored)

## Betriebsmodi

- Live‑Handel (MT5): Strategien laufen gegen Brokeradapter und Datenprovider; Start via Launcher.
- Datafeed‑Server: FastAPI‑Service für OHLC/Tick‑Daten; kann separat laufen (remote/local Provider).
- Backtests: Event‑getriebene Candle‑ oder Tick‑Simulation, Single‑/Multi‑Symbol, Multi‑Timeframe.
- Optimierung/Walkforward: Parameter‑Suche (Grid/Optuna) und robuste Auswertungen.

## Datenflüsse

- Live‑Pfad
  1) Broker‑ und Datenprovider‑Initialisierung (MT5, optional RemoteDataProvider)
  2) Strategien generieren Signale; Risk‑Gates und ExecutionEngine prüfen/setzen Orders
  3) Logging (System/Trades/Orders) und Heartbeats; optionale Telegram‑Benachrichtigungen

- Backtest‑Pfad
  1) Laden historischer Daten (CSV/Parquet/Ticks) pro Symbol/TF mit Market‑Hours‑Filter
  2) Zeitstempel‑Alignment (Primary TF strikt; höhere TFs look‑ahead‑sicher „previous completed“)
  3) Event‑Engines treiben Strategie und ExecutionSimulator; Portfolio aktualisiert Equity/PNL
  4) Ergebnisse/Logs werden in `var/results` bzw. `var/logs` gespeichert

## Komponenten (hochgeordnet)

- Launcher: `src/engine_launcher.py`
  - Startet Account‑ oder Datafeed‑Prozesse (Multiprocessing, spawn).
  - Heartbeats (`var/tmp/heartbeat_<id>.txt`), Stop‑Signale (`var/tmp/stop_<id>.signal`).
  - Initialisiert LogService, lädt News‑Cache, orchestriert geordneten Shutdown.

- UI/Controller: `src/ui_engine/*`
  - API‑App: `src/ui_engine/main.py` (FastAPI, CORS, Lifespan/Watchdog, WS‑Logstream)
  - Prozess‑Steuerung: `src/ui_engine/controller.py` (Start/Stop/Status, Heartbeat‑Auswertung, Ressourcen)
  - Manager: `datafeeds/*.py`, `strategies/*.py` (Factory/Mappings/Aliase)
  - Registry: `registry/strategy_alias.py` (Alias → technische ID)
  - Modelle/Utils: `models.py`, `utils.py`, `config.py`

- Datafeed‑Service: `src/hf_engine/adapter/fastapi/mt5_feed_server.py`
  - Endpunkte für OHLC‑Serien, einzelne geschlossene Kerzen, Zeitbereiche, Ticks, Health.
  - Optionaler API‑Key‑Header (`DATAFEED_API_KEY`).
  - Initialisierung via `DATAFEED_CONFIG` (Launcher schreibt Konfigurationsdatei).

- Live‑Engine: `src/hf_engine/*`
  - Brokeradapter: `adapter/broker/` (Interface, MT5Adapter, Utils, FSM)
  - Datenprovider: `adapter/data/` (Interface, MT5DataProvider, RemoteDataProvider)
  - Execution: `core/execution/` (ExecutionEngine, Tracker, SL/TP‑Hilfen, Session‑State)
  - Risiko: `core/risk/` (RiskManager, LotSizeCalculator, News‑Filter)
  - Controlling: `core/controlling/` (EventBus, StrategyRunner, MultiStrategyController, SessionRunner)
  - Infrastruktur/Logging/Monitoring: `infra/*` (LogService, Pfade, Environment, Health‑Server, Telegram)

- Backtest‑Engine: `src/backtest_engine/*`
  - Core: `core/` (EventEngines für Single/Cross‑Symbol/Tick, ExecutionSimulator, Portfolio, Slippage/Fee‑Modelle)
  - Daten: `data/` (Candle/Tick‑Modelle, CSV/Parquet‑Loader, Market‑Hours, Tick‑Loader)
  - Strategy‑Wrapper/Filter: `strategy/` (Wrapper, Session‑Filter, Zeitfenster‑Utils, Validatoren)
  - Sizing & Gebühren: `sizing/` (LotSizer, CommissionModel, FX‑Rate‑Provider, SymbolSpecs‑Registry)
  - Reporting/Result‑Saver: `report/` (Metrics, TP/SL‑Stress, Result‑Exporter)
  - Optimizer: `optimizer/` (Grid-Searcher, Optuna, Walkforward, Final-Param-Selector, Robust-Zone-Analyzer, Instrumentation/StageRecorder, Symbol-Grid)

- Strategien: `src/strategies/*`
  - Basisklassen: `_base/*` (Strategy, Position‑Manager, Szenarien)
  - Template: `_template/*` (Vorlage für neue Strategien mit Live/Backtest-Struktur)
  - Konkrete Strategien (jeweils mit `live/` und `backtest/` Modulen):
    - `ema_rsi_trend_follow` – EMA+RSI Trend-Following mit Higher-TF-Filter
    - `ema_rejection_trend_follow` – EMA Trend-Following mit Rejection-Pattern-Erkennung
    - `trading_the_flow` – Price Action mit Engulfing-Patterns und Psych-Levels
    - `mean_reversion_z_score` – Z-Score-basierte Mean Reversion
    - `mean_reversion_bollinger_bands_plus_macd` – Bollinger Bands + MACD Reversion
    - `statistical_arbitrage` – Spread-Trading zwischen korrelierten Paaren
    - `macd_trend_follow` – MACD-basiertes Trend-System
    - `session_momentum` – Session-basierte Momentum-Strategie
    - `pre_session_momentum` – Pre-Market Momentum-Strategie

## Konfiguration & Dateien

- Systempfade: `src/hf_engine/infra/config/paths.py`
  - Legt Projekt‑/Daten‑/Log‑/Result‑/Tmp‑Verzeichnisse fest und erzeugt sie.
  - Wichtige Orte: `var/logs` (System/Trades/Entry/Optuna), `var/results` (Backtests), `var/tmp` (Heartbeats/Stop), `data/raw|parquet|csv`.

- Environment: `src/hf_engine/infra/config/environment.py`
  - Lädt `.env` (optional), zentrale Variablen (u. a. Zeitzonen, Telegram, MT5‑Flags, Log‑Level).

- Live‑Konfigurationen: `configs/live/*.json`
  - Account‑Daten, Broker‑Klassennamen, Datafeed‑Einstellungen (lokal/remote), Strategieliste/Parameter.
  - Spezieller Datafeed‑Modus via `"data_provider_only": true`.

- Backtest‑Konfigurationen: `configs/backtest/*.json`
  - Pflichtfelder: Zeitraum, Datenpfad/Format, `symbol` oder `multi_symbols`, `timeframes`, Strategie/Parameter, Slippage/Fee.
  - Validatoren in `configs/backtest/_config_validator.py`.

- Ausführungskosten: `configs/execution_costs.yaml`
  - Slippage‑Defaults, Legacy‑Fees, neues CommissionModel (Schemata pro Symbol möglich).

- Symbol‑Spezifikationen: `configs/symbol_specs.yaml`
  - Kontraktgröße, Tick‑Größe/‑Wert, Volumen‑Grenzen, Pip‑Größe, Währungen je Symbol.

## Kerngleichungen & Logik (Auszug)

- Zeitfenster/Warmup: `backtest_engine/runner.py`
  - `prepare_time_window`: bestimmt `extended_start` anhand größtem TF×Warmup‑Bars.

- Timestamp‑Alignment: `backtest_engine/runner.py`
  - Primary‑TF: harte Schnittmenge Bid/Ask‑Timestamps, strikte Validierung.
  - Höhere TFs: Carry‑Forward nur mit „previous completed bar“ (look‑ahead‑sicher); optional Stale‑Limit in Bars; Zeitstempel‑Semantik pro TF konfigurierbar (open/close).
  - Alignment‑Cache (LRU) über Signatur von Sequenzköpfen/-enden; Größe via `ALIGN_CACHE_MAX`.

- Market‑Hours‑Filter: `backtest_engine/data/market_hours.py`
  - DST‑aware FX‑Sitzung in Sydney‑Zeit: Mo ab 07:00 bis Sa 07:00 gültig; vektorisiert oder row‑wise Fallback.

- Datenlader: `backtest_engine/data/data_handler.py`
  - CSV/Parquet‑Laden mit UTC‑Normalisierung; optional Downcast (`HF_FLOAT32=1`), TF‑Normalisierung via `.dt.floor`.
  - Daily‑TF: entfernt Wochenenden und 0‑Volumen‑Flat‑Platzhalter.
  - LRU‑Caches für Candle‑Builds (Parquet/Preloaded‑DFs), Größe via `HF_CACHE_MAX_*`.

- Execution (Live): `hf_engine/core/execution/execution_engine.py`
  - Idempotency‑Cache (TTL) für Order‑Keys; Vor‑Trade‑Guards (SL‑Mindestabstand, Pending‑Abstände, Typ‑Kohärenz).
  - Thread‑Pool für Broker‑IO mit Timeout; Tracking über ExecutionTracker.

- Brokeradapter (MT5): `hf_engine/adapter/broker/mt5_adapter.py`
  - Initialisierung (Terminal, Login, Server, Datenpfad), Symbol‑Selektion, Preisabfragen/Spread, Market/Pending Orders, Änderung/Cancel.
  - Kommentar‑Sanitizer, Runden gemäß Tick‑Größe, Volumen‑Rundung basierend auf Step/Min.

- Risk & Sizing (Live/Backtest):
  - LotSizeCalculator: Größenberechnung aus Risiko‑% und SL‑Distanz (Pip‑Werte, Cross‑Kurs ggf. erforderlich).
  - CommissionModel + FX‑Rate‑Provider (statisch/zeitreihe/composite) für Gebühren je Konto‑Währung.

## APIs & Endpunkte

- UI (FastAPI): `src/ui_engine/main.py`
  - `GET /` – Ping
  - `POST /start/{name}` – Prozess (Strategie/Datafeed) starten
  - `POST /stop/{name}` – Prozess stoppen
  - `POST /restart/{name}` – Prozess neu starten
  - `GET /status/{name}` – Status (inkl. Heartbeat‑Prüfung)
  - `GET /logs/{account_id}` – Log‑Tail als Text
  - `WS /ws/logs/{account_id}` – Live‑Logstream
  - `GET /resource/{name}` – CPU/RAM/Threads/Startzeit
  - `POST /datafeed/start|stop`, `GET /datafeed/health` – Datafeed‑Steuerung/Health

- Datafeed (FastAPI): `src/hf_engine/adapter/fastapi/mt5_feed_server.py`
  - `POST /ohlc` – Liste abgeschlossener Kerzen (limitierbar, `MAX_BARS`)
  - `POST /ohlc_closed` – einzelne abgeschlossene Kerze (Offset)
  - `POST /rates_range`, `POST /rates_from_pos` – OHLC‑Bereiche/Positionen
  - `POST /tick_data` – Tickdaten Zeitraum
  - `GET /health` – Health/Ping (offen)

- Health‑Server: `src/hf_engine/infra/monitoring/health_server.py`
  - `GET /health` – Systemstatus (Umgebung, Strategien mit Heartbeats, Zeit in UTC)

## Laufzeit‑Ablage & Pfade

- `var/tmp` – Heartbeats/Stop‑Signale, temporäre Konfigurationen
- `var/logs` – System‑, Trade‑, Entry‑, Optuna‑Logs; SQLite‑DB (`engine_logs.db`)
- `var/results` – Backtests/Walkforwards/Optimierungen
- `data/raw|csv|parquet` – Marktdaten (Ein-/Ausgabe der Konverter/Loader)

## Tests & Tooling

- Tests: `tests/*` decken UI‑Controller, Datafeed‑Server, Risk/Lot‑Sizing, Broker‑FSM/Utils, Backtest‑Runner/Optimizer u. a. ab.
- Numerik: BLAS/NumPy Threads werden in Backtests auf 1 begrenzt (Reproduzierbarkeit).
- Formatierung/Packaging: `pyproject.toml` (setuptools, isort Black‑Profil); Abhängigkeiten via `pyproject.toml`.

## Abhängigkeiten (Auszug)

- Server/Frameworks: FastAPI, Uvicorn, Pydantic v2
- Numerik/DS: NumPy, pandas, matplotlib, joblib
- Optimierung: Optuna
- I/O & Utils: PyYAML, requests, python‑dotenv, filelock
- Live: MetaTrader5 (nur Windows)
- Optional Dev: pytest, httpx; Optional ML: torch

## Typische Aufrufe

- Live/Datafeed starten: `python src/engine_launcher.py --config <configs/live/*.json>`
- UI starten: `uvicorn ui_engine.main:app --reload --port 8000`
- Backtest ausführen: `python src/backtest_engine/runner.py <configs/backtest/*.json>`

---

Diese Zusammenfassung bildet die Struktur, Datenwege und Kernlogik des Repositories ab. Detaillierte Beispiele und Anwendungsanweisungen befinden sich in `README.md` und den jeweiligen Modul‑Docs/Tests.

