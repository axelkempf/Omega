# Omega V2 – Observability & Profiling Plan

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation der Observability-Strategie für Omega V2 (Logging/Tracing/Profiling, Artefakte, CI-Integration, Determinismus-Guardrails)  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, FFI-Grenze, Modul-Verantwortlichkeiten |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Timestamp-Contract, Fail-Fast Checkpoints |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Data-Quality-Policies, Snapshots/Manifests, Provenance |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell (Bid/Ask, SL/TP, deterministisch) |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Rust Workspace/Crates, Test-Strategie |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema (run_mode, data_mode, rng_seed) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Output-Artefakte (trades/equity/metrics/meta), Zeit/Units |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Trading-Metriken, Keys/Units, Rundung |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains (PyO3/maturin), Logging (`tracing`), Error-Contract |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Determinismus-Tests, Golden-Files, PR-Gates vs nightly |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI/CD Workflow, Quality Gates, Build-Matrix, Security |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für Observability und Profiling in Omega V2.

**Normative Ziele (MUSS):**

- **Debuggability-first**: Reproduzierbare Ursachenanalyse via `run_id`, deterministische Logs/Traces und klare Fehlerpfade
- **Performance-first**: Profiling/Flamegraphs für Hotspots + optionale Budgets/Regression (CI/nightly)
- **Determinismus-Safe**: Keine Logging-/Tracing-Calls, die Berechnungslogik oder Reproduzierbarkeit beeinflussen
- **Security-Aware**: Keine Secrets in Logs; Config-Values nur gehasht oder explizit erlaubte Felder

**Bewusste Entscheidungen (festgezogen):**

- Logging/Tracing via **`tracing`** (Rust-Standard)
- **STDOUT = human**, **File = JSON** (maschinenlesbar für CI/Automation)
- Zwei Log-Dateien: `<run_id>.log` (human-readable) + `<run_id>.jsonl` (JSON Lines)
- Profiling-Artefakte im **Run-Ordner** unter `var/results/backtests/<run_id>/profiling/`
- Profiling-Artefakte sind **optional** (nie Teil des MVP-Output-Contracts)
- Profiling-Jobs nur **manuell + nightly** (kein PR-Gate)

---

## 2. Geltungsbereich

### 2.1 In Scope

- **Omega V2 Backtest-Core** (Rust + Python Wrapper)
- Logging/Tracing während Data Load, Strategy Init, Backtest Loop, Metrics, Write
- Performance-Counter (Timings/Counts/Resources) als separates Artefakt (`profiling.json`)
- CPU/Memory-Profiling (Flamegraphs, pprof, Peak RSS)
- CI-Integration (manuell + nightly)

### 2.2 Out of Scope (vorerst)

- UI-Engine / Live-Trading Observability (separater Plan)
- Ops-Monitoring / Dashboards / Alerts (Post-MVP)
- Distributed Tracing über mehrere Services (Single-Process MVP)

---

## 3. Grundprinzipien (Normativ)

### 3.1 Determinismus & Reproduzierbarkeit

**Normativ (MUSS):**

- Logging/Tracing darf **nie** Berechnungslogik oder Execution-Entscheidungen beeinflussen.
- Profiling-Spans im Hot-Path (z.B. per Bar) sind nur in `profiling_mode=true` erlaubt und müssen deterministisch sein (keine Timestamps in Strategy-Logik).
- Standard-Runs ohne Profiling-Mode sind bitgenau reproduzierbar (gemäß DEV/PROD Policy aus Config-Schema-Plan).

### 3.2 Security & Secrets

**Normativ (MUSS):**

- Secrets (API-Keys, Tokens, Passwörter) dürfen **nie** in Logs/Traces erscheinen.
- Config-Values werden nur gehasht ausgegeben **oder** explizit erlaubte Felder (`symbol`, `timeframe`, `strategy_name`).
- Redaction ist Allowlist-only (kein Blacklist-Ansatz).

### 3.3 Performance-Overhead

**Normativ (SOLL):**

- Logging im Hot-Path (Backtest-Loop) ist auf `DEBUG`/`TRACE` Level und Default-aus.
- Strukturiertes Logging bevorzugt über String-Interpolation (effizientere Serialisierung).

---

## 4. Logging-Strategie (Rust `tracing`)

### 4.1 Format & Sinks

**Normativ:**

- Logging-Framework: **`tracing`** + **`tracing-subscriber`** (siehe Tech-Stack-Plan).
- **STDOUT**: Human-readable Format (entwicklerfreundlich, Console).
- **File**: JSON Lines (`.jsonl`) für maschinenlesbare Ingest/Analyse.

### 4.2 Log-Dateien

**Normativ:**

- Zwei Dateien pro Run (unter `var/logs/`):
  - `<run_id>.log` (human-readable)
  - `<run_id>.jsonl` (JSON Lines)

**Datei-Layout:**

- `var/logs/<run_id>.log`
- `var/logs/<run_id>.jsonl`

### 4.3 Minimaler Log-Kontext

**Normativ (MUSS in jedem Log-Eintrag):**

- `run_id`
- `strategy_name`
- `symbol`
- `timeframe`

**Optionale Kontextfelder (SOLL):**

- `component` (z.B. `data_load`, `execution`, `metrics`)
- `phase` (z.B. `init`, `loop`, `finalize`)

### 4.4 Log-Level & Defaults

**Normativ:**

| `run_mode` | Default Level | Override |
|-----------|--------------|----------|
| `dev` | `INFO` | via `RUST_LOG` |
| `prod` | `WARNING` | via `RUST_LOG` |

**Beispiel:**

- DEV: `RUST_LOG=info` (default)
- PROD: `RUST_LOG=warn` (default)
- Override: `RUST_LOG=debug` (für Debugging)

### 4.5 Strukturiertes Logging (Best Practice)

**Normativ (SOLL):**

```rust
// GOOD: Strukturiert
tracing::info!(
    run_id = %run_id,
    symbol = %symbol,
    bars_processed = total_bars,
    "Backtest completed"
);

// BAD: String-Interpolation
tracing::info!("Backtest completed for {} with {} bars", symbol, total_bars);
```

---

## 5. Tracing-Strategie (Spans & Sampling)

### 5.1 Span-Granularität

**Normativ:**

- **Phase-Level (immer)**: Data Load, Strategy Init, Backtest Loop, Metrics Calculation, Write Artefacts.
- **Fein-granular (nur `profiling_mode=true`)**: Per-Bar Spans, Per-Event Spans.

**Beispiel (Phase-Level):**

```rust
let _span = tracing::info_span!("data_load", run_id = %run_id).entered();
// ... load data ...
drop(_span);

let _span = tracing::info_span!("backtest_loop", run_id = %run_id).entered();
// ... backtest loop ...
drop(_span);
```

### 5.2 Profiling-Mode Aktivierung

**Normativ:**

- Aktivierung via **Config-Feld**: `observability.profiling_mode = true`
- Default: `false`

**Config-Schema (Erweiterung für Config-Schema-Plan):**

```json
{
  "observability": {
    "profiling_mode": false
  }
}
```

### 5.3 Sampling-Policy (run_mode-abhängig)

**Normativ:**

| `run_mode` | Tracing-Spans | Logs |
|-----------|---------------|------|
| `dev` | Phase-Level aktiv | `INFO` |
| `prod` | **Spans aus** | `WARNING` |

**Begründung:** PROD-Runs priorisieren Performance und minimalen Overhead; Debugging erfolgt via DEV-Mode mit Reproduktion.

---

## 6. Profiling-Tooling (CPU & Memory)

### 6.1 CPU Profiling (Rust Core)

**Normativ (zwei Stufen):**

- **Lokal**: `cargo flamegraph` (einfach, etabliert, entwicklerfreundlich)
- **CI/Nightly**: `pprof` (programmatisch, report als Artifact)

**Empfohlene Workflow:**

1. Entwickler lokal: `cargo flamegraph --bin <target>` → `flamegraph.svg`
2. CI nightly: `pprof` → `profile.pb.gz` als Artifact

### 6.2 Memory Profiling (Optional)

**Normativ:**

- Memory profiling via **jemalloc + jeprof** bei Bedarf aktivierbar.
- **Standard-Runs tracken nur Peak RSS** (Lightweight-Metrik).
- Heap-Profiling **nicht** Teil des MVP; Tooling und Aktivierung dokumentiert, aber optional.

**Dokumentation (für später):**

```bash
# Aktivierung (Beispiel)
cargo build --features jemalloc-profiling
MALLOC_CONF=prof:true ./target/release/omega_v2 ...
jeprof --svg target/release/omega_v2 jeprof.*.heap > heap.svg
```

### 6.3 Python Profiling (Wrapper/Orchestrator)

**Normativ:**

- Python-Layer: **`cProfile`** (Standard-Library) nur für I/O/Serialisierung.
- Fokus liegt auf **Rust-Core**, nicht Python (Wrapper ist dünn).

**Beispiel:**

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... run backtest ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

---

## 7. Performance-Counter & Observability-Metrics

### 7.1 Welche Counter? (Normativ)

**Minimal (MUSS):**

- `wall_time_total_ms` (Gesamtlaufzeit)
- `time_data_load_ms` (Data Loading Phase)
- `time_backtest_ms` (Backtest Loop Phase)
- `time_write_ms` (Artefakte schreiben)
- `peak_rss_mb` (Peak Resident Set Size)

**Engine-Zähler (SOLL):**

- `bars_processed` (Anzahl verarbeitete Bars)
- `events_processed` (Anzahl Events, falls Event-Engine aktiv)
- `trades_emitted` (Anzahl abgeschlossene Trades)
- `orders_filled` (Anzahl gefüllte Orders)

### 7.2 Speicherort & Schema

**Normativ:**

- Separates Artefakt: **`profiling.json`**
- Pfad: `var/results/backtests/<run_id>/profiling/profiling.json`

**Schema (angelehnt an `metrics.json`):**

```json
{
  "metrics": {
    "wall_time_total_ms": 1234,
    "wall_time_total_ns": 1234567890,
    "time_data_load_ms": 100,
    "time_data_load_ns": 100000000,
    "time_backtest_ms": 1000,
    "time_backtest_ns": 1000000000,
    "time_write_ms": 50,
    "time_write_ns": 50000000,
    "peak_rss_mb": 256,
    "bars_processed": 10000,
    "events_processed": 5000,
    "trades_emitted": 42,
    "orders_filled": 84
  },
  "definitions": {
    "wall_time_total_ms": {"unit": "ms", "description": "Total wall-clock time"},
    "wall_time_total_ns": {"unit": "ns", "description": "Total wall-clock time (nanoseconds)"},
    "time_data_load_ms": {"unit": "ms", "description": "Data loading phase duration"},
    "time_data_load_ns": {"unit": "ns", "description": "Data loading phase duration (nanoseconds)"},
    "time_backtest_ms": {"unit": "ms", "description": "Backtest loop phase duration"},
    "time_backtest_ns": {"unit": "ns", "description": "Backtest loop phase duration (nanoseconds)"},
    "time_write_ms": {"unit": "ms", "description": "Artefact writing phase duration"},
    "time_write_ns": {"unit": "ns", "description": "Artefact writing phase duration (nanoseconds)"},
    "peak_rss_mb": {"unit": "MB", "description": "Peak resident set size"},
    "bars_processed": {"unit": "count", "description": "Number of bars processed"},
    "events_processed": {"unit": "count", "description": "Number of events processed"},
    "trades_emitted": {"unit": "count", "description": "Number of closed trades"},
    "orders_filled": {"unit": "count", "description": "Number of filled orders"}
  }
}
```

**Zeit-Units:**

- **Beide** `*_ms` **und** `*_ns` (für Vergleichbarkeit mit Time-Contract, siehe Output-Contract-Plan)

### 7.3 Optional vs. MUSS

**Normativ:**

- `profiling.json` ist **optional** (nie Teil des MVP-Output-Contracts).
- Wird nur erzeugt, wenn `observability.profiling_mode=true` oder explizit angefordert.

---

## 8. Profiling-Artefakte (Flamegraphs, pprof, Logs)

### 8.1 Ablageort

**Normativ:**

- Alle Profiling-Artefakte unter: `var/results/backtests/<run_id>/profiling/`

**Beispiel-Layout:**

```
var/results/backtests/<run_id>/
├── trades.json
├── equity.csv
├── metrics.json
├── meta.json
└── profiling/
    ├── profiling.json
    ├── flamegraph.svg
    ├── profile.pb.gz (pprof)
    └── heap.svg (optional)
```

### 8.2 Naming-Konventionen

**Normativ:**

- `profiling.json` (Performance-Counter)
- `flamegraph.svg` (CPU Flamegraph)
- `profile.pb.gz` (pprof Protobuf)
- `heap.svg` (Memory Heap Flamegraph, optional)

### 8.3 Retention

**Normativ:**

- Profiling-Artefakte folgen der gleichen Retention-Policy wie andere Run-Artefakte (projektspezifisch, nicht im Plan normiert).
- In CI: 7 Tage (siehe CI-Workflow-Plan).

---

## 9. CI-Integration (Performance Jobs)

### 9.1 Wann laufen Profiling-Jobs?

**Normativ:**

- **Manuell** (`workflow_dispatch`)
- **Optional nightly** (z.B. via `schedule: cron`)
- **Nie als PR-Gate** (Performance-Jobs sind langsam und optional)

### 9.2 Performance-Regression Policy

**Normativ:**

- **Keine harten Budgets** (nur Reports/Trends).
- CI sammelt `profiling.json` als Artifact und erlaubt manuelle Vergleiche.
- Spätere Erweiterung: Automated Regression Detection (post-MVP).

### 9.3 CI-Artefakte

**Normativ:**

- `profiling.json` (Performance-Counter)
- `flamegraph.svg` oder `profile.pb.gz` (CPU-Profil)
- Logs (`.log` + `.jsonl`)

**Retention:** 7 Tage (siehe CI-Workflow-Plan).

---

## 10. Logging/Tracing-Konfiguration (Rust Runtime)

### 10.1 Subscriber-Setup (Normativ)

**Empfohlene Implementierung:**

```rust
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

fn setup_tracing(run_id: &str, run_mode: &str) {
    let default_level = if run_mode == "prod" { "warn" } else { "info" };
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    // STDOUT: Human-readable
    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    // File: JSON Lines
    let log_file = std::fs::File::create(format!("var/logs/{}.jsonl", run_id)).unwrap();
    let json_layer = fmt::layer()
        .json()
        .with_writer(log_file);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .with(json_layer)
        .init();
}
```

### 10.2 Context-Enrichment

**Normativ:**

```rust
use tracing::Span;

let span = tracing::info_span!(
    "backtest_run",
    run_id = %run_id,
    strategy_name = %config.strategy_name,
    symbol = %config.symbol,
    timeframe = %config.timeframe
);
let _guard = span.enter();
// ... run backtest ...
```

---

## 11. Python-Wrapper Logging

### 11.1 Python Logging-Bridge (SOLL)

**Normativ:**

- Python-Wrapper loggt via Standard-`logging`-Modul.
- Logs werden **nicht** in die Rust-Logs gemischt (separate Streams).

**Beispiel:**

```python
import logging

logger = logging.getLogger("omega_v2.wrapper")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(f"var/logs/{run_id}.python.log")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
```

---

## 12. Security & Redaction (Normativ)

### 12.1 Config-Logging Policy

**Normativ (MUSS):**

- Config wird **nur gehasht** in Logs referenziert (`config_hash: <sha256>`).
- Erlaubte Felder (Allowlist):
  - `symbol`
  - `timeframe`
  - `strategy_name`
  - `run_mode`
  - `data_mode`
  - `start_date` / `end_date`

**Verbotene Felder (dürfen NIE geloggt werden):**

- `rng_seed` (könnte deterministisches Verhalten offenlegen)
- API-Keys, Tokens, Passwörter (wenn später erweitert)

### 12.2 Error-Messages & Stack-Traces

**Normativ (MUSS):**

- Error-Messages dürfen **keine** Secrets enthalten.
- Stack-Traces sind erlaubt, aber nur in DEV-Mode (PROD: nur Error-Message, kein Backtrace).

---

## 13. Nicht-Ziele / Guardrails

**Out of Scope (vorerst):**

- Distributed Tracing (OpenTelemetry, Jaeger, Zipkin)
- Real-time Streaming / WebSocket-Logs (UI-Engine separat)
- Ops-Monitoring / Dashboards / Alerting (post-MVP)

**Guardrails:**

- Profiling darf **nie** Berechnungslogik beeinflussen.
- Logging im Hot-Path ist Default-aus (nur DEBUG/TRACE).
- Secrets dürfen **nie** in Logs/Traces erscheinen.

---

## 14. Checkliste (Definition of Done)

- [ ] `tracing` + `tracing-subscriber` im Rust Core integriert
- [ ] Zwei Log-Dateien pro Run: `.log` (human) + `.jsonl` (JSON Lines)
- [ ] Minimaler Kontext in jedem Log-Eintrag (`run_id`, `strategy_name`, `symbol`, `timeframe`)
- [ ] Phase-Level Spans (Data Load, Backtest Loop, Metrics, Write) implementiert
- [ ] `profiling_mode` in Config-Schema definiert
- [ ] `profiling.json` Schema dokumentiert und implementiert
- [ ] Profiling-Artefakte unter `var/results/backtests/<run_id>/profiling/`
- [ ] CI-Job für Profiling (manuell + nightly) konfiguriert
- [ ] Flamegraph-Workflow (`cargo flamegraph` lokal, `pprof` CI) dokumentiert
- [ ] Security: Config-Hashing + Allowlist für Logging verifiziert
- [ ] Determinismus: Profiling-Spans greifen nicht in Berechnungslogik ein

---

## 15. Erweiterungen (Post-MVP)

- **OpenTelemetry**: Distributed Tracing für Multi-Service Setups
- **Performance Budgets**: Automatische Regression Detection (Hard Fail bei Überschreitung)
- **Real-time Monitoring**: Prometheus Metrics Export + Grafana Dashboards
- **Alerting**: Integration mit PagerDuty/Slack für Production-Runs

---

## Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 13.01.2026 | Initiale Version: Logging/Tracing (tracing), Profiling (flamegraph/pprof), Performance-Counter (profiling.json), CI-Integration, Security/Determinismus-Guardrails |

