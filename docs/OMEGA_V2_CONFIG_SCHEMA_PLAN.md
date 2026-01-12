# Omega V2 – Config Schema Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Normative Spezifikation der Omega V2 Backtest-Konfiguration (JSON) inkl. Pflichtfeldern, Defaults, Validierungsregeln, Normalisierung und Migrationspfad von V1  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Vision, strategische Ziele, Erfolgskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Blueprint, Module, Regeln, Single FFI Boundary |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Phasen, Datenqualitäts-Checkpoints |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Struktur, Zuständigkeiten, Config/Result Location |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Normative Metrik-Keys + Parameterisierung (`metrics.*`) |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell (Bid/Ask, Fills, SL/TP, Slippage/Fees, Sizing) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (Artefakte, Schema, Zeit/Units, Pfade) |

---

## 1. Scope und Leitplanken

### 1.1 Ziel des Config Schemas

Die V2-Konfiguration ist die **einzige** normative Eingabe für den Backtest-Core.

- Sie muss **klar**, **minimal**, **validierbar** und **forward-compatible** sein.
- Sie soll **ohne Pfadangaben** auskommen (Pfade werden *fixed* bzw. *environment-driven* aufgelöst).
- Sie trennt explizit:
  - **Run-Determinismus** (dev/prod)
  - **Daten-Granularität** (candle/tick)

### 1.2 Nicht-Ziele

- Kein vollständiges JSON-Schema (Draft 2020-12) als Artefakt in dieser Phase.
- Keine Ausprägung aller zukünftigen V2-Pläne (z.B. Execution Model Details); diese werden in separaten Plänen normiert.

---

## 2. Grundprinzipien

### 2.1 Forward-Compatibility

- **Unbekannte Keys sind erlaubt** und werden vom Parser ignoriert.
- Breaking Changes werden über `schema_version` gesteuert.

### 2.2 Pfade sind nicht Teil der Config

Die Config enthält **keine** absoluten/relativen Pfade zu:

- Marktdaten (Parquet)
- Kosten/Specs YAMLs
- News-Dateien

Stattdessen wird über **Environment Defaults** und **fixed Layout** gearbeitet (siehe Abschnitt 6).

### 2.3 Determinismus-Policy

- `run_mode = "dev"`:
  - Backtests sollen **deterministisch reproduzierbar** sein.
  - Randomness (z.B. Slippage-Jitter, Fees-Jitter) nutzt **`rng_seed`**.
- `run_mode = "prod"`:
  - Randomness soll **stochastisch** sein (Seed aus OS-RNG).
  - `rng_seed` ist optional; wenn gesetzt, darf er zur Reproduzierbarkeit genutzt werden.

---

## 3. Top-Level Schema (Normativ)

### 3.1 Pflichtfelder

| Feld | Typ | Beispiel | Beschreibung |
|------|-----|----------|--------------|
| `schema_version` | `string` | `"2"` | Schema-Version der V2-Config |
| `strategy_name` | `string` | `"mean_reversion_z_score"` | Name der Rust-Strategie im Strategy-Registry |
| `symbol` | `string` | `"EURUSD"` | Single-Symbol Backtest |
| `start_date` | `string` | `"2020-01-01"` | Start (inklusive), Format `YYYY-MM-DD` |
| `end_date` | `string` | `"2021-01-01"` | Ende (inklusive), Format `YYYY-MM-DD` |
| `run_mode` | `"dev"|"prod"` | `"dev"` | Determinismus-Policy |
| `data_mode` | `"candle"|"tick"` | `"candle"` | Daten-Granularität |
| `timeframes` | `object` | siehe unten | Primär-TF + optionale zusätzliche TFs |

### 3.2 Optionale Felder (Core)

| Feld | Typ | Default | Beschreibung |
|------|-----|---------|--------------|
| `rng_seed` | `integer \| null` | `42` (nur `dev`) | RNG-Seed für deterministische Runs |
| `warmup_bars` | `integer` | `500` | Warmup-Bars pro Timeframe (global angewendet) |
| `sessions` | `array \| null` | `null` | Trading-Sessions in UTC |
| `account` | `object \| null` | defaults | Konto-/Sizing-Defaults (Backtest) |
| `costs` | `object \| null` | defaults | Kostenmodell-Toggles/Multipliers (ohne Pfade) |
| `news_filter` | `object \| null` | defaults | News-Filter (enabled + Parameter, ohne Pfade) |
| `logging` | `object \| null` | defaults | Logging-Toggles (artefaktbezogen) |
| `metrics` | `object \| null` | defaults | Parameter für Metrik-/Robustness-Orchestrierung (Optimizer/Rating) |

---

## 4. Sub-Schemas

### 4.1 `timeframes`

```json
{
  "timeframes": {
    "primary": "M5",
    "additional": ["H1", "D1"],
    "additional_source": "separate_parquet"
  }
}
```

| Feld | Typ | Default | Regeln |
|------|-----|---------|-------|
| `primary` | `string` | `"M15"` | Uppercase; muss in erlaubter Menge sein (z.B. `M1, M5, M15, M30, H1, H4, D1`) |
| `additional` | `array<string>` | `[]` | Uppercase, dedupliziert, darf `primary` nicht enthalten |
| `additional_source` | `"separate_parquet"|"aggregate_from_primary"` | `"separate_parquet"` | Quelle für zusätzliche TFs (HTF) |

**Entscheidung**: HTF kann via separate Parquet-Dateien oder Aggregation aus Primary generiert werden.

### 4.2 `sessions`

```json
{
  "sessions": [
    {"start": "08:00", "end": "17:00"},
    {"start": "19:00", "end": "22:00"}
  ]
}
```

| Feld | Typ | Regeln |
|------|-----|-------|
| `start` | `string` | UTC-Zeit, Format `HH:MM` |
| `end` | `string` | UTC-Zeit, Format `HH:MM`; wenn `end <= start`, gilt Session als *cross-midnight* |

### 4.3 `account`

```json
{
  "account": {
    "initial_balance": 10000.0,
    "account_currency": "EUR",
    "risk_per_trade": 100.0
  }
}
```

| Feld | Typ | Default | Validierung |
|------|-----|---------|------------|
| `initial_balance` | `number` | `10000.0` | `> 0` |
| `account_currency` | `string` | `"EUR"` | Uppercase, 3-letter |
| `risk_per_trade` | `number` | `100.0` | `> 0` |

### 4.4 `costs`

```json
{
  "costs": {
    "enabled": true,
    "fee_multiplier": 1.0,
    "slippage_multiplier": 1.0,
    "spread_multiplier": 1.0
  }
}
```

| Feld | Typ | Default | Validierung |
|------|-----|---------|------------|
| `enabled` | `boolean` | `true` | - |
| `fee_multiplier` | `number` | `1.0` | `>= 0` |
| `slippage_multiplier` | `number` | `1.0` | `>= 0` |
| `spread_multiplier` | `number` | `1.0` | `>= 0` |

**Wichtig**: Es gibt **keine** Pfadfelder für `execution_costs.yaml` oder `symbol_specs.yaml`.

### 4.5 `news_filter`

```json
{
  "news_filter": {
    "enabled": false,
    "minutes_before": 30,
    "minutes_after": 30,
    "min_impact": "medium",
    "currencies": ["EUR", "USD"]
  }
}
```

| Feld | Typ | Default | Validierung |
|------|-----|---------|------------|
| `enabled` | `boolean` | `false` | - |
| `minutes_before` | `integer` | `30` | `>= 0` |
| `minutes_after` | `integer` | `30` | `>= 0` |
| `min_impact` | `"low"|"medium"|"high"` | `"medium"` | - |
| `currencies` | `array<string> \| null` | `null` | Wenn `null`, aus `symbol` abgeleitet: 1. und 2. Währung |

**Wichtig**: Kein Pfad in der Config. Quelle/Location ist environment-driven (siehe Abschnitt 6).

### 4.6 `logging`

```json
{
  "logging": {
    "enable_entry_logging": false,
    "logging_mode": "trades_only"
  }
}
```

| Feld | Typ | Default | Beschreibung |
|------|-----|---------|--------------|
| `enable_entry_logging` | `boolean` | `false` | Aktiviert Entry-Logs/Debug-Artefakte |
| `logging_mode` | `string` | `"trades_only"` | Erweiterbar (z.B. `"all"`) |

---

### 4.7 `metrics`

Dieses Objekt steuert **metrische Auswertungen und Robustness-/Stress-Orchestrierung**.

**Wichtig:** Die Parameter werden primär im **Optimizer-/Rating-Layer** verwendet (Multi-Run). Single-Run `metrics.json` wird davon nicht beeinflusst.

Referenz: `OMEGA_V2_METRICS_DEFINITION_PLAN.md`.

```json
{
  "metrics": {
    "robustness": {
      "enabled": false,
      "mode": "full",
      "jitter_frac": 0.05,
      "jitter_repeats": 5,
      "dropout_frac": 0.10,
      "dropout_runs": 1,
      "cost_shock_factors": [1.25, 1.50, 2.00]
    },
    "data_jitter": {
      "repeats": 5,
      "atr_period": 14,
      "sigma_atr": 0.10,
      "penalty_cap": 0.5,
      "min_price": 1e-9,
      "fraq": 0.0
    },
    "timing_jitter": {
      "divisors": [10, 5, 20],
      "min_months": 1
    },
    "ulcer": {
      "ulcer_cap": 10.0
    }
  }
}
```

#### 4.7.1 `metrics.robustness`

| Feld | Typ | Default | Validierung | Beschreibung |
|------|-----|---------|------------|--------------|
| `enabled` | `boolean` | `false` | - | Aktiviert Robustness/Stress-Scoring im Optimizer |
| `mode` | `string` | `"full"` | `"full"|"fast"|"off"` | Auswahl des Robustness-Sets (Implementierungsdetails) |
| `jitter_frac` | `number` | `0.05` | `>=0` | Relative Param-Jitter Stärke (V1-kompatibel) |
| `jitter_repeats` | `integer` | `5` | `>=0` | Wiederholungen für Param-Jitter |
| `dropout_frac` | `number` | `0.10` | `0..1` | Anteil entfallender Trades (Trade-Dropout) |
| `dropout_runs` | `integer` | `1` | `>=0` | Anzahl Runs für Trade-Dropout |
| `cost_shock_factors` | `array<number>` | `[1.25, 1.50, 2.00]` | alle `>=1` | Multiplikative Kosten-Schocks |

#### 4.7.2 `metrics.data_jitter`

| Feld | Typ | Default | Validierung | Beschreibung |
|------|-----|---------|------------|--------------|
| `repeats` | `integer` | `5` | `>=0` | Anzahl Jitter-Samples |
| `atr_period` | `integer` | `14` | `>0` | ATR-Periode für Skalierung |
| `sigma_atr` | `number` | `0.10` | `>=0` | Jitter-Sigma relativ zu ATR |
| `penalty_cap` | `number` | `0.5` | `>0` | Cap im Penalty-Modell (V1-kompatibel) |
| `min_price` | `number` | `1e-9` | `>0` | Untere Schranke für Preise |
| `fraq` | `number` | `0.0` | `>=0` | V1-Kompatibilitätsparameter (Data-Jitter) |

#### 4.7.3 `metrics.timing_jitter`

| Feld | Typ | Default | Validierung | Beschreibung |
|------|-----|---------|------------|--------------|
| `divisors` | `array<integer>` | `[10, 5, 20]` | alle `>0` | Window→Monatsshift Ableitung (V1-kompatibel) |
| `min_months` | `integer` | `1` | `>=0` | Minimaler Monatsshift |

#### 4.7.4 `metrics.ulcer`

| Feld | Typ | Default | Validierung | Beschreibung |
|------|-----|---------|------------|--------------|
| `ulcer_cap` | `number` | `10.0` | `>0` | Cap für `ulcer_index_score = 1 - ulcer/ulcer_cap` |

## 5. Validierung und Normalisierung (Normativ)

### 5.1 Typ- und Wertevalidierung

- `start_date <= end_date`
- `symbol` und Timeframes werden **uppercase-normalisiert**
- `timeframes.additional` wird **dedupliziert** und darf `primary` nicht enthalten
- `warmup_bars >= 0`
- **Kein** `MIN_TRADING_BARS`-Gate. Es gilt nur: es müssen ausreichend Bars vorhanden sein, um nach Warmup starten zu können.

### 5.2 Datenverfügbarkeits-Checks (an Data Loader gekoppelt)

- Für `data_mode=candle` müssen die Parquet-Dateien im expected Layout vorhanden sein.
- Für `data_mode=tick` gelten separate Tick-Layouts (nicht Teil des MVP; Platzhalter).

---

## 6. Environment-driven Path Resolution (Normativ)

### 6.1 Marktdaten (Parquet)

- Root wird über `OMEGA_DATA_PARQUET_ROOT` bestimmt.
- Default (wenn Env nicht gesetzt): `data/parquet`

Expected Layout:

- `{ROOT}/{SYMBOL}/{SYMBOL}_{TF}_BID.parquet`
- `{ROOT}/{SYMBOL}/{SYMBOL}_{TF}_ASK.parquet`

Beispiel:

- `data/parquet/EURUSD/EURUSD_M5_BID.parquet`
- `data/parquet/EURUSD/EURUSD_M5_ASK.parquet`

### 6.2 Kosten- und Symbol-Specs

- Default Locations:
  - `configs/execution_costs.yaml`
  - `configs/symbol_specs.yaml`

Optional übersteuert via Env:

- `OMEGA_EXECUTION_COSTS_FILE`
- `OMEGA_SYMBOL_SPECS_FILE`

### 6.3 News Calendar

- Default Location: `news/news_calender_history.parquet` (oder äquivalentes, versioniertes Format)
- Optional via Env: `OMEGA_NEWS_CALENDAR_FILE`

**Hinweis**: Das aktuelle Repo enthält `news/news_calender_history.csv`. Für V2 wird ein Parquet/Arrow-Format erwartet; der Konvertierungsschritt ist Teil des Data-Layers bzw. der Datenpipeline, nicht der Config.

---

## 7. Beispielkonfigurationen

### 7.1 Minimal (MVP Candle)

```json
{
  "schema_version": "2",
  "strategy_name": "mean_reversion_z_score",
  "symbol": "EURUSD",
  "start_date": "2020-01-01",
  "end_date": "2021-01-01",
  "run_mode": "dev",
  "data_mode": "candle",
  "rng_seed": 42,
  "timeframes": {"primary": "M5", "additional": ["D1"], "additional_source": "separate_parquet"},
  "warmup_bars": 500
}
```

### 7.2 Candle + Sessions + Costs + News

```json
{
  "schema_version": "2",
  "strategy_name": "mean_reversion_z_score",
  "symbol": "EURUSD",
  "start_date": "2020-01-01",
  "end_date": "2021-01-01",
  "run_mode": "prod",
  "data_mode": "candle",
  "timeframes": {"primary": "M5", "additional": ["H1", "D1"], "additional_source": "aggregate_from_primary"},
  "warmup_bars": 500,
  "sessions": [{"start": "08:00", "end": "17:00"}],
  "account": {"initial_balance": 10000.0, "account_currency": "EUR", "risk_per_trade": 100.0},
  "costs": {"enabled": true, "fee_multiplier": 1.0, "slippage_multiplier": 1.0, "spread_multiplier": 1.0},
  "news_filter": {"enabled": true, "minutes_before": 30, "minutes_after": 30, "min_impact": "high", "currencies": null},
  "logging": {"enable_entry_logging": false, "logging_mode": "trades_only"}
}
```

---

## 8. Migration von V1-Configs (Guidance)

### 8.1 Feld-Mapping (V1 → V2)

| V1 Feld | V2 Feld | Notes |
|--------|---------|-------|
| `mode` (`candle|tick`) | `data_mode` | Begriff entkoppelt von `run_mode` |
| (kein V1 Äquivalent) | `run_mode` | Neu: `dev|prod` Determinismus-Policy |
| `timeframes.primary` | `timeframes.primary` | Semantik identisch (uppercase-normalisiert) |
| `timeframes.additional` | `timeframes.additional` | Deduplizierung, ohne `primary` |
| `warmup_bars` | `warmup_bars` | Direkte Übernahme |
| `strategy.module`/`strategy.class` | `strategy_name` | V2: Rust-Registry statt Python import |
| `strategy.parameters` | (impl-spezifisch) | Wird im V2-Core als strategy-params interpretiert (Format bleibt JSON object) |
| `execution_costs_file`/`symbol_specs_file` | (Env) | Keine Pfade in V2-Config |
| `news_filter.*_path` | (Env) | Keine Pfade in V2-Config |

### 8.2 Empfohlene Übergangsstrategie

- Bestehende V1-Configs bleiben für V1-Backtests unverändert.
- Für V2 wird eine neue, schlanke JSON-Config genutzt.
- Optional kann ein Python-Wrapper eine V1-Config nach V2 „transpilieren“ (nur für Migration/Convenience).

---

## 9. Offene Punkte (für spätere Pläne)

Diese Themen sind absichtlich **nicht** vollständig im Config Schema normiert und werden in dedizierten Plänen behandelt:

- Alignment-Loss-Policy, Gap-Policy, Duplicate-Timestamp-Policy → [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md)
- Execution-Prioritäten, Stop/TP Tie-Breaks, Intrabar-Regeln → OMEGA_V2_EXECUTION_MODEL_PLAN.md
- Output-Artefakte/JSON Felder → OMEGA_V2_OUTPUT_CONTRACT_PLAN.md
