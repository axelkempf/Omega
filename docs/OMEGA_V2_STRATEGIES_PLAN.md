# Omega V2 – Strategies Plan (Mean Reversion Z-Score)

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation der Strategie-Schicht (MVP: Mean Reversion Z-Score), inklusive Szenario-Logik 1–6, Filter/Guards, benötigter Indikatoren, Modul-Zerlegung und Paritätsanforderungen zu V1.

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope (MVP-Strategie: Mean Reversion Z-Score) |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Single-FFI, Verantwortlichkeiten |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Crates/Ordner, Strategy-Registry, BarContext |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Bar-/Zeit-Contract, Event-Loop, Warmup |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Datenqualität, News-Quelle/Normalisierung |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Input-Schema (`strategy.parameters`, `sessions`, `news_filter`) |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Fill/Exit-Semantik, SL/TP Prioritäten, Spread/Fees |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Output-Artefakte + Metadaten-Kontrakte |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Metriken/Keys (Single-Run vs. Optimizer) |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Paritäts-/Golden-Tests, Determinismus |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing, Profiling, Debug-Artefakte |
| [OMEGA_V2_FORMATTING_PLAN.md](OMEGA_V2_FORMATTING_PLAN.md) | Dokumentations-/Code-Standards |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI-Gates inkl. Paritätschecks |

---

## 1. Zusammenfassung (universale Wahrheit)

Omega V2 definiert eine **Rust-native Strategie-Schicht**, die im MVP genau **eine Strategie** enthält:

- **Mean Reversion Z-Score** (MRZ)

Diese Strategie besteht aus **Szenario-Regeln 1–6**, die jeweils ein Entry-Setup erzeugen können.

**Normative Leitplanken:**

- **Single FFI Boundary**: Strategieausführung läuft vollständig in Rust; Python orchestriert nur (`run_backtest(config_json)`).
- **Determinismus**: Keine Randomness in der Strategie. Gleiche Inputs → identische Trades.
- **Zeit-Kontrakt**: Entry-Entscheidungen gelten für die **abgeschlossene Bar** (bar-close). (Siehe `OMEGA_V2_DATA_FLOW_PLAN.md`)
- **Separation of Concerns**:
  - Daten-/News-/Sessions-Guards liegen **nicht** als Monolith in der Strategie-Logik.
  - Strategie konsumiert `BarContext` Flags (z.B. `news_blocked`) statt selbst Datenquellen zu parsen.

---

## 2. Scope

### 2.1 In Scope (MVP)

- MRZ-Strategie (Szenarien **1–6**, Long/Short)
- Konfigurationsoberfläche für MRZ unter `strategy.parameters` (als objekt, schema-validiert in V2)
- Filter/Guards, soweit sie Entry-Signale beeinflussen:
  - Sessions (`config.sessions`)
  - News-Blackout (`config.news_filter`)
  - One-position-per-symbol Gate
  - Cooldown (zeit-/bar-basiert)
- HTF-Filter (D1 + optional H4/H1) sowie Scenario-6 Multi-TF Overlay

### 2.2 Out of Scope (vorerst)

- **Position Manager / Trade Management** (separater Plan gewünscht; offener Punkt, siehe Abschnitt 11)
- Multi-Symbol Strategien
- Portfolio-level Risk (über `risk_per_trade` hinaus)
- Live-Trading Port (MT5) – unverändert in Python

---

## 3. Normative Entscheidungen (aus der Klärungsrunde)

### 3.1 Szenario-Nummerierung & Aktivierung

- **Szenario 1 ist aktiv** und Teil des MVP (nicht „nur vorhanden“).
- Szenario-IDs sind stabil: `1..6`.
- Scenario-Labeling (für Meta/Logs):
  - `long_1`, `short_1`, … `long_6`, `short_6`
  - Zusätzlich kann ein menschenlesbarer Name geführt werden (z.B. `szenario_2_long`), aber die numerische ID ist maßgeblich.

### 3.2 Parameter-Overrides (Backtest-Parität)

**Entscheidung (Empfehlung):** Override-Hierarchie bleibt 1:1 wie in V1 (Backtest), um Backtest-Ergebnisse nicht zu verändern.

Normative Merge-Order (last-wins):

1. Global Defaults (Strategy-Defaults)
2. `overrides["*/*"]` (wildcard symbol + wildcard timeframe)
3. `overrides["<SYMBOL>/*"]`
4. `overrides["*/<TF>"]`
5. `overrides["<SYMBOL>/<TF>"]`

Zusätzlich:
- Long/Short Parameter sind getrennt (`long`/`short`).
- Szenario-spezifische Overrides dürfen darüber liegen (z.B. `scenario6_params`).

### 3.3 Scenario-6 (Multi-TF Overlay)

- `scenario6_mode = "all" | "any"` (gewählt: **all**)
- Timeframes werden normalisiert (uppercase)
- Overlay nutzt **Kalman-Z + Bollinger** pro TF mit TF-spezifischen Parametern.

### 3.4 Intraday Vol Cluster (Scenario-5)

- Feature ist **GARCH-basiert** (`intraday_vol_feature = "garch_forecast"`).
- `atr_points` bleibt V2-MVP **nicht** relevant (keine weitere Variante in der Spezifikation).
- Hysterese-Default: `cluster_hysteresis_bars = 1` (Backtest-ähnlich).

### 3.5 Session/News Zeitbezug

- Session-/News-Blocking bezieht sich auf den Entry-Zeitpunkt (bar-close).
- Umsetzung erfolgt nicht in der Strategie, sondern über `BarContext` Flags.

---

## 4. Strategie-API (Rust, im Sinne der Crate-Struktur)

Referenz: `OMEGA_V2_MODULE_STRUCTURE_PLAN.md` (`crates/strategy`).

### 4.1 Eingaben

Die Strategie erhält pro Bar einen read-only Snapshot:

- `BarContext` (MVP-Felder, normativ):
  - `idx` (bar index)
  - `timestamp` (bar close time)
  - `candles` (primary bid/ask candle)
  - `htf_data` (optional: zusätzliche Timeframes)
  - `indicators` (precomputed, kein FFI)
  - `session_open: bool`
  - `news_blocked: bool`

### 4.2 Ausgabe

Die Strategie liefert optional ein Signal/Intent:

- Typ: **stark typisiert** (Rust struct/enum im `types` Crate, serialisierbar via `serde`).
  - Referenz: `crates/types/src/signal.rs`.

Minimal benötigte Signal-Felder (konzeptionell):

- `direction` (`Long|Short`)
- `order_type` (`Market` im MVP)
- `sl` (Preis)
- `tp` (Preis)
- `scenario_id` (1..6)
- `tags` (z.B. `scenario2`, `kalman`, `bollinger`, `vol_cluster`)
- `meta` (indikator-/filter-relevante Snapshots für Output)

**Wichtig:** Trades entstehen erst in `execution`/`backtest` Crate (siehe Execution-Plan). Strategie erzeugt kein Trade-Objekt direkt.

---

## 5. Guards / Filters (nicht monolithisch)

### 5.1 Session Filter (deaktivierbar)

- Konfig: `config.sessions: array | null` (siehe `OMEGA_V2_CONFIG_SCHEMA_PLAN.md`).
- Semantik:
  - `sessions == null` ⇒ **kein** Session-Blocking.
  - sonst: `BarContext.session_open` ist true, falls bar-close Zeitpunkt in mindestens einer Session liegt.
  - Cross-midnight ist erlaubt (`end <= start`).

### 5.2 News Filter (blockt X Minuten vor/nach News)

**Ziel:** Signal wird geblockt, wenn bar-close innerhalb eines Blackout-Intervalls liegt:

- `[news_time - minutes_before, news_time + minutes_after]`

**Architektur (performant, nicht monolithisch):**

- News werden im `data` Crate geladen und zu einer per-Bar Maske kompiliert:
  - `news_mask: Vec<bool>` mit `len == primary_candles.len()`
  - `BarContext.news_blocked = news_mask[idx]`
- Konfig: `config.news_filter` (enabled + minutes_before/after + impact + currencies).
- Strategie prüft ausschließlich `BarContext.news_blocked`.

Damit bleibt die Strategie frei von IO, Parsing und Index-Building.

### 5.3 One-position-per-symbol Gate

- MVP-Regel: pro Symbol nur eine offene Position.
- Gate liegt im Backtest-Orchestrator/Portfolio-Layer.
- Strategie wird nicht aufgerufen, wenn das Gate im jeweiligen Modus ein neues Signal verhindern soll (oder die Strategie gibt intern `None` zurück – beide sind erlaubt, aber eine Implementationsentscheidung).

### 5.4 Cooldown

- Cooldown ist Bestandteil der „Entry Eligibility“.
- Empfehlung: im `backtest` Orchestrator als symbol-lokaler Zustand (z.B. `last_entry_ts`, `last_exit_ts`) implementieren, nicht im Indikator-Stack.

---

## 6. Indikatoren – benötigte Menge (V1-Parität)

Die MRZ-Strategie benötigt (MVP):

### 6.1 Core (Primary TF)

- EMA (`ema_length`)
- ATR (Wilder) (`atr_length`, `atr_mult`)
- Bollinger Bands (`b_b_length`, `std_factor`) – SMA-basiert
- Z-Score (klassisch) (`window_length`)
- Kalman Z-Score (`window_length`, `kalman_r`, `kalman_q`)
- GARCH Volatilität (GARCH(1,1)):
  - `garch_alpha`, `garch_beta`, optional `garch_omega`, `garch_use_log_returns`, `garch_scale`, `garch_min_periods`, `garch_sigma_floor`
- Kalman+GARCH Z-Score (Kalman-Residual / GARCH-Sigma)

### 6.2 Scenario-5 (Intraday Vol Cluster)

- Vol-Cluster-State über **GARCH-Sigma-Serie** (Feature: `garch_forecast`).
- `calculate_vol_cluster_state(series, window, k, min_points, log_transform)`
- `cluster_hysteresis_bars` muss berücksichtigt werden.

### 6.3 Scenario-6 (Multi-TF Overlay)

Pro Overlay-TF:

- Kalman-Z (TF-spezifischer `window_length`, `kalman_r`, `kalman_q`)
- Bollinger (TF-spezifisch `b_b_length`, `std_factor`)

**Wichtig:** Overlay-TFs werden aus separaten Parquets geladen (MVP, keine Aggregation aus primary).

---

## 7. Szenarien – vollständige Spezifikation (1–6)

Allgemeine Vorbedingungen für alle Szenarien (sofern nicht explizit anders):

- Warmup erfüllt (mindestens so viele Bars, dass alle Indikatoren valide Werte liefern).
- `session_open == true` (wenn `sessions != null`).
- `news_blocked == false` (wenn Newsfilter enabled).
- One-position-per-symbol Gate erlaubt.
- Cooldown erlaubt.
- Direction Filter erlaubt (`direction_filter: long|short|both`).

### 7.1 Szenario 1 – Z-Score + EMA Take Profit

**Long (1):**

Entry, wenn:

- `zscore_now <= z_score_long` (negative Schwelle)

SL/TP:

- `sl = low - atr_mult * atr`
- `tp = ema_now`

**Short (1):**

Entry, wenn:

- `zscore_now >= z_score_short` (positive Schwelle)

SL/TP:

- `sl = high + atr_mult * atr`
- `tp = ema_now`

Hinweis zur Parität:

- V2 orientiert sich an der **Backtest-Semantik** (Preisunits, nicht pip-units für `tp_min_distance`, das nur Szenario 3 betrifft).

### 7.2 Szenario 2 – Kalman-Z + Bollinger, TP=BB-Mid

**Long (2):**

Entry, wenn alle zutreffen:

- HTF Trend Bias erlaubt (siehe Abschnitt 8)
- `kalman_z_now <= z_score_long`
- `close <= bb_lower_now`

SL/TP:

- `sl = low - atr_mult * atr`
- `tp = bb_mid_now`

**Short (2):**

Entry, wenn:

- HTF Trend Bias erlaubt
- `kalman_z_now >= z_score_short`
- `close >= bb_upper_now`

SL/TP:

- `sl = high + atr_mult * atr`
- `tp = bb_mid_now`

### 7.3 Szenario 3 – Kalman-Z + Bollinger, TP=EMA mit Mindestabstand

Wie Szenario 2, aber TP ist EMA und es gilt eine Mindestdistanz.

**Long (3):**

Zusätzliche Bedingung:

- `ema_now > ask_close + tp_min_distance`

TP:

- `tp = ema_now`

**Short (3):**

Zusätzliche Bedingung:

- `ema_now < bid_close - tp_min_distance`

TP:

- `tp = ema_now`

**Normativ (V2 Backtest):** `tp_min_distance` ist **Preisabstand** (nicht Pips).

### 7.4 Szenario 4 – Kalman+GARCH Z + Bollinger, TP=BB-Mid

Wie Szenario 2, aber Z-Signal ist `kalman_garch_z`.

**Long (4):**

- HTF Trend Bias erlaubt
- `kalman_garch_z_now <= z_score_long`
- `close <= bb_lower_now`

SL/TP: wie Szenario 2.

**Short (4):**

- HTF Trend Bias erlaubt
- `kalman_garch_z_now >= z_score_short`
- `close >= bb_upper_now`

SL/TP: wie Szenario 2.

### 7.5 Szenario 5 – Szenario 2 + Intraday Vol Cluster Guard

Basis ist Szenario 2. Zusätzlich muss der Vol-Cluster Guard passieren.

Vol-Cluster Guard (garch_forecast):

- Berechne Vol-Cluster-State aus GARCH-Sigma-Serie.
- `label ∈ intraday_vol_allowed` (Default: `low|mid`).
- Hysterese:
  - wenn `cluster_hysteresis_bars > 1`, muss das Cluster-Label für die letzten `h` Bars konstant sein.

Meta:

- `meta.vol_cluster` wird im Signal abgelegt (label, centers, sigma, status, allowed_labels, hysteresis_ok).

### 7.6 Szenario 6 – Szenario 2 + Multi-TF Overlay

Basis ist Szenario 2. Zusätzlich muss eine Overlay-Kette über weitere TFs passieren.

Konfig:

- `scenario6_timeframes`: Liste der TFs
- `scenario6_mode`: `all|any`
- `scenario6_params`: pro TF (und optional pro direction) Parameter für Kalman/Bollinger/Z-Schwelle

Pro TF wird geprüft:

- (Long) `kalman_z_now <= z_score_long` und `price_now <= lower_band_now`
- (Short) `kalman_z_now >= z_score_short` und `price_now >= upper_band_now`

Entscheidung:

- `all`: alle TFs müssen `ok` sein
- `any`: mindestens ein TF muss `ok` sein

Meta:

- `meta.scenario6 = { mode, chain: [ { tf, ok, status, z, threshold, price, upper/lower, params } ... ] }`

---

## 8. HTF Trend Filter (Bias) – Zerlegung & Semantik

Ziel: Entry-Signale in Richtung eines höherzeitlichen Bias erlauben.

### 8.1 Inputs

- D1 Bias (primary HTF) mit EMA-Länge `htf_ema`
- Optional: zusätzlicher Bias (H4/H1) mit `extra_htf_ema` und Relation-Regeln

### 8.2 Semantik (kompatibel zum V1 Backtest)

- Bias A: `price` vs `EMA(htf_tf)` je nach Filter `above|below|both|none`.
- Bias B: optionaler Filter `extra_htf_filter`.
- Zusätzlich kann eine „Min-Filters-Required“ Policy existieren (z.B. 1 von 2 muss passen).

**Wichtig:** Die konkrete HTF-Logik ist Teil der Strategie-Implementierung, aber die HTF-Daten werden vom Data-Layer geliefert (separate Parquets).

---

## 9. Konfigurationsoberfläche für MRZ (strategy.parameters)

Dieser Plan beschreibt die Parameter **inhaltlich**; das Top-Level Schema ist normativ in `OMEGA_V2_CONFIG_SCHEMA_PLAN.md`.

### 9.1 Kernparameter (Primary TF)

- `ema_length: int`
- `atr_length: int`
- `atr_mult: float`
- `b_b_length: int`
- `std_factor: float`
- `window_length: int`
- `z_score_long: float` (negativ)
- `z_score_short: float` (positiv)
- `kalman_q: float`
- `kalman_r: float`

### 9.2 HTF

- `htf_tf: string` (z.B. `D1`)
- `htf_ema: int`
- `htf_filter: above|below|both|none`
- `extra_htf_tf: string` (z.B. `H4`)
- `extra_htf_ema: int`
- `extra_htf_filter: above|below|both|none`

### 9.3 GARCH

- `garch_alpha: float`
- `garch_beta: float`
- `garch_omega: float|null`
- `garch_use_log_returns: bool`
- `garch_scale: float`
- `garch_min_periods: int`
- `garch_sigma_floor: float`

### 9.4 Scenario 5

- `intraday_vol_cluster_window: int`
- `intraday_vol_cluster_k: int`
- `intraday_vol_min_points: int`
- `intraday_vol_log_transform: bool`
- `intraday_vol_allowed: array<string>` (Default: `low|mid`)
- `intraday_vol_feature: "garch_forecast"` (MVP)
- `cluster_hysteresis_bars: int`
- `intraday_vol_garch_lookback: int`

### 9.5 Scenario 6

- `scenario6_mode: all|any`
- `scenario6_timeframes: array<string>`
- `scenario6_params: object` (per timeframe, optional direction)

### 9.6 Gating

- `direction_filter: long|short|both`
- `enabled_scenarios: array<int>` (IDs 1..6)
- `use_position_manager: bool` (MVP: false; Position Manager separat geplant)

---

## 10. Modul-Zerlegung (ohne Widerspruch zur Modul-Struktur)

Referenz: `crates/strategy/src/strategies/mean_reversion_z_score.rs` (Module Structure Plan).

Normative Empfehlung:

- `mean_reversion_z_score.rs` bleibt als Modul-Entry erhalten.
- Interne Submodule dürfen als Ordnerstruktur existieren, z.B.:
  - `mean_reversion_z_score/scenarios.rs`
  - `mean_reversion_z_score/params.rs`
  - `mean_reversion_z_score/htf.rs`
  - `mean_reversion_z_score/meta.rs`

So bleibt die globale Struktur stabil, während die Implementierung nicht monolithisch wird.

---

## 11. Offene Punkte / Follow-ups

### 11.1 Position Manager Plan (offen)

Ein eigener Plan für Trade-/Position-Management wird erstellt.

Bis dahin gilt für MRZ im MVP:

- `use_position_manager = false`
- Exits erfolgen ausschließlich über SL/TP/Timeouts gemäß Execution/Backtest-Orchestrierung.

### 11.2 Indikator-Plan (noch nicht vorhanden)

Ein dedizierter Indikator-Plan ist noch nicht geschrieben.

Bis dahin ist dieses Dokument die normative Referenz für die **benötigte Indikator-Menge** der MRZ-Strategie.

---

## 12. Testanforderungen (Parität & Determinismus)

Referenz: `OMEGA_V2_TESTING_VALIDATION_PLAN.md`.

Pflicht-Tests für MRZ:

1. **Golden Parity** gegen V1 Backtest-Implementierung:
   - identische Inputs (Parquet, Config) ⇒ identische `trades.json` (normalisiert) und `metrics.json` Kernwerte.
2. **Scenario Unit Tests** (synthetische Candle-Sequenzen):
   - Jede Szenario-Regel 1–6 erzeugt ein Signal, wenn Bedingungen erfüllt sind.
   - Jede Regel erzeugt **kein** Signal, wenn ein Guard greift (News/Sessions).
3. **News mask invariants**:
   - `news_mask.len() == primary.len()` und deterministisch.

---

## 13. Implementationshinweise (Paritäts-Fallen vermeiden)

- **Scenario-3 `tp_min_distance`**: In V1 Backtest ist dies ein **Preisabstand**. V2 muss diese Semantik übernehmen.
- **ATR**: Wilder ATR (EMA-ähnliche Glättung), nicht Simple ATR.
- **Bollinger**: SMA + std über `period`, gleiche NaN-/Warmup-Policy.
- **Kalman**: Algorithmus/Initialisierung muss V1 entsprechen (sonst drift in Entry-Signalen).
- **GARCH**: gleiche Return-Definition (log vs simple) und `scale`.
- **Zeit**: Entscheidend ist der bar-close Zeitstempel; keine Off-by-one Fehler.
