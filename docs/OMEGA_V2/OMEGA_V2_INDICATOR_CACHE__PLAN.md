# Omega V2 – Indicator Cache Plan (MRZ Szenarien 1–6)

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Spezifikation des Indikator-Cache (Multi-TF, Stepwise-Semantik, Cache-Keys, Paritätsanforderungen zu V1), inklusive vollständigem Indikator-Inventar für Mean Reversion Z-Score (Szenarien 1–6).

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Single-FFI, Verantwortlichkeiten |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Crates/Ordner, BarContext, Indicator-Engine Platzierung |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | MRZ Szenarien 1–6, benötigte Indikatoren, Guard-Logik |
| [OMEGA_V2_TRADE_MANAGER_PLAN.md](OMEGA_V2_TRADE_MANAGER_PLAN.md) | Trade-/Position-Management: Actions/Meta-Labels für Exits |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Bar-/Zeit-Contract, Multi-TF Alignment, Warmup |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Datenqualität, Monotonie, News-Mask |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Config-Schema: Parameterflächen (Indikator- und Szenario-Parameter) |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Paritäts-/Golden-Tests, Determinismus, Vergleichsregeln |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Debug-/Profiling-Artefakte, Performance Counter |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Rust/Python Boundary |

---

## 1. Zusammenfassung (universale Wahrheit)

Omega V2 benötigt für die MVP-Strategie **Mean Reversion Z-Score (MRZ)** einen Indikator-Cache, der:

- **Multi-Timeframe** Daten konsumiert (z.B. M1 primär + H1/H4/D1 Filter/Overlays).
- Indikatoren **auf ihrer nativen Timeframe-Granularität** berechnet.
- Ergebnisse **auf die primäre Timeframe** abbildet (Forward-Fill), ohne Lookahead.
- **Stepwise-Varianten** (EMA/Bollinger/Kalman-Z) bereitstellt, die nur bei *neuen Candles* des Ziel-TF neu berechnen und dazwischen deterministisch fortschreiben.
- Ergebnisse **lazy** (on first access) berechnet und **cached**.
- **Parität** zu V1 sicherstellt.

**Paritäts-Referenz (normativ):** Wenn V1 Python und V1 Rust voneinander abweichen, gilt **V1 Python** als Referenz („golden behavior“). V1-Rust-Implementierungen sind für Paritätsentscheidungen nicht relevant.

---

## 2. Scope

### 2.1 In Scope

- Indikator-Cache für MRZ Szenarien **1–6** (inkl. HTF-Filter + Scenario-6 Overlay).
- Indikator-Funktionen und Stepwise-Varianten, die in MRZ benötigt werden.
- Lokale Fenster-Varianten, soweit sie in MRZ genutzt werden:
  - lokale GARCH-Volatilität (`garch_volatility_local`) als Window-Serie
  - lokaler Kalman+GARCH Z-Score am Index (`kalman_garch_zscore_local`) als Skalar
  - Vol-Cluster Feature-Serie (`vol_cluster_series`) als Strategy-Helper
- Normative Regeln für:
  - Cache-Keys
  - NaN/Undefined Werte
  - Multi-TF Mapping
  - Test-/Paritätsstrategie

### 2.2 Out of Scope (vorerst)

- „Beliebige“ Indikatoren außerhalb MRZ (z.B. RSI, MACD, Ichimoku, etc.).
- Export von Indikator-Zeitreihen als Teil des V2 Output-Contracts (Outputs sind Trades/Equity/Metrics/Meta; siehe Output-Contract).

---

## 3. Normative Entscheidungen (aus der Klärungsrunde)

### 3.1 Struktur

- **Single crate**: `crates/indicators`.
- **One file per indicator** unter `crates/indicators/src/impl/`.

### 3.2 Parität & Vergleiche

- Paritätsbaseline: **V1 Python**.
- Vergleiche sind **toleranzbasiert**, nicht bitweise:
  - `NaN` gilt als gleich, wenn beide Seiten `NaN` an derselben Position sind.
  - Für endliche Werte gilt: $|a-b| \le \text{atol}$ (Default: `atol=1e-10`).

### 3.3 Multi-Timeframe & Mapping

- Indikatoren werden auf **nativer TF** berechnet.
- Mapping auf primäre TF erfolgt über **Forward-Fill** (ffill) nach dem Prinzip „letzter bekannter Wert“, ohne Lookahead.

### 3.4 Stepwise-APIs

- Stepwise-Varianten werden als explizite APIs bereitgestellt und sind *semantisch normativ*.

### 3.5 Missing Values (offener Punkt 4.1 – entschieden)

- Interne Indikator-Ausgaben sind `f64`-Arrays.
- **Undefined/Warmup-Werte** werden als `NaN` repräsentiert.
- Forward-Fill gilt nur **ab dem ersten endlichen Wert**; führende NaNs bleiben NaN.

---

## 4. Begriffe, Datenmodell, Preis-Seite

### 4.1 Candle-Input

Der Indikator-Cache arbeitet auf Candle-Daten (OHLCV) aus dem Data-Layer.

- Candle-Daten sind **NaN-frei** und monoton in der Zeit (Data Governance).
- Indikatoren dürfen `NaN` erzeugen (Warmup/Undefined).

### 4.2 Preis-Seite (`price_type`)

Alle Indikatoren sind mindestens für `BID` und `ASK` definierbar.

- `price_type` ist normativ `"bid" | "ask"` (case-insensitiv, intern normalisiert).

---

## 5. Indikator-Inventar (MRZ Szenarien 1–6)

Diese Liste ist die **Implementierungs-Checkliste** für `omega_indicators` (MVP).

### 5.1 Datenzugriff (Helper)

| API | Output | Zweck |
|-----|--------|-------|
| `get_closes(tf, price_type)` | `Vec<f64>` | Close-Serie für Indikatoren und Strategy-Checks |
| `get_candles(tf, price_type)` | `&[Candle]` | OHLC Zugriff (ATR/Bollinger; Debug; Warmup) |

### 5.2 Kernindikatoren (global, vektorisiert)

| Indikator | Datei | Output | Wichtige Parameter | Stepwise? | MRZ Nutzung |
|----------|-------|--------|--------------------|-----------|------------|
| EMA | `impl/ema.rs` | `Vec<f64>` | `period` | Ja (`ema_stepwise`) | Trendfilter, Mittelwert, Trigger |
| ATR (Wilder) | `impl/atr.rs` | `Vec<f64>` | `period` | Nein | SL/TP, Vol-Filter, ATR-Points |
| Bollinger Bands | `impl/bollinger.rs` | `(mid, upper, lower)` als `Vec<f64>` | `period`, `std_factor` | Ja (`bollinger_stepwise`) | Mean-Reversion Gates, Bands |
| Z-Score | `impl/z_score.rs` | `Vec<f64>` | `window`, `mean_source`, `ema_period` | Nein | Scenario 1 (rolling/EMA), Thresholds |
| Kalman Mean | `impl/kalman_mean.rs` | `Vec<f64>` | `R`, `Q` | Nein | Residual-Basis für Kalman-Z / Kalman+GARCH-Z |
| Kalman Z-Score | `impl/kalman_zscore.rs` | `Vec<f64>` | `window`, `R`, `Q` | Ja (`kalman_zscore_stepwise`) | Scenario 2/6, Mean-Reversion Trigger |
| GARCH Volatility (Returns) | `impl/garch_volatility.rs` | `Vec<f64>` | `alpha`, `beta`, `omega?`, `use_log_returns`, `scale`, `min_periods`, `sigma_floor` | Nein | Scenario 4/5, Vol-Regime |
| Kalman+GARCH Z-Score | `impl/kalman_garch_zscore.rs` | `Vec<f64>` | `R`, `Q` + GARCH params | Nein | Scenario 4 (global) |

### 5.3 Lokale Varianten (Window-basiert)

| Indikator/Helper | Datei | Output | Parameter | MRZ Nutzung |
|------------------|-------|--------|-----------|------------|
| GARCH Volatility (local window) | `impl/garch_volatility_local.rs` | `Vec<f64>` (Fenster, bis inkl. `idx`) | `idx`, `lookback` + GARCH params | Scenario 5 (Clustering Feature Window) |
| Kalman+GARCH Z (local scalar) | `impl/kalman_garch_zscore_local.rs` | `Option<f64>` | `idx`, `lookback` + Kalman/GARCH params | Scenario 4/5/6 (robust local gating) |
| Vol-Cluster Feature Serie | `impl/vol_cluster_series.rs` | `Option<VolFeatureSeries>` | `idx`, `feature`, `atr_length`, `garch_lookback` + params | Scenario 5 (Vol-Regime Cluster) |

**Hinweis:** `VolFeatureSeries` ist ein kleines, typisiertes Ergebnis (z.B. `enum`), das entweder eine „volle“ Serie (ATR) oder ein lokales Fenster (GARCH local) repräsentiert.

---

## 6. API-Oberfläche (Rust intern)

### 6.1 Ziele

- Kein Lookahead.
- Deterministisch.
- O(1) Zugriff auf bereits berechnete Ergebnisse.
- Ergebnislängen sind normativ definiert.

### 6.2 Kern-Interface (konzeptionell)

Der Indikator-Cache ist **kein** dynamisches Plugin-System nach außen, sondern eine interne Komponente der Rust Engine.

Normativ (konzeptionell):

- `IndicatorCache::new(store: &CandleStore, primary_tf: Timeframe)`
- `IndicatorCache::{ema, ema_stepwise, atr, bollinger, bollinger_stepwise, zscore, kalman_mean, kalman_zscore, kalman_zscore_stepwise, garch_volatility, kalman_garch_zscore, garch_volatility_local, kalman_garch_zscore_local, vol_cluster_series}`

**Wichtig:** Die konkrete Signatur (Enums/Structs) ist Implementation Detail, aber die **Semantik** der Outputs ist Teil dieses Plans.

---

## 7. Multi-TF Semantik (Compute + Mapping)

### 7.1 Compute auf nativer TF

Ein Indikator $I$ für Timeframe `tf` wird auf den Candles dieser TF berechnet und erzeugt eine Serie $I_{tf}[k]$ mit Länge $N_{tf}$.

### 7.2 Mapping auf primäre TF

Für die Strategy-Auswertung auf der primären TF (z.B. `M1`) werden Werte auf Bar-Index-Ebene benötigt.

Normativ:

- Es existiert eine Mapping-Funktion `primary_index -> tf_index` ohne Lookahead.
- Für jeden primären Bar-Index $i$ gilt:
  - $k = \text{map}(i)$ ist der letzte TF-Bar, dessen Close-Zeit $\le$ primäre Close-Zeit.
  - `mapped[i] = series_tf[k]`.
- Ist `series_tf[k]` `NaN`, bleibt das Ergebnis `NaN` (kein Ffill über NaN hinweg, außer über bereits endliche Werte).

### 7.3 Stepwise-Semantik

Stepwise bedeutet: Der Indikator wird nur an den „Update-Punkten“ des TF neu berechnet und dazwischen fortgeschrieben.

Normativ:

- Update-Punkt = „neue Candle auf Ziel-TF“.
- Bei Update wird der neue Wert berechnet und dann auf alle primären Bars bis zum nächsten Update **forward-filled**.
- Vor dem ersten validen Wert bleibt alles `NaN`.

---

## 8. Cache-Design

### 8.1 Key-Schema

Alle gecachten Ergebnisse werden über einen stabilen Key adressiert:

- `name` (z.B. `"ema"`, `"kalman_garch_zscore"`)
- `tf`
- `price_type`
- `params_hash` (stabiler Hash über normalisierte Parameter)

Für lokale Indikatoren (idx/Window):

- zusätzlich `idx` und `lookback` im Key.

### 8.2 Lazy Compute

Normativ:

- Beim ersten Zugriff auf einen Key wird das Ergebnis berechnet und gespeichert.
- Folgezugriffe sind reine HashMap-Lookups.

**Hinweis:** Lokale Window-Keys können in der Praxis viele Varianten erzeugen. Implementierungen dürfen hierfür eine LRU/Cap einführen, solange Ergebnisse identisch bleiben.

---

## 9. Parität, Determinismus, Tests

### 9.1 Paritätsstrategie

- Für jeden Indikator existieren Paritätstests gegen **V1 Python**. V1-Rust-Implementierungen werden nicht als Referenz verwendet.
- Parität gilt für:
  - numerische Werte (toleranzbasiert)
  - `NaN`-Positionen
  - Stepwise-Semantik (Update-Punkte, ffill-Verhalten)

### 9.2 Vergleichsregeln (normativ)

- `NaN` == `NaN` an gleicher Position.
- Endliche Werte: `abs(a-b) <= 1e-10`.

### 9.3 Determinismus

- Keine Randomness.
- Keine Abhängigkeit von Systemzeit.
- Keine parallel-nichtdeterministische Reduktionen (keine nicht-stabile Summe ohne festes Verfahren).

---

## 10. Observability & Performance

### 10.1 Debug-Modus (SOLL)

Für Debug/Regressionsanalyse SOLL es möglich sein:

- ausgewählte Indikator-Serien (oder Hashes davon) in Debug-Artefakte zu schreiben,
- Hot-Paths (z.B. GARCH) als Profiling-Spans zu markieren.

Referenz: `OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md`.

### 10.2 Performance Leitlinien

- Berechnungen sind vektorisiert/iterativ in Rust und vermeiden Allocation-Churn.
- GARCH/Kalman sind die Hotspots; dort sind Single-Pass Implementierungen mit stabilen Guards normativ.
- Cache verhindert N-fache Recomputes pro Bar.

---

## 11. Offene Punkte

- Keine offenen Punkte im MVP-Scope dieses Plans.
