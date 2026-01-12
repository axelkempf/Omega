# Omega V2 – Output-Contract (Backtest Artefakte)

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Normative Spezifikation der Output-Artefakte für Backtests (Schema, Naming, Pfade, Zeit-/Units-Contract)  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, FFI-Grenze, Verantwortlichkeiten |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Bar/Time-Kontrakte, Result-Building |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell: Fills/Exits erzeugen Trades + Reasons |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Module/Crates, Result-Typen, Serialisierung |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Normative Metrik-Keys, Definitionen/Units, Scores, Rundung |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema (Input) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für Omega V2 Backtest-Outputs.

**Normative Artefakte pro Run (MVP):**

1. `trades.json` – abgeschlossene Trades (Root ist ein JSON-Array)
2. `equity.csv` – Equity-Curve **pro Bar** (Zeitstempel pro Bar)
3. `metrics.json` – Kernmetriken (Werte + Units/Definition)
4. `meta.json` – Run-Metadaten (Kontext, Versions-/Config-Infos)

**Zentrale Designentscheidungen (bereits entschieden):**

- **Dateinamen**: `metrics.json` (kein `summary.json` im V2-Contract).
- **Zeit**: Outputs enthalten **ISO-8601** *und* einen **Integer-Zeitstempel**.
- **Währungen/Units**: keine `_eur`-Suffixe; Units werden explizit beschrieben.
- **Warnings**: kein separates Warning-Artefakt im MVP; Fehler sind **hard fail**.
- **Output-Pfad**: flach nach `run_id`: `var/results/backtests/<run_id>/`.

---

## 2. Geltungsbereich

### 2.1 In Scope

- Backtest-Outputs unter `var/results/backtests/<run_id>/`.
- Schema-/Naming-Contracts für `trades.json`, `equity.csv`, `metrics.json`, `meta.json`.
- Zeit-/Units-/Encoding-Regeln, die alle Artefakte konsistent halten.

### 2.2 Out of Scope (vorerst)

- Optimizer-/Walkforward-spezifische Aggregate/Reports (z.B. Multi-Run Summary).
- UI-/Live-Engine Artefakte.
- Streaming-/Realtime-Outputs während eines laufenden Backtests.

---

## 3. Output-Layout & Naming

### 3.1 Run-Ordner

Jeder Backtest-Run schreibt genau einen Output-Ordner:

- `var/results/backtests/<run_id>/`

**Normativ:** `<run_id>` ist ein **stabiler, eindeutiger Identifier** (z.B. UUIDv4 oder Hash), der in `meta.json` wiederholt wird.

### 3.2 Dateien im Run-Ordner (MVP)

| Datei | Pflicht? | Typ | Kurzbeschreibung |
|------|----------|-----|------------------|
| `meta.json` | MUSS | JSON | Kontext/Provenance (Config, Versionen, run_id, Zeitrahmen, Symbol) |
| `trades.json` | MUSS | JSON | Liste abgeschlossener Trades (Entry/Exit) |
| `equity.csv` | MUSS | CSV | Equity/Balances **pro Bar** |
| `metrics.json` | MUSS | JSON | Kernmetriken für Vergleich/Optimizer |

---

## 4. Gemeinsame Konventionen (alle Artefakte)

### 4.1 Encoding & Files

- UTF-8.
- Zeilenenden: `\n`.
- Datei endet mit Newline.

### 4.2 Zeit-Contract

Omega V2 nutzt UTC als einzige Zeitzone.

**Normativ pro Zeitstempel-Feld:**

- Ein ISO-8601 Feld (String) **UND** ein Integer-Feld (Epoch-Zeit).

**Integer-Zeitstempel:**

- Name: `*_time_ns` bzw. `timestamp_ns`.
- Typ: signed 64-bit integer.
- Einheit: **Nanoseconds since Unix epoch (UTC)**.

**ISO-8601:**

- Name: `*_time` bzw. `timestamp`.
- Muss UTC ausdrücken: entweder mit Suffix `Z` oder Offset `+00:00`.

### 4.3 Zahlen & Units

- Dezimalzahlen: JSON number / CSV float.
- Units werden entweder
  - im Feldnamen **nicht** kodiert (keine `_eur`, `_usd`, etc.), sondern
  - über Meta/Schema beschrieben (z.B. `account_currency` in `meta.json`) oder
  - über explizite Unit-Felder in `metrics.json`.

---

## 5. Artefakt: `meta.json`

### 5.1 Zweck

`meta.json` macht einen Run reproduzierbar und auditierbar: Welche Config, welche Datenbasis, welche Versionen, welcher Zeitbereich.

### 5.2 Format

- JSON Object (Root ist ein Objekt).

### 5.3 Schema (MVP)

| Feld | Pflicht? | Typ | Bedeutung |
|------|----------|-----|----------|
| `run_id` | MUSS | string | Ordnername/Run-Identifier |
| `generated_at` | MUSS | string (ISO) | Zeitpunkt der Artefakt-Erzeugung (UTC) |
| `generated_at_ns` | MUSS | integer | Zeitpunkt der Artefakt-Erzeugung (UTC, ns) |
| `engine` | MUSS | object | Engine-Metadaten |
| `engine.name` | MUSS | string | z.B. `omega-v2` |
| `engine.version` | SOLL | string | SemVer oder Git describe |
| `git` | SOLL | object | Git/Build-Provenance |
| `git.commit` | SOLL | string | Commit SHA |
| `git.dirty` | SOLL | boolean | Working tree dirty? |
| `config` | MUSS | object | Snapshot oder Referenz der verwendeten Config |
| `config.source` | SOLL | string | z.B. Pfad zur JSON-Datei |
| `config.hash` | MUSS | string | Hash des normierten Config-JSON |
| `dataset` | SOLL | object | Datenbasis (Symbol/TF/Provider) |
| `dataset.symbol` | MUSS | string | z.B. `EURUSD` |
| `dataset.timeframe` | MUSS | string | z.B. `M1` |
| `dataset.start_time` | MUSS | string (ISO) | Startzeit des Backtests (UTC) |
| `dataset.start_time_ns` | MUSS | integer | Startzeit (UTC, ns) |
| `dataset.end_time` | MUSS | string (ISO) | Endzeit des Backtests (UTC) |
| `dataset.end_time_ns` | MUSS | integer | Endzeit (UTC, ns) |
| `account` | SOLL | object | Währung/Initialwerte |
| `account.account_currency` | SOLL | string | z.B. `EUR` |
| `account.initial_balance` | SOLL | number | Start-Balance in `account_currency` |

**Hinweis:** `config` darf im MVP entweder den kompletten Config-Snapshot enthalten oder einen Minimal-Snapshot + Hash. Normativ ist mindestens `config.hash`.

---

## 6. Artefakt: `trades.json`

### 6.1 Zweck

`trades.json` enthält alle abgeschlossenen Trades (Entry/Exit) als Grundlage für Analyse, Debugging, Regression und spätere Walkforward-/Optimizer-Auswertungen.

### 6.2 Format

- JSON Array (Root ist ein Array).
- Elemente sind Trade-Objekte.

### 6.3 Schema (MVP, V1-kompatibler Superset)

Die folgenden Felder sind **MUSS** (minimaler, stabiler Contract):

| Feld | Pflicht? | Typ | Bedeutung |
|------|----------|-----|----------|
| `entry_time` | MUSS | string (ISO) | Entry-Zeit (UTC) |
| `entry_time_ns` | MUSS | integer | Entry-Zeit (UTC, ns) |
| `exit_time` | MUSS | string (ISO) | Exit-Zeit (UTC) |
| `exit_time_ns` | MUSS | integer | Exit-Zeit (UTC, ns) |
| `direction` | MUSS | string | `long` oder `short` |
| `symbol` | SOLL | string | z.B. `EURUSD` |
| `entry_price` | MUSS | number | Entry-Preis (Quote) |
| `exit_price` | MUSS | number | Exit-Preis (Quote) |
| `stop_loss` | SOLL | number | SL-Preis (Quote) |
| `take_profit` | SOLL | number | TP-Preis (Quote) |
| `size` | SOLL | number | Positionsgröße (Lot/Units, siehe Strategy/Config) |
| `result` | SOLL | number | Realisierte PnL in `account_currency` **vor** expliziten Fees/Commission (Fees werden separat aggregiert) |
| `r_multiple` | SOLL | number | R-Multiple (dimensionslos) |
| `reason` | SOLL | string | Exit-Grund (z.B. `take_profit`, `stop_loss`, `manual`, `timeout`) |
| `meta` | SOLL | object | Freie, serialisierbare Zusatzinfos (szenario/labels/etc.) |

**Sortierung (SOLL):** Trades sind nach `exit_time_ns` aufsteigend sortiert.

### 6.4 Beispiel

```json
[
  {
    "entry_time": "2026-01-01T00:05:00Z",
    "entry_time_ns": 1767225900000000000,
    "exit_time": "2026-01-01T00:20:00Z",
    "exit_time_ns": 1767226800000000000,
    "direction": "long",
    "symbol": "EURUSD",
    "entry_price": 1.08123,
    "exit_price": 1.08210,
    "stop_loss": 1.08050,
    "take_profit": 1.08280,
    "size": 0.10,
    "result": 12.34,
    "r_multiple": 1.25,
    "reason": "take_profit",
    "meta": {
      "scenario": "base",
      "magic_number": 123456
    }
  }
]
```

---

## 7. Artefakt: `equity.csv`

### 7.1 Zweck

`equity.csv` ist die kanonische Equity-Curve als Zeitreihe **pro Bar**.

### 7.2 Format

- CSV mit Header.
- Eine Zeile pro Bar im primären Backtest-Timeframe.

### 7.3 Schema (MVP)

| Spalte | Pflicht? | Typ | Bedeutung |
|--------|----------|-----|----------|
| `timestamp` | MUSS | string (ISO) | Bar-Zeit (UTC) |
| `timestamp_ns` | MUSS | integer | Bar-Zeit (UTC, ns) |
| `equity` | MUSS | number | Equity nach Verarbeitung dieser Bar (in `account_currency`) |
| `balance` | SOLL | number | Balance/Cash nach Bar (in `account_currency`) |
| `drawdown` | SOLL | number | Drawdown relativ zum High-Water (0..1) |
| `high_water` | SOLL | number | High-Water-Mark der Equity (in `account_currency`) |

**Kompatibilität:** Für einfache V1-Tools sind mindestens `timestamp` und `equity` vorhanden.

### 7.4 Beispiel

```csv
timestamp,timestamp_ns,equity,balance,drawdown,high_water
2026-01-01T00:00:00Z,1767225600000000000,10000.00,10000.00,0.0,10000.00
2026-01-01T00:01:00Z,1767225660000000000,10002.50,10000.00,0.0,10002.50
```

---

## 8. Artefakt: `metrics.json`

### 8.1 Zweck

`metrics.json` liefert die Metriken für Vergleichbarkeit, Optimizer-Scoring und Regression-Tests.

### 8.2 Format

- JSON Object (Root ist ein Objekt).

### 8.3 Schema (MVP)

Das konkrete Set der Kernmetriken ist inhaltlich Teil des MVP-Scopes. Für den Output-Contract gilt:

- `metrics.json` enthält ein flaches Objekt `metrics` und ein Objekt `definitions`.
- `metrics` enthält nur Zahlen/Booleans/Strings (keine Arrays), damit es leicht aggregierbar ist.
- Das **konkrete Key-Set inkl. Semantik und `definitions`-Felder** ist normiert in: `OMEGA_V2_METRICS_DEFINITION_PLAN.md`.

Hinweis: Das folgende JSON-Beispiel ist **minimal**; in V2 werden in `definitions` zusätzliche Felder wie `description`, `domain`, `source` und `type` erwartet (siehe Metrics-Definition-Plan).

| Feld | Pflicht? | Typ | Bedeutung |
|------|----------|-----|----------|
| `metrics` | MUSS | object | Key → Value |
| `definitions` | MUSS | object | Key → (Unit/Definition) |

**Empfohlene Kern-Keys (SOLL, MVP-Minimum):**

- `total_trades`, `wins`, `losses`, `win_rate`
- `profit_gross`, `profit_net`, `fees_total`
- `max_drawdown`, `max_drawdown_abs`, `max_drawdown_duration_bars`
- `avg_r_multiple`, `profit_factor`

**Optionale Kern-Keys (SOLL, MVP+):**

- `avg_trade_pnl`, `expectancy`, `active_days`, `trades_per_day`

**Units/Definition (Beispiel):**

```json
{
  "metrics": {
    "total_trades": 42,
    "win_rate": 0.52,
    "profit_net": 123.45,
    "fees_total": 12.34,
    "max_drawdown": 0.18
  },
  "definitions": {
    "total_trades": {"unit": "count"},
    "win_rate": {"unit": "ratio"},
    "profit_net": {"unit": "account_currency"},
    "fees_total": {"unit": "account_currency"},
    "max_drawdown": {"unit": "ratio"}
  }
}
```

---

## 9. Fehlerpolitik (MVP)

- Es gibt **kein** `warnings.json` im MVP.
- Bei Contract-Verletzungen (z.B. fehlende Pflichtfelder) ist das Verhalten: **Run fails hard**.
- Fehlerdetails werden über Logs/Exception transportiert (nicht als Artefakt).

---

## 10. Validierung & Tests

### 10.1 Schema-Validierung

- `meta.json`, `trades.json`, `metrics.json`: JSON-Schema-Validierung in Python (CI) und/oder Rust (Runtime) ist vorgesehen.
- `equity.csv`: Validierung über Header + Typchecks.

### 10.2 Golden-File Tests

- Für deterministische Backtests (fixe Daten, fixe Seeds) werden Golden-Files (Artefakte) in CI verglichen.
- Vergleichs-Strategie:
  - JSON: sortierte Keys/normalisierte Floats.
  - CSV: strikte Header-Reihenfolge, numerische Toleranzen bei floats.

---

## 11. Erweiterungen (nicht MVP)

- Zusätzliche Artefakte (optional): `orders.json`, `fills.json`, `positions.json`, `events.json`.
- Streaming-Logs/Tracing Artefakte.
- Aggregation über Runs (Optimizer/Walkforward Reports) in separaten Contracts.

---

## 12. Checkliste (Definition of Done)

- Output-Ordner ist `var/results/backtests/<run_id>/`.
- Alle vier Artefakte existieren (`meta.json`, `trades.json`, `equity.csv`, `metrics.json`).
- Zeitstempel enthalten ISO + `*_ns`.
- `trades.json` Root ist Array.
- `equity.csv` ist per Bar.
- Keine `_eur`-Suffixe.
