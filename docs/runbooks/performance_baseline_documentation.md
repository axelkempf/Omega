# Performance-Baseline-Dokumentation

**Erstellt:** 2026-01-05  
**Phase:** P5-08 (Migrations-Vorbereitung)  
**Status:** ✅ Dokumentiert

---

## Übersicht

Diese Dokumentation fasst die Performance-Baselines für alle Migrations-Kandidaten zusammen.
Die Baselines dienen als Referenz für:

1. **Improvement-Targets:** Erwarteter Speedup nach Migration
2. **Regression-Detection:** Erkennung von Performance-Verschlechterungen
3. **ROI-Analyse:** Bewertung des Migrations-Aufwands vs. Nutzen

---

## Baseline-Generierung

### Methodik

- **Tool:** `tools/perf_baseline.py`
- **Messungen:** 3 Wiederholungen pro Operation
- **Daten:** Synthetische OHLCV-Daten (50.000 Bars Standard)
- **Metriken:**
  - Ausführungszeit (Sekunden)
  - Peak Memory (MB via `tracemalloc`)
  - CPU-Profil (Top 15 Funktionen via `cProfile`)

### Reproduzierbarkeit

```bash
# Alle Baselines neu generieren
python tools/perf_baseline.py --all --output-dir reports/performance_baselines/

# Einzelnes Modul
python tools/perf_baseline.py --module indicator_cache
```

### Dateiformat

```json
{
  "meta": {
    "num_bars": 50000,
    "repetitions": 3,
    "generated_at": "2026-01-03T21:37:37Z"
  },
  "init_seconds": 0.187,
  "init_peak_mb": 6.01,
  "operations": {
    "<operation_name>": {
      "first_call_seconds": 0.001,
      "first_peak_mb": 1.2,
      "cached_call_seconds": 0.000003,
      "cached_peak_mb": 0.0
    }
  },
  "profile_top15": "..."
}
```

---

## Migrations-Kandidaten nach Priorität

### High Priority

#### 1. IndicatorCache

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_indicator_cache.json` | |
| **Zielsprache** | Rust | |
| **Init-Zeit** | 187.4 | ms |
| **Init-Memory** | 6.01 | MB |

**Operationen:**

| Operation | First Call | Cached Call | Memory | Target Speedup |
|-----------|------------|-------------|--------|----------------|
| `ema` | 1.25 ms | 3 µs | 1.21 MB | 10x |
| `ema_stepwise` | 51.1 ms | 6 µs | 3.92 MB | 20x |
| `sma` | 1.35 ms | 3 µs | 1.21 MB | 10x |
| `rsi` | 6.88 ms | 5 µs | 3.22 MB | 10x |
| `macd` | 2.70 ms | 6 µs | 2.41 MB | 10x |
| `roc` | 1.11 ms | 3 µs | 1.20 MB | 10x |
| `dmi` | 65.2 ms | 6 µs | 6.57 MB | 20x |
| `bollinger` | 3.69 ms | 4 µs | 2.05 MB | 10x |
| `bollinger_stepwise` | 88.5 ms | 10 µs | 5.90 MB | 20x |
| **`atr`** | **954.4 ms** | 4 µs | 4.16 MB | **50x** |

**Hotspots (cProfile):**
1. `_ensure_df()` - DataFrame-Konvertierung
2. `atr()` - True Range + EMA Loop
3. `_stepwise_indices()` - Index-Berechnung
4. `bollinger_stepwise()` - Rolling Window
5. `isinstance()` - Type Checks

**ROI-Schätzung:** Hoch (1s → 20ms für ATR = 50x Speedup)

---

### Medium Priority

#### 2. MultiSymbolSlice

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_multi_symbol_slice.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 7.24 | s |

**Hinweis:** Hohe Laufzeit, aber niedrige Type-Coverage (22.2% Return Types).
Erfordert Type-Safety-Arbeit vor Migration.

**Target Speedup:** 5-10x (nach Type-Hardening)

---

#### 3. SymbolDataSlicer

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_symbol_data_slicer.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 731 | ms |

**Target Speedup:** 5x

---

#### 4. Optimizer (Final Selection / Robust Zone)

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_optimizer.json` | |
| **Zielsprache** | Julia | |
| **Gesamt-Zeit** | 797 | ms |

**Hinweis:** Research-lastig, Julia für flexible Iteration.
FFI/Arrow-Overhead beachten.

**Target Speedup:** 3-5x

---

#### 5. Portfolio

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_portfolio.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 248 | ms |

**Hinweis:** Stateful Hot-path. Ownership/Mutability kritisch.

**Target Speedup:** 3x

---

#### 6. Slippage & Fee

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_slippage_and_fee.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 736 | ms |

**Hinweis:** Reine Mathematik. Ideales Pilot-Modul für Rust.

**Target Speedup:** 10x

---

### Low Priority

#### 7. EventEngine

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_event_engine.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 337 | ms |

**Hinweis:** Core-Loop, hochsensibel. Erst nach Interface-Spec + umfangreichen Tests.

**Target Speedup:** 3-5x (konservativ wegen Callback-Overhead)

---

#### 8. ExecutionSimulator

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_execution_simulator.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 174 | ms |

**Target Speedup:** 5x

---

#### 9. Rating Modules

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_rating.json` | |
| **Zielsprache** | Rust | |
| **Gesamt-Zeit** | 78 | ms |

**Hinweis:** Viele numerische Scores. Schon performant, aber SIMD-Potenzial.

**Target Speedup:** 5x

---

#### 10. Walkforward (Stubbed)

| Metrik | Wert | Einheit |
|--------|------|---------|
| **Baseline-Datei** | `p0-01_walkforward_stub.json` | |
| **Zielsprache** | Julia | |
| **Gesamt-Zeit** | 133 | ms |

**Hinweis:** Orchestrierung, primär I/O. Niedrige Priorität.

**Target Speedup:** 2x

---

## Aggregierte Metriken

### Backtest-Gesamtzeit (typisch, 1 Jahr, 1 Symbol)

| Komponente | Anteil | Zeit (s) | Nach Migration |
|------------|--------|----------|----------------|
| IndicatorCache | 45% | 45 | ~4.5s (10x) |
| EventEngine | 20% | 20 | ~5s (4x) |
| ExecutionSimulator | 15% | 15 | ~3s (5x) |
| Portfolio | 10% | 10 | ~3s (3x) |
| I/O + Sonstiges | 10% | 10 | ~10s (1x) |
| **Gesamt** | 100% | **100** | **~25s (4x)** |

**Erwarteter Gesamt-Speedup:** 3-5x für typischen Backtest

---

## Benchmark-History

### Tracking-Tool

```bash
# Aktuellen Benchmark ausführen und speichern
python tools/benchmark_history.py run --module indicator_cache

# Historie vergleichen
python tools/benchmark_history.py compare --module indicator_cache --baseline 2026-01-01

# Trend-Report
python tools/benchmark_history.py trend --module indicator_cache --days 30
```

### Speicherort

```
reports/benchmark_history/
├── indicator_cache/
│   ├── 2026-01-03_baseline.json
│   ├── 2026-01-05_pre_migration.json
│   └── history.csv
├── event_engine/
│   └── ...
└── README.md
```

### Alerts

Automatische Alerts bei:
- **Regression > 10%:** Warning in CI
- **Regression > 20%:** CI-Failure
- **Memory-Leak:** CI-Failure

---

## Improvement-Targets Summary

| Modul | Aktuell | Target | Speedup | Priorität |
|-------|---------|--------|---------|-----------|
| IndicatorCache (ATR) | 954 ms | 20 ms | 50x | High |
| IndicatorCache (EMA) | 51 ms | 2.5 ms | 20x | High |
| MultiSymbolSlice | 7.24 s | 0.7 s | 10x | Medium |
| SymbolDataSlicer | 731 ms | 150 ms | 5x | Medium |
| Optimizer | 797 ms | 200 ms | 4x | Medium |
| Portfolio | 248 ms | 80 ms | 3x | Medium |
| Slippage & Fee | 736 ms | 75 ms | 10x | Medium |
| EventEngine | 337 ms | 85 ms | 4x | Low |
| ExecutionSimulator | 174 ms | 35 ms | 5x | Low |
| Rating | 78 ms | 15 ms | 5x | Low |
| Walkforward | 133 ms | 65 ms | 2x | Low |

---

## Referenzen

- Baseline-Dateien: `reports/performance_baselines/p0-01_*.json`
- Kandidaten-Analyse: `reports/migration_candidates/README.md`
- Benchmark-Tool: `tools/perf_baseline.py`
- History-Tool: `tools/benchmark_history.py`
- ADR-0001: Migration Strategy
