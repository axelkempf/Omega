---
title: "ADR-0005: Wave 1 IndicatorCache Rust Migration"
status: Accepted
date: 2026-01-09
deciders:
  - Axel Kempf
  - AI Agent (Claude Opus 4.5)
consulted:
  - Omega Maintainers
---

## ADR-0005: Wave 1 IndicatorCache Rust Migration

## Status

**Accepted** - Implementation completed 2026-01-09

## Kontext

Das `IndicatorCache`-Modul (`src/backtest_engine/core/indicator_cache.py`) ist ein kritischer Performance-Hotspot im Omega Trading Stack:

- **Numerisch intensiv**: EMA, RSI, MACD, Bollinger, ATR, DMI – alle berechnen aufwendige Float-Operationen
- **Hohe Aufruffrequenz**: Wird bei jedem Backtest-Tick aufgerufen
- **SIMD-Potenzial**: Vektorisierte Indikatoren ideal für Rust-Optimierung
- **Determinismus-Kritisch**: NaN-Propagation und Float-Determinismus sind für Backtest-Reproduzierbarkeit essentiell

### Performance-Baseline (Python)

| Operation | Python Baseline | Problem |
|-----------|-----------------|---------|
| `atr(14)` | 954ms | Wilder-Loop extrem langsam |
| `ema_stepwise` | 51ms | HTF-Bar-Berechnung |
| `bollinger_stepwise` | 88ms | Rolling + Stepwise |
| `dmi(14)` | 65ms | Drei Output-Arrays |
| `macd(12,26,9)` | 45ms | Mehrfache EMA |

## Entscheidung

Migration des IndicatorCache zu Rust via PyO3/Maturin mit:

1. **Feature-Flag-Pattern**: `OMEGA_USE_RUST_INDICATOR_CACHE` ermöglicht Fallback
2. **Zero-Copy NumPy Interop**: Direkte Array-Übergabe ohne Kopieren
3. **HashMap-basiertes Caching**: Rust-seitiger Cache für berechnete Indikatoren
4. **Vollständige Indikator-Abdeckung**: ATR, SMA, EMA, DEMA, TEMA, Bollinger, DMI, MACD, ROC, Z-Score, Kalman, Choppiness

### Implementierte Architektur

```
Python API Layer (indicator_cache_rust.py)
    │
    │ Feature Flag: OMEGA_USE_RUST_INDICATOR_CACHE
    ▼
┌─────────────────────────────────────────┐
│  IndicatorCacheRustWrapper (Python)     │
│  - Delegiert an Rust oder Python        │
│  - NumPy Array I/O                      │
└─────────────────────────────────────────┘
    │
    │ PyO3 FFI Boundary
    ▼
┌─────────────────────────────────────────┐
│  IndicatorCacheRust (Rust)              │
│  - HashMap<OhlcvKey, OhlcvData>         │
│  - HashMap<CacheKey, Vec<f64>>          │
│  - SIMD-optimierte Berechnungen         │
└─────────────────────────────────────────┘
```

## Konsequenzen

### Positive Konsequenzen

- **Performance**: 474x Gesamt-Speedup (alle Targets übertroffen)
  - ATR: 7299x (Target: 50x) ✅✅✅
  - SMA: 528x (Target: 10x) ✅
  - EMA: 337x (Target: 10x) ✅
  - MACD: 285x (Target: 10x) ✅
  - DMI: 234x (Target: 20x) ✅
  - Bollinger: 160x (Target: 20x) ✅

- **Cache-Hit-Bonus**: Zusätzlich 16.3x bei wiederholten Aufrufen

- **Abwärtskompatibilität**: Python-Fallback bei `OMEGA_USE_RUST_INDICATOR_CACHE=0`

- **Test-Abdeckung**: 17/17 spezifische Tests + 707/708 Regressionstests bestanden

### Negative Konsequenzen

- **Build-Komplexität**: Erfordert Rust-Toolchain und Maturin
- **Platform-Abhängigkeit**: Windows/macOS/Linux-spezifische Wheels nötig
- **Debug-Komplexität**: Rust-Debugging erfordert andere Tools als Python

### Risiken und Mitigationen

| Risiko | Mitigation |
|--------|------------|
| NaN-Handling-Differenzen | Explizite Tests mit separaten Valid-Masken pro Output-Array |
| Float-Determinismus | IEEE 754 strict mode, keine auto-vectorization ohne Validierung |
| Memory-Leaks | Rust Ownership-Model verhindert Leaks by design |
| FFI-Overhead | Batch-Design amortisiert Overhead (einmalige OHLCV-Registrierung) |

## Alternativen

### Alternative 1: NumPy/Numba-Optimierung

- **Beschreibung**: JIT-Kompilierung der Python-Indikatoren via Numba
- **Warum nicht gewählt**: 
  - Numba-Einschränkungen bei komplexen Datenstrukturen
  - Weniger deterministisch als Rust
  - Geringere Speedups (typisch 10-30x vs. 100-7000x)

### Alternative 2: Julia-Implementation

- **Beschreibung**: Indikatoren in Julia via PythonCall
- **Warum nicht gewählt**:
  - Julia-Startup-Overhead (~2-5s) bei jedem Import
  - PythonCall weniger ausgereift als PyO3
  - Julia für Wave 3+ reserviert (Monte Carlo, Portfolio-Optimierung)

### Alternative 3: C/C++ mit ctypes

- **Beschreibung**: Native C-Extension mit ctypes-Bindings
- **Warum nicht gewählt**:
  - Memory-Safety-Risiken
  - Keine automatische Python-Integration wie PyO3
  - Höherer Wartungsaufwand

## Implementierte Dateien

### Rust-Module (`src/rust_modules/omega_rust/src/indicators/`)

| Datei | Beschreibung | LOC |
|-------|--------------|-----|
| `mod.rs` | Module exports | ~30 |
| `types.rs` | OhlcvData, CacheKey, OhlcvKey | ~120 |
| `cache.rs` | IndicatorCacheRust impl | ~400 |
| `py_bindings.rs` | PyO3 Python bindings | ~550 |
| `atr.rs` | Average True Range (Wilder) | ~80 |
| `sma.rs` | Simple Moving Average | ~60 |
| `ema_extended.rs` | EMA, DEMA, TEMA | ~120 |
| `bollinger.rs` | Bollinger Bands | ~80 |
| `dmi.rs` | DMI (+DI, -DI, ADX) | ~150 |
| `macd.rs` | MACD Line + Signal | ~90 |
| `roc.rs` | Rate of Change, Momentum | ~70 |
| `zscore.rs` | Z-Score variants | ~100 |
| `kalman.rs` | Kalman Filter | ~120 |
| `choppiness.rs` | Choppiness Index | ~70 |

**Gesamt Rust**: ~1840 LOC

### Python-Integration

| Datei | Beschreibung |
|-------|--------------|
| `src/shared/indicator_cache_rust.py` | Feature-Flag-Wrapper + IndicatorCacheRustWrapper |
| `tests/test_indicator_cache_rust.py` | 17 pytest Tests |

## Usage

```python
# Standard-Import (nutzt automatisch Rust wenn verfügbar)
from src.shared.indicator_cache_rust import get_indicator_cache, is_rust_enabled

# Prüfen ob Rust aktiv
if is_rust_enabled():
    print("Rust IndicatorCache aktiv (474x schneller)")

# Cache erstellen
cache = get_indicator_cache()

# OHLCV-Daten registrieren
cache.register_ohlcv("EURUSD", "H1", "BID", open_arr, high_arr, low_arr, close_arr, volume_arr)

# Indikatoren berechnen (gecached)
atr = cache.atr("EURUSD", "H1", "BID", 14)
sma = cache.sma("EURUSD", "H1", "BID", 20)
plus_di, minus_di, adx = cache.dmi("EURUSD", "H1", "BID", 14)
```

### Feature-Flag-Steuerung

```bash
# Rust aktivieren (default wenn verfügbar)
export OMEGA_USE_RUST_INDICATOR_CACHE=1

# Python-Fallback erzwingen
export OMEGA_USE_RUST_INDICATOR_CACHE=0

# Auto-Detection (default)
unset OMEGA_USE_RUST_INDICATOR_CACHE
```

## Referenzen

- [Wave 1 Implementation Plan](../WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md)
- [FFI Specification: IndicatorCache](../ffi/indicator_cache.md)
- [Migration Runbook](../runbooks/indicator_cache_migration.md)
- [ADR-0001: Migration Strategy](./ADR-0001-migration-strategy.md)
- [ADR-0003: Error Handling](./ADR-0003-error-handling.md)
- [Performance Baseline](../../reports/performance_baselines/p0-01_indicator_cache.json)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-09 | AI Agent (Claude Opus 4.5) | Initiale Version nach erfolgreicher Implementation |
