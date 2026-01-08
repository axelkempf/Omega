# FFI Interface Specification: Rating Modules

**Modulpfad:** `src/backtest_engine/rating/`  
**Migrations-Ziel:** Rust (via PyO3/maturin) oder Julia (via PythonCall.jl)  
**Phase 2 Task:** P2-04  
**Status:** ✅ Spezifiziert (2026-01-05), Updated (2026-01-08)

---

## Executive Summary

Das Rating-Modul berechnet **Qualitäts-Scores** für Trading-Strategien basierend auf
Performance-Metriken. Diese Scores werden verwendet für:
- Strategie-Ranking (Final Selection)
- Robustheit gegen Parameter-Jitter
- Stabilität über Zeit (Jahresvergleich)
- Stress-Resistenz (Kosten-Schocks, Trade-Dropout)

**Hinweis:** `strategy_rating.py` wurde entfernt als Teil der Wave 1 Migration Preparation.
Die `rate_strategy_performance` Funktionalität wurde inline in die verwendenden Module
verschoben (`backtest_engine.optimizer.walkforward`). Die verbleibenden 10 Rating-Module
sind die primären Migrationskanditaten für Wave 1.

Migrations-Kandidat aufgrund:
- Numerisch intensive Berechnungen (Mittelwerte, Standardabweichungen)
- Monte-Carlo-artige Berechnungen (Dropout-Simulation)
- Pure Functions ohne Side Effects
- Klare Input/Output-Schemas

---

## Module Overview

| Modul | Funktion | Input | Output |
|-------|----------|-------|--------|
| `robustness_score_1.py` | Parameter-Jitter-Robustheit | base + jitter Metrics | Score [0,1] |
| `stability_score.py` | Jahres-Stabilität | profits_by_year | Score [0,1] |
| `cost_shock_score.py` | Kosten-Schock-Resistenz | base + shocked Metrics | Score [0,1] |
| `trade_dropout_score.py` | Trade-Dropout-Simulation | trades_df + dropout_frac | Metrics Dict |
| `stress_penalty.py` | Gemeinsame Penalty-Logik | base + stress Metrics | Penalty [0, cap] |
| `data_jitter_score.py` | Daten-Jitter-Robustheit | base + jittered Data | Score [0,1] |
| `timing_jitter_score.py` | Timing-Shift-Robustheit | base + shifted Metrics | Score [0,1] |
| `tp_sl_stress_score.py` | TP/SL-Stress-Test | trades_df + TP/SL Arrays | Score [0,1] |
| `ulcer_index_score.py` | Ulcer Index Score | equity_curve | Score [0,1] + Index |
| `p_values.py` | Statistische Signifikanz | trades_df | p-values Dict |

---

## Common Types

### MetricsDict

```python
# @ffi_boundary: Input

MetricsDict: TypeAlias = Mapping[str, float]

# Erwartete Schlüssel variieren pro Score-Typ:

# Für robustness_score_1:
{
    "profit": float,      # Netto-Profit in Konto-Währung
    "avg_r": float,       # Durchschnittliches R-Multiple
    "winrate": float,     # Win-Rate in Prozent (z.B. 55.0)
    "drawdown": float,    # Max Drawdown in Konto-Währung
}

# Für stress_penalty / cost_shock / trade_dropout:
{
    "profit": float,      # Netto-Profit
    "drawdown": float,    # Max Drawdown
    "sharpe": float,      # Sharpe Ratio
}

# Für Penalty-Berechnung (stress_penalty, cost_shock, timing_jitter, data_jitter):
{
    "profit": float,      # Netto-Profit
    "drawdown": float,    # Max Drawdown
    "sharpe": float,      # Sharpe Ratio
}
```

### Arrow Schema für MetricsDict

```
MetricsDict {
  keys: list<utf8>
  values: list<float64>
}

# Alternative: Fixed Schema für häufige Keys
PerformanceMetrics {
  profit: float64
  drawdown: float64
  sharpe: float64
  avg_r: float64
  winrate: float64
  profit_factor: float64
}
```

### ScoreResult

```python
# @ffi_boundary: Output

ScoreResult: TypeAlias = float  # [0.0, 1.0]

# Semantik:
# - 1.0 = Perfekt (keine Degradation)
# - 0.5 = Grenzwertig
# - 0.0 = Vollständig degradiert
```

---

## 1. robustness_score_1.py

### compute_robustness_score_1()

```python
def compute_robustness_score_1(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Robustheit gegen Parameter-Jitter.
    
    @ffi_boundary: Input/Output
    
    Args:
        base_metrics: Baseline-Performance (ohne Jitter)
            Keys: profit, avg_r, winrate, drawdown
        jitter_metrics: Liste von Runs mit leicht veränderten Parametern
        penalty_cap: Maximale Penalty [0.0, 0.5]
        
    Returns:
        Score [0.0, 1.0]
        
    Algorithm:
        1. Für jeden jitter Run:
           - pct_drop(profit) = max(0, (base - jitter) / base)
           - pct_drop(avg_r) = max(0, (base - jitter) / base)
           - pct_drop(winrate) = max(0, (base - jitter) / base)
           - pct_increase(drawdown) = max(0, (jitter - base) / base)
           - drop_i = mean(4 drops)
           
        2. penalty = mean(drops) clamped to [0, penalty_cap]
        
        3. score = 1.0 - penalty
        
    Edge Cases:
        - Empty jitter_metrics → return 1.0 - penalty_cap
        - Base metric <= 0 → denominator guarded with 1e-9
        - NaN/Inf values → treated as 0.0
    """
```

### Helper: _pct_drop()

```python
def _pct_drop(
    base: float,
    x: float,
    *,
    invert: bool = False,
) -> float:
    """
    Berechnet relative Verschlechterung.
    
    @ffi_boundary: Internal
    
    Normal (invert=False): max(0, (base - x) / base)
    Inverted (invert=True): max(0, (x - base) / base)  # für Drawdown
    """
```

---

## 2. stability_score.py

### compute_stability_score_and_wmape_from_yearly_profits()

```python
def compute_stability_score_and_wmape_from_yearly_profits(
    profits_by_year: Mapping[int, float],
    *,
    durations_by_year: Optional[Mapping[int, float]] = None,
) -> Tuple[float, float]:
    """
    Stabilität der Jahres-Profits relativ zum Erwartungswert.
    
    @ffi_boundary: Input/Output
    
    Args:
        profits_by_year: {year: profit_eur}
        durations_by_year: Optional {year: trading_days}
        
    Returns:
        (score, wmape)
        
    Algorithm:
        1. Berechne µ = total_profit / total_days (tägliche Rate)
        
        2. Für jedes Jahr y:
           E_y = µ * days_y  (erwarteter Profit)
           S_min = max(100, 0.02 * |P_total|)
           
        3. WMAPE (Weighted Mean Absolute Percentage Error):
           wmape = Σ w_y * |P_y - E_y| / max(|E_y|, S_min)
           wo w_y = days_y / total_days
           
        4. score = 1 / (1 + wmape)
        
    Edge Cases:
        - Empty profits_by_year → return (1.0, 0.0)
        - Invalid durations → fallback to calendar year days
    """
```

### compute_stability_score_from_yearly_profits()

```python
def compute_stability_score_from_yearly_profits(
    profits_by_year: Mapping[int, float],
    *,
    durations_by_year: Optional[Mapping[int, float]] = None,
) -> float:
    """Convenience wrapper, returns nur Score."""
```

### Arrow Schema für YearlyProfits

```
YearlyProfits {
  years: list<int32>
  profits: list<float64>
  durations: list<float64> (nullable)
}

# Alternativ: Struct per Year
YearData {
  year: int32
  profit: float64
  duration: float64 (nullable)
}
```

---

## 3. stress_penalty.py (Shared Foundation)

### compute_penalty_profit_drawdown_sharpe()

```python
def compute_penalty_profit_drawdown_sharpe(
    base_metrics: Mapping[str, float],
    stress_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Gemeinsame Penalty-Berechnung für Stress-Tests.
    
    @ffi_boundary: Input/Output
    
    Args:
        base_metrics: Baseline (profit, drawdown, sharpe)
        stress_metrics: Liste gestresster Runs
        penalty_cap: Max Penalty [0.0, 0.5]
        
    Returns:
        Penalty [0.0, penalty_cap]
        
    Algorithm:
        Für jeden stress Run:
          p = rel_drop(base_profit, stress_profit)
          d = rel_increase(base_drawdown, stress_drawdown)
          s = rel_drop(base_sharpe, stress_sharpe)
          penalty_i = (p + d + s) / 3
          
        penalty = mean(penalty_i) clamped to [0, cap]
    """
```

### score_from_penalty()

```python
def score_from_penalty(
    penalty: float,
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Konvertiert Penalty zu Score.
    
    @ffi_boundary: Input/Output
    
    Returns:
        score = max(0, 1.0 - penalty)
    """
```

---

## 4. cost_shock_score.py

### COST_SHOCK_FACTORS

```python
# @ffi_boundary: Config

COST_SHOCK_FACTORS: Tuple[float, ...] = (1.25, 1.50, 2.00)
# +25%, +50%, +100% Kosten-Erhöhung
```

### apply_cost_shock_inplace()

```python
def apply_cost_shock_inplace(
    cfg: Dict[str, Any],
    *,
    factor: float,
) -> None:
    """
    Modifiziert Config für Cost-Shock-Simulation.
    
    @ffi_boundary: Input (mutiert)
    
    Setzt:
        cfg["execution"]["slippage_multiplier"] *= factor
        cfg["execution"]["fee_multiplier"] *= factor
        
    Note:
        Wirkt auf die Multiplier, nicht direkt auf Slippage/Fee-Werte,
        um Doppel-Anwendung zu vermeiden.
    """
```

### compute_cost_shock_score()

```python
def compute_cost_shock_score(
    base_metrics: Mapping[str, float],
    shocked_metrics: Mapping[str, float],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Score für einzelnen Cost-Shock.
    
    @ffi_boundary: Input/Output
    
    Delegiert an compute_penalty_profit_drawdown_sharpe + score_from_penalty.
    """
```

### compute_multi_factor_cost_shock_score()

```python
def compute_multi_factor_cost_shock_score(
    base_metrics: Mapping[str, float],
    shocked_metrics_list: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Aggregierter Score über mehrere Shock-Faktoren.
    
    @ffi_boundary: Input/Output
    
    Returns:
        mean(score_per_shock)
        
    Für COST_SHOCK_FACTORS = (1.25, 1.50, 2.00):
        3 Backtests mit erhöhten Kosten
        → 3 Scores gemittelt
    """
```

---

## 5. trade_dropout_score.py

### simulate_trade_dropout_metrics()

```python
def simulate_trade_dropout_metrics(
    trades_df: Optional[pd.DataFrame],
    *,
    dropout_frac: float,
    base_metrics: Optional[Mapping[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    pnl_col: str = "result",
    r_col: str = "r_multiple",
    debug: bool | None = None,
) -> Dict[str, float]:
    """
    Simuliert Trade-Dropout und berechnet resultierende Metriken.
    
    @ffi_boundary: Input/Output
    
    Args:
        trades_df: DataFrame mit Trade-Historie
        dropout_frac: Anteil zu entfernender Trades [0.0, 1.0]
        base_metrics: Optional Fallback bei leerem Input
        rng: NumPy Random Generator
        seed: Optional Seed für Reproduzierbarkeit
        pnl_col: Spaltenname für P&L ("result")
        r_col: Spaltenname für R-Multiple ("r_multiple")
        debug: Enable Debug-Output
        
    Returns:
        {
            "profit": float,    # Nach Dropout
            "drawdown": float,  # Nach Dropout
            "sharpe": float,    # Nach Dropout
        }
        
    Algorithm:
        1. Sortiere Trades chronologisch (exit_time)
        2. Entferne dropout_frac * n Trades zufällig
        3. Berechne auf verbleibenden Trades:
           - profit = sum(pnl) - sum(fees)
           - drawdown = max peak-to-trough
           - sharpe = mean(r) / std(r)
           
    Fee-Handling:
        - Net-of-fee wenn Fee-Spalten vorhanden
        - Konsistent mit Portfolio.cash-Berechnung
    """
```

### Arrow Schema für TradesInput

```
TradesInput {
  exit_time: timestamp[us, tz=UTC]
  entry_time: timestamp[us, tz=UTC]
  result: float64          # P&L
  r_multiple: float64      # R-Multiple
  fee_entry: float64 (nullable)
  fee_exit: float64 (nullable)
}
```

### Internal Helpers

```python
def _drawdown_from_results(
    results: np.ndarray | None,
) -> float:
    """
    Max Drawdown aus P&L-Sequenz.
    
    @ffi_boundary: Internal
    
    Algorithm:
        cum = cumsum([0] + results)
        peaks = maximum.accumulate(cum)
        drawdown = max(peaks - cum)
    """

def _sharpe_from_r_multiples(
    r: np.ndarray,
) -> float:
    """
    Sharpe Ratio aus R-Multiples.
    
    @ffi_boundary: Internal
    
    Returns:
        mean(r) / std(r, ddof=1)
        0.0 wenn < 2 Trades oder std == 0
    """
```

---

## 6. data_jitter_score.py

### compute_data_jitter_score()

```python
def compute_data_jitter_score(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Robustheit gegen Daten-Jitter (simulierte Marktdaten-Rauschen).
    
    @ffi_boundary: Input/Output
    
    Args:
        base_metrics: Baseline-Performance (ohne Jitter)
            Keys: profit, drawdown, sharpe
        jitter_metrics: Liste von Runs mit leicht verrauschten Marktdaten
        penalty_cap: Maximale Penalty [0.0, 0.5]
        
    Returns:
        Score [0.0, 1.0]
        
    Algorithm:
        Delegiert an compute_penalty_profit_drawdown_sharpe + score_from_penalty.
    """
```

### Helper: build_jittered_preloaded_data()

```python
def build_jittered_preloaded_data(
    base_preloaded_data: Mapping[Tuple[str, str], pd.DataFrame],
    *,
    atr_cache: Mapping[str, pd.Series],
    sigma_atr: float = 0.10,
    seed: int,
    min_price: float = 1e-9,
    fraq: float = 0.0,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Erzeugt jittered OHLC-Daten basierend auf ATR-Skalierung.
    
    @ffi_boundary: Internal (nicht für FFI Migration)
    
    Algorithm:
        - Für jeden Candle: noise = N(0, sigma_atr * ATR)
        - Open, High, Low, Close += noise (mit Konsistenz-Korrektur)
    """
```

### Arrow Schema für DataJitterInput

```
DataJitterInput {
  base_metrics: PerformanceMetrics
  jitter_metrics: list<PerformanceMetrics>
  penalty_cap: float64
}
```

---

## 7. timing_jitter_score.py

### compute_timing_jitter_score()

```python
def compute_timing_jitter_score(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Robustheit gegen Timing-Shifts (verschobene Backtest-Fenster).
    
    @ffi_boundary: Input/Output
    
    Args:
        base_metrics: Baseline-Performance
        jitter_metrics: Performance mit verschobenen Zeitfenstern
        penalty_cap: Maximale Penalty [0.0, 0.5]
        
    Returns:
        Score [0.0, 1.0]
        
    Algorithm:
        Delegiert an compute_penalty_profit_drawdown_sharpe + score_from_penalty.
    """
```

### Helper: get_timing_jitter_backward_shift_months()

```python
def get_timing_jitter_backward_shift_months(
    start_date: str | datetime,
    end_date: str | datetime,
) -> List[int]:
    """
    Berechnet sinnvolle Rückwärts-Shifts in Monaten.
    
    @ffi_boundary: Internal
    
    Returns:
        Liste von Shift-Werten [6, 12, 3] oder angepasst je nach Fenstergröße.
    """
```

---

## 8. tp_sl_stress_score.py

### compute_tp_sl_stress_score()

```python
def compute_tp_sl_stress_score(
    trades_df: pd.DataFrame,
    primary_candle_arrays: Optional[PrimaryCandleArrays],
    *,
    tp_shift_pct_list: Sequence[float] = (-0.1, -0.2, -0.3),
    sl_shift_pct_list: Sequence[float] = (-0.1, -0.2, -0.3),
    penalty_cap: float = 0.5,
) -> float:
    """
    Stress-Test für TP/SL-Level Sensitivität.
    
    @ffi_boundary: Input/Output
    
    Args:
        trades_df: Trade-Historie mit entry_price, tp_price, sl_price, direction
        primary_candle_arrays: Aligned bid/ask Candles für Hit-Simulation
        tp_shift_pct_list: Liste von TP-Shift-Prozenten (negativ = enger)
        sl_shift_pct_list: Liste von SL-Shift-Prozenten (negativ = enger)
        penalty_cap: Maximale Penalty
        
    Returns:
        Score [0.0, 1.0]
        
    Algorithm:
        1. Für jeden Trade simuliere veränderte TP/SL-Hits
        2. Berechne neue P&L basierend auf Candle-Durchbrüchen
        3. Vergleiche mit Original-Performance
        4. Score = 1.0 - aggregierte Penalty
    """
```

### Arrow Schema für TPSLStressInput

```
TPSLStressInput {
  trades: list<TradeRecord>
  candle_times_ns: list<int64>
  bid_high: list<float64>
  bid_low: list<float64>
  ask_high: list<float64>
  ask_low: list<float64>
}

TradeRecord {
  entry_price: float64
  tp_price: float64
  sl_price: float64
  direction: utf8  # "long" | "short"
  entry_time_ns: int64
  exit_time_ns: int64
}
```

---

## 9. ulcer_index_score.py

### compute_ulcer_index_and_score()

```python
def compute_ulcer_index_and_score(
    equity_curve: Sequence[object] | Iterable[object],
    *,
    ulcer_cap: float = 10.0,
) -> Tuple[float, float]:
    """
    Berechnet Ulcer Index (wöchentlich, in Prozent) und mapped Score.
    
    @ffi_boundary: Input/Output
    
    Args:
        equity_curve: Sequenz von Equity-Werten oder (timestamp, equity) Tupeln
        ulcer_cap: Cap für lineare Score-Mapping (in Prozent-Einheiten)
        
    Returns:
        (ulcer_index, ulcer_score)
        - ulcer_index: NaN wenn keine Daten, sonst sqrt(mean(dd_pct^2))
        - ulcer_score: [0.0, 1.0], höher = besser
        
    Algorithm:
        1. Resample zu wöchentlichen Closes (W-SUN)
        2. Berechne Drawdown in Prozent: (weekly - cummax) / cummax * 100
        3. Ulcer Index = sqrt(mean(dd_pct^2))
        4. Score = 1.0 - ulcer_index / ulcer_cap (clamped)
    """
```

### Arrow Schema für EquityCurveInput

```
EquityCurveInput {
  timestamps: list<timestamp[us, tz=UTC]> (nullable)
  values: list<float64>
}

UlcerResult {
  ulcer_index: float64
  ulcer_score: float64
}
```

---

## 10. p_values.py

### compute_p_values()

```python
def compute_p_values(
    trades_df: Optional[pd.DataFrame],
    *,
    r_col: str = "r_multiple",
    pnl_col: str = "result",
    fees_col: str = "total_fee",
    net_of_fees_pnl: bool = True,
    n_boot: int = 2000,
    seed_r: int = 123,
    seed_pnl: int = 456,
) -> Mapping[str, float]:
    """
    Berechnet Bootstrap p-Werte für Strategie-Signifikanz.
    
    @ffi_boundary: Input/Output
    
    Args:
        trades_df: Trade-Historie
        r_col: Spaltenname für R-Multiple
        pnl_col: Spaltenname für P&L
        fees_col: Spaltenname für Gebühren
        net_of_fees_pnl: PnL netto nach Gebühren?
        n_boot: Anzahl Bootstrap-Samples
        seed_r: Seed für R-Multiple p-Wert
        seed_pnl: Seed für PnL p-Wert
        
    Returns:
        {
            "p_mean_r_gt_0": float,      # P(mean_boot(r) <= 0)
            "p_net_profit_gt_0": float,  # P(mean_boot(pnl) <= 0)
        }
        
    Algorithm:
        IID Bootstrap: Ziehe n_boot Samples mit Replacement,
        berechne mean für jedes Sample, zähle Anteil <= 0.
        
    Note:
        Nicht für multiple testing korrigiert. Kann optimistisch sein
        nach intensiver Parametersuche.
    """
```

### Helper: bootstrap_p_value_mean_gt_zero()

```python
def bootstrap_p_value_mean_gt_zero(
    x: Any,
    *,
    n_boot: int = 2000,
    seed: int = 123,
) -> float:
    """
    Bootstrap p-value für H0: mean(x) <= 0.
    
    @ffi_boundary: Internal
    
    Returns:
        p = P(mean_boot <= 0)
        1.0 für leere/ungültige Inputs
    """
```

### Arrow Schema für PValuesInput

```
PValuesInput {
  r_multiples: list<float64>
  pnl_values: list<float64>
  n_boot: int32
  seed_r: int32
  seed_pnl: int32
}

PValuesResult {
  p_mean_r_gt_0: float64
  p_net_profit_gt_0: float64
}
```

---

## FFI Migration Strategy

### Rust Implementation

```rust
use std::collections::HashMap;

/// Common metrics input
pub type MetricsDict = HashMap<String, f64>;

/// Helper: finite float
fn to_finite(x: f64, default: f64) -> f64 {
    if x.is_finite() { x } else { default }
}

/// Relative drop calculation
fn pct_drop(base: f64, x: f64, invert: bool) -> f64 {
    let base = base.max(1e-9);
    let x = x.max(0.0);
    if invert {
        ((x - base) / base).max(0.0)
    } else {
        ((base - x) / base).max(0.0)
    }
}

/// Robustness Score 1
#[pyfunction]
pub fn compute_robustness_score_1(
    base_metrics: MetricsDict,
    jitter_metrics: Vec<MetricsDict>,
    penalty_cap: f64,
) -> f64 {
    let cap = penalty_cap.max(0.0);
    if cap == 0.0 { return 1.0; }
    
    let base_profit = to_finite(*base_metrics.get("profit").unwrap_or(&0.0), 0.0);
    let base_avg_r = to_finite(*base_metrics.get("avg_r").unwrap_or(&0.0), 0.0);
    let base_winrate = to_finite(*base_metrics.get("winrate").unwrap_or(&0.0), 0.0);
    let base_drawdown = to_finite(*base_metrics.get("drawdown").unwrap_or(&0.0), 0.0);
    
    if jitter_metrics.is_empty() {
        return (1.0 - cap).max(0.0);
    }
    
    let mut drops: Vec<f64> = Vec::new();
    for m in &jitter_metrics {
        let profit = to_finite(*m.get("profit").unwrap_or(&0.0), 0.0);
        let avg_r = to_finite(*m.get("avg_r").unwrap_or(&0.0), 0.0);
        let winrate = to_finite(*m.get("winrate").unwrap_or(&0.0), 0.0);
        let drawdown = to_finite(*m.get("drawdown").unwrap_or(&0.0), 0.0);
        
        let d = (
            pct_drop(base_profit, profit, false) +
            pct_drop(base_avg_r, avg_r, false) +
            pct_drop(base_winrate, winrate, false) +
            pct_drop(base_drawdown, drawdown, true)
        ) / 4.0;
        
        if d.is_finite() {
            drops.push(d);
        }
    }
    
    let penalty = if drops.is_empty() {
        cap
    } else {
        let sum: f64 = drops.iter().sum();
        (sum / drops.len() as f64).clamp(0.0, cap)
    };
    
    (1.0 - penalty).max(0.0)
}
```

### Benchmark Targets

| Function | Python Baseline | Rust Target |
|----------|-----------------|-------------|
| robustness_score_1 (100 jitter runs) | ~2ms | <100µs |
| stability_score (5 years) | ~50µs | <5µs |
| cost_shock_score (3 factors) | ~100µs | <10µs |
| trade_dropout_metrics (1000 trades) | ~5ms | <500µs |
| data_jitter_score (10 jitter runs) | ~1.5ms | <150µs |
| timing_jitter_score (3 shifts) | ~0.8ms | <80µs |
| tp_sl_stress_score (1000 trades) | ~50ms | <5ms |
| ulcer_index_score (365 points) | ~20ms | <2ms |
| p_values (2000 bootstrap) | ~3ms | <300µs |

---

## Critical Invariants

1. **Score Range:** Alle Scores müssen in [0.0, 1.0] liegen
2. **Penalty Capping:** Penalties werden auf [0, penalty_cap] geclippt
3. **NaN Safety:** Alle Inputs werden auf Finite geprüft
4. **Determinism:** Bei gleichem Seed produziert simulate_trade_dropout identische Ergebnisse
5. **Empty Input Handling:** Leere Inputs → definierte Default-Werte

---

## Test Coverage Requirements

```python
# Für jeden Score-Typ:

def test_score_range():
    """Score muss in [0, 1] liegen."""
    
def test_empty_input():
    """Leerer Input → definierter Default."""
    
def test_nan_handling():
    """NaN/Inf in Input → graceful fallback."""
    
def test_determinism():
    """Gleicher Input → gleicher Output."""
    
def test_penalty_cap():
    """Penalty wird auf cap geclippt."""
    
def test_edge_cases():
    """Base metric = 0, negative values, etc."""
```

---

## Related Modules

- `src/backtest_engine/optimizer/final_param_selector.py` - Verwendet Rating-Scores
- `src/backtest_engine/optimizer/walkforward.py` - Generiert Metriken für Rating
- `src/backtest_engine/analysis/walkforward_analyzer.py` - Analysiert Rating-Ergebnisse
- `src/backtest_engine/core/portfolio.py` - Liefert Basis-Metriken
