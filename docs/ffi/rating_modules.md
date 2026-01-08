# FFI Interface Specification: Rating Modules

**Modulpfad:** `src/backtest_engine/rating/`  
**Migrations-Ziel:** Rust (via PyO3/maturin) oder Julia (via PythonCall.jl)  
**Phase 2 Task:** P2-04  
**Status:** ✅ Spezifiziert (2026-01-05)

---

## Executive Summary

Das Rating-Modul berechnet **Qualitäts-Scores** für Trading-Strategien basierend auf
Performance-Metriken. Diese Scores werden verwendet für:
- Deployment-Entscheidungen (ja/nein)
- Strategie-Ranking (Final Selection)
- Robustheit gegen Parameter-Jitter
- Stabilität über Zeit (Jahresvergleich)
- Stress-Resistenz (Kosten-Schocks, Trade-Dropout)

Migrations-Kandidat aufgrund:
- Numerisch intensive Berechnungen (Mittelwerte, Standardabweichungen)
- Monte-Carlo-artige Berechnungen (Dropout-Simulation)
- Pure Functions ohne Side Effects
- Klare Input/Output-Schemas

---

## Module Overview

| Modul | Funktion | Input | Output |
|-------|----------|-------|--------|
| `strategy_rating.py` | Deployment-Entscheidung | summary Dict | Score + Deployment Bool |
| `robustness_score_1.py` | Parameter-Jitter-Robustheit | base + jitter Metrics | Score [0,1] |
| `stability_score.py` | Jahres-Stabilität | profits_by_year | Score [0,1] |
| `cost_shock_score.py` | Kosten-Schock-Resistenz | base + shocked Metrics | Score [0,1] |
| `trade_dropout_score.py` | Trade-Dropout-Simulation | trades_df + dropout_frac | Metrics Dict |
| `stress_penalty.py` | Gemeinsame Penalty-Logik | base + stress Metrics | Penalty [0, cap] |

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

# Für strategy_rating:
{
    "Winrate (%)": float,     # z.B. 55.0
    "Avg R-Multiple": float,  # z.B. 0.8
    "Net Profit": float,      # z.B. 1500.0
    "profit_factor": float,   # z.B. 1.5
    "drawdown_eur": float,    # z.B. 500.0
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

## 1. strategy_rating.py

### rate_strategy_performance()

```python
def rate_strategy_performance(
    summary: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Bewertet Strategie gegen fixe Schwellenwerte.
    
    @ffi_boundary: Input/Output
    
    Args:
        summary: Performance-Metriken aus Backtest
        thresholds: Optional Override für Schwellen
        
    Default Thresholds:
        min_winrate: 45        # Prozent
        min_avg_r: 0.6         # R-Multiple
        min_profit: 500        # EUR
        min_profit_factor: 1.2
        max_drawdown: 1000     # EUR
        
    Returns:
        {
            "Score": float,           # [0.0, 1.0] - Anteil bestandener Checks
            "Deployment": bool,       # True wenn alle Checks bestanden
            "Deployment_Fails": str,  # Pipe-separierte Liste: "Winrate|Drawdown"
        }
        
    Scoring Logic:
        - 5 unabhängige Checks
        - Score = (5 - Anzahl Fails) / 5
        - Deployment = True nur wenn alle Checks bestanden
    """
```

### Arrow Schema für DeploymentResult

```
DeploymentResult {
  score: float64
  deployment: bool
  deployment_fails: utf8  # Pipe-separated
}
```

---

## 2. robustness_score_1.py

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

## 3. stability_score.py

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

## 4. stress_penalty.py (Shared Foundation)

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

## 5. cost_shock_score.py

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

## 6. trade_dropout_score.py

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
