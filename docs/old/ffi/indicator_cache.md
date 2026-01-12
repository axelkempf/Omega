# FFI Interface Specification: IndicatorCache

**Modul:** `src/backtest_engine/core/indicator_cache.py`  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Phase 2 Task:** P2-01  
**Status:** ✅ Spezifiziert (2026-01-05)

---

## Executive Summary

`IndicatorCache` ist ein **High-Performance Indikator-Cache** für aligned Multi-Timeframe OHLCV-Daten.
Das Modul ist ein Haupt-Kandidat für Rust-Migration aufgrund:
- Intensiver numerischer Berechnungen (EMA, RSI, MACD, Bollinger, ATR, etc.)
- Hoher Aufruffrequenz in der Backtest-Loop
- Klarer Input/Output-Grenzen (OHLCV → Indicator-Serien)

---

## Data Structures

### Input: MultiCandleData (Aligned)

```python
# @ffi_boundary: Input
# Typ-Definition aus src/backtest_engine/core/types.py

AlignedMultiCandleData: TypeAlias = Mapping[
    Timeframe,                           # str, z.B. "M1", "H1", "D1"
    Mapping[
        PriceType,                       # Literal["bid", "ask"]
        Sequence[AlignedCandle]          # Candle | CandleDict | None
    ]
]

# Länge ALLER Sequences ist identisch (= Anzahl Primary Bars)
# None-Werte markieren fehlende Bars (carry_forward stale oder strict)
```

**Arrow Schema (für FFI):**

```
AlignedMultiCandleData {
  timeframe: utf8 (dict-encoded)
  price_type: utf8 (enum: "bid" | "ask")
  n_bars: uint64
  
  # Columnar OHLCV arrays (length = n_bars)
  timestamp: timestamp[us, tz=UTC]  # oder int64 (epoch micros)
  open: float64
  high: float64
  low: float64
  close: float64
  volume: float64
  
  # Validity mask für None-Candles
  valid: bool  # true wenn Candle vorhanden, false wenn None
}
```

### Candle Representation

```python
# @ffi_boundary: Input
class CandleDict(TypedDict, total=False):
    """Dict-basierte Candle-Repräsentation (JSON-serialisierbar)."""
    timestamp: datetime          # UTC timezone-aware
    open: float                  # np.float64 kompatibel
    high: float
    low: float
    close: float
    volume: float
    candle_type: PriceType       # Optional: "bid" | "ask"

# Alternativ: Candle-Objekt mit identischen Attributen
CandleLike: TypeAlias = Candle | CandleDict
AlignedCandle: TypeAlias = CandleLike | None
```

### Internal: DataFrame Cache

```python
# @ffi_boundary: Internal (nicht über FFI exponiert)
# Pandas DataFrame mit OHLCV-Spalten

_df_cache: Dict[Tuple[str, str], pd.DataFrame]
# Key: (timeframe, price_type)
# Value: DataFrame mit columns ["open", "high", "low", "close", "volume"]
#        dtype: float64
#        index: RangeIndex (0..n_bars-1)
#        NaN für fehlende Bars
```

---

## Public API Signatures

### Constructor

```python
def __init__(
    self,
    multi_candle_data: AlignedMultiCandleData,
) -> None:
    """
    Initialisiert den IndicatorCache.
    
    @ffi_boundary: Input
    
    Args:
        multi_candle_data: Aligned Multi-TF Candle-Daten.
            Schlüssel: Timeframe-String (z.B. "M1", "H1", "D1")
            Werte: Dict mit "bid"/"ask" → Liste von Candles (oder None)
            
    Constraints:
        - Alle Candle-Listen MÜSSEN gleiche Länge haben (aligned)
        - None-Werte sind erlaubt (fehlende Bars)
        
    Performance:
        - DataFrames werden im Constructor erstellt (O(n) pro TF/Side)
        - Nachfolgende Zugriffe sind O(1) cached
    """
```

### Low-Level: OHLCV Access

```python
def get_df(
    self,
    tf: str,                    # Timeframe, z.B. "M1", "H1"
    price_type: str = "bid",    # "bid" | "ask"
) -> pd.DataFrame:
    """
    Gibt OHLCV-DataFrame zurück (kopierfrei).
    
    @ffi_boundary: Output
    
    Returns:
        pd.DataFrame mit columns:
            - open:   float64, shape (n,)
            - high:   float64, shape (n,)
            - low:    float64, shape (n,)
            - close:  float64, shape (n,)
            - volume: float64, shape (n,)
        NaN für fehlende Bars.
        
    Arrow Output Schema:
        struct {
            open: float64[n]
            high: float64[n]
            low: float64[n]
            close: float64[n]
            volume: float64[n]
        }
    """

def get_closes(
    self,
    tf: str,
    price_type: str = "bid",
) -> pd.Series:
    """
    Schnellzugriff auf Close-Serie.
    
    @ffi_boundary: Output
    
    Returns:
        pd.Series[float64], shape (n,)
        NaN für fehlende Bars.
        
    Arrow Output: float64[n]
    """
```

### Indicator APIs (Vektorisiert + Gecached)

#### EMA (Exponential Moving Average)

```python
def ema(
    self,
    tf: str,                    # Timeframe
    price_type: str,            # "bid" | "ask"
    period: int,                # EMA-Periode, > 0
) -> pd.Series:
    """
    Berechnet EMA über Close-Serie.
    
    @ffi_boundary: Output
    
    Formel: ewm(span=period, adjust=False).mean()
    
    Args:
        tf: Timeframe-String
        price_type: "bid" oder "ask"
        period: Glättungsperiode (muss > 0 sein)
        
    Returns:
        pd.Series[float64], shape (n,)
        NaN für initiale Warmup-Bars und fehlende Daten.
        
    Raises:
        ValueError: wenn period <= 0
        
    Arrow Output: float64[n]
    
    Cache-Key: ("ema", tf, price_type, period)
    """

def ema_stepwise(
    self,
    tf: str,
    price_type: str,
    period: int,
) -> pd.Series:
    """
    EMA mit HTF-Bar-Update-Semantik (verhindert carry_forward Drift).
    
    @ffi_boundary: Output
    
    Berechnung:
        1. Identifiziere Indizes mit *neuer* HTF-Bar
        2. Berechne EMA nur auf reduzierten Closes
        3. Forward-fill auf Primary-Raster
        
    Returns:
        pd.Series[float64], shape (n,)
        
    Arrow Output: float64[n]
    """
```

#### SMA (Simple Moving Average)

```python
def sma(
    self,
    tf: str,
    price_type: str,
    period: int,
) -> pd.Series:
    """
    @ffi_boundary: Output
    
    Formel: rolling(window=period, min_periods=period).mean()
    
    Returns:
        pd.Series[float64], shape (n,)
        NaN für erste (period-1) Bars.
        
    Arrow Output: float64[n]
    """
```

#### RSI (Relative Strength Index)

```python
def rsi(
    self,
    tf: str,
    price_type: str,
    period: int = 14,
) -> pd.Series:
    """
    Wilder RSI Berechnung.
    
    @ffi_boundary: Output
    
    Formel:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = ewm(alpha=1/period, adjust=False).mean()
        avg_loss = ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    Returns:
        pd.Series[float64], shape (n,)
        Range: [0, 100], NaN für Warmup
        
    Arrow Output: float64[n]
    """
```

#### MACD (Moving Average Convergence/Divergence)

```python
def macd(
    self,
    tf: str,
    price_type: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    @ffi_boundary: Output
    
    Returns:
        Tuple[macd_line, signal_line]
        - macd_line: pd.Series[float64], shape (n,)
        - signal_line: pd.Series[float64], shape (n,)
        
    Arrow Output:
        struct {
            macd_line: float64[n]
            signal_line: float64[n]
        }
    """
```

#### Bollinger Bands

```python
def bollinger(
    self,
    tf: str,
    price_type: str,
    period: int = 20,
    std_factor: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    @ffi_boundary: Output
    
    Returns:
        Tuple[upper, mid, lower]
        - upper: pd.Series[float64], shape (n,) - Oberes Band
        - mid:   pd.Series[float64], shape (n,) - Mittlere Linie (SMA)
        - lower: pd.Series[float64], shape (n,) - Unteres Band
        
    Arrow Output:
        struct {
            upper: float64[n]
            mid: float64[n]
            lower: float64[n]
        }
    """

def bollinger_stepwise(
    self,
    tf: str,
    price_type: str,
    period: int = 20,
    std_factor: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger mit HTF-Bar-Update-Semantik.
    
    @ffi_boundary: Output
    
    Returns: Identisch zu bollinger()
    """
```

#### ATR (Average True Range)

```python
def atr(
    self,
    tf: str,
    price_type: str,
    period: int = 14,
) -> pd.Series:
    """
    Wilder-ATR (Bloomberg/TradingView-kompatibel).
    
    @ffi_boundary: Output
    
    Formel:
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR_0 = SMA(TR[0:period])
        ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
        
    Returns:
        pd.Series[float64], shape (n,)
        NaN für Warmup-Periode
        
    Arrow Output: float64[n]
    """
```

#### DMI (Directional Movement Index)

```python
def dmi(
    self,
    tf: str,
    price_type: str,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    @ffi_boundary: Output
    
    Returns:
        Tuple[plus_di, minus_di, adx]
        - plus_di:  pd.Series[float64], shape (n,) - +DI
        - minus_di: pd.Series[float64], shape (n,) - -DI  
        - adx:      pd.Series[float64], shape (n,) - ADX
        
    Arrow Output:
        struct {
            plus_di: float64[n]
            minus_di: float64[n]
            adx: float64[n]
        }
    """
```

#### Z-Score Varianten

```python
def zscore(
    self,
    tf: str,
    price_type: str,
    window: int = 100,
    mean_source: str = "rolling",    # "rolling" | "ema"
    ema_period: Optional[int] = None,
) -> pd.Series:
    """
    Flexibler Z-Score.
    
    @ffi_boundary: Output
    
    Formel:
        mean_source="rolling": z = (x - SMA(window)) / STD(window)
        mean_source="ema":     z = (x - EMA(ema_period)) / STD(window)
        
    Returns:
        pd.Series[float64], shape (n,)
        
    Raises:
        ValueError: wenn mean_source="ema" und ema_period nicht gesetzt
        
    Arrow Output: float64[n]
    """

def kalman_zscore(
    self,
    tf: str,
    price_type: str,
    window: int = 100,
    R: float = 0.01,    # Measurement noise
    Q: float = 1.0,     # Process noise
) -> pd.Series:
    """
    Z-Score basierend auf Kalman-Filter Mean.
    
    @ffi_boundary: Output
    
    Returns:
        pd.Series[float64], shape (n,)
        
    Arrow Output: float64[n]
    """

def kalman_garch_zscore(
    self,
    tf: str,
    price_type: str,
    R: float = 0.01,
    Q: float = 1.0,
    alpha: float = 0.05,    # GARCH alpha
    beta: float = 0.90,     # GARCH beta
    omega: Optional[float] = None,
    use_log_returns: bool = True,
    scale: float = 100.0,
    min_periods: int = 50,
    sigma_floor: float = 1e-6,
) -> pd.Series:
    """
    Kalman-GARCH Z-Score für volatilitäts-adjustierte Mean-Reversion.
    
    @ffi_boundary: Output
    
    Returns:
        pd.Series[float64], shape (n,)
        
    Arrow Output: float64[n]
    """
```

#### Choppiness Index

```python
def choppiness(
    self,
    tf: str,
    price_type: str,
    period: int = 14,
) -> pd.Series:
    """
    Choppiness Index (Markt-Regime-Indikator).
    
    @ffi_boundary: Output
    
    Formel:
        chop = 100 * log10(ATR_sum / (high_max - low_min)) / log10(period)
        
    Returns:
        pd.Series[float64], shape (n,)
        Range: typisch [0, 100], höhere Werte = mehr Seitwärtsbewegung
        
    Arrow Output: float64[n]
    """
```

---

## Internal Helper Methods

```python
# @ffi_boundary: Internal

def _ensure_df(self, tf: str, side: str) -> pd.DataFrame:
    """Lazy DataFrame construction. Nicht über FFI exponiert."""

def _get_np_closes(self, tf: str, price_type: str) -> np.ndarray:
    """Cached NumPy Close-Array. dtype: float64, shape: (n,)"""

def _stepwise_indices(self, tf: str, price_type: str) -> List[int]:
    """Indizes für neue HTF-Bars (für stepwise Indikatoren)."""

@staticmethod
def _ema(series: pd.Series, period: int) -> pd.Series:
    """Statischer EMA-Helper."""

@staticmethod
def _kalman_mean_from_series(series: pd.Series, R: float, Q: float) -> pd.Series:
    """Kalman-Filter Mean Estimation."""
```

---

## Cache Structure

```python
# Indicator Cache Keys (Tuple-basiert für Hashability)

_ind_cache: Dict[Tuple[Any, ...], Any]

# Key-Formate:
# ("ema", tf, price_type, period) -> pd.Series
# ("sma", tf, price_type, period) -> pd.Series
# ("rsi", tf, price_type, period) -> pd.Series
# ("macd", tf, price_type, fast, slow, signal) -> Tuple[pd.Series, pd.Series]
# ("bb", tf, price_type, period, std_factor) -> Tuple[pd.Series, pd.Series, pd.Series]
# ("atr", tf, price_type, period) -> pd.Series
# ("dmi", tf, price_type, period) -> Tuple[pd.Series, pd.Series, pd.Series]
# ("zscore", tf, price_type, window, mean_source, ema_period) -> pd.Series
# ("kalman_z", tf, price_type, window, R, Q) -> pd.Series
# ("chop", tf, price_type, period) -> pd.Series
# ("_np_closes", tf, price_type) -> np.ndarray
```

---

## Performance Characteristics

| Operation | Complexity | Cached |
|-----------|-----------|--------|
| Constructor | O(n × m) | N/A |
| get_df() | O(1) | ✅ |
| get_closes() | O(1) | ✅ |
| ema() | O(n) first, O(1) cached | ✅ |
| rsi() | O(n) first, O(1) cached | ✅ |
| bollinger() | O(n) first, O(1) cached | ✅ |
| atr() | O(n) first, O(1) cached | ✅ |

n = Anzahl Bars, m = Anzahl Timeframes × 2 (bid/ask)

---

## FFI Migration Notes

### Rust Implementation Strategy

1. **Data Input**: Arrow IPC für Multi-Candle-Data Transfer
2. **Internal Storage**: `ndarray::Array1<f64>` für OHLCV-Arrays
3. **Indicator Functions**: Pure Rust mit `#[no_mangle]` Export
4. **Python Bindings**: PyO3 mit numpy/Arrow Interop

### Critical Invariants

- **NaN Propagation**: Rust-Implementierung MUSS identisches NaN-Handling haben
- **Floating Point**: IEEE 754 float64 für Determinismus
- **Cache Keys**: Identisches Hashing für Cross-Language Kompatibilität

### Benchmark Targets

| Indicator | Python Baseline | Rust Target |
|-----------|-----------------|-------------|
| EMA(14) | ~5ms/100k bars | <0.5ms |
| RSI(14) | ~10ms/100k bars | <1ms |
| Bollinger(20) | ~15ms/100k bars | <1.5ms |
| ATR(14) | ~20ms/100k bars | <2ms |

---

## Related Types

- `src/backtest_engine/core/types.py`: TypedDict/TypeAlias Definitionen
- `src/shared/protocols.py`: `IndicatorCacheProtocol`
- `src/backtest_engine/core/symbol_data_slicer.py`: Consumer des IndicatorCache
