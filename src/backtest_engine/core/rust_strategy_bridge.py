"""Pure Rust Strategy Bridge - Wave 4 Integration.

This module provides a Python interface for running backtests entirely in Rust,
eliminating the ~150,000 FFI calls per backtest that occur with the Python
callback approach (Wave 3).

## Architecture

Wave 3 (Event Engine):
    Python → Rust → Python (strategy.evaluate()) → Rust → ...
    FFI Calls: ~7 per bar × ~20,000 bars = ~140,000+ calls

Wave 4 (Pure Rust Strategy):
    Python → Rust (entire backtest) → Python
    FFI Calls: 2 (init + result)

## Feature Flag

Enable Pure Rust Strategy via environment variable:

    export OMEGA_USE_RUST_STRATEGY=1

Values:
- "auto": Use Rust if available (default)
- "true" / "1": Force Rust (error if unavailable)
- "false" / "0": Force Python (Wave 3 fallback)

## Performance Targets

| Metric              | Wave 3 (Python) | Wave 4 (Rust) | Improvement |
|---------------------|-----------------|---------------|-------------|
| FFI calls/backtest  | ~150,000        | 2             | 75,000x     |
| 20K bars time       | ~36s            | ~3s           | 12x         |
| Strategy eval       | ~70% of time    | ~10% of time  | 7x          |

## Usage

```python
from src.backtest_engine.core.rust_strategy_bridge import (
    is_rust_strategy_available,
    should_use_rust_strategy,
    run_rust_backtest,
)

# Check availability
if should_use_rust_strategy():
    result = run_rust_backtest(
        strategy_name="mean_reversion_z_score",
        config=backtest_config,
        bid_candles=bid_data,
        ask_candles=ask_data,
    )
else:
    # Fall back to Wave 3 (Python callbacks)
    result = run_python_backtest(...)
```

## Supported Strategies

- mean_reversion_z_score: Mean Reversion with Kalman-filtered Z-Score

## Implementation Notes

This bridge converts Python data structures to Rust-compatible formats:
1. Config: Converted to Rust StrategyConfig struct
2. Candles: Converted to Vec<CandleData> per timeframe
3. Indicators: Uses IndicatorCacheRust for pre-computed values
4. Result: BacktestResult contains all trades and metrics
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.backtest_engine.core.indicator_cache import IndicatorCache
    from src.backtest_engine.data.candle import Candle

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flag Configuration
# =============================================================================

FEATURE_FLAG = "OMEGA_USE_RUST_STRATEGY"
_RUST_AVAILABLE: Optional[bool] = None
_RUST_MODULE: Any = None


def _check_rust_available() -> bool:
    """Check if Rust Pure Strategy module is available.
    
    Returns:
        True if omega_rust.BacktestResult and omega_rust.run_backtest_rust
        are importable.
    """
    global _RUST_AVAILABLE, _RUST_MODULE
    
    if _RUST_AVAILABLE is not None:
        return _RUST_AVAILABLE
    
    try:
        import omega_rust
        
        # Check for Wave 4 exports
        if not hasattr(omega_rust, "BacktestResult"):
            logger.debug("omega_rust.BacktestResult not found")
            _RUST_AVAILABLE = False
            return False
            
        if not hasattr(omega_rust, "run_backtest_rust"):
            logger.debug("omega_rust.run_backtest_rust not found")
            _RUST_AVAILABLE = False
            return False
        
        _RUST_MODULE = omega_rust
        _RUST_AVAILABLE = True
        logger.debug("Rust Pure Strategy module available")
        
    except ImportError as e:
        logger.debug(f"Rust Pure Strategy not available: {e}")
        _RUST_AVAILABLE = False
    
    return _RUST_AVAILABLE


def is_rust_strategy_available() -> bool:
    """Check if Rust Pure Strategy is available.
    
    Returns:
        True if the Rust module is properly installed and accessible.
    """
    return _check_rust_available()


def is_rust_enabled() -> bool:
    """Check if Rust Pure Strategy is enabled via feature flag.
    
    Environment variable OMEGA_USE_RUST_STRATEGY:
    - "auto": Use Rust if available (default)
    - "true" / "1" / "yes" / "on": Force Rust
    - "false" / "0" / "no" / "off": Force Python
    
    Returns:
        True if Rust should be used based on the feature flag.
    """
    flag = os.environ.get(FEATURE_FLAG, "auto").lower()
    
    if flag in ("false", "0", "no", "off"):
        return False
    
    if flag in ("true", "1", "yes", "on"):
        return True
    
    # "auto" - use Rust if available
    return True


def should_use_rust_strategy() -> bool:
    """Determine if Rust Pure Strategy should be used.
    
    Combines feature flag check with availability check.
    
    Returns:
        True if Rust should be used (enabled AND available).
    """
    enabled = is_rust_enabled()
    available = is_rust_strategy_available()
    
    if enabled and not available:
        flag = os.environ.get(FEATURE_FLAG, "auto").lower()
        if flag in ("true", "1", "yes", "on"):
            raise RuntimeError(
                f"Rust Pure Strategy forced via {FEATURE_FLAG}={flag} "
                "but omega_rust module is not available. "
                "Please build the Rust module or set OMEGA_USE_RUST_STRATEGY=auto"
            )
    
    return enabled and available


def get_active_backend() -> str:
    """Get the currently active strategy backend.
    
    Returns:
        "rust" if using Pure Rust Strategy, "python" otherwise.
    """
    return "rust" if should_use_rust_strategy() else "python"


# =============================================================================
# Data Conversion
# =============================================================================

@dataclass
class RustCandle:
    """Rust-compatible candle representation."""
    timestamp_us: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def convert_candle_to_rust(candle: "Candle") -> Dict[str, Any]:
    """Convert Python Candle to Rust-compatible dict.
    
    Args:
        candle: Python Candle object with timestamp, OHLCV fields.
        
    Returns:
        Dictionary matching Rust CandleData struct.
    """
    # Handle timestamp conversion
    if isinstance(candle.timestamp, datetime):
        timestamp_us = int(candle.timestamp.timestamp() * 1_000_000)
    elif isinstance(candle.timestamp, (int, float)):
        # Assume already in some timestamp format
        timestamp_us = int(candle.timestamp * 1_000_000)
    else:
        # Try to convert
        timestamp_us = int(candle.timestamp.timestamp() * 1_000_000)
    
    return {
        "timestamp_us": timestamp_us,
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": float(getattr(candle, "volume", 0.0)),
    }


def convert_candles_to_rust(
    candles: List["Candle"],
) -> List[Dict[str, Any]]:
    """Convert list of Python Candles to Rust-compatible format.
    
    Args:
        candles: List of Python Candle objects.
        
    Returns:
        List of dictionaries matching Rust CandleData struct.
    """
    return [convert_candle_to_rust(c) for c in candles]


def convert_multi_candles_to_rust(
    candle_data: Dict[str, List["Candle"]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert multi-timeframe candle data to Rust format.
    
    Args:
        candle_data: Dict mapping timeframe -> list of candles.
        
    Returns:
        Dict mapping timeframe string -> list of Rust CandleData dicts.
    """
    result = {}
    for timeframe, candles in candle_data.items():
        result[timeframe] = convert_candles_to_rust(candles)
    return result


def convert_config_to_rust(
    strategy_name: str,
    config: Dict[str, Any],
) -> Any:
    """Convert Python backtest config to Rust StrategyConfig.
    
    Args:
        strategy_name: Name of the strategy to use.
        config: Python backtest configuration dictionary.
        
    Returns:
        Native Rust StrategyConfig object.
    """
    if not _RUST_MODULE:
        raise RuntimeError("Rust module not available")
    
    # Extract strategy params
    strategy_params = config.get("strategy_params", {})
    # Ensure all params are floats for Rust HashMap<String, f64>
    params_float = {str(k): float(v) for k, v in strategy_params.items() if isinstance(v, (int, float))}
    
    # Create native Rust StrategyConfig
    rust_config = _RUST_MODULE.StrategyConfig(
        config.get("symbol", "EURUSD"),
        config.get("primary_timeframe", "H1"),
        float(config.get("initial_capital", 100000.0)),
        float(config.get("risk_per_trade", 0.01)),
        params_float,
    )
    
    return rust_config


# =============================================================================
# Result Conversion
# =============================================================================

@dataclass
class TradeResultPython:
    """Python representation of a trade result."""
    id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    size: float
    pnl: float
    pnl_pips: float
    exit_reason: str
    scenario: int


@dataclass
class BacktestResultPython:
    """Python representation of backtest results."""
    strategy_name: str
    symbol: str
    trades: List[TradeResultPython]
    initial_capital: float
    final_capital: float
    bars_processed: int
    execution_time_ms: float
    strategy_time_ms: float
    open_positions: int
    
    # Computed metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    roi: float


def convert_result_from_rust(rust_result: Any) -> BacktestResultPython:
    """Convert Rust BacktestResult to Python format.
    
    Args:
        rust_result: omega_rust.BacktestResult object.
        
    Returns:
        BacktestResultPython with all metrics.
    """
    # Convert trades
    trades = []
    for rust_trade in rust_result.trades:
        trade = TradeResultPython(
            id=rust_trade.id,
            symbol=rust_trade.symbol,
            direction=rust_trade.direction,
            entry_price=rust_trade.entry_price,
            exit_price=rust_trade.exit_price,
            entry_timestamp=datetime.fromtimestamp(
                rust_trade.entry_timestamp_us / 1_000_000
            ),
            exit_timestamp=datetime.fromtimestamp(
                rust_trade.exit_timestamp_us / 1_000_000
            ),
            size=rust_trade.size,
            pnl=rust_trade.pnl,
            pnl_pips=rust_trade.pnl_pips,
            exit_reason=rust_trade.exit_reason,
            scenario=rust_trade.scenario,
        )
        trades.append(trade)
    
    return BacktestResultPython(
        strategy_name=rust_result.strategy_name,
        symbol=rust_result.symbol,
        trades=trades,
        initial_capital=rust_result.initial_capital,
        final_capital=rust_result.final_capital,
        bars_processed=rust_result.bars_processed,
        execution_time_ms=rust_result.execution_time_ms,
        strategy_time_ms=rust_result.strategy_time_ms,
        open_positions=rust_result.open_positions,
        # Computed from Rust getters
        total_trades=rust_result.total_trades,
        winning_trades=rust_result.winning_trades,
        losing_trades=rust_result.losing_trades,
        win_rate=rust_result.win_rate,
        total_pnl=rust_result.total_pnl,
        profit_factor=rust_result.profit_factor,
        roi=rust_result.roi,
    )


# =============================================================================
# Main API
# =============================================================================

def run_rust_backtest(
    strategy_name: str,
    config: Dict[str, Any],
    bid_candles: Dict[str, List["Candle"]],
    ask_candles: Dict[str, List["Candle"]],
    indicator_cache: Optional["IndicatorCache"] = None,
) -> BacktestResultPython:
    """Run a complete backtest using Pure Rust Strategy.
    
    This function executes the entire backtest loop in Rust, eliminating
    the ~150,000 FFI calls that occur with the Python callback approach.
    
    Args:
        strategy_name: Name of the registered Rust strategy 
            (e.g., "mean_reversion_z_score").
        config: Backtest configuration dictionary containing:
            - symbol: Trading symbol (e.g., "EURUSD")
            - primary_timeframe: Main timeframe (e.g., "H1")
            - initial_capital: Starting capital
            - risk_per_trade: Risk per trade as fraction
            - strategy_params: Strategy-specific parameters
        bid_candles: Dict mapping timeframe -> list of bid candles.
        ask_candles: Dict mapping timeframe -> list of ask candles.
        indicator_cache: Optional pre-computed indicator cache.
            If not provided, indicators will be computed in Rust.
    
    Returns:
        BacktestResultPython containing all trades and performance metrics.
        
    Raises:
        RuntimeError: If Rust strategy is not available.
        ValueError: If strategy_name is not registered.
    
    Example:
        >>> result = run_rust_backtest(
        ...     strategy_name="mean_reversion_z_score",
        ...     config={
        ...         "symbol": "EURUSD",
        ...         "primary_timeframe": "H1",
        ...         "initial_capital": 100000,
        ...         "risk_per_trade": 0.01,
        ...         "strategy_params": {
        ...             "z_score_entry_threshold": 2.0,
        ...             "z_score_lookback": 100,
        ...         },
        ...     },
        ...     bid_candles={"H1": h1_bid_candles},
        ...     ask_candles={"H1": h1_ask_candles},
        ... )
        >>> print(f"Win rate: {result.win_rate:.1%}")
    """
    if not should_use_rust_strategy():
        raise RuntimeError(
            "Rust Pure Strategy not available. "
            "Either build omega_rust or use Python fallback."
        )
    
    assert _RUST_MODULE is not None
    
    # Convert config to native Rust StrategyConfig
    strategy_params = config.get("strategy_params", {})
    # Ensure all params are floats for Rust HashMap<String, f64>
    params_float = {str(k): float(v) for k, v in strategy_params.items() if isinstance(v, (int, float))}
    
    rust_config = _RUST_MODULE.StrategyConfig(
        config.get("symbol", "EURUSD"),
        config.get("primary_timeframe", "H1"),
        float(config.get("initial_capital", 100000.0)),
        float(config.get("risk_per_trade", 0.01)),
        params_float,
    )
    
    # Convert candles to native Rust CandleData objects
    # Filter out None values that may appear from alignment gaps
    rust_bid: Dict[str, List[Any]] = {}
    rust_ask: Dict[str, List[Any]] = {}
    
    def _candle_to_rust(c: Any) -> Any:
        """Convert a single candle to Rust CandleData, handling None safely."""
        if c is None:
            return None
        ts = c.timestamp
        if hasattr(ts, 'timestamp'):
            ts_us = int(ts.timestamp() * 1_000_000)
        else:
            ts_us = int(ts * 1_000_000)
        return _RUST_MODULE.CandleData(
            ts_us,
            float(c.open),
            float(c.high),
            float(c.low),
            float(c.close),
            float(getattr(c, "volume", 0.0)),
        )
    
    for timeframe, candles in bid_candles.items():
        # Filter None values and convert to Rust
        rust_bid[timeframe] = [
            rc for rc in (_candle_to_rust(c) for c in candles) if rc is not None
        ]
    
    for timeframe, candles in ask_candles.items():
        rust_ask[timeframe] = [
            rc for rc in (_candle_to_rust(c) for c in candles) if rc is not None
        ]
    
    # Get or create Rust indicator cache
    rust_indicator_cache = None
    if indicator_cache is not None:
        # Try to get Rust backend
        if hasattr(indicator_cache, "_rust_cache"):
            rust_indicator_cache = indicator_cache._rust_cache
        elif hasattr(indicator_cache, "rust_cache"):
            rust_indicator_cache = indicator_cache.rust_cache
    
    if rust_indicator_cache is None:
        # Create new Rust cache
        rust_indicator_cache = _RUST_MODULE.IndicatorCacheRust()
        
        # Register OHLCV data (filter None values from alignment)
        for timeframe, candles in bid_candles.items():
            valid_candles = [c for c in candles if c is not None]
            if valid_candles:
                opens = np.array([c.open for c in valid_candles], dtype=np.float64)
                highs = np.array([c.high for c in valid_candles], dtype=np.float64)
                lows = np.array([c.low for c in valid_candles], dtype=np.float64)
                closes = np.array([c.close for c in valid_candles], dtype=np.float64)
                volumes = np.array(
                    [getattr(c, "volume", 0.0) for c in valid_candles], 
                    dtype=np.float64
                )
                
                rust_indicator_cache.register_ohlcv(
                    symbol=config.get("symbol", "EURUSD"),
                    timeframe=timeframe,
                    price_type="BID",
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    volume=volumes,
                )
        
        for timeframe, candles in ask_candles.items():
            valid_candles = [c for c in candles if c is not None]
            if valid_candles:
                opens = np.array([c.open for c in valid_candles], dtype=np.float64)
                highs = np.array([c.high for c in valid_candles], dtype=np.float64)
                lows = np.array([c.low for c in valid_candles], dtype=np.float64)
                closes = np.array([c.close for c in valid_candles], dtype=np.float64)
                volumes = np.array(
                    [getattr(c, "volume", 0.0) for c in valid_candles],
                    dtype=np.float64
                )
                
                rust_indicator_cache.register_ohlcv(
                    symbol=config.get("symbol", "EURUSD"),
                    timeframe=timeframe,
                    price_type="ASK",
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    volume=volumes,
                )
    
    logger.debug(
        f"Running Pure Rust backtest: strategy={strategy_name}, "
        f"symbol={config.get('symbol')}, bars={len(next(iter(bid_candles.values()), []))}"
    )
    
    # Execute backtest in Rust (positional args)
    rust_result = _RUST_MODULE.run_backtest_rust(
        strategy_name,
        rust_config,
        rust_bid,
        rust_ask,
        rust_indicator_cache,
    )
    
    # Convert result to Python
    result = convert_result_from_rust(rust_result)
    
    logger.info(
        f"Pure Rust backtest complete: {result.bars_processed} bars in "
        f"{result.execution_time_ms:.1f}ms ({result.bars_processed / (result.execution_time_ms / 1000):.0f} bars/s)"
    )
    
    return result


# =============================================================================
# Utility Functions
# =============================================================================

def list_available_strategies() -> List[str]:
    """List all registered Rust strategies.
    
    Returns:
        List of strategy names that can be used with run_rust_backtest.
    """
    if not is_rust_strategy_available():
        return []
    
    # Currently hardcoded - could expose registry from Rust
    return [
        "mean_reversion_z_score",
    ]


def get_strategy_default_params(strategy_name: str) -> Dict[str, Any]:
    """Get default parameters for a Rust strategy.
    
    Args:
        strategy_name: Name of the registered strategy.
        
    Returns:
        Dictionary of default parameter values.
    """
    defaults = {
        "mean_reversion_z_score": {
            # Z-Score Parameters
            "z_score_lookback": 100,
            "z_score_entry_threshold": 2.0,
            "z_score_exit_threshold": 0.5,
            # Kalman Parameters
            "kalman_q": 0.01,
            "kalman_r": 1.0,
            # ATR Parameters
            "atr_period": 14,
            "atr_sl_multiplier": 2.0,
            "atr_tp_multiplier": 3.0,
            # EMA/ADX Parameters
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "adx_period": 14,
            "adx_threshold": 25.0,
            # Scenario Thresholds
            "volatility_low_threshold": 0.3,
            "volatility_high_threshold": 0.7,
            "trend_strength_threshold": 0.6,
            # Risk Management
            "max_daily_trades": 5,
            "min_rr_ratio": 1.5,
            "max_spread_pips": 3.0,
        },
    }
    return defaults.get(strategy_name, {})


# =============================================================================
# Module Initialization
# =============================================================================

# Eagerly check availability on import (for logging)
_check_rust_available()
