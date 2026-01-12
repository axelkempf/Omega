#!/usr/bin/env python
"""
Debug-Script für Wave 4 Pure Rust Strategy
==========================================

Untersucht warum die Strategie 0 Trades produziert.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

# Enable Rust strategy
os.environ["OMEGA_USE_RUST_STRATEGY"] = "1"


def generate_trending_candles(num_bars: int, seed: int = 42) -> tuple:
    """Generate synthetic data with clear trends and mean reversion patterns."""
    np.random.seed(seed)
    
    # Create price series with distinct phases
    prices = [1.10]  # Start at 1.10
    
    for i in range(1, num_bars):
        phase = (i // 100) % 4  # 4 phases, 100 bars each
        
        if phase == 0:  # Strong uptrend
            drift = 0.0003  # 3 pips positive drift
            volatility = 0.0005
        elif phase == 1:  # Mean reversion (consolidation)
            drift = 0.0
            # Add explicit oscillation around mean
            base_price = prices[-100] if i >= 100 else prices[0]
            volatility = 0.001
            # Force price back toward mean
            deviation = (prices[-1] - base_price) / base_price
            if abs(deviation) > 0.005:  # > 0.5%
                correction = -deviation * 0.3
                prices.append(prices[-1] * (1 + correction + np.random.randn() * 0.0002))
                continue
        elif phase == 2:  # Strong downtrend  
            drift = -0.0003
            volatility = 0.0005
        else:  # High volatility with mean reversion
            drift = 0.0
            volatility = 0.002
        
        returns = drift + np.random.randn() * volatility
        prices.append(prices[-1] * (1 + returns))
    
    # Timestamps (5-minute intervals)
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    timestamps = [base_ts + i * 300_000_000 for i in range(num_bars)]
    
    class MockCandle:
        def __init__(self, ts, o, h, l, c, v):
            self.timestamp = datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc)
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.volume = v
    
    bid_candles = []
    ask_candles = []
    
    for i in range(num_bars):
        p = prices[i]
        spread = 0.0002  # 2 pip spread
        bar_range = np.random.uniform(0.0005, 0.002)
        
        high = p * (1 + bar_range / 2)
        low = p * (1 - bar_range / 2)
        open_price = p * (1 + np.random.uniform(-bar_range/3, bar_range/3))
        
        bid_candles.append(MockCandle(timestamps[i], open_price, high, low, p, 100.0 + np.random.rand() * 50))
        ask_candles.append(MockCandle(timestamps[i], open_price + spread, high + spread, low + spread, p + spread, 100.0 + np.random.rand() * 50))
    
    return {"M5": bid_candles}, {"M5": ask_candles}


def debug_indicator_cache():
    """Debug indicator cache functionality."""
    print("\n" + "=" * 70)
    print("1. Debug Indicator Cache")
    print("=" * 70)
    
    from omega_rust import IndicatorCacheRust
    
    bid_candles, _ = generate_trending_candles(500)
    m5_bid = bid_candles["M5"]
    
    # Create arrays
    closes = np.array([c.close for c in m5_bid], dtype=np.float64)
    opens = np.array([c.open for c in m5_bid], dtype=np.float64)
    highs = np.array([c.high for c in m5_bid], dtype=np.float64)
    lows = np.array([c.low for c in m5_bid], dtype=np.float64)
    volumes = np.array([c.volume for c in m5_bid], dtype=np.float64)
    
    cache = IndicatorCacheRust()
    cache.register_ohlcv("EURUSD", "M5", "BID", opens, highs, lows, closes, volumes)
    
    print(f"  Registered OHLCV data: {len(closes)} bars")
    print(f"  Close price range: {closes.min():.5f} - {closes.max():.5f}")
    print(f"  Price std dev: {closes.std():.6f}")
    
    # Test EMA
    try:
        ema_fast = cache.ema("EURUSD", "M5", "BID", 20, None)
        ema_slow = cache.ema("EURUSD", "M5", "BID", 50, None)
        print(f"  EMA(20) length: {len(ema_fast)}")
        print(f"  EMA(50) length: {len(ema_slow)}")
        if len(ema_fast) > 0:
            print(f"  EMA(20) last 5: {ema_fast[-5:]}")
    except Exception as e:
        print(f"  EMA ERROR: {e}")
    
    # Test ATR
    try:
        atr = cache.atr("EURUSD", "M5", "BID", 14)
        print(f"  ATR(14) length: {len(atr)}")
        if len(atr) > 0:
            print(f"  ATR(14) last value: {atr[-1]:.6f}")
    except Exception as e:
        print(f"  ATR ERROR: {e}")
    
    # Test DMI/ADX
    try:
        plus_di, minus_di, adx = cache.dmi("EURUSD", "M5", "BID", 14)
        print(f"  ADX(14) length: {len(adx)}")
        if len(adx) > 0:
            print(f"  ADX last value: {adx[-1]:.2f}")
    except Exception as e:
        print(f"  DMI ERROR: {e}")


def debug_strategy_params():
    """Debug strategy parameter values."""
    print("\n" + "=" * 70)
    print("2. Debug Strategy Parameters")
    print("=" * 70)
    
    from backtest_engine.core.rust_strategy_bridge import get_strategy_default_params
    
    params = get_strategy_default_params("mean_reversion_zscore")
    print("  Default parameters:")
    for key, value in sorted(params.items()):
        print(f"    {key}: {value}")


def debug_zscore_calculation():
    """Debug Z-Score calculation with synthetic data."""
    print("\n" + "=" * 70)
    print("3. Debug Z-Score Calculation")
    print("=" * 70)
    
    bid_candles, _ = generate_trending_candles(500)
    closes = np.array([c.close for c in bid_candles["M5"]], dtype=np.float64)
    
    # Calculate Z-Score manually (Python)
    lookback = 100
    zscore_values = []
    
    for i in range(lookback, len(closes)):
        window = closes[i - lookback:i]
        mean = window.mean()
        std = window.std()
        if std > 1e-10:
            zscore = (closes[i] - mean) / std
            zscore_values.append(zscore)
    
    zscore_array = np.array(zscore_values)
    print(f"  Z-Score stats:")
    print(f"    Min: {zscore_array.min():.4f}")
    print(f"    Max: {zscore_array.max():.4f}")
    print(f"    Mean: {zscore_array.mean():.4f}")
    print(f"    Std: {zscore_array.std():.4f}")
    
    # Count potential signals
    entry_threshold = 1.5
    long_signals = np.sum(zscore_array < -entry_threshold)
    short_signals = np.sum(zscore_array > entry_threshold)
    print(f"\n  Potential signals (threshold = {entry_threshold}):")
    print(f"    Long signals (Z < -{entry_threshold}): {long_signals}")
    print(f"    Short signals (Z > {entry_threshold}): {short_signals}")


def debug_backtest_run():
    """Debug a single backtest run with verbose output."""
    print("\n" + "=" * 70)
    print("4. Debug Backtest Run")
    print("=" * 70)
    
    from backtest_engine.core.rust_strategy_bridge import run_rust_backtest
    
    bid_candles, ask_candles = generate_trending_candles(500)
    
    # Very aggressive parameters for debugging
    config = {
        "symbol": "EURUSD",
        "primary_timeframe": "M5",
        "initial_capital": 100000.0,
        "risk_per_trade": 0.02,  # 2% risk
        "strategy_params": {
            "z_score_lookback": 50,
            "z_score_entry_threshold": 1.0,  # Very low threshold
            "z_score_exit_threshold": 0.2,
            "atr_period": 14,
            "atr_sl_multiplier": 1.0,  # Tighter SL
            "atr_tp_multiplier": 2.0,
            "ema_fast_period": 10,
            "ema_slow_period": 20,
            "adx_period": 14,
            "adx_threshold": 10.0,  # Very low ADX threshold
            "max_daily_trades": 100,  # Allow many trades
            "min_rr_ratio": 1.0,  # Minimum R:R
            "max_spread_pips": 10.0,  # Allow wider spreads
        }
    }
    
    print(f"  Running backtest with 500 bars...")
    print(f"  Config: {config['strategy_params']}")
    
    result = run_rust_backtest(
        strategy_name="mean_reversion_zscore",
        config=config,
        bid_candles=bid_candles,
        ask_candles=ask_candles,
    )
    
    print(f"\n  Results:")
    print(f"    Bars processed: {result.bars_processed}")
    print(f"    Total trades: {result.total_trades}")
    print(f"    Win rate: {result.win_rate * 100:.1f}%")
    print(f"    Final capital: ${result.final_capital:,.2f}")
    print(f"    Execution time: {result.execution_time_ms:.2f}ms")
    
    if result.total_trades > 0:
        print(f"\n  First 5 trades:")
        for trade in result.trades[:5]:
            print(f"    #{trade.id}: {trade.direction} @ {trade.entry_price:.5f} -> {trade.exit_price:.5f} ({trade.exit_reason})")


def debug_warmup_period():
    """Check if warmup period is causing issues."""
    print("\n" + "=" * 70)
    print("5. Debug Warmup Period")
    print("=" * 70)
    
    from backtest_engine.core.rust_strategy_bridge import list_available_strategies
    
    strategies = list_available_strategies()
    print(f"  Registered strategies: {strategies}")
    
    # The default warmup is max(z_score_lookback, ema_slow_period, 100) = 100
    print(f"  Expected warmup period: ~100 bars")
    print(f"  With 500 bars, ~400 bars should be evaluated")
    print(f"  With 1000 bars, ~900 bars should be evaluated")


if __name__ == "__main__":
    print("=" * 70)
    print("Wave 4 Pure Rust Strategy Debug")
    print("=" * 70)
    
    debug_indicator_cache()
    debug_strategy_params()
    debug_zscore_calculation()
    debug_warmup_period()
    debug_backtest_run()
    
    print("\n" + "=" * 70)
    print("Debug Complete")
    print("=" * 70)
