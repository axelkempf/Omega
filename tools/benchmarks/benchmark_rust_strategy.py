#!/usr/bin/env python
"""
Performance Benchmark für Wave 4 Pure Rust Strategy
====================================================

Vergleicht:
1. Python->Rust FFI Overhead (Datenkonvertierung)
2. Reiner Rust-Backtest (Strategie-Evaluation + Trade Management)
3. Indikatoren-Berechnung im Rust Cache

Target Metriken:
- Bars pro Sekunde
- FFI-Overhead vs. Rust-Kernzeit
- Memory-Footprint
"""

import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone

import numpy as np

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

# Enable Rust strategy
os.environ["OMEGA_USE_RUST_STRATEGY"] = "1"

def generate_synthetic_candles(num_bars: int, seed: int = 42, volatility: float = 0.001) -> tuple:
    """Generate synthetic OHLCV data for benchmarking.
    
    Args:
        num_bars: Number of bars to generate
        seed: Random seed for reproducibility
        volatility: Price return std dev (0.001 = ~10 pips for EURUSD)
    
    Returns:
        Tuple of (bid_candles, ask_candles) dicts with M5 timeframe
    """
    np.random.seed(seed)
    
    # Random walk price series with higher volatility for trade generation
    returns = np.random.randn(num_bars) * volatility
    prices = np.cumprod(1 + returns) * 1.10  # Start at 1.10
    
    # Add mean-reversion component for strategy signals
    # Create regime shifts every 200-500 bars
    regime_changes = np.cumsum(np.random.randint(200, 500, size=num_bars // 100 + 1))
    regime_idx = 0
    regime_price = prices[0]
    
    for i in range(num_bars):
        if regime_idx < len(regime_changes) - 1 and i >= regime_changes[regime_idx]:
            regime_idx += 1
            regime_price = prices[i]
        
        # Add mean-reversion component (price oscillates around regime mean)
        deviation = (prices[i] - regime_price) / regime_price
        if abs(deviation) > 0.01:  # > 1% deviation triggers pullback
            prices[i] = prices[i] * (1 - deviation * 0.1)  # Partial reversion
    
    # Timestamps (5-minute intervals)
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    timestamps = [base_ts + i * 300_000_000 for i in range(num_bars)]
    
    # Create mock candle objects
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
        # Create realistic OHLC with intra-bar volatility
        spread = 0.0002  # 2 pip spread
        bar_range = np.random.uniform(0.0005, 0.002)  # 5-20 pip range
        
        high = p * (1 + bar_range / 2)
        low = p * (1 - bar_range / 2)
        open_price = p * (1 + np.random.uniform(-bar_range/3, bar_range/3))
        
        bid_candles.append(MockCandle(
            timestamps[i], open_price, high, low, p, 100.0 + np.random.rand() * 50
        ))
        ask_candles.append(MockCandle(
            timestamps[i], open_price + spread, high + spread, low + spread, p + spread, 100.0 + np.random.rand() * 50
        ))
    
    return {"M5": bid_candles}, {"M5": ask_candles}


def benchmark_data_conversion(bid_candles: dict, ask_candles: dict, iterations: int = 3) -> dict:
    """Benchmark Python->Rust data conversion."""
    from omega_rust import StrategyConfig, CandleData, IndicatorCacheRust
    
    m5_bid = bid_candles["M5"]
    m5_ask = ask_candles["M5"]
    num_bars = len(m5_bid)
    
    results = {"candle_times": [], "cache_times": [], "config_time": 0}
    
    for _ in range(iterations):
        # Benchmark CandleData creation
        t0 = time.perf_counter()
        rust_bid = [
            CandleData(
                int(c.timestamp.timestamp() * 1_000_000),
                float(c.open), float(c.high), float(c.low), float(c.close), float(c.volume)
            )
            for c in m5_bid
        ]
        rust_ask = [
            CandleData(
                int(c.timestamp.timestamp() * 1_000_000),
                float(c.open), float(c.high), float(c.low), float(c.close), float(c.volume)
            )
            for c in m5_ask
        ]
        results["candle_times"].append(time.perf_counter() - t0)
        
        # Benchmark IndicatorCache setup
        t0 = time.perf_counter()
        cache = IndicatorCacheRust()
        
        # Register OHLCV data
        opens = np.array([c.open for c in m5_bid], dtype=np.float64)
        highs = np.array([c.high for c in m5_bid], dtype=np.float64)
        lows = np.array([c.low for c in m5_bid], dtype=np.float64)
        closes = np.array([c.close for c in m5_bid], dtype=np.float64)
        volumes = np.array([c.volume for c in m5_bid], dtype=np.float64)
        
        cache.register_ohlcv("EURUSD", "M5", "BID", opens, highs, lows, closes, volumes)
        
        opens_ask = np.array([c.open for c in m5_ask], dtype=np.float64)
        highs_ask = np.array([c.high for c in m5_ask], dtype=np.float64)
        lows_ask = np.array([c.low for c in m5_ask], dtype=np.float64)
        closes_ask = np.array([c.close for c in m5_ask], dtype=np.float64)
        volumes_ask = np.array([c.volume for c in m5_ask], dtype=np.float64)
        
        cache.register_ohlcv("EURUSD", "M5", "ASK", opens_ask, highs_ask, lows_ask, closes_ask, volumes_ask)
        results["cache_times"].append(time.perf_counter() - t0)
    
    # Config is fast, just measure once
    t0 = time.perf_counter()
    config = StrategyConfig("EURUSD", "M5", 100000.0, 0.01, {
        "z_score_lookback": 100.0,
        "z_score_entry_threshold": 2.0,
    })
    results["config_time"] = time.perf_counter() - t0
    
    return {
        "num_bars": num_bars,
        "candle_avg_ms": np.mean(results["candle_times"]) * 1000,
        "cache_avg_ms": np.mean(results["cache_times"]) * 1000,
        "config_ms": results["config_time"] * 1000,
        "total_conversion_ms": (np.mean(results["candle_times"]) + np.mean(results["cache_times"])) * 1000,
    }


def benchmark_rust_backtest(bid_candles: dict, ask_candles: dict, iterations: int = 3) -> dict:
    """Benchmark pure Rust backtest execution."""
    from backtest_engine.core.rust_strategy_bridge import run_rust_backtest
    
    # Use more aggressive parameters to generate trades with synthetic data
    config = {
        "symbol": "EURUSD",
        "primary_timeframe": "M5",
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,
        "strategy_params": {
            "z_score_lookback": 50,        # Shorter lookback for faster signals
            "z_score_entry_threshold": 1.5, # Lower threshold for more trades
            "z_score_exit_threshold": 0.3,  # Lower exit threshold
            "atr_period": 14,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 2.0,
            "ema_fast_period": 12,
            "ema_slow_period": 26,
            "adx_period": 14,
            "adx_threshold": 15.0,         # Lower ADX threshold for more trades
        }
    }
    
    num_bars = len(bid_candles["M5"])
    times = []
    rust_times = []
    trades_count = 0
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        total_time = time.perf_counter() - t0
        times.append(total_time)
        rust_times.append(result.execution_time_ms)
        trades_count = result.total_trades
    
    return {
        "num_bars": num_bars,
        "total_avg_ms": np.mean(times) * 1000,
        "rust_core_avg_ms": np.mean(rust_times),
        "ffi_overhead_ms": np.mean(times) * 1000 - np.mean(rust_times),
        "trades": trades_count,
        "bars_per_sec": num_bars / np.mean(times),
    }


def benchmark_memory_usage(bid_candles: dict, ask_candles: dict) -> dict:
    """Benchmark memory usage of Rust strategy."""
    from backtest_engine.core.rust_strategy_bridge import run_rust_backtest
    
    config = {
        "symbol": "EURUSD",
        "primary_timeframe": "M5",
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,
        "strategy_params": {}
    }
    
    tracemalloc.start()
    
    result = run_rust_backtest(
        strategy_name="mean_reversion_z_score",
        config=config,
        bid_candles=bid_candles,
        ask_candles=ask_candles,
    )
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "bars": result.bars_processed,
        "trades": result.total_trades,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("Wave 4 Pure Rust Strategy Performance Benchmark")
    print("=" * 70)
    
    # Test with different data sizes
    sizes = [1000, 5000, 10000, 20000]
    
    print("\n1. Data Conversion Overhead (Python -> Rust FFI)")
    print("-" * 50)
    print(f"{'Bars':>8} | {'Candles':>10} | {'Cache':>10} | {'Total':>10} | {'Rate':>12}")
    print(f"{'':>8} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(bars/s)':>12}")
    print("-" * 50)
    
    for size in sizes:
        bid, ask = generate_synthetic_candles(size)
        result = benchmark_data_conversion(bid, ask)
        rate = size / (result["total_conversion_ms"] / 1000)
        print(f"{size:>8,} | {result['candle_avg_ms']:>10.1f} | {result['cache_avg_ms']:>10.1f} | {result['total_conversion_ms']:>10.1f} | {rate:>12,.0f}")
    
    print("\n2. Full Backtest Performance")
    print("-" * 60)
    print(f"{'Bars':>8} | {'Total':>10} | {'Rust Core':>10} | {'FFI OH':>10} | {'Trades':>7} | {'Rate':>12}")
    print(f"{'':>8} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'':>7} | {'(bars/s)':>12}")
    print("-" * 60)
    
    for size in sizes:
        bid, ask = generate_synthetic_candles(size)
        result = benchmark_rust_backtest(bid, ask)
        print(f"{size:>8,} | {result['total_avg_ms']:>10.1f} | {result['rust_core_avg_ms']:>10.1f} | {result['ffi_overhead_ms']:>10.1f} | {result['trades']:>7} | {result['bars_per_sec']:>12,.0f}")
    
    print("\n3. Memory Usage")
    print("-" * 40)
    
    # Use largest dataset for memory test
    bid, ask = generate_synthetic_candles(20000)
    mem_result = benchmark_memory_usage(bid, ask)
    print(f"Peak Memory:    {mem_result['peak_mb']:.2f} MB")
    print(f"Current Memory: {mem_result['current_mb']:.2f} MB")
    print(f"Bars Processed: {mem_result['bars']:,}")
    print(f"Trades:         {mem_result['trades']}")
    
    print("\n4. Analysis Summary")
    print("-" * 40)
    
    # Calculate FFI overhead percentage
    bid, ask = generate_synthetic_candles(10000)
    backtest_result = benchmark_rust_backtest(bid, ask)
    ffi_pct = (backtest_result['ffi_overhead_ms'] / backtest_result['total_avg_ms']) * 100
    
    print(f"FFI Overhead:   {ffi_pct:.1f}% of total time")
    print(f"Rust Core:      {100 - ffi_pct:.1f}% of total time")
    print(f"Throughput:     {backtest_result['bars_per_sec']:,.0f} bars/second")
    
    # Extrapolate for 1 year of M5 data
    bars_per_year = 365 * 24 * 12  # ~105k bars
    estimated_time = bars_per_year / backtest_result['bars_per_sec']
    print(f"\nEstimated time for 1 year M5 data: {estimated_time:.1f}s ({bars_per_year:,} bars)")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
