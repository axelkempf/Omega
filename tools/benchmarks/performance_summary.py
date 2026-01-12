#!/usr/bin/env python3
"""
Wave 4 Pure Rust Strategy - Performance Summary

Generates comprehensive performance metrics for the Rust-based
Mean Reversion Z-Score strategy.
"""

import time
from datetime import datetime, timezone

import numpy as np


def create_mock_candles(num_bars: int, seed: int = 42) -> tuple[dict, dict]:
    """Generate realistic OHLCV data for backtesting."""
    np.random.seed(seed)
    returns = np.random.randn(num_bars) * 0.001
    prices = np.cumprod(1 + returns) * 1.10

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

    bid_candles = {"M5": []}
    ask_candles = {"M5": []}

    for i in range(num_bars):
        p = prices[i]
        spread = 0.0002
        bar_range = np.random.uniform(0.0005, 0.002)
        high = p * (1 + bar_range / 2)
        low = p * (1 - bar_range / 2)
        open_price = p * (1 + np.random.uniform(-bar_range / 3, bar_range / 3))

        bid_candles["M5"].append(MockCandle(timestamps[i], open_price, high, low, p, 100.0))
        ask_candles["M5"].append(
            MockCandle(timestamps[i], open_price + spread, high + spread, low + spread, p + spread, 100.0)
        )

    return bid_candles, ask_candles


def run_performance_summary():
    """Run comprehensive performance analysis."""
    from backtest_engine.core.rust_strategy_bridge import run_rust_backtest

    config = {
        "symbol": "EURUSD",
        "primary_timeframe": "M5",
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,
        "strategy_params": {
            "z_score_lookback": 50,
            "z_score_entry_threshold": 1.5,
            "z_score_exit_threshold": 0.3,
            "atr_period": 14,
            "adx_threshold": 15.0,
        },
    }

    print("=" * 70)
    print("Wave 4 Pure Rust Strategy - Performance Summary")
    print("=" * 70)
    print()

    # Test with different data sizes
    test_sizes = [1000, 5000, 10000]

    print("Performance Scaling Analysis")
    print("-" * 70)
    print(f"{'Bars':>10} {'Time (ms)':>12} {'Trades':>10} {'Throughput':>15} {'Win Rate':>12}")
    print("-" * 70)

    for num_bars in test_sizes:
        bid_candles, ask_candles = create_mock_candles(num_bars)

        start = time.perf_counter()
        result = run_rust_backtest("mean_reversion_zscore", config, bid_candles, ask_candles)
        elapsed = (time.perf_counter() - start) * 1000

        throughput = f"{num_bars / (elapsed / 1000):,.0f} bars/s"
        win_rate = f"{result.win_rate * 100:.1f}%"

        print(f"{num_bars:>10,} {elapsed:>12.2f} {result.total_trades:>10} {throughput:>15} {win_rate:>12}")

    print("-" * 70)
    print()

    # Detailed analysis with 10k bars
    print("Detailed Backtest Results (10,000 bars)")
    print("-" * 70)

    bid_candles, ask_candles = create_mock_candles(10000)
    result = run_rust_backtest("mean_reversion_zscore", config, bid_candles, ask_candles)

    print(f"  Bars Processed:     {result.bars_processed:>15,}")
    print(f"  Total Trades:       {result.total_trades:>15,}")
    print(f"  Win Rate:           {result.win_rate * 100:>14.1f}%")
    print(f"  Profit Factor:      {result.profit_factor:>15.2f}")
    print(f"  Total PnL:          ${result.total_pnl:>14,.2f}")
    print(f"  Final Capital:      ${result.final_capital:>14,.2f}")
    print(f"  ROI:                {result.roi:>14.2f}%")
    print("-" * 70)
    print()

    # Timing breakdown
    print("Timing Breakdown")
    print("-" * 70)
    print(f"  Total Execution:    {result.execution_time_ms:>14.2f} ms")
    print(f"  Strategy Eval:      {result.strategy_time_ms:>14.2f} ms")
    ffi_overhead = result.execution_time_ms - result.strategy_time_ms
    print(f"  FFI Overhead:       {ffi_overhead:>14.2f} ms")
    throughput = result.bars_processed / (result.execution_time_ms / 1000) if result.execution_time_ms > 0 else 0
    print(f"  Throughput:         {throughput:>14,.0f} bars/sec")
    print("-" * 70)
    print()

    # Strategy parameters
    print("Strategy Configuration")
    print("-" * 70)
    print(f"  Symbol:             {config['symbol']}")
    print(f"  Timeframe:          {config['primary_timeframe']}")
    print(f"  Initial Capital:    ${config['initial_capital']:,.2f}")
    print(f"  Risk per Trade:     {config['risk_per_trade'] * 100:.1f}%")
    print(f"  Z-Score Lookback:   {config['strategy_params']['z_score_lookback']}")
    print(f"  Entry Threshold:    {config['strategy_params']['z_score_entry_threshold']}")
    print(f"  Exit Threshold:     {config['strategy_params']['z_score_exit_threshold']}")
    print(f"  ATR Period:         {config['strategy_params']['atr_period']}")
    print(f"  ADX Threshold:      {config['strategy_params']['adx_threshold']}")
    print("-" * 70)
    print()

    print("=" * 70)
    print("✅ Wave 4 Pure Rust Strategy Integration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_performance_summary()
