#!/usr/bin/env python3
"""
Backtest Pipeline Profiler
==========================

Dieses Tool führt eine detaillierte Zeitmessung der Backtest-Pipeline durch,
um zu identifizieren, wo die meiste Zeit verbracht wird.

Gemessene Phasen:
1. Data Loading (CSV/Parquet)
2. Alignment & Preprocessing
3. Strategy Preparation (Import, Wrapper, Models)
4. Event Engine Loop (pro-Bar Iteration)
   - Strategy Evaluation (on_data / check_entry)
   - Indicator Cache Lookups
   - Execution Simulation
   - Portfolio Updates
   - Position Management
5. Reporting & Metrics

Usage:
    python tools/profile_backtest_pipeline.py configs/backtest/<config>.json [--detailed] [--cprofile]
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Setup path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# ==============================================================================
# Timing Utilities
# ==============================================================================


@dataclass
class TimingStats:
    """Statistics for a single phase."""
    
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    samples: List[float] = field(default_factory=list)
    
    def record(self, duration: float) -> None:
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        if len(self.samples) < 1000:  # Limit memory usage
            self.samples.append(duration)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def std_time(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return float(np.std(self.samples))
    
    @property
    def median_time(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.median(self.samples))


class PipelineProfiler:
    """Collects timing data for the backtest pipeline."""
    
    def __init__(self) -> None:
        self.stats: Dict[str, TimingStats] = {}
        self._start_times: Dict[str, float] = {}
        self.detailed_mode = False
        self.per_bar_samples: Dict[str, List[float]] = defaultdict(list)
        
    def start(self, phase: str) -> None:
        self._start_times[phase] = time.perf_counter()
        
    def stop(self, phase: str) -> float:
        if phase not in self._start_times:
            return 0.0
        duration = time.perf_counter() - self._start_times.pop(phase)
        if phase not in self.stats:
            self.stats[phase] = TimingStats(name=phase)
        self.stats[phase].record(duration)
        return duration
    
    @contextmanager
    def measure(self, phase: str):
        """Context manager for timing a phase."""
        self.start(phase)
        try:
            yield
        finally:
            self.stop(phase)
    
    def time_function(self, phase: str) -> Callable:
        """Decorator for timing a function."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start(phase)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.stop(phase)
            return wrapper
        return decorator
    
    def record_bar_timing(self, bar_idx: int, phase: str, duration: float) -> None:
        """Record per-bar timing for detailed analysis."""
        if self.detailed_mode:
            key = f"bar_{phase}"
            self.per_bar_samples[key].append(duration)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_time = sum(s.total_time for s in self.stats.values())
        
        summary = {
            "total_time_seconds": round(total_time, 3),
            "phases": {}
        }
        
        # Sort by total time descending
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s.total_time,
            reverse=True
        )
        
        for stat in sorted_stats:
            pct = (stat.total_time / total_time * 100) if total_time > 0 else 0
            summary["phases"][stat.name] = {
                "total_seconds": round(stat.total_time, 6),
                "percentage": round(pct, 2),
                "call_count": stat.call_count,
                "avg_ms": round(stat.avg_time * 1000, 4),
                "median_ms": round(stat.median_time * 1000, 4),
                "min_ms": round(stat.min_time * 1000, 4) if stat.min_time != float("inf") else 0,
                "max_ms": round(stat.max_time * 1000, 4),
                "std_ms": round(stat.std_time * 1000, 4),
            }
        
        return summary
    
    def print_report(self, title: str = "Pipeline Profiling Report") -> None:
        """Print a formatted profiling report."""
        summary = self.summary()
        
        print("\n" + "=" * 80)
        print(f"📊 {title}")
        print("=" * 80)
        print(f"\n⏱️  Total Time: {summary['total_time_seconds']:.3f}s")
        print("\n" + "-" * 80)
        print(f"{'Phase':<40} {'Time (s)':<12} {'%':<8} {'Calls':<10} {'Avg (ms)':<12}")
        print("-" * 80)
        
        for name, data in summary["phases"].items():
            print(f"{name:<40} {data['total_seconds']:<12.4f} {data['percentage']:<8.1f} "
                  f"{data['call_count']:<10} {data['avg_ms']:<12.4f}")
        
        print("-" * 80)
        
        # Highlight top time consumers
        if summary["phases"]:
            top_3 = list(summary["phases"].items())[:3]
            print("\n🔥 Top 3 Time Consumers:")
            for i, (name, data) in enumerate(top_3, 1):
                print(f"   {i}. {name}: {data['total_seconds']:.3f}s ({data['percentage']:.1f}%)")
        
        print()


# ==============================================================================
# Instrumented Components
# ==============================================================================


class InstrumentedEventEngine:
    """Wrapper around EventEngine with detailed timing instrumentation."""
    
    def __init__(
        self,
        original_engine,
        profiler: PipelineProfiler,
        detailed: bool = False
    ):
        self.engine = original_engine
        self.profiler = profiler
        self.detailed = detailed
        
    def run(self) -> None:
        """Run with instrumented timing."""
        total = len(self.engine.bid_candles)
        
        if self.engine.original_start_dt is None:
            raise ValueError("original_start_dt muss gesetzt werden!")
        
        start_index = next(
            (i for i, c in enumerate(self.engine.bid_candles) 
             if c.timestamp >= self.engine.original_start_dt),
            None
        )
        if start_index is None:
            raise ValueError("Kein Startindex gefunden!")
        
        # Import here to avoid circular imports
        from backtest_engine.core.indicator_cache import get_cached_indicator_cache
        from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
        
        # Time indicator cache init
        with self.profiler.measure("indicator_cache_init"):
            ind_cache = get_cached_indicator_cache(self.engine.multi_candle_data)
        
        # Time slice creation
        with self.profiler.measure("slice_creation"):
            symbol_slice = SymbolDataSlice(
                multi_candle_data=self.engine.multi_candle_data,
                index=start_index,
                indicator_cache=ind_cache,
            )
            slice_map = {self.engine.symbol: symbol_slice}
        
        # Main loop timing
        strategy_times = []
        execution_times = []
        portfolio_times = []
        position_mgmt_times = []
        
        num_bars = total - start_index
        
        with self.profiler.measure("event_loop_total"):
            for i in range(start_index, total):
                bid_candle = self.engine.bid_candles[i]
                ask_candle = self.engine.ask_candles[i]
                timestamp = bid_candle.timestamp
                
                # Slice index update
                t0 = time.perf_counter()
                symbol_slice.set_index(i)
                slice_time = time.perf_counter() - t0
                
                # === STRATEGY EVALUATION ===
                t0 = time.perf_counter()
                signals = self.engine.strategy.evaluate(i, slice_map)
                strategy_time = time.perf_counter() - t0
                strategy_times.append(strategy_time)
                
                if signals:
                    if not isinstance(signals, list):
                        signals = [signals]
                    for signal in signals:
                        t0 = time.perf_counter()
                        self.engine.executor.process_signal(signal)
                        execution_times.append(time.perf_counter() - t0)
                
                # === EXITS ===
                if self.engine.executor.active_positions:
                    t0 = time.perf_counter()
                    self.engine.executor.evaluate_exits(bid_candle, ask_candle)
                    execution_times.append(time.perf_counter() - t0)
                
                # === POSITION MANAGEMENT ===
                if self.engine.executor.active_positions:
                    t0 = time.perf_counter()
                    strategy_instance = getattr(self.engine.strategy.strategy, "strategy", None)
                    pm = getattr(strategy_instance, "position_manager", None)
                    if pm:
                        if not getattr(pm, "portfolio", None):
                            pm.attach_portfolio(self.engine.portfolio)
                        open_pos = self.engine.portfolio.get_open_positions(self.engine.symbol)
                        all_pos = self.engine.executor.active_positions
                        pm.manage_positions(
                            open_positions=open_pos,
                            symbol_slice=symbol_slice,
                            bid_candle=bid_candle,
                            ask_candle=ask_candle,
                            all_positions=all_pos,
                        )
                    position_mgmt_times.append(time.perf_counter() - t0)
                
                # === PORTFOLIO UPDATE ===
                t0 = time.perf_counter()
                self.engine.portfolio.update(timestamp)
                portfolio_times.append(time.perf_counter() - t0)
        
        # Record aggregated stats
        if strategy_times:
            stat = TimingStats(name="strategy_evaluate")
            for t in strategy_times:
                stat.record(t)
            self.profiler.stats["strategy_evaluate"] = stat
        
        if execution_times:
            stat = TimingStats(name="execution_simulator")
            for t in execution_times:
                stat.record(t)
            self.profiler.stats["execution_simulator"] = stat
        
        if portfolio_times:
            stat = TimingStats(name="portfolio_update")
            for t in portfolio_times:
                stat.record(t)
            self.profiler.stats["portfolio_update"] = stat
        
        if position_mgmt_times:
            stat = TimingStats(name="position_management")
            for t in position_mgmt_times:
                stat.record(t)
            self.profiler.stats["position_management"] = stat


def run_profiled_backtest(
    config: dict,
    profiler: PipelineProfiler,
    detailed: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """Run a backtest with full pipeline profiling."""
    
    from backtest_engine.core.event_engine import EventEngine
    from backtest_engine.core.execution_simulator import ExecutionSimulator
    from backtest_engine.core.portfolio import Portfolio
    from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
    from backtest_engine.runner import (
        _build_central_registry,
        _get_or_build_alignment,
        _load_execution_costs,
        _load_symbol_specs,
        diagnose_alignment,
        load_data,
        prepare_strategies,
        prepare_time_window,
    )
    from backtest_engine.sizing.commission import CommissionModel
    from backtest_engine.sizing.lot_sizer import LotSizer
    from backtest_engine.sizing.rate_provider import (
        CompositeRateProvider,
        StaticRateProvider,
        TimeSeriesRateProvider,
    )
    
    mode = config.get("mode", "candle")
    enable_logging = config.get("enable_entry_logging", False)
    tf_config = config.get("timeframes", {})
    primary_tf = tf_config.get("primary", "M15")
    
    # === Phase 1: Time Window Preparation ===
    with profiler.measure("1_prepare_time_window"):
        start_dt, end_dt, extended_start, warmup_bars = prepare_time_window(config)
    
    # === Phase 2: Load Execution Costs & Symbol Specs ===
    with profiler.measure("2_load_configs"):
        exec_costs = _load_execution_costs(config)
        symbol_specs = _load_symbol_specs(config)
        central_registry = _build_central_registry(symbol_specs)
    
    # === Phase 3: Build Models ===
    with profiler.measure("3_build_models"):
        defaults = exec_costs.get("defaults", {})
        slip_conf = {**defaults.get("slippage", {}), **config.get("slippage", {})}
        fee_conf = {**defaults.get("fees", {}), **config.get("fees", {})}
        
        slip_mult = float(config.get("execution", {}).get("slippage_multiplier", 1.0))
        fee_mult = float(config.get("execution", {}).get("fee_multiplier", 1.0))
        
        slippage_model = SlippageModel(
            fixed_pips=float(slip_conf.get("fixed_pips", 0.0)) * slip_mult,
            random_pips=float(slip_conf.get("random_pips", 0.0)) * slip_mult,
        )
        fee_model = FeeModel(
            per_million=float(fee_conf.get("per_million", 0.0)) * fee_mult,
            lot_size=float(fee_conf.get("lot_size", 100_000.0)),
            min_fee=float(fee_conf.get("min_fee", 0.0)),
        )
        
        account_ccy = config.get("account_currency", "EUR")
        rates_cfg = config.get("rates", {}) or {}
        rates_mode = (
            rates_cfg.get("mode") or ("static" if "rates_static" in config else "timeseries")
        ).lower()
        rates_static = {k.upper(): float(v) for k, v in config.get("rates_static", {}).items()}
        
        if rates_mode == "static":
            rate_provider = StaticRateProvider(rates_static, strict=bool(rates_cfg.get("strict", True)))
        else:
            ts_pairs = [p.upper() for p in rates_cfg.get("pairs", [])]
            ts_tf = primary_tf
            rate_provider = TimeSeriesRateProvider(
                pairs=ts_pairs,
                timeframe=ts_tf,
                start_dt=extended_start,
                end_dt=end_dt,
                use_price=rates_cfg.get("use_price", "close"),
                stale_limit_bars=rates_cfg.get("stale_limit_bars", 2),
                strict=bool(rates_cfg.get("strict", True)),
            )
        
        lot_sizer = LotSizer(account_ccy, rate_provider, central_registry)
        commission_model = CommissionModel(
            account_ccy, rate_provider, exec_costs, central_registry, multiplier=fee_mult
        )
    
    # === Phase 4: Load Data ===
    with profiler.measure("4_load_data"):
        symbol_map, bid_candles, ask_candles, tick_data = load_data(
            config, mode, extended_start, end_dt
        )
    
    # === Phase 5: Data Alignment ===
    with profiler.measure("5_data_alignment"):
        bid_aligned, ask_aligned, multi_candle_data_aligned = _get_or_build_alignment(
            symbol_map=symbol_map,
            primary_tf=primary_tf,
            config=config,
            start_dt=start_dt,
        )
    
    # === Phase 6: Strategy Preparation ===
    with profiler.measure("6_strategy_preparation"):
        envs = prepare_strategies(
            config,
            symbol_map,
            slippage_model,
            fee_model,
            enable_logging,
            symbol_specs=symbol_specs,
            lot_sizer=lot_sizer,
            commission_model=commission_model,
        )
    
    env = envs[0]
    
    # === Phase 7: Engine Loop (instrumented) ===
    engine = EventEngine(
        bid_candles=bid_aligned,
        ask_candles=ask_aligned,
        strategy=env.strategy,
        executor=env.executor,
        portfolio=env.portfolio,
        multi_candle_data=multi_candle_data_aligned,
        symbol=config["symbol"],
    )
    engine.original_start_dt = start_dt
    
    instrumented_engine = InstrumentedEventEngine(engine, profiler, detailed=detailed)
    instrumented_engine.run()
    
    # === Phase 8: Metrics Calculation ===
    with profiler.measure("8_metrics_calculation"):
        summary = env.portfolio.get_summary()
    
    return env.portfolio, profiler.summary()


def run_cprofile_analysis(config: dict, output_file: Optional[str] = None) -> str:
    """Run cProfile analysis on the backtest."""
    from backtest_engine.runner import run_backtest_and_return_portfolio
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        portfolio, _ = run_backtest_and_return_portfolio(config)
    finally:
        profiler.disable()
    
    # Generate stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(50)
    
    result = stream.getvalue()
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(result)
        print(f"📄 cProfile output saved to: {output_file}")
    
    return result


def analyze_strategy_overhead(config: dict) -> Dict[str, Any]:
    """Analyze strategy-specific overhead by measuring on_data calls."""
    from importlib import import_module
    
    from backtest_engine.data.data_handler import CSVDataHandler
    from backtest_engine.runner import prepare_time_window
    
    # Load strategy
    strat_conf = config["strategy"]
    module_path = strat_conf["module"]
    if not module_path.startswith("strategies."):
        module_path = f"strategies.{module_path}"
    
    strat_module = import_module(module_path)
    strat_class = getattr(strat_module, strat_conf["class"])
    parameters = strat_conf.get("parameters", {})
    
    # Create strategy instance
    if "symbol" in parameters:
        pass
    elif "symbol" in config:
        from inspect import signature
        if "symbol" in signature(strat_class.__init__).parameters:
            parameters["symbol"] = config["symbol"]
    
    # Add timeframe if required
    from inspect import signature
    sig = signature(strat_class.__init__)
    if "timeframe" in sig.parameters and "timeframe" not in parameters:
        parameters["timeframe"] = config.get("timeframes", {}).get("primary", "M15")
    
    strategy = strat_class(**parameters)
    
    # Load minimal data
    start_dt, end_dt, extended_start, warmup_bars = prepare_time_window(config)
    symbol = config["symbol"]
    primary_tf = config.get("timeframes", {}).get("primary", "M15")
    
    dh = CSVDataHandler(symbol=symbol, timeframe=primary_tf)
    candles = dh.load_candles(start_dt=extended_start, end_dt=end_dt)
    
    # Measure on_data calls
    from backtest_engine.core.indicator_cache import IndicatorCache
    from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
    
    multi_candle_data = {primary_tf: {"bid": candles["bid"], "ask": candles["ask"]}}
    ind_cache = IndicatorCache(multi_candle_data)
    
    num_bars = len(candles["bid"])
    
    # Warmup timing
    warmup_times = []
    for i in range(min(100, num_bars)):
        symbol_slice = SymbolDataSlice(multi_candle_data, i, ind_cache)
        slice_map = {symbol: symbol_slice}
        
        t0 = time.perf_counter()
        _ = strategy.on_data(slice_map)
        warmup_times.append(time.perf_counter() - t0)
    
    # Steady-state timing
    steady_times = []
    gc.disable()
    for i in range(100, min(1000, num_bars)):
        symbol_slice = SymbolDataSlice(multi_candle_data, i, ind_cache)
        slice_map = {symbol: symbol_slice}
        
        t0 = time.perf_counter()
        _ = strategy.on_data(slice_map)
        steady_times.append(time.perf_counter() - t0)
    gc.enable()
    
    return {
        "strategy_class": strat_conf["class"],
        "warmup_avg_ms": round(np.mean(warmup_times) * 1000, 4),
        "steady_avg_ms": round(np.mean(steady_times) * 1000, 4),
        "steady_median_ms": round(np.median(steady_times) * 1000, 4),
        "steady_std_ms": round(np.std(steady_times) * 1000, 4),
        "steady_max_ms": round(max(steady_times) * 1000, 4),
        "total_bars_tested": len(steady_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile backtest pipeline to identify performance bottlenecks"
    )
    parser.add_argument("config", help="Path to backtest config JSON")
    parser.add_argument("--detailed", action="store_true", help="Enable per-bar detailed timing")
    parser.add_argument("--cprofile", action="store_true", help="Run cProfile analysis")
    parser.add_argument("--strategy-only", action="store_true", help="Only analyze strategy overhead")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"📂 Loading config: {config_path}")
    print(f"📈 Symbol: {config.get('symbol', 'N/A')}")
    print(f"📅 Period: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}")
    
    if args.strategy_only:
        print("\n🔍 Analyzing strategy overhead...")
        results = analyze_strategy_overhead(config)
        print("\n📊 Strategy Analysis Results:")
        for k, v in results.items():
            print(f"   {k}: {v}")
        return
    
    if args.cprofile:
        print("\n🔬 Running cProfile analysis...")
        output_file = args.output or "var/profiling/cprofile_output.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result = run_cprofile_analysis(config, output_file)
        print("\n" + result[:5000])  # Print first 5000 chars
        return
    
    # Run profiled backtest
    print("\n🚀 Running profiled backtest...")
    profiler = PipelineProfiler()
    profiler.detailed_mode = args.detailed
    
    gc.collect()
    portfolio, summary = run_profiled_backtest(config, profiler, detailed=args.detailed)
    
    # Print report
    profiler.print_report("Backtest Pipeline Profiling Report")
    
    # Print trade summary
    trades = len(getattr(portfolio, "closed_positions", []))
    print(f"📈 Trades: {trades}")
    print(f"💰 Final Balance: {portfolio.get_summary().get('final_balance', 'N/A')}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
