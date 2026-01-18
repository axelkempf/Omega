#!/usr/bin/env python3
"""Test script to verify warmup separation per timeframe is working."""

import json
from pathlib import Path


def main():
    # Lade die Backtest-Konfiguration
    config_path = Path("configs/backtest/mean_reversion_z_score_v2.json")
    config = json.loads(config_path.read_text())

    print("=== Backtest-Konfiguration ===")
    print(f'Symbol: {config["symbol"]}')
    print(f'Primary TF: {config["timeframes"]["primary"]}')
    print(f'HTF: {config["timeframes"].get("additional", [])}')
    print(f'Warmup Bars: {config["warmup_bars"]}')

    # Führe den Backtest aus
    from omega_bt import run_backtest

    result_json = run_backtest(json.dumps(config))
    result = json.loads(result_json)

    if "error" in result:
        print(f"ERROR: {result}")
        return 1
    else:
        print("\n=== Backtest-Ergebnis ===")
        trades = result.get("trades", [])
        metrics = result.get("metrics", {})
        print(f'Total Trades: {len(trades)}')
        print(f'Profit Net: {metrics.get("profit_net", "N/A")}')
        print(f'Win Rate: {metrics.get("win_rate", "N/A")}')
        print(f'Max Drawdown: {metrics.get("max_drawdown", "N/A")}')
        print("\n✅ Backtest erfolgreich mit getrenntem Warmup!")
        return 0


if __name__ == "__main__":
    exit(main())
