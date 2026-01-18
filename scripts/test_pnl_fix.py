#!/usr/bin/env python3
"""Test script to verify that trade.result is now in USD (account currency)."""

import json

import omega_bt

config = {
    "schema_version": "2.0",
    "strategy_name": "mean_reversion_z_score",
    "symbol": "EURUSD",
    "start_date": "2024-01-02",
    "end_date": "2024-01-02",
    "run_mode": "dev",
    "data_mode": "candle",
    "execution_variant": "v2",
    "rng_seed": 42,
    "warmup_bars": 500,
    "timeframes": {
        "primary": "M5",
    },
    "account": {
        "initial_balance": 10000.0,
        "account_currency": "USD",
        "risk_per_trade": 100.0,
        "max_positions": 5,
    },
    "costs": {
        "enabled": False,
    },
    "strategy_parameters": {
        "ema_length": 18,
        "atr_length": 14,
        "atr_mult": 1.4,
        "b_b_length": 48,
        "std_factor": 2.0,
        "window_length": 180,
        "z_score_long": -0.31,
        "z_score_short": 1.5,
        "kalman_q": 1.3,
        "kalman_r": 0.546,
        "htf_tf": "D1",
        "htf_ema": 75,
        "htf_filter": "above",
        "garch_alpha": 0.1,
        "garch_beta": 0.76,
        "garch_omega": 0.0,
        "garch_use_log_returns": True,
    },
}

result_json = omega_bt.run_backtest(json.dumps(config))
result = json.loads(result_json)

if "trades" in result and len(result["trades"]) > 0:
    print("=== Erste 3 Trades ===")
    for i, t in enumerate(result["trades"][:3]):
        print(f"Trade {i+1}:")
        print(f"  Direction: {t.get('direction', 'N/A')}")
        print(f"  Entry: {t['entry_price']:.5f}")
        print(f"  Exit:  {t['exit_price']:.5f}")
        print(f"  Size:  {t['size']}")
        print(f"  Result: {t['result']:.2f} USD")
        print(f"  R-Multiple: {t['r_multiple']:.2f}")
        print()

    # Verifiziere die Berechnung
    t = result["trades"][0]
    price_diff = t["exit_price"] - t["entry_price"]
    if t.get("direction") == "Short":
        price_diff = -price_diff
    expected = price_diff * t["size"] * 100000  # EURUSD: tick_value/tick_size = 100,000
    print("=== Verifikation Trade 1 ===")
    print(f"Price diff: {price_diff:.5f}")
    print(f"Berechnet: {price_diff:.5f} * {t['size']} * 100000 = {expected:.2f} USD")
    print(f"trade.result = {t['result']:.2f} USD")
    print(f"Match: {abs(expected - t['result']) < 0.01}")
    # Equity check
    if "equity_curve" in result:
        equity_curve = result["equity_curve"]
        final_equity = equity_curve[-1]["equity"] if equity_curve else None
        print("\n=== Equity ===")
        equity_msg = (
            f"Final Equity: {final_equity:.2f} USD"
            if final_equity
            else "No equity data"
        )
        print(equity_msg)
else:
    print("Keine Trades oder Fehler:")
    print(json.dumps(result, indent=2))
