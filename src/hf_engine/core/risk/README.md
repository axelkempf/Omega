# Risk Management Layer

This directory contains the risk management logic, ensuring that all trades adhere to defined safety rules and constraints.

## Overview

The Risk Management Layer acts as a gatekeeper for all trading activities. It validates trade requests against a set of rules (e.g., equity limits, news events) and calculates appropriate position sizes to manage exposure.

## Key Components

### `RiskManager`
The central component for risk validation.
- **Pre-Trade Checks**:
    - **Equity**: Ensures account equity is above the minimum threshold.
    - **Trade Limits**: Checks daily trade counts and maximum open positions.
    - **News**: Blocks trades during high-impact news events (via `NewsFilter`).
    - **Direction**: Prevents duplicate trades in the same direction if configured.
- **Status Reporting**: Returns `RiskStatus` enums (e.g., `OK`, `NEWS_BLOCKED`, `EQUITY_MIN_VIOLATION`).

### `NewsFilter`
Filters trades based on economic calendar events.
- **Impact**: Blocks trading during "High" impact events.
- **Timing**: Configurable blackout windows before and after events.

### `LotSizeCalculator`
Calculates the appropriate position size (volume) for a trade.
- **Risk Models**: Supports fixed lots, percentage of equity, or risk per trade.
- **Constraints**: Respects broker limits (min/max lot size, step size).

## Usage

Before any trade is executed, the `ExecutionEngine` or `StrategyRunner` consults the `RiskManager`. If the risk check fails, the trade is rejected, and the reason is logged.

```python
# Conceptual usage
risk_status = risk_manager.check_trade_risk(symbol, direction)
if risk_status != RiskStatus.OK:
    logger.warning(f"Trade blocked: {risk_status}")
    return
```
