# Execution Layer

This directory contains the logic for executing trades, managing orders, and tracking position states.

## Overview

The Execution Layer is responsible for translating high-level trade setups from strategies into concrete broker orders. It handles the complexities of order placement, sizing, and state tracking, ensuring reliable and safe execution.

## Key Components

### `ExecutionEngine`
The core component for executing trades.
- **Order Placement**: Converts `TradeSetup` objects into broker orders.
- **Idempotency**: Uses `IdempotencyCache` to prevent duplicate orders for the same signal.
- **Safety**: Wraps broker calls in safe execution blocks to handle exceptions gracefully.
- **Notifications**: Integrates with Telegram to send alerts for executed trades.

### `ExecutionTracker`
Tracks the state of orders and positions.
- **State Management**: Maintains a local view of open positions and active orders.
- **Synchronization**: Updates state based on `OrderResult` feedback from the broker.

### `sl_tp_utils`
Utilities for calculating and validating Stop Loss (SL) and Take Profit (TP) levels.
- **Normalization**: Ensures price levels are valid (e.g., aligned with pip size).
- **Distance Calculation**: Computes distances in pips or price units.

## Features

- **Idempotency**: Prevents accidental double-execution of the same signal within a short time window.
- **Risk Integration**: Collaborates with `RiskManager` (via `StrategyRunner`) to validate trades before execution.
- **Logging**: Detailed logging of all execution attempts and results.
