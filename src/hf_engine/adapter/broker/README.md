# Broker Adapter Layer

This directory contains the abstraction layer for broker interactions, ensuring a clean separation between the core trading logic and specific broker APIs.

## Overview

The Broker Adapter Layer provides a unified interface for executing trades, managing orders, and handling broker connections. It is designed to support multiple broker implementations, with MetaTrader 5 (MT5) being the primary implementation for live trading.

## Key Components

### `BrokerInterface`
The abstract base class that defines the contract for all broker adapters. It enforces consistent types and signatures for:
- **Order Execution**: Market, Limit, and Stop orders.
- **Order Management**: Modification and cancellation.
- **Position Management**: Retrieving open positions and account information.
- **Timezone Handling**: All timestamps are strictly **UTC** and **timezone-aware**.

### `MT5Adapter`
The concrete implementation of `BrokerInterface` for the MetaTrader 5 platform.
- **Defensive Checks**: Validates inputs and connection state before API calls.
- **Symbol Mapping**: Translates between internal symbol names and broker-specific tickers using `SymbolMapper`.
- **Resilience**: Handles API quirks and connection issues.

### `BrokerConnectionFSM`
A Finite State Machine (FSM) responsible for maintaining a stable connection to the broker.
- **Automatic Reconnection**: Implements exponential backoff with jitter.
- **Status Management**: Tracks connection states (`INITIALIZING`, `CONNECTED`, `DEGRADED`, `DISCONNECTED`).
- **Policies**: Configurable retry limits and timeouts via `ReconnectPolicy`.

## Usage

The trading engine interacts exclusively with `BrokerInterface`, allowing for easy swapping of broker implementations (e.g., for backtesting or different providers) without changing the core logic.

```python
# Example usage (conceptual)
broker: BrokerInterface = MT5Adapter(...)
result = broker.place_order(
    symbol="EURUSD",
    volume=1.0,
    direction=Direction.BUY,
    order_type=OrderType.MARKET
)
```
