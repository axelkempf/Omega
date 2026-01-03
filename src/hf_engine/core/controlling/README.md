# Controlling Layer

This directory contains the orchestration logic for the trading engine, managing the lifecycle of strategies, sessions, and events.

## Overview

The Controlling Layer acts as the central nervous system of the trading engine. It coordinates the flow of data and events between the Broker Adapter, Data Provider, and Strategy logic. It ensures that strategies run within their defined sessions and react to market events in real-time.

## Key Components

### `EventBus`
A lightweight, thread-safe event bus that drives the event-driven architecture.
- **Event Types**: `TIMER_TICK`, `BAR_CLOSE`, `NEWS`, `BROKER_STATUS`, `SHUTDOWN`, etc.
- **Features**:
    - **Backpressure Monitoring**: Warns when the event queue fills up.
    - **Slow Handler Detection**: Logs warnings if event handlers take too long.
    - **Thread Pool**: Distributes event processing across worker threads.

### `StrategyRunner`
The main controller for a single strategy instance.
- **Responsibility**: Binds the Strategy, Broker, and Data Provider together.
- **Lifecycle**: Initializes the strategy, starts the event loop, and manages shutdown.
- **Integration**: Uses `ExecutionEngine` for trades and `RiskManager` for validation.

### `SessionRunner`
Manages trading sessions (e.g., specific hours of the day).
- **Function**: Checks if the current time falls within allowed trading windows.
- **Events**: Triggers `SESSION_OPEN` and `SESSION_CLOSE` events.

### `PositionMonitorController`
Monitors open positions and manages their lifecycle.
- **Updates**: Periodically syncs position state with the broker.
- **Exits**: Triggers exit logic based on strategy rules or risk limits.

## Architecture

The system follows an event-driven pattern:
1. **Sources** (Timer, Data Provider) publish events to the `EventBus`.
2. **Subscribers** (Strategy, Position Monitor) react to these events.
3. **Controllers** (`StrategyRunner`) manage the overall state and coordination.
