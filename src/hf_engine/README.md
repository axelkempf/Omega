# Live Engine (HF Engine)

The `hf_engine` (High-Frequency Engine) is the live trading component of the stack. It acts as the bridge between your trading strategies and the market, specifically designed to interface with MetaTrader 5 (MT5).

## Architecture

### 🔌 Adapter Layer (`adapter/`)
Handles the direct communication with the MetaTrader 5 terminal.
- **MT5 Connectivity**: Manages connection state, login, and keep-alive.
- **Data Streaming**: Subscribes to market data and normalizes it for the internal event bus.
- **Order Management**: Translates internal order requests into MT5 trade operations.

### ⚙️ Core Logic (`core/`)
The brain of the live execution.
- **Risk Management**: Enforces pre-trade risk checks (e.g., max drawdown, exposure limits) to protect capital.
- **Execution**: Manages the lifecycle of orders, from signal generation to fill confirmation.
- **Event Bus**: Distributes market events to subscribed strategies.

### 🏗️ Infrastructure (`infra/`)
Support services for the engine.
- **Configuration**: Loads and validates runtime configurations.
- **Logging**: Structured logging for audit trails and debugging.
- **State Management**: Handles persistence of engine state (e.g., heartbeats, stop signals).

## Operational Safety

> [!WARNING]
> **Windows Only**: The MT5 Python integration is officially supported only on Windows. Ensure this engine is deployed in a compatible environment.

- **Heartbeats**: The engine emits regular heartbeat signals to `var/tmp/`.
- **Graceful Shutdown**: Listens for stop signals to close connections and save state cleanly.
- **Guardrails**: Built-in circuit breakers and risk limits prevent runaway algorithms.

## Usage

The live engine is typically launched via the central launcher:

```bash
python src/engine_launcher.py --config configs/live/strategy_config_<id>.json
```
