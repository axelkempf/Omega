# Adapter Layer

The **Adapter Layer** serves as the interface between the core trading engine and external systems. It abstracts the complexities of broker connections, data feeds, and API communication, ensuring that the core logic remains agnostic to specific implementations.

## 📂 Structure

### `broker/`
Handles all interactions with the brokerage platform.
- **`broker_interface.py`**: Defines the abstract contract for broker adapters.
- **`mt5_adapter.py`**: Concrete implementation for MetaTrader 5 (Windows-only).
- **`broker_connection_fsm.py`**: Finite State Machine managing connection lifecycle and recovery.

### `data/`
Responsible for fetching and streaming market data.
- **`data_provider_interface.py`**: Abstract base class for data providers.
- **`mt5_data_provider.py`**: Fetches live market data directly from MT5.
- **`remote_data_provider.py`**: Client for consuming data from a remote data feed service.

### `fastapi/`
Provides the HTTP/WebSocket interface for the engine.
- Exposes endpoints for control (start/stop), status monitoring, and log streaming.

## 🚀 Key Features

- **Abstraction**: Switch between different brokers or data sources without changing core logic.
- **Resilience**: The Broker FSM handles connection drops and re-logins automatically.
- **Flexibility**: Supports both local (direct MT5) and remote (networked) data modes.

> [!NOTE]
> The `mt5_adapter` requires a Windows environment or a compatible compatibility layer to function, as it depends on the `MetaTrader5` Python package.
