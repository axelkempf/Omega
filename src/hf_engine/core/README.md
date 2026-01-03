# Core Engine

The **Core Engine** contains the central business logic of the trading system. It orchestrates trade execution, risk management, and session state, ensuring that trading strategies are executed safely and effectively.

## 📂 Structure

### `execution/`
Manages the lifecycle of trades and orders.
- **`execution_engine.py`**: The brain of the operation. Coordinates signals, risk checks, and broker orders.
- **`execution_tracker.py`**: Tracks the state of active orders and positions.
- **`session_state.py`**: Maintains the runtime state of the trading session.
- **`sl_tp_utils.py`**: Utilities for calculating Stop Loss and Take Profit levels.

### `risk/`
Enforces safety rules and capital preservation.
- **`risk_manager.py`**: Validates trades against risk parameters (drawdown, exposure, etc.).
- **`lot_size_calculator.py`**: Determines the appropriate position size based on account equity and risk per trade.
- **`news_filter.py`**: Blocks trading during high-impact economic events.

### `controlling/`
High-level supervision and control logic.
- Manages the overall flow and coordination between components.

## 🛡️ Design Principles

- **Safety First**: Risk checks are performed *before* any order is sent to the broker.
- **Statefulness**: The engine maintains a robust internal state to handle restarts and connection interruptions.
- **Separation of Concerns**: Execution logic is decoupled from strategy signal generation and broker implementation.

> [!IMPORTANT]
> The `risk_manager` is the final gatekeeper. It has the authority to reject any trade that violates defined risk limits, regardless of the strategy signal.
