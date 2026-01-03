# Source Code (`src`)

This directory contains the core source code for the Omega trading stack. The system is designed with a modular architecture, separating concerns between live execution, backtesting, user interface, and strategy logic.

## Directory Structure

| Component | Description |
|-----------|-------------|
| **[`backtest_engine/`](backtest_engine/)** | Event-driven backtesting and optimization framework. Handles historical data replay, simulation, and performance analysis. |
| **[`hf_engine/`](hf_engine/)** | High-Frequency / Live Trading Engine. Manages connectivity with MetaTrader 5, risk management, and order execution. |
| **[`strategies/`](strategies/)** | Collection of trading strategies. Includes base classes and templates for developing new strategies. |
| **[`ui_engine/`](ui_engine/)** | FastAPI-based backend for the user interface. Provides endpoints for process control, monitoring, and log streaming. |
| **`engine_launcher.py`** | The central entry point for launching live trading sessions, datafeeds, or backtests via configuration. |

## Key Concepts

- **Event-Driven Architecture**: Both live and backtest engines operate on an event loop, processing ticks and signals as they occur.
- **Configuration Driven**: All executions are controlled by JSON configuration files located in the root `configs/` directory.
- **Separation of Concerns**: Strategies are agnostic to the execution environment (live vs. backtest) where possible, relying on abstract interfaces.

> [!NOTE]
> Ensure you have the correct environment variables set up in your `.env` file before running any engine. See the root `README.md` for setup instructions.
