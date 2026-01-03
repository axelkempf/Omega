# MT5 Feed Server

This directory contains the FastAPI application that exposes MetaTrader 5 market data via a REST API.

## Overview

The `mt5_feed_server` acts as a bridge, allowing external components or remote services to access MT5 market data over HTTP. This is particularly useful for decoupling the data source (which requires Windows for MT5) from consumers running on other operating systems (e.g., Linux containers).

## Key Features

- **REST API**: Provides endpoints to query historical data and current market state.
- **Structured Logging**: Integrates with the project's `log_service` for consistent observability.
- **DoS Protection**: Implements `DATAFEED_MAX_BARS` limit to prevent excessive data requests.
- **Symbol Mapping**: Uses `SymbolMapper` to ensure consistent symbol naming across the system.

## Configuration

The server is configured via environment variables:
- `LOG_LEVEL`: Sets the logging verbosity (default: INFO).
- `DATAFEED_MAX_BARS`: Maximum number of bars allowed in a single request (default: 10000).

## Architecture

The server wraps the `MT5DataProvider` and exposes its functionality via FastAPI endpoints. It handles request validation using Pydantic models and ensures responses are JSON-serializable.
