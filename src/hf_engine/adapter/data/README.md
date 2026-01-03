# Data Adapter Layer

This directory contains the data provider abstraction layer, responsible for fetching and normalizing market data from various sources.

## Overview

The Data Adapter Layer ensures that the trading engine receives market data in a consistent, provider-agnostic format. It handles the retrieval of historical and real-time OHLC (Open, High, Low, Close) data.

## Key Components

### `DataProviderInterface`
The abstract base class defining the contract for data providers.
- **Schema Enforcement**: Defines `OHLC` and `OHLCSeries` TypedDicts for consistent data structures.
- **Timezone Consistency**: Enforces **UTC** and **timezone-aware** timestamps for all data.
- **Methods**: Standardizes methods for fetching historical bars and current ticks.

### `MT5DataProvider`
The concrete implementation for MetaTrader 5.
- **Timeframe Mapping**: Maps internal timeframe strings (e.g., "M1", "H1") to MT5 constants.
- **Data Normalization**: Converts MT5's native data format into the standard `OHLCSeries` structure.
- **Broker Time Conversion**: Handles conversion between Broker time (used for API calls) and UTC (used internally).

### `RemoteDataProvider`
An implementation designed to fetch data from a remote source, such as the `mt5_feed_server`. This allows the trading engine to run in environments where a direct broker connection is not available (e.g., non-Windows containers).

## Data Structures

### `OHLC`
Represents a single closed candle.
- `time`: datetime (UTC, tz-aware)
- `open`, `high`, `low`, `close`, `volume`: float

### `OHLCSeries`
Represents a series of candles in a columnar format (lists), optimized for processing.
