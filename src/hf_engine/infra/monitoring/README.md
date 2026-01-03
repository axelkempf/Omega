# Monitoring Layer

This directory contains components for monitoring the health and status of the trading engine.

## Overview

The Monitoring Layer ensures that the system is running correctly and alerts operators to any issues. It provides both active monitoring (health checks) and passive alerting (Telegram notifications).

## Key Components

### `health_server.py`
A FastAPI application that exposes health endpoints.
- **Endpoints**:
    - `/health`: Simple liveness check.
    - `/status`: Detailed status of all running strategies (active, stopped, error).
    - `/heartbeat`: Updates the system heartbeat timestamp.
- **Integration**: Used by the UI Engine and external watchdogs to verify system uptime.

### `telegram_bot.py`
Integration with the Telegram API for real-time alerts.
- **Notifications**:
    - **Trade Execution**: Entry and exit alerts with profit/loss.
    - **Errors**: Critical exceptions and system warnings.
    - **Status**: Daily summaries or on-demand status reports.
- **Configuration**: Requires `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

## Usage

The monitoring components run alongside the main trading engine. The `health_server` is typically started as a separate process or thread, while the `telegram_bot` is invoked by the `ExecutionEngine` and `log_service`.
