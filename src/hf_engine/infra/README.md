# Infrastructure Layer

The **Infrastructure Layer** provides the foundational services and utilities that support the entire application. It handles configuration, logging, metrics collection, and system monitoring.

## 📂 Structure

### `config/`
Centralized configuration management.
- **`environment.py`**: Loads and validates environment variables (secrets, flags).
- **`paths.py`**: Defines standard file system paths for logs, data, and artifacts.
- **`symbol_mapper.py`**: Normalizes symbol names across different brokers and data sources.
- **`branding.py`**: Application branding and display utilities.

### `logging/`
Robust logging and error handling.
- **`log_manager.py`**: Configures loggers, handlers, and formatters.
- **`log_service.py`**: Service for structured logging and log rotation.
- **`error_handler.py`**: Centralized exception handling and reporting.

### `metrics/`
Performance and health tracking.
- Collects operational metrics (latency, uptime, resource usage) for monitoring.

### `monitoring/`
System health and watchdog services.
- Monitors the health of the engine and triggers alerts or restarts if issues are detected.

## 🔧 Capabilities

- **Environment Aware**: Adapts behavior based on `dev`, `staging`, or `prod` environments.
- **Observability**: Provides deep visibility into system behavior through structured logs and metrics.
- **Standardization**: Enforces consistent paths and naming conventions across the project.

> [!TIP]
> Always use `paths.py` for file operations to ensure compatibility with the project's directory structure and deployment environments.
