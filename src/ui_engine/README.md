# UI Engine

The `ui_engine` provides a FastAPI-based backend service for monitoring and controlling the trading system. It serves as the interface between the operator (or a frontend application) and the underlying trading engines.

## Features

- **Process Control**: Start, stop, and restart trading strategies and datafeeds via REST endpoints.
- **Monitoring**: Real-time status checks, resource usage monitoring (CPU/RAM), and heartbeat verification.
- **Log Streaming**: WebSocket endpoints for streaming live logs from the trading engines to the UI.
- **Configuration Management**: View and manage strategy configurations.

## Key Components

- **`main.py`**: The FastAPI application entry point. Defines the API routes and middleware.
- **`controller.py`**: The core logic for managing subprocesses. Handles spawning `engine_launcher.py` instances and tracking their lifecycle.
- **`models.py`**: Pydantic models defining the API request and response schemas.
- **`utils.py`**: Utility functions for file handling, log reading, and system interactions.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start/{name}` | Start a strategy or process by alias. |
| `POST` | `/stop/{name}` | Gracefully stop a running process. |
| `GET` | `/status/{name}` | Get the current status and heartbeat freshness. |
| `WS` | `/ws/logs/{id}` | WebSocket connection for live log streaming. |

## Running the UI

Start the server using `uvicorn`:

```bash
uvicorn src.ui_engine.main:app --reload --port 8000
```

> [!NOTE]
> The UI Engine relies on the `var/` directory for state management (heartbeats, stop signals). Ensure the directory structure is initialized.
