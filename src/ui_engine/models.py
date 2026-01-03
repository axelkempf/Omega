"""
Pydantic Request/Response Models for UI Engine API.

This module provides type-safe request and response models for all FastAPI
endpoints in the UI Engine. Using Pydantic models ensures validation,
serialization, and automatic OpenAPI documentation.

Example:
    >>> from ui_engine.models import ProcessStatus, HealthResponse
    >>> status = ProcessStatus(name="my_strategy", status="running", pid=12345)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Base Response Models
# =============================================================================


class StrategyResponse(BaseModel):
    """
    Basic response model for strategy operations.

    Used for simple start/stop/restart responses where only
    name and status are needed.

    Attributes:
        name: Strategy name or alias.
        status: Current status string (e.g., "Running", "Stopped").
    """

    name: str = Field(..., description="Strategy name or alias")
    status: str = Field(..., description="Current status of the strategy")


class StrategyProcessInfo(BaseModel):
    """
    Extended process information for a strategy.

    Includes PID and start time for detailed process tracking.

    Attributes:
        name: Strategy name or alias.
        pid: Process ID if running, None otherwise.
        status: Current status string.
        start_time: Process start timestamp.
    """

    name: str = Field(..., description="Strategy name or alias")
    pid: int | None = Field(None, description="Process ID if running")
    status: str = Field(..., description="Current status of the strategy")
    start_time: datetime | None = Field(None, description="Process start timestamp")


# =============================================================================
# Detailed Status Models
# =============================================================================


class ProcessStatus(BaseModel):
    """
    Comprehensive process status response.

    Includes all relevant information about a running process including
    heartbeat status, resource usage, and timing information.

    Attributes:
        name: Strategy name or alias.
        alias: Resolved alias name.
        pid: Process ID if running.
        status: Process status (running, stopped, unknown, unresponsive).
        heartbeat_age_seconds: Seconds since last heartbeat.
        last_heartbeat: Timestamp of last heartbeat.
        cpu_percent: Current CPU usage percentage.
        memory_mb: Current memory usage in megabytes.
    """

    name: str = Field(..., description="Strategy name or alias")
    alias: str = Field("", description="Resolved alias name")
    pid: int | None = Field(None, description="Process ID if running")
    status: Literal["running", "stopped", "unknown", "unresponsive", "no_heartbeat"] = (
        Field(..., description="Current process status")
    )
    heartbeat_age_seconds: float | None = Field(
        None, description="Seconds since last heartbeat"
    )
    last_heartbeat: datetime | None = Field(
        None, description="Last heartbeat timestamp"
    )
    cpu_percent: float | None = Field(None, ge=0, description="CPU usage percentage")
    memory_mb: float | None = Field(None, ge=0, description="Memory usage in MB")


class ResourceUsage(BaseModel):
    """
    Resource usage statistics for a process.

    Provides CPU, memory, and thread information for monitoring.

    Attributes:
        status: Process status string.
        cpu_percent: CPU usage percentage (0-100+).
        memory_mb: Memory usage in megabytes.
        threads: Number of active threads.
        start_time: Process start time as formatted string.
    """

    status: str = Field(..., description="Process status")
    cpu_percent: float | None = Field(None, ge=0, description="CPU usage percentage")
    memory_mb: float | None = Field(None, ge=0, description="Memory usage in MB")
    threads: int | None = Field(None, ge=0, description="Number of active threads")
    start_time: str | None = Field(None, description="Process start time (formatted)")


# =============================================================================
# Request Models
# =============================================================================


class StartRequest(BaseModel):
    """
    Request body for starting a strategy with optional overrides.

    Allows passing configuration overrides when starting a strategy.

    Attributes:
        config_override: Optional dictionary of config values to override.
    """

    config_override: dict[str, Any] | None = Field(
        None, description="Optional configuration overrides"
    )


class RestartRequest(BaseModel):
    """
    Request body for restarting a strategy.

    Allows configuring restart behavior like delay and force options.

    Attributes:
        delay_seconds: Seconds to wait between stop and start.
        force: Whether to force kill if graceful shutdown fails.
    """

    delay_seconds: float = Field(
        6.0, ge=0, le=60, description="Delay between stop and start"
    )
    force: bool = Field(False, description="Force kill if graceful shutdown fails")


# =============================================================================
# Log Models
# =============================================================================


class LogsResponse(BaseModel):
    """
    Response model for log retrieval.

    Contains log lines and metadata about the log file.

    Attributes:
        account_id: Account ID or strategy name.
        lines: List of log lines.
        total_lines: Total number of lines returned.
        log_file: Path to the log file.
    """

    account_id: str = Field(..., description="Account ID or strategy name")
    lines: list[str] = Field(default_factory=list, description="Log lines")
    total_lines: int = Field(0, ge=0, description="Number of lines returned")
    log_file: str | None = Field(None, description="Path to log file")


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """
    Response model for health check endpoints.

    Provides overall system health status and key metrics.

    Attributes:
        status: Overall health status.
        uptime_seconds: Server uptime in seconds.
        active_processes: Number of active strategy processes.
        datafeed_status: Status of the datafeed server.
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall health status"
    )
    uptime_seconds: float = Field(0, ge=0, description="Server uptime in seconds")
    active_processes: int = Field(0, ge=0, description="Number of active processes")
    datafeed_status: Literal["ok", "offline", "unknown"] = Field(
        "unknown", description="Datafeed server status"
    )


class DatafeedHealth(BaseModel):
    """
    Response model for datafeed health check.

    Attributes:
        status: Datafeed status (ok or offline).
    """

    status: Literal["ok", "offline"] = Field(..., description="Datafeed status")


class DatafeedActionResponse(BaseModel):
    """
    Response model for datafeed start/stop actions.

    Attributes:
        status: Action result (started, stopping, stopped, error).
    """

    status: Literal["started", "stopping", "stopped", "error"] = Field(
        ..., description="Action result"
    )


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Provides consistent error format across all endpoints.

    Attributes:
        detail: Error message or description.
        error_code: Optional error code for programmatic handling.
    """

    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code for programmatic use")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base responses
    "StrategyResponse",
    "StrategyProcessInfo",
    # Status models
    "ProcessStatus",
    "ResourceUsage",
    # Request models
    "StartRequest",
    "RestartRequest",
    # Log models
    "LogsResponse",
    # Health models
    "HealthResponse",
    "DatafeedHealth",
    "DatafeedActionResponse",
    # Error models
    "ErrorResponse",
]
