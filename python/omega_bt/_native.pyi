"""Type stubs for omega_bt._native FFI module."""

def run_backtest(config_json: str) -> str:
    """Run a V2 backtest from JSON config.

    Args:
        config_json: JSON string containing backtest configuration.
            Must conform to V2 config schema (schema_version: "2").

    Returns:
        JSON string containing BacktestResult with fields:
        - ok: bool - Success flag
        - error: Optional error information if not ok
        - trades: List of completed trades
        - metrics: Performance metrics
        - metric_definitions: Metric metadata
        - equity_curve: Equity curve data
        - meta: Result metadata

    Raises:
        ValueError: If config_json is invalid JSON or fails validation.
    """
    ...
