"""Command-line interface for Omega V2 backtests.

Usage:
    python -m bt run config.json
    python -m bt run config.json --output-dir var/results/custom
    python -m bt validate config.json --strict
    python -m bt schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .config import load_config, validate_config
from .reporting import REPORT_METRICS, extract_key_metrics, format_metric
from .runner import run_backtest

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_CONFIG_NOT_FOUND = 2
EXIT_VALIDATION_FAILED = 3
EXIT_FFI_UNAVAILABLE = 4


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="omega-bt",
        description="Omega V2 Backtest Runner - High-performance Rust-based backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run configs/backtest/mean_reversion_z_score.json
  %(prog)s run config.json -o var/results/my_run
  %(prog)s run config.json --json
  %(prog)s validate config.json --strict
  %(prog)s schema
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a backtest",
        description="Execute a backtest using the Rust V2 engine.",
    )
    run_parser.add_argument(
        "config",
        type=Path,
        help="Path to config JSON file",
    )
    run_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory override (default: var/results/backtests/<run_id>)",
    )
    run_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output (only errors)",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    run_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict config validation (requires jsonschema)",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate config file without running",
        description="Validate a backtest config file.",
    )
    validate_parser.add_argument(
        "config",
        type=Path,
        help="Path to config JSON file",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict schema validation (requires jsonschema)",
    )

    # schema command
    subparsers.add_parser(
        "schema",
        help="Print JSON schema to stdout",
        description="Print the V2 config JSON schema.",
    )

    return parser


def print_summary(result: Mapping[str, Any], config_path: Path) -> None:
    """Print human-readable backtest summary."""
    meta = result.get("meta", {})
    extra = meta.get("extra", {})
    config_meta = extra.get("config", {})

    # Try to get config info from meta or result directly
    strategy_name = config_meta.get("strategy_name", meta.get("strategy_name", "unknown"))
    symbol = config_meta.get("symbol", meta.get("symbol", "unknown"))
    start_date = config_meta.get("start_date", meta.get("start_date", ""))
    end_date = config_meta.get("end_date", meta.get("end_date", ""))

    print()
    print("Omega V2 Backtest Results")
    print("=" * 60)
    print()
    print(f"  Config:      {config_path}")
    print(f"  Strategy:    {strategy_name}")
    print(f"  Symbol:      {symbol}")
    if start_date and end_date:
        print(f"  Period:      {start_date[:10]} to {end_date[:10]}")

    # Metrics
    metrics = result.get("metrics", {})
    definitions = result.get("metric_definitions", {})
    key_metrics = extract_key_metrics(metrics)

    if key_metrics:
        print()
        print("Performance Metrics:")
        for key in REPORT_METRICS:
            if key in key_metrics:
                value = key_metrics[key]
                formatted = format_metric(key, value, definitions)
                print(f"    {formatted}")

    # Output location
    output_dir = meta.get("output_dir") or extra.get("output_dir")
    if output_dir:
        print()
        print(f"  Output:      {output_dir}")
    print()


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' command."""
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return EXIT_CONFIG_NOT_FOUND

    try:
        # Validate first if strict mode
        if args.strict:
            config = load_config(config_path)
            validate_config(config, strict=True)

        # Run backtest
        result = run_backtest(
            config_path=config_path,
            output_dir=args.output_dir,
        )

        # Check result
        if not result.get("ok", False):
            error = result.get("error", {})
            if isinstance(error, dict):
                msg = error.get("message", "Unknown error")
                category = error.get("category", "")
                if category:
                    msg = f"[{category}] {msg}"
            else:
                msg = str(error)
            print(f"Backtest failed: {msg}", file=sys.stderr)
            return EXIT_ERROR

        # Output results
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        elif not args.quiet:
            print_summary(result, config_path)

        return EXIT_SUCCESS

    except ImportError as e:
        print(
            "Error: FFI module not available. Run 'maturin develop' first.",
            file=sys.stderr,
        )
        print(f"  Details: {e}", file=sys.stderr)
        return EXIT_FFI_UNAVAILABLE

    except ValueError as e:
        print(f"Config validation error: {e}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' command."""
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return EXIT_CONFIG_NOT_FOUND

    try:
        config = load_config(config_path)
        validate_config(config, strict=args.strict)
        print(f"Config valid: {config_path}")
        return EXIT_SUCCESS

    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR


def cmd_schema(args: argparse.Namespace) -> int:
    """Print JSON schema to stdout."""
    schema_path = Path(__file__).parent / "schema" / "v2_config.json"

    if schema_path.exists():
        print(schema_path.read_text())
        return EXIT_SUCCESS

    print("Error: Schema file not found", file=sys.stderr)
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "schema":
        return cmd_schema(args)

    parser.print_help()
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
