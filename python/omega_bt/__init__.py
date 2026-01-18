"""Omega V2 Backtest Engine - Python bindings for Rust core."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from omega_bt._native import run_backtest

__all__ = ["run_backtest"]
