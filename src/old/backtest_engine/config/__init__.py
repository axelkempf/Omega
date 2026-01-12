"""Backtest-Engine Konfigurationsmodelle.

Dieses Package kapselt (Pydantic-)Modelle für JSON/YAML Configs, damit
Validierung und Normalisierung zentral erfolgt.
"""

from __future__ import annotations

from .models import BacktestConfig, StrategyConfig

__all__ = ["BacktestConfig", "StrategyConfig"]
