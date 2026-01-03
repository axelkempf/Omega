"""
Centralized rating/score modules used across the backtest and analysis pipeline.

Keep `__init__` intentionally lightweight: other parts of the system import
`backtest_engine.rating.*` (e.g. `strategy_rating`) and we want to avoid eager
imports/circular dependencies here.
"""

from __future__ import annotations

__all__: list[str] = []
