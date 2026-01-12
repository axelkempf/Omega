"""
Centralized rating/score modules used across the backtest and analysis pipeline.

This module provides various scoring functions for evaluating trading strategy
performance, robustness, and stability. Key modules include:

- `robustness_score_1`: Parameter jitter robustness scoring
- `stability_score`: Yearly profit stability scoring
- `cost_shock_score`: Cost sensitivity analysis
- `trade_dropout_score`: Trade dropout simulation
- `stress_penalty`: Shared stress penalty computation
- `timing_jitter_score`: Timing shift sensitivity
- `data_jitter_score`: Data noise sensitivity
- `tp_sl_stress_score`: TP/SL stress testing
- `ulcer_index_score`: Ulcer Index calculation
- `p_values`: Statistical significance testing

Note: `strategy_rating.py` has been removed as part of Wave 1 migration preparation.
The `rate_strategy_performance` functionality has been moved inline to consuming
modules (e.g., `backtest_engine.optimizer.walkforward`).

Keep `__init__` intentionally lightweight: other parts of the system import
`backtest_engine.rating.*` and we want to avoid eager imports/circular dependencies.
"""

from __future__ import annotations

__all__: list[str] = []
