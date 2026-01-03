# hf_engine/core/execution/sl_tp_utils.py

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class SlTpLevels:
    sl: float
    tp: float


class SlTpError(ValueError):
    pass


# Small tolerance to avoid floating-point edge cases (prices equal to entry)
EPSILON: float = 1e-6


def ensure_abs_levels(
    entry: float,
    sl: float,
    tp: float,
    direction: Literal["buy", "sell"],
) -> SlTpLevels:
    """
    Enforces engine-wide semantics:
      - sl and tp are ABSOLUTE price levels.
      - direction in {"buy","sell"}.
    Validates minimum distance to entry (> EPSILON) and directional logic.
    """
    dir_l = str(direction or "").lower()
    if dir_l not in ("buy", "sell"):
        raise SlTpError("direction must be 'buy' or 'sell'")

    # Validate numerics and finiteness
    for name, v in (("entry", entry), ("sl", sl), ("tp", tp)):
        try:
            fv = float(v)
        except Exception as _:
            raise SlTpError(f"invalid price for {name}")
        if not isfinite(fv):
            raise SlTpError(f"invalid price for {name}")

    # Minimum distance from entry
    if abs(entry - sl) <= EPSILON or abs(entry - tp) <= EPSILON:
        raise SlTpError("SL/TP must not be equal to Entry")

    # Directional logic with strict inequalities buffered by EPSILON
    if dir_l == "buy":
        if not ((sl + EPSILON) < entry < (tp - EPSILON)):
            raise SlTpError("For BUY: SL < Entry < TP")
    else:  # sell
        if not ((tp + EPSILON) < entry < (sl - EPSILON)):
            raise SlTpError("For SELL: TP < Entry < SL")

    return SlTpLevels(sl=float(sl), tp=float(tp))


def distance_to_sl(entry: float, sl: float, precision: int = 6) -> float:
    """Absolute price distance from Entry to SL (positive)."""
    return round(abs(float(entry) - float(sl)), precision)
