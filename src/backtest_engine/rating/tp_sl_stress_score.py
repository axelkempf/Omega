from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from hf_engine.infra.config.paths import PARQUET_DIR

PrimaryCandleArrays = Dict[str, np.ndarray]


def align_primary_candles(
    bid_df: pd.DataFrame, ask_df: pd.DataFrame
) -> Optional[PrimaryCandleArrays]:
    """
    Align bid/ask candle DataFrames on UTC timestamps and expose arrays for fast lookups.
    """
    if bid_df is None or ask_df is None:
        return None
    try:
        b = bid_df.copy()
        a = ask_df.copy()
        if "UTC time" not in b.columns or "UTC time" not in a.columns:
            return None
        b["UTC time"] = pd.to_datetime(b["UTC time"], utc=True, errors="coerce")
        a["UTC time"] = pd.to_datetime(a["UTC time"], utc=True, errors="coerce")
        b = b.dropna(subset=["UTC time"])
        a = a.dropna(subset=["UTC time"])
        if b.empty or a.empty:
            return None
        # Keep only required columns if present; tolerate extras.
        for col in ("High", "Low"):
            if col not in b.columns or col not in a.columns:
                return None
        b = b[["UTC time", "High", "Low"]]
        a = a[["UTC time", "High", "Low"]]
        df = b.merge(
            a,
            on="UTC time",
            how="inner",
            suffixes=("_bid", "_ask"),
        )
        if df.empty:
            return None
        df = df.sort_values("UTC time").reset_index(drop=True)
        times = pd.to_datetime(df["UTC time"], utc=True, errors="coerce")
        times = times.dropna()
        if times.empty:
            return None
        df = df.loc[times.index]
        times_ns = times.astype("int64").to_numpy()
        bid_high = pd.to_numeric(df["High_bid"], errors="coerce").to_numpy(dtype=float)
        bid_low = pd.to_numeric(df["Low_bid"], errors="coerce").to_numpy(dtype=float)
        ask_high = pd.to_numeric(df["High_ask"], errors="coerce").to_numpy(dtype=float)
        ask_low = pd.to_numeric(df["Low_ask"], errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None

    if times_ns.size == 0:
        return None

    return {
        "times_ns": times_ns,
        "bid_high": bid_high,
        "bid_low": bid_low,
        "ask_high": ask_high,
        "ask_low": ask_low,
    }


def load_primary_candle_arrays_from_parquet(
    symbol: str, timeframe: str, parquet_dir: Optional[Path] = None
) -> Optional[PrimaryCandleArrays]:
    """
    Load primary timeframe bid/ask candles from PARQUET_DIR and align them for scoring.
    """
    if not symbol or not timeframe:
        return None
    base_dir = (parquet_dir or PARQUET_DIR) / str(symbol)

    # Robust path resolution: prefer BID/ASK, fallback to bid/ask
    def _find_parquet(base: Path, sym: str, tf: str, side: str) -> Optional[Path]:
        upper_path = base / f"{sym}_{tf}_{side.upper()}.parquet"
        if upper_path.exists():
            return upper_path
        lower_path = base / f"{sym}_{tf}_{side.lower()}.parquet"
        if lower_path.exists():
            return lower_path
        return None

    bid_path = _find_parquet(base_dir, symbol, timeframe, "bid")
    ask_path = _find_parquet(base_dir, symbol, timeframe, "ask")
    if not bid_path or not ask_path:
        return None
    try:
        bid_df = pd.read_parquet(bid_path)
        ask_df = pd.read_parquet(ask_path)
    except Exception:
        return None
    return align_primary_candles(bid_df, ask_df)


def compute_tp_sl_stress_score(
    trades_df: Optional[pd.DataFrame],
    arrays: Optional[PrimaryCandleArrays],
    *,
    debug: bool | None = None,
) -> float:
    """
    Evaluate TP/SL robustness per trade. Return 1.0 when data is missing.
    """
    if debug is None:
        debug = str(os.getenv("TP_SL_STRESS_DEBUG", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _dbg(msg: str) -> None:
        if debug:
            print(f"[tp_sl-stress] {msg}", flush=True)

    if trades_df is None:
        _dbg("no trades_df provided -> returning 1.0")
        return 1.0
    try:
        if trades_df.empty:
            _dbg("trades_df empty -> returning 1.0")
            return 1.0
    except Exception:
        _dbg("trades_df check failed -> returning 1.0")
        return 1.0
    if not arrays:
        _dbg("primary arrays missing -> returning 1.0")
        return 1.0

    times_ns = arrays.get("times_ns")
    bid_high = arrays.get("bid_high")
    bid_low = arrays.get("bid_low")
    ask_high = arrays.get("ask_high")
    ask_low = arrays.get("ask_low")
    if (
        times_ns is None
        or bid_high is None
        or bid_low is None
        or ask_high is None
        or ask_low is None
    ):
        return 1.0
    n_candles = int(times_ns.size)
    if n_candles <= 0:
        _dbg("no candles in primary arrays -> returning 1.0")
        return 1.0

    _dbg(f"starting loop: trades={len(trades_df)} n_candles={n_candles}")
    skip_counts = defaultdict(int)
    logged_skips: Set[str] = set()

    def _ts_to_ns(val: Any) -> Optional[int]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            ts = (
                val
                if isinstance(val, pd.Timestamp)
                else pd.to_datetime(val, utc=True, errors="coerce")
            )
        except Exception:
            return None
        if ts is pd.NaT:
            return None
        return int(ts.value)

    def _spread_from_meta(meta: Any) -> Optional[float]:
        if not isinstance(meta, dict):
            return None
        prices = meta.get("prices") or {}
        if isinstance(prices, dict):
            sp = prices.get("spread")
            if sp is not None:
                try:
                    return float(sp)
                except Exception:
                    pass
            if "ask_close" in prices and "bid_close" in prices:
                try:
                    return float(prices["ask_close"]) - float(prices["bid_close"])
                except Exception:
                    pass
        return None

    scores: list[float] = []

    for idx, trade in enumerate(trades_df.itertuples(index=False)):
        last_step = f"trade[{idx}] init"
        try:
            last_step = f"trade[{idx}] reason check"
            reason = str(getattr(trade, "reason", "") or "").strip().lower()
            if reason != "take_profit":
                skip_counts["not_tp_exit"] += 1
                continue

            direction = str(getattr(trade, "direction", "") or "").strip().lower()
            last_step = f"trade[{idx}] direction check"
            if direction not in ("long", "short"):
                skip_counts["invalid_direction"] += 1
                continue

            tp_val = getattr(trade, "take_profit", None)
            sl_val = getattr(trade, "stop_loss", None)
            if sl_val is None:
                sl_val = getattr(trade, "initial_stop_loss", None)
            try:
                tp = float(tp_val)
                sl = float(sl_val)
            except Exception:
                skip_counts["invalid_tp_sl"] += 1
                continue
            if not (np.isfinite(tp) and np.isfinite(sl)):
                skip_counts["non_finite_tp_sl"] += 1
                continue
            spread = _spread_from_meta(getattr(trade, "meta", None))
            if spread is None or not np.isfinite(spread) or spread <= 0.0:
                skip_counts["invalid_spread"] += 1
                continue

            entry_ns = _ts_to_ns(getattr(trade, "entry_time", None))
            exit_ns = _ts_to_ns(getattr(trade, "exit_time", None))
            if entry_ns is None or exit_ns is None:
                skip_counts["invalid_times"] += 1
                continue

            start_idx = int(np.searchsorted(times_ns, entry_ns, side="left"))
            if start_idx >= n_candles:
                skip_counts["start_idx_oob"] += 1
                continue

            orig_exit_idx = int(np.searchsorted(times_ns, exit_ns, side="right") - 1)
            if orig_exit_idx < start_idx:
                orig_exit_idx = start_idx

            if direction == "long":
                tp_var = tp + spread
                sl_var = sl + spread
                hi = bid_high
                lo = bid_low
            else:
                tp_var = tp - spread
                sl_var = sl - spread
                hi = ask_high
                lo = ask_low

            # Ensure we never count the entry candle itself as a TP/SL hit.
            # Start checking from the candle AFTER the entry bar.
            check_start = start_idx + 1
            if check_start >= hi.size or check_start >= lo.size:
                # No future candles to check after entry
                skip_counts["hi_lo_oob"] += 1
                continue

            hi_seg = hi[check_start:]
            lo_seg = lo[check_start:]
            if hi_seg.size == 0 or lo_seg.size == 0:
                skip_counts["empty_segments"] += 1
                continue

            # For debug, report the first checked absolute index
            first_checked_idx = check_start

            if direction == "long":
                tp_hits = np.nonzero(hi_seg >= tp_var)[0]
                sl_hits = np.nonzero(lo_seg <= sl_var)[0]
            else:
                tp_hits = np.nonzero(lo_seg <= tp_var)[0]
                sl_hits = np.nonzero(hi_seg >= sl_var)[0]

            tp_first = int(tp_hits[0]) if tp_hits.size else None
            sl_first = int(sl_hits[0]) if sl_hits.size else None

            _dbg(f"--- Trade #{idx} ({direction.upper()}) ---")
            _dbg(f"  TP={tp:.5f}, SL={sl:.5f}, Spread={spread:.5f}")
            _dbg(f"  TP_var={tp_var:.5f}, SL_var={sl_var:.5f}")
            _dbg(
                f"  Entry_idx={start_idx}, First_checked_idx={first_checked_idx}, Orig_exit_idx={orig_exit_idx}"
            )
            # tp_first/sl_first are relative to hi_seg/lo_seg which start at first_checked_idx
            _dbg(
                f"  TP_first_hit={tp_first}, SL_first_hit={sl_first} (relative to first_checked_idx)"
            )

            if tp_first is None and sl_first is None:
                penalty = 1.0
                _dbg(f"  → No TP/SL hit found → penalty=1.0 → score=0.0")
            elif tp_first is not None and (sl_first is None or tp_first < sl_first):
                tp_idx = start_idx + tp_first
                new_exit_idx = tp_idx
                sl_same_bar = (
                    sl_first is not None and (start_idx + sl_first) == new_exit_idx
                )
                _dbg(f"  TP hit first at bar {tp_first} (abs_idx={tp_idx})")
                if sl_same_bar:
                    penalty = 1.0
                    _dbg(f"  → SL also hit on same bar → penalty=1.0 → score=0.0")
                else:
                    if new_exit_idx <= orig_exit_idx:
                        penalty = 0.0
                        _dbg(f"  → TP at or before orig exit → penalty=0.0 → score=1.0")
                    else:
                        delay_bars = max(0, int(new_exit_idx - orig_exit_idx))
                        penalty = min(0.1 * float(delay_bars), 0.5)
                        _dbg(
                            f"  → TP delayed by {delay_bars} bars → penalty={penalty:.3f} → score={1.0-penalty:.3f}"
                        )
            else:
                penalty = 1.0
                _dbg(f"  SL hit first or equal → penalty=1.0 → score=0.0")

            penalty = float(max(0.0, min(1.0, penalty)))
            score = 1.0 - penalty
            scores.append(score)
            _dbg(f"  Final score for trade #{idx}: {score:.4f}")
        except Exception as exc:
            _dbg(
                f"trade[{idx}] exception at {last_step} -> {type(exc).__name__}: {exc}"
            )
            skip_counts["exception"] += 1
            continue

    if not scores:
        _dbg("no valid TP/SL scores -> returning 1.0")
        if skip_counts:
            skip_summary = ", ".join(f"{k}={v}" for k, v in sorted(skip_counts.items()))
            _dbg(f"skip breakdown: {skip_summary}")
        return 1.0
    try:
        mean_score = float(np.mean(scores))
        _dbg(f"computed mean score={mean_score} from {len(scores)} valid entries")
        return mean_score
    except Exception:
        _dbg("mean calculation failed -> returning 1.0")
        return 1.0
