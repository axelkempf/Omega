# runner.py
import os

# Begrenze BLAS/NumPy Threads frühzeitig (schadet nie, hilft bei Parallel-Trials)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from dateutil import tz

from backtest_engine.core.event_engine import CrossSymbolEventEngine, EventEngine
from backtest_engine.core.execution_simulator import ExecutionSimulator, SymbolSpec
from backtest_engine.core.multi_strategy_controller import (
    MultiStrategyController,
    StrategyEnvironment,
)
from backtest_engine.core.multi_tick_controller import MultiTickController
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.core.tick_event_engine import TickEventEngine
from backtest_engine.data.data_handler import CSVDataHandler
from backtest_engine.data.tick_data_handler import TickDataHandler
from backtest_engine.report.result_saver import _json_default, save_backtest_result
from backtest_engine.sizing.commission import CommissionModel
from backtest_engine.sizing.lot_sizer import LotSizer
from backtest_engine.sizing.rate_provider import (
    CompositeRateProvider,
    StaticRateProvider,
    TimeSeriesRateProvider,
)
from backtest_engine.sizing.symbol_specs_registry import SymbolSpec as CentralSpec
from backtest_engine.sizing.symbol_specs_registry import (
    SymbolSpecsRegistry as CentralRegistry,
)
from backtest_engine.strategy.session_filter import (
    AnchoredSessionFilterWrapper,
    UniversalSessionFilterWrapper,
)
from backtest_engine.strategy.session_time_utils import (
    generate_anchored_windows,
    generate_anchored_windows_combined,
)
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper
from backtest_engine.strategy.validators import validate_strategy_config
from configs.backtest._config_validator import validate_config
from hf_engine.infra.config.paths import BACKTEST_RESULTS_DIR

# Optional: YAML für Symbol-Spezifikationen laden


def _maybe_seed_deterministic_rng(config: dict) -> None:
    """
    Optional determinism hook for backtests.

    If `execution.random_seed` is set, it takes precedence.
    Otherwise, if `reporting.dev_mode` is enabled and `reporting.dev_seed` is set,
    we use that seed.

    This primarily affects backtest slippage randomness (Python's `random`) and any
    legacy `np.random.*` calls.
    """
    seed: Optional[int] = None
    try:
        exec_cfg = config.get("execution", {}) or {}
        if isinstance(exec_cfg, dict) and exec_cfg.get("random_seed", None) is not None:
            seed = int(exec_cfg.get("random_seed"))
        else:
            rep_cfg = config.get("reporting", {}) or {}
            if isinstance(rep_cfg, dict) and bool(rep_cfg.get("dev_mode", False)):
                if rep_cfg.get("dev_seed", None) is not None:
                    seed = int(rep_cfg.get("dev_seed"))
    except Exception:
        seed = None

    if seed is None:
        return

    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        # numpy expects 0 <= seed < 2**32
        np.random.seed(int(seed) % (2**32))
    except Exception:
        pass


try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# --- Windows/Console: sichere UTF-8-Ausgabe (Emojis/Unicode) ----------------------
def _ensure_utf8_stdout():
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_ensure_utf8_stdout()


def _first_strategy_parameters(config: dict) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    strategy = config.get("strategy")
    if isinstance(strategy, dict):
        params = strategy.get("parameters")
        if isinstance(params, dict):
            return params
    strategies = config.get("strategies")
    if isinstance(strategies, list):
        for strat in strategies:
            if not isinstance(strat, dict):
                continue
            params = strat.get("parameters")
            if isinstance(params, dict):
                return params
    return {}


def _resolve_robust_setting(
    config: dict,
    keys: Tuple[str, ...],
    default: Any,
    *,
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    rep = config.get("reporting", {}) or {}
    sources: List[Dict[str, Any]] = []
    if isinstance(rep, dict):
        sources.append(rep)
    if isinstance(config, dict):
        sources.append(config)
    if parameters is not None and isinstance(parameters, dict):
        sources.append(parameters)
    else:
        params = _first_strategy_parameters(config)
        if params:
            sources.append(params)
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _ts_head_tail(seq, k=3):
    if not seq:
        return []
    seq = list(seq)
    head = seq[: min(k, len(seq))]
    tail = seq[-min(k, len(seq)) :]
    return head + (["..."] if len(seq) > 2 * k else []) + tail


def _print_tf_diag(tf_name, bid_ts, ask_ts):
    bid_set, ask_set = set(bid_ts), set(ask_ts)
    inter = bid_set & ask_set
    union = bid_set | ask_set
    dup_bid = len(bid_ts) - len(bid_set)
    dup_ask = len(ask_ts) - len(ask_set)
    # print(f"🔎 TF={tf_name}: bid={len(bid_ts)} ask={len(ask_ts)} inter={len(inter)} union={len(union)} overlap={len(inter)/(len(union) or 1):.3f}")
    if dup_bid or dup_ask:
        print(f"   ⚠️ Duplikate: bid={dup_bid}, ask={dup_ask}")
    # only_bid = sorted(list(bid_set - ask_set))
    # only_ask = sorted(list(ask_set - bid_set))
    # if only_bid:
    #     print(f"   ↪️ Nur in BID (Sample): {_ts_head_tail(only_bid, 2)}")
    # if only_ask:
    #     print(f"   ↩️ Nur in ASK (Sample): {_ts_head_tail(only_ask, 2)}")


class _BacktestTimer:
    """Lightweight timer for coarse backtest stage durations."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = self._start
        self._durations: OrderedDict[str, float] = OrderedDict()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._durations[name] = round(now - self._last, 6)
        self._last = now

    def add(self, name: str, delta: float) -> None:
        if delta < 0:
            return
        self._durations[name] = round(self._durations.get(name, 0.0) + delta, 6)
        self._last = time.perf_counter()

    def summary(self) -> Dict[str, float]:
        total = round(time.perf_counter() - self._start, 6)
        return {**self._durations, "total": total}

    def print_summary(self, prefix: str = "⏱️ Backtest Timing") -> None:
        summary = self.summary()
        parts = [f"{k}={v:.2f}s" for k, v in summary.items()]
        print(f"{prefix}: " + " | ".join(parts))


def diagnose_alignment(symbol_map: dict, primary_tf: str):
    # print("\n🧪 Timestamp-Diagnostik (Bid/Ask):")
    for symbol, tf_map in symbol_map.items():
        # print(f"— Symbol: {symbol}")
        for tf, sides in tf_map.items():
            bid_ts = [c.timestamp for c in sides["bid"]]
            ask_ts = [c.timestamp for c in sides["ask"]]
            # monotone? (nur Heuristik, echte Sortierung in Schritt 3)
            if bid_ts and any(
                bid_ts[i] > bid_ts[i + 1] for i in range(len(bid_ts) - 1)
            ):
                print(f"   ⚠️ BID nicht sortiert in TF={tf}")
            if ask_ts and any(
                ask_ts[i] > ask_ts[i + 1] for i in range(len(ask_ts) - 1)
            ):
                print(f"   ⚠️ ASK nicht sortiert in TF={tf}")
            _print_tf_diag(tf, bid_ts, ask_ts)
    # print("🧪 Ende Diagnostik\n")


def validate_strict_alignment(
    bid_seq, ask_seq, tf_name: str, max_missing_ratio: float = 0.0
):
    n_bid, n_ask = len(bid_seq), len(ask_seq)
    inter_len = len(
        set(c.timestamp for c in bid_seq) & set(c.timestamp for c in ask_seq)
    )
    union_len = len(
        set(c.timestamp for c in bid_seq) | set(c.timestamp for c in ask_seq)
    )
    overlap = inter_len / (union_len or 1)
    missing_ratio = 1.0 - overlap
    if missing_ratio > max_missing_ratio:
        sample_bid_only = sorted(
            list(set(c.timestamp for c in bid_seq) - set(c.timestamp for c in ask_seq))
        )[:5]
        sample_ask_only = sorted(
            list(set(c.timestamp for c in ask_seq) - set(c.timestamp for c in bid_seq))
        )[:5]
        raise ValueError(
            f"❌ Striktes Alignment fehlgeschlagen in TF={tf_name}: "
            f"overlap={overlap:.3f} (max_missing_ratio={max_missing_ratio:.3f})\n"
            f"   BID nur: {sample_bid_only}\n"
            f"   ASK nur: {sample_ask_only}"
        )


def prepare_time_window(config: dict) -> Tuple[datetime, datetime, datetime, int]:
    """
    Calculate the time window for data loading, including warmup period.

    Args:
        config (dict): Backtest configuration.

    Returns:
        Tuple[start_dt, end_dt, extended_start, warmup_bars]
    """
    tf_config = config.get("timeframes", {})
    all_tfs = [tf_config.get("primary", "M1")] + tf_config.get("additional", [])
    warmup_bars = config.get("warmup_bars", 500)

    def tf_minutes(tf: str) -> int:
        if tf.startswith("M"):
            return int(tf[1:])
        elif tf.startswith("H"):
            return int(tf[1:]) * 60
        elif tf.startswith("D"):
            return int(tf[1:]) * 60 * 24
        else:
            raise ValueError(f"Unbekannter TF: {tf}")

    start_dt = datetime.strptime(config["start_date"], "%Y-%m-%d").replace(
        tzinfo=tz.UTC
    )
    end_dt = datetime.strptime(config["end_date"], "%Y-%m-%d").replace(tzinfo=tz.UTC)

    warmup_deltas = [
        timedelta(minutes=warmup_bars * tf_minutes(tf)) for tf in all_tfs if tf
    ]
    max_warmup_delta = max(warmup_deltas) if warmup_deltas else timedelta(0)
    extended_start = start_dt - max_warmup_delta

    return start_dt, end_dt, extended_start, warmup_bars


def build_candle_lookup(
    symbol_map: Dict[str, Any], primary_tf: str
) -> Tuple[Dict, List[set]]:
    """
    Build fast lookup structures for candles by symbol and timestamp.

    Args:
        symbol_map (dict): Symbol/timeframe structure.
        primary_tf (str): Primary timeframe.

    Returns:
        Tuple: (candle_lookups, all_timestamp_sets)
    """
    candle_lookups = defaultdict(lambda: defaultdict(dict))
    all_timestamp_sets = []

    for symbol, tfs in symbol_map.items():
        tf_data = tfs[primary_tf]
        for typ in ["bid", "ask"]:
            candles = tf_data[typ]
            lookup = {c.timestamp: c for c in candles}
            candle_lookups[symbol][typ] = lookup
            all_timestamp_sets.append(set(lookup.keys()))
    return candle_lookups, all_timestamp_sets


def unwrap_strategy(strategy: Any) -> Any:
    """
    Recursively unwraps strategy from wrappers, except StrategyWrapper.

    Args:
        strategy: Possibly wrapped strategy object.

    Returns:
        Unwrapped strategy.
    """
    while hasattr(strategy, "strategy") and not isinstance(strategy, StrategyWrapper):
        strategy = strategy.strategy
    return strategy


def _load_execution_costs(config: dict) -> dict:
    """
    Lädt zentrale Ausführungskosten:
      - defaults:   globale Vorgaben
      - per_symbol: symbol-spezifische Overrides
    Datei: config/execution_costs.yaml (Pfad via config['execution_costs_file'] überschreibbar)
    """
    path = config.get("execution_costs_file", "configs/execution_costs.yaml")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return (
                yaml.safe_load(f)
                if (yaml and path.lower().endswith((".yml", ".yaml")))
                else json.load(f)
            )
    except Exception as e:
        print(f"⚠️ Konnte Execution-Costs nicht laden ({path}): {e}")
        return {}


def _load_symbol_specs(config: dict) -> Dict[str, SymbolSpec]:
    """
    Zentraler Loader für Symbol-Spezifikationen.
    Erwartet standardmäßig 'config/symbol_specs.yaml' (Pfad via config['symbol_specs_file'] überschreibbar).
    YAML/JSON-Struktur:
      SYMBOL:
        contract_size: float
        tick_size: float
        tick_value: float
        volume_min: float
        volume_step: float
        volume_max: float
        base_currency: str (opt)
        quote_currency: str (opt)
        profit_currency: str (opt)
        pip_size: float (opt)
    """
    path = config.get("symbol_specs_file", "configs/symbol_specs.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = (
                yaml.safe_load(f)
                if (yaml and path.lower().endswith((".yml", ".yaml")))
                else json.load(f)
            )
        specs: Dict[str, SymbolSpec] = {}
        for sym, s in (raw or {}).items():
            specs[sym] = SymbolSpec(
                symbol=sym,
                contract_size=float(s["contract_size"]),
                tick_size=float(s["tick_size"]),
                tick_value=float(s["tick_value"]),
                volume_min=float(s.get("volume_min", 0.01)),
                volume_step=float(s.get("volume_step", 0.01)),
                volume_max=float(s.get("volume_max", 100.0)),
                base_currency=s.get("base_currency"),
                quote_currency=s.get("quote_currency"),
                profit_currency=s.get("profit_currency"),
                pip_size=float(s["pip_size"]) if "pip_size" in s else None,
            )
        return specs
    except Exception as e:
        print(f"⚠️ Konnte Symbol-Spezifikationen nicht laden ({path}): {e}")
        return {}


def _build_central_registry(raw_specs: Dict[str, SymbolSpec]) -> CentralRegistry:
    """
    Adapter: konvertiert die lokal verwendete SymbolSpec (execution_simulator.py)
    in die zentrale Registry-Klasse (symbol_specs_registry.py) für Sizing/Fees.
    """
    converted = {}
    for sym, s in (raw_specs or {}).items():
        converted[sym] = CentralSpec(
            symbol=s.symbol,
            contract_size=s.contract_size,
            tick_size=s.tick_size,
            tick_value=s.tick_value,
            volume_min=s.volume_min,
            volume_step=s.volume_step,
            volume_max=s.volume_max,
            base_currency=s.base_currency or "",
            quote_currency=s.quote_currency or "",
            profit_currency=s.profit_currency or "",
            pip_size=s.pip_size or (0.01 if "JPY" in sym else 0.0001),
        )
    return CentralRegistry(converted)


def get_common_timestamps(
    all_timestamp_sets: List[set], method: str = "intersection"
) -> List:
    """
    Returns sorted list of common timestamps.

    Args:
        all_timestamp_sets (list): List of sets of timestamps.
        method (str): "intersection" or "union".

    Returns:
        List: Sorted timestamps.
    """
    if method == "intersection":
        return sorted(set.intersection(*all_timestamp_sets))
    elif method == "union":
        return sorted(set.union(*all_timestamp_sets))
    else:
        raise ValueError("Methode muss 'intersection' oder 'union' sein")


def _sequence_signature(
    seq: List[Any], sample: int = 2
) -> Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]:
    ts = [c.timestamp for c in seq]
    length = len(ts)
    head = tuple(ts[:sample])
    tail = tuple(ts[-sample:]) if length >= sample else tuple(ts)
    return length, head, tail


def _collect_tf_signatures(
    symbol_map: dict,
) -> Dict[str, Dict[str, Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]]]:
    sym = next(iter(symbol_map), None)
    if sym is None:
        return {}
    signatures: Dict[str, Dict[str, Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]]] = {}
    for tf, sides in symbol_map[sym].items():
        signatures[tf] = {
            side: _sequence_signature(seq)
            for side, seq in sides.items()
            if isinstance(seq, list)
        }
    return signatures


def _serialize_signature(sig: Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]) -> str:
    length, head, tail = sig
    fmt = lambda t: t.isoformat() if hasattr(t, "isoformat") else str(t)
    head_str = ",".join(fmt(t) for t in head)
    tail_str = ",".join(fmt(t) for t in tail)
    return f"{length}|{head_str}|{tail_str}"


@dataclass(frozen=True)
class AlignmentPlan:
    common_ts: List[datetime]
    primary_bid_idx: List[int]
    primary_ask_idx: List[int]
    per_tf: Dict[str, Dict[str, List[int]]]
    signatures: Dict[str, Dict[str, Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]]]


def _build_alignment_plan(
    symbol_map: dict, primary_tf: str, config: dict
) -> AlignmentPlan:
    """
    Baut einen Alignment-Plan auf Basis der aktuellen Candle-Sequenzen.
    Rückgabe enthält ausschließlich Indizes/Timestamps – keine Candle-Objekte.
    """
    ta_cfg = config.get("timestamp_alignment", {}) or {}
    entry_mode = str(ta_cfg.get("entry_timestamp_mode", "open")).lower()
    use_prev_completed = bool(ta_cfg.get("use_previous_completed_higher_tf", True))

    additional_mode = ta_cfg.get("additional_tfs_mode", "carry_forward")
    stale_limit = int(ta_cfg.get("stale_bar_limit_bars", 0))
    max_missing = float(ta_cfg.get("max_missing_ratio", 0.0))
    ts_mode_map = ta_cfg.get("higher_tf_timestamps_mode", {}) or {}

    if additional_mode not in {"carry_forward", "strict"}:
        raise ValueError(
            "timestamp_alignment.additional_tfs_mode muss 'carry_forward' oder 'strict' sein"
        )

    def _tf_minutes(tf: str) -> int:
        tf = tf.upper()
        if tf.startswith("M"):
            return int(tf[1:])
        if tf.startswith("H"):
            return int(tf[1:]) * 60
        if tf.startswith("D"):
            return int(tf[1:]) * 1440
        raise ValueError(f"Unbekannter TF: {tf}")

    primary_minutes = _tf_minutes(primary_tf)

    def _effective_stale_limit(tf: str) -> int:
        if stale_limit > 0:
            return stale_limit
        ratio = max(1, _tf_minutes(tf) // max(1, primary_minutes))
        return max(1, 2 * ratio)

    primary_delta = timedelta(minutes=primary_minutes)

    def _bar_end_time(open_ts, tf_name: str, tf_delta: timedelta):
        mode = str(ts_mode_map.get(tf_name, "open")).lower()
        return open_ts if mode == "close" else (open_ts + tf_delta)

    def _cf_completed_indices(
        lookup: dict,
        common_ts: list,
        tf_name: str,
        tf_delta: timedelta,
        eff_stale: int,
        decision_shift: timedelta,
    ) -> list:
        if not use_prev_completed:
            out, last, last_seen_idx = [], None, -(10**9)
            for i, ts in enumerate(common_ts):
                if ts in lookup:
                    last = lookup[ts]
                    last_seen_idx = i
                out.append(last if (i - last_seen_idx) <= eff_stale else -1)
            return out

        higher_opens_sorted = sorted(lookup.keys())
        k = 0
        out, last, last_seen_idx = [], None, -(10**9)
        n = len(higher_opens_sorted)

        for i, ts in enumerate(common_ts):
            eff_ts = ts + decision_shift
            while k < n:
                open_t = higher_opens_sorted[k]
                if _bar_end_time(open_t, tf_name, tf_delta) <= eff_ts:
                    last = lookup[open_t]
                    last_seen_idx = i
                    k += 1
                else:
                    break
            out.append(last if (i - last_seen_idx) <= eff_stale else -1)
        return out

    candle_lookups, all_ts_sets = build_candle_lookup(symbol_map, primary_tf)
    common_ts = get_common_timestamps(all_ts_sets, method="intersection")
    if not common_ts:
        raise ValueError("Kein gemeinsamer Timestamp zwischen Bid/Ask im Zeitraum.")

    sym = list(symbol_map.keys())[0]
    primary_bid_seq = symbol_map[sym][primary_tf]["bid"]
    primary_ask_seq = symbol_map[sym][primary_tf]["ask"]
    primary_bid_idx_lookup = {c.timestamp: idx for idx, c in enumerate(primary_bid_seq)}
    primary_ask_idx_lookup = {c.timestamp: idx for idx, c in enumerate(primary_ask_seq)}

    primary_bid_idx = [primary_bid_idx_lookup[ts] for ts in common_ts]
    primary_ask_idx = [primary_ask_idx_lookup[ts] for ts in common_ts]

    bid_aligned = [primary_bid_seq[i] for i in primary_bid_idx]
    ask_aligned = [primary_ask_seq[i] for i in primary_ask_idx]
    validate_strict_alignment(
        bid_aligned, ask_aligned, primary_tf, max_missing_ratio=max_missing
    )

    per_tf: Dict[str, Dict[str, List[int]]] = {}
    decision_shift = primary_delta if entry_mode == "close" else timedelta(0)
    for tf, sides in symbol_map[sym].items():
        if tf == primary_tf:
            continue

        bid_tf_lookup = {c.timestamp: idx for idx, c in enumerate(sides["bid"])}
        ask_tf_lookup = {c.timestamp: idx for idx, c in enumerate(sides["ask"])}

        eff_stale = _effective_stale_limit(tf)
        tf_delta = timedelta(minutes=_tf_minutes(tf))

        per_tf[tf] = {
            "bid": _cf_completed_indices(
                bid_tf_lookup, common_ts, tf, tf_delta, eff_stale, decision_shift
            ),
            "ask": _cf_completed_indices(
                ask_tf_lookup, common_ts, tf, tf_delta, eff_stale, decision_shift
            ),
        }

    signatures = _collect_tf_signatures(symbol_map)
    return AlignmentPlan(
        common_ts=common_ts,
        primary_bid_idx=primary_bid_idx,
        primary_ask_idx=primary_ask_idx,
        per_tf=per_tf,
        signatures=signatures,
    )


def _apply_alignment_plan(
    plan: AlignmentPlan, symbol_map: dict, primary_tf: str
) -> Tuple[List[Any], List[Any], Dict[str, Dict[str, List[Any]]]]:
    sym = next(iter(symbol_map), None)
    if sym is None:
        raise ValueError("Symbol Map ist leer – kein Alignment möglich.")

    tf_map = symbol_map[sym]
    current_signatures = _collect_tf_signatures(symbol_map)

    for tf, sides in plan.signatures.items():
        if tf not in current_signatures:
            raise ValueError(f"AlignmentPlan invalid: TF {tf} nicht im aktuellen Map.")
        for side, expected_sig in sides.items():
            if side not in current_signatures[tf]:
                raise ValueError(
                    f"AlignmentPlan invalid: Side {side} fehlt in TF {tf}."
                )
            if current_signatures[tf][side] != expected_sig:
                raise ValueError(
                    f"AlignmentPlan Signatur-Mismatch für TF={tf}, side={side}."
                )

    primary_bid_seq = tf_map[primary_tf]["bid"]
    primary_ask_seq = tf_map[primary_tf]["ask"]
    expected_len = len(plan.common_ts)

    if (
        len(plan.primary_bid_idx) != expected_len
        or len(plan.primary_ask_idx) != expected_len
    ):
        raise ValueError("AlignmentPlan Länge passt nicht zu Common-Timestamps.")

    def _ensure_range(idx_list: List[int], seq_len: int, label: str) -> None:
        if any(idx < 0 or idx >= seq_len for idx in idx_list):
            raise ValueError(f"AlignmentPlan {label} außerhalb Range (len={seq_len}).")

    _ensure_range(plan.primary_bid_idx, len(primary_bid_seq), "primary_bid_idx")
    _ensure_range(plan.primary_ask_idx, len(primary_ask_seq), "primary_ask_idx")

    bid_aligned = [primary_bid_seq[idx] for idx in plan.primary_bid_idx]
    ask_aligned = [primary_ask_seq[idx] for idx in plan.primary_ask_idx]

    multi_candle_data_aligned: Dict[str, Dict[str, List[Any]]] = {
        primary_tf: {"bid": bid_aligned, "ask": ask_aligned}
    }

    def _map_indices(seq: List[Any], idx_list: List[int], label: str) -> List[Any]:
        out: List[Any] = []
        for idx in idx_list:
            if idx < 0:
                out.append(None)
            elif idx < len(seq):
                out.append(seq[idx])
            else:
                raise ValueError(
                    f"AlignmentPlan Index {idx} außerhalb Range für {label} (len={len(seq)})."
                )
        return out

    for tf, sides in plan.per_tf.items():
        if tf not in tf_map:
            raise ValueError(f"AlignmentPlan invalid: TF {tf} fehlt im aktuellen Map.")

        bid_seq_tf = tf_map[tf]["bid"]
        ask_seq_tf = tf_map[tf]["ask"]
        bid_idx_list = sides.get("bid", [])
        ask_idx_list = sides.get("ask", [])

        if len(bid_idx_list) != expected_len or len(ask_idx_list) != expected_len:
            raise ValueError(
                f"AlignmentPlan Länge stimmt nicht für TF {tf} mit Common-Timestamps überein."
            )

        multi_candle_data_aligned[tf] = {
            "bid": _map_indices(bid_seq_tf, bid_idx_list, f"{tf}/bid"),
            "ask": _map_indices(ask_seq_tf, ask_idx_list, f"{tf}/ask"),
        }

    return bid_aligned, ask_aligned, multi_candle_data_aligned


def _align_single_symbol_primary_and_multi(
    symbol_map: dict, primary_tf: str, config: dict
):
    """
    Kompatibilitäts-Wrapper: baut einen Alignment-Plan und wendet ihn direkt an.
    """
    plan = _build_alignment_plan(
        symbol_map=symbol_map, primary_tf=primary_tf, config=config
    )
    return _apply_alignment_plan(plan, symbol_map, primary_tf)


# === NEU: Alignment Cache =====================================================

_ALIGNMENT_CACHE: "OrderedDict[str, AlignmentPlan]" = OrderedDict()
_ALIGNMENT_CACHE_MAX = int(os.getenv("ALIGN_CACHE_MAX", "2"))


def clear_alignment_cache(keep_last: int = 0) -> None:
    """
    Clear the alignment cache to free RAM.
    keep_last: keep the most-recent N entries (LRU); 0 clears everything.
    """
    global _ALIGNMENT_CACHE
    if keep_last <= 0:
        _ALIGNMENT_CACHE.clear()
        return
    # trim to last N
    while len(_ALIGNMENT_CACHE) > keep_last:
        _ALIGNMENT_CACHE.popitem(last=False)


def _alignment_cache_key(
    symbol_map: dict, primary_tf: str, config: dict, start_dt: datetime
) -> str:
    """
    Baut einen stabilen Key für den Alignment-Plan.
    Nutzt deterministische Signaturen aller relevanten TFs (Bid/Ask).
    """
    import hashlib

    try:
        sym = next(iter(symbol_map))
        tf_signatures = _collect_tf_signatures(symbol_map)
        parts: List[str] = []
        for tf in sorted(tf_signatures):
            for side in ("bid", "ask"):
                if side in tf_signatures[tf]:
                    parts.append(
                        f"{tf}:{side}:{_serialize_signature(tf_signatures[tf][side])}"
                    )
        sig_blob = ";".join(parts)
        align_cfg = json.dumps(config.get("timestamp_alignment", {}), sort_keys=True)
        raw_key = f"{sym}|{primary_tf}|{sig_blob}|{align_cfg}|{start_dt.isoformat()}"
        return hashlib.sha1(raw_key.encode("utf-8"), usedforsecurity=False).hexdigest()
    except Exception:
        # Fallback: kein Cache
        return ""


def _get_or_build_alignment(
    symbol_map: dict, primary_tf: str, config: dict, start_dt: datetime
):
    """
    Holen oder bauen: AlignmentPlan anwenden und aligned Candle-Listen liefern.
    """
    key = _alignment_cache_key(symbol_map, primary_tf, config, start_dt)
    plan: Optional[AlignmentPlan] = None
    if key and key in _ALIGNMENT_CACHE:
        _ALIGNMENT_CACHE.move_to_end(key, last=True)
        plan = _ALIGNMENT_CACHE[key]
        try:
            return _apply_alignment_plan(plan, symbol_map, primary_tf)
        except Exception:
            _ALIGNMENT_CACHE.pop(key, None)

    plan = _build_alignment_plan(
        symbol_map=symbol_map, primary_tf=primary_tf, config=config
    )
    if key:
        _ALIGNMENT_CACHE[key] = plan
        _ALIGNMENT_CACHE.move_to_end(key, last=True)
        if _ALIGNMENT_CACHE_MAX >= 1:
            while len(_ALIGNMENT_CACHE) > _ALIGNMENT_CACHE_MAX:
                _ALIGNMENT_CACHE.popitem(last=False)
    return _apply_alignment_plan(plan, symbol_map, primary_tf)


def load_data(
    config: dict,
    mode: str,
    extended_start: datetime,
    end_dt: datetime,
    preloaded_data: Optional[dict] = None,
) -> Tuple[Dict, Any, Any, Any]:
    """
    Loads candles or tick data as needed for the backtest.

    Returns:
        symbol_map, bid, ask, tick_data
    """
    symbol_map = {}
    tick_data = None

    if mode == "tick":
        if "multi_symbols" in config:
            print("🔁 Tick-Modus mit mehreren Symbolen")
            tick_data_map = {}
            for symbol in config["multi_symbols"].keys():
                dh = TickDataHandler(symbol=symbol)
                ticks = dh.load_ticks(start_dt=extended_start, end_dt=end_dt)
                tick_data_map[symbol] = ticks
                print(f"✅ {symbol}: {len(ticks)} Ticks")
            return {s: {} for s in tick_data_map}, None, None, tick_data_map
        else:
            print("📥 Tick-Modus aktiviert")
            symbol = config["symbol"]
            dh = TickDataHandler(symbol=symbol)
            tick_data = dh.load_ticks(start_dt=extended_start, end_dt=end_dt)
            print(f"✅ Ticks geladen: {len(tick_data)} für {symbol}")
            return {symbol: {}}, None, None, tick_data

    elif "multi_symbols" in config:
        print("🔁 Multi-Symbol Candle-Modus")
        for symbol, tfs in config["multi_symbols"].items():
            print(f"🔄 Symbol: {symbol}")
            tf_data = {}
            for tf in tfs:
                normalize = config.get("timestamp_alignment", {}).get(
                    "normalize_to_timeframe", False
                )
                dh = CSVDataHandler(
                    symbol=symbol,
                    timeframe=tf,
                    preloaded_data=preloaded_data,
                    normalize_to_timeframe=normalize,
                )
                candles = dh.load_candles(start_dt=extended_start, end_dt=end_dt)
                tf_data[tf] = {"bid": candles["bid"], "ask": candles["ask"]}
            symbol_map[symbol] = tf_data

    else:
        # print("📥 Single-Symbol Candle-Modus")
        symbol = config["symbol"]
        tfs = config.get("timeframes", {"primary": "M1", "additional": []})
        tf_data = {}
        for tf in [tfs["primary"]] + tfs["additional"]:
            normalize = config.get("timestamp_alignment", {}).get(
                "normalize_to_timeframe", False
            )
            dh = CSVDataHandler(
                symbol=symbol,
                timeframe=tf,
                preloaded_data=preloaded_data,
                normalize_to_timeframe=normalize,
            )
            candles = dh.load_candles(start_dt=extended_start, end_dt=end_dt)
            tf_data[tf] = {"bid": candles["bid"], "ask": candles["ask"]}
        symbol_map[symbol] = tf_data

    if "multi_symbols" in config:
        return symbol_map, None, None, tick_data
    else:
        first_symbol = next(iter(symbol_map))
        bid = {
            tf: symbol_map[first_symbol][tf]["bid"] for tf in symbol_map[first_symbol]
        }
        ask = {
            tf: symbol_map[first_symbol][tf]["ask"] for tf in symbol_map[first_symbol]
        }
        return symbol_map, bid, ask, tick_data


def prepare_strategies(
    config: dict,
    symbol_map: dict,
    slippage_model: SlippageModel,
    fee_model: FeeModel,
    enable_logging: bool,
    symbol_specs: Optional[Dict[str, SymbolSpec]] = None,
    lot_sizer: Optional[LotSizer] = None,
    commission_model: Optional[CommissionModel] = None,
) -> List[StrategyEnvironment]:
    """
    Instantiates all strategies as configured, applies session/cooldown wrappers.

    Returns:
        List[StrategyEnvironment]
    """
    cooldown_candles = config.get("cooldown_candles", 0)
    cooldown_minutes = config.get("cooldown_minutes", 0)
    cooldown_minutes_trade = config.get("cooldown_minutes_trade", 0)
    envs = []

    def apply_wrappers(wrapper: Any, strat_conf: dict, global_conf: dict) -> Any:
        session_filter = strat_conf.get("session_filter") or global_conf.get(
            "session_filter"
        )
        if isinstance(session_filter, dict):
            wrapper = UniversalSessionFilterWrapper(
                wrapper, session_filter=session_filter
            )
        anchored_conf = strat_conf.get("anchored_session_filter") or global_conf.get(
            "anchored_session_filter"
        )
        if isinstance(anchored_conf, dict):
            if "windows" in anchored_conf:
                calendar = generate_anchored_windows_combined(
                    global_conf["start_date"],
                    global_conf["end_date"],
                    anchored_conf["windows"],
                )
            else:
                calendar = generate_anchored_windows(
                    global_conf["start_date"],
                    global_conf["end_date"],
                    anchored_conf["anchor"],
                    anchored_conf["offset_window"],
                )
            wrapper = AnchoredSessionFilterWrapper(wrapper, anchored_calendar=calendar)
        return wrapper

    if "strategies" in config:
        for strat_conf in config["strategies"]:
            errors = validate_strategy_config(
                strat_conf, mode=config.get("mode", "candle")
            )
            if errors:
                raise ValueError(
                    f"❌ Fehler in Strategie '{strat_conf.get('name', '?')}':\n"
                    + "\n".join(errors)
                )

            module_path = strat_conf["module"]
            if not module_path.startswith("strategies."):
                module_path = f"strategies.{module_path}"
            strat_module = import_module(module_path)
            strat_class = getattr(strat_module, strat_conf["class"])
            strat_instance = strat_class(**strat_conf.get("parameters", {}))
            strat_instance.symbol = strat_conf.get("symbol")
            portfolio = Portfolio(
                initial_balance=config.get("initial_balance", 10000.0)
            )
            # Optional: Backtest-Robustheitsmetriken per Config enablen
            try:
                rep = config.get("reporting", {}) or {}
                strat_params = strat_conf.get("parameters", {})
                if not isinstance(strat_params, dict):
                    strat_params = {}
                enabled_flag = bool(
                    rep.get("enable_backtest_robust_metrics", False)
                    or config.get("enable_backtest_robust_metrics", False)
                    or (strat_params.get("enable_backtest_robust_metrics", False))
                )
                portfolio.enable_backtest_robust_metrics = enabled_flag
                jitter_frac = float(
                    _resolve_robust_setting(
                        config,
                        ("jitter_frac", "robust_jitter_frac"),
                        0.05,
                        parameters=strat_params,
                    )
                )
                jitter_repeats = int(
                    _resolve_robust_setting(
                        config,
                        ("robust_jitter_repeats", "jitter_repeats"),
                        5,
                        parameters=strat_params,
                    )
                )
                portfolio.robust_backtest_options = {
                    "dropout_frac": float(rep.get("robust_dropout_frac", 0.10)),
                    "cost_shock_factor": float(
                        rep.get("robust_cost_shock_factor", 0.50)
                    ),
                    "jitter_frac": jitter_frac,
                    "jitter_repeats": jitter_repeats,
                }
            except Exception:
                pass
            entry_mode = (config.get("timestamp_alignment", {}) or {}).get(
                "entry_timestamp_mode", "open"
            )
            wrapper = StrategyWrapper(
                strat_instance,
                cooldown_candles=cooldown_candles,
                cooldown_minutes=cooldown_minutes,
                cooldown_minutes_trade=cooldown_minutes_trade,
                enable_logging=enable_logging,
                logging_mode=config.get("logging_mode", "trades_only"),
                portfolio=portfolio,
                entry_timestamp_mode=entry_mode,
            )
            portfolio.strategy_wrapper = wrapper
            wrapper = apply_wrappers(wrapper, strat_conf, config)
            env = StrategyEnvironment(
                name=strat_conf["name"], strategy=wrapper, multi_candle_data=symbol_map
            )
            env.portfolio = portfolio
            env.executor = ExecutionSimulator(
                portfolio=env.portfolio,
                risk_per_trade=config.get("risk_per_trade", 100.0),
                slippage_model=slippage_model,
                fee_model=fee_model,
                symbol_specs=symbol_specs or {},
                lot_sizer=lot_sizer,
                commission_model=commission_model,
            )
            envs.append(env)
    else:
        strat_conf = config["strategy"]
        errors = validate_strategy_config(strat_conf, mode=config.get("mode", "candle"))
        if errors:
            raise ValueError("❌ Fehler in Einzelstrategie:\n" + "\n".join(errors))
        module_path = strat_conf["module"]
        if not module_path.startswith("strategies."):
            module_path = f"strategies.{module_path}"
        strat_module = import_module(module_path)
        strat_class = getattr(strat_module, strat_conf["class"])
        global_symbol = config.get("symbol")
        parameters = strat_conf.get("parameters", {})
        if "symbol" not in parameters and global_symbol:
            from inspect import signature

            if "symbol" in signature(strat_class.__init__).parameters:
                parameters["symbol"] = global_symbol
        tf_config = config.get("timeframes", {})
        primary_tf = tf_config.get("primary", "M15")
        if "timeframe" not in parameters:
            from inspect import signature

            if "timeframe" in signature(strat_class.__init__).parameters:
                parameters["timeframe"] = primary_tf
        strat_instance = strat_class(**parameters)
        portfolio = Portfolio(initial_balance=config.get("initial_balance", 10000.0))
        # Optional: Backtest-Robustheitsmetriken per Config enablen
        try:
            rep = config.get("reporting", {}) or {}
            enabled_flag = bool(
                rep.get("enable_backtest_robust_metrics", False)
                or config.get("enable_backtest_robust_metrics", False)
                or (parameters.get("enable_backtest_robust_metrics", False))
            )
            portfolio.enable_backtest_robust_metrics = enabled_flag
            jitter_frac = float(
                _resolve_robust_setting(
                    config,
                    ("jitter_frac", "robust_jitter_frac"),
                    0.05,
                    parameters=parameters,
                )
            )
            jitter_repeats = int(
                _resolve_robust_setting(
                    config,
                    ("robust_jitter_repeats", "jitter_repeats"),
                    5,
                    parameters=parameters,
                )
            )
            portfolio.robust_backtest_options = {
                "dropout_frac": float(rep.get("robust_dropout_frac", 0.10)),
                "cost_shock_factor": float(rep.get("robust_cost_shock_factor", 0.50)),
                "jitter_frac": jitter_frac,
                "jitter_repeats": jitter_repeats,
            }
        except Exception:
            pass
        entry_mode = (config.get("timestamp_alignment", {}) or {}).get(
            "entry_timestamp_mode", "open"
        )
        wrapper = StrategyWrapper(
            strat_instance,
            cooldown_candles=cooldown_candles,
            cooldown_minutes=cooldown_minutes,
            cooldown_minutes_trade=cooldown_minutes_trade,
            enable_logging=enable_logging,
            logging_mode=config.get("logging_mode", "trades_only"),
            portfolio=portfolio,
            entry_timestamp_mode=entry_mode,
        )
        portfolio.strategy_wrapper = wrapper
        wrapper = apply_wrappers(wrapper, strat_conf, config)
        env = StrategyEnvironment(
            name=strat_conf["class"], strategy=wrapper, multi_candle_data=symbol_map
        )
        env.portfolio = portfolio
        env.executor = ExecutionSimulator(
            portfolio=env.portfolio,
            risk_per_trade=config.get("risk_per_trade", 100.0),
            slippage_model=slippage_model,
            fee_model=fee_model,
            symbol_specs=symbol_specs or {},
            lot_sizer=lot_sizer,
            commission_model=commission_model,
        )
        envs.append(env)
    return envs


def run_backtest(config: dict) -> None:
    """
    Main backtest entrypoint (CLI). Handles all modes and reporting.

    Args:
        config (dict): Backtest configuration.
    """
    print("🚀 Starte Backtest...")
    timer = _BacktestTimer()
    _maybe_seed_deterministic_rng(config)

    mode = config.get("mode", "candle")
    enable_logging = config.get("enable_entry_logging", False)
    tf_config = config.get("timeframes", {})
    primary_tf = tf_config.get("primary")
    additional_tfs = tf_config.get("additional", [])
    all_tfs = [primary_tf] + additional_tfs

    # Zeitfenster & Daten vorbereiten
    start_dt, end_dt, extended_start, warmup_bars = prepare_time_window(config)

    # Account Currency (Default EUR)
    account_ccy = config.get("account_currency", "EUR")

    # Slippage & Fee Modelle (zentral + Config-Overrides)
    exec_costs = _load_execution_costs(config)
    defaults = exec_costs.get("defaults", {})
    # Merge: zentrale Defaults → JSON-Config überschreibt
    slip_conf = {**defaults.get("slippage", {}), **config.get("slippage", {})}
    fee_conf = {**defaults.get("fees", {}), **config.get("fees", {})}

    # Optional execution multipliers (default 1.0). These are the primary knobs used by
    # robustness stress tests (e.g. cost shock) and are safe no-ops when unset.
    try:
        exec_cfg = config.get("execution", {}) or {}
        slip_mult = float(exec_cfg.get("slippage_multiplier", 1.0))
        fee_mult = float(exec_cfg.get("fee_multiplier", 1.0))
    except Exception:
        slip_mult = 1.0
        fee_mult = 1.0
    config.setdefault("execution", {})
    config["execution"]["slippage_multiplier"] = slip_mult
    config["execution"]["fee_multiplier"] = fee_mult

    slippage_model = SlippageModel(
        fixed_pips=float(slip_conf.get("fixed_pips", 0.0)) * slip_mult,
        random_pips=float(slip_conf.get("random_pips", 0.0)) * slip_mult,
    )
    fee_model = FeeModel(  # legacy (will be ignored if CommissionModel used)
        per_million=float(fee_conf.get("per_million", 0.0)) * fee_mult,
        lot_size=float(fee_conf.get("lot_size", 100_000.0)),
        min_fee=float(fee_conf.get("min_fee", 0.0)),
    )
    # Daten laden
    symbol_map, bid_candles, ask_candles, tick_data = load_data(
        config, mode, extended_start, end_dt
    )
    timer.mark("data_load")

    if config.get("timestamp_alignment", {}).get("diagnostics", True):
        diagnose_alignment(symbol_map, primary_tf)

    # Strategie/Env vorbereiten
    symbol_specs = _load_symbol_specs(config)
    central_registry = _build_central_registry(symbol_specs)

    # --- WICHTIG: Sicherstellen, dass Higher-TF Kerzen nur abgeschlossen genutzt werden ---
    ta_cfg = config.setdefault("timestamp_alignment", {})
    # Default auf True (nur abgeschlossene Higher-TF Bars werden sichtbar)
    if "use_previous_completed_higher_tf" not in ta_cfg:
        ta_cfg["use_previous_completed_higher_tf"] = True

    # ==== FX Rate Provider (historisch / composite / statisch) ====
    rates_cfg = config.get("rates", {}) or {}
    mode = (
        rates_cfg.get("mode")
        or ("static" if "rates_static" in config else "timeseries")
    ).lower()
    rates_static = {
        k.upper(): float(v) for k, v in config.get("rates_static", {}).items()
    }

    def _build_rate_provider():
        if mode == "static":
            return StaticRateProvider(
                rates_static, strict=bool(rates_cfg.get("strict", True))
            )
        # Default/Timeseries/Composite
        ts_pairs = [p.upper() for p in rates_cfg.get("pairs", [])]
        ts_tf_cfg = (rates_cfg.get("timeframe") or "").upper()
        ts_tf = primary_tf if ts_tf_cfg in ("", "AUTO") else ts_tf_cfg
        use_price = rates_cfg.get("use_price", "close")
        stale_bars = rates_cfg.get("stale_limit_bars", 2)
        strict_ts = bool(rates_cfg.get("strict", True))
        rp_ts = TimeSeriesRateProvider(
            pairs=ts_pairs,
            timeframe=ts_tf,
            start_dt=extended_start,
            end_dt=end_dt,
            use_price=use_price,
            stale_limit_bars=stale_bars,
            strict=strict_ts,
        )
        if mode == "composite":
            rp_static = StaticRateProvider(rates_static, strict=False)
            return CompositeRateProvider([rp_ts, rp_static])
        return rp_ts

    rate_provider = _build_rate_provider()
    lot_sizer = LotSizer(account_ccy, rate_provider, central_registry)
    commission_model = CommissionModel(
        account_ccy,
        rate_provider,
        exec_costs,
        central_registry,
        multiplier=fee_mult,
    )
    envs = prepare_strategies(
        config,
        symbol_map,
        slippage_model,
        fee_model,
        enable_logging,
        symbol_specs=symbol_specs,
        lot_sizer=lot_sizer,
        commission_model=commission_model,
    )
    timer.mark("strategy_prep")

    def progress(i: int, total: int) -> None:
        if i % 1000 == 0:
            print(f"⏳ Fortschritt: {i}/{total}")

    # --------- Backtest Execution --------- #

    is_cross_symbol = False
    if len(envs) == 1:
        env = envs[0]
        strat_core = unwrap_strategy(env.strategy)
        if getattr(strat_core, "cross_symbol", False) or "multi_symbols" in config:
            is_cross_symbol = True

    # Cross-Symbol Candle Engine
    if len(envs) == 1 and is_cross_symbol and mode != "tick":
        primary_tf = config["timeframes"]["primary"]
        candle_lookups, all_timestamp_sets = build_candle_lookup(symbol_map, primary_tf)
        common_timestamps = get_common_timestamps(
            all_timestamp_sets, method="intersection"
        )
        timer.mark("data_align")
        engine = CrossSymbolEventEngine(
            candle_lookups=candle_lookups,
            common_timestamps=common_timestamps,
            strategy=env.strategy,
            executor=env.executor,
            portfolio=env.portfolio,
            primary_tf=primary_tf,
            on_progress=progress,
        )
        engine.original_start_dt = start_dt
        engine.run()
        timer.mark("engine_loop")

    # Tick-Modus
    elif mode == "tick":
        timer.mark("data_align")
        if isinstance(tick_data, dict):
            controller = MultiTickController(
                envs=envs, tick_data_map=tick_data, on_progress=progress
            )
            controller.run()
        else:
            env = envs[0]
            engine = TickEventEngine(
                ticks=tick_data,
                strategy=env.strategy,
                executor=env.executor,
                portfolio=env.portfolio,
                multi_candle_data={
                    primary_tf: {"bid": bid_candles, "ask": ask_candles}
                },
                on_progress=progress,
                symbol=config.get("symbol", ""),
            )
            engine.original_start_dt = start_dt
            engine.run()
        timer.mark("engine_loop")

    # Single-Symbol Candle Engine
    elif len(envs) == 1:
        env = envs[0]

        # Vereinheitlichtes Alignment für Primary + Additional TFs
        bid_aligned, ask_aligned, multi_candle_data_aligned = _get_or_build_alignment(
            symbol_map=symbol_map,
            primary_tf=primary_tf,
            config=config,
            start_dt=start_dt,
        )
        timer.mark("data_align")

        engine = EventEngine(
            bid_candles=bid_aligned,
            ask_candles=ask_aligned,
            strategy=env.strategy,
            executor=env.executor,
            portfolio=env.portfolio,
            multi_candle_data=multi_candle_data_aligned,
            on_progress=progress,
            symbol=config["symbol"],
        )
        engine.original_start_dt = start_dt
        engine.run()
        timer.mark("engine_loop")

    # Multi-Strategy Controller
    # Multi-Strategy (Candle) – sichere Datenzuordnung je Strategie/Symbol
    else:
        align_time = 0.0
        engine_time = 0.0
        controller = MultiStrategyController(envs)
        tf_config = config.get("timeframes", {})
        primary_tf = tf_config.get("primary")
        for env in envs:
            sym = getattr(env.strategy, "symbol", None) or config.get("symbol")
            if not sym or sym not in symbol_map:
                raise ValueError(
                    f"❌ Kein passendes Symbol für Env '{env.name}' gefunden (sym={sym})."
                )

            single_sym_map = {sym: symbol_map[sym]}
            if config.get("timestamp_alignment", {}).get("diagnostics", True):
                diagnose_alignment(single_sym_map, primary_tf)

            t_align_start = time.perf_counter()
            bid_aligned, ask_aligned, multi_candle_data_aligned = (
                _get_or_build_alignment(
                    symbol_map=single_sym_map,
                    primary_tf=primary_tf,
                    config=config,
                    start_dt=start_dt,
                )
            )
            align_time += time.perf_counter() - t_align_start

            engine = EventEngine(
                bid_candles=bid_aligned,
                ask_candles=ask_aligned,
                strategy=env.strategy,
                executor=env.executor,
                portfolio=env.portfolio,
                multi_candle_data=multi_candle_data_aligned,
                on_progress=progress,
                symbol=sym,
            )
            # Kontextsensitiver Startzeitpunkt (kein start_dt erzwingen)
            engine.original_start_dt = start_dt
            t_engine_start = time.perf_counter()
            engine.run()
            engine_time += time.perf_counter() - t_engine_start

        timer.add("data_align", align_time)
        timer.add("engine_loop", engine_time)

    # --------- Reporting --------- #
    for env in envs:
        # Exakte Step-5-Robustness-Metriken für normalem Backtest (Feature-Flag)
        try:
            rep_cfg = config.get("reporting", {}) or {}
            enabled_flag = bool(
                rep_cfg.get("enable_backtest_robust_metrics", False)
                or config.get("enable_backtest_robust_metrics", False)
            )

            if enabled_flag and not getattr(
                env.portfolio, "backtest_robust_metrics", None
            ):
                env.portfolio.backtest_robust_metrics = (
                    _compute_backtest_robust_metrics(
                        config,
                        env.portfolio,
                    )
                )
        except Exception:
            pass
        print(f"\n📈 Ergebnisse für {env.name}:")
        for k, v in env.portfolio.get_summary().items():
            print(f"{k}: {v}")

        strategy_core = unwrap_strategy(env.strategy)
        from hf_engine.infra.logging.log_manager import log_entry

        if enable_logging and hasattr(strategy_core, "logger") and strategy_core.logger:
            log_entry(
                strategy_core.logger,
                strategy_name=env.name,
                symbol=config.get("symbol", ""),
            )
        save_backtest_result(
            env.portfolio,
            config,
            strategy_name=env.name,
            strategy_wrapper=strategy_core,
        )

    timer.mark("reporting")
    timings = timer.summary()
    for env in envs:
        try:
            setattr(env.portfolio, "backtest_timings", timings)
        except Exception:
            pass
    timer.print_summary()

    print("\n✅ Backtest abgeschlossen.")


def run_backtest_and_return_portfolio(
    config: dict,
    preloaded_data: Optional[dict] = None,
    prealigned: Optional[
        Tuple[List[Any], List[Any], Dict[str, Dict[str, List[Any]]]]
    ] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Runs a single-strategy backtest and returns Portfolio and optional EntryLog DataFrame.

    Args:
        config (dict): Backtest configuration.
        preloaded_data (dict, optional): Preloaded candle data for speedup.

    Returns:
        Tuple: (portfolio, entry_log_df or None)
    """
    timer = _BacktestTimer()
    _maybe_seed_deterministic_rng(config)
    mode = config.get("mode", "candle")
    profiling_enabled = bool(config.get("profiling", {}).get("enabled", False))
    t0 = time.perf_counter() if profiling_enabled else 0.0
    enable_logging = config.get("enable_entry_logging", False)
    start_dt, end_dt, extended_start, warmup_bars = prepare_time_window(config)
    symbol_map, bid_candles, ask_candles, tick_data = load_data(
        config, mode, extended_start, end_dt, preloaded_data=preloaded_data
    )
    timer.mark("data_load")
    t_load = time.perf_counter() - t0 if profiling_enabled else 0.0
    # --- Slippage/Fee Modelle inkl. Multiplikatoren (zentral + Overrides) ---
    exec_costs = _load_execution_costs(config)
    defaults = exec_costs.get("defaults", {})
    slip_base = {**defaults.get("slippage", {}), **config.get("slippage", {})}
    fee_base = {**defaults.get("fees", {}), **config.get("fees", {})}

    spread_mult = float(config.get("execution", {}).get("spread_multiplier", 1.0))
    fee_mult = float(config.get("execution", {}).get("fee_multiplier", 1.0))
    slip_mult = float(config.get("execution", {}).get("slippage_multiplier", 1.0))

    slippage_model = SlippageModel(
        fixed_pips=(float(slip_base.get("fixed_pips", 0.0)) * slip_mult),
        random_pips=(float(slip_base.get("random_pips", 0.0)) * slip_mult),
    )
    fee_model = FeeModel(
        per_million=(float(fee_base.get("per_million", 0.0)) * fee_mult),
        lot_size=float(fee_base.get("lot_size", 100_000.0)),
        min_fee=float(fee_base.get("min_fee", 0.0)),
    )
    # Spread-Multiplikator im Config-Objekt hinterlegen, falls Engine ihn ausliest
    config.setdefault("execution", {})
    config["execution"]["spread_multiplier"] = spread_mult
    config["execution"]["fee_multiplier"] = fee_mult
    config["execution"]["slippage_multiplier"] = slip_mult
    assert "strategy" in config, "Diese Funktion unterstützt nur Einzelstrategien"
    # --- Timeframe-Kontext VOR inneren Funktionen binden (wichtig für _build_rate_provider) ---
    tf_config = config.get("timeframes", {}) or {}
    primary_tf = tf_config.get("primary", "M15")
    additional_tfs = tf_config.get("additional", [])
    # -------------------------------------------------------------------------------------------
    # Account currency & central models
    account_ccy = config.get("account_currency", "EUR")
    symbol_specs = _load_symbol_specs(config)
    central_registry = _build_central_registry(symbol_specs)
    rates_cfg = config.get("rates", {}) or {}
    mode = (
        rates_cfg.get("mode")
        or ("static" if "rates_static" in config else "timeseries")
    ).lower()
    rates_static = {
        k.upper(): float(v) for k, v in config.get("rates_static", {}).items()
    }

    def _build_rate_provider():
        if mode == "static":
            return StaticRateProvider(
                rates_static, strict=bool(rates_cfg.get("strict", True))
            )
        ts_pairs = [p.upper() for p in rates_cfg.get("pairs", [])]
        # Dynamik: wenn kein timeframe gesetzt (oder "AUTO"), nimm primary TF
        ts_tf_cfg = (rates_cfg.get("timeframe") or "").upper()
        ts_tf = primary_tf if ts_tf_cfg in ("", "AUTO") else ts_tf_cfg
        use_price = rates_cfg.get("use_price", "close")
        stale_bars = rates_cfg.get("stale_limit_bars", 2)
        strict_ts = bool(rates_cfg.get("strict", True))
        rp_ts = TimeSeriesRateProvider(
            pairs=ts_pairs,
            timeframe=ts_tf,
            start_dt=start_dt,  # hier haben wir start_dt aus prepare_time_window
            end_dt=end_dt,
            use_price=use_price,
            stale_limit_bars=stale_bars,
            strict=strict_ts,
        )
        if mode == "composite":
            rp_static = StaticRateProvider(rates_static, strict=False)
            return CompositeRateProvider([rp_ts, rp_static])
        return rp_ts

    rate_provider = _build_rate_provider()
    lot_sizer = LotSizer(account_ccy, rate_provider, central_registry)
    commission_model = CommissionModel(
        account_ccy,
        rate_provider,
        exec_costs,
        central_registry,
        multiplier=fee_mult,
    )
    envs = prepare_strategies(
        config,
        symbol_map,
        slippage_model,
        fee_model,
        enable_logging,
        symbol_specs=symbol_specs,
        lot_sizer=lot_sizer,
        commission_model=commission_model,
    )
    timer.mark("strategy_prep")
    env = envs[0]
    # primary_tf/additional_tfs wurden bereits oben gebunden
    all_tfs = [primary_tf] + additional_tfs
    if isinstance(bid_candles, list):
        multi_candle_data = {primary_tf: {"bid": bid_candles, "ask": ask_candles}}
    else:
        multi_candle_data = {
            tf: {"bid": bid_candles.get(tf, []), "ask": ask_candles.get(tf, [])}
            for tf in all_tfs
        }
    # Engine
    if mode == "tick":
        timer.mark("data_align")
        if isinstance(tick_data, dict):
            strategy_symbol = getattr(env.strategy, "symbol", None)
            if not strategy_symbol:
                raise ValueError(
                    "❌ Strategy benötigt ein .symbol Attribut im Tick-Multi-Modus"
                )
            ticks = tick_data.get(strategy_symbol, [])
            if not ticks:
                raise ValueError(f"❌ Keine Tickdaten für Symbol: {strategy_symbol}")
        else:
            ticks = tick_data
        engine = TickEventEngine(
            ticks=ticks,
            strategy=env.strategy,
            executor=env.executor,
            portfolio=env.portfolio,
            multi_candle_data=multi_candle_data,
        )
    else:
        # Striktes Alignment wie in run_backtest (Single-Symbol Candle Engine)
        primary_tf = config["timeframes"]["primary"]
        if config.get("timestamp_alignment", {}).get("diagnostics", True):
            diagnose_alignment(symbol_map, primary_tf)

        if prealigned is not None:
            bid_aligned, ask_aligned, multi_candle_data_aligned = prealigned
            t_align = 0.0
        else:
            t_align_start = time.perf_counter() if profiling_enabled else 0.0
            bid_aligned, ask_aligned, multi_candle_data_aligned = (
                _get_or_build_alignment(
                    symbol_map=symbol_map,
                    primary_tf=primary_tf,
                    config=config,
                    start_dt=start_dt,
                )
            )
            t_align = time.perf_counter() - t_align_start if profiling_enabled else 0.0
        timer.mark("data_align")

        engine = EventEngine(
            bid_candles=bid_aligned,
            ask_candles=ask_aligned,
            strategy=env.strategy,
            executor=env.executor,
            portfolio=env.portfolio,
            multi_candle_data=multi_candle_data_aligned,
            symbol=config["symbol"],
        )
    engine.original_start_dt = start_dt
    t_loop_start = time.perf_counter() if profiling_enabled else 0.0
    engine.run()
    timer.mark("engine_loop")
    t_loop = time.perf_counter() - t_loop_start if profiling_enabled else 0.0
    # Profiling Info am Portfolio hinterlegen (kein API-Bruch)
    if profiling_enabled:
        setattr(
            env.portfolio,
            "profiling",
            {
                "load_seconds": round(t_load, 6),
                "align_seconds": round(t_align if "t_align" in locals() else 0.0, 6),
                "loop_seconds": round(t_loop, 6),
                "total_seconds": round((time.perf_counter() - t0), 6),
            },
        )

    timer.mark("reporting")
    timings = timer.summary()
    try:
        setattr(env.portfolio, "backtest_timings", timings)
    except Exception:
        pass
    timer.print_summary()

    if enable_logging and hasattr(env.strategy, "logger") and env.strategy.logger:
        return env.portfolio, env.strategy.logger.to_dataframe()
    return env.portfolio, None


# -------------------------------
# Robust Metrics (Backtest only)
# -------------------------------


def _compute_backtest_robust_metrics(
    config: dict,
    base_portfolio,
) -> Dict[str, float]:
    """
    Compute Step-5 equivalent metrics in a single backtest context by performing
    additional targeted runs (param jitter, cost shock, time jitter, yearly re-runs).

        Returns keys:
            - robustness_1
            - cost_shock_score
            - timing_jitter_score
            - trade_dropout_score
            - p_mean_gt
            - stability_score
            - tp_sl_stress_score
            - data_jitter_score
            - data_jitter_num_samples
            - data_jitter_failures
            - ulcer_index
            - ulcer_index_score
    """
    t_debug_start = time.perf_counter()
    stage_elapsed = {
        "base": 0.0,
        "tp_sl": 0.0,
        "r1": 0.0,
        "data_jitter": 0.0,
        "cost_shock": 0.0,
        "timing_jitter": 0.0,
        "trade_dropout": 0.0,
        "stability": 0.0,
    }

    from backtest_engine.rating.cost_shock_score import (
        COST_SHOCK_FACTORS,
        apply_cost_shock_inplace,
        compute_multi_factor_cost_shock_score,
    )
    from backtest_engine.rating.data_jitter_score import (
        _stable_data_jitter_seed,
        build_jittered_preloaded_data,
        compute_data_jitter_score,
        precompute_atr_cache,
    )
    from backtest_engine.rating.p_values import compute_p_mean_r_gt_0
    from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
    from backtest_engine.rating.stability_score import (
        compute_stability_score_from_yearly_profits,
    )
    from backtest_engine.rating.timing_jitter_score import (
        apply_timing_jitter_month_shift_inplace,
        compute_timing_jitter_score,
        get_timing_jitter_backward_shift_months,
    )
    from backtest_engine.rating.tp_sl_stress_score import (
        PrimaryCandleArrays,
        compute_tp_sl_stress_score,
        load_primary_candle_arrays_from_parquet,
    )
    from backtest_engine.rating.trade_dropout_score import (
        compute_multi_run_trade_dropout_score,
        simulate_trade_dropout_metrics_multi,
    )
    from backtest_engine.rating.ulcer_index_score import compute_ulcer_index_and_score
    from backtest_engine.report.metrics import calculate_metrics

    rep = config.get("reporting", {}) or {}
    try:
        mode = str(rep.get("robust_metrics_mode", "full") or "full").lower()
    except Exception:
        mode = "full"
    strat_params = _first_strategy_parameters(config)
    jitter_frac = float(
        _resolve_robust_setting(
            config,
            ("jitter_frac", "robust_jitter_frac"),
            0.05,
            parameters=strat_params,
        )
    )
    jitter_repeats = int(
        _resolve_robust_setting(
            config,
            ("robust_jitter_repeats", "jitter_repeats"),
            5,
            parameters=strat_params,
        )
    )
    dropout_frac = float(rep.get("robust_dropout_frac", 0.10))
    cost_shock_factor = float(rep.get("robust_cost_shock_factor", 0.50))
    param_bounds: Dict[str, Any] = rep.get("param_jitter_bounds", {}) or {}

    try:
        ulcer_cap = float(rep.get("ulcer_cap", 10.0))
    except Exception:
        ulcer_cap = 10.0

    ulcer_index = math.nan
    ulcer_index_score = 0.0

    try:
        debug_trade_dropout = bool(rep.get("debug_trade_dropout", False))
    except Exception:
        debug_trade_dropout = False

    # Optional: Fine-grained control over which params get jittered (Backtest-only)
    # - reporting.param_jitter_include: ["param_a", "param_b"]
    # - reporting.param_jitter_exclude: ["param_c"]
    # - reporting.param_jitter_scenario: "scenario_key" (hint)
    # - reporting.param_jitter_include_by_scenario: {"scenario_key": ["param_a", ...]}
    # - reporting.param_jitter_exclude_by_scenario: {"scenario_key": ["param_x", ...]}
    try:
        _inc = rep.get("param_jitter_include")
        include_global_set: Optional[Set[str]] = (
            set(map(str, _inc)) if isinstance(_inc, (list, tuple, set)) else None
        )
    except Exception:
        include_global_set = None
    try:
        _exc = rep.get("param_jitter_exclude")
        exclude_global_set: Set[str] = (
            set(map(str, _exc)) if isinstance(_exc, (list, tuple, set)) else set()
        )
    except Exception:
        exclude_global_set = set()
    scenario_hint = rep.get("param_jitter_scenario")
    include_by_scn = rep.get("param_jitter_include_by_scenario", {}) or {}
    exclude_by_scn = rep.get("param_jitter_exclude_by_scenario", {}) or {}

    # Auto-detect scenario from executed trades if requested or unset
    def _detect_scenario_from_portfolio(port) -> Optional[str]:
        try:
            counts: Counter = Counter()
            positions = []
            try:
                positions.extend(getattr(port, "closed_positions", []) or [])
                positions.extend(getattr(port, "partial_closed_positions", []) or [])
            except Exception:
                positions = []
            for p in positions:
                meta = getattr(p, "metadata", {}) or {}
                # tags like ["scenario2", ...]
                tags = meta.get("tags")
                if isinstance(tags, (list, tuple)):
                    for t in tags:
                        t_s = str(t or "").strip().lower()
                        if t_s.startswith("scenario"):
                            counts[t_s] += 1
                # scenario key like "long_2" / "short_3"
                scn = str(meta.get("scenario", "")).strip().lower()
                for d in ("2", "3", "4", "5", "6"):
                    if f"_{d}" in scn:
                        counts[f"scenario{d}"] += 1
            if counts:
                return counts.most_common(1)[0][0]
        except Exception:
            return None
        return None

    def _effective_scenario_hint(hint) -> Optional[str]:
        h = str(hint or "").strip().lower()
        if h in ("", "auto"):
            return _detect_scenario_from_portfolio(base_portfolio)
        # normalize e.g. "long_2" -> "scenario2"
        if h.endswith("_2"):
            return "scenario2"
        if h.endswith("_3"):
            return "scenario3"
        if h.endswith("_4"):
            return "scenario4"
        if h.endswith("_5"):
            return "scenario5"
        if h.endswith("_6"):
            return "scenario6"
        return h if h else None

    scenario_hint_eff = _effective_scenario_hint(scenario_hint)
    include_eff: Optional[Set[str]] = None
    try:
        if isinstance(scenario_hint_eff, str) and scenario_hint_eff:
            scn_vals = include_by_scn.get(scenario_hint_eff)
            if isinstance(scn_vals, (list, tuple, set)):
                include_eff = set(map(str, scn_vals))
        if include_eff is None:
            include_eff = include_global_set
    except Exception:
        include_eff = include_global_set
    # Special case: if scenario6 is active and no explicit include list is set,
    # restrict top-level jitter to scenario2 include set (scenario6 builds on scenario2).
    try:
        if (not include_eff) and scenario_hint_eff == "scenario6":
            scn2_vals = include_by_scn.get("scenario2")
            if isinstance(scn2_vals, (list, tuple, set)):
                include_eff = set(map(str, scn2_vals))
    except Exception:
        pass
    try:
        exclude_eff = set(exclude_global_set)
        if isinstance(scenario_hint_eff, str) and scenario_hint_eff:
            scn_ex = exclude_by_scn.get(scenario_hint_eff)
            if isinstance(scn_ex, (list, tuple, set)):
                exclude_eff |= set(map(str, scn_ex))
    except Exception:
        exclude_eff = exclude_global_set

    # Conditional HTF jittering based on filters (Backtest-only convenience)
    # - htf_ema only if htf_filter in {"above", "below"}
    # - extra_htf_ema only if extra_htf_filter in {"above", "below"}
    try:
        params_now = strat_params if isinstance(strat_params, dict) else {}
        htf_filter = str(params_now.get("htf_filter", "")).strip().lower()
        if htf_filter not in {"above", "below"}:
            if include_eff is not None:
                include_eff.discard("htf_ema")
            exclude_eff.add("htf_ema")
        extra_htf_filter = str(params_now.get("extra_htf_filter", "")).strip().lower()
        if extra_htf_filter not in {"above", "below"}:
            if include_eff is not None:
                include_eff.discard("extra_htf_ema")
            exclude_eff.add("extra_htf_ema")
    except Exception:
        pass

    # Detect Scenario 6 traded directions (long/short) to scope nested jitter
    scenario6_dirs: Optional[Set[str]] = None
    if str(scenario_hint_eff).lower() == "scenario6":
        try:

            def _detect_s6_dirs(port) -> Optional[Set[str]]:
                dirs: Set[str] = set()
                positions = []
                positions.extend(getattr(port, "closed_positions", []) or [])
                positions.extend(getattr(port, "partial_closed_positions", []) or [])
                positions.extend(getattr(port, "open_positions", []) or [])
                for p in positions:
                    meta = getattr(p, "metadata", {}) or {}
                    tags = meta.get("tags")
                    scn = str(meta.get("scenario", "")).strip().lower()
                    is_s6 = False
                    if isinstance(tags, (list, tuple)):
                        is_s6 = any(
                            str(t or "").strip().lower() == "scenario6" for t in tags
                        )
                    if (not is_s6) and ("_6" in scn):
                        is_s6 = True
                    if not is_s6:
                        continue
                    d = str(getattr(p, "direction", "")).strip().lower()
                    if d in ("long", "short"):
                        dirs.add(d)
                return dirs or None

            scenario6_dirs = _detect_s6_dirs(base_portfolio)
        except Exception:
            scenario6_dirs = None

    # Optional debug: print the final set of parameter names that will be jittered
    try:
        debug_names = bool(rep.get("param_jitter_debug", False))
    except Exception:
        debug_names = False
    try:
        debug_verbose = bool(rep.get("param_jitter_debug_verbose", False))
    except Exception:
        debug_verbose = False
    try:
        debug_per_repeat = bool(rep.get("param_jitter_debug_per_repeat", False))
    except Exception:
        debug_per_repeat = False
    if debug_names:
        try:
            params_now = strat_params if isinstance(strat_params, dict) else {}
            jitter_names: List[str] = []
            for k, v in params_now.items() if isinstance(params_now, dict) else []:
                if include_eff is not None and k not in include_eff:
                    continue
                if k in exclude_eff:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    jitter_names.append(str(k))
            jitter_names = sorted(set(jitter_names))
            scenolog = scenario_hint_eff or "auto"
            # Scenario plan info
            source = (
                "explicit"
                if str(rep.get("param_jitter_scenario", "")).strip().lower()
                not in ("", "auto")
                else "auto"
            )
            # Scenario counts from portfolio (for transparency)
            scen_counts: Dict[str, int] = {}
            try:
                positions = []
                positions.extend(getattr(base_portfolio, "closed_positions", []) or [])
                positions.extend(
                    getattr(base_portfolio, "partial_closed_positions", []) or []
                )
                positions.extend(getattr(base_portfolio, "open_positions", []) or [])
                for p in positions:
                    meta = getattr(p, "metadata", {}) or {}
                    tags = meta.get("tags")
                    if isinstance(tags, (list, tuple)):
                        for t in tags:
                            t_s = str(t or "").strip().lower()
                            if t_s.startswith("scenario"):
                                scen_counts[t_s] = scen_counts.get(t_s, 0) + 1
                    scn = str(meta.get("scenario", "")).strip().lower()
                    for d in ("2", "3", "4", "5", "6"):
                        if f"_{d}" in scn:
                            key = f"scenario{d}"
                            scen_counts[key] = scen_counts.get(key, 0) + 1
            except Exception:
                scen_counts = {}

            print(
                f"🔎 Robustness1 Jitter-Parameter (scenario={scenolog}, repeats={jitter_repeats}, frac={jitter_frac}): "
                + ", ".join(jitter_names)
            )
            # Plan + gating info
            try:
                print(
                    f"   plan: scenario_hint={repr(rep.get('param_jitter_scenario', None))}, scenario_eff={scenolog}, source={source}"
                )
                if scen_counts:
                    counts_str = ", ".join(
                        f"{k}:{v}" for k, v in sorted(scen_counts.items())
                    )
                    print(f"   scenario_counts: {counts_str}")
                print(f"   jitter: frac={jitter_frac}, repeats={jitter_repeats}")
                inc_str = ", ".join(sorted(include_eff)) if include_eff else "(none)"
                exc_str = ", ".join(sorted(exclude_eff)) if exclude_eff else "(none)"
                print(f"   include_eff: {inc_str}")
                print(f"   exclude_eff: {exc_str}")
                htf_filter_val = str(strat_params.get("htf_filter", "")).lower()
                extra_htf_filter_val = str(
                    strat_params.get("extra_htf_filter", "")
                ).lower()
                print(
                    f"   htf_filter={htf_filter_val} => htf_ema={'yes' if 'htf_ema' not in (exclude_eff or set()) else 'no'}"
                )
                print(
                    f"   extra_htf_filter={extra_htf_filter_val} => extra_htf_ema={'yes' if 'extra_htf_ema' not in (exclude_eff or set()) else 'no'}"
                )
                if str(scenolog).lower() == "scenario6" and include_by_scn.get(
                    "scenario6"
                ) in (None, []):
                    print(
                        "   note: scenario6 top-level include derived from scenario2 include (fallback)"
                    )
            except Exception:
                pass

            # Optional verbose: span descriptors (no samples), top-level
            if debug_verbose:

                def _span_descr(val, frac):
                    try:
                        if isinstance(val, int):
                            base = int(val)
                            span = max(1, int(round(abs(base) * float(frac))))
                            return f"±{span} (int)"
                        if isinstance(val, float):
                            base = float(val)
                            span = abs(base) * float(frac)
                            if span == 0.0:
                                span = float(frac)
                            return f"±{span:.6g} (float)"
                    except Exception:
                        return ""
                    return ""

                spans = []
                try:
                    for k, v in (
                        params_now.items() if isinstance(params_now, dict) else []
                    ):
                        if include_eff is not None and k not in include_eff:
                            continue
                        if k in (exclude_eff or set()):
                            continue
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            spans.append(f"{k}({_span_descr(v, jitter_frac)})")
                    spans = sorted(spans)
                    print(
                        "   top-level spans:", ", ".join(spans) if spans else "(none)"
                    )
                except Exception:
                    pass

            # Scenario-6 nested names: log once (not per repeat)
            if str(scenolog).lower() == "scenario6":
                try:
                    tfs = params_now.get("scenario6_timeframes") or []
                    scen6 = params_now.get("scenario6_params") or {}
                    nested_names: List[str] = []
                    if isinstance(scen6, dict):
                        tf_keys = [
                            str(tf).strip().upper()
                            for tf in (tfs or [])
                            if str(tf or "").strip()
                        ]
                        # mismatch warnings & direction plan
                        try:
                            available = {k for k in scen6.keys()}
                            tf_missing = [tf for tf in tf_keys if tf not in available]
                            if tf_missing:
                                print(
                                    "   WARNING: scenario6_timeframes not found in scenario6_params:",
                                    ", ".join(tf_missing),
                                )
                            used_dirs = sorted(
                                list(scenario6_dirs or {"long", "short"})
                            )
                            print(
                                "   scenario6_tfs:",
                                ", ".join(tf_keys) if tf_keys else "(none)",
                            )
                            print(
                                "   scenario6_dirs:",
                                ", ".join(used_dirs) if used_dirs else "(none)",
                            )
                        except Exception:
                            pass
                        for tf in tf_keys:
                            node = scen6.get(tf)
                            if not isinstance(node, dict):
                                continue
                            for direction in ("long", "short"):
                                if (
                                    scenario6_dirs is not None
                                    and direction not in scenario6_dirs
                                ):
                                    continue
                                d = node.get(direction)
                                if not isinstance(d, dict):
                                    continue
                                for pk, pv in list(d.items()):
                                    if isinstance(pv, (int, float)) and not isinstance(
                                        pv, bool
                                    ):
                                        nested_names.append(
                                            f"scenario6:{tf}.{direction}.{pk}"
                                        )
                    if nested_names:
                        print(
                            "🔎 Robustness1 Jitter-Parameter (scenario6 nested): "
                            + ", ".join(sorted(set(nested_names)))
                        )
                    # Verbose spans for nested
                    if debug_verbose and isinstance(scen6, dict):
                        try:
                            ns: List[str] = []
                            for tf in tf_keys:
                                node = scen6.get(tf, {})
                                if not isinstance(node, dict):
                                    continue
                                for direction in scenario6_dirs or {"long", "short"}:
                                    d = node.get(direction, {})
                                    if not isinstance(d, dict):
                                        continue
                                    for pk, pv in d.items():
                                        if isinstance(
                                            pv, (int, float)
                                        ) and not isinstance(pv, bool):
                                            ns.append(
                                                f"{tf}.{direction}.{pk}({_span_descr(pv, jitter_frac)})"
                                            )
                            if ns:
                                print(
                                    "   scenario6 nested spans:",
                                    ", ".join(sorted(ns)),
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # Debug darf den Lauf niemals gefährden
            pass

    # Base metrics from the executed portfolio
    t_stage = time.perf_counter()
    base_summary = calculate_metrics(base_portfolio)
    base_profit = float(base_summary.get("net_profit_after_fees_eur", 0.0) or 0.0)
    base_avg_r = float(base_summary.get("avg_r_multiple", 0.0) or 0.0)
    base_winrate = float(base_summary.get("winrate_percent", 0.0) or 0.0)
    base_drawdown = float(base_summary.get("drawdown_eur", 0.0) or 0.0)
    base_commission = float(base_summary.get("fees_total_eur", 0.0) or 0.0)
    base_sharpe = float(base_summary.get("sharpe_trade", 0.0) or 0.0)

    base_metrics = {
        "profit": base_profit,
        "drawdown": base_drawdown,
        "profit_over_dd": (base_profit / base_drawdown) if base_drawdown > 0 else 0.0,
        "commission": base_commission,
        "trades": int(base_summary.get("total_trades", 0) or 0),
        "sharpe": base_sharpe,
    }

    # Trades DF for bootstrap, Dropout und TP/SL-Stress
    try:
        trades_df = base_portfolio.trades_to_dataframe()
    except Exception:
        trades_df = pd.DataFrame()

    # p_mean_gt based on R-Multiples
    p_mean_gt = compute_p_mean_r_gt_0(
        trades_df, r_col="r_multiple", n_boot=2000, seed=123
    )

    # TP/SL-Stress-Score aus Primär-TF-Candles (Parquet) berechnen
    try:
        symbol = str(config.get("symbol") or "").strip()
        tf_conf = config.get("timeframes", {}) or {}
        primary_tf = str(tf_conf.get("primary") or "").strip()
        arrays: Optional[PrimaryCandleArrays] = None
        if symbol and primary_tf:
            arrays = load_primary_candle_arrays_from_parquet(symbol, primary_tf)
        t_tp_sl = time.perf_counter()
        tp_sl_score = compute_tp_sl_stress_score(trades_df, arrays)
        stage_elapsed["tp_sl"] = time.perf_counter() - t_tp_sl
    except Exception:
        tp_sl_score = 1.0

    # Ulcer Index (Tail/Drawdown metric)
    try:
        equity_curve = (
            base_portfolio.get_equity_curve()
            if hasattr(base_portfolio, "get_equity_curve")
            else []
        )
    except Exception:
        equity_curve = []
    try:
        ulcer_index, ulcer_index_score = compute_ulcer_index_and_score(
            equity_curve, ulcer_cap=ulcer_cap
        )
    except Exception:
        ulcer_index, ulcer_index_score = math.nan, 0.0

    stage_elapsed["base"] = time.perf_counter() - t_stage

    # Data Jitter Score (ATR-skaliertes OHLC-Rauschen)
    t_stage = time.perf_counter()
    data_jitter_repeats = int(
        _resolve_robust_setting(
            config,
            (
                "robust_data_jitter_repeats",
                "data_jitter_repeats",
                # Backward-compatible aliases
                "robust_data_jitter_repeat",
                "data_jitter_repeat",
            ),
            5,
            parameters=strat_params,
        )
    )

    try:
        debug_data_jitter = bool(rep.get("data_jitter_debug", False))
    except Exception:
        debug_data_jitter = False
    try:
        debug_data_jitter_per_repeat = bool(
            rep.get("data_jitter_debug_per_repeat", False)
        )
    except Exception:
        debug_data_jitter_per_repeat = False

    config_mode = str(config.get("mode", "") or "").lower()
    skip_data_jitter = (
        data_jitter_repeats <= 0
        or mode == "r1_only"
        or config_mode == "tick"
        or "multi_symbols" in config
    )

    if skip_data_jitter:
        data_jitter_score = 0.0
        data_jitter_num_samples = 0
        data_jitter_failures = 0
        stage_elapsed["data_jitter"] = time.perf_counter() - t_stage
    else:
        atr_period = int(
            _resolve_robust_setting(
                config,
                (
                    "robust_data_jitter_atr_period",
                    "data_jitter_atr_period",
                    # Backward-compatible aliases (configs may use plural naming)
                    "robust_data_jitter_atr_periods",
                    "data_jitter_atr_periods",
                ),
                14,
                parameters=strat_params,
            )
        )
        sigma_atr = float(
            _resolve_robust_setting(
                config,
                (
                    "robust_data_jitter_sigma_atr",
                    "data_jitter_sigma_atr",
                    # Backward-compatible aliases (configs may use plural naming)
                    "robust_data_jitter_sigma_atrs",
                    "data_jitter_sigma_atrs",
                ),
                0.10,
                parameters=strat_params,
            )
        )
        penalty_cap_dj = float(
            _resolve_robust_setting(
                config,
                ("robust_data_jitter_penalty_cap", "data_jitter_penalty_cap"),
                0.5,
                parameters=strat_params,
            )
        )
        min_price = float(
            _resolve_robust_setting(
                config,
                ("robust_data_jitter_min_price", "data_jitter_min_price"),
                1e-9,
                parameters=strat_params,
            )
        )
        fraq = float(
            _resolve_robust_setting(
                config,
                (
                    "robust_data_jitter_fraq",
                    "robust_data_jitter_frac",
                    "data_jitter_fraq",
                    "data_jitter_frac",
                ),
                0.0,
                parameters=strat_params,
            )
        )

        base_seed = _get_data_jitter_base_seed(config)
        start_dt, end_dt, extended_start, _ = prepare_time_window(config)
        base_preloaded_data = _load_base_preloaded_data(config, extended_start, end_dt)
        atr_cache = precompute_atr_cache(base_preloaded_data, period=atr_period)

        base_data_jitter_metrics = {
            "profit": float(base_profit),
            "avg_r": float(base_avg_r),
            "winrate": float(base_winrate),
            "drawdown": float(base_drawdown),
        }

        if debug_data_jitter:
            try:
                print(
                    "[data-jitter] repeats={} atr_period={} sigma_atr={} penalty_cap={} base_profit={:.2f} base_avg_r={:.4f} base_winrate={:.2f} base_dd={:.2f}".format(
                        data_jitter_repeats,
                        atr_period,
                        sigma_atr,
                        penalty_cap_dj,
                        float(base_profit),
                        float(base_avg_r),
                        float(base_winrate),
                        float(base_drawdown),
                    )
                )
            except Exception:
                pass

        jitter_metrics_list: List[Dict[str, float]] = []
        data_jitter_failures = 0

        for i in range(max(0, data_jitter_repeats)):
            seed_i = _stable_data_jitter_seed(base_seed, i)
            cfg_i = _config_without_robust(config)
            try:
                exec_cfg = cfg_i.setdefault("execution", {})
                if isinstance(exec_cfg, dict):
                    exec_cfg["random_seed"] = int(seed_i)
                cfg_i["execution"] = exec_cfg
            except Exception:
                pass

            if debug_data_jitter_per_repeat:
                try:
                    print(
                        f"[data-jitter] repeat {i + 1}/{data_jitter_repeats} seed={int(seed_i)}"
                    )
                except Exception:
                    pass

            preloaded_i = None
            try:
                preloaded_i = build_jittered_preloaded_data(
                    base_preloaded_data,
                    atr_cache=atr_cache,
                    sigma_atr=sigma_atr,
                    seed=seed_i,
                    min_price=min_price,
                    fraq=fraq,
                )
                port_i, _ = run_backtest_and_return_portfolio(
                    cfg_i, preloaded_data=preloaded_i
                )

                summ_i = calculate_metrics(port_i)
                met_i = {
                    "profit": float(
                        summ_i.get("net_profit_after_fees_eur", 0.0) or 0.0
                    ),
                    "avg_r": float(summ_i.get("avg_r_multiple", 0.0) or 0.0),
                    "winrate": float(summ_i.get("winrate_percent", 0.0) or 0.0),
                    "drawdown": float(summ_i.get("drawdown_eur", 0.0) or 0.0),
                }
                jitter_metrics_list.append(met_i)

                if debug_data_jitter_per_repeat:
                    try:
                        print(
                            "[data-jitter]   metrics profit={:.2f} avg_r={:.4f} winrate={:.2f} dd={:.2f}".format(
                                met_i["profit"],
                                met_i["avg_r"],
                                met_i["winrate"],
                                met_i["drawdown"],
                            )
                        )
                    except Exception:
                        pass
            except Exception:
                data_jitter_failures += 1
                if debug_data_jitter_per_repeat:
                    try:
                        print(
                            f"[data-jitter]   ERROR repeat {i + 1}/{data_jitter_repeats} (failures={data_jitter_failures})"
                        )
                    except Exception:
                        pass
            finally:
                del preloaded_i
                gc.collect()

        data_jitter_num_samples = len(jitter_metrics_list)
        data_jitter_score = compute_data_jitter_score(
            base_data_jitter_metrics,
            jitter_metrics_list,
            penalty_cap=penalty_cap_dj,
        )
        stage_elapsed["data_jitter"] = time.perf_counter() - t_stage

    # Robustness 1: Parameter jitter repeats
    t_stage = time.perf_counter()
    param_jitter_samples: List[Dict[str, float]] = []
    # Enable nested jitter for scenario6 (TF-specific params) if detected
    jitter_s6_nested = bool(scenario_hint_eff == "scenario6")

    # Jitter-Failure-Tracking: Bei zu vielen Fehlern frühzeitig abbrechen
    # um aussagelose 0.5 Scores durch leere param_jitter_samples zu vermeiden
    jitter_failures = 0
    max_jitter_failures = max(1, jitter_repeats // 4)  # 25% Toleranz

    for i_repeat in range(max(0, jitter_repeats)):
        cfg_j = _config_without_robust(config)
        _apply_param_jitter(
            cfg_j,
            jitter_frac=jitter_frac,
            bounds=param_bounds,
            include=include_eff,
            exclude=exclude_eff,
            jitter_scenario6_nested=jitter_s6_nested,
            scenario6_dirs=scenario6_dirs,
        )
        if debug_names and debug_per_repeat:
            try:
                print(
                    f"🔁 Jitter repeat {i_repeat + 1}/{jitter_repeats} (scenario6_nested={'on' if jitter_s6_nested else 'off'})"
                )
            except Exception:
                pass
        try:
            port_j, _ = run_backtest_and_return_portfolio(cfg_j)
        except Exception as exc:
            jitter_failures += 1
            if jitter_failures >= max_jitter_failures:
                try:
                    print(
                        f"[robust-metrics] Zu viele Jitter-Fehler ({jitter_failures}/{jitter_repeats}), "
                        f"breche Jitter-Phase ab. Letzter Fehler: {type(exc).__name__}: {exc}"
                    )
                except Exception:
                    pass
                break
            continue
        summ_j = calculate_metrics(port_j)
        profit_j = float(summ_j.get("net_profit_after_fees_eur", 0.0) or 0.0)
        avg_r_j = float(summ_j.get("avg_r_multiple", 0.0) or 0.0)
        winrate_j = float(summ_j.get("winrate_percent", 0.0) or 0.0)
        dd_j = float(summ_j.get("drawdown_eur", 0.0) or 0.0)
        param_jitter_samples.append(
            {
                "profit": float(profit_j),
                "avg_r": float(avg_r_j),
                "winrate": float(winrate_j),
                "drawdown": float(dd_j),
            }
        )
    base_r1 = {
        "profit": float(base_profit),
        "avg_r": float(base_avg_r),
        "winrate": float(base_winrate),
        "drawdown": float(base_drawdown),
    }
    robustness_1 = float(
        compute_robustness_score_1(base_r1, param_jitter_samples, penalty_cap=0.5)
    )
    stage_elapsed["r1"] = time.perf_counter() - t_stage

    # Warnung bei hoher Failure-Rate oder wenigen Samples
    if jitter_repeats > 0:
        success_rate = len(param_jitter_samples) / jitter_repeats
        if success_rate < 0.5:
            try:
                print(
                    f"[robust-metrics] Warnung: Nur {len(param_jitter_samples)}/{jitter_repeats} "
                    f"Jitter-Runs erfolgreich ({success_rate:.0%}). "
                    f"Robustness-Score könnte unzuverlässig sein."
                )
            except Exception:
                pass

    if mode == "r1_only":
        try:
            total_elapsed = time.perf_counter() - t_debug_start
            print(
                "[robust-metrics] base={:.2f}s tp_sl={:.2f}s r1={:.2f}s data_jitter={:.2f}s cost={:.2f}s timing_jitter={:.2f}s dropout={:.2f}s stability={:.2f}s total={:.2f}s".format(
                    stage_elapsed["base"],
                    stage_elapsed["tp_sl"],
                    stage_elapsed["r1"],
                    stage_elapsed["data_jitter"],
                    0.0,  # cost shock
                    0.0,  # timing_jitter
                    0.0,  # trade dropout
                    0.0,  # stability
                    total_elapsed,
                )
            )
        except Exception:
            pass
        return {
            "robustness_1": round(robustness_1, 4),
            "robustness_1_num_samples": len(param_jitter_samples),
            "robustness_1_jitter_failures": jitter_failures,
            "cost_shock_score": 0.0,
            "timing_jitter_score": 0.0,
            "trade_dropout_score": 0.0,
            "p_mean_gt": round(float(p_mean_gt), 4),
            "stability_score": 1.0,
            "tp_sl_stress_score": round(float(tp_sl_score), 4),
            "data_jitter_score": round(float(data_jitter_score), 4),
            "data_jitter_num_samples": data_jitter_num_samples,
            "data_jitter_failures": data_jitter_failures,
            "ulcer_index": float(ulcer_index),
            "ulcer_index_score": float(ulcer_index_score),
        }

    # Stress scores (independent; each uses penalty_cap=0.5)

    # Cost shock (multi-factor, deterministic)
    t_stage = time.perf_counter()
    shocked_metrics_list: List[Dict[str, float]] = []
    for factor in COST_SHOCK_FACTORS:
        cfg_cost = _config_without_robust(config)
        apply_cost_shock_inplace(cfg_cost, factor=factor)
        try:
            port_c, _ = run_backtest_and_return_portfolio(cfg_cost)
            met_c = _metrics_from_portfolio(port_c)
        except Exception:
            # On failure, assume worst-case degradation relative to base drawdown
            met_c = {
                "profit": 0.0,
                "drawdown": float(base_metrics.get("drawdown", 0.0) or 0.0) * 2,
                "sharpe": 0.0,
            }
        shocked_metrics_list.append(met_c)

    cost_shock_score = float(
        compute_multi_factor_cost_shock_score(
            base_metrics, shocked_metrics_list, penalty_cap=0.5
        )
    )
    stage_elapsed["cost_shock"] = time.perf_counter() - t_stage

    # Timing jitter (BACKWARD month shifts): /10, /5, /20 of the overall window length
    # (minimum 1 month). We do NOT shift forward to avoid any future leakage.
    t_stage = time.perf_counter()
    jitter_metrics: List[Dict[str, float]] = []

    # Optional debug output for validating the timing-jitter window mutation.
    # Enable via config:
    #   reporting: { debug_timing_jitter: true }
    # or via env var:
    #   TIMING_JITTER_DEBUG=1
    try:
        rep_cfg = config.get("reporting", {}) or {}
        debug_timing_jitter = bool(rep_cfg.get("debug_timing_jitter", False))
    except Exception:
        debug_timing_jitter = False
    try:
        env_flag = str(os.getenv("TIMING_JITTER_DEBUG", "") or "").strip().lower()
        if env_flag in {"1", "true", "yes", "y", "on"}:
            debug_timing_jitter = True
    except Exception:
        pass

    shift_months_list = get_timing_jitter_backward_shift_months(
        start_date=str(config.get("start_date") or ""),
        end_date=str(config.get("end_date") or ""),
        divisors=(10, 5, 20),
        min_months=1,
    )

    if debug_timing_jitter:
        try:
            print(
                "[timing-jitter][debug] base_window={}..{} shifts_months={}".format(
                    str(config.get("start_date") or ""),
                    str(config.get("end_date") or ""),
                    list(map(int, shift_months_list)) if shift_months_list else [],
                )
            )
        except Exception:
            pass

    for shift_months in shift_months_list:
        cfg_t = _config_without_robust(config)
        before_start = str(cfg_t.get("start_date") or "")
        before_end = str(cfg_t.get("end_date") or "")
        apply_timing_jitter_month_shift_inplace(
            cfg_t, shift_months_backward=int(shift_months)
        )
        if debug_timing_jitter:
            try:
                print(
                    "[timing-jitter][debug] shift_months={} {}..{} -> {}..{}".format(
                        int(shift_months),
                        before_start,
                        before_end,
                        str(cfg_t.get("start_date") or ""),
                        str(cfg_t.get("end_date") or ""),
                    )
                )
            except Exception:
                pass
        port_t, _ = run_backtest_and_return_portfolio(cfg_t)
        jitter_metrics.append(_metrics_from_portfolio(port_t))
    timing_jitter_score = float(
        compute_timing_jitter_score(base_metrics, jitter_metrics, penalty_cap=0.5)
        if jitter_metrics
        else 0.0
    )
    stage_elapsed["timing_jitter"] = time.perf_counter() - t_stage

    # Trade dropout metric from base trades (multi-run average)
    t_stage = time.perf_counter()
    try:
        dropout_runs = int(rep.get("robust_dropout_runs", 1) or 1)
    except Exception:
        dropout_runs = 1

    dropout_metrics_list = simulate_trade_dropout_metrics_multi(
        trades_df,
        dropout_frac=dropout_frac,
        base_metrics=base_metrics,
        n_runs=dropout_runs,
        seed=987654321,
        debug=debug_trade_dropout,
    )
    trade_dropout_score = float(
        compute_multi_run_trade_dropout_score(
            base_metrics, dropout_metrics_list, penalty_cap=0.5
        )
    )
    stage_elapsed["trade_dropout"] = time.perf_counter() - t_stage

    # Stability Score: exact yearly re-runs as in Step 5
    t_stage = time.perf_counter()
    stability_score = _stability_score_yearly_reruns(config)
    stage_elapsed["stability"] = time.perf_counter() - t_stage

    try:
        total_elapsed = time.perf_counter() - t_debug_start
        print(
            "[robust-metrics] base={:.2f}s tp_sl={:.2f}s r1={:.2f}s data_jitter={:.2f}s cost={:.2f}s timing_jitter={:.2f}s dropout={:.2f}s stability={:.2f}s total={:.2f}s".format(
                stage_elapsed["base"],
                stage_elapsed["tp_sl"],
                stage_elapsed["r1"],
                stage_elapsed["data_jitter"],
                stage_elapsed["cost_shock"],
                stage_elapsed["timing_jitter"],
                stage_elapsed["trade_dropout"],
                stage_elapsed["stability"],
                total_elapsed,
            )
        )
    except Exception:
        pass

    return {
        "robustness_1": round(robustness_1, 4),
        "robustness_1_num_samples": len(param_jitter_samples),
        "robustness_1_jitter_failures": jitter_failures,
        "cost_shock_score": round(float(cost_shock_score), 4),
        "timing_jitter_score": round(float(timing_jitter_score), 4),
        "trade_dropout_score": round(float(trade_dropout_score), 4),
        "p_mean_gt": round(float(p_mean_gt), 4),
        "stability_score": round(float(stability_score), 4),
        "tp_sl_stress_score": round(float(tp_sl_score), 4),
        "data_jitter_score": round(float(data_jitter_score), 4),
        "data_jitter_num_samples": data_jitter_num_samples,
        "data_jitter_failures": data_jitter_failures,
        "ulcer_index": float(ulcer_index),
        "ulcer_index_score": float(ulcer_index_score),
    }


def _config_without_robust(config: dict) -> dict:
    cfg = deepcopy(config)
    try:
        rep = cfg.get("reporting", {}) or {}
        rep["enable_backtest_robust_metrics"] = False
        cfg["reporting"] = rep
        # logging optional drosseln
        cfg["enable_entry_logging"] = False
    except Exception:
        pass
    return cfg


def _get_data_jitter_base_seed(config: dict) -> int:
    """Bestimmt den Basisseed für Data-Jitter (execution.random_seed > dev_seed > zufällig)."""

    seed: Optional[int] = None
    try:
        exec_cfg = config.get("execution", {}) or {}
        if isinstance(exec_cfg, dict) and exec_cfg.get("random_seed", None) is not None:
            seed = int(exec_cfg.get("random_seed"))
        else:
            rep_cfg = config.get("reporting", {}) or {}
            if isinstance(rep_cfg, dict) and bool(rep_cfg.get("dev_mode", False)):
                if rep_cfg.get("dev_seed", None) is not None:
                    seed = int(rep_cfg.get("dev_seed"))
    except Exception:
        seed = None

    if seed is None:
        try:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        except Exception:
            seed = random.randint(0, 2**32 - 1)

    return int(seed) % (2**32)


def _load_base_preloaded_data(
    config: dict, start_dt: datetime, end_dt: datetime
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Lädt OHLC-Parquet-Daten (BID/ASK) für alle konfigurierten Timeframes."""

    from hf_engine.infra.config.paths import PARQUET_DIR

    try:
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tz.UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=tz.UTC)
    except Exception:
        pass

    symbol = str(config.get("symbol", "") or "").strip()
    tf_cfg = config.get("timeframes", {}) or {}
    primary_tf = tf_cfg.get("primary")
    additional = tf_cfg.get("additional", []) or []
    all_tfs = [primary_tf] + list(additional)

    preloaded: Dict[Tuple[str, str], pd.DataFrame] = {}
    parquet_dir = PARQUET_DIR / symbol

    for tf in all_tfs:
        if not tf:
            continue
        tf_str = str(tf)
        for candle_type in ("bid", "ask"):
            path_upper = (
                parquet_dir / f"{symbol}_{tf_str}_{candle_type.upper()}.parquet"
            )
            path_lower = (
                parquet_dir / f"{symbol}_{tf_str}_{candle_type.lower()}.parquet"
            )
            path = path_upper if path_upper.exists() else path_lower
            if not path.exists():
                continue

            try:
                df = pd.read_parquet(
                    path, columns=["UTC time", "Open", "High", "Low", "Close", "Volume"]
                )
            except Exception:
                continue

            try:
                df["UTC time"] = pd.to_datetime(df["UTC time"])
                if df["UTC time"].dt.tz is None:
                    df["UTC time"] = df["UTC time"].dt.tz_localize("UTC")
            except Exception:
                pass

            try:
                mask = (df["UTC time"] >= start_dt) & (df["UTC time"] <= end_dt)
                df = df.loc[mask]
            except Exception:
                pass

            preloaded[(tf_str, candle_type)] = df.reset_index(drop=True)

    return preloaded


def _apply_param_jitter(
    cfg: dict,
    *,
    jitter_frac: float,
    bounds: Dict[str, Any],
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
    jitter_scenario6_nested: bool = False,
    scenario6_dirs: Optional[Set[str]] = None,
) -> None:
    """
    Jittert Parameter identisch zur Logik in final_param_selector.py:
    - Spannweite wird relativ zum aktuellen Wert bestimmt (|value| * jitter_frac).
    - Sampling erfolgt aus [value - span, value + span] ohne zusätzliche Bounds.
    """
    params = (cfg.get("strategy", {}) or {}).get("parameters", {}) or {}
    if not isinstance(params, dict):
        return
    for k, v in list(params.items()):
        # Optional: nur ausgewählte Parameter jittern (Backtest-only Feature)
        if include is not None and k not in include:
            continue
        if exclude is not None and k in exclude:
            continue
        if isinstance(v, bool) or v is None:
            continue

        if isinstance(v, int):
            base_val = int(v)
            span = max(1, int(round(abs(base_val) * float(jitter_frac))))
            lo = base_val - span
            hi = base_val + span
            if lo > hi:
                jittered = int(v)
            else:
                jittered = int(np.random.randint(lo, hi + 1))
            params[k] = jittered

        elif isinstance(v, float):
            base_val = float(v)
            span = abs(base_val) * float(jitter_frac)
            if span == 0.0:
                span = float(jitter_frac)
            lo = base_val - span
            hi = base_val + span
            if hi < lo:
                jittered = float(v)
            else:
                jittered = float(np.random.uniform(lo, hi))
            params[k] = jittered

        else:
            # Kategorische Parameter unverändert lassen (Deckungsgleich mit final_param_selector)
            params[k] = v

    # Scenario 6 nested jitter: iterate TF-specific parameter overrides
    if jitter_scenario6_nested:
        try:
            tfs = params.get("scenario6_timeframes") or []
            scen6 = params.get("scenario6_params") or {}
            if isinstance(scen6, dict):
                tf_keys = [
                    str(tf).strip().upper()
                    for tf in (tfs or [])
                    if str(tf or "").strip()
                ]

                def _jit_num(val):
                    if isinstance(val, int):
                        base_val = int(val)
                        span = max(1, int(round(abs(base_val) * float(jitter_frac))))
                        lo, hi = base_val - span, base_val + span
                        return (
                            int(np.random.randint(lo, hi + 1)) if lo <= hi else base_val
                        )
                    if isinstance(val, float):
                        base_val = float(val)
                        span = abs(base_val) * float(jitter_frac)
                        if span == 0.0:
                            span = float(jitter_frac)
                        lo, hi = base_val - span, base_val + span
                        return (
                            float(np.random.uniform(lo, hi)) if hi >= lo else base_val
                        )
                    return val

                jittered_nested_keys: List[str] = []
                for tf in tf_keys:
                    node = scen6.get(tf)
                    if not isinstance(node, dict):
                        continue
                    for direction in ("long", "short"):
                        if (
                            scenario6_dirs is not None
                            and direction not in scenario6_dirs
                        ):
                            continue
                        d = node.get(direction)
                        if not isinstance(d, dict):
                            continue
                        for pk, pv in list(d.items()):
                            if isinstance(pv, bool) or pv is None:
                                continue
                            if isinstance(pv, (int, float)):
                                d[pk] = _jit_num(pv)
                                jittered_nested_keys.append(f"{tf}.{direction}.{pk}")

                # Nested debug logging moved to once-per-run in _compute_backtest_robust_metrics
        except Exception:
            # Nested jitter must not break the run
            pass


def _metrics_from_portfolio(portfolio) -> Dict[str, float]:
    from backtest_engine.report.metrics import calculate_metrics

    s = calculate_metrics(portfolio)
    profit = float(s.get("net_profit_after_fees_eur", 0.0) or 0.0)
    dd = float(s.get("drawdown_eur", 0.0) or 0.0)
    pod = (profit / dd) if dd > 0 else 0.0
    commission = float(s.get("fees_total_eur", 0.0) or 0.0)
    trades = int(s.get("total_trades", 0) or 0)
    sharpe = float(s.get("sharpe_trade", 0.0) or 0.0)
    comm_over_profit = (commission / profit) if profit not in (0, np.nan) else np.nan
    return {
        "profit": profit,
        "drawdown": dd,
        "profit_over_dd": pod,
        "commission": commission,
        "trades": trades,
        "comm_over_profit": comm_over_profit,
        "sharpe": sharpe,
    }


def _yearly_segments_exact(
    start_s: str, end_s: str
) -> List[Tuple[str, datetime, datetime]]:
    if not start_s or not end_s:
        return []
    start_dt = datetime.strptime(start_s, "%Y-%m-%d")
    end_dt = datetime.strptime(end_s, "%Y-%m-%d")
    if end_dt < start_dt:
        return []
    segments: List[Tuple[str, datetime, datetime]] = []
    for year in range(start_dt.year, end_dt.year + 1):
        seg_start = max(start_dt, datetime(year, 1, 1))
        seg_end = min(end_dt, datetime(year, 12, 31))
        if seg_start <= seg_end:
            segments.append((str(year), seg_start, seg_end))
    return segments


def _stability_score_yearly_reruns(config: dict) -> float:
    # Re-run yearly segments and take Net Profit (after fees) per year
    start_s = config.get("start_date")
    end_s = config.get("end_date")
    segments = _yearly_segments_exact(start_s, end_s)
    if not segments:
        return 1.0
    profits: List[Tuple[str, float]] = []
    durations_by_year: Dict[int, float] = {}
    for label, seg_start, seg_end in segments:
        cfg_y = _config_without_robust(config)
        cfg_y["start_date"] = seg_start.strftime("%Y-%m-%d")
        cfg_y["end_date"] = seg_end.strftime("%Y-%m-%d")
        port_y, _ = run_backtest_and_return_portfolio(cfg_y)
        from backtest_engine.report.metrics import calculate_metrics

        summ_y = calculate_metrics(port_y)
        profits.append(
            (label, float(summ_y.get("net_profit_after_fees_eur", 0.0) or 0.0))
        )
        # Inclusive day count for partial-year segments
        try:
            durations_by_year[int(label)] = float(
                (seg_end.date() - seg_start.date()).days + 1
            )
        except Exception:
            pass

    # Compute stability score via centralized helper
    if not profits:
        return 1.0
    from backtest_engine.rating.stability_score import (
        compute_stability_score_from_yearly_profits,
    )

    profits_by_year: Dict[int, float] = {}
    for y, p in profits:
        try:
            profits_by_year[int(y)] = float(p)
        except Exception:
            continue
    if not profits_by_year:
        return 1.0
    return float(
        compute_stability_score_from_yearly_profits(
            profits_by_year, durations_by_year=durations_by_year
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️ Verwendung: python runner.py <pfad/zur/config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"❌ Datei nicht gefunden: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = json.load(f)
    errors = validate_config(config)
    if errors:
        for err in errors:
            print(err)
        raise ValueError("❌ Konfigurationsfehler erkannt. Backtest abgebrochen.")
    run_backtest(config)
