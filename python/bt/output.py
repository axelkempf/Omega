"""Artifact output writer."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

NS_PER_SECOND = 1_000_000_000
EQUITY_FIELDS = [
    "timestamp",
    "timestamp_ns",
    "equity",
    "balance",
    "drawdown",
    "high_water",
]


def write_artifacts(result: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write output artifacts to a directory.

    Args:
        result: Backtest result dictionary.
        output_dir: Target output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    now_dt = datetime.now(timezone.utc)
    now_ns = int(now_dt.timestamp() * NS_PER_SECOND)

    meta = _build_meta(result, now_dt, now_ns)
    write_json(output_path / "meta.json", meta)

    trades = result.get("trades") or []
    write_json(output_path / "trades.json", _normalize_trades(trades))

    metrics_output = {
        "metrics": result.get("metrics") or {},
        "definitions": result.get("metric_definitions") or {},
    }
    write_json(output_path / "metrics.json", metrics_output)

    equity = result.get("equity_curve") or []
    write_equity_csv(output_path / "equity.csv", equity)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_equity_csv(path: Path, equity: Iterable[Mapping[str, Any]]) -> None:
    """Write equity curve as CSV."""
    rows = list(equity)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EQUITY_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for point in rows:
            ts_ns = _coerce_ns(point.get("timestamp_ns"))
            row = {
                "timestamp": _iso_from_ns(ts_ns),
                "timestamp_ns": ts_ns,
                "equity": point.get("equity", 0),
                "balance": point.get("balance", 0),
                "drawdown": point.get("drawdown", 0),
                "high_water": point.get("high_water", 0),
            }
            writer.writerow(row)


def _normalize_trades(trades: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ensure trade entries include ISO timestamps."""
    normalized: list[dict[str, Any]] = []
    for trade in trades:
        item = dict(trade)
        entry_ns = item.get("entry_time_ns")
        if "entry_time" not in item and entry_ns is not None:
            item["entry_time"] = _iso_from_ns(entry_ns)
        exit_ns = item.get("exit_time_ns")
        if "exit_time" not in item and exit_ns is not None:
            item["exit_time"] = _iso_from_ns(exit_ns)
        normalized.append(item)
    return normalized


def _iso_from_ns(timestamp_ns: Any) -> str:
    """Convert nanosecond epoch to ISO-8601 UTC."""
    ts_ns = _coerce_ns(timestamp_ns)
    return datetime.fromtimestamp(ts_ns / NS_PER_SECOND, tz=timezone.utc).isoformat()


def _coerce_ns(value: Any) -> int:
    """Coerce timestamp input into an integer nanoseconds value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_meta(
    result: Mapping[str, Any],
    generated_at: datetime,
    generated_at_ns: int,
) -> dict[str, Any]:
    meta_src = _as_dict(result.get("meta"))
    extra = _as_dict(meta_src.get("extra"))

    run_id = extra.get("run_id") or meta_src.get("run_id") or "unknown"
    engine = _as_dict(extra.get("engine"))
    if "name" not in engine:
        engine["name"] = "omega-v2"

    config_meta = _as_dict(extra.get("config"))
    config_hash = config_meta.get("hash") or extra.get("config_hash") or "unknown"
    config_meta["hash"] = config_hash

    dataset_meta = _as_dict(extra.get("dataset"))
    start_ns = _coerce_ns(
        meta_src.get("start_timestamp") or dataset_meta.get("start_time_ns")
    )
    end_ns = _coerce_ns(
        meta_src.get("end_timestamp") or dataset_meta.get("end_time_ns")
        )
    dataset_out: dict[str, Any] = {
        "symbol": dataset_meta.get("symbol", ""),
        "timeframe": dataset_meta.get("timeframe", ""),
        "start_time": _iso_from_ns(start_ns),
        "start_time_ns": start_ns,
        "end_time": _iso_from_ns(end_ns),
        "end_time_ns": end_ns,
        "manifest_sha256": dataset_meta.get("manifest_sha256", "unknown"),
    }
    if "manifest_ref" in dataset_meta:
        dataset_out["manifest_ref"] = dataset_meta["manifest_ref"]
    if "governance" in dataset_meta:
        dataset_out["governance"] = dataset_meta["governance"]

    meta_out: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "generated_at_ns": generated_at_ns,
        "engine": engine,
        "config": config_meta,
        "dataset": dataset_out,
    }

    account = _as_dict(extra.get("account"))
    if account:
        meta_out["account"] = account

    git = extra.get("git")
    if isinstance(git, Mapping):
        meta_out["git"] = dict(git)

    return meta_out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
