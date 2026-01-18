#!/usr/bin/env python3
"""Regenerate Omega V2 golden fixtures.

Usage:
  python scripts/generate_v2_golden.py
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_root = repo_root / "python" / "tests" / "fixtures" / "golden"
    data_root = repo_root / "python" / "tests" / "fixtures" / "data" / "parquet"

    if not fixture_root.exists():
        print(f"Missing golden fixture root: {fixture_root}")
        return 1
    if not data_root.exists():
        print(f"Missing fixture data root: {data_root}")
        return 1

    os.environ["OMEGA_DATA_PARQUET_ROOT"] = str(data_root)

    try:
        from bt import run_backtest
    except Exception as exc:
        print("Failed to import bt; run `maturin develop` first.")
        print(str(exc))
        return 1

    scenario_dirs = sorted([p for p in fixture_root.iterdir() if p.is_dir()])
    if not scenario_dirs:
        print("No scenario directories found.")
        return 1

    for scenario_dir in scenario_dirs:
        config_path = scenario_dir / "config.json"
        if not config_path.exists():
            print(f"Skipping {scenario_dir.name}: missing config.json")
            continue
        print(f"Generating {scenario_dir.name}...")
        run_backtest(config_path=config_path, output_dir=scenario_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
