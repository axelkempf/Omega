# backtest_engine/batch_runner.py

import json
import multiprocessing
import os
from typing import List

from backtest_engine.runner import run_backtest
from configs.backtest._config_validator import validate_config


def worker(config_path: str) -> None:
    """
    Arbeiterprozess für einen einzelnen Backtest.

    Args:
        config_path (str): Pfad zur JSON-Konfigurationsdatei.
    """
    print(f"🚀 Starte Backtest: {config_path}")
    if not os.path.isfile(config_path):
        print(f"❌ Datei nicht gefunden: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    errors = validate_config(config)
    if errors:
        print(f"❌ Fehler in {config_path}:")
        for err in errors:
            print("   " + err)
        return

    try:
        run_backtest(config)
    except Exception as e:
        print(f"❌ Fehler bei {config_path}: {e}")


def run_batch(config_paths: List[str], max_workers: int = 4) -> None:
    """
    Führt mehrere Backtests parallel aus (Multiprocessing).

    Args:
        config_paths (List[str]): Liste von Pfaden zu JSON-Konfigurationsdateien.
        max_workers (int): Maximale Anzahl paralleler Prozesse.
    """
    processes: List[multiprocessing.Process] = []

    for path in config_paths:
        p = multiprocessing.Process(target=worker, args=(path,))
        processes.append(p)
        p.start()

        if len(processes) >= max_workers:
            for proc in processes:
                proc.join()
            processes.clear()

    # Schließe verbleibende Prozesse ab
    for proc in processes:
        proc.join()
