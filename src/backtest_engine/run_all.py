# run_all.py

from backtest_engine.batch_runner import run_batch


def main() -> None:
    """
    Startet Batch-Backtest für alle angegebenen Konfigurationen.
    """
    configs = [
        "batch_configs/eurusd_2023.json",
        "batch_configs/eurusd_2024.json",
        "batch_configs/gbpusd_2023.json",
    ]
    run_batch(configs, max_workers=2)


if __name__ == "__main__":
    main()
