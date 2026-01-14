import csv
from pathlib import Path
from typing import Optional

from backtest_engine.core.portfolio import Portfolio

from hf_engine.infra.config.paths import TRADE_LOGS_DIR


def export_trades_to_csv(
    portfolio: Portfolio, strategy_name: str, export_dir: Optional[Path] = None
) -> None:
    """
    Exportiert alle abgeschlossenen Trades eines Portfolios als CSV-File.

    Args:
        portfolio: Portfolio-Objekt mit abgeschlossenen Trades.
        strategy_name: Name der Strategie (für Dateinamen).
        export_dir: Zielverzeichnis (Default: TRADE_LOGS_DIR).

    Prints:
        Statusmeldung nach erfolgreichem Export.
    """
    export_dir = export_dir or TRADE_LOGS_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    file_path = export_dir / f"{strategy_name}_trades.csv"

    with open(file_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Entry Time",
                "Exit Time",
                "Direction",
                "Entry",
                "Exit",
                "SL",
                "TP",
                "Result",
                "Reason",
            ]
        )
        for p in portfolio.closed_positions:
            writer.writerow(
                [
                    p.entry_time,
                    p.exit_time,
                    p.direction,
                    p.entry_price,
                    p.exit_price,
                    p.stop_loss,
                    p.take_profit,
                    p.result,
                    p.reason,
                ]
            )
    print(f"✅ CSV-Export abgeschlossen: {file_path}")
