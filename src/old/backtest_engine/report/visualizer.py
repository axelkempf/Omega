from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from backtest_engine.core.portfolio import PortfolioPosition

from hf_engine.infra.config.paths import TRADE_LOGS_DIR


def plot_equity_curve(
    positions: List[PortfolioPosition],
    strategy_name: str,
    symbol: str,
    initial_balance: float = 10000.0,
) -> None:
    """
    Erstellt und speichert die Equity-Kurve als PNG.

    Args:
        positions: Liste abgeschlossener PortfolioPosition-Objekte (müssen .exit_time und .result haben).
        strategy_name: Name der Strategie (für Dateinamen).
        symbol: Symbol (für Dateinamen).
        initial_balance: Startkontostand (default: 10000).

    Prints:
        Exportpfad nach erfolgreichem Speichern.
    """
    times = []
    equity = []
    current = initial_balance

    for p in positions:
        times.append(p.exit_time)
        current += p.result
        equity.append(current)

    plt.figure(figsize=(12, 5))
    plt.plot(times, equity, label="Equity", linewidth=2)
    plt.title("📈 Equity Curve")
    plt.xlabel("Zeit")
    plt.ylabel("Kontostand")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = TRADE_LOGS_DIR / f"{strategy_name}_{symbol}_equity_{timestamp}.png"
    plt.savefig(file_path)
    plt.close()
    print(f"✅ Equity-Curve gespeichert: {file_path}")


def plot_r_multiples(
    positions: List[PortfolioPosition], strategy_name: str, symbol: str
) -> None:
    """
    Erstellt und speichert einen Barplot der R-Multiples für alle Trades.

    Args:
        positions: Liste abgeschlossener PortfolioPosition-Objekte (müssen .result, .entry_price, .stop_loss haben).
        strategy_name: Name der Strategie (für Dateinamen).
        symbol: Symbol (für Dateinamen).

    Prints:
        Exportpfad nach erfolgreichem Speichern.
    """
    r_values = []
    labels = []

    for i, p in enumerate(positions):
        risk = abs(p.entry_price - p.stop_loss)
        if risk > 0:
            r = p.result / risk
            r_values.append(r)
            labels.append(f"{i+1}")

    plt.figure(figsize=(12, 4))
    plt.bar(labels, r_values, color=["green" if r > 0 else "red" for r in r_values])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("🎯 R-Multiples pro Trade")
    plt.xlabel("Trade #")
    plt.ylabel("R")
    plt.xticks(rotation=90)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = TRADE_LOGS_DIR / f"{strategy_name}_{symbol}_r_multiples_{timestamp}.png"
    plt.savefig(file_path)
    plt.close()
    print(f"✅ R-Multiples Plot gespeichert: {file_path}")
