from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from backtest_engine.core.portfolio import PortfolioPosition
from backtest_engine.data.candle import Candle
from matplotlib.patches import Rectangle


def plot_trades_on_candles(
    candles: List[Candle], trades: List[PortfolioPosition], title: str = "Trade Overlay"
) -> None:
    """
    Visualisiert Trades als Overlay auf Candlestick-Chart.

    Args:
        candles: Liste von Candle-Objekten (möglichst homogenes Intervall, z. B. 1min).
        trades: Liste von PortfolioPosition-Objekten (Ein- und Ausstiege werden markiert).
        title: Plot-Titel.

    Shows:
        Matplotlib-Figur mit Overlays für Ein-/Ausstieg, SL/TP und Trade-Pfade.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Candlesticks zeichnen
    for c in candles:
        color = "green" if c.close >= c.open else "red"
        ax.plot(
            [c.timestamp, c.timestamp], [c.low, c.high], color="black", linewidth=0.5
        )
        ax.add_patch(
            Rectangle(
                (c.timestamp, min(c.open, c.close)),
                width=pd.Timedelta(minutes=1),
                height=abs(c.close - c.open),
                color=color,
                alpha=0.6,
            )
        )

    # Trades einzeichnen
    for t in trades:
        color = "green" if getattr(t, "result", 0) > 0 else "red"
        # Entry/Exit markieren
        ax.plot(
            [t.entry_time],
            [t.entry_price],
            marker="^",
            color=color,
            markersize=8,
            label="Entry",
        )
        ax.plot(
            [t.exit_time],
            [t.exit_price],
            marker="x",
            color=color,
            markersize=8,
            label="Exit",
        )
        # Linie vom Entry zum Exit
        ax.plot(
            [t.entry_time, t.exit_time],
            [t.entry_price, t.exit_price],
            linestyle="--",
            color=color,
            alpha=0.5,
        )
        # SL/TP als horizontale Linien
        ax.axhline(t.stop_loss, color="gray", linestyle=":", linewidth=0.5)
        ax.axhline(t.take_profit, color="gray", linestyle=":", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis")
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
