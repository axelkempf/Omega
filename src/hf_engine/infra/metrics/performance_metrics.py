"""
Berechnet Performance-Kennzahlen aus dem Trade-Log.
Kennzahlen: Trefferquote, Profit-Faktor, Drawdown, Sharpe Ratio etc.
"""

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from hf_engine.infra.config.paths import TRADE_LOG_CSV


def load_trade_log() -> pd.DataFrame:
    """
    Lädt das Trade-Log als DataFrame, angereichert mit Tages-Datum.
    """
    try:
        df = pd.read_csv(TRADE_LOG_CSV, parse_dates=["datetime"])
        if df.empty or df is None:
            print("⚠️ Trade-Log ist leer.")
            return pd.DataFrame()

        df["date"] = df["datetime"].dt.date
        return df

    except FileNotFoundError:
        print("⚠️ Trade-Log-Datei nicht gefunden.")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Fehler beim Laden der Trade-Log-Datei: {e}")
        return pd.DataFrame()


def calculate_metrics(df: pd.DataFrame, start_capital: float = 100_000) -> dict:
    """
    Berechnet Performance-Metriken aus dem DataFrame.
    """
    total_trades = len(df)
    if total_trades == 0:
        return {}

    wins = df[df["profit_abs"] > 0]
    losses = df[df["profit_abs"] < 0]

    win_rate = len(wins) / total_trades
    avg_profit = df["profit_abs"].mean()
    total_profit = df["profit_abs"].sum()
    profit_factor = (
        wins["profit_abs"].sum() / abs(losses["profit_abs"].sum())
        if not losses.empty
        else np.inf
    )

    df["equity"] = df["profit_abs"].cumsum()
    df["drawdown"] = df["equity"] - df["equity"].cummax()
    max_drawdown = df["drawdown"].min()

    daily_returns = df.groupby("date")["profit_abs"].sum() / start_capital
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std()
        else np.nan
    )

    return {
        "Anzahl Trades": total_trades,
        "Trefferquote (%)": round(win_rate * 100, 2),
        "Ø Profit": round(avg_profit, 2),
        "Gesamtprofit": round(total_profit, 2),
        "Profitfaktor": round(profit_factor, 2),
        "Max Drawdown": round(max_drawdown, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
    }


def print_summary_by(
    group_field: str, title: str, start_capital: float = 100_000
) -> None:
    """
    Gibt Metriken je Gruppe (z. B. Strategie, Symbol) im Terminal aus.
    """
    df = load_trade_log()
    if df.empty or df is None or group_field not in df.columns:
        print(f"⚠️ Keine Daten oder Spalte '{group_field}' nicht gefunden.")
        return

    print(f"\n📊 PERFORMANCE NACH {title.upper()}")
    print("-" * 45)

    for value in sorted(df[group_field].dropna().unique()):
        sub_df = df[df[group_field] == value]
        metrics = calculate_metrics(sub_df, start_capital=start_capital)
        print(f"\n🔹 {title}: {value}")
        for key, val in metrics.items():
            print(f"{key}: {val}")


if __name__ == "__main__":
    print_summary_by("strategy", "Strategie")
    print_summary_by("symbol", "Symbol")
