import os
from datetime import datetime


def validate_config(config: dict) -> list[str]:
    errors = []

    # Pflichtfelder
    if "start_date" not in config:
        errors.append("🟥 'start_date' fehlt")
    else:
        try:
            datetime.strptime(config["start_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("🟥 'start_date' muss Format 'YYYY-MM-DD' haben")

    if "end_date" not in config:
        errors.append("🟥 'end_date' fehlt")
    else:
        try:
            datetime.strptime(config["end_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("🟥 'end_date' muss Format 'YYYY-MM-DD' haben")

    # Backtest-Modus
    mode = config.get("mode", "candle")
    if mode not in ["candle", "tick"]:
        errors.append("🟥 'mode' muss 'tick' oder 'candle' sein")

    # Datenpfad prüfen
    data_path = config.get("data", {}).get("path")
    if not data_path:
        errors.append("🟥 'data.path' fehlt")
    elif not os.path.isdir(data_path):
        errors.append(f"🟥 Datenpfad nicht gefunden: {data_path}")

    # Strategie(n)
    if "strategy" not in config and "strategies" not in config:
        errors.append("🟥 Weder 'strategy' noch 'strategies' definiert")

    # Gebühren/Slippage optional, aber prüfen wenn vorhanden
    if "fees" in config:
        if not isinstance(config["fees"].get("per_million", 0), (int, float)):
            errors.append("🟥 'fees.per_million' muss eine Zahl sein")

    if "slippage" in config:
        for key in ["fixed_pips", "random_pips"]:
            if key in config["slippage"] and not isinstance(
                config["slippage"][key], (int, float)
            ):
                errors.append(f"🟥 'slippage.{key}' muss eine Zahl sein")

    return errors
