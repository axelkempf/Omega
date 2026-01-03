import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import numpy as np
except Exception:
    np = None

from backtest_engine.core.portfolio import Portfolio
from backtest_engine.report.metrics import calculate_metrics
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper
from hf_engine.infra.config.paths import BACKTEST_RESULTS_DIR


# -- Hedgefonds-sicherer JSON-Konverter ---------------------------------------
def _json_default(o):
    # pandas Timestamp -> ISO
    if isinstance(o, pd.Timestamp):
        # konvertiert tz-aware/naive korrekt
        try:
            return o.to_pydatetime().isoformat()
        except Exception:
            return str(o)
    # datetime / date -> ISO
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # numpy Skalar -> Python native
    if np is not None:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            # NaN/Inf -> None (oder String, je nach Policy)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    # Sets -> Listen
    if isinstance(o, (set, frozenset)):
        return list(o)
    # Letzter Fallback: String-Repräsentation
    return str(o)


def save_backtest_result(
    portfolio: Portfolio,
    config: Dict[str, Any],
    strategy_name: str,
    strategy_wrapper: Optional[StrategyWrapper] = None,
) -> None:
    """
    Speichert alle relevanten Backtest-Resultate (Summary, Trades, Equity, Meta)
    im Ordnerstruktur BACKTEST_RESULTS_DIR/strategy_symbol_tf/timestamp.

    Args:
        portfolio: Portfolio-Objekt mit allen Positionen.
        config: Verwendete Backtest-Konfiguration (Dict).
        strategy_name: Name der Strategie.
        strategy_wrapper: Optional, falls ein TradeLogger im Wrapper genutzt wird.

    Prints:
        Statusmeldung mit Speicherpfad.
    """
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    symbol = config.get("strategy", {}).get("symbol", "unknown")
    tf = (
        config.get("data", {}).get("timeframe")
        or config.get("strategy", {}).get("timeframe")
        or config.get("timeframes", {}).get("primary", "M1")
    )

    result_dir = BACKTEST_RESULTS_DIR / f"{strategy_name}_{symbol}_{tf}" / now
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary (Kennzahlen)
    summary = calculate_metrics(portfolio)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=_json_default)

    # 2. Trades (alle abgeschlossenen Trades)
    trades = [p.to_dict() for p in portfolio.closed_positions]
    with open(result_dir / "trades.json", "w") as f:
        json.dump(trades, f, indent=2, default=_json_default)

    # 3. Equity-Kurve (CSV)
    equity_data = [
        {"timestamp": timestamp.isoformat(), "equity": equity}
        for timestamp, equity in portfolio.get_equity_curve()
    ]
    pd.DataFrame(equity_data).to_csv(result_dir / "equity.csv", index=False)

    # 4. Meta-Info
    meta = {
        "strategy": strategy_name,
        "symbol": symbol,
        "config_used": config,
        "generated_at": now,
        "total_trades": len(portfolio.closed_positions),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
    }
    with open(result_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    # 5. TradeLogger speichern (falls vorhanden)
    if (
        strategy_wrapper
        and hasattr(strategy_wrapper, "trade_logger")
        and strategy_wrapper.trade_logger
    ):
        strategy_wrapper.trade_logger.save(
            strategy_name=strategy_name, symbol=symbol, timeframe=tf
        )

    print(f"✅ Ergebnis gespeichert in: {result_dir}")
