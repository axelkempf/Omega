from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_engine.core.portfolio import Portfolio


def calculate_metrics(portfolio: Portfolio) -> Dict[str, Any]:
    """
    Berechnet die wichtigsten Metriken eines Portfolios für Reporting und Walkforward/Optimierung.
    Gibt sichere Defaults zurück, um Pipeline-Robustheit zu garantieren.
    """
    positions = portfolio.closed_positions
    partial_closed_positions = portfolio.partial_closed_positions

    # Hilfsfunktion für aktive Tage
    def _active_days_from_positions(pos_list):
        days = set()
        for p in pos_list:
            for attr in ("open_time", "close_time", "entry_time", "exit_time"):
                if hasattr(p, attr):
                    t = getattr(p, attr)
                    if t is not None:
                        try:
                            days.add(t.date())
                        except AttributeError:
                            days.add(t)
        return days

    days_pos = _active_days_from_positions(positions) | _active_days_from_positions(
        partial_closed_positions
    )
    active_days = len(days_pos)

    # Fallback über Equity (Tage mit Veränderung)
    if active_days == 0:
        try:
            eq_series = _extract_equity_series(portfolio)
            if eq_series is not None and len(eq_series) > 1:
                daily = (
                    eq_series.resample("1D").last()
                    if isinstance(eq_series.index, pd.DatetimeIndex)
                    else eq_series
                )
                rets = daily.diff().fillna(0.0)
                active_days = int((rets != 0).sum())
        except Exception:
            pass

    if not positions or len(positions) == 0:
        return {
            "net_profit_eur": 0.0,
            "net_profit_after_fees_eur": 0.0,
            "fees_total_eur": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "avg_result_eur": 0.0,
            "winrate_percent": 0.0,
            "avg_r_multiple": 0.0,
            "total_r_multiple": 0.0,
            "profit_factor": 0.0,
            "drawdown_eur": 0.0,
            "drawdown_percent": 0.0,
            "initial_drawdown_eur": 0.0,
            "total_trades": 0,
            "active_days": 0,
            "sharpe_trade": 0.0,
            "sortino_trade": 0.0,
        }

    results = [p.result for p in positions if p.result is not None]
    r_values = [p.r_multiple for p in positions if p.r_multiple is not None]
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r < 0]

    partial_results = [
        p.result for p in partial_closed_positions if p.result is not None
    ]
    partial_wins = [r for r in partial_results if r > 0]
    partial_losses = [r for r in partial_results if r < 0]

    gross_profit = sum(wins) + sum(partial_wins)
    gross_loss = abs(sum(losses) + sum(partial_losses))
    net = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    total_r = sum(r_values)
    avg_r = total_r / len(r_values) if r_values else 0.0
    winrate = round(len(wins) / len(results) * 100, 2) if results else 0.0

    combined_results = results + partial_results
    avg_count = len(combined_results)
    avg_pnl = (sum(combined_results) / avg_count) if avg_count > 0 else 0.0

    # NEW -------- Sharpe/Sortino (per Trade, auf R-Multiples) ----------
    def _sharpe_sortino_from_r(r_list, *, risk_free: float = 0.0, mar: float = 0.0):
        """
        Berechnet Sharpe- und Sortino-Ratio auf *pro-Trade*-Basis aus R-Multiples.
        - risk_free: risikofreier Return pro Trade (auf R-Basis, i.d.R. 0)
        - mar: Minimum Acceptable Return für Sortino (auf R-Basis, i.d.R. 0)
        Hinweis: Keine Annualisierung – Kennzahlen gelten pro Trade.
        """
        # Filter: None/NaN/Inf entfernen
        arr = np.asarray([x for x in r_list if x is not None], dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            # Zu wenig Daten für sinnvolle Streuung
            mu_single = float((arr - risk_free).mean()) if arr.size == 1 else 0.0
            return 0.0, 0.0

        # Excess-Return relativ zu risk-free (pro Trade)
        excess = arr - risk_free
        mu = excess.mean()

        # Sharpe: Stichproben-Std (ddof=1). Kein 1e-12-Hack: bei sigma=0 -> 0.0
        sigma = excess.std(ddof=1)
        sharpe = float(mu / sigma) if sigma > 0.0 else 0.0

        # Sortino: Semideviation relativ zu MAR über die *gesamte* Stichprobe
        # downside = min(0, r - MAR)
        downside_diff = np.minimum(excess - mar, 0.0)
        # Semideviation = sqrt(mean(downside^2))  -> Teiler N, nicht N-1
        semi_dev = float(np.sqrt(np.mean(downside_diff**2)))
        sortino = float(mu / semi_dev) if semi_dev > 0.0 else 0.0

        return sharpe, sortino

    sharpe_trade, sortino_trade = _sharpe_sortino_from_r(
        r_values, risk_free=0.0, mar=0.0
    )

    # --- Drawdown, Drawdown % und Initial-Drawdown robust berechnen ---
    eq_series = _extract_equity_series(portfolio)
    # Fallback: Equity-Kurve aus Trades rekonstruieren, falls zu kurz/leer
    try:
        if (eq_series is None) or (len(eq_series) < 2):
            trades_curve = getattr(portfolio, "get_equity_curve", None)
            if callable(trades_curve):
                tc = trades_curve()
                if tc and len(tc) >= 2:
                    ts = pd.to_datetime([t[0] for t in tc], errors="coerce")
                    vals = pd.to_numeric([t[1] for t in tc], errors="coerce")
                    eq_series = pd.Series(vals, index=ts).dropna()
    except Exception:
        pass

    dd_eur = round(getattr(portfolio, "max_drawdown", 0.0), 2)
    init_eq = getattr(
        portfolio, "initial_equity", getattr(portfolio, "initial_balance", None)
    )

    # Fallback: aus Equity-Kurve max Drawdown berechnen
    def _max_drawdown_from_series(s):
        import numpy as np

        if s is None or len(s) < 2:
            return 0.0
        vals = s.astype(float).values
        run_max = np.maximum.accumulate(vals)
        dd = run_max - vals
        return float(np.max(dd))

    try:
        if (dd_eur == 0.0 or not np.isfinite(dd_eur)) and eq_series is not None:
            dd_eur = round(_max_drawdown_from_series(eq_series), 2)
    except Exception:
        pass

    # Drawdown %: robust mit init_eq, sonst aus Peak->Trough relativ
    dd_pct = 0.0
    try:
        if init_eq not in (None, 0):
            dd_pct = float(round((dd_eur / float(init_eq)) * 100.0, 2))
        elif eq_series is not None and len(eq_series) > 1:
            peak = float(eq_series.max())
            dd_pct = float(round((dd_eur / peak) * 100.0, 2)) if peak > 0 else 0.0
    except Exception:
        dd_pct = 0.0

    # Initial Drawdown: Abstand von initial_equity zum Minimum
    initial_drawdown_eur = 0.0
    try:
        if init_eq not in (None, 0):
            if eq_series is not None and len(eq_series) > 0:
                min_eq = float(eq_series.min())
                initial_drawdown_eur = float(max(0.0, init_eq - min_eq))
            else:
                # Fallback auf vom Portfolio gepflegten Wert
                initial_drawdown_eur = float(
                    getattr(portfolio, "initial_max_drawdown", 0.0)
                )
    except Exception:
        pass

    return {
        "net_profit_eur": round(net, 2),
        "net_profit_after_fees_eur": round(
            getattr(portfolio, "cash", 0.0)
            - getattr(portfolio, "initial_balance", 0.0),
            2,
        ),
        "fees_total_eur": round(getattr(portfolio, "total_fees", 0.0), 2),
        "gross_profit_eur": round(gross_profit, 2),
        "gross_loss_eur": round(gross_loss, 2),
        "avg_result_eur": round(avg_pnl, 2),
        "winrate_percent": winrate,
        "avg_r_multiple": round(avg_r, 3),
        "total_r_multiple": round(total_r, 3),
        "profit_factor": round(profit_factor, 2),
        "drawdown_eur": round(dd_eur, 2),
        "drawdown_percent": dd_pct,
        "initial_drawdown_eur": round(initial_drawdown_eur, 2),
        "total_trades": len(positions),
        "active_days": active_days,
        "sharpe_trade": round(sharpe_trade, 3),
        "sortino_trade": round(sortino_trade, 3),
    }


# --- NEU: flexible Equity-Kurven-Extraktion ---
def _extract_equity_series(portfolio):
    import pandas as pd

    eq = getattr(portfolio, "equity_curve", None)
    if eq is None:
        return None
    # Unterstütze: pd.Series (mit/ohne DatetimeIndex), DataFrame mit time/equity,
    # Liste/Tuples [(ts, equity)] oder nur [equity,...]
    try:
        # 1) Series
        if isinstance(eq, pd.Series):
            if isinstance(eq.index, pd.DatetimeIndex):
                return eq.astype(float)
            s = pd.Series(eq).astype(float)
            s.index = pd.date_range("2000-01-01", periods=len(s), freq="D")
            return s
        # 2) DataFrame mit 'time'/'timestamp' und 'equity' Spalten
        if isinstance(eq, pd.DataFrame):
            cols = {c.lower(): c for c in eq.columns}
            tcol = cols.get("time") or cols.get("timestamp") or cols.get("datetime")
            ecol = cols.get("equity") or cols.get("balance") or cols.get("equity_value")
            if tcol and ecol:
                s = pd.Series(
                    pd.to_numeric(eq[ecol], errors="coerce").astype(float).values,
                    index=pd.to_datetime(eq[tcol]),
                )
                return s.dropna()
            # Fallback: erste Spalte Index, zweite Werte
            if len(eq.columns) >= 2:
                s = pd.Series(
                    pd.to_numeric(eq.iloc[:, 1], errors="coerce").astype(float).values,
                    index=pd.to_datetime(eq.iloc[:, 0], errors="coerce"),
                )
                return s.dropna()
        # 3) Liste von (ts, value) / Liste von values
        if isinstance(eq, (list, tuple)) and len(eq) > 0:
            first = eq[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                idx = pd.to_datetime([r[0] for r in eq], errors="coerce")
                vals = pd.to_numeric([r[1] for r in eq], errors="coerce")
                s = pd.Series(vals, index=idx).dropna()
                return s
            # Nur Werte → künstlicher Tagesindex
            vals = pd.to_numeric(pd.Series(eq), errors="coerce").astype(float)
            s = pd.Series(
                vals.values,
                index=pd.date_range("2000-01-01", periods=len(vals), freq="D"),
            )
            return s.dropna()
    except Exception:
        return None
    return None
