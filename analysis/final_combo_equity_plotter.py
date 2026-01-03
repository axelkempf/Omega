"""
Final Combo Equity Plotter

Dieses Skript lädt die Top-10 finalen Kombinationen aus der combined_walkforward_matrix_analyzer
und führt für jede finale Kombination die Equity-Curve-Visualisierung und KPI-Berechnung durch.

Workflow:
1. Lade top_10_final_combo_pairs.csv
2. Für jede finale combo_pair_id:
   - Lade equity.csv und trades.json aus final_combos/<id>/
   - Berechne KPIs (analog zu combine_equity_curves.py)
   - Erstelle Equity-Plot mit Drawdown
   - Erstelle Trade-Distribution-Plots
   - Speichere Ergebnisse im gleichen Ordner

Features:
- Wiederverwendung der bewährten Logik aus combine_equity_curves.py
- Automatische Verarbeitung aller Top-10 Kombinationen
- Harmonische Integration mit der Matrix-Analyse
- Debug-Logging zur Validierung
- Robust gegen fehlende oder inkonsistente Dateien

Autor: AI Agent
Datum: 2025-12-01
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

# Pfade relativ zum Projektroot
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WALKFORWARD_ROOT = PROJECT_ROOT / "var" / "results" / "analysis"
COMBINED_MATRIX_DIR = WALKFORWARD_ROOT / "combined_matrix"
FINAL_COMBOS_DIR = COMBINED_MATRIX_DIR / "final_combos"
# NEU (Dez 2025): Nutze final_combos_categorical_champions.csv mit 11 Champion-Kategorien
# (enthält die unique final_combo_pair_ids mit kategorialen Scores)
CHAMPIONS_CSV = COMBINED_MATRIX_DIR / "final_combos_categorical_champions.csv"


def detect_equity_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Findet die Timestamp- und Equity-Spalten im DataFrame.

    Args:
        df: DataFrame mit Equity-Daten

    Returns:
        (timestamp_column, equity_column)

    Raises:
        ValueError: Wenn benötigte Spalten nicht gefunden werden
    """
    ts_candidates = [
        "timestamp",
        "time",
        "date",
        "datetime",
        "Date",
        "Timestamp",
    ]
    eq_candidates = [
        "equity",
        "Equity",
        "equity_value",
        "equity_total",
        "value",
    ]

    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"Keine Zeitspalte gefunden. Verfügbare Spalten: {list(df.columns)}"
        )

    eq_col = next((c for c in eq_candidates if c in df.columns), None)
    if eq_col is None:
        raise ValueError(
            f"Keine Equity-Spalte gefunden. Verfügbare Spalten: {list(df.columns)}"
        )

    return ts_col, eq_col


def load_equity_series(path: Path) -> Optional[pd.Series]:
    """
    Lädt eine Equity-CSV und gibt sie als Zeitreihe zurück.

    Args:
        path: Pfad zur equity.csv

    Returns:
        pd.Series mit Timestamp-Index und Equity-Werten oder None bei Fehler
    """
    try:
        if not path.exists():
            print(f"[Load] Warnung: Equity-Datei nicht gefunden: {path}")
            return None

        df = pd.read_csv(path)
        ts_col, eq_col = detect_equity_columns(df)

        # Parse Timestamps
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        eq_vals = pd.to_numeric(df[eq_col], errors="coerce")

        # Nur gültige Zeilen behalten
        mask = ts.notna() & eq_vals.notna()
        if not mask.any():
            print(f"[Load] Warnung: Keine gültigen Daten in {path}")
            return None

        series = pd.Series(eq_vals[mask].values, index=ts[mask])
        series = series.sort_index()

        # Duplikate entfernen (letzten Wert behalten)
        series = series[~series.index.duplicated(keep="last")]

        return series

    except Exception as e:
        print(f"[Load] Fehler beim Laden von {path}: {e}")
        return None


def load_trades_data(path: Path) -> Optional[pd.DataFrame]:
    """
    Lädt trades.json und gibt sie als DataFrame zurück.

    Args:
        path: Pfad zur trades.json

    Returns:
        pd.DataFrame mit Trade-Daten oder None bei Fehler
    """
    try:
        if not path.exists():
            print(f"[Load] Warnung: Trades-Datei nicht gefunden: {path}")
            return None

        with path.open("r", encoding="utf-8") as f:
            trades = json.load(f)

        if not isinstance(trades, list) or not trades:
            print(f"[Load] Warnung: Leere oder ungültige Trades-Datei: {path}")
            return None

        df = pd.DataFrame(trades)

        # Parse Timestamps
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        return df

    except Exception as e:
        print(f"[Load] Fehler beim Laden von {path}: {e}")
        return None


def compute_kpis(
    equity_series: pd.Series,
    trades_df: Optional[pd.DataFrame],
    baseline: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Berechnet KPIs aus Equity-Serie und Trades.

    Args:
        equity_series: Zeitreihe der Equity
        trades_df: DataFrame mit Trades (optional)
        baseline: Startkapital

    Returns:
        Dict mit berechneten KPIs
    """
    kpis = {}

    # Equity-basierte Metriken
    if equity_series is not None and not equity_series.empty:
        # Total Profit
        if len(equity_series) >= 2:
            total_profit = float(equity_series.iloc[-1] - equity_series.iloc[0])
            kpis["total_profit"] = total_profit
        else:
            kpis["total_profit"] = 0.0

        # Max Drawdown
        roll_max = equity_series.cummax()
        dd = roll_max - equity_series
        max_dd = float(dd.max())
        kpis["max_drawdown"] = max_dd

        # Drawdown Date
        if max_dd > 0:
            max_dd_ts = dd.idxmax()
            kpis["max_drawdown_date"] = max_dd_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            kpis["max_drawdown_date"] = ""

        # Profit over Drawdown
        if max_dd > 0:
            pod = kpis["total_profit"] / max_dd
            kpis["profit_over_dd"] = float(pod)
        else:
            kpis["profit_over_dd"] = float("inf") if kpis["total_profit"] > 0 else 0.0

        # Yearly Profits
        years = sorted(pd.unique(equity_series.index.year))
        kpis["yearly_profits"] = {}

        for year in years:
            year_mask = equity_series.index.year == year
            year_equity = equity_series[year_mask]

            if len(year_equity) >= 2:
                year_profit = float(year_equity.iloc[-1] - year_equity.iloc[0])
                kpis["yearly_profits"][int(year)] = year_profit

        # Startdatum und Enddatum
        kpis["start_date"] = equity_series.index[0].strftime("%Y-%m-%d")
        kpis["end_date"] = equity_series.index[-1].strftime("%Y-%m-%d")

        # Dauer in Tagen
        duration_days = (equity_series.index[-1] - equity_series.index[0]).days
        kpis["duration_days"] = int(duration_days)

    else:
        kpis["total_profit"] = 0.0
        kpis["max_drawdown"] = 0.0
        kpis["max_drawdown_date"] = ""
        kpis["profit_over_dd"] = 0.0
        kpis["yearly_profits"] = {}
        kpis["start_date"] = ""
        kpis["end_date"] = ""
        kpis["duration_days"] = 0

    # Trade-basierte Metriken
    if trades_df is not None and not trades_df.empty:
        kpis["total_trades"] = len(trades_df)

        # Winrate
        if "result" in trades_df.columns:
            results = pd.to_numeric(trades_df["result"], errors="coerce")
            wins = (results > 0).sum()
            total_valid = results.notna().sum()
            if total_valid > 0:
                kpis["winrate"] = float(wins / total_valid * 100)
            else:
                kpis["winrate"] = 0.0
        else:
            kpis["winrate"] = np.nan

        # Average R-Multiple
        if "r_multiple" in trades_df.columns:
            r_vals = pd.to_numeric(trades_df["r_multiple"], errors="coerce")
            if not r_vals.isna().all():
                kpis["avg_r"] = float(r_vals.mean())
            else:
                kpis["avg_r"] = np.nan
        else:
            kpis["avg_r"] = np.nan

        # Average Trade Duration
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            entry_times = pd.to_datetime(
                trades_df["entry_time"], utc=True, errors="coerce"
            )
            exit_times = pd.to_datetime(
                trades_df["exit_time"], utc=True, errors="coerce"
            )
            durations = (exit_times - entry_times).dt.total_seconds() / 3600  # Stunden
            valid_durations = durations[durations.notna() & (durations > 0)]
            if not valid_durations.empty:
                kpis["avg_trade_duration_hours"] = float(valid_durations.mean())
            else:
                kpis["avg_trade_duration_hours"] = np.nan
        else:
            kpis["avg_trade_duration_hours"] = np.nan

        # Commission (wenn vorhanden)
        if "commission" in trades_df.columns:
            commissions = pd.to_numeric(trades_df["commission"], errors="coerce")
            kpis["total_commission"] = float(commissions.sum())
        else:
            kpis["total_commission"] = np.nan

    else:
        kpis["total_trades"] = 0
        kpis["winrate"] = np.nan
        kpis["avg_r"] = np.nan
        kpis["avg_trade_duration_hours"] = np.nan
        kpis["total_commission"] = np.nan

    return kpis


def _resample_closes(eq: pd.Series, freq: str) -> pd.Series:
    """Hilfsfunktion: resamplet die Equity-Serie und nimmt den letzten Wert je Periode."""
    return eq.sort_index().resample(freq).last().dropna()


def _max_drawdown_vs_prev_period(
    eq: pd.Series,
    resample_freq: str,
    period_freq: Optional[str] = None,
) -> Tuple[float, Optional[pd.Timestamp]]:
    """
    Maximaler negativer Intraperioden-Drawdown relativ zur Equity des jeweils
    vorherigen Zeitraums. Logik gespiegelt aus analysis/combine_equity_curves.py.
    """
    if period_freq is None:
        period_freq = resample_freq

    closes = _resample_closes(eq, resample_freq)
    if len(closes) < 2:
        return np.nan, None

    try:
        closes_period = closes.copy()
        closes_period.index = (
            closes_period.index.tz_convert(None).to_period(period_freq)
            if hasattr(closes_period.index, "tz")
            else closes_period.index.to_period(period_freq)
        )
    except Exception:
        return np.nan, None

    prev_closes = closes_period.shift(1)

    # Erste Periode gegen Start-Equity referenzieren
    if len(prev_closes):
        first_val = prev_closes.iloc[0]
        if pd.isna(first_val) and len(eq):
            prev_closes.iloc[0] = float(eq.iloc[0])

    df = pd.DataFrame({"equity": eq})
    try:
        idx = df.index
        if hasattr(idx, "tz"):
            idx = idx.tz_convert(None)
        df["period"] = idx.to_period(period_freq)
    except Exception:
        return np.nan, None

    mapping = prev_closes.to_dict()
    df["prev_close"] = df["period"].map(mapping)
    df = df.dropna(subset=["prev_close"])
    if df.empty:
        return np.nan, None

    rel = df["equity"] - df["prev_close"]
    if rel.empty:
        return np.nan, None

    idx_min = rel.idxmin()
    try:
        dd_val = float(rel.loc[idx_min])
    except Exception:
        return np.nan, None

    return dd_val, idx_min


def write_expected_profit_summary_single(
    equity_series: pd.Series, out_csv: Path
) -> None:
    """
    Schreibt expected_profit_summary.csv für eine einzelne Equity-Serie.

    Spalten: timeframe, mean_abs_profit, worst_abs_profit, worst_abs_profit_date,
    max_drawdown, max_drawdown_date
    """
    if equity_series is None or equity_series.empty:
        # leere Datei mit Header schreiben
        df_empty = pd.DataFrame(
            columns=[
                "timeframe",
                "mean_abs_profit",
                "worst_abs_profit",
                "worst_abs_profit_date",
                "max_drawdown",
                "max_drawdown_date",
            ]
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_empty.to_csv(out_csv, index=False)
        return

    eq = equity_series.sort_index()
    specs = [
        ("daily", "D", "D"),
        ("weekly", "W-FRI", "W-FRI"),
        ("monthly", "ME", "M"),
        ("yearly", "YE-DEC", "Y-DEC"),
    ]

    rows = []
    for name, resample_freq, period_freq in specs:
        closes = _resample_closes(eq, resample_freq)
        worst_profit_date_str = ""

        if len(closes) >= 2:
            chg = closes.diff().dropna()
            mean_profit = float(chg.mean())
            worst_profit = float(chg.min())
            try:
                worst_ts = chg.idxmin()
                if isinstance(worst_ts, pd.Timestamp):
                    ts_utc = worst_ts.tz_convert("UTC") if worst_ts.tzinfo else worst_ts
                    worst_profit_date_str = ts_utc.date().isoformat()
                else:
                    worst_profit_date_str = str(worst_ts)
            except Exception:
                worst_profit_date_str = ""
        else:
            mean_profit = np.nan
            worst_profit = np.nan

        max_dd, max_dd_ts = _max_drawdown_vs_prev_period(
            eq, resample_freq, period_freq=period_freq
        )
        max_dd_date_str = ""
        if max_dd_ts is not None:
            try:
                if isinstance(max_dd_ts, pd.Timestamp):
                    ts_utc = (
                        max_dd_ts.tz_convert("UTC") if max_dd_ts.tzinfo else max_dd_ts
                    )
                    max_dd_date_str = ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    max_dd_date_str = str(max_dd_ts)
            except Exception:
                max_dd_date_str = ""

        rows.append(
            {
                "timeframe": name,
                "mean_abs_profit": mean_profit,
                "worst_abs_profit": worst_profit,
                "worst_abs_profit_date": worst_profit_date_str,
                "max_drawdown": max_dd,
                "max_drawdown_date": max_dd_date_str,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "timeframe",
            "mean_abs_profit",
            "worst_abs_profit",
            "worst_abs_profit_date",
            "max_drawdown",
            "max_drawdown_date",
        ],
    )
    for col in ("mean_abs_profit", "worst_abs_profit", "max_drawdown"):
        if col in out.columns:
            out[col] = out[col].round(2)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _extract_symbol_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    """Extrahiert Symbol aus meta.json (robust, analog zur Vorlage)."""
    sym = meta.get("symbol")
    if isinstance(sym, str) and sym and sym.lower() != "unknown":
        return sym
    cfg = meta.get("config_used") or {}
    if isinstance(cfg, dict):
        sym = cfg.get("symbol")
        if isinstance(sym, str) and sym:
            return sym
        strat = cfg.get("strategy") or {}
        if isinstance(strat, dict):
            params = strat.get("parameters") or {}
            if isinstance(params, dict):
                sym = params.get("symbol")
                if isinstance(sym, str) and sym:
                    return sym
    return None


def write_max_simult_trades_per_symbol_single(
    trades_df: pd.DataFrame, combo_dir: Path, out_csv: Path
) -> None:
    """
    Berechnet für eine einzelne Kombination die maximale Anzahl simultaner Trades
    pro Symbol sowie Episoden mit gleichzeitigen Long- und Short-Positionen.

    Output-CSV mit Spalten:
    symbol, max_trades_simult, timestamp_start, timestamp_end, long_short_overlap_episodes
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if trades_df is None or trades_df.empty:
        pd.DataFrame(
            columns=[
                "symbol",
                "max_trades_simult",
                "timestamp_start",
                "timestamp_end",
                "long_short_overlap_episodes",
            ]
        ).to_csv(out_csv, index=False)
        return

    # Ermitteln, ob pro-Trade ein Symbol vorliegt; falls ja, pro Symbol berechnen
    symbols: List[str] = []
    if "symbol" in trades_df.columns:
        syms = trades_df["symbol"].fillna("").astype(str).str.strip()
        symbols = sorted([s for s in syms.unique().tolist() if s])
    if not symbols:
        # Fallback: meta.json prüfen
        meta_path = combo_dir / "meta.json"
        sym: Optional[str] = None
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                sym = _extract_symbol_from_meta(meta)
            except Exception:
                sym = None
        symbols = [sym or "unknown"]

    def _fmt_ts(ts: Optional[pd.Timestamp]) -> str:
        if ts is None:
            return ""
        try:
            ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts
            return ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return ""

    rows = []
    for sym in symbols:
        sub_df = trades_df
        if "symbol" in trades_df.columns:
            sub_df = trades_df[trades_df["symbol"].astype(str).str.strip() == sym]
        if sub_df is None or sub_df.empty:
            rows.append(
                {
                    "symbol": sym,
                    "max_trades_simult": 0,
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "long_short_overlap_episodes": 0,
                }
            )
            continue

        events: List[Tuple[pd.Timestamp, int]] = []
        dir_events: List[Tuple[pd.Timestamp, int, str]] = []
        for _, tr in sub_df.iterrows():
            entry_ts = pd.to_datetime(tr.get("entry_time"), utc=True, errors="coerce")
            exit_ts = pd.to_datetime(tr.get("exit_time"), utc=True, errors="coerce")
            if pd.isna(entry_ts) or pd.isna(exit_ts):
                continue
            if exit_ts <= entry_ts:
                continue
            events.append((entry_ts, 1))
            events.append((exit_ts, -1))
            direction_raw = (
                tr.get("direction") if "direction" in sub_df.columns else None
            )
            direction = str(direction_raw).lower() if direction_raw is not None else ""
            if direction in ("long", "short"):
                dir_events.append((entry_ts, 1, direction))
                dir_events.append((exit_ts, -1, direction))

        if not events:
            rows.append(
                {
                    "symbol": sym,
                    "max_trades_simult": 0,
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "long_short_overlap_episodes": 0,
                }
            )
            continue

        events.sort(key=lambda x: (x[0], x[1]))
        active = 0
        max_active = 0
        event_states: List[Tuple[pd.Timestamp, int, int]] = []
        for ts, delta in events:
            active_after = active + delta
            event_states.append((ts, delta, active_after))
            if active_after > max_active:
                max_active = active_after
            active = active_after

        overlap_episodes = 0
        if dir_events:
            dir_events.sort(key=lambda x: (x[0], x[1]))
            active_long = 0
            active_short = 0
            for ts, delta, direction in dir_events:
                had_overlap_before = active_long > 0 and active_short > 0
                if direction == "long":
                    active_long += delta
                elif direction == "short":
                    active_short += delta
                had_overlap_after = active_long > 0 and active_short > 0
                if (not had_overlap_before) and had_overlap_after:
                    overlap_episodes += 1

        ts_start: Optional[pd.Timestamp] = None
        ts_end: Optional[pd.Timestamp] = None
        in_window = False
        for ts, _delta, active_after in event_states:
            if not in_window and active_after == max_active:
                ts_start = ts
                in_window = True
            elif in_window and active_after < max_active:
                ts_end = ts
                break

        rows.append(
            {
                "symbol": sym,
                "max_trades_simult": int(max_active),
                "timestamp_start": _fmt_ts(ts_start),
                "timestamp_end": _fmt_ts(ts_end),
                "long_short_overlap_episodes": int(overlap_episodes),
            }
        )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "max_trades_simult",
            "timestamp_start",
            "timestamp_end",
            "long_short_overlap_episodes",
        ],
    )
    out_df.sort_values("symbol", inplace=True)
    out_df.to_csv(out_csv, index=False)


def plot_equity_with_drawdown(
    equity_series: pd.Series,
    output_path: Path,
    title: str = "Equity Curve",
    baseline: float = 100_000.0,
    yearly_profits: Optional[Dict[int, float]] = None,
) -> None:
    """
    Erstellt einen Equity-Plot mit Drawdown-Panel (analog zu combine_equity_curves.py).

    Args:
        equity_series: Zeitreihe der Equity
        output_path: Pfad zum Output-PNG
        title: Plot-Titel
        baseline: Startkapital für Baseline-Linie
        yearly_profits: Optional Dict {year: profit} für Annotation
    """
    try:
        if equity_series is None or equity_series.empty:
            print(
                f"[Plot] Warnung: Leere Equity-Serie, überspringe Plot: {output_path}"
            )
            return

        # Drawdown berechnen
        roll_max = equity_series.cummax()
        dd_abs = equity_series - roll_max  # <= 0

        # Figure mit zwei Panels (Equity + Drawdown)
        fig = plt.figure(figsize=(12, 7), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.14)
        ax_eq = fig.add_subplot(gs[0])
        ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

        # Style
        fig.patch.set_facecolor("white")
        for ax in (ax_eq, ax_dd):
            ax.set_facecolor("white")
            ax.grid(True, which="major", linestyle="-", alpha=0.15)
            ax.grid(True, which="minor", linestyle=":", alpha=0.08)

        # Plot Equity
        ax_eq.plot(
            equity_series.index,
            equity_series.values,
            color="#1f77b4",
            linewidth=2.0,
            label="Equity",
        )
        ax_eq.fill_between(
            equity_series.index, equity_series.values, color="#1f77b4", alpha=0.06
        )
        ax_eq.set_ylabel("Equity")
        ax_eq.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax_eq.legend(loc="upper left", frameon=False)

        # Y-Limits für Equity (asymmetrisch)
        max_eq = float(equity_series.max())
        min_eq = float(equity_series.min())
        if max_eq == min_eq:
            pad = max(1.0, 0.001 * baseline)
            y_min, y_max = min_eq - pad, max_eq + pad
        else:
            span = max_eq - min_eq
            pad = max(span * 0.02, 0.001 * baseline)
            y_min, y_max = min_eq - pad, max_eq + pad
        ax_eq.set_ylim(y_min, y_max)

        # Baseline-Linie
        ax_eq.axhline(
            baseline, color="#888888", linewidth=0.9, linestyle="--", alpha=0.7
        )

        # Titel
        title_y = 0.96
        fig.suptitle(title, fontsize=13, y=title_y)
        fig.subplots_adjust(top=0.90)

        # Jährliche Profits als Annotation (zwischen Titel und Plot)
        if yearly_profits:
            try:
                ax_top = ax_eq.get_position().y1
                frac_from_axis_to_title = 0.25
                y_pos = ax_top + (title_y - ax_top) * frac_from_axis_to_title
                trans = mtransforms.blended_transform_factory(
                    ax_eq.transData, fig.transFigure
                )

                for year in sorted(yearly_profits.keys()):
                    profit = yearly_profits[year]
                    year_mask = equity_series.index.year == year
                    year_equity = equity_series[year_mask]

                    if len(year_equity) > 0:
                        x_mid = (
                            year_equity.index[0]
                            + (year_equity.index[-1] - year_equity.index[0]) / 2
                        )
                        color = "#2ca02c" if profit >= 0 else "#8c1c13"
                        ax_eq.text(
                            x_mid,
                            y_pos,
                            f"{year}: {profit:+,.0f}",
                            transform=trans,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color=color,
                            clip_on=False,
                        )
            except Exception as e:
                print(f"[Plot] Warnung: Konnte Jahres-Annotation nicht erstellen: {e}")

        # Plot Drawdown (Underwater)
        ax_dd.fill_between(
            dd_abs.index,
            dd_abs.values,
            0,
            where=dd_abs.values < 0,
            color="#d62728",
            alpha=0.35,
        )
        ax_dd.axhline(0, color="#333333", linewidth=0.8)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        # Max Drawdown markieren
        if len(dd_abs) > 0:
            dd_min_val = float(dd_abs.min())
            dd_min_ts = dd_abs.idxmin()
            ax_dd.scatter([dd_min_ts], [dd_min_val], color="#8c1c13", s=20, zorder=5)

            # Datum formatieren
            try:
                dd_date_str = (
                    dd_min_ts.tz_convert("UTC").date().isoformat()
                    if dd_min_ts.tzinfo
                    else dd_min_ts.date().isoformat()
                )
            except Exception:
                dd_date_str = ""

            label = f"Max DD: {dd_min_val:,.0f}"
            if dd_date_str:
                label = f"{label} ({dd_date_str})"

            ax_dd.annotate(
                label,
                xy=(dd_min_ts, dd_min_val),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=9,
                color="#8c1c13",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", ec="#8c1c13", alpha=0.8
                ),
                arrowprops=dict(arrowstyle="-", color="#8c1c13", lw=0.8),
            )

            # Y-Limits für Drawdown (0 oben, max DD unten + Padding)
            pad_dd = max(abs(dd_min_val) * 0.05, 1.0)
            ax_dd.set_ylim(dd_min_val - pad_dd, 0.0)

        # X-Achse formatieren (shared)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for ax in (ax_eq, ax_dd):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis="x", labelsize=9)

        # Speichern
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Equity-Plot gespeichert: {output_path}")

    except Exception as e:
        print(f"[Plot] Fehler beim Erstellen des Equity-Plots: {e}")
        import traceback

        traceback.print_exc()


def _plot_distribution(
    times: pd.Series,
    output_path: Path,
    title: str,
    color: str,
    freq: str,  # "daily", "weekly", "monthly"
) -> None:
    """
    Erstellt ein Histogramm der zeitlichen Verteilung.

    Args:
        times: Zeitstempel-Serie
        output_path: Pfad zum Output-PNG
        title: Plot-Titel
        color: Farbe für die Balken
        freq: "daily", "weekly" oder "monthly"
    """
    try:
        times = pd.to_datetime(times, utc=True, errors="coerce").dropna()
        if times.empty:
            return

        fig, ax = plt.subplots(
            figsize=(10, 4) if freq != "monthly" else (12, 4), dpi=150
        )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if freq == "daily":
            hours = np.arange(24)
            # Wenn times bereits ein DatetimeIndex ist, direkt .hour verwenden
            if isinstance(times, pd.DatetimeIndex):
                counts = (
                    times.hour.value_counts().reindex(hours, fill_value=0).sort_index()
                )
            else:
                counts = (
                    times.dt.hour.value_counts()
                    .reindex(hours, fill_value=0)
                    .sort_index()
                )
            ax.bar(hours, counts.values, color=color, alpha=0.85)
            ax.set_xticks(hours)
            ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
            ax.set_xlabel("Uhrzeit (UTC)")

        elif freq == "weekly":
            days = np.arange(7)
            # Wenn times bereits ein DatetimeIndex ist, direkt .dayofweek verwenden
            if isinstance(times, pd.DatetimeIndex):
                counts = (
                    times.dayofweek.value_counts()
                    .reindex(days, fill_value=0)
                    .sort_index()
                )
            else:
                counts = (
                    times.dt.dayofweek.value_counts()
                    .reindex(days, fill_value=0)
                    .sort_index()
                )
            labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
            ax.bar(days, counts.values, color=color, alpha=0.85)
            ax.set_xticks(days)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Wochentag (UTC)")

        elif freq == "monthly":
            days = np.arange(1, 32)
            # Wenn times bereits ein DatetimeIndex ist, direkt .day verwenden
            if isinstance(times, pd.DatetimeIndex):
                counts = (
                    times.day.value_counts().reindex(days, fill_value=0).sort_index()
                )
            else:
                counts = (
                    times.dt.day.value_counts().reindex(days, fill_value=0).sort_index()
                )
            ax.bar(days, counts.values, color=color, alpha=0.85)
            ax.set_xticks(days)
            ax.set_xticklabels([str(d) for d in days], rotation=0, ha="center")
            ax.set_xlabel("Tag im Monat (1–31)")

        ax.set_ylabel("Anzahl Trades")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="-", alpha=0.2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        print(f"[Plot] Fehler beim Erstellen des Distribution-Plots {output_path}: {e}")


def create_trade_distribution_plots(trades_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Erstellt 18 Trade-Distribution-Plots (analog zu combine_equity_curves.py).

    Args:
        trades_df: DataFrame mit Trade-Daten
        output_dir: Basis-Verzeichnis für die Plots
    """
    if trades_df is None or trades_df.empty:
        print("[Plot] Keine Trades vorhanden, überspringe Trade-Distribution-Plots")
        return

    # Wins und Losses filtern
    wins_mask = trades_df["result"] > 0
    losses_mask = trades_df["result"] < 0

    # Konfigurationen für alle 18 Plots
    configs = [
        # Daily
        (
            "daily_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "daily",
            "Wins Entry – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "daily_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "daily",
            "Wins Exit – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "daily_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "daily",
            "Losses Entry – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "daily_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "daily",
            "Losses Exit – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "daily_trades_entry",
            trades_df["entry_time"],
            "daily",
            "Trades Entry – zeitliche Verteilung",
            "#1f77b4",
        ),
        (
            "daily_trades_exit",
            trades_df["exit_time"],
            "daily",
            "Trades Exit – zeitliche Verteilung",
            "#1f77b4",
        ),
        # Weekly
        (
            "weekly_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "weekly",
            "Wins Entry – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "weekly_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "weekly",
            "Wins Exit – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "weekly_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "weekly",
            "Losses Entry – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "weekly_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "weekly",
            "Losses Exit – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "weekly_trades_entry",
            trades_df["entry_time"],
            "weekly",
            "Trades Entry – zeitliche Verteilung",
            "#1f77b4",
        ),
        (
            "weekly_trades_exit",
            trades_df["exit_time"],
            "weekly",
            "Trades Exit – zeitliche Verteilung",
            "#1f77b4",
        ),
        # Monthly
        (
            "monthly_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "monthly",
            "Wins Entry – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "monthly_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "monthly",
            "Wins Exit – zeitliche Verteilung",
            "#2ca02c",
        ),
        (
            "monthly_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "monthly",
            "Losses Entry – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "monthly_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "monthly",
            "Losses Exit – zeitliche Verteilung",
            "#d62728",
        ),
        (
            "monthly_trades_entry",
            trades_df["entry_time"],
            "monthly",
            "Trades Entry – zeitliche Verteilung",
            "#1f77b4",
        ),
        (
            "monthly_trades_exit",
            trades_df["exit_time"],
            "monthly",
            "Trades Exit – zeitliche Verteilung",
            "#1f77b4",
        ),
    ]

    trade_distr_dir = output_dir / "trade_distr"
    trade_distr_dir.mkdir(parents=True, exist_ok=True)

    for filename, series, freq, title, color in configs:
        output_path = trade_distr_dir / f"{filename}.png"
        _plot_distribution(series, output_path, title, color, freq)


def process_final_combo(
    combo_id: str, combo_dir: Path, baseline: float = 100_000.0
) -> Dict[str, Any]:
    """
    Verarbeitet eine finale Kombination: lädt Daten, berechnet KPIs, erstellt Plots.

    Args:
        combo_id: ID der finalen Kombination
        combo_dir: Pfad zum Verzeichnis der Kombination
        baseline: Startkapital

    Returns:
        Dict mit Verarbeitungsergebnis (success, kpis, errors)
    """
    print(f"\n{'='*80}")
    print(f"[Process] Verarbeite finale Kombination: {combo_id}")
    print(f"{'='*80}")

    result = {
        "combo_id": combo_id,
        "success": False,
        "kpis": {},
        "plots_created": [],
        "errors": [],
    }

    # Pfade zu Daten
    equity_path = combo_dir / "equity.csv"
    trades_path = combo_dir / "trades.json"

    # Equity laden
    equity_series = load_equity_series(equity_path)
    if equity_series is None:
        result["errors"].append("Konnte equity.csv nicht laden")
        return result

    print(
        f"[Process] Equity geladen: {len(equity_series)} Datenpunkte, Zeitraum {equity_series.index[0].date()} bis {equity_series.index[-1].date()}"
    )

    # Trades laden (optional)
    trades_df = load_trades_data(trades_path)
    if trades_df is not None:
        print(f"[Process] Trades geladen: {len(trades_df)} Trades")
    else:
        print("[Process] Keine Trades vorhanden")

    # KPIs berechnen
    print("[Process] Berechne KPIs...")
    kpis = compute_kpis(equity_series, trades_df, baseline)
    result["kpis"] = kpis

    # KPIs ausgeben
    print(f"[KPIs] Total Profit: {kpis.get('total_profit', 0):,.2f}")
    print(f"[KPIs] Max Drawdown: {kpis.get('max_drawdown', 0):,.2f}")
    print(f"[KPIs] Profit/DD: {kpis.get('profit_over_dd', 0):,.2f}")
    print(f"[KPIs] Winrate: {kpis.get('winrate', 0):.2f}%")
    print(f"[KPIs] Avg R: {kpis.get('avg_r', 0):.4f}")
    print(f"[KPIs] Total Trades: {kpis.get('total_trades', 0)}")

    # KPIs speichern
    kpis_path = combo_dir / "kpis.json"
    try:
        with kpis_path.open("w", encoding="utf-8") as f:
            json.dump(kpis, f, indent=2, default=str)
        print(f"[Process] KPIs gespeichert: {kpis_path}")
    except Exception as e:
        result["errors"].append(f"Konnte KPIs nicht speichern: {e}")

    # Equity-Plot erstellen
    print("[Process] Erstelle Equity-Plot...")
    equity_plot_path = combo_dir / "equity_plot.png"
    try:
        plot_equity_with_drawdown(
            equity_series,
            equity_plot_path,
            title=f"Final Combo {combo_id} – Equity Curve",
            baseline=baseline,
            yearly_profits=kpis.get("yearly_profits"),
        )
        result["plots_created"].append("equity_plot.png")
    except Exception as e:
        result["errors"].append(f"Konnte Equity-Plot nicht erstellen: {e}")

    # Expected Profit Summary (CSV)
    try:
        eps_path = combo_dir / "expected_profit_summary.csv"
        write_expected_profit_summary_single(equity_series, eps_path)
        print(f"[Process] Expected-Profit-Summary gespeichert: {eps_path}")
    except Exception as e:
        result["errors"].append(
            f"Konnte expected_profit_summary.csv nicht erstellen: {e}"
        )

    # Trade-Distribution-Plots erstellen
    if trades_df is not None and not trades_df.empty:
        print("[Process] Erstelle Trade-Distribution-Plots...")
        try:
            create_trade_distribution_plots(trades_df, combo_dir)
            result["plots_created"].append("trade_distr/")
        except Exception as e:
            result["errors"].append(
                f"Konnte Trade-Distribution-Plots nicht erstellen: {e}"
            )

        # Max simultane Trades pro Symbol (CSV)
        try:
            mts_path = combo_dir / "max_trades_simult_per_symbol.csv"
            write_max_simult_trades_per_symbol_single(trades_df, combo_dir, mts_path)
            print(f"[Process] Max simultane Trades pro Symbol gespeichert: {mts_path}")
        except Exception as e:
            result["errors"].append(
                f"Konnte max_trades_simult_per_symbol.csv nicht erstellen: {e}"
            )

    result["success"] = len(result["errors"]) == 0

    if result["success"]:
        print(f"[Process] ✓ Erfolgreich verarbeitet: {combo_id}")
    else:
        print(f"[Process] ✗ Fehler bei der Verarbeitung: {combo_id}")
        for err in result["errors"]:
            print(f"  - {err}")

    return result


def load_champions(csv_path: Path) -> List[str]:
    """
    Lädt die Liste der finalen champion combo_pair_ids aus der Champions-CSV.

    Args:
        csv_path: Pfad zur final_combos_categorical_champions.csv

    Returns:
        Liste von final_combo_pair_ids (11 Champions mit kategorialem Ranking)
    """
    try:
        if not csv_path.exists():
            print(f"[Load] Warnung: Champions-CSV nicht gefunden: {csv_path}")
            return []

        df = pd.read_csv(csv_path, skip_blank_lines=True)

        # Entferne Leerzeilen
        df = df.dropna(how="all")

        # Nutze final_combo_pair_id (neue Spalte aus kategorischem Ranking)
        if "final_combo_pair_id" in df.columns:
            combo_ids = df["final_combo_pair_id"].dropna().unique().tolist()
            print(f"[Load] {len(combo_ids)} Champions aus Champions-CSV geladen")
            return [str(cid) for cid in combo_ids]
        else:
            print(
                f"[Load] Warnung: Spalte 'final_combo_pair_id' nicht in {csv_path} gefunden"
            )
            return []

    except Exception as e:
        print(f"[Load] Fehler beim Laden der Champions-CSV: {e}")
        return []


def main():
    """
    Hauptfunktion: Verarbeitet alle finalen Kombinationen aus der Top-10-Liste.
    """
    parser = argparse.ArgumentParser(
        description="Erstellt Equity-Plots und KPIs für finale Kombinationen aus der Matrix-Analyse"
    )
    parser.add_argument(
        "--champions-csv",
        type=str,
        default=str(CHAMPIONS_CSV),
        help="Pfad zur final_combos_categorical_champions.csv",
    )
    parser.add_argument(
        "--final-combos-dir",
        type=str,
        default=str(FINAL_COMBOS_DIR),
        help="Basis-Verzeichnis für finale Kombinationen",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=100_000.0,
        help="Startkapital für Baseline-Linie",
    )
    parser.add_argument(
        "--combo-id",
        type=str,
        default=None,
        help="Optional: Verarbeite nur eine spezifische combo_id",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FINAL COMBO EQUITY PLOTTER")
    print("=" * 80 + "\n")

    # Pfade konvertieren
    champions_csv = Path(args.champions_csv)
    final_combos_dir = Path(args.final_combos_dir)

    # Prüfe ob final_combos Verzeichnis existiert
    if not final_combos_dir.exists():
        print(f"[Error] final_combos Verzeichnis nicht gefunden: {final_combos_dir}")
        print("Führe zuerst combined_walkforward_matrix_analyzer.py aus!")
        return

    # Lade Champions oder verarbeite einzelne combo_id
    if args.combo_id:
        combo_ids = [args.combo_id]
        print(f"[Mode] Verarbeite einzelne Kombination: {args.combo_id}")
    else:
        combo_ids = load_champions(champions_csv)
        if not combo_ids:
            print("[Error] Keine finalen Kombinationen gefunden!")
            return

    # Verarbeite jede Kombination
    results = []
    for combo_id in combo_ids:
        combo_dir = final_combos_dir / combo_id

        if not combo_dir.exists():
            print(f"[Warnung] Verzeichnis nicht gefunden: {combo_dir}")
            continue

        result = process_final_combo(combo_id, combo_dir, args.baseline)
        results.append(result)

    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nVerarbeitet: {len(results)} finale Kombinationen")
    print(f"  ✓ Erfolgreich: {len(successful)}")
    print(f"  ✗ Fehler: {len(failed)}")

    if failed:
        print("\nFehlgeschlagene Kombinationen:")
        for r in failed:
            print(f"  - {r['combo_id']}: {', '.join(r['errors'])}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
