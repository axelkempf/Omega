import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


def find_equity_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "equity.csv":
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    # Try to find timestamp/date column
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


def load_equity_series(path: str, label: Optional[str] = None) -> pd.Series:
    """Load equity as an event-based time series indexed by timestamp (UTC).

    - Parses timestamps
    - Drops unparsable rows
    - Sorts by time and keeps the last value per exact timestamp
    """
    df = pd.read_csv(path)
    ts_col, eq_col = detect_columns(df)

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    mask = ts.notna()
    df = df.loc[mask, [eq_col]].copy()
    ts = ts.loc[mask]

    df["timestamp"] = ts.dt.tz_convert("UTC")
    df = df.sort_values("timestamp")
    # If there are duplicate timestamps, keep the last occurrence
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    s = pd.Series(df[eq_col].values, index=df["timestamp"].values)
    if not label:
        label = os.path.basename(os.path.dirname(path))
    s.name = label
    return s


def sanitize_label(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    # Drop the trailing filename and use the directory as label
    rel_dir = os.path.dirname(rel)
    return rel_dir.replace(os.sep, "__") or rel


def combine_equity(
    root: str,
    output_csv: str,
    wide_csv: Optional[str] = None,
    baseline: float = 100_000.0,
) -> pd.DataFrame:
    files = find_equity_files(root)
    if not files:
        raise FileNotFoundError(f"Keine equity.csv Dateien unter '{root}' gefunden.")
    # Debug-Ausgabe: Wie viele Backtest-Ordner (mit equity.csv) werden berücksichtigt?
    try:
        backtest_dirs = {os.path.dirname(p) for p in files}
        print(
            f"Debug: Greife auf {len(backtest_dirs)} Backtest-Ordner zu (Quelle: equity.csv)"
        )
    except Exception:
        # Auf Debug-Ausgabe verzichten, falls etwas Unerwartetes passiert
        pass

    series_list = []
    for p in files:
        label = sanitize_label(root, p)
        try:
            s = load_equity_series(p, label=label)
        except Exception as e:
            # Skip malformed files but continue
            print(f"Warnung: Überspringe {p}: {e}")
            continue
        series_list.append(s)

    if not series_list:
        raise RuntimeError("Keine gültigen Equity-Serien geladen.")

    # Outer-join all timestamps (union), keep full span earliest -> latest
    wide = pd.concat(series_list, axis=1).sort_index()

    # Forward-fill within each series so equity holds when no updates
    wide_ffill = wide.ffill()

    # For timestamps before a series starts, set to baseline (PnL=0)
    wide_union = wide_ffill.fillna(float(baseline))

    # Convert equities to PnL deltas relative to a single baseline per strategy
    # and sum PnL across strategies; final total = baseline + sum(PnL)
    deltas = wide_union - float(baseline)
    total = float(baseline) + deltas.sum(axis=1)
    total.name = "equity_total"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    out_df = total.reset_index()
    # timestamp as ISO 8601 (UTC) string
    out_df.rename(columns={"index": "timestamp"}, inplace=True)
    # Ensure tz-aware UTC and format with 'Z'
    out_df["timestamp"] = (
        pd.to_datetime(out_df["timestamp"], utc=True)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    out_df = out_df[["timestamp", "equity_total"]]
    out_df.to_csv(output_csv, index=False)

    if wide_csv:
        # Save the per-strategy equity (nicht PnL) aligned on timestamp union
        wide_out = wide_union.copy()
        # Format index as UTC ISO 8601
        idx = (
            pd.to_datetime(wide_out.index, utc=True)
            .tz_convert("UTC")
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        wide_out.index = idx
        wide_out.index.name = "timestamp"
        wide_out.to_csv(wide_csv)

    return out_df


def plot_equity(
    df: pd.DataFrame,
    plot_path: str,
    title: str = "Portfolio Equity",
    baseline: float = 100_000.0,
):
    # Prepare data
    d = df.copy()
    time_col = (
        "timestamp"
        if "timestamp" in d.columns
        else ("date" if "date" in d.columns else None)
    )
    if time_col is None:
        raise ValueError(
            "Erwarte eine Spalte 'timestamp' oder 'date' für die Zeitachse."
        )
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col)

    equity = d["equity_total"].values
    eq = pd.Series(equity, index=d[time_col])
    roll_max = eq.cummax()
    dd_abs = eq - roll_max  # <= 0

    # Figure with two panels (equity + absolute drawdown)
    fig = plt.figure(figsize=(12, 7), dpi=150)
    # Slightly increase vertical spacing between panels
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.14)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    # Style
    fig.patch.set_facecolor("white")
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor("white")
        ax.grid(True, which="major", linestyle="-", alpha=0.15)
        ax.grid(True, which="minor", linestyle=":", alpha=0.08)

    # Plot equity
    ax_eq.plot(eq.index, eq.values, color="#1f77b4", linewidth=2.0, label="Equity")
    ax_eq.fill_between(eq.index, eq.values, color="#1f77b4", alpha=0.06)
    ax_eq.set_ylabel("Equity")
    ax_eq.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    # Title and annual profits displayed above the graphs
    ax_eq.legend(loc="upper left", frameon=False)

    # Asymmetric y-limits for equity
    max_eq = float(eq.max()) if len(eq) else baseline
    min_eq = float(eq.min()) if len(eq) else baseline
    if max_eq == min_eq:
        pad = max(1.0, 0.001 * baseline)
        y_min, y_max = min_eq - pad, max_eq + pad
    else:
        span = max_eq - min_eq
        pad = max(span * 0.02, 0.001 * baseline)
        y_min, y_max = min_eq - pad, max_eq + pad
    ax_eq.set_ylim(y_min, y_max)
    ax_eq.axhline(baseline, color="#888888", linewidth=0.9, linestyle="--", alpha=0.7)

    # Title and annual profit numbers positioned between title and plot
    title_y = 0.96
    fig.suptitle(title, fontsize=13, y=title_y)
    # Leave room for title and labels
    fig.subplots_adjust(top=0.90)
    try:
        ax_top = ax_eq.get_position().y1
        # Place labels closer to the plot (further from the title)
        frac_from_axis_to_title = 0.25  # 0..1 between axis top and title
        y_pos = ax_top + (title_y - ax_top) * frac_from_axis_to_title
        years = sorted(pd.unique(eq.index.year))
        if years:
            trans = mtransforms.blended_transform_factory(
                ax_eq.transData, fig.transFigure
            )
            for y in years:
                mask = eq.index.year == y
                if not mask.any():
                    continue
                eq_year = eq.loc[mask]
                if len(eq_year) == 0:
                    continue
                prof = float(eq_year.iloc[-1] - eq_year.iloc[0])
                x_mid = eq_year.index[0] + (eq_year.index[-1] - eq_year.index[0]) / 2
                color = "#2ca02c" if prof >= 0 else "#8c1c13"
                ax_eq.text(
                    x_mid,
                    y_pos,
                    f"{y}: {prof:+,.0f}",
                    transform=trans,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    clip_on=False,
                )
    except Exception:
        # Ignore annotation errors to not break plotting
        pass

    # Plot absolute drawdown (underwater)
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

    # Highlight maximum drawdown
    if len(dd_abs):
        dd_min_val = float(dd_abs.min())
        dd_min_ts = dd_abs.idxmin()
        ax_dd.scatter([dd_min_ts], [dd_min_val], color="#8c1c13", s=20, zorder=5)
        # Datum, an dem der maximale Drawdown erreicht wurde
        try:
            if isinstance(dd_min_ts, pd.Timestamp):
                ts_utc = dd_min_ts.tz_convert("UTC") if dd_min_ts.tzinfo else dd_min_ts
                dd_date_str = ts_utc.date().isoformat()
            else:
                dd_date_str = str(dd_min_ts)
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
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8c1c13", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="#8c1c13", lw=0.8),
        )

        # Set y-limits so 0 is exactly at the top (no space above),
        # and add only a small margin below the maximum drawdown.
        pad_dd = max(abs(dd_min_val) * 0.05, 1.0)
        ax_dd.set_ylim(dd_min_val - pad_dd, 0.0)

    # X axis formatting (shared)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in (ax_eq, ax_dd):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelsize=9)

    # Removed in-plot yearly profit annotations per latest requirements

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def _resample_closes(eq: pd.Series, freq: str) -> pd.Series:
    return eq.sort_index().resample(freq).last().dropna()


def _max_drawdown_vs_prev_period(
    eq: pd.Series,
    resample_freq: str,
    period_freq: Optional[str] = None,
) -> Tuple[float, Optional[pd.Timestamp]]:
    """
    Maximaler negativer Intraperioden-Drawdown relativ zur Equity
    des jeweils vorherigen Zeitraums.

    Beispiel (daily):
    - Für jeden Tag wird die Equity innerhalb des Tages betrachtet.
    - Referenz ist der Schlusskurs des Vortags.
    - Pro Tag: min(Equity - Equity_vortag)
    - max_drawdown ist dann das Minimum über alle Tage (negativ).
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

    # Für die erste Periode: prev_close explizit auf die Start-Equity setzen,
    # damit auch diese Periode in die Drawdown-Berechnung einfließt.
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
        return np.nan

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


def _find_meta_paths_for_equity(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        if "equity.csv" in filenames and "meta.json" in filenames:
            paths.append(os.path.join(dirpath, "meta.json"))
    return paths


def _parse_meta_dates(meta_path: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        meta = pd.read_json(meta_path, typ="series")
        # Prefer root-level keys; fall back to config_used
        start = meta.get("start_date")
        end = meta.get("end_date")
        if start is None or end is None:
            cfg = meta.get("config_used", {})
            if isinstance(cfg, dict):
                start = start or cfg.get("start_date")
                end = end or cfg.get("end_date")
        if start is None or end is None:
            return None
        s = pd.to_datetime(str(start)).normalize()
        e = pd.to_datetime(str(end)).normalize()
        return s, e
    except Exception:
        return None


def _full_years_intersection_from_meta(root: str) -> List[int]:
    meta_paths = _find_meta_paths_for_equity(root)
    if not meta_paths:
        return []

    per_run_full_years: List[set] = []
    for mp in meta_paths:
        se = _parse_meta_dates(mp)
        if se is None:
            # Skip runs without parsable dates (conservative)
            continue
        s, e = se
        # Build set of fully covered years for this run
        full_years = set()
        for y in range(s.year, e.year + 1):
            y_start = pd.Timestamp(year=y, month=1, day=1)
            y_end = pd.Timestamp(year=y, month=12, day=31)
            if s <= y_start and e >= y_end:
                full_years.add(y)
        per_run_full_years.append(full_years)

    if not per_run_full_years:
        return []
    # Intersection across all contributing runs
    inter = set.intersection(*per_run_full_years) if per_run_full_years else set()
    return sorted(list(inter))


def write_expected_profit_summary(
    df: pd.DataFrame, out_csv: str, root: Optional[str] = None
) -> None:
    """Compute expected values for daily/weekly/monthly/yearly segments.

    Outputs: timeframe, mean_abs_profit, worst_abs_profit, max_drawdown
    - mean_abs_profit: arithmetic mean of absolute PnL per period (Δ = close_t - close_{t-1})
    - worst_abs_profit: minimum (worst) period PnL across all segments
    - max_drawdown: maximal negativer Intraperioden-Drawdown relativ zur Equity des
      vorherigen Zeitraums (z.B. Vortag / Vorwoche / Vormonat / Vorjahr)
    """
    d = df.copy()
    time_col = (
        "timestamp"
        if "timestamp" in d.columns
        else ("date" if "date" in d.columns else None)
    )
    if time_col is None:
        raise ValueError(
            "Erwarte eine Spalte 'timestamp' oder 'date' für die Zeitachse."
        )
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col)

    eq = pd.Series(d["equity_total"].values, index=d[time_col])
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
        if name == "yearly":
            # Include only full calendar years (intersection across runs from meta.json)
            # and compute per-year profit as last-within-year minus first-within-year.
            if root is not None:
                full_years = set(_full_years_intersection_from_meta(root))
            else:
                full_years = set()
            if full_years:
                profits = []
                profit_dates: List[pd.Timestamp] = []
                for y in sorted(pd.unique(eq.index.year)):
                    if y not in full_years:
                        continue
                    mask_y = eq.index.year == y
                    eq_y = eq.loc[mask_y]
                    if len(eq_y) >= 2:
                        profits.append(float(eq_y.iloc[-1] - eq_y.iloc[0]))
                        profit_dates.append(eq_y.index[-1])
                if len(profits) == 0:
                    mean_profit = np.nan
                    worst_profit = np.nan
                else:
                    arr = np.asarray(profits, dtype=float)
                    mean_profit = float(arr.mean())
                    worst_idx = int(np.argmin(arr))
                    worst_profit = float(arr[worst_idx])
                    worst_ts = profit_dates[worst_idx]
                    try:
                        if isinstance(worst_ts, pd.Timestamp):
                            ts_utc = (
                                worst_ts.tz_convert("UTC")
                                if worst_ts.tzinfo
                                else worst_ts
                            )
                            worst_profit_date_str = ts_utc.date().isoformat()
                        else:
                            worst_profit_date_str = str(worst_ts)
                    except Exception:
                        worst_profit_date_str = ""
            else:
                mean_profit = np.nan
                worst_profit = np.nan
        else:
            if len(closes) >= 2:
                chg = closes.diff().dropna()
                mean_profit = float(chg.mean())
                worst_profit = float(chg.min())
                # Timestamp des schlechtesten Perioden-Profits -> Datum
                try:
                    worst_ts = chg.idxmin()
                    if isinstance(worst_ts, pd.Timestamp):
                        ts_utc = (
                            worst_ts.tz_convert("UTC") if worst_ts.tzinfo else worst_ts
                        )
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
        # Vollständiger Timestamp des maximalen Intraperioden-Drawdowns
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
    # Round numeric columns to 2 decimals for cleaner CSV output
    for col in ("mean_abs_profit", "worst_abs_profit", "max_drawdown"):
        if col in out.columns:
            out[col] = out[col].round(2)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)


def _extract_symbol_from_meta(meta: Dict) -> Optional[str]:
    """Versucht, das Symbol aus einer meta.json-Struktur robust zu extrahieren."""
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


def write_max_simult_trades_per_symbol(root: str, out_csv: str) -> None:
    """
    Liest alle trades.json unterhalb von root ein und berechnet pro Symbol,
    wie viele Trades maximal gleichzeitig offen waren.

    Zusätzlich wird gezählt, wie oft pro Symbol gleichzeitig Long- und Short-Positionen
    offen waren (Overlaps als diskrete Episoden, nicht nach Dauer gewichtet).

    Output-CSV: Spalten
    'symbol', 'max_trades_simult', 'timestamp_start', 'timestamp_end',
    'long_short_overlap_episodes'.
    """
    events_by_symbol: Dict[str, List[Tuple[pd.Timestamp, int]]] = {}
    dir_events_by_symbol: Dict[str, List[Tuple[pd.Timestamp, int, str]]] = {}

    for dirpath, _, filenames in os.walk(root):
        if "trades.json" not in filenames:
            continue

        trades_path = os.path.join(dirpath, "trades.json")
        meta_path = os.path.join(dirpath, "meta.json")

        symbol: Optional[str] = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                symbol = _extract_symbol_from_meta(meta)
            except Exception:
                symbol = None

        if not symbol:
            # Fallback: Symbol aus Verzeichnisname ableiten (strategy_symbol_tf)
            rel = os.path.relpath(dirpath, root)
            parts = rel.split(os.sep)
            if parts:
                base = parts[0]
                fields = base.split("_")
                if len(fields) >= 3:
                    symbol = "_".join(fields[1:-1])
                elif len(fields) >= 2:
                    symbol = fields[1]

        if not symbol:
            symbol = "unknown"

        try:
            with open(trades_path, "r") as f:
                trades = json.load(f)
        except Exception as e:
            print(f"Warnung: Konnte Trades aus {trades_path} nicht lesen: {e}")
            continue

        if not isinstance(trades, list):
            continue

        for tr in trades:
            if not isinstance(tr, dict):
                continue
            entry_raw = tr.get("entry_time")
            exit_raw = tr.get("exit_time")
            if not entry_raw or not exit_raw:
                continue
            entry_ts = pd.to_datetime(entry_raw, utc=True, errors="coerce")
            exit_ts = pd.to_datetime(exit_raw, utc=True, errors="coerce")
            if pd.isna(entry_ts) or pd.isna(exit_ts):
                continue
            # Nur Trades mit positiver Dauer berücksichtigen (Halb-offene Intervalle [entry, exit))
            if exit_ts <= entry_ts:
                continue

            ev_list = events_by_symbol.setdefault(symbol, [])
            ev_list.append((entry_ts, 1))  # +1 bei Entry
            ev_list.append((exit_ts, -1))  # -1 bei Exit

            # Richtung (long/short) für Overlap-Zählung berücksichtigen
            direction_raw = tr.get("direction")
            direction = str(direction_raw).lower() if direction_raw is not None else ""
            if direction in ("long", "short"):
                dev_list = dir_events_by_symbol.setdefault(symbol, [])
                dev_list.append((entry_ts, 1, direction))
                dev_list.append((exit_ts, -1, direction))

    rows = []
    for symbol, events in events_by_symbol.items():
        if not events:
            continue
        # Exit-Events (-1) sollen bei gleichem Timestamp vor Entry-Events (+1) kommen,
        # damit ein neuer Trade zum exakt gleichen Zeitpunkt nicht als simultan gezählt wird.
        events.sort(key=lambda x: (x[0], x[1]))

        # Erster Durchlauf: Verlauf der aktiven Trades und globales Maximum bestimmen
        active = 0
        max_active = 0
        event_states: List[Tuple[pd.Timestamp, int, int]] = (
            []
        )  # (ts, delta, active_after)
        for ts, delta in events:
            active_before = active
            active_after = active_before + delta
            event_states.append((ts, delta, active_after))
            if active_after > max_active:
                max_active = active_after
            active = active_after

        if max_active <= 0:
            # Keine offenen Trades gefunden
            rows.append(
                {
                    "symbol": symbol,
                    "max_trades_simult": 0,
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "long_short_overlap_episodes": 0,
                }
            )
            continue

        # Long/Short-Overlap als Episoden zählen (0 -> >0 Zustandswechsel von gleichzeitigen
        # Long- und Short-Positionen). Falls keine Richtungsinformationen vorliegen, 0.
        overlap_episodes = 0
        dir_events = dir_events_by_symbol.get(symbol, [])
        if dir_events:
            # Auch hier: Exit-Events (-1) vor Entry-Events (+1) pro Timestamp
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

        # Zweiter Durchlauf: erstes Intervall bestimmen, in dem max_active erfüllt ist
        ts_start: Optional[pd.Timestamp] = None
        ts_end: Optional[pd.Timestamp] = None
        in_max_window = False
        for ts, _delta, active_after in event_states:
            if not in_max_window and active_after == max_active:
                ts_start = ts
                in_max_window = True
            elif in_max_window and active_after < max_active:
                ts_end = ts
                break

        # Formatierung der Timestamps
        def _fmt_ts(ts: Optional[pd.Timestamp]) -> str:
            if ts is None:
                return ""
            try:
                if isinstance(ts, pd.Timestamp):
                    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts
                    return ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                return str(ts)
            except Exception:
                return ""

        rows.append(
            {
                "symbol": symbol,
                "max_trades_simult": int(max_active),
                "timestamp_start": _fmt_ts(ts_start),
                "timestamp_end": _fmt_ts(ts_end),
                "long_short_overlap_episodes": int(overlap_episodes),
            }
        )

    rows.sort(key=lambda r: r["symbol"])
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
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)


def _load_all_trades(root: str) -> pd.DataFrame:
    """
    Lädt alle trades.json unterhalb von root und gibt ein DataFrame
    mit Spalten entry_time, exit_time, result zurück.
    """
    rows: List[Dict[str, object]] = []

    for dirpath, _, filenames in os.walk(root):
        if "trades.json" not in filenames:
            continue
        trades_path = os.path.join(dirpath, "trades.json")
        try:
            with open(trades_path, "r") as f:
                trades = json.load(f)
        except Exception as e:
            print(f"Warnung: Konnte Trades aus {trades_path} nicht lesen: {e}")
            continue

        if not isinstance(trades, list):
            continue

        for tr in trades:
            if not isinstance(tr, dict):
                continue
            entry_raw = tr.get("entry_time")
            exit_raw = tr.get("exit_time")
            if not entry_raw or not exit_raw:
                continue

            entry_ts = pd.to_datetime(entry_raw, utc=True, errors="coerce")
            exit_ts = pd.to_datetime(exit_raw, utc=True, errors="coerce")
            if pd.isna(entry_ts) or pd.isna(exit_ts):
                continue

            res_raw = tr.get("result")
            try:
                res_val = float(res_raw) if res_raw is not None else np.nan
            except Exception:
                res_val = np.nan

            rows.append(
                {
                    "entry_time": entry_ts,
                    "exit_time": exit_ts,
                    "result": res_val,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["entry_time", "exit_time", "result"])

    df = pd.DataFrame(rows)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    return df


def _plot_daily_distribution(
    times: pd.Series, out_path: str, title: str, color: str = "#1f77b4"
) -> None:
    times = pd.to_datetime(times, utc=True, errors="coerce").dropna()
    if times.empty:
        return

    hours = np.arange(24)
    counts = times.dt.hour.value_counts().reindex(hours, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(hours, counts.values, color=color, alpha=0.85)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
    ax.set_xlabel("Uhrzeit (UTC)")
    ax.set_ylabel("Anzahl Trades")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="-", alpha=0.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_weekly_distribution(
    times: pd.Series, out_path: str, title: str, color: str = "#1f77b4"
) -> None:
    times = pd.to_datetime(times, utc=True, errors="coerce").dropna()
    if times.empty:
        return

    days = np.arange(7)
    counts = times.dt.dayofweek.value_counts().reindex(days, fill_value=0).sort_index()
    labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(days, counts.values, color=color, alpha=0.85)
    ax.set_xticks(days)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Wochentag (UTC)")
    ax.set_ylabel("Anzahl Trades")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="-", alpha=0.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_monthly_distribution(
    times: pd.Series, out_path: str, title: str, color: str = "#1f77b4"
) -> None:
    times = pd.to_datetime(times, utc=True, errors="coerce").dropna()
    if times.empty:
        return

    # Aggregation nach Tag im Monat (1–31), über alle Monate/Jahre hinweg
    days = np.arange(1, 32)
    counts = times.dt.day.value_counts().reindex(days, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(days, counts.values, color=color, alpha=0.85)
    ax.set_xticks(days)
    ax.set_xticklabels([str(d) for d in days], rotation=0, ha="center")
    ax.set_xlabel("Tag im Monat (1–31, aggregiert über alle Monate)")
    ax.set_ylabel("Anzahl Trades")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="-", alpha=0.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_trade_distribution_plots(root: str, out_dir: str) -> None:
    """
    Erzeugt 18 PNGs mit zeitlicher Verteilung der Trades:
    - daily/weekly/monthly
    - wins/losses/trades
    - entry/exit

    Die Dateien werden als <timeframe>_<klasse>_<entry|exit>.png unter out_dir gespeichert.
    """
    trades_df = _load_all_trades(root)
    if trades_df.empty:
        print(
            "Warnung: Keine Trades gefunden – Trade-Distribution-Plots werden übersprungen."
        )
        return

    wins_mask = trades_df["result"] > 0
    losses_mask = trades_df["result"] < 0

    configs = [
        # Daily
        (
            "daily_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "daily",
            "Wins",
            "Entry",
            "#2ca02c",
        ),
        (
            "daily_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "daily",
            "Wins",
            "Exit",
            "#2ca02c",
        ),
        (
            "daily_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "daily",
            "Losses",
            "Entry",
            "#d62728",
        ),
        (
            "daily_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "daily",
            "Losses",
            "Exit",
            "#d62728",
        ),
        (
            "daily_trades_entry",
            trades_df["entry_time"],
            "daily",
            "Trades",
            "Entry",
            "#1f77b4",
        ),
        (
            "daily_trades_exit",
            trades_df["exit_time"],
            "daily",
            "Trades",
            "Exit",
            "#1f77b4",
        ),
        # Weekly
        (
            "weekly_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "weekly",
            "Wins",
            "Entry",
            "#2ca02c",
        ),
        (
            "weekly_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "weekly",
            "Wins",
            "Exit",
            "#2ca02c",
        ),
        (
            "weekly_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "weekly",
            "Losses",
            "Entry",
            "#d62728",
        ),
        (
            "weekly_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "weekly",
            "Losses",
            "Exit",
            "#d62728",
        ),
        (
            "weekly_trades_entry",
            trades_df["entry_time"],
            "weekly",
            "Trades",
            "Entry",
            "#1f77b4",
        ),
        (
            "weekly_trades_exit",
            trades_df["exit_time"],
            "weekly",
            "Trades",
            "Exit",
            "#1f77b4",
        ),
        # Monthly
        (
            "monthly_wins_entry",
            trades_df.loc[wins_mask, "entry_time"],
            "monthly",
            "Wins",
            "Entry",
            "#2ca02c",
        ),
        (
            "monthly_wins_exit",
            trades_df.loc[wins_mask, "exit_time"],
            "monthly",
            "Wins",
            "Exit",
            "#2ca02c",
        ),
        (
            "monthly_losses_entry",
            trades_df.loc[losses_mask, "entry_time"],
            "monthly",
            "Losses",
            "Entry",
            "#d62728",
        ),
        (
            "monthly_losses_exit",
            trades_df.loc[losses_mask, "exit_time"],
            "monthly",
            "Losses",
            "Exit",
            "#d62728",
        ),
        (
            "monthly_trades_entry",
            trades_df["entry_time"],
            "monthly",
            "Trades",
            "Entry",
            "#1f77b4",
        ),
        (
            "monthly_trades_exit",
            trades_df["exit_time"],
            "monthly",
            "Trades",
            "Exit",
            "#1f77b4",
        ),
    ]

    os.makedirs(out_dir, exist_ok=True)

    for filename, series, timeframe, cls, side, color in configs:
        series = pd.to_datetime(series, utc=True, errors="coerce").dropna()
        if series.empty:
            continue
        title = f"{timeframe.capitalize()} {cls} {side} – zeitliche Verteilung"
        out_path = os.path.join(out_dir, f"{filename}.png")
        if timeframe == "daily":
            _plot_daily_distribution(series, out_path, title=title, color=color)
        elif timeframe == "weekly":
            _plot_weekly_distribution(series, out_path, title=title, color=color)
        elif timeframe == "monthly":
            _plot_monthly_distribution(series, out_path, title=title, color=color)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fasst alle equity.csv-Dateien zusammen, normalisiert das Datum auf "
            "YYYY-MM-DD und summiert die Equity über Strategien pro Datum."
        )
    )
    parser.add_argument(
        "--root",
        default="var/results/backtests",
        help="Wurzelverzeichnis, unter dem nach equity.csv gesucht wird.",
    )
    parser.add_argument(
        "--out",
        default="var/results/backtests/summary/equity_summary.csv",
        help="Zieldatei für die Gesamt-Equity-Kurve (CSV)",
    )
    parser.add_argument(
        "--wide-out",
        default=None,
        help="Optional: Pfad für eine Wide-CSV mit allen Einzelserien",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=100_000.0,
        help="Basiswert (Startkapital), gegen den Deltas berechnet werden.",
    )
    parser.add_argument(
        "--plot-out",
        default="var/results/backtests/summary/equity_summary.png",
        help="Pfad für die Equity-Visualisierung (PNG/PDF)",
    )
    parser.add_argument(
        "--expected-profit-out",
        default="var/results/backtests/summary/expected_profit_summary.csv",
        help="Pfad für Erwartungswerte (wöchentlich/monatlich: Mittelwert Profit und schlechtester Profit)",
    )
    parser.add_argument(
        "--max-simult-trades-out",
        default="var/results/backtests/summary/max_trades_simult_per_symbol.csv",
        help="Pfad für Statistik: maximale simultane Trades pro Symbol (CSV)",
    )
    parser.add_argument(
        "--trade-distr-dir",
        default="var/results/backtests/summary/trade_distr",
        help="Basisverzeichnis für Trade-Distributions-Grafiken (PNG)",
    )
    parser.add_argument(
        "--title",
        default="Portfolio Equity (Aggregiert)",
        help="Titel der Visualisierung",
    )
    args = parser.parse_args()

    df = combine_equity(args.root, args.out, args.wide_out, baseline=args.baseline)
    print(f"Gesamt-Equity-Kurve geschrieben: {args.out}  (Zeilen: {len(df)})")

    try:
        plot_equity(df, args.plot_out, title=args.title, baseline=args.baseline)
        print(f"Visualisierung gespeichert: {args.plot_out}")
    except Exception as e:
        print(f"Warnung: Konnte Plot nicht erstellen: {e}")
    try:
        write_expected_profit_summary(df, args.expected_profit_out, root=args.root)
        print(f"Erwartungswerte gespeichert: {args.expected_profit_out}")
    except Exception as e:
        print(f"Warnung: Konnte Erwartungswerte nicht erstellen: {e}")
    try:
        write_max_simult_trades_per_symbol(args.root, args.max_simult_trades_out)
        print(
            f"Maximale simultane Trades pro Symbol gespeichert: {args.max_simult_trades_out}"
        )
    except Exception as e:
        print(
            f"Warnung: Konnte Statistik 'max simultane Trades pro Symbol' nicht erstellen: {e}"
        )
    try:
        write_trade_distribution_plots(args.root, args.trade_distr_dir)
        print(f"Trade-Distribution-Grafiken gespeichert unter: {args.trade_distr_dir}")
    except Exception as e:
        print(f"Warnung: Konnte Trade-Distribution-Grafiken nicht erstellen: {e}")


if __name__ == "__main__":
    main()
