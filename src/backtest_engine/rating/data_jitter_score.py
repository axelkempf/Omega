from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Berechnet die ATR-Serie pro Bar.

    True Range: max(H-L, |H-C_prev|, |L-C_prev|)
    ATR: Rolling-Mean mit Fenster=period und min_periods=1 (expanding mean bis period erreicht).
    """

    if df.empty:
        return pd.Series(dtype=float)

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr_high_low = high - low
    tr_high_prev_close = np.abs(high - prev_close)
    tr_low_prev_close = np.abs(low - prev_close)

    true_range = np.maximum.reduce([tr_high_low, tr_high_prev_close, tr_low_prev_close])
    atr = pd.Series(true_range).rolling(window=period, min_periods=1).mean()
    atr.index = df.index
    return atr


def precompute_atr_cache(
    base_preloaded_data: Mapping[Tuple[str, str], pd.DataFrame],
    period: int = 14,
) -> Dict[str, pd.Series]:
    """Berechnet einmalig ATR-Serien pro Timeframe (BID-only)."""

    atr_cache: Dict[str, pd.Series] = {}
    for (tf, candle_type), df in base_preloaded_data.items():
        if candle_type.lower() != "bid":
            continue
        if tf in atr_cache:
            continue
        atr_cache[tf] = compute_atr_series(df, period=period)
    return atr_cache


def build_jittered_preloaded_data(
    base_preloaded_data: Mapping[Tuple[str, str], pd.DataFrame],
    *,
    atr_cache: Mapping[str, pd.Series],
    sigma_atr: float = 0.10,
    seed: int,
    min_price: float = 1e-9,
    fraq: float = 0.0,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Erzeugt jittered OHLC-DataFrames für alle (timeframe, candle_type) Keys.

    Jitter-Mechanik v2 (siehe design.md):

    - Pro Timeframe wird ein ε-Delta mit alternierendem Vorzeichen erzeugt.
    - Delta-Magnitude: |N(0, sigma_atr * ATR * scale)|, wobei scale aus `fraq`
      stammt (Uniform[1-fraq, 1+fraq], deterministisch via seed).
    - Ketten-Clamp: kumulativer Shift (Prefix-Sum der Deltas) wird auf OHLC
      angewandt und für BID/ASK identisch ausgerichtet.
    - Candle-Constraints werden repariert (High>=Open/Close, Low<=Open/Close),
      anschließend wird ein globaler Offset gesetzt, um `min_price` sicherzustellen.
    """

    rng = np.random.default_rng(seed)
    result: Dict[Tuple[str, str], pd.DataFrame] = {}

    timeframes = sorted({tf for (tf, _) in base_preloaded_data.keys()})
    for tf in timeframes:
        atr_series = atr_cache.get(tf)
        if atr_series is None:
            continue
        atr_array = np.nan_to_num(
            np.asarray(atr_series.to_numpy(dtype=float)), nan=0.0, posinf=0.0, neginf=0.0
        )

        scale_atr = 1.0
        scale_sigma = 1.0
        if fraq and fraq > 0:
            scale_atr = float(rng.uniform(1.0 - fraq, 1.0 + fraq))
            scale_sigma = float(rng.uniform(1.0 - fraq, 1.0 + fraq))

        sigma = abs(float(sigma_atr)) * scale_sigma
        std = sigma * atr_array * scale_atr

        n = len(std)
        if n == 0:
            continue

        alt_pattern = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        start_sign = float(rng.choice([1.0, -1.0]))
        signs = start_sign * alt_pattern

        # Deterministisches Delta: ATR * sigma (ohne Normalverteilung)
        mag = std
        delta = signs * mag
        shift = np.cumsum(delta)
        shift_series = pd.Series(shift, index=atr_series.index)

        # Debug: Erste 5 Kerzen ausgeben
        # print(f"\n[Jitter Debug] TF={tf}, scale_atr={scale_atr:.4f}, scale_sigma={scale_sigma:.4f}, start_sign={start_sign:+.0f}")
        # print(f"{'#':>2} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'ATR':>10} {'σATR':>10} {'Delta':>10} {'Shift':>10}")
        base_df_bid = next((df for (t, s), df in base_preloaded_data.items() if t == tf and s.lower() == "bid"), None)
        if base_df_bid is not None:
            for i in range(min(5, n)):
                o_val = float(base_df_bid["Open"].iloc[i]) if i < len(base_df_bid) else np.nan
                h_val = float(base_df_bid["High"].iloc[i]) if i < len(base_df_bid) else np.nan
                l_val = float(base_df_bid["Low"].iloc[i]) if i < len(base_df_bid) else np.nan
                c_val = float(base_df_bid["Close"].iloc[i]) if i < len(base_df_bid) else np.nan
                atr_val = float(atr_array[i])
                atr_sigma_val = float(std[i])
                delta_val = float(delta[i])
                shift_val = float(shift[i])
                # print(f"{i+1:2d} {o_val:10.5f} {h_val:10.5f} {l_val:10.5f} {c_val:10.5f} {atr_val:10.5f} {atr_sigma_val:10.5f} {delta_val:+10.5f} {shift_val:+10.5f}")

        for candle_type in ("bid", "ask"):
            key = (tf, candle_type)
            if key not in base_preloaded_data:
                continue

            df = base_preloaded_data[key].copy()
            base_df = base_preloaded_data[key]
            shift_aligned = shift_series.reindex(df.index)
            if shift_aligned.isnull().any():
                shift_aligned = shift_aligned.ffill().bfill().fillna(0.0)

            shift_vals = shift_aligned.to_numpy(dtype=float)
            shift_prev = np.roll(shift_vals, 1)
            shift_prev[0] = 0.0

            open_base = df["Open"].to_numpy(dtype=float)
            high_base = df["High"].to_numpy(dtype=float)
            low_base = df["Low"].to_numpy(dtype=float)
            close_base = df["Close"].to_numpy(dtype=float)

            df["Open"] = open_base + shift_prev
            df["High"] = high_base + shift_vals
            df["Low"] = low_base + shift_vals
            df["Close"] = close_base + shift_vals

            open_vals = df["Open"].to_numpy(dtype=float)
            close_vals = df["Close"].to_numpy(dtype=float)
            high_vals = df["High"].to_numpy(dtype=float)
            low_vals = df["Low"].to_numpy(dtype=float)

            high_fixed = np.maximum.reduce([high_vals, open_vals, close_vals])
            low_fixed = np.minimum.reduce([low_vals, open_vals, close_vals])

            # Guard gegen numerische Anomalien
            high_fixed = np.maximum(high_fixed, low_fixed)
            low_fixed = np.minimum(low_fixed, high_fixed)

            df["High"] = high_fixed
            df["Low"] = low_fixed

            global_min = float(np.min([open_vals.min(), close_vals.min(), high_fixed.min(), low_fixed.min()]))
            offset = 0.0
            if global_min < min_price:
                offset = float(min_price - global_min)
                for col in ("Open", "High", "Low", "Close"):
                    df[col] = df[col].to_numpy(dtype=float) + offset

            # Debug: Vor/Nach Vergleich und Validierung (nur BID)
            if candle_type.lower() == "bid":
                # print(f"\n[Jitter Validation] TF={tf}, side={candle_type}, offset={offset:.6f}")
                # print(f"{'#':>2} {'Base_O':>10} {'Base_H':>10} {'Base_L':>10} {'Base_C':>10} {'→':^3} {'Jit_O':>10} {'Jit_H':>10} {'Jit_L':>10} {'Jit_C':>10} {'✓':^3}")
                
                base_vals_open = base_df["Open"].to_numpy(dtype=float)
                base_vals_high = base_df["High"].to_numpy(dtype=float)
                base_vals_low = base_df["Low"].to_numpy(dtype=float)
                base_vals_close = base_df["Close"].to_numpy(dtype=float)
                
                final_open = df["Open"].to_numpy(dtype=float)
                final_high = df["High"].to_numpy(dtype=float)
                final_low = df["Low"].to_numpy(dtype=float)
                final_close = df["Close"].to_numpy(dtype=float)
                
                for i in range(min(5, len(df))):
                    # Validierung: High >= max(Open, Close), Low <= min(Open, Close), Low <= High
                    high_ok = final_high[i] >= max(final_open[i], final_close[i])
                    low_ok = final_low[i] <= min(final_open[i], final_close[i])
                    low_high_ok = final_low[i] <= final_high[i]
                    min_price_ok = final_open[i] >= min_price and final_close[i] >= min_price and final_high[i] >= min_price and final_low[i] >= min_price
                    all_ok = high_ok and low_ok and low_high_ok and min_price_ok
                    
                    check_mark = "✓" if all_ok else "✗"
                    # print(f"{i+1:2d} {base_vals_open[i]:10.5f} {base_vals_high[i]:10.5f} {base_vals_low[i]:10.5f} {base_vals_close[i]:10.5f} {'→':^3} {final_open[i]:10.5f} {final_high[i]:10.5f} {final_low[i]:10.5f} {final_close[i]:10.5f} {check_mark:^3}")
                    
                    if not all_ok:
                        issues = []
                        if not high_ok:
                            issues.append(f"High={final_high[i]:.5f} < max(O,C)={max(final_open[i], final_close[i]):.5f}")
                        if not low_ok:
                            issues.append(f"Low={final_low[i]:.5f} > min(O,C)={min(final_open[i], final_close[i]):.5f}")
                        if not low_high_ok:
                            issues.append(f"Low={final_low[i]:.5f} > High={final_high[i]:.5f}")
                        if not min_price_ok:
                            issues.append(f"min(OHLC)={min(final_open[i], final_close[i], final_high[i], final_low[i]):.5f} < min_price={min_price}")
                        print(f"     ⚠️ ISSUE: {', '.join(issues)}")

            result[key] = df

    return result


def compute_data_jitter_score(
    base_metrics: Mapping[str, float],
    jitter_metrics_list: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Berechnet den Data-Jitter-Score via Robustness-1-Penalty."""

    from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1

    return float(
        compute_robustness_score_1(
            base_metrics,
            jitter_metrics_list,
            penalty_cap=penalty_cap,
        )
    )


def _stable_data_jitter_seed(base_seed: int, repeat_idx: int) -> int:
    """Erzeugt stabilen 32-bit Seed pro Repeat."""

    payload = f"{int(base_seed)}|data_jitter|{int(repeat_idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**32)
