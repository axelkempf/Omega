from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "old"))

from src.old.backtest_engine.core.indicator_cache import IndicatorCache  # noqa: E402


class Candle:
    def __init__(self, close: float) -> None:
        self.open = close - 0.01
        self.high = close + 0.02
        self.low = close - 0.02
        self.close = close
        self.volume = 1.0


def make_candle(close: float) -> Candle:
    return Candle(close)


class Slice:
    def __init__(
        self,
        indicators: IndicatorCache,
        bid_candles: list[Candle],
        ask_candles: list[Candle],
        idx: int,
    ) -> None:
        self.indicators = indicators
        self._bid = bid_candles
        self._ask = ask_candles
        self.index = idx

    def latest(self, tf: str, price_type: str = "bid") -> Candle:
        if price_type == "bid":
            return self._bid[self.index]
        return self._ask[self.index]


def series_value(series, idx: int) -> float | None:
    if series is None:
        return None
    if idx < 0 or idx >= len(series):
        return None
    val = series.iloc[idx]
    if not np.isfinite(val):
        return None
    return float(val)


def signal_payload(signal: dict[str, Any] | None) -> dict[str, Any] | None:
    if not signal:
        return None
    return {
        "direction": str(signal.get("direction")),
        "entry": float(signal.get("entry")),
        "sl": float(signal.get("sl")),
        "tp": float(signal.get("tp")),
    }


def build_data(
    closes: list[float],
    h1_closes: list[float],
    ask_spread: float = 0.0002,
) -> dict[str, Any]:
    bid_candles = [make_candle(c) for c in closes]
    ask_candles = [make_candle(c + ask_spread) for c in closes]

    h1_candles = [make_candle(c) for c in h1_closes]
    rep = max(1, len(closes) // max(1, len(h1_candles)))
    h1_bid = []
    for candle in h1_candles:
        h1_bid.extend([candle] * rep)
    h1_bid = h1_bid[: len(closes)]
    h1_ask = [make_candle(c.close + ask_spread) for c in h1_bid]

    return {
        "bid_closes": closes,
        "ask_closes": [c + ask_spread for c in closes],
        "h1_bid_closes": h1_closes,
        "h1_ask_closes": [c + ask_spread for c in h1_closes],
        "bid_candles": bid_candles,
        "ask_candles": ask_candles,
        "h1_bid": h1_bid,
        "h1_ask": h1_ask,
    }


def build_payload(
    strategy_cls: type,
    params: dict[str, Any],
    closes: list[float],
    h1_closes: list[float],
) -> dict[str, Any]:
    data = build_data(closes, h1_closes)
    multi = {
        "M5": {"bid": data["bid_candles"], "ask": data["ask_candles"]},
        "H1": {"bid": data["h1_bid"], "ask": data["h1_ask"]},
    }
    cache = IndicatorCache(multi)
    idx = len(data["bid_closes"]) - 1
    slice_ = Slice(cache, data["bid_candles"], data["ask_candles"], idx)

    strat = build_strategy(strategy_cls, params)

    signals: dict[str, Any] = {}
    for scenario_id in range(1, 7):
        long_fn = getattr(strat, f"_evaluate_long_{scenario_id}")
        short_fn = getattr(strat, f"_evaluate_short_{scenario_id}")
        signals[f"{scenario_id}_long"] = signal_payload(
            long_fn(
                slice_,
                data["bid_candles"][idx],
                data["ask_candles"][idx],
            )
        )
        signals[f"{scenario_id}_short"] = signal_payload(
            short_fn(
                slice_,
                data["bid_candles"][idx],
                data["ask_candles"][idx],
            )
        )

    indicators = {
        "atr": series_value(cache.atr("M5", "bid", params["atr_length"]), idx),
        "ema": series_value(cache.ema("M5", "bid", params["ema_length"]), idx),
        "kalman_z": series_value(
            cache.kalman_zscore(
                "M5",
                "bid",
                window=params["window_length"],
                R=params["kalman_r"],
                Q=params["kalman_q"],
            ),
            idx,
        ),
        "kalman_garch_z": series_value(
            cache.kalman_garch_zscore(
                "M5",
                "bid",
                R=params["kalman_r"],
                Q=params["kalman_q"],
                alpha=params["garch_alpha"],
                beta=params["garch_beta"],
                omega=params["garch_omega"],
                use_log_returns=params["garch_use_log_returns"],
                scale=params["garch_scale"],
                min_periods=params["garch_min_periods"],
                sigma_floor=params["garch_sigma_floor"],
            ),
            idx,
        ),
    }

    bb_upper, bb_mid, bb_lower = cache.bollinger(
        "M5", "bid", period=params["b_b_length"], std_factor=params["std_factor"]
    )
    indicators["bollinger"] = {
        "upper": series_value(bb_upper, idx),
        "middle": series_value(bb_mid, idx),
        "lower": series_value(bb_lower, idx),
    }

    h1_kz = cache.kalman_zscore_stepwise(
        "H1",
        "bid",
        window=params["scenario6_params"]["H1"]["window_length"],
        R=params["scenario6_params"]["H1"]["kalman_r"],
        Q=params["scenario6_params"]["H1"]["kalman_q"],
    )
    h1_upper, h1_mid, h1_lower = cache.bollinger_stepwise(
        "H1",
        "bid",
        period=params["scenario6_params"]["H1"]["b_b_length"],
        std_factor=params["scenario6_params"]["H1"]["std_factor"],
    )
    h1_price = cache.get_closes("H1", "bid")
    indicators["scenario6_debug"] = {
        "kalman_z_step": series_value(h1_kz, idx),
        "lower": series_value(h1_lower, idx),
        "price": series_value(h1_price, idx),
        "upper": series_value(h1_upper, idx),
        "middle": series_value(h1_mid, idx),
    }
    chain_ok, chain_meta = strat._scenario6_evaluate_chain(slice_, idx, "long")
    indicators["scenario6_chain_ok"] = chain_ok
    indicators["scenario6_chain_meta"] = chain_meta

    return {
        "index": idx,
        "params": params,
        "data": {
            "bid_closes": data["bid_closes"],
            "ask_closes": data["ask_closes"],
            "h1_bid_closes": data["h1_bid_closes"],
            "h1_ask_closes": data["h1_ask_closes"],
        },
        "signals": signals,
        "indicators": indicators,
    }


def build_strategy(cls, params: dict[str, Any]) -> Any:
    strat = cls.__new__(cls)
    strat.symbol = "EURUSD"
    strat.timeframe = "M5"
    strat.pip_size = 0.0001

    strat.atr_length = int(params["atr_length"])
    strat.atr_mult = float(params["atr_mult"])
    strat.b_b_length = int(params["b_b_length"])
    strat.std_factor = float(params["std_factor"])
    strat.window_length = int(params["window_length"])
    strat.z_score_long = float(params["z_score_long"])
    strat.z_score_short = float(params["z_score_short"])
    strat.ema_length = int(params["ema_length"])
    strat.kalman_r = float(params["kalman_r"])
    strat.kalman_q = float(params["kalman_q"])
    strat.tp_min_distance = float(params["tp_min_distance"])

    strat.direction_filter = "both"
    strat.enabled_scenarios = {1, 2, 3, 4, 5, 6}
    strat.allowed_labels = set()
    strat.use_position_manager = False
    strat.position_manager = None

    strat.htfA_tf = str(params.get("htf_tf", "NONE")).upper()
    strat.htfA_ema = int(params.get("htf_ema", 0))
    strat.htfA_filter = str(params.get("htf_filter", "none")).lower()
    strat.htfB_tf = str(params.get("extra_htf_tf", "NONE")).upper()
    strat.htfB_ema = int(params.get("extra_htf_ema", 0))
    strat.htfB_filter = str(params.get("extra_htf_filter", "none")).lower()

    strat.garch_alpha = float(params["garch_alpha"])
    strat.garch_beta = float(params["garch_beta"])
    strat.garch_omega = float(params["garch_omega"])
    strat.garch_use_log_returns = bool(params.get("garch_use_log_returns", True))
    strat.garch_scale = float(params.get("garch_scale", 100.0))
    strat.garch_min_periods = int(params.get("garch_min_periods", 50))
    strat.garch_sigma_floor = float(params.get("garch_sigma_floor", 1e-6))
    strat.local_z_lookback = int(params.get("local_z_lookback", 0))

    strat.intraday_vol_feature = str(params["intraday_vol_feature"]).lower()
    strat.intraday_vol_cluster_window = int(params["intraday_vol_cluster_window"])
    strat.intraday_vol_cluster_k = int(params["intraday_vol_cluster_k"])
    strat.intraday_vol_log_transform = bool(
        params.get("intraday_vol_log_transform", True)
    )
    strat.intraday_vol_min_points = int(params.get("intraday_vol_min_points", 1))
    strat.intraday_vol_garch_lookback = int(
        params.get("intraday_vol_garch_lookback", 500)
    )
    strat.cluster_hysteresis_bars = int(params.get("cluster_hysteresis_bars", 0))
    strat.intraday_vol_allowed = list(params.get("intraday_vol_allowed", []))
    strat._vol_cluster_cache = {}
    strat._local_z_cache = {}

    scenario6_mode = str(params.get("scenario6_mode", "all")).lower()
    strat.scenario6_mode = scenario6_mode if scenario6_mode in ("all", "any") else "all"
    strat.scenario6_timeframes = [
        str(tf).upper() for tf in params.get("scenario6_timeframes", [])
    ]
    strat.scenario6_params = params.get("scenario6_params", {})

    strat.news_filter = None
    strat.portfolio = None
    return strat


def main() -> None:
    mrz_module = importlib.import_module(
        "src.old.strategies.mean_reversion_z_score.backtest.backtest_strategy"
    )
    MeanReversionZScoreStrategy = mrz_module.MeanReversionZScoreStrategy
    params = {
        "atr_length": 3,
        "atr_mult": 2.0,
        "b_b_length": 3,
        "std_factor": 1.0,
        "window_length": 3,
        "z_score_long": -0.5,
        "z_score_short": 0.5,
        "ema_length": 3,
        "kalman_r": 0.5,
        "kalman_q": 0.1,
        "garch_alpha": 0.1,
        "garch_beta": 0.8,
        "garch_omega": 0.00001,
        "garch_min_periods": 2,
        "garch_use_log_returns": True,
        "garch_scale": 100.0,
        "garch_sigma_floor": 1e-6,
        "tp_min_distance": 0.0001,
        "htf_filter": "none",
        "htf_tf": "H1",
        "htf_ema": 3,
        "extra_htf_tf": "NONE",
        "extra_htf_filter": "none",
        "scenario6_mode": "all",
        "scenario6_timeframes": ["H1"],
        "scenario6_params": {
            "H1": {
                "window_length": 3,
                "b_b_length": 3,
                "std_factor": 0.5,
                "kalman_r": 0.5,
                "kalman_q": 0.1,
                "z_score_long": -0.5,
                "z_score_short": 0.5,
            }
        },
        "intraday_vol_feature": "atr_points",
        "intraday_vol_cluster_window": 5,
        "intraday_vol_cluster_k": 1,
        "intraday_vol_allowed": [
            "low",
            "mid",
            "high",
            "very_high",
            "extreme",
        ],
        "intraday_vol_min_points": 1,
        "cluster_hysteresis_bars": 0,
    }

    long_closes = [1.0] * 8 + [0.95, 0.94, 0.93, 0.92]
    long_h1 = [1.0, 0.96, 0.92]
    short_closes = [1.0] * 8 + [1.06, 1.07, 1.08, 1.09]
    short_h1 = [1.0, 1.04, 1.08]

    payload = {
        "cases": {
            "long": build_payload(
                MeanReversionZScoreStrategy, params, long_closes, long_h1
            ),
            "short": build_payload(
                MeanReversionZScoreStrategy, params, short_closes, short_h1
            ),
        }
    }

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
