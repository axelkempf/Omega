# hf_engine/strategies/mean_reversion_z_score/live/master_config.py
from datetime import time

MASTER_CONFIG = {
    "global_defaults": {
        "strategy_name": "Mean_Reversion_Z_Score",
        "order_type": "market",
    },
    "setups": [
        {
            "name": "eurusd_m5_m15_long_s2",
            "enabled": True,
            # Markt-Kontext
            "symbol": "EURUSD",
            "timeframes": ["M5"],
            # Stabile, fixe Magic-Nummern pro Timeframe
            "magic_numbers": 10035,
            "directions": ["long"],  # ["long"], ["short"], ["long","short"]
            # Szenario-Whitelist (nur diese Signale dürfen am Ende wirklich handeln)
            "scenarios": ["szenario_2_long"],
            # Eigene Handelszeiten
            "session": {
                "session_start": time(8, 0),
                "session_end": time(17, 30),
            },
            # Eigenes Risiko (voll separater Risiko-Topf)
            "risk": {
                "start_capital": 25_000,
                "risk_per_trade_pct": 0.20,
                "max_drawdown_pct": 5.0,
            },
            # Trend-/Filter-Defaults für dieses Setup
            # Diese Infos gehen später in runtime_cfg als Root-Keys,
            # damit SzenarioEvaluator sie wie bisher lesen kann.
            "trend": {
                "daily_trend_ema_period": 50,
                "daily_trend_relation_long": "above",
                "daily_trend_relation_short": "below",
                "h4_trend_ema_period": 50,
                "h4_trend_relation_long": "above",
                "h4_trend_relation_short": "below",
                "h1_trend_ema_period": 50,
                "h1_trend_relation_long": "above",
                "h1_trend_relation_short": "below",
                "trend_min_filters_required": 2,
            },
            # Strategy-Parameter pro Richtung
            "params": {
                "long": {
                    "ema_period": 13,
                    "atr_length": 14,
                    "atr_mult": 1.2,
                    "bb_length": 15,
                    "bb_std": 1.0,
                    "zscore_length": 50,
                    "kalman_window": 80,
                    "kalman_r": 0.01,
                    "kalman_q": 1,
                    "z_score": -1.6,
                    "cooldown_minutes": 10,
                    "max_holding_minutes": 40,
                    # Trendfilter pro Richtung (überschreibt ggf. trend.*)
                    "daily_trend_ema_period": 50,
                    "daily_trend_relation": "above",
                    "h4_trend_ema_period": 50,
                    "h4_trend_relation": "above",
                    "h1_trend_ema_period": 50,
                    "h1_trend_relation": "above",
                    "trend_min_filters_required": 2,
                },
            },
        },
        {
            "name": "S5_m5_all",
            "enabled": True,
            "symbol": "EURUSD",
            "timeframes": ["M5"],
            "magic_numbers": 100025,
            "directions": ["long", "short"],
            "scenarios": ["szenario_5_short", "szenario_5_long"],
            "session": {
                "session_start": time(23, 0),
                "session_end": time(21, 0),
            },
            "risk": {
                "start_capital": 100_000,
                "risk_per_trade_pct": 0.05,
                "max_drawdown_pct": 50.0,
            },
            "trend": {
                "daily_trend_ema_period": 50,
                "daily_trend_relation_long": "any",
                "daily_trend_relation_short": "any",
                "h4_trend_ema_period": 50,
                "h4_trend_relation_long": "any",
                "h4_trend_relation_short": "any",
                "h1_trend_ema_period": 50,
                "h1_trend_relation_long": "any",
                "h1_trend_relation_short": "any",
            },
            "params": {
                "short": {
                    "ema_period": 13,
                    "atr_length": 14,
                    "atr_mult": 1.5,
                    "bb_length": 20,
                    "bb_std": 1.5,
                    "zscore_length": 110,
                    "kalman_window": 110,
                    "kalman_r": 0.01,
                    "kalman_q": 1,
                    "z_score": 1.0,
                    "cooldown_minutes": 10,
                    "max_holding_minutes": 40,
                    # Intraday-Vol-Cluster-Parameter (Szenario 5, identisch zum Backtest)
                    "intraday_vol_feature": "garch_forecast",
                    "intraday_vol_cluster_window": 80,
                    "intraday_vol_cluster_k": 3,
                    "intraday_vol_min_points": 50,
                    "intraday_vol_log_transform": True,
                    "intraday_vol_allowed": ["low", "mid"],
                    "cluster_hysteresis_bars": 1,
                    # GARCH-Parameter für Volatilitäts-Feature (Szenario 4/5)
                    "garch_alpha": 0.01,
                    "garch_beta": 0.85,
                    "garch_omega": None,
                    "garch_use_log_returns": True,
                    "garch_scale": 100.0,
                    "garch_min_periods": 50,
                    "garch_sigma_floor": 1e-6,
                },
                "long": {
                    "ema_period": 13,
                    "atr_length": 14,
                    "atr_mult": 1.5,
                    "bb_length": 20,
                    "bb_std": 1.5,
                    "zscore_length": 110,
                    "kalman_window": 110,
                    "kalman_r": 0.01,
                    "kalman_q": 1,
                    "z_score": -1.0,
                    "cooldown_minutes": 10,
                    "max_holding_minutes": 40,
                    # Intraday-Vol-Cluster-Parameter (Szenario 5, identisch zum Backtest)
                    "intraday_vol_feature": "garch_forecast",
                    "intraday_vol_cluster_window": 80,
                    "intraday_vol_cluster_k": 3,
                    "intraday_vol_min_points": 50,
                    "intraday_vol_log_transform": True,
                    "intraday_vol_allowed": ["low", "mid"],
                    "cluster_hysteresis_bars": 1,
                    # GARCH-Parameter für Volatilitäts-Feature (Szenario 4/5)
                    "garch_alpha": 0.01,
                    "garch_beta": 0.85,
                    "garch_omega": None,
                    "garch_use_log_returns": True,
                    "garch_scale": 100.0,
                    "garch_min_periods": 50,
                    "garch_sigma_floor": 1e-6,
                },
            },
        },
        # Beispiel‑Setup: Szenario 6 (Multi‑TF Overlay wie im Backtest)
        {
            "name": "eurusd_m5_s6_multi_tf",
            "enabled": True,
            "symbol": "EURUSD",
            "timeframes": ["M5"],
            # Ein einzelner stabiler Magic‑Wert für dieses Setup
            "magic_number": 1090601,
            "directions": ["long", "short"],
            # Nur Szenario 6 handeln
            "scenarios": ["szenario_6_long", "szenario_6_short"],
            "session": {
                "session_start": time(8, 0),
                "session_end": time(21, 0),
            },
            "risk": {
                "start_capital": 25_000,
                "risk_per_trade_pct": 0.10,
                "max_drawdown_pct": 5.0,
            },
            # Globale Trendfilter Defaults
            "trend": {
                "daily_trend_ema_period": 50,
                "daily_trend_relation_long": "above",
                "daily_trend_relation_short": "below",
                "h4_trend_ema_period": 50,
                "h4_trend_relation_long": "above",
                "h4_trend_relation_short": "below",
                "h1_trend_ema_period": 50,
                "h1_trend_relation_long": "above",
                "h1_trend_relation_short": "below",
                # Anzahl aktiver Filter, die erfüllt sein müssen
                "trend_min_filters_required": 2,
            },
            # Baseline‑Parameter je Richtung (Primär‑TF) – für Szenario 2 Basis‑Signal
            "params": {
                "long": {
                    "ema_period": 18,
                    "atr_length": 14,
                    "atr_mult": 4.0,
                    "bb_length": 52,
                    "bb_std": 2.8,
                    "zscore_length": 90,
                    "kalman_window": 90,
                    "kalman_q": 1.5,
                    "kalman_r": 0.49,
                    "z_score": -0.47,
                    "tp_min_distance": 0.0007,
                    "max_holding_minutes": 2160,
                },
                "short": {
                    "ema_period": 18,
                    "atr_length": 14,
                    "atr_mult": 4.0,
                    "bb_length": 52,
                    "bb_std": 2.8,
                    "zscore_length": 90,
                    "kalman_window": 90,
                    "kalman_q": 1.5,
                    "kalman_r": 0.49,
                    "z_score": 1.3,
                    "tp_min_distance": 0.0007,
                    "max_holding_minutes": 2160,
                },
            },
            # Szenario 6: Extra‑Timeframes + TF‑spezifische Parameter
            "scenario6_mode": "all",  # "all" (alle TF müssen ok sein) | "any" (mind. eins)
            "scenario6_timeframes": ["M30", "H1"],
            "scenario6_params": {
                "M30": {
                    "long": {
                        "b_b_length": 30,
                        "std_factor": 2.0,
                        "window_length": 80,
                        "z_score_long": -1.60,
                    },
                    "short": {
                        "b_b_length": 30,
                        "std_factor": 2.0,
                        "window_length": 80,
                        "z_score_short": 1.60,
                    },
                },
                "H1": {
                    "long": {
                        "b_b_length": 40,
                        "std_factor": 2.5,
                        "window_length": 100,
                        "z_score_long": -1.8,
                    },
                    "short": {
                        "b_b_length": 40,
                        "std_factor": 2.5,
                        "window_length": 100,
                        "z_score_short": 1.8,
                    },
                },
            },
        },
        {
            "name": "crypto_m5_s2",
            "enabled": True,
            # Markt-Kontext
            "asset_class": "crypto",
            "symbols": [
                "SOLUSDT",
                "ETHUSDT",
                "BTCUSDT",
            ],
            "timeframes": ["M5"],
            # Stabile, fixe Magic-Nummern pro Timeframe
            "magic_numbers": 1000012034,
            "directions": ["long", "short"],  # ["long"], ["short"], ["long","short"]
            # Szenario-Whitelist (nur diese Signale dürfen am Ende wirklich handeln)
            "scenarios": ["szenario_2_long", "szenario_2_short"],
            # Eigene Handelszeiten
            "session": {
                "session_start": time(0, 0),
                "session_end": time(0, 0),
            },
            # Eigenes Risiko (voll separater Risiko-Topf)
            "risk": {
                "start_capital": 100_000,
                "risk_per_trade_pct": 0.1,
                "max_drawdown_pct": 50.0,
            },
            # Trend-/Filter-Defaults für dieses Setup
            # Diese Infos gehen später in runtime_cfg als Root-Keys,
            # damit SzenarioEvaluator sie wie bisher lesen kann.
            "trend": {
                "daily_trend_ema_period": 50,
                "daily_trend_relation_long": "any",
                "daily_trend_relation_short": "any",
                "h4_trend_ema_period": 50,
                "h4_trend_relation_long": "any",
                "h4_trend_relation_short": "any",
                "h1_trend_ema_period": 50,
                "h1_trend_relation_long": "any",
                "h1_trend_relation_short": "any",
            },
            # Strategy-Parameter pro Richtung
            "params": {
                "long": {
                    "ema_period": 13,
                    "atr_length": 14,
                    "atr_mult": 1.2,
                    "bb_length": 15,
                    "bb_std": 1.0,
                    "zscore_length": 50,
                    "kalman_window": 80,
                    "kalman_r": 0.01,
                    "kalman_q": 1,
                    "z_score": -1.6,
                    "cooldown_minutes": 10,
                    "max_holding_minutes": 40,
                    # Trendfilter pro Richtung (überschreibt ggf. trend.*)
                    "daily_trend_ema_period": 50,
                    "daily_trend_relation": "above",
                    "h4_trend_ema_period": 50,
                    "h4_trend_relation": "above",
                    "h1_trend_ema_period": 50,
                    "h1_trend_relation": "above",
                    "trend_min_filters_required": 2,
                },
            },
        },
    ],
}
