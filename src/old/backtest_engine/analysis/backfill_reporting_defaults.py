"""Zentrale Reporting-Konstante für Backfill-Backtests.

Diese Datei definiert die Reporting-Einstellungen, die beim Erstellen eines
Backfill-Snapshots (frozen_snapshot_backfill.json) hart überschrieben werden.
Dadurch sind alle Backfill-Runs konsistent und unabhängig von externen Configs.

Hinweis: dev_mode ist hier bewusst auf False gesetzt, um versehentliche
Debug-Einstellungen in produktiven Backfill-Läufen zu vermeiden.
"""

from __future__ import annotations

from typing import Any, Dict

BACKFILL_REPORTING_DEFAULTS: Dict[str, Any] = {
    "param_jitter_include_by_scenario": {
        "scenario2": [
            "atr_length",
            "atr_mult",
            "b_b_length",
            "std_factor",
            "window_length",
            "z_score_long",
            "z_score_short",
            "kalman_q",
            "kalman_r",
            "htf_ema",
            "extra_htf_ema",
        ],
        "scenario3": [
            "atr_length",
            "atr_mult",
            "b_b_length",
            "std_factor",
            "window_length",
            "z_score_long",
            "z_score_short",
            "kalman_q",
            "kalman_r",
            "htf_ema",
            "extra_htf_ema",
            "ema_length",
            "tp_min_distance",
            "max_holding_minutes",
        ],
        "scenario4": [
            "atr_length",
            "atr_mult",
            "b_b_length",
            "std_factor",
            "z_score_long",
            "z_score_short",
            "kalman_q",
            "kalman_r",
            "htf_ema",
            "extra_htf_ema",
            "garch_alpha",
            "garch_beta",
            "garch_min_periods",
        ],
        "scenario5": [
            "atr_length",
            "atr_mult",
            "b_b_length",
            "std_factor",
            "window_length",
            "z_score_long",
            "z_score_short",
            "kalman_q",
            "kalman_r",
            "htf_ema",
            "extra_htf_ema",
            "garch_alpha",
            "garch_beta",
            "garch_min_periods",
            "intraday_vol_cluster_window",
            "intraday_vol_min_points",
        ],
    },
    "param_jitter_debug": False,
    "param_jitter_debug_verbose": False,
    "param_jitter_debug_per_repeat": False,
    "data_jitter_debug_per_repeat": False,
    "debug_timing_jitter": False,
    "robust_dropout_frac": 0.10,
    "robust_dropout_runs": 3,
    "debug_trade_dropout": False,
    "dev_mode": False,
    "dev_seed": 123,
}
