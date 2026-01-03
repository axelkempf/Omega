import json
from datetime import datetime, timezone

from backtest_engine.optimizer.walkforward import walkforward_optimization
from backtest_engine.optimizer.walkforward_utils import update_master_index
from hf_engine.infra.config.paths import BACKTEST_CONFIG_DIR, WALKFORWARD_RESULTS_DIR

# === Basis-Config laden =======================================================
config_template_path = BACKTEST_CONFIG_DIR / "mean_reversion_z_score.json"
with open(config_template_path, "r") as f:
    base_config = json.load(f)

strategy_name = base_config["strategy"]["class"]
symbol = base_config["symbol"]

# Ergebnisverzeichnisse
# Uniques Run-Verzeichnis, damit vorherige WF-Läufe nicht überschrieben werden
run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
walkforward_root = WALKFORWARD_RESULTS_DIR / f"run_{run_id}"
walkforward_root.mkdir(parents=True, exist_ok=True)
export_results_path = walkforward_root / "ratings_summary.json"

# === Parameterraum (korrigiert auf die echten Typen) ==========================
# Siehe Defaults in der JSON: ema_* (int), bollinger_period (int),
# bollinger_std_factor (float), ratio_threshold (float),
# min_pip_mult (int), sl_mult (float), max_holding_minutes (int).  # JSON-Quelle
param_grid = {
    # EMA-Längen
    # "ema_length": {"type": "int", "low": 10, "high": 50, "step": 2},
    "htf_ema": {"type": "int", "low": 70, "high": 90, "step": 10},
    # Bollinger
    "b_b_length": {"type": "int", "low": 60, "high": 64, "step": 2},
    "std_factor": {"type": "float", "low": 1.4, "high": 1.8, "step": 0.1},
    # Z-Score
    # "window_length": {"type": "int", "low": 40, "high": 250, "step": 10},
    # "z_score_short": {"type": "float", "low": 1.0, "high": 4.5, "step": 0.1},
    "z_score_long": {"type": "float", "low": -0.85, "high": -0.75, "step": 0.01},
    "kalman_r": {"type": "float", "low": 0.35, "high": 0.38, "step": 0.006},
    "kalman_q": {"type": "float", "low": 0.2, "high": 0.4, "step": 0.1},
    # Risk/Exit
    # "atr_length": {"type": "int", "low": 7, "high": 21},
    "atr_mult": {"type": "float", "low": 1.4, "high": 1.8, "step": 0.2},
    # GARCH
    "garch_alpha": {"type": "float", "low": 0.11, "high": 0.15, "step": 0.01},
    "garch_beta": {"type": "float", "low": 0.74, "high": 0.82, "step": 0.02},
    "garch_min_periods": {"type": "int", "low": 40, "high": 60, "step": 10},
    # Intraday Volatility Clustering
    # "cluster_hysteresis_bars": {"type": "int", "low": 0, "high": 6, "step": 1},
    # "intraday_vol_min_points": {"type": "int", "low": 60, "high": 140, "step": 10},
    # "intraday_vol_cluster_window": {"type": "int", "low": 100, "high": 300, "step": 10},
    # --- Szenario 6: Timeframes & Mode ---------------------------------------
    # Optuna benötigt hashbare Kategorien → Tupel statt Listen.
    # Die Strategie akzeptiert jede Iterable (wird per for-Schleife iteriert).
    # "scenario6_timeframes": {
    #     "type": "categorical",
    #     "choices": [
    #         ("M30", "H1"),
    #         ("M30",),
    #         ("H1",),
    #         ("H1", "H4"),
    #         ("M30", "H1", "H4"),
    #     ],
    # },
    # "scenario6_mode": {
    #     "type": "categorical",
    #     "choices": ["all", "any"],
    # },
    # --- Szenario 6: Per‑TF Parameter (flattened keys) ------------------------
    # M30 – Long/Short
    # "scenario6_M30_long_window_length": {"type": "int", "low": 40, "high": 140, "step": 10},
    # "scenario6_M30_short_window_length": {"type": "int", "low": 40, "high": 140, "step": 10},
    # "scenario6_M30_long_b_b_length": {"type": "int", "low": 18, "high": 30, "step": 2},
    # "scenario6_M30_short_b_b_length": {"type": "int", "low": 18, "high": 30, "step": 2},
    # "scenario6_M30_long_std_factor": {"type": "float", "low": 1.4, "high": 2.6, "step": 0.2},
    # "scenario6_M30_short_std_factor": {"type": "float", "low": 1.4, "high": 2.6, "step": 0.2},
    # "scenario6_M30_long_z_score_long": {"type": "float", "low": -1.2, "high": -0.1, "step": 0.1},
    # "scenario6_M30_short_z_score_short": {"type": "float", "low": 0.1, "high": 1.2, "step": 0.1},
    # H1 – Long/Short
    # "scenario6_H1_long_window_length": {"type": "int", "low": 60, "high": 200, "step": 10},
    # "scenario6_H1_short_window_length": {"type": "int", "low": 60, "high": 200, "step": 10},
    # "scenario6_H1_long_b_b_length": {"type": "int", "low": 18, "high": 30, "step": 2},
    # "scenario6_H1_short_b_b_length": {"type": "int", "low": 18, "high": 30, "step": 2},
    # "scenario6_H1_long_std_factor": {"type": "float", "low": 1.4, "high": 2.6, "step": 0.2},
    # "scenario6_H1_short_std_factor": {"type": "float", "low": 1.4, "high": 2.6, "step": 0.2},
    # "scenario6_H1_long_z_score_long": {"type": "float", "low": -1.2, "high": -0.1, "step": 0.1},
    # "scenario6_H1_short_z_score_short": {"type": "float", "low": 0.1, "high": 1.2, "step": 0.1},
}

# === Bewertungs-Schwellen (realistisch & nicht zu scharf) =====================
rating_thresholds = {
    "min_winrate": 55,  # nicht zu hoch, um robuste Parameter nicht auszuschließen
    "min_avg_r": 0.80,
    "min_profit": 0,
    "min_profit_factor": 1.10,
    "max_drawdown": 3000,
}

# === Walkforward-Run starten (mit neuen Optionen) =============================
# Wichtige Punkte:
# - n_trials wird in der neuen walkforward.py mind. auf die geschätzten min_trials angehoben
# - kfold_splits=3: zeitbasierte CV im Trainingsfenster
# - robustness_*: Parameter-Jitter-Stabilitätstests
# - preload_mode="window": lädt nur das benötigte Zeitfenster (RAM/I/O-schonend)
results = walkforward_optimization(
    config_template_path=str(config_template_path),
    param_grid=param_grid,
    train_days=180,
    test_days=60,
    buffer_days=2,  # kleiner Puffer zwischen Train und Test
    roll_interval_days=60,  # dichteres Rollen für mehr OOS-Stichproben
    rating_thresholds=rating_thresholds,
    # Beschleunigte Basis (A/B‑Check): Trials reduzieren
    n_trials=300,
    walkforward_root=str(walkforward_root),
    export_results_path=str(export_results_path),
    seed=42,
    # Neue Optionen aus der überarbeiteten Pipeline:
    preload_mode="window",  # "full" | "window"
    min_trades=7,
    min_days_active=5,
    kfold_splits=1,
    # Robustness‑Jitter für Messlauf deaktivieren (reine Optimierungszeit)
    robustness_jitter_frac=0.08,
    robustness_repeats=1,
    n_jobs=2,  # None → automatisch (os.cpu_count()-1)
    final_mode="smart",  # "smart" | "grid"
)

# === master_index pflegen =====================================================
# update_master_index(
#     strategy_name=strategy_name,
#     run_path=str(walkforward_root),
#     summary_path=str(export_results_path),
# )
