from __future__ import annotations

from typing import Any, Dict

from .portfolio_runtime import build_runtime_configs
from .strategy import MeanReversionZScoreStrategy


class MeanReversionZScoreSetupStrategy(MeanReversionZScoreStrategy):
    """
    Strategy-Wrapper pro Setup×Timeframe basierend auf MASTER_CONFIG.

    Verwendung (engine JSON):
      module: "mean_reversion_z_score.live.portfolio_strategy"
      class:  "MeanReversionZScoreSetupStrategy"
      init_args: ["<setup_name>", "<timeframe>"]

    Beispiel:
      init_args: ["eurusd_m5_m15_long_s2", "M15"]

    Die Klasse erzeugt eine runtime-config wie in portfolio_runtime.build_runtime_configs()
    und übernimmt die Logik von MeanReversionZScoreStrategy (generate_signal etc.).
    """

    def __init__(self, setup_name: str, timeframe: str | None = None):
        if not setup_name:
            raise ValueError("setup_name ist ein Pflichtparameter")

        # Finde passende Job-Config(s) aus dem Master-Setup
        jobs = build_runtime_configs()
        setup_jobs = [j for j in jobs if j.get("setup_name") == setup_name]
        if not setup_jobs:
            combos = [f"{j.get('setup_name')}:{j.get('timeframe')}" for j in jobs]
            raise ValueError(
                f"Setup nicht gefunden: {setup_name}. Verfügbar: {', '.join(combos)}"
            )

        # Timeframe bestimmen: Falls angegeben → nutzen, sonst auto-infer bei eindeutiger Auswahl
        tf_norm = None
        if timeframe is not None:
            tf_norm = str(timeframe).strip().upper()
        else:
            unique_tfs = sorted(
                {str(j.get("timeframe", "")).upper() for j in setup_jobs}
            )
            if len(unique_tfs) == 1:
                tf_norm = unique_tfs[0]
            else:
                raise ValueError(
                    f"Mehrere Timeframes für Setup '{setup_name}' gefunden: {', '.join(unique_tfs)}. "
                    "Bitte 'timeframe' als zweiten Init-Parameter angeben."
                )

        selected: Dict[str, Any] | None = None
        for job in setup_jobs:
            if str(job.get("timeframe", "")).upper() == tf_norm:
                selected = job
                break

        if not selected:
            combos = [f"{j.get('setup_name')}:{j.get('timeframe')}" for j in setup_jobs]
            raise ValueError(
                f"Setup×TF nicht gefunden: {setup_name}:{tf_norm}. Verfügbar: {', '.join(combos)}"
            )

        cfg = dict(selected.get("config") or {})
        # defensive Sicherstellung zentraler Keys
        if not cfg.get("symbols"):
            sym = selected.get("symbol")
            cfg["symbols"] = [sym] if sym else []
        cfg["timeframe"] = tf_norm

        # KEIN super().__init__ – Eltern-__init__ erwartet Modulnamen.
        # Stattdessen direkt die Felder setzen, die generate_signal benötigt.
        self.config = cfg
        self.magic_number = cfg.get("magic_number")
        self.timeframe = cfg.get("timeframe")
        self.cooldown_minutes = cfg.get("cooldown_minutes")
        self.szenarien = None
