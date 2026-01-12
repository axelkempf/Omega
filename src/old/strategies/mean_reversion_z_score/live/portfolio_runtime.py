from datetime import datetime
from typing import Any, Dict, List

from .master_config import MASTER_CONFIG
from .scenarios import SzenarioEvaluator


def derive_direction_filter(directions: List[str]) -> str:
    dset = {d.lower() for d in directions}
    if dset == {"long"}:
        return "long"
    if dset == {"short"}:
        return "short"
    return "both"


def within_session(session_cfg: Dict[str, Any], now: datetime) -> bool:
    """
    Prüft, ob wir gerade im Handelsfenster liegen.
    session_cfg: {"session_start": time(...), "session_end": time(...)}
    Achtung: einfache Variante ohne Overnight-Handel (23 -> 07). Kannst du bei Bedarf erweitern.
    """
    start = session_cfg["session_start"]
    end = session_cfg["session_end"]

    current_t = now.time()

    # Standardfall: Start < End am selben Tag
    if start <= end:
        return start <= current_t <= end

    # Overnight-Fall (z.B. 23:00 bis 07:00)
    # Dann ist erlaubt, wenn current >= start ODER current <= end
    return (current_t >= start) or (current_t <= end)


def build_runtime_configs() -> List[Dict[str, Any]]:
    """
    Erzeugt eine Liste von Jobs.
    Jeder Job enthält:
      - setup_name
      - symbol
      - timeframe
      - config   (diese config ist direkt an SzenarioEvaluator übergebbar)
    """
    jobs: List[Dict[str, Any]] = []

    g = MASTER_CONFIG["global_defaults"]
    strategy_name = g["strategy_name"]
    order_type = g["order_type"]

    # Magic-Vergabe erfolgt ausschließlich per Setup-Konfiguration (ohne globalen Base):
    #  - Bevorzugt: setup["magic_numbers"][TF]
    #  - Alternativ: setup["magic_number"] (gilt für alle TF dieses Setups)

    for setup_idx, setup in enumerate(MASTER_CONFIG["setups"]):
        if not setup.get("enabled", True):
            continue

        # Ein Setup kann entweder ein einzelnes Symbol oder eine Symbol-Liste definieren
        symbols_cfg = setup.get("symbols")
        if isinstance(symbols_cfg, list) and len(symbols_cfg) > 0:
            symbols_list = [
                str(s).upper() for s in symbols_cfg if isinstance(s, str) and s.strip()
            ]
            # Für Rückwärtskompatibilität zusätzlich 'symbol' in Jobs speichern (erstes Element)
            symbol_single = symbols_list[0]
        else:
            symbol_single = setup["symbol"]
            symbols_list = [str(symbol_single).upper()]
        timeframes = setup["timeframes"]
        dir_filter = derive_direction_filter(setup["directions"])
        trend_cfg = setup.get("trend", {}) or {}
        risk_cfg = setup.get("risk", {}) or {}
        session_cfg = setup.get("session", {}) or {}

        # param_overrides für _resolve_params nachbauen:
        # Wir bringen dein "params" in die alte Struktur.
        base_long = setup.get("params", {}).get("long", {})
        base_short = setup.get("params", {}).get("short", {})

        # Szenario-Whitelist als Set für schnellen Lookup
        allowed_scenarios = set(setup.get("scenarios", []))

        # Magic Number pro Setup×TF eindeutig machen (Stamm + Setup-Bucket + TF-Offset)
        # Bucket-Größe 100 hält TF-Offsets sauber im Bereich und schafft Platz für Erweiterungen.

        for tf in timeframes:
            tf_norm = str(tf).upper()

            # 1) Feste, setup-spezifische Vergabe pro Timeframe (stabil, explizit)
            magic_number = None
            fixed_map = setup.get("magic_numbers") or {}
            if isinstance(fixed_map, dict) and tf_norm in fixed_map:
                try:
                    magic_number = int(fixed_map[tf_norm])
                except Exception:
                    magic_number = None

            # 2) Alternativ: ein einzelner fester Wert pro Setup (gilt dann für alle TFs)
            if magic_number is None:
                single = setup.get("magic_number")
                if single is not None:
                    try:
                        magic_number = int(single)
                    except Exception:
                        magic_number = None

            # 3) Keine implizite Berechnung mehr – Magic ist Pflicht
            if magic_number is None:
                raise ValueError(
                    f"Magic-Nummer fehlt für Setup '{setup.get('name')}' TF '{tf_norm}'. "
                    "Bitte 'magic_numbers' pro TF oder 'magic_number' im Setup definieren."
                )

            param_overrides = {
                "*": {
                    "*": {
                        "buy": base_long,
                        "sell": base_short,
                    }
                }
            }
            # Symbol-spezifische Blöcke für alle Symbole des Setups hinzufügen
            for sym in symbols_list:
                param_overrides.setdefault(sym.upper(), {})[tf_norm] = {
                    "buy": base_long,
                    "sell": base_short,
                }

            # Jetzt bauen wir die runtime_cfg so,
            # dass sie 1:1 die Keys enthält, die SzenarioEvaluator erwartet.
            runtime_cfg = {
                "strategy_name": strategy_name,
                "order_type": order_type,
                "magic_number": magic_number,
                # optionale Asset-Klassifizierung (z.B. "crypto")
                "asset_class": setup.get("asset_class"),
                # Evaluator erwartet z.B.:
                "symbols": symbols_list,
                "timeframe": tf_norm,
                "direction_filter": dir_filter,
                # Session & Risk
                "session": session_cfg,
                "risk": risk_cfg,
                # Trendfilter (flatten aus trend_cfg auf Root-Level)
                "daily_trend_ema_period": trend_cfg.get("daily_trend_ema_period", 50),
                "h4_trend_ema_period": trend_cfg.get("h4_trend_ema_period", 50),
                "h1_trend_ema_period": trend_cfg.get("h1_trend_ema_period", 50),
                "daily_trend_relation_long": trend_cfg.get(
                    "daily_trend_relation_long", "above"
                ),
                "daily_trend_relation_short": trend_cfg.get(
                    "daily_trend_relation_short", "below"
                ),
                "h4_trend_relation_long": trend_cfg.get(
                    "h4_trend_relation_long", "above"
                ),
                "h4_trend_relation_short": trend_cfg.get(
                    "h4_trend_relation_short", "below"
                ),
                "h1_trend_relation_long": trend_cfg.get(
                    "h1_trend_relation_long", "above"
                ),
                "h1_trend_relation_short": trend_cfg.get(
                    "h1_trend_relation_short", "below"
                ),
                # Param overrides kompatibel zum alten Resolver
                "param_overrides": param_overrides,
                # Für die finale Handlungskontrolle
                "allowed_scenarios": allowed_scenarios,
            }

            # Optionale Erweiterungen: GARCH-/Cluster-Parameter direkt auf Root-Level
            extra_keys = [
                # GARCH long/short (für Szenario 4/5)
                "garch_alpha_long",
                "garch_beta_long",
                "garch_omega_long",
                "garch_use_log_returns_long",
                "garch_scale_long",
                "garch_min_periods_long",
                "garch_sigma_floor_long",
                "garch_alpha_short",
                "garch_beta_short",
                "garch_omega_short",
                "garch_use_log_returns_short",
                "garch_scale_short",
                "garch_min_periods_short",
                "garch_sigma_floor_short",
                # Intraday-Vol-Cluster (Szenario 5)
                "intraday_vol_feature",
                "intraday_vol_cluster_window",
                "intraday_vol_cluster_k",
                "intraday_vol_min_points",
                "intraday_vol_log_transform",
                "intraday_vol_allowed",
                "cluster_hysteresis_bars",
            ]
            for k in extra_keys:
                if k in setup:
                    runtime_cfg[k] = setup[k]

            # Optional: Szenario 6 (Multi‑TF) aus dem Setup in die Runtime‑Config übernehmen
            # Erwartete Struktur wie im Backtest:
            #   scenario6_mode: "all" | "any"
            #   scenario6_timeframes: ["M30", "H1", ...]
            #   scenario6_params: { "M30": {"long": {..}, "short": {..}}, ... }
            s6_mode = setup.get("scenario6_mode")
            s6_tfs = setup.get("scenario6_timeframes")
            s6_params = setup.get("scenario6_params")
            if s6_mode is not None:
                runtime_cfg["scenario6_mode"] = s6_mode
            if s6_tfs is not None:
                runtime_cfg["scenario6_timeframes"] = s6_tfs
            if s6_params is not None:
                runtime_cfg["scenario6_params"] = s6_params

            jobs.append(
                {
                    "setup_name": setup["name"],
                    "symbol": symbol_single,
                    "timeframe": tf_norm,
                    "config": runtime_cfg,
                }
            )

    return jobs


def run_all_setups_once(data_provider, now: datetime):
    """
    Das ist dein neue Main-Schicht für's Live-Trading pro Tick/Loop.
    - baut alle Jobs
    - prüft Session je Job
    - ruft SzenarioEvaluator
    - whitelistet Szenarien
    - gibt fertige Signale zurück (damit du Orders platzierst)
    """
    results = []

    for job in build_runtime_configs():
        cfg = job["config"]
        tf = job["timeframe"]

        # Session-Check pro Setup
        if not within_session(cfg["session"], now):
            continue

        evaluator = SzenarioEvaluator(cfg, data_provider)

        for symbol in cfg.get("symbols", []):
            signal = evaluator.evaluate_all(symbol, tf)
            if not signal:
                continue

            # Szenario-Whitelist Check
            scenario_name = signal.get("scenario")
            allowed = cfg.get("allowed_scenarios", set())
            if allowed and scenario_name not in allowed:
                continue

            signal_out = {
                "setup_name": job["setup_name"],
                "symbol": symbol,
                "timeframe": tf,
                "scenario": scenario_name,
                "direction": signal.get("direction"),
                "sl": signal.get("sl"),
                "tp": signal.get("tp"),
                "risk": cfg["risk"],
                "magic_number": cfg["magic_number"],
                "indicators": signal.get("indicators"),
                "meta": signal.get("meta", {}),
            }

            results.append(signal_out)

    return results
