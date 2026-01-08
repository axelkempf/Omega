# engine_launcher.py
import argparse
import atexit
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from importlib import import_module
from pathlib import Path

import hf_engine.core.risk.news_filter as news_filter
from hf_engine.infra.config.paths import PROJECT_ROOT, TMP_DIR
from hf_engine.infra.logging.log_service import LogService

# NEU: Controller-Funktionen für Datafeed-Healthcheck
from ui_engine.controller import check_datafeed_health

# ----------------------
# Modulweite Zustände
# ----------------------
log_service = None  # bleibt global für atexit-kompatiblen Broker-Shutdown
shared_broker = None
active_processes = []

project_root = PROJECT_ROOT
src_root = project_root / "src"
for candidate in (project_root, src_root):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Verzeichnisse/Dateien
SHUTDOWN_SIGNAL_DIR: Path = Path(TMP_DIR)
SHUTDOWN_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# Defaults (konfigurierbar via conf)
DEFAULT_HEARTBEAT_INTERVAL_SEC = 10
DEFAULT_HEALTHCHECK_TIMEOUT_SEC = 15  # Gesamtwartezeit Datafeed-Health
DEFAULT_HEALTHCHECK_INTERVAL_SEC = 1
DEFAULT_CONTROLLER_TICK_SEC = 20

# Multiprocessing robust setzen
multiprocessing.set_start_method("spawn", force=True)


# ----------------------
# Hilfsfunktionen
# ----------------------
def _log_exc(prefix: str, level: str = "ERROR"):
    """Loggt aktuelle Exception inkl. Stacktrace konsistent über LogService."""
    msg = f"{prefix}: {traceback.format_exc()}"
    if log_service:
        log_service.log_system(msg, level=level)
    else:
        # Fallback auf stderr, falls LogService noch nicht initialisiert
        sys.stderr.write(msg + "\n")


def _cleanup_files(*paths: Path):
    for p in paths:
        try:
            if p and p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            _log_exc(f"[Launcher] Cleanup-Fehler bei {p}", level="WARNING")


def _validate_conf(conf: dict) -> None:
    """
    Minimalistische Validierung ohne externe Dependencies.
    Wir prüfen nur zwingende Felder, abhängig vom Modus.
    """
    required_base = ["account_id"]
    for key in required_base:
        if key not in conf:
            raise ValueError(f"Konfiguration unvollständig: '{key}' fehlt")

    if conf.get("data_provider_only", False):
        # Datafeed-only: MT5-Feed-Server benötigt keine Strategieangaben
        return

    # Strategiemodus
    required_broker = ["account_id", "password", "server", "terminal_path", "data_path"]
    for key in required_broker:
        if key not in conf:
            raise ValueError(f"Broker-Konfiguration unvollständig: '{key}' fehlt")

    if (
        "strategies" not in conf
        or not isinstance(conf["strategies"], list)
        or len(conf["strategies"]) == 0
    ):
        raise ValueError("Es wurden keine Strategien konfiguriert ('strategies' leer).")


def _shutdown_paths(account_id: str) -> dict:
    base = {
        "stop_file": SHUTDOWN_SIGNAL_DIR / f"stop_{account_id}.signal",
        "heartbeat_file": SHUTDOWN_SIGNAL_DIR / f"heartbeat_{account_id}.txt",
        "datafeed_conf": project_root / f"tmp/datafeed_conf_{account_id}.json",
    }
    return base


def _monitor_stop_file(
    stop_file: Path, stop_event: threading.Event, poll_sec: float = 1.0
):
    """Überwacht das Stop-Signal (Datei) und setzt das Event, sobald vorhanden."""
    try:
        while not stop_event.is_set():
            if stop_file.exists():
                stop_event.set()
                break
            time.sleep(poll_sec)
    except Exception:
        _log_exc("[Launcher] Fehler in _monitor_stop_file", level="WARNING")


def _start_heartbeat(account_id: str, stop_event: threading.Event, interval_sec: float):
    """
    Schreibt kompatibel weiterhin NUR den Unix-Timestamp in die Heartbeat-Datei,
    um bestehende Konsumenten nicht zu brechen.
    """
    hb_path = _shutdown_paths(account_id)["heartbeat_file"]

    def loop():
        try:
            while not stop_event.is_set():
                hb_path.write_text(str(time.time()))
                # Sleep mit Interrupt-Möglichkeit
                stop_event.wait(interval_sec)
        except Exception:
            _log_exc(
                f"[Launcher|{account_id}] Fehler im Heartbeat-Thread", level="WARNING"
            )

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_heartbeat(
    account_id: str, interval_sec: float | None = None
) -> threading.Event:
    """Öffentliches API (Tests/UI) für einfachen Heartbeat ohne separaten Stop-Event."""
    stop_event = threading.Event()
    interval = (
        float(interval_sec)
        if interval_sec is not None
        else DEFAULT_HEARTBEAT_INTERVAL_SEC
    )
    threading.Thread(
        target=_monitor_stop_file,
        args=(_shutdown_paths(account_id)["stop_file"], stop_event, 1.0),
        daemon=True,
    ).start()
    _start_heartbeat(account_id, stop_event, interval)
    return stop_event


def _with_log_service(account_id: str) -> LogService:
    """
    Initialisiert LogService und patcht Modul-Referenz, damit alle Komponenten
    auf dieselbe Instanz loggen.
    """
    global log_service
    log_service = LogService(account_id=account_id)
    import hf_engine.infra.logging.log_service as ls_mod

    ls_mod.log_service = log_service
    log_service.log_system(
        f"[Launcher|{account_id}] 📓 Logging gestartet in {account_id}.log"
    )
    return log_service


# ----------------------
# Hauptfunktion Child-Prozess
# ----------------------
def launch_strategy_process(conf: dict):
    global shared_broker
    global log_service

    stop_event = threading.Event()

    try:
        _validate_conf(conf)
    except Exception:
        _log_exc("[Launcher] Ungültige Konfiguration")
        sys.exit(2)

    account_id = conf.get("account_id", "default")
    paths = _shutdown_paths(account_id)

    # LogService initialisieren
    _with_log_service(account_id)

    # Alte Stop-Datei aufräumen
    if paths["stop_file"].exists():
        log_service.log_system(
            f"[Launcher|{account_id}] ⚠️ Alte Shutdown-Signaldatei gefunden – wird gelöscht."
        )
        _cleanup_files(paths["stop_file"])

    # Hintergrund-Überwachung auf Stop-Datei
    threading.Thread(
        target=_monitor_stop_file,
        args=(paths["stop_file"], stop_event, 1.0),
        daemon=True,
    ).start()

    # News Warm-Cache
    try:
        news_filter.load_news_csv()
        log_service.log_system(
            f"[Launcher|{account_id}] 📰 News‑Kalender geladen (Warm‑Cache)."
        )
    except Exception:
        _log_exc(
            f"[Launcher|{account_id}] News‑Warm‑Cache fehlgeschlagen", level="WARNING"
        )

    # Heartbeat starten (konfigurierbar)
    hb_interval = int(
        conf.get("heartbeat_interval_sec", DEFAULT_HEARTBEAT_INTERVAL_SEC)
    )
    _start_heartbeat(account_id, stop_event, hb_interval)

    try:
        # ----------------------
        # Datafeed-only Modus
        # ----------------------
        if conf.get("data_provider_only", False):
            paths["datafeed_conf"].parent.mkdir(parents=True, exist_ok=True)
            with paths["datafeed_conf"].open("w", encoding="utf-8") as f:
                json.dump(conf, f)
            os.environ["DATAFEED_CONFIG"] = str(paths["datafeed_conf"])

            import uvicorn

            from hf_engine.adapter.fastapi.mt5_feed_server import app

            log_service.log_system(
                f"[Launcher|{account_id}] 🚀 Starte eingebetteten Uvicorn-Server für Datafeed..."
            )

            server_config = uvicorn.Config(
                app=app,
                host=conf.get("datafeed_host", "127.0.0.1"),
                port=int(conf.get("datafeed_port", 8081)),
                log_level="warning",
            )
            server = uvicorn.Server(config=server_config)

            def run_server():
                try:
                    server.run()
                except Exception:
                    _log_exc(f"[Launcher|{account_id}] Uvicorn-Serverfehler")

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            # Healthcheck mit Gesamttimeout
            total_wait = int(
                conf.get("datafeed_health_timeout_sec", DEFAULT_HEALTHCHECK_TIMEOUT_SEC)
            )
            step = int(
                conf.get(
                    "datafeed_health_interval_sec", DEFAULT_HEALTHCHECK_INTERVAL_SEC
                )
            )
            waited = 0
            while waited < total_wait and not stop_event.is_set():
                try:
                    if check_datafeed_health():
                        log_service.log_system(
                            f"[Launcher|{account_id}] ✅ Datafeed-Server erreichbar."
                        )
                        break
                except Exception:
                    _log_exc(
                        f"[Launcher|{account_id}] Fehler beim Datafeed-Healthcheck",
                        level="WARNING",
                    )
                time.sleep(step)
                waited += step
            else:
                log_service.log_system(
                    f"[Launcher|{account_id}] ⚠️ Keine Antwort vom Datafeed-Server innerhalb {total_wait}s – beende.",
                    level="WARNING",
                )
                try:
                    server.should_exit = True
                except Exception:
                    pass
                sys.exit(1)

            # Hauptwarte-Loop: reagiert auf Stop-Datei
            while not stop_event.wait(timeout=5):
                pass

            # Shutdown & Cleanup
            try:
                server.should_exit = True
            except Exception:
                pass

            # Wartet kurz auf Server-Thread-Ende
            server_thread.join(timeout=5)
            _cleanup_files(
                paths["datafeed_conf"], paths["heartbeat_file"], paths["stop_file"]
            )

            log_service.log_system(
                f"[Launcher|{account_id}] 📴 Datafeed wird endgültig gestoppt."
            )
            sys.exit(0)

        # ----------------------
        # Strategiemodus
        # ----------------------
        from hf_engine.adapter.data.mt5_data_provider import MT5DataProvider
        from hf_engine.core.controlling.multi_strategy_controller import (
            MultiStrategyController,
        )
        from hf_engine.core.controlling.strategy_runner import StrategyRunner
        from hf_engine.infra.config.symbol_mapper import SymbolMapper

        broker_map = conf.get("symbol_map", {}) or {}
        data_map = {}

        if isinstance(conf.get("data_provider"), dict):
            data_map = conf["data_provider"].get("symbol_map", broker_map) or {}
            dp_conf = conf["data_provider"]
        else:
            data_map = broker_map
            dp_conf = conf

        symbol_mapper = SymbolMapper(broker_map=broker_map, data_map=data_map)

        # Data Provider: lokal oder remote
        if conf.get("data_provider_remote", False):
            from hf_engine.adapter.data.remote_data_provider import RemoteDataProvider

            data_provider = RemoteDataProvider(
                host=conf.get("data_provider_host", "127.0.0.1"),
                port=int(conf.get("data_provider_port", 8081)),
            )
        else:
            data_provider = MT5DataProvider(
                terminal_path=dp_conf.get("terminal_path"),
                login=dp_conf.get("account_id"),
                password=dp_conf.get("password"),
                server=dp_conf.get("server"),
                data_path=dp_conf.get("data_path"),
                symbol_mapper=symbol_mapper,
            )

        # Broker-Instanz
        broker_module = import_module("hf_engine.adapter.broker.mt5_adapter")
        broker_class = getattr(broker_module, conf.get("broker_class", "MT5Adapter"))
        broker_args = {
            "account_id": conf["account_id"],
            "password": conf["password"],
            "server": conf["server"],
            "terminal_path": conf["terminal_path"],
            "data_path": conf["data_path"],
            "symbol_mapper": symbol_mapper,
        }
        shared_broker = broker_class(**broker_args)

        # Strategien erstellen
        runners = []
        for strategy_conf in conf["strategies"]:
            if not strategy_conf.get("active", True):
                continue

            magic_number = strategy_conf.get("magic_number", 0)
            strategy_module = import_module(f"strategies.{strategy_conf['module']}")
            strategy_class = getattr(strategy_module, strategy_conf["class"])
            strategy_instance = strategy_class(*strategy_conf.get("init_args", []))
            # Safety: Magic-Nummer zwischen Runner (JSON) und Strategy angleichen
            try:
                strategy_instance.magic_number = magic_number
                if hasattr(strategy_instance, "config") and isinstance(
                    strategy_instance.config, dict
                ):
                    strategy_instance.config["magic_number"] = magic_number
            except Exception:
                pass

            runner = StrategyRunner(
                strategy_instance,
                shared_broker,
                data_provider,
                magic_number=magic_number,
                symbol_mapper=symbol_mapper,
            )
            runners.append(runner)

            log_service.log_system(
                f"[System|{conf['account_id']}] ✅ Strategie {strategy_conf.get('class')} geladen mit Magic {magic_number}"
            )

        controller = MultiStrategyController(runners)
        for r in runners:
            r.controller = controller

        # Start Controller
        interval_sec = int(conf.get("controller_tick_sec", DEFAULT_CONTROLLER_TICK_SEC))
        controller.start_all(interval_sec=interval_sec)

        # Hauptwarte-Loop: reagiert auf Stop-Datei
        while not stop_event.wait(timeout=5):
            pass

        # Geordneter Shutdown
        try:
            controller.stop_all()
            _cleanup_files(paths["stop_file"])
            log_service.log_system(
                f"[Launcher|{account_id}] 🧹 Shutdown-Datei entfernt"
            )
        except Exception:
            _log_exc(
                f"[Launcher|{account_id}] Fehler beim Shutdown/Cleanup", level="WARNING"
            )
        finally:
            _cleanup_files(paths["heartbeat_file"])
            sys.exit(0)

    except SystemExit:
        # bewusstes sys.exit() durchreichen
        raise
    except Exception:
        _log_exc(f"[Launcher|{account_id}] ❌ Fehler im Subprozess")
        sys.exit(1)


# ----------------------
# Parent-Prozess
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Engine Launcher für einen Account oder Datenprovider"
    )
    parser.add_argument(
        "--config", required=True, help="Pfad zur JSON-Konfigurationsdatei"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = json.load(f)

    # Basic Signal-Handling im Parent
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    process_name = (
        f"datafeed_{conf['account_id']}"
        if conf.get("data_provider_only", False)
        else f"account_{conf['account_id']}"
    )

    p = multiprocessing.Process(
        target=launch_strategy_process, args=(conf,), name=process_name
    )
    p.start()
    active_processes.append(p)

    # Auf Subprozess warten und Exit-Code auswerten
    for proc in active_processes:
        proc.join()
        if proc.exitcode not in (0, None):
            # Parent hat keinen LogService – wir schreiben klar nach stderr
            sys.stderr.write(
                f"[Launcher] Subprozess '{proc.name}' beendete sich mit Exit-Code {proc.exitcode}\n"
            )
            # Exit-Code des Subprozesses übernehmen
            sys.exit(proc.exitcode)


def _graceful_shutdown():
    try:
        if shared_broker:
            shared_broker.shutdown()
            if log_service:
                log_service.log_system(
                    f"[Launcher] 📴 Broker {shared_broker.__class__.__name__} erfolgreich beendet"
                )
    except Exception:
        if log_service:
            _log_exc("[Launcher] Fehler beim Broker-Shutdown", level="ERROR")


atexit.register(_graceful_shutdown)


if __name__ == "__main__":
    main()
