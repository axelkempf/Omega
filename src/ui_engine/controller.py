# ui_engine/controller.py
import os
import signal
import subprocess
import threading
import time
from typing import Any, Dict, TypedDict

import psutil
import requests

from hf_engine.infra.config.paths import TMP_DIR
from ui_engine.registry.strategy_alias import resolve_alias

# Alle gestarteten Prozesse nach Name
running_processes: Dict[str, subprocess.Popen[Any]] = {}


_CREATE_NEW_CONSOLE: int = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


class ResourceUsageData(TypedDict, total=False):
    status: str
    cpu_percent: float
    memory_mb: float
    threads: int
    start_time: str


def stream_process_output(process: subprocess.Popen[Any], name: str) -> None:
    def _stream() -> None:
        stdout = process.stdout
        if stdout is None:
            return

        for line in iter(stdout.readline, b""):
            print(f"[{name}] {line.strip()}")
        stdout.close()

    threading.Thread(target=_stream, daemon=True).start()


def validate_config(config_path: str) -> tuple[bool, str]:
    import json

    try:
        with open(config_path, "r") as f:
            conf = json.load(f)

        required_fields = ["account_id", "password", "server", "terminal_path"]
        for field in required_fields:
            if field not in conf:
                return False, f"Missing required field: {field}"

        if not conf.get("strategies"):
            return False, "Keine Strategien definiert."

        if not isinstance(conf["strategies"], list):
            return False, "strategies muss eine Liste sein."

        for s in conf["strategies"]:
            if "module" not in s or "class" not in s:
                return False, "Jede Strategie muss `module` und `class` enthalten."

    except Exception as e:
        return False, f"⚠️ JSON-Konfig fehlerhaft: {e}"

    return True, "OK"


def start_strategy(name: str, config_path: str) -> bool:
    is_valid, reason = validate_config(config_path)
    if not is_valid:
        print(f"❌ Strategie-Konfiguration ungültig: {reason}")
        return False

    real_name = resolve_alias(name)

    if real_name in running_processes and running_processes[real_name].poll() is None:
        print(f"❗ Strategie {real_name} läuft bereits.")
        return False

    process = subprocess.Popen(
        ["python", "engine_launcher.py", "--config", config_path],
        cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
        creationflags=_CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        close_fds=True,
    )

    time.sleep(1)
    if process.poll() is not None:
        print(
            f"❌ Prozess {real_name} ist sofort beendet (Exit Code: {process.returncode})"
        )
        return False

    running_processes[real_name] = process
    print(f"✅ Strategie {real_name} gestartet mit PID {process.pid}")
    return True


def stop_strategy(name: str) -> bool:
    real_name = resolve_alias(name)
    if real_name not in running_processes:
        print(f"❗ Strategie {real_name} ist nicht gestartet.")
        return False

    process = running_processes[real_name]
    if process.poll() is None:
        shutdown_path = TMP_DIR / f"stop_{real_name}.signal"
        shutdown_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shutdown_path, "w") as f:
            f.write("stop")

        print(
            f"📡 Shutdown-Signal an {real_name} gesendet – warte auf Graceful Shutdown..."
        )

        try:
            process.wait(timeout=15)
            print(f"✅ {real_name} sauber beendet.")
            return True
        except subprocess.TimeoutExpired:
            print(f"⚠️ {real_name} hat nicht reagiert – .kill() wird erzwungen.")
            process.kill()
            return False

    # Prozess ist bereits beendet.
    running_processes.pop(real_name, None)
    return True


def get_status(name: str) -> str:
    real_name = resolve_alias(name)
    if real_name not in running_processes:
        return "Not Started"

    process = running_processes[real_name]
    if process.poll() is not None:
        return "Stopped"

    heartbeat_file = TMP_DIR / f"heartbeat_{real_name}.txt"
    if heartbeat_file.exists():
        try:
            with open(heartbeat_file, "r") as f:
                last = float(f.read().strip())
                delay = time.time() - last
                if delay > 30:
                    return f"Unresponsive ({int(delay)}s)"
        except (ValueError, IOError, OSError) as e:
            print(f"⚠️ Heartbeat-Fehler für {real_name}: {e}")
            return "Heartbeat Error"
    else:
        return "No Heartbeat"

    return "Running"


def start_datafeed_server() -> bool:
    name = "datafeed"
    if name in running_processes and running_processes[name].poll() is None:
        print("✅ Datafeed-Server läuft bereits.")
        return True

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "config",
        "strategy_config_15582434.json",
    )

    process = subprocess.Popen(
        ["python", "engine_launcher.py", "--config", config_path],
        cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
        creationflags=_CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        close_fds=True,
    )

    time.sleep(1)
    if process.poll() is not None:
        print(f"❌ Datafeed-Prozess beendete sich sofort (Exit {process.returncode})")
        return False

    running_processes[name] = process
    print(f"🚀 Datafeed gestartet mit PID {process.pid}")
    return True


def check_datafeed_health() -> bool:
    try:
        resp = requests.get("http://127.0.0.1:8081/health", timeout=2)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def restart_unresponsive_strategies(interval: int = 30) -> None:
    print("🛡️ Starte Watchdog für automatische Strategieüberwachung...")
    while True:
        for heartbeat_path in TMP_DIR.glob("heartbeat_*.txt"):
            try:
                name = heartbeat_path.name.replace("heartbeat_", "").replace(".txt", "")
                real_name = resolve_alias(name)
                if real_name not in running_processes:
                    continue

                with open(heartbeat_path, "r") as f:
                    last = float(f.read().strip())
                    delay = time.time() - last

                    if delay > 60:
                        print(
                            f"⚠️ {real_name} ist unresponsive ({int(delay)}s) – versuche Neustart..."
                        )

                        stop_strategy(real_name)
                        time.sleep(2)
                        config_path = os.path.join(
                            "config", f"strategy_config_{real_name}.json"
                        )
                        start_strategy(real_name, config_path)

            except Exception as e:
                print(f"❌ Watchdog-Fehler bei {heartbeat_path}: {e}")

        time.sleep(interval)


def get_resource_usage(name: str) -> ResourceUsageData:
    real_name = resolve_alias(name)
    proc = running_processes.get(real_name)
    if not proc:
        return {"status": "Not Started"}

    if proc.poll() is not None:
        return {"status": "Stopped"}

    try:
        p = psutil.Process(proc.pid)
        p.cpu_percent(interval=None)
        time.sleep(0.1)
        cpu = p.cpu_percent(interval=None)
        return {
            "status": "Running",
            "cpu_percent": cpu,
            "memory_mb": round(p.memory_info().rss / 1024 / 1024, 1),
            "threads": p.num_threads(),
            "start_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(p.create_time())
            ),
        }
    except Exception as e:
        return {"status": f"Fehler: {e}"}
