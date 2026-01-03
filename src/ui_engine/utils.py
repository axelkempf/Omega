# engine_ui/utils.py
import datetime
import subprocess


def format_timestamp(ts: datetime.datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def is_process_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def read_log_tail(log_path: str, lines: int = 50) -> str:
    """
    Gibt die letzten X Zeilen einer Logdatei zurück.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])
    except Exception as e:
        return f"⚠️ Fehler beim Lesen von {log_path}: {str(e)}"
