import io
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from hf_engine.infra.config.paths import WALKFORWARD_RESULTS_DIR

_HEAD_MAX = int(os.getenv("WF_MASTER_INDEX_HEAD_MAX", "200"))
_BACKEND = os.getenv("WF_MASTER_INDEX_BACKEND", "jsonl").lower()  # "jsonl" | "json"


@contextmanager
def _file_lock_advisory(lock_path: Path) -> Generator[None, None, None]:
    """
    Sehr einfache, systemneutrale Advisory-Lock-Strategie:
    - erstellt eine .lock-Datei exklusiv; bei Kollision kurz warten/retry.
    - vermeidet schwere Abhängigkeiten (fcntl/portalocker).
    Reentrant/Best-Effort (nur kurze, kleine Writes).
    """
    import time
    import uuid

    token = str(uuid.uuid4())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            # exklusiv anlegen
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(token)
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            # nur löschen, wenn noch "unser" Token drin steht
            content = ""
            try:
                content = Path(lock_path).read_text()
            except Exception:
                pass
            if token == content:
                os.unlink(str(lock_path))
        except Exception:
            pass


def update_master_index(strategy_name: str, run_path: Path, summary_path: Path) -> None:
    """
    Schnelles, skalierendes Master-Index-Update.
    - Default: JSONL-Append (master_index.jsonl), O(1)
    - Optional: JSON (alter Modus) via WF_MASTER_INDEX_BACKEND=json
    - Schreibt zusätzlich eine kleine Kopfdatei (master_index_head.json) mit den letzten N Einträgen.
    """
    run_path = Path(run_path)
    results_dir = WALKFORWARD_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "master_index.json"  # Alt
    jsonl_path = results_dir / "master_index.jsonl"  # Neu
    head_path = results_dir / "master_index_head.json"
    lock_path = results_dir / ".master_index.lock"

    entry = {
        "strategy_name": strategy_name,
        "run_time": datetime.utcnow().isoformat(),
        "run_path": str(run_path),
        "ratings_summary": str(summary_path),
    }

    # 1) Einmalige Migration von JSON -> JSONL (falls nötig)
    if _BACKEND == "jsonl":
        try:
            if (not jsonl_path.exists()) and json_path.exists():
                with _file_lock_advisory(lock_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        with open(jsonl_path, "a", encoding="utf-8") as out:
                            for row in data or []:
                                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        # Backup statt löschen
                        json_path.rename(json_path.with_suffix(".json.bak"))
                        print(
                            f"ℹ️ master_index.json → master_index.jsonl migriert (Backup angelegt)."
                        )
                    except Exception as me:
                        print(
                            f"⚠️ Migration master_index.json → .jsonl fehlgeschlagen: {me}"
                        )
        except Exception:
            pass

    # 2) Schreiben je nach Backend
    if _BACKEND == "json":
        # Alter Modus (vollständig) – für Kompatibilität verfügbar
        try:
            with _file_lock_advisory(lock_path):
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        master_data = json.load(f)
                else:
                    master_data = []
                master_data.append(entry)
                # atomic write
                tmp = json_path.with_suffix(".tmp.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(master_data, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp, json_path)
        except Exception as e:
            print(f"⚠️ master_index(JSON) Update fehlgeschlagen: {e}")
            return
        print(f"✅ master_index (JSON) aktualisiert: {json_path}")
        return

    # Neuer Modus: JSONL (Append-only, O(1))
    try:
        # 2a) Append Entry als eine Zeile
        with _file_lock_advisory(lock_path):
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ master_index(JSONL) Append fehlgeschlagen: {e}")
        return

    # 2b) Kleiner Head (letzte N Einträge) für schnelle Übersicht
    try:
        # Effizient: von hinten lesen ohne volle Datei im Speicher zu laden
        last: List[Dict[str, Any]] = []
        # Tail-Read: wir lesen in Blöcken rückwärts
        block_size = 1 << 14  # 16 KiB
        needed = _HEAD_MAX
        with open(jsonl_path, "rb") as f:
            f.seek(0, io.SEEK_END)
            pos = f.tell()
            buf = b""
            while pos > 0 and len(last) < needed:
                step = min(block_size, pos)
                pos -= step
                f.seek(pos)
                chunk = f.read(step)
                buf = chunk + buf
                lines = buf.split(b"\n")
                # halte erste (evtl. angebrochene) Zeile im Buffer, verarbeite Rest
                buf = lines[0]
                for line in lines[-1:0:-1]:
                    if not line.strip():
                        continue
                    try:
                        last.append(json.loads(line.decode("utf-8")))
                        if len(last) >= needed:
                            break
                    except Exception:
                        continue
        last = list(reversed(last))
        tmp = head_path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(last, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, head_path)
    except Exception as e:
        # Head ist rein optional; keine harte Fehlersituation
        print(f"⚠️ master_index_head konnte nicht aktualisiert werden: {e}")

    print(f"✅ master_index (JSONL) appended: {jsonl_path} | head: {head_path}")


def estimate_n_trials(
    param_grid: Dict[str, Dict[str, Any]],
    float_trials: int = 10,
    base_min_per_param: int = 10,
    base_opt_per_param: int = 30,
) -> Tuple[int, int, List[Tuple[str, str, int]]]:
    """
    Empfiehlt minimale und optimale n_trials-Anzahl für Optuna,
    basierend auf Parametertyp, Wertebereich und Schrittweite.
    """
    param_details: List[Tuple[str, str, int]] = []
    total_effective = 1
    param_count = 0

    for k, v in param_grid.items():
        if v["type"] == "int":
            step = v.get("step", 1)
            n = (v["high"] - v["low"]) // step + 1
        elif v["type"] == "float":
            n = float_trials
        elif v["type"] == "categorical":
            n = len(v["choices"])
        else:
            n = 10  # Fallback
        param_details.append((k, v["type"], n))
        total_effective *= max(1, n)
        param_count += 1

    min_trials = max(base_min_per_param * param_count, 50)
    opt_trials = max(base_opt_per_param * param_count, 100)

    print("Parameter-Details:")
    for name, typ, eff in param_details:
        print(f"- {name:22s} {typ:10s}: ca. {eff} Werte")
    print(f"\nEffektiver Gesamt-Raum: ~{int(total_effective):,} Kombinationen (grob)")
    print(
        f"Empfohlene n_trials: min {min_trials}, optimal {opt_trials} (je nach Hardware und Ziel)"
    )

    return min_trials, opt_trials, param_details


def recommend_n_jobs(default_fallback: int = 3) -> int:
    """Wählt n_jobs dynamisch basierend auf Hardware."""
    try:
        n = os.cpu_count() or default_fallback
        return max(1, n - 1)
    except Exception:
        return default_fallback


# ==== Beispiel-CLI ====
if __name__ == "__main__":
    param_grid = {
        "min_dist_pips": {"type": "float", "low": 3, "high": 6},
        "max_dist_pips": {"type": "float", "low": 10, "high": 40},
        "sl_mult": {"type": "float", "low": 1.0, "high": 4.0},
        "pip_threshold": {"type": "float", "low": 0.00005, "high": 0.0006},
        "start": {"type": "int", "low": 15, "high": 150, "step": 15},
        "max_holding_minutes": {"type": "int", "low": 30, "high": 240, "step": 15},
    }
    estimate_n_trials(param_grid)
    print("Empfohlene n_jobs:", recommend_n_jobs())
