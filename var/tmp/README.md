# var/tmp/ – Temporary Runtime Files

Holds transient runtime state for process coordination.

## Contents

| File Pattern | Purpose |
|--------------|---------|
| `heartbeat_<account_id>.txt` | Updated every ~30s by running engine; checked by UI watchdog |
| `stop_<account_id>.signal` | Created to trigger graceful shutdown of an engine |

## Usage

```python
from hf_engine.infra.config.paths import TMP_DIR

heartbeat_file = TMP_DIR / f"heartbeat_{account_id}.txt"
stop_signal = TMP_DIR / f"stop_{account_id}.signal"
```

## Notes

- Files here are ephemeral and should be cleaned up on engine exit
- UI Engine monitors heartbeats to detect stalled processes
