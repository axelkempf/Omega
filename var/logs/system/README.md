# var/logs/system/ – System Logs

SQLite-based structured logging for the engine.

## Key File

- `engine_logs.db` – SQLite database with structured log entries

## Viewing Logs

```bash
python -m src.hf_engine.infra.logging.log_sqlite_viewer
```

Or use any SQLite browser to query the database directly.
