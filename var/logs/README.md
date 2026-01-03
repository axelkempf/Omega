# var/logs/ – Application Logs

Central logging directory for all Omega components.

## Structure

| Subdirectory | Purpose |
|--------------|---------|
| `system/` | SQLite log database (`engine_logs.db`), system-level logs |
| `entry_logs/` | Trade entry decision logs (debugging signal generation) |
| `trade_logs/` | Execution logs (`trade_log.csv`, `executions.json`) |
| `optuna/` | Optuna optimization trial logs |

## Key Files

- `system/engine_logs.db` – Structured SQLite logs, viewable via `log_sqlite_viewer.py`
- `trade_logs/trade_log.csv` – CSV export of all trade executions
- `trade_logs/executions.json` – Detailed execution tracking

## Log Retention

Consider periodic archival to `var/archive/` for old logs.
