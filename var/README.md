# var/ – Runtime State Directory

This directory holds all **runtime state** for the Omega Trading Stack.
Contents are git-ignored; only the directory structure (README files) is tracked.

## Structure

| Subdirectory | Purpose |
|--------------|---------|
| `tmp/` | Heartbeats (`heartbeat_<account_id>.txt`), stop signals (`stop_<account_id>.signal`) |
| `logs/` | Application logs (system, trade, entry, optuna) |
| `results/` | Backtest results, walkforward outputs, analysis artifacts |
| `archive/` | Archived data (old results, rotated logs) |

## Operational Guardrails

- **Heartbeats**: `var/tmp/heartbeat_<account_id>.txt` – updated every ~30s by running engines
- **Stop Signals**: `var/tmp/stop_<account_id>.signal` – triggers graceful shutdown
- **Logs**: `var/logs/system/engine_logs.db` – SQLite log database

## Auto-Creation

Directories are created automatically by `src/hf_engine/infra/config/paths.py` on import.
This tracked skeleton ensures the structure exists even without running the code.

## CI Validation

The directory structure is validated in CI via `tests/test_directory_structure.py`.
