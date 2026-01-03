# var/logs/trade_logs/ – Execution Logs

Records of all trade executions and their outcomes.

## Key Files

| File | Format | Purpose |
|------|--------|---------|
| `trade_log.csv` | CSV | Flat export of all trades |
| `executions.json` | JSON | Detailed execution tracking with metadata |

## Schema (trade_log.csv)

Typical columns: `timestamp`, `symbol`, `direction`, `entry_price`, `exit_price`, `pnl`, `magic_number`, etc.
