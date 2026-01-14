# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Omega is a Python-based trading stack with three main components:
- **Live Engine** (`src/hf_engine/`): MetaTrader 5 adapter with risk management and order execution (Windows-only)
- **Backtest Engine** (`src/backtest_engine/`): Event-driven backtesting, optimization, and walkforward validation
- **UI Engine** (`src/ui_engine/`): FastAPI backend for process control, monitoring, and log streaming

## Common Commands

```bash
# Development install
python -m venv .venv && source .venv/bin/activate && pip install -e .[dev,analysis]

# Full environment (includes ML/torch)
pip install -e .[all]

# Run tests
pytest -q                              # All tests
pytest tests/test_<name>.py -v         # Specific test file
pytest -k "pattern" -v                 # Tests matching pattern
pytest --cov=src --cov-report=term-missing  # With coverage

# Code formatting (pre-commit runs black, isort, flake8, mypy, bandit)
pre-commit run -a

# Run backtest
python src/backtest_engine/runner.py configs/backtest/<name>.json

# Run walkforward backtest
python -m src.strategies.<strategy_name>.backtest.walkforward_backtest

# Start UI (FastAPI)
uvicorn src.ui_engine.main:app --reload --port 8000

# Start live/datafeed process
python src/engine_launcher.py --config configs/live/strategy_config_<account_id>.json
```

## Architecture

### Event-Driven Design
Both live and backtest engines use an event loop architecture. Strategies are designed to be agnostic to the execution environment (live vs backtest) through abstract interfaces.

### Configuration-Driven Execution
All executions are controlled by JSON configs in `configs/`:
- Live strategies: `configs/live/strategy_config_<account_id>.json`
- Backtest configs: `configs/backtest/*.json`
- Execution costs: `configs/execution_costs.yaml`
- Symbol specs: `configs/symbol_specs.yaml`

### Runtime State (`var/` directory)
All runtime state is managed under `var/` (gitignored):
- Heartbeats: `var/tmp/heartbeat_<account_id>.txt`
- Stop signals: `var/tmp/stop_<account_id>.signal`
- Logs: `var/logs/`
- Results: `var/results/`

### Rust/Julia Extensions (Optional)
Performance-critical modules can use Rust (PyO3/Maturin) or Julia (PythonCall.jl). Feature flags control backend selection:
- `OMEGA_USE_RUST_SLIPPAGE_FEE`, `OMEGA_USE_RUST_INDICATOR_CACHE`, `OMEGA_USE_RUST_PORTFOLIO`
- Values: `auto` (default, uses Rust if available), `true` (Rust only), `false` (Python only)

## Critical Constraints

### Platform Requirements
- **Python >= 3.12** with modern type hints (`|` union syntax, `TypedDict`, `Literal`)
- **MetaTrader5 is Windows-only**: Live trading requires Windows; backtests must run on macOS/Linux without MT5

### Trading Safety Invariants
- **Resume semantics**: Position matching via `magic_number` must not break
- **No silent live changes**: New trading logic requires config flag or explicit migration
- **Reproducibility**: Backtests must be deterministic (fixed seeds, no network calls, no lookahead bias)

### Market Data Layout
```
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet
```
Schema: `UTC time`, `Open`, `High`, `Low`, `Close`, `Volume`
Timeframes: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`

## Key Entry Points

- `src/engine_launcher.py` - Central launcher for live/datafeed/backtest modes
- `src/backtest_engine/runner.py` - Single backtest execution
- `src/ui_engine/main.py` - FastAPI application

## Type Checking Tiers

mypy configuration uses tiered strictness:
- **Strict**: `backtest_engine.core`, `backtest_engine.optimizer`, `backtest_engine.rating`, `shared.*`
- **Relaxed**: `hf_engine.*`, `ui_engine.*` (production safety, not FFI-relevant)

## Dependencies

Single source of truth: `pyproject.toml`
- New imports must be added to `dependencies` or appropriate `optional-dependencies` extra
- Optional deps should use defensive imports (`try/except` with fallback)

## Agent Roles

For specialized tasks, refer to `AGENT_ROLES.md` which defines 7 distinct agent roles:
- **Architect**: System-Design und ADRs
- **Implementer**: Code schreiben (Default-Rolle für Claude Code)
- **Reviewer**: Code Review
- **Tester**: Test-Generierung
- **Researcher**: Bibliotheks-Recherche
- **DevOps**: CI/CD, Deployment
- **Safety Auditor**: Sicherheits-Reviews

Default role for Claude Code: **Implementer**
