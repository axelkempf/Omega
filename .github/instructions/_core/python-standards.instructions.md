---
description: 'Canonical Python coding standards for Omega project - Single Source of Truth'
applyTo: '**/*.py'
---

# Python Standards

> Kanonische Python-Standards für das Omega-Projekt.
> Diese Datei ist die Single Source of Truth – alle anderen Instruktionen referenzieren hierher.

---

## Version & Compatibility

- **Required:** Python ≥3.12 (spezifiziert in `pyproject.toml`)
- **Type Hints:** Mandatory für alle öffentlichen Funktionen
- **Union Syntax:** Verwende `X | Y` statt `Union[X, Y]`

---

## Style Guide

### Allgemein

| Regel | Standard |
|-------|----------|
| Standard | PEP 8 |
| Zeilenlänge | 88 Zeichen (Black default) |
| Einrückung | 4 Spaces |
| String Quotes | Doppelte Anführungszeichen `"` (Black default) |

### Naming Conventions

| Element | Konvention | Beispiel |
|---------|-----------|----------|
| Variablen | `snake_case` | `user_data`, `trade_count` |
| Funktionen | `snake_case` | `calculate_lot_size()` |
| Klassen | `CamelCase` | `TradeManager`, `StrategyConfig` |
| Konstanten | `UPPER_CASE` | `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT` |
| Private | `_` Präfix | `_internal_state`, `_helper_method()` |

### Anti-Patterns

- ❌ Bedeutungslose Namen: `data`, `temp`, `stuff`, `x`
- ❌ Single-letter außer Loop-Indizes: `i`, `j`, `k` sind OK
- ❌ Funktionen >50 Zeilen
- ❌ Nesting >3 Ebenen
- ❌ Magic Numbers ohne Konstante

---

## Type Hints

### Grundregeln

```python
from __future__ import annotations  # Für Forward References
from typing import TypedDict, Literal, Final

# Moderne Union-Syntax (Python 3.10+)
def process(value: str | None) -> dict[str, int]:
    ...

# TypedDict für strukturierte Dicts
class StrategyConfig(TypedDict):
    symbol: str
    timeframe: str
    magic_number: int

# Literal für feste String-Werte
Direction = Literal["long", "short"]

# Final für Konstanten
MAX_RETRIES: Final = 3
```

### Container Types

```python
# Moderne Syntax (Python 3.9+)
def process_trades(trades: list[dict[str, float]]) -> dict[str, list[float]]:
    ...

# Optional ist None Union
def find_user(user_id: int) -> User | None:
    ...
```

---

## Imports

### Reihenfolge (isort)

1. Standard Library
2. Third-Party
3. Local Imports

```python
# 1. Standard Library
import os
from pathlib import Path
from typing import TypedDict

# 2. Third-Party
import numpy as np
import pandas as pd
from fastapi import FastAPI

# 3. Local Imports
from src.hf_engine.core import TradeManager
from src.strategies.base import BaseStrategy
```

### isort-Konfiguration

```toml
# pyproject.toml
[tool.isort]
profile = "black"
```

---

## Docstrings

### Google Style (Standard)

```python
def calculate_lot_size(
    balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value: float,
) -> float:
    """Calculate position size based on risk parameters.

    Args:
        balance: Account balance in base currency.
        risk_percent: Risk per trade as decimal (0.01 = 1%).
        stop_loss_pips: Distance to stop loss in pips.
        pip_value: Value of one pip in base currency.

    Returns:
        Calculated lot size rounded to 2 decimal places.

    Raises:
        ValueError: If any parameter is negative or zero.

    Example:
        >>> calculate_lot_size(10000, 0.01, 20, 10)
        0.5
    """
    if any(v <= 0 for v in [balance, risk_percent, stop_loss_pips, pip_value]):
        raise ValueError("All parameters must be positive")
    
    risk_amount = balance * risk_percent
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    return round(lot_size, 2)
```

### Wann Docstrings?

| Element | Pflicht? |
|---------|----------|
| Public Functions | ✅ Ja |
| Public Classes | ✅ Ja |
| Public Methods | ✅ Ja |
| Module-Level | ✅ Ja (kurz) |
| Private Helpers | ❌ Optional |
| Offensichtliche One-Liner | ❌ Nein |

---

## Common Patterns

### Defensive Optional Imports

```python
# MT5 ist Windows-only – defensiv importieren
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # Type: ignore

# Nutzung
def get_account_info() -> dict | None:
    if not MT5_AVAILABLE:
        return None
    return mt5.account_info()._asdict()
```

### Config Loading

```python
from pathlib import Path
from typing import TypedDict
import json

class StrategyConfig(TypedDict):
    symbol: str
    timeframe: str
    magic_number: int

def load_config(path: Path) -> StrategyConfig:
    """Load and validate strategy configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path) as f:
        config = json.load(f)
    
    # Validation
    required = {"symbol", "timeframe", "magic_number"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    
    return config
```

### Context Manager Pattern

```python
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def managed_connection(host: str) -> Iterator[Connection]:
    """Context manager for database connections."""
    conn = Connection(host)
    try:
        conn.connect()
        yield conn
    finally:
        conn.disconnect()

# Nutzung
with managed_connection("localhost") as conn:
    conn.execute("SELECT * FROM trades")
```

---

## Performance Guidelines

### Bevorzuge Built-ins

```python
from collections import Counter, defaultdict
from itertools import chain

# Zählen
word_counts = Counter(words)

# Gruppieren
grouped = defaultdict(list)
for item in items:
    grouped[item.category].append(item)

# Chains
all_items = list(chain(list1, list2, list3))
```

### List Comprehensions über Loops

```python
# ✅ Gut
squares = [x**2 for x in range(10)]
filtered = [x for x in items if x.is_valid]

# ❌ Vermeiden
squares = []
for x in range(10):
    squares.append(x**2)
```

### Generator für große Daten

```python
# ✅ Memory-effizient
def process_large_file(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)

# Nutzung
for record in process_large_file(data_path):
    process(record)
```

---

## Tools & Enforcement

### Formatierung

```bash
# Black (Formatter)
black src/ tests/

# isort (Import Sorting)
isort src/ tests/

# Kombiniert via pre-commit
pre-commit run -a
```

### Linting

```bash
# flake8 (Style)
flake8 src/ tests/

# mypy (Type Checking)
mypy src/
```

### IDE-Konfiguration

```json
// .vscode/settings.json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

---

## Quick Reference

| Regel | Standard |
|-------|----------|
| Python Version | ≥3.12 |
| Formatter | Black (88 chars) |
| Import Sorter | isort (profile: black) |
| Linter | flake8 |
| Type Checker | mypy |
| Docstring Style | Google |
| Test Framework | pytest |
