---
description: 'Error handling standards for Omega project - Single Source of Truth'
applyTo: '**/*.py'
---

# Error Handling Standards

> Error-Handling-Standards für das Omega-Projekt.
> Diese Datei ist die Single Source of Truth – alle anderen Instruktionen referenzieren hierher.

---

## Grundprinzipien

| Prinzip | Beschreibung |
|---------|--------------|
| **Fail Fast** | Exceptions sofort werfen, nicht verschlucken |
| **Be Specific** | Spezifische Exceptions statt generische |
| **Be Informative** | Hilfreiche Fehlermeldungen mit Kontext |
| **Don't Leak** | Keine internen Details nach außen |

---

## Exception Hierarchy

### Standard Exceptions verwenden

```python
# ✅ Spezifische Built-in Exceptions
raise ValueError("balance must be positive, got: -100")
raise TypeError("expected str, got: int")
raise KeyError(f"missing required key: {key}")
raise FileNotFoundError(f"config not found: {path}")

# ❌ Generische Exception
raise Exception("something went wrong")
```

### Custom Exceptions für Domain

```python
# src/exceptions.py
class OmegaError(Exception):
    """Base exception for Omega project."""
    pass

class TradingError(OmegaError):
    """Base exception for trading operations."""
    pass

class InsufficientMarginError(TradingError):
    """Raised when margin is insufficient for operation."""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient margin: required {required}, available {available}"
        )

class ConfigurationError(OmegaError):
    """Raised for configuration issues."""
    pass

class DataValidationError(OmegaError):
    """Raised when data fails validation."""
    pass
```

---

## Error Messages

### Struktur

```python
# ✅ Gut: Was + Warum + Wert
raise ValueError(
    f"risk_percent must be between 0 and 1, got: {risk_percent}"
)

# ✅ Gut: Kontext hinzufügen
raise FileNotFoundError(
    f"Strategy config not found at {config_path}. "
    f"Expected format: configs/live/strategy_config_<account_id>.json"
)

# ❌ Schlecht: Nicht hilfreich
raise ValueError("invalid value")
raise Exception("error")
```

### Checkliste für Messages

- [ ] Was ist der Fehler?
- [ ] Welcher Wert wurde übergeben?
- [ ] Was wäre korrekt gewesen?
- [ ] Wie kann man es beheben?

---

## Try/Except Patterns

### Spezifische Exceptions fangen

```python
# ✅ Spezifisch fangen
try:
    config = load_config(path)
except FileNotFoundError:
    logger.error(f"Config file not found: {path}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in config: {e}")
    raise ConfigurationError(f"Invalid config format: {path}") from e

# ❌ Alles fangen (verschluckt Bugs)
try:
    config = load_config(path)
except:
    pass
```

### Re-raise mit Kontext

```python
# ✅ Exception Chain beibehalten
try:
    result = external_api.fetch(symbol)
except RequestException as e:
    raise DataFetchError(f"Failed to fetch {symbol}") from e

# ✅ Logging vor re-raise
try:
    process_trade(order)
except Exception as e:
    logger.exception(f"Failed to process order {order.id}")
    raise
```

### Cleanup mit finally

```python
# ✅ finally für Cleanup
connection = None
try:
    connection = database.connect()
    connection.execute(query)
except DatabaseError:
    logger.error("Database query failed")
    raise
finally:
    if connection:
        connection.close()
```

---

## Context Managers

### Bevorzuge with-Statement

```python
# ✅ Context Manager
with open(config_path) as f:
    config = json.load(f)

# ✅ Eigener Context Manager
from contextlib import contextmanager

@contextmanager
def managed_mt5_connection():
    """Context manager for MT5 connection."""
    import MetaTrader5 as mt5
    
    if not mt5.initialize():
        raise ConnectionError("Failed to initialize MT5")
    try:
        yield mt5
    finally:
        mt5.shutdown()

# Nutzung
with managed_mt5_connection() as mt5:
    account = mt5.account_info()
```

### Multiple Resources

```python
# ✅ Mehrere Resources
with (
    open(input_path) as infile,
    open(output_path, "w") as outfile,
):
    data = json.load(infile)
    json.dump(process(data), outfile)
```

---

## Logging von Errors

### Levels richtig verwenden

```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: Detailed diagnostic info
logger.debug(f"Processing order: {order}")

# INFO: Normal operations
logger.info(f"Order executed: {order.id}")

# WARNING: Unexpected but handled
logger.warning(f"Retrying failed request, attempt {attempt}")

# ERROR: Operation failed
logger.error(f"Failed to execute order: {order.id}")

# EXCEPTION: Error with traceback
try:
    process(data)
except Exception:
    logger.exception("Unexpected error in process")
    raise
```

### Was NICHT loggen

```python
# ❌ Keine Secrets
logger.info(f"Connecting with password: {password}")

# ❌ Keine PII
logger.info(f"User data: {user_details}")

# ✅ Stattdessen
logger.info(f"Connecting to server: {server_name}")
logger.info(f"Processing user: {user_id}")
```

---

## Validation Pattern

### Input Validation früh

```python
from typing import TypedDict

class OrderParams(TypedDict):
    symbol: str
    lot_size: float
    stop_loss: float
    take_profit: float

def validate_order_params(params: dict) -> OrderParams:
    """Validate and return typed order parameters."""
    # Required fields
    required = {"symbol", "lot_size", "stop_loss", "take_profit"}
    missing = required - params.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Type validation
    if not isinstance(params["symbol"], str):
        raise TypeError(f"symbol must be str, got: {type(params['symbol'])}")
    
    # Range validation
    if params["lot_size"] <= 0:
        raise ValueError(f"lot_size must be positive, got: {params['lot_size']}")
    
    if params["stop_loss"] <= 0:
        raise ValueError(f"stop_loss must be positive, got: {params['stop_loss']}")
    
    return OrderParams(**{k: params[k] for k in required})
```

### Validation Decorator

```python
from functools import wraps
from typing import Callable

def validate_positive(*param_names: str) -> Callable:
    """Decorator to validate positive parameters."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter values
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate
            for name in param_names:
                value = bound.arguments.get(name)
                if value is not None and value <= 0:
                    raise ValueError(f"{name} must be positive, got: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Nutzung
@validate_positive("balance", "risk_percent")
def calculate_position_size(balance: float, risk_percent: float) -> float:
    ...
```

---

## Error Recovery Patterns

### Retry mit Backoff

```python
import time
from typing import TypeVar, Callable

T = TypeVar("T")

def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """Retry function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                time.sleep(delay)
    
    raise last_exception

# Nutzung
result = retry_with_backoff(lambda: api.fetch_data())
```

### Graceful Degradation

```python
def get_market_data(symbol: str) -> dict:
    """Get market data with fallback sources."""
    # Primary source
    try:
        return primary_api.fetch(symbol)
    except APIError as e:
        logger.warning(f"Primary API failed: {e}")
    
    # Fallback source
    try:
        return backup_api.fetch(symbol)
    except APIError as e:
        logger.warning(f"Backup API failed: {e}")
    
    # Last resort: cached data
    cached = cache.get(symbol)
    if cached:
        logger.warning(f"Using cached data for {symbol}")
        return cached
    
    raise DataUnavailableError(f"No data available for {symbol}")
```

---

## Anti-Patterns

```python
# ❌ Silent failure
try:
    risky_operation()
except:
    pass

# ❌ Generic exception
raise Exception("something went wrong")

# ❌ Exception as flow control
try:
    return items[index]
except IndexError:
    return None  # Use len() check instead

# ❌ Catching and ignoring
try:
    data = fetch_data()
except Exception as e:
    print(e)  # Not logging, no re-raise

# ❌ Bare except
try:
    process()
except:  # Catches SystemExit, KeyboardInterrupt too!
    handle()
```

---

## Quick Reference

| Situation | Pattern |
|-----------|---------|
| Validation | Raise ValueError/TypeError früh |
| Missing Resource | Raise FileNotFoundError/KeyError |
| Domain Error | Custom Exception mit Kontext |
| External Call | Try/except + logging + re-raise |
| Resource Cleanup | Context Manager (with) |
| Retry Logic | Exponential Backoff |
| Logging | exception() für Tracebacks |
