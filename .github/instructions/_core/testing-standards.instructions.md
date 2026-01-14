---
description: 'Testing standards for Omega project - Single Source of Truth'
applyTo: '**/test_*.py,**/tests/**/*.py'
---

# Testing Standards

> Teststandards für das Omega-Projekt.
> Diese Datei ist die Single Source of Truth – alle anderen Instruktionen referenzieren hierher.

---

## Framework & Tools

| Tool | Version | Zweck |
|------|---------|-------|
| pytest | ≥7.4 | Test Framework |
| pytest-cov | - | Coverage Reporting |
| pytest-mock | - | Mocking |
| httpx | ≥0.28 | Async HTTP Testing |

---

## Naming Conventions

### Dateien

```
tests/
├── test_<module>.py          # Unit Tests
├── test_<feature>_integration.py  # Integration Tests
└── conftest.py               # Shared Fixtures
```

### Funktionen

```python
def test_<function>_<scenario>():
    """Test <function> when <scenario>."""
    ...

# Beispiele
def test_calculate_lot_size_with_zero_balance():
def test_process_order_when_market_closed():
def test_validate_config_with_missing_fields():
```

### Klassen (optional für Gruppierung)

```python
class TestTradeManager:
    """Tests for TradeManager class."""
    
    def test_open_position_success(self):
        ...
    
    def test_open_position_insufficient_margin(self):
        ...
```

---

## Test Structure (AAA Pattern)

```python
def test_calculate_lot_size_with_valid_inputs():
    """Calculate lot size returns correct value for valid inputs."""
    # Arrange
    balance = 10000.0
    risk_percent = 0.01
    stop_loss_pips = 20.0
    pip_value = 10.0
    
    # Act
    result = calculate_lot_size(balance, risk_percent, stop_loss_pips, pip_value)
    
    # Assert
    assert result == 0.5
```

---

## Determinismus (Kritisch für Trading)

### Fixierte Seeds

```python
import random
import numpy as np

def test_strategy_signals():
    """Test strategy produces consistent signals."""
    # Fixiere alle Random Sources
    random.seed(42)
    np.random.seed(42)
    
    # Test
    signals = strategy.generate_signals(data)
    assert signals == expected_signals
```

### Keine Netzwerk-Calls

```python
import pytest
from unittest.mock import patch

def test_fetch_market_data(mocker):
    """Test market data fetching without real network."""
    # Mock external API
    mock_response = {"EURUSD": {"bid": 1.1000, "ask": 1.1002}}
    mocker.patch("src.api.client.fetch_data", return_value=mock_response)
    
    # Test
    result = fetch_market_data("EURUSD")
    assert result["bid"] == 1.1000
```

### Keine Zeit-Dependencies

```python
from datetime import datetime
from unittest.mock import patch

def test_is_market_open():
    """Test market hours check with fixed time."""
    # Mock datetime.now()
    fixed_time = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10:00
    
    with patch("src.market.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        
        result = is_market_open()
        assert result is True
```

---

## MT5 Handling

### Skip wenn nicht verfügbar

```python
import pytest

MT5_AVAILABLE = False
try:
    import MetaTrader5
    MT5_AVAILABLE = True
except ImportError:
    pass

@pytest.mark.skipif(not MT5_AVAILABLE, reason="MT5 not available")
def test_mt5_account_info():
    """Test MT5 account info retrieval."""
    ...
```

### Mock MT5 Calls

```python
# conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_mt5(mocker):
    """Mock MT5 module."""
    mock = MagicMock()
    mock.account_info.return_value = MagicMock(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
    )
    mocker.patch.dict("sys.modules", {"MetaTrader5": mock})
    return mock

# test_file.py
def test_get_balance(mock_mt5):
    """Test balance retrieval with mocked MT5."""
    from src.adapter import get_balance
    
    balance = get_balance()
    assert balance == 10000.0
```

---

## Fixtures (conftest.py)

### Beispiel-Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
import pandas as pd

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "UTC time": pd.date_range("2024-01-01", periods=100, freq="1min"),
        "Open": [1.1000] * 100,
        "High": [1.1010] * 100,
        "Low": [1.0990] * 100,
        "Close": [1.1005] * 100,
        "Volume": [100] * 100,
    })

@pytest.fixture
def sample_config() -> dict:
    """Sample strategy configuration."""
    return {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "magic_number": 12345,
        "risk_percent": 0.01,
    }

@pytest.fixture
def tmp_config_file(tmp_path, sample_config) -> Path:
    """Temporary config file for testing."""
    import json
    
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(sample_config))
    return config_path
```

---

## Coverage Requirements

### Zielwerte

| Bereich | Minimum Coverage |
|---------|-----------------|
| Core Execution | ≥80% |
| Risk Management | ≥90% |
| Utilities | ≥70% |
| Neuer Code | ≥80% |

### Commands

```bash
# Coverage Report
pytest --cov=src --cov-report=term-missing

# HTML Report
pytest --cov=src --cov-report=html

# Mit Threshold
pytest --cov=src --cov-fail-under=80
```

---

## Anti-Patterns

### Was NICHT tun

```python
# ❌ Always passes
def test_something():
    assert True

# ❌ Silent failures
def test_with_silent_catch():
    try:
        risky_operation()
    except:
        pass

# ❌ Test depends on execution order
class TestOrdered:
    state = []
    
    def test_first(self):
        self.state.append(1)
    
    def test_second(self):
        assert len(self.state) == 1  # Fails if run alone!

# ❌ Modifies global state
GLOBAL_CONFIG = {}

def test_modifies_global():
    GLOBAL_CONFIG["key"] = "value"  # Leaks to other tests!

# ❌ Sleeps in tests
def test_with_sleep():
    import time
    start_process()
    time.sleep(5)  # Slow and flaky!
    check_result()
```

### Was STATTDESSEN tun

```python
# ✅ Specific assertion
def test_calculate_lot_size():
    result = calculate_lot_size(10000, 0.01, 20, 10)
    assert result == 0.5

# ✅ Test expects exception
def test_invalid_input_raises():
    with pytest.raises(ValueError, match="must be positive"):
        calculate_lot_size(-100, 0.01, 20, 10)

# ✅ Isolated tests with fixtures
@pytest.fixture
def fresh_config():
    return {"key": "value"}

def test_with_isolated_config(fresh_config):
    fresh_config["key"] = "modified"
    assert fresh_config["key"] == "modified"

# ✅ Mock instead of sleep
def test_async_process(mocker):
    mock_wait = mocker.patch("src.process.wait_for_completion")
    mock_wait.return_value = True
    
    result = start_and_wait()
    assert result is True
```

---

## Test Categories

### Unit Tests

```python
# Fokus: Einzelne Funktionen isoliert
def test_calculate_pip_value():
    """Test pip value calculation for EURUSD."""
    result = calculate_pip_value("EURUSD", 1.0)
    assert result == pytest.approx(10.0, rel=1e-2)
```

### Integration Tests

```python
# Fokus: Zusammenspiel mehrerer Komponenten
@pytest.mark.integration
def test_strategy_full_cycle(sample_ohlcv_data, sample_config):
    """Test complete strategy execution cycle."""
    strategy = MeanReversionStrategy(sample_config)
    strategy.on_data(sample_ohlcv_data)
    
    signals = strategy.get_signals()
    assert len(signals) > 0
```

### Regression Tests

```python
# Fokus: Bug-Fixes verifizieren
def test_magic_number_matching_issue_42():
    """Regression test for issue #42: magic number mismatch."""
    # This bug caused position matching to fail
    position = Position(magic_number=12345)
    order = Order(magic_number=12345)
    
    assert position.matches(order)  # Previously returned False!
```

---

## Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("balance,risk,expected", [
    (10000, 0.01, 0.5),
    (5000, 0.02, 0.5),
    (20000, 0.01, 1.0),
])
def test_lot_size_calculation(balance, risk, expected):
    """Test lot size with various inputs."""
    result = calculate_lot_size(balance, risk, 20, 10)
    assert result == expected

@pytest.mark.parametrize("invalid_input,error_msg", [
    (-100, "balance must be positive"),
    (0, "balance must be positive"),
])
def test_lot_size_invalid_inputs(invalid_input, error_msg):
    """Test lot size rejects invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        calculate_lot_size(invalid_input, 0.01, 20, 10)
```

---

## Async Tests

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_fetch_data():
    """Test async data fetching."""
    result = await fetch_data_async("EURUSD")
    assert "bid" in result
    assert "ask" in result

# Fixture für Event Loop
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

---

## Quick Reference

| Aspekt | Standard |
|--------|----------|
| Framework | pytest ≥7.4 |
| Naming | `test_<function>_<scenario>` |
| Pattern | Arrange-Act-Assert (AAA) |
| Seeds | Immer fixieren (42) |
| Network | Immer mocken |
| Time | datetime.now() mocken |
| MT5 | skipif oder mock |
| Coverage Ziel | ≥80% für neuen Code |
