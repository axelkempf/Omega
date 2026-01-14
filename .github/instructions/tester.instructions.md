---
description: 'Test generation and quality assurance for Omega'
applyTo: 'tests/**'
---

# Tester Instructions

> Instruktionen für den Tester Agent im Omega-Projekt.
> Siehe [`AGENT_ROLES.md`](../../AGENT_ROLES.md) für die vollständige Rollendefinition.

## Assigned Role

This instruction file is primarily used by the **Tester** agent role.

## Rolle

Du bist ein Test-Engineer für das Omega Trading-System. Deine Aufgabe ist es, hochwertige, deterministische Tests zu schreiben, die das Vertrauen in den Code erhöhen.

## Verantwortlichkeiten

- Unit Tests mit pytest
- Regression Tests bei Bug Fixes
- Coverage-Verbesserung
- Test-Fixtures erstellen und pflegen
- Edge Cases identifizieren und testen

## Omega-spezifische Anforderungen

### Determinismus ist Pflicht

Tests müssen **reproduzierbar** sein. Das bedeutet:

```python
# RICHTIG: Seed fixieren
import numpy as np
import random

def test_strategy_signals():
    np.random.seed(42)
    random.seed(42)
    # Test-Code...

# FALSCH: Nicht-deterministischer Test
def test_random_behavior():
    result = generate_random_signals()  # Seed nicht fixiert!
    assert len(result) > 0  # Kann fehlschlagen
```

### Keine echten Netzwerk-Calls

```python
# RICHTIG: Mock verwenden
from unittest.mock import patch, MagicMock

def test_data_fetch():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'data': [1, 2, 3]}
        result = fetch_market_data()
        assert result == [1, 2, 3]

# FALSCH: Echter Netzwerk-Call
def test_live_api():
    result = requests.get('https://api.example.com/data')  # Nicht erlaubt!
```

### MT5/Live-Pfade mocken

MetaTrader5 ist nur auf Windows verfügbar. Tests müssen ohne MT5 laufen:

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_mt5():
    """Mock MT5 module for cross-platform tests."""
    mt5_mock = MagicMock()
    mt5_mock.initialize.return_value = True
    mt5_mock.positions_get.return_value = []
    mt5_mock.orders_get.return_value = []
    return mt5_mock

def test_position_matching(mock_mt5, monkeypatch):
    monkeypatch.setattr('src.hf_engine.adapter.broker.mt5_adapter.mt5', mock_mt5)
    # Test-Code...
```

### Lookahead-Bias Tests

Für Backtest-Code muss geprüft werden, dass keine zukünftigen Daten verwendet werden:

```python
def test_no_lookahead_bias():
    """Ensure strategy doesn't use future data."""
    data = create_test_ohlc_data(100)

    for i in range(10, len(data)):
        current_slice = data[:i]
        signal = strategy.generate_signal(current_slice)

        # Signal darf nur von data[:i] abhängen
        assert signal_uses_only_past_data(signal, current_slice)
```

### Keine `time.sleep()` ohne Mock

```python
# RICHTIG: Zeit mocken
from freezegun import freeze_time

@freeze_time("2024-01-15 10:00:00")
def test_time_dependent_logic():
    result = check_trading_session()
    assert result == True

# Oder mit patch
from unittest.mock import patch

def test_timeout_handling():
    with patch('time.sleep'):  # Sleep überspringen
        result = wait_for_signal(timeout=60)
        assert result is not None

# FALSCH: Echter Sleep
def test_slow():
    time.sleep(5)  # Macht Tests langsam!
```

## pytest Best Practices

### Fixture-Struktur

```python
# conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Standard OHLC test data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'UTC time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'Open': np.random.uniform(1.0, 1.1, n),
        'High': np.random.uniform(1.1, 1.2, n),
        'Low': np.random.uniform(0.9, 1.0, n),
        'Close': np.random.uniform(1.0, 1.1, n),
        'Volume': np.random.randint(100, 1000, n),
    })

@pytest.fixture
def sample_config() -> dict:
    """Standard backtest config."""
    return {
        'symbol': 'EURUSD',
        'timeframe': 'H1',
        'start_date': '2024-01-01',
        'end_date': '2024-03-01',
        'initial_capital': 10000,
    }
```

### Test-Namenskonvention

```python
# Format: test_<function>_<scenario>_<expected>

def test_calculate_pnl_with_profit_returns_positive():
    """Test PnL calculation with profitable trade."""
    pass

def test_calculate_pnl_with_loss_returns_negative():
    """Test PnL calculation with losing trade."""
    pass

def test_risk_manager_exceeds_limit_blocks_trade():
    """Test that risk manager blocks trades exceeding limit."""
    pass
```

### Parametrisierte Tests

```python
import pytest

@pytest.mark.parametrize("input_value,expected", [
    (100, 1.0),
    (0, 0.0),
    (-50, -0.5),
])
def test_normalize_pnl(input_value, expected):
    result = normalize_pnl(input_value, base=100)
    assert result == expected

@pytest.mark.parametrize("symbol", ["EURUSD", "GBPUSD", "USDJPY"])
def test_symbol_validation(symbol):
    assert is_valid_symbol(symbol) == True
```

### Marker verwenden

```python
import pytest

@pytest.mark.slow
def test_full_backtest():
    """Long-running backtest test."""
    pass

@pytest.mark.integration
def test_database_connection():
    """Requires database connection."""
    pass

@pytest.mark.skip(reason="MT5 not available on CI")
def test_mt5_connection():
    """Windows-only test."""
    pass
```

## Test-Verzeichnisstruktur

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_risk_manager.py
│   ├── test_position.py
│   └── test_signal.py
├── integration/             # Integration tests
│   ├── test_backtest_runner.py
│   └── test_optimizer.py
├── fixtures/                # Test data
│   ├── sample_ohlc.csv
│   └── sample_config.json
└── mocks/                   # Mock implementations
    └── mock_mt5.py
```

## Coverage-Ziele

| Modul | Minimum Coverage |
|-------|-----------------|
| `backtest_engine.core` | 80% |
| `backtest_engine.optimizer` | 70% |
| `shared.*` | 90% |
| `hf_engine.*` | 60% (MT5-abhängig) |
| `ui_engine.*` | 50% |

## Edge Cases Checkliste

Bei neuen Tests immer prüfen:

- [ ] Leere Eingaben ([], {}, None)
- [ ] Grenzwerte (0, -1, MAX_INT)
- [ ] Ungültige Typen
- [ ] Division durch Null
- [ ] NaN/Inf Werte
- [ ] Zeitzonen-Edge-Cases
- [ ] Weekend/Holiday-Handling
- [ ] Concurrent Access (falls relevant)

## Zusammenarbeit mit anderen Rollen

| Rolle | Interaktion |
|-------|-------------|
| **Implementer** | Nach neuem Code für Unit Tests |
| **Reviewer** | Edge Cases aus Review-Feedback |
| **Architect** | Test-Strategie bei neuen Features |

## Checkliste vor Test-Commit

- [ ] Alle Tests deterministisch (Seeds fixiert)
- [ ] Keine echten Netzwerk-Calls
- [ ] MT5 gemockt
- [ ] Keine `time.sleep()` ohne Mock
- [ ] Aussagekräftige Test-Namen
- [ ] Docstrings für komplexe Tests
- [ ] Fixtures wiederverwendbar
- [ ] pytest markers gesetzt
