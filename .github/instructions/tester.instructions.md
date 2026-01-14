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

---

## ⚠️ V1 vs V2 Test-Kontexte

Dieses Projekt hat **zwei parallele Test-Kontexte**:

| Aspekt | V1 (Live-Engine, Analysis) | V2 (Backtest-Core) |
|--------|---------------------------|-------------------|
| **Sprache** | Python only | Python + Rust |
| **Test-Framework** | pytest | pytest + `cargo test` + `proptest` |
| **Pfade** | `tests/` | `python/bt/tests/` + `rust_core/crates/*/tests/` |
| **Spezial-Tests** | MT5-Mocking, Lookahead-Bias | **Golden Files**, V1↔V2 Parität |
| **Determinismus** | Seeds fixieren | Seeds + `rng_seed` Config |
| **Instructions** | Dieses Dokument | Dieses Dokument + `omega-v2-backtest.instructions.md` |

**Für V2-Backtest-Tests zusätzlich lesen:** [omega-v2-backtest.instructions.md](omega-v2-backtest.instructions.md)

---

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

---

## V2 Backtest-Core: Golden File Testing (NEU)

### Was sind Golden Files?

Golden Files sind **erwartete Referenz-Outputs** für deterministische Backtests. Sie dienen als Regressionsschutz: Wenn sich ein Artefakt unerwartet ändert, schlägt der Test fehl.

### Golden-Artefakte (MVP)

| Datei | Format | Beschreibung |
|-------|--------|--------------|
| `trades.json` | JSON Array | Alle Trades mit Entry/Exit/Reason |
| `equity.csv` | CSV | Equity-Kurve pro Bar |
| `metrics.json` | JSON Object | Sharpe, Sortino, Drawdown, etc. |
| `meta.json` | JSON Object | Run-Metadaten, Timestamps |

### Golden File Workflow

```bash
# 1. Golden-Smoke (PR-Gate, schnell)
pytest python/bt/tests/test_golden.py -k "smoke"

# 2. Full Golden (Nightly/Release)
pytest python/bt/tests/test_golden.py

# 3. Golden-Update (NUR mit Review!)
pytest python/bt/tests/test_golden.py --update-golden
```

### Golden Test Struktur

```python
# python/bt/tests/test_golden.py
import json
import pytest
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURES_DIR = GOLDEN_DIR / "fixtures"
EXPECTED_DIR = GOLDEN_DIR / "expected"


@pytest.mark.smoke
def test_golden_mean_reversion_basic():
    """Golden test for basic mean reversion scenario."""
    # 1. Load config
    config_path = FIXTURES_DIR / "configs/mean_reversion_basic.json"
    config = json.loads(config_path.read_text())
    
    # 2. Run backtest
    from bt import run
    result = run(config)
    
    # 3. Load expected outputs
    expected_dir = EXPECTED_DIR / "mean_reversion_basic"
    expected_trades = json.loads((expected_dir / "trades.json").read_text())
    expected_metrics = json.loads((expected_dir / "metrics.json").read_text())
    
    # 4. Compare (nach Normalisierung)
    assert normalize_trades(result.trades) == normalize_trades(expected_trades)
    assert normalize_metrics(result.metrics) == normalize_metrics(expected_metrics)


def normalize_trades(trades: list) -> list:
    """Normalize trades for comparison."""
    # Sortiere nach exit_time_ns für stabile Reihenfolge
    return sorted(trades, key=lambda t: t["exit_time_ns"])


def normalize_metrics(metrics: dict) -> dict:
    """Normalize metrics for comparison."""
    result = metrics.copy()
    # Entferne nicht-deterministische Felder
    result.pop("generated_at", None)
    result.pop("generated_at_ns", None)
    return result
```

### Vergleichsregeln

1. **Normalisierung vor Vergleich:**
   - `meta.json`: `generated_at` und `generated_at_ns` werden ignoriert
   - JSON: stabile Key-Order (kanonische Serialisierung)

2. **Float-Vergleich:**
   - Nach Contract-Rundung (2/6 Dezimalstellen) wird **exakt** verglichen
   - Keine Toleranzen nach Rundung

3. **Trades-Vergleich:**
   - Sortierung nach `exit_time_ns`
   - Alle Pflichtfelder müssen übereinstimmen

### Golden Update Policy

**WICHTIG:** Golden-Updates sind Breaking Changes!

```python
# NIEMALS automatisch updaten!
# Golden-Updates nur wenn:
# 1. Bewusste Änderung der Execution-Logik
# 2. Bug-Fix der falsche Golden-Werte korrigiert
# 3. Review und Begründung im PR

# Update-Prozess:
# 1. PR mit --update-golden lokal ausführen
# 2. Diff der Golden-Files reviewen
# 3. Begründung im PR dokumentieren
# 4. Mindestens 1 Reviewer muss Golden-Diff prüfen
```

### V1↔V2 Paritäts-Tests

```python
@pytest.mark.parity
def test_v1_v2_parity_scenario_1():
    """Market-Entry Long → Take-Profit.
    
    V1 und V2 müssen identische Events produzieren.
    """
    config = load_parity_config("scenario_1")
    
    # V1 Referenzlauf
    v1_result = run_v1_backtest(config)
    
    # V2 Lauf im Parity-Mode
    config["execution_variant"] = "v1_parity"
    v2_result = run_v2_backtest(config)
    
    # Events MÜSSEN übereinstimmen
    assert v1_result.trade_count == v2_result.trade_count
    assert v1_result.trades == v2_result.trades
    
    # PnL/Fees innerhalb Toleranz (nach 2dp Rundung)
    assert abs(v1_result.profit_net - v2_result.profit_net) < 0.01
```

### 6 Kanonische Szenarien (MUSS)

1. **Market-Entry Long → Take-Profit**
2. **Market-Entry Long → Stop-Loss**
3. **Pending Entry (Limit/Stop) → Trigger ab `next_bar` → Exit**
4. **Same-Bar SL/TP Tie → SL-Priorität**
5. **`in_entry_candle` Spezialfall inkl. Limit-TP Regel**
6. **Mix aus Sessions/Warmup/HTF-Einflüssen**

---

## V2 Rust Tests: `cargo test` + `proptest`

### Unit Tests in Rust

```rust
// rust_core/crates/execution/src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_long_at_ask() {
        let fill = execute_market_order(
            Direction::Long,
            bid: 1.08000,
            ask: 1.08005,
        );
        assert_eq!(fill.price, 1.08005);
    }

    #[test]
    fn test_sl_priority_on_same_bar() {
        // SL und TP beide getriggert → SL gewinnt
        let result = check_exits(
            position: &long_position,
            bar: &bar_with_both_triggers,
        );
        assert_eq!(result.reason, ExitReason::StopLoss);
    }
}
```

### Property Tests mit `proptest`

```rust
// rust_core/crates/execution/tests/prop_tests.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_exit_time_never_before_entry(
        entry_ns in 0i64..i64::MAX,
        hold_bars in 1u32..1000,
    ) {
        let trade = create_trade(entry_ns, hold_bars);
        prop_assert!(trade.exit_time_ns >= trade.entry_time_ns);
    }

    #[test]
    fn test_pnl_sign_matches_direction(
        direction in prop_oneof![Just(Direction::Long), Just(Direction::Short)],
        entry_price in 1.0f64..2.0,
        exit_price in 1.0f64..2.0,
    ) {
        let pnl = calculate_pnl(direction, entry_price, exit_price);
        match direction {
            Direction::Long => {
                if exit_price > entry_price {
                    prop_assert!(pnl > 0.0);
                }
            }
            Direction::Short => {
                if exit_price < entry_price {
                    prop_assert!(pnl > 0.0);
                }
            }
        }
    }
}
```

### Rust Test Commands

```bash
# Alle Rust-Tests
cargo test --all

# Spezifisches Crate
cargo test -p omega-execution

# Mit Output
cargo test -- --nocapture

# Property Tests (langsamer)
cargo test --all -- --ignored proptest
```

---

## V2 Coverage-Ziele

| Bereich/Crate | Ziel-Coverage | Begründung |
|--------------|---------------|------------|
| `types` | 95% | Fundament: Datenmodelle & Invarianten |
| `data` | 90% | Data Governance ist Fail-Fast Kernrisiko |
| `execution` | 90% | Fill-/Tie-Breaks sind correctness-kritisch |
| `portfolio` | 90% | State Machine + Equity-Konsistenz |
| `strategy` | 85% | Strategie-Regeln, aber oft fixture-lastig |
| `backtest` | 80% | Event Loop ist schwer granular zu testen |
| `metrics` | 85% | Formeln/Edge-Cases müssen stabil sein |
| `ffi` | 60% | Glue-Code, Schwerpunkt auf Contract/E2E |
| `python/bt` | 70% | Orchestrator/Reporting, Contract-lastig |

---
