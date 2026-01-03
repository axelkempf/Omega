# Sizing Module

The **Sizing Module** is responsible for calculating position sizes, managing risk, and handling symbol-specific specifications (e.g., contract size, tick value). It ensures that trades are sized correctly according to the account currency and risk parameters.

## Features

- **Position Sizing**: `lot_sizer.py` calculates the appropriate lot size based on risk per trade (e.g., % of equity) and stop loss distance.
- **Symbol Specifications**: `symbol_specs_registry.py` manages a registry of symbol properties (contract size, pip size, currencies) essential for accurate calculations.
- **Currency Conversion**: `rate_provider.py` handles conversion between quote currencies and the account currency.
- **Commission Calculation**: `commission.py` calculates trading costs based on volume or value.

## Key Components

| File | Description |
|------|-------------|
| `lot_sizer.py` | Calculates position sizes based on risk and market conditions. |
| `symbol_specs_registry.py` | Registry for symbol-specific data (contract size, min volume, etc.). |
| `rate_provider.py` | Provides exchange rates for currency conversion. |
| `commission.py` | Calculates commissions and fees for trades. |

## Usage

```python
from backtest_engine.sizing.lot_sizer import LotSizer

# Initialize sizer
sizer = LotSizer(account_ccy="EUR", rate_provider=rate_provider, specs=specs_registry)

# Calculate lot size for 1% risk
lots = sizer.size_risk_based(
    symbol="EURUSD",
    price=1.1050,
    stop_loss=1.1000,
    risk_amount=100.0  # 100 EUR risk
)
```
