---
description: 'Trading safety and risk management rules for Omega trading stack'
applyTo: 'src/hf_engine/**,src/strategies/**,configs/live/**'
---

# Trading Safety Standards

> Trading-spezifische Sicherheitsregeln für das Omega-Projekt.
> Diese Datei ergänzt die allgemeinen Security-Standards um domänenspezifische Regeln.

---

## Kritische Pfade

Die folgenden Pfade erfordern erhöhte Aufmerksamkeit und zusätzliche Review-Prozesse:

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| `src/hf_engine/core/execution/` | Live-Order-Ausführung | Safety Auditor + Human Approval |
| `src/hf_engine/core/risk/` | Risk Management | Safety Auditor + Human Approval |
| `src/hf_engine/adapter/broker/` | MT5-Kommunikation | Reviewer + Integration Tests |
| `configs/live/` | Live-Trading-Konfiguration | Human Approval Required |
| `src/strategies/*/live/` | Live-Strategie-Logik | Safety Auditor Review |

---

## Non-Negotiable Rules

### 1. Keine stillen Live-Änderungen

Änderungen am Trading-Verhalten MÜSSEN:

- [ ] Hinter einem Config-Flag stehen ODER
- [ ] Explizite Migration mit Dokumentation haben
- [ ] Im PR-Title/Description als "Breaking Change" markiert sein

```python
# ✅ Neue Logik hinter Flag
if config.get("use_new_sl_logic", False):
    calculate_stop_loss_v2(...)
else:
    calculate_stop_loss_v1(...)  # Legacy bleibt Default

# ❌ Silent change
def calculate_stop_loss(...):  # Geändertes Verhalten ohne Flag
    ...
```

### 2. Resume-Semantik wahren

Das Matching offener Positionen via `magic_number` ist eine **Invariante**:

```python
# Diese Logik darf NICHT brechen
def find_matching_position(magic_number: int) -> Position | None:
    """Find position by magic number - CRITICAL PATH."""
    for position in get_open_positions():
        if position.magic == magic_number:
            return position
    return None
```

Bei Änderungen an Position-Matching:
- Regression-Test erforderlich
- Safety Auditor Review
- Human Approval

### 3. var/-Layout Invarianten

Der Runtime-State in `var/` ist operational kritisch:

```
var/
├── tmp/
│   ├── heartbeat_<account_id>.txt  # Heartbeat-Dateien
│   └── stop_<account_id>.signal    # Stop-Signale
├── logs/                           # Log-Dateien
│   ├── system/
│   ├── trade_logs/
│   └── entry_logs/
└── results/                        # Backtest-Ergebnisse
```

**Regeln:**
- Pfade NICHT ändern ohne DevOps-Abstimmung
- Heartbeat-Format ist Contract mit UI-Engine
- Stop-Signal-Erkennung darf nicht brechen

---

## Risk Management Limits

### Hard Limits (nicht überschreitbar)

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class RiskLimits:
    """Hard limits for risk management - do not override."""
    
    # Position Sizing
    MAX_LOT_SIZE: Decimal = Decimal("10.0")
    MAX_RISK_PER_TRADE: Decimal = Decimal("0.02")  # 2%
    MAX_TOTAL_EXPOSURE: Decimal = Decimal("0.10")  # 10%
    
    # Stop Loss
    MIN_STOP_LOSS_PIPS: int = 5
    MAX_STOP_LOSS_PIPS: int = 500
    
    # Take Profit
    MIN_RISK_REWARD_RATIO: Decimal = Decimal("1.0")
    
    # Session
    MAX_TRADES_PER_SESSION: int = 10
    MAX_CONSECUTIVE_LOSSES: int = 3

RISK_LIMITS = RiskLimits()
```

### Validation Pattern

```python
def validate_trade_risk(
    trade: TradeOrder,
    account: AccountInfo,
    limits: RiskLimits = RISK_LIMITS,
) -> TradeOrder:
    """Validate trade against risk limits - MUST be called before execution."""
    
    # Lot Size Check
    if trade.lot_size > limits.MAX_LOT_SIZE:
        raise RiskLimitExceeded(
            f"Lot size {trade.lot_size} exceeds max {limits.MAX_LOT_SIZE}"
        )
    
    # Risk Per Trade Check
    risk_amount = calculate_risk_amount(trade)
    risk_percent = risk_amount / account.balance
    if risk_percent > limits.MAX_RISK_PER_TRADE:
        raise RiskLimitExceeded(
            f"Risk {risk_percent:.2%} exceeds max {limits.MAX_RISK_PER_TRADE:.2%}"
        )
    
    # Stop Loss Check
    if trade.stop_loss_pips < limits.MIN_STOP_LOSS_PIPS:
        raise RiskLimitExceeded(
            f"Stop loss {trade.stop_loss_pips} pips below minimum {limits.MIN_STOP_LOSS_PIPS}"
        )
    
    # Must have Stop Loss
    if trade.stop_loss is None:
        raise RiskLimitExceeded("Stop loss is required for all trades")
    
    return trade
```

---

## MT5 Spezifika

### Windows-Only Handling

```python
import sys

# MT5 ist Windows-only
MT5_AVAILABLE = False
if sys.platform == "win32":
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        pass

def require_mt5(func):
    """Decorator requiring MT5 availability."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not MT5_AVAILABLE:
            raise EnvironmentError(
                "MT5 required but not available. "
                "This operation requires Windows with MetaTrader5 installed."
            )
        return func(*args, **kwargs)
    return wrapper
```

### Backtests ohne MT5

Backtests MÜSSEN auf macOS/Linux ohne MT5 laufen:

```python
# ✅ Backtest mit Mock-Daten
def run_backtest(config: dict) -> BacktestResult:
    """Run backtest using historical data files."""
    # Keine MT5-Dependency
    data = load_parquet_data(config["symbol"], config["timeframe"])
    ...

# ❌ Backtest mit Live-Daten
def run_backtest(config: dict) -> BacktestResult:
    """Run backtest with live MT5 connection."""
    mt5.initialize()  # Bricht auf macOS/Linux!
    ...
```

---

## Execution Safety

### Order Validation Pipeline

```
Order Creation → Risk Validation → Symbol Validation → Execution → Confirmation
      ↓               ↓                  ↓               ↓            ↓
   Logging        Hard Limits      Spread/Hours      MT5 Call    Result Check
```

### Pre-Execution Checks

```python
def execute_order(order: Order) -> ExecutionResult:
    """Execute order with full validation pipeline."""
    
    # 1. Risk Validation
    validate_trade_risk(order, get_account_info())
    
    # 2. Symbol Validation
    symbol_info = mt5.symbol_info(order.symbol)
    if symbol_info is None:
        raise SymbolError(f"Symbol not found: {order.symbol}")
    
    if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        raise SymbolError(f"Trading disabled for {order.symbol}")
    
    # 3. Spread Check
    spread = symbol_info.ask - symbol_info.bid
    max_spread = get_max_spread(order.symbol)
    if spread > max_spread:
        raise SpreadTooWide(f"Spread {spread} exceeds max {max_spread}")
    
    # 4. Market Hours Check
    if not is_market_open(order.symbol):
        raise MarketClosed(f"Market closed for {order.symbol}")
    
    # 5. Execute
    result = mt5.order_send(order.to_mt5_request())
    
    # 6. Verify
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise ExecutionError(f"Order failed: {result.comment}")
    
    return ExecutionResult.from_mt5(result)
```

---

## Logging Requirements

### Trade Events (MUSS)

```python
import logging
from datetime import datetime

trade_logger = logging.getLogger("trade_logs")

def log_trade_event(event_type: str, trade: Trade, details: dict = None):
    """Log trade event with full context - REQUIRED."""
    trade_logger.info(
        f"{event_type}",
        extra={
            "timestamp": datetime.utcnow().isoformat(),
            "account_id": trade.account_id,
            "magic_number": trade.magic_number,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "lot_size": float(trade.lot_size),
            "entry_price": float(trade.entry_price),
            "stop_loss": float(trade.stop_loss) if trade.stop_loss else None,
            "take_profit": float(trade.take_profit) if trade.take_profit else None,
            "details": details or {},
        }
    )
```

### Was IMMER loggen

- Order Send (Entry)
- Order Modify (SL/TP Änderungen)
- Order Close (Exit)
- Fehler und Rejections
- Risk Limit Violations

### Was NIE loggen

- Account Passwörter
- API Keys
- Vollständige Account Balance (nur relative Änderungen)

---

## Code Review Checklist (Trading)

### Vor Merge prüfen

- [ ] Keine stillen Verhaltensänderungen in Trading-Logik
- [ ] Risk Limits werden eingehalten
- [ ] Stop Loss ist immer gesetzt
- [ ] Magic Number Matching unverändert
- [ ] var/-Pfade unverändert oder mit Migration
- [ ] Läuft auf macOS/Linux ohne MT5 (für Backtests)
- [ ] Trade Events werden geloggt
- [ ] Keine Secrets im Code

### Bei kritischen Pfaden zusätzlich

- [ ] Safety Auditor Review angefordert
- [ ] Human Approval eingeholt
- [ ] Regression Tests vorhanden
- [ ] Rollback Plan dokumentiert

---

## Quick Reference

| Regel | Requirement |
|-------|-------------|
| Stop Loss | IMMER setzen |
| Max Lot Size | 10.0 |
| Max Risk/Trade | 2% |
| Magic Number | NIEMALS brechen |
| var/ Layout | NICHT ändern |
| MT5 auf macOS | NIEMALS voraussetzen |
| Trade Logging | IMMER |
| Silent Changes | NIEMALS |
