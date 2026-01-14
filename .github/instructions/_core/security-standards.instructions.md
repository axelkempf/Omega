---
description: 'Security standards based on OWASP guidelines - Single Source of Truth'
applyTo: '**'
---

# Security Standards

> Sicherheitsstandards basierend auf OWASP Top 10.
> Diese Datei ist die Single Source of Truth – alle anderen Instruktionen referenzieren hierher.

---

## OWASP Top 10 Compliance

### 1. Injection Prevention

#### SQL Injection

```python
# ❌ NIEMALS: String Concatenation
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ IMMER: Parameterized Queries
cursor.execute("SELECT * FROM users WHERE email = ?", (email,))

# ✅ SQLAlchemy
session.query(User).filter(User.email == email).first()
```

#### Command Injection

```python
import subprocess
import shlex

# ❌ NIEMALS: shell=True mit User Input
subprocess.run(f"ls {user_dir}", shell=True)

# ✅ IMMER: Liste ohne Shell
subprocess.run(["ls", user_dir], shell=False)

# ✅ Wenn Shell nötig: shlex.quote()
subprocess.run(f"ls {shlex.quote(user_dir)}", shell=True)
```

#### Template Injection

```python
from jinja2 import Environment, select_autoescape

# ✅ Auto-Escaping aktivieren
env = Environment(autoescape=select_autoescape(['html', 'xml']))
```

---

### 2. Secrets Management

#### Grundregeln

| Regel | Umsetzung |
|-------|-----------|
| Nie committen | Keine Passwörter, API-Keys, Tokens im Code |
| Environment Variables | Via `python-dotenv` laden |
| Dokumentation | Required ENV vars in README dokumentieren |
| Rotation | Secrets regelmäßig rotieren |

#### Pattern

```python
import os
from dotenv import load_dotenv

# .env laden (lokal)
load_dotenv()

# Secrets aus Environment
API_KEY = os.getenv("API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Validation
if not API_KEY:
    raise EnvironmentError("API_KEY not set in environment")
```

#### .env Template

```bash
# .env.example (wird committet)
API_KEY=your_api_key_here
DB_PASSWORD=your_password_here
MT5_LOGIN=your_login_here
MT5_PASSWORD=your_password_here
MT5_SERVER=your_server_here
```

---

### 3. Input Validation

#### Grundregeln

- Validiere an **jeder Boundary** (User Input, External APIs, Files)
- **Fail Fast**: Exceptions sofort werfen
- **Sanitize**: HTML, SQL, Shell-Zeichen

#### Patterns

```python
import re
from typing import TypedDict

class UserInput(TypedDict):
    email: str
    username: str

def validate_email(email: str) -> str:
    """Validate and return sanitized email."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"Invalid email format: {email}")
    return email.lower().strip()

def validate_username(username: str) -> str:
    """Validate username (alphanumeric only)."""
    if not username.isalnum():
        raise ValueError("Username must be alphanumeric")
    if len(username) < 3 or len(username) > 20:
        raise ValueError("Username must be 3-20 characters")
    return username.lower()

def validate_user_input(data: dict) -> UserInput:
    """Validate complete user input."""
    return UserInput(
        email=validate_email(data.get("email", "")),
        username=validate_username(data.get("username", "")),
    )
```

#### Path Traversal Prevention

```python
from pathlib import Path

def safe_file_path(base_dir: Path, user_path: str) -> Path:
    """Ensure path stays within base directory."""
    # Resolve to absolute path
    full_path = (base_dir / user_path).resolve()
    
    # Check it's still within base
    if not str(full_path).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal attempt detected")
    
    return full_path
```

---

### 4. Authentication & Authorization

#### Grundregeln

- Permissions **vor** jeder geschützten Operation prüfen
- Etablierte Libraries verwenden (kein Custom Crypto)
- Security Events loggen (Failed Logins, Permission Denials)

#### Pattern

```python
from functools import wraps
from typing import Callable
import logging

security_logger = logging.getLogger("security")

def require_permission(permission: str) -> Callable:
    """Decorator to check permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(user, *args, **kwargs):
            if not user.has_permission(permission):
                security_logger.warning(
                    f"Permission denied: {user.id} tried to access {func.__name__}"
                )
                raise PermissionError(f"Missing permission: {permission}")
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

# Nutzung
@require_permission("trade.execute")
def execute_trade(user, trade_params):
    ...
```

---

### 5. Error Handling

#### Information Disclosure Prevention

```python
import logging
from typing import Any

logger = logging.getLogger(__name__)

def safe_error_response(error: Exception, context: str) -> dict[str, Any]:
    """Create safe error response without internal details."""
    # Log full details internally
    logger.error(f"Error in {context}: {error}", exc_info=True)
    
    # Return safe message externally
    return {
        "error": True,
        "message": "An error occurred. Please try again later.",
        "reference": generate_error_id(),  # For support lookup
    }
```

#### Was NICHT loggen

- Passwörter
- API Keys / Tokens
- PII (Personally Identifiable Information)
- Credit Card Numbers
- Session Tokens

---

### 6. Dependency Security

#### Grundregeln

| Regel | Tool |
|-------|------|
| Vulnerability Scanning | `pip-audit`, `safety` |
| Version Pinning | `pyproject.toml` mit exakten Versionen |
| Update Schedule | Monatliche Security Updates |
| License Check | Vor neuen Dependencies prüfen |

#### Commands

```bash
# Vulnerability Check
pip-audit

# Safety Check
safety check

# Update Dependencies
pip install --upgrade <package>
```

---

## Code Review Security Checklist

### Pre-Commit Checks

- [ ] Keine hardcodierten Secrets
- [ ] Input Validation vorhanden
- [ ] Parameterized Queries verwendet
- [ ] Error Messages leaken keine internen Details
- [ ] Dependencies sind aktuell

### API Endpoints

- [ ] Authentication erforderlich
- [ ] Authorization geprüft
- [ ] Rate Limiting implementiert
- [ ] Input validiert und sanitized
- [ ] Output escaped (HTML/JSON)

### File Operations

- [ ] Path Traversal Prevention
- [ ] File Type Validation
- [ ] Size Limits
- [ ] Keine ausführbaren Uploads

### Database

- [ ] Prepared Statements
- [ ] Least Privilege Principle
- [ ] Connection Pooling mit Limits
- [ ] Keine SQL in Logs

---

## Trading-Spezifische Sicherheit

### Kritische Pfade

| Pfad | Risiko | Anforderung |
|------|--------|-------------|
| Order Execution | Finanzielle Verluste | Double Validation |
| Position Sizing | Überhöhte Risiken | Hard Limits |
| Stop Loss | Unbegrenzte Verluste | Always Set |
| API Keys | Unauthorized Access | Environment Only |

### Pattern: Order Validation

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class OrderLimits:
    max_lot_size: Decimal = Decimal("10.0")
    max_risk_percent: Decimal = Decimal("0.02")
    min_stop_loss_pips: int = 5

def validate_order(order: Order, limits: OrderLimits) -> Order:
    """Validate order against safety limits."""
    if order.lot_size > limits.max_lot_size:
        raise ValueError(f"Lot size {order.lot_size} exceeds max {limits.max_lot_size}")
    
    if order.risk_percent > limits.max_risk_percent:
        raise ValueError(f"Risk {order.risk_percent} exceeds max {limits.max_risk_percent}")
    
    if order.stop_loss_pips < limits.min_stop_loss_pips:
        raise ValueError(f"Stop loss too tight: {order.stop_loss_pips} pips")
    
    return order
```

---

## Quick Reference

| Bereich | Regel |
|---------|-------|
| Secrets | Nur via Environment Variables |
| SQL | Nur Parameterized Queries |
| Input | Validate at every boundary |
| Errors | Don't leak internal details |
| Logging | No PII, no secrets |
| Dependencies | Monthly security updates |
| Crypto | Use established libraries only |
