# 06 - Agent Permissions

> Least Privilege Principle für KI-Agenten

**Status:** 🔴 Offen
**Priorität:** Niedrig
**Komplexität:** Hoch
**Geschätzter Aufwand:** 2-3 Tage

---

## Objective

Implementiere ein **Permission-System** für KI-Agenten das:
- Zugriff auf sensible Bereiche einschränkt
- Audit-Trails für alle Änderungen erstellt
- Sandbox-Umgebungen für Tests bereitstellt
- Rollback bei Fehlern ermöglicht

---

## Current State

### Problem

Aktuell haben alle Agenten **uneingeschränkten Zugriff**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT STATE                                │
│                                                                  │
│  ┌─────────────┐                                                │
│  │   Agent     │ ─── READ/WRITE ──► Gesamte Codebase            │
│  │             │ ─── EXECUTE ────► Alle Commands                 │
│  │             │ ─── MODIFY ─────► Configs, Secrets, Live Code  │
│  └─────────────┘                                                │
│                                                                  │
│  ⚠️ Risiken:                                                     │
│  • Agent kann versehentlich Live-Trading-Code ändern            │
│  • Agent kann Secrets in Code committen                         │
│  • Agent kann Breaking Changes ohne Warnung einführen           │
│  • Keine Nachverfolgbarkeit welcher Agent was geändert hat     │
└─────────────────────────────────────────────────────────────────┘
```

### Identifizierte Risiko-Bereiche

| Bereich | Risiko-Level | Begründung |
|---------|--------------|------------|
| `src/hf_engine/core/execution/` | 🔴 Kritisch | Live-Order-Ausführung |
| `src/hf_engine/core/risk/` | 🔴 Kritisch | Risk Management |
| `configs/live/` | 🔴 Kritisch | Live-Trading-Konfiguration |
| `.env`, Secrets | 🔴 Kritisch | Zugangsdaten |
| `src/strategies/*/live/` | 🟡 Hoch | Live-Strategie-Logik |
| `src/backtest_engine/` | 🟢 Niedrig | Nur Simulation |
| `tests/` | 🟢 Niedrig | Test-Code |
| `docs/` | 🟢 Niedrig | Dokumentation |

---

## Target State

### Permission Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TARGET STATE                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Permission Manager                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   Roles     │  │   Policies  │  │   Audit Log    │  │   │
│  │  │  (YAML)     │  │  (YAML)     │  │  (SQLite)      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                   Access Control                          │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  Agent: Implementer                               │   │   │
│  │  │  ├── READ:  src/**, tests/**, docs/**            │   │   │
│  │  │  ├── WRITE: src/backtest_engine/**, tests/**     │   │   │
│  │  │  └── DENY:  src/hf_engine/core/execution/**      │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  Agent: Reviewer                                  │   │   │
│  │  │  ├── READ:  **                                   │   │   │
│  │  │  └── WRITE: (none)                               │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  Agent: Live-Maintainer (special role)            │   │   │
│  │  │  ├── READ:  **                                   │   │   │
│  │  │  ├── WRITE: src/hf_engine/**, configs/live/**    │   │   │
│  │  │  └── REQUIRES: Human approval                    │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Permission Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                   PERMISSION LEVELS                              │
│                                                                  │
│  Level 0: SANDBOX (Experimental)                                │
│  ├── Can: Create temporary files, run in isolated env          │
│  └── Cannot: Modify any tracked files                          │
│                                                                  │
│  Level 1: CONTRIBUTOR (Default Agent)                           │
│  ├── Can: Modify docs, tests, backtest engine                  │
│  └── Cannot: Modify live engine, configs, strategies           │
│                                                                  │
│  Level 2: DEVELOPER (Trusted Agent)                             │
│  ├── Can: Modify most code except live execution               │
│  └── Cannot: Modify execution engine, risk manager             │
│                                                                  │
│  Level 3: MAINTAINER (With Human Approval)                      │
│  ├── Can: Modify any code                                      │
│  └── Requires: Human approval for live-critical changes        │
│                                                                  │
│  Level 4: ADMIN (Human Only)                                    │
│  ├── Can: Everything                                           │
│  └── Used for: Emergency fixes, permission changes             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Schritt 1: Permission Config

Erstelle `.github/agent_permissions.yaml`:

```yaml
# .github/agent_permissions.yaml
# Agent Permission Configuration

version: "1.0"

# Zone Definitions
zones:
  critical:
    description: "Live trading and risk management"
    paths:
      - "src/hf_engine/core/execution/**"
      - "src/hf_engine/core/risk/**"
      - "src/hf_engine/adapter/broker/**"
      - "configs/live/**"
    requires_approval: true
    audit_level: full

  sensitive:
    description: "Live strategies and configuration"
    paths:
      - "src/strategies/*/live/**"
      - "src/hf_engine/**"
      - ".env*"
      - "**/secrets/**"
    requires_approval: false
    audit_level: detailed

  standard:
    description: "Backtest, analysis, tests"
    paths:
      - "src/backtest_engine/**"
      - "src/ui_engine/**"
      - "tests/**"
      - "docs/**"
    requires_approval: false
    audit_level: basic

  unrestricted:
    description: "Documentation and scripts"
    paths:
      - "*.md"
      - "scripts/**"
      - ".github/instructions/**"
    requires_approval: false
    audit_level: none

# Role Definitions
roles:
  sandbox:
    level: 0
    description: "Experimental - no persistent changes"
    permissions:
      read: ["**"]
      write: []
      execute: ["pytest", "python -c"]
    restrictions:
      - "Cannot modify tracked files"
      - "Cannot commit changes"

  contributor:
    level: 1
    description: "Default agent role"
    permissions:
      read: ["**"]
      write:
        - "docs/**"
        - "tests/**"
        - "src/backtest_engine/analysis/**"
      execute: ["pytest", "pre-commit", "black", "isort"]
    restrictions:
      - "Cannot modify live engine"
      - "Cannot modify strategies"

  developer:
    level: 2
    description: "Trusted agent with broader access"
    permissions:
      read: ["**"]
      write:
        - "src/backtest_engine/**"
        - "src/ui_engine/**"
        - "src/shared/**"
        - "tests/**"
        - "docs/**"
      execute: ["pytest", "pre-commit", "python"]
    restrictions:
      - "Cannot modify hf_engine/core/execution"
      - "Cannot modify hf_engine/core/risk"
      - "Cannot modify live configs"

  maintainer:
    level: 3
    description: "Full access with approval workflow"
    permissions:
      read: ["**"]
      write: ["**"]
      execute: ["**"]
    approval_required_for:
      - "src/hf_engine/core/execution/**"
      - "src/hf_engine/core/risk/**"
      - "configs/live/**"
    restrictions:
      - "Critical changes require human approval"

# Agent-Role Mappings
agent_roles:
  architect:
    default_role: developer
    can_request: [maintainer]

  implementer:
    default_role: contributor
    can_request: [developer]

  reviewer:
    default_role: sandbox  # Read-only
    can_request: []

  tester:
    default_role: contributor
    can_request: [developer]

  researcher:
    default_role: sandbox
    can_request: []

  devops:
    default_role: developer
    can_request: [maintainer]

# Audit Configuration
audit:
  storage: "var/logs/agent_audit.db"
  retention_days: 90
  log_fields:
    - timestamp
    - agent_id
    - agent_role
    - action
    - target_path
    - result
    - approval_status
```

### Schritt 2: Permission Manager

```python
# src/agent_permissions/manager.py
"""Agent permission management."""

from __future__ import annotations

import fnmatch
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Action(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"


class PermissionResult(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class PermissionCheck:
    """Result of a permission check."""

    result: PermissionResult
    reason: str
    zone: str | None = None
    required_role: str | None = None


@dataclass
class AuditEntry:
    """Audit log entry."""

    timestamp: datetime
    agent_id: str
    agent_role: str
    action: Action
    target_path: str
    result: PermissionResult
    approval_status: str | None = None
    details: dict[str, Any] | None = None


class PermissionManager:
    """Manages agent permissions and audit logging."""

    def __init__(self, config_path: Path = Path(".github/agent_permissions.yaml")):
        self.config_path = config_path
        self.config = self._load_config()
        self._init_audit_db()

    def _load_config(self) -> dict:
        """Load permission configuration."""

        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        """Return default (permissive) configuration."""

        return {
            "roles": {
                "default": {
                    "level": 1,
                    "permissions": {
                        "read": ["**"],
                        "write": ["**"],
                        "execute": ["**"]
                    }
                }
            },
            "zones": {},
            "agent_roles": {}
        }

    def _init_audit_db(self) -> None:
        """Initialize audit database."""

        audit_config = self.config.get("audit", {})
        db_path = Path(audit_config.get("storage", "var/logs/agent_audit.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(db_path))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                action TEXT NOT NULL,
                target_path TEXT NOT NULL,
                result TEXT NOT NULL,
                approval_status TEXT,
                details TEXT
            )
        """)
        self.db.commit()

    def check_permission(
        self,
        agent_id: str,
        agent_role: str,
        action: Action,
        target_path: str
    ) -> PermissionCheck:
        """Check if an agent has permission for an action."""

        # Get role configuration
        role_mapping = self.config.get("agent_roles", {}).get(agent_role, {})
        effective_role = role_mapping.get("default_role", "contributor")
        role_config = self.config.get("roles", {}).get(effective_role, {})

        # Check zone
        zone = self._get_zone(target_path)

        # Check if zone requires approval
        if zone:
            zone_config = self.config["zones"][zone]
            if zone_config.get("requires_approval"):
                return PermissionCheck(
                    result=PermissionResult.APPROVAL_REQUIRED,
                    reason=f"Zone '{zone}' requires human approval",
                    zone=zone,
                    required_role="maintainer"
                )

        # Check role permissions
        allowed_patterns = role_config.get("permissions", {}).get(action.value, [])

        for pattern in allowed_patterns:
            if fnmatch.fnmatch(target_path, pattern):
                return PermissionCheck(
                    result=PermissionResult.ALLOWED,
                    reason=f"Allowed by role '{effective_role}' pattern: {pattern}",
                    zone=zone
                )

        return PermissionCheck(
            result=PermissionResult.DENIED,
            reason=f"Role '{effective_role}' does not have {action.value} permission for {target_path}",
            zone=zone,
            required_role=self._find_required_role(action, target_path)
        )

    def _get_zone(self, path: str) -> str | None:
        """Determine which zone a path belongs to."""

        for zone_name, zone_config in self.config.get("zones", {}).items():
            for pattern in zone_config.get("paths", []):
                if fnmatch.fnmatch(path, pattern):
                    return zone_name
        return None

    def _find_required_role(self, action: Action, path: str) -> str | None:
        """Find the minimum role required for an action on a path."""

        for role_name, role_config in self.config.get("roles", {}).items():
            patterns = role_config.get("permissions", {}).get(action.value, [])
            for pattern in patterns:
                if fnmatch.fnmatch(path, pattern):
                    return role_name
        return "maintainer"

    def log_action(
        self,
        agent_id: str,
        agent_role: str,
        action: Action,
        target_path: str,
        result: PermissionResult,
        approval_status: str | None = None,
        details: dict | None = None
    ) -> None:
        """Log an action to the audit database."""

        import json

        self.db.execute(
            """
            INSERT INTO audit_log
            (timestamp, agent_id, agent_role, action, target_path, result, approval_status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                agent_id,
                agent_role,
                action.value,
                target_path,
                result.value,
                approval_status,
                json.dumps(details) if details else None
            )
        )
        self.db.commit()

    def get_audit_log(
        self,
        agent_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100
    ) -> list[AuditEntry]:
        """Retrieve audit log entries."""

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        cursor = self.db.execute(query, params)
        rows = cursor.fetchall()

        import json

        return [
            AuditEntry(
                timestamp=datetime.fromisoformat(row[1]),
                agent_id=row[2],
                agent_role=row[3],
                action=Action(row[4]),
                target_path=row[5],
                result=PermissionResult(row[6]),
                approval_status=row[7],
                details=json.loads(row[8]) if row[8] else None
            )
            for row in rows
        ]

    def request_approval(
        self,
        agent_id: str,
        agent_role: str,
        action: Action,
        target_path: str,
        reason: str
    ) -> str:
        """Request human approval for a restricted action."""

        # Create approval request
        request_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent_id[:8]}"

        # Log the request
        self.log_action(
            agent_id=agent_id,
            agent_role=agent_role,
            action=action,
            target_path=target_path,
            result=PermissionResult.APPROVAL_REQUIRED,
            approval_status="pending",
            details={"request_id": request_id, "reason": reason}
        )

        return request_id
```

### Schritt 3: Integration Hooks

```python
# src/agent_permissions/hooks.py
"""Pre-commit hooks for permission enforcement."""

from pathlib import Path
from .manager import PermissionManager, Action, PermissionResult


def check_file_permission(file_path: str, agent_id: str, agent_role: str) -> bool:
    """Check if agent can write to a file (for pre-commit)."""

    manager = PermissionManager()
    result = manager.check_permission(
        agent_id=agent_id,
        agent_role=agent_role,
        action=Action.WRITE,
        target_path=file_path
    )

    if result.result == PermissionResult.DENIED:
        print(f"❌ Permission denied: {result.reason}")
        return False

    if result.result == PermissionResult.APPROVAL_REQUIRED:
        print(f"⚠️  Approval required: {result.reason}")
        print(f"   Zone: {result.zone}")
        print(f"   Request approval with: omega-permissions request-approval {file_path}")
        return False

    return True
```

### Schritt 4: CLI

```python
# src/agent_permissions/cli.py
"""CLI for permission management."""

import argparse
from datetime import datetime, timedelta

from .manager import PermissionManager, Action


def main():
    parser = argparse.ArgumentParser(description="Agent Permissions")
    subparsers = parser.add_subparsers(dest="command")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check permission")
    check_parser.add_argument("agent_role", help="Agent role")
    check_parser.add_argument("action", choices=["read", "write", "execute"])
    check_parser.add_argument("path", help="Target path")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="View audit log")
    audit_parser.add_argument("--agent", help="Filter by agent ID")
    audit_parser.add_argument("--days", type=int, default=7, help="Days to show")

    # Request approval command
    request_parser = subparsers.add_parser("request-approval", help="Request approval")
    request_parser.add_argument("path", help="Target path")
    request_parser.add_argument("--reason", required=True, help="Reason for request")

    args = parser.parse_args()

    manager = PermissionManager()

    if args.command == "check":
        result = manager.check_permission(
            agent_id="cli",
            agent_role=args.agent_role,
            action=Action(args.action),
            target_path=args.path
        )
        print(f"Result: {result.result.value}")
        print(f"Reason: {result.reason}")
        if result.zone:
            print(f"Zone: {result.zone}")

    elif args.command == "audit":
        since = datetime.now() - timedelta(days=args.days)
        entries = manager.get_audit_log(agent_id=args.agent, since=since)

        for entry in entries:
            print(f"{entry.timestamp} | {entry.agent_role} | {entry.action.value} | {entry.result.value}")
            print(f"  Path: {entry.target_path}")

    elif args.command == "request-approval":
        request_id = manager.request_approval(
            agent_id="cli",
            agent_role="implementer",
            action=Action.WRITE,
            target_path=args.path,
            reason=args.reason
        )
        print(f"Approval request created: {request_id}")


if __name__ == "__main__":
    main()
```

---

## Acceptance Criteria

- [ ] `agent_permissions.yaml` definiert alle Zonen und Rollen
- [ ] Permission Manager kann Berechtigungen prüfen
- [ ] Audit Log speichert alle Aktionen
- [ ] CLI kann Berechtigungen abfragen
- [ ] Pre-Commit Hook blockiert unerlaubte Änderungen
- [ ] Approval-Workflow für kritische Bereiche

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Zu restriktiv | Hoch | Mittel | Default "contributor" mit breitem Zugriff |
| Umgehung möglich | Mittel | Hoch | Git Hooks + CI/CD Enforcement |
| Performance Overhead | Niedrig | Niedrig | Caching der Konfiguration |

---

## Dependencies

- `01_agent_roles.md` muss implementiert sein
- `04_precommit_validation.md` für Enforcement

---

## Future Enhancements

1. **Web UI** für Approval-Workflows
2. **Temporary Permissions** mit Ablaufdatum
3. **Role Escalation** mit 2FA
4. **Anomaly Detection** für verdächtige Aktivitäten
