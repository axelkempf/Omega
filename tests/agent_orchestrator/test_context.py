"""Unit tests for ContextManager component.

Tests hierarchical scopes, thread safety, and context entry management.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.agent_orchestrator.context import (
    ContextEntry,
    ContextManager,
    ContextScope,
    TaskContext,
)


class TestContextScope:
    """Test ContextScope enum."""

    def test_scope_values(self) -> None:
        """Verify scope values are defined."""
        assert ContextScope.GLOBAL is not None
        assert ContextScope.TASK is not None
        assert ContextScope.AGENT is not None

    def test_scope_hierarchy(self) -> None:
        """Verify scope hierarchy (GLOBAL > TASK > AGENT)."""
        # GLOBAL should have lower value (higher priority)
        assert ContextScope.GLOBAL.value < ContextScope.TASK.value
        assert ContextScope.TASK.value < ContextScope.AGENT.value


class TestContextEntry:
    """Test ContextEntry dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test default values."""
        entry = ContextEntry(key="test_key", value="test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.scope == ContextScope.TASK  # Default
        assert entry.created_at is not None
        assert entry.ttl_seconds is None
        assert entry.metadata == {}

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields populated."""
        now = datetime.utcnow()
        entry = ContextEntry(
            key="config",
            value={"symbol": "EURUSD"},
            scope=ContextScope.GLOBAL,
            created_at=now,
            ttl_seconds=3600,
            metadata={"source": "user"},
        )
        assert entry.key == "config"
        assert entry.value == {"symbol": "EURUSD"}
        assert entry.scope == ContextScope.GLOBAL
        assert entry.created_at == now
        assert entry.ttl_seconds == 3600
        assert entry.metadata["source"] == "user"

    def test_is_expired_without_ttl(self) -> None:
        """Test that entries without TTL never expire."""
        entry = ContextEntry(key="test", value="data", ttl_seconds=None)
        assert entry.is_expired() is False

    def test_is_expired_with_future_ttl(self) -> None:
        """Test that entries with future TTL are not expired."""
        entry = ContextEntry(key="test", value="data", ttl_seconds=3600)
        assert entry.is_expired() is False

    def test_is_expired_with_past_ttl(self) -> None:
        """Test that entries with past TTL are expired."""
        past = datetime.utcnow() - timedelta(hours=2)
        entry = ContextEntry(
            key="test",
            value="data",
            created_at=past,
            ttl_seconds=3600,  # 1 hour TTL, but 2 hours old
        )
        assert entry.is_expired() is True


class TestTaskContext:
    """Test TaskContext dataclass."""

    def test_creation(self) -> None:
        """Test TaskContext creation."""
        ctx = TaskContext(
            task_id="task-123",
            task_type="implementation",
            is_v2_context=True,
            affected_files=["/path/to/file.rs"],
        )
        assert ctx.task_id == "task-123"
        assert ctx.task_type == "implementation"
        assert ctx.is_v2_context is True
        assert len(ctx.affected_files) == 1

    def test_default_values(self) -> None:
        """Test TaskContext default values."""
        ctx = TaskContext(task_id="task-456", task_type="review")
        assert ctx.is_v2_context is False
        assert ctx.affected_files == []
        assert ctx.metadata == {}


class TestContextManager:
    """Test ContextManager class."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a fresh ContextManager instance."""
        return ContextManager()

    def test_set_and_get_global(self, manager: ContextManager) -> None:
        """Test setting and getting global context."""
        manager.set("global_key", "global_value", scope=ContextScope.GLOBAL)
        result = manager.get("global_key")
        assert result == "global_value"

    def test_set_and_get_task_scope(self, manager: ContextManager) -> None:
        """Test setting and getting task-scoped context."""
        manager.set(
            "task_key", "task_value", scope=ContextScope.TASK, task_id="task-123"
        )
        result = manager.get("task_key", task_id="task-123")
        assert result == "task_value"

    def test_set_and_get_agent_scope(self, manager: ContextManager) -> None:
        """Test setting and getting agent-scoped context."""
        manager.set(
            "agent_key",
            "agent_value",
            scope=ContextScope.AGENT,
            task_id="task-123",
            agent_id="implementer",
        )
        result = manager.get("agent_key", task_id="task-123", agent_id="implementer")
        assert result == "agent_value"

    def test_scope_hierarchy_lookup(self, manager: ContextManager) -> None:
        """Test that lookup follows scope hierarchy."""
        # Set at global scope
        manager.set("shared_key", "global_value", scope=ContextScope.GLOBAL)
        
        # Should be accessible from task scope
        result = manager.get("shared_key", task_id="any-task")
        assert result == "global_value"

    def test_task_scope_overrides_global(self, manager: ContextManager) -> None:
        """Test that task scope overrides global scope."""
        manager.set("key", "global", scope=ContextScope.GLOBAL)
        manager.set("key", "task", scope=ContextScope.TASK, task_id="task-123")
        
        # Task-specific lookup should get task value
        result = manager.get("key", task_id="task-123")
        assert result == "task"
        
        # Global lookup should still get global value
        result = manager.get("key")
        assert result == "global"

    def test_get_nonexistent_key(self, manager: ContextManager) -> None:
        """Test getting nonexistent key returns None."""
        result = manager.get("nonexistent")
        assert result is None

    def test_get_with_default(self, manager: ContextManager) -> None:
        """Test getting nonexistent key with default."""
        result = manager.get("nonexistent", default="fallback")
        assert result == "fallback"

    def test_delete_entry(self, manager: ContextManager) -> None:
        """Test deleting context entry."""
        manager.set("to_delete", "value")
        assert manager.get("to_delete") == "value"
        
        manager.delete("to_delete")
        assert manager.get("to_delete") is None

    def test_clear_task_context(self, manager: ContextManager) -> None:
        """Test clearing all context for a task."""
        manager.set("key1", "value1", scope=ContextScope.TASK, task_id="task-123")
        manager.set("key2", "value2", scope=ContextScope.TASK, task_id="task-123")
        manager.set("key3", "value3", scope=ContextScope.GLOBAL)
        
        manager.clear_task("task-123")
        
        # Task entries should be gone
        assert manager.get("key1", task_id="task-123") is None
        assert manager.get("key2", task_id="task-123") is None
        
        # Global should remain
        assert manager.get("key3") == "value3"

    def test_get_all_for_task(self, manager: ContextManager) -> None:
        """Test getting all context for a task."""
        manager.set("global", "g_value", scope=ContextScope.GLOBAL)
        manager.set("task", "t_value", scope=ContextScope.TASK, task_id="task-123")
        manager.set(
            "agent",
            "a_value",
            scope=ContextScope.AGENT,
            task_id="task-123",
            agent_id="impl",
        )
        
        all_context = manager.get_all_for_task("task-123", agent_id="impl")
        
        assert "global" in all_context
        assert "task" in all_context
        assert "agent" in all_context

    def test_ttl_expiration(self, manager: ContextManager) -> None:
        """Test that expired entries are not returned."""
        # Create already-expired entry by manipulating created_at
        past = datetime.utcnow() - timedelta(hours=2)
        entry = ContextEntry(
            key="expired",
            value="old_data",
            scope=ContextScope.GLOBAL,
            created_at=past,
            ttl_seconds=3600,
        )
        # Directly insert into manager (bypassing set)
        manager._global_context["expired"] = entry
        
        # Should return None because expired
        result = manager.get("expired")
        assert result is None

    def test_cleanup_expired(self, manager: ContextManager) -> None:
        """Test cleanup of expired entries."""
        # Add non-expired entry
        manager.set("fresh", "data", ttl_seconds=3600)
        
        # Add expired entry directly
        past = datetime.utcnow() - timedelta(hours=2)
        expired_entry = ContextEntry(
            key="stale",
            value="old",
            scope=ContextScope.GLOBAL,
            created_at=past,
            ttl_seconds=3600,
        )
        manager._global_context["stale"] = expired_entry
        
        # Cleanup
        cleaned = manager.cleanup_expired()
        
        assert cleaned >= 1
        assert "fresh" in manager._global_context
        assert "stale" not in manager._global_context


class TestContextManagerThreadSafety:
    """Test thread safety of ContextManager."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a fresh ContextManager instance."""
        return ContextManager()

    def test_concurrent_writes(self, manager: ContextManager) -> None:
        """Test concurrent writes don't corrupt state."""
        errors: list[Exception] = []
        
        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    manager.set(key, f"value_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_reads_writes(self, manager: ContextManager) -> None:
        """Test concurrent reads and writes."""
        manager.set("shared", "initial")
        errors: list[Exception] = []
        read_values: list[str] = []
        
        def reader() -> None:
            try:
                for _ in range(100):
                    val = manager.get("shared")
                    if val is not None:
                        read_values.append(val)
            except Exception as e:
                errors.append(e)
        
        def writer() -> None:
            try:
                for i in range(100):
                    manager.set("shared", f"value_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Should have read some values
        assert len(read_values) > 0

    def test_concurrent_task_operations(self, manager: ContextManager) -> None:
        """Test concurrent operations on different tasks."""
        errors: list[Exception] = []
        
        def task_worker(task_id: str) -> None:
            try:
                for i in range(50):
                    manager.set(
                        f"key_{i}",
                        f"value_{i}",
                        scope=ContextScope.TASK,
                        task_id=task_id,
                    )
                    _ = manager.get(f"key_{i}", task_id=task_id)
                
                manager.clear_task(task_id)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=task_worker, args=(f"task-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestContextManagerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a fresh ContextManager instance."""
        return ContextManager()

    def test_set_none_value(self, manager: ContextManager) -> None:
        """Test setting None value."""
        manager.set("key", None)
        result = manager.get("key")
        assert result is None

    def test_set_complex_value(self, manager: ContextManager) -> None:
        """Test setting complex nested value."""
        complex_value = {
            "config": {
                "symbol": "EURUSD",
                "params": [1, 2, 3],
            },
            "metadata": {"timestamp": datetime.utcnow().isoformat()},
        }
        manager.set("complex", complex_value)
        result = manager.get("complex")
        assert result == complex_value

    def test_overwrite_existing_key(self, manager: ContextManager) -> None:
        """Test overwriting existing key."""
        manager.set("key", "value1")
        manager.set("key", "value2")
        assert manager.get("key") == "value2"

    def test_empty_task_id(self, manager: ContextManager) -> None:
        """Test with empty task_id."""
        manager.set("key", "value", scope=ContextScope.TASK, task_id="")
        result = manager.get("key", task_id="")
        assert result == "value"

    def test_special_characters_in_key(self, manager: ContextManager) -> None:
        """Test keys with special characters."""
        manager.set("key:with:colons", "value1")
        manager.set("key.with.dots", "value2")
        manager.set("key/with/slashes", "value3")
        
        assert manager.get("key:with:colons") == "value1"
        assert manager.get("key.with.dots") == "value2"
        assert manager.get("key/with/slashes") == "value3"
