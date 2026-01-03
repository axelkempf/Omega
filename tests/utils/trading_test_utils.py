"""Trading test utilities for reusable test helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tests.mocks.mock_broker import MockPosition


@dataclass
class FileWaitResult:
    """Result of waiting for a file."""

    path: Path
    content: str | None


def create_mock_position(
    ticket: int,
    symbol: str = "EURUSD",
    magic: int = 12345,
    direction: str = "buy",
    volume: float = 0.01,
    price_open: float = 1.1000,
    sl: float = 1.0950,
    tp: float = 1.1100,
    profit: float = 0.0,
    comment: str = "",
) -> MockPosition:
    """Create a mock position with sensible defaults."""

    return MockPosition(
        ticket=ticket,
        symbol=symbol,
        magic=magic,
        direction=direction,
        volume=volume,
        price_open=price_open,
        sl=sl,
        tp=tp,
        profit=profit,
        comment=comment,
    )


def wait_for_file(
    path: Path,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> bool:
    """Wait for a file to exist, with timeout."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(poll_interval)
    return path.exists()


def wait_for_file_content(
    path: Path,
    expected_content: str | None = None,
    timeout: float = 5.0,
    require_non_empty: bool = True,
) -> str | None:
    """Wait for file to exist and optionally contain expected content.

    Args:
        path: File path to wait for
        expected_content: If set, wait until file contains this content
        timeout: Maximum seconds to wait
        require_non_empty: If True and expected_content is None, wait for non-empty content
    """

    if not wait_for_file(path, timeout=timeout):
        return None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = path.read_text()
        except Exception:
            text = None
        if text is not None:
            # If expecting specific content, check for match
            if expected_content is not None:
                if expected_content in text or text.strip() == expected_content.strip():
                    return text
            # If no specific content expected but require non-empty
            elif require_non_empty:
                if text.strip():
                    return text
            # No requirements - return immediately
            else:
                return text
        time.sleep(0.05)
    try:
        return path.read_text()
    except Exception:
        return None


def assert_position_matched(
    position: Any,
    expected_magic: int,
    msg: str = "",
) -> None:
    """Assert that a position has the expected magic number."""

    assert position is not None, msg or "Expected position, got None"
    try:
        pos_magic = int(getattr(position, "magic"))
    except Exception as exc:  # pragma: no cover - defensive path
        raise AssertionError(msg or f"Position missing magic: {exc}") from exc
    assert pos_magic == expected_magic, (
        msg or f"Unexpected magic {pos_magic}, expected {expected_magic}"
    )


def assert_positions_filtered_correctly(
    positions: Iterable[Any],
    expected_magic: int,
    msg: str = "",
) -> None:
    """Assert all positions in list have expected magic number."""

    for pos in positions:
        assert_position_matched(pos, expected_magic, msg=msg)


def create_heartbeat_file(
    directory: Path,
    account_id: str,
    timestamp: str | None = None,
) -> Path:
    """Create a heartbeat file for testing in the given directory."""

    directory.mkdir(parents=True, exist_ok=True)
    hb_path = directory / f"heartbeat_{account_id}.txt"
    hb_path.write_text(timestamp if timestamp is not None else str(time.time()))
    return hb_path


def create_stop_signal_file(
    directory: Path,
    account_id: str,
) -> Path:
    """Create a stop signal file for testing."""

    directory.mkdir(parents=True, exist_ok=True)
    stop_path = directory / f"stop_{account_id}.signal"
    stop_path.write_text("stop")
    return stop_path
