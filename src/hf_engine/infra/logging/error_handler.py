# error_handler.py
from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Optional, TypeVar

from hf_engine.infra.logging.log_service import log_service

T = TypeVar("T")


def _safe_repr(value: Any, max_len: int = 200) -> str:
    """
    Create a safe, length‑limited representation for logging.
    Ensures that very large args/kwargs do not flood the logs.
    """
    try:
        s = repr(value)
    except Exception:
        return "<unrepr-able>"
    return s if len(s) <= max_len else f"{s[:max_len]}…(+{len(s)-max_len} chars)"


def log_exception(
    label: str,
    exception: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
    log_trace: bool = True,
) -> None:
    """
    Log an exception in a consistent, compact way.

    Args:
        label: Logical component or purpose of the call (e.g., "OrderSubmit").
        exception: The exception instance to log.
        context: Optional structured context (e.g., {"func": "foo", "args": "..."}).
        log_trace: If True, also log the full traceback.
    """
    base_msg = f"[{label}] {type(exception).__name__}: {exception}"
    if context:
        # Keep context single line to avoid multi-line log spam.
        ctx_pairs = ", ".join(f"{k}={_safe_repr(v)}" for k, v in context.items())
        base_msg = f"{base_msg} | context: {ctx_pairs}"

    log_service.log_system(base_msg, level="ERROR")

    if log_trace:
        log_service.log_system(traceback.format_exc(), level="ERROR")


def safe_execute(
    label: str,
    func: Callable[..., T],
    *args: Any,
    raise_on_error: bool = False,
    log_trace: bool = True,
    log_args: bool = False,
    **kwargs: Any,
) -> Optional[T]:
    """
    Execute a callable safely:
      - Logs any exception with context.
      - Returns the callable's result on success.
      - Returns None on failure (unless raise_on_error=True).

    Args:
        label: Logical component or purpose of the call (e.g., "DataFetch").
        func: The callable to execute.
        *args, **kwargs: Arguments forwarded to the callable.
        raise_on_error: Re-raise the exception after logging if True.
        log_trace: Include traceback in logs if True.
        log_args: Include (limited) args/kwargs in logs if True (helps debugging).

    Returns:
        The callable's return value, or None on error (if raise_on_error is False).
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        ctx: Dict[str, Any] = {"func": getattr(func, "__name__", "<callable>")}
        if log_args:
            # Only log compact representations to avoid sensitive data / log floods.
            ctx["args"] = [_safe_repr(a) for a in args] if args else []
            ctx["kwargs"] = (
                {k: _safe_repr(v) for k, v in kwargs.items()} if kwargs else {}
            )
        log_exception(label, e, context=ctx, log_trace=log_trace)
        if raise_on_error:
            raise
        return None
