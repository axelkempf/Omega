# Type stubs for joblib (verwendet in backtest_engine.optimizer)
#
# joblib hat teilweise Type Hints, aber nicht vollständig.
# Diese Stubs decken die im Projekt verwendeten APIs ab.

from collections.abc import Sequence
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

_T = TypeVar("_T")

# Memory/Caching
class Memory:
    def __init__(
        self,
        location: Optional[str] = ...,
        backend: str = ...,
        mmap_mode: Optional[str] = ...,
        compress: Union[bool, int] = ...,
        verbose: int = ...,
        bytes_limit: Union[int, str] = ...,
        backend_options: Optional[dict[str, Any]] = ...,
    ) -> None: ...
    def cache(
        self,
        func: Optional[Callable[..., _T]] = ...,
        ignore: Optional[Sequence[str]] = ...,
        verbose: Optional[int] = ...,
        mmap_mode: Optional[str] = ...,
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...
    def clear(self, warn: bool = ...) -> None: ...
    def reduce_size(self, bytes_limit: Union[int, str] = ...) -> None: ...

# Parallel execution
class Parallel:
    def __init__(
        self,
        n_jobs: Optional[int] = ...,
        backend: Optional[str] = ...,
        verbose: int = ...,
        timeout: Optional[float] = ...,
        pre_dispatch: Union[str, int] = ...,
        batch_size: Union[str, int] = ...,
        temp_folder: Optional[str] = ...,
        max_nbytes: Union[int, str, None] = ...,
        mmap_mode: Optional[str] = ...,
        prefer: Optional[str] = ...,
        require: Optional[str] = ...,
        return_as: str = ...,
    ) -> None: ...
    def __call__(self, iterable: Iterable[Any]) -> list[Any]: ...
    def __enter__(self) -> Parallel: ...
    def __exit__(self, *args: Any) -> None: ...

# Note: joblib.delayed returns a special DelayedFunction wrapper object.
# We approximate this as Any here to avoid an incorrect callable/tuple signature.
def delayed(function: Callable[..., _T]) -> Any: ...

# Persistence
def dump(
    value: Any,
    filename: str,
    compress: Union[bool, int, tuple[str, int]] = ...,
    protocol: Optional[int] = ...,
    cache_size: Optional[int] = ...,
) -> list[str]: ...
def load(
    filename: str,
    mmap_mode: Optional[str] = ...,
) -> Any: ...

# Hash utilities
def hash(
    obj: Any,
    hash_name: str = ...,
    coerce_mmap: bool = ...,
) -> str: ...
