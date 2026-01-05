# Type stubs for optuna (verwendet in backtest_engine.optimizer)
#
# Optuna hat Type Hints, aber mypy kann sie manchmal nicht finden.
# Diese Stubs decken die im Projekt verwendeten Core-APIs ab.

from typing import Any, Callable, Optional, Union, Sequence
from enum import Enum

# Study & Trial
class Study:
    def optimize(
        self,
        func: Callable[["Trial"], Union[float, Sequence[float]]],
        n_trials: Optional[int] = ...,
        timeout: Optional[float] = ...,
        n_jobs: int = ...,
        catch: tuple[type[Exception], ...] = ...,
        callbacks: Optional[list[Callable[[Study, "Trial"], None]]] = ...,
        gc_after_trial: bool = ...,
        show_progress_bar: bool = ...,
    ) -> None: ...
    
    @property
    def best_params(self) -> dict[str, Any]: ...
    
    @property
    def best_value(self) -> float: ...
    
    @property
    def best_trial(self) -> "Trial": ...
    
    @property
    def trials(self) -> list["Trial"]: ...
    
    def trials_dataframe(
        self,
        attrs: Sequence[str] = ...,
        multi_index: bool = ...,
    ) -> Any: ...  # pandas.DataFrame

class Trial:
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = ...,
        log: bool = ...,
    ) -> float: ...
    
    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = ...,
        log: bool = ...,
    ) -> int: ...
    
    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Union[str, int, float, bool, None]],
    ) -> Union[str, int, float, bool, None]: ...
    
    def suggest_uniform(self, name: str, low: float, high: float) -> float: ...
    def suggest_loguniform(self, name: str, low: float, high: float) -> float: ...
    
    @property
    def params(self) -> dict[str, Any]: ...
    
    @property
    def number(self) -> int: ...
    
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...
    
    def set_user_attr(self, key: str, value: Any) -> None: ...
    def user_attrs(self) -> dict[str, Any]: ...

class TrialState(Enum):
    RUNNING = ...
    COMPLETE = ...
    PRUNED = ...
    FAIL = ...
    WAITING = ...

# Study creation
def create_study(
    *,
    storage: Optional[Union[str, "storages.BaseStorage"]] = ...,
    sampler: Optional["samplers.BaseSampler"] = ...,
    pruner: Optional["pruners.BasePruner"] = ...,
    study_name: Optional[str] = ...,
    direction: Optional[Union[str, "StudyDirection"]] = ...,
    directions: Optional[Sequence[Union[str, "StudyDirection"]]] = ...,
    load_if_exists: bool = ...,
) -> Study: ...

def load_study(
    *,
    study_name: str,
    storage: Union[str, "storages.BaseStorage"],
    sampler: Optional["samplers.BaseSampler"] = ...,
    pruner: Optional["pruners.BasePruner"] = ...,
) -> Study: ...

class StudyDirection(Enum):
    MINIMIZE = ...
    MAXIMIZE = ...
    NOT_SET = ...

# Samplers
class samplers:
    class BaseSampler: ...
    
    class TPESampler(BaseSampler):
        def __init__(
            self,
            *,
            consider_prior: bool = ...,
            prior_weight: float = ...,
            consider_magic_clip: bool = ...,
            consider_endpoints: bool = ...,
            n_startup_trials: int = ...,
            n_ei_candidates: int = ...,
            seed: Optional[int] = ...,
            multivariate: bool = ...,
            warn_independent_sampling: bool = ...,
        ) -> None: ...
    
    class RandomSampler(BaseSampler):
        def __init__(self, seed: Optional[int] = ...) -> None: ...
    
    class GridSampler(BaseSampler):
        def __init__(
            self,
            search_space: dict[str, Sequence[Any]],
            seed: Optional[int] = ...,
        ) -> None: ...

# Pruners
class pruners:
    class BasePruner: ...
    
    class MedianPruner(BasePruner):
        def __init__(
            self,
            n_startup_trials: int = ...,
            n_warmup_steps: int = ...,
            interval_steps: int = ...,
        ) -> None: ...
    
    class PercentilePruner(BasePruner):
        def __init__(
            self,
            percentile: float,
            n_startup_trials: int = ...,
            n_warmup_steps: int = ...,
            interval_steps: int = ...,
            n_min_trials: int = ...,
        ) -> None: ...

# Storage
class storages:
    class BaseStorage: ...
    
    class InMemoryStorage(BaseStorage):
        def __init__(self) -> None: ...
    
    class RDBStorage(BaseStorage):
        def __init__(
            self,
            url: str,
            engine_kwargs: Optional[dict[str, Any]] = ...,
            skip_compatibility_check: bool = ...,
        ) -> None: ...

# Logging
class logging:
    @staticmethod
    def set_verbosity(verbosity: int) -> None: ...
    
    @staticmethod
    def get_verbosity() -> int: ...
    
    DEBUG: int
    INFO: int
    WARNING: int
    ERROR: int
    CRITICAL: int

# Exceptions
class TrialPruned(Exception): ...
class OptunaError(Exception): ...
