# engine_ui/strategies/base.py

from abc import ABC, abstractmethod


class BaseStrategyManager(ABC):
    def __init__(self, alias: str, resolved_id: str):
        self.alias = alias
        self.resolved_id = resolved_id

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self) -> bool:
        pass

    @abstractmethod
    def status(self) -> str:
        pass
