from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ReviewerInterface(ABC):
    def __init__(self, data: Any) -> None:
        self.data: Any = data
    
    def review(self) -> Any:
        return self._review_implementation()