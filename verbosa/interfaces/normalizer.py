from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any, TypeVar
import logging


import pandas as pd



logger = logging.getLogger(__name__)



T = TypeVar("T")


class NormalizerInterface(ABC):
    def __init__(
        self,
        data: T,
        autonorm_settings: Optional[Any] = None
    ) -> None:
        self.data: T = data
        self.autonorm_settings: Optional[Any] = autonorm_settings
    
    def autonorm(self) -> T:
        if self.autonorm_settings is None:
            raise ValueError("No normalization details provided.")
        
        return self._autonorm_implementation()
    
    @abstractmethod
    def _autonorm_implementation(self) -> T:
        return self.data