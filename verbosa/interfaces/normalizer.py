from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any, TypeVar, Protocol, runtime_checkable, Generic
import logging

import pandas as pd

from verbosa.utils.typings import Pathlike


logger = logging.getLogger(__name__)


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


@runtime_checkable
class ConfigurableNormalizer(Protocol):
    """Protocol for normalizers that use configuration objects."""
    
    def from_config(self, config_path: Pathlike) -> Any:
        """Load configuration from a file path."""
        ...


class NormalizerInterface(ABC, Generic[T]):
    """
    Abstract base class for data normalization implementations.
    
    This interface defines the contract that all normalizers must follow,
    providing a consistent API for data normalization operations across
    different data types and formats.
    
    Type Parameters
    ---------------
    T
        The type of data being normalized (e.g., pd.DataFrame, np.ndarray)
    
    Parameters
    ----------
    data : T
        The data to be normalized. Type depends on the concrete implementation.
    autonorm_settings : Optional[Pathlike | ConfigT], default None
        Settings for automatic normalization. Can be a path to a configuration
        file or a configuration object, depending on the implementation.
        
    Attributes
    ----------
    data : T
        The data being normalized
    autonorm_settings : Optional[Pathlike | ConfigT]
        The normalization configuration
    """
    
    def __init__(
        self,
        data: T,
        autonorm_settings: Optional[Pathlike | ConfigT] = None
    ) -> None:
        self.data: T = data
        self.autonorm_settings: Optional[Pathlike | ConfigT] = autonorm_settings
    
    def autonorm(self) -> T:
        """
        Apply automatic normalization to the data.
        
        This method triggers the automatic normalization process using the
        settings provided during initialization. It validates that settings
        are available before proceeding.
        
        Returns
        -------
        T
            The normalized data of the same type as input
            
        Raises
        ------
        ValueError
            If no normalization settings are provided
        """
        if self.autonorm_settings is None:
            raise ValueError(
                "No normalization settings provided. Cannot perform automatic "
                "normalization without configuration."
            )
        
        logger.info("Starting automatic normalization process.")
        result = self._autonorm_implementation()
        logger.info("Completed automatic normalization process.")
        return result
    
    @abstractmethod
    def _autonorm_implementation(self) -> T:
        """
        Concrete implementation of the automatic normalization logic.
        
        This method must be implemented by subclasses to provide the actual
        normalization logic specific to the data type being handled.
        
        Returns
        -------
        T
            The normalized data
        """
        return self.data
    
    def validate_data(self) -> bool:
        """
        Validate that the data is in an acceptable state for normalization.
        
        This method can be overridden by subclasses to implement specific
        validation logic for their data types.
        
        Returns
        -------
        bool
            True if data is valid for normalization, False otherwise
        """
        if self.data is None:
            logger.warning("Data is None, cannot perform normalization.")
            return False
        return True
    
    def get_data_info(self) -> dict[str, Any]:
        """
        Get basic information about the data being normalized.
        
        This method can be overridden by subclasses to provide specific
        information relevant to their data type.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing basic information about the data
        """
        return {
            "data_type": type(self.data).__name__,
            "has_autonorm_settings": self.autonorm_settings is not None,
            "is_valid": self.validate_data()
        }