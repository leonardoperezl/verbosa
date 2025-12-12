from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import logging


from verbosa.utils.global_typings import (
    TDType,
    TDReviewMethod,
    TDNormalizationMethod
)
from verbosa.utils.serialization_helpers import str_sequence_param


logger = logging.getLogger(__name__)



@dataclass
class ColumnConfig:
    """
    Stores data related to a column.
    """
    name: str  # The main reference to the column
    dtype: TDType
    description: Optional[str] = None
    # Other possible names for the column
    aliases: Optional[Sequence[str] | str] = None
    na_values: Optional[Sequence[Any] | Any] = None
    fill_na: Optional[str] = None
    reviews: Optional[dict[TDReviewMethod, dict] | TDReviewMethod] = None
    normalization: Optional[
        dict[TDNormalizationMethod, dict] | TDNormalizationMethod] = None
    
    def __post_init__(self) -> None:
        self.aliases: set = set(self.aliases or [])
        self.aliases.add(self.name)
    
    # --------------------------- Instance methods ------------------------- #
    def is_alias(self, alias: str) -> bool:
        """
        Check if given column name is an alias for this column.
        """
        return alias in self.aliases
    
    # ------------------------- Serialization Methods ---------------------- #
    @classmethod
    def from_dict(
        cls,
        name: str,
        data: dict[str, Any]
    ) -> ColumnConfig:
        """Parse column config from dict."""
        # Handle checks - can be None, list of dicts, or list with None
        parameters = {
            "name": name,
            "dtype": data.get("dtype"),
            "description": data.get("description"),
            "aliases": str_sequence_param(data.get("aliases"), tuple),
            "na_values": str_sequence_param(data.get("na_values", tuple)),
            "fill_na": data.get("fill_na"),
            "reviews": data.get("reviews"),
            "normalization": data.get("normalization")
        }
        return cls(**parameters)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert back to YAML-compatible dict."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "aliases": tuple(self.aliases),
            "na_values": self.na_values,
            "fill_na": self.fill_na,
            "reviews": self.reviews,
            "normalization": self.normalization
        }