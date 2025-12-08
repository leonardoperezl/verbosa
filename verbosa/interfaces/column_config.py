from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import logging


from verbosa.utils.global_typings import (
    TDType, TDCheckCallable
)


logger = logging.getLogger(__name__)



@dataclass
class ColumnCheck:
    """
    Represents a single validation check for a column.
    """
    name: TDCheckCallable
    parameters: Any = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert back to YAML-compatible dict."""
        return {self.name: self.parameters}


@dataclass
class ColumnConfig:
    """
    Stores data related to a column.
    """
    name: str  # The main reference to the column
    dtype: TDType
    description: Optional[str] = None
    # Other possible names for the column
    aliases: Optional[Sequence[str]] = None
    nullable: bool = True
    allow_duplicates: bool = True
    fill_na: Optional[Any] = None
    checks: Optional[Sequence[ColumnCheck]] = None
    normalization: Optional[str] = None  # The normalization method to apply
    
    def __post_init__(self) -> None:
        self.aliases: set = set(self.aliases or [])
        self.aliases.add(self.name)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnConfig:
        """Parse column config from YAML dict."""
        # Handle checks - can be None, list of dicts, or list with None
        raw_checks: dict[str, Any] = data.get("checks") or dict()
        checks = []
        for check_method, parameters in raw_checks.items():
            check = ColumnCheck(check_method, parameters)
            checks.append(check)
        
        checks = checks if checks else None
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            description=data.get("description"),
            aliases=data.get("aliases"),
            nullable=data.get("nullable", True),
            allow_duplicates=data.get("allow_duplicates", True),
            fill_na=data.get("fill_na"),
            checks=checks,
            normalization=data.get("normalization")
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert back to YAML-compatible dict."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "aliases": self.aliases if self.aliases else None,
            "nullable": self.nullable,
            "allow_duplicates": self.allow_duplicates,
            "fill_na": self.fill_na,
            "checks": {
                c.name: c.parameters for c in self.checks
            } if self.checks else None,
            "normalization": self.normalization
        }
    
    def is_alias(self, alias: str) -> bool:
        """
        Check if given column name is an alias for this column.
        """
        return alias in self.aliases