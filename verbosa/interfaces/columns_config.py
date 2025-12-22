from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Iterator, Optional, Sequence
import logging


from verbosa.utils.typings import Pathlike
from verbosa.interfaces.column_config import (
    ColumnConfig, CallSpec, StrCastedDTypes
)
from verbosa.data.readers.local import FileDataReader


logger = logging.getLogger(__name__)


class ColumnsConfig(Mapping[str, ColumnConfig]):
    """
    A representation of the data found at a columns configuration file.
    
    > Note. Parameters do not recive validation.
    
    Parameters
    ----------
    name : str
        Name of the configuration (for identification)
    
    description : str
        Description of the configuration
    
    author : str
        Author of the configuration
    
    date : str
        Date of creation/modification
    
    columns : Sequence[ColumnConfig]
        List of column configurations
    """
    def __init__(
        self,
        name: str,
        description: str,
        author: str,
        date: str,
        columns: Sequence[ColumnConfig]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.author: str = author
        self.date: str = date
        self._columns = tuple(columns)
        self.columns: tuple[str] = tuple(col.name for col in self._columns)
        self._build_index()
    
    def _build_index(self) -> None:
        """
        Build index for column lookup by name/alias.
        """
        self._index: dict[str, ColumnConfig] = {}
        
        for col in self._columns:
            for alias in col.aliases:
                if alias not in self._index:
                    self._index[alias] = col
                    continue
                
                logger.debug(
                    f"Conflict while indexing alias {alias} for column "
                    f"{col.name}",
                )
            
            # (self._index[col.name] is self._index[alias]) == True
    
    # ------------------------- Dunder Methods ----------------------------- #
    def __getitem__(self, key: str) -> ColumnConfig:
        """
        Mapping access by name or alias (case-insensitive).
        
        Raises
        ------
        TypeError
            If key is not a string.
        KeyError
            If no column with the given name/alias exists.
        """
        if not isinstance(key, str):
            raise TypeError(
                f"Column key must be str, not {type(key).__name__}"
            )
        
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(f"Column {key!r} not found") from None
    
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over primary column names in their original order.
        """
        for col in self._columns: yield col.name
    
    def __len__(self) -> int:
        """
        Number of *columns* (primary entries), preserving sequence semantics.
        """
        return len(self._columns)
    
    def __contains__(self, key: object) -> bool:
        """
        Membership by name or alias, case-insensitive.
        """
        if not isinstance(key, str): return False
        return key in self._index
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"columns={len(self._columns)})"
        )
    
    # ------------------------- Serialization Methods ---------------------- #
    @classmethod
    def from_yaml(cls, file_path: Pathlike) -> ColumnsConfig:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        file_path : Pathlike
            Path to the YAML configuration file.
        
        Returns
        -------
        ColumnsConfigFile
            Loaded configuration instance.
        """
        data = FileDataReader.read_yaml(file_path=file_path)
        if not data:
            raise ValueError(f"Empty or invalid YAML file: {file_path}")
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnsConfig:
        """
        Create instance from dictionary.
        """
        columns_data: dict[str, Any] = data.get("columns", dict())
        if not columns_data:
            raise ValueError("Configuration must contain 'columns' list")
        
        columns = tuple(
            ColumnConfig.from_dict(name=cname, data=cdata)
            for cname, cdata in columns_data.items()
        )
        
        return cls(
            name=data["name"],
            description=data["description"],
            author=data["author"],
            date=data["date"],
            columns=columns
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert instance to a dictionary.
        """
        result = {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "date": self.date
        }
        
        # Add columns
        result["columns"] = {
            col.name: col.to_dict()
            for col in self._columns
        }
        return result
    
    def validate_aliases(self) -> list[str]:
        """
        Validate configuration and return list of issues.
        
        It checks for:
        - Duplicate column names
        - Overlapping aliases between columns
        
        Returns
        -------
        list[str]
            Validation error messages (empty if valid)
        """
        issues = []
        
        seen_names: dict[str, list[str]] = {}
        for col in self._columns:
            seen_names.setdefault(col.name, []).append(col.name)
        
        duplicate_names = {
            column_name: aliases
            for column_name, aliases in seen_names.items()
            if len(aliases) > 1
        }
        if duplicate_names:
            formatted = ", ".join(
                f"{'/'.join(sorted(set(names)))}"
                for names in duplicate_names.values()
            )
            issues.append(
                f"Duplicate column names (case-insensitive): {formatted}"
            )
        
        # 2) Overlapping names/aliases between columns (case-sensitive)
        #    We track where each normalized identifier came from.
        used: set = set()
        for col in self._columns:
            # Aliases
            for alias in col.aliases:
                if alias not in used:
                    used.add(alias)
                    continue
                
                issues.append(
                    f"Alias '{alias}' in column '{col.name}' "
                    "overlaps with another column"
                )
        
        return issues
    
    def is_valid(self) -> bool:
        """
        Check if configuration is valid.
        
        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        return len(self.validate_aliases()) == 0
    
    def group_by_normalization(self) -> tuple[tuple[CallSpec, tuple[str, ...]], ...]:
        """
        Group columns by each individual normalization "step".
        
        Returns
        -------
        tuple[tuple[CallSpec, tuple[str, ...]], ...]
            Each entry is a pair:
            (normalization_call_spec, (column_names...)).
        
        Notes
        -----
        - A single column may appear in multiple groups if it has a pipeline
        with multiple steps.
        - If you need *pipeline* grouping (columns sharing the same full
        pipeline), use `group_by_normalization_pipeline`.
        """
        groups: OrderedDict[CallSpec, list[str]] = OrderedDict()
        
        for col in self._columns:
            pipeline = col.normalization
            if pipeline is None:
                continue
            for spec in pipeline:
                groups.setdefault(spec, []).append(col.name)
        
        return tuple((spec, tuple(cols)) for spec, cols in groups.items())
    
    def group_by_normalization_pipeline(
        self,
    ) -> tuple[tuple[tuple[CallSpec, ...], tuple[str, ...]], ...]:
        """
        Group columns by their full normalization pipeline.
        
        Returns
        -------
        tuple[tuple[tuple[CallSpec, ...], tuple[str, ...]], ...]
            Each entry is a pair: (pipeline, (column_names...)).
        """
        groups: OrderedDict[tuple[CallSpec, ...], list[str]] = OrderedDict()
        
        for col in self._columns:
            pipeline = col.normalization or ()
            groups.setdefault(pipeline, []).append(col.name)
        
        return tuple(
            (pipeline, tuple(cols)) for pipeline, cols in groups.items()
        )
    
    def get_na_values_dict(
        self
    ) -> dict[str, Optional[tuple[StrCastedDTypes]]]:
        """
        Get a dictionary mapping column names to their na_values.
        
        Returns
        -------
        dict[str, Optional[tuple[AllowedCastingDTypes]]]
            Mapping of column names to their na_values tuples.
        """
        result: dict[str, Optional[tuple[StrCastedDTypes]]] = {}
        for col in self._columns:
            result[col.name] = col.na_values
        return result
    
    def get_columns_fill_na_dict(
        self
    ) -> dict[str, Optional[StrCastedDTypes]]:
        """
        Get a dictionary mapping column names to their fill_na values.
        
        Returns
        -------
        dict[str, Optional[AllowedCastingDTypes]]
            Mapping of column names to their fill_na values.
        """
        result: dict[str, Optional[StrCastedDTypes]] = {}
        for col in self._columns:
            result[col.name] = col.fill_na
        return result