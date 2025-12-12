from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import logging


import yaml


from verbosa.utils.global_typings import Pathlike
from verbosa.interfaces.column_config import ColumnConfig
from verbosa.data.readers.local import FileDataReader


logger = logging.getLogger(__name__)


@dataclass
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
    name: str
    description: str
    author: str
    date: str
    columns: Sequence[ColumnConfig]
    
    def __post_init__(self) -> None:
        """
        Build index for column lookup by name/alias.
        """
        self._index: dict[str, ColumnConfig] = {}
        
        for col in self.columns:
            # Index by primary name
            if col.name not in self._index:
                self._index[col.name] = col
            else:
                logger.debug(
                    f"Duplicated name detected while indexing {col.name}"
                )
            
            #* Index by all aliases. Comment to avoid name/alias conflicts
            for alias in col.aliases:
                if alias not in self._index:
                    self._index[alias] = col
                else:
                    logger.debug(
                        f"Conflict while indexing alias {alias} for column "
                        f"{col.name}",
                    )
            
            # (self._index[col.name] is self._index[alias]) == True
        
        self.column_names: tuple[str] = tuple(
            col.name for col in self.columns
        )
    
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
        for col in self.columns:
            yield col.name
    
    def __len__(self) -> int:
        """
        Number of *columns* (primary entries), preserving sequence semantics.
        """
        return len(self.columns)
    
    def __contains__(self, key: object) -> bool:
        """
        Membership by name or alias, case-insensitive.
        """
        if not isinstance(key, str):
            return False
        return key in self._index
    
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
        
        columns = [
            ColumnConfig.from_dict(name=cname, data=cdata)
            for cname, cdata in columns_data.items()
        ]
        
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
        result = {}
        
        # Add metadata if present
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.author:
            result["author"] = self.author
        if self.date:
            result["date"] = self.date
        
        # Add columns
        result["columns"] = {
            col.name: col.to_dict()
            for col in self.columns
        }
        return result
    
    # TODO. To write the file use the data writer class
    def to_yaml_file(self, file_path: Pathlike) -> None:
        """
        Save configuration to YAML file.
        
        Parameters
        ----------
        file_path : Pathlike
            Path to save the YAML configuration file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
        
        logger.info(f"Configuration saved to {path}")
    
    def validate(self) -> list[str]:
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
        for col in self.columns:
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
        
        # 2) Overlapping names/aliases between columns (case-insensitive)
        #    We track where each normalized identifier came from.
        used: dict[str, tuple[str, str]] = {}  # norm -> (kind, column_name)
        
        for col in self.columns:
            # Primary name
            if col.name in used:
                prev_kind, prev_owner = used[col.name]
                if prev_owner != col.name:
                    # This is already covered by duplicate names, but we
                    # keep this here in case you want a more detailed message.
                    issues.append(
                        f"Primary name {col.name!r} conflicts with "
                        f"{prev_kind} of column {prev_owner!r}"
                    )
            else:
                used[col.name] = ("name", col.name)
            
            # Aliases
            for alias in col.aliases:
                if alias in used:
                    prev_kind, prev_owner = used[alias]
                    if prev_owner != col.name:
                        issues.append(
                            f"Identifier {alias!r} on column {col.name!r} "
                            f"conflicts with {prev_kind} of column "
                            f"{prev_owner!r}"
                        )
                else:
                    used[alias] = ("alias", col.name)
        
        return issues
    
    def __len__(self) -> int:
        """
        """
        return len(self.columns)
    
    def __getitem__(self, key: int | str) -> ColumnConfig | None:
        if isinstance(key, int):
            return self.columns[key]  # Access by index
        elif isinstance(key, str):
            search_name = key
            col: ColumnConfig | None  = self._index.get(search_name)
            if col is None:
                return None
            return col  # Access by name/alias
        else:
            raise TypeError(
                f"Column key must be int or str, not {type(key).__name__}"
            )
    
    def __iter__(self):
        """Iterate over columns."""
        return iter(self.columns)