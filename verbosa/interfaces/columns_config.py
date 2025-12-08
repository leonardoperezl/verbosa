from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence 
import logging


import yaml


from verbosa.utils.global_typings import Pathlike
from verbosa.interfaces.column_config import ColumnConfig
from verbosa.data.readers.local import FileDataReader


logger = logging.getLogger(__name__)


@dataclass
class ColumnsConfig:
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
    name: str  # Name of the configuration (for identification)
    description: str  # Description of the configuration
    author: str  # Author of the configuration
    date: str  # Date of creation/modification
    columns: Sequence[ColumnConfig]  # List of column configurations
    
    def __post_init__(self) -> None:
        """
        Build index for column lookup by name/alias.
        """
        self._column_index: dict[str, ColumnConfig] = {}
        for col in self.columns:
            # Index by primary name
            self._column_index[col.name] = col
            #* Index by all aliases. Comment to avoid name/alias conflicts
            for alias in col.aliases:
                self._column_index[alias] = col
    
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
        columns_data = data.get("columns", [])
        if not columns_data:
            raise ValueError("Configuration must contain 'columns' list")
        
        columns = [ColumnConfig.from_dict(col) for col in columns_data]
        
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
        result["columns"] = [col.to_dict() for col in self.columns]
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
    
    def column_names(self) -> list[str]:
        """Get list of all column names."""
        return [col.name for col in self.columns]
    
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
        
        # 1) Check for duplicate column names
        names = [col.name for col in self.columns]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            issues.append(f"Duplicate column names: {set(duplicates)}")
        
        # 2) Check for overlapping aliases
        all_column_names = set()
        for col in self.columns:
            col_aliases = col.aliases
            overlap = all_column_names & col_aliases
            if overlap:
                issues.append(
                    f"Column '{col.name}' has names/aliases overlapping "
                    f"with other columns: {overlap}"
                )
            all_column_names.update(col_aliases)
        
        return issues
    
    def __len__(self) -> int:
        """
        """
        return len(self.columns)
    
    def __getitem__(self, key: int | str) -> ColumnConfig | None:
        """
        Access column by index or name/alias.
        """
        if isinstance(key, int):
            return self.columns[key]  # Access by index
        elif isinstance(key, str):
            search_name = key.strip().lower()
            col = self._column_index.get(search_name)
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