from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING, Any
import json
import logging
import shutil
from pathlib import Path


import pandas as pd
import yaml


from verbosa.utils.typings import Pathlike


logger = logging.getLogger(__name__)


class FileSystemNavigator:
    def __init__(self, start: Pathlike) -> None:
        self._wd: Path = Path(start)  # Working Directory
        logger.info(
            f"TextFileDataReader initialized with start path: {self._wd}"
        )
    
    def cd(self, path: Pathlike) -> None:
        """
        """
        target_path = Path(path)
        
        # Handle relative paths
        if not target_path.is_absolute():
            target_path = self._wd / target_path
        
        # Resolve any symbolic links or relative components
        target_path = target_path.resolve()
        
        if not target_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {target_path}")
        
        if not target_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {target_path}")
        
        old_path = self._wd
        self._wd = target_path
        logger.info(f"Changed directory from {old_path} to {self._wd}")
    
    def mv(self, source: Pathlike, destination: Pathlike) -> None:
        """
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        # Handle relative paths
        if not source_path.is_absolute():
            source_path = self._wd / source_path
        if not dest_path.is_absolute():
            dest_path = self._wd / dest_path
            
        source_path = source_path.resolve()
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
            
        try:
            # Use shutil.move for cross-filesystem moves
            moved_path = shutil.move(str(source_path), str(dest_path))
            logger.info(f"Moved {source_path} to {moved_path}")
        except PermissionError as e:
            logger.error(f"Permission denied when moving {source_path} to {dest_path}")
            raise e
        except Exception as e:
            logger.error(f"Error moving {source_path} to {dest_path}: {e}")
            raise e
    
    def pwd(self) -> Path:
        """
        """
        return self._wd
    
    def ls(self) -> dict[str, Path]:
        """
        """
        if not self._wd.is_dir():
            raise NotADirectoryError(
                f"Current path is not a directory: {self._wd}"
            )
        
        files_at_wd = self._wd.iterdir()
        listing = {item.name: item for item in files_at_wd}
        
        formatted_listing = ", ".join(listing.keys())
        logger.debug(
            f"Directory listing for {self._wd}: {formatted_listing}".rstrip(", ")
        )
        return listing
    
    def rm(self, target: Pathlike, *, confirm: bool | None = None) -> None:
        """
        """
        if confirm is not True:
            logger.info("rm called without confirm=True; no action taken")
            return
        
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = (self._wd / target_path).resolve()
        else:
            target_path = target_path.resolve()
        
        if not target_path.exists():
            raise FileNotFoundError(f"Target does not exist: {target_path}")
        
        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
                logger.info(f"Removed directory: {target_path}")
            else:
                target_path.unlink()
                logger.info(f"Removed file: {target_path}")
        except PermissionError:
            logger.error(f"Permission denied when removing {target_path}")
            raise
        except Exception as e:
            logger.error(f"Error removing {target_path}: {e}")
            raise


class FileDataReader(FileSystemNavigator):
    
    # ------------------------ Formatted data storage ------------------------
    @classmethod
    def read_csv(cls, file_path: Pathlike) -> pd.DataFrame:
        pass
    
    @classmethod
    def read_json(cls, file_path: Pathlike) -> dict[str, Any] | list[Any]:
        with open(file_path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    
    @classmethod
    def read_yaml(cls, file_path: Pathlike) -> dict[str, Any] | list[Any]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        with open(file_path, mode="r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # --------------------------- Simple text data ---------------------------
    @classmethod
    def read_txt(cls, file_path: Pathlike) -> str:
        with open(file_path, mode="r", encoding="utf-8") as f:
            return f.read()
    
    @classmethod
    def read_sql(cls, file_path: Pathlike) -> Template:
        file_txt: str = FileDataReader.read_txt(file_path)
        return Template(file_txt)
    
    # ------------------------ Non-binary data storage -----------------------
    @classmethod
    def read_excel(cls, file_path: Pathlike) -> pd.DataFrame:
        pass
    
    @classmethod
    def read_file(
        cls,
        file_path: Pathlike
    ) -> dict | pd.DataFrame | str | Any:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower().replace(".", "")
        method_name = f"read_{suffix}"
        
        if not hasattr(cls, method_name):
            raise ValueError(f"Unsupported file type: .{suffix}")
        
        read_method = getattr(cls, method_name)
        return read_method(file_path)