from __future__ import annotations
from pathlib import Path
from typing import Optional


def is_file_path(path: str, extension: Optional[str] = None) -> bool:
    """Check if the given path is a valid file path."""
    is_file = Path(path).is_file()
    
    extension = extension.lstrip(".") if extension else None
    if extension:
        return is_file and Path(path).suffix == extension
    return is_file