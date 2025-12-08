from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Literal


TDType = Literal[
    "string",
    "datetime64[ns]",
    "Float64",
    "Int32",
]

TDCheckCallable = Literal[
    "min_value",
    "max_value",
    "min_length",
    "max_length",
    "contains_only",
]

TDViewer = Literal[
    "excel",
    "vscode",
    "console"
]


TDNormalizationMethod = Literal[
    "text_stressed",
    "text_relaxed"
]


Pathlike = str | Path