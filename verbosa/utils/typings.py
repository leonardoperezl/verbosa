from __future__ import annotations
from pathlib import Path
from typing import Literal


TDType = Literal[
    "string",
    "datetime64[ns]",
    "Float64",
    "Int32",
    "boolean"
]

TDReviewMethod = Literal[
    "min_value",
    "max_value",
    "min_length",
    "max_length",
    "contains_only",
    "no_na"
]

TDViewer = Literal[
    "excel",
    "vscode",
    "console"
]

TDNormalizationMethod = Literal[
    "text",
    "numeric",
    "date",
    "categorical",
    "boolean",
    "text_stressed",
    "text_relaxed",
    "numeric_float",
    "numeric_int",
    "date_yearfirst",
    "date_dayfirst"
]

Pathlike = str | Path