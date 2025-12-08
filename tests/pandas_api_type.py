from __future__ import annotations

from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd

example_data: pd.DataFrame = pd.DataFrame({
    "integers": [1, 2, 3, None, 5],
    "floats": [1.0, 2.5, None, 4.5, 5.0],
    "nullable_ints": pd.Series([1, None, 3, 4, None], dtype="Int64"),
    "nullable_floats": pd.Series([1.0, None, 3.5, None, 5.0], dtype="Float64"),
    "stings": ["1", "2", "3", None, "5"],
})


numeric_columns: set[str] = set()
for column in example_data.columns:
    if is_numeric_dtype(example_data[column]):
        numeric_columns.add(column)