from __future__ import annotations
from typing import Optional, Sequence


def str_sequence_param(
    str_list: str | Sequence[str] | None,
    as_type: Optional[type] = None
) -> Sequence[str]:
    """
    Quick helper to parse a parameter that can be either a single string or a
    sequence of strings. If `str_list` is a single string, it will be
    converted to a sequence containing that string. The final result will be
    returned as the specified Sequnce-like type (list, set, tuple, etc).
    """
    
    if isinstance(str_list, str):
        str_list = [str_list]
    elif str_list is None:
        str_list = []
    
    if as_type:
        return as_type(str_list)
    return str_list