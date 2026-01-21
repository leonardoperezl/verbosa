"""
Selection widget for filtering a dataframe
"""

from __future__ import annotations
from typing import Any, Sequence
import logging


import pandas as pd


logger = logging.getLogger(__name__)


class SelectionMenu:
    """
    A dataframe used as a menu, mainly because one of the rows it contains
    needs to be selected by a user or process.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def search(
        self,
        value: Any,
        at: Sequence[str] | str = "all",
        *,
        case_sensitive: bool = False,
        whole_match: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Get all rows that have the searched value at one of the provided
        columns. Provided columns to search 
        
        Parameters
        ----------
        value : str
            The value to search for. If the value is not a string, it will be
            casted to one.
        
        at: Sequence[str] or str
            The names of the columns to search in
        
        case_sensitive: bool, default False
            If the search should treat upper and lower values the same.
        
        whole_match: bool, default False
            Get all rows which values match from start to finish the provided
            value. If False, row values that contains part of the provided
            value will be considered a proper match.
        
        Returns
        -------
        pd.DataFrame or pd.Series
            All rows considered matches.
        """
        
        # 1) Parameter checks and normalizations
        if value is None:
            return self.data
        
        if not isinstance(value, str):
            value: str = str(value)
        
        columns: list[str] = list(at)
        if at != "all" and isinstance(at, str):
            columns = [at]
        
        if at == "all":
            columns = self.data.columns.tolist()
        
        # 2) Cast all search columns to string
        for column in columns.copy():
            if column not in self.data.columns:
                columns.remove(column)
                continue
            
            try:
                s = self.data[column].astype("string")
            except Exception as e:
                columns.remove(column)
                continue
            
            self.data[column] = s
        
        # 3) Get a boolean mask composed of each column match evaluation
        matches_mask: pd.Series = pd.Series(False, index=self.data.index)
        
        if not whole_match:
            for column in columns:
                s = self.data[column]
                matches_mask = (
                    matches_mask |
                    s.str.contains(value, case=case_sensitive, regex=False)
                )
            
            return self.data.loc[matches_mask, :]
        
        for column in columns:
            s = self.data[column]
            if case_sensitive:
                s = s.str.lower()
                value = value.lower()
            
            matches_mask = matches_mask | s.eq(value)
        
        return self.data.loc[matches_mask, :]