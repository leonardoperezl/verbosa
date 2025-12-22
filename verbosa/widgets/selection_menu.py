"""Selection menu widget for interactive DataFrame navigation and filtering."""

from typing import Optional, Sequence, Union, List, Any
import logging
import pandas as pd


logger = logging.getLogger(__name__)


class SelectionMenu:
    """Interactive menu for DataFrame selection and filtering.
    
    This class provides an interface to filter, view, and select rows from
    a pandas DataFrame with automatic type conversion and index management.
    
    Attributes:
        data: The filtered DataFrame containing only specified columns with
            string types and ordered RangeIndex.
        _current_view: A working copy of data that reflects filtering operations.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        filter_columns: Optional[Sequence[str]] = None,
        order_by: Optional[str] = None,
        ascending: bool = True
    ) -> None:
        """Initialize the selection menu.
        
        Args:
            data: DataFrame containing the data to display in the menu.
            filter_columns: Columns to use from the DataFrame. If None, uses all.
            order_by: Column name to order the data by. Orders alphabetically.
            ascending: Whether to sort in ascending order.
        
        Raises:
            ValueError: If filter_columns contains non-existent columns.
            ValueError: If order_by column doesn't exist in the filtered data.
        """
        logger.info(f"Initializing SelectionMenu with {len(data)} rows")
        
        # Select columns
        if filter_columns is None:
            self.data = data.copy()
            logger.debug("Using all columns from DataFrame")
        else:
            missing_cols = set(filter_columns) - set(data.columns)
            if missing_cols:
                logger.error(f"Columns not found in DataFrame: {missing_cols}")
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.data = data[list(filter_columns)].copy()
            logger.debug(f"Selected columns: {list(filter_columns)}")
        
        # Convert all columns to string type
        for col in self.data.columns:
            self.data[col] = self.data[col].astype(str)
        logger.debug("All columns converted to string type")
        
        # Reset index to ordered RangeIndex
        self.data = self.data.reset_index(drop=True)
        
        # Apply ordering if specified
        if order_by is not None:
            if order_by not in self.data.columns:
                logger.error(f"Order column '{order_by}' not found in data")
                raise ValueError(f"Order column '{order_by}' not found in data")
            self.data = self.data.sort_values(by=order_by, ascending=ascending)
            self.data = self.data.reset_index(drop=True)
            logger.debug(f"Data ordered by '{order_by}' (ascending={ascending})")
        
        # Create current view
        self._current_view = self.data.copy()
        logger.info(f"SelectionMenu initialized with {len(self.data)} rows and {len(self.data.columns)} columns")
    
    @property
    def current_view(self) -> pd.DataFrame:
        return self._current_view
    
    def filter(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[Any, Sequence[Any]]] = None,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Filter rows in the current view.
        
        Supports multiple filtering modes:
        1. Column-value pairs via kwargs (e.g., filter(name='John', age='30'))
        2. Explicit columns and values lists
        3. Pandas query string
        
        Filtering checks if the value is contained within (substring of) the
        column's string values. Since all columns are strings, this performs
        a substring match.
        
        Args:
            columns: Column name(s) to filter by. Can be a single string or
                sequence of strings.
            values: Value(s) to match. Should correspond to columns. Values
                are checked as substrings within the column values.
            query: A pandas query string for complex filtering.
            **kwargs: Column-value pairs for filtering. Values are checked
                as substrings within the column values.
        
        Returns:
            The filtered DataFrame (_current_view).
        
        Examples:
            >>> menu.filter(name='John')  # Contains 'John'
            >>> menu.filter(age='3')  # Contains '3' (matches '30', '35', '300')
            >>> menu.filter(columns='age', values='30')  # Contains '30'
            >>> menu.filter(columns=['name', 'city'], values=['John', 'NY'])
            >>> menu.filter(query='age == "30" and name == "John"')
        """
        logger.debug(f"Applying filter with columns={columns}, values={values}, kwargs={kwargs}")
        
        # Start with current view
        result = self._current_view.copy()
        initial_rows = len(result)
        
        # Handle query string
        if query is not None:
            logger.debug(f"Applying query filter: {query}")
            result = result.query(query)
        
        # Handle explicit columns and values
        if columns is not None and values is not None:
            if isinstance(columns, str):
                columns = [columns]
                values = [values]
            
            for col, val in zip(columns, values):
                if col not in result.columns:
                    logger.error(f"Column '{col}' not found in data")
                    raise ValueError(f"Column '{col}' not found in data")
                
                # Check if val is a sequence (but not string)
                if isinstance(val, (list, tuple, set)):
                    # Multiple values: check if any substring matches
                    mask = pd.Series([False] * len(result))
                    for v in val:
                        mask |= result[col].str.contains(str(v), case=False, na=False, regex=False)
                    result = result[mask]
                    logger.debug(f"Applied 'contains' filter on '{col}' with {len(val)} values")
                else:
                    # Single value: check if substring is contained
                    result = result[result[col].str.contains(str(val), case=False, na=False, regex=False)]
                    logger.debug(f"Applied 'contains' filter on '{col}' with value '{val}'")
        
        # Handle kwargs
        for col, val in kwargs.items():
            if col not in result.columns:
                logger.error(f"Column '{col}' not found in data")
                raise ValueError(f"Column '{col}' not found in data")
            
            # Check if val is a sequence (but not string)
            if isinstance(val, (list, tuple, set)):
                # Multiple values: check if any substring matches
                mask = pd.Series([False] * len(result))
                for v in val:
                    mask |= result[col].str.contains(str(v), case=False, na=False, regex=False)
                result = result[mask]
                logger.debug(f"Applied 'contains' filter on '{col}' with {len(val)} values")
            else:
                # Single value: check if substring is contained
                result = result[result[col].str.contains(str(val), case=False, na=False, regex=False)]
                logger.debug(f"Applied 'contains' filter on '{col}' with value '{val}'")
        
        # Update current view and ensure ordered index
        self._current_view = result.reset_index(drop=True)
        filtered_rows = len(self._current_view)
        logger.info(f"Filter complete: {initial_rows} rows -> {filtered_rows} rows ({initial_rows - filtered_rows} filtered out)")
        
        return self._current_view
    
    def single_selection(self, index: int) -> Optional[pd.Series]:
        """Select a single row from the current view by index.
        
        Args:
            index: The index of the row to select (0-based).
        
        Returns:
            The selected row as a pandas Series, or None if invalid index.
        """
        if self._current_view.empty:
            logger.warning("Single selection attempted on empty view")
            return None
        
        logger.debug(f"Attempting single selection of row {index} from {len(self._current_view)} rows")
        
        if index < 0 or index >= len(self._current_view):
            logger.warning(f"Invalid index: {index}. Valid range: 0-{len(self._current_view) - 1}")
            return None
        
        logger.info(f"Row {index} selected successfully")
        return self._current_view.iloc[index]
    
    def multiple_selection(self, indices: Sequence[int]) -> Optional[pd.DataFrame]:
        """Select multiple rows from the current view by indices.
        
        Args:
            indices: Sequence of row indices to select (0-based).
        
        Returns:
            DataFrame containing the selected rows, or None if any invalid index.
        """
        if self._current_view.empty:
            logger.warning("Multiple selection attempted on empty view")
            return None
        
        logger.debug(f"Attempting multiple selection of {len(indices)} rows from {len(self._current_view)} rows")
        
        # Validate all indices
        invalid = [idx for idx in indices if idx < 0 or idx >= len(self._current_view)]
        if invalid:
            logger.warning(f"Invalid indices: {invalid}. Valid range: 0-{len(self._current_view) - 1}")
            return None
        
        logger.info(f"{len(indices)} rows selected successfully: {list(indices)}")
        return self._current_view.iloc[list(indices)].reset_index(drop=True)
    
    def reset_view(self) -> pd.DataFrame:
        """Reset the current view to the original data.
        
        Returns:
            The reset current view.
        """
        logger.debug("Resetting view to original data")
        self._current_view = self.data.copy()
        logger.info(f"View reset complete: {len(self._current_view)} rows")
        return self._current_view
    
    
    
    def get_column_values(self, column: str) -> List[str]:
        """Get unique values from a column in the current view.
        
        Args:
            column: Name of the column.
        
        Returns:
            List of unique values sorted alphabetically.
        
        Raises:
            ValueError: If column doesn't exist.
        """
        if column not in self._current_view.columns:
            logger.error(f"Column '{column}' not found in data")
            raise ValueError(f"Column '{column}' not found in data")
        
        unique_vals = self._current_view[column].unique()
        logger.debug(f"Retrieved {len(unique_vals)} unique values from column '{column}'")
        
        # Sort alphabetically (all values are strings)
        return sorted(unique_vals)
    
    def search(self, term: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Search for a term across specified columns (or all columns).
        
        Args:
            term: Search term (case-insensitive).
            columns: Columns to search in. If None, searches all columns.
        
        Returns:
            Filtered DataFrame containing matching rows.
        """
        logger.debug(f"Searching for term '{term}' in columns {columns}")
        initial_rows = len(self._current_view)
        
        if columns is None:
            # Search in all columns (all are strings)
            columns = list(self._current_view.columns)
            logger.debug(f"No columns specified, searching all {len(columns)} columns")
        
        if not columns:
            logger.warning("No columns available for search")
            return self._current_view
        
        mask = pd.Series([False] * len(self._current_view))
        for col in columns:
            if col in self._current_view.columns:
                mask |= self._current_view[col].str.contains(
                    term, case=False, na=False
                )
        
        self._current_view = self._current_view[mask].reset_index(drop=True)
        filtered_rows = len(self._current_view)
        logger.info(f"Search complete: {initial_rows} rows -> {filtered_rows} rows (found {filtered_rows} matches for '{term}')")
        
        return self._current_view
    
    def __len__(self) -> int:
        """Return the number of rows in the current view."""
        return len(self._current_view)
    
    def __repr__(self) -> str:
        """Return a string representation of the selection menu."""
        return (
            f"SelectionMenu(rows={len(self.data)}, "
            f"columns={list(self.data.columns)}, "
            f"current_view_rows={len(self._current_view)})"
        )
