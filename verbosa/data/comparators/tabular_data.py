from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd


class TabularDataComparator:
    def __init__(
        self,
        right: pd.DataFrame,
        sort_columns: Sequence[str],
        diff_columns: Sequence[str],
        visual_column: Optional[str] = None,
        right_label: str = "right",
        left_label: str = "left",
    ):
        # Attributes only (no business logic here).
        self.right = right
        self.sort_columns = list(sort_columns)
        self.diff_columns = list(diff_columns)
        self.visual_column = visual_column
        self.right_label = str(right_label)
        self.left_label = str(left_label)
        
        self._validate_initial_state()
    
    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_initial_state(self) -> None:
        self._validate_sort_columns(self.right, df_name="right")
        self._validate_diff_columns(self.right, df_name="right")
        self._validate_visual_column(self.right, df_name="right")
        self._validate_no_overlap_sort_diff()
    
    def _validate_sort_columns(
        self,
        df: pd.DataFrame,
        *,
        df_name: str,
    ) -> None:
        if not self.sort_columns:
            raise ValueError("sort_columns cannot be empty")
        
        missing = [c for c in self.sort_columns if c not in df.columns]
        if missing:
            msg = f"sort_columns not found in {df_name}: {missing}"
            raise ValueError(msg)
    
    def _validate_diff_columns(
        self,
        df: pd.DataFrame,
        *,
        df_name: str,
    ) -> None:
        missing = [c for c in self.diff_columns if c not in df.columns]
        if missing:
            msg = f"diff_columns not found in {df_name}: {missing}"
            raise ValueError(msg)
    
    def _validate_visual_column(
        self,
        df: pd.DataFrame,
        *,
        df_name: str,
    ) -> None:
        if self.visual_column is None:
            return
        if self.visual_column not in df.columns:
            msg = f"visual_column '{self.visual_column}' not in {df_name}"
            raise ValueError(msg)
    
    def _validate_no_overlap_sort_diff(self) -> None:
        overlap = set(self.sort_columns) & set(self.diff_columns)
        if overlap:
            msg = (
                "diff_columns cannot intersect sort_columns: "
                f"{sorted(overlap)}"
            )
            raise ValueError(msg)
    
    # ------------------------------------------------------------------
    # DataFrame preparation helpers
    # ------------------------------------------------------------------
    def _prepare(
        self,
        df: pd.DataFrame,
        *,
        name: str,
    ) -> pd.DataFrame:
        self._validate_sort_columns(df, df_name=name)
        self._validate_diff_columns(df, df_name=name)
        self._validate_visual_column(df, df_name=name)
        
        return (
            df.sort_values(by=self.sort_columns, kind="mergesort")
            .reset_index(drop=True)
        )
    
    def _suffix(self, col: str, label: str) -> str:
        return f"{col}_{label}"
    
    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------
    def _equal_mask_with_nan(
        self,
        a: pd.Series,
        b: pd.Series,
    ) -> pd.Series:
        return a.eq(b) | (a.isna() & b.isna())
    
    def _changed_mask_for_column(
        self,
        right_df: pd.DataFrame,
        left_df: pd.DataFrame,
        col: str,
    ) -> pd.Series:
        return ~self._equal_mask_with_nan(
            right_df[col],
            left_df[col],
        )
    
    def _changed_diff_columns(
        self,
        right_df: pd.DataFrame,
        left_df: pd.DataFrame,
    ) -> List[str]:
        changed: List[str] = []
        for col in self.diff_columns:
            if self._changed_mask_for_column(
                right_df,
                left_df,
                col,
            ).any():
                changed.append(col)
        return changed
    
    def _changed_rows_mask(
        self,
        right_df: pd.DataFrame,
        left_df: pd.DataFrame,
        cols: Sequence[str],
    ) -> pd.Series:
        if not cols:
            return pd.Series(False, index=right_df.index)
        
        mask = self._changed_mask_for_column(
            right_df,
            left_df,
            cols[0],
        )
        for col in cols[1:]:
            mask = mask | self._changed_mask_for_column(
                right_df,
                left_df,
                col,
            )
        return mask
    
    def _build_result_dataframe(
        self,
        right_df: pd.DataFrame,
        left_df: pd.DataFrame,
        diff_only: bool,
        diff_cols: Sequence[str],
    ) -> pd.DataFrame:
        series_list: List[pd.Series] = []
        
        # First column: visual helper or reset index.
        if self.visual_column is not None:
            vis = right_df[self.visual_column].combine_first(
                left_df[self.visual_column]
            )
            series_list.append(vis.rename(self.visual_column))
        else:
            series_list.append(
                pd.Series(right_df.index, name="index")
            )
        
        # Interleave only changed diff columns.
        for col in diff_cols:
            r_name = self._suffix(col, self.right_label)
            l_name = self._suffix(col, self.left_label)
            series_list.append(right_df[col].rename(r_name))
            series_list.append(left_df[col].rename(l_name))
        
        # Optional context columns from right.
        if not diff_only:
            excluded = set(diff_cols)
            if self.visual_column is not None:
                excluded.add(self.visual_column)
            for col in self.right.columns:
                if col not in excluded:
                    r_name = self._suffix(col, self.right_label)
                    series_list.append(right_df[col].rename(r_name))
        
        return pd.concat(series_list, axis=1)
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compare(
        self,
        left: pd.DataFrame,
        *,
        diff_only: bool = True,
    ) -> Union[str, pd.DataFrame]:
        right_s = self._prepare(self.right, name="right")
        left_s = self._prepare(left, name="left")
        
        flags: List[str] = []
        
        if len(left_s) > len(right_s):
            flags.append("rows_added")
        elif len(left_s) < len(right_s):
            flags.append("rows_removed")
        
        if left_s.shape[1] > right_s.shape[1]:
            flags.append("cols_added")
        elif left_s.shape[1] < right_s.shape[1]:
            flags.append("cols_removed")
        
        if flags:
            return ", ".join(flags)
        
        changed_cols = self._changed_diff_columns(right_s, left_s)
        if not changed_cols:
            return "equal"
        
        changed_rows = self._changed_rows_mask(
            right_s,
            left_s,
            changed_cols,
        )
        
        right_c = right_s.loc[changed_rows]
        left_c = left_s.loc[changed_rows]
        
        return self._build_result_dataframe(
            right_c,
            left_c,
            diff_only=diff_only,
            diff_cols=changed_cols,
        )
