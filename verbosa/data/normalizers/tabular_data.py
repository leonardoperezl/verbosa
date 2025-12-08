from __future__ import annotations
from typing import Any, Literal, Optional, Sequence
import logging
import re


from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype
)
import pandas as pd


from verbosa.utils.global_typings import Pathlike
from verbosa.interfaces.normalizer import NormalizerInterface
from verbosa.interfaces.columns_config import ColumnsConfig


logger = logging.getLogger(__name__)



TEXT_TYPES = {"string", "category"}
NUMERIC_TYPES = {"Int32", "Int64", "Float32", "Float64"}
DATE_TYPES = {"datetime64[ns]"}


_RE_COMBINING_MARKS: re.Pattern = re.compile(r"[\u0300-\u036f]+")
_RE_NON_ASCII: re.Pattern = re.compile(r"[^\x00-\x7F]+")



class TabularDataNormalizer(NormalizerInterface):
    data: pd.DataFrame
    
    def __init__(
        self,
        data: pd.DataFrame,
        columns_config_path: Pathlike = None
    ) -> None:
        super().__init__(data, columns_config_path)
        self.columns_config: ColumnsConfig = (
            ColumnsConfig.from_yaml(self.autonorm_settings)
        )
    
    # ------------------------- Non-public methods ------------------------- #
    def _adjust_column_labels(self) -> None:
        """
        Encapsulates the logic to normalize column names based on the
        columns configuration. Only usable at `_autonorm_implementation()`.
        """
        
        original_columns = self.data.columns.tolist()
        logger.debug(
            f"Adjusting column labels based on configuration.\n"
            f"- Original columns: {original_columns}\n"
            f"- Configuration columns: "
            f"{[col_config.name for col_config in self.columns_config]}"
        )
        
        # 1) Rename columns based on configuration
        new_column_names = []
        for col in self.data.columns:
            col_config = self.columns_config[col]
            if col_config is None:
                # If no configuration found, keep original name
                new_column_names.append(col)
                continue
            # Else, rename to the configured name
            new_column_names.append(col_config.name)
        
        self.data.columns = new_column_names
        logger.debug(
            f"Renamed columns: {self.data.columns.tolist()}"
        )
        
        # 2) Sort columns based on configuration order
        config_columns = [
            col_config.name for col_config in self.columns_config
            if col_config.name in self.data.columns
        ]
        not_in_config = [
            col for col in self.data.columns
            if col not in config_columns
        ]
        new_order = config_columns + not_in_config
        self.data = self.data.loc[:, new_order]
        logger.debug(
            f"Reordered columns: {self.data.columns.tolist()}"
        )
    
    def _apply_config_fill_na(self) -> None:
        """
        """
        logger.debug("Applying fill_na values from column configuration")
        
        for col_config in self.columns_config:
            col_name = col_config.name
            
            # Skip if column not in data or no fill_na specified
            is_col_missing = col_name not in self.data.columns
            is_fill_na_missing = col_config.fill_na is None
            if is_col_missing or is_fill_na_missing:
                continue
            
            fill_na = col_config.fill_na
            col_dtype = self.data[col_name].dtype
            
            # Type validation and coercion
            try:
                # For numeric columns, ensure fill_value is numeric
                if is_numeric_dtype(col_dtype):
                    fill_na = pd.to_numeric(fill_na)
                    logger.debug(
                        f"Column '{col_name}': coerced fill_na value to "
                        f"numeric: {fill_na}"
                    )
                
                # For datetime columns, ensure fill_value is Timestamp
                elif is_datetime64_any_dtype(col_dtype):
                    fill_na = pd.Timestamp(fill_na)
                    logger.debug(
                        f"Column '{col_name}': coerced fill_na value to "
                        f"Timestamp: {fill_na}"
                    )
                
                # For string/categorical, convert to string
                elif col_dtype.name == "string":
                    fill_na = str(fill_na)
                    logger.debug(
                        f"Column '{col_name}': coerced fill_na value to "
                        f"string: {fill_na}"
                    )
                
                # For categorical, ensure fill_na is in categories
                elif col_dtype.name == "category":
                    fill_na = str(fill_na)
                    # Add fill_na as a category if not already present
                    if fill_na not in self.data[col_name].cat.categories:
                        self.data[col_name] = (
                            self.data[col_name].cat.add_categories([fill_na])
                        )
                        logger.debug(
                            f"Column '{col_name}': added fill_na value "
                            f"'{fill_na}' to categories"
                        )
                    else:
                        logger.debug(
                            f"Column '{col_name}': fill_na value '{fill_na}' "
                            f"already in categories"
                        )
                
                # Apply fill
                na_count_before = self.data[col_name].isna().sum()
                self.data[col_name] = self.data[col_name].fillna(fill_na)
                na_count_after = self.data[col_name].isna().sum()
                
                logger.info(
                    f"Filled {na_count_before - na_count_after} NA values "
                    f"in column '{col_name}' with config value: {fill_na}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to apply fill_na for column '{col_name}': {e}. "
                    f"Skipping column."
                )
    
    def _autonorm_implementation(self) -> pd.DataFrame:
        # 1) Normalize column names
        self._adjust_column_labels()
        
        # 2) Group columns by normalization method
        grouping_by_norm: dict[str, list[str]] = {}
        for col_config in self.columns_config:
            col_name = col_config.name
            if col_name not in self.data.columns:
                logger.warning(
                    f"Column '{col_name}' not found in data. Skipping."
                )
                continue
            
            col_normalization = col_config.normalization
            if col_normalization is None:
                logger.info(
                    f"No normalization method specified for column "
                    f"'{col_name}'. Skipping."
                )
                continue
            
            grouping_by_norm.setdefault(
                col_normalization, []
            ).append(col_name)
        
        # 3) Apply normalization methods
        for norm_method, columns in grouping_by_norm.items():
            if not hasattr(self, norm_method):
                logger.warning(
                    f"Normalization method '{norm_method}' not implemented. "
                    f"Skipping columns: {columns}"
                )
                continue
            
            norm_func = getattr(self, norm_method)
            self.data = norm_func(columns)
        
        # 4) Apply fill_na from configuration
        self._apply_config_fill_na()
        
        return self.data
    
    # ----------------- API. General normalization methods ----------------- #
    def text(
        self,
        columns: Sequence[str],
        strip: Optional[Literal["both", "left", "right"]] = None,
        compact_whitespace: Optional[Any] = None,
        case: Optional[Literal["lower", "upper", "title"]] = None,
        empty_to_na: bool = False,
        delete_diacritics: bool = False,
        delete_non_ascii: bool = False
    ) -> pd.DataFrame:
        logger.info(
            f"Normalizing text columns: {columns} with options:\n"
            f"- strip: {strip}\n"
            f"- compact_whitespace: {compact_whitespace}\n"
            f"- case: {case}\n"
            f"- empty_to_na: {empty_to_na}\n"
            f"- delete_diacritics: {delete_diacritics}\n"
            f"- delete_non_ascii: {delete_non_ascii}"
        )
        
        for column in columns:
            s = self.data[column].astype("string")
            
            if strip is not None:
                if strip == "both": s = s.str.strip()
                elif strip == "left": s = s.str.lstrip()
                elif strip == "right": s = s.str.rstrip()
                else: raise ValueError(f"Invalid strip option: {strip}")
            
            if compact_whitespace is not None:
                s = s.replace(r"\s{2,}", compact_whitespace, regex=True)
            
            if case is not None:
                if case == "lower": s = s.str.lower()
                elif case == "upper": s = s.str.upper()
                elif case == "title": s = s.str.title()
                else: raise ValueError(f"Invalid case option: {case}")
            
            if empty_to_na:
                s = s.replace(r"^\s*$", pd.NA, regex=True)
            
            if delete_diacritics:
                s = (
                    s.str.normalize("NFD")
                    .str.replace(_RE_COMBINING_MARKS, "", regex=True)
                )
            
            if delete_non_ascii:
                s = s.str.replace(_RE_NON_ASCII, "", regex=True)
            
            self.data[column] = s
            logger.debug(f"Succesfully normalized text column: {column}")
        
        logger.info("Completed text normalization.")
        return self.data
    
    def numeric(
        self,
        columns: Sequence[str],
        dtype: Optional[Literal["Int64", "Float64"]] = "Float64",
        fill_na: Optional[float | int] = None,
        cleanup_pattern: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info(
            f"Normalizing numeric columns: {columns} with options:\n"
            f"- dtype: {dtype}\n"
            f"- fill_na: {fill_na}\n"
            f"- cleanup_pattern: {cleanup_pattern}"
        )
        
        # 1) Separate numeric and non-numeric columns (preserve order)
        numeric: list[str] = []
        non_numeric: list[str] = []
        for col in columns:
            if is_numeric_dtype(self.data[col]): numeric.append(col)
            else: non_numeric.append(col)
        
        logger.debug(
            f"Detected:\n"
            f"- Numeric columns: {numeric}.\n"
            f"- Non-numeric columns: {non_numeric}"
        )
        
        # 2) Convert all numeric columns to desired dtype (vectorized)
        if numeric:
            self.data[numeric] = self.data[numeric].astype(dtype)
        
        # 3) For non-numeric columns, attempt conversion to numeric
        # Compile cleanup pattern once if provided
        cleanup_re = (
            re.compile(cleanup_pattern)
            if cleanup_pattern is not None else None
        )
        
        for col in non_numeric:
            s = self.data[col].astype("string")
            if cleanup_re is not None:
                s = s.str.replace(cleanup_re, "", regex=True)
            
            try:
                self.data[col] = s.astype(dtype, errors="raise")
                numeric.append(col)
            except Exception as e:
                logger.warning(
                    f"Failed to convert column '{col}' to numeric type "
                    f"'{dtype}'. Skipping column, try to improve the "
                    f"cleanup_pattern for better parsing. Error: {e}"
                )
            else:
                logger.debug(
                    f"Successfully converted column '{col}' to {dtype}."
                )
        
        # 4) Fill NA values if specified (vectorized)
        if fill_na is not None and numeric:
            self.data[numeric] = self.data[numeric].fillna(fill_na)
        
        logger.info("Completed numeric normalization.")
        return self.data
    
    def date(
        self,
        columns: Sequence[str],
        formats: Optional[Sequence[str] | str] = None,
        cleanup_pattern: Optional[str] = None,
        fill_na: Optional[pd.Timestamp] = None,
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: bool = False
    ) -> pd.DataFrame:
        """
        """
        logger.info(
            f"Normalizing date columns: {columns} with options:\n"
            f"- formats: {formats}\n"
            f"- cleanup_pattern: {cleanup_pattern}\n"
            f"- fill_na: {fill_na}\n"
            f"- dayfirst: {dayfirst}, yearfirst: {yearfirst}, utc: {utc}"
        )
        
        # Normalize formats to list
        if formats is None:
            formats_list = []
        elif isinstance(formats, str):
            formats_list = [formats]
        else:
            formats_list = list(formats)
        
        for column in columns:
            # Check if already datetime
            if is_datetime64_any_dtype(self.data[column]):
                logger.debug(
                    f"Column '{column}' is already datetime, skipping "
                    f"conversion"
                )
                if utc and self.data[column].dt.tz is None:
                    self.data[column] = (
                        self.data[column].dt.tz_localize('UTC')
                    )
                elif utc:
                    self.data[column] = (
                        self.data[column].dt.tz_convert('UTC')
                    )
                continue
            
            # Convert to string for processing
            s = self.data[column].astype("string")
            original_na_count = s.isna().sum()
            
            # Apply cleanup pattern if provided
            if cleanup_pattern is not None:
                cleanup_re: re.Pattern = re.compile(cleanup_pattern)
                s = s.str.replace(cleanup_re, "", regex=True)
                logger.debug(f"Applied cleanup pattern to column '{column}'")
            
            best_series = None
            best_na_count = len(s) + 1  # Worst case: all NAs
            best_format = None
            
            # Try parsing without explicit format first (pandas inference)
            if not formats_list:
                try:
                    parsed = pd.to_datetime(
                        s,
                        errors='coerce',
                        dayfirst=dayfirst,
                        yearfirst=yearfirst,
                        utc=utc
                    )
                    na_count = parsed.isna().sum()
                    logger.debug(
                        f"Column '{column}': inferred parsing produced "
                        f"{na_count} NAs"
                    )
                    if na_count < best_na_count:
                        best_series = parsed
                        best_na_count = na_count
                        best_format = "inferred"
                except Exception as e:
                    logger.warning(
                        f"Failed to parse column '{column}' with inferred "
                        f"format: {e}"
                    )
            
            # Try each explicit format
            for fmt in formats_list:
                try:
                    parsed = pd.to_datetime(
                        s,
                        format=fmt,
                        errors='coerce',
                        utc=utc
                    )
                    na_count = parsed.isna().sum()
                    logger.debug(
                        f"Column '{column}': format '{fmt}' produced "
                        f"{na_count} NAs"
                    )
                    
                    if na_count < best_na_count:
                        best_series = parsed
                        best_na_count = na_count
                        best_format = fmt
                except Exception as e:
                    logger.warning(
                        f"Failed to parse column '{column}' with format "
                        f"'{fmt}': {e}"
                    )
                    continue
            
            # Check if we successfully parsed anything
            if best_series is None:
                logger.error(
                    f"Failed to parse column '{column}' to datetime with any "
                    f"method. Skipping column."
                )
                continue
            
            # Warn if we created more NAs than original
            new_na_count = best_na_count
            if new_na_count > original_na_count:
                logger.warning(
                    f"Column '{column}': parsing created "
                    f"{new_na_count - original_na_count} additional NA  "
                    f"values (original: {original_na_count}, after parsing: "
                    f"{new_na_count}). Best format: {best_format}"
                )
            else:
                logger.debug(
                    f"Successfully parsed column '{column}' using format: "
                    f"{best_format}"
                )
            
            self.data[column] = best_series
            
            # Fill NA values if specified
            if fill_na is not None:
                self.data[column] = self.data[column].fillna(fill_na)
                logger.debug(
                    f"Filled NA values in column '{column}' with {fill_na}"
                )
        
        logger.info("Completed date normalization.")
        return self.data
    
    def categorical(
        self,
        columns: Sequence[str],
        strip: Optional[Literal["both", "left", "right"]] = None,
        compact_whitespace: Optional[Any] = None,
        case: Optional[Literal["lower", "upper", "title"]] = None,
        empty_to_na: bool = False,
        delete_diacritics: bool = False,
        delete_non_ascii: bool = False,
        ordered: bool = False,
        sort_categories: bool = False
    ) -> pd.DataFrame:
        """
        """
        logger.info(
            f"Normalizing categorical columns: {columns} with options:\n"
            f"- strip: {strip}, case: {case}\n"
            f"- compact_whitespace: {compact_whitespace}\n"
            f"- empty_to_na: {empty_to_na}\n"
            f"- delete_diacritics: {delete_diacritics}\n"
            f"- delete_non_ascii: {delete_non_ascii}\n"
            f"- ordered: {ordered}, sort_categories: {sort_categories}"
        )
        
        # 1) Apply text normalization first using the text method
        self.data = self.text(
            columns=columns,
            strip=strip,
            compact_whitespace=compact_whitespace,
            case=case,
            empty_to_na=empty_to_na,
            delete_diacritics=delete_diacritics,
            delete_non_ascii=delete_non_ascii
        )
        
        # 2) Convert each column to categorical with inferred categories
        for column in columns:
            # Get unique values (categories) from the column
            unique_values = self.data[column].dropna().unique()
            
            # Sort categories if requested
            if sort_categories:
                try:
                    categories = sorted(unique_values)
                except TypeError:
                    # If sorting fails (mixed types), keep original order
                    logger.warning(
                        f"Cannot sort categories for column '{column}' "
                        f"(mixed types). Using natural order."
                    )
                    categories = unique_values
            else:
                categories = unique_values
            
            # Convert to categorical
            self.data[column] = pd.Categorical(
                self.data[column],
                categories=categories,
                ordered=ordered
            )
            
            tmp = '(ordered)' if ordered else ''
            has_na = self.data[column].isna().any()
            logger.debug(
                f"Successfully normalized categorical column '{column}' "
                f"{tmp} with {len(categories)} categories "
                f"(has_na: {has_na})"
            )
        
        logger.info("Completed categorical normalization.")
        return self.data
    
    # -------------------- Specific normalization methods ------------------ #
    def text_stressed(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.text(
            columns=columns,
            strip="both",
            compact_whitespace=" ",
            case="upper",
            empty_to_na=True,
            delete_diacritics=True,
            delete_non_ascii=True
        )
    
    def numeric_float(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.numeric(
            columns=columns,
            dtype="Float64",
            cleanup_pattern=r"[$\s,%]"
        )
    
    def numeric_int(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.numeric(
            columns=columns,
            dtype="Int64",
            cleanup_pattern=r"[$\s,%]"
        )
    
    def date_dayfirst(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.date(
            columns=columns,
            formats=[
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y"
            ],
            dayfirst=True,
            yearfirst=False
        )
    
    def date_yearfirst(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.date(
            columns=columns,
            formats=[
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y"
            ],
            dayfirst=False,
            yearfirst=True
        )
    
    def categorical_relaxed(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.categorical(
            columns=columns,
            strip="both",
            compact_whitespace=" ",
            case="upper",
            empty_to_na=True,
            delete_diacritics=True,
            delete_non_ascii=True,
            ordered=False,
            sort_categories=True
        )