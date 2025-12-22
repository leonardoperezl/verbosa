# Python standard library imports
from __future__ import annotations
from typing import Any, Callable, Literal, Optional, Sequence, TypeAlias
import logging
import re

# Third-party imports
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)
import pandas as pd

# Library imports
from verbosa.utils.typings import Pathlike
from verbosa.interfaces.normalizer import NormalizerInterface
from verbosa.interfaces.columns_config import ColumnsConfig



##############################################################################
#                            LOGGER CONFIGURATION                            #
##############################################################################

logger = logging.getLogger(__name__)



##############################################################################
#                               CUSTOM TYPINGS                               #
##############################################################################

NAValues: TypeAlias = Optional[Sequence[str | re.Pattern | int | float]]

NormalizedStringDType: TypeAlias = Literal["string"]
NormalizedNumericDType: TypeAlias = Literal["Int64", "Float64"]
NormalizedDateDType: TypeAlias = (
    Literal["datetime64[ns]", "datetime64[ns, UTC]"]
)
NormalizedCategoricalDType: TypeAlias = Literal["category"]
NormalizedBooleanDType: TypeAlias = Literal["boolean"]

NormalizedDType: TypeAlias = (
    NormalizedStringDType | 
    NormalizedNumericDType |
    NormalizedDateDType |
    NormalizedCategoricalDType |
    NormalizedBooleanDType
)



##############################################################################
#                              MODULE CONSTANTS                              #
##############################################################################

_RE_COMBINING_MARKS: re.Pattern = re.compile(r"[\u0300-\u036f]+")
_RE_NON_ASCII: re.Pattern = re.compile(r"[^\x00-\x7F]+")



##############################################################################
#                            MAIN CLASS DEFINITION                           #
##############################################################################

class TabularDataNormalizer(NormalizerInterface[pd.DataFrame]):
    """
    Tabular data normalizer implementation for pandas DataFrames.
    
    This class provides comprehensive normalization capabilities for tabular data,
    including text, numeric, date, categorical, and boolean data type normalization.
    It uses a YAML configuration file to define normalization rules and parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be normalized
    columns_config_path : Pathlike, optional
        Path to the YAML configuration file containing normalization rules
        
    Attributes
    ----------
    data : pd.DataFrame
        The DataFrame being normalized
    columns_config : ColumnsConfig
        Loaded configuration object containing normalization specifications
    """
    
    data: pd.DataFrame
    
    def __init__(
        self,
        data: pd.DataFrame,
        columns_config_path: Optional[Pathlike] = None
    ) -> None:
        super().__init__(data, columns_config_path)
        if self.autonorm_settings is not None:
            self.columns_config: ColumnsConfig = (
                ColumnsConfig.from_yaml(self.autonorm_settings)
            )
        else:
            self.columns_config = None
    
    # ------------------------- Non-public methods ------------------------- #
    def _autonorm_implementation(self) -> pd.DataFrame:
        """
        Apply normalization methods defined in the columns configuration.
        
        This method reads the normalization settings from `self.columns_config`
        and applies the specified normalization methods to the corresponding
        columns in the dataframe. It processes columns in the order defined
        in the configuration file.
        
        Returns
        -------
        pd.DataFrame
            The normalized dataframe.
        
        Raises
        ------
        ValueError
            If no columns configuration is available
        
        Notes
        -----
        - Columns not present in the dataframe will be skipped with a warning
        - If a column has no normalization defined, it will be skipped
        - Invalid normalization methods will raise an AttributeError
        """
        if self.columns_config is None:
            raise ValueError(
                "No columns configuration available. Cannot perform automatic "
                "normalization without configuration."
            )
        
        logger.info(
            f"Beginning automatic normalization using configuration: "
            f"{self.columns_config.name}"
        )
        
        # 1) Sort the columns according to the order in the configuration
        self._sort_columns_as_config()
        
        # 2) Convert defined NA values to pd.NA before normalization
        self._convert_na_values()
        
        # 3) Apply normalization methods group by group
        self._apply_norm_methods()
        
        # 4) Convert defined NA values to pd.NA
        self._convert_na_values()
        
        # 5) Fill NA values as defined in the configuration
        self._fill_na_values()
        
        logger.info("Ended automatic normalization process successfully.")
        return self.data
    
    def _sort_columns_as_config(self) -> None:
        logger.info(
            "Beggining column sorting as in the configuration provided."
        )
        
        not_in_config = [
            col for col in self.data.columns
            if col not in self.columns_config
        ]
        new_order = list(self.columns_config) + list(not_in_config)
        self.data = self.data.loc[:, new_order]
        
        logger.info(
            "Ended column sorting according to configuration."
        )
    
    def _convert_na_values(self) -> None:
        logger.info("Beginning conversion of defined NA values to pd.NA.")
        
        na_values_dict = self.columns_config.get_na_values_dict()
        self.convert_to_na(column_na_values=na_values_dict)
        
        logger.info("Ended conversion of defined NA values to pd.NA.")
    
    def _apply_norm_methods(self) -> None:
        logger.info(
            "Beginning application of normalization methods to column groups."
        )
        
        norm_groups = self.columns_config.group_by_normalization()
        for spec, columns in norm_groups:
            method_name = spec.method_name
            if not hasattr(self, method_name):
                logger.warning(
                    f"Normalization method '{method_name}' not found in "
                    f"TabularDataNormalizer. Skipping columns: {columns}"
                )
                continue
            
            method_params = spec.params_to_dict()
            method_params.update({"columns": columns})
            method: Callable = getattr(self, method_name)
            logger.debug(
                f"Applying normalization method '{method_name}' to columns: "
                f"{columns} with parameters: {method_params}"
            )
            self.data = method(**method_params)
        
        logger.info("Completed applying normalization methods to column groups.")
    
    def _fill_na_values(self) -> None:
        logger.info("Filling defined NA values in columns.")
        
        fill_values_dict = (
            self.columns_config.get_columns_fill_na_dict()
        )
        self.fill_na(column_fill_values=fill_values_dict)
        
        logger.info("Completed filling defined NA values in columns.")
    
    # ----------------- API. General normalization methods ----------------- #
    def text(
        self,
        columns: Sequence[str],
        strip: Optional[Literal["both", "left", "right"]] = None,
        compact_whitespace: Optional[Any] = None,
        case: Optional[Literal["lower", "upper", "title"]] = None,
        empty_to_na: bool = False,
        delete_diacritics: bool = False,
        delete_non_ascii: bool = False,
        cleanup_pattern: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info(
            f"Normalizing text columns: {columns} with options:\n"
            f"- strip: {strip}\n"
            f"- compact_whitespace: {compact_whitespace}\n"
            f"- case: {case}\n"
            f"- empty_to_na: {empty_to_na}\n"
            f"- delete_diacritics: {delete_diacritics}\n"
            f"- delete_non_ascii: {delete_non_ascii}\n"
            f"- cleanup_pattern: {cleanup_pattern}"
        )
        
        # Compile cleanup pattern once if provided
        cleanup_re = (
            re.compile(cleanup_pattern)
            if cleanup_pattern is not None else None
        )
        
        for column in columns:
            s = self.data[column].astype("string")
            
            if cleanup_re is not None:
                s = s.str.replace(cleanup_re, "", regex=True)
            
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
        cleanup_pattern: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info(
            f"Normalizing numeric columns: {columns} with options:\n"
            f"- dtype: {dtype}\n"
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
        
        logger.info("Completed numeric normalization.")
        return self.data
    
    def date(
        self,
        columns: Sequence[str],
        formats: Optional[Sequence[str] | str] = None,
        cleanup_pattern: Optional[str] = None,
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
        cleanup_pattern: Optional[str] = None,
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
            f"- cleanup_pattern: {cleanup_pattern}\n"
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
            delete_non_ascii=delete_non_ascii,
            cleanup_pattern=cleanup_pattern
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
    
    def boolean(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        logger.info(f"Normalizing boolean columns: {columns}")
        
        for column in columns:
            s = self.data[column].astype("boolean", errors="ignore")
            self.data[column] = s
        
        logger.info("Completed boolean normalization.")
        return self.data
    
    # --------------------------- API. NA methods -------------------------- #
    def convert_to_na(
        self,
        column_na_values: dict[str, NAValues]
    ) -> None:
        logger.info(
            "Converting provided na values in columns: "
            f"{', '.join(column_na_values.keys())}"
        )
        
        for column, na_values in column_na_values.items():
            if column not in self.data.columns:
                logger.warning(
                    f"The column '{column}' is not present in the data. "
                    f"Skipping."
                )
                continue
            
            if na_values is None:
                logger.debug(
                    f"Column '{column}': no NA values provided, skipping."
                )
                continue
            
            # Convert na_values to tuple for consistency
            if isinstance(na_values, (str, re.Pattern)):
                na_values = (na_values,)
            elif not isinstance(na_values, Sequence):
                na_values = (na_values,)
            na_values = tuple(na_values)
            
            s = self.data[column]
            is_cat = isinstance(s.dtype, pd.CategoricalDtype)
            
            literal_nas = [
                na for na in na_values if not isinstance(na, re.Pattern)
            ]
            pattern_nas = [
                na for na in na_values if isinstance(na, re.Pattern)
            ]
            
            # 
            if is_cat and literal_nas:
                categories = s.dtype.categories
                # Keep only those literal NA markers that actually exist as
                # categories
                literal_nas = [
                    v for v in literal_nas if v in categories
                ]
            
            # Replace literal na values
            if literal_nas:
                na_count = int(s.isin(literal_nas).sum())
                mask = s.isin(literal_nas)
                s = s.mask(mask, pd.NA)
                logger.debug(
                    f"Column '{column}': replaced {na_count} identified NA "
                    f"values ({literal_nas})"
                )
            
            # Pattern-based replacement
            if pattern_nas:
                # For categoricals, operate on string view, then mask into c
                s = s.astype("string")
                for pattern in pattern_nas:
                    matches = s.str.match(pattern).fillna(False)
                    na_count = int(matches.sum())
                    s = s.mask(matches, pd.NA)
                    logger.debug(
                        f"Column '{column}': replaced {na_count} identified "
                        f"NA values (pattern: {pattern.pattern})"
                    )
            
            self.data[column] = s
    
    def fill_na(
        self,
        column_fill_values: dict[str, Any]
    ) -> None:
        logger.info(
            "Filling NA values in columns: "
            f"{', '.join(column_fill_values.keys())}"
        )
        
        for column, fill_value in column_fill_values.items():
            if column not in self.data.columns:
                logger.warning(
                    f"The column '{column}' is not present in the data. "
                    f"Skipping."
                )
                continue
            
            if pd.isna(fill_value) or (fill_value is None):
                logger.debug(
                    f"Column '{column}': fill value is NA, skipping."
                )
                continue
            
            c = self.data[column]
            na_count = int(c.isna().sum())
            if na_count == 0:
                logger.debug(f"Column '{column}': no NA values to fill.")
                continue
            
            # --- Categorical handling ---
            if pd.api.types.is_categorical_dtype(c):
                cat = c.dtype  # pandas.CategoricalDtype
                categories = cat.categories
                
                # If fill_value is not an existing category, add it.
                # (Categorical cannot accept new values unless they are in
                # categories.)
                if fill_value not in categories:
                    logger.debug(
                        f"Column '{column}': adding {fill_value!r} to "
                        f"categories before filling NA."
                    )
                    self.data[column] = c.cat.add_categories([fill_value])
                    c = self.data[column]  # refresh reference
            
            self.data[column] = c.fillna(fill_value)
            logger.debug(
                f"Column '{column}': filled {na_count} NA values with "
                f"{fill_value!r}"
            )
    
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
    
    def text_relaxed(
        self,
        columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.text(
            columns=columns,
            strip="both",
            compact_whitespace=" ",
            empty_to_na=True
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
    
    # ---------------------- Interface implementation ---------------------- #
    def validate_data(self) -> bool:
        """
        Validate that the DataFrame is in an acceptable state for normalization.
        
        Returns
        -------
        bool
            True if data is valid for normalization, False otherwise
        """
        if not super().validate_data():
            return False
            
        if not isinstance(self.data, pd.DataFrame):
            logger.error(
                f"Expected pandas DataFrame, got {type(self.data).__name__}"
            )
            return False
            
        if self.data.empty:
            logger.warning("DataFrame is empty")
            return False
            
        logger.debug("DataFrame validation passed")
        return True
    
    def get_data_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the DataFrame being normalized.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing detailed information about the DataFrame
        """
        base_info = super().get_data_info()
        
        if not isinstance(self.data, pd.DataFrame):
            return base_info
            
        df_info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": self.data.isnull().sum().to_dict(),
            "has_config": self.columns_config is not None,
        }
        
        if self.columns_config is not None:
            df_info.update({
                "config_name": self.columns_config.name,
                "config_description": self.columns_config.description,
                "configured_columns": list(self.columns_config.keys()),
            })
        
        base_info.update(df_info)
        return base_info