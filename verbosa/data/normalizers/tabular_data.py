# Python standard library imports
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, TypeAlias
import logging
import re

# Third-party imports
from pandas.api.types import is_datetime64_any_dtype
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

NAValues: TypeAlias = Optional[str | re.Pattern | int | float]

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

CastingErrorHandling: TypeAlias = Literal["raise", "ignore", "coerce"]


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
    
    # Class attributes
    STRING_DTYPES: set = {"string"}
    NUMERIC_DTYPES: set = {"Int64", "Float64"}
    DATE_DTYPES: set = {"datetime64[ns]", "datetime64[ns, UTC]"}
    CATEGORICAL_DTYPES: set = {"category"}
    BOOLEAN_DTYPES: set = {"boolean"}
    
    ALL_DTYPES = (
        STRING_DTYPES
        .union(NUMERIC_DTYPES)
        .union(DATE_DTYPES)
        .union(CATEGORICAL_DTYPES)
        .union(BOOLEAN_DTYPES)
    )
    
    data: pd.DataFrame
    
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        config_path: Optional[Pathlike | ColumnsConfig] = None
    ) -> None:
        super().__init__(data, config_path)
        if isinstance(config_path, (str, Path)):
            self.autonorm_settings: ColumnsConfig = (
                ColumnsConfig.from_yaml(self.autonorm_settings)
            )
    
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
        if self.autonorm_settings is None:
            raise ValueError(
                "No columns configuration available. Cannot perform automatic "
                "normalization without configuration."
            )
        
        logger.info(
            f"Beginning automatic normalization using configuration: "
            f"{self.autonorm_settings.name}"
        )
        
        # 1) Sort the columns according to the order in the configuration
        self._sort_columns_as_config()
        
        # 2) Apply normalization methods group by group
        self._apply_norm_methods()
        
        # 3) Convert defined NA values to pd.NA
        self._convert_na_values()
        
        # 4) Fill NA values as defined in the configuration
        self._fill_na_values()
        
        logger.info("Ended automatic normalization process successfully.")
        return self.data
    
    def _sort_columns_as_config(self) -> None:
        logger.info(
            "Beggining column sorting as in the configuration provided."
        )
        
        not_in_config = [
            col for col in self.data.columns
            if col not in self.autonorm_settings
        ]
        new_order = list(self.autonorm_settings) + list(not_in_config)
        
        # Sort columns into the provided config order. This action will also
        # rename each column name to each ColumnConfig.name attribute.
        self.data = self.data.loc[:, new_order]
        
        logger.info(
            "Ended column sorting according to configuration."
        )
    
    def _convert_na_values(self) -> None:
        logger.info("Beginning conversion of defined NA values to pd.NA.")
        
        na_values_dict = self.autonorm_settings.get_na_values_dict()
        self.convert_to_na(columns_and_nas=na_values_dict)
        
        logger.info("Ended conversion of defined NA values to pd.NA.")
    
    def _apply_norm_methods(self) -> None:
        logger.info(
            "Beginning application of normalization methods to column groups."
        )
        
        norm_groups = self.autonorm_settings.group_by_normalization()
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
        
        logger.info(
            "Completed applying normalization methods to column groups."
        )
    
    def _fill_na_values(self) -> None:
        logger.info("Filling defined NA values in columns.")
        
        fill_values_dict = self.autonorm_settings.get_columns_fill_na_dict()
        self.fill_na(columns_and_fills=fill_values_dict)
        
        logger.info("Completed filling defined NA values in columns.")
    
    # ----------------- API. General normalization methods ----------------- #
    def text(
        self,
        columns: Sequence[str],
        error: CastingErrorHandling = "coerce",
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
            s = self.data[column]
            
            if error == "coerce":
                str_mask = s.apply(lambda x: isinstance(x, str))
                s = s.mask(~str_mask, pd.NA).astype("string")
            else:
                s = s.astype("string", errors=error)
            
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
        dtype: NormalizedNumericDType = "Float64",
        errors: CastingErrorHandling = "coerce",
        cleanup_pattern: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalizes numeric columns in the DataFrame. Useful for giving the
        provided columns a consistent numeric dtype, and for attempting to
        convert non-numeric columns to numeric by cleaning up unwanted
        characters. Use it when the expected dtype of the column(s) is known.
        
        Normalization rules
        -------------------
        1. Check if a cleanup pattern was provided, if so, remove all matches
        from the column and then cast it to the provided dtype.
        2. If no cleanup pattern was provided, check if the column is a
        non-numeric dtype. If so, attempt to convert it to numeric using
        `pd.to_numeric` with the provided error handling strategy.
        3. Finally, cast the column to the provided dtype.
        
        Parameters
        ----------
        columns : Sequence[str]
            List of column names to normalize as numeric
        
        dtype : NormalizedNumericDType, optional
            The target numeric dtype for the columns. Default is "Float64".
        
        errors : CastingErrorHandling, optional
            Error handling strategy when converting to numeric.
        
        cleanup_pattern : Optional[str], optional
            Regular expression pattern to clean up unwanted characters
            from the columns before conversion.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame with normalized numeric columns.
        """
        
        logger.info(
            f"Normalizing numeric columns: {columns} with options:\n"
            f"- dtype: {dtype}\n"
            f"- cleanup_pattern: {cleanup_pattern}"
        )
        
        for column in columns:
            s: pd.Series = self.data[column]
            
            # 1) Apply cleanup pattern if provided
            if cleanup_pattern is not None:
                cleanup_re: re.Pattern = re.compile(cleanup_pattern)
                
                try:
                    # 1.1) Apply cleanup if the column can be casted to string
                    s = s.astype("string")
                    s = s.str.replace(cleanup_re, "", regex=True)
                    logger.debug(
                        f"Applied cleanup pattern to column '{column}'"
                    )
                except Exception as e:
                    # 1.2) Log the column and skip this step
                    logger.warning(
                        f"Failed to apply cleanup pattern to column "
                        f"'{column}': {e}"
                    )
            
            # 2) Attempt conversion, if cleanup_pattern was provided, the
            # column will be string at this point (always)
            if errors != "coerce":
                s = s.astype(dtype, errors=errors)
                continue
            
            pre_na_count = int(s.isna().sum())
            s = pd.to_numeric(s, errors="coerce")
            na_count = int(s.isna().sum())
            
            if na_count > pre_na_count:
                logger.debug(
                    f"Column '{column}': converted to numeric with "
                    f"{na_count - pre_na_count} additional NA values "
                    f"(original: {pre_na_count}, after conversion: "
                    f"{na_count})"
                )
            
            s = s.astype(dtype)
            self.data[column] = s
            logger.debug(f"Successfully normalized numeric column: {column}")
        
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
            f"- ordered: {ordered}\n"
            f"- sort_categories: {sort_categories}"
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
            s: pd.Series = self.data[column]
            
            # 2.1) Get unique values (categories) from the column
            unique_values = s.dropna().unique()
            
            # 2.2) Sort categories if requested
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
            
            # 2.3) Convert to categorical dtype
            self.data[column] = pd.Categorical(
                s, categories=categories, ordered=ordered
            )
            
            tmp = "(ordered)" if ordered else ""
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
        columns: Sequence[str],
        true_values: Optional[Sequence[Any] | Any] = None,
        false_values: Optional[Sequence[Any] | Any] = None
    ) -> pd.DataFrame:
        logger.info(f"Normalizing boolean columns: {columns}")
        
        for column in columns:
            s: pd.Series = self.data[column]
            
            # 1) Convert true_values and false_values to lists if provided
            if true_values is not isinstance(
                true_values, (list, tuple, set, frozenset)
            ):
                true_values = [true_values]
            if false_values is not isinstance(
                false_values, (list, tuple, set, frozenset)
            ):
                false_values = [false_values]
            
            # 2) Map true and false values if provided
            if true_values is not None:
                s = s.replace(true_values, True)
            if false_values is not None:
                s = s.replace(false_values, False)
            
            # 3) Convert to boolean dtype with coercion
            pre_na_count = int(s.isna().sum())
            s = s.astype("boolean", errors="coerce")
            na_count = int(s.isna().sum())
            
            if na_count > pre_na_count:
                logger.debug(
                    f"Column '{column}': converted to boolean with "
                    f"{na_count - pre_na_count} additional NA values "
                    f"(original: {pre_na_count}, after conversion: "
                    f"{na_count})"
                )
            
            self.data[column] = s
            logger.debug(f"Successfully normalized boolean column: {column}")
        
        logger.info("Completed boolean normalization.")
        return self.data
    
    # --------------------------- API. NA methods -------------------------- #
    def convert_to_na(
        self,
        columns_and_nas: dict[str, Optional[NAValues | Sequence[NAValues]]]
    ) -> pd.DataFrame:
        """
        """
        
        logger.info(
            f"Converting defined NA values to pd.NA in columns: "
            f"{", ".join(columns_and_nas.keys())}"
        )
        
        for column, na_values in columns_and_nas.items():
            if na_values is None:
                continue
            
            # 1) If single value provided, convert to list
            if not isinstance(na_values, (list, tuple, set, frozenset)):
                na_values = [na_values]
            
            s: pd.Series = self.data[column]
            dtype: str = str(s.dtype)
            
            # 2) If the column dtype is not normalized, skip conversion
            if dtype not in self.ALL_DTYPES:
                logger.warning(
                    f"Column '{column}' of type {dtype} is not of a "
                    f"normalized dtype, please use a normalization method "
                    f"and try again. Skipping NA conversion."
                )
                continue
            
            # 3) Get a mask for all NA values
            na_mask: pd.Series = s.isin(na_values)
            na_count: int = int(na_mask.sum())
            
            if dtype in self.CATEGORICAL_DTYPES:
                # 3.1) Remove categories that are being converted to NA
                current_categories = s.cat.categories.tolist()
                self.data[column] = s.cat.remove_categories([
                    val for val in na_values if val in current_categories
                ])
            
            self.data.loc[na_mask, column] = pd.NA
            
            logger.debug(
                f"Column '{column}': converted {na_count} values to pd.NA "
                f"using defined NA values: {na_values}"
            )
        
        logger.info("Completed conversion of defined NA values to pd.NA.")
        return self.data
    
    def fill_na(
        self,
        columns_and_fills: dict[str, Any]
    ) -> pd.DataFrame:
        logger.info(
            f"Filling NA values in columns: "
            f"{", ".join(columns_and_fills.keys())} with the following "
            f"values respectively: {columns_and_fills.values()}"
        )
        
        for column, fill_value in columns_and_fills.items():
            s: pd.Series = self.data[column]
            dtype = str(s.dtype)
            
            # 1) Check if the column dtype is already normalized
            if dtype not in self.ALL_DTYPES:
                logger.warning(
                    f"Column '{column}' of type {dtype} is not of a "
                    f"normalized dtype, please use a normalization method "
                    f"and try again. Skipping NA filling."
                )
                continue
            
            if fill_value is None:
                continue
            
            # 2) Apply filling based on dtype, if not supported, use pandas
            # default behavior
            if dtype in self.STRING_DTYPES:
                s = self._string_fill_na(column, fill_value)
            elif dtype in self.NUMERIC_DTYPES:
                fill_value = float(fill_value)
                s = s.fillna(fill_value)
            elif dtype in self.CATEGORICAL_DTYPES:
                s = self._categorical_fill_na(column, fill_value)
            else:
                logger.warning(
                    f"NA filling for dtype '{dtype}' not implemented yet. "
                    f"Using pandas default fillna behavior."
                )
                s = s.fillna(fill_value)
            
            self.data[column] = s
            logger.debug(f"Successfully filled NA values in column: {column}")
        
        logger.info("Completed filling NA values.")
        return self.data
    
    def _string_fill_na(
        self,
        column: str,
        fill_value: str
    ) -> pd.Series:
        fill_value = str(fill_value)
        s = self.data[column]
        
        na_count = int(s.isna().sum())
        s = s.fillna(fill_value)
        s = s.astype("string")
        
        logger.debug(
            f"Column '{column}': filled {na_count} NA values with "
            f"'{fill_value}'"
        )
        return s
    
    def _categorical_fill_na(
        self,
        column: str,
        fill_value: Any
    ) -> pd.Series:
        s = self.data[column]
        
        # 1) Add fill_value to categories if not present
        if fill_value not in s.cat.categories:
            s = s.cat.add_categories([fill_value])
            logger.debug(
                f"Column '{column}': added fill value '{fill_value}' to "
                f"categories"
            )
        
        # 2) Fill NA values
        na_count = int(s.isna().sum())
        s = s.fillna(fill_value)
        
        logger.debug(
            f"Column '{column}': filled {na_count} NA values with "
            f"'{fill_value}'"
        )
        return s
    
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
        Validate that the DataFrame is in an acceptable state for
        normalization.
        
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
            logger.debug("DataFrame is empty")
            return False
        
        logger.debug("DataFrame validation passed")
        return True