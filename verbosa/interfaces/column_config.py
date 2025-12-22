from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypeAlias
import re
import logging


import pandas as pd


from verbosa.utils.typings import (
    TDType,
    TDReviewMethod,
    TDNormalizationMethod
)
from verbosa.utils.serialization_helpers import str_sequence_param


logger = logging.getLogger(__name__)


AllowedNA: TypeAlias = str | float | int
StrCastedDTypes: TypeAlias = re.Pattern | pd.Timestamp | str
ReviewSpecInput: TypeAlias = (
    tuple["CallSpec", ...]
    | dict[TDReviewMethod, dict]
    | TDReviewMethod
)
NormalizationSpecInput: TypeAlias = (
    tuple["CallSpec", ...]
    | dict[TDNormalizationMethod, dict]
    | TDNormalizationMethod
)

_CASTING_PATTERN: re.Pattern = re.compile(
    r"^(?P<dtype>.+?)\('(?P<value>.+?)'\)$"
)


def _cast_string(value: Any) -> StrCastedDTypes:
    value = value
    
    if not isinstance(value, str):
        logger.debug(
            f"The '{value}' of type '{type(value)}' "
            f"can't be casted since is not string"
        )
        return value
    
    match = _CASTING_PATTERN.fullmatch(value)
    if match is None:
        return value
    
    dtype_str = match.group("dtype").strip()
    cast_value = match.group("value").strip()
    
    if dtype_str == "re.Pattern": return re.compile(cast_value)
    elif dtype_str == "pd.Timestamp": return pd.Timestamp(cast_value)
    
    logger.warning(
        f"Unknown casting type: {dtype_str}. "
        f"Returning original string."
    )
    return value


def _freeze(
    value: Any,
    map_sort_key: Callable
) -> Any:
    """
    Convert potentially-unhashable values into hashable equivalents
    recursively.
    
    This is needed because frozen dataclasses derive __hash__ from their
    fields. If any field contains an unhashable element (e.g., list inside a
    tuple), hashing the instance fails.
    """
    # Handle mappings (dict-like)
    if isinstance(value, Mapping):
        # Freeze keys+values and sort for determinism
        return tuple(
            sorted((k, _freeze(v, map_sort_key)) for k, v in value.items())
        )
    
    # Handle sequences
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v, map_sort_key) for v in value)
    
    # Handle sets
    if isinstance(value, set):
        return frozenset(_freeze(v, map_sort_key) for v in value)
    
    # Leave everything else as-is (strings, ints, regex patterns,
    # pd.Timestamp, etc.)
    return value


def _unfreeze(value: Any) -> Any:
    """
    Reconstruct user-friendly (mutable) objects from frozen/hashable ones.
    
    This is the inverse of `_freeze` and is intended ONLY for presentation,
    debugging, or serialization (e.g. to YAML).
    
    Rules:
    - tuple of (key, value) pairs -> dict
    - tuple of non-pairs -> list
    - frozenset -> set
    - everything else -> unchanged
    """
    # Frozen mapping: tuple of (key, value) pairs
    if isinstance(value, tuple):
        if value and all(
            isinstance(item, tuple) and len(item) == 2 for item in value
        ):
            return { k: _unfreeze(v) for k, v in value }
        
        # Frozen sequence
        return [ _unfreeze(v) for v in value ]
    
    # Frozen set
    if isinstance(value, frozenset):
        return { _unfreeze(v) for v in value }
    
    return value


@dataclass(frozen=True, slots=True)
class CallSpec:
    """
    A hashable "method call" specification.
    
    This replaces the previous dict-of-dict shape used by ColumnConfig for
    normalization / reviews.
    
    Parameters
    ----------
    method : TDNormalizationMethod | TDReviewMethod
        The name of the method to call.
    
    params : tuple[tuple[str, Any], ...], default=()
        The parameters to pass to the method, stored as an ordered tuple of
        (key, value) pairs.
    
    Notes
    -----
    - `params` is stored as an ordered tuple of (key, value) pairs so it is
    hashable and stable for grouping.
    - Values are casted (when possible) to support YAML-friendly strings such
    as "re.Pattern('...')" or "pd.Timestamp('...')".
    """
    
    method_name: TDNormalizationMethod | TDReviewMethod
    params: tuple[tuple[str, Any], ...]
    
    # ------------------------ Class Methods ------------------------------- #
    @classmethod
    def from_map(
        cls,
        method_name: TDNormalizationMethod | TDReviewMethod,
        params: Mapping[str, Any] | None,
        *,
        sort_key: Callable[[tuple[str, Any]], Any] | None = None
    ) -> "CallSpec":
        sort_key = sort_key or (lambda kv: kv[0]) 
        if params is None:
            logger.debug(
                f"CallSpec parameters for method '{method_name}' is None. "
                "Using empty parameters."
            )
            return cls(method_name=method_name, params=tuple())
        
        if not isinstance(params, Mapping):
            logger.debug(
                "CallSpec parameters must be a mapping (dict-like). "
                f"Got {type(params).__name__}."
            )
            return cls(method_name=method_name, params=tuple())
        
        keys_and_values: list[tuple[str, Any]] = []
        for k, v in params.items():
            v = _cast_string(v)
            v = _freeze(v, sort_key)
            keys_and_values.append((k, v))
        
        return cls(
            method_name=method_name,
            params=tuple(sorted(keys_and_values, key=sort_key))
        )
    
    # ------------------------ Instance Methods ---------------------------- #
    def params_to_dict(self) -> dict[str, Any]:
        return {k: _unfreeze(v) for k, v in self.params}
    
    def to_hash(self) -> str:
        """Deterministic human-readable hash used for logs / grouping."""
        if not self.params:
            return f"{self.method_name}"
        hashed_params = " - ".join(str(item) for item in self.params)
        return f"{self.method_name}: {hashed_params}"


@dataclass
class ColumnConfig:
    """
    Stores data related to a column.
    
    Parameters
    ----------
    name : str
        The main reference to the column.
    
    dtype : TDType
        The target data type for the column.
    
    description : str, default None
        A human-readable description of the column.
    
    aliases : Sequence[str], str, default None
        Alternative names for the column.
    
    na_values : AllowedNA, Sequence[AllowedNA], default None
        Additional values to consider as NA/missing.
    
    fill_na : AllowedNA, default None
        Value to use to fill missing values.
    
    reviews : ReviewSpecInput, default None
        Data review pipeline to apply to the column.
    
    normalization : NormalizationSpecInput, default None
        Data normalization pipeline to apply to the column.
    """
    name: str
    dtype: TDType
    description: Optional[str] = None
    aliases: Optional[str | Sequence[str]] = None
    na_values: Optional[AllowedNA | Sequence[AllowedNA]] = None
    fill_na: Optional[AllowedNA] = None
    
    # Stored as a pipeline of CallSpec objects.
    # Backwards compatible: legacy shapes (str or dict[str, dict]) are
    # accepted and normalized into CallSpec pipelines in __post_init__.
    reviews: Optional[ReviewSpecInput] = None
    normalization: Optional[NormalizationSpecInput] = None
    
    def __post_init__(self) -> None:
        # 1) Ensure aliases is a set and includes the main name
        aliases: set[str] = set()
        if isinstance(self.aliases, str):
            aliases.add(self.aliases)
        elif isinstance(self.aliases, Sequence):
            aliases = set(self.aliases)
        elif self.aliases is None:
            aliases = set()
        else:
            logger.warning(
                f"Invalid type for aliases: {type(self.aliases).__name__}. "
                "Expected str or Sequence[str]."
            )
            aliases = set()
        
        self.aliases: set = aliases
        self.aliases.add(self.name)
        
        # 2) Cast every na value to its correct dtype
        na_values: Optional[tuple[StrCastedDTypes]] = None
        if isinstance(self.na_values, Sequence):
            na_values = tuple(_cast_string(nas) for nas in self.na_values)
        elif self.na_values is not None:
            na_values = (_cast_string(self.na_values), )
        
        self.na_values: Optional[tuple[StrCastedDTypes]] = na_values
        
        # 3) Cast fill_na to its correct dtype
        if self.fill_na is not None:
            self.fill_na: StrCastedDTypes = _cast_string(self.fill_na)
        
        # 4) Normalize legacy review/normalization shapes into CallSpec
        self.reviews = self._parse_pipeline(self.reviews)
        self.normalization = self._parse_pipeline(self.normalization)
    
    @staticmethod
    def _parse_pipeline(
        value: Any,
    ) -> Optional[tuple[CallSpec, ...]]:
        """
        Parse a normalization/review pipeline into a tuple[CallSpec, ...].
        
        Accepted inputs
        ---------------
        - None
        - "method" (string literal)
        - {"method_a": {"param1": value1, ...}, ...} (dict-of-dict)
        - (CallSpec, ...)  (already parsed)
        """
        if value is None:
            return None
        
        # Already normalized
        is_callspec_tuple = (
            isinstance(value, tuple) and
            all(isinstance(x, CallSpec) for x in value)
        )
        if is_callspec_tuple:
            return value
        
        # A single method with no params
        if isinstance(value, str):
            return (CallSpec.from_map(value, None),)
        
        # Legacy dict-of-dict
        if isinstance(value, Mapping):
            specs: list[CallSpec] = []
            for method, params in value.items():
                spec = CallSpec.from_map(method, params)
                specs.append(spec)
            return tuple(specs)
        
        raise TypeError(
            "Invalid pipeline type. Expected None, str, mapping, or "
            f"tuple[CallSpec, ...]. Got {type(value).__name__}."
        )
    
    # --------------------------- Instance methods ------------------------- #
    def is_alias(self, alias: str) -> bool:
        """
        Check if given column name is an alias for this column.
        """
        return alias in self.aliases
    
    # ------------------------- Serialization Methods ---------------------- #
    @classmethod
    def from_dict(
        cls,
        name: str,
        data: dict[str, Any]
    ) -> ColumnConfig:
        """
        Parse column config from dict.
        """
        # Handle checks - can be None, list of dicts, or list with None
        parameters = {
            "name": name,
            "dtype": data.get("dtype"),
            "description": data.get("description"),
            "aliases": data.get("aliases"),
            "na_values": data.get("na_values"),
            "fill_na": data.get("fill_na"),
            "reviews": data.get("reviews"),
            "normalization": data.get("normalization")
        }
        return cls(**parameters)
    
    @staticmethod
    def _pipeline_to_yaml(pipeline: Optional[tuple[CallSpec, ...]]) -> Any:
        """Convert a CallSpec pipeline into a YAML-friendly shape.
        
        Output mirrors the legacy config format:
        - None
        - "method" (single method, no params)
        - {"method": {param: value, ...}, ...}
        """
        if pipeline is None:
            return None
        if (len(pipeline) == 1) and (pipeline[0].params == tuple()):
            return pipeline[0].method_name
        return {spec.method_name: spec.params_to_dict() for spec in pipeline}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert back to YAML-compatible dict."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "aliases": tuple(self.aliases),
            "na_values": self.na_values,
            "fill_na": self.fill_na,
            "reviews": self._pipeline_to_yaml(self.reviews),
            "normalization": self._pipeline_to_yaml(self.normalization)
        }
    
    def get_normalization_hashes(
        self,
        sort_key: Optional[Callable] = None
    ) -> tuple[str]:
        """
        Transforms each normalization set of method-parameters into an ordered
        hash.
        
        Returns
        -------
        tuple[str]
            Each entry corresponds to a normalization method and its
            parameters, in the format:
            ```
            method_name: (param1, value1) - (param2, value2) - ...
            ```
        """
        
        pipeline = self.normalization
        if pipeline is None:
            return ("None",)
        
        # If a custom sort_key is provided, we re-order each spec's params
        # for hashing. (Normally params are already stored sorted.)
        if sort_key is None:
            return tuple(spec.to_hash() for spec in pipeline)
        
        rehashed: list[str] = []
        for spec in pipeline:
            if not spec.params:
                rehashed.append(f"{spec.method_name}")
                continue
            hashed_params = " - ".join(
                str(item) for item in sorted(spec.params, key=sort_key)
            )
            rehashed.append(f"{spec.method_name}: {hashed_params}")
        
        return tuple(rehashed)