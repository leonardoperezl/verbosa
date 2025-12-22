from re import compile
from typing import Any

from datetime import datetime
import pandas as pd

from verbosa.interfaces.column_config import ColumnConfig, CallSpec
from tests.fixtures.dictionaries_test import (
    concept_column,
    notes_column,
    is_spei_column,
    date_column
)


def test_from_dict(concept_column) -> None:
    """
    Checks that all column configurations are being created correctly via the
    ColumnConfig.from_dict() method. See fixtures above to see used
    dictionaries.
    """
    
    concept = ColumnConfig.from_dict(
        name=concept_column["name"],
        data=concept_column
    )
    assert concept.name == "concept"
    assert concept.dtype == "string"
    assert concept.description == "The concept of the transaction"
    assert concept.aliases == {"concept", "concepto", "transaction_concept"}
    assert concept.na_values == ("N/A", "unknown")
    assert concept.fill_na == "READING ERROR"
    
    # Check that reviews are stored as CallSpec objects
    assert isinstance(concept.reviews, tuple)
    assert len(concept.reviews) == 2
    assert all(isinstance(spec, CallSpec) for spec in concept.reviews)
    
    # Check specific review CallSpecs
    review_methods = {spec.method_name for spec in concept.reviews}
    assert review_methods == {"no_na", "max_length"}
    
    # Check that they convert back to dict format correctly
    reviews_dict = concept._pipeline_to_yaml(concept.reviews)
    assert reviews_dict == {
        "no_na": {"tolerance": 0.0},
        "max_length": {"max_length": 100}
    }
    
    # Check normalization
    assert isinstance(concept.normalization, tuple)
    assert len(concept.normalization) == 1
    norm_dict = concept._pipeline_to_yaml(concept.normalization)
    assert norm_dict == {
        "text": {
            "strip": "both",
            "compact_whitespace": " ",
            "case": "title",
            "empty_to_na": True,
            "delete_diacritics": True,
            "delete_non_ascii": True
        }
    }


def test_equality(concept_column, notes_column) -> None:
    """
    Checks that ConlumnConfig dict-like attributes can be compared between
    instances, such attributes are: `normalization` and `reviews`
    """
    
    concept = ColumnConfig.from_dict(
        name=concept_column["name"],
        data=concept_column
    )
    notes = ColumnConfig.from_dict(
        name=notes_column["name"],
        data=notes_column
    )
    
    assert concept is not notes
    assert concept.normalization != notes.normalization
    assert concept.reviews == notes.reviews
    
    # Test that CallSpec objects are properly hashable and comparable
    concept_reviews_set = set(concept.reviews)
    notes_reviews_set = set(notes.reviews)
    assert concept_reviews_set == notes_reviews_set


def test_get_normalization_hashes(concept_column, is_spei_column) -> None:
    concept: ColumnConfig = ColumnConfig.from_dict(
        name=concept_column["name"], data=concept_column
    )
    is_spei: ColumnConfig = ColumnConfig.from_dict(
        name=is_spei_column["name"], data=is_spei_column
    )
    
    concept_hashes: tuple[str] = concept.get_normalization_hashes()
    is_spei_hashes: tuple[str] = is_spei.get_normalization_hashes()
    
    concept_expexted_hash = (
        "text: "
        "('case', 'title') - "
        "('compact_whitespace', ' ') - "
        "('delete_diacritics', True) - "
        "('delete_non_ascii', True) - "
        "('empty_to_na', True) - "
        "('strip', 'both')"
    )
    is_spei_exepected_hash = "None"
    
    assert isinstance(concept_hashes, tuple)
    assert isinstance(is_spei_hashes, tuple)
    
    assert len(concept_hashes) == 1
    assert len(is_spei_hashes) == 1
    
    assert concept_hashes[0] == concept_expexted_hash
    assert is_spei_hashes[0] == is_spei_exepected_hash


def test_call_spec_creation() -> None:
    """
    Test CallSpec creation and methods.
    """
    # Test from_map with parameters
    spec_with_params = CallSpec.from_map(
        "max_length",
        {"max_length": 100, "tolerance": 0.05}
    )
    assert spec_with_params.method_name == "max_length"
    assert len(spec_with_params.params) == 2
    
    # Parameters should be sorted by key
    param_keys = [param[0] for param in spec_with_params.params]
    assert param_keys == ["max_length", "tolerance"]
    
    # Test params_to_dict
    params_dict = spec_with_params.params_to_dict()
    assert params_dict == {"max_length": 100, "tolerance": 0.05}
    
    # Test to_hash
    expected_hash = "max_length: ('max_length', 100) - ('tolerance', 0.05)"
    assert spec_with_params.to_hash() == expected_hash
    
    # Test from_map with no parameters
    spec_no_params = CallSpec.from_map("no_na", None)
    assert spec_no_params.method_name == "no_na"
    assert spec_no_params.params == tuple()
    assert spec_no_params.to_hash() == "no_na"


def test_call_spec_hashability() -> None:
    """
    Test that CallSpec objects are hashable and can be used in sets/dicts.
    """
    spec1 = CallSpec.from_map("text", {"case": "title", "strip": "both"})
    spec2 = CallSpec.from_map("text", {"strip": "both", "case": "title"})  # Same params, different order
    spec3 = CallSpec.from_map("text", {"case": "lower", "strip": "both"})
    
    # Same specs should be equal and have same hash
    assert spec1 == spec2
    assert hash(spec1) == hash(spec2)
    
    # Different specs should not be equal
    assert spec1 != spec3
    
    # Should be able to use in sets
    spec_set = {spec1, spec2, spec3}
    assert len(spec_set) == 2  # spec1 and spec2 are the same


def test_column_config_serialization() -> None:
    """
    Test to_dict method for serialization.
    """
    config = ColumnConfig(
        name="test_col",
        dtype="string",
        description="Test column",
        aliases=["alias1", "alias2"],
        na_values=["N/A", "NULL"],
        fill_na="DEFAULT",
        reviews={"no_na": {"tolerance": 0.0}},  # Using legacy dict format that gets converted
        normalization={"text": {"case": "title"}}
    )
    
    result_dict = config.to_dict()
    
    assert result_dict["name"] == "test_col"
    assert result_dict["dtype"] == "string"
    assert result_dict["description"] == "Test column"
    assert set(result_dict["aliases"]) == {"test_col", "alias1", "alias2"}
    assert result_dict["na_values"] == ("N/A", "NULL")
    assert result_dict["fill_na"] == "DEFAULT"


def test_legacy_pipeline_formats() -> None:
    """
    Test that legacy pipeline formats are properly converted.
    """
    # Test string format (single method, no params)
    config_str = ColumnConfig(
        name="test",
        dtype="string",
        normalization="text"
    )
    assert isinstance(config_str.normalization, tuple)
    assert len(config_str.normalization) == 1
    assert config_str.normalization[0].method_name == "text"
    assert config_str.normalization[0].params == tuple()
    
    # Test dict format (multiple methods with params)
    config_dict = ColumnConfig(
        name="test",
        dtype="string",
        reviews={
            "no_na": {"tolerance": 0.0},
            "max_length": {"max_length": 50}
        }
    )
    assert isinstance(config_dict.reviews, tuple)
    assert len(config_dict.reviews) == 2
    
    # Check that methods are present
    method_names = {spec.method_name for spec in config_dict.reviews}
    assert method_names == {"no_na", "max_length"}


def test_is_alias_method() -> None:
    """
    Test the is_alias method.
    """
    config = ColumnConfig(
        name="primary_name",
        dtype="string",
        aliases=["alt_name", "another_alias"]
    )
    
    # Primary name should be an alias
    assert config.is_alias("primary_name")
    
    # Explicit aliases should work
    assert config.is_alias("alt_name")
    assert config.is_alias("another_alias")
    
    # Non-aliases should return False
    assert not config.is_alias("not_an_alias")
    assert not config.is_alias("random_name")


def test_casting_functionality() -> None:
    """
    Test string casting for special types.
    """
    # Test regex pattern casting
    config = ColumnConfig(
        name="test",
        dtype="string",
        na_values=["re.Pattern('[Nn][Aa]')"]
    )
    
    # Should be converted to actual regex pattern
    assert len(config.na_values) == 1
    pattern = config.na_values[0]
    assert isinstance(pattern, compile("test").__class__)  # Check it's a regex Pattern
    
    # Test timestamp casting
    config_date = ColumnConfig(
        name="test_date",
        dtype="datetime64[ns]",
        fill_na="pd.Timestamp('2000-01-01')"
    )
    
    assert isinstance(config_date.fill_na, pd.Timestamp)
    assert config_date.fill_na == pd.Timestamp('2000-01-01')


def test_frozen_dataclass_behavior() -> None:
    """
    Test that CallSpec objects behave as frozen dataclasses.
    """
    spec = CallSpec.from_map("text", {"case": "title"})
    
    # Should not be able to modify frozen dataclass
    import pytest
    with pytest.raises(AttributeError):  # FrozenInstanceError in newer versions
        spec.method_name = "other_method"  # This should fail
    
    # Should be hashable
    hash_val = hash(spec)
    assert isinstance(hash_val, int)


def test_pipeline_conversion_edge_cases() -> None:
    """
    Test edge cases in pipeline conversion.
    """
    # Test None pipeline
    config_none = ColumnConfig(
        name="test",
        dtype="string",
        normalization=None
    )
    assert config_none.normalization is None
    assert config_none._pipeline_to_yaml(config_none.normalization) is None
    
    # Test single method with no parameters
    config_single = ColumnConfig(
        name="test",
        dtype="string",
        normalization="text"
    )
    yaml_output = config_single._pipeline_to_yaml(config_single.normalization)
    assert yaml_output == "text"
    
    # Test multiple methods
    config_multi = ColumnConfig(
        name="test", 
        dtype="string",
        normalization={
            "text": {"case": "upper"},
            "no_special": {"chars": "@#$%"}
        }
    )
    yaml_output = config_multi._pipeline_to_yaml(config_multi.normalization)
    assert isinstance(yaml_output, dict)
    assert "text" in yaml_output
    assert "no_special" in yaml_output


def test_complex_parameter_freezing() -> None:
    """
    Test that complex nested parameters are properly frozen and unfrozen.
    """
    complex_config = ColumnConfig(
        name="test",
        dtype="string",
        normalization={
            "complex_method": {
                "nested_dict": {"a": 1, "b": [2, 3]},
                "list_param": [4, 5, {"c": 6}],
                "set_param": {7, 8, 9}
            }
        }
    )
    
    # Should be able to create without errors
    assert isinstance(complex_config.normalization, tuple)
    assert len(complex_config.normalization) == 1
    
    spec = complex_config.normalization[0]
    params_dict = spec.params_to_dict()
    
    # Check that complex structures are properly reconstructed
    assert "nested_dict" in params_dict
    assert "list_param" in params_dict
    assert "set_param" in params_dict
    
    # Check the reconstructed types
    assert isinstance(params_dict["nested_dict"], dict)
    assert isinstance(params_dict["list_param"], list)
    assert isinstance(params_dict["set_param"], set)


def test_error_handling_in_parse_pipeline() -> None:
    """
    Test error handling in _parse_pipeline method.
    """
    import pytest
    
    # Test invalid pipeline type
    with pytest.raises(TypeError):
        ColumnConfig(
            name="test",
            dtype="string",
            normalization=123  # Invalid type
        )


def test_round_trip_serialization() -> None:
    """
    Test that serialization and deserialization produce equivalent objects.
    """
    original = ColumnConfig(
        name="test_column",
        dtype="float64",
        description="A test column for serialization",
        aliases=["test_alias", "another_alias"],
        na_values=["N/A", "NULL", "re.Pattern('missing')"],
        fill_na="pd.Timestamp('2000-01-01')",
        reviews={
            "no_na": {"tolerance": 0.05},
            "range_check": {"min_val": 0.0, "max_val": 100.0}
        },
        normalization={
            "numeric": {"dtype": "Float64"},
            "outlier_removal": {"method": "iqr", "factor": 1.5}
        }
    )
    
    # Convert to dict and back
    as_dict = original.to_dict()
    reconstructed = ColumnConfig.from_dict(
        name=as_dict["name"],
        data=as_dict
    )
    
    # Check that key properties are preserved
    assert original.name == reconstructed.name
    assert original.dtype == reconstructed.dtype
    assert original.description == reconstructed.description
    assert original.aliases == reconstructed.aliases
    assert original.na_values == reconstructed.na_values
    assert original.fill_na == reconstructed.fill_na
    
    # Check that pipelines are equivalent
    assert len(original.reviews) == len(reconstructed.reviews)
    assert len(original.normalization) == len(reconstructed.normalization)
    
    # Check that the hash representations are the same
    assert (original.get_normalization_hashes() == 
            reconstructed.get_normalization_hashes())


def test_na_values_casting(date_column) -> None:
    date: ColumnConfig = ColumnConfig.from_dict(
        name=date_column["name"], data=date_column
    )
    expected_na_values: tuple = (
        "0000-00-00",
        "9999-99-99",
        compile("[Na](/)?[Aa]")
    )
    expected_fill_na: pd.Timestamp = pd.Timestamp("1970-01-01")
    
    assert isinstance(date.na_values, tuple)
    assert date.na_values == expected_na_values
    assert date.fill_na == expected_fill_na