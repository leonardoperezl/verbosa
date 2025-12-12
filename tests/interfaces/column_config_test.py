from typing import Any

import pytest

from verbosa.interfaces.column_config import ColumnConfig


@pytest.fixture
def text_column() -> dict[str, Any]:
    return {
        "name": "concept",
        "dtype": "string",
        "description": "The concept of the transaction",
        "aliases": ["concepto", "transaction_concept"],
        "na_values": ["N/A", "unknown"],
        "fill_na": "READING ERROR",
        "reviews": {
            "no_na": {"tolerance": 0.0},
            "max_length": {"max_length": 100}
        },
        "normalization": {
            "text": {
                "strip": "both",
                "compact_whitespace": " ",
                "case": "title",
                "empty_to_na": True,
                "delete_diacritics": True,
                "delete_non_ascii": True
            }
        }
    }

@pytest.fixture
def other_text_column() -> dict[str, Any]:
    return {
        "name": "notes",
        "dtype": "string",
        "description": "Additional notes",
        "aliases": ["nota", "additional_notes"],
        "na_values": ["N/A", "none"],
        "fill_na": "NO NOTES",
        "reviews": {
            "no_na": {"tolerance": 0.0},
            "max_length": {"max_length": 100}
        },
        "normalization": {
            "text": {
                "strip": "both",
                "compact_whitespace": " ",
                "case": "title",
                "empty_to_na": True,
                "delete_diacritics": True,
                "delete_non_ascii": False
            }
        }
    }


def test_column_config_from_dict(text_column) -> None:
    """
    Checks that all column configurations are being created correctly via the
    ColumnConfig.from_dict() method. See fixtures above to see used
    dictionaries.
    """
    
    column_config = ColumnConfig.from_dict(
        name=text_column["name"],
        data=text_column
    )
    assert column_config.name == "concept"
    assert column_config.dtype == "string"
    assert column_config.description == "The concept of the transaction"
    assert column_config.aliases == {"concept", "concepto", "transaction_concept"}
    assert column_config.na_values == ["N/A", "unknown"]
    assert column_config.fill_na == "READING ERROR"
    assert column_config.reviews == {
        "no_na": {"tolerance": 0.0},
        "max_length": {"max_length": 100}
    }
    assert column_config.normalization == {
        "text": {
            "strip": "both",
            "compact_whitespace": " ",
            "case": "title",
            "empty_to_na": True,
            "delete_diacritics": True,
            "delete_non_ascii": True
        }
    }


def test_column_configs_norm_equality(text_column, other_text_column) -> None:
    """
    Checks that ConlumnConfig dict-like attributes can be compared between
    instances, such attributes are: `normalization` and `reviews`
    """
    
    concept = ColumnConfig.from_dict(
        name=text_column["name"],
        data=text_column
    )
    notes = ColumnConfig.from_dict(
        name=other_text_column["name"],
        data=other_text_column
    )
    
    assert concept is not notes
    assert concept.normalization != notes.normalization
    assert concept.reviews == notes.reviews