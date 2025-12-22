from typing import Any

import pytest


@pytest.fixture
def concept_column() -> dict[str, Any]:
    return {
        "name": "concept",
        "dtype": "string",
        "description": "The concept of the transaction",
        "aliases": ["concepto", "transaction_concept"],
        "na_values": [
            "N/A",
            "unknown"
        ],
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
def notes_column() -> dict[str, Any]:
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


@pytest.fixture
def date_column() -> dict[str, Any]:
    return {
        "name": "transaction_date",
        "dtype": "datetime64[ns]",
        "description": "The date of the transaction",
        "aliases": ["fecha_transaccion", "date_of_transaction"],
        "na_values": [
            "0000-00-00",
            "9999-99-99",
            r"re.Pattern('[Na](/)?[Aa]')"
        ],
        "fill_na": "pd.Timestamp('1970-01-01')",
        "reviews": {
            "no_na": {
                "tolerance": 0.0
            },
            "date_range": {
                "start_date": "2000-01-01",
                "end_date": "2025-12-31"
            }
        },
        "normalization": {
            "date": {
                "format": "%Y-%m-%d",
                "coerce_errors": True
            }
        }
    }


@pytest.fixture
def is_spei_column() -> dict[str, Any]:
    return {
        "name": "is_spei",
        "dtype": "boolean",
        "description": "Indicates if the transaction was made via SPEI",
        "aliases": ["es_spei", "via_spei"],
        "na_values": None,
        "fill_na": None,
        "reviews": None,
        "normalization": None
    }