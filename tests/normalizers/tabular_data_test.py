from pandas.testing import assert_series_equal
import pandas as pd

from tests.fixtures.dataframes_test import custom, client
from verbosa.data.normalizers.tabular_data import TabularDataNormalizer


# ------------------------ Text normalization tests ------------------------ #
def test_text_normalization(custom: pd.DataFrame) -> None:
    normalizer = TabularDataNormalizer(data=custom)
    normalizer.text(["noisy_text"], error="coerce")
    normalized_data = normalizer.data
    
    # Expected results
    expected_texts = pd.Series([
        "normal text",
        pd.NA,
        pd.NA,
        pd.NA,
        "another text"
    ], name="noisy_text").astype("string")
    
    assert_series_equal(
        normalized_data["noisy_text"],
        expected_texts
    )


def test_text_fill_na(custom: pd.DataFrame) -> None:
    normalizer = TabularDataNormalizer(data=custom)
    normalizer.text(["noisy_text"], error="coerce")
    normalized_column = normalizer.fill_na(
        columns_and_fills={"noisy_text": "missing"},
    )
    
    # Expected results
    expected_texts = pd.Series([
        "normal text",
        "missing",
        "missing",
        "missing",
        "another text"
    ], name="noisy_text").astype("string")
    
    assert_series_equal(
        normalized_column,
        expected_texts
    )


# ----------------------- Numeric normalization tests ---------------------- #
def test_numeric_normalization(custom: pd.DataFrame) -> None:
    normalizer = TabularDataNormalizer(data=custom)
    normalizer.numeric(
        ["noisy_numbers"],
        errors="coerce",
        dtype="Int64",
        cleanup_pattern=r"[$,\s]+"
    )
    normalized_data = normalizer.data
    
    # Expected results
    expected_numbers = pd.Series([
        1_000,
        pd.NA,
        pd.NA,
        pd.NA,
        3_000,
    ], name="noisy_numbers").astype("Int64")
    
    assert_series_equal(
        normalized_data["noisy_numbers"],
        expected_numbers
    )


# --------------------- Categorical normalization tests -------------------- #
def test_categorical_fill_na(client: pd.DataFrame) -> None:
    normalizer = TabularDataNormalizer(data=client)
    normalizer.data.loc[
        normalizer.data["clasificacion"] == "BASE", "clasificacion"
    ] = pd.NA
    
    normalizer.categorical(
        ["clasificacion"],
        strip="both",
        compact_whitespace=" ",
        case="upper",
        empty_to_na=True,
        delete_diacritics=True,
        delete_non_ascii=True,
        cleanup_pattern=None,
        sort_categories=True
    )
    
    print()
    print(normalizer.data.loc[normalizer.data["clasificacion"] == "BASE", :])
    
    normalizer.fill_na(
        columns_and_fills={"clasificacion": "SIN CLASIFICACION"}
    )
    
    print()
    print(normalizer.data["clasificacion"].dtype)
    print(normalizer.data["clasificacion"].cat.categories)
    print(normalizer.data["clasificacion"].unique().tolist())
    
    print()
    print(normalizer.data.loc[normalizer.data["clasificacion"] == "UNKNOWN", :])


def test_categorical_convert_na(client: pd.DataFrame) -> None:
    normalizer = TabularDataNormalizer(data=client)
    normalizer.categorical(
        ["clasificacion"],
        strip="both",
        compact_whitespace=" ",
        case="upper",
        empty_to_na=True,
        delete_diacritics=True,
        delete_non_ascii=True,
        cleanup_pattern=None,
        sort_categories=True
    )
    
    print()
    print(normalizer.data.loc[normalizer.data["clasificacion"] == "BASE", :])
    
    normalizer.convert_to_na(
        columns_and_nas={"clasificacion": "BASE"}
    )
    
    print()
    print(normalizer.data["clasificacion"].dtype)
    print(normalizer.data["clasificacion"].cat.categories)
    print(normalizer.data["clasificacion"].unique().tolist())
    
    print()
    print(normalizer.data.loc[normalizer.data["clasificacion"].isna(), :])