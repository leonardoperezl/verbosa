from collections import OrderedDict

from pandas import read_csv, DataFrame
import pytest
import streamlit as st


from verbosa.utils.config import EXAMPLES_DIRECTORY


EXAMPLES_DATA_DIRECTORY = EXAMPLES_DIRECTORY / "data"

CLIENT_PATH = EXAMPLES_DATA_DIRECTORY / "client.csv"
CLIENT_MODIFIED_PATH = EXAMPLES_DATA_DIRECTORY / "client_modified.csv"
CLIENT_TEST_PATH = EXAMPLES_DATA_DIRECTORY / "client_test.csv"

# Reading the dataframes using pandas `read_csv` function
CLIENT_DF: DataFrame = read_csv(CLIENT_PATH, quoting=1)
CLIENT_MODIFIED_DF: DataFrame = read_csv(CLIENT_MODIFIED_PATH, quoting=1)
CLIENT_TEST_DF: DataFrame = read_csv(CLIENT_TEST_PATH, quoting=1)


@pytest.fixture
def client() -> DataFrame:
    return CLIENT_DF


@pytest.fixture
def client_modified() -> DataFrame:
    return CLIENT_MODIFIED_DF


@pytest.fixture
def client_test() -> DataFrame:
    return CLIENT_TEST_DF