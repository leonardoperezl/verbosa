from pathlib import Path
from string import Template
import pytest

from tests.utils.config import QUERIES_EXAMPLES_DIRECTORY


@pytest.fixture
def queries_directory() -> str:
    return QUERIES_EXAMPLES_DIRECTORY


@pytest.fixture
def basic_table_query() -> Path:
    return QUERIES_EXAMPLES_DIRECTORY / "basic_table_query.sql"