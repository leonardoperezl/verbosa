import pytest

from verbosa.utils.config import CONFIG_EXAMPLES_DIRECTORY
from verbosa.interfaces.columns_config import ColumnsConfig
from verbosa.interfaces.column_config import CallSpec


@pytest.fixture
def path() -> str:
    return CONFIG_EXAMPLES_DIRECTORY / "column_norm_config.yaml"


def test_from_yaml(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    
    # Metadata
    assert config.name == "Example. Column normalization configuration"
    assert config.author == "Leonardo Pérez Lázaro"
    assert config.date == "06/12/2025"
    
    # Columns' data
    date = config["date"]
    amount = config["amount"]
    is_spei = config["is_spei"]
    type_config = config["type"]
    classification = config["classification"]
    
    # 1) Test column lecture and order
    assert len(config) == 7
    
    # 2) Check that column values are read correcltly
    assert date.aliases == {"date", "fecha", "transaction_date"}
    assert date.fill_na == "1970-01-01"
    assert isinstance(date.reviews, tuple)
    assert isinstance(date.normalization, tuple)
    assert all(isinstance(spec, CallSpec) for spec in date.reviews)
    assert all(isinstance(spec, CallSpec) for spec in date.normalization)
    
    # Convert to dict format for easier testing
    date_reviews_dict = date._pipeline_to_yaml(date.reviews)
    date_norm_dict = date._pipeline_to_yaml(date.normalization)
    assert isinstance(date_reviews_dict["no_na"], dict)
    assert isinstance(date_reviews_dict["no_na"]["tolerance"], float)
    assert isinstance(date_norm_dict["date"]["dayfirst"], bool)
    
    # 3) Check that column dict values can be compared from lecture
    assert classification.normalization == type_config.normalization
    # Amount normalization should be a single CallSpec with no params
    assert isinstance(amount.normalization, tuple)
    assert len(amount.normalization) == 1
    assert amount.normalization[0].method_name == "numeric_float"
    assert amount.normalization[0].params == tuple()
    assert is_spei.normalization is None


def test_get_column_by_alias(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    
    col1 = config["fecha"]
    col2 = config["transaction_date"]
    col3 = config["date"]
    
    assert col1.name == "fecha"
    assert col2.name == "fecha"
    assert col3.name == "fecha"
    assert col1 is col2 is col3
    
    with pytest.raises(KeyError):
        config["non_existent_column"]


def test_columns_attribute(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    
    names = tuple(config)
    expected_names = (
        "fecha",
        "concepto",
        "monto",
        "tipo",
        "saldo",
        "clasificacion",
        "is_spei"
    )
    
    assert names == expected_names


def test_len(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    assert len(config) == 7


def test_contains(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    
    assert "monto" in config
    assert "amount" in config
    assert "transaction_date" in config
    assert "non_existent_column" not in config
    assert 123 not in config


def test_iter(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    
    for col_name, col_config in config.items():
        print(f"{col_name}: {col_config}")


def test_group_by_normalization(path: str) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(path)
    groups = config.group_by_normalization()
    
    # The method now returns (CallSpec, tuple[str, ...]) pairs
    # Extract just the column names for comparison
    column_groups = tuple(group[1] for group in groups)
    
    expected_groups = (
        ("fecha",),
        ("concepto",),
        ("monto",),
        ("tipo", "clasificacion"),
        ("saldo",),
    )
    
    assert len(column_groups) == 5  # is_spei has no normalization
    
    # Verify that each expected group is present
    for expected_group in expected_groups:
        assert expected_group in column_groups
    
    # Verify the structure: each group should have a CallSpec and column names
    for callspec, columns in groups:
        assert isinstance(callspec, CallSpec)
        assert isinstance(columns, tuple)
        assert all(isinstance(col_name, str) for col_name in columns)