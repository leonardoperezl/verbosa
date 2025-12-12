import pytest

from verbosa.interfaces.columns_config import ColumnsConfig


@pytest.fixture
def columns_config_path() -> str:
    return "./verbosa/assets/examples/column_norm_config.yaml"


def test_columns_config_from_yaml(columns_config_path) -> None:
    config: ColumnsConfig = ColumnsConfig.from_yaml(columns_config_path)
    
    # Metadata
    assert config.name == "Example. Column normalization configuration"
    assert config.author == "Leonardo Pérez Lázaro"
    assert config.date == "06/12/2025"
    
    # Columns' data
    date = config["date"]
    amount = config["amount"]
    balance = config["balance"]
    
    # 1) Test column lecture and order
    assert len(config.columns) == 6
    assert config.column_names == (
        "fecha", "concepto", "monto", "tipo", "saldo", "clasificacion"
    )
    
    # 2) Check that column values are read correcltly
    assert date.aliases == {"date", "fecha", "transaction_date"}
    assert date.fill_na is None
    assert isinstance(date.reviews, dict)
    assert isinstance(date.normalization, dict)
    assert isinstance(date.reviews["no_na"], dict)
    assert isinstance(date.reviews["no_na"]["tolerance"], float)
    assert isinstance(date.normalization["date"]["dayfirst"], bool)
    
    # 3) Check that column dict values can be compared from lecture
    assert amount.reviews != balance.reviews
    assert amount.normalization == balance.normalization