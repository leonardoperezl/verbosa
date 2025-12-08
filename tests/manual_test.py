import pandas as pd

# Modules to test manually
from verbosa.data.normalizers.tabular_data import TabularDataNormalizer
from verbosa.utils.logger_machine import LogsMachine


logs_config_path = "./verbosa/assets/examples/logs_config.yaml"
logs_machine = LogsMachine(config_path=logs_config_path)
logs_machine.on()


data_file = "./verbosa/assets/examples/client_data.csv"
yaml_file = "./verbosa/assets/examples/column_norm_config.yaml"

data = pd.read_csv(data_file, quoting=1)

# columns_config = ColumnsConfig.from_yaml(yaml_file)
tabular_normalizer = TabularDataNormalizer(
    data=data,
    columns_config_path=yaml_file
)
new_data: pd.DataFrame = tabular_normalizer.autonorm()
new_data.to_csv("./output/client_data_normalized.csv", index=False, quoting=1)