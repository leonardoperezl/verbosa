import pandas as pd

from tests.fixtures.dataframes_test import client
from verbosa.widgets.selection_menu import SelectionMenu


def test_selection_menu_search(client: pd.DataFrame) -> None:
    menu: SelectionMenu = SelectionMenu(data=client)
    
    # Search the number 6 at the specified columns
    search_value = "72"  # Value is int, expected to be casted to str
    search_columns = ["concepto", "monto", "saldo", "rfc_entidad"]
    search_results = menu.search(
        value=search_value,
        at=search_columns,
        case_sensitive=False,
        whole_match=False
    )
    
    # Send result to a file at the output folder
    search_results.to_csv("./output/selection_menu_test.csv", index=True)