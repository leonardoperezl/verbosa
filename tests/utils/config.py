from __future__ import annotations
from pathlib import Path


TESTS_DIRECTORY: Path = Path(__file__).resolve().parent.parent


ASSETS_DIRECTORY: Path = TESTS_DIRECTORY / "assets"
EXAMPLES_DIRECTORY: Path = ASSETS_DIRECTORY / "examples"
CONFIG_EXAMPLES_DIRECTORY: Path = EXAMPLES_DIRECTORY / "configs"
DATA_EXAMPLES_DIRECTORY: Path = EXAMPLES_DIRECTORY / "data"
QUERIES_EXAMPLES_DIRECTORY: Path = EXAMPLES_DIRECTORY / "queries"


SECRETS_DIRECTORY: Path = ASSETS_DIRECTORY / "secrets"