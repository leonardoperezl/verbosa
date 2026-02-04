from pathlib import Path
import json

import pytest

from tests.utils.config import SECRETS_DIRECTORY
from verbosa.interfaces.aws import AthenaDataBaseDetails, AWSCredentials


@pytest.fixture
def aws_secrets() -> dict:
    secrets_file: Path = SECRETS_DIRECTORY / "aws.json"
    with open(secrets_file, "r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def aws_credentials() -> AWSCredentials:
    secrets_file: Path = SECRETS_DIRECTORY / "aws.json"
    with open(secrets_file, "r", encoding="utf-8") as file:
        secrets: dict =  json.load(file)
    
    return AWSCredentials(
        access_key_id=secrets["access_key_id"],
        secret_access_key=secrets["secret_access_key"],
        region_name=secrets["region_name"],
        session_token=secrets["session_token"]
    )


@pytest.fixture
def aws_db_details() -> AthenaDataBaseDetails:
    secrets_file: Path = SECRETS_DIRECTORY / "aws.json"
    with open(secrets_file, "r", encoding="utf-8") as file:
        secrets: dict =  json.load(file)
    
    return AthenaDataBaseDetails(
        database=secrets["database"],
        workgroup=secrets["workgroup"],
        s3_output_location=secrets["s3_output_location"],
        ctas_approach=secrets["ctas_approach"],
        ctas_parameters=secrets["ctas_parameters"],
        unload_approach=secrets["unload_approach"],
        unload_parameters=secrets["unload_parameters"]
    )