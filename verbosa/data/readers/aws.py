from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import logging


import awswrangler as wr
import pandas as pd


from verbosa.utils.global_typings import TDViewer
from verbosa.utils.validation_helpers import is_file_path


if TYPE_CHECKING:
    import boto3
    from verbosa.interfaces.aws import AWSCredentials, AthenaTableDetails


logger = logging.getLogger(__name__)


def _read_query_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        sql = file.read()
    return sql


class AthenaDataReader:
    def __init__(
        self,
        boto3_session: boto3.Session,
        table_details: AthenaTableDetails,
    ) -> None:
        self.session: boto3.Session = boto3_session
        self.table_details: AthenaTableDetails = table_details
    
    def execute_query(self, query: str) -> pd.DataFrame:
        if is_file_path(query): query = _read_query_file(query)
        
        df: pd.DataFrame = wr.athena.read_sql_query(
            sql=query,
            database=self.table_details.database,
            ctas_approach=self.table_details.ctas_approach,
            workgroup=self.table_details.workgroup,
            s3_output=self.table_details.s3_output_location,
            boto3_session=self.session
        )
        return df


class AWSDataReader:
    def __init__(
        self,
        aws_credentials: AWSCredentials
    ) -> None:
        self.session: boto3.Session = aws_credentials.to_boto3_session()