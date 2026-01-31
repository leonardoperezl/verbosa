from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING, Any, Optional, Sequence
import logging


import awswrangler as wr
import pandas as pd


from verbosa.utils.typings import Pathlike
from verbosa.utils.validation_helpers import is_file_path


if TYPE_CHECKING:
    import boto3
    from verbosa.interfaces.aws import AWSCredentials, AthenaDataBaseDetails


logger = logging.getLogger(__name__)


def _read_query_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        sql = file.read()
    return sql


class AthenaDataReader:
    def __init__(
        self,
        boto3_session: boto3.Session,
        db_details: AthenaDataBaseDetails,
    ) -> None:
        self.session: boto3.Session = boto3_session
        self.db_details: AthenaDataBaseDetails = db_details
    
    def execute_query(
        self, query: Pathlike | str, kwargs: str
    ) -> Optional[pd.DataFrame]:
        """
        Run the provided query at the Athena query system. All queries made
        should belong to a single database.
        
        Parameters
        ----------
        query : Pathlike or str
            Query to run as if ran at the Athena console. If the value is a 
            valid path to a file, it will read and execute its content.
        
        kwargs: str
            Key value pairs that specify a placeholder and its value at the
            query.
        
        Returns
        -------
        pd.DataFrame
            The result of the query made. If a an empty DataFrame was returned
            it means that the query resulted in no rows returned.
        
        None
            When the query failed because the query had invalid actions /
            values.
        
        Examples
        --------
        >>> reader = AthenaDataReader(boto3_session, db_details)
        code_output
        """
        
        if is_file_path(query): query = _read_query_file(query)
        
        try:
            df: pd.DataFrame = wr.athena.read_sql_query(
                sql=query,
                database=self.db_details.database,
                ctas_approach=self.db_details.ctas_approach,
                workgroup=self.db_details.workgroup,
                s3_output=self.db_details.s3_output_location,
                boto3_session=self.session
            )
        except Exception as e:
            logger.exception(
                f"The query provided: {query} could not be executed as an "
                f"Athena query. Returning an empty DataFrame"
            )
            return pd.DataFrame()
        
        return df
    
    def simple_query(
        self,
        table_name: str,
        columns: Sequence[str] | str,
        *,
        filter_by: Optional[str] = None,
        value: Optional[str] = None
    ) -> pd.DataFrame:
        if not isinstance(columns, (list, tuple, set)):
            columns = [columns]
        
        query: str = f"""
        SELECT {", ".join(columns)}
        FROM {self.db_details.database}.{table_name}
        """
        
        if (filter_by is not None) and (value is not None):
            query += f" WHERE {filter_by} = {value};"
        
        query += ";"
        
        return self.execute_query(query=query)
    
    def get_unique_values(self, table_name:str, column: str) -> list[Any]:
        query: str = f"""
        SELECT DISTINCT({column})
        FROM {self.db_details.database}.{table_name};
        """
        
        df: pd.DataFrame = self.execute_query(query=query)
        return df[column].tolist()


class AWSDataReader:
    def __init__(
        self,
        aws_credentials: AWSCredentials
    ) -> None:
        self.session: boto3.Session = aws_credentials.to_boto3_session()