import pandas as pd


from tests.fixtures.secrets_test import aws_credentials, aws_db_details, aws_secrets
from tests.fixtures.queries_test import basic_table_query
from verbosa.data.readers.aws import AthenaDataReader


def test_optimization_method(aws_credentials, aws_db_details):
    reader: AthenaDataReader = AthenaDataReader(
        aws_credentials.to_boto3_session(),
        db_details=aws_db_details
    )
    
    reader.optimize_for(output_size="small")
    
    assert not reader.db_details.ctas_approach
    assert not reader.db_details.unload_approach


def test_queries_via_template(
    basic_table_query,
    aws_credentials,
    aws_db_details,
    aws_secrets
) -> None:
    reader: AthenaDataReader = AthenaDataReader(
        aws_credentials.to_boto3_session(),
        db_details=aws_db_details
    )
    
    reader.optimize_for(output_size="large")
    
    output = reader.execute_query(
        query=basic_table_query,
        columns="*",
        table_name=aws_secrets["table"]
    )
    
    assert isinstance(output, pd.DataFrame)