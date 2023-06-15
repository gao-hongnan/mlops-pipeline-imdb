from typing import TypeVar, Union

from google.cloud.bigquery.dbapi.connection import Connection as BigQueryConnection
from psycopg2.extensions import connection as PostgresConnection

# Connection = TypeVar('Connection', bound=Union[BigQueryConnection, PostgresConnection])
