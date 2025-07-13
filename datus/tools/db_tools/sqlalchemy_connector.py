from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from pandas import DataFrame
from pyarrow import DataType, RecordBatch, Table, array, ipc
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Inspector, Result
from sqlalchemy.exc import SQLAlchemyError

from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLAlchemyConnector(BaseSqlConnector):
    """
    SQLAlchemy connector with Arrow support.
    Implements BaseSqlConnector interface.
    """

    def __init__(self, connection_string: str, dialect: str = "", batch_size: int = 1024):
        """
        Initialize SQLAlchemyConnector.

        Args:
            connection_string: SQLAlchemy connection string or Engine/Connection object
            batch_size: Rows per batch for streaming
        """
        prefix = connection_string.split(":")[0] if isinstance(connection_string, str) else "Unknown"
        if dialect:
            self.dialect = dialect
        else:
            if prefix == "mysql+pymysql":
                self.dialect = DBType.MYSQL
            else:
                self.dialect = prefix
        super().__init__(self.dialect, batch_size)
        self.connection_string = connection_string
        self.engine = None
        self._conn = None
        self._owns_engine = False
        self.use_arrow_dtype_mapping = False

    def __del__(self):
        """Destructor to ensure connections are properly closed."""
        try:
            self.close()
        except Exception:
            # Ignore any errors during cleanup
            pass

    def connect(self):
        """Establish connection to the database."""
        if self.engine and self._owns_engine:
            return
        try:
            self.engine = create_engine(self.connection_string)
            self._conn = self.engine.connect()
            self._owns_engine = True
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "connect",
                    "error_message": str(e),
                    "uri": self.connection_string,
                },
            ) from e
        if self.engine is None or self._conn is None:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "initialize",
                    "uri": self.connection_string,
                    "error_message": "Engine not established",
                },
            )

    def close(self):
        """Close the database connection."""
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
            if self.engine:
                self.engine.dispose()
                self.engine = None
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")

    def validate_input(self, input_params: Any):
        """Validate the input parameters before execution."""
        super().validate_input(input_params)

    def do_execute(
        self,
        input_params: Union[ExecuteSQLInput, Dict[str, Any]],
        result_format: Literal["csv", "arrow", "list"] = "csv",
    ) -> ExecuteSQLResult:
        if isinstance(input_params, ExecuteSQLInput):
            sql_query = input_params.sql_query.strip()
        else:
            sql_query = input_params["sql_query"].strip()
        self.connect()
        try:
            sql_lower = sql_query.lower()
            if sql_lower.startswith("insert"):
                lastrowid, rowcount = self.insert(sql_query)
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql_query,
                    sql_return=str(lastrowid),
                    row_count=rowcount,
                    result_format=result_format,
                )
            elif sql_lower.startswith("update") or sql_lower.startswith("delete"):
                rowcount = self.update(sql_query)
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql_query,
                    sql_return=str(rowcount),
                    row_count=rowcount,
                    result_format=result_format,
                )
            else:
                df = self._execute_query(sql_query)
                result = ExecuteSQLResult(
                    success=True,
                    sql_query=sql_query,
                    row_count=len(df),
                    result_format=result_format,
                )
                if result_format == "csv":
                    result.sql_return = df.to_csv(index=False)
                elif result_format == "arrow":
                    result.sql_return = Table.from_pandas(df)
                else:
                    result.sql_return = df.to_dict(orient="records")
                return result
        except Exception as e:
            logger.error(f"Execute sql error: {str(e)}")
            return ExecuteSQLResult(
                success=True,
                error=str(e),
                sql_query=sql_query,
                sql_return="",
                row_count=0,
                result_format=result_format,
            )

    def execute_csv(self, query: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results with csv format."""
        self.connect()

        try:
            result = self._conn.execute(text(query))

            if result.returns_rows:  # rows returned: select
                rows = result.fetchall()
                return ExecuteSQLResult(
                    success=True,
                    error=None,
                    sql_query=query,
                    sql_return=str(rows),
                    row_count=len(rows),
                    result_format="csv",
                )
            else:  # no rows returned: insert, update, delete
                return ExecuteSQLResult(
                    success=True,
                    error=None,
                    sql_query=query,
                    sql_return="",
                    row_count=result.rowcount,
                    result_format="csv",
                )
        except SQLAlchemyError as e:
            return ExecuteSQLResult(
                success=True,
                error=str(e),
                sql_query=query,
                sql_return="",
                row_count=0,
                result_format="csv",
            )
        except Exception as e:
            return ExecuteSQLResult(
                success=False,
                sql_query=query,
                sql_return="",
                row_count=0,
                error=str(e),
                result_format="csv",
            )

    def execute_arrow(self, query: str) -> ExecuteSQLResult:
        """Execute query and return results as Arrow table."""
        self.connect()

        try:
            result = self._conn.execute(text(query))
            if result.returns_rows:
                # TODO: improve the performance of this function with ADBC or remove pandas dependency
                df = DataFrame(result.fetchall(), columns=result.keys())
                table = Table.from_pandas(df)
                return ExecuteSQLResult(
                    success=True,
                    error=None,
                    sql_query=query,
                    sql_return=table,
                    row_count=len(df),
                    result_format="arrow",
                )
            # no rows returned: insert, update, delete
            return ExecuteSQLResult(
                success=True,
                error=None,
                sql_query=query,
                sql_return=result.rowcount,
                row_count=0,
                result_format="arrow",
            )
        except SQLAlchemyError as e:
            return ExecuteSQLResult(
                success=False,
                error=str(e),
                sql_query=query,
                sql_return="",
                row_count=0,
                result_format="arrow",
            )

    def execute_arrow_iterator(self, query: str, max_rows: int = 100) -> Iterator[Tuple]:
        """Execute query and return results as tuples in batches."""
        try:
            self.connect()
            result = self._conn.execute(text(query).execution_options(stream_results=True, max_row_buffer=max_rows))
            if result.returns_rows:
                while True:
                    batch_rows = result.fetchmany(max_rows)
                    if not batch_rows:
                        break
                    for row in batch_rows:
                        yield row
            else:
                # Return empty iterator
                yield from []
        except SQLAlchemyError as e:
            logger.error(f"Error executing arrow iterator: {e}")
            raise e

    def execute_csv_iterator(self, query: str, max_rows: int = 100, with_header: bool = True) -> Iterator[Tuple]:
        """Execute a SQL query and return results as tuples in batches."""
        try:
            self.connect()
            result = self._conn.execute(text(query).execution_options(stream_results=True, max_row_buffer=max_rows))
            if result.returns_rows:
                if with_header:
                    columns = result.keys()
                    yield columns
                while True:
                    batch_rows = result.fetchmany(max_rows)
                    if not batch_rows:
                        break
                    for row in batch_rows:
                        yield row
            else:
                # For non-SELECT queries, yield empty tuple
                if with_header:
                    yield ()
                yield from []
        except SQLAlchemyError as e:
            logger.error(f"Error executing csv iterator: {e}")
            raise e

    def execute_to_arrow_stream(self, query: str, output_stream: Any, compression: Optional[str] = "lz4") -> None:
        """Execute query and stream results as Arrow format."""
        self.connect()
        try:
            result = self._conn.execute(text(query))
            if result.returns_rows:
                while True:
                    batch = result.fetchmany(self.batch_size)
                    if not batch:
                        break
                    df = DataFrame(batch, columns=result.keys())
                    table = Table.from_pandas(df)
                    with ipc.new_stream(output_stream, table.schema) as writer:
                        writer.write_table(table)
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED,
                message_args={
                    "error_message": str(e),
                    "uri": self.connection_string,
                    "sql": query,
                },
            ) from e

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            self.connect()
            self.execute_query("SELECT 1")
            return True
        except DatusException as e:
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "test_connection",
                    "error_message": "Connection failed",
                    "uri": self.connection_string,
                },
            ) from e
        finally:
            self.close()

    # TODO execute_update

    def _execute_query(self, query: str) -> DataFrame:
        result = self._conn.execute(text(query))
        return DataFrame(result.fetchall(), columns=list(result.keys()))

    def insert(self, sql: str) -> Tuple[int, int]:
        """Insert the database.
        Args:
            sql: insert sql
        Returns:
            lastrowid
        """
        self.connect()
        try:
            res = self._conn.execute(text(sql))
            return (res.lastrowid, res.rowcount)
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED,
                message_args={
                    "error_message": str(e),
                    "uri": self.connection_string,
                    "sql": sql,
                },
            ) from e

    def update(self, sql: str) -> int:
        """Update the database.
        Args:
            sql: update sql
        Returns:
            The number of rows updated
        """
        return self._update_or_delete(sql)

    def _update_or_delete(self, sql: str) -> int:
        self.connect()
        try:
            res = self._conn.execute(text(sql))
            return res.rowcount
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED,
                message_args={
                    "error_message": str(e),
                    "uri": self.connection_string,
                    "sql": sql,
                },
            ) from e

    def delete(self, sql: str) -> int:
        """Delete the database.
        Args:
            sql: delete sql
        Returns:
            The number of rows deleted
        """
        return self._update_or_delete(sql)

    def execute_query(self, query: str) -> DataFrame:
        """Execute a query and return the result."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        self.connect()
        try:
            return self._execute_query(query)
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED,
                message_args={
                    "error_message": str(e),
                    "uri": self.connection_string,
                    "sql": query,
                },
            ) from e

    def _execute(self, query: str) -> Any:
        result = self._conn.execute(text(query))
        if result.returns_rows:
            df = DataFrame(result.fetchall(), columns=list(result.keys()))
            return df.to_dict(orient="records")
        else:
            query = query.strip().lower()
            if query.startswith("insert"):
                return result.lastrowid
            elif query.startswith("update") or query.startswith("delete"):
                return result.rowcount
            else:
                return None

    def execute_queries(self, queries: List[str]) -> List[Any]:
        """Execute multiple queries and return results."""
        results = []
        self.connect()
        try:
            for query in queries:
                results.append(self._execute(query))
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED,
                message_args={
                    "error_message": str(e),
                    "uri": self.connection_string,
                    "sql": "\n".join(queries),
                },
            ) from e
        return results

    def get_tables(self, **kwargs) -> List[str]:
        """Get list of tables in the database."""
        schema_name = self.sqlalchemy_schema(**kwargs)
        inspector = self._inspector()

        return inspector.get_table_names(schema=schema_name if schema_name else None)

    def get_schema(self, **kwargs) -> List[Dict[str, Any]]:
        """Get database schema information."""
        schema_name = self.sqlalchemy_schema(**kwargs)
        inspector = self._inspector()
        try:
            schemas: List[Dict[str, Any]] = []
            for table_name in inspector.get_table_names(schema=schema_name):
                columns = inspector.get_columns(table_name=table_name, schema=schema_name)
                schemas.append(
                    {
                        "table": table_name,
                        "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
                    }
                )
            return schemas
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message=f"Get schema error, uri={self.connection_string}, error_mesage={str(e)}",
            ) from e

    def get_views(self, **kwargs) -> List[str]:
        """Get list of views in the database."""
        inspector = self._inspector()
        schema_name = self.sqlalchemy_schema(**kwargs)
        try:
            return inspector.get_view_names(schema=schema_name)
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED, message=f"Get views error, uri={self.connection_string}"
            ) from e

    def _inspector(self) -> Inspector:
        self.connect()
        try:
            return inspect(self.engine)
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "connect",
                    "uri": self.connection_string,
                    "error_message": str(e),
                },
            ) from e

    def sqlalchemy_schema(self, **kwargs) -> Optional[str]:
        """
        Get the schema name from the kwargs for Inspector.
        return None if no schema is specified.
        """
        return kwargs.get("schema_name")

    def get_materialized_views(self, schema_name: Optional[str] = None) -> List[str]:
        """Get list of materialized views in the database."""
        inspector = self._inspector()
        try:
            # Try to get materialized views - this may not be supported by all databases
            if hasattr(inspector, "get_materialized_view_names"):
                return inspector.get_materialized_view_names(schema=schema_name if schema_name else None)
            else:
                # unsupported
                return []
        except Exception as e:
            logger.error(f"Error getting materialized views: {e}")
            return []

    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Get sample values from tables.
        The caller should fill catalog_name, database_name, schema_name themselves.
        """
        catalog_name = kwargs.get("catalog_name", "")
        database_name = kwargs.get("database_name", "")
        schema_name = kwargs.get("schema_name", "")
        self._inspector()
        try:
            samples = []
            if not tables:
                tables = self.get_tables(schema_name=self.sqlalchemy_schema(**kwargs))
            logger.info(f"Getting sample data from tables {tables} LIMIT {top_n}")
            for table_name in tables:
                full_table_name = self.full_name(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                query = f"SELECT * FROM {full_table_name} LIMIT {top_n}"
                result = self.execute_query(query)
                if not result.empty:
                    samples.append(
                        {
                            "identifier": self.identifier(
                                catalog_name=catalog_name,
                                database_name=database_name,
                                schema_name=schema_name,
                                table_name=table_name,
                            ),
                            "catalog_name": catalog_name,
                            "database_name": database_name,
                            "schema_name": schema_name,
                            "table_name": table_name,
                            "sample_rows": result.to_csv(index=False),
                        }
                    )
            return samples
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED, message=f"Database connection error, uri={self.connection_string}"
            ) from e

    def get_columns(self, table_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List[Dict]: List of column information dictionaries
        """
        inspector = self._inspector()
        schema_name = self.sqlalchemy_schema(**kwargs)
        try:
            columns = inspector.get_columns(table_name=table_name, schema=schema_name)

            # Standardize the output format
            return [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None),
                    "primary_key": col.get("primary_key", False),
                }
                for col in columns
            ]
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
            return []

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return self.identifier(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    # Additional methods from SQLAlchemyArrow that can be useful

    def _execute_arrow_batches(self, query: str, params: Optional[Dict[str, Any]] = None) -> Iterator[RecordBatch]:
        """
        Execute a query and yield results as Arrow RecordBatches.
        This is memory-efficient for large result sets.

        Args:
            query: SQL query string or SQLAlchemy text object
            params: Parameters for the query

        Yields:
            Arrow RecordBatch objects, each containing a batch of rows
        """
        self.connect()
        with self._conn.execution_options(stream_results=True).execute(
            text(query) if isinstance(query, str) else query, parameters=params
        ) as result:
            if result.returns_rows:
                while True:
                    batch_rows = result.fetchmany(self.batch_size)
                    if not batch_rows:
                        break
                    batch = self._result_to_arrow_batch(result, batch_rows)
                    if batch is not None:
                        yield batch

    def stream_to_parquet(
        self,
        query: str,
        output_file: str,
        params: Optional[Dict[str, Any]] = None,
        parquet_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Stream query results directly to a Parquet file without loading everything into memory.

        Args:
            query: SQL query string or SQLAlchemy text object
            output_file: Path to output Parquet file
            params: Parameters for the query
            parquet_kwargs: Optional arguments to pass to pyarrow.parquet.write_table
        """
        import pyarrow.parquet as pq

        parquet_kwargs = parquet_kwargs or {}
        schema = None
        writer = None

        try:
            for i, batch in enumerate(self._execute_arrow_batches(query, params)):
                if i == 0:
                    # Initialize schema and writer from the first batch
                    schema = batch.schema
                    writer = pq.ParquetWriter(output_file, schema, **parquet_kwargs)

                writer.write_batch(batch)
        finally:
            if writer:
                writer.close()

    def _get_arrow_type_for_sql_type(self, sql_type_name: str) -> Optional[DataType]:
        """Map SQL types to Arrow types."""
        type_map = {
            "int": DataType.int32(),
            "integer": DataType.int32(),
            "bigint": DataType.int64(),
            "smallint": DataType.int16(),
            "tinyint": DataType.int8(),
            "float": DataType.float32(),
            "real": DataType.float32(),
            "double": DataType.float64(),
            "boolean": DataType.bool_(),
            "bool": DataType.bool_(),
            "varchar": DataType.string(),
            "char": DataType.string(),
            "text": DataType.string(),
            "string": DataType.string(),
            "date": DataType.date32(),
            "timestamp": DataType.timestamp("ns"),
            "datetime": DataType.timestamp("ns"),
            "time": DataType.time32("ms"),
            "binary": DataType.binary(),
            "varbinary": DataType.binary(),
            "blob": DataType.binary(),
            "decimal": DataType.decimal128(38, 10),
        }

        for key in type_map:
            if sql_type_name.lower().startswith(key):
                return type_map[key]
        return None

    def _result_to_arrow_batch(self, result: Result, batch_rows: List[Tuple]) -> RecordBatch:
        """Convert SQL result batch to Arrow RecordBatch."""
        if not batch_rows:
            return None

        column_names = list(result.keys())

        # Use default Arrow types
        columns = list(zip(*batch_rows))
        arrays = [array(column_data) for column_data in columns]

        return RecordBatch.from_arrays(arrays, names=column_names)
