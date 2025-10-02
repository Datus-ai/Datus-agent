from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, override

from pandas import DataFrame
from pyarrow import DataType, RecordBatch, Table, array, ipc
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Inspector, Result
from sqlalchemy.exc import (
    DatabaseError,
    DataError,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    TimeoutError,
)

from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType, SQLType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type

logger = get_logger(__name__)


class SQLAlchemyConnector(BaseSqlConnector):
    """
    SQLAlchemy connector with Arrow support.
    Implements BaseSqlConnector interface.
    """

    def __init__(self, connection_string: str, dialect: str = "", batch_size: int = 1024, timeout_seconds: int = 30):
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
        super().__init__(self.dialect, batch_size, timeout_seconds)
        self.connection_string = connection_string
        self.engine = None
        self.connection = None
        self._owns_engine = False
        self.use_arrow_dtype_mapping = False

    def __del__(self):
        """Destructor to ensure connections are properly closed."""
        try:
            self.close()
        except Exception as e:
            # Ignore any errors during cleanup
            logger.debug(f"Failed to close connection: {str(e)}")

    def _trans_sqlalchemy_exception(
        self, e: Exception, sql: str = None, operation: str = "SQL execution"
    ) -> DatusException:
        if isinstance(e, DatusException):
            return e
        """Map SQLAlchemy exceptions to specific Datus ErrorCode values."""
        # Use .orig attribute to get original database error without SQLAlchemy's background links
        if hasattr(e, "orig") and e.orig is not None:
            error_message = str(e.orig)
        else:
            error_message = str(e)
        message_args = {"error_message": error_message, "sql": sql}

        error_msg_lower = error_message.lower()
        if any(keyword in error_msg_lower for keyword in ["syntax", "parse error", "sql error"]):
            return DatusException(
                ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                message_args=message_args,
            )
        # Connection-related errors
        if isinstance(e, (OperationalError, InterfaceError)):
            # Handle transaction rollback errors specifically
            if any(
                keyword in error_msg_lower
                for keyword in ["can't reconnect until invalid transaction is rolled back", "invalid transaction"]
            ):
                logger.warning("Detected invalid transaction state, resetting connection")
                self._force_reset_connection()
                return DatusException(
                    ErrorCode.DB_TRANSACTION_FAILED,
                    message_args=message_args,
                )
            elif any(keyword in error_msg_lower for keyword in ["timeout", "timed out", "connection timeout"]):
                return DatusException(
                    ErrorCode.DB_CONNECTION_TIMEOUT,
                    message_args=message_args,
                )
            elif any(keyword in error_msg_lower for keyword in ["authentication", "access denied", "login failed"]):
                return DatusException(
                    ErrorCode.DB_AUTHENTICATION_FAILED,
                    message_args=message_args,
                )
            elif any(keyword in error_msg_lower for keyword in ["permission denied", "insufficient privilege"]):
                message_args["operation"] = operation
                return DatusException(
                    ErrorCode.DB_PERMISSION_DENIED,
                    message_args=message_args,
                )
            elif any(keyword in error_msg_lower for keyword in ["syntax", "parse error", "sql error"]):
                return DatusException(
                    ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                    message_args=message_args,
                )
            elif any(
                keyword in error_msg_lower
                for keyword in ["connection refused", "connection failed", "connect", "unable to open database"]
            ):
                return DatusException(ErrorCode.DB_CONNECTION_FAILED, message_args=message_args)
            else:
                return DatusException(
                    ErrorCode.DB_EXECUTION_ERROR,
                    message_args=message_args,
                )

        # SQL syntax and programming errors
        elif isinstance(e, ProgrammingError):
            if any(keyword in error_msg_lower for keyword in ["syntax", "parse error", "sql error"]):
                return DatusException(
                    ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                    message_args=message_args,
                )
            else:
                return DatusException(
                    ErrorCode.DB_EXECUTION_ERROR,
                    message_args=message_args,
                )

        # Integrity constraint violations
        elif isinstance(e, IntegrityError):
            return DatusException(
                ErrorCode.DB_CONSTRAINT_VIOLATION,
                message_args=message_args,
            )

        # Timeout errors
        elif isinstance(e, TimeoutError):
            return DatusException(
                ErrorCode.DB_EXECUTION_TIMEOUT,
                message_args=message_args,
            )

        # Other database errors
        elif isinstance(e, (DatabaseError, DataError, InternalError, NotSupportedError)):
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args=message_args,
            )

        # Fallback to generic SQLAlchemy error
        else:
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args=message_args,
            )

    def _extract_table_name_from_error(self, error_message: str) -> Optional[str]:
        """Extract table name from SQLAlchemy error message."""
        import re

        patterns = [
            r"table ['\"]([^'\"]+)['\"]",
            r"relation ['\"]([^'\"]+)['\"]",
            r"table (\w+)",
            r"relation (\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_column_name_from_error(self, error_message: str) -> Optional[str]:
        """Extract column name from SQLAlchemy error message."""
        import re

        patterns = [
            r"column ['\"]([^'\"]+)['\"]",
            r"column (\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_schema_name_from_error(self, error_message: str) -> Optional[str]:
        """Extract schema name from SQLAlchemy error message."""
        import re

        patterns = [
            r"schema ['\"]([^'\"]+)['\"]",
            r"database ['\"]([^'\"]+)['\"]",
            r"schema (\w+)",
            r"database (\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @override
    def connect(self):
        """Establish connection to the database."""
        if self.engine and self.connection and self._owns_engine:
            return

        try:
            # Clean up any existing connection
            self._safe_close()

            # Create engine based on dialect
            if self.dialect not in (DBType.DUCKDB, DBType.SQLITE):
                self.engine = create_engine(
                    self.connection_string,
                    pool_size=3,
                    max_overflow=5,
                    pool_timeout=self.timeout_seconds * 1000,
                    pool_recycle=3600,
                )
            else:
                self.engine = create_engine(self.connection_string)

            # Create connection
            self.connection = self.engine.connect().execution_options(statement_timeout=self.timeout_seconds * 1000)
            self._owns_engine = True

        except Exception as e:
            # Clean up on failure
            self._force_reset_connection()
            raise self._trans_sqlalchemy_exception(e, sql="", operation="CONNECTION_INITIALIZATION") from e

        if not (self.engine and self.connection):
            self._force_reset_connection()
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": "Failed to establish engine and connection"},
            )

    @override
    def close(self):
        """Close the database connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            if self.engine:
                self.engine.dispose()
                self.engine = None
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")

    def reset_connection(self):
        """Reset the database connection to clear any stuck transaction state."""
        try:
            # Force close and dispose of existing connection and engine
            if self.connection:
                self._safe_rollback()
                try:
                    self.connection.close()
                except Exception:
                    logger.debug(f"Close failed for {self.connection}")
                self.connection = None

            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    logger.debug(f"Dispose failed for {self.engine}")
                self.engine = None

            self._owns_engine = False
            logger.info("Database connection has been reset")
        except Exception as e:
            logger.warning(f"Error during connection reset: {str(e)}")

    def _safe_rollback(self, operation_name: str = "operation"):
        """Safely rollback transaction with error logging."""
        if self.connection:
            try:
                self.connection.rollback()
            except Exception:
                logger.warning(f"Failed to rollback transaction after {operation_name} error")

    def _handle_sql_exception(self, e: Exception, sql: str = None, operation: str = "SQL execution"):
        """Handle SQL exceptions with rollback and proper error mapping."""
        self._safe_rollback(operation)
        return self._trans_sqlalchemy_exception(e, sql, operation)

    def validate_input(self, input_params: Any):
        """Validate the input parameters before execution."""
        super().validate_input(input_params)

    @override
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        try:
            self.connect()
            res = self.connection.execute(text(sql))
            return ExecuteSQLResult(success=True, sql_query=sql, sql_return=str(res.rowcount), row_count=res.rowcount)
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(success=False, sql_query=sql, error=str(ex))

    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        try:
            df = self._execute_pandas(sql)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=df,
                row_count=len(df),
                result_format="pandas",
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(success=False, error=str(ex), sql_query=sql)

    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results with csv format."""

        try:
            self.connect()
            df = self._execute_pandas(sql)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=df.to_csv(index=False),
                row_count=len(df),
                result_format="csv",
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                sql_query=sql,
                sql_return="",
                row_count=0,
                error=str(ex),
                result_format="csv",
            )

    def execute_arrow(self, query: str) -> ExecuteSQLResult:
        """Execute query and return results as Arrow table."""

        try:
            self.connect()
            result = self.connection.execute(text(query))
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
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, query)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=query,
                sql_return="",
                row_count=0,
                result_format="arrow",
            )

    def execute_arrow_iterator(self, sql: str, max_rows: int = 100) -> Iterator[Tuple]:
        """Execute query and return results as tuples in batches."""
        self.connect()
        try:
            result = self.connection.execute(text(sql).execution_options(stream_results=True, max_row_buffer=max_rows))
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
        except Exception as e:
            if isinstance(e, DatusException):
                raise e
            raise self._trans_sqlalchemy_exception(e) from e

    def execute_csv_iterator(self, query: str, max_rows: int = 100, with_header: bool = True) -> Iterator[Tuple]:
        """Execute a SQL query and return results as tuples in batches."""
        self.connect()
        try:
            result = self.connection.execute(
                text(query).execution_options(stream_results=True, max_row_buffer=max_rows)
            )
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
        except Exception as e:
            raise self._trans_sqlalchemy_exception(e) from e

    def execute_to_arrow_stream(self, query: str, output_stream: Any, compression: Optional[str] = "lz4") -> None:
        """Execute query and stream results as Arrow format."""
        self.connect()
        try:
            result = self.connection.execute(text(query))
            if result.returns_rows:
                while True:
                    batch = result.fetchmany(self.batch_size)
                    if not batch:
                        break
                    df = DataFrame(batch, columns=result.keys())
                    table = Table.from_pandas(df)
                    with ipc.new_stream(output_stream, table.schema) as writer:
                        writer.write_table(table)
        except Exception as e:
            raise self._trans_sqlalchemy_exception(e, query) from e

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            self.connect()
            self._execute_query("SELECT 1")
            return True
        except DatusException:
            # Reset connection on test failure to clear any invalid transaction state
            self._safe_close()
            raise
        except Exception as e:
            # Reset connection on test failure to clear any invalid transaction state
            self._safe_close()
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={
                    "error_message": "Connection failed during test",
                },
            ) from e
        finally:
            self._safe_close()

    def _safe_close(self):
        """Safely close connection, ignoring any errors."""
        try:
            self.close()
        except Exception:
            pass

    def _force_reset_connection(self):
        """Force reset connection, used internally by exception handlers."""
        try:
            self.reset_connection()
        except Exception:
            # If reset fails, at least try to clear the references
            self.connection = None
            self.engine = None
            self._owns_engine = False

    @override
    def execute_insert(
        self,
        sql: str,
    ) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement.

        Args:
            sql: The INSERT SQL statement to execute

        Returns:
            A dictionary containing the insert operation results
        """

        try:
            self.connect()
            res = self.connection.execute(text(sql))
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(res.lastrowid),
                row_count=res.rowcount,
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
                sql_return="",
                row_count=0,
            )

    def execute_update(
        self,
        sql: str,
    ) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement.

        Args:
            sql: The UPDATE SQL statement to execute

        Returns:
            A dictionary containing the update operation results
        """
        try:
            self.connect()
            rowcount = self._update(sql)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(rowcount),
                row_count=rowcount,
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
                sql_return="",
                row_count=0,
            )

    @override
    def execute_delete(
        self,
        sql: str,
    ) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement.

        Args:
            sql: The DELETE SQL statement to execute

        Returns:
            A dictionary containing the delete operation results
        """

        try:
            self.connect()
            rowcount = self.delete(sql)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(rowcount),
                row_count=rowcount,
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
                sql_return="",
                row_count=0,
            )

    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        try:
            self.connect()
            result = self._execute_query(sql)
            row_count = len(result)
            if result_format == "csv":
                df = DataFrame(result)
                result = df.to_csv(index=False)
            elif result_format == "arrow":
                result = Table.from_pylist(result)
            elif result_format == "pandas":
                result = DataFrame(result)

            return ExecuteSQLResult(
                success=True, sql_query=sql, sql_return=result, row_count=row_count, result_format=result_format
            )
        except Exception as e:
            if isinstance(e, DatusException):
                ex = e
            else:
                ex = self._trans_sqlalchemy_exception(e, sql)
            return ExecuteSQLResult(success=False, error=str(ex), sql_query=sql)

    def _execute_query(self, query_sql: str) -> List[Dict[str, Any]]:
        if parse_sql_type(query_sql, self.dialect) in (
            SQLType.INSERT,
            SQLType.UPDATE,
            SQLType.DELETE,
            SQLType.MERGE,
            SQLType.CONTENT_SET,
            SQLType.UNKNOWN,
        ):
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR, message="Only supports normal queries and metadata queries."
            )
        self.connect()
        try:
            result = self.connection.execute(text(query_sql))
            rows = result.fetchall()
            return [row._asdict() for row in rows]
        except DatusException:
            raise
        except Exception as e:
            raise self._trans_sqlalchemy_exception(e, query_sql, "query execution") from e

    def _execute_pandas(self, query_sql: str) -> DataFrame:
        return DataFrame(self._execute_query(query_sql))

    @override
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        schemas = self._inspector().get_schema_names()
        if not include_sys:
            system_schemas = self._sys_schemas()
            # Filter out common system schemas
            schemas = [s for s in schemas if s.lower() not in system_schemas]
        return schemas

    def _update(self, sql: str) -> int:
        """Update the database.
        Args:
            sql: update sql
        Returns:
            The number of rows updated
        """
        return self._update_or_delete(sql, "update")

    def _update_or_delete(self, sql: str, operation: str) -> int:
        self.connect()
        try:
            res = self.connection.execute(text(sql))
            return res.rowcount
        except Exception as e:
            raise self._handle_sql_exception(e, sql, operation) from e

    def delete(self, sql: str) -> int:
        """Delete the database.
        Args:
            sql: delete sql
        Returns:
            The number of rows deleted
        """
        return self._update_or_delete(sql, "delete")

    def _execute(self, query: str) -> Any:
        result = self.connection.execute(text(query))
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
            raise self._handle_sql_exception(e, "\n".join(queries), "batch query execution") from e
        return results

    @override
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        self.connect()
        try:
            self.connection.execute(text(sql_query))
            return ExecuteSQLResult(success=True, sql_query=sql_query, sql_return="Successful", row_count=0)
        except Exception as e:
            ex = self._handle_sql_exception(e, sql_query)
            return ExecuteSQLResult(success=False, error=str(ex), sql_query=sql_query)

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get list of tables in the database."""
        self.connect()
        sqlalchemy_schema = self.sqlalchemy_schema(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        inspector = self._inspector()

        return inspector.get_table_names(schema=sqlalchemy_schema)

    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, Any]]:
        """Get Table schema information."""
        sqlalchemy_schema = self.sqlalchemy_schema(
            catalog_name=catalog_name or self.catalog_name,
            database_name=database_name or self.database_name,
            schema_name=schema_name or self.schema_name,
        )
        inspector = self._inspector()
        try:
            schemas: List[Dict[str, Any]] = []
            pk_columns = set(
                inspector.get_pk_constraint(table_name=table_name, schema=sqlalchemy_schema)["constrained_columns"]
            )
            columns = inspector.get_columns(table_name=table_name, schema=sqlalchemy_schema)
            for i, col in enumerate(columns):
                schemas.append(
                    {
                        "cid": i,
                        "name": col["name"],
                        "type": str(col["type"]),
                        "comment": str(col["comment"]) if "comment" in col else None,
                        "nullable": col["nullable"],
                        "pk": col["name"] in pk_columns,
                        "default_value": col["default"],
                    }
                )
            # schemas.append(
            #     {
            #         "table": table_name,
            #         "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
            #     }
            # )

            return schemas
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_FAILED,
                message_args={
                    "operation": "get_schema",
                    "error_message": str(e),
                },
            ) from e

    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get list of views in the database."""
        inspector = self._inspector()
        sqlalchemy_schema = self.sqlalchemy_schema(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        try:
            return inspector.get_view_names(schema=sqlalchemy_schema)
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_FAILED,
                message_args={
                    "operation": "get_views",
                    "error_message": str(e),
                },
            ) from e

    def _inspector(self) -> Inspector:
        self.connect()
        try:
            return inspect(self.engine)
        except Exception as e:
            raise self._trans_sqlalchemy_exception(e, operation="Connection") from e

    def sqlalchemy_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> Optional[str]:
        """
        Get the schema name from the kwargs for Inspector.
        return None if no schema is specified.
        """
        return database_name or schema_name

    def get_materialized_views(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
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
            logger.error(f"Error getting materialized views: {str(e)}")
            return []

    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[Dict[str, str]]:
        """
        Get sample values from tables.
        The caller should fill catalog_name, database_name, schema_name themselves.
        """
        self._inspector()
        try:
            samples = []
            if not tables:
                tables = self.get_tables(
                    catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
                )
            logger.info(f"Getting sample data from tables {tables} LIMIT {top_n}")
            for table_name in tables:
                full_table_name = self.full_name(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                query = f"SELECT * FROM {full_table_name} LIMIT {top_n}"
                result = self._execute_pandas(query)
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
        except DatusException:
            raise
        except Exception as e:
            if isinstance(e, DatusException):
                raise e
            raise self._trans_sqlalchemy_exception(e) from e

    def get_columns(
        self, table_name: str, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List[Dict]: List of column information dictionaries
        """
        inspector = self._inspector()
        sqlalchemy_schema = self.sqlalchemy_schema(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        try:
            columns = inspector.get_columns(table_name=table_name, schema=sqlalchemy_schema)

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
        with self.connection.execution_options(stream_results=True).execute(
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
