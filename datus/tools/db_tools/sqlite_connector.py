# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import sqlite3
from typing import Any, Dict, List, Literal, Optional, override

from pandas import DataFrame
from pyarrow import Table

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.config import SQLiteConfig
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLiteConnector(BaseSqlConnector):
    """
    Connector for SQLite databases using native sqlite3 SDK.
    """

    def __init__(self, config: SQLiteConfig):
        super().__init__(config, dialect=DBType.SQLITE)
        self.db_path = config.db_path.replace("sqlite:///", "")
        self.check_same_thread = config.check_same_thread
        self.connection: Optional[sqlite3.Connection] = None

        if config.database_name:
            self.database_name = config.database_name
        else:
            from datus.configuration.agent_config import file_stem_from_uri

            self.database_name = file_stem_from_uri(self.db_path)

    @override
    def connect(self):
        """Establish connection to SQLite database."""
        if self.connection:
            return

        try:
            self.connection = sqlite3.connect(
                self.db_path,
                timeout=self.timeout_seconds,
                check_same_thread=self.check_same_thread,
            )
            self.connection.row_factory = sqlite3.Row
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e

    @override
    def close(self):
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")
            finally:
                self.connection = None

    @override
    def test_connection(self) -> bool:
        """Test the database connection."""
        opened_here = self.connection is None
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e
        finally:
            if opened_here:
                self.close()

    def _handle_exception(self, e: Exception, sql: str = "") -> DatusException:
        """Handle SQLite exceptions and map to appropriate Datus ErrorCode."""
        if isinstance(e, DatusException):
            return e

        error_msg = str(e).lower()

        if isinstance(e, sqlite3.OperationalError):
            if "syntax error" in error_msg or "near" in error_msg:
                return DatusException(
                    ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                    message_args={"sql": sql, "error_message": str(e)},
                )
            elif "no such table" in error_msg:
                return DatusException(
                    ErrorCode.DB_TABLE_NOT_EXISTS,
                    message_args={"table_name": sql, "error_message": str(e)},
                )
            elif "locked" in error_msg or "database is locked" in error_msg:
                return DatusException(
                    ErrorCode.DB_CONNECTION_TIMEOUT,
                    message_args={"error_message": str(e)},
                )
            else:
                return DatusException(
                    ErrorCode.DB_EXECUTION_ERROR,
                    message_args={"sql": sql, "error_message": str(e)},
                )
        elif isinstance(e, sqlite3.IntegrityError):
            return DatusException(
                ErrorCode.DB_CONSTRAINT_VIOLATION,
                message_args={"sql": sql, "error_message": str(e)},
            )
        elif isinstance(e, sqlite3.ProgrammingError):
            return DatusException(
                ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )
        else:
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )

    @override
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.lastrowid),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.rowcount),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.rowcount),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        """Execute a DDL SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SELECT query."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert to list of dicts
            result_list = [dict(zip(columns, row)) for row in rows]
            row_count = len(rows)

            # Explicitly pass columns to preserve schema for empty results
            df = DataFrame(result_list, columns=columns)

            if result_format == "csv":
                result = df.to_csv(index=False)
            elif result_format == "arrow":
                result = Table.from_pandas(df)
            elif result_format == "pandas":
                result = df
            else:  # list
                result = result_list

            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=result,
                row_count=row_count,
                result_format=result_format,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return pandas DataFrame."""
        return self.execute_query(sql, result_format="pandas")

    @override
    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return CSV format."""
        return self.execute_query(sql, result_format="csv")

    @override
    def execute_queries(self, queries: List[str]) -> List[Any]:
        """Execute multiple queries."""
        results = []
        self.connect()
        try:
            for query in queries:
                cursor = self.connection.cursor()
                cursor.execute(query)
                if cursor.description:
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    results.append([dict(zip(columns, row)) for row in rows])
                else:
                    results.append(cursor.rowcount)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise self._handle_exception(e, "\n".join(queries))
        return results

    @override
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """Execute SET/USE commands (SQLite doesn't support these)."""
        return ExecuteSQLResult(
            success=True,
            sql_query=sql_query,
            sql_return="SQLite does not support context switching",
            row_count=0,
        )

    @override
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names."""
        self.connect()
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return [row[0] for row in cursor.fetchall()]

    @override
    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names."""
        self.connect()
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [row[0] for row in cursor.fetchall()]

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return f'"{table_name}"'

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """SQLite does not support switch context"""

    def _get_schema_with_ddl(
        self, database_name: str = "", table_type: str = "table", filter_tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get schema with DDL for tables or views."""
        self.connect()
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='{table_type}'")

        schema_list = []
        for row in cursor.fetchall():
            table_name = row[0]
            definition = row[1]

            # Skip SQLite system tables
            if table_name.startswith("sqlite_"):
                continue

            if filter_tables and table_name not in filter_tables:
                continue

            schema_list.append(
                {
                    "identifier": self.identifier(
                        database_name=database_name,
                        table_name=table_name,
                    ),
                    "catalog_name": "",
                    "database_name": database_name,
                    "schema_name": "",
                    "table_name": table_name,
                    "definition": definition,
                    "table_type": table_type,
                }
            )

        return schema_list

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get tables with DDL definitions."""
        return self._get_schema_with_ddl(
            database_name=database_name or self.database_name,
            table_type="table",
            filter_tables=tables,
        )

    @override
    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get views with DDL definitions."""
        return self._get_schema_with_ddl(
            database_name=database_name or self.database_name,
            table_type="view",
        )

    @override
    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        if not table_name:
            return []

        self.connect()
        cursor = self.connection.cursor()
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()
            return [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "nullable": not bool(col[3]),  # Normalize: invert notnull to nullable
                    "default_value": col[4],  # Normalize: dflt_value to default_value
                    "pk": col[5],
                }
                for col in columns
            ]
        except Exception as e:
            raise self._handle_exception(e, f'PRAGMA table_info("{table_name}")')

    @override
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, Any]]:
        """Get sample rows from tables."""
        self.connect()
        samples = []

        if tables:
            for table_name in tables:
                try:
                    cursor = self.connection.cursor()
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {top_n}')
                    rows = cursor.fetchall()
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        df = DataFrame([dict(zip(columns, row)) for row in rows])
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": database_name or self.database_name,
                                "schema_name": "",
                                "table_name": table_name,
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get sample rows for table {table_name}: {e}")
        else:
            # Get all tables/views
            all_tables = []
            if table_type in ("full", "table"):
                all_tables.extend(self.get_tables())
            if table_type in ("full", "view"):
                all_tables.extend(self.get_views())

            for table_name in all_tables:
                try:
                    cursor = self.connection.cursor()
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {top_n}')
                    rows = cursor.fetchall()
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        df = DataFrame([dict(zip(columns, row)) for row in rows])
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": database_name or self.database_name,
                                "schema_name": "",
                                "table_name": table_name,
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get sample rows for table {table_name}: {e}")

        return samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary."""
        return {"db_type": DBType.SQLITE, "db_path": self.db_path}

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """SQLite has only one database (the file itself)."""
        return ["main"]

    def get_type(self) -> str:
        return DBType.SQLITE
