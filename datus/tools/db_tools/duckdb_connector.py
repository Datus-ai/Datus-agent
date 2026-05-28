# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import functools
import re
import threading
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypeVar, override

import duckdb
from datus_db_core import BaseSqlConnector, SchemaNamespaceMixin, list_to_in_str
from pydantic import BaseModel, Field

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools._migration_compat import MigrationTargetMixin
from datus.tools.db_tools.config import DuckDBConfig
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _DBMetadataNames(BaseModel):
    """
    The corresponding database commands are SHOW/SHOW CREAT/INFORMATION_SCHEMA.<TABLES>
    """

    info_table: str = Field(..., init=True, description="The name of metadata table")
    name_field: str = Field(..., init=True, description="Fields corresponding to names in metadata table")
    has_sql_field: bool = Field(True, init=True, description="Is there a SQL field.")


METADATA_DICT: Dict[str, _DBMetadataNames] = {
    "database": _DBMetadataNames(info_table="duckdb_databases", name_field="database_name", has_sql_field=False),
    "schema": _DBMetadataNames(info_table="duckdb_schemas", name_field="schema_name", has_sql_field=True),
    "table": _DBMetadataNames(info_table="duckdb_tables", name_field="table_name", has_sql_field=True),
    "view": _DBMetadataNames(info_table="duckdb_views", name_field="view_name", has_sql_field=True),
}


def _metadata_names(_type: str) -> _DBMetadataNames:
    if _type not in METADATA_DICT:
        raise DatusException(ErrorCode.COMMON_FIELD_INVALID, f"Invalid type `{_type}` for Database table type")
    return METADATA_DICT[_type]


_F = TypeVar("_F", bound=Callable[..., Any])


def _serialised(method: _F) -> _F:
    """Serialise calls that touch ``self.connection`` against the connector's RLock.

    ``DuckDBPyConnection`` is not thread-safe — concurrent ``execute()``
    on the same instance races on the shared result/statement state and
    manifests as ``description=None``, empty rows, or DuckDB's NULL-
    shared_ptr INTERNAL Error (and in our local repro, a Python SIGSEGV).
    Dashboard's multi-chart fan-out hits this hard because every chart's
    filter query lands on the same cached connector via ``asyncio.to_thread``.

    Applied as a decorator (rather than ``with self._lock:`` inside each
    method body) so the lock change is one line per method instead of a
    full re-indent — the connector's body stays diffable and reviewable.
    """

    @functools.wraps(method)
    def wrapper(self: "DuckdbConnector", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class DuckdbConnector(BaseSqlConnector, SchemaNamespaceMixin, MigrationTargetMixin):
    """
    Connector for DuckDB databases with schema support using native DuckDB SDK.
    """

    def __init__(self, config: DuckDBConfig):
        super().__init__(config, dialect=DBType.DUCKDB)
        self.db_path = config.db_path.replace("duckdb:///", "")
        # ``RLock`` so same-thread re-entry (e.g. ``get_sample_rows`` →
        # ``get_tables_with_ddl`` → ``_get_meta_with_ddl``) doesn't deadlock.
        self._lock = threading.RLock()
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self.enable_external_access = config.enable_external_access
        self.memory_limit = config.memory_limit
        self.read_only = config.read_only
        self.iceberg_config = config.iceberg or {}

        if config.database_name:
            self.database_name = config.database_name
        else:
            from datus.configuration.agent_config import file_stem_from_uri

            self.database_name = file_stem_from_uri(self.db_path)

    @override
    def connect(self):
        """Establish connection to DuckDB database."""
        # Double-checked locking: skip the lock on the hot path (already
        # connected) and only acquire it for the first-time race window.
        if self.connection:
            return
        with self._lock:
            if self.connection:
                return
            self._connect_locked()

    def _connect_locked(self):
        try:
            # Align with the `custom_user_agent` that duckdb_engine auto-injects on every connect:
            # without this, any same-process SQLAlchemy+duckdb_engine client (metricflow validator,
            # dbt, etc.) hits DuckDB's config-consistency check and fails with
            # "Can't open a connection to same database file with a different configuration".
            try:
                import sqlalchemy as _sa
                from duckdb_engine import __version__ as _de_ver

                config = {"custom_user_agent": f"duckdb_engine/{_de_ver}(sqlalchemy/{_sa.__version__})"}
                self.connection = duckdb.connect(self.db_path, read_only=self.read_only, config=config)
            except ImportError:
                self.connection = duckdb.connect(self.db_path, read_only=self.read_only)

            # Per-session settings — kept as post-connect SET so they don't enter
            # DuckDB's instance-config equality check.
            if self.memory_limit:
                self.connection.execute(f"SET memory_limit='{self.memory_limit}'")

            if not self.enable_external_access:
                self.connection.execute("SET enable_external_access=false")

            if self.iceberg_config:
                self._setup_iceberg()

        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e

    @staticmethod
    def _sql_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _safe_identifier(value: Any, field_name: str) -> str:
        text = str(value)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message_args={
                    "field_name": field_name,
                    "except_values": "SQL identifier matching [A-Za-z_][A-Za-z0-9_]*",
                    "your_value": text,
                },
            )
        return text

    @staticmethod
    def _bool_value(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _s3_use_ssl(cls, cfg: Dict[str, Any], endpoint: Any) -> bool:
        configured = cfg.get("s3_use_ssl")
        if configured is not None and configured != "":
            return cls._bool_value(configured)
        if not endpoint:
            return True
        endpoint_text = str(endpoint).strip().lower()
        return endpoint_text.startswith("https://")

    def _load_extension(self, name: str) -> None:
        try:
            self.connection.execute(f"LOAD {name}")
        except Exception as load_error:
            try:
                self.connection.execute(f"INSTALL {name}")
                self.connection.execute(f"LOAD {name}")
            except Exception as retry_error:
                raise RuntimeError(
                    f"Failed to load DuckDB extension {name}; "
                    f"initial LOAD error: {load_error}; INSTALL/LOAD retry error: {retry_error}"
                ) from retry_error

    @classmethod
    def _sql_option(cls, name: str, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return f"{name} " + ("true" if value else "false")
        return f"{name} {cls._sql_literal(value)}"

    @staticmethod
    def _rewrite_iceberg_create_or_replace_table(sql: str) -> Optional[str]:
        quoted_identifier = r'"(?:[^"]|"")+"'
        identifier = rf"(?:[A-Za-z_][A-Za-z0-9_]*|{quoted_identifier})"
        match = re.match(
            r"^\s*CREATE\s+OR\s+REPLACE\s+TABLE\s+"
            rf"({identifier}(?:\s*\.\s*{identifier})*)\s+(.*)\Z",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        table_name = match.group(1)
        remainder = match.group(2)
        return f"DROP TABLE IF EXISTS {table_name};\nCREATE TABLE {table_name} {remainder}"

    def _create_iceberg_secret_if_needed(self, cfg: Dict[str, Any]) -> Optional[str]:
        """Create a DuckDB Iceberg catalog secret when auth credentials are supplied."""
        if not self.connection:
            return None

        secret_name = cfg.get("iceberg_secret_name") or cfg.get("catalog_secret_name") or cfg.get("secret_name")
        create_secret = self._bool_value(cfg.get("create_iceberg_secret"), default=True)
        credential_keys = {
            "client_id",
            "client_secret",
            "oauth2_server_uri",
            "oauth2_scope",
            "oauth2_grant_type",
            "token",
        }
        has_credentials = any(cfg.get(key) for key in credential_keys)
        if not secret_name and has_credentials and create_secret:
            secret_name = "datus_iceberg"
        if not secret_name:
            return None

        secret_name = self._safe_identifier(secret_name, "iceberg_secret_name")
        if not create_secret:
            return secret_name

        secret_option_map = {
            "client_id": "CLIENT_ID",
            "client_secret": "CLIENT_SECRET",
            "oauth2_server_uri": "OAUTH2_SERVER_URI",
            "oauth2_scope": "OAUTH2_SCOPE",
            "oauth2_grant_type": "OAUTH2_GRANT_TYPE",
            "token": "TOKEN",
        }
        secret_parts = ["TYPE iceberg"]
        for cfg_key, sql_name in secret_option_map.items():
            option = self._sql_option(sql_name, cfg.get(cfg_key))
            if option:
                secret_parts.append(option)
        if len(secret_parts) == 1:
            return None

        self.connection.execute("CREATE OR REPLACE SECRET " + secret_name + " (" + ", ".join(secret_parts) + ")")
        return secret_name

    def _setup_iceberg(self) -> None:
        """Attach an Iceberg REST catalog for DuckDB-as-query-engine mode."""
        if not self.connection:
            return

        cfg = self.iceberg_config
        catalog_alias = self._safe_identifier(cfg.get("catalog_alias") or "lake", "catalog_alias")
        catalog_uri = cfg.get("catalog_uri") or cfg.get("iceberg_catalog_uri") or cfg.get("endpoint")
        warehouse = cfg.get("warehouse")
        if not catalog_uri:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": "DuckDB iceberg config requires catalog_uri"},
            )
        if not warehouse:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": "DuckDB iceberg config requires warehouse"},
            )

        self._load_extension("httpfs")
        self._load_extension("iceberg")

        iceberg_secret_name = self._create_iceberg_secret_if_needed(cfg)

        key_id = cfg.get("s3_access_key_id") or cfg.get("aws_access_key_id")
        secret = cfg.get("s3_secret_access_key") or cfg.get("aws_secret_access_key")
        region = cfg.get("s3_region") or cfg.get("aws_region") or "us-east-1"
        provider = str(cfg.get("s3_provider") or cfg.get("aws_provider") or "").strip().lower()
        endpoint = cfg.get("s3_endpoint")
        url_style = cfg.get("s3_url_style") or cfg.get("url_style")
        use_ssl = self._s3_use_ssl(cfg, endpoint)

        if provider:
            provider = self._safe_identifier(provider, "s3_provider")
            secret_parts = [
                "TYPE s3",
                f"PROVIDER {provider}",
                f"REGION {self._sql_literal(region)}",
            ]
            if endpoint:
                endpoint_value = str(endpoint).removeprefix("http://").removeprefix("https://")
                secret_parts.append(f"ENDPOINT {self._sql_literal(endpoint_value)}")
            if url_style:
                secret_parts.append(f"URL_STYLE {self._sql_literal(url_style)}")
            secret_parts.append("USE_SSL " + ("true" if use_ssl else "false"))
            self.connection.execute("CREATE OR REPLACE SECRET datus_s3 (" + ", ".join(secret_parts) + ")")
        elif key_id and secret:
            secret_parts = [
                "TYPE s3",
                f"KEY_ID {self._sql_literal(key_id)}",
                f"SECRET {self._sql_literal(secret)}",
                f"REGION {self._sql_literal(region)}",
            ]
            if endpoint:
                endpoint_value = str(endpoint).removeprefix("http://").removeprefix("https://")
                secret_parts.append(f"ENDPOINT {self._sql_literal(endpoint_value)}")
            if url_style:
                secret_parts.append(f"URL_STYLE {self._sql_literal(url_style)}")
            secret_parts.append("USE_SSL " + ("true" if use_ssl else "false"))
            self.connection.execute("CREATE OR REPLACE SECRET datus_s3 (" + ", ".join(secret_parts) + ")")

        attach_options = [
            "TYPE iceberg",
            f"ENDPOINT {self._sql_literal(catalog_uri)}",
        ]
        if iceberg_secret_name:
            attach_options.append(f"SECRET {iceberg_secret_name}")

        option_aliases = {
            "endpoint_type": "ENDPOINT_TYPE",
            "default_region": "DEFAULT_REGION",
            "authorization_type": "AUTHORIZATION_TYPE",
            "access_delegation_mode": "ACCESS_DELEGATION_MODE",
            "support_nested_namespaces": "SUPPORT_NESTED_NAMESPACES",
            "support_stage_create": "SUPPORT_STAGE_CREATE",
            "max_table_staleness": "MAX_TABLE_STALENESS",
            "purge_requested": "PURGE_REQUESTED",
        }
        has_catalog_auth = bool(iceberg_secret_name or cfg.get("authorization_type") or cfg.get("endpoint_type"))
        if not has_catalog_auth:
            if not cfg.get("authorization_type"):
                attach_options.append("AUTHORIZATION_TYPE none")
            if not cfg.get("access_delegation_mode"):
                attach_options.append("ACCESS_DELEGATION_MODE none")
        for cfg_key, sql_name in option_aliases.items():
            option = self._sql_option(sql_name, cfg.get(cfg_key))
            if option:
                attach_options.append(option)
        attach_options.append(
            "READ_ONLY " + ("true" if self._bool_value(cfg.get("read_only"), default=False) else "false")
        )
        self.connection.execute(
            f"ATTACH {self._sql_literal(warehouse)} AS {catalog_alias} (" + ", ".join(attach_options) + ")"
        )

    @override
    @_serialised
    def close(self):
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")
            finally:
                self.connection = None

    @override
    @_serialised
    def test_connection(self) -> bool:
        """Test the database connection."""
        opened_here = self.connection is None
        try:
            self.connect()
            self.connection.execute("SELECT 1").fetchone()
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
        """Handle DuckDB exceptions and map to appropriate Datus ErrorCode."""
        if isinstance(e, DatusException):
            return e

        error_msg = str(e).lower()

        # Check for common error patterns
        if "syntax error" in error_msg or "parser error" in error_msg:
            return DatusException(
                ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )
        elif "table" in error_msg and "does not exist" in error_msg:
            return DatusException(
                ErrorCode.DB_TABLE_NOT_EXISTS,
                message_args={"table_name": sql, "error_message": str(e)},
            )
        elif "constraint" in error_msg or "unique" in error_msg:
            return DatusException(
                ErrorCode.DB_CONSTRAINT_VIOLATION,
                message_args={"sql": sql, "error_message": str(e)},
            )
        elif "timeout" in error_msg:
            return DatusException(
                ErrorCode.DB_EXECUTION_TIMEOUT,
                message_args={"sql": sql, "error_message": str(e)},
            )
        else:
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )

    @override
    @_serialised
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    @_serialised
    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    @_serialised
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    @_serialised
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        """Execute a DDL SQL statement.

        DuckDB-Iceberg does not natively support CREATE OR REPLACE TABLE.
        For that extension, the connector emulates it with DROP + CREATE
        inside an explicit transaction.
        """
        try:
            self.connect()
            try:
                self.connection.execute(sql)
            except Exception as e:
                rewritten_sql = None
                if self.iceberg_config and "create or replace not supported in duckdb-iceberg" in str(e).lower():
                    rewritten_sql = self._rewrite_iceberg_create_or_replace_table(sql)
                if rewritten_sql is None:
                    raise
                try:
                    self.connection.execute("BEGIN")
                    self.connection.execute(rewritten_sql)
                    self.connection.execute("COMMIT")
                except Exception as rewrite_error:
                    try:
                        self.connection.execute("ROLLBACK")
                    except Exception as rollback_error:
                        logger.warning(
                            "Failed to rollback DuckDB-Iceberg CREATE OR REPLACE emulation: %s",
                            rollback_error,
                        )
                    raise RuntimeError(
                        "DuckDB-Iceberg CREATE OR REPLACE TABLE emulation failed; "
                        f"original error: {e}; rewrite error: {rewrite_error}"
                    ) from e
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
    @_serialised
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SELECT query.

        Each call uses a fresh ``cursor()`` so its result handle is
        independent — defence in depth on top of the connector-level lock,
        so a future regression that drops the lock can't silently return
        rows that belong to a sibling query whose state was just reset.
        """
        try:
            self.connect()
            result = self.connection.cursor().execute(sql)

            if result_format == "csv":
                df = result.df()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=df.to_csv(index=False),
                    row_count=len(df),
                    result_format=result_format,
                )
            elif result_format == "arrow":
                arrow_table = result.fetch_arrow_table()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=arrow_table,
                    row_count=arrow_table.num_rows,
                    result_format=result_format,
                )
            elif result_format == "pandas":
                df = result.df()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=df,
                    row_count=len(df),
                    result_format=result_format,
                )
            else:  # list
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                result_list = [dict(zip(columns, row)) for row in rows]
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=result_list,
                    row_count=len(rows),
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
    @_serialised
    def execute_queries(self, queries: List[str]) -> List[Any]:
        """Execute multiple queries."""
        results = []
        self.connect()
        try:
            for query in queries:
                result = self.connection.execute(query)
                if result.description:
                    rows = result.fetchall()
                    columns = [desc[0] for desc in result.description]
                    results.append([dict(zip(columns, row)) for row in rows])
                else:
                    results.append(0)
        except Exception as e:
            raise self._handle_exception(e, "\n".join(queries))
        return results

    @override
    @_serialised
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """Execute SET/USE commands."""
        try:
            self.connect()
            self.connection.execute(sql_query)

            # Parse context switch
            from datus.utils.sql_utils import parse_context_switch

            switch_context = parse_context_switch(sql=sql_query, dialect=self.dialect)
            if switch_context:
                if database_name := switch_context.get("database_name"):
                    self.database_name = database_name
                if schema_name := switch_context.get("schema_name"):
                    self.schema_name = schema_name

            return ExecuteSQLResult(
                success=True,
                sql_query=sql_query,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql_query)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql_query,
            )

    @override
    @_serialised
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names."""
        self.connect()
        sql = "SELECT table_name FROM duckdb_tables() WHERE database_name != 'system'"
        if database_name:
            sql += f" AND database_name = '{database_name}'"
        if schema_name:
            sql += f" AND schema_name = '{schema_name}'"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    @_serialised
    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names."""
        self.connect()
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name
        sql = "SELECT view_name FROM duckdb_views() WHERE database_name != 'system'"
        if database_name:
            sql += f" AND database_name = '{database_name}'"
        if schema_name:
            sql += f" AND schema_name = '{schema_name}'"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    @_serialised
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """Get list of database names."""
        self.connect()
        sql = "SELECT database_name FROM duckdb_databases()"
        if not include_sys:
            sql += " WHERE database_name not in ('system', 'temp')"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "main", table_name: str = ""
    ) -> str:
        if database_name:
            if schema_name:
                return f'"{database_name}"."{schema_name}"."{table_name}"'
            return f'"{database_name}"."{table_name}"'
        return f'"{schema_name}"."{table_name}"' if schema_name else table_name

    @override
    @_serialised
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        self.connect()
        sql = "SELECT schema_name FROM duckdb_schemas()"
        has_where = False
        database_name = database_name or self.database_name
        if database_name:
            sql += f" WHERE database_name='{database_name}'"
            has_where = True

        if not include_sys:
            sys_schemas = list(self._sys_schemas())
            if not has_where:
                sql += list_to_in_str(" WHERE schema_name NOT IN", sys_schemas)
            else:
                sql += list_to_in_str(" AND schema_name NOT IN", sys_schemas)

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def _sys_schemas(self) -> Set[str]:
        return {"system", "temp", "information_schema"}

    @override
    @_serialised
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        self.connect()
        if schema_name:
            self.connection.execute(f'USE "{schema_name}"')

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get tables with DDL definitions."""
        filter_tables = self._reset_filter_tables(
            tables, catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
            _type="table",
            filter_tables=filter_tables,
        )

    @_serialised
    def _get_meta_with_ddl(
        self,
        database_name: str = "",
        schema_name: str = "",
        _type: str = "",
        filter_tables: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Get metadata with DDL for tables or views."""
        self.connect()
        metadata_names = _metadata_names(_type)
        sql_field = "" if not metadata_names.has_sql_field else ', "sql"'
        query_sql = (
            f"SELECT database_name, schema_name, {metadata_names.name_field}{sql_field}"
            f" FROM {metadata_names.info_table}() WHERE database_name != 'system'"
        )
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name
        if database_name:
            query_sql += f" AND database_name = '{database_name}'"
        if schema_name:
            query_sql += f" AND schema_name = '{schema_name}'"

        result_set = self.connection.execute(query_sql)
        rows = result_set.fetchall()
        columns = [desc[0] for desc in result_set.description]

        result = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            table_name = str(row_dict[metadata_names.name_field])
            full_name = self.full_name(
                database_name=str(row_dict["database_name"]),
                schema_name=str(row_dict["schema_name"]),
                table_name=table_name,
            )
            if not database_name:
                full_name = ".".join(full_name.split(".")[1:])
            if filter_tables and full_name not in filter_tables:
                continue

            result.append(
                {
                    "identifier": self.identifier(
                        database_name=str(row_dict["database_name"]),
                        schema_name=str(row_dict["schema_name"]),
                        table_name=table_name,
                    ),
                    "catalog_name": "",
                    "database_name": row_dict["database_name"],
                    "schema_name": row_dict["schema_name"],
                    "table_name": table_name,
                    "definition": row_dict.get("sql", ""),
                    "table_type": _type,
                }
            )
        return result

    @override
    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get views with DDL definitions."""
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
            _type="view",
        )

    @override
    @_serialised
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, str]]:
        """Get sample values from tables."""
        self.connect()
        try:
            samples = []
            if tables:
                logger.debug(f"Getting sample data from tables {tables} LIMIT {top_n}")
                for table_name in tables:
                    if schema_name:
                        if database_name:
                            prefix = f'"{database_name}"."{schema_name}"'
                        else:
                            prefix = f'"{schema_name}"'
                    else:
                        prefix = "" if not schema_name else f'"{schema_name}"'
                    if prefix:
                        query = f"""SELECT * FROM {prefix}."{table_name}" LIMIT {top_n}"""
                    else:
                        query = f"""SELECT * FROM "{table_name}" LIMIT {top_n}"""

                    result = self.connection.execute(query)
                    df = result.df()
                    if len(df) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": database_name,
                                "table_name": table_name,
                                "schema_name": schema_name,
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
            else:
                tables_with_ddl = []
                if table_type == "mv":
                    return []
                if table_type in ("full", "table"):
                    tables_with_ddl.extend(
                        self.get_tables_with_ddl(database_name=database_name, schema_name=schema_name)
                    )
                if table_type in ("full", "view"):
                    tables_with_ddl.extend(
                        self.get_views_with_ddl(database_name=database_name, schema_name=schema_name)
                    )
                for table in tables_with_ddl:
                    query = (
                        f'SELECT * FROM "{table["database_name"]}"."{table["schema_name"]}"."{table["table_name"]}" '
                        f"LIMIT {top_n}"
                    )
                    result = self.connection.execute(query)
                    df = result.df()
                    if len(df) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": table["database_name"],
                                "table_name": table["table_name"],
                                "schema_name": table["schema_name"],
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
            return samples
        except DatusException:
            raise
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": str(e),
                },
            ) from e

    @override
    @_serialised
    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        if not table_name:
            return []

        self.connect()
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name or "main"
        full_name = self.full_name(database_name=database_name, schema_name=schema_name, table_name=table_name)

        escaped_name = full_name.replace("'", "''")
        sql = f"PRAGMA table_info('{escaped_name}')"
        try:
            try:
                result = self.connection.execute(sql)
            except duckdb.CatalogException:
                # In common single-file DuckDB usage, callers may pass the
                # connector's logical/current database name even though PRAGMA
                # resolution only needs "schema.table". Retry without the
                # database qualifier before surfacing the error.
                if database_name:
                    fallback_full_name = self.full_name(schema_name=schema_name, table_name=table_name)
                    fallback_sql = f"PRAGMA table_info('{fallback_full_name}')"
                    logger.warning(
                        "DuckDB get_schema retrying without database qualification: "
                        "database_name=%r schema_name=%r table_name=%r original=%r fallback=%r",
                        database_name,
                        schema_name,
                        table_name,
                        full_name,
                        fallback_full_name,
                    )
                    result = self.connection.execute(fallback_sql)
                    sql = fallback_sql
                else:
                    raise
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]
            # Normalize field names to match standard schema
            schema_list = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert notnull to nullable and dflt_value to default_value
                normalized = {
                    "cid": row_dict.get("cid"),
                    "name": row_dict.get("name"),
                    "type": row_dict.get("type"),
                    "nullable": not bool(row_dict.get("notnull", 0)),  # Invert notnull to nullable
                    "default_value": row_dict.get("dflt_value"),  # Rename dflt_value to default_value
                    "pk": row_dict.get("pk"),
                }
                schema_list.append(normalized)
            return schema_list
        except DatusException as e:
            if "error_message" in e.message_args:
                message = e.message_args["error_message"]
            else:
                message = e.message
            raise DatusException(ErrorCode.DB_QUERY_METADATA_FAILED, message=message)
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_QUERY_METADATA_FAILED,
                message_args={"error_message": str(e), "sql": sql},
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary."""
        return {"db_type": DBType.DUCKDB, "db_path": self.db_path}

    def get_type(self) -> str:
        return DBType.DUCKDB

    # ==================== MigrationTargetMixin ====================

    def describe_migration_capabilities(self) -> Dict[str, Any]:
        return {
            "supported": True,
            "dialect_family": "duckdb",
            "requires": [],  # DuckDB is single-node; no distribution required
            "forbids": [
                "DUPLICATE KEY (StarRocks-only)",
                "DISTRIBUTED BY HASH ... BUCKETS (StarRocks-only)",
                "ENGINE = ... (MySQL/ClickHouse syntax)",
            ],
            "type_hints": {
                "unbounded VARCHAR": "VARCHAR (no length limit)",
                "TEXT": "VARCHAR",
                "JSON": "JSON (native type)",
                "JSONB": "JSON",
                "VARIANT": "JSON (Snowflake VARIANT maps to native JSON)",
                "HUGEINT": "HUGEINT (native 128-bit integer)",
                "LARGEINT": "HUGEINT",
                "LIST<T>": "T[] (DuckDB array syntax)",
                "STRUCT": "STRUCT(field_name field_type, ...)",
                "MAP": "MAP(key_type, value_type)",
                "BOOLEAN": "BOOLEAN",
                "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
            },
            "example_ddl": (
                "CREATE TABLE main.t (\n"
                "  id BIGINT,\n"
                "  name VARCHAR,\n"
                "  tags VARCHAR[],\n"
                "  payload JSON,\n"
                "  created_at TIMESTAMP\n"
                ")"
            ),
        }

    def suggest_table_layout(self, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        # DuckDB is embedded/single-node — no distribution keys or partition hints needed.
        return {}

    def validate_ddl(self, ddl: str) -> List[str]:
        import re as _re

        errors: List[str] = []
        upper = ddl.upper()
        # Match across arbitrary whitespace (spaces, tabs, newlines) so irregular
        # formatting still trips the dialect checks — e.g. `ENGINE   =` and
        # `DISTRIBUTED\nBY` should both be caught.
        if _re.search(r"DUPLICATE\s+KEY", upper):
            errors.append("DUPLICATE KEY is StarRocks-only syntax; DuckDB does not support it")
        if _re.search(r"DISTRIBUTED\s+BY", upper) and "BUCKETS" in upper:
            errors.append("DISTRIBUTED BY ... BUCKETS is StarRocks syntax; DuckDB does not support it")
        if _re.search(r"\bENGINE\s*=", upper):
            errors.append("ENGINE clause is MySQL/ClickHouse syntax; DuckDB does not support it")
        return errors

    def map_source_type(self, source_dialect: str, source_type: str) -> Optional[str]:
        import re as _re

        base = _re.sub(r"\(.*\)", "", source_type.strip().upper()).strip()
        overrides = {
            "JSONB": "JSON",
            "VARIANT": "JSON",
            "SUPER": "JSON",  # Redshift SUPER → DuckDB JSON
            "OBJECT": "JSON",  # Snowflake OBJECT → DuckDB JSON
        }
        return overrides.get(base)
