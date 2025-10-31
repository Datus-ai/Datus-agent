from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

import pandas as pd
import pyarrow as pa

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType, SQLType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import metadata_identifier, parse_context_switch, parse_sql_type
from datus.utils.typing_compat import override

try:
    from clickzetta.zettapark.session import Session
except ImportError as exc:  # pragma: no cover - optional dependency
    Session = None  # type: ignore[assignment]
    _CLICKZETTA_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CLICKZETTA_IMPORT_ERROR = None

logger = get_logger(__name__)

_DEFAULT_HINTS: Dict[str, Any] = {
    "sdk.job.timeout": 300,
    "query_tag": "Query from Datus Agent",
    "cz.storage.parquet.vector.index.read.memory.cache": "true",
    "cz.storage.parquet.vector.index.read.local.cache": "false",
    "cz.sql.table.scan.push.down.filter": "true",
    "cz.sql.table.scan.enable.ensure.filter": "true",
    "cz.storage.always.prefetch.internal": "true",
    "cz.optimizer.generate.columns.always.valid": "true",
    "cz.sql.index.prewhere.enabled": "true",
    "cz.storage.parquet.enable.io.prefetch": "false",
}


def _safe_escape(value: Optional[str]) -> str:
    """Escape single quotes in string values for SQL literals."""
    if value is None:
        return ""
    return value.replace("'", "''")


def _safe_escape_identifier(identifier: Optional[str]) -> str:
    """Escape backticks in identifiers for SQL identifiers using backtick quoting."""
    if identifier is None:
        return ""
    return str(identifier).replace("`", "``")


class ClickzettaConnector(BaseSqlConnector):
    """
    Connector implementation for ClickZetta Lakehouse.
    Wraps the official `clickzetta.zettapark` session with the BaseSqlConnector interface.
    """

    AUTH_EXPIRATION_SECONDS = 1800

    def __init__(
        self,
        service: str,
        username: str,
        password: str,
        instance: str,
        workspace: str,
        schema: str = "",
        vcluster: str = "",
        secure: Optional[bool] = None,
        hints: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(DBType.CLICKZETTA)
        schema = schema or "PUBLIC"
        vcluster = vcluster or "DEFAULT_AP"
        if Session is None:
            raise DatusException(
                ErrorCode.COMMON_MISSING_DEPENDENCY,
                message=(
                    "ClickZetta connector requires the packages "
                    "`clickzetta-connector-python` and `clickzetta-zettapark-python`. "
                    "Please install them before using the connector."
                ),
            ) from _CLICKZETTA_IMPORT_ERROR

        required_fields = {
            "service": service,
            "username": username,
            "password": password,
            "instance": instance,
            "workspace": workspace,
            "schema": schema,
            "vcluster": vcluster,
        }
        missing_fields = [name for name, value in required_fields.items() if not value]
        if missing_fields:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": f"Missing ClickZetta connection fields: {', '.join(sorted(missing_fields))}"
                },
            )

        self.service = service
        self.user = username
        self.password = password
        self.instance = instance
        self._workspace = workspace
        self.schema_name = schema
        self.vcluster = vcluster

        self.catalog_name = ""
        self.database_name = workspace or ""

        merged_hints: Dict[str, Any] = dict(_DEFAULT_HINTS)
        if hints:
            merged_hints.update(hints)

        connection_config: Dict[str, Any] = {
            "service": service,
            "username": username,
            "password": password,
            "instance": instance,
            "workspace": workspace,
            "schema": self.schema_name,
            "vcluster": self.vcluster,
            "hints": merged_hints,
        }
        if secure is not None:
            connection_config["secure"] = secure
        if extra:
            connection_config.update(extra)

        self._connection_config = connection_config
        self._session: Optional[Session] = None
        self._auth_timestamp = 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_connection(self) -> Session:
        if self._session is None or (time.time() - self._auth_timestamp) > self.AUTH_EXPIRATION_SECONDS:
            self.connect()
        return self._session  # type: ignore[return-value]

    def connect(self):
        """Initialize or refresh the ClickZetta session."""
        try:
            self._session = Session.builder.configs(self._connection_config).create()
            self._auth_timestamp = time.time()
            self.connection = self._session  # Maintain BaseSqlConnector.connection reference
            if self.schema_name:
                escaped_schema = _safe_escape_identifier(self.schema_name.upper())
                self._session.sql(f"USE SCHEMA `{escaped_schema}`")
            if self.vcluster:
                escaped_vc = _safe_escape_identifier(self.vcluster.upper())
                self._session.sql(f"USE VCLUSTER `{escaped_vc}`")
        except Exception as exc:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(exc)},
            ) from exc

    def close(self):
        if self._session:
            try:
                self._session.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug(f"Failed to close ClickZetta session cleanly: {exc}")
            finally:
                self._session = None
                self.connection = None
        super().close()

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """Execute context switching in ClickZetta session.

        Note:
        - Workspace (database_name) switching is NOT supported and requires separate namespace configuration
        - Only schema switching within the same workspace is supported
        - VCluster switching is supported via USE VCLUSTER command
        """
        session = self._ensure_connection()

        # Reject workspace switching requests
        if database_name and database_name != self.database_name:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": (
                        f"Workspace switching from '{self.database_name}' to '{database_name}' "
                        "is not supported. Please configure separate namespaces for different workspaces."
                    ),
                    "sql": f"USE DATABASE {database_name}",
                },
            )

        # Support schema switching within the same workspace
        if schema_name and schema_name != self.schema_name:
            try:
                escaped_schema = _safe_escape_identifier(schema_name.upper())
                session.sql(f"USE SCHEMA `{escaped_schema}`")
                logger.info(f"Switched to schema: {schema_name}")
            except Exception as exc:
                escaped_schema = _safe_escape_identifier(schema_name.upper())
                self._wrap_exception(exc, f"USE SCHEMA `{escaped_schema}`", ErrorCode.DB_EXECUTION_ERROR)

    def _wrap_exception(self, exc: Exception, sql: str = "", error_code: ErrorCode = ErrorCode.DB_EXECUTION_ERROR):
        if isinstance(exc, DatusException):
            raise exc
        raise DatusException(error_code, message_args={"error_message": str(exc), "sql": sql}) from exc

    def _run_query(self, sql: str) -> pd.DataFrame:
        try:
            session = self._ensure_connection()
            result = session.sql(sql)
            if hasattr(result, "to_pandas"):
                return result.to_pandas()
            # Fallback to empty DataFrame if result has no tabular output
            return pd.DataFrame()
        except Exception as exc:
            self._wrap_exception(exc, sql)

    def _run_command(self, sql: str) -> pd.DataFrame:
        try:
            session = self._ensure_connection()
            result = session.sql(sql)
            if hasattr(result, "to_pandas"):
                try:
                    return result.to_pandas()
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()
        except Exception as exc:
            self._wrap_exception(exc, sql)

    @staticmethod
    def _normalize_volume_uri(volume: str, relative_path: str) -> str:
        base = (volume or "").strip()
        if not base:
            raise ValueError("Volume name must not be empty when reading semantic model files.")
        if base.lower().startswith("volume:"):
            base = base.rstrip("/")
            relative = (relative_path or "").lstrip("/")
            return f"{base}/{relative}" if relative else base
        if base.startswith("@"):
            relative = (relative_path or "").lstrip("/")
            return f"{base.rstrip('/')}/{relative}" if relative else base
        raise ValueError(f"Unsupported volume/stage format: {volume}")

    def read_volume_file(self, volume: str, relative_path: str) -> str:
        """Download and return the contents of a file stored inside a ClickZetta volume or stage."""
        source_uri = self._normalize_volume_uri(volume, relative_path)
        session = self._ensure_connection()

        with tempfile.TemporaryDirectory() as tmp_dir:
            session.file.get(source_uri, tmp_dir)

            candidate = Path(tmp_dir) / Path(relative_path).name
            if not candidate.exists():
                nested_candidate = Path(tmp_dir) / Path(relative_path)
                if nested_candidate.exists():
                    candidate = nested_candidate
                else:
                    matches = list(Path(tmp_dir).rglob(Path(relative_path).name))
                    if not matches:
                        raise FileNotFoundError(f"File '{relative_path}' not found in {volume}")
                    candidate = matches[0]

            return candidate.read_text(encoding="utf-8")

    def list_volume_files(
        self,
        volume: str,
        directory: str = "",
        suffixes: tuple[str, ...] = (".yaml", ".yml"),
    ) -> List[str]:
        """List files stored inside a ClickZetta volume or legacy stage."""
        directory = directory.strip().lstrip("/").rstrip("/")
        volume_uri = self._normalize_volume_uri(volume, directory or "")
        session = self._ensure_connection()

        if volume.lower().startswith("volume:user://"):
            list_sql = "LIST USER VOLUME"
            if directory:
                # Escape single quotes to avoid breaking the SQL literal
                escaped_dir = _safe_escape(directory.rstrip("/"))
                list_sql += f" SUBDIRECTORY '{escaped_dir}/'"
        else:
            list_sql = f"LIST {volume_uri}"

        result = session.sql(list_sql)
        try:
            df = result.to_pandas()
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return []

        column_name = None
        for candidate in ("relative_path", "name", "path"):
            if candidate in df.columns:
                column_name = candidate
                break
        if column_name is None:
            column_name = df.columns[0]

        discovered: List[str] = []
        for value in df[column_name].tolist():
            if not value:
                continue
            path_str = str(value).strip()
            candidate_name = path_str.split("/")[-1]
            lower = candidate_name.lower()
            if suffixes and not any(lower.endswith(suffix) for suffix in suffixes):
                continue
            if candidate_name not in discovered:
                discovered.append(candidate_name)
        return sorted(discovered)

    @staticmethod
    def _extract_row_count(df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        for field in ("rows", "row_count", "rows_affected", "affected_rows", "count"):
            if field in df.columns:
                try:
                    return int(df[field].iloc[0])
                except Exception as exc:
                    # Log and continue to try other common row count fields
                    logger.debug(f"Failed to extract row count from field '{field}': {exc}")
                    continue
        return len(df)

    def _build_definition(
        self,
        workspace: str,
        schema_name: str,
        table_name: str,
        columns: List[Dict[str, Any]],
        table_comment: Optional[str] = "",
        table_type: str = "table",
    ) -> str:
        column_lines: List[str] = []
        for column in columns:
            col_name = column.get("column_name") or column.get("name") or ""
            data_type = column.get("data_type") or column.get("type") or "STRING"
            comment = column.get("comment")
            col_def = f"`{_safe_escape_identifier(col_name)}` {data_type}"
            if comment:
                col_def += f" COMMENT '{_safe_escape(str(comment))}'"
            column_lines.append(col_def)

        columns_section = ",\n  ".join(column_lines) if column_lines else ""
        # Build table name parts
        escaped_workspace = _safe_escape_identifier(workspace)
        escaped_schema = _safe_escape_identifier(schema_name)
        escaped_table = _safe_escape_identifier(table_name)
        table_full_name = f"`{escaped_workspace}`.`{escaped_schema}`.`{escaped_table}`"

        definition = (
            f"CREATE {table_type.upper()} {table_full_name} (\n  {columns_section}\n)"
            if columns_section
            else f"CREATE {table_type.upper()} {table_full_name}"
        )
        if table_comment:
            definition += f"\nCOMMENT = '{_safe_escape(str(table_comment))}'"
        return definition

    # ------------------------------------------------------------------ #
    # BaseSqlConnector abstract methods
    # ------------------------------------------------------------------ #
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        try:
            df = self._run_command(sql)
            row_count = self._extract_row_count(df)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except DatusException as exc:
            return ExecuteSQLResult(success=False, error=str(exc), sql_query=sql, sql_return="", row_count=0)

    def execute_update(self, sql: str) -> ExecuteSQLResult:
        return self.execute_insert(sql)

    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        return self.execute_insert(sql)

    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        try:
            df = self._run_query(sql)
            row_count = len(df)
            if result_format == "csv":
                sql_return: Any = df.to_csv(index=False)
            elif result_format == "arrow":
                sql_return = pa.Table.from_pandas(df)
            elif result_format == "list":
                sql_return = df.to_dict(orient="records")
            else:
                sql_return = df

            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=sql_return,
                row_count=row_count,
                result_format=result_format,
            )
        except DatusException as exc:
            return ExecuteSQLResult(success=False, error=str(exc), sql_query=sql)
        except Exception as exc:  # Ensure unexpected errors are also surfaced in result
            # Surface unexpected errors in logs while keeping graceful result contract
            logger.error(f"Unexpected error in execute_query: {exc}", exc_info=True)
            return ExecuteSQLResult(success=False, error=str(exc), sql_query=sql)

    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        result = self.execute_query(sql, result_format="pandas")
        if not result.success:
            return result
        if hasattr(result.sql_return, "empty"):
            result.row_count = len(result.sql_return)
        return result

    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        try:
            self._run_command(sql)
            return ExecuteSQLResult(success=True, sql_query=sql, sql_return="Successful", row_count=0)
        except DatusException as exc:
            return ExecuteSQLResult(success=False, error=str(exc), sql_query=sql)

    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        result = self.execute_query(sql, result_format="csv")
        return result

    def execute_arrow_iterator(self, sql: str, max_rows: int = 100) -> Iterator[pa.Table]:
        df = self._run_query(sql)
        if df.empty:
            return iter(())
        table = pa.Table.from_pandas(df)
        return iter([table.slice(0, max_rows)])

    def test_connection(self):
        self.execute_query("SELECT 1")

    def execute_queries(self, queries: List[str]) -> List[Any]:
        results: List[Any] = []
        for query in queries:
            if parse_sql_type(query, self.dialect) == SQLType.SELECT:
                df = self._run_query(query)
                results.append(df.to_dict(orient="records"))
            else:
                command_df = self._run_command(query)
                results.append(self._extract_row_count(command_df))
        return results

    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        try:
            self._run_command(sql_query)
            switch_context = parse_context_switch(sql=sql_query, dialect=self.dialect)
            if switch_context:
                if catalog_name := switch_context.get("catalog_name"):
                    self.catalog_name = catalog_name
                if database_name := switch_context.get("database_name"):
                    self.database_name = database_name
                if schema_name := switch_context.get("schema_name"):
                    self.schema_name = schema_name
            return ExecuteSQLResult(success=True, sql_query=sql_query, sql_return="Successful", row_count=0)
        except DatusException as exc:
            return ExecuteSQLResult(success=False, error=str(exc), sql_query=sql_query)

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def _info_schema(self, workspace: Optional[str] = None) -> str:
        workspace = workspace or self.workspace
        # Quote workspace with backticks to handle special characters or case sensitivity
        escaped_workspace = _safe_escape_identifier(workspace)
        return f"`{escaped_workspace}`.information_schema"

    def _normalized_schema(self, schema_name: Optional[str] = None) -> str:
        schema = schema_name or self.schema_name
        return schema.upper() if schema else ""

    @property
    def workspace(self) -> str:
        return self.database_name or self._workspace

    def get_catalogs(self) -> List[str]:
        try:
            df = self._run_query("SHOW CATALOGS")
            if "catalog_name" in df.columns:
                return df["catalog_name"].dropna().tolist()
            if "name" in df.columns:
                return df["name"].dropna().tolist()
        except DatusException:
            logger.debug("SHOW CATALOGS not supported, returning workspace as catalog fallback")
        return [self.database_name] if self.database_name else []

    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        # In ClickZetta workspace ~= database concept
        if self.database_name:
            return [self.database_name]
        return []

    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        workspace = database_name or self.database_name
        if not workspace:
            return []
        info_schema = self._info_schema(workspace)
        sql = f"SELECT DISTINCT table_schema FROM {info_schema}.tables"
        try:
            df = self._run_query(sql)
            schemas = df["table_schema"].dropna().tolist()
            if not include_sys:
                schemas = [s for s in schemas if not str(s).startswith("INFORMATION_SCHEMA")]
            return schemas
        except DatusException:
            return [self.schema_name] if self.schema_name else []

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        workspace = database_name or self.database_name
        schema = self._normalized_schema(schema_name)
        if not workspace or not schema:
            return []
        info_schema = self._info_schema(workspace)
        sql = (
            f"SELECT table_name, table_type FROM {info_schema}.tables "
            f"WHERE upper(table_schema) = '{_safe_escape(schema)}'"
        )
        df = self._run_query(sql)
        if df.empty:
            return []
        valid_types = {"MANAGED_TABLE", "EXTERNAL_TABLE", "BASE TABLE", "TABLE"}
        return [row.table_name for row in df.itertuples() if str(row.table_type).upper() in valid_types]

    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        workspace = database_name or self.database_name
        schema = self._normalized_schema(schema_name)
        if not workspace or not schema:
            return []
        info_schema = self._info_schema(workspace)
        sql = (
            f"SELECT table_name, table_type FROM {info_schema}.tables "
            f"WHERE upper(table_schema) = '{_safe_escape(schema)}'"
        )
        try:
            df = self._run_query(sql)
            if df.empty:
                return []
            view_types = {"VIEW", "DYNAMIC_TABLE"}
            return [row.table_name for row in df.itertuples() if str(row.table_type).upper() in view_types]
        except DatusException:
            return []

    def get_materialized_views(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        workspace = database_name or self.database_name
        schema = self._normalized_schema(schema_name)
        if not workspace or not schema:
            return []
        info_schema = self._info_schema(workspace)
        sql = (
            f"SELECT table_name, table_type FROM {info_schema}.tables "
            f"WHERE upper(table_schema) = '{_safe_escape(schema)}'"
        )
        try:
            df = self._run_query(sql)
            if df.empty:
                return []
            return [row.table_name for row in df.itertuples() if str(row.table_type).upper() == "MATERIALIZED_VIEW"]
        except DatusException:
            return []

    def get_tables_with_ddl(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        tables: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        return self._collect_table_definitions(
            database_name=database_name,
            schema_name=schema_name,
            tables=tables,
            include_views=False,
        )

    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        return self._collect_table_definitions(
            database_name=database_name,
            schema_name=schema_name,
            include_views=True,
        )

    def _collect_table_definitions(
        self,
        database_name: str = "",
        schema_name: str = "",
        tables: Optional[List[str]] = None,
        include_views: bool = False,
    ) -> List[Dict[str, str]]:
        workspace = database_name or self.database_name
        schema = self._normalized_schema(schema_name)
        if not workspace or not schema:
            return []

        info_schema = self._info_schema(workspace)
        base_query = (
            f"SELECT table_name, comment, table_type "
            f"FROM {info_schema}.tables WHERE upper(table_schema) = '{_safe_escape(schema)}'"
        )
        if tables:
            table_list = ",".join(f"'{_safe_escape(tbl)}'" for tbl in tables)
            base_query += f" AND table_name IN ({table_list})"
        try:
            tables_df = self._run_query(base_query)
        except DatusException:
            return []

        column_query = (
            f"SELECT table_name, column_name, data_type, comment "
            f"FROM {info_schema}.columns WHERE upper(table_schema) = '{_safe_escape(schema)}'"
        )
        if tables:
            table_list = ",".join(f"'{_safe_escape(tbl)}'" for tbl in tables)
            column_query += f" AND table_name IN ({table_list})"
        column_query += " ORDER BY table_name, column_name"

        try:
            columns_df = self._run_query(column_query)
        except DatusException:
            columns_df = pd.DataFrame()

        columns_map: Dict[str, List[Dict[str, Any]]] = {}
        if not columns_df.empty:
            for item in columns_df.to_dict(orient="records"):
                table_name = item.get("table_name")
                if not table_name:
                    continue
                columns_map.setdefault(table_name, []).append(item)

        records: List[Dict[str, str]] = []
        for table_item in tables_df.to_dict(orient="records"):
            table_name = table_item.get("table_name")
            if not table_name:
                continue
            table_type_raw = str(table_item.get("table_type", "")).upper()
            is_view = table_type_raw in {"VIEW", "DYNAMIC_TABLE"}
            is_mv = table_type_raw == "MATERIALIZED_VIEW"
            if include_views and not is_view:
                continue
            if not include_views and (is_view or is_mv):
                continue
            table_type = "view" if is_view else "table"
            if is_mv:
                table_type = "mv"
            definition = self._build_definition(
                workspace=workspace,
                schema_name=schema,
                table_name=table_name,
                columns=columns_map.get(table_name, []),
                table_comment=table_item.get("comment"),
                table_type=table_type if table_type != "mv" else "table",
            )
            records.append(
                {
                    "identifier": metadata_identifier(
                        database_name=workspace,
                        schema_name=schema,
                        table_name=table_name,
                        dialect=self.dialect,
                    ),
                    "catalog_name": "",
                    "database_name": workspace,
                    "schema_name": schema,
                    "table_name": table_name,
                    "definition": definition,
                    "table_type": table_type,
                }
            )
        return records

    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, Any]]:
        if not table_name:
            return []
        workspace = database_name or self.database_name
        schema = self._normalized_schema(schema_name)
        if not workspace or not schema:
            return []

        info_schema = self._info_schema(workspace)
        sql = (
            f"SELECT column_name, data_type, comment "
            f"FROM {info_schema}.columns "
            f"WHERE upper(table_schema) = '{_safe_escape(schema)}' "
            f"AND table_name = '{_safe_escape(table_name)}' "
            f"ORDER BY column_name"
        )
        df = self._run_query(sql)
        result: List[Dict[str, Any]] = []
        for idx, item in enumerate(df.to_dict(orient="records")):
            result.append(
                {
                    "cid": idx,
                    "name": item.get("column_name"),
                    "type": item.get("data_type"),
                    "comment": item.get("comment"),
                    "nullable": True,
                    "pk": False,
                    "default_value": None,
                }
            )
        return result

    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, Any]]:
        workspace = database_name or self.database_name
        schema = schema_name or self.schema_name
        if not workspace or not schema:
            return []

        tables_to_sample = tables or self.get_tables(database_name=workspace, schema_name=schema)
        samples: List[Dict[str, Any]] = []
        for table_name in tables_to_sample:
            # Build table name parts for sample query
            escaped_workspace = _safe_escape_identifier(workspace)
            escaped_schema = _safe_escape_identifier(schema)
            escaped_table = _safe_escape_identifier(table_name)
            table_full_name = f"`{escaped_workspace}`.`{escaped_schema}`.`{escaped_table}`"
            sql = f"SELECT * FROM {table_full_name} LIMIT {top_n}"
            try:
                df = self._run_query(sql)
            except DatusException:
                continue
            if df.empty:
                continue
            samples.append(
                {
                    "catalog_name": "",
                    "database_name": workspace,
                    "schema_name": schema,
                    "table_name": table_name,
                    "sample_rows": df.to_csv(index=False),
                }
            )
        return samples

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        workspace = database_name or self.database_name
        schema = schema_name or self.schema_name
        if workspace and schema:
            return f"{workspace}.{schema}.{table_name}"
        if schema:
            return f"{schema}.{table_name}"
        return table_name

    def identifier(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        db = database_name or self.database_name
        schema = schema_name or self.schema_name
        # For ClickZetta, prefer a clear 3-part identifier when possible
        if db and schema:
            return f"{db}.{schema}.{table_name}"
        return metadata_identifier(
            catalog_name=catalog_name,
            database_name=db,
            schema_name=schema,
            table_name=table_name,
            dialect=self.dialect,
        )
