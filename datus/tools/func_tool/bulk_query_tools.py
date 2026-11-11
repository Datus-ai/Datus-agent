"""File exports with bounded tool results."""

from __future__ import annotations

import gzip
import hashlib
from contextlib import closing
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pyarrow as pa

from datus.tools.db_tools.query_spool import QueryLimits, QuerySpool, bulk_slot, check_cancelled
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.policy_runtime import PolicyRuntime
from datus.tools.read_observer import ReadObservation, notify_read
from datus.tools.registry.contracts import tool_contract
from datus.utils.bounded_tool_output import bounded_text, error_summary
from datus.utils.project_files import MAX_SQL_FILE_BYTES, ProjectFiles
from datus.utils.sql_utils import looks_like_sql_file_ref


class BulkQueryTools:
    def _bulk_root(self) -> Path:
        return Path(self._filesystem_root or getattr(self.agent_config, "project_root", None) or Path.cwd()).resolve()

    def _bulk_sql(self, value: str, files: ProjectFiles) -> tuple[str, str]:
        if looks_like_sql_file_ref(value.lower()):
            return files.read_sql(value.strip(), MAX_SQL_FILE_BYTES)
        return value, "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _bulk_failure(self, exc, *, file_scope=None, **state) -> FuncToolResult:
        diagnostic = None
        owned_scope = None
        try:
            if file_scope is None:
                owned_scope = file_scope = ProjectFiles(self._bulk_root())
            with file_scope.directory("exports/.diagnostics", create=True) as directory:
                name = uuid4().hex + ".txt.gz"
                with directory.open(name, "xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as output:
                    output.write(bounded_text(exc, 16 * 1024 * 1024).encode("utf-8"))
                directory.verify_current()
                diagnostic = str((directory.path / name).relative_to(file_scope.root))
        except Exception:
            pass  # An unwritable diagnostic directory must never expose the raw error.
        finally:
            if owned_scope is not None:
                owned_scope.close()
        return FuncToolResult(
            success=0,
            error=error_summary(exc),
            result={
                "error_code": type(exc).__name__,
                "diagnostic_ref": diagnostic,
                "diagnostic_encoding": "gzip",
                "complete": False,
                **state,
            },
        )

    def _stage_bulk_query(self, sql: str, connector, datasource: str, spool: QuerySpool) -> dict:
        guard_error = self._reject_unsafe_uploads_sql(sql, datasource, connector=connector)
        if guard_error is not None:
            raise PermissionError(guard_error.error)
        invalid, _ = self._validate_read_sql(sql, connector)
        if invalid:
            raise ValueError(invalid.error)
        runtime = PolicyRuntime(self.agent_config)
        streaming = getattr(connector, "supports_streaming_read", False) is True and runtime.supports_streaming_results
        observation = dict(
            agent_config=self.agent_config,
            policy_context=self.policy_context,
            datasource=datasource,
            runtime_names=runtime.runtime_names,
            datasource_configs=getattr(self._db_manager, "_db_configs", None),
        )
        notify_read(ReadObservation(phase="before_sql", sql=sql, **observation))
        decision = runtime.before_sql_read(
            sql,
            datasource=datasource,
            dialect=connector.dialect,
            connector=connector,
            policy_context=self.policy_context,
        )
        if not decision.allowed:
            raise PermissionError(decision.reason or "Policy denied the query")
        original_sql = sql
        sql = decision.sql if decision.sql is not None else sql
        if sql != original_sql:
            guard_error = self._reject_unsafe_uploads_sql(sql, datasource, connector=connector)
            if guard_error is not None:
                raise PermissionError(guard_error.error)
        invalid, _ = self._validate_read_sql(sql, connector)
        if invalid:
            raise ValueError(invalid.error)
        if sql != original_sql:
            from collections import Counter

            import sqlglot
            from sqlglot import exp

            from datus.utils.sql_utils import parse_dialect

            def windows(text):
                tree = sqlglot.parse_one(text, read=parse_dialect(connector.dialect))
                return Counter(
                    node.sql() for node in tree.walk() if isinstance(node, (exp.Limit, exp.Offset, exp.Fetch))
                )

            if windows(original_sql) != windows(sql):
                raise PermissionError("Read policy changed query row limits; a complete export is unavailable")
        if decision.verify_context is not None:
            decision.verify_context()
        applied = set(decision.applied_policies)
        with bulk_slot(connector):
            check_cancelled()
            if decision.verify_context is not None:
                decision.verify_context()
            if streaming:
                self._stage_streaming_query(sql, connector, datasource, runtime, spool, applied)
            else:
                result = connector.execute_query(sql, result_format="arrow")
                check_cancelled()
                if not result.success:
                    raise RuntimeError(result.error or "Source query failed")
                table = result.sql_return
                self._validate_bulk_table(table, spool)
                if decision.verify_context is not None:
                    decision.verify_context()
                result = runtime.after_read_result(
                    table,
                    sql=sql,
                    datasource=datasource,
                    dialect=connector.dialect,
                    policy_context=self.policy_context,
                )
                if not result.allowed:
                    raise PermissionError(result.reason or "Policy denied the query result")
                applied.update(result.applied_policies)
                table = result.result
                self._validate_bulk_table(table, spool)
                if table.num_rows == 0:
                    spool.add(table)
                else:
                    with closing(table.to_reader(max_chunksize=spool.limits.batch_rows)) as batches:
                        for batch in batches:
                            spool.add(pa.Table.from_batches([batch]))
        if spool.schema is None:
            raise ValueError("Datasource adapter did not return result metadata")
        if decision.verify_context is not None:
            decision.verify_context()
        notify_read(
            ReadObservation(phase="after_sql", sql=sql, applied_policy_ids=tuple(sorted(applied)), **observation)
        )
        return {
            "scope": "authorized_query",
            "query_rewritten": sql != original_sql,
            "executed_sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "applied_policies": sorted(applied),
            "read_mode": "streaming" if streaming else "buffered",
        }

    @staticmethod
    def _validate_bulk_table(table: pa.Table, spool: QuerySpool) -> None:
        if not isinstance(table, pa.Table):
            raise TypeError("Query results and result policies must return Arrow tables")
        if table.num_rows > spool.limits.max_rows or table.nbytes > spool.limits.max_bytes:
            raise ValueError("Source query or result policy exceeds max_rows/max_bytes")

    def _stage_streaming_query(
        self, sql: str, connector, datasource: str, runtime: PolicyRuntime, spool: QuerySpool, applied: set[str]
    ) -> None:
        read_rows = read_bytes = 0
        with closing(connector.iter_query_batches(sql, batch_size=spool.limits.batch_rows)) as batches:
            for table in batches:
                check_cancelled()
                self._validate_bulk_table(table, spool)
                read_rows += table.num_rows
                read_bytes += table.nbytes
                if read_rows > spool.limits.max_rows or read_bytes > spool.limits.max_bytes:
                    raise ValueError("Source query exceeds max_rows/max_bytes before result policies")
                if table.nbytes > spool.limits.max_batch_bytes:
                    raise ValueError("Source query batch exceeds max_batch_bytes")
                result = runtime.after_read_batch(
                    table, sql=sql, datasource=datasource, dialect=connector.dialect, policy_context=self.policy_context
                )
                if not result.allowed:
                    raise PermissionError(result.reason or "Policy denied the query result")
                applied.update(result.applied_policies)
                spool.add(result.result)

    @tool_contract(access="write", direct_call=False, category="db_tools")
    def export_query(
        self,
        sql: str,
        output_file: Optional[str] = "",
        datasource: Optional[str] = "",
        database: Optional[str] = "",
    ) -> FuncToolResult:
        """Export one complete read-only query directly to a project CSV/Parquet file.

        datasource and database select the connection, as in execute_sql;
        omitted selectors use the current connection. Use for downloads or
        files consumed by Python, including large results.
        sql accepts SQL text or a project .sql file path. output_file is a
        project-relative .csv/.parquet path; omitted paths use exports/<id>.csv.
        Returns only file/count/completion metadata, never rows or SQL text.
        Stream when the adapter and policies support it; otherwise fetch the
        complete result into Agent memory and stage it in bounded batches.
        Inspect small query samples with execute_sql instead. Existing files
        are never overwritten. A failed/cancelled export is not complete.
        """
        if getattr(self.agent_config, "_client_source", None) == "vscode":
            return FuncToolResult(
                success=0,
                error=(
                    "export_query needs server-side project files, but this VS Code workspace lives on the client. "
                    "Use a project in the web IDE to export query results."
                ),
            )
        files = directory = None
        try:
            effective_datasource = self._resolve_effective_datasource(datasource)
            if self._db_manager is None:
                if datasource and effective_datasource != self._resolve_effective_datasource(None):
                    raise ValueError("Cannot select another datasource from a fixed connector")
                if database and database != self._active_database_of(self.connector):
                    raise ValueError("Cannot select another database from a fixed connector")
            files = ProjectFiles(self._bulk_root())
            sql, sql_ref = self._bulk_sql(sql, files)
            destination = files.resolve(output_file or f"exports/{uuid4().hex}.csv")
            if destination.suffix.lower() not in (".csv", ".parquet"):
                raise ValueError("output_file must end in .csv or .parquet")
            directory = files.directory(destination.parent, create=True)
            if directory.exists(destination.name) or directory.exists(destination.name + ".meta.json"):
                raise FileExistsError("Output already exists; choose another output_file")
            connector = self._get_connector(datasource, database or "")
            limits = QueryLimits.from_dict(getattr(self.agent_config, "bulk_query", None))
            metadata = {
                "file": str(destination.relative_to(files.root)),
                "format": destination.suffix[1:].lower(),
                "sql_ref": sql_ref,
                "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            }
            with QuerySpool(directory, limits) as spool:
                metadata.update(self._stage_bulk_query(sql, connector, effective_datasource, spool))
                spool.publish(destination.name, metadata)
            return FuncToolResult(result=metadata)
        except Exception as exc:
            return self._bulk_failure(exc, file_scope=files)
        finally:
            try:
                if directory is not None:
                    directory.close()
            finally:
                if files is not None:
                    files.close()
