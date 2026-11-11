"""Complete file exports and bounded tool results exercised against real DuckDB."""

import gzip
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.policy_runtime import PolicyRuntime
from datus.tools.read_observer import observe_reads
from datus.tools.registry.contracts import tool_definition
from datus.utils.bounded_tool_output import MAX_RESULT_BYTES, bound_data_tool_result


@pytest.fixture
def environment(tmp_path):
    source = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "source.duckdb")))
    target = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
    tool = DBFuncTool(source, filesystem_root=str(tmp_path))
    tool.agent_config = SimpleNamespace(bulk_query={"batch_rows": 2}, policy_context={}, sql_read_only=False)
    tool._default_datasource = "source"
    tool._get_connector = lambda datasource="", database="": {
        "source": source,
        "target": target,
        "local_files": source,
        "": source,
    }[datasource or ""]
    try:
        yield tool, source, target, tmp_path
    finally:
        source.close()
        target.close()


@pytest.fixture
def routed_environment(environment):
    from datus.configuration.agent_config import DbConfig

    tool, _, _, root = environment
    configs = {}
    for datasource, default in [("warehouse", "one"), ("other", "alternate"), ("local_files", "one")]:
        directory = root / datasource
        directory.mkdir()
        for database in {default, "two"}:
            connector = DuckdbConnector(DuckDBConfig(db_path=str(directory / (database + ".duckdb"))))
            try:
                assert (
                    connector.execute_ddl(f"CREATE TABLE marker AS SELECT '{datasource}/{database}' AS id").success
                    is True
                )
                assert connector.execute_ddl(f"CREATE TABLE only_in_{database} AS SELECT 1 AS id").success is True
            finally:
                connector.close()
        configs[datasource] = DbConfig(type="duckdb", path_pattern=str(directory / "*.duckdb"), database=default)
    manager = DBManager(configs)
    tool._db_manager = manager
    tool._is_multi_connector = True
    tool._default_datasource = "warehouse"
    tool._default_database = "one"
    tool._datasources = list(configs)
    tool._execution_context = {}
    tool._get_connector = DBFuncTool._get_connector.__get__(tool)
    tool._primary_connector = tool._get_connector()
    try:
        yield tool, root
    finally:
        manager.close()


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_complete_export_preserves_ids_and_outgrows_display_compression(environment, suffix):
    tool, _, _, root = environment
    result = tool.export_query(
        "SELECT '0012345678901234567' AS id, i, NULL AS absent FROM range(31) t(i)",
        f"exports/orders.{suffix}",
    )
    assert result.success == 1, result.error
    assert result.result["row_count"] == 31
    assert result.result["complete"] is True
    path = root / result.result["file"]
    frame = pd.read_csv(path, dtype={"id": str}) if suffix == "csv" else pd.read_parquet(path)
    assert frame["id"].tolist() == ["0012345678901234567"] * 31
    assert frame["i"].tolist() == list(range(31))
    assert frame["absent"].isna().all()
    metadata = json.loads(path.with_name(path.name + ".meta.json").read_text())
    assert metadata == result.result
    assert len(json.dumps(result.model_dump()).encode()) < MAX_RESULT_BYTES
    assert not list(path.parent.glob(".query-*"))


@pytest.mark.parametrize("sql_ref", ["query.sql", " \nquery.sql\t ", "query.SQL"])
def test_sql_file_and_empty_result(environment, sql_ref):
    tool, _, _, root = environment
    (root / sql_ref.strip()).write_text("SELECT 'x' AS id, 1 AS n WHERE FALSE")
    result = tool.export_query(sql_ref, "exports/empty.parquet")
    assert result.success == 1, result.error
    assert result.result["sql_ref"] == sql_ref.strip()
    table = pq.read_table(root / result.result["file"])
    assert table.num_rows == 0
    assert table.column_names == ["id", "n"]


@pytest.mark.parametrize("sql", ["SELECT * FROM staging.sql", "SELECT *\nFROM staging.sql"])
def test_inline_sql_ending_in_sql_table_name_is_exported(environment, sql):
    tool, source, _, root = environment
    assert source.execute_ddl("CREATE SCHEMA staging").success is True
    assert source.execute_ddl("CREATE TABLE staging.sql AS SELECT '0012345678901234567' AS id").success is True

    result = tool.export_query(sql, "exports/matched.parquet")

    assert result.success == 1, result.error
    assert result.result["complete"] is True
    assert result.result["row_count"] == 1
    assert pq.read_table(root / result.result["file"]).to_pylist() == [{"id": "0012345678901234567"}]


def test_source_failure_and_limits_never_publish_partial_files(environment, monkeypatch):
    tool, source, _, root = environment
    tool.agent_config.bulk_query["max_rows"] = 3
    result = tool.export_query("SELECT * FROM range(5)", "exports/limited.csv")
    assert result.success == 0
    assert "max_rows" in result.error
    assert not (root / "exports/limited.csv").exists()

    def failed_stream(*args, **kwargs):
        yield pa.table({"id": [1, 2]})
        raise RuntimeError("Binder error: missing column\nLINE 1: INSERT INTO t VALUES " + "x" * 4_000_000)

    monkeypatch.setattr(source, "iter_query_batches", failed_stream)
    result = tool.export_query("SELECT 1", "exports/failed.csv")
    assert result.success == 0
    assert "Binder error" in result.error
    assert len(json.dumps(result.model_dump()).encode()) < MAX_RESULT_BYTES
    with gzip.open(root / result.result["diagnostic_ref"], "rt") as diagnostic:
        assert len(diagnostic.read()) > 4_000_000
    assert result.result["diagnostic_encoding"] == "gzip"
    assert not (root / "exports/failed.csv").exists()
    assert not list((root / "exports").glob(".query-*"))


def test_export_refuses_overwrite_and_escape(environment, tmp_path):
    tool, _, _, root = environment
    (root / "existing.csv").write_text("original")
    result = tool.export_query("SELECT 1", "existing.csv")
    assert result.success == 0
    assert (root / "existing.csv").read_text() == "original"
    outside = tmp_path.parent / "outside-export"
    (root / "escape").symlink_to(outside, target_is_directory=True)
    result = tool.export_query("SELECT 1", "escape/stolen.csv")
    assert result.success == 0
    assert "inside the current project" in result.error


@pytest.mark.parametrize("when", ["connect", "read", "metadata"])
@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_swapped_export_directory_never_receives_data_or_cleanup(environment, monkeypatch, when, replacement):
    tool, source, _, root = environment
    exports = root / "exports"
    exports.mkdir()
    outside = root.parent / (root.name + "-other-project")
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("other-project")

    def swap():
        exports.rename(root / "saved-exports")
        if replacement == "symlink":
            exports.symlink_to(outside, target_is_directory=True)
        else:
            exports.mkdir()

    if when == "connect":
        original = tool._get_connector

        def connect(*args, **kwargs):
            swap()
            return original(*args, **kwargs)

        monkeypatch.setattr(tool, "_get_connector", connect)
    elif when == "read":
        original = source.iter_query_batches

        def read(*args, **kwargs):
            swap()
            yield from original(*args, **kwargs)

        monkeypatch.setattr(source, "iter_query_batches", read)
    else:
        original = json.dump

        def write_metadata(*args, **kwargs):
            swap()
            return original(*args, **kwargs)

        monkeypatch.setattr("datus.tools.db_tools.query_spool.json.dump", write_metadata)

    result = tool.export_query("SELECT 'private-query-fixture' AS id", "exports/leak.csv")
    assert result.success == 0
    assert result.result["complete"] is False
    assert sentinel.read_text() == "other-project"
    assert sorted(path.name for path in outside.iterdir()) == ["keep.txt"]
    assert not (root / "saved-exports/leak.csv").exists()
    assert not (root / "saved-exports/leak.csv.meta.json").exists()
    assert not list((root / "saved-exports").glob(".query-*"))


def test_project_local_links_still_resolve_to_authorized_files(environment):
    tool, _, _, root = environment
    sql = root / "input.sql"
    sql.write_text("SELECT '0012345678901234567' AS id")
    (root / "input-link.sql").symlink_to(sql)
    exports = root / "real-exports"
    exports.mkdir()
    (root / "linked-exports").symlink_to(exports, target_is_directory=True)
    result = tool.export_query("input-link.sql", "linked-exports/orders.parquet")
    assert result.success == 1, result.error
    assert result.result["file"] == "real-exports/orders.parquet"
    assert pq.read_table(exports / "orders.parquet").to_pylist() == [{"id": "0012345678901234567"}]


def test_project_file_descriptors_are_closed_after_success_and_failure(tmp_path, monkeypatch):
    import os

    from datus.utils.project_files import ProjectFiles

    (tmp_path / "query.sql").write_text("SELECT 1 AS id")
    handles = []
    original = os.open

    def track(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        handles.append(descriptor)
        return descriptor

    monkeypatch.setattr("datus.utils.project_files.os.open", track)
    with ProjectFiles(tmp_path) as files:
        assert files.read_sql("query.sql", 1024)[0] == "SELECT 1 AS id"
        with files.directory("output", create=True) as directory:
            with directory.open("file.csv", "xb") as output:
                output.write(b"id\n1\n")
            with pytest.raises(FileExistsError):
                directory.open("file.csv", "xb")
        with pytest.raises(FileNotFoundError):
            files.read_sql("missing/file.sql", 1024)
    assert handles
    for descriptor in handles:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("phase", ["directory", "file"])
def test_staging_initialization_error_releases_handles_and_temporary_files(environment, monkeypatch, phase):
    import os

    from datus.utils.project_files import ProjectDirectory

    tool, _, _, root = environment
    captured = None
    injected = False
    original_stat = os.fstat
    original_child = ProjectDirectory.child
    original_open = ProjectDirectory.open

    def child(directory, name):
        nonlocal captured
        handle = original_child(directory, name)
        if phase == "directory" and name.startswith(".query-"):
            captured = handle.fd
        return handle

    def open_file(directory, name, mode="rb"):
        nonlocal captured
        handle = original_open(directory, name, mode)
        if phase == "file" and name == "000000.arrow":
            captured = handle.fileno()
        return handle

    def stat(descriptor):
        nonlocal injected
        if descriptor == captured and not injected:
            injected = True
            raise OSError("Injected staging stat failure")
        return original_stat(descriptor)

    monkeypatch.setattr(ProjectDirectory, "child", child)
    monkeypatch.setattr(ProjectDirectory, "open", open_file)
    monkeypatch.setattr("datus.tools.db_tools.query_spool.os.fstat", stat)
    result = tool.export_query("SELECT 1 AS id", "exports/failed.csv")
    assert injected is True
    assert result.success == 0
    assert "Injected staging stat failure" in result.error
    assert not list((root / "exports").glob(".query-*"))
    assert not (root / "exports/failed.csv").exists()
    with pytest.raises(OSError):
        original_stat(captured)


def test_sql_file_swapped_after_resolution_is_not_read(environment, monkeypatch):
    from datus.utils.project_files import ProjectFiles

    tool, source, _, root = environment
    sql = root / "input.sql"
    sql.write_text("SELECT 1 AS id")
    outside = root.parent / (root.name + "-private.sql")
    outside.write_text("SELECT 'other-project-secret' AS id")
    original = ProjectFiles.resolve
    swapped = False

    def resolve(files, value):
        nonlocal swapped
        path = original(files, value)
        if path == sql and not swapped:
            swapped = True
            sql.unlink()
            sql.symlink_to(outside)
        return path

    monkeypatch.setattr(ProjectFiles, "resolve", resolve)
    result = tool.export_query("input.sql", "exports/leak.csv")
    assert swapped is True
    assert result.success == 0
    assert source.connection is None
    assert "other-project-secret" not in json.dumps(result.model_dump())
    assert not (root / "exports/leak.csv").exists()


def test_diagnostic_directory_swapped_after_resolution_stays_scoped(environment, monkeypatch):
    from datus.utils.project_files import ProjectFiles

    tool, _, _, root = environment
    diagnostic = root / "exports/.diagnostics"
    diagnostic.mkdir(parents=True)
    outside = root.parent / (root.name + "-diagnostics")
    outside.mkdir()
    original = ProjectFiles.resolve
    swapped = False

    def resolve(files, value):
        nonlocal swapped
        path = original(files, value)
        if path == diagnostic and not swapped:
            swapped = True
            diagnostic.rename(root / "exports/saved-diagnostics")
            diagnostic.symlink_to(outside, target_is_directory=True)
        return path

    monkeypatch.setattr(ProjectFiles, "resolve", resolve)
    result = tool.export_query("SELECT * FROM nonexistent")
    assert swapped is True
    assert result.success == 0
    assert result.result["diagnostic_ref"] is None
    assert not list(outside.iterdir())


def test_read_only_agent_can_export_but_cannot_transfer(environment):
    tool, _, _, _ = environment
    tool.read_only = True
    assert tool.export_query("SELECT 1 AS id").success == 1
    assert tool.transfer_query_result("SELECT 1 AS id", "source", "orders", "target").success == 0


def test_result_policy_masking_and_observation_apply_to_export(environment, monkeypatch):
    tool, _, _, root = environment
    runtime = PolicyRuntime(None)

    class Mask:
        def after_read_batch(self, table, **kwargs):
            return SimpleNamespace(
                allowed=True, result=table.set_column(0, "secret", pa.array(["masked"] * table.num_rows))
            )

    runtime._runtimes = [("mask", Mask())]
    monkeypatch.setattr("datus.tools.func_tool.bulk_query_tools.PolicyRuntime", lambda config: runtime)
    observations = []
    with observe_reads(observations.append):
        result = tool.export_query("SELECT 'private' AS secret FROM range(3)", "masked.parquet")
    assert result.success == 1, result.error
    assert pq.read_table(root / "masked.parquet").to_pylist() == [{"secret": "masked"}] * 3
    assert [item.phase for item in observations] == ["before_sql", "after_sql"]


def test_unknown_whole_result_policy_is_rejected_before_read(environment, monkeypatch):
    tool, source, _, _ = environment
    runtime = PolicyRuntime(None)
    runtime._runtimes = [("whole", SimpleNamespace(after_read_result=lambda *a, **k: None))]
    monkeypatch.setattr("datus.tools.func_tool.bulk_query_tools.PolicyRuntime", lambda config: runtime)
    result = tool.export_query("SELECT 1 AS id")
    assert result.success == 0
    assert "does not support streaming" in result.error
    assert source.connection is None


def test_budget_preserves_valid_json_for_multibyte_driver_errors():
    result = bound_data_tool_result({"success": 0, "error": "数据库错误" * 1_000_000, "result": None})
    assert len(result["error"].encode()) <= 2048
    assert json.loads(json.dumps(result)) == result


def test_data_tool_catalog_exposes_export_connection_selectors(environment):
    tool, _, _, _ = environment
    export = tool.to_function_tool(tool.export_query)
    definition = tool_definition(export)
    assert definition.tool_id == "db_tools.export_query"
    assert definition.access == "write"
    assert definition.direct_call is False
    assert set(export.params_json_schema["properties"]) == {"sql", "output_file", "datasource", "database"}
    assert "export_query" in DBFuncTool.all_tools_name()


def test_export_uses_the_current_connection_without_a_datasource_override(environment, monkeypatch):
    tool, _, _, _ = environment
    connector = Mock(wraps=tool._get_connector)
    monkeypatch.setattr(tool, "_get_connector", connector)
    result = tool.export_query("SELECT 'current' AS id", "current.csv")
    assert result.success == 1, result.error
    connector.assert_called_once_with("", "")


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [{"datasource": "target"}, {"database": "another_database"}])
async def test_export_rejects_unavailable_selection_on_a_fixed_connector(environment, monkeypatch, selector):
    tool, _, _, root = environment
    connector = Mock(wraps=tool._get_connector)
    monkeypatch.setattr(tool, "_get_connector", connector)
    function = tool.to_function_tool(tool.export_query)
    result = await function.on_invoke_tool(
        None, json.dumps({"sql": "SELECT 1", "output_file": "override.csv", **selector})
    )
    assert result["success"] == 0
    assert "fixed connector" in result["error"]
    connector.assert_not_called()
    assert not (root / "override.csv").exists()


@pytest.mark.asyncio
async def test_export_routes_datasource_and_database_like_execute_sql(routed_environment):
    tool, root = routed_environment
    cases = [
        ({}, "warehouse/one"),
        ({"datasource": None, "database": None}, "warehouse/one"),
        ({"database": "two"}, "warehouse/two"),
        ({"datasource": "other"}, "other/alternate"),
        ({"datasource": "other", "database": "two"}, "other/two"),
        ({"datasource": "warehouse"}, "warehouse/one"),
    ]
    function = tool.to_function_tool(tool.export_query)
    for index, (selectors, expected) in enumerate(cases):
        output = f"routed/{index}.parquet"
        observations = []
        with observe_reads(observations.append):
            result = await function.on_invoke_tool(
                None, json.dumps({"sql": "SELECT * FROM marker", "output_file": output, **selectors})
            )
        assert result["success"] == 1, result["error"]
        assert pq.read_table(root / output).to_pylist() == [{"id": expected}]
        assert {item.datasource for item in observations} == {expected.split("/")[0]}
        assert tool._default_datasource == "warehouse"
        assert tool._default_database == "one"


@pytest.mark.parametrize("selectors", [{"datasource": "missing"}, {"database": "missing"}])
def test_export_unknown_connection_never_falls_back(routed_environment, selectors):
    tool, root = routed_environment
    result = tool.export_query("SELECT * FROM marker", "missing.parquet", **selectors)
    assert result.success == 0
    assert "missing" in result.error
    assert not (root / "missing.parquet").exists()


def test_export_uploads_guard_uses_the_selected_database(routed_environment):
    tool, root = routed_environment
    result = tool.export_query(
        "SELECT * FROM only_in_two", "selected_upload.parquet", datasource="local_files", database="two"
    )
    assert result.success == 1, result.error
    assert pq.read_table(root / "selected_upload.parquet").to_pylist() == [{"id": 1}]
    result = tool.export_query(
        "SELECT * FROM only_in_one", "wrong_upload.parquet", datasource="local_files", database="two"
    )
    assert result.success == 0
    assert "not a table registered" in result.error
    foreign = root.parent / (root.name + "-selected-upload.csv")
    foreign.write_text("id\n99\n")
    result = tool.export_query(
        f"SELECT * FROM read_csv_auto('{foreign}')",
        "foreign_upload.csv",
        datasource="local_files",
        database="two",
    )
    assert result.success == 0
    assert "cannot be called directly" in result.error
    assert not (root / "foreign_upload.csv").exists()


@pytest.mark.parametrize("sql_file", [False, True])
@pytest.mark.parametrize(
    ("reader", "error"),
    [("function", "cannot be called directly"), ("replacement_scan", "not a table registered")],
)
def test_export_reuses_uploads_guard_for_project_external_files(environment, sql_file, reader, error):
    tool, _, _, root = environment
    tool._default_datasource = "local_files"
    foreign = root.parent / (root.name + "-foreign.csv")
    foreign.write_text("id,marker\n1,FOREIGN_PROJECT_FIXTURE\n")
    target = f"read_csv_auto('{foreign}')" if reader == "function" else f"'{foreign}'"
    sql = f"SELECT * FROM {target}"
    if sql_file:
        (root / "external.sql").write_text(sql)
        sql = "external.sql"
    result = tool.export_query(sql, "external.csv")
    assert result.success == 0
    assert error in result.error
    assert not (root / "external.csv").exists()
    assert not (root / "external.csv.meta.json").exists()


def test_export_keeps_registered_upload_views_readable(environment):
    tool, source, _, root = environment
    tool._default_datasource = "local_files"
    source_file = root / "registered.csv"
    source_file.write_text("id\n0012345678901234567\n")
    created = source.execute_ddl(
        f"CREATE VIEW registered_upload AS SELECT * FROM read_csv('{source_file}', all_varchar=true)"
    )
    assert created.success is True, created.error
    result = tool.export_query("SELECT * FROM registered_upload", "registered.parquet")
    assert result.success == 1, result.error
    assert pq.read_table(root / "registered.parquet").to_pylist() == [{"id": "0012345678901234567"}]


def test_export_checks_uploads_guard_after_policy_rewrite(environment, monkeypatch):
    tool, source, _, root = environment
    tool._default_datasource = "local_files"
    assert source.execute_ddl("CREATE TABLE registered_upload AS SELECT 1 AS id").success is True
    foreign = root.parent / (root.name + "-rewrite.csv")
    foreign.write_text("id\n99\n")
    runtime = PolicyRuntime(None)
    runtime._runtimes = [
        (
            "rewrite",
            SimpleNamespace(
                before_sql_read=lambda *a, **k: SimpleNamespace(
                    allowed=True, sql=f"SELECT * FROM read_csv_auto('{foreign}')"
                )
            ),
        )
    ]
    monkeypatch.setattr("datus.tools.func_tool.bulk_query_tools.PolicyRuntime", lambda config: runtime)
    result = tool.export_query("SELECT * FROM registered_upload", "rewritten.csv")
    assert result.success == 0
    assert "cannot be called directly" in result.error
    assert not (root / "rewritten.csv").exists()


def test_export_preserves_regular_duckdb_file_read_behavior(environment):
    tool, _, _, root = environment
    foreign = root.parent / (root.name + "-configured-source.csv")
    foreign.write_text("id\n1\n")
    result = tool.export_query(f"SELECT * FROM read_csv_auto('{foreign}')", "regular.csv")
    assert result.success == 1, result.error
    assert result.result["row_count"] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 10001}, "batch_size"),
        ({"mode": "unknown"}, "mode"),
        ({"target_table": "orders; DROP TABLE orders"}, "identifier"),
        ({"source_sql": "SELECT 1; DELETE FROM orders"}, "Multi-statement"),
        ({"source_sql": "DELETE FROM orders"}, "read-only"),
    ],
)
def test_invalid_transfer_request_preserves_destination(environment, kwargs, message):
    tool, _, target, _ = environment
    assert target.execute_ddl("CREATE TABLE orders(id BIGINT)").success is True
    assert target.execute_insert("INSERT INTO orders VALUES (99)").success is True
    request = dict(
        source_sql="SELECT 1 AS id", source_datasource="source", target_table="orders", target_datasource="target"
    )
    request.update(kwargs)
    result = tool.transfer_query_result(**request)
    assert result.success == 0
    assert message in result.error
    assert target.execute_query("SELECT * FROM orders", result_format="list").sql_return == [{"id": 99}]


def test_unknown_datasource_is_reported_without_fallback(environment):
    tool, source, target, _ = environment
    result = tool.transfer_query_result("SELECT 1 AS id", "unavailable", "orders", "target")
    assert result.success == 0
    assert source.connection is None
    assert target.connection is None


def test_cancellation_does_not_publish_export(environment):
    import threading

    from datus.tools.db_tools.query_spool import ACTIVE_BULK_CANCEL

    tool, _, target, root = environment
    assert target.execute_ddl("CREATE TABLE orders(id BIGINT)").success is True
    assert target.execute_insert("INSERT INTO orders VALUES (99)").success is True
    event = threading.Event()
    event.set()
    token = ACTIVE_BULK_CANCEL.set(event)
    try:
        exported = tool.export_query("SELECT 1 AS id", "cancelled.csv")
    finally:
        ACTIVE_BULK_CANCEL.reset(token)
    assert exported.success == 0
    assert not (root / "cancelled.csv").exists()
    assert target.execute_query("SELECT * FROM orders", result_format="list").sql_return == [{"id": 99}]


@pytest.mark.asyncio
async def test_tool_timeout_prevents_late_file_publication(environment, monkeypatch):
    import asyncio
    import threading

    tool, source, _, root = environment
    assert tool.export_query("SELECT 1 AS id", "warmup.csv").success == 1
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    original_failure = tool._bulk_failure

    def slow_stream(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=2):
            raise TimeoutError("Test did not release the blocked query")
        yield pa.table({"id": [1]})

    def failed_export(*args, **kwargs):
        try:
            return original_failure(*args, **kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(source, "iter_query_batches", slow_stream)
    monkeypatch.setattr(tool, "_bulk_failure", failed_export)
    monkeypatch.setattr("datus.tools.func_tool.base.DATA_TOOL_TIMEOUT_SECONDS", 0.1)
    export = tool.to_function_tool(tool.export_query)
    try:
        result = await export.on_invoke_tool(None, json.dumps({"sql": "SELECT 1 AS id", "output_file": "late.csv"}))
        assert entered.is_set() is True
        assert result["success"] == 0
        assert result["result"]["complete"] is False
        assert result["result"]["write_outcome"] == "unknown"
    finally:
        release.set()
        assert await asyncio.to_thread(completed.wait, 2) is True
    assert not (root / "late.csv").exists()
    assert not list(root.glob(".query-*"))


@pytest.mark.parametrize("limits", [{"batch_rows": 0}, {"max_bytes": True}, {"unexpected": 1}])
def test_host_limits_are_validated(limits):
    from datus.tools.db_tools.query_spool import QueryLimits

    with pytest.raises(ValueError):
        QueryLimits.from_dict(limits)


def test_null_only_first_batch_promotes_without_losing_later_values(environment, monkeypatch):
    tool, source, _, root = environment

    def batches(*args, **kwargs):
        yield pa.table({"id": [None, None]})
        yield pa.table({"id": ["0012345678901234567"]})

    monkeypatch.setattr(source, "iter_query_batches", batches)
    result = tool.export_query("SELECT 'x' AS id", "promoted.parquet")
    assert result.success == 1, result.error
    assert pq.read_table(root / "promoted.parquet").to_pylist() == [
        {"id": None},
        {"id": None},
        {"id": "0012345678901234567"},
    ]


def test_policy_row_limit_does_not_silently_publish_a_complete_export(environment, monkeypatch):
    tool, source, _, root = environment
    runtime = PolicyRuntime(None)
    runtime._runtimes = [
        (
            "cap",
            SimpleNamespace(
                before_sql_read=lambda *args, **kwargs: SimpleNamespace(
                    allowed=True, sql="SELECT 1 AS id LIMIT 1", applied_policies=["row-cap"]
                )
            ),
        )
    ]
    monkeypatch.setattr("datus.tools.func_tool.bulk_query_tools.PolicyRuntime", lambda config: runtime)
    result = tool.export_query("SELECT 1 AS id", "limited.parquet")
    assert result.success == 0
    assert "changed query row limits" in result.error
    assert not (root / "limited.parquet").exists()
    assert source.connection is None


def test_large_policy_metadata_keeps_delivery_and_reconciliation_fields():
    original = {
        "success": 1,
        "error": None,
        "result": {
            "file": "exports/orders.parquet",
            "row_count": 123,
            "complete": True,
            "applied_policies": ["policy"] * 100000,
        },
    }
    result = bound_data_tool_result(original)
    assert result["success"] == 1
    assert result["result"]["file"] == "exports/orders.parquet"
    assert result["result"]["row_count"] == 123
    assert result["result"]["complete"] is True
    assert result["result"]["applied_policy_count"] == 100000
    assert len(json.dumps(result).encode()) <= MAX_RESULT_BYTES


def test_cancellation_during_metadata_publication_removes_both_files(environment, monkeypatch):
    import threading

    from datus.tools.db_tools.query_spool import ACTIVE_BULK_CANCEL

    tool, _, _, root = environment
    event = threading.Event()
    original = json.dump

    def interrupted_dump(*args, **kwargs):
        event.set()
        return original(*args, **kwargs)

    monkeypatch.setattr("datus.tools.db_tools.query_spool.json.dump", interrupted_dump)
    token = ACTIVE_BULK_CANCEL.set(event)
    try:
        result = tool.export_query("SELECT 1 AS id", "interrupted.csv")
    finally:
        ACTIVE_BULK_CANCEL.reset(token)
    assert result.success == 0
    assert not (root / "interrupted.csv").exists()
    assert not (root / "interrupted.csv.meta.json").exists()


@pytest.mark.asyncio
async def test_queued_bulk_calls_do_not_starve_the_default_thread_pool(environment, monkeypatch):
    import asyncio
    import threading

    tool, source, _, root = environment
    release = threading.Event()
    entered = []

    def waiting_stream(*args, **kwargs):
        entered.append(threading.get_ident())
        release.wait(timeout=2)
        yield pa.table({"id": [1]})

    monkeypatch.setattr(source, "iter_query_batches", waiting_stream)
    export = tool.to_function_tool(tool.export_query)
    tasks = [
        asyncio.create_task(
            export.on_invoke_tool(None, json.dumps({"sql": "SELECT 1 AS id", "output_file": f"queued_{i}.csv"}))
        )
        for i in range(40)
    ]
    try:
        await asyncio.sleep(0.1)
        assert len(entered) == 2
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "responsive"), timeout=0.5) == "responsive"
    finally:
        for task in tasks:
            task.cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0.1)
    assert not list(root.glob("queued_*.csv"))


@pytest.mark.asyncio
async def test_timed_out_workers_keep_slots_until_the_driver_exits(environment, monkeypatch):
    import asyncio
    import threading

    tool, source, _, root = environment
    assert tool.export_query("SELECT 1 AS id", "warmup.csv").success == 1
    release = threading.Event()
    entered = []

    def waiting_stream(*args, **kwargs):
        entered.append(threading.get_ident())
        release.wait(timeout=2)
        yield pa.table({"id": [1]})

    monkeypatch.setattr(source, "iter_query_batches", waiting_stream)
    monkeypatch.setattr("datus.tools.func_tool.base.DATA_TOOL_TIMEOUT_SECONDS", 0.05)
    export = tool.to_function_tool(tool.export_query)

    async def invoke(index):
        return await export.on_invoke_tool(
            None, json.dumps({"sql": "SELECT 1 AS id", "output_file": f"timeout_{index}.csv"})
        )

    try:
        first = await asyncio.gather(invoke(0), invoke(1))
        assert len(entered) == 2
        assert all(result["result"]["write_outcome"] == "unknown" for result in first)
        queued = await asyncio.gather(*(invoke(i) for i in range(2, 42)))
        assert all(result["result"]["write_outcome"] == "not_started" for result in queued)
        assert len(entered) == 2
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "responsive"), timeout=0.5) == "responsive"
    finally:
        release.set()
        await asyncio.sleep(0.15)
    assert not list(root.glob("timeout_*.csv"))
