# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Explicit scopes constrain real reads, writes, searches and DB routing."""

import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.agent.node.configurable_scope import ConfigurableScope, ScopedAgentFilesystem
from datus.configuration.scoped_context_overrides import effective_subagent
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.schemas.agent_spec import AgentSpec
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool
from datus.utils.exceptions import DatusException


def _scope(config, scoped_context=None):
    return ConfigurableScope(AgentSpec(name="research", scoped_context=scoped_context), config)


def _filesystem(scope):
    return ScopedAgentFilesystem(execution_scope=scope, root_path=str(scope.root), strict=True)


def test_omitted_scope_denies_business_data_and_project_files(real_agent_config):
    scope = _scope(real_agent_config)
    root = Path(real_agent_config.project_root)
    root.mkdir(parents=True, exist_ok=True)
    secret = root / "private.txt"
    secret.write_text("private revenue")
    assert scope.config.services.datasources == {}
    assert not _filesystem(scope).read_file("private.txt").success
    assert not _filesystem(scope).write_file("new.txt", "write").success
    assert not (root / "new.txt").exists()
    with pytest.raises(DatusException, match="scoped_context.datasource"):
        scope.require_tool_scope("db_tools.execute_sql", {"sql": "select 1"})


@pytest.mark.parametrize("method", ["read_file", "write_file", "edit_file", "delete_file", "grep", "glob"])
def test_project_file_filters_protect_sibling_files_at_every_entrypoint(real_agent_config, method):
    scope = _scope(real_agent_config, {"project_files": ["public/**"]})
    scope.root.mkdir(parents=True, exist_ok=True)
    (scope.root / "public").mkdir()
    allowed = scope.root / "public" / "sales.txt"
    allowed.write_text("public revenue")
    private = scope.root / "private.txt"
    private.write_text("private revenue")
    tools = _filesystem(scope)
    if method == "read_file":
        assert tools.read_file("public/sales.txt").success == 1
        assert not tools.read_file("private.txt").success
    elif method == "write_file":
        assert tools.write_file("public/new.txt", "new").success == 1
        assert not tools.write_file("private.txt", "replaced").success
    elif method == "edit_file":
        assert tools.edit_file("public/sales.txt", "public", "published").success == 1
        assert not tools.edit_file("private.txt", "private", "leaked").success
    elif method == "delete_file":
        assert tools.delete_file("public/sales.txt").success == 1
        assert not tools.delete_file("private.txt").success
    else:
        output = tools.grep("revenue") if method == "grep" else tools.glob("**/*.txt")
        assert output.success == 1, output.error
        assert "sales.txt" in str(output.result)
        assert "private.txt" not in str(output.result)
    assert private.read_text() == "private revenue"


def test_symlink_and_traversal_cannot_escape_a_broad_project_glob(real_agent_config, tmp_path):
    scope = _scope(real_agent_config, {"project_files": ["**"]})
    scope.root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("private credentials")
    (scope.root / "inside.txt").write_text("authorized content")
    (scope.root / "alias.txt").symlink_to(outside)
    fs = _filesystem(scope)
    assert not fs.read_file("alias.txt").success
    assert not fs.read_file("../outside.txt").success
    output = fs.glob("**/*.txt")
    assert output.success == 1, output.error
    assert output.result["files"] == ["inside.txt"]


def test_runtime_broad_glob_still_intersects_static_file_selection(real_agent_config):
    scope = _scope(real_agent_config, {"project_files": ["public/**"]})
    scope.prepare(ScopedContext(project_files=["**"]))
    assert scope.allows_file(scope.root / "public" / "nested" / "sale.txt")
    assert not scope.allows_file(scope.root / "private.txt")
    scope.prepare(ScopedContext(project_files=[]))
    assert not scope.allows_file(scope.root / "public" / "sale.txt")
    scope.prepare(None)
    assert scope.allows_file(scope.root / "public" / "sale.txt")


def test_grouped_caller_scope_is_intersected_without_mutating_shared_config(real_agent_config):
    real_agent_config.agentic_nodes["research"] = {"scoped_context": {"tables_by_datasource": {}}}
    scope = _scope(real_agent_config, {"datasource": "california_schools"})
    assert scope.datasources == ()
    assert "california_schools" in real_agent_config.services.datasources
    with pytest.raises(DatusException, match="scoped_context.datasource"):
        scope.require_tool_scope("db_tools.execute_sql", {"sql": "select 1"})


def test_empty_grouped_table_selection_is_not_a_wildcard(real_agent_config):
    with pytest.raises(DatusException, match="requires a table selection"):
        _scope(real_agent_config, {"tables_by_datasource": {"california_schools": ""}})

    real_agent_config.agentic_nodes["research"] = {
        "scoped_context": {"tables_by_datasource": {"california_schools": ""}}
    }
    with pytest.raises(DatusException, match="requires a table selection"):
        _scope(real_agent_config, {"datasource": "california_schools"})


def test_delegated_configurable_child_intersects_parent_tables_and_files(real_agent_config):
    real_agent_config.agentic_nodes["research"] = {
        "node_class": "configurable",
        "scoped_context": {"datasource": "california_schools", "tables": "*", "project_files": ["**"]},
    }
    task = SubAgentTaskTool(real_agent_config)
    task.set_parent_node(
        SimpleNamespace(
            node_config={
                "scoped_context": {
                    "datasource": "california_schools",
                    "tables": "schools",
                    "project_files": ["public/**"],
                }
            }
        )
    )
    effective = task._resolve_effective_sub_agent_config("research")
    with effective_subagent("research", effective):
        scope = _scope(real_agent_config, real_agent_config.agentic_nodes["research"]["scoped_context"])
    public = scope.root / "public/reference.md"
    public.parent.mkdir(parents=True)
    public.write_text("Public evidence")
    private = scope.root / "private.md"
    private.write_text("Private evidence")
    assert _filesystem(scope).read_file("public/reference.md").success == 1
    assert _filesystem(scope).read_file("private.md").success == 0
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            assert tools.execute_sql("SELECT COUNT(*) FROM schools").success == 1
            denied = tools.execute_sql("SELECT COUNT(*) FROM satscores")
            assert denied.success == 0
            assert "outside scoped context" in denied.error
        finally:
            tools.close()


def test_grouped_child_cannot_replace_an_inherited_single_source_table_filter(real_agent_config):
    parent = SubAgentConfig(scoped_context=ScopedContext(datasource="california_schools", tables="schools"))
    with effective_subagent("research", parent):
        broad = _scope(real_agent_config, {"tables_by_datasource": {"california_schools": "*"}})
        allowed = _scope(real_agent_config, {"tables_by_datasource": {"california_schools": "schools"}})
    assert broad.effective.tables_by_datasource == {"california_schools": "schools"}
    assert allowed.effective.tables_by_datasource == {"california_schools": "schools"}


@pytest.mark.parametrize(
    "identity,args",
    [
        ("db_tools.execute_sql", {"sql": "private.sql"}),
        ("db_tools.load_file_as_table", {"path": "private.csv"}),
        ("db_tools.import_database_file", {"path": "private.duckdb"}),
        ("db_tools.export_query", {"sql": "select 1", "output_file": "private.csv"}),
        ("db_tools.transfer_query_result", {"target_datasource": "other"}),
    ],
)
def test_indirect_file_and_cross_datasource_operations_cannot_bypass_scope(real_agent_config, identity, args):
    scope = _scope(real_agent_config, {"datasource": "california_schools"})
    with pytest.raises(DatusException, match="scoped_context"):
        scope.require_tool_scope(identity, args)


@pytest.mark.parametrize("selection", ["../secret", "/private/file", "public/../../private"])
def test_invalid_file_patterns_fail_before_execution(real_agent_config, selection):
    with pytest.raises(DatusException, match="relative paths"):
        _scope(real_agent_config, {"project_files": [selection]})


def test_single_datasource_selects_the_actual_connection_without_changing_caller(real_agent_config, tmp_path):
    database = tmp_path / "warehouse.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schools (id INTEGER)")
        connection.executemany("INSERT INTO schools VALUES (?)", [(1,), (2,), (3,)])
        connection.execute("CREATE TABLE satscores (id INTEGER)")
    real_agent_config.services.datasources["warehouse"] = replace(
        real_agent_config.services.datasources["california_schools"], uri=str(database), default=False
    )
    scope = _scope(real_agent_config, {"datasource": "warehouse", "tables": "schools"})
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            result = tools.execute_sql("SELECT COUNT(*) AS n FROM schools")
            denied = tools.execute_sql("SELECT COUNT(*) FROM satscores")
            assert result.success == 1, result.error
            assert int(result.result["compressed_data"].splitlines()[-1].split(",")[-1]) == 3
            assert denied.success == 0
            assert "outside scoped context" in denied.error
            assert tools.execute_sql("SELECT COUNT(*) FROM schools", datasource="california_schools").success == 0
        finally:
            tools.close()
    assert scope.config.current_datasource == "warehouse"
    assert real_agent_config.current_datasource == "california_schools"
    assert set(real_agent_config.services.datasources) == {"warehouse", "california_schools"}


def test_grouped_scope_routes_each_source_with_its_own_table_filter(real_agent_config):
    real_agent_config.services.datasources["warehouse"] = replace(
        real_agent_config.services.datasources["california_schools"]
    )
    scope = _scope(
        real_agent_config,
        {
            "datasource": "warehouse",
            "tables_by_datasource": {"warehouse": "schools", "california_schools": "satscores"},
        },
    )
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            assert tools.execute_sql("SELECT COUNT(*) FROM schools").success == 1
            denied = tools.execute_sql("SELECT COUNT(*) FROM schools", datasource="california_schools")
            assert denied.success == 0
            assert "outside scoped context" in denied.error
            assert tools.execute_sql("SELECT COUNT(*) FROM satscores", datasource="california_schools").success == 1
        finally:
            tools.close()
    scope.prepare(
        ScopedContext(datasource="california_schools", tables_by_datasource={"california_schools": "satscores"})
    )
    assert scope.datasources == ("california_schools",)
    assert scope.config.current_datasource == "california_schools"
    with pytest.raises(DatusException, match="datasource"):
        scope.prepare(ScopedContext(datasource="warehouse", tables_by_datasource={"california_schools": "satscores"}))


def test_runtime_table_filter_restricts_every_grouped_datasource_and_resets_next_turn(real_agent_config):
    real_agent_config.services.datasources["warehouse"] = replace(
        real_agent_config.services.datasources["california_schools"]
    )
    scope = _scope(
        real_agent_config,
        {"tables_by_datasource": {"warehouse": "*", "california_schools": "*"}},
    )
    scope.prepare(ScopedContext(tables="schools"))
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            for datasource in ("warehouse", "california_schools"):
                allowed = tools.execute_sql("SELECT COUNT(*) FROM schools", datasource=datasource)
                assert allowed.success == 1, allowed.error
                denied = tools.execute_sql("SELECT COUNT(*) FROM satscores", datasource=datasource)
                assert denied.success == 0
                assert "outside scoped context" in denied.error
        finally:
            tools.close()
    scope.prepare(None)
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            restored = tools.execute_sql("SELECT COUNT(*) FROM satscores", datasource="warehouse")
            assert restored.success == 1, restored.error
        finally:
            tools.close()


def test_runtime_grouped_filter_further_narrows_the_common_table_filter(real_agent_config):
    real_agent_config.services.datasources["warehouse"] = replace(
        real_agent_config.services.datasources["california_schools"]
    )
    scope = _scope(real_agent_config, {"tables_by_datasource": {"warehouse": "*", "california_schools": "satscores"}})
    scope.prepare(
        ScopedContext(datasource="warehouse", tables="schools,satscores", tables_by_datasource={"warehouse": "schools"})
    )
    with scope.activate():
        tools = DBFuncTool.create_dynamic(scope.config, sub_agent_name="research")
        try:
            assert tools.execute_sql("SELECT COUNT(*) FROM schools").success == 1
            denied = tools.execute_sql("SELECT COUNT(*) FROM satscores")
            assert denied.success == 0
            assert "outside scoped context" in denied.error
            assert tools.execute_sql("SELECT COUNT(*) FROM satscores", datasource="california_schools").success == 0
        finally:
            tools.close()


def test_runtime_common_table_filter_cannot_widen_a_grouped_source(real_agent_config):
    scope = _scope(real_agent_config, {"tables_by_datasource": {"california_schools": "schools"}})
    with pytest.raises(DatusException, match="must narrow"):
        scope.prepare(ScopedContext(tables="*"))


@pytest.mark.parametrize(
    "field,parent,child",
    [
        ("tables", "sch*,satscores", "schools"),
        ("metrics", "Finance.Revenue", "Finance.Revenue.Daily"),
        ("sqls", "Finance.Reports", "Finance.Reports.Monthly"),
    ],
)
def test_runtime_partial_scope_preserves_static_bounds_and_rejects_widening(real_agent_config, field, parent, child):
    scope = _scope(
        real_agent_config, {"datasource": "california_schools", field: parent, "project_files": ["public/**"]}
    )
    scope.prepare(ScopedContext.model_validate({field: child}))
    assert scope.config.current_datasource == "california_schools"
    assert getattr(scope.effective, field) == child
    assert scope.allows_file(scope.root / "public" / "report.txt")
    with pytest.raises(DatusException, match="must narrow"):
        scope.prepare(ScopedContext.model_validate({field: "Unrelated"}))
    with pytest.raises(DatusException, match="cannot clear"):
        scope.prepare(ScopedContext.model_validate({field: ""}))
    scope.prepare(None)
    assert getattr(scope.effective, field) == parent


@pytest.mark.parametrize("kind", ["metrics", "sqls"])
def test_runtime_subject_filter_reaches_real_rag_reads_inside_a_parent_override(real_agent_config, kind):
    rag_class = MetricRAG if kind == "metrics" else ReferenceSqlRAG
    writer = rag_class(real_agent_config)
    rows = []
    for name in ("Allowed", "Private"):
        row = {
            "id": name,
            "name": name,
            "subject_path": ["Finance", name],
            "sql": "SELECT 1",
        }
        if kind == "metrics":
            row.update(description=name, semantic_model_name="orders", metric_type="simple", measure_expr="COUNT(*)")
        else:
            row.update(summary=name, search_text=name)
        rows.append(row)
    writer.store_batch(rows)
    parent = SubAgentConfig(scoped_context=ScopedContext.model_validate({kind: "Finance"}))
    with effective_subagent("research", parent):
        scope = _scope(real_agent_config, {"datasource": "california_schools", kind: "Finance"})
        scope.prepare(ScopedContext.model_validate({kind: "Finance.Allowed"}))
        reader = rag_class(scope.config, sub_agent_name="research")
        search = reader.search_all_metrics if kind == "metrics" else reader.search_all_reference_sql
        with scope.activate():
            assert {row["name"] for row in search()} == {"Allowed"}
        assert {row["name"] for row in search()} == {"Allowed", "Private"}
