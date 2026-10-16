"""The optional analysis contract works without installing any vendor SDK."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from datus_db_core import ExecuteSQLResult

from datus.tools.db_tools import connector_registry
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.lineage_tools import LineageTools
from datus.utils.constants import SQLType
from datus.utils.sql_utils import (
    extract_table_names,
    extract_table_names_strict,
    format_sql_to_pretty,
    parse_sql_type,
    validate_read_only_sql,
)


@dataclass
class Analysis:
    status: str
    sql_type: SQLType = SQLType.UNKNOWN
    tables: tuple = ()
    diagnostic: str = "unsupported syntax"

    def require_complete(self):
        if self.status != "complete":
            raise ValueError(self.diagnostic)
        return self


@pytest.fixture
def analyzer(monkeypatch):
    def analyze(sql):
        if sql == 'SELECT * FROM "S"."t"':
            return Analysis("complete", SQLType.SELECT, ('"S"."t"',))
        if sql == "SELECT 1":
            return Analysis("complete", SQLType.SELECT)
        if sql == "SELECT * FROM t":
            return Analysis("complete", SQLType.SELECT, ('"t"',))
        return Analysis("unsupported")

    # The registry is the external adapter boundary. Old db-core installations
    # need not expose this optional method; the Agent remains compatible.
    monkeypatch.setattr(
        connector_registry,
        "get_sql_analyzer",
        lambda dialect: analyze if dialect == "native-test" else None,
        raising=False,
    )


def test_adapter_supplies_table_facts_without_generator(analyzer):
    sql = 'SELECT * FROM "S"."t"'
    assert extract_table_names(sql, "native-test") == ['"S"."t"']
    assert extract_table_names_strict(sql, "native-test") == ['"S"."t"']
    assert parse_sql_type(sql, "native-test") == SQLType.SELECT
    assert format_sql_to_pretty(sql, "native-test") == sql


def test_complete_no_table_query_is_distinct_from_failed_analysis(analyzer):
    assert extract_table_names("SELECT 1", "native-test") == []
    for extract in (extract_table_names, extract_table_names_strict):
        with pytest.raises(ValueError, match="unsupported syntax"):
            extract("opaque statement", "native-test")
    assert parse_sql_type("SELECT vendor syntax", "native-test") == SQLType.UNKNOWN
    assert validate_read_only_sql("SELECT vendor syntax", "native-test")[0] is not None


def test_unregistered_dialects_keep_existing_path(analyzer):
    assert extract_table_names("SELECT * FROM test_table", "sqlite", ignore_empty=True) == ["test_table"]
    assert parse_sql_type("INSERT INTO test_table VALUES (1)", "sqlite") == SQLType.INSERT


@pytest.fixture
def native_connector(analyzer):
    return SimpleNamespace(
        dialect="native-test",
        database_name="",
        schema_name="",
        get_effective_capabilities=lambda: {"database"},
        execute_query=Mock(return_value=ExecuteSQLResult(success=True, sql_return=[{"id": 1}], result_format="list")),
    )


def test_complete_analysis_allows_read_and_preserves_original_sql(native_connector):
    sql = 'SELECT * FROM "S"."t"'
    result = DBFuncTool(native_connector, read_only=True).execute_read_enforced(sql, native_connector)
    assert result.success is True, result.error
    assert result.sql_return == [{"id": 1}]
    native_connector.execute_query.assert_called_once_with(sql, result_format="list")


def test_unsupported_analysis_rejects_read_before_connector_dispatch(native_connector):
    result = DBFuncTool(native_connector, read_only=True).execute_read_enforced("opaque statement", native_connector)
    assert not result.success
    assert "SQL analysis unsupported" in result.error
    native_connector.execute_query.assert_not_called()


@pytest.mark.parametrize("database", ["", "S"])
def test_scoped_native_read_requires_resolved_namespace(native_connector, monkeypatch, database):
    # SQLite supplies a real identifier parser without any optional vendor SDK.
    getter = connector_registry.get_sql_analyzer
    monkeypatch.setattr(connector_registry, "get_sql_analyzer", lambda dialect: getter("native-test"))
    native_connector.dialect = "sqlite"
    native_connector.database_name = database
    tool = DBFuncTool(native_connector, read_only=True, scoped_tables=["S.t"])
    result = tool.execute_read_enforced("SELECT * FROM t", native_connector)
    if database:
        assert result.success is True, result.error
        native_connector.execute_query.assert_called_once()
    else:
        assert not result.success
        assert "Unresolved database" in result.error
        native_connector.execute_query.assert_not_called()


def test_limited_analyzer_cannot_execute_policy_generated_sql(native_connector):
    decision = SimpleNamespace(allowed=True, sql="SELECT 1 LIMIT 1", applied_policies=[])
    with patch("datus.tools.policy_runtime.PolicyRuntime.before_sql_read", return_value=decision):
        result = DBFuncTool(native_connector, read_only=True).execute_read_enforced("SELECT 1", native_connector)
    assert not result.success
    assert "rewriting is unsupported" in result.error
    native_connector.execute_query.assert_not_called()


def test_static_lineage_requires_a_supported_analysis_backend(analyzer, tmp_path):
    result = LineageTools(root_path=str(tmp_path)).upsert_lineage(sql="SELECT 1", dialect="native-test")
    assert not result.success
    assert "lineage is unsupported" in result.error
