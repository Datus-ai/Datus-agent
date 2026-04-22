# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the DBFuncTool migration-target wrapper tools.

These three tools duck-type the ``MigrationTargetMixin`` API on the underlying
connector. If the connector implements the Mixin, the tool forwards to it.
Otherwise, the tool returns a safe fallback so the migration agent can
continue in pure-LLM mode.

The tests use mock connectors rather than real adapters to keep them
CI-safe and decoupled from adapter version bumps.
"""

from unittest.mock import Mock

import pytest

from datus.tools.func_tool.database import DBFuncTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_tool_with_connector(connector: Mock) -> DBFuncTool:
    """Build a minimal DBFuncTool and attach the mocked connector.

    We bypass the full DBFuncTool constructor (which wants real adapters)
    by using __new__ and injecting the attributes the wrapper methods read.
    Single-connector mode: ``_db_manager = None`` → ``_primary_connector``
    is used regardless of the ``database`` argument.
    """
    tool = DBFuncTool.__new__(DBFuncTool)
    tool._primary_connector = connector
    tool._db_manager = None
    tool._default_database = "default"
    return tool


@pytest.fixture
def mixin_connector():
    """Connector that implements the MigrationTargetMixin contract."""
    c = Mock()
    c.dialect = "starrocks"
    c.describe_migration_capabilities.return_value = {
        "supported": True,
        "dialect_family": "mysql-like",
        "requires": ["DUPLICATE KEY", "DISTRIBUTED BY"],
        "forbids": ["AUTO_INCREMENT"],
        "type_hints": {"unbounded VARCHAR": "VARCHAR(65533)"},
        "example_ddl": "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10",
    }
    c.suggest_table_layout.return_value = {"duplicate_key": ["id"], "distributed_by": ["id"], "buckets": 10}

    def _validate(ddl):
        return [] if "DUPLICATE KEY" in ddl.upper() else ["missing DUPLICATE KEY"]

    c.validate_ddl.side_effect = _validate
    c.dry_run_ddl.return_value = []
    return c


@pytest.fixture
def non_mixin_connector():
    """Connector that does NOT implement MigrationTargetMixin."""
    c = Mock(spec=["dialect"])  # spec limits attributes to just `dialect`
    c.dialect = "sqlite"
    return c


# ---------------------------------------------------------------------------
# get_migration_capabilities
# ---------------------------------------------------------------------------


class TestGetMigrationCapabilities:
    def test_forwards_to_mixin(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.get_migration_capabilities(database="default")
        assert result.success == 1
        assert result.result["supported"] is True
        assert result.result["dialect_family"] == "mysql-like"
        assert "DUPLICATE KEY" in " ".join(result.result["requires"])

    def test_fallback_when_no_mixin(self, non_mixin_connector):
        tool = _build_tool_with_connector(non_mixin_connector)
        result = tool.get_migration_capabilities(database="default")
        assert result.success == 1
        assert result.result["supported"] is False
        # Pin the exact fallback-mode marker the production code emits so the
        # prompt rendering stays stable when the LLM reads the warning.
        assert "falling back to pure LLM mode" in result.result["warning"]


# ---------------------------------------------------------------------------
# suggest_table_layout
# ---------------------------------------------------------------------------


class TestSuggestTableLayout:
    def test_forwards_to_mixin(self, mixin_connector):
        import json as _json

        tool = _build_tool_with_connector(mixin_connector)
        columns_json = _json.dumps([{"name": "id", "type": "BIGINT", "nullable": False}])
        result = tool.suggest_table_layout(database="default", columns_json=columns_json)
        assert result.success == 1
        assert result.result["duplicate_key"] == ["id"]
        assert result.result["buckets"] == 10

    def test_fallback_returns_empty_dict(self, non_mixin_connector):
        import json as _json

        tool = _build_tool_with_connector(non_mixin_connector)
        columns_json = _json.dumps([{"name": "id", "type": "BIGINT", "nullable": False}])
        result = tool.suggest_table_layout(database="default", columns_json=columns_json)
        assert result.success == 1
        assert result.result == {}

    def test_invalid_json_is_rejected(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.suggest_table_layout(database="default", columns_json="not valid json")
        assert result.success == 0
        # Pin the exact prefix the production code emits.
        assert result.error.startswith("Invalid columns_json:")

    def test_non_array_json_is_rejected(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.suggest_table_layout(database="default", columns_json='{"not": "array"}')
        assert result.success == 0
        assert "array" in result.error.lower()


# ---------------------------------------------------------------------------
# validate_ddl
# ---------------------------------------------------------------------------


class TestValidateDdl:
    def test_forwards_to_mixin_returns_errors(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.validate_ddl(
            database="default",
            ddl="CREATE TABLE t (id BIGINT) DISTRIBUTED BY HASH(id) BUCKETS 10",  # missing DUPLICATE KEY
        )
        assert result.success == 1
        assert result.result["errors"]
        assert any("DUPLICATE KEY" in e for e in result.result["errors"])

    def test_forwards_to_mixin_no_errors(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        result = tool.validate_ddl(database="default", ddl=ddl)
        assert result.success == 1
        assert result.result["errors"] == []

    def test_dry_run_runs_when_target_table_provided(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        result = tool.validate_ddl(database="default", ddl=ddl, target_table="t")
        assert result.success == 1
        mixin_connector.dry_run_ddl.assert_called_once()

    def test_dry_run_skipped_when_target_table_absent(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        tool.validate_ddl(database="default", ddl=ddl)
        mixin_connector.dry_run_ddl.assert_not_called()

    def test_dry_run_not_implemented_is_swallowed(self, mixin_connector):
        mixin_connector.dry_run_ddl.side_effect = NotImplementedError
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        result = tool.validate_ddl(database="default", ddl=ddl, target_table="t")
        # Static validation succeeded; dry_run absence must not fail the tool.
        assert result.success == 1
        assert result.result["errors"] == []

    def test_fallback_returns_no_errors_when_no_mixin(self, non_mixin_connector):
        """No Mixin → no static checks. LLM is solely responsible."""
        tool = _build_tool_with_connector(non_mixin_connector)
        result = tool.validate_ddl(database="default", ddl="CREATE TABLE t (id BIGINT)")
        assert result.success == 1
        assert result.result["errors"] == []
        assert result.result.get("validated") is False

    def test_empty_ddl_is_rejected(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.validate_ddl(database="default", ddl="")
        assert result.success == 0
        assert result.error == "Empty DDL statement"

    def test_whitespace_only_ddl_is_rejected(self, mixin_connector):
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.validate_ddl(database="default", ddl="   \n  \t  ")
        assert result.success == 0

    def test_static_check_raises_unexpectedly_is_recorded(self, mixin_connector):
        """If connector.validate_ddl raises, the error is swallowed and recorded."""
        mixin_connector.validate_ddl.side_effect = RuntimeError("boom")
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.validate_ddl(
            database="default",
            ddl="CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10",
        )
        assert result.success == 1
        assert any("Static check raised" in e for e in result.result["errors"])

    def test_dry_run_unexpected_error_is_recorded(self, mixin_connector):
        """Unexpected dry_run error (not NotImplementedError) is swallowed and recorded."""
        mixin_connector.dry_run_ddl.side_effect = RuntimeError("permission denied")
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        result = tool.validate_ddl(database="default", ddl=ddl, target_table="t")
        assert result.success == 1
        assert any("Dry-run raised" in e for e in result.result["errors"])

    def test_dry_run_returns_errors_appends_them(self, mixin_connector):
        """If dry_run_ddl returns a list of errors, they are appended."""
        mixin_connector.dry_run_ddl.return_value = ["schema already exists"]
        tool = _build_tool_with_connector(mixin_connector)
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        result = tool.validate_ddl(database="default", ddl=ddl, target_table="t")
        assert result.success == 1
        assert "schema already exists" in result.result["errors"]


# ---------------------------------------------------------------------------
# Error paths in get_migration_capabilities / suggest_table_layout
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_get_migration_capabilities_swallows_unexpected_error(self, mixin_connector):
        mixin_connector.describe_migration_capabilities.side_effect = RuntimeError("adapter bug")
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.get_migration_capabilities(database="default")
        assert result.success == 1
        assert result.result["supported"] is False
        assert "adapter bug" in result.result["warning"]

    def test_suggest_table_layout_swallows_unexpected_error(self, mixin_connector):
        mixin_connector.suggest_table_layout.side_effect = RuntimeError("adapter bug")
        tool = _build_tool_with_connector(mixin_connector)
        result = tool.suggest_table_layout(
            database="default",
            columns_json='[{"name": "id", "type": "BIGINT", "nullable": false}]',
        )
        # Swallowed → treated as "no suggestion", returning empty dict.
        assert result.success == 1
        assert result.result == {}


# ---------------------------------------------------------------------------
# _get_connector failures (e.g. database name is not configured)
# ---------------------------------------------------------------------------


class TestUnknownDatabaseRaises:
    """When _get_connector raises DatusException, all three wrappers propagate it."""

    def _tool_with_raising_get_connector(self, connector: Mock) -> DBFuncTool:
        from datus.utils.exceptions import DatusException, ErrorCode

        tool = DBFuncTool.__new__(DBFuncTool)
        tool._primary_connector = connector
        tool._db_manager = None
        tool._default_database = "default"

        def _raise(_database=None):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message="Database 'nonexistent' is not configured.",
            )

        tool._get_connector = _raise  # type: ignore[method-assign]
        return tool

    def test_get_migration_capabilities_propagates_datus_exception(self, mixin_connector):
        tool = self._tool_with_raising_get_connector(mixin_connector)
        result = tool.get_migration_capabilities(database="nonexistent")
        assert result.success == 0
        assert "not configured" in result.error

    def test_suggest_table_layout_propagates_datus_exception(self, mixin_connector):
        tool = self._tool_with_raising_get_connector(mixin_connector)
        result = tool.suggest_table_layout(database="nonexistent", columns_json="[]")
        assert result.success == 0
        assert "not configured" in result.error

    def test_validate_ddl_propagates_datus_exception(self, mixin_connector):
        tool = self._tool_with_raising_get_connector(mixin_connector)
        result = tool.validate_ddl(database="nonexistent", ddl="CREATE TABLE t (id BIGINT)")
        assert result.success == 0
        assert "not configured" in result.error
