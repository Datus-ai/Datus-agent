# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/metadata_commands.py — MetadataCommands.

All external dependencies (db_connector, agent_config, console) are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.cli.metadata_commands import MetadataCommands
from datus.utils.constants import DBType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_db_config(db_type=DBType.SQLITE, logic_name="mydb", database="mydb.db", uri="sqlite:///mydb.db"):
    cfg = MagicMock()
    cfg.type = db_type
    cfg.logic_name = logic_name
    cfg.database = database
    cfg.uri = uri
    return cfg


def _make_cli(db_type=DBType.SQLITE):
    cli = MagicMock()
    cli.console = MagicMock()

    # db_connector
    connector = MagicMock()
    connector.dialect = db_type
    connector.get_type.return_value = db_type
    cli.db_connector = connector

    # agent_config
    db_cfg = _make_db_config(db_type=db_type)
    cli.agent_config.current_namespace = "test_ns"
    cli.agent_config.namespaces = {"test_ns": {"mydb": db_cfg}}
    cli.agent_config.db_type = db_type

    # cli_context
    cli.cli_context.current_catalog = None
    cli.cli_context.current_db_name = "mydb"
    cli.cli_context.current_schema = None
    cli.cli_context.current_logic_db_name = "mydb"

    return cli


@pytest.fixture
def meta():
    cli = _make_cli()
    return MetadataCommands(cli)


# ---------------------------------------------------------------------------
# Tests: cmd_list_databases
# ---------------------------------------------------------------------------


class TestCmdListDatabases:
    def test_single_sqlite_db(self, meta):
        meta.cmd_list_databases()
        meta.cli.console.print.assert_called()

    def test_multi_db(self):
        cli = _make_cli()
        db_cfg1 = _make_db_config(logic_name="db1")
        db_cfg2 = _make_db_config(logic_name="db2")
        cli.agent_config.namespaces = {"test_ns": {"db1": db_cfg1, "db2": db_cfg2}}
        meta = MetadataCommands(cli)
        meta.cmd_list_databases()
        cli.console.print.assert_called()

    def test_exception_prints_error(self, meta):
        meta.cli.agent_config.current_namespace = None
        meta.cli.agent_config.namespaces = {}
        meta.cmd_list_databases()
        meta.cli.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: cmd_switch_database
# ---------------------------------------------------------------------------


class TestCmdSwitchDatabase:
    def test_no_name_lists_databases(self):
        cli = _make_cli()
        meta = MetadataCommands(cli)
        meta.cmd_list_databases = MagicMock()
        meta.cmd_switch_database("")
        cli.console.print.assert_called()

    def test_same_db_sqlite_already_current(self):
        cli = _make_cli(db_type=DBType.SQLITE)
        cli.cli_context.current_logic_db_name = "mydb"
        cli.db_connector.dialect = DBType.SQLITE
        meta = MetadataCommands(cli)
        meta.cmd_switch_database("mydb")
        calls = [str(c) for c in cli.console.print.call_args_list]
        assert any("doesn't need" in c for c in calls)

    def test_same_db_name_already_current(self):
        cli = _make_cli(db_type="snowflake")
        cli.cli_context.current_db_name = "targetdb"
        cli.cli_context.current_logic_db_name = "other"
        cli.db_connector.dialect = "snowflake"
        cli.agent_config.db_type = "snowflake"
        meta = MetadataCommands(cli)
        meta.cmd_switch_database("targetdb")
        calls = [str(c) for c in cli.console.print.call_args_list]
        assert any("doesn't need" in c for c in calls)

    def test_switch_non_sqlite(self):
        cli = _make_cli(db_type="snowflake")
        cli.cli_context.current_db_name = "old_db"
        cli.cli_context.current_logic_db_name = "old_db"
        cli.db_connector.dialect = "snowflake"
        cli.agent_config.db_type = "snowflake"
        meta = MetadataCommands(cli)
        meta.cmd_switch_database("new_db")
        calls = [str(c) for c in cli.console.print.call_args_list]
        assert any("new_db" in c for c in calls)

    def test_switch_sqlite_db_not_found(self):
        cli = _make_cli(db_type=DBType.SQLITE)
        cli.cli_context.current_logic_db_name = "old_db"
        cli.db_connector.dialect = DBType.SQLITE
        cli.agent_config.db_type = DBType.SQLITE
        cli.agent_config.current_db_configs.return_value = {}  # empty -> not found
        meta = MetadataCommands(cli)
        meta.cmd_switch_database("unknown_db")
        calls = [str(c) for c in cli.console.print.call_args_list]
        assert any("No corresponding database" in c or "not found" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_tables
# ---------------------------------------------------------------------------


class TestCmdTables:
    def test_no_connector(self):
        cli = _make_cli()
        cli.db_connector = None
        meta = MetadataCommands(cli)
        meta.cmd_tables("")
        cli.console.print.assert_called_with("[bold red]Error:[/] No database connection.")

    def test_empty_result(self, meta):
        meta.cli.db_connector.get_tables.return_value = []
        meta.cmd_tables("")
        meta.cli.console.print.assert_called_with("[yellow]Empty set.[/]")

    def test_with_tables(self, meta):
        meta.cli.db_connector.get_tables.return_value = ["users", "orders"]
        meta.cmd_tables("")
        meta.cli.console.print.assert_called()

    def test_exception_prints_error(self, meta):
        meta.cli.db_connector.get_tables.side_effect = RuntimeError("db error")
        meta.cmd_tables("")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("db error" in c for c in calls)

    def test_with_schema_context(self, meta):
        meta.cli.cli_context.current_schema = "myschema"
        meta.cli.db_connector.get_tables.return_value = ["t1"]
        meta.cmd_tables("")
        meta.cli.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: cmd_schemas
# ---------------------------------------------------------------------------


class TestCmdSchemas:
    def test_unsupported_dialect(self):
        cli = _make_cli(db_type=DBType.SQLITE)
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = False
            meta = MetadataCommands(cli)
            meta.cmd_schemas("")
        cli.console.print.assert_called()

    def test_empty_schemas(self):
        cli = _make_cli(db_type="snowflake")
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = True
            cli.db_connector.get_schemas.return_value = []
            meta = MetadataCommands(cli)
            meta.cmd_schemas("")
        cli.console.print.assert_called_with("[yellow]Empty set.[/]")

    def test_with_schemas(self):
        cli = _make_cli(db_type="snowflake")
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = True
            cli.db_connector.get_schemas.return_value = ["public", "private"]
            meta = MetadataCommands(cli)
            meta.cmd_schemas("")
        cli.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: cmd_switch_schema
# ---------------------------------------------------------------------------


class TestCmdSwitchSchema:
    def test_unsupported_dialect(self):
        cli = _make_cli(db_type=DBType.SQLITE)
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = False
            meta = MetadataCommands(cli)
            meta.cmd_switch_schema("public")
        cli.console.print.assert_called()

    def test_empty_name(self):
        cli = _make_cli()
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = True
            meta = MetadataCommands(cli)
            meta.cmd_switch_schema("")
        cli.console.print.assert_called()

    def test_success(self):
        cli = _make_cli()
        with patch("datus.cli.metadata_commands.connector_registry") as mock_reg:
            mock_reg.support_schema.return_value = True
            meta = MetadataCommands(cli)
            meta.cmd_switch_schema("myschema")
        assert cli.cli_context.current_schema == "myschema"
        calls = [str(c) for c in cli.console.print.call_args_list]
        assert any("myschema" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_table_schema
# ---------------------------------------------------------------------------


class TestCmdTableSchema:
    def test_no_connector(self):
        cli = _make_cli()
        cli.db_connector = None
        meta = MetadataCommands(cli)
        meta.cmd_table_schema("users")
        cli.console.print.assert_called_with("[bold red]Error:[/] No database connection.")

    def test_specific_table(self, meta):
        meta.cli.db_connector.get_schema.return_value = [
            {"cid": 0, "name": "id", "type": "INTEGER", "nullable": False, "default_value": None, "pk": 1}
        ]
        meta.cmd_table_schema("users")
        meta.cli.console.print.assert_called()

    def test_no_table_lists_all(self, meta):
        meta.cli.db_connector.get_tables.return_value = ["users", "orders"]
        meta.cmd_table_schema("")
        meta.cli.console.print.assert_called()

    def test_exception_prints_error(self, meta):
        meta.cli.db_connector.get_schema.side_effect = RuntimeError("schema error")
        meta.cmd_table_schema("users")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("schema error" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_indexes
# ---------------------------------------------------------------------------


class TestCmdIndexes:
    def test_no_table_name(self, meta):
        meta.cmd_indexes("")
        meta.cli.console.print.assert_called_with("[bold red]Error:[/] Table name required")

    def test_no_connector(self):
        cli = _make_cli()
        cli.db_connector = None
        meta = MetadataCommands(cli)
        meta.cmd_indexes("users")
        cli.console.print.assert_called_with("[bold red]Error:[/] No database connection.")

    def test_non_sqlite_not_supported(self, meta):
        meta.cli.db_connector.get_type.return_value = "snowflake"
        meta.cmd_indexes("users")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("not yet supported" in c for c in calls)

    def test_sqlite_no_indexes(self, meta):
        meta.cli.db_connector.get_type.return_value = DBType.SQLITE
        import pandas as pd

        mock_result = MagicMock()
        mock_result.success = True
        empty_df = pd.DataFrame({"name": []})
        mock_result.sql_return = empty_df
        meta.cli.db_connector.execute_pandas.return_value = mock_result
        meta.cmd_indexes("users")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("no indexes" in c for c in calls)

    def test_sqlite_with_indexes(self, meta):
        meta.cli.db_connector.get_type.return_value = DBType.SQLITE
        import pandas as pd

        mock_result = MagicMock()
        mock_result.success = True
        df = pd.DataFrame({"name": ["idx_users_email"]})
        mock_result.sql_return = df
        meta.cli.db_connector.execute_pandas.return_value = mock_result
        meta.cmd_indexes("users")
        meta.cli.console.print.assert_called()

    def test_sqlite_query_failed(self, meta):
        meta.cli.db_connector.get_type.return_value = DBType.SQLITE
        mock_result = MagicMock()
        mock_result.success = False
        meta.cli.db_connector.execute_pandas.return_value = mock_result
        meta.cmd_indexes("users")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("Query failed" in c for c in calls)

    def test_exception_prints_error(self, meta):
        meta.cli.db_connector.get_type.return_value = DBType.SQLITE
        meta.cli.db_connector.execute_pandas.side_effect = RuntimeError("exec error")
        meta.cmd_indexes("users")
        calls = [str(c) for c in meta.cli.console.print.call_args_list]
        assert any("exec error" in c for c in calls)
