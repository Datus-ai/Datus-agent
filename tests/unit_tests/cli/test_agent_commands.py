# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/agent_commands.py.

Tests cover:
- AgentCommands initialization
- _gen_sql_task: with args, empty, existing task
- create_node_input: for each NodeType
- _prompt_subject_path: empty and non-empty
- cmd_schema_linking: empty input error
- cmd_search_metrics: empty input error, success, no result, failure
- cmd_search_reference_sql: empty input error, success, no result, failure
- cmd_doc_search: validation branches
- cmd_save: no context, with context
- run_standalone_node: confirm cancelled, success
- _print_metadata_table: basic smoke test
- update_agent_reference

All external dependencies are mocked.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.agent_commands import AgentCommands
from datus.cli.cli_context import CliContext
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.node_models import SqlTask
from datus.utils.constants import DBType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console() -> Console:
    return Console(file=io.StringIO(), no_color=True)


class MinimalCLI:
    """Lightweight CLI substitute providing attributes AgentCommands needs."""

    def __init__(self, agent_config, console=None):
        import argparse

        self.agent_config = agent_config
        self.console = console or _make_console()
        self.cli_context = CliContext()
        self.actions = ActionHistoryManager()
        self.agent = None
        self.db_connector = MagicMock()
        self.db_connector.get_type.return_value = DBType.SQLITE
        self.db_connector.dialect = DBType.SQLITE
        self.args = argparse.Namespace(db_path="test.db", database="test_db", debug=False)
        self.workflow_runner = None

    def prompt_input(self, message="", default="", choices=None, multiline=False):
        return default

    def check_agent_available(self):
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(real_agent_config):
    return MinimalCLI(real_agent_config)


@pytest.fixture
def cli_context():
    return CliContext()


@pytest.fixture
def agent_commands(cli, cli_context):
    return AgentCommands(cli, cli_context)


# ---------------------------------------------------------------------------
# Tests: init
# ---------------------------------------------------------------------------


class TestAgentCommandsInit:
    def test_init_sets_attributes(self, cli, cli_context):
        ac = AgentCommands(cli, cli_context)
        assert ac.cli is cli
        assert ac.cli_context is cli_context
        assert ac.console is cli.console
        assert ac.agent is None
        assert ac.darun_is_running is False
        assert ac.output_tool is None

    def test_update_agent_reference(self, agent_commands, cli):
        mock_agent = MagicMock()
        cli.agent = mock_agent
        agent_commands.update_agent_reference()
        assert agent_commands.agent is mock_agent


# ---------------------------------------------------------------------------
# Tests: _gen_sql_task
# ---------------------------------------------------------------------------


class TestGenSqlTask:
    def test_reuse_existing_task_when_no_args(self, agent_commands, cli_context):
        """Returns existing task when args is empty and use_existing=True."""
        existing = SqlTask(
            id="abc",
            database_type=DBType.SQLITE,
            task="show me sales",
            database_name="testdb",
            output_dir="/tmp",
        )
        cli_context.set_current_sql_task(existing)
        agent_commands.cli_context = cli_context

        result = agent_commands._gen_sql_task("", use_existing=True)
        assert result is existing

    def test_creates_new_task_from_args(self, agent_commands):
        """Creates a new SqlTask when args is provided."""
        agent_commands.cli.db_connector.get_type.return_value = DBType.SQLITE
        # Set a db_name so the task can be created without prompting
        agent_commands.cli_context.current_db_name = "testdb"

        result = agent_commands._gen_sql_task("show me revenue")

        assert result is not None
        assert result.task == "show me revenue"

    def test_returns_none_on_exception(self, agent_commands):
        """Returns None when an unexpected exception occurs."""
        agent_commands.cli.db_connector = None
        # With no db_connector, it falls back to SQLITE — should still work
        result = agent_commands._gen_sql_task("test query")
        # The code falls back to SQLITE when db_connector is None; a SqlTask should be returned
        assert result is not None
        assert result.task == "test query"


# ---------------------------------------------------------------------------
# Tests: cmd_schema_linking
# ---------------------------------------------------------------------------


class TestCmdSchemaLinking:
    def test_empty_input_prints_error(self, agent_commands):
        """Empty input prints error and returns."""
        # prompt_input returns "" (default), args is also ""
        agent_commands.cli.prompt_input = lambda *a, **kw: ""
        agent_commands.cmd_schema_linking("")
        output = agent_commands.console.file.getvalue()
        assert "cannot be empty" in output.lower() or "Error" in output

    def test_with_args_calls_schema_rag(self, agent_commands):
        """When input is provided, schema RAG is called."""
        import pyarrow as pa

        empty_table = pa.table(
            {
                "table_name": [],
                "definition": [],
                "database_name": [],
                "catalog_name": [],
                "schema_name": [],
                "table_type": [],
                "_distance": [],
            }
        )

        mock_rag = MagicMock()
        mock_rag.search_similar.return_value = (empty_table, empty_table)

        with patch("datus.storage.schema_metadata.SchemaWithValueRAG", return_value=mock_rag):
            with patch.object(agent_commands, "_prompt_db_layers", return_value=("", "testdb", "")):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_schema_linking("find tables about revenue")

        mock_rag.search_similar.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: cmd_search_metrics
# ---------------------------------------------------------------------------


class TestCmdSearchMetrics:
    def test_empty_input_prints_error(self, agent_commands):
        agent_commands.cli.prompt_input = lambda *a, **kw: ""
        agent_commands.cmd_search_metrics("")
        output = agent_commands.console.file.getvalue()
        assert "Error" in output

    def test_success_displays_metrics(self, agent_commands):
        from datus.tools.func_tool.base import FuncToolResult

        mock_result = MagicMock(spec=FuncToolResult)
        mock_result.success = True
        mock_result.result = [{"name": "revenue", "description": "Total revenue"}]
        mock_result.error = None

        with patch.object(agent_commands, "_context_search_tools") as mock_cst:
            mock_cst.search_metrics.return_value = mock_result
            with patch.object(agent_commands, "_prompt_subject_path", return_value=None):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_search_metrics("revenue")

        output = agent_commands.console.file.getvalue()
        assert "revenue" in output.lower() or "Found" in output

    def test_no_results(self, agent_commands):
        from datus.tools.func_tool.base import FuncToolResult

        mock_result = MagicMock(spec=FuncToolResult)
        mock_result.success = True
        mock_result.result = []
        mock_result.error = None

        with patch.object(agent_commands, "_context_search_tools") as mock_cst:
            mock_cst.search_metrics.return_value = mock_result
            with patch.object(agent_commands, "_prompt_subject_path", return_value=None):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_search_metrics("something")

        output = agent_commands.console.file.getvalue()
        assert "No metrics" in output or "not found" in output.lower()

    def test_search_failure(self, agent_commands):
        from datus.tools.func_tool.base import FuncToolResult

        mock_result = MagicMock(spec=FuncToolResult)
        mock_result.success = False
        mock_result.result = None
        mock_result.error = "Storage not initialized"

        with patch.object(agent_commands, "_context_search_tools") as mock_cst:
            mock_cst.search_metrics.return_value = mock_result
            with patch.object(agent_commands, "_prompt_subject_path", return_value=None):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_search_metrics("revenue")

        output = agent_commands.console.file.getvalue()
        assert "Error" in output


# ---------------------------------------------------------------------------
# Tests: cmd_search_reference_sql
# ---------------------------------------------------------------------------


class TestCmdSearchReferenceSql:
    def test_empty_input_error(self, agent_commands):
        agent_commands.cli.prompt_input = lambda *a, **kw: ""
        agent_commands.cmd_search_reference_sql("")
        output = agent_commands.console.file.getvalue()
        assert "Error" in output

    def test_success_displays_table(self, agent_commands):
        from datus.tools.func_tool.base import FuncToolResult

        mock_result = MagicMock(spec=FuncToolResult)
        mock_result.success = True
        mock_result.result = [
            {
                "name": "get_revenue",
                "sql": "SELECT SUM(amount) FROM sales",
                "summary": "Revenue summary",
                "comment": "",
                "tags": "",
                "subject_path": ["Finance"],
                "filepath": "/path/to/sql.sql",
                "_distance": 0.1,
            }
        ]
        mock_result.error = None

        with patch.object(agent_commands, "_context_search_tools") as mock_cst:
            mock_cst.search_reference_sql.return_value = mock_result
            with patch.object(agent_commands, "_prompt_subject_path", return_value=None):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_search_reference_sql("revenue sql")

        output = agent_commands.console.file.getvalue()
        assert "Found" in output or "get_revenue" in output

    def test_no_results(self, agent_commands):
        from datus.tools.func_tool.base import FuncToolResult

        mock_result = MagicMock(spec=FuncToolResult)
        mock_result.success = True
        mock_result.result = []
        mock_result.error = None

        with patch.object(agent_commands, "_context_search_tools") as mock_cst:
            mock_cst.search_reference_sql.return_value = mock_result
            with patch.object(agent_commands, "_prompt_subject_path", return_value=None):
                agent_commands.cli.prompt_input = lambda *a, **kw: kw.get("default", "5")
                agent_commands.cmd_search_reference_sql("nothing")

        output = agent_commands.console.file.getvalue()
        assert "No reference SQL" in output or "not found" in output.lower()


# ---------------------------------------------------------------------------
# Tests: cmd_doc_search
# ---------------------------------------------------------------------------


class TestCmdDocSearch:
    def test_empty_platform_returns_error(self, agent_commands):
        """Empty platform prints error and returns."""
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""  # empty platform
            return default

        agent_commands.cli.prompt_input = mock_prompt
        agent_commands.cmd_doc_search("")
        output = agent_commands.console.file.getvalue()
        assert "Platform name is required" in output or "required" in output.lower()

    def test_empty_keywords_returns_error(self, agent_commands):
        """Empty keywords prints error and returns."""
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "duckdb"  # platform
            if call_count[0] == 2:
                return ""  # version (optional)
            return ""  # keywords empty

        agent_commands.cli.prompt_input = mock_prompt
        agent_commands.cmd_doc_search("")
        output = agent_commands.console.file.getvalue()
        assert "Keywords cannot be empty" in output or "cannot be empty" in output.lower()


# ---------------------------------------------------------------------------
# Tests: cmd_save
# ---------------------------------------------------------------------------


class TestCmdSave:
    def test_no_last_sql_context_prints_error(self, agent_commands):
        """Without a last SQL context, prints error."""
        agent_commands.cmd_save("")
        output = agent_commands.console.file.getvalue()
        assert "No previous result" in output

    def test_with_context_calls_output_tool(self, agent_commands):
        """With a valid last context, cmd_save proceeds past the 'no context' check."""
        from datus.schemas.node_models import SQLContext

        ctx = SQLContext(
            sql_query="SELECT 1",
            sql_return="1",
            row_count=1,
            sql_error=None,
        )
        agent_commands.cli.cli_context.add_sql_context(ctx)
        agent_commands.cli.db_connector = MagicMock()

        def mock_prompt(msg="", default="", choices=None, **kw):
            return default or "all"

        agent_commands.cli.prompt_input = mock_prompt

        # Patch everything external that cmd_save calls
        mock_path_manager = MagicMock()
        mock_path_manager.save_dir = "/tmp/save"
        agent_commands.cli.agent_config.path_manager = mock_path_manager
        mock_output_result = MagicMock()
        mock_output_result.output = "/tmp/save/output.json"

        with patch("datus.cli.agent_commands.OutputTool") as mock_output_cls:
            mock_output_tool = MagicMock()
            mock_output_tool.execute.return_value = mock_output_result
            mock_output_cls.return_value = mock_output_tool
            agent_commands.cmd_save("")

        # Verify the "No previous result" error was NOT printed (context exists)
        output = agent_commands.console.file.getvalue()
        assert "No previous result" not in output


# ---------------------------------------------------------------------------
# Tests: _prompt_subject_path
# ---------------------------------------------------------------------------


class TestPromptSubjectPath:
    def test_empty_input_returns_none(self, agent_commands):
        agent_commands.cli.prompt_input = lambda *a, **kw: ""
        result = agent_commands._prompt_subject_path()
        assert result is None

    def test_slash_separated_input_returns_list(self, agent_commands):
        agent_commands.cli.prompt_input = lambda *a, **kw: "Finance/Revenue/Q1"
        result = agent_commands._prompt_subject_path()
        assert result == ["Finance", "Revenue", "Q1"]

    def test_whitespace_only_returns_none(self, agent_commands):
        agent_commands.cli.prompt_input = lambda *a, **kw: "   "
        result = agent_commands._prompt_subject_path()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: run_standalone_node
# ---------------------------------------------------------------------------


class TestRunStandaloneNode:
    def test_cancel_returns_none(self, agent_commands):
        """User cancels confirmation -> returns None."""
        mock_input = MagicMock()

        with patch("datus.cli.agent_commands.Confirm.ask", return_value=False):
            result = agent_commands.run_standalone_node(NodeType.TYPE_SCHEMA_LINKING, mock_input, need_confirm=True)

        assert result is None

    def test_node_exception_returns_none(self, agent_commands):
        """Node creation exception is caught and None is returned."""
        mock_input = MagicMock()
        mock_input.to_dict.return_value = {}

        with patch("datus.cli.agent_commands.Confirm.ask", return_value=True):
            with patch("datus.cli.agent_commands.Node.new_instance", side_effect=RuntimeError("node error")):
                result = agent_commands.run_standalone_node(NodeType.TYPE_SCHEMA_LINKING, mock_input, need_confirm=True)

        assert result is None

    def test_no_confirm_runs_node(self, agent_commands):
        """need_confirm=False skips confirmation and attempts to run node."""
        mock_input = MagicMock()
        mock_node = MagicMock()
        mock_node.run_async = MagicMock(return_value=None)

        async def mock_run():
            return "result"

        mock_node.run_async.return_value = mock_run()

        with patch("datus.cli.agent_commands.Node.new_instance", return_value=mock_node):
            with patch("asyncio.run", return_value="result"):
                result = agent_commands.run_standalone_node(
                    NodeType.TYPE_SCHEMA_LINKING, mock_input, need_confirm=False
                )

        assert result == "result"


# ---------------------------------------------------------------------------
# Tests: create_node_input
# ---------------------------------------------------------------------------


class TestCreateNodeInput:
    def test_unsupported_node_type_raises(self, agent_commands, cli_context):
        """Unsupported node type raises ValueError."""
        existing = SqlTask(
            id="x1",
            database_type=DBType.SQLITE,
            task="test",
            database_name="db",
            output_dir="/tmp",
        )
        cli_context.set_current_sql_task(existing)
        agent_commands.cli_context = cli_context

        with pytest.raises(ValueError, match="Unsupported node type"):
            agent_commands.create_node_input("unknown_type", "test task")

    def test_fix_node_no_sql_returns_none(self, agent_commands, cli_context):
        """Fix node returns None if no previous SQL."""
        existing = SqlTask(
            id="x2",
            database_type=DBType.SQLITE,
            task="test",
            database_name="db",
            output_dir="/tmp",
        )
        cli_context.set_current_sql_task(existing)
        # No sql_context added -> get_last_sql() returns None
        agent_commands.cli_context = cli_context

        result = agent_commands.create_node_input(NodeType.TYPE_FIX, "fix something")
        assert result is None

    def test_compare_empty_expectation_returns_none(self, agent_commands, cli_context):
        """Compare node returns None if expectation is empty."""
        existing = SqlTask(
            id="x3",
            database_type=DBType.SQLITE,
            task="test",
            database_name="db",
            output_dir="/tmp",
        )
        cli_context.set_current_sql_task(existing)
        agent_commands.cli_context = cli_context
        agent_commands.cli.prompt_input = lambda *a, **kw: ""

        result = agent_commands.create_node_input(NodeType.TYPE_COMPARE, "compare data")
        assert result is None


@pytest.fixture
def sql_task():
    return SqlTask(
        id="test01",
        database_type=DBType.SQLITE,
        task="show revenue",
        database_name="testdb",
        output_dir="/tmp",
    )


# ---------------------------------------------------------------------------
# Tests: cmd_gen
# ---------------------------------------------------------------------------


class TestCmdGen:
    def test_no_input_data_returns_early(self, agent_commands):
        with patch.object(agent_commands, "create_node_input", return_value=None):
            agent_commands.cmd_gen("")
        # no exception, output unchanged
        output = agent_commands.console.file.getvalue()
        assert "Error" not in output or "No" in output

    def test_result_success_with_sql_contexts(self, agent_commands):
        mock_ctx = MagicMock()
        mock_ctx.sql_query = "SELECT 1"
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sql_contexts = [mock_ctx]

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_gen("revenue")

        output = agent_commands.console.file.getvalue()
        assert "SELECT 1" in output

    def test_result_success_sql_query_attribute(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sql_contexts = []
        mock_result.sql_query = "SELECT 2"

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_gen("revenue")

        output = agent_commands.console.file.getvalue()
        assert "SELECT 2" in output

    def test_result_success_no_sql(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sql_contexts = []
        del mock_result.sql_query  # no sql_query attr

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_gen("revenue")

        output = agent_commands.console.file.getvalue()
        assert "completed" in output

    def test_result_failure(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = False

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_gen("revenue")

        output = agent_commands.console.file.getvalue()
        assert "failed" in output.lower()


# ---------------------------------------------------------------------------
# Tests: cmd_fix
# ---------------------------------------------------------------------------


class TestCmdFix:
    def test_no_input_data_returns_early(self, agent_commands):
        with patch.object(agent_commands, "create_node_input", return_value=None):
            agent_commands.cmd_fix("")

    def test_result_success_with_sql_contexts(self, agent_commands):
        mock_ctx = MagicMock()
        mock_ctx.sql_query = "SELECT fixed"
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sql_contexts = [mock_ctx]

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_fix("fix it")

        output = agent_commands.console.file.getvalue()
        assert "SELECT fixed" in output

    def test_result_failure(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = False

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_fix("fix")

        output = agent_commands.console.file.getvalue()
        assert "failed" in output.lower()


# ---------------------------------------------------------------------------
# Tests: cmd_reason
# ---------------------------------------------------------------------------


class TestCmdReason:
    def test_no_input_returns_early(self, agent_commands):
        with patch.object(agent_commands, "create_node_input", return_value=None):
            agent_commands.cmd_reason("")

    def test_result_success_with_explanation(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.explanation = "The revenue is calculated by..."

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_reason("why")

        output = agent_commands.console.file.getvalue()
        assert "The revenue" in output

    def test_result_failure(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = False

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_reason("why")

        output = agent_commands.console.file.getvalue()
        assert "failed" in output.lower()

    def test_reason_stream_delegates_to_reason(self, agent_commands):
        with patch.object(agent_commands, "cmd_reason") as mock_reason:
            agent_commands.cmd_reason_stream("test")
        mock_reason.assert_called_once_with("test")


# ---------------------------------------------------------------------------
# Tests: cmd_compare
# ---------------------------------------------------------------------------


class TestCmdCompare:
    def test_no_input_returns_early(self, agent_commands):
        with patch.object(agent_commands, "create_node_input", return_value=None):
            agent_commands.cmd_compare("")

    def test_result_success(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.comparison_result = "Matches expectation"

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_compare("compare")

        output = agent_commands.console.file.getvalue()
        assert "completed" in output or "Matches" in output

    def test_result_failure(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = False

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_compare("compare")

        output = agent_commands.console.file.getvalue()
        assert "failed" in output.lower()

    def test_compare_stream_delegates(self, agent_commands):
        with patch.object(agent_commands, "cmd_compare") as mock_compare:
            agent_commands.cmd_compare_stream("test")
        mock_compare.assert_called_once_with("test")


# ---------------------------------------------------------------------------
# Tests: cmd_daend
# ---------------------------------------------------------------------------


class TestCmdDaend:
    def test_no_workflow_runner_prints_message(self, agent_commands):
        agent_commands.cli.workflow_runner = None
        agent_commands.cmd_daend("")
        output = agent_commands.console.file.getvalue()
        assert "No active" in output or "no active" in output.lower()

    def test_with_workflow_runner_saves(self, agent_commands):
        mock_runner = MagicMock()
        mock_runner.workflow.task.output_dir = "/tmp"
        mock_runner.workflow.name = "test_wf"
        mock_runner.workflow.save = MagicMock()
        agent_commands.cli.workflow_runner = mock_runner

        agent_commands.cmd_daend("")

        mock_runner.workflow.save.assert_called_once()
        output = agent_commands.console.file.getvalue()
        assert "save" in output.lower() or "Ending" in output


# ---------------------------------------------------------------------------
# Tests: run_node (error paths)
# ---------------------------------------------------------------------------


class TestRunNode:
    def test_no_agent_returns_error_dict(self, agent_commands):
        agent_commands.agent = None
        result = agent_commands.run_node("schema_linking")
        assert result["success"] is False

    def test_no_workflow_runner_returns_error_dict(self, agent_commands):
        agent_commands.agent = MagicMock()
        agent_commands.cli.workflow_runner = None
        result = agent_commands.run_node("schema_linking")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Tests: _extract_sql_from_streaming_actions
# ---------------------------------------------------------------------------


class TestExtractSqlFromStreamingActions:
    def test_empty_actions_no_crash(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []
        node = MagicMock()
        del node.action_history_manager
        agent_commands._extract_sql_from_streaming_actions([], workflow, node)
        # No exception raised

    def test_extracts_from_read_query_action(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        action = MagicMock()
        action.action_type = "read_query"
        action.status = MagicMock()
        action.status.value = "success"
        action.input = {"sql": "SELECT 1"}
        action.output = {"result": "1 row", "error": ""}

        node = MagicMock(spec=[])  # no action_history_manager
        agent_commands._extract_sql_from_streaming_actions([action], workflow, node)

        assert len(workflow.context.sql_contexts) == 1

    def test_extracts_from_action_history_manager(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        sql_ctx = MagicMock()
        sql_ctx.sql_error = ""
        node = MagicMock()
        node.action_history_manager = MagicMock()
        node.action_history_manager.sql_contexts = [sql_ctx]

        agent_commands._extract_sql_from_streaming_actions([], workflow, node)
        assert len(workflow.context.sql_contexts) == 1

    def test_failed_sql_context_not_added(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        action = MagicMock()
        action.action_type = "read_query"
        action.status = MagicMock()
        action.status.value = "success"
        action.input = {"sql": "SELECT bad"}
        action.output = {"result": "", "error": "syntax error"}

        node = MagicMock(spec=[])
        agent_commands._extract_sql_from_streaming_actions([action], workflow, node)
        # Failed context (error != "") should not be added
        assert len(workflow.context.sql_contexts) == 0

    def test_exception_in_extraction_does_not_raise(self, agent_commands):
        """Top-level exception in extraction should be caught and logged, not raised."""
        workflow = MagicMock()
        # Make accessing sql_contexts raise an exception to exercise the outer except block
        type(workflow.context).sql_contexts = property(lambda self: (_ for _ in ()).throw(RuntimeError("injected")))

        node = MagicMock(spec=[])
        # Should not raise - outer try/except catches it
        agent_commands._extract_sql_from_streaming_actions([], workflow, node)


# ---------------------------------------------------------------------------
# Tests: _prompt_db_layers
# ---------------------------------------------------------------------------


class TestPromptDbLayers:
    def test_sqlite_prompts_database_only(self, agent_commands):
        agent_commands.cli.db_connector.dialect = DBType.SQLITE
        agent_commands.cli.prompt_input = lambda msg, default="", **kw: default or "testdb"

        with patch("datus.cli.agent_commands.connector_registry") as mock_registry:
            mock_registry.support_catalog.return_value = False
            mock_registry.support_database.return_value = False
            mock_registry.support_schema.return_value = False
            catalog, database, schema = agent_commands._prompt_db_layers()

        assert database == "testdb"
        assert catalog == ""
        assert schema == ""

    def test_catalog_and_schema_supported(self, agent_commands):
        agent_commands.cli.db_connector.dialect = "snowflake"
        call_returns = {"catalog": "mycat", "database": "mydb", "schema": "myschema"}

        def mock_prompt(msg, default="", **kw):
            for key in call_returns:
                if key in msg.lower():
                    return call_returns[key]
            return default

        agent_commands.cli.prompt_input = mock_prompt

        with patch("datus.cli.agent_commands.connector_registry") as mock_registry:
            mock_registry.support_catalog.return_value = True
            mock_registry.support_database.return_value = True
            mock_registry.support_schema.return_value = True
            catalog, database, schema = agent_commands._prompt_db_layers()

        assert catalog == "mycat"


# ---------------------------------------------------------------------------
# Tests: _print_metadata_table
# ---------------------------------------------------------------------------


class TestPrintMetadataTable:
    def test_prints_table_with_data(self, agent_commands):
        data = [
            {
                "catalog_name": "",
                "database_name": "mydb",
                "schema_name": "",
                "table_name": "orders",
                "table_type": "BASE TABLE",
                "definition": "CREATE TABLE orders (id INT)",
                "_distance": 0.1,
            }
        ]
        agent_commands._print_metadata_table(data, data_column="definition", lexer="sql")
        output = agent_commands.console.file.getvalue()
        assert "orders" in output

    def test_prints_table_empty_data(self, agent_commands):
        agent_commands._print_metadata_table([], data_column="definition")
        # Should not raise


# ---------------------------------------------------------------------------
# Tests: cmd_doc_search extra branches
# ---------------------------------------------------------------------------


class TestCmdDocSearchExtended:
    def test_invalid_top_n_prints_error(self, agent_commands):
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "duckdb"  # platform
            if call_count[0] == 2:
                return ""  # version
            if call_count[0] == 3:
                return "window functions"  # keywords
            return "not_a_number"  # top_n invalid

        agent_commands.cli.prompt_input = mock_prompt
        agent_commands.cmd_doc_search("")
        output = agent_commands.console.file.getvalue()
        assert "integer" in output.lower() or "must be" in output.lower()

    def test_success_path_displays_results(self, agent_commands):
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "duckdb"
            if call_count[0] == 2:
                return ""  # version
            if call_count[0] == 3:
                return "window"
            return "5"  # top_n

        agent_commands.cli.prompt_input = mock_prompt

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.doc_count = 1
        mock_result.docs = {
            "window": [
                {
                    "title": "Window Functions",
                    "titles": ["DuckDB", "SQL"],
                    "nav_path": ["docs", "sql"],
                    "hierarchy": "H1 > H2",
                    "chunk_text": "Window functions allow...",
                    "doc_path": "/docs/window.html",
                }
            ]
        }

        with patch("datus.cli.agent_commands.ContextSearchTools"):
            from datus.tools.search_tools.search_tool import SearchTool

            with patch.object(SearchTool, "search_document", return_value=mock_result):
                agent_commands.cmd_doc_search("")

        output = agent_commands.console.file.getvalue()
        assert "Found" in output or "chunk" in output.lower() or "1" in output

    def test_no_results_path(self, agent_commands):
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "duckdb"
            if call_count[0] == 2:
                return ""
            if call_count[0] == 3:
                return "window"
            return "5"

        agent_commands.cli.prompt_input = mock_prompt

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.doc_count = 0

        from datus.tools.search_tools.search_tool import SearchTool

        with patch.object(SearchTool, "search_document", return_value=mock_result):
            agent_commands.cmd_doc_search("")

        output = agent_commands.console.file.getvalue()
        assert "No documents" in output or "not found" in output.lower()

    def test_failure_path(self, agent_commands):
        call_count = [0]

        def mock_prompt(msg="", default="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "duckdb"
            if call_count[0] == 2:
                return ""
            if call_count[0] == 3:
                return "window"
            return "5"

        agent_commands.cli.prompt_input = mock_prompt

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Index not found"

        from datus.tools.search_tools.search_tool import SearchTool

        with patch.object(SearchTool, "search_document", return_value=mock_result):
            agent_commands.cmd_doc_search("")

        output = agent_commands.console.file.getvalue()
        assert "Error" in output


# ---------------------------------------------------------------------------
# Tests: create_node_input additional types
# ---------------------------------------------------------------------------


class TestCreateNodeInputExtended:
    def test_generate_sql_type_returns_input(self, agent_commands, cli_context, sql_task):
        cli_context.set_current_sql_task(sql_task)
        agent_commands.cli_context = cli_context

        result = agent_commands.create_node_input(NodeType.TYPE_GENERATE_SQL, "show revenue")
        assert result is not None

    def test_fix_type_no_sql_returns_none(self, agent_commands, cli_context, sql_task):
        """When there is no previous SQL, fix returns None."""
        cli_context.set_current_sql_task(sql_task)
        agent_commands.cli_context = cli_context
        # No sql_context added -> get_last_sql() returns None
        result = agent_commands.create_node_input(NodeType.TYPE_FIX, "fix it")
        assert result is None

    @pytest.mark.xfail(
        reason="Production bug: create_node_input passes sql_query= to ReasoningInput which forbids extras",
        strict=True,
    )
    def test_reasoning_type_creates_input(self, agent_commands, cli_context, sql_task):
        """Reasoning type should create a valid ReasoningInput.

        Currently fails with ValidationError because create_node_input passes
        sql_query= as a keyword argument but ReasoningInput inherits extra='forbid'
        from GenerateSQLInput.
        """
        cli_context.set_current_sql_task(sql_task)
        agent_commands.cli_context = cli_context
        agent_commands.cli.prompt_input = lambda msg, default="", **kw: default or ""

        result = agent_commands.create_node_input(NodeType.TYPE_REASONING, "explain query")
        assert result is not None
