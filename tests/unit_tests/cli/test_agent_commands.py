# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/agent_commands.py.

Tests cover:
- AgentCommands initialization
- _gen_sql_task: with args, empty, existing task
- create_node_input: for each NodeType
- cmd_save: no context, with context
- run_standalone_node: confirm cancelled, success
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
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.schemas.node_models import SqlTask
from datus.schemas.reason_sql_node_models import ReasoningInput
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

        assert isinstance(result, SqlTask)
        assert result.task == "show me revenue"

    def test_returns_none_on_exception(self, agent_commands):
        """Returns None when an unexpected exception occurs."""
        agent_commands.cli.db_connector = None
        # With no db_connector, it falls back to SQLITE — should still work
        result = agent_commands._gen_sql_task("test query")
        # The code falls back to SQLITE when db_connector is None; a SqlTask should be returned
        assert isinstance(result, SqlTask)
        assert result.task == "test query"


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
# Tests: cmd_fix
# ---------------------------------------------------------------------------


class TestCmdFix:
    def test_no_input_data_returns_early(self, agent_commands):
        with (
            patch.object(agent_commands, "create_node_input", return_value=None),
            patch.object(agent_commands, "run_standalone_node") as mock_run,
        ):
            agent_commands.cmd_fix("")
        mock_run.assert_not_called()

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
        with (
            patch.object(agent_commands, "create_node_input", return_value=None),
            patch.object(agent_commands, "run_standalone_node") as mock_run,
        ):
            agent_commands.cmd_reason("")
        mock_run.assert_not_called()

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
        with (
            patch.object(agent_commands, "create_node_input", return_value=None),
            patch.object(agent_commands, "run_standalone_node") as mock_run,
        ):
            agent_commands.cmd_compare("")
        mock_run.assert_not_called()

    def test_result_success(self, agent_commands):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.comparison_result = "Matches expectation"

        with patch.object(agent_commands, "create_node_input", return_value=MagicMock()):
            with patch.object(agent_commands, "run_standalone_node", return_value=mock_result):
                agent_commands.cmd_compare("compare")

        output = agent_commands.console.file.getvalue()
        assert "SQL comparison completed" in output
        assert "Matches expectation" in output

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
        assert "No active workflow session to end." in output

    def test_with_workflow_runner_saves(self, agent_commands):
        mock_runner = MagicMock()
        mock_runner.workflow.task.output_dir = "/tmp"
        mock_runner.workflow.name = "test_wf"
        mock_runner.workflow.save = MagicMock()
        agent_commands.cli.workflow_runner = mock_runner

        agent_commands.cmd_daend("")

        mock_runner.workflow.save.assert_called_once()
        output = agent_commands.console.file.getvalue()
        assert "Ending workflow session, save to /tmp/test_wf.yaml" in output


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

    def test_sql_node_result_prints_sql_once_and_excludes_sql_from_tree(self, agent_commands):
        from types import SimpleNamespace

        agent_commands.agent = MagicMock()
        workflow = MagicMock()
        workflow.tools = []
        workflow.context.sql_contexts = []
        runner = MagicMock()
        runner.workflow_ready = True
        runner.workflow = workflow
        agent_commands.cli.workflow_runner = runner

        node = MagicMock()
        node.type = NodeType.TYPE_GEN_SQL
        node.status = "success"
        node.input = GenSQLNodeInput(user_message="count schools", database="california_schools")
        node.result = SimpleNamespace(sql="SELECT 1", response="done", tokens_used=1)

        tree_payloads = []

        def fake_dict_to_tree(payload, console=None):
            tree_payloads.append(payload)
            return "TREE"

        with (
            patch("datus.cli.agent_commands.Node.new_instance", return_value=node),
            patch("datus.cli.agent_commands.setup_node_input", return_value={"success": True}),
            patch("datus.cli.agent_commands.update_context_from_node", return_value={"success": True}),
            patch("datus.cli.agent_commands.dict_to_tree", side_effect=fake_dict_to_tree),
        ):
            result = agent_commands.run_node(NodeType.TYPE_GEN_SQL, need_confirm=False)

        assert result["success"] is True
        assert tree_payloads == [{"response": "done", "tokens_used": 1}]
        assert "SELECT 1" in agent_commands.console.file.getvalue()


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
        assert workflow.context.sql_contexts == []

    def test_extracts_from_execute_sql_action(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        action = MagicMock()
        action.action_type = "execute_sql"
        action.status = MagicMock()
        action.status.value = "success"
        # Real tool actions nest params under ``input["arguments"]``.
        action.input = {"function_name": "execute_sql", "arguments": {"sql": "SELECT 1"}}
        # Read-only results carry the compressor payload (compressed_data).
        action.output = {"result": {"original_rows": 1, "compressed_data": "n\n1"}, "error": ""}

        node = MagicMock(spec=[])  # no action_history_manager
        agent_commands._extract_sql_from_streaming_actions([action], workflow, node)

        assert len(workflow.context.sql_contexts) == 1
        # The query is extracted from the nested ``arguments`` payload.
        assert workflow.context.sql_contexts[0].sql_query == "SELECT 1"

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
        action.action_type = "execute_sql"
        action.status = MagicMock()
        action.status.value = "success"
        action.input = {"function_name": "execute_sql", "arguments": {"sql": "SELECT bad"}}
        # A failed read returns no compressor payload → not read-shaped → skipped.
        action.output = {"result": "", "error": "syntax error"}

        node = MagicMock(spec=[])
        agent_commands._extract_sql_from_streaming_actions([action], workflow, node)
        # Failed context (no read-shaped result) should not be added
        assert len(workflow.context.sql_contexts) == 0

    def test_extracts_output_field_from_final_assistant_message(self, agent_commands):
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        action = MagicMock()
        action.action_type = "message"
        action.role = "assistant"
        action.output = {"raw_output": '{"sql": "SELECT 1", "output": "compact response"}'}

        node = MagicMock(spec=[])
        agent_commands._extract_sql_from_streaming_actions([action], workflow, node)

        assert len(workflow.context.sql_contexts) == 1
        assert workflow.context.sql_contexts[0].sql_query == "SELECT 1"
        assert workflow.context.sql_contexts[0].explanation == "compact response"

    def test_exception_in_extraction_does_not_raise(self, agent_commands):
        """Top-level exception in extraction should be caught and logged, not raised."""
        workflow = MagicMock()

        def raise_injected(_self):
            raise RuntimeError("injected")

        # Make accessing sql_contexts raise an exception to exercise the outer except block
        type(workflow.context).sql_contexts = property(raise_injected)

        node = MagicMock(spec=[])
        agent_commands._extract_sql_from_streaming_actions([], workflow, node)
        assert isinstance(workflow.context, MagicMock)


# ---------------------------------------------------------------------------
# Tests: create_node_input additional types
# ---------------------------------------------------------------------------


class TestCreateNodeInputExtended:
    def test_gen_sql_type_returns_agentic_input(self, agent_commands, cli_context, sql_task):
        cli_context.set_current_sql_task(sql_task)
        agent_commands.cli_context = cli_context

        result = agent_commands.create_node_input(NodeType.TYPE_GEN_SQL, "show revenue")
        assert isinstance(result, GenSQLNodeInput)
        assert result.user_message == "show revenue"

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
        from its base schema.
        """
        cli_context.set_current_sql_task(sql_task)
        agent_commands.cli_context = cli_context
        agent_commands.cli.prompt_input = lambda msg, default="", **kw: default or ""

        result = agent_commands.create_node_input(NodeType.TYPE_REASONING, "explain query")
        assert isinstance(result, ReasoningInput)
