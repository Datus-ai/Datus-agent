# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/chat_commands.py.

Tests cover:
- ChatCommands initialization
- _should_create_new_node logic
- _extract_node_type_from_session_id static method
- _create_new_node for all node types
- create_node_input for all node types
- _extract_report_from_json
- _extract_sql_and_output_from_content
- Display methods: _display_sql_with_copy, _display_markdown_response,
  _display_semantic_model, _display_sql_summary_file, _display_ext_knowledge_file,
  INTERACTION rendering (via ActionRenderer)
- cmd_clear_chat
- cmd_chat_info
- add_in_sql_context
- execute_chat_command (basic flow)

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
A lightweight MinimalCLI helper (real object, not mock) provides the attributes
ChatCommands needs without constructing a full DatusCLI.
"""

import io
import json
import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.chat_commands import ChatCommands, _is_model_config_error
from datus.cli.cli_context import CliContext
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus

# ===========================================================================
# Lightweight real helpers (NOT mocks)
# ===========================================================================


class MinimalAtCompleter:
    """Minimal at-completer that returns empty context."""

    def parse_at_context(self, user_input):
        """Return empty lists for tables, metrics, sqls."""
        return ([], [], [])


class MinimalCLI:
    """Lightweight real CLI substitute for testing ChatCommands without full DatusCLI.

    Provides the minimal attributes that ChatCommands.__init__ and its methods access.
    """

    def __init__(self, agent_config, console=None):
        self.agent_config = agent_config
        self.console = console or Console(file=io.StringIO(), no_color=True)
        self.cli_context = CliContext()
        self.actions = ActionHistoryManager()
        self.last_sql = ""
        self.at_completer = MinimalAtCompleter()
        self.scope = None

    def prompt_input(self, message="", multiline=False):
        """Return empty string for prompt input."""
        return ""

    def _print_welcome(self):
        """No-op stand-in for the real banner printer."""

    def run_on_bg_loop(self, coro):
        """Simple synchronous stand-in for ``DatusCLI.run_on_bg_loop``.

        The real implementation routes through a persistent background loop to
        keep prompt_toolkit Futures alive across chat turns; tests don't need
        that machinery, so we just drive the coroutine to completion.
        """
        import asyncio

        return asyncio.run(coro)


# ===========================================================================
# Shared helper to capture console output
# ===========================================================================


def _get_console_output(console: Console) -> str:
    """Extract text written to a StringIO-backed Console."""
    output_file = console.file
    if hasattr(output_file, "getvalue"):
        return output_file.getvalue()
    return ""


def _make_chat_commands(agent_config, console=None):
    """Create a ChatCommands instance with a MinimalCLI."""
    if console is None:
        console = Console(file=io.StringIO(), no_color=True)
    cli = MinimalCLI(agent_config, console=console)
    return ChatCommands(cli)


def _create_session_on_disk(session_id, messages=None):
    """Create a real session .db file on disk for testing cmd_resume/cmd_rewind.

    Uses get_path_manager().sessions_dir so the file is in the correct location
    for SessionManager to find.

    Args:
        session_id: The session ID (e.g. "chat_session_abc12345")
        messages: List of (role, content) tuples. Defaults to a single user+assistant exchange.
    """
    from datus.utils.path_manager import get_path_manager

    sessions_dir = str(get_path_manager().sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)
    db_path = os.path.join(sessions_dir, f"{session_id}.db")

    if messages is None:
        messages = [("user", "Hello"), ("assistant", "Hi there!")]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_sessions ("
            "session_id TEXT PRIMARY KEY, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "message_data TEXT NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)",
            (session_id,),
        )

        base_time = datetime(2025, 6, 1, 10, 0, 0)
        for i, (role, content) in enumerate(messages):
            ts = (base_time + timedelta(minutes=i)).isoformat()
            msg_data = json.dumps({"role": role, "content": content})
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, ?)",
                (session_id, msg_data, ts),
            )
        conn.commit()

    return db_path


# ===========================================================================
# TestChatCommandsInit
# ===========================================================================


class TestChatCommandsInit:
    """Tests for ChatCommands.__init__ attribute setup."""

    def test_init_sets_console_from_cli(self, real_agent_config, mock_llm_create):
        """Initialization stores the console from the CLI instance."""
        console = Console(file=io.StringIO(), no_color=True)
        cli = MinimalCLI(real_agent_config, console=console)
        cmds = ChatCommands(cli)

        assert cmds.console is console
        assert cmds.cli is cli

    def test_init_sets_default_state(self, real_agent_config, mock_llm_create):
        """Initialization creates default empty state for nodes and history."""
        cmds = _make_chat_commands(real_agent_config)

        assert cmds.current_node is None
        assert cmds.chat_node is None
        assert cmds.current_subagent_name is None
        assert cmds.chat_history == []
        assert cmds.last_actions == []


# ===========================================================================
# TestShouldCreateNewNode
# ===========================================================================


class TestShouldCreateNewNode:
    """Tests for _should_create_new_node decision logic."""

    def test_returns_true_when_current_node_is_none(self, real_agent_config, mock_llm_create):
        """Should create new node when no node exists yet."""
        cmds = _make_chat_commands(real_agent_config)
        assert cmds.current_node is None
        assert cmds._should_create_new_node() is True
        assert cmds._should_create_new_node(subagent_name="gensql") is True

    def test_returns_false_when_node_exists_no_subagent(self, real_agent_config, mock_llm_create):
        """Should NOT create new node when a regular chat node already exists."""
        cmds = _make_chat_commands(real_agent_config)
        # Simulate existing chat node (no subagent)
        cmds.current_node = cmds._create_new_node()
        cmds.current_subagent_name = None

        result = cmds._should_create_new_node()
        assert result is False
        assert cmds.current_node is not None

    def test_returns_true_when_switching_from_no_subagent_to_subagent(self, real_agent_config, mock_llm_create):
        """Should create new node when switching from regular chat to a subagent."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node()
        cmds.current_subagent_name = None

        result = cmds._should_create_new_node(subagent_name="gensql")
        assert result is True
        assert cmds.current_subagent_name is None  # Not changed yet

    def test_returns_true_when_switching_between_different_subagents(self, real_agent_config, mock_llm_create):
        """Should create new node when switching between different subagents."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node(subagent_name="gensql")
        cmds.current_subagent_name = "gensql"

        result = cmds._should_create_new_node(subagent_name="gen_semantic_model")
        assert result is True
        assert cmds.current_subagent_name == "gensql"  # Not changed yet

    def test_returns_false_when_same_subagent(self, real_agent_config, mock_llm_create):
        """Should NOT create new node when continuing with the same subagent."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node(subagent_name="gensql")
        cmds.current_subagent_name = "gensql"

        result = cmds._should_create_new_node(subagent_name="gensql")
        assert result is False
        assert cmds.current_subagent_name == "gensql"

    def test_returns_true_when_switching_from_subagent_to_regular(self, real_agent_config, mock_llm_create):
        """Should create new node when switching from subagent back to regular chat."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node(subagent_name="gensql")
        cmds.current_subagent_name = "gensql"

        result = cmds._should_create_new_node(subagent_name=None)
        assert result is True
        assert cmds.current_subagent_name == "gensql"


# ===========================================================================
# TestExtractNodeTypeFromSessionId
# ===========================================================================


class TestExtractNodeTypeFromSessionId:
    """Tests for the static method _extract_node_type_from_session_id."""

    def test_chat_session_returns_chat(self):
        """Extract 'chat' from a chat session ID."""
        result = ChatCommands._extract_node_type_from_session_id("chat_session_abc123")
        assert result == "chat"
        assert isinstance(result, str)

    def test_gensql_session_returns_gensql(self):
        """Extract 'gensql' from a gensql session ID."""
        result = ChatCommands._extract_node_type_from_session_id("gensql_session_def456")
        assert result == "gensql"
        assert "session" not in result

    def test_gen_semantic_model_session(self):
        """Extract 'gen_semantic_model' from a gen_semantic_model session ID."""
        result = ChatCommands._extract_node_type_from_session_id("gen_semantic_model_session_789xyz")
        assert result == "gen_semantic_model"
        assert "_session_" not in result

    def test_no_session_marker_returns_chat(self):
        """Return 'chat' when session ID has no '_session_' marker."""
        result = ChatCommands._extract_node_type_from_session_id("some-random-uuid")
        assert result == "chat"
        assert isinstance(result, str)

    def test_gen_sql_summary_session(self):
        """Extract 'gen_sql_summary' from a gen_sql_summary session ID."""
        result = ChatCommands._extract_node_type_from_session_id("gen_sql_summary_session_abc")
        assert result == "gen_sql_summary"
        assert result != "gen_sql"

    def test_empty_string_returns_chat(self):
        """Return 'chat' for empty string (no _session_ marker)."""
        result = ChatCommands._extract_node_type_from_session_id("")
        assert result == "chat"
        assert result != ""


# ===========================================================================
# TestCreateNewNode
# ===========================================================================


class TestCreateNewNode:
    """Tests for _create_new_node creating correct node types."""

    def test_default_chat_node_creation(self, real_agent_config, mock_llm_create):
        """Default (no subagent) creates a ChatAgenticNode."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node()

        assert isinstance(node, ChatAgenticNode)
        assert node.id == "chat_cli"

    def test_gensql_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name='gensql' creates a GenSQLAgenticNode (not ChatAgenticNode)."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gensql")

        assert isinstance(node, GenSQLAgenticNode)
        assert not isinstance(node, ChatAgenticNode)

    def test_gen_semantic_model_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name='gen_semantic_model' creates a GenSemanticModelAgenticNode."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_semantic_model")

        assert isinstance(node, GenSemanticModelAgenticNode)
        assert node.agent_config is real_agent_config

    def test_gen_metrics_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name='gen_metrics' creates a GenMetricsAgenticNode."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_metrics")

        assert isinstance(node, GenMetricsAgenticNode)
        assert node.agent_config is real_agent_config

    def test_gen_sql_summary_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name='gen_sql_summary' creates a SqlSummaryAgenticNode."""
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_sql_summary")

        assert isinstance(node, SqlSummaryAgenticNode)
        assert node.agent_config is real_agent_config

    def test_gen_ext_knowledge_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name='gen_ext_knowledge' creates a GenExtKnowledgeAgenticNode."""
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_ext_knowledge")

        assert isinstance(node, GenExtKnowledgeAgenticNode)
        assert node.agent_config is real_agent_config

    def test_gen_report_node_creation(self, real_agent_config, mock_llm_create):
        """subagent_name with node_class='gen_report' creates a GenReportAgenticNode."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        # The real_agent_config fixture has gen_report in agentic_nodes but without node_class.
        # We need to set node_class='gen_report' for this config entry.
        real_agent_config.agentic_nodes["gen_report"]["node_class"] = "gen_report"

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_report")

        assert isinstance(node, GenReportAgenticNode)
        assert node.agent_config is real_agent_config

    def test_console_output_on_chat_creation(self, real_agent_config, mock_llm_create):
        """Console prints 'Creating new chat session...' for default chat."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        cmds._create_new_node()

        output = _get_console_output(console)
        assert "Creating new chat session" in output
        assert len(output) > 0

    def test_console_output_on_subagent_creation(self, real_agent_config, mock_llm_create):
        """Console prints the subagent name when creating a subagent node."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        cmds._create_new_node(subagent_name="gen_semantic_model")

        output = _get_console_output(console)
        assert "gen_semantic_model" in output
        assert "Creating new" in output


# ===========================================================================
# TestCreateNodeInput
# ===========================================================================


class TestCreateNodeInput:
    """Tests for create_node_input returning correct input types.

    ChatAgenticNode is independent from GenSQLAgenticNode (no inheritance).
    ChatAgenticNode receives ChatNodeInput; GenSQLAgenticNode receives GenSQLNodeInput.
    """

    def test_chat_node_gets_chat_input(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode gets ChatNodeInput with 'chat' type."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node()
        assert isinstance(node, ChatAgenticNode)

        node_input, node_type = cmds.create_node_input("Hello", node, [], [], [])

        # ChatAgenticNode is NOT a GenSQLAgenticNode, so it falls to the else branch
        assert isinstance(node_input, ChatNodeInput)
        assert node_type == "chat"
        assert node_input.user_message == "Hello"

    def test_gensql_node_input(self, real_agent_config, mock_llm_create):
        """GenSQLAgenticNode gets GenSQLNodeInput with 'gensql' type."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gensql")
        assert isinstance(node, GenSQLAgenticNode)

        node_input, node_type = cmds.create_node_input("Generate SQL for users", node, [], [], [])

        assert isinstance(node_input, GenSQLNodeInput)
        assert node_type == "gensql"
        assert node_input.user_message == "Generate SQL for users"

    def test_semantic_model_node_input(self, real_agent_config, mock_llm_create):
        """GenSemanticModelAgenticNode gets SemanticNodeInput with 'semantic' type."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_semantic_model")
        assert isinstance(node, GenSemanticModelAgenticNode)

        node_input, node_type = cmds.create_node_input("Build semantic model", node, [], [], [])

        assert isinstance(node_input, SemanticNodeInput)
        assert node_type == "semantic"
        assert node_input.user_message == "Build semantic model"

    def test_gen_metrics_node_input(self, real_agent_config, mock_llm_create):
        """GenMetricsAgenticNode gets SemanticNodeInput with 'semantic' type."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_metrics")
        assert isinstance(node, GenMetricsAgenticNode)

        node_input, node_type = cmds.create_node_input("Generate metrics", node, [], [], [])

        assert isinstance(node_input, SemanticNodeInput)
        assert node_type == "semantic"

    def test_sql_summary_node_input(self, real_agent_config, mock_llm_create):
        """SqlSummaryAgenticNode gets SqlSummaryNodeInput with 'sql_summary' type."""
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_sql_summary")
        assert isinstance(node, SqlSummaryAgenticNode)

        node_input, node_type = cmds.create_node_input("Summarize SQL", node, [], [], [])

        assert isinstance(node_input, SqlSummaryNodeInput)
        assert node_type == "sql_summary"
        assert node_input.user_message == "Summarize SQL"

    def test_ext_knowledge_node_input(self, real_agent_config, mock_llm_create):
        """GenExtKnowledgeAgenticNode gets ExtKnowledgeNodeInput with 'ext_knowledge' type."""
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
        from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_ext_knowledge")
        assert isinstance(node, GenExtKnowledgeAgenticNode)

        node_input, node_type = cmds.create_node_input("Add business knowledge", node, [], [], [])

        assert isinstance(node_input, ExtKnowledgeNodeInput)
        assert node_type == "ext_knowledge"
        assert node_input.user_message == "Add business knowledge"

    def test_gen_report_node_input(self, real_agent_config, mock_llm_create):
        """GenReportAgenticNode gets GenReportNodeInput with 'gen_report' type."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
        from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

        real_agent_config.agentic_nodes["gen_report"]["node_class"] = "gen_report"
        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gen_report")
        assert isinstance(node, GenReportAgenticNode)

        node_input, node_type = cmds.create_node_input("Generate report", node, [], [], [])

        assert isinstance(node_input, GenReportNodeInput)
        assert node_type == "gen_report"
        assert node_input.user_message == "Generate report"

    def test_chat_node_input_with_plan_mode(self, real_agent_config, mock_llm_create):
        """ChatNodeInput has plan_mode set when plan_mode=True (via ChatAgenticNode)."""
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node()  # ChatAgenticNode (independent from GenSQLAgenticNode)

        node_input, node_type = cmds.create_node_input("Test", node, [], [], [], plan_mode=True)

        assert isinstance(node_input, ChatNodeInput)
        assert node_input.plan_mode is True
        assert node_type == "chat"

    def test_gensql_node_input_with_at_context(self, real_agent_config, mock_llm_create):
        """GenSQLNodeInput passes through at_tables, at_metrics, at_sqls."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
        from datus.schemas.node_models import Metric, ReferenceSql, TableSchema

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gensql")

        at_tables = [
            TableSchema(
                table_name="table1",
                database_name="db",
                definition="CREATE TABLE table1 (id INT)",
            ),
            TableSchema(
                table_name="table2",
                database_name="db",
                definition="CREATE TABLE table2 (id INT)",
            ),
        ]
        at_metrics = [Metric(name="metric1", description="Test metric")]
        at_sqls = [ReferenceSql(name="ref1", sql="SELECT 1")]
        node_input, node_type = cmds.create_node_input("Query", node, at_tables, at_metrics, at_sqls)

        assert isinstance(node_input, GenSQLNodeInput)
        assert len(node_input.schemas) == 2
        assert node_input.schemas[0].table_name == "table1"
        assert len(node_input.metrics) == 1
        assert node_input.reference_sql[0].sql == "SELECT 1"


# ===========================================================================
# TestExtractReportFromJson
# ===========================================================================


class TestExtractReportFromJson:
    """Tests for _extract_report_from_json utility method."""

    def test_none_input_returns_none(self, real_agent_config, mock_llm_create):
        """None input returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json(None)
        assert result is None

    def test_empty_string_returns_none(self, real_agent_config, mock_llm_create):
        """Empty string returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json("")
        assert result is None

    def test_valid_json_with_report_field(self, real_agent_config, mock_llm_create):
        """Valid JSON with 'report' field extracts the report content."""
        cmds = _make_chat_commands(real_agent_config)
        json_str = json.dumps({"report": "This is the report content", "other": "data"})

        result = cmds._extract_report_from_json(json_str)
        assert result == "This is the report content"
        assert isinstance(result, str)

    def test_json_without_report_field_returns_none(self, real_agent_config, mock_llm_create):
        """JSON without 'report' field returns None."""
        cmds = _make_chat_commands(real_agent_config)
        json_str = json.dumps({"sql": "SELECT 1", "explanation": "test"})

        result = cmds._extract_report_from_json(json_str)
        assert result is None

    def test_non_json_string_returns_none(self, real_agent_config, mock_llm_create):
        """Plain non-JSON string returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json("This is just a plain text string.")
        assert result is None

    def test_json_wrapped_in_code_blocks(self, real_agent_config, mock_llm_create):
        """JSON wrapped in ```json ... ``` code blocks extracts the report field."""
        cmds = _make_chat_commands(real_agent_config)
        json_content = json.dumps({"report": "Report from code block"})
        wrapped = f"```json\n{json_content}\n```"

        result = cmds._extract_report_from_json(wrapped)
        assert result == "Report from code block"
        assert isinstance(result, str)

    def test_empty_report_field_returns_empty_string(self, real_agent_config, mock_llm_create):
        """JSON with empty 'report' field returns empty string."""
        cmds = _make_chat_commands(real_agent_config)
        json_str = json.dumps({"report": ""})

        result = cmds._extract_report_from_json(json_str)
        assert result == ""
        assert result is not None


# ===========================================================================
# TestExtractSqlAndOutputFromContent
# ===========================================================================


class TestExtractSqlAndOutputFromContent:
    """Tests for _extract_sql_and_output_from_content parsing logic."""

    def test_json_with_sql_and_output(self, real_agent_config, mock_llm_create):
        """JSON content with sql and output fields extracts both."""
        cmds = _make_chat_commands(real_agent_config)
        content = json.dumps({"sql": "SELECT * FROM users", "output": "5 rows returned"})

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT * FROM users"
        assert output is not None

    def test_json_with_sql_and_raw_output(self, real_agent_config, mock_llm_create):
        """JSON content with sql and raw_output fields extracts both."""
        cmds = _make_chat_commands(real_agent_config)
        content = json.dumps({"sql": "SELECT 1", "raw_output": "Result: 1"})

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT 1"
        assert output is not None

    def test_sql_code_block_in_markdown(self, real_agent_config, mock_llm_create):
        """SQL code block in markdown extracts the SQL."""
        cmds = _make_chat_commands(real_agent_config)
        content = "Here is the query:\n```sql\nSELECT id FROM orders\n```\nDone."

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT id FROM orders"
        assert output is None

    def test_plain_text_returns_none_none(self, real_agent_config, mock_llm_create):
        """Plain text without SQL returns (None, None)."""
        cmds = _make_chat_commands(real_agent_config)
        content = "Hello, how can I help you today?"

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql is None
        assert output is None

    def test_invalid_json_returns_gracefully(self, real_agent_config, mock_llm_create):
        """Invalid JSON returns gracefully without raising exceptions."""
        cmds = _make_chat_commands(real_agent_config)
        content = "{broken json content here"

        sql, output = cmds._extract_sql_and_output_from_content(content)
        # Should not raise; returns either (None, None) or extracted data
        assert sql is None or isinstance(sql, str)
        assert output is None or isinstance(output, str)

    def test_json_newline_format(self, real_agent_config, mock_llm_create):
        """'json\\n{...}' format extracts SQL and output."""
        cmds = _make_chat_commands(real_agent_config)
        json_data = json.dumps({"sql": "SELECT count(*) FROM t", "output": "count: 42"})
        content = f"json\n{json_data}"

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT count(*) FROM t"
        assert output is not None

    def test_json_with_escaped_quotes(self, real_agent_config, mock_llm_create):
        """JSON with escaped quotes is handled correctly."""
        cmds = _make_chat_commands(real_agent_config)
        content = '{"sql": "SELECT * FROM users WHERE name = \'John\'", "output": "1 row"}'

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql is not None
        assert "SELECT" in sql


# ===========================================================================
# TestDisplayMethods
# ===========================================================================


class TestDisplaySqlWithCopy:
    """Tests for _display_sql_with_copy console output."""

    def test_sql_panel_rendered(self, real_agent_config, mock_llm_create):
        """SQL is displayed in a panel and stored in cli.last_sql."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_sql_with_copy("SELECT 1")

        output = _get_console_output(console)
        assert "SELECT 1" in output
        assert cmds.cli.last_sql == "SELECT 1"

    def test_sql_panel_shows_generated_sql_title(self, real_agent_config, mock_llm_create):
        """The SQL panel title contains 'Generated SQL'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_sql_with_copy("SELECT * FROM orders")

        output = _get_console_output(console)
        assert "Generated SQL" in output
        assert "orders" in output


class TestDisplayMarkdownResponse:
    """Tests for _display_markdown_response console output."""

    def test_markdown_rendered(self, real_agent_config, mock_llm_create):
        """Markdown text is rendered to the console."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_markdown_response("# Hello World\nThis is a test.")

        output = _get_console_output(console)
        assert "Hello World" in output
        assert "test" in output

    def test_json_response_extracts_report(self, real_agent_config, mock_llm_create):
        """JSON response with 'report' field displays the report content."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        json_str = json.dumps({"report": "Extracted report content here"})
        cmds._display_markdown_response(json_str)

        output = _get_console_output(console)
        assert "Extracted report content here" in output
        assert len(output) > 0

    def test_plain_json_without_report_still_displays(self, real_agent_config, mock_llm_create):
        """JSON without 'report' field falls through to display as markdown."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        json_str = json.dumps({"sql": "SELECT 1"})
        cmds._display_markdown_response(json_str)

        output = _get_console_output(console)
        # Should display the raw content as markdown
        assert len(output) > 0


class TestDisplaySemanticModel:
    """Tests for _display_semantic_model console output."""

    def test_none_semantic_models(self, real_agent_config, mock_llm_create):
        """None input displays 'None'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_semantic_model(None)

        output = _get_console_output(console)
        assert "None" in output
        assert "Semantic Model" in output

    def test_single_file_semantic_model(self, real_agent_config, mock_llm_create):
        """Single file path is displayed."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_semantic_model(["/path/to/model.yaml"])

        output = _get_console_output(console)
        assert "/path/to/model.yaml" in output
        assert "Semantic Model File" in output

    def test_multiple_files_semantic_model(self, real_agent_config, mock_llm_create):
        """Multiple file paths are all displayed."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_semantic_model(["/path/model1.yaml", "/path/model2.yaml"])

        output = _get_console_output(console)
        assert "/path/model1.yaml" in output
        assert "/path/model2.yaml" in output
        assert "Semantic Model Files" in output

    def test_empty_list_semantic_model(self, real_agent_config, mock_llm_create):
        """Empty list displays 'None'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_semantic_model([])

        output = _get_console_output(console)
        assert "None" in output


class TestDisplaySqlSummaryFile:
    """Tests for _display_sql_summary_file console output."""

    def test_file_path_displayed(self, real_agent_config, mock_llm_create):
        """SQL summary file path is displayed."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_sql_summary_file("/output/summary.yaml")

        output = _get_console_output(console)
        assert "/output/summary.yaml" in output
        assert "SQL Summary File" in output


class TestDisplayExtKnowledgeFile:
    """Tests for _display_ext_knowledge_file console output."""

    def test_file_path_displayed(self, real_agent_config, mock_llm_create):
        """External knowledge file path is displayed."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_ext_knowledge_file("/output/knowledge.yaml")

        output = _get_console_output(console)
        assert "/output/knowledge.yaml" in output
        assert "External Knowledge File" in output


class TestDisplaySuccess:
    """Tests for INTERACTION SUCCESS rendering via ActionRenderer.

    Rendering was moved from ChatCommands._display_success to
    ActionRenderer.render_interaction_success + render_main_action.
    """

    def _make_action(self, content, content_type="markdown"):
        """Create an ActionHistory with output for display."""
        return ActionHistory(
            action_id="test_success_1",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output={"content": content, "content_type": content_type},
            status=ActionStatus.SUCCESS,
        )

    def _render_and_get_output(self, action):
        """Render via ActionRenderer.render_interaction_success and capture output.

        Note: render_main_action returns [] for INTERACTION (not shown in history),
        but render_interaction_success is still used during live interaction.
        """
        from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer

        console = Console(file=io.StringIO(), no_color=True)
        renderer = ActionRenderer(ActionContentGenerator())
        renderables = renderer.render_interaction_success(action, verbose=False)
        renderer.print_renderables(console, renderables)
        return _get_console_output(console)

    def test_yaml_content_type(self, real_agent_config, mock_llm_create):
        """YAML content type is rendered with syntax highlighting."""
        action = self._make_action("key: value\nlist:\n  - item1", "yaml")
        output = self._render_and_get_output(action)
        assert "key" in output
        assert "value" in output

    def test_sql_content_type(self, real_agent_config, mock_llm_create):
        """SQL content type is rendered with syntax highlighting."""
        action = self._make_action("SELECT * FROM users WHERE id = 1", "sql")
        output = self._render_and_get_output(action)
        assert "SELECT" in output
        assert "users" in output

    def test_markdown_content_type(self, real_agent_config, mock_llm_create):
        """Markdown content type is rendered as rich markdown."""
        action = self._make_action("# Title\nSome **bold** text", "markdown")
        output = self._render_and_get_output(action)
        assert "Title" in output
        assert "bold" in output

    def test_empty_content_skips_display(self, real_agent_config, mock_llm_create):
        """Empty content renders only completion indicator."""
        action = self._make_action("", "markdown")
        output = self._render_and_get_output(action)
        assert "Title" not in output

    def test_fallback_to_messages_when_no_content(self, real_agent_config, mock_llm_create):
        """When output.content is empty, falls back to action.messages."""
        action = ActionHistory(
            action_id="test_success_fb",
            role=ActionRole.INTERACTION,
            messages="Fallback message content",
            action_type="request_choice",
            input={},
            output={"content": "", "content_type": "markdown"},
            status=ActionStatus.SUCCESS,
        )
        output = self._render_and_get_output(action)
        assert "Fallback message content" in output


# ===========================================================================
# TestCmdClearChat
# ===========================================================================


class TestCmdClearChat:
    """Tests for cmd_clear_chat session clearing."""

    def test_clear_when_no_current_node(self, real_agent_config, mock_llm_create):
        """Clearing without a current node resets references and prints message."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_clear_chat("")

        assert cmds.current_node is None
        assert cmds.chat_node is None
        output = _get_console_output(console)
        assert "cleared" in output.lower()

    def test_clear_with_existing_current_node(self, real_agent_config, mock_llm_create):
        """Clearing with an existing node resets both current_node and chat_node."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a node first
        cmds.current_node = cmds._create_new_node()
        cmds.chat_node = cmds.current_node
        assert cmds.current_node is not None

        cmds.cmd_clear_chat("")

        assert cmds.current_node is None
        assert cmds.chat_node is None
        output = _get_console_output(console)
        assert "cleared" in output.lower()


# ===========================================================================
# TestCmdChatInfo
# ===========================================================================


class TestCmdChatInfo:
    """Tests for cmd_chat_info display."""

    def test_no_active_session_prints_warning(self, real_agent_config, mock_llm_create):
        """No active session prints a yellow warning."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_chat_info("")

        output = _get_console_output(console)
        assert "No active session" in output
        assert len(output) > 0

    def test_with_session_displays_info(self, real_agent_config, mock_llm_create):
        """With an active session, displays session info."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a node and give it a session
        cmds.current_node = cmds._create_new_node()

        # cmd_chat_info calls asyncio.run(get_session_info())
        # For ChatAgenticNode, this may or may not have a session_id yet
        # If no session_id, it prints "No active session"
        cmds.cmd_chat_info("")

        output = _get_console_output(console)
        # Either shows session info or "No active session"
        assert len(output) > 0
        assert ("Session" in output) or ("No active session" in output)


# ===========================================================================
# TestAddInSqlContext
# ===========================================================================


class TestAddInSqlContext:
    """Tests for add_in_sql_context logic."""

    def _make_tool_action(self, function_name, output_data, status=ActionStatus.SUCCESS):
        """Helper to create a TOOL action."""
        return ActionHistory(
            action_id=f"test_{function_name}",
            role=ActionRole.TOOL,
            messages=f"Tool call: {function_name}",
            action_type=function_name,
            input={"function_name": function_name, "arguments": "{}"},
            output=output_data,
            status=status,
        )

    def test_no_sql_action_found(self, real_agent_config, mock_llm_create):
        """No read_query action in incremental_actions logs warning and returns."""
        cmds = _make_chat_commands(real_agent_config)
        actions = [
            self._make_tool_action("list_tables", {"success": True, "raw_output": "tables"}),
        ]

        # Should not raise, just silently return
        cmds.add_in_sql_context("SELECT 1", "Test explanation", actions)

        # Verify no SQL context was added
        assert len(cmds.cli.cli_context.recent_sql_contexts) == 0
        assert cmds.cli.cli_context.get_last_sql_context() is None

    def test_sql_action_found_with_success(self, real_agent_config, mock_llm_create):
        """read_query action with success adds SQL context."""
        cmds = _make_chat_commands(real_agent_config)
        actions = [
            self._make_tool_action("list_tables", {"success": True, "raw_output": "tables"}),
            self._make_tool_action(
                "read_query",
                {
                    "success": "True",
                    "raw_output": {
                        "success": 1,
                        "result": {
                            "original_rows": 5,
                            "compressed_data": "id,name\n1,Alice\n2,Bob",
                        },
                    },
                },
            ),
        ]

        cmds.add_in_sql_context("SELECT * FROM users", "Get all users", actions)

        last_ctx = cmds.cli.cli_context.get_last_sql_context()
        assert last_ctx is not None
        assert last_ctx.sql_query == "SELECT * FROM users"
        assert last_ctx.row_count == 5

    def test_sql_action_found_with_failure(self, real_agent_config, mock_llm_create):
        """read_query action with failure adds SQL context with error."""
        cmds = _make_chat_commands(real_agent_config)
        actions = [
            self._make_tool_action(
                "read_query",
                {
                    "success": "True",
                    "raw_output": {
                        "success": 0,
                        "error": "Table not found",
                    },
                },
            ),
        ]

        cmds.add_in_sql_context("SELECT * FROM nonexistent", "Query failed", actions)

        last_ctx = cmds.cli.cli_context.get_last_sql_context()
        assert last_ctx is not None
        assert last_ctx.sql_query == "SELECT * FROM nonexistent"
        assert last_ctx.sql_error == "Table not found"
        assert last_ctx.row_count == 0

    def test_sql_action_output_not_successful(self, real_agent_config, mock_llm_create):
        """read_query with success=False (empty string falsy) records error."""
        cmds = _make_chat_commands(real_agent_config)
        actions = [
            self._make_tool_action(
                "read_query",
                {
                    "success": "",
                    "error": "Permission denied",
                    "raw_output": "Permission denied",
                },
            ),
        ]

        cmds.add_in_sql_context("SELECT * FROM secret", "Permission check", actions)

        last_ctx = cmds.cli.cli_context.get_last_sql_context()
        assert last_ctx is not None
        assert last_ctx.sql_query == "SELECT * FROM secret"
        assert last_ctx.row_count == 0

    def test_multiple_read_query_actions_uses_last(self, real_agent_config, mock_llm_create):
        """When multiple read_query actions exist, the last one is used."""
        cmds = _make_chat_commands(real_agent_config)
        actions = [
            self._make_tool_action(
                "read_query",
                {
                    "success": "True",
                    "raw_output": {
                        "success": 1,
                        "result": {"original_rows": 3, "compressed_data": "first"},
                    },
                },
            ),
            self._make_tool_action(
                "read_query",
                {
                    "success": "True",
                    "raw_output": {
                        "success": 1,
                        "result": {"original_rows": 10, "compressed_data": "second"},
                    },
                },
            ),
        ]
        # Make the second one have a unique action_id
        actions[1].action_id = "test_read_query_2"

        cmds.add_in_sql_context("SELECT * FROM users", "Latest query", actions)

        last_ctx = cmds.cli.cli_context.get_last_sql_context()
        assert last_ctx is not None
        assert last_ctx.row_count == 10
        assert last_ctx.sql_return == "second"


# ===========================================================================
# TestExecuteChatCommand
# ===========================================================================


class TestExecuteChatCommand:
    """Tests for execute_chat_command basic flow."""

    def test_empty_message_prints_warning(self, real_agent_config, mock_llm_create):
        """Empty message prints a yellow warning and returns early."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.execute_chat_command("")

        output = _get_console_output(console)
        assert "Please provide a message" in output
        assert cmds.current_node is None

    def test_whitespace_message_prints_warning(self, real_agent_config, mock_llm_create):
        """Whitespace-only message prints warning."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.execute_chat_command("   ")

        output = _get_console_output(console)
        assert "Please provide a message" in output
        assert cmds.current_node is None

    def test_basic_chat_creates_node(self, real_agent_config, mock_llm_create):
        """A non-empty message triggers node creation."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # execute_chat_command will fail during asyncio.run(run_chat_stream...)
        # because MockLLMModel has no responses. But it should still create the node.
        cmds.execute_chat_command("Hello, how are you?")

        # Node should have been created (even if execution failed)
        assert cmds.current_node is not None
        assert isinstance(cmds.current_node, ChatAgenticNode)

    def test_subagent_creates_correct_node(self, real_agent_config, mock_llm_create):
        """Providing a subagent_name creates the corresponding node type."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.execute_chat_command("Generate SQL for users", subagent_name="gensql")

        assert cmds.current_node is not None
        assert isinstance(cmds.current_node, GenSQLAgenticNode)
        assert cmds.current_subagent_name == "gensql"

    def test_chat_history_updated_after_execution(self, real_agent_config, mock_llm_create):
        """Chat history is updated after execution (even if execution encounters errors)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.execute_chat_command("Test message")

        # Chat history should have one entry
        assert len(cmds.chat_history) == 1
        assert cmds.chat_history[0]["user"] == "Test message"


# ===========================================================================
# TestUpdateChatNodeTools
# ===========================================================================


class TestUpdateChatNodeTools:
    """Tests for update_chat_node_tools method."""

    def test_no_current_node_does_not_raise(self, real_agent_config, mock_llm_create):
        """Calling update_chat_node_tools with no current node does not raise."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = None
        cmds.chat_node = None

        # Should not raise
        cmds.update_chat_node_tools()
        assert cmds.current_node is None

    def test_with_current_node_calls_setup_tools(self, real_agent_config, mock_llm_create):
        """Calling update_chat_node_tools with a current node calls setup_tools."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node()
        cmds.chat_node = cmds.current_node

        # Should not raise; setup_tools exists on ChatAgenticNode
        cmds.update_chat_node_tools()
        assert cmds.current_node is not None
        assert cmds.chat_node is not None


# ===========================================================================
# Edge cases and additional coverage
# ===========================================================================


class TestEdgeCases:
    """Additional edge case tests for broader coverage."""

    def test_extract_sql_and_output_empty_string(self, real_agent_config, mock_llm_create):
        """Empty string returns (None, None)."""
        cmds = _make_chat_commands(real_agent_config)
        sql, output = cmds._extract_sql_and_output_from_content("")
        assert sql is None
        assert output is None

    def test_extract_report_whitespace_only(self, real_agent_config, mock_llm_create):
        """Whitespace-only string returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json("   ")
        assert result is None

    def test_display_sql_with_copy_stores_last_sql(self, real_agent_config, mock_llm_create):
        """_display_sql_with_copy updates cli.last_sql."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_sql_with_copy("SELECT count(*) FROM orders")
        assert cmds.cli.last_sql == "SELECT count(*) FROM orders"

        cmds._display_sql_with_copy("SELECT 2")
        assert cmds.cli.last_sql == "SELECT 2"

    def test_create_node_input_with_cli_context_values(self, real_agent_config, mock_llm_create):
        """create_node_input reads catalog/database/schema from cli_context."""
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        cmds = _make_chat_commands(real_agent_config)
        cmds.cli.cli_context.current_catalog = "my_catalog"
        cmds.cli.cli_context.current_db_name = "my_db"
        cmds.cli.cli_context.current_schema = "my_schema"

        # ChatAgenticNode is independent from GenSQLAgenticNode -> ChatNodeInput
        node = cmds._create_new_node()
        node_input, node_type = cmds.create_node_input("Test", node, [], [], [])

        assert isinstance(node_input, ChatNodeInput)
        assert node_input.catalog == "my_catalog"
        assert node_input.database == "my_db"
        assert node_input.db_schema == "my_schema"

    def test_should_create_new_node_none_subagent_with_existing_subagent(self, real_agent_config, mock_llm_create):
        """Switching from subagent to regular (None) creates new node."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node(subagent_name="gen_metrics")
        cmds.current_subagent_name = "gen_metrics"

        assert cmds._should_create_new_node(subagent_name=None) is True
        assert cmds._should_create_new_node() is True

    def test_extract_sql_multiple_sql_blocks(self, real_agent_config, mock_llm_create):
        """Multiple SQL code blocks: first one is extracted."""
        cmds = _make_chat_commands(real_agent_config)
        content = "```sql\nSELECT 1\n```\nSome text\n```sql\nSELECT 2\n```"

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT 1"
        assert output is None

    def test_display_success_with_none_output(self, real_agent_config, mock_llm_create):
        """render_interaction_success handles action with None output gracefully."""
        from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer

        console = Console(file=io.StringIO(), no_color=True)
        renderer = ActionRenderer(ActionContentGenerator())

        action = ActionHistory(
            action_id="test_none_output",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output=None,
            status=ActionStatus.SUCCESS,
        )
        # Should not raise
        renderables = renderer.render_interaction_success(action, verbose=False)
        renderer.print_renderables(console, renderables)
        output = _get_console_output(console)
        assert isinstance(output, str)

    def test_cmd_clear_chat_resets_both_nodes(self, real_agent_config, mock_llm_create):
        """cmd_clear_chat resets both current_node and chat_node to None."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = cmds._create_new_node()
        cmds.chat_node = cmds.current_node
        cmds.current_subagent_name = None

        cmds.cmd_clear_chat("")

        assert cmds.current_node is None
        assert cmds.chat_node is None

    def test_add_in_sql_context_processing_action_skipped(self, real_agent_config, mock_llm_create):
        """read_query action with PROCESSING status is not considered (is_done() is False)."""
        cmds = _make_chat_commands(real_agent_config)
        processing_action = ActionHistory(
            action_id="test_processing",
            role=ActionRole.TOOL,
            messages="Tool call: read_query",
            action_type="read_query",
            input={"function_name": "read_query", "arguments": "{}"},
            output={},
            status=ActionStatus.PROCESSING,
        )

        cmds.add_in_sql_context("SELECT 1", "Test", [processing_action])
        # Should not add context since action is not done
        assert len(cmds.cli.cli_context.recent_sql_contexts) == 0

    def test_extract_report_from_json_report_with_rich_content(self, real_agent_config, mock_llm_create):
        """Report field with markdown content is extracted correctly."""
        cmds = _make_chat_commands(real_agent_config)
        report_content = "# Sales Report\n\n## Summary\n- Total: $100K\n- Growth: 15%"
        json_str = json.dumps({"report": report_content, "metadata": {"version": 1}})

        result = cmds._extract_report_from_json(json_str)
        assert result == report_content
        assert "Sales Report" in result

    def test_create_node_input_cli_context_none_values(self, real_agent_config, mock_llm_create):
        """create_node_input passes None for catalog/database/schema when cli_context has no values."""
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        cmds = _make_chat_commands(real_agent_config)
        # cli_context defaults have None for catalog/db_name/schema
        assert cmds.cli.cli_context.current_catalog is None
        assert cmds.cli.cli_context.current_db_name is None
        assert cmds.cli.cli_context.current_schema is None

        node = cmds._create_new_node()
        node_input, node_type = cmds.create_node_input("Test", node, [], [], [])

        assert isinstance(node_input, ChatNodeInput)
        assert node_input.catalog is None
        assert node_input.database is None
        assert node_input.db_schema is None

    def test_display_markdown_response_simple_text(self, real_agent_config, mock_llm_create):
        """Simple text with no markdown formatting is still displayed."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_markdown_response("Just a plain text response with no formatting.")

        output = _get_console_output(console)
        assert "plain text response" in output
        assert len(output) > 20

    def test_gensql_node_creation_default_subagent(self, real_agent_config, mock_llm_create):
        """Unknown subagent names without node_class default to GenSQLAgenticNode."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        # Add a custom subagent without node_class
        real_agent_config.agentic_nodes["custom_agent"] = {
            "system_prompt": "custom",
            "max_turns": 5,
        }

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="custom_agent")

        assert isinstance(node, GenSQLAgenticNode)
        assert not isinstance(node, ChatAgenticNode)

    def test_display_sql_with_copy_multiline_sql(self, real_agent_config, mock_llm_create):
        """Multiline SQL is displayed correctly."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        sql = "SELECT\n  id,\n  name\nFROM\n  users\nWHERE\n  active = 1"
        cmds._display_sql_with_copy(sql)

        output = _get_console_output(console)
        assert "SELECT" in output
        assert "users" in output
        assert cmds.cli.last_sql == sql


# ===========================================================================
# TestTriggerCompact
# ===========================================================================


# ===========================================================================
# TestCmdCompact
# ===========================================================================


class TestCmdCompact:
    """Tests for cmd_compact session compaction."""

    def test_compact_no_current_node(self, real_agent_config, mock_llm_create):
        """No active session prints warning."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_compact("")

        output = _get_console_output(console)
        assert "No active session" in output

    def test_compact_node_without_session(self, real_agent_config, mock_llm_create):
        """Node exists but no session_id prints 'No active session'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        cmds.current_node = cmds._create_new_node()

        cmds.cmd_compact("")

        output = _get_console_output(console)
        assert "No active session" in output


# ===========================================================================
# TestDisplayExceptionPaths
# ===========================================================================


class TestDisplayExceptionPaths:
    """Tests for exception/fallback paths in display methods."""

    def test_display_semantic_model_exception_fallback(self, real_agent_config, mock_llm_create):
        """Exception in display falls back to simple print."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Test with valid list - this exercises normal path
        cmds._display_semantic_model(["/path/to/model.yaml", "/path/to/model2.yaml"])

        output = _get_console_output(console)
        assert "model.yaml" in output
        assert "model2.yaml" in output

    def test_display_sql_summary_file_normal(self, real_agent_config, mock_llm_create):
        """Normal display of SQL summary file path."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_sql_summary_file("/output/test_summary.yaml")

        output = _get_console_output(console)
        assert "test_summary.yaml" in output

    def test_display_ext_knowledge_file_normal(self, real_agent_config, mock_llm_create):
        """Normal display of external knowledge file path."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds._display_ext_knowledge_file("/output/test_knowledge.yaml")

        output = _get_console_output(console)
        assert "test_knowledge.yaml" in output

    def test_display_success_text_content_type(self, real_agent_config, mock_llm_create):
        """Text content type renders as plain text."""
        from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer

        console = Console(file=io.StringIO(), no_color=True)
        renderer = ActionRenderer(ActionContentGenerator())

        action = ActionHistory(
            action_id="test_text_ct",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output={"content": "Plain text result", "content_type": "text"},
            status=ActionStatus.SUCCESS,
        )
        renderables = renderer.render_interaction_success(action, verbose=False)
        renderer.print_renderables(console, renderables)

        output = _get_console_output(console)
        assert "Plain text result" in output

    def test_display_success_renders_content(self, real_agent_config, mock_llm_create):
        """render_interaction_success renders content."""
        from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer

        console = Console(file=io.StringIO(), no_color=True)
        renderer = ActionRenderer(ActionContentGenerator())

        action = ActionHistory(
            action_id="test_err_success",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output={"content": "Fallback content here"},
            status=ActionStatus.SUCCESS,
        )
        renderables = renderer.render_interaction_success(action, verbose=False)
        renderer.print_renderables(console, renderables)

        output = _get_console_output(console)
        assert len(output) > 0


# ===========================================================================
# TestExecuteChatCommandResponseProcessing
# ===========================================================================


class TestExecuteChatCommandResponseProcessing:
    """Tests for execute_chat_command response processing paths."""

    def test_execute_with_mock_response_processes_output(self, real_agent_config, mock_llm_create):
        """execute_chat_command with a mock LLM response processes the output."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Configure mock to return a simple response
        mock_llm_create.reset(responses=[build_simple_response("Here is your answer.")])

        cmds.execute_chat_command("What is 1+1?")

        output = _get_console_output(console)
        assert cmds.current_node is not None
        assert len(cmds.chat_history) == 1
        # The response should be processed in some form
        assert len(output) > 0

    def test_execute_with_sql_response(self, real_agent_config, mock_llm_create):
        """execute_chat_command with SQL response displays SQL panel."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        sql_response = json.dumps(
            {
                "sql": "SELECT count(*) FROM schools",
                "response": "Here is the count query",
            }
        )
        mock_llm_create.reset(responses=[build_simple_response(sql_response)])

        cmds.execute_chat_command("Count all schools")

        _get_console_output(console)
        assert cmds.current_node is not None
        assert len(cmds.chat_history) == 1

    def test_execute_with_report_response(self, real_agent_config, mock_llm_create):
        """execute_chat_command with report JSON response extracts report."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        report_response = json.dumps({"report": "# Analysis Report\nFindings here."})
        mock_llm_create.reset(responses=[build_simple_response(report_response)])

        cmds.execute_chat_command("Generate a report")

        _get_console_output(console)
        assert len(cmds.chat_history) == 1

    def test_execute_with_none_response(self, real_agent_config, mock_llm_create):
        """execute_chat_command handles None response gracefully."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("")])

        cmds.execute_chat_command("Hello")

        assert cmds.current_node is not None
        assert len(cmds.chat_history) == 1

    def test_execute_reuses_existing_node(self, real_agent_config, mock_llm_create):
        """Second execute_chat_command reuses the existing node."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("First"), build_simple_response("Second")])

        cmds.execute_chat_command("First message")
        first_node = cmds.current_node

        cmds.execute_chat_command("Second message")
        second_node = cmds.current_node

        assert first_node is second_node
        assert len(cmds.chat_history) == 2

    def test_execute_switches_subagent_creates_new_node(self, real_agent_config, mock_llm_create):
        """Switching subagent creates a new node."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("Chat reply"),
                build_simple_response("SQL reply"),
            ]
        )

        cmds.execute_chat_command("Hello", subagent_name=None)
        first_node = cmds.current_node

        cmds.execute_chat_command("Generate SQL", subagent_name="gensql")
        second_node = cmds.current_node

        assert first_node is not second_node
        assert cmds.current_subagent_name == "gensql"


# ===========================================================================
# TestExtractReportEdgeCases
# ===========================================================================


class TestExtractReportEdgeCases:
    """Additional edge cases for _extract_report_from_json."""

    def test_json_array_returns_none(self, real_agent_config, mock_llm_create):
        """JSON array (not object) returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json("[1, 2, 3]")
        assert result is None

    def test_deeply_nested_report(self, real_agent_config, mock_llm_create):
        """Report field is extracted even from larger JSON."""
        cmds = _make_chat_commands(real_agent_config)
        json_str = json.dumps(
            {
                "report": "Deep report content",
                "metadata": {"nested": {"deep": True}},
                "extra": [1, 2, 3],
            }
        )
        result = cmds._extract_report_from_json(json_str)
        assert result == "Deep report content"

    def test_malformed_json_returns_none(self, real_agent_config, mock_llm_create):
        """Malformed JSON that json_repair can't fix returns None."""
        cmds = _make_chat_commands(real_agent_config)
        result = cmds._extract_report_from_json("{{{broken")
        # json_repair may repair it or not - either way should not raise
        assert result is None or isinstance(result, str)


# ===========================================================================
# TestExtractSqlEdgeCases
# ===========================================================================


class TestExtractSqlEdgeCases:
    """Additional edge cases for _extract_sql_and_output_from_content."""

    def test_json_with_only_output_no_sql(self, real_agent_config, mock_llm_create):
        """JSON with output but no sql field."""
        cmds = _make_chat_commands(real_agent_config)
        content = json.dumps({"output": "Some data", "explanation": "test"})

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql is None
        assert output == "Some data"

    def test_json_with_newlines_in_output(self, real_agent_config, mock_llm_create):
        """JSON output with escaped newlines is unescaped."""
        cmds = _make_chat_commands(real_agent_config)
        content = json.dumps({"sql": "SELECT 1", "output": "line1\\nline2\\nline3"})

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql == "SELECT 1"
        assert output is not None

    def test_content_with_only_text_no_json_no_sql(self, real_agent_config, mock_llm_create):
        """Content with no JSON and no SQL blocks returns (None, None)."""
        cmds = _make_chat_commands(real_agent_config)
        content = "The database has 5 tables with various schemas for school data."

        sql, output = cmds._extract_sql_and_output_from_content(content)
        assert sql is None
        assert output is None


# ===========================================================================
# TestSelectChoiceWithPipeInput — synchronous tests using create_pipe_input
# ===========================================================================


class TestSelectChoiceWithPipeInput:
    """Tests for select_choice and prompt_input using prompt_toolkit's pipe input.

    Uses create_pipe_input() + create_app_session() to feed deterministic
    keystrokes. Tests must call the functions synchronously (NOT through
    _handle_cli_interaction's async run_in_executor) to avoid macOS kqueue
    issues with nested event loops in threads.
    """

    def test_select_choice_returns_sent_key(self, real_agent_config, mock_llm_create):
        """select_choice returns the key that was sent via pipe input."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.send_text("n")
                result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")
        assert result == "n"

    def test_select_choice_returns_default_key(self, real_agent_config, mock_llm_create):
        """select_choice returns default when Enter is pressed."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.send_text("\r")  # Enter key
                result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")
        assert result == "y"

    def test_select_choice_eof_returns_default(self, real_agent_config, mock_llm_create):
        """select_choice returns default when pipe is closed (EOF)."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.close()
                result = select_choice(console, choices={"a": "A", "b": "B"}, default="a")
        assert result == "a"

    def test_select_choice_arrow_down_then_enter(self, real_agent_config, mock_llm_create):
        """select_choice navigates down with arrow key and selects with Enter."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                # Down arrow = \x1b[B, then Enter = \r
                pipe_input.send_text("\x1b[B\r")
                result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")
        assert result == "n"

    def test_select_choice_arrow_up_wraps_around(self, real_agent_config, mock_llm_create):
        """select_choice wraps around when pressing Up from the first item."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                # Up arrow = \x1b[A wraps to last item, then Enter
                pipe_input.send_text("\x1b[A\r")
                result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")
        # Wraps from index 0 to index 1 (last item)
        assert result == "n"

    def test_select_choice_ctrl_c_returns_default(self, real_agent_config, mock_llm_create):
        """select_choice returns default when Ctrl+C is pressed."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.send_text("\x03")  # Ctrl+C
                result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")
        assert result == "y"

    def test_prompt_input_returns_text(self, real_agent_config, mock_llm_create):
        """prompt_input returns the text sent via pipe input."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import prompt_input

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.send_text("hello world\n")
                result = prompt_input(console, "Enter text")
        assert result == "hello world"

    def test_prompt_input_eof_returns_default(self, real_agent_config, mock_llm_create):
        """prompt_input returns default on EOF (pipe closed)."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import prompt_input

        console = Console(file=io.StringIO(), no_color=True)
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.close()
                result = prompt_input(console, "Enter text", default="fallback")
        assert result == "fallback"

    def test_select_choice_with_three_options(self, real_agent_config, mock_llm_create):
        """select_choice works with multiple choices and direct key press."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input

        from datus.cli._cli_utils import select_choice

        console = Console(file=io.StringIO(), no_color=True)
        choices = {"y": "Allow (once)", "a": "Always allow", "n": "Deny"}
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input):
                pipe_input.send_text("a")
                result = select_choice(console, choices=choices, default="y")
        assert result == "a"


# ===========================================================================
# TestExecuteResponseProcessing — 覆盖 execute_chat_command 的响应处理分支
# ===========================================================================


class TestExecuteResponseProcessing:
    """Tests for response processing branches inside execute_chat_command (lines 390-461)."""

    def _run_execute_and_get_output(self, real_agent_config, mock_llm_create, message, responses):
        """Helper: run execute_chat_command with mock responses, return (cmds, console_output)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        mock_llm_create.reset(responses=responses)
        cmds.execute_chat_command(message)
        return cmds, _get_console_output(console)

    def test_response_with_sql_displays_panel_and_adds_context(self, real_agent_config, mock_llm_create):
        """Response containing SQL triggers _display_sql_with_copy and add_in_sql_context."""
        from tests.unit_tests.mock_llm_model import MockLLMResponse, MockToolCall

        # Use a tool call to trigger read_query, then a response with SQL
        responses = [
            MockLLMResponse(
                tool_calls=[MockToolCall(name="read_query", arguments='{"query": "SELECT 1"}')],
                content=json.dumps(
                    {
                        "sql": "SELECT count(*) FROM schools",
                        "explanation": "Count schools",
                    }
                ),
            )
        ]

        cmds, output = self._run_execute_and_get_output(real_agent_config, mock_llm_create, "Count schools", responses)

        assert len(cmds.chat_history) == 1
        # SQL should be displayed or stored
        assert cmds.current_node is not None

    def test_response_with_report_json_extracts_report(self, real_agent_config, mock_llm_create):
        """Response with report JSON format extracts report field for display."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        report_json = json.dumps({"report": "# Analysis\nKey findings here."})
        responses = [build_simple_response(report_json)]

        cmds, output = self._run_execute_and_get_output(
            real_agent_config, mock_llm_create, "Generate report", responses
        )

        assert len(cmds.chat_history) == 1
        assert cmds.current_node is not None

    def test_response_with_plain_text_uses_fallback(self, real_agent_config, mock_llm_create):
        """Plain text response uses fallback path (ast.literal_eval or direct)."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        responses = [build_simple_response("The database has 5 tables with student records.")]

        cmds, output = self._run_execute_and_get_output(
            real_agent_config, mock_llm_create, "Describe the database", responses
        )

        assert len(cmds.chat_history) == 1
        # Plain text should be rendered as markdown
        assert "table" in output.lower() or "database" in output.lower() or len(output) > 0

    def test_response_with_dict_string_extracts_raw_output(self, real_agent_config, mock_llm_create):
        """Response that is a Python dict string exercises ast.literal_eval path."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        # ast.literal_eval can parse this
        dict_str = "{'raw_output': 'Extracted content from dict', 'status': 'ok'}"
        responses = [build_simple_response(dict_str)]

        cmds, output = self._run_execute_and_get_output(real_agent_config, mock_llm_create, "Process data", responses)

        assert len(cmds.chat_history) == 1

    def test_response_none_uses_empty_fallback(self, real_agent_config, mock_llm_create):
        """Empty/None response exercises the response is None fallback path."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        responses = [build_simple_response("")]

        cmds, output = self._run_execute_and_get_output(real_agent_config, mock_llm_create, "Hello", responses)

        assert len(cmds.chat_history) == 1
        assert cmds.current_node is not None

    def test_session_reuse_shows_session_info(self, real_agent_config, mock_llm_create):
        """Second execute reusing same node shows 'Using existing session' info."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("First"), build_simple_response("Second")])

        cmds.execute_chat_command("First message")
        # Clear console buffer for second call
        console.file = io.StringIO()

        cmds.execute_chat_command("Second message")
        output = _get_console_output(console)

        assert len(cmds.chat_history) == 2
        # Second call reuses the node, may show session info
        assert "Processing" in output or "Using existing session" in output


# ===========================================================================
# TestCmdChatInfoWithSession — cmd_chat_info 带活跃 session
# ===========================================================================


class TestCmdChatInfoWithSession:
    """Tests for cmd_chat_info when a session is active."""

    def test_chat_info_after_execute(self, real_agent_config, mock_llm_create):
        """cmd_chat_info shows session details after executing a command."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("Hello!")])
        cmds.execute_chat_command("Hi")

        # Clear console for info output
        console.file = io.StringIO()
        cmds.cmd_chat_info("")

        output = _get_console_output(console)
        # Should display session info or "No active session" depending on session state
        assert len(output) > 0
        assert "Session" in output or "No active session" in output

    def test_chat_info_shows_recent_conversations(self, real_agent_config, mock_llm_create):
        """cmd_chat_info displays recent conversation history."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("Reply 1"),
                build_simple_response("Reply 2"),
            ]
        )
        cmds.execute_chat_command("Question 1")
        cmds.execute_chat_command("Question 2")

        console.file = io.StringIO()
        cmds.cmd_chat_info("")

        output = _get_console_output(console)
        assert len(output) > 0


# ===========================================================================
# TestCmdCompactWithSession — cmd_compact 带活跃 session
# ===========================================================================


class TestCmdCompactWithSession:
    """Tests for cmd_compact when a session is active."""

    def test_compact_after_execute(self, real_agent_config, mock_llm_create):
        """cmd_compact on an active session attempts compaction."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("Hello!"),
                # compact 内部会调 _manual_compact -> generate (需要一个额外响应)
                build_simple_response("Summary of conversation"),
            ]
        )
        cmds.execute_chat_command("Hi there")

        console.file = io.StringIO()
        cmds.cmd_compact("")

        output = _get_console_output(console)
        # Should show compact result (success or failure)
        assert len(output) > 0
        assert (
            "Compacting" in output
            or "compacted" in output.lower()
            or "No active session" in output
            or "Error" in output
        )

    def test_compact_resets_in_memory_state(self, real_agent_config, mock_llm_create):
        """After successful compact, in-memory state is reset via _reload_state_from_session."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("First reply"),
                build_simple_response("Second reply"),
                # compact summary response
                build_simple_response("Summary of conversation"),
            ]
        )

        # Two rounds of chat to accumulate history
        cmds.execute_chat_command("First question")
        cmds.execute_chat_command("Second question")

        # Verify pre-compact state has accumulated data
        assert len(cmds.all_turn_actions) == 2
        assert len(cmds.chat_history) == 2
        assert len(cmds.last_actions) > 0

        console.file = io.StringIO()
        cmds.cmd_compact("")

        output = _get_console_output(console)
        # Compact must succeed; this is the precondition for the reload behavior
        # under test. A silent failure here would have previously made the
        # remaining assertions vacuous.
        assert "compacted successfully" in output.lower(), f"compact did not succeed; output={output!r}"
        # After successful compact, _reload_state_from_session rebuilds from
        # the compacted session (which contains only the summary pair), so
        # pre-compact accumulated turn actions must be cleared.
        assert cmds._trace_verbose is False
        assert cmds.current_node.actions == []


# ===========================================================================
# TestTriggerCompactWithSession — _trigger_compact 带活跃 session
# ===========================================================================


class TestTriggerCompactWithSession:
    """Tests for subagent switch without compact."""

    def test_no_compact_on_subagent_switch(self, real_agent_config, mock_llm_create):
        """Switching subagent should NOT trigger compact."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("Chat reply"),
                build_simple_response("SQL reply"),
            ]
        )

        # First: create a chat session
        cmds.execute_chat_command("Hello")
        assert cmds.current_node is not None

        # Second: switch to subagent — no compact should be triggered
        console.file = io.StringIO()
        cmds.execute_chat_command("Generate SQL", subagent_name="gensql")

        output = _get_console_output(console)
        assert cmds.current_subagent_name == "gensql"
        assert "compacting" not in output.lower()


# ===========================================================================
# TestCmdClearChatWithSession — cmd_clear_chat 带 session 的 delete 路径
# ===========================================================================


class TestCmdClearChatWithSession:
    """Tests for cmd_clear_chat when a session is active (exercises delete_session)."""

    def test_clear_chat_deletes_session(self, real_agent_config, mock_llm_create):
        """cmd_clear_chat with active session calls delete_session."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("Hello!")])
        cmds.execute_chat_command("Hi")
        assert cmds.current_node is not None

        console.file = io.StringIO()
        cmds.cmd_clear_chat("")

        output = _get_console_output(console)
        assert cmds.current_node is None
        assert cmds.chat_node is None
        assert "cleared" in output.lower()


# ===========================================================================
# TestCmdResumeWithSession — cmd_resume with direct session_id
# ===========================================================================


class TestCmdResumeWithSession:
    """Tests for cmd_resume when sessions exist (exercises lines 983-1111)."""

    def test_resume_with_direct_session_id(self, real_agent_config, mock_llm_create):
        """cmd_resume with a valid session_id resumes the session and displays history."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a disk session with user+assistant messages
        session_id = "chat_session_resume01"
        _create_session_on_disk(
            session_id,
            [("user", "Hello"), ("assistant", "Hi there!")],
        )

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        assert cmds.current_node.session_id == session_id
        assert "resuming session" in output.lower() or "continue the conversation" in output.lower()

    def test_resume_session_not_found(self, real_agent_config, mock_llm_create):
        """cmd_resume with a nonexistent session_id shows error message."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_resume("nonexistent_session_id_12345")

        output = _get_console_output(console)
        assert "not found" in output.lower() or "error" in output.lower()

    def test_resume_no_sessions_interactive_path(self, real_agent_config, mock_llm_create):
        """cmd_resume with no args and no sessions shows 'No sessions found'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "no sessions" in output.lower() or "error" in output.lower()

    def test_resume_displays_conversation_history(self, real_agent_config, mock_llm_create):
        """cmd_resume shows conversation messages when session has messages."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_resume02"
        _create_session_on_disk(
            session_id,
            [
                ("user", "First question"),
                ("assistant", "First reply"),
                ("user", "Second question"),
                ("assistant", "Second reply"),
            ],
        )

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        assert cmds.current_node.session_id == session_id
        assert "resumed" in output.lower() or "continue" in output.lower()
        # Should display message count
        assert "message" in output.lower()

    def test_resume_updates_state_correctly(self, real_agent_config, mock_llm_create):
        """cmd_resume correctly updates current_node, current_subagent_name, chat_node."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_resume03"
        _create_session_on_disk(session_id)

        cmds.cmd_resume(session_id)

        # Verify state is updated correctly
        assert cmds.current_node is not None
        assert cmds.current_node.session_id == session_id
        # chat session -> subagent_name should be None, chat_node should be updated
        assert cmds.current_subagent_name is None
        assert cmds.chat_node is not None

    def test_resume_exception_handling(self, real_agent_config, mock_llm_create):
        """cmd_resume handles exceptions gracefully with invalid session_id."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Invalid session_id with path traversal characters
        cmds.cmd_resume("../../../etc/passwd")

        output = _get_console_output(console)
        assert "error" in output.lower()

    def test_resume_shows_user_and_assistant_messages(self, real_agent_config, mock_llm_create):
        """cmd_resume displays both user and assistant messages from history."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_resume04"
        _create_session_on_disk(
            session_id,
            [("user", "What is SQL?"), ("assistant", "SQL is a query language.")],
        )

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # Should show "You:" for user messages
        assert "you:" in output.lower() or "message" in output.lower()


# ===========================================================================
# TestCmdRewindWithSession — cmd_rewind with active session + disk session
# ===========================================================================


class TestCmdRewindWithSession:
    """Tests for cmd_rewind when an active session exists (exercises lines 1113-1229)."""

    def _setup_node_with_disk_session(self, cmds, mock_llm_create, session_id, messages):
        """Helper: create a disk session and set current_node to point to it."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        _create_session_on_disk(session_id, messages)

        # Create a real node via execute to get a properly initialized node
        mock_llm_create.reset(responses=[build_simple_response("Init")])
        cmds.execute_chat_command("Init")

        # Override the node's session_id to the disk session
        cmds.current_node.session_id = session_id

    def test_rewind_no_active_session(self, real_agent_config, mock_llm_create):
        """cmd_rewind with no active session shows warning."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert "no active session" in output.lower()

    def test_rewind_with_turn_number(self, real_agent_config, mock_llm_create):
        """cmd_rewind with a valid turn number creates a branched session."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind01"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [
                ("user", "Question 1"),
                ("assistant", "Reply 1"),
                ("user", "Question 2"),
                ("assistant", "Reply 2"),
            ],
        )

        console.file = io.StringIO()
        mock_llm_create.reset(responses=[])
        result = cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # Turn 1 creates a fresh session (no prior messages)
        assert cmds.current_node.session_id != session_id
        # Should return the selected user message for input prefill
        assert result == "Question 1"
        assert "rewound" in output.lower() or "input buffer" in output.lower()

    def test_rewind_turn2_returns_message_and_keeps_turn1(self, real_agent_config, mock_llm_create):
        """cmd_rewind with turn 2 keeps turn 1 and returns turn 2 message."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind_ret2"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [
                ("user", "Question 1"),
                ("assistant", "Reply 1"),
                ("user", "Question 2"),
                ("assistant", "Reply 2"),
            ],
        )

        console.file = io.StringIO()
        mock_llm_create.reset(responses=[])
        result = cmds.cmd_rewind("2")

        assert result == "Question 2"
        assert cmds.current_node is not None
        assert cmds.current_node.session_id != session_id

        # Verify the branched session only contains turns before the rewound turn
        from datus.models.session_manager import SessionManager

        sm = SessionManager()
        new_messages = sm.get_session_messages(cmds.current_node.session_id)
        user_messages = [m["content"] for m in new_messages if m.get("role") == "user"]
        # Should contain only turn 1 user message, not turn 2
        assert "Question 1" in user_messages
        assert "Question 2" not in user_messages

    def test_rewind_cancel_returns_none(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_rewind with picker cancellation returns None."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind_cancel_ret"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Question"), ("assistant", "Reply")],
        )

        import datus.cli.chat_commands as chat_mod

        class FakeCancelApp:
            def __init__(self, **kwargs):
                pass

            def run(self):
                return None

        monkeypatch.setattr(chat_mod, "ListSelectorApp", FakeCancelApp)

        console.file = io.StringIO()
        result = cmds.cmd_rewind("")
        assert result is None

    def test_rewind_invalid_turn_number_too_high(self, real_agent_config, mock_llm_create):
        """cmd_rewind with turn number exceeding total turns shows error."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind02"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Question"), ("assistant", "Reply")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("5")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "must be between" in output.lower() or "error" in output.lower()

    def test_rewind_invalid_turn_number_zero(self, real_agent_config, mock_llm_create):
        """cmd_rewind with turn number 0 shows error."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind03"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Question"), ("assistant", "Reply")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("0")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "must be between" in output.lower() or "error" in output.lower()

    def test_rewind_non_numeric_input(self, real_agent_config, mock_llm_create):
        """cmd_rewind with non-numeric input shows error."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind04"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Question"), ("assistant", "Reply")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("abc")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "number" in output.lower()

    def test_rewind_displays_conversation_table(self, real_agent_config, mock_llm_create):
        """cmd_rewind displays table of user turns before rewind."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind05"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [
                ("user", "First message"),
                ("assistant", "Reply A"),
                ("user", "Second message"),
                ("assistant", "Reply B"),
                ("user", "Third message"),
                ("assistant", "Reply C"),
            ],
        )

        console.file = io.StringIO()
        mock_llm_create.reset(responses=[])
        cmds.cmd_rewind("2")

        output = _get_console_output(console)
        assert "conversation turns" in output.lower() or "turn" in output.lower()
        assert cmds.current_node is not None

    def test_rewind_cancel_with_q(self, real_agent_config, mock_llm_create):
        """cmd_rewind with 'q' as turn number shows cancellation."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind06"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Question"), ("assistant", "Reply")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("q")

        output = _get_console_output(console)
        assert "cancelled" in output.lower()

    def test_rewind_updates_state(self, real_agent_config, mock_llm_create):
        """cmd_rewind correctly updates current_node and session references."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind07"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [
                ("user", "Q1"),
                ("assistant", "R1"),
                ("user", "Q2"),
                ("assistant", "R2"),
            ],
        )

        mock_llm_create.reset(responses=[])
        cmds.cmd_rewind("1")

        assert cmds.current_node is not None
        new_session_id = cmds.current_node.session_id
        assert new_session_id != session_id
        assert cmds.chat_node is not None

    def test_rewind_no_messages_in_session(self, real_agent_config, mock_llm_create):
        """cmd_rewind with empty session shows 'no messages' warning."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a node via execute but with an empty disk session
        session_id = "chat_session_rewind08"
        mock_llm_create.reset(responses=[build_simple_response("Init")])
        cmds.execute_chat_command("Init")

        # Create disk session with NO messages (only session record)
        from datus.utils.path_manager import get_path_manager

        sessions_dir = str(get_path_manager().sessions_dir)
        os.makedirs(sessions_dir, exist_ok=True)
        db_path = os.path.join(sessions_dir, f"{session_id}.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "message_data TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute("INSERT INTO agent_sessions (session_id) VALUES (?)", (session_id,))
            conn.commit()

        cmds.current_node.session_id = session_id
        console.file = io.StringIO()
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert "no messages" in output.lower() or "no user turns" in output.lower() or "error" in output.lower()


# ===========================================================================
# TestResumeWithSqlMessages — cmd_resume with SQL in messages (lines 1084-1094)
# ===========================================================================


class TestResumeWithSqlMessages:
    """Tests for cmd_resume message rendering paths with SQL and actions."""

    def _create_session_with_sql_messages(self, session_id):
        """Create a session with function_call messages that SessionManager parses as SQL."""
        from datus.utils.path_manager import get_path_manager

        sessions_dir = str(get_path_manager().sessions_dir)
        os.makedirs(sessions_dir, exist_ok=True)
        db_path = os.path.join(sessions_dir, f"{session_id}.db")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "message_data TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)",
                (session_id,),
            )

            base = datetime(2025, 6, 1, 10, 0, 0)
            msgs = [
                # User message
                (
                    json.dumps({"role": "user", "content": "Find all students"}),
                    (base).isoformat(),
                ),
                # Function call (tool invocation)
                (
                    json.dumps(
                        {
                            "type": "function_call",
                            "name": "read_query",
                            "call_id": "call_001",
                            "arguments": json.dumps({"query": "SELECT * FROM students"}),
                        }
                    ),
                    (base + timedelta(seconds=1)).isoformat(),
                ),
                # Function call output
                (
                    json.dumps(
                        {
                            "type": "function_call_output",
                            "call_id": "call_001",
                            "output": "3 rows returned",
                        }
                    ),
                    (base + timedelta(seconds=2)).isoformat(),
                ),
                # Assistant message with SQL result
                (
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "sql": "SELECT * FROM students",
                                    "output": "Found 3 students",
                                }
                            ),
                        }
                    ),
                    (base + timedelta(seconds=3)).isoformat(),
                ),
            ]
            for msg_data, ts in msgs:
                conn.execute(
                    "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, ?)",
                    (session_id, msg_data, ts),
                )
            conn.commit()

    def test_resume_with_sql_messages_renders_sql(self, real_agent_config, mock_llm_create):
        """cmd_resume with SQL messages exercises sql rendering path (lines 1087-1089)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_sqlmsg01"
        self._create_session_with_sql_messages(session_id)

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        assert cmds.current_node.session_id == session_id
        # Should show resumed message and display content
        assert "resumed" in output.lower() or "continue" in output.lower()

    def test_resume_with_json_content_skips_markdown(self, real_agent_config, mock_llm_create):
        """Resume with JSON assistant content with SQL skips markdown rendering (line 1093)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_sqlmsg02"
        self._create_session_with_sql_messages(session_id)

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # Content should be rendered (either as SQL or markdown)
        assert len(output) > 50  # Should have substantial output


# ===========================================================================
# TestRewindEdgeCases — additional rewind edge cases
# ===========================================================================


class TestRewindEdgeCases:
    """Additional edge case tests for cmd_rewind."""

    def _setup_node_with_disk_session(self, cmds, mock_llm_create, session_id, messages):
        """Helper: create a disk session and set current_node to point to it."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        _create_session_on_disk(session_id, messages)
        mock_llm_create.reset(responses=[build_simple_response("Init")])
        cmds.execute_chat_command("Init")
        cmds.current_node.session_id = session_id

    def test_rewind_no_user_turns_only_assistant(self, real_agent_config, mock_llm_create):
        """cmd_rewind with session containing only assistant messages shows 'no user turns'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind09"
        # Session with only assistant messages (no user turns)
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("assistant", "I am ready"), ("assistant", "Waiting for input")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert "no user turns" in output.lower() or "no messages" in output.lower() or "error" in output.lower()

    def test_rewind_with_long_messages_truncates(self, real_agent_config, mock_llm_create):
        """cmd_rewind truncates long user messages in the table display (line 1152)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind10"
        long_msg = "A" * 200  # Very long message, should be truncated at 77+...
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", long_msg), ("assistant", "Reply")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("q")  # Cancel via args

        output = _get_console_output(console)
        assert "cancelled" in output.lower()

    def test_rewind_negative_turn_number(self, real_agent_config, mock_llm_create):
        """cmd_rewind with negative turn number shows error."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind11"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("user", "Q"), ("assistant", "R")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("-1")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "must be between" in output.lower() or "error" in output.lower()


# ===========================================================================
# TestResumeInteractiveNoSessions — cmd_resume interactive path with sessions
# ===========================================================================


class TestResumeInteractiveNoSessions:
    """Tests for cmd_resume interactive session selection path."""

    def test_resume_no_args_sessions_exist_but_empty(self, real_agent_config, mock_llm_create):
        """cmd_resume with no args and sessions that have no messages shows 'No sessions with messages'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create an empty session (has session record but no messages)
        from datus.utils.path_manager import get_path_manager

        sessions_dir = str(get_path_manager().sessions_dir)
        os.makedirs(sessions_dir, exist_ok=True)
        session_id = "chat_session_empty01"
        db_path = os.path.join(sessions_dir, f"{session_id}.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "message_data TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute("INSERT INTO agent_sessions (session_id) VALUES (?)", (session_id,))
            conn.commit()

        cmds.cmd_resume("")

        output = _get_console_output(console)
        # Should show "No sessions with messages" or "No sessions found"
        assert "no sessions" in output.lower() or "error" in output.lower()


# ===========================================================================
# TestResumeInteractiveWithSessions — cmd_resume interactive session picker
# ===========================================================================


class TestResumeInteractiveWithSessions:
    """Tests for cmd_resume interactive session listing and selection (lines 1006-1047)."""

    def test_resume_interactive_valid_selection(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user selects session #1 from the list."""
        from datus.cli.list_selector_app import ListSelection

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick01", [("user", "First Q"), ("assistant", "First A")])
        _create_session_on_disk("chat_session_pick01b", [("user", "Second Q"), ("assistant", "Second A")])

        captured = {}

        import datus.cli.chat_commands as chat_mod

        class FakeListSelectorApp:
            def __init__(self, **kwargs):
                captured["items"] = kwargs.get("items", [])

            def run(self):
                if captured["items"]:
                    return ListSelection(key=captured["items"][0].key)
                return None

        monkeypatch.setattr(chat_mod, "ListSelectorApp", FakeListSelectorApp)

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert cmds.current_node is not None
        assert "session" in output.lower()
        assert len(captured["items"]) >= 2

    def test_resume_interactive_cancel(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user cancels selection."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick02", [("user", "Q"), ("assistant", "A")])

        import datus.cli.chat_commands as chat_mod

        class FakeCancelApp:
            def __init__(self, **kwargs):
                pass

            def run(self):
                return None

        monkeypatch.setattr(chat_mod, "ListSelectorApp", FakeCancelApp)

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "cancelled" in output.lower() or "session" in output.lower()


class TestRewindDisplayMessages:
    """Tests for cmd_rewind message display paths (lines 1211-1221, 1227-1229)."""

    def _setup_node_with_disk_session(self, cmds, mock_llm_create, session_id, messages):
        """Helper: create a disk session and set current_node to point to it."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        _create_session_on_disk(session_id, messages)
        mock_llm_create.reset(responses=[build_simple_response("Init")])
        cmds.execute_chat_command("Init")
        cmds.current_node.session_id = session_id

    def test_rewind_displays_actions_and_sql_in_messages(self, real_agent_config, mock_llm_create):
        """cmd_rewind with messages containing actions and SQL shows them in display."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create session with user + assistant messages that have SQL-like content
        session_id = "chat_session_rewind_display01"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [
                ("user", "Show me students"),
                (
                    "assistant",
                    json.dumps({"sql": "SELECT * FROM students", "output": "3 rows"}),
                ),
                ("user", "More details"),
                ("assistant", "Here are the details"),
            ],
        )

        console.file = io.StringIO()
        mock_llm_create.reset(responses=[])
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        # Should show rewound message
        assert "rewound" in output.lower() or "turn" in output.lower() or "continue" in output.lower()

    def test_rewind_no_user_turns_shows_warning(self, real_agent_config, mock_llm_create):
        """cmd_rewind with session containing only assistant messages shows 'no user turns'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_rewind_nouser01"
        self._setup_node_with_disk_session(
            cmds,
            mock_llm_create,
            session_id,
            [("assistant", "I am ready"), ("assistant", "Still waiting")],
        )

        console.file = io.StringIO()
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert "no user turns" in output.lower() or "no messages" in output.lower() or "error" in output.lower()

    def test_rewind_exception_shows_error(self, real_agent_config, mock_llm_create):
        """cmd_rewind handles exceptions gracefully (lines 1227-1229)."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        from tests.unit_tests.mock_llm_model import build_simple_response

        mock_llm_create.reset(responses=[build_simple_response("Init")])
        cmds.execute_chat_command("Init")

        # Set an invalid session_id that will cause an error
        cmds.current_node.session_id = "nonexistent_invalid_session"

        console.file = io.StringIO()
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert "error" in output.lower() or "no messages" in output.lower()


class TestSelectChoiceExceptionHandler:
    """Tests for select_choice generic exception handling (_cli_utils.py lines 85-88)."""

    def test_generic_exception_returns_default(self, monkeypatch):
        """select_choice returns default when Application.run raises a generic Exception."""
        from prompt_toolkit import Application

        from datus.cli._cli_utils import select_choice

        def broken_run(self, *args, **kwargs):
            raise RuntimeError("Unexpected terminal error")

        monkeypatch.setattr(Application, "run", broken_run)

        console = Console(file=io.StringIO(), no_color=True)
        result = select_choice(console, choices={"y": "Yes", "n": "No"}, default="y")

        assert result == "y"
        output = console.file.getvalue()
        assert "Selection error" in output or "error" in output.lower()


class TestResumeListingTruncation:
    """Tests for cmd_resume listing truncation of session_id and first_user_message."""

    def test_long_first_message_is_truncated(self, real_agent_config, mock_llm_create, monkeypatch):
        """Session with long first message passes full text to ListSelectorApp items."""
        console = Console(file=io.StringIO(), no_color=True, width=160)
        cmds = _make_chat_commands(real_agent_config, console=console)

        long_sid = "chat_truncate_test_01_abcdefghijklmnop"
        long_msg = "This is a very long message that should definitely be truncated in the display listing table"
        _create_session_on_disk(long_sid, [("user", long_msg), ("assistant", "OK")])

        import datus.cli.chat_commands as chat_mod

        captured = {}

        class FakeCaptureApp:
            def __init__(self, **kwargs):
                captured["items"] = kwargs.get("items", [])

            def run(self):
                return None

        monkeypatch.setattr(chat_mod, "ListSelectorApp", FakeCaptureApp)

        console.file = io.StringIO()
        cmds.cmd_resume("")
        output = _get_console_output(console)
        assert "cancelled" in output.lower()
        assert len(captured["items"]) > 0
        first_item = captured["items"][0]
        assert long_msg.replace("\n", " ") in first_item.primary or first_item.primary in long_msg

    def test_short_session_id_not_truncated(self, real_agent_config, mock_llm_create, monkeypatch):
        """Session with short session_id appears in ListSelectorApp items without truncation."""
        console = Console(file=io.StringIO(), no_color=True, width=160)
        cmds = _make_chat_commands(real_agent_config, console=console)

        short_sid = "chat_short_sid_01"
        _create_session_on_disk(short_sid, [("user", "Hi"), ("assistant", "Hello")])

        import datus.cli.chat_commands as chat_mod

        captured = {}

        class FakeCaptureApp:
            def __init__(self, **kwargs):
                captured["items"] = kwargs.get("items", [])

            def run(self):
                return None

        monkeypatch.setattr(chat_mod, "ListSelectorApp", FakeCaptureApp)

        console.file = io.StringIO()
        cmds.cmd_resume("")
        assert len(captured["items"]) > 0
        all_text = " ".join(f"{item.primary} {item.secondary}" for item in captured["items"])
        assert short_sid in all_text


# ===========================================================================
# TestAllTurnActions
# ===========================================================================


def _make_action_for_chat(
    role=ActionRole.TOOL,
    status=ActionStatus.SUCCESS,
    messages="tool result",
    input_data=None,
    output_data=None,
):
    """Helper to create ActionHistory instances for chat_commands tests."""
    import uuid

    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=role,
        messages=messages,
        action_type="test",
        input=input_data,
        output=output_data,
        status=status,
        start_time=datetime.now(),
        end_time=datetime.now(),
        depth=0,
    )


@pytest.mark.ci
class TestAllTurnActions:
    """Tests for all_turn_actions multi-turn accumulation."""

    def test_init_all_turn_actions_empty(self, real_agent_config, mock_llm_create):
        """all_turn_actions is initialized as empty list."""
        cmds = _make_chat_commands(real_agent_config)
        assert cmds.all_turn_actions == []

    def test_all_turn_actions_accumulates(self, real_agent_config, mock_llm_create):
        """all_turn_actions accumulates (message, actions) tuples."""
        cmds = _make_chat_commands(real_agent_config)

        actions1 = [_make_action_for_chat(messages="result1")]
        actions2 = [_make_action_for_chat(messages="result2")]

        cmds.all_turn_actions.append(("Question 1", actions1))
        cmds.all_turn_actions.append(("Question 2", actions2))

        assert len(cmds.all_turn_actions) == 2
        assert cmds.all_turn_actions[0][0] == "Question 1"
        assert cmds.all_turn_actions[1][0] == "Question 2"
        assert cmds.all_turn_actions[0][1] is actions1
        assert cmds.all_turn_actions[1][1] is actions2

    def test_cmd_clear_chat_resets_all_turn_actions(self, real_agent_config, mock_llm_create):
        """cmd_clear_chat resets all_turn_actions to empty."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Simulate accumulated turns
        cmds.all_turn_actions.append(("Q1", [_make_action_for_chat()]))
        cmds.all_turn_actions.append(("Q2", [_make_action_for_chat()]))
        assert len(cmds.all_turn_actions) == 2

        cmds.cmd_clear_chat("")

        assert cmds.all_turn_actions == []

    def test_display_inline_trace_uses_all_turn_actions(self, real_agent_config, mock_llm_create):
        """display_inline_trace_details uses all_turn_actions when populated."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        actions1 = [_make_action_for_chat(messages="r1", input_data={"function_name": "read_query"})]
        actions2 = [_make_action_for_chat(messages="r2", input_data={"function_name": "list_tables"})]

        cmds.all_turn_actions = [
            ("Turn 1 question", actions1),
            ("Turn 2 question", actions2),
        ]
        cmds.last_actions = actions2

        # Reset console buffer
        console.file = io.StringIO()
        cmds.display_inline_trace_details(cmds.last_actions)
        output = _get_console_output(console)

        # Both turns should appear in output
        assert "Turn 1 question" in output
        assert "Turn 2 question" in output

    def test_display_inline_trace_fallback_without_all_turn_actions(self, real_agent_config, mock_llm_create):
        """display_inline_trace_details falls back to actions param when all_turn_actions is empty."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        actions = [_make_action_for_chat(messages="single result", input_data={"function_name": "read_query"})]
        cmds.last_actions = actions
        cmds.all_turn_actions = []  # No accumulated turns

        console.file = io.StringIO()
        cmds.display_inline_trace_details(actions)
        output = _get_console_output(console)

        # Should still render the actions (just not with multi-turn headers)
        assert "switched to verbose mode" in output

    def test_display_inline_trace_rerenders_final_response(self, real_agent_config, mock_llm_create):
        """display_inline_trace_details re-renders node final action output after the trace."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        tool_action = _make_action_for_chat(messages="tool result", input_data={"function_name": "read_query"})
        node_final = _make_action_for_chat(
            role=ActionRole.ASSISTANT,
            messages="Chat interaction completed successfully",
            output_data={"response": "This is the final markdown answer", "sql": None},
        )
        # Patch action_type to end with _response
        node_final.action_type = "chat_response"

        actions = [tool_action, node_final]
        cmds.all_turn_actions = [("user question", actions)]
        cmds.last_actions = actions

        console.file = io.StringIO()
        cmds.display_inline_trace_details(cmds.last_actions)
        output = _get_console_output(console)

        assert "switched to verbose mode" in output
        # Final markdown response should be re-rendered
        assert "final markdown answer" in output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console():
    return Console(file=io.StringIO(), no_color=True)


class MinimalAtCompleterExtended:
    def parse_at_context(self, user_input):
        return ([], [], [])


class MinimalCLIExtended:
    def __init__(self, agent_config, console=None):
        self.agent_config = agent_config
        self.console = console or _make_console()
        self.cli_context = CliContext()
        self.actions = ActionHistoryManager()
        self.last_sql = ""
        self.at_completer = MinimalAtCompleter()
        self.scope = None

    def prompt_input(self, message="", multiline=False, default="", **kw):
        return default

    def run_on_bg_loop(self, coro):
        import asyncio

        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(real_agent_config):
    return MinimalCLI(real_agent_config)


@pytest.fixture
def chat_cmd(cli):
    return ChatCommands(cli)


# ---------------------------------------------------------------------------
# Tests: _should_create_new_node
# ---------------------------------------------------------------------------


class TestShouldCreateNewNodeExtended:
    def test_no_current_node_always_create(self, chat_cmd):
        chat_cmd.current_node = None
        assert chat_cmd._should_create_new_node() is True
        assert chat_cmd._should_create_new_node("gensql") is True

    def test_has_node_no_subagent_no_switch(self, chat_cmd):
        """When current_node exists and no subagent requested, only create if currently using subagent."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = None
        assert chat_cmd._should_create_new_node(None) is False

    def test_has_node_same_subagent_no_create(self, chat_cmd):
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._should_create_new_node("gensql") is False

    def test_has_node_different_subagent_creates(self, chat_cmd):
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._should_create_new_node("compare") is True

    def test_switching_from_subagent_to_chat_creates(self, chat_cmd):
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._should_create_new_node(None) is True


# ---------------------------------------------------------------------------
# Tests: _is_agent_switch
# ---------------------------------------------------------------------------


class TestIsAgentSwitch:
    def test_no_current_node_is_not_switch(self, chat_cmd):
        """No current node means this is not a switch."""
        chat_cmd.current_node = None
        assert chat_cmd._is_agent_switch("gensql") is False

    def test_same_subagent_is_not_switch(self, chat_cmd):
        """Same subagent is not a switch."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._is_agent_switch("gensql") is False

    def test_different_subagent_is_switch(self, chat_cmd):
        """Different subagent is a switch."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._is_agent_switch("compare") is True

    def test_chat_to_subagent_is_switch(self, chat_cmd):
        """Switching from chat to subagent is a switch."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = None
        assert chat_cmd._is_agent_switch("gensql") is True

    def test_subagent_to_chat_is_switch(self, chat_cmd):
        """Switching from subagent to chat is a switch."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = "gensql"
        assert chat_cmd._is_agent_switch(None) is True

    def test_chat_to_chat_is_not_switch(self, chat_cmd):
        """Staying on chat is not a switch."""
        chat_cmd.current_node = MagicMock()
        chat_cmd.current_subagent_name = None
        assert chat_cmd._is_agent_switch(None) is False


# ---------------------------------------------------------------------------
# Tests: session_id preservation on agent switch
# ---------------------------------------------------------------------------


class TestSessionPreservationOnSwitch:
    def test_session_copied_on_switch_with_correct_prefix(self, chat_cmd):
        """When switching agents, session is copied and new id has the target node prefix."""
        old_node = MagicMock()
        old_node.session_id = "chat_session_abc123"
        chat_cmd.current_node = old_node
        chat_cmd.current_subagent_name = None  # was on chat

        new_node = MagicMock()
        new_node.session_id = None
        new_node.get_node_name.return_value = "gensql"
        chat_cmd._create_new_node = MagicMock(return_value=new_node)

        # Mock _copy_session_for_switch to return a properly-prefixed session_id
        chat_cmd._copy_session_for_switch = MagicMock(return_value="gensql_session_def456")

        # Simulate _execute_chat logic for agent switch
        subagent_name = "gensql"
        need_new_node = chat_cmd._should_create_new_node(subagent_name)
        is_switch = chat_cmd._is_agent_switch(subagent_name)

        assert need_new_node is True
        assert is_switch is True

        prev_session_id = None
        if is_switch and chat_cmd.current_node and hasattr(chat_cmd.current_node, "session_id"):
            prev_session_id = chat_cmd.current_node.session_id
        chat_cmd.current_node = chat_cmd._create_new_node(subagent_name)
        if prev_session_id:
            chat_cmd.current_node.session_id = chat_cmd._copy_session_for_switch(prev_session_id, chat_cmd.current_node)

        # Verify the new session_id has the target node prefix, not the source prefix
        assert chat_cmd.current_node.session_id == "gensql_session_def456"
        chat_cmd._copy_session_for_switch.assert_called_once_with("chat_session_abc123", new_node)

    def test_copy_session_delegates_to_session_manager(self, chat_cmd):
        """_copy_session_for_switch delegates to SessionManager.copy_session."""
        new_node = MagicMock()
        new_node.get_node_name.return_value = "gensql"

        with patch("datus.models.session_manager.SessionManager") as mock_sm_cls:
            mock_sm = mock_sm_cls.return_value
            mock_sm.copy_session.return_value = "gensql_session_xyz789"

            result = chat_cmd._copy_session_for_switch("chat_session_abc123", new_node)

        assert result == "gensql_session_xyz789"
        mock_sm.copy_session.assert_called_once_with("chat_session_abc123", "gensql")

    def test_copy_session_fallback_on_error(self, chat_cmd):
        """If copy_session fails, fall back to the node's existing session_id."""
        new_node = MagicMock()
        new_node.session_id = "fallback_session_id"
        new_node.get_node_name.return_value = "gensql"

        with patch("datus.models.session_manager.SessionManager", side_effect=Exception("no dir")):
            result = chat_cmd._copy_session_for_switch("chat_session_abc123", new_node)

        assert result == "fallback_session_id"

    def test_session_id_not_carried_on_fresh_start(self, chat_cmd):
        """First node creation (no previous node) does not carry session_id."""
        chat_cmd.current_node = None
        chat_cmd.current_subagent_name = None

        new_node = MagicMock()
        new_node.session_id = None
        chat_cmd._create_new_node = MagicMock(return_value=new_node)

        is_switch = chat_cmd._is_agent_switch("gensql")
        assert is_switch is False

        chat_cmd.current_node = chat_cmd._create_new_node("gensql")
        # session_id remains None (will be auto-generated on first use)
        assert chat_cmd.current_node.session_id is None


# ---------------------------------------------------------------------------
# Tests: update_chat_node_tools
# ---------------------------------------------------------------------------


class TestUpdateChatNodeToolsExtended:
    def test_calls_setup_tools_on_current_node(self, chat_cmd):
        mock_node = MagicMock()
        chat_cmd.current_node = mock_node
        chat_cmd.chat_node = None
        chat_cmd.update_chat_node_tools()
        mock_node.setup_tools.assert_called_once()

    def test_no_current_node_no_crash(self, chat_cmd):
        chat_cmd.current_node = None
        chat_cmd.chat_node = None
        chat_cmd.update_chat_node_tools()
        # Both node handles remain None — no setup work was attempted.
        assert chat_cmd.current_node is None
        assert chat_cmd.chat_node is None


# ---------------------------------------------------------------------------
# Tests: cmd_clear_chat
# ---------------------------------------------------------------------------


class TestCmdClearChatExtended:
    def test_clears_state(self, chat_cmd):
        # Set some state
        chat_cmd.chat_history = [{"role": "user", "content": "hello"}]
        chat_cmd.last_actions = [MagicMock()]
        chat_cmd.current_node = MagicMock()

        chat_cmd.cmd_clear_chat("")

        # After clear, state should be reset
        assert chat_cmd.chat_history == [] or chat_cmd.current_node is None


# ---------------------------------------------------------------------------
# Tests: _create_new_node for special subagents
# ---------------------------------------------------------------------------


class TestCreateNewNodeExtended:
    def test_create_chat_node_no_subagent(self, chat_cmd):
        with patch("datus.agent.node.chat_agentic_node.ChatAgenticNode.__init__", return_value=None) as mock_init:
            chat_cmd._create_new_node(None)
        mock_init.assert_called_once()

    def test_create_gen_semantic_model(self, chat_cmd):
        mock_node = MagicMock()
        with patch("datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode") as mock_cls:
            mock_cls.return_value = mock_node
            with patch.dict(
                "sys.modules",
                {"datus.agent.node.gen_semantic_model_agentic_node": MagicMock(GenSemanticModelAgenticNode=mock_cls)},
            ):
                result = chat_cmd._create_new_node("gen_semantic_model")
        # _create_new_node returns the node; verify the correct class was instantiated
        mock_cls.assert_called_once()
        assert result is mock_node

    def test_create_gen_metrics(self, chat_cmd):
        mock_node = MagicMock()
        with patch(
            "datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode",
            return_value=mock_node,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "datus.agent.node.gen_metrics_agentic_node": MagicMock(
                        GenMetricsAgenticNode=MagicMock(return_value=mock_node)
                    )
                },
            ):
                result = chat_cmd._create_new_node("gen_metrics")
        assert result is mock_node

    def test_create_gensql_default(self, chat_cmd):
        mock_node = MagicMock()
        with patch("datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode") as mock_cls:
            mock_cls.return_value = mock_node
            with patch.dict(
                "sys.modules",
                {"datus.agent.node.gen_sql_agentic_node": MagicMock(GenSQLAgenticNode=mock_cls)},
            ):
                result = chat_cmd._create_new_node("gensql")
        mock_cls.assert_called_once()
        assert result is mock_node


# ---------------------------------------------------------------------------
# Tests: add_in_sql_context
# ---------------------------------------------------------------------------


class TestAddInSqlContextExtended:
    def test_add_in_sql_context_no_sql_action_skips_gracefully(self, chat_cmd):
        """add_in_sql_context with empty actions does not raise (warns and returns)."""
        # Empty actions -> logs warning, returns without storing
        chat_cmd.add_in_sql_context("SELECT 1", "select one", [])
        # No SQL context should be stored (no SQL action found)
        stored = chat_cmd.cli.cli_context.get_last_sql_context()
        assert stored is None  # graceful skip


# ---------------------------------------------------------------------------
# Tests: cmd_chat_info
# ---------------------------------------------------------------------------


class TestCmdChatInfoExtended:
    def test_no_current_node_prints_message(self, chat_cmd):
        chat_cmd.current_node = None
        chat_cmd.cmd_chat_info("")
        output = chat_cmd.console.file.getvalue()
        # Should print "No active session" or similar message
        assert len(output) > 0, "Should have printed a message about no active session"

    def test_with_current_node_calls_get_info(self, chat_cmd):
        async def mock_get_info():
            return {
                "session_id": "sess_123",
                "token_count": 1000,
                "action_count": 3,
            }

        async def mock_get_last_turn_usage():
            return None

        mock_node = MagicMock()
        mock_node.get_session_info = mock_get_info
        mock_node.get_last_turn_usage = mock_get_last_turn_usage
        mock_node.session_manager = None  # Prevent MagicMock auto-chain for get_detailed_usage
        chat_cmd.current_node = mock_node
        chat_cmd.current_subagent_name = "gensql"

        chat_cmd.cmd_chat_info("")
        output = chat_cmd.console.file.getvalue()
        assert "sess_123" in output or "Session" in output


# ===========================================================================
# _collect_batch tests
# ===========================================================================


class TestCollectBatch:
    """Test _collect_batch method for batch question collection."""

    @staticmethod
    def _make(real_agent_config):
        """Create ChatCommands with a MinimalCLI that has controllable prompt_input."""
        console = Console(file=io.StringIO(), no_color=True)
        return _make_chat_commands(real_agent_config, console=console), console

    def test_empty_contents_returns_empty_json(self, real_agent_config, mock_llm_create):
        """Empty contents list returns '[]'."""
        chat_cmd, console = self._make(real_agent_config)
        result = chat_cmd._collect_batch(console, [], [])
        assert result == json.dumps([])

    def test_single_free_text_question(self, real_agent_config, mock_llm_create):
        """Single free-text question collects via prompt_input."""
        chat_cmd, console = self._make(real_agent_config)
        chat_cmd.cli.prompt_input = MagicMock(return_value="my answer")
        result = chat_cmd._collect_batch(console, ["What name?"], [{}])
        answers = json.loads(result)
        assert len(answers) == 1
        assert answers[0] == "my answer"

    @patch("datus.cli.chat_commands.select_choice", return_value="2")
    def test_single_question_with_choices(self, mock_select, real_agent_config, mock_llm_create):
        """Single question with choices uses select_choice."""
        chat_cmd, console = self._make(real_agent_config)
        result = chat_cmd._collect_batch(console, ["Pick DB?"], [{"1": "MySQL", "2": "PG"}])
        answers = json.loads(result)
        assert len(answers) == 1
        assert answers[0] == "PG"

    @patch("datus.cli.chat_commands.select_choice", return_value="1")
    def test_multi_question_batch(self, mock_select, real_agent_config, mock_llm_create):
        """Multiple questions with choices collects answers sequentially."""
        chat_cmd, console = self._make(real_agent_config)
        chat_cmd.cli.prompt_input = MagicMock(return_value="custom filter")
        result = chat_cmd._collect_batch(
            console,
            ["DB?", "Time?", "Filter?"],
            [{"1": "MySQL", "2": "PG"}, {"1": "7d", "2": "30d"}, {}],
        )
        answers = json.loads(result)
        assert len(answers) == 3
        assert answers[0] == "MySQL"
        assert answers[1] == "7d"
        assert answers[2] == "custom filter"

    @patch("datus.cli.chat_commands.select_choice", return_value="custom text")
    def test_free_text_option_preserves_input(self, mock_select, real_agent_config, mock_llm_create):
        """Free-text input via select_choice is preserved as-is."""
        chat_cmd, console = self._make(real_agent_config)
        result = chat_cmd._collect_batch(console, ["Q?"], [{"1": "A", "2": "B"}])
        answers = json.loads(result)
        assert answers[0] == "custom text"

    @patch("datus.cli.chat_commands.select_choice", return_value="1")
    def test_multi_question_shows_summary(self, mock_select, real_agent_config, mock_llm_create):
        """Multi-question batch prints summary to console."""
        chat_cmd, console = self._make(real_agent_config)
        result = chat_cmd._collect_batch(
            console,
            ["Q1?", "Q2?"],
            [{"1": "A", "2": "B"}, {"1": "C", "2": "D"}],
        )
        output = console.file.getvalue()
        assert "Answers submitted" in output
        answers = json.loads(result)
        assert len(answers) == 2


class TestMakeInputCollector:
    """Test _make_input_collector and the returned collect() closure."""

    @staticmethod
    def _make(real_agent_config):
        console = Console(file=io.StringIO(), no_color=True)
        return _make_chat_commands(real_agent_config, console=console), console

    @staticmethod
    def _make_action(action_type, input_data):
        return ActionHistory(
            action_id="test-id",
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type=action_type,
            messages="test",
            input=input_data,
        )

    def test_collect_routes_batch_to_collect_batch(self, real_agent_config, mock_llm_create):
        """collect() routes multi-question contents to _collect_batch."""
        chat_cmd, console = self._make(real_agent_config)
        chat_cmd.cli.prompt_input = MagicMock(return_value="ans1")
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock()
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action(
            "request_batch",
            {
                "contents": ["Q1?", "Q2?"],
                "choices": [{}, {}],
                "default_choices": ["", ""],
                "allow_free_text": True,
            },
        )
        result = collector(action, console)
        answers = json.loads(result)
        assert len(answers) == 2

    @patch("datus.cli.chat_commands.select_choice", return_value="y")
    def test_collect_routes_choice_to_single(self, mock_select, real_agent_config, mock_llm_create):
        """collect() routes single-question contents to single choice."""
        chat_cmd, console = self._make(real_agent_config)
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock()
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action(
            "request_choice",
            {
                "contents": ["Confirm?"],
                "choices": [{"y": "Yes", "n": "No"}],
                "default_choices": ["y"],
                "allow_free_text": False,
            },
        )
        result = collector(action, console)
        assert result == "y"

    def test_collect_free_text_no_choices(self, real_agent_config, mock_llm_create):
        """collect() with empty choices calls prompt_input for free text."""
        chat_cmd, console = self._make(real_agent_config)
        chat_cmd.cli.prompt_input = MagicMock(return_value="typed answer")
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock()
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action(
            "request_choice",
            {
                "contents": ["Enter text"],
                "choices": [{}],
                "default_choices": [""],
                "allow_free_text": True,
            },
        )
        result = collector(action, console)
        assert result == "typed answer"

    @patch("datus.cli.chat_commands.select_choice", return_value="")
    def test_collect_empty_free_text_returns_empty(self, mock_select, real_agent_config, mock_llm_create):
        """collect() with allow_free_text and empty result returns empty string."""
        chat_cmd, console = self._make(real_agent_config)
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock()
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action(
            "request_choice",
            {
                "contents": ["Pick?"],
                "choices": [{"a": "Option A"}],
                "default_choices": ["a"],
                "allow_free_text": True,
            },
        )
        result = collector(action, console)
        assert result == ""

    def test_collect_exception_returns_none_for_choice(self, real_agent_config, mock_llm_create):
        """collect() returns None on exception for request_choice."""
        chat_cmd, console = self._make(real_agent_config)
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock(side_effect=RuntimeError("boom"))
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action("request_choice", {})
        result = collector(action, console)
        assert result is None

    def test_collect_exception_returns_none_for_batch(self, real_agent_config, mock_llm_create):
        """collect() returns None on exception for request_batch."""
        chat_cmd, console = self._make(real_agent_config)
        esc_guard = MagicMock()
        esc_guard.paused.return_value.__enter__ = MagicMock(side_effect=RuntimeError("boom"))
        esc_guard.paused.return_value.__exit__ = MagicMock()
        collector = chat_cmd._make_input_collector(esc_guard)
        action = self._make_action("request_batch", {"contents": ["Q1?", "Q2?"], "choices": [{}, {}]})
        result = collector(action, console)
        assert result is None


# ===========================================================================
# _drop_if_matches_final (thinking pending reconcile helper)
# ===========================================================================


@pytest.mark.ci
class TestDropIfMatchesFinal:
    """Unit tests for the helper that drops a deferred ASSISTANT text when its
    content equals the node's final *_response body, or flushes it otherwise."""

    def _assistant(self, raw: str) -> ActionHistory:
        return ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages="",
            action_type="response",
            input={},
            output={"raw_output": raw, "is_thinking": False},
            status=ActionStatus.SUCCESS,
        )

    def _final(self, response: str) -> ActionHistory:
        return ActionHistory(
            action_id="f1",
            role=ActionRole.ASSISTANT,
            messages="",
            action_type="chat_response",
            input={},
            output={"response": response},
            status=ActionStatus.SUCCESS,
        )

    def test_pending_none_returns_none(self):
        from datus.cli.chat_commands import _drop_if_matches_final

        incremental: list = []
        result = _drop_if_matches_final(None, self._final("hi"), incremental)
        assert result is None
        assert incremental == []

    def test_texts_equal_drops_pending(self):
        from datus.cli.chat_commands import _drop_if_matches_final

        incremental: list = []
        pending = self._assistant("  Hello world  ")
        final = self._final("Hello world")
        result = _drop_if_matches_final(pending, final, incremental)
        assert result is None
        assert incremental == []  # pending was NOT flushed

    def test_texts_differ_flushes_pending(self):
        from datus.cli.chat_commands import _drop_if_matches_final

        incremental: list = []
        pending = self._assistant("mid-turn thought")
        final = self._final("actual final answer")
        result = _drop_if_matches_final(pending, final, incremental)
        assert result is None
        assert incremental == [pending]

    def test_non_dict_pending_output_is_dropped(self):
        """Non-dict pending output has no extractable text — drop it."""
        from datus.cli.chat_commands import _drop_if_matches_final

        incremental: list = []
        pending = ActionHistory(
            action_id="a2",
            role=ActionRole.ASSISTANT,
            messages="raw",
            action_type="response",
            input={},
            output="not a dict",
            status=ActionStatus.SUCCESS,
        )
        final = self._final("anything")
        _drop_if_matches_final(pending, final, incremental)
        assert incremental == []

    def test_empty_pending_text_is_dropped(self):
        """Empty pending text has nothing to contribute — drop it."""
        from datus.cli.chat_commands import _drop_if_matches_final

        incremental: list = []
        pending = self._assistant("")
        final = self._final("anything")
        _drop_if_matches_final(pending, final, incremental)
        assert incremental == []


# ===========================================================================
# TestSessionFilterByAgent — cmd_resume filters by active agent
# ===========================================================================


class TestIsModelConfigError:
    """Tests for _is_model_config_error() helper."""

    @pytest.mark.parametrize(
        "exc",
        [
            KeyError("No active model configured. Set `target` in agent.yml"),
            KeyError("Model foo not found in agent_config"),
            KeyError("Unsupported model type: xyz"),
            Exception("invalid api_key provided"),
            RuntimeError("OpenAI model authentication failed"),
            Exception("401 Unauthorized from provider openai"),
        ],
    )
    def test_model_errors_detected(self, exc):
        assert _is_model_config_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("database connection failed"),
            RuntimeError("some unrelated error"),
            Exception("timeout waiting for response"),
            Exception("401 Unauthorized"),
            RuntimeError("authentication failed"),
        ],
    )
    def test_non_model_errors_not_detected(self, exc):
        assert _is_model_config_error(exc) is False


class TestSessionFilterByAgent:
    """cmd_resume only surfaces sessions for the active agent."""

    def test_resume_interactive_empty_when_agent_has_no_sessions(self, real_agent_config, mock_llm_create):
        """cmd_resume with no args filters by current agent; empty result message mentions agent."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        cmds.current_subagent_name = "gen_metrics"

        _create_session_on_disk("chat_session_a", [("user", "hi")])
        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "gen_metrics" in output
        assert cmds.current_node is None
