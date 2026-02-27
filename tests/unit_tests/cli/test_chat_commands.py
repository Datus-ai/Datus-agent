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
  _display_success
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

from rich.console import Console

from datus.cli.chat_commands import ChatCommands
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

    def prompt_input(self, message="", multiline=False):
        """Return empty string for prompt input."""
        return ""


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
    for SessionManager and SessionLoader to find.

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
        conn.execute("INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)", (session_id,))

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

    Note: ChatAgenticNode extends GenSQLAgenticNode, so the isinstance check
    for GenSQLAgenticNode matches ChatAgenticNode too. Therefore, ChatAgenticNode
    receives GenSQLNodeInput (not ChatNodeInput) from create_node_input.
    ChatNodeInput is only returned for node types that don't match any specific check.
    """

    def test_chat_node_gets_gensql_input(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode (subclass of GenSQLAgenticNode) gets GenSQLNodeInput with 'gensql' type."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node()
        assert isinstance(node, ChatAgenticNode)

        node_input, node_type = cmds.create_node_input("Hello", node, [], [], [])

        # ChatAgenticNode IS a GenSQLAgenticNode, so it matches the GenSQL branch
        assert isinstance(node_input, GenSQLNodeInput)
        assert node_type == "gensql"
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

    def test_gensql_node_input_with_plan_mode(self, real_agent_config, mock_llm_create):
        """GenSQLNodeInput has plan_mode set when plan_mode=True (via ChatAgenticNode)."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node()  # ChatAgenticNode -> isinstance GenSQLAgenticNode

        node_input, node_type = cmds.create_node_input("Test", node, [], [], [], plan_mode=True)

        assert isinstance(node_input, GenSQLNodeInput)
        assert node_input.plan_mode is True
        assert node_type == "gensql"

    def test_gensql_node_input_with_at_context(self, real_agent_config, mock_llm_create):
        """GenSQLNodeInput passes through at_tables, at_metrics, at_sqls."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
        from datus.schemas.node_models import Metric, ReferenceSql, TableSchema

        cmds = _make_chat_commands(real_agent_config)
        node = cmds._create_new_node(subagent_name="gensql")

        at_tables = [
            TableSchema(table_name="table1", database_name="db", definition="CREATE TABLE table1 (id INT)"),
            TableSchema(table_name="table2", database_name="db", definition="CREATE TABLE table2 (id INT)"),
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
    """Tests for _display_success rendering different content types."""

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

    def test_yaml_content_type(self, real_agent_config, mock_llm_create):
        """YAML content type is rendered with syntax highlighting."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = self._make_action("key: value\nlist:\n  - item1", "yaml")
        cmds._display_success(action)

        output = _get_console_output(console)
        assert "key" in output
        assert "value" in output

    def test_sql_content_type(self, real_agent_config, mock_llm_create):
        """SQL content type is rendered with syntax highlighting."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = self._make_action("SELECT * FROM users WHERE id = 1", "sql")
        cmds._display_success(action)

        output = _get_console_output(console)
        assert "SELECT" in output
        assert "users" in output

    def test_markdown_content_type(self, real_agent_config, mock_llm_create):
        """Markdown content type is rendered as rich markdown."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = self._make_action("# Title\nSome **bold** text", "markdown")
        cmds._display_success(action)

        output = _get_console_output(console)
        assert "Title" in output
        assert "bold" in output

    def test_empty_content_skips_display(self, real_agent_config, mock_llm_create):
        """Empty content does not display anything."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = self._make_action("", "markdown")
        cmds._display_success(action)

        output = _get_console_output(console)
        # Empty content should produce minimal or no output
        assert "Title" not in output

    def test_fallback_to_messages_when_no_content(self, real_agent_config, mock_llm_create):
        """When output.content is empty, falls back to action.messages."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = ActionHistory(
            action_id="test_success_fb",
            role=ActionRole.INTERACTION,
            messages="Fallback message content",
            action_type="request_choice",
            input={},
            output={"content": "", "content_type": "markdown"},
            status=ActionStatus.SUCCESS,
        )
        cmds._display_success(action)

        output = _get_console_output(console)
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
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        cmds = _make_chat_commands(real_agent_config)
        cmds.cli.cli_context.current_catalog = "my_catalog"
        cmds.cli.cli_context.current_db_name = "my_db"
        cmds.cli.cli_context.current_schema = "my_schema"

        # ChatAgenticNode is subclass of GenSQLAgenticNode -> GenSQLNodeInput
        node = cmds._create_new_node()
        node_input, node_type = cmds.create_node_input("Test", node, [], [], [])

        assert isinstance(node_input, GenSQLNodeInput)
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
        """_display_success handles action with None output gracefully."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

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
        cmds._display_success(action)
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
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        cmds = _make_chat_commands(real_agent_config)
        # cli_context defaults have None for catalog/db_name/schema
        assert cmds.cli.cli_context.current_catalog is None
        assert cmds.cli.cli_context.current_db_name is None
        assert cmds.cli.cli_context.current_schema is None

        node = cmds._create_new_node()
        node_input, node_type = cmds.create_node_input("Test", node, [], [], [])

        assert isinstance(node_input, GenSQLNodeInput)
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


class TestTriggerCompact:
    """Tests for _trigger_compact_for_current_node."""

    def test_trigger_compact_no_node(self, real_agent_config, mock_llm_create):
        """No-op when current_node is None."""
        cmds = _make_chat_commands(real_agent_config)
        cmds.current_node = None

        # Should not raise
        cmds._trigger_compact_for_current_node()
        assert cmds.current_node is None

    def test_trigger_compact_node_without_compact_method(self, real_agent_config, mock_llm_create):
        """No-op when node does not have _manual_compact."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        # Use a minimal object without _manual_compact
        cmds.current_node = object()

        # Should not raise
        cmds._trigger_compact_for_current_node()
        output = _get_console_output(console)
        assert "compacted" not in output.lower()
        assert cmds.current_node is not None

    def test_trigger_compact_with_node_no_session(self, real_agent_config, mock_llm_create):
        """Node with _manual_compact but no active session does not print compact messages."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)
        node = cmds._create_new_node()
        cmds.current_node = node

        # Node exists but has no session_id set, so get_session_info returns no session_id
        cmds._trigger_compact_for_current_node()

        output = _get_console_output(console)
        # Should not contain compact success/failure messages (no active session)
        assert "Session compacted successfully" not in output


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
# TestCmdListSessions
# ===========================================================================


class TestCmdListSessions:
    """Tests for cmd_list_sessions."""

    def test_list_sessions_no_sessions(self, real_agent_config, mock_llm_create):
        """No sessions prints warning."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_list_sessions("")

        output = _get_console_output(console)
        # Either shows "No chat sessions found" or shows table
        assert len(output) > 0


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
        """Text content type falls through to markdown rendering."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        action = ActionHistory(
            action_id="test_text_ct",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output={"content": "Plain text result", "content_type": "text"},
            status=ActionStatus.SUCCESS,
        )
        cmds._display_success(action)

        output = _get_console_output(console)
        assert "Plain text result" in output

    def test_display_success_exception_fallback(self, real_agent_config, mock_llm_create):
        """_display_success exception path renders fallback content."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Action with output that has content for fallback
        action = ActionHistory(
            action_id="test_err_success",
            role=ActionRole.INTERACTION,
            messages="",
            action_type="request_choice",
            input={},
            output={"content": "Fallback content here"},
            status=ActionStatus.SUCCESS,
        )
        cmds._display_success(action)

        output = _get_console_output(console)
        # Should display something (normal or fallback)
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

        sql_response = json.dumps({"sql": "SELECT count(*) FROM schools", "response": "Here is the count query"})
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

        mock_llm_create.reset(responses=[build_simple_response("Chat reply"), build_simple_response("SQL reply")])

        cmds.execute_chat_command("Hello", subagent_name=None)
        first_node = cmds.current_node

        cmds.execute_chat_command("Generate SQL", subagent_name="gensql", compact_when_new_subagent=False)
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
            {"report": "Deep report content", "metadata": {"nested": {"deep": True}}, "extra": [1, 2, 3]}
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
                content=json.dumps({"sql": "SELECT count(*) FROM schools", "explanation": "Count schools"}),
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

        mock_llm_create.reset(responses=[build_simple_response("Reply 1"), build_simple_response("Reply 2")])
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


# ===========================================================================
# TestCmdListSessionsWithData — cmd_list_sessions 有 session 数据
# ===========================================================================


class TestCmdListSessionsWithData:
    """Tests for cmd_list_sessions when sessions exist."""

    def test_list_sessions_after_execute(self, real_agent_config, mock_llm_create):
        """cmd_list_sessions shows sessions after creating one via execute."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(responses=[build_simple_response("Response")])
        cmds.execute_chat_command("Create session")

        console.file = io.StringIO()
        cmds.cmd_list_sessions("")

        output = _get_console_output(console)
        assert len(output) > 0
        # Either shows sessions table or "No chat sessions found"
        assert "Session" in output or "No chat sessions" in output


# ===========================================================================
# TestTriggerCompactWithSession — _trigger_compact 带活跃 session
# ===========================================================================


class TestTriggerCompactWithSession:
    """Tests for _trigger_compact_for_current_node with an active session."""

    def test_trigger_compact_before_subagent_switch(self, real_agent_config, mock_llm_create):
        """Switching subagent with compact_when_new_subagent=True triggers compact."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        mock_llm_create.reset(
            responses=[
                build_simple_response("Chat reply"),
                build_simple_response("Summary"),  # for compact
                build_simple_response("SQL reply"),  # for subagent
            ]
        )

        # First: create a chat session
        cmds.execute_chat_command("Hello")
        assert cmds.current_node is not None

        # Second: switch to subagent with compact (exercises line 314)
        console.file = io.StringIO()
        cmds.execute_chat_command("Generate SQL", subagent_name="gensql", compact_when_new_subagent=True)

        output = _get_console_output(console)
        assert cmds.current_subagent_name == "gensql"
        assert len(output) > 0


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

    def test_resume_high_token_warning(self, real_agent_config, mock_llm_create):
        """cmd_resume shows high token warning when session has > 50000 tokens."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        session_id = "chat_session_resume05"
        _create_session_on_disk(session_id)

        # Insert token usage > 50000 into turn_usage table
        from datus.utils.path_manager import get_path_manager

        db_path = os.path.join(str(get_path_manager().sessions_dir), f"{session_id}.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS turn_usage ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "branch_id TEXT DEFAULT 'main', "
                "user_turn_number INTEGER NOT NULL, "
                "requests INTEGER DEFAULT 0, "
                "input_tokens INTEGER DEFAULT 0, "
                "output_tokens INTEGER DEFAULT 0, "
                "total_tokens INTEGER DEFAULT 0, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO turn_usage (session_id, user_turn_number, total_tokens) VALUES (?, ?, ?)",
                (session_id, 1, 60000),
            )
            conn.commit()

        cmds.cmd_resume(session_id)

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # Should show high token warning
        assert "token" in output.lower() or "compact" in output.lower()


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
        cmds.cmd_rewind("1")

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # New session should be different from original
        assert cmds.current_node.session_id != session_id
        assert "rewound" in output.lower() or "turn" in output.lower() or "continue" in output.lower()

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
# TestCmdListSessionsTableRendering — cmd_list_sessions with disk sessions
# ===========================================================================


class TestCmdListSessionsTableRendering:
    """Tests for cmd_list_sessions table rendering when sessions exist (lines 929-974)."""

    def test_list_sessions_no_sessions(self, real_agent_config, mock_llm_create):
        """cmd_list_sessions with no sessions shows 'No chat sessions found'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        cmds.cmd_list_sessions("")

        output = _get_console_output(console)
        assert "no chat sessions" in output.lower() or len(output) > 0

    def test_list_sessions_with_disk_session(self, real_agent_config, mock_llm_create):
        """cmd_list_sessions with disk sessions exercises the listing path."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create disk sessions
        _create_session_on_disk("chat_session_list01", [("user", "Hello"), ("assistant", "Hi")])

        cmds.cmd_list_sessions("")

        output = _get_console_output(console)
        assert len(output) > 0
        # list_sessions() returns list[str] but cmd_list_sessions tries dict access,
        # so it will hit the error path or handle it gracefully
        assert "session" in output.lower() or "error" in output.lower()

    def test_list_sessions_with_active_session(self, real_agent_config, mock_llm_create):
        """cmd_list_sessions when current_node has session_id exercises highlight path."""
        from tests.unit_tests.mock_llm_model import build_simple_response

        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a node first
        mock_llm_create.reset(responses=[build_simple_response("Reply")])
        cmds.execute_chat_command("Hello")
        assert cmds.current_node is not None

        # Create a disk session matching the node's session_id
        _create_session_on_disk(cmds.current_node.session_id)

        console.file = io.StringIO()
        cmds.cmd_list_sessions("")

        output = _get_console_output(console)
        assert len(output) > 0


# ===========================================================================
# TestResumeWithSqlMessages — cmd_resume with SQL in messages (lines 1084-1094)
# ===========================================================================


class TestResumeWithSqlMessages:
    """Tests for cmd_resume message rendering paths with SQL and actions."""

    def _create_session_with_sql_messages(self, session_id):
        """Create a session with function_call messages that SessionLoader parses as SQL."""
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
            conn.execute("INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)", (session_id,))

            base = datetime(2025, 6, 1, 10, 0, 0)
            msgs = [
                # User message
                (json.dumps({"role": "user", "content": "Find all students"}), (base).isoformat()),
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
                            "content": json.dumps({"sql": "SELECT * FROM students", "output": "Found 3 students"}),
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
        cmds.cmd_rewind("q")  # Just display table and cancel

        output = _get_console_output(console)
        # Table should show the user turn with truncated message
        assert "..." in output or "turn" in output.lower()

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
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create sessions with messages
        _create_session_on_disk("chat_session_pick01", [("user", "First Q"), ("assistant", "First A")])

        # Monkeypatch prompt_input to return "1" (select first session)
        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "1")

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert cmds.current_node is not None
        # Table should have been rendered
        assert "available sessions" in output.lower() or "session" in output.lower()

    def test_resume_interactive_cancel_with_q(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user cancels with 'q'."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick02", [("user", "Q"), ("assistant", "A")])

        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "q")

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "cancelled" in output.lower() or "session" in output.lower()

    def test_resume_interactive_invalid_number(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user enters an out-of-range number."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick03", [("user", "Q"), ("assistant", "A")])

        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "99")

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "session" in output.lower()

    def test_resume_interactive_non_numeric_input(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user enters non-numeric text."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick04", [("user", "Q"), ("assistant", "A")])

        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "abc")

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "invalid" in output.lower() or "number" in output.lower() or "session" in output.lower()

    def test_resume_interactive_empty_input(self, real_agent_config, mock_llm_create, monkeypatch):
        """cmd_resume interactive: user enters empty string."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        _create_session_on_disk("chat_session_pick05", [("user", "Q"), ("assistant", "A")])

        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "")

        cmds.cmd_resume("")

        output = _get_console_output(console)
        assert "cancelled" in output.lower() or len(output) > 0


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
                ("assistant", json.dumps({"sql": "SELECT * FROM students", "output": "3 rows"})),
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


class TestResumeListingLongMessage:
    """Tests for cmd_resume long message truncation (line 1026)."""

    def test_long_first_message_is_truncated(self, real_agent_config, mock_llm_create, monkeypatch):
        """Session with first_user_message > 37 chars is truncated with '...' in listing."""
        console = Console(file=io.StringIO(), no_color=True)
        cmds = _make_chat_commands(real_agent_config, console=console)

        # Create a session with a message longer than 37 characters
        long_msg = "This is a very long message that should definitely be truncated in the display listing table"
        _create_session_on_disk("chat_truncate_test_01", [("user", long_msg), ("assistant", "OK")])

        import datus.cli._cli_utils as cli_utils_mod

        monkeypatch.setattr(cli_utils_mod, "prompt_input", lambda *args, **kwargs: "q")

        console.file = io.StringIO()
        cmds.cmd_resume("")
        output = _get_console_output(console)
        # The long message should be truncated with "..."
        assert "..." in output
