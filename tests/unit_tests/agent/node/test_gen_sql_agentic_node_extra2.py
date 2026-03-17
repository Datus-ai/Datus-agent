# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for GenSQLAgenticNode (part 2) covering more uncovered methods.

CI-level: zero external deps, zero network, zero API keys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, node_name="gensql"):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

    return GenSQLAgenticNode(
        node_id="gensql_extra2",
        description="Extra gensql test 2",
        node_type=NodeType.TYPE_GENSQL,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestSetupToolPattern
# ---------------------------------------------------------------------------


class TestSetupToolPatternGenSQL:
    def test_wildcard_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_date_parsing_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_date_parsing_tools") as mock_setup:
            node._setup_tool_pattern("date_parsing_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_filesystem_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_filesystem_tools") as mock_setup:
            node._setup_tool_pattern("filesystem_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_platform_doc_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_platform_doc_tools") as mock_setup:
            node._setup_tool_pattern("platform_doc_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_unknown_type_logs_warning(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Should not raise
        node._setup_tool_pattern("unknown_tool_type.*")

    def test_exact_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools")
        mock_setup.assert_called_once()

    def test_exact_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools")
        mock_setup.assert_called_once()

    def test_exact_date_parsing_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_date_parsing_tools") as mock_setup:
            node._setup_tool_pattern("date_parsing_tools")
        mock_setup.assert_called_once()

    def test_exact_filesystem_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_filesystem_tools") as mock_setup:
            node._setup_tool_pattern("filesystem_tools")
        mock_setup.assert_called_once()

    def test_exact_platform_doc_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_platform_doc_tools") as mock_setup:
            node._setup_tool_pattern("platform_doc_tools")
        mock_setup.assert_called_once()

    def test_specific_method_pattern(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_specific_tool_method") as mock_method:
            node._setup_tool_pattern("db_tools.list_tables")
        mock_method.assert_called_once_with("db_tools", "list_tables")

    def test_unknown_pattern_no_raise(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Should not raise
        node._setup_tool_pattern("some_random_pattern")

    def test_exception_in_setup_is_caught(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools", side_effect=RuntimeError("db error")):
            # Should not raise - exception is caught internally
            node._setup_tool_pattern("db_tools.*")


# ---------------------------------------------------------------------------
# TestExtractSqlAndOutputFromResponse
# ---------------------------------------------------------------------------


class TestExtractSqlAndOutputFromResponse:
    def test_extracts_sql_from_json_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "sql": "SELECT id FROM users",
                "explanation": "Returns all user IDs",
                "tables": ["users"],
            }
        )
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT id FROM users"
        assert "Returns all user IDs" in output
        assert "users" in output

    def test_returns_none_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": ""})
        assert sql is None
        assert output is None

    def test_returns_none_on_non_string_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": {"nested": "dict"}})
        assert sql is None
        assert output is None

    def test_returns_none_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": "not json at all"})
        assert sql is None

    def test_sql_only_no_explanation(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps({"sql": "SELECT 1", "output": "Query result"})
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT 1"
        assert output == "Query result"

    def test_unescape_newlines_in_output(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "sql": "SELECT 1",
                "explanation": "line1\\nline2",
                "tables": [],
            }
        )
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert "\n" in output  # \\n should be unescaped to \n


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextGenSQL:
    def test_returns_failure_when_no_result(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.result = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is False

    def test_updates_context_with_sql(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_result = MagicMock()
        mock_result.sql = "SELECT id FROM users"
        mock_result.response = ""
        node.result = mock_result

        wf = MagicMock()
        wf.context.sql_contexts = []

        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1

    def test_updates_context_without_sql(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_result = MagicMock(spec=[])  # no 'sql' attribute
        node.result = mock_result

        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is True

    def test_update_context_exception_returns_failure(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_result = MagicMock()
        mock_result.sql = "SELECT 1"
        mock_result.response = ""
        node.result = mock_result

        wf = MagicMock()
        wf.context.sql_contexts.append = MagicMock(side_effect=RuntimeError("append error"))

        result = node.update_context(wf)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestGetExecutionConfig
# ---------------------------------------------------------------------------


class TestGetExecutionConfig:
    def test_normal_mode_returns_tools_and_instruction(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.tools = [MagicMock()]
        user_input = GenSQLNodeInput(user_message="query")

        with patch.object(node, "_get_system_instruction", return_value="system instruction"):
            config = node._get_execution_config("normal", user_input)

        assert config["tools"] == node.tools
        assert config["instruction"] == "system instruction"
        assert config["hooks"] is None

    def test_plan_mode_returns_combined_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        base_tool = MagicMock()
        plan_tool = MagicMock()
        node.tools = [base_tool]
        node.plan_hooks = MagicMock()
        node.plan_hooks.get_plan_tools.return_value = [plan_tool]
        user_input = GenSQLNodeInput(user_message="query", plan_mode=True)

        with patch.object(node, "_get_system_instruction", return_value="base instruction"):
            config = node._get_execution_config("plan", user_input)

        assert base_tool in config["tools"]
        assert plan_tool in config["tools"]
        assert config["hooks"] == node.plan_hooks

    def test_unknown_mode_raises(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenSQLNodeInput(user_message="query")
        with pytest.raises(ValueError, match="Unknown execution mode"):
            node._get_execution_config("invalid_mode", user_input)


# ---------------------------------------------------------------------------
# TestSetupMcpServersExtra
# ---------------------------------------------------------------------------


class TestSetupMcpServersExtra:
    def test_mcp_server_from_config_called_for_unknown_server(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "my_custom_mcp"}

        mock_server = MagicMock()
        with patch.object(node, "_setup_mcp_server_from_config", return_value=mock_server) as mock_from_config:
            result = node._setup_mcp_servers()

        mock_from_config.assert_called_once_with("my_custom_mcp")
        assert "my_custom_mcp" in result

    def test_none_server_not_added(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "missing_server"}

        with patch.object(node, "_setup_mcp_server_from_config", return_value=None):
            result = node._setup_mcp_servers()

        assert "missing_server" not in result

    def test_exception_in_server_setup_is_caught(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "bad_server"}

        with patch.object(node, "_setup_mcp_server_from_config", side_effect=RuntimeError("mcp error")):
            # Should not raise
            result = node._setup_mcp_servers()

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestExecuteStreamGenSQL (error path)
# ---------------------------------------------------------------------------


class TestExecuteStreamGenSQLError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.action_history import ActionStatus

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenSQLAgenticNode(
            node_id="gensql_error",
            description="Error test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )
        node.input = GenSQLNodeInput(user_message="Find total sales")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"
