# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for CompareAgenticNode.

NO MOCK except LLM: uses real AgentConfig, real SQLite database, real tools,
real PathManager, real prompt templates. Only LLMBaseModel.create_model is mocked
via the conftest mock_llm_create fixture.
"""

import json

import pytest

from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.node_models import SQLContext, SqlTask
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_compare_input():
    """Create a standard CompareInput for testing."""
    sql_task = SqlTask(
        database_type="sqlite",
        database_name="california_schools",
        task="Find average SAT reading score",
    )
    sql_context = SQLContext(
        sql_query="SELECT AVG(AvgScrRead) FROM satscores",
        explanation="Aggregate average SAT reading score across all schools",
        sql_return="AvgScrRead\n479.699",
        sql_error="",
    )
    return CompareInput(
        sql_task=sql_task,
        sql_context=sql_context,
        expectation="Expected: average SAT reading score from the satscores table",
    )


def _create_compare_node(real_agent_config, **kwargs):
    """Create a CompareAgenticNode with real config and real dependencies."""
    from datus.agent.node.compare_agentic_node import CompareAgenticNode

    defaults = dict(
        node_name="compare",
        agent_config=real_agent_config,
    )
    defaults.update(kwargs)
    return CompareAgenticNode(**defaults)


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestCompareAgenticNodeInit:
    """Tests for CompareAgenticNode initialization with real dependencies."""

    def test_compare_init(self, real_agent_config, mock_llm_create):
        """Node can be initialized with real config."""
        node = _create_compare_node(real_agent_config)

        assert node.configured_node_name == "compare"
        assert node.get_node_name() == "compare"

    def test_compare_has_db_tools(self, real_agent_config, mock_llm_create):
        """Node has real database tools from DBFuncTool."""
        node = _create_compare_node(real_agent_config)

        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "read_query" in tool_names
        assert "describe_table" in tool_names

    def test_compare_max_turns(self, real_agent_config, mock_llm_create):
        """max_turns is read from agentic_nodes config (5 in test config)."""
        node = _create_compare_node(real_agent_config)
        assert node.max_turns == 5  # Set in conftest real_agent_config

    def test_compare_node_name_override(self, real_agent_config, mock_llm_create):
        """Node name can be overridden."""
        node = _create_compare_node(real_agent_config, node_name="compare_v2")
        assert node.get_node_name() == "compare_v2"


# ===========================================================================
# Test Execution
# ===========================================================================


class TestCompareAgenticNodeExecution:
    """Tests for CompareAgenticNode.execute_stream() with real tools."""

    @pytest.mark.asyncio
    async def test_compare_simple_comparison(self, real_agent_config, mock_llm_create):
        """execute_stream with a JSON comparison response."""
        node = _create_compare_node(real_agent_config)

        response_content = json.dumps(
            {
                "explanation": "The SQL correctly aggregates average SAT reading score.",
                "suggest": "No changes needed.",
            }
        )
        mock_llm_create.reset(
            responses=[
                build_simple_response(response_content),
            ]
        )
        node.model = mock_llm_create

        node.input = _create_compare_input()

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Verify action sequence
        assert len(actions) >= 2
        roles = [a.role for a in actions]
        assert ActionRole.USER in roles

        # Final action should be SUCCESS with comparison result
        final = actions[-1]
        assert final.status == ActionStatus.SUCCESS
        assert final.action_type == "compare_sql_response"

        # Verify result data in final action output
        output = final.output
        assert output["success"] is True
        assert "explanation" in output
        assert "suggest" in output

    @pytest.mark.asyncio
    async def test_compare_with_tool_calls(self, real_agent_config, mock_llm_create):
        """LLM calls read_query tool to verify, then responds with comparison."""
        node = _create_compare_node(real_agent_config)

        response_content = json.dumps(
            {
                "explanation": "After checking the table, the SQL is correct.",
                "suggest": "Consider adding an index on AvgScrRead column.",
            }
        )
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="read_query",
                            arguments=json.dumps({"sql": "SELECT COUNT(*) FROM satscores"}),
                        ),
                    ],
                    content=response_content,
                ),
            ]
        )
        node.model = mock_llm_create
        node.input = _create_compare_input()

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should include TOOL actions from real tool execution
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2  # PROCESSING + SUCCESS for the tool call

        # Verify read_query was actually executed against real SQLite
        tool_success_actions = [a for a in tool_actions if a.status == ActionStatus.SUCCESS]
        assert len(tool_success_actions) >= 1

        # Final action should still be SUCCESS
        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "compare_sql_response"

    @pytest.mark.asyncio
    async def test_compare_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Should raise ValueError when input is not set."""
        node = _create_compare_node(real_agent_config)
        node.input = None

        with pytest.raises(ValueError, match="Compare input not set"):
            async for _ in node.execute_stream():
                pass

    @pytest.mark.asyncio
    async def test_compare_wrong_input_type_raises(self, real_agent_config, mock_llm_create):
        """Should raise ValueError when input is not a CompareInput."""
        node = _create_compare_node(real_agent_config)
        # Set input to an invalid type (a plain string)
        node.input = "not a CompareInput"

        with pytest.raises(ValueError, match="Input must be a CompareInput"):
            async for _ in node.execute_stream():
                pass


# ===========================================================================
# Test Static/Utility Methods
# ===========================================================================


class TestCompareStaticMethods:
    """Tests for CompareAgenticNode static methods."""

    def test_parse_comparison_output_dict(self, real_agent_config, mock_llm_create):
        """_parse_comparison_output returns dict as-is."""
        from datus.agent.node.compare_agentic_node import CompareAgenticNode

        result = CompareAgenticNode._parse_comparison_output({"explanation": "Good", "suggest": "None needed"})
        assert result["explanation"] == "Good"
        assert result["suggest"] == "None needed"

    def test_parse_comparison_output_none(self, real_agent_config, mock_llm_create):
        """_parse_comparison_output returns empty dict for None."""
        from datus.agent.node.compare_agentic_node import CompareAgenticNode

        result = CompareAgenticNode._parse_comparison_output(None)
        assert result == {}

    def test_parse_comparison_output_json_string(self, real_agent_config, mock_llm_create):
        """_parse_comparison_output parses valid JSON string with output key."""
        from datus.agent.node.compare_agentic_node import CompareAgenticNode

        # llm_result2json expects an "output" or "sql" key to succeed
        json_str = json.dumps({"explanation": "Match", "suggest": "OK", "output": "done"})
        result = CompareAgenticNode._parse_comparison_output(json_str)
        assert result["explanation"] == "Match"

    def test_prepare_prompt_components(self, real_agent_config, mock_llm_create):
        """_prepare_prompt_components returns system instruction, user prompt, and messages."""
        from datus.agent.node.compare_agentic_node import CompareAgenticNode

        compare_input = _create_compare_input()

        system_instruction, user_prompt, messages = CompareAgenticNode._prepare_prompt_components(compare_input)

        # System instruction should be a non-empty string from the real template
        assert isinstance(system_instruction, str)
        assert len(system_instruction) > 0

        # User prompt should contain task-related text
        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0

        # Messages should have system and user entries
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
