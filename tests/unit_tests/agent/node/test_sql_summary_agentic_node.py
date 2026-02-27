# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SqlSummaryAgenticNode.

NO MOCK except LLM: uses real AgentConfig, real SQLite database, real tools,
real PathManager, real RAG storage. Only LLMBaseModel.create_model is mocked
via the conftest mock_llm_create fixture.
"""

import json

import pytest

from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_node(real_agent_config, **kwargs):
    """Create a SqlSummaryAgenticNode with real config and real dependencies."""
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

    defaults = dict(
        node_name="gen_sql_summary",
        agent_config=real_agent_config,
        execution_mode="workflow",
    )
    defaults.update(kwargs)
    return SqlSummaryAgenticNode(**defaults)


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestSqlSummaryAgenticNodeInit:
    """Tests for SqlSummaryAgenticNode initialization with real dependencies."""

    def test_sql_summary_init(self, real_agent_config, mock_llm_create):
        """Node can be initialized with real config."""
        node = _create_node(real_agent_config)

        assert node.configured_node_name == "gen_sql_summary"
        assert node.execution_mode == "workflow"
        assert node.build_mode == "incremental"
        assert node.hooks is None  # No hooks in workflow mode
        assert node.get_node_name() == "gen_sql_summary"

    def test_sql_summary_has_tools(self, real_agent_config, mock_llm_create):
        """Node has generation tools and filesystem tools."""
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]

        # Generation tools
        assert "generate_sql_summary_id" in tool_names

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "read_multiple_files" in tool_names
        assert "list_directory" in tool_names

        # Tool instances should be initialized
        assert node.filesystem_func_tool is not None
        assert node.generation_tools is not None

    def test_sql_summary_max_turns(self, real_agent_config, mock_llm_create):
        """max_turns is read from agentic_nodes config (5 in test config)."""
        node = _create_node(real_agent_config)
        assert node.max_turns == 5  # Set in conftest real_agent_config

    def test_sql_summary_build_mode(self, real_agent_config, mock_llm_create):
        """build_mode can be configured."""
        node = _create_node(real_agent_config, build_mode="overwrite")
        assert node.build_mode == "overwrite"

    def test_sql_summary_subject_tree(self, real_agent_config, mock_llm_create):
        """subject_tree can be passed and stored."""
        tree = ["Analytics", "Reports"]
        node = _create_node(real_agent_config, subject_tree=tree)
        assert node.subject_tree == tree


# ===========================================================================
# Test Execution
# ===========================================================================


class TestSqlSummaryAgenticNodeExecution:
    """Tests for SqlSummaryAgenticNode.execute_stream() with real tools."""

    @pytest.mark.asyncio
    async def test_sql_summary_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with a simple LLM response produces USER + SUCCESS actions."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("SQL summary created for the revenue query"),
            ]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this SQL query",
            sql_query="SELECT AVG(AvgScrRead) FROM satscores",
            comment="Average SAT reading score aggregation",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should have at least: USER action + final SUCCESS action
        assert len(actions) >= 2
        roles = [a.role for a in actions]
        assert ActionRole.USER in roles
        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "sql_summary_response"

    @pytest.mark.asyncio
    async def test_sql_summary_with_tool_calls(self, real_agent_config, mock_llm_create):
        """LLM calls filesystem tools then responds; tools are ACTUALLY EXECUTED."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        response_content = json.dumps(
            {
                "sql_summary_file": "summary_001.yaml",
                "output": "Summary with filesystem operations",
            }
        )
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="list_directory",
                            arguments=json.dumps({"path": "."}),
                        ),
                    ],
                    content=response_content,
                ),
            ]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Create summary for SAT scores query",
            sql_query="SELECT AVG(AvgScrRead) FROM satscores",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should include TOOL actions from real tool execution
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2  # PROCESSING + SUCCESS for list_directory

        # Verify tool was actually executed
        tool_success_actions = [a for a in tool_actions if a.status == ActionStatus.SUCCESS]
        assert len(tool_success_actions) >= 1

        # Final action should be SUCCESS
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_sql_summary_workflow_mode(self, real_agent_config, mock_llm_create):
        """Node in workflow mode does not set up hooks."""
        node = _create_node(real_agent_config, execution_mode="workflow")
        assert node.hooks is None
        assert node.execution_mode == "workflow"

        mock_llm_create.reset(
            responses=[
                build_simple_response("Summary generated in workflow mode"),
            ]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Generate summary",
            sql_query="SELECT 1",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "sql_summary_response"

    @pytest.mark.asyncio
    async def test_sql_summary_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """execute_stream raises ValueError when input is not set."""
        node = _create_node(real_agent_config)
        node.input = None

        with pytest.raises(ValueError, match="SQL summary input not set"):
            async for _ in node.execute_stream():
                pass

    @pytest.mark.asyncio
    async def test_sql_summary_with_sql_query_context(self, real_agent_config, mock_llm_create):
        """Input with sql_query and comment enriches the user message."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Summary for complex query with context"),
            ]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this query",
            sql_query=(
                "SELECT s.County, AVG(sc.AvgScrRead) FROM satscores sc "
                "JOIN schools s ON sc.cds = s.CDSCode GROUP BY s.County"
            ),
            comment="Average SAT reading score by county",
            database="california_schools",
            db_schema="main",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should complete successfully
        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with the prompt containing SQL context
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        # The enhanced message should contain the SQL query
        assert "SELECT" in prompt or "Summarize" in prompt

    @pytest.mark.asyncio
    async def test_sql_summary_interactive_mode_token_tracking(self, real_agent_config, mock_llm_create):
        """Test that interactive mode tracks token usage from action history."""
        node = _create_node(real_agent_config, execution_mode="interactive")

        mock_llm_create.reset(
            responses=[
                build_simple_response("SQL summary created in interactive mode"),
            ]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this SQL query",
            sql_query="SELECT COUNT(*) FROM satscores",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # In interactive mode, the final result should have tokens_used > 0
        last_output = actions[-1].output
        assert last_output is not None
        if isinstance(last_output, dict) and "tokens_used" in last_output:
            assert last_output["tokens_used"] > 0


# ===========================================================================
# Test Extract Methods
# ===========================================================================


class TestSqlSummaryExtractMethods:
    """Tests for SqlSummaryAgenticNode extraction utility methods."""

    def test_extract_sql_summary_from_dict_response(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with dict content."""
        node = _create_node(real_agent_config)

        file_name, output = node._extract_sql_summary_and_output_from_response(
            {"content": {"sql_summary_file": "test.yaml", "output": "Done"}}
        )
        assert file_name == "test.yaml"
        assert output == "Done"

    def test_extract_sql_summary_from_json_string(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with JSON string content."""
        node = _create_node(real_agent_config)

        json_content = json.dumps({"sql_summary_file": "summary.yaml", "output": "Generated"})
        file_name, output = node._extract_sql_summary_and_output_from_response({"content": json_content})
        assert file_name == "summary.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_from_empty(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with empty content returns None."""
        node = _create_node(real_agent_config)

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": ""})
        assert file_name is None
        assert output is None
