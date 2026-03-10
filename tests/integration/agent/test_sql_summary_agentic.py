# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for SqlSummaryAgenticNode.

Tests the full SQL summary generation workflow with real LLM, real config,
and real tools. Also covers the gen_sql_summary_system_1.1.j2 prompt template.
"""

import pytest

from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestSqlSummaryAgentic:
    """Integration tests for SqlSummaryAgenticNode with real LLM."""

    def test_node_initialization(self, nightly_agent_config):
        """N9-01: Node initializes with correct tools."""
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_sql_summary"
        assert node.execution_mode == "workflow"
        assert node.hooks is None

        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names, f"Missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file, got: {tool_names}"
        assert "list_directory" in tool_names, f"Missing list_directory, got: {tool_names}"
        assert "edit_file" in tool_names, f"Missing edit_file, got: {tool_names}"
        assert "read_multiple_files" in tool_names, f"Missing read_multiple_files, got: {tool_names}"

        # Generation tools
        assert "generate_sql_summary_id" in tool_names, f"Missing generate_sql_summary_id, got: {tool_names}"

        logger.info(f"Node initialized with {len(node.tools)} tools: {tool_names}")

    def test_interactive_mode_has_hooks(self, nightly_agent_config):
        """N9-02: Interactive mode initializes hooks."""
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=nightly_agent_config,
            execution_mode="interactive",
        )

        assert node.hooks is not None, "Interactive mode should have hooks"

    def test_template_context_preparation(self, nightly_agent_config):
        """N9-03: Template context includes tools and subject tree info."""
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this SQL",
            sql_query="SELECT COUNT(*) FROM schools WHERE county = 'Fresno'",
        )

        context = node._prepare_template_context(node.input)

        assert "native_tools" in context, "Context should include native_tools"
        assert "sql_summary_dir" in context, "Context should include sql_summary_dir"
        assert "has_subject_tree" in context, "Context should include has_subject_tree"
        assert "similar_items" in context, "Context should include similar_items"

        logger.info(f"Template context keys: {list(context.keys())}")

    @pytest.mark.asyncio
    async def test_execute_stream_produces_actions(self, nightly_agent_config):
        """N9-04: execute_stream produces valid action sequence with real LLM."""
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SqlSummaryNodeInput(
            user_message="Summarize the following SQL query and classify it.",
            sql_query="SELECT county, COUNT(*) as school_count FROM schools GROUP BY county ORDER BY school_count DESC",
            comment="Count schools by county",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        assert (
            actions[-1].status == ActionStatus.SUCCESS
        ), f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"

    @pytest.mark.asyncio
    async def test_execute_stream_with_complex_sql(self, nightly_agent_config):
        """N9-05: execute_stream handles complex SQL with joins."""
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this complex SQL query.",
            sql_query=(
                "SELECT s.school_name, s.county, sat.avg_score "
                "FROM schools s "
                "JOIN sat_scores sat ON s.cds_code = sat.cds "
                "WHERE s.county = 'Los Angeles' "
                "ORDER BY sat.avg_score DESC "
                "LIMIT 10"
            ),
            comment="Top 10 schools by SAT score in Los Angeles",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS
