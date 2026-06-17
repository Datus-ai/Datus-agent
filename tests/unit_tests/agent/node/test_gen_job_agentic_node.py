# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenJobAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool + execute_ddl + execute_write + transfer_query_result
  + get_migration_capabilities + suggest_table_layout + validate_ddl)
- Max turns configuration
- Node type registration and factory creation

Since migration subagent was merged into gen_job, this node now covers both
single-database ETL and cross-database migration. Tests reflect both paths.

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite)
- Real Tools (DBFuncTool)
- Real PromptManager (using built-in templates)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.agent.node._builtin_node_test_helpers import (
    check_dynamic_db_func_tool,
    check_execute_stream_basic_workflow,
    check_execute_stream_error_handling,
    check_execute_stream_raises_without_input,
    check_filesystem_tools,
    check_inherits_agentic_node,
    check_max_turns,
    check_node_factory,
    check_node_factory_with_input,
    check_node_id,
    check_node_name,
    check_node_type_constant,
    check_node_type_in_action_types,
    check_standard_db_tools,
    check_tools_include,
)
from tests.unit_tests.mock_llm_model import MockToolCall, build_tool_then_response

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenJobAgenticNodeInit:
    """Tests for GenJobAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_node_name(GenJobAgenticNode, real_agent_config, "gen_job")

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_inherits_agentic_node(GenJobAgenticNode, real_agent_config)

    def test_node_id(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_node_id(GenJobAgenticNode, real_agent_config, "gen_job_node")

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_include(GenJobAgenticNode, real_agent_config, "execute_ddl")

    def test_setup_tools_includes_execute_write(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_include(GenJobAgenticNode, real_agent_config, "execute_write")

    def test_setup_tools_includes_transfer_query_result(self, real_agent_config, mock_llm_create):  # audit-noqa
        """gen_job absorbed migration — transfer_query_result is required for cross-DB flows."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_include(GenJobAgenticNode, real_agent_config, "transfer_query_result")

    def test_setup_tools_registers_three_migration_target_wrappers(self, real_agent_config, mock_llm_create):
        """gen_job absorbed migration — all three Mixin wrappers must be wired
        as tools so the LLM can read dialect hints, pick layout, and validate
        DDL end to end on the same turn.
        """
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = {tool.name for tool in node.tools}
        assert {"get_migration_capabilities", "suggest_table_layout", "validate_ddl"} <= tool_names

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_standard_db_tools(GenJobAgenticNode, real_agent_config)

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_filesystem_tools(GenJobAgenticNode, real_agent_config)

    def test_default_max_turns(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_max_turns(GenJobAgenticNode, real_agent_config, 50)

    def test_uses_dynamic_db_func_tool(self, real_agent_config, mock_llm_create):  # audit-noqa
        """gen_job should use create_dynamic for multi-connector support."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_dynamic_db_func_tool(GenJobAgenticNode, real_agent_config)

    def test_tool_registry_keeps_db_write_helpers_under_db_tools(self, real_agent_config, mock_llm_create):
        """``execute_write`` / ``execute_ddl`` / ``transfer_query_result`` must
        stay in ``db_tools`` so profile ASK rules fire. They are mounted as
        method-level wrappers, so the registry must cover the full
        ``DBFuncTool.all_tools_name()`` surface, not just ``available_tools()``."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry.get("execute_ddl") == "db_tools"
        assert registry.get("execute_write") == "db_tools"
        assert registry.get("transfer_query_result") == "db_tools"

    def test_system_prompt_requires_explicit_authorization_for_replacement(self, real_agent_config, mock_llm_create):
        """gen_job inherits gen-table behavior and must not overwrite in workflow mode by default."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        context = node._prepare_template_context(SemanticNodeInput(user_message="Build ETL table"))
        prompt = node._get_system_prompt(context)

        assert "interactive vs workflow authorization rules" in prompt
        assert "replace existing tables only when explicitly authorized" in prompt
        assert "Use `mode='replace'` only for a new target table" in prompt


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.llm_harness
class TestGenJobExecution:
    """Test execute_stream error paths and basic workflow."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_raises_without_input(GenJobAgenticNode, real_agent_config)

    @pytest.mark.asyncio
    async def test_execute_stream_basic_workflow(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_basic_workflow(
            GenJobAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Create an ETL job to load data into summary table",
        )

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_error_handling(
            GenJobAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Build ETL job",
        )


@pytest.mark.acceptance
@pytest.mark.llm_harness
class TestGenJobWritePathAcceptance:
    """Deterministic coverage for the gen_job DDL + DML path."""

    @pytest.mark.asyncio
    async def test_gen_job_creates_table_inserts_rows_and_reads_back(self, mutable_real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        create_sql = "CREATE TABLE job_school_summary (school_count INTEGER)"
        insert_sql = "INSERT INTO job_school_summary (school_count) SELECT COUNT(*) FROM schools"
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall("execute_ddl", {"sql": create_sql}),
                        MockToolCall(
                            "execute_write",
                            {
                                "sql": insert_sql,
                                "min_rows": 1,
                                "max_rows": 1,
                            },
                        ),
                    ],
                    content=json.dumps(
                        {
                            "status": "created",
                            "target_table": "job_school_summary",
                        }
                    ),
                )
            ]
        )

        node = GenJobAgenticNode(agent_config=mutable_real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Create a summary job table with the number of schools.")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        executed_tools = [item["tool"] for item in mock_llm_create.tool_results if item["executed"]]
        assert executed_tools == ["execute_ddl", "execute_write"]
        query_result = node.db_func_tool.read_query("SELECT school_count FROM job_school_summary")
        assert query_result.success == 1
        assert "school_count" in str(query_result.result)
        assert actions[-1].status == ActionStatus.SUCCESS
        assert "job_school_summary" in str(actions[-1].output)


# ---------------------------------------------------------------------------
# Node Type Integration Tests
# ---------------------------------------------------------------------------


class TestGenJobNodeType:
    """Tests for GenJobAgenticNode type registration."""

    def test_node_type_constant_exists(self):  # audit-noqa
        check_node_type_constant("TYPE_GEN_JOB", "gen_job")

    def test_node_type_in_action_types(self):  # audit-noqa
        check_node_type_in_action_types("TYPE_GEN_JOB")

    def test_node_factory_creates_gen_job(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory(GenJobAgenticNode, NodeType.TYPE_GEN_JOB, real_agent_config)

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):  # audit-noqa
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory_with_input(
            GenJobAgenticNode,
            NodeType.TYPE_GEN_JOB,
            real_agent_config,
            "Build a summary table",
        )
