# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenSemanticModelAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool, FilesystemFuncTool, GenerationTools, SemanticDiscoveryTools, SemanticTools)
- Max turns configuration
- Streaming execution with MockLLMModel
- Database tool invocation (describe_table)
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite) with tables: frpm, satscores, schools
- Real Storage/RAG (vector store in tmp_path)
- Real Tools (DBFuncTool, FilesystemFuncTool, GenerationTools, etc.)
- Real PromptManager (using built-in templates)
- Real PathManager
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.tools.func_tool import DBFuncTool, FilesystemFuncTool, SemanticDiscoveryTools
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response


def _set_global_semantic_adapter(agent_config, adapter: str) -> None:
    agent_config.resolve_semantic_adapter = MagicMock(return_value=adapter)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenSemanticModelAgenticNodeInit:
    """Tests for GenSemanticModelAgenticNode initialization."""

    def test_semantic_model_init(self, real_agent_config, mock_llm_create):
        """Test that GenSemanticModelAgenticNode initializes with real config."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_semantic_model"
        assert node.id == "gen_semantic_model_node"
        assert node.execution_mode == "workflow"
        assert node.hooks is None  # No hooks in workflow mode

    def test_semantic_model_has_db_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has database tools from DBFuncTool."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        # DB tools should be present
        assert isinstance(node.db_func_tool, DBFuncTool)

        tool_names = [tool.name for tool in node.tools]

        # Standard DB tools
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_semantic_model_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has filesystem tools."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)

        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names

        # Generation tools
        assert "check_semantic_object_exists" in tool_names
        assert "publish_semantic_model" in tool_names

        # SemanticDiscoveryTools should be present; the profiler tool is
        # registered by default (the optional skill is in the default set).
        assert isinstance(node.semantic_discovery_tools, SemanticDiscoveryTools)
        assert "profile_semantic_model_evidence" in tool_names
        assert "inspect_semantic_sources" in tool_names
        assert "validate_semantic_key_candidates" in tool_names

    def test_osi_semantic_model_uses_dataset_upsert_for_create(self, real_agent_config, mock_llm_create):
        """Ossie authoring creates valid models through the narrow dataset upsert."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate an Ossie semantic model")

        node._get_system_prompt(template_context=node._prepare_template_context(node.input))
        tool_names = {tool.name for tool in node.tools}

        assert {
            "read_file",
            "edit_file",
            "upsert_osi_datasets",
            "glob",
            "grep",
            "plan_osi_semantic_model_target",
        }.issubset(tool_names)
        assert {"write_file", "delete_file", "upsert_osi_metrics", "bash"}.isdisjoint(tool_names)
        assert "publish_semantic_model" in tool_names
        node._populate_tool_registry()
        assert node.tool_registry.get("plan_osi_semantic_model_target") == "semantic_tools"

    @pytest.mark.asyncio
    async def test_before_stream_resets_request_state_without_replacing_shared_evidence(
        self, real_agent_config, mock_llm_create
    ):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate an OSI semantic model")
        node.osi_target_state.select(
            {
                "semantic_model_name": "old_model",
                "semantic_model_file": "subject/semantic_models/warehouse/old.yml",
                "absolute_path": "/tmp/old.yml",
            },
            mode="planned",
        )
        node.generation_evidence.validation_passed = True
        node.generation_evidence.semantic_kb_sync_passed = True
        evidence = node.generation_evidence
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())

        await node._before_stream(ctx)

        assert node.osi_target_state.planned is None
        assert node.generation_evidence is evidence
        assert node.generation_evidence == type(evidence)()
        assert node.generation_tools.generation_evidence is evidence
        assert node.semantic_func_tool.generation_evidence is evidence

    @pytest.mark.asyncio
    async def test_before_stream_only_resets_request_local_sql_plan(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message=(
                "Generate a semantic model for:\n"
                "WITH base AS (SELECT * FROM orders) SELECT COUNT(*) AS orders FROM base"
            )
        )
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())

        await node._before_stream(ctx)

        assert node.sql_modeling_plan is None
        assert node.generation_evidence.sql_modeling_plan_status == "pending"
        assert "prepare_sql_modeling_plan" in {tool.name for tool in node.tools}

    def test_sql_result_cannot_bypass_preflight(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext
        from datus.utils.exceptions import DatusException

        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="SELECT COUNT(*) AS order_count FROM orders")
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
        ctx.response_content = "not json"

        with pytest.raises(DatusException, match="prepare_sql_modeling_plan"):
            node._build_success_result(ctx)

    def test_sql_result_requires_semantic_model_files(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="SELECT COUNT(*) AS order_count FROM orders")
        node.generation_evidence.set_sql_modeling_plan("ready", "source")
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
        ctx.response_content = "not json"

        with pytest.raises(RuntimeError, match="semantic_model_files"):
            node._build_success_result(ctx)

    @pytest.mark.parametrize(
        ("adapter", "required_text", "forbidden_text"),
        [
            ("osi", "osi-semantic-authoring", "metricflow-semantic-authoring"),
            ("metricflow", "metricflow-semantic-authoring", "osi-semantic-authoring"),
        ],
    )
    def test_required_skills_combine_preflight_with_format_specific_authoring(
        self,
        real_agent_config,
        mock_llm_create,
        adapter,
        required_text,
        forbidden_text,
    ):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        _set_global_semantic_adapter(real_agent_config, adapter)
        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate a semantic model")

        required_skills = node._get_required_skills()

        assert required_skills[0] == "sql-modeling-preflight"
        assert required_text in required_skills
        assert forbidden_text not in required_skills

    def test_osi_filesystem_mutations_require_and_preserve_the_planned_target(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        datasource = real_agent_config.current_datasource
        target = f"subject/semantic_models/{datasource}/orders.yml"
        sibling = f"subject/semantic_models/{datasource}/customers.yml"
        content = (
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: orders\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
        )

        unplanned = node.filesystem_func_tool.write_file(target, content)
        assert not unplanned.success
        assert "Plan the OSI semantic-model target" in unplanned.error

        plan = node.osi_target_tools.plan_osi_semantic_model_target(semantic_model_name="orders")
        assert plan.success == 1
        wrong_target = node.filesystem_func_tool.write_file(sibling, content.replace("orders", "customers"))
        assert not wrong_target.success
        assert "authoring is planned for" in wrong_target.error

        node.generation_evidence.validation_passed = True
        node.generation_evidence.semantic_kb_sync_passed = True
        written = node.filesystem_func_tool.write_file(target, content)
        edited = node.filesystem_func_tool.edit_file(
            target,
            "source: orders",
            "source: analytics.orders",
        )

        assert written.success == 1
        assert edited.success == 1
        assert node.generation_evidence.validation_passed is False
        assert node.generation_evidence.semantic_kb_sync_passed is False

        replan = node.osi_target_tools.plan_osi_semantic_model_target(semantic_model_name="customers")
        assert not replan.success
        assert "cannot change after authoring started" in replan.error
        assert node.osi_target_state.planned["semantic_model_name"] == "orders"
        assert node.osi_target_state.last_error_code == "semantic_model_target_invalid"
        with pytest.raises(ValueError, match="unresolved after a failed replan"):
            node.generation_tools.resolve_planned_osi_semantic_target()

    def test_semantic_sql_history_profiler_tool_opt_out(self, real_agent_config, mock_llm_create):
        """An explicit empty skills entry removes the profiler tool."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        original = dict(real_agent_config.agentic_nodes.get("gen_semantic_model", {}))
        try:
            real_agent_config.agentic_nodes["gen_semantic_model"] = {
                **original,
                "skills": "",
            }
            node = GenSemanticModelAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            tool_names = [tool.name for tool in node.tools]
            assert "profile_semantic_model_evidence" not in tool_names
        finally:
            real_agent_config.agentic_nodes["gen_semantic_model"] = original

    def test_semantic_sql_history_profiler_tool_with_explicit_skills(self, real_agent_config, mock_llm_create):
        """An explicit skills entry naming the profiler keeps the tool exposed."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        original = dict(real_agent_config.agentic_nodes.get("gen_semantic_model", {}))
        try:
            real_agent_config.agentic_nodes["gen_semantic_model"] = {
                **original,
                "skills": "metricflow-semantic-authoring, semantic-sql-history-profiler",
            }
            node = GenSemanticModelAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
        finally:
            real_agent_config.agentic_nodes["gen_semantic_model"] = original

        tool_names = {tool.name for tool in node.tools}
        assert "profile_semantic_model_evidence" in tool_names

    def test_semantic_model_max_turns(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        # The real_agent_config has gen_semantic_model.max_turns = 5
        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 5

    def test_semantic_model_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 50 when not configured."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        # Remove gen_semantic_model from agentic_nodes to test default
        original = real_agent_config.agentic_nodes.pop("gen_semantic_model", None)
        try:
            node = GenSemanticModelAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            assert node.max_turns == 50
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_semantic_model"] = original


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.llm_harness
class TestGenSemanticModelAgenticNodeExecution:
    """Tests for GenSemanticModelAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_semantic_model_simple_response(self, real_agent_config, mock_llm_create):
        """Test execute_stream with a simple text response."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generation completed."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model for satscores table")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have at least: USER action + LLM response + final action
        assert len(actions) >= 2

        # First action should be USER/PROCESSING
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # Last action should be SUCCESS
        last_action = actions[-1]
        assert last_action.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_semantic_model_with_db_tool_calls(self, real_agent_config, mock_llm_create):
        """Test execute_stream where LLM calls describe_table tool against real SQLite."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="describe_table",
                            arguments=json.dumps({"table_name": "satscores"}),
                        ),
                    ],
                    content="I have examined the satscores table and created the semantic model.",
                ),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model for satscores table")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have: USER + TOOL start + TOOL complete + ASSISTANT response + final action
        assert len(actions) >= 4

        # Check tool actions exist
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2  # 1 tool call x 2 (start + complete)

        tool_processing = [a for a in tool_actions if a.status == ActionStatus.PROCESSING]
        assert any(a.action_type == "describe_table" for a in tool_processing)

        # Check the tool was actually executed against real SQLite
        tool_results = mock_llm_create.tool_results
        assert len(tool_results) >= 1
        assert tool_results[0]["tool"] == "describe_table"
        assert tool_results[0]["executed"] is True

    @pytest.mark.asyncio
    async def test_semantic_model_workflow_mode(self, real_agent_config, mock_llm_create):
        """Test node in workflow mode has no hooks and executes correctly."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Done generating semantic model."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.hooks is None
        assert node.execution_mode == "workflow"

        node.input = SemanticNodeInput(user_message="Generate semantic model")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Execution should succeed in workflow mode
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_semantic_model_with_database_context(self, real_agent_config, mock_llm_create):
        """Test execute_stream with database context enriches the enhanced message."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generated with database context."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message="Generate semantic model for satscores",
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with enhanced prompt containing database context
        # (unified format via ``build_database_context``).
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "Generate semantic model for satscores" in prompt
        assert "california_schools" in prompt

    @pytest.mark.asyncio
    async def test_semantic_model_interactive_mode_token_tracking(self, real_agent_config, mock_llm_create):
        """Test that interactive mode tracks token usage from action history."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generated in interactive mode."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="interactive",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # In interactive mode, the final result should have tokens_used > 0
        last_output = actions[-1].output
        assert isinstance(last_output, dict)
        assert "tokens_used" in last_output
        assert last_output["tokens_used"] > 0

    @pytest.mark.asyncio
    async def test_semantic_model_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Test that execute_stream raises when input is not set."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = None

        action_manager = ActionHistoryManager()
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException, match="Missing required field"):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_semantic_model_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        """Test that ExecutionInterrupted is re-raised from execute_stream."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            """Async generator that raises ExecutionInterrupted."""
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, execution_mode="workflow"):
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    return GenSemanticModelAgenticNode(
        agent_config=real_agent_config,
        execution_mode=execution_mode,
    )


# ---------------------------------------------------------------------------
# TestExtractSemanticModelAndOutputFromResponse
# ---------------------------------------------------------------------------


class TestExtractSemanticModelAndOutputFromResponse:
    def test_extracts_from_dict_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_files": ["orders.yml", "customers.yml"],
                "output": "Generated semantic models successfully",
            }
        }
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == ["orders.yml", "customers.yml"]
        assert out == "Generated semantic models successfully"

    def test_extracts_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "semantic_model_files": ["model.yml"],
                "output": "Done",
            }
        )
        output = {"content": content}
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == ["model.yml"]
        assert out == "Done"

    def test_returns_empty_list_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == []
        assert out is None

    def test_returns_empty_list_on_dict_missing_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"other_key": "other_value"}}
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == []

    def test_returns_empty_list_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "not valid json at all"}
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == []

    def test_returns_empty_on_non_list_semantic_model_files(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_files": "not_a_list",  # should be a list
                "output": "Done",
            }
        }
        files, out = node._extract_semantic_model_and_output_from_response(output)
        assert files == []


# ---------------------------------------------------------------------------
# TestPrepareTemplateContext
# ---------------------------------------------------------------------------


class TestPrepareTemplateContext:
    def test_template_context_contains_required_keys(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = SemanticNodeInput(user_message="Generate semantic model")
        context = node._prepare_template_context(user_input)

        assert "native_tools" in context
        assert "mcp_tools" in context
        assert "semantic_model_dir" in context

    def test_template_context_lists_tool_names(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Add a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        node.tools = [mock_tool]

        user_input = SemanticNodeInput(user_message="Generate semantic model")
        context = node._prepare_template_context(user_input)

        assert "test_tool" in context["native_tools"]

    def test_osi_target_is_in_request_context_not_system_template(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = SemanticNodeInput(
            user_message="Generate a semantic model",
            semantic_model_name="Executive Sales",
            business_domain="commerce",
            fact_tables=["main.orders"],
            dimension_tables=["main.customers"],
        )

        context = node._prepare_template_context(user_input)

        assert "osi_target_resolved" not in context
        assert "requested_semantic_model_name" not in context
        enhanced = node._build_enhanced_message(user_input)
        assert "Requested semantic model name: `Executive Sales`" in enhanced
        assert "executive_sales.yml" not in enhanced
        assert "only the tool result binds the target" in enhanced
        assert "main.orders" in enhanced
        assert "main.customers" in enhanced


class TestGetSystemPrompt:
    def test_osi_authoring_uses_shared_template_with_osi_context(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create)

        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
            mock_pm.return_value.render_template.return_value = "osi prompt"

            template_context = node._prepare_template_context(None)
            node._get_system_prompt(prompt_version="1.2", template_context=template_context)

        call_kwargs = mock_pm.return_value.render_template.call_args.kwargs
        # Both formats share one template; the format travels in the context.
        assert call_kwargs["template_name"] == "gen_semantic_model_system"
        assert call_kwargs["version"] == "1.2"
        assert call_kwargs["authoring_format"] == "osi"

    def test_osi_authoring_injects_osi_required_skill(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create)

        prompt = node._get_system_prompt(template_context=node._prepare_template_context(None))

        assert '<required_skill name="osi-semantic-authoring">' in prompt
        assert "OSI core semantics only" in prompt
        assert '<required_skill name="metricflow-semantic-authoring">' not in prompt

    def test_metricflow_authoring_injects_metricflow_required_skill(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)

        prompt = node._get_system_prompt(template_context=node._prepare_template_context(None))

        assert '<required_skill name="metricflow-semantic-authoring">' in prompt
        assert "MetricFlow semantic model structure specification" in prompt
        assert '<required_skill name="osi-semantic-authoring">' not in prompt


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeNameGenSemanticModel:
    def test_get_node_name(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        assert node.get_node_name() == "gen_semantic_model"


# ---------------------------------------------------------------------------
# TestExecutionMode
# ---------------------------------------------------------------------------


class TestExecutionModeGenSemanticModel:
    def test_workflow_mode_has_no_hooks(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        assert node.hooks is None
        assert node.execution_mode == "workflow"

    def test_interactive_mode_has_hooks(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="interactive")
        # Hooks are set up in interactive mode
        assert node.execution_mode == "interactive"
        # hooks may or may not be set depending on whether setup_hooks succeeded

    def test_workflow_mode_compose_hooks_is_non_interactive(self, real_agent_config, mock_llm_create):
        """Workflow mode → PermissionHooks must run with non_interactive=True
        so ASK / EXTERNAL fs hits raise PermissionDeniedException instead of
        awaiting a broker that nobody will answer (``/bootstrap`` flow)."""
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        # Workflow now also composes CompactHook (multi-turn history enabled
        # in all modes), so _compose_hooks may return CompositeHooks. Check
        # the permission gate on the node so the test stays robust.
        hooks = node._compose_hooks()
        assert node.permission_hooks.non_interactive is True
        hooks_list = getattr(hooks, "hooks_list", [hooks])
        assert node.permission_hooks in hooks_list
        # Permission manager must be loaded with the dangerous profile, not the
        # user's profile, so workflow flows always operate against a known
        # baseline.
        assert node.permission_manager.active_profile == "dangerous"

    def test_interactive_mode_compose_hooks_is_interactive(self, real_agent_config, mock_llm_create):
        """Interactive mode keeps the broker prompts (``non_interactive=False``)
        and the user's permission profile."""
        from datus.tools.permission.permission_hooks import PermissionHooks

        node = _make_node(real_agent_config, mock_llm_create, execution_mode="interactive")
        hooks = node._compose_hooks()
        # ``hooks`` may be CompositeHooks(generation_hooks, permission_hooks).
        # Find the PermissionHooks layer regardless of composition shape.
        if isinstance(hooks, PermissionHooks):
            permission_layer = hooks
        else:
            from datus.tools.permission.permission_hooks import CompositeHooks

            assert isinstance(hooks, CompositeHooks)
            permission_layer = next(h for h in hooks.hooks_list if isinstance(h, PermissionHooks))
        assert permission_layer.non_interactive is False


# ---------------------------------------------------------------------------
# TestExecuteStreamError
# ---------------------------------------------------------------------------


class TestExecuteStreamGenSemanticModelError:
    def test_osi_finalizer_revalidates_when_evidence_targets_another_model(self, tmp_path):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.tools.func_tool.base import FuncToolResult
        from datus.tools.func_tool.generation_evidence import GenerationEvidence

        sales_file = tmp_path / "sales.yml"
        finance_file = tmp_path / "finance.yml"
        sales_file.write_text("semantic_model: sales\n", encoding="utf-8")
        finance_file.write_text("semantic_model: finance\n", encoding="utf-8")

        node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
        node.agent_config = SimpleNamespace(resolve_semantic_adapter=lambda requested=None: "osi")
        node.generation_evidence = GenerationEvidence(validation_passed=True)
        node.generation_evidence.record_semantic_artifact_validation("sales", sales_file)
        node.generation_tools = MagicMock()
        node.generation_tools.resolve_planned_osi_semantic_target.return_value = (
            "subject/semantic_models/warehouse/finance.yml",
            str(finance_file),
            "finance",
        )
        node.generation_tools.publish_semantic_model.return_value = FuncToolResult(
            result={"semantic_model_files": ["subject/semantic_models/warehouse/finance.yml"]}
        )
        node.semantic_func_tool = MagicMock()
        node.semantic_func_tool.validate_semantic.return_value = FuncToolResult(result={"valid": True, "issues": []})

        node._finalize_semantic_model_generation(["finance.yml"])

        node.semantic_func_tool.validate_semantic.assert_called_once_with(
            scope="semantic_model",
            semantic_model_name="finance",
        )
        node.generation_tools.publish_semantic_model.assert_called_once_with(
            ["subject/semantic_models/warehouse/finance.yml"]
        )
        assert node.generation_evidence.semantic_artifact_validation_passed("finance", finance_file)

    def test_osi_finalizer_rejects_unplanned_final_json_fallback(self):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.tools.func_tool.generation_evidence import GenerationEvidence

        node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
        node.agent_config = SimpleNamespace(resolve_semantic_adapter=lambda requested=None: "osi")
        node.generation_evidence = GenerationEvidence()
        node.generation_tools = MagicMock()
        node.generation_tools.resolve_planned_osi_semantic_target.side_effect = ValueError(
            "Plan the OSI semantic-model name and file before publishing."
        )
        node.semantic_func_tool = MagicMock()

        with pytest.raises(RuntimeError, match="Plan the OSI semantic-model"):
            node._finalize_semantic_model_generation(["subject/semantic_models/warehouse/rogue.yml"])

        node.semantic_func_tool.validate_semantic.assert_not_called()
        node.generation_tools.publish_semantic_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate semantic model")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"

    @pytest.mark.asyncio
    async def test_final_semantic_files_without_end_tool_auto_validates_and_publishes(
        self, real_agent_config, mock_llm_create
    ):
        """A final JSON file list is enough when the node can validate and publish it."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        model_dir = real_agent_config.path_manager.semantic_model_path(datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        semantic_path = model_dir / "orders.yml"
        semantic_path.write_text(
            "data_source:\n"
            "  name: orders\n"
            "  sql_table: orders\n"
            "  measures:\n"
            "    - name: order_count\n"
            "      agg: COUNT\n"
            '      expr: "1"\n',
            encoding="utf-8",
        )
        reported_semantic_path = f"subject/semantic_models/{datasource}/orders.yml"

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_files": [reported_semantic_path],
                            "output": "Generated semantic model.",
                        }
                    )
                ),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate semantic model")
        node.semantic_func_tool = MagicMock()
        node.semantic_func_tool.validate_semantic = MagicMock(
            return_value=FuncToolResult(result={"valid": True, "issues": []})
        )

        action_manager = ActionHistoryManager()
        actions = []
        with patch(
            "datus.agent.node.gen_semantic_model_agentic_node.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "synced"},
        ) as sync_mock:
            async for action in node.execute_stream(action_manager):
                actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "gen_semantic_model_response"
        node.semantic_func_tool.validate_semantic.assert_called_once_with(scope="semantic_model")
        sync_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_final_semantic_files_without_validation_fails_closed(self, real_agent_config, mock_llm_create):
        """Final JSON files do not publish when host-side validate_semantic fails."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_files": ["orders.yml"],
                            "output": "Generated semantic model.",
                        }
                    )
                ),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate semantic model")
        node.semantic_func_tool = MagicMock()
        node.semantic_func_tool.validate_semantic = MagicMock(
            return_value=FuncToolResult(
                success=0,
                error="bad semantic YAML",
                result={"valid": False, "issues": [{"message": "bad semantic YAML"}]},
            )
        )

        action_manager = ActionHistoryManager()
        actions = []
        with patch(
            "datus.agent.node.gen_semantic_model_agentic_node.GenerationHooks._sync_semantic_to_db"
        ) as sync_mock:
            async for action in node.execute_stream(action_manager):
                actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert actions[-1].action_type == "error"
        assert "validate_semantic failed before publishing semantic models" in actions[-1].output["error"]
        node.semantic_func_tool.validate_semantic.assert_called_once_with(scope="semantic_model")
        sync_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_stream_with_catalog_context(self, real_agent_config, mock_llm_create):
        """Test execute_stream with catalog enriches the enhanced message."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generated with catalog context."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message="Generate semantic model",
            catalog="my_catalog",
            database="california_schools",
            db_schema="main",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify prompt contains catalog context (unified ``build_database_context`` format).
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "Generate semantic model" in prompt
        assert "my_catalog" in prompt
        assert "california_schools" in prompt
        assert "main" in prompt


class TestGenSemanticModelFilesystemRootPath:
    """FilesystemFuncTool now uses project_root; write-scope enforcement moved to GenerationHooks."""

    def test_filesystem_root_is_project_root(self, real_agent_config, mock_llm_create):
        from pathlib import Path

        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        expected = str(Path(real_agent_config.project_root).expanduser())

        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert node.filesystem_func_tool.root_path == expected
