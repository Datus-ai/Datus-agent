# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenMetricsAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (FilesystemFuncTool, GenerationTools, SemanticTools)
- Max turns configuration
- Streaming execution with MockLLMModel
- Filesystem tool invocation
- Thinking content in responses
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real Storage/RAG (vector store in tmp_path)
- Real Tools (FilesystemFuncTool, GenerationTools, SemanticTools)
- Real PromptManager (using built-in templates)
- Real PathManager
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import semantic_authoring
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response


@pytest.fixture(autouse=True)
def _stub_osi_schema_validation(monkeypatch):
    monkeypatch.setattr(semantic_authoring, "validate_osi_core_document", lambda document: None)


def _set_global_semantic_adapter(agent_config, adapter: str) -> None:
    agent_config.resolve_semantic_adapter = MagicMock(return_value=adapter)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenMetricsAgenticNodeInit:
    """Tests for GenMetricsAgenticNode initialization."""

    def test_metrics_init(self, real_agent_config, mock_llm_create):
        """Test that GenMetricsAgenticNode initializes with real config."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_metrics"
        assert node.id == "gen_metrics_node"
        assert node.execution_mode == "workflow"
        assert node.hooks is None

    def test_metrics_has_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has filesystem and generation tools."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = {tool.name for tool in node.tools}

        # Filesystem tools
        assert {"read_file", "write_file", "edit_file", "glob", "grep"}.issubset(tool_names)

        # Generation tools
        assert "check_semantic_object_exists" in tool_names
        assert "publish_metrics" in tool_names

        # Tool instances should be initialized
        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert isinstance(node.generation_tools, GenerationTools)

    def test_osi_metrics_has_only_metric_mutation_tools(self, real_agent_config, mock_llm_create):
        """OSI metrics cannot invoke general filesystem or semantic-model authoring tools."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")

        node._get_system_prompt(template_context=node._prepare_template_context(node.input))
        tool_names = {tool.name for tool in node.tools}

        assert {"read_file", "upsert_osi_metrics", "glob", "grep"}.issubset(tool_names)
        assert {"write_file", "edit_file", "delete_file", "publish_semantic_model", "bash"}.isdisjoint(tool_names)
        assert "task" not in tool_names
        assert node.sub_agent_task_tool is None
        assert "list_existing_osi_semantic_models" in tool_names
        assert "bind_osi_semantic_model_target" in tool_names
        assert "plan_osi_semantic_model_target" not in tool_names
        assert "publish_metrics" in tool_names
        node._populate_tool_registry()
        assert node.tool_registry.get("list_existing_osi_semantic_models") == "semantic_tools"
        assert node.tool_registry.get("bind_osi_semantic_model_target") == "semantic_tools"

    def test_metrics_max_turns(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        # The real_agent_config has gen_metrics.max_turns = 5
        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 5

    def test_metrics_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 50 when not configured."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        # Remove gen_metrics from agentic_nodes to test default
        original = real_agent_config.agentic_nodes.pop("gen_metrics", None)
        try:
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            assert node.max_turns == 50
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_metrics"] = original

    def test_tool_registry_splits_semantic_and_db(self, real_agent_config, mock_llm_create):
        """The registry buckets semantic helpers into ``semantic_tools`` and
        DB helpers into ``db_tools`` so profile rules for each match correctly."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry.get("publish_metrics") == "semantic_tools"
        assert registry.get("check_semantic_object_exists") == "semantic_tools"
        assert registry.get("execute_sql") == "db_tools"
        assert "read_query" not in registry
        assert registry.get("write_file") == "filesystem_tools"


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.llm_harness
class TestGenMetricsAgenticNodeExecution:
    """Tests for GenMetricsAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_metrics_simple_response(self, real_agent_config, mock_llm_create):
        """Test execute_stream with a simple text response."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Metrics generation completed successfully."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

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
    async def test_before_stream_resets_authoring_state_and_preserves_request_hints(
        self, real_agent_config, mock_llm_create
    ):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message="Create order metrics. The semantic model is at subject/semantic_models/warehouse/orders.yml.",
            semantic_model_name="requested_target",
        )
        node.osi_target_state.select(
            {
                "semantic_model_name": "old_target",
                "semantic_model_file": "subject/semantic_models/warehouse/old.yml",
                "absolute_path": "/tmp/old.yml",
            },
            mode="bound",
        )
        node.osi_target_state.authored_metric_names = ["old_metric"]
        node.generation_evidence.validation_passed = True
        node.generation_evidence.metric_kb_sync_passed = True
        evidence = node.generation_evidence
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())

        await node._before_stream(ctx)

        assert node.osi_target_state.bound is None
        assert node.osi_target_state.authored_metric_names == []
        assert node.generation_evidence is evidence
        assert node.generation_evidence == type(evidence)()
        assert node.generation_tools.generation_evidence is evidence
        assert node.semantic_tools.generation_evidence is evidence
        assert node.input.semantic_model_name == "requested_target"
        assert mock_llm_create.call_history == []

    @pytest.mark.asyncio
    async def test_before_stream_leaves_sql_preflight_pending(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate metrics for SELECT SUM(amount) AS revenue FROM orders")
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())

        await node._before_stream(ctx)

        assert node.sql_modeling_plan is None
        assert node.generation_evidence.sql_modeling_plan_status == "pending"
        assert node.generation_evidence.required_metric_output_ids == []
        assert "prepare_sql_modeling_plan" in {tool.name for tool in node.tools}

    def test_sql_result_cannot_bypass_preflight(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext
        from datus.utils.exceptions import DatusException

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="SELECT COUNT(*) AS order_count FROM orders")
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
        ctx.response_content = "not json"

        with pytest.raises(DatusException, match="prepare_sql_modeling_plan"):
            node._build_success_result(ctx)

    def test_sql_result_requires_structured_terminal_response(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="SELECT COUNT(*) AS order_count FROM orders")
        node.generation_evidence.set_sql_modeling_plan("ready", "source")
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
        ctx.response_content = json.dumps({"metric_file": "metrics/orders.yml"})

        with pytest.raises(RuntimeError, match="structured final response"):
            node._build_success_result(ctx)

    def test_sql_result_cannot_skip_required_metric_outputs(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="SELECT COUNT(*) AS order_count FROM orders")
        node.generation_evidence.set_sql_modeling_plan("ready", "source")
        node.generation_evidence.set_required_metric_outputs([{"output_id": "orders:output_1"}])
        ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
        ctx.response_content = json.dumps(
            {
                "metric_file": None,
                "status": "skipped",
                "skip_reason": "not_a_metric",
                "output": "Skipped.",
            }
        )

        with pytest.raises(RuntimeError, match="required metric outputs"):
            node._build_success_result(ctx)

    @pytest.mark.asyncio
    async def test_metrics_with_filesystem_tool(self, real_agent_config, mock_llm_create):
        """Test execute_stream where LLM calls write_file tool."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="write_file",
                            arguments=json.dumps(
                                {
                                    "path": "revenue_metrics.yml",
                                    "content": "metric:\n  name: revenue\n  type: simple",
                                }
                            ),
                        ),
                    ],
                    content="I have generated the revenue metrics file.",
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

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
        assert any(a.action_type == "write_file" for a in tool_processing)

        # Check the tool was actually executed
        tool_results = mock_llm_create.tool_results
        assert len(tool_results) >= 1
        assert tool_results[0]["tool"] == "write_file"
        assert tool_results[0]["executed"] is True

    @pytest.mark.asyncio
    async def test_metrics_workflow_mode(self, real_agent_config, mock_llm_create):
        """Test node in workflow mode has no hooks."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Done generating metrics."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.hooks is None
        assert node.execution_mode == "workflow"

        node.input = SemanticNodeInput(user_message="Generate metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Execution should succeed in workflow mode
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_metrics_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Test that execute_stream raises when input is not set."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
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
    async def test_metrics_with_database_context(self, real_agent_config, mock_llm_create):
        """Test execute_stream with database context enriches the enhanced message."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Metrics generated with database context."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message="Generate revenue metrics",
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with enhanced prompt containing database context
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "Generate revenue metrics" in prompt
        assert "california_schools" in prompt

    @pytest.mark.asyncio
    async def test_metrics_interactive_mode_token_tracking(self, real_agent_config, mock_llm_create):
        """Test that interactive mode tracks token usage from action history."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Metrics generated in interactive mode."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="interactive",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # In interactive mode, the final action output should be present
        last_output = actions[-1].output
        assert isinstance(last_output, dict), f"Expected dict output in interactive mode, got {type(last_output)}"
        # Interactive mode must report token usage for cost tracking
        assert "tokens_used" in last_output, f"Expected 'tokens_used' in output keys, got: {list(last_output.keys())}"
        assert last_output["tokens_used"] >= 0

    @pytest.mark.asyncio
    async def test_metrics_with_thinking(self, real_agent_config, mock_llm_create):
        """Test response with thinking content yields a thinking action."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    content="Generated revenue metrics.",
                    thinking="I need to analyze the revenue data and create appropriate metrics.",
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have thinking action among the assistant actions
        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) >= 2  # thinking + response + final

        # Check that thinking content appears somewhere in the action stream
        all_action_text = " ".join(str(a.output) + " " + str(getattr(a, "messages", "")) for a in assistant_actions)
        assert "analyze the revenue data" in all_action_text, (
            f"Expected thinking content in actions, got: {all_action_text[:200]}"
        )

    @pytest.mark.asyncio
    async def test_metrics_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        """Test that ExecutionInterrupted is re-raised from execute_stream."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            """Async generator that raises ExecutionInterrupted."""
            raise ExecutionInterrupted("User pressed ESC")
            yield  # pragma: no cover - makes this an async generator

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate metrics")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, execution_mode="workflow"):
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

    return GenMetricsAgenticNode(
        agent_config=real_agent_config,
        execution_mode=execution_mode,
    )


# ---------------------------------------------------------------------------
# TestSetupDbTools
# ---------------------------------------------------------------------------


class TestSetupDbTools:
    """Tests for _setup_db_tools() method."""

    def test_db_tools_added_when_available(self, real_agent_config, mock_llm_create):
        """When db_manager can connect, DB tools should be in node.tools."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        # SQLite connector provides these tools via DBFuncTool
        assert "describe_table" in tool_names, f"Missing describe_table, got: {tool_names}"
        assert "list_tables" in tool_names, f"Missing list_tables, got: {tool_names}"
        assert isinstance(node.db_func_tool, DBFuncTool)

    def test_db_tools_failure_does_not_break_init(self, real_agent_config, mock_llm_create):
        """When DBFuncTool() constructor raises, node still initializes with other tools."""
        from unittest.mock import patch as _patch

        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        with _patch(
            "datus.tools.func_tool.DBFuncTool",
            side_effect=RuntimeError("no connection"),
        ):
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )

        tool_names = [tool.name for tool in node.tools]
        # DB tools should be absent, but filesystem/generation tools still present
        assert "describe_table" not in tool_names
        assert "read_file" in tool_names
        assert "check_semantic_object_exists" in tool_names
        assert node.db_func_tool is None


# ---------------------------------------------------------------------------
# TestSetupSemanticDiscoveryTools
# ---------------------------------------------------------------------------


class TestSetupSemanticDiscoveryTools:
    """Tests for _setup_semantic_discovery_tools() method."""

    def test_semantic_discovery_tools_added_when_db_available(self, real_agent_config, mock_llm_create):
        """When db_func_tool is initialized, semantic_discovery_tools should be mounted."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "inspect_semantic_sources" in tool_names
        assert "validate_semantic_key_candidates" in tool_names
        assert "analyze_table_relationships" not in tool_names
        assert "validate_semantic_key_candidate" not in tool_names
        assert "get_multiple_tables_ddl" not in tool_names
        assert "analyze_column_usage_patterns" not in tool_names
        assert "prepare_sql_modeling_plan" in tool_names
        assert "analyze_metric_candidates_from_history" not in tool_names
        assert isinstance(node.semantic_discovery_tools, SemanticDiscoveryTools)

    def test_sql_modeling_preflight_remains_when_no_db(self, real_agent_config, mock_llm_create):
        """Request-local SQL planning remains available without a live DB tool."""
        from unittest.mock import patch as _patch

        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        with _patch(
            "datus.tools.func_tool.DBFuncTool",
            side_effect=RuntimeError("no connection"),
        ):
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )

        tool_names = [tool.name for tool in node.tools]
        assert "inspect_semantic_sources" not in tool_names
        assert "validate_semantic_key_candidates" not in tool_names
        assert "prepare_sql_modeling_plan" in tool_names
        assert "analyze_metric_candidates_from_history" not in tool_names
        assert isinstance(node.semantic_discovery_tools, SemanticDiscoveryTools)
        assert "read_file" in tool_names
        assert "check_semantic_object_exists" in tool_names


# ---------------------------------------------------------------------------
# TestExtractMetricAndOutputFromResponse
# ---------------------------------------------------------------------------


class TestExtractMetricAndOutputFromResponse:
    def test_extracts_from_dict_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": "revenue_metrics.yml",
                "output": "Generated successfully",
            }
        }
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert metric_file == "revenue_metrics.yml"
        assert sem_model == ["model.yml"]
        assert status is None
        assert blocker_code is None
        assert skip_reason is None
        assert out == "Generated successfully"

    def test_retains_metric_output_bindings_for_host_publish_fallback(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        node = _make_node(real_agent_config, mock_llm_create)
        bindings = [{"output_id": "output_1", "metric_name": "revenue"}]

        node._extract_metric_and_output_from_response(
            {
                "content": {
                    "metric_file": "revenue_metrics.yml",
                    "metric_output_bindings": bindings,
                    "status": "generated",
                    "output": "Generated successfully",
                }
            }
        )

        assert node.response_metric_output_bindings == bindings

    def test_extracts_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "semantic_model_file": "model.yml",
                "metric_file": "sales_metrics.yml",
                "output": "Done",
            }
        )
        output = {"content": content}
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert metric_file == "sales_metrics.yml"
        assert sem_model == ["model.yml"]
        assert status is None
        assert blocker_code is None
        assert skip_reason is None
        assert out == "Done"

    def test_extracts_status_from_dict_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "skipped",
                "skip_reason": "not_a_metric",
                "output": "The request is not a metric.",
            }
        }
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "skipped"
        assert blocker_code is None
        assert skip_reason == "not_a_metric"
        assert out == "The request is not a metric."

    def test_extracts_generated_status_without_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "generated",
                "output": "Generated successfully.",
            }
        }
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "generated"
        assert blocker_code is None
        assert skip_reason is None
        assert out == "Generated successfully."

    def test_extracts_explicit_status_without_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "done",
                "output": "Done.",
            }
        }
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "done"
        assert blocker_code is None
        assert skip_reason is None
        assert out == "Done."

    def test_extracts_status_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "skipped",
                "skip_reason": "not_a_metric",
                "output": "Skipped: not a metric.",
            }
        )
        output = {"content": content}
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "skipped"
        assert blocker_code is None
        assert skip_reason == "not_a_metric"
        assert out == "Skipped: not a metric."

    def test_extracts_blocker_code(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_files": [],
                "metric_file": None,
                "status": "blocked",
                "blocker_code": "semantic_model_selection_required",
                "output": "Multiple semantic models match.",
            }
        }

        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )

        assert sem_model == []
        assert metric_file is None
        assert status == "blocked"
        assert blocker_code == "semantic_model_selection_required"
        assert skip_reason is None
        assert out == "Multiple semantic models match."

    def test_returns_none_tuple_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert metric_file is None
        assert sem_model is None
        assert status is None
        assert blocker_code is None
        assert skip_reason is None
        assert out is None

    def test_returns_none_tuple_on_dict_missing_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert metric_file is None
        assert status is None
        assert blocker_code is None
        assert skip_reason is None

    def test_returns_none_tuple_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "not json at all !!!"}
        sem_model, metric_file, status, blocker_code, skip_reason, out = node._extract_metric_and_output_from_response(
            output
        )
        assert metric_file is None
        assert status is None
        assert blocker_code is None
        assert skip_reason is None


# ---------------------------------------------------------------------------
# TestExtractMetricSqlsFromActions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestPrepareTemplateContext
# ---------------------------------------------------------------------------


class TestPrepareTemplateContext:
    def test_prepare_template_context_no_subject_tree(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = None

        # Mock the storage to return empty subject trees
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.return_value = []

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert "semantic_model_dir" in context
        assert context["has_subject_tree"] is False
        assert "existing_subject_trees" in context

    def test_prepare_template_context_with_predefined_subject_tree(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = ["Finance", "Revenue"]

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert context["has_subject_tree"] is True
        assert context["subject_tree"] == ["Finance", "Revenue"]

    def test_prepare_template_context_includes_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = None
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.return_value = []

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert "native_tools" in context
        assert "mcp_tools" in context

    def test_osi_structured_target_is_an_unresolved_turn_hint(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = SemanticNodeInput(
            user_message="Generate order metrics",
            semantic_model_name="orders_model",
            semantic_model_file="subject/semantic_models/warehouse/orders.yml",
        )

        enhanced = node._build_enhanced_message(user_input)

        assert "Requested semantic model name: `orders_model`" in enhanced
        assert "Requested semantic model file: `subject/semantic_models/warehouse/orders.yml`" in enhanced
        assert "bind_osi_semantic_model_target" in enhanced

    def test_sql_modeling_plan_is_not_reinjected_into_the_user_message(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        from datus.tools.func_tool.sql_modeling_planner import SqlModelingPlan

        node = _make_node(real_agent_config, mock_llm_create)
        node.sql_modeling_plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={
                "metric_requirements": [
                    {
                        "output_id": "output_1",
                        "dataset_requirement_id": "query_dataset:abc",
                        "dataset_name_hint": "retained_users_query_dataset",
                    }
                ],
                "dataset_requirements": [
                    {
                        "requirement_id": "query_dataset:abc",
                        "sql": "SELECT 1",
                    }
                ],
            },
        )

        enhanced = node._build_enhanced_message(SemanticNodeInput(user_message="Generate retention metrics"))

        assert "Generate retention metrics" in enhanced
        assert "dataset_requirement_id" not in enhanced
        assert "retained_users_query_dataset" not in enhanced
        assert node.sql_modeling_plan.candidate_plan["dataset_requirements"][0]["sql"] == "SELECT 1"

    def test_query_backed_plan_uses_final_output_grain_for_queryability(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        from datus.tools.func_tool.sql_modeling_planner import SqlModelingPlan

        output_id = "retention:statement_1:output_2:retained_players"
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = SemanticNodeInput(user_message="Generate retention metrics")
        node._accept_sql_modeling_plan(
            SqlModelingPlan(
                source_fingerprint="source",
                metric_catalog_fingerprint="catalog",
                candidate_plan={
                    "metric_requirements": [
                        {
                            "output_id": output_id,
                            "preferred_name": "retained_players",
                        }
                    ],
                    "dataset_requirements": [
                        {
                            "source_sql_name": "retention_metrics",
                            "output_grain": ["cohort_date", "retention_day"],
                            "metric_output_ids": [output_id],
                        }
                    ],
                    "source_classifications": [
                        {
                            "source_sql_name": "retention_metrics",
                            "classification": "metric_plus_derived_datasource",
                        }
                    ],
                    "blocked_direct_metric_candidates": [
                        {
                            "source_sql_name": "retention_metrics",
                            "name": "intermediate_count",
                        }
                    ],
                },
            )
        )

        assert node.generation_evidence.metric_queryability_contracts == [
            {
                "source": "retention_metrics",
                "dimension_hints": ["cohort_date", "retention_day"],
                "metric_hints": ["retained_players"],
                "metric_output_ids": [output_id],
                "contract_source": "query_backed_output_grain",
            }
        ]


class TestGetSystemPrompt:
    def test_workflow_uses_latest_prompt_when_version_unset(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate metrics")

        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
            mock_pm.return_value.render_template.return_value = "test prompt"

            node._get_system_prompt(template_context={})

            call_kwargs = mock_pm.return_value.render_template.call_args
            version = call_kwargs.kwargs.get("version")
            assert version is None

    def test_input_prompt_version_overrides_config(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate metrics", prompt_version="1.2")

        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
            mock_pm.return_value.render_template.return_value = "test prompt"

            node._get_system_prompt(template_context={})

            call_kwargs = mock_pm.return_value.render_template.call_args
            version = call_kwargs.kwargs.get("version")
            assert version == "1.2", f"Expected explicit version '1.2', got '{version}'"

    def test_osi_authoring_uses_shared_template_with_osi_context(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate OSI metrics", prompt_version="1.2")

        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
            mock_pm.return_value.render_template.return_value = "osi prompt"

            template_context = node._prepare_template_context(node.input)
            node._get_system_prompt(template_context=template_context)

        call_kwargs = mock_pm.return_value.render_template.call_args.kwargs
        # Both formats share one template; the format travels in the context and
        # the pinned prompt_version applies to the shared template.
        assert call_kwargs["template_name"] == "gen_metrics_system"
        assert call_kwargs["version"] == "1.2"
        assert call_kwargs["authoring_format"] == "osi"

    def test_osi_authoring_injects_osi_required_skill(self, real_agent_config, mock_llm_create):
        _set_global_semantic_adapter(real_agent_config, "osi")
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")

        prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))

        assert '<required_skill name="osi-metrics-authoring">' in prompt
        assert "OSI core semantics only" in prompt
        assert '<required_skill name="gen-metrics">' not in prompt

    def test_metricflow_authoring_injects_gen_metrics_required_skill(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate metrics")

        prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))

        assert '<required_skill name="gen-metrics">' in prompt
        assert "measure_proxy" in prompt
        assert '<required_skill name="osi-metrics-authoring">' not in prompt


# ---------------------------------------------------------------------------
# TestGetExistingSubjectTrees
# ---------------------------------------------------------------------------


class TestGetExistingSubjectTrees:
    def test_returns_subject_paths(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_storage = MagicMock()
        mock_storage.get_subject_tree_flat.return_value = ["Finance/Revenue", "Sales/Quarterly"]
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = mock_storage

        result = node._get_existing_subject_trees()
        assert result == ["Finance/Revenue", "Sales/Quarterly"]

    def test_returns_empty_when_no_storage(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = None

        result = node._get_existing_subject_trees()
        assert result == []

    def test_returns_empty_on_exception(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.side_effect = RuntimeError("storage error")

        result = node._get_existing_subject_trees()
        assert result == []


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeNameGenMetrics:
    def test_get_node_name(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        assert node.get_node_name() == "gen_metrics"


# ---------------------------------------------------------------------------
# TestExecuteStreamError
# ---------------------------------------------------------------------------


class TestExecuteStreamGenMetricsError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics")
        mock_llm_create.generate_with_tools_stream = _raise_error
        node.filesystem_func_tool.rollback_failed_metric_authoring = MagicMock(return_value=True)

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have initial USER action + error action
        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"
        node.filesystem_func_tool.rollback_failed_metric_authoring.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_final_metric_file_without_end_tool_auto_publishes(self, real_agent_config, mock_llm_create):
        """A final JSON metric_file is enough when the node can validate, dry-run, and publish it."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_path = metric_dir / "orders_metrics.yml"
        metric_path.write_text(
            "metric:\n  name: orders_total\n  type: measure_proxy\n  type_params:\n    measure: orders\n",
            encoding="utf-8",
        )
        reported_semantic_path = f"subject/semantic_models/{datasource}/orders.yml"
        reported_metric_path = f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml"

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": reported_semantic_path,
                            "metric_file": reported_metric_path,
                            "status": "generated",
                            "output": "Generated metrics.",
                        }
                    )
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics")
        node.permission_manager = None
        node.permission_hooks = None
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(
            return_value=FuncToolResult(result={"valid": True, "issues": []})
        )
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "columns": ["sql"],
                    "data": [],
                    "metadata": {
                        "sql": "SELECT 1",
                        "warehouse_dry_run": {"status": "success"},
                    },
                }
            )
        )
        node.generation_tools.publish_metrics = MagicMock(
            return_value=FuncToolResult(result={"message": "Metric generation completed and synced to Knowledge Base"})
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "gen_metrics_response"
        node.semantic_tools.validate_semantic.assert_called_once()
        node.semantic_tools.query_metrics.assert_called_once_with(metrics=["orders_total"], dry_run=True)
        node.generation_tools.publish_metrics.assert_called_once_with(
            metric_file=str(metric_path),
        )

    def test_final_metric_publish_automatically_dry_runs_grouped_source_sql(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult
        from datus.tools.func_tool.sql_modeling_planner import SqlModelingPlan

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_path = metric_dir / "revenue_metrics.yml"
        metric_path.write_text(
            "metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n",
            encoding="utf-8",
        )
        reported_semantic_path = f"subject/semantic_models/{datasource}/orders.yml"
        reported_metric_path = f"subject/semantic_models/{datasource}/metrics/revenue_metrics.yml"

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message=(
                "Create this metric from SQL: "
                "SELECT activity_date AS metric_time__day, reporting_region, "
                "SUM(revenue) AS revenue_total FROM ("
                "SELECT event_date AS activity_date, raw_region AS reporting_region, revenue FROM orders"
                ") metric_source GROUP BY activity_date, reporting_region;"
            )
        )
        node._accept_sql_modeling_plan(
            SqlModelingPlan(
                source_fingerprint="source",
                metric_catalog_fingerprint="catalog",
                candidate_plan={
                    "queryability_contracts": [
                        {
                            "source": "revenue_by_customer_segment",
                            "dimension_hints": ["metric_time__day", "reporting_region"],
                            "dimension_expr_hints": [
                                {
                                    "alias": "reporting_region",
                                    "expr": "raw_region",
                                    "column": "raw_region",
                                }
                            ],
                            "time_group_hints": [
                                {
                                    "alias": "metric_time__day",
                                    "base_expr": "event_date",
                                    "grain": "day",
                                }
                            ],
                            "metric_hints": ["revenue_total"],
                        }
                    ]
                },
            )
        )
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.get_dimensions = MagicMock(
            return_value=FuncToolResult(
                result={
                    "items": [{"name": "event_date"}, {"name": "raw_region"}],
                    "extra": {"time_dimension": "event_date", "time_granularities": ["day"]},
                }
            )
        )
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "metadata": {
                        "sql": "SELECT 1",
                        "warehouse_dry_run": {"status": "success"},
                    }
                }
            )
        )
        node.generation_tools.publish_metrics = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(reported_semantic_path, reported_metric_path, "generated")

        node.semantic_tools.get_dimensions.assert_called_once_with(metric_name="revenue_total")
        node.semantic_tools.query_metrics.assert_called_once_with(
            metrics=["revenue_total"],
            dimensions=["event_date", "raw_region"],
            time_granularity="day",
            dry_run=True,
        )
        node.generation_tools.publish_metrics.assert_called_once_with(metric_file=str(metric_path))

    def test_metric_publish_requires_warehouse_dry_run_evidence(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.semantic_tools = MagicMock()
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT COUNT(*) FROM orders"}})
        )

        with pytest.raises(RuntimeError, match="did not complete a warehouse dry-run"):
            node._ensure_metric_dry_runs(["order_count"])

    def test_publish_metrics_returns_preflight_error_to_tool_caller(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.generation_tools = MagicMock()
        node.generation_tools._extract_metric_names_from_file.return_value = ["order_count"]
        node.generation_tools._extract_metric_definitions_from_file.return_value = {}
        node.generation_tools._metric_names_requiring_dry_run.return_value = ["order_count"]
        node._resolve_metric_artifact_path = MagicMock(return_value="/tmp/order_metrics.yml")
        error = (
            "query_metrics(dry_run=True) failed for generated metric(s) order_count: "
            "Warehouse dry-run failed: invalid identifier 'DISC_PRICE'"
        )
        node._ensure_metric_dry_runs = MagicMock(side_effect=RuntimeError(error))

        result = node.publish_metrics("metrics/order_metrics.yml")

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert result.error == error
        assert result.result == {
            "code": "metric_publish_preflight_failed",
            "stage": "query_metrics_dry_run",
            "metric_file": "metrics/order_metrics.yml",
            "metrics": ["order_count"],
        }
        node.generation_tools.publish_metrics.assert_not_called()

    def test_publish_metrics_falls_back_to_exception_type_for_empty_error(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.generation_tools = MagicMock()
        node.generation_tools._extract_metric_names_from_file.return_value = ["order_count"]
        node.generation_tools._extract_metric_definitions_from_file.return_value = {}
        node.generation_tools._metric_names_requiring_dry_run.return_value = ["order_count"]
        node._resolve_metric_artifact_path = MagicMock(return_value="/tmp/order_metrics.yml")
        node._ensure_metric_dry_runs = MagicMock(side_effect=RuntimeError())

        result = node.publish_metrics("metrics/order_metrics.yml")

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert result.error == "RuntimeError"
        assert result.result == {
            "code": "metric_publish_preflight_failed",
            "stage": "query_metrics_dry_run",
            "metric_file": "metrics/order_metrics.yml",
            "metrics": ["order_count"],
        }
        node.generation_tools.publish_metrics.assert_not_called()

    def test_final_metric_publish_accepts_grouped_source_sql_dry_run(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
        from datus.tools.func_tool.base import FuncToolResult
        from datus.tools.func_tool.sql_modeling_planner import SqlModelingPlan

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_path = metric_dir / "revenue_metrics.yml"
        metric_path.write_text(
            "metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n",
            encoding="utf-8",
        )
        reported_semantic_path = f"subject/semantic_models/{datasource}/orders.yml"
        reported_metric_path = f"subject/semantic_models/{datasource}/metrics/revenue_metrics.yml"

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message=(
                "Create this metric from SQL: "
                "SELECT customer_segment, SUM(revenue) AS revenue_total FROM orders GROUP BY customer_segment;"
            )
        )
        node._accept_sql_modeling_plan(
            SqlModelingPlan(
                source_fingerprint="source",
                metric_catalog_fingerprint="catalog",
                source_queries=[
                    SourceQueryEvidence(
                        source_sql_name="revenue_by_customer_segment",
                        sql=(
                            "SELECT customer_segment, SUM(revenue) AS revenue_total "
                            "FROM orders GROUP BY customer_segment"
                        ),
                    )
                ],
                candidate_plan={},
            )
        )
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.get_dimensions = MagicMock(
            return_value=FuncToolResult(
                result={
                    "items": [{"name": "customer_segment"}],
                    "extra": {"time_dimension": None, "time_granularities": []},
                }
            )
        )
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "metadata": {
                        "sql": "SELECT 1",
                        "warehouse_dry_run": {"status": "success"},
                    }
                }
            )
        )
        node.generation_evidence.record_metric_dry_run(
            ["revenue_total"],
            FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}}),
            dimensions=["customer_segment"],
        )
        node.generation_tools.publish_metrics = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(reported_semantic_path, reported_metric_path, "generated")

        node.generation_tools.publish_metrics.assert_called_once_with(
            metric_file=str(metric_path),
        )

    @staticmethod
    def _bind_authored_osi_target(node, real_agent_config, metric_name="order_count"):
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "shop.yml"
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: commerce.orders\n",
            encoding="utf-8",
        )
        bind_result = node.osi_target_tools.bind_osi_semantic_model_target(
            semantic_model_file=str(target),
            semantic_model_name="shop",
        )
        assert bind_result.success == 1

        authored = (
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: commerce.orders\n"
            "    metrics:\n"
            f"      - name: {metric_name}\n"
        )
        target.write_text(authored, encoding="utf-8")
        node.osi_target_state.record_metric_write(target, authored.encode("utf-8"), [metric_name])
        return target

    def test_osi_finalizer_publishes_only_the_bound_target(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")
        target = self._bind_authored_osi_target(node, real_agent_config)
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "metadata": {
                        "sql": "SELECT 1",
                        "warehouse_dry_run": {"status": "success"},
                    }
                }
            )
        )
        node.generation_tools.extract_osi_model_names = MagicMock(return_value=["shop"])
        node.generation_tools.publish_metrics = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(
            semantic_model_files=["subject/semantic_models/ignored.yml"],
            metric_file="subject/semantic_models/ignored/metrics/wrong.yml",
            status="generated",
        )

        node.semantic_tools.validate_semantic.assert_called_once_with(
            semantic_model_name="shop",
        )
        node.semantic_tools.query_metrics.assert_called_once_with(metrics=["order_count"], dry_run=True)
        node.generation_tools.publish_metrics.assert_called_once_with(
            metric_file=str(target),
        )

    def test_osi_retry_republishes_an_identical_existing_metric(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        _set_global_semantic_adapter(real_agent_config, "osi")
        metric = {
            "name": "order_count",
            "description": "Count orders",
            "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
        }
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "shop.yml"
        target.write_text(
            json.dumps(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "shop",
                            "datasets": [{"name": "orders", "source": "commerce.orders"}],
                            "metrics": [metric],
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        original_content = target.read_bytes()
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate order_count")
        bind_result = node.osi_target_tools.bind_osi_semantic_model_target(str(target), "shop")
        assert bind_result.success == 1

        upsert_result = node.filesystem_func_tool.upsert_osi_metrics(str(target), json.dumps([metric]))

        assert upsert_result.success == 1
        assert upsert_result.result["unchanged"] == ["order_count"]
        assert target.read_bytes() == original_content
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "metadata": {
                        "sql": "SELECT COUNT(*) FROM orders",
                        "warehouse_dry_run": {"status": "success"},
                    }
                }
            )
        )
        node.generation_tools.extract_osi_model_names = MagicMock(return_value=["shop"])
        node.generation_tools.publish_metrics = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(None, None, "generated")

        node.semantic_tools.query_metrics.assert_called_once_with(metrics=["order_count"], dry_run=True)
        node.generation_tools.publish_metrics.assert_called_once_with(
            metric_file=str(target),
        )

    def test_osi_publish_scope_expansion_does_not_reuse_partial_sync(
        self,
        real_agent_config,
        mock_llm_create,
    ):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        _set_global_semantic_adapter(real_agent_config, "osi")
        metrics = [
            {
                "name": "order_count",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
            },
            {
                "name": "revenue",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(amount)"}]},
            },
        ]
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "shop.yml"
        target.write_text(
            json.dumps(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "shop",
                            "datasets": [{"name": "orders", "source": "commerce.orders"}],
                            "metrics": metrics,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate order_count and revenue")
        assert node.osi_target_tools.bind_osi_semantic_model_target(str(target), "shop").success == 1
        assert node.filesystem_func_tool.upsert_osi_metrics(str(target), json.dumps([metrics[0]])).success == 1
        node.generation_evidence.record_semantic_artifact_validation("shop", target)
        node.generation_evidence.record_metric_dry_run(
            ["order_count"],
            FuncToolResult(result={"metadata": {"sql": "SELECT COUNT(*) FROM orders"}}),
        )
        node.generation_evidence.mark_kb_sync("metric", ["order_count"])

        assert node.filesystem_func_tool.upsert_osi_metrics(str(target), json.dumps([metrics[1]])).success == 1
        assert node.osi_target_state.authored_metric_names == ["order_count", "revenue"]
        node.semantic_tools = MagicMock()
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(
                result={
                    "metadata": {
                        "sql": "SELECT 1",
                        "warehouse_dry_run": {"status": "success"},
                    }
                }
            )
        )
        node.generation_tools.extract_osi_model_names = MagicMock(return_value=["shop"])
        node.generation_tools.publish_metrics = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(None, None, "generated")

        node.semantic_tools.validate_semantic.assert_not_called()
        node.semantic_tools.query_metrics.assert_called_once_with(
            metrics=["order_count", "revenue"],
            dry_run=True,
        )
        node.generation_tools.publish_metrics.assert_called_once_with(
            metric_file=str(target),
        )

    def test_osi_skipped_requires_non_metric_reason(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        with pytest.raises(RuntimeError, match="skip_reason='not_a_metric'"):
            node._finalize_metric_generation(None, None, "skipped")

        node._finalize_metric_generation(None, None, "skipped", skip_reason="not_a_metric")

    def test_osi_generated_result_requires_a_bound_target(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        with pytest.raises(RuntimeError, match="must bind an existing semantic model"):
            node._finalize_metric_generation(None, None, "generated")

    def test_osi_implicit_generated_result_cannot_bypass_target_discovery(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        with pytest.raises(RuntimeError, match="must bind an existing semantic model"):
            node._finalize_metric_generation(None, None, None)

    def test_osi_blocked_result_does_not_reintroduce_host_model_matching(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate metrics for another business domain")
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "unrelated.yml").write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: unrelated\n"
            "    datasets:\n"
            "      - name: inventory\n"
            "        source: warehouse.inventory\n",
            encoding="utf-8",
        )
        ctx = StreamRunContext(
            user_input=node.input,
            action_history_manager=ActionHistoryManager(),
        )
        ctx.response_content = json.dumps(
            {
                "metric_file": None,
                "status": "blocked",
                "blocker_code": " Semantic_Model_Required ",
                "output": "No existing model matches this metric domain.",
            }
        )

        result = node._build_success_result(ctx)

        assert result.success is False
        assert result.status == "blocked"
        assert result.blocker_code == "semantic_model_required"

    def test_osi_skipped_result_normalizes_skip_reason(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Explain the source data")
        ctx = StreamRunContext(
            user_input=node.input,
            action_history_manager=ActionHistoryManager(),
        )
        ctx.response_content = json.dumps(
            {
                "metric_file": None,
                "status": " SKIPPED ",
                "skip_reason": " Not_A_Metric ",
                "output": "The request does not define a metric.",
            }
        )

        result = node._build_success_result(ctx)

        assert result.status == "skipped"
        assert result.skip_reason == "not_a_metric"

    def test_osi_bound_target_can_report_a_missing_semantic_prerequisite(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate a metric that requires a missing relationship")
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "commerce.yml"
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: commerce\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: warehouse.orders\n",
            encoding="utf-8",
        )
        assert node.osi_target_tools.bind_osi_semantic_model_target(str(target), "commerce").success == 1
        ctx = StreamRunContext(
            user_input=node.input,
            action_history_manager=ActionHistoryManager(),
        )
        ctx.response_content = json.dumps(
            {
                "metric_file": None,
                "status": "blocked",
                "blocker_code": "semantic_model_target_invalid",
                "output": "The bound model is missing a required relationship.",
            }
        )

        result = node._build_success_result(ctx)

        assert result.success is False
        assert result.status == "blocked"
        assert result.blocker_code == "semantic_model_target_invalid"

    def test_osi_generated_result_requires_an_authored_metric(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "shop.yml"
        target.write_text("semantic_model:\n  - name: shop\n", encoding="utf-8")
        result = node.osi_target_tools.bind_osi_semantic_model_target(str(target), "shop")
        assert result.success == 1

        with pytest.raises(RuntimeError, match="No metrics were authored"):
            node._finalize_metric_generation(None, None, "generated")

    def test_osi_finalizer_rejects_a_stale_bound_revision(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        target = self._bind_authored_osi_target(node, real_agent_config)
        target.write_text(target.read_text(encoding="utf-8") + "# external change\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="changed after selection"):
            node._finalize_metric_generation(None, None, "generated")

    def test_osi_blocked_result_is_rejected_after_metric_authoring(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.stream_run_context import StreamRunContext

        _set_global_semantic_adapter(real_agent_config, "osi")
        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")
        self._bind_authored_osi_target(node, real_agent_config)
        ctx = StreamRunContext(
            user_input=node.input,
            action_history_manager=ActionHistoryManager(),
        )
        ctx.response_content = json.dumps(
            {
                "metric_file": None,
                "status": "blocked",
                "blocker_code": "semantic_model_target_invalid",
                "output": "Cannot select a model.",
            }
        )

        with pytest.raises(RuntimeError, match="only valid before metric authoring"):
            node._build_success_result(ctx)

    @pytest.mark.asyncio
    async def test_final_metric_file_rejects_out_of_sandbox_absolute_path(
        self, real_agent_config, mock_llm_create, tmp_path
    ):
        """Final JSON fallback must reject fabricated metric paths before opening them."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        outside = tmp_path / "outside_metrics.yml"
        outside.write_text(
            "metric:\n  name: outside_metric\n  type: measure_proxy\n  type_params:\n    measure: outside\n",
            encoding="utf-8",
        )

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": None,
                            "metric_file": str(outside),
                            "status": "generated",
                            "output": "Generated metrics.",
                        }
                    )
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics")
        node.permission_manager = None
        node.permission_hooks = None
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(
            return_value=FuncToolResult(result={"valid": True, "issues": []})
        )
        node.generation_tools._validate_metric_file_has_blocks = MagicMock(return_value=None)
        node.generation_tools.publish_metrics = MagicMock(
            return_value=FuncToolResult(result={"message": "should not publish"})
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert actions[-1].action_type == "error"
        assert "outside Knowledge Base sandbox" in actions[-1].output["error"]
        node.generation_tools._validate_metric_file_has_blocks.assert_not_called()
        node.generation_tools.publish_metrics.assert_not_called()

    def test_final_metric_path_resolver_rejects_parent_traversal(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        with pytest.raises(RuntimeError, match="outside Knowledge Base sandbox"):
            node._resolve_metric_artifact_path("../outside_metrics.yml", "metric")

    @pytest.mark.asyncio
    async def test_skipped_status_bypasses_publish_gate(self, real_agent_config, mock_llm_create):
        """``status: 'skipped'`` with ``metric_file: null`` is a clean exit, not a publish failure."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": "orders.yml",
                            "metric_file": None,
                            "status": "skipped",
                            "output": "All requested metrics already exist; nothing generated.",
                        }
                    )
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics that already exist")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "gen_metrics_response"

    @pytest.mark.asyncio
    async def test_skipped_status_with_metric_file_fails_closed(self, real_agent_config, mock_llm_create):
        """``status: 'skipped'`` is only valid when no metric file was generated."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": "orders.yml",
                            "metric_file": "orders_metrics.yml",
                            "status": "skipped",
                            "output": "Metric already exists; reused existing definition.",
                        }
                    )
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics that already exist")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert actions[-1].action_type == "error"
        assert "status='skipped' with a non-null metric_file" in actions[-1].output["error"]

    @pytest.mark.asyncio
    async def test_generated_status_without_metric_file_fails_closed(self, real_agent_config, mock_llm_create):
        """``status: 'generated'`` must name a metric file unless sync already happened."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": "orders.yml",
                            "metric_file": None,
                            "status": "generated",
                            "output": "Generated metrics.",
                        }
                    )
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert actions[-1].action_type == "error"
        assert "status='generated' without a metric_file" in actions[-1].output["error"]


class TestGenMetricsFilesystemRootPath:
    """FilesystemFuncTool now uses project_root; write-scope enforcement moved to GenerationHooks."""

    def test_filesystem_root_is_project_root(self, real_agent_config, mock_llm_create):
        from pathlib import Path

        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        expected = str(Path(real_agent_config.project_root).expanduser())

        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert node.filesystem_func_tool.root_path == expected


class TestGenMetricsNonInteractiveBridge:
    """Workflow mode → ``PermissionHooks.non_interactive=True``.

    Ensures ``/bootstrap`` Metrics tab and other workflow-mode callers cannot
    be paused by ASK / EXTERNAL fs broker prompts.
    """

    def test_workflow_mode_compose_hooks_is_non_interactive(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.permission.permission_hooks import CompositeHooks, PermissionHooks

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        # Workflow mode may now compose CompositeHooks (permission + compact)
        # because multi-turn history is enabled for all modes. Validate the
        # permission gate via ``node.permission_hooks`` instead of the bundle.
        hooks = node._compose_hooks()
        assert isinstance(hooks, CompositeHooks)
        assert isinstance(node.permission_hooks, PermissionHooks)
        assert node.permission_hooks.non_interactive is True
        assert node.permission_manager.active_profile == "dangerous"
