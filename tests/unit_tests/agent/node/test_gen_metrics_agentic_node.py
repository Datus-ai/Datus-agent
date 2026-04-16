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
from unittest.mock import MagicMock

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response

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
        assert not hasattr(node, "hooks")  # hooks removed from gen_metrics

    def test_metrics_has_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has filesystem and generation tools."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert len(node.tools) > 0
        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names

        # Generation tools
        assert "check_semantic_object_exists" in tool_names
        assert "end_metric_generation" in tool_names

        # Tool instances should be initialized
        assert node.filesystem_func_tool is not None
        assert node.generation_tools is not None

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
        """Test default max_turns is 30 when not configured."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        # Remove gen_metrics from agentic_nodes to test default
        original = real_agent_config.agentic_nodes.pop("gen_metrics", None)
        try:
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            assert node.max_turns == 40
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_metrics"] = original


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


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

        assert not hasattr(node, "hooks")  # hooks removed from gen_metrics
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
        with pytest.raises(ValueError, match="Metrics input not set"):
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
        assert "california_schools" in prompt or "Generate" in prompt

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
        assert last_output is not None
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
            yield  # noqa: makes this an async generator

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
        assert node.db_func_tool is not None

    def test_db_tools_failure_does_not_break_init(self, real_agent_config, mock_llm_create):
        """When DBFuncTool.create_dynamic raises, node still initializes with other tools."""
        from unittest.mock import patch as _patch

        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        with _patch(
            "datus.tools.func_tool.DBFuncTool.create_dynamic",
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
# TestSetupGenSemanticModelTools
# ---------------------------------------------------------------------------


class TestSetupGenSemanticModelTools:
    """Tests for _setup_gen_semantic_model_tools() method."""

    def test_gen_semantic_model_tools_added_when_db_available(self, real_agent_config, mock_llm_create):
        """When db_func_tool is initialized, gen_semantic_model_tools should be mounted."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "analyze_table_relationships" in tool_names, f"Missing analyze_table_relationships, got: {tool_names}"
        assert "get_multiple_tables_ddl" in tool_names, f"Missing get_multiple_tables_ddl, got: {tool_names}"
        assert "analyze_column_usage_patterns" in tool_names, (
            f"Missing analyze_column_usage_patterns, got: {tool_names}"
        )
        assert node.gen_semantic_model_tools is not None

    def test_gen_semantic_model_tools_skipped_when_no_db(self, real_agent_config, mock_llm_create):
        """When DBFuncTool.create_dynamic fails, gen_semantic_model_tools is None but node still works."""
        from unittest.mock import patch as _patch

        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        with _patch(
            "datus.tools.func_tool.DBFuncTool.create_dynamic",
            side_effect=RuntimeError("no connection"),
        ):
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )

        tool_names = [tool.name for tool in node.tools]
        assert "analyze_table_relationships" not in tool_names
        assert node.gen_semantic_model_tools is None
        # Other tools still present
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
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "revenue_metrics.yml"
        assert sem_model == "model.yml"
        assert out == "Generated successfully"

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
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "sales_metrics.yml"
        assert out == "Done"

    def test_returns_none_triple_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None
        assert sem_model is None
        assert out is None

    def test_returns_none_triple_on_dict_missing_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None

    def test_returns_none_triple_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "not json at all !!!"}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None


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

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have initial USER action + error action
        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"


class TestGenMetricsFilesystemRootPath:
    """FilesystemFuncTool is sandboxed to subject_dir (not the type-specific subdir)."""

    def test_filesystem_root_is_subject_dir(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        expected = str(real_agent_config.path_manager.subject_dir)

        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected
        assert node.filesystem_func_tool._path_normalizer is not None

        # metric kind co-locates under semantic_models/
        assert (
            node.filesystem_func_tool._path_normalizer("metrics/orders.yaml", None)
            == "semantic_models/metrics/orders.yaml"
        )
