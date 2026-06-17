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

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools
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
        assert "end_metric_generation" in tool_names

        # Tool instances should be initialized
        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert isinstance(node.generation_tools, GenerationTools)

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
        assert registry.get("end_metric_generation") == "semantic_tools"
        assert registry.get("check_semantic_object_exists") == "semantic_tools"
        assert registry.get("read_query") == "db_tools"
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
        assert "analyze_table_relationships" in tool_names, f"Missing analyze_table_relationships, got: {tool_names}"
        assert "get_multiple_tables_ddl" in tool_names, f"Missing get_multiple_tables_ddl, got: {tool_names}"
        assert "analyze_column_usage_patterns" in tool_names, (
            f"Missing analyze_column_usage_patterns, got: {tool_names}"
        )
        assert "analyze_metric_candidates_from_history" in tool_names, (
            f"Missing analyze_metric_candidates_from_history, got: {tool_names}"
        )
        assert isinstance(node.semantic_discovery_tools, SemanticDiscoveryTools)

    def test_semantic_discovery_tools_skipped_when_no_db(self, real_agent_config, mock_llm_create):
        """When DBFuncTool() constructor fails, semantic_discovery_tools is None but node still works."""
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
        assert "analyze_table_relationships" not in tool_names
        assert node.semantic_discovery_tools is None
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
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "revenue_metrics.yml"
        assert sem_model == ["model.yml"]
        assert status is None
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
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "sales_metrics.yml"
        assert sem_model == ["model.yml"]
        assert status is None
        assert out == "Done"

    def test_extracts_status_from_dict_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "skipped",
                "output": "All requested metrics already exist; skipped per Step 4.",
            }
        }
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "skipped"
        assert out.startswith("All requested metrics already exist")

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
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "generated"
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
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "done"
        assert out == "Done."

    def test_extracts_status_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "semantic_model_file": "model.yml",
                "metric_file": None,
                "status": "skipped",
                "output": "Skipped: metric already exists.",
            }
        )
        output = {"content": content}
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert sem_model == ["model.yml"]
        assert metric_file is None
        assert status == "skipped"
        assert out == "Skipped: metric already exists."

    def test_returns_none_quad_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None
        assert sem_model is None
        assert status is None
        assert out is None

    def test_returns_none_quad_on_dict_missing_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None
        assert status is None

    def test_returns_none_quad_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "not json at all !!!"}
        sem_model, metric_file, status, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None
        assert status is None


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

    def test_osi_authoring_uses_osi_prompt_template(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.input = SemanticNodeInput(user_message="Generate OSI metrics", prompt_version="1.2")

        with (
            patch("datus.agent.node.semantic_authoring.osi_prompt_version", return_value="osi-latest") as version_mock,
            patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm,
        ):
            mock_pm.return_value.render_template.return_value = "osi prompt"

            node._get_system_prompt(template_context={})

        version_mock.assert_called_once_with(real_agent_config, "gen_metrics", "1.2")
        call_kwargs = mock_pm.return_value.render_template.call_args.kwargs
        assert call_kwargs["template_name"] == "gen_metrics_osi_system"
        assert call_kwargs["version"] == "osi-latest"


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
            return_value=FuncToolResult(result={"columns": ["sql"], "data": [], "metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock(
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
        node.generation_tools.end_metric_generation.assert_called_once_with(
            metric_file=str(metric_path),
            semantic_model_files=[str(real_agent_config.path_manager.semantic_model_path(datasource) / "orders.yml")],
        )

    def test_final_metric_publish_requires_grouped_source_sql_dry_run(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult
        from datus.utils.exceptions import DatusException

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
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        with pytest.raises(DatusException, match="source SQL group-by dimensions"):
            node._finalize_metric_generation(reported_semantic_path, reported_metric_path, "generated")

        node.semantic_tools.query_metrics.assert_called_once_with(metrics=["revenue_total"], dry_run=True)
        node.generation_tools.end_metric_generation.assert_not_called()

    def test_final_metric_publish_accepts_grouped_source_sql_dry_run(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

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
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.generation_evidence.record_metric_dry_run(
            ["revenue_total"],
            FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}}),
            dimensions=["customer_segment"],
        )
        node.generation_tools.end_metric_generation = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(reported_semantic_path, reported_metric_path, "generated")

        node.generation_tools.end_metric_generation.assert_called_once_with(
            metric_file=str(metric_path),
            semantic_model_files=[str(real_agent_config.path_manager.semantic_model_path(datasource) / "orders.yml")],
        )

    def test_osi_final_metric_publish_uses_semantic_model_files(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        semantic_dir = real_agent_config.path_manager.semantic_model_path(datasource)
        metric_dir = semantic_dir / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_path = metric_dir / "orders_metrics.yml"
        metric_path.write_text(
            """
version: 0.2.0.dev0
semantic_model:
  - name: shop
    datasets:
      - name: orders
        source: orders
    metrics:
      - name: order_count
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: COUNT(DISTINCT order_id)
""",
            encoding="utf-8",
        )
        reported_semantic_path = f"subject/semantic_models/{datasource}/orders.yml"
        reported_metric_path = f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml"

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation([reported_semantic_path], reported_metric_path, "generated")

        node.semantic_tools.query_metrics.assert_called_once_with(metrics=["order_count"], dry_run=True)
        node.generation_tools.end_metric_generation.assert_called_once_with(
            metric_file=str(metric_path),
            semantic_model_files=[str(semantic_dir / "orders.yml")],
        )

    def test_osi_final_metric_publish_handles_string_semantic_model_file(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        semantic_dir = real_agent_config.path_manager.semantic_model_path(datasource)
        metric_dir = semantic_dir / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_path = metric_dir / "orders_metrics.yml"
        metric_path.write_text(
            """
version: 0.2.0.dev0
semantic_model:
  - name: shop
    metrics:
      - name: order_count
""",
            encoding="utf-8",
        )

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.input = SemanticNodeInput(user_message="Generate OSI metrics")
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock(return_value=FuncToolResult(result={"message": "ok"}))

        node._finalize_metric_generation(
            f"subject/semantic_models/{datasource}/orders.yml",
            f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml",
            "generated",
        )

        node.generation_tools.end_metric_generation.assert_called_once_with(
            metric_file=str(metric_path),
            semantic_model_files=[str(semantic_dir / "orders.yml")],
        )

    def test_osi_final_metric_publish_skips_when_already_synced(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.generation_evidence.mark_kb_sync("metric")
        node.generation_tools.end_metric_generation = MagicMock()

        node._finalize_metric_generation(None, "orders_metrics.yml", "generated")

        node.generation_tools.end_metric_generation.assert_not_called()

    def test_osi_skipped_status_without_metric_file_returns_cleanly(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.generation_tools.end_metric_generation = MagicMock()

        node._finalize_metric_generation(None, None, "skipped")

        node.generation_tools.end_metric_generation.assert_not_called()

    def test_osi_skipped_status_with_metric_file_fails_closed(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"

        with pytest.raises(RuntimeError, match="status='skipped' with a non-null metric_file"):
            node._finalize_metric_generation(None, "orders_metrics.yml", "skipped")

    def test_osi_generated_status_without_metric_file_fails_closed(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"

        with pytest.raises(RuntimeError, match="status='generated' without a metric_file"):
            node._finalize_metric_generation(None, None, "generated")

    def test_osi_empty_final_response_without_metric_file_is_noop(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.generation_tools.end_metric_generation = MagicMock()

        node._finalize_metric_generation(None, None, None)

        node.generation_tools.end_metric_generation.assert_not_called()

    def test_osi_final_metric_publish_requires_generation_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.generation_tools = None

        with pytest.raises(RuntimeError, match="generation tools are unavailable"):
            node._finalize_metric_generation(None, "orders_metrics.yml", "generated")

    def test_osi_final_metric_publish_requires_validation_tool(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.semantic_tools = None

        with pytest.raises(RuntimeError, match="validate_semantic is unavailable"):
            node._finalize_metric_generation(None, "orders_metrics.yml", "generated")

    def test_osi_final_metric_publish_reports_validation_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(
            return_value=FuncToolResult(success=0, error="invalid OSI", result={"valid": False})
        )

        with pytest.raises(RuntimeError, match="validate_semantic failed"):
            node._finalize_metric_generation(None, "orders_metrics.yml", "generated")

    def test_osi_final_metric_publish_requires_query_metrics_tool(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "orders_metrics.yml").write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n",
            encoding="utf-8",
        )

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.generation_evidence.validation_passed = True
        node.semantic_tools = None

        with pytest.raises(RuntimeError, match="query_metrics is unavailable"):
            node._finalize_metric_generation(
                None,
                f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml",
                "generated",
            )

    def test_osi_final_metric_publish_reports_dry_run_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "orders_metrics.yml").write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n",
            encoding="utf-8",
        )

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(return_value=FuncToolResult(success=0, error="compile failed"))

        with pytest.raises(RuntimeError, match="query_metrics\\(dry_run=True\\) failed"):
            node._finalize_metric_generation(
                None, f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml", "generated"
            )

    def test_osi_final_metric_publish_requires_queryability_contracts(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult
        from datus.utils.exceptions import DatusException

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "revenue_metrics.yml").write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: revenue_total\n",
            encoding="utf-8",
        )

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.input = SemanticNodeInput(
            user_message="SELECT customer_segment, SUM(revenue) FROM orders GROUP BY customer_segment"
        )
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock()

        with pytest.raises(DatusException, match="source SQL group-by dimensions"):
            node._finalize_metric_generation(
                None,
                f"subject/semantic_models/{datasource}/metrics/revenue_metrics.yml",
                "generated",
            )

        node.generation_tools.end_metric_generation.assert_not_called()

    def test_osi_final_metric_publish_reports_sync_failure(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.tools.func_tool.base import FuncToolResult

        datasource = real_agent_config.current_datasource
        metric_dir = real_agent_config.path_manager.semantic_model_path(datasource) / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "orders_metrics.yml").write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n",
            encoding="utf-8",
        )

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.node_config["authoring_format"] = "osi"
        node.semantic_tools = MagicMock()
        node.semantic_tools.validate_semantic = MagicMock(return_value=FuncToolResult(result={"valid": True}))
        node.semantic_tools.query_metrics = MagicMock(
            return_value=FuncToolResult(result={"metadata": {"sql": "SELECT 1"}})
        )
        node.generation_tools.end_metric_generation = MagicMock(
            return_value=FuncToolResult(success=0, error="sync failed")
        )

        with pytest.raises(RuntimeError, match="OSI metric KB sync failed"):
            node._finalize_metric_generation(
                None,
                f"subject/semantic_models/{datasource}/metrics/orders_metrics.yml",
                "generated",
            )

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
        node.generation_tools.end_metric_generation = MagicMock(
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
        node.generation_tools.end_metric_generation.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_explicit_non_skipped_status_without_metric_file_fails_closed(
        self, real_agent_config, mock_llm_create
    ):
        """Any explicit non-skipped final status must not silently bypass publishing."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    json.dumps(
                        {
                            "semantic_model_file": "orders.yml",
                            "metric_file": None,
                            "status": "done",
                            "output": "Done.",
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
        assert "status='done' without a metric_file" in actions[-1].output["error"]


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

        node = GenMetricsAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        # Workflow mode may now compose CompositeHooks (permission + compact)
        # because multi-turn history is enabled for all modes. Validate the
        # permission gate via ``node.permission_hooks`` instead of the bundle.
        hooks = node._compose_hooks()
        assert hooks is not None
        assert node.permission_hooks is not None
        assert node.permission_hooks.non_interactive is True
        assert node.permission_manager.active_profile == "dangerous"
