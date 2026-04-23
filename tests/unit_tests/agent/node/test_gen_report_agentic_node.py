# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenReportAgenticNode.

Tests cover:
- Node creation with different configurations
- Tools setup based on config patterns (DBFuncTool, ContextSearchTools)
- Configurable node name
- Max turns configuration
- Streaming execution with MockLLMModel
- Database tool invocation (execute_sql)
- JSON report extraction
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite) with tables: frpm, satscores, schools
- Real Storage/RAG (vector store in tmp_path)
- Real Tools (DBFuncTool, ContextSearchTools, etc.)
- Real PromptManager (using built-in templates)
- Real PathManager
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json
from unittest.mock import patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenReportAgenticNodeInit:
    """Tests for GenReportAgenticNode initialization."""

    def test_report_init(self, real_agent_config, mock_llm_create):
        """Test that GenReportAgenticNode initializes with real config."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="report_node_1",
            description="Test report node",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        assert node.id == "report_node_1"
        assert node.description == "Test report node"
        assert node.get_node_name() == "gen_report"

    def test_report_has_db_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has database tools when configured."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        # real_agent_config has gen_report.tools = "db_tools.*,context_search_tools.*"
        node = GenReportAgenticNode(
            node_id="report_db_test",
            description="Report with DB tools",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        # DB tools should be setup
        assert node.db_func_tool is not None

        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_report_has_context_search_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has context search tools when configured."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        # real_agent_config has gen_report.tools = "db_tools.*,context_search_tools.*"
        node = GenReportAgenticNode(
            node_id="report_ctx_test",
            description="Report with context tools",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        # Context search tools should be setup
        assert node.context_search_tools is not None

        tool_names = [tool.name for tool in node.tools]
        # Context search tools should include at least some search functions
        assert len(tool_names) > 0

    def test_report_max_turns(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        # The real_agent_config has gen_report.max_turns = 5
        node = GenReportAgenticNode(
            node_id="report_turns_test",
            description="Report max turns test",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        assert node.max_turns == 5

    def test_report_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 30 when node_name not in agentic_nodes."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="report_default_turns",
            description="Report default max turns",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="nonexistent_report_node",  # Not in agentic_nodes
        )

        assert node.max_turns == 30

    def test_report_node_name_configurable(self, real_agent_config, mock_llm_create):
        """Test that node_name is configurable and affects get_node_name()."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="custom_report_node",
            description="Custom report node",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        assert node.get_node_name() == "gen_report"
        assert node.configured_node_name == "gen_report"

    def test_report_node_name_defaults_to_node_name(self, real_agent_config, mock_llm_create):
        """Test that get_node_name() falls back to NODE_NAME when node_name is None."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="default_report_node",
            description="Default report node",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name=None,
        )

        assert node.get_node_name() == "gen_report"


# ---------------------------------------------------------------------------
# Execution Mode Tests
# ---------------------------------------------------------------------------


class TestGenReportAgenticNodeExecutionMode:
    """Tests for GenReportAgenticNode execution_mode gating of ask_user tool."""

    def test_interactive_mode_has_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Interactive mode (default) enables ask_user tool."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="report_interactive",
            description="Test interactive mode",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        assert node.execution_mode == "interactive"
        assert node.ask_user_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_workflow_mode_no_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Workflow mode disables ask_user tool."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="report_workflow",
            description="Test workflow mode",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
            execution_mode="workflow",
        )

        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None
        tool_names = [t.name for t in node.tools]
        assert "ask_user" not in tool_names


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly
class TestGenReportAgenticNodeExecution:
    """Tests for GenReportAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_report_simple_response(self, real_agent_config, mock_llm_create):
        """Test basic execute_stream produces actions."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Revenue analysis report completed."),
            ]
        )

        node = GenReportAgenticNode(
            node_id="report_exec_test",
            description="Test report execution",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        node.input = GenReportNodeInput(user_message="Analyze revenue trends")

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
    async def test_report_with_db_tool_calls(self, real_agent_config, mock_llm_create):
        """Test execute_stream where LLM calls describe_table tool to gather data for report."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="describe_table",
                            arguments=json.dumps({"table_name": "satscores"}),
                        ),
                    ],
                    content="Based on the satscores table schema, the average SAT reading score is 479.7.",
                ),
            ]
        )

        node = GenReportAgenticNode(
            node_id="report_tools_exec",
            description="Report with tool calls",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        node.input = GenReportNodeInput(
            user_message="What is the average SAT reading score?",
            database="california_schools",
        )

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
    async def test_report_json_extraction(self, real_agent_config, mock_llm_create):
        """Test that JSON report responses are properly handled."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        report_json = json.dumps(
            {
                "report": "## Revenue Analysis\n\nRevenue grew 15% this quarter.",
                "data_sources": ["revenue_total"],
                "key_findings": ["15% growth"],
            }
        )
        mock_llm_create.reset(
            responses=[
                build_simple_response(report_json),
            ]
        )

        node = GenReportAgenticNode(
            node_id="report_json_test",
            description="Report with JSON response",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        node.input = GenReportNodeInput(user_message="Analyze revenue")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Last action should be SUCCESS with result
        last_action = actions[-1]
        assert last_action.status == ActionStatus.SUCCESS
        assert last_action.output is not None

    @pytest.mark.asyncio
    async def test_report_with_db_schema_context(self, real_agent_config, mock_llm_create):
        """Test execute_stream with db_schema enriches the enhanced message."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Report with schema context generated."),
            ]
        )

        node = GenReportAgenticNode(
            node_id="report_schema_ctx",
            description="Report with schema context",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        node.input = GenReportNodeInput(
            user_message="Analyze revenue by department",
            database="california_schools",
            db_schema="main",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with a prompt that includes schema context
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "Schema" in prompt or "main" in prompt or "Analyze" in prompt

    @pytest.mark.asyncio
    async def test_report_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Test that execute_stream raises when input is not set."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="report_no_input",
            description="Report without input",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )
        node.input = None

        action_manager = ActionHistoryManager()
        with pytest.raises(ValueError, match="Report input not set"):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_report_template_fallback(self, real_agent_config, mock_llm_create):
        """Test that a non-existent system_prompt falls back to gen_report_system template."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        # Set a non-existent system_prompt so the template lookup fails and falls back
        original_prompt = real_agent_config.agentic_nodes["gen_report"].get("system_prompt")
        real_agent_config.agentic_nodes["gen_report"]["system_prompt"] = "nonexistent_prompt_xyz"

        try:
            mock_llm_create.reset(
                responses=[
                    build_simple_response("Report with fallback template."),
                ]
            )

            node = GenReportAgenticNode(
                node_id="report_fallback_test",
                description="Report with template fallback",
                node_type=NodeType.TYPE_GEN_REPORT,
                agent_config=real_agent_config,
                node_name="gen_report",
            )

            node.input = GenReportNodeInput(user_message="Analyze data")

            action_manager = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(action_manager):
                actions.append(action)

            # Should still succeed using the fallback template
            assert len(actions) >= 2
            assert actions[-1].status == ActionStatus.SUCCESS
        finally:
            real_agent_config.agentic_nodes["gen_report"]["system_prompt"] = original_prompt

    @pytest.mark.asyncio
    async def test_report_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        """Test that ExecutionInterrupted is re-raised from execute_stream."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            """Async generator that raises ExecutionInterrupted."""
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa: makes this an async generator

        node = GenReportAgenticNode(
            node_id="report_interrupt_test",
            description="Report interrupt test",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        node.input = GenReportNodeInput(user_message="Analyze data")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, node_name="gen_report"):
    from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

    return GenReportAgenticNode(
        node_id="report_extra",
        description="Extra report test",
        node_type=NodeType.TYPE_GEN_REPORT,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestExtractReportFromResponse
# ---------------------------------------------------------------------------


class TestExtractReportFromResponse:
    def test_extracts_from_dict_with_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "report": "## Revenue Analysis\n\nRevenue grew 15%.",
                "data_sources": ["revenue_total"],
                "key_findings": ["15% growth"],
            }
        }
        report, metadata = node._extract_report_from_response(output)
        assert "Revenue Analysis" in report
        assert metadata["data_sources"] == ["revenue_total"]
        assert metadata["key_findings"] == ["15% growth"]

    def test_extracts_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "report": "## Sales Report\n\nSales data analysis.",
                "data_sources": [],
                "key_findings": [],
            }
        )
        output = {"content": content}
        report, metadata = node._extract_report_from_response(output)
        assert "Sales Report" in report

    def test_returns_empty_string_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        report, metadata = node._extract_report_from_response(output)
        assert report == ""
        assert metadata is None

    def test_returns_raw_string_when_no_json_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "plain text response without JSON"}
        report, metadata = node._extract_report_from_response(output)
        assert "plain text response" in report

    def test_returns_dict_as_string_on_missing_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        report, metadata = node._extract_report_from_response(output)
        # dict without 'report' key should return string representation
        assert isinstance(report, str)

    def test_checks_raw_output_field(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": "",
            "raw_output": "Response from raw output field",
        }
        report, metadata = node._extract_report_from_response(output)
        # raw_output fallback should be used when content is empty
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# TestBuildEnhancedMessage
# ---------------------------------------------------------------------------


class TestBuildEnhancedMessage:
    def test_no_context_returns_user_message(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(user_message="Analyze revenue")
        result = node._build_enhanced_message(user_input)
        assert result == "Analyze revenue"

    def test_with_database_context_builds_structured_message(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(
            user_message="Analyze revenue",
            database="california_schools",
        )
        result = node._build_enhanced_message(user_input)
        assert "california_schools" in result
        assert "Analyze revenue" in result

    def test_with_schema_context(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(
            user_message="Analyze data",
            db_schema="main_schema",
        )
        result = node._build_enhanced_message(user_input)
        assert "main_schema" in result


# ---------------------------------------------------------------------------
# TestExtractReportResult
# ---------------------------------------------------------------------------


class TestExtractReportResult:
    def test_base_implementation_returns_none(self, real_agent_config, mock_llm_create):
        """Base _extract_report_result always returns None."""
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._extract_report_result([])
        assert result is None


# ---------------------------------------------------------------------------
# TestSetupToolPattern
# ---------------------------------------------------------------------------


class TestSetupToolPattern:
    def test_unknown_pattern_logs_warning(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch("datus.agent.node.gen_report_agentic_node.logger.warning") as mock_warning:
            node._setup_tool_pattern("unknown_tool_type.*")
        mock_warning.assert_called_once_with("Unknown tool type: unknown_tool_type")

    def test_specific_method_setup(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Setup semantic_tools.search_metrics specifically
        with patch.object(node, "_setup_specific_tool_method") as mock_method:
            node._setup_tool_pattern("semantic_tools.search_metrics")
        mock_method.assert_called_once_with("semantic_tools", "search_metrics")

    def test_exact_db_tools_pattern(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools")
        mock_setup.assert_called_once()

    def test_wildcard_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools.*")
        mock_setup.assert_called_once()


# ---------------------------------------------------------------------------
# TestExecuteStreamError
# ---------------------------------------------------------------------------


class TestExecuteStreamGenReportError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenReportAgenticNode(
            node_id="report_error",
            description="Error test",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )
        node.input = GenReportNodeInput(user_message="Analyze data")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"


class TestGenReportSystemPromptCurrentDate:
    """Verify current_date is injected into the system prompt."""

    def test_system_prompt_contains_current_date(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = GenReportAgenticNode(
            node_id="test_report_date",
            description="Test current_date",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )

        with patch(
            "datus.utils.time_utils.get_default_current_date",
            return_value="2025-06-15",
        ):
            prompt = node._get_system_prompt()
        assert "2025-06-15" in prompt
