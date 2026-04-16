# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SchedulerAgenticNode.

Tests cover:
- Node creation with/without scheduler_config
- Tools setup (scheduler tools only, no DB/BI/filesystem tools exposed)
- Max turns configuration
- Node name
- Graceful handling when no scheduler config present
- Registration in node.py, node_type.py, sub_agent_task_tool.py
- execute_stream happy path, error handling, ExecutionInterrupted
- _prepare_template_context and _fallback_system_prompt

Design principle: Mock SchedulerTools since datus-scheduler-core is optional.
- Real AgentConfig (from conftest `real_agent_config`)
- The ONLY LLM mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Patch path for SchedulerTools (imported inside method body via
# `from datus.tools.func_tool.scheduler_tools import SchedulerTools`)
_SCHEDULER_TOOLS_PATCH = "datus.tools.func_tool.scheduler_tools.SchedulerTools"

# ---- Helper: add scheduler config to agent_config ----


def _make_mock_scheduler_tools():
    """Create a mock SchedulerTools that returns fake tools."""
    mock_tools = MagicMock()
    # Create fake tool objects with .name attribute
    fake_tool_names = [
        "submit_sql_job",
        "submit_sparksql_job",
        "trigger_scheduler_job",
        "pause_job",
        "resume_job",
        "delete_job",
        "update_job",
        "get_scheduler_job",
        "list_scheduler_jobs",
        "list_scheduler_connections",
        "list_job_runs",
        "get_run_log",
    ]
    fake_tools = []
    for name in fake_tool_names:
        tool = MagicMock()
        tool.name = name
        fake_tools.append(tool)
    mock_tools.available_tools.return_value = fake_tools
    return mock_tools


def _add_scheduler_config(agent_config, max_turns=25):
    """Add scheduler config and scheduler agentic node config to an AgentConfig."""
    agent_config.scheduler_config = {
        "name": "airflow_local",
        "type": "airflow",
        "api_base_url": "http://localhost:8080/api/v1",
        "username": "admin",
        "password": "admin123",
        "dags_folder": "/tmp/dags",
    }
    agent_config.agentic_nodes["scheduler"] = {
        "system_prompt": "scheduler",
        "max_turns": max_turns,
    }


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestSchedulerAgenticNodeInit:
    """Tests for SchedulerAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.NODE_NAME == "scheduler"
            assert node.get_node_name() == "scheduler"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.agentic_node import AgenticNode
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.id == "scheduler_node"

    def test_max_turns_from_config(self, real_agent_config, mock_llm_create):
        _add_scheduler_config(real_agent_config, max_turns=25)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.max_turns == 25

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        """Default max_turns is 30 when scheduler not in agentic_nodes."""
        # Add scheduler_config but no scheduler agentic node config
        real_agent_config.scheduler_config = {
            "name": "airflow_local",
            "type": "airflow",
            "api_base_url": "http://localhost:8080/api/v1",
        }
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.max_turns == 30


# ---------------------------------------------------------------------------
# Tool Setup Tests
# ---------------------------------------------------------------------------


class TestSchedulerToolSetup:
    """Tests for tool setup — scheduler tools only, no DB/BI/filesystem tools exposed."""

    def test_has_scheduler_tools_with_config(self, real_agent_config, mock_llm_create):
        """With scheduler_config, node should have scheduler tools."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "submit_sql_job" in tool_names
            assert "submit_sparksql_job" in tool_names
            assert "trigger_scheduler_job" in tool_names
            assert "get_scheduler_job" in tool_names
            assert "list_scheduler_jobs" in tool_names

    def test_no_db_tools_exposed(self, real_agent_config, mock_llm_create):
        """DB tools should NOT be in tools list."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "list_tables" not in tool_names
            assert "describe_table" not in tool_names
            assert "read_query" not in tool_names
            assert "get_table_ddl" not in tool_names

    def test_no_bi_tools_exposed(self, real_agent_config, mock_llm_create):
        """BI tools should NOT be in tools list."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "list_dashboards" not in tool_names
            assert "create_dashboard" not in tool_names

    def test_no_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Filesystem tools should NOT be in tools list."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "read_file" not in tool_names
            assert "write_file" not in tool_names

    def test_no_scheduler_config_no_tools(self, real_agent_config, mock_llm_create):
        """Without scheduler_config, node should have 0 scheduler tools (graceful no-op)."""
        # Ensure no scheduler_config
        real_agent_config.scheduler_config = None
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]

        assert "submit_sql_job" not in tool_names
        assert "trigger_scheduler_job" not in tool_names

    def test_import_error_yields_no_tools(self, real_agent_config, mock_llm_create):
        """When scheduler tools import fails, node should have no scheduler tools."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, side_effect=ImportError("datus-scheduler-core not installed")):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert "submit_sql_job" not in [t.name for t in node.tools]


# ---------------------------------------------------------------------------
# Registration Coverage Tests (node.py, node_type.py, sub_agent_task_tool.py)
# ---------------------------------------------------------------------------


class TestSchedulerRegistration:
    """Tests covering the registration code paths in existing files."""

    def test_node_type_input(self):
        """NodeType.type_input should return SchedulerNodeInput for TYPE_SCHEDULER."""
        from datus.configuration.node_type import NodeType
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

        result = NodeType.type_input(NodeType.TYPE_SCHEDULER, {"user_message": "test"})
        assert isinstance(result, SchedulerNodeInput)
        assert result.user_message == "test"

    def test_node_type_input_optional(self):
        """NodeType.type_input with ignore_require_check should accept empty input."""
        from datus.configuration.node_type import NodeType
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

        result = NodeType.type_input(NodeType.TYPE_SCHEDULER, {}, ignore_require_check=True)
        assert result is not None
        assert isinstance(result, SchedulerNodeInput)
        assert result.database is None

    def test_node_factory_creates_scheduler(self, real_agent_config, mock_llm_create):
        """Node.new_instance should create SchedulerAgenticNode for TYPE_SCHEDULER."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.node import Node
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
            from datus.configuration.node_type import NodeType

            node = Node.new_instance(
                node_id="test_factory",
                description="Factory test",
                node_type=NodeType.TYPE_SCHEDULER,
                agent_config=real_agent_config,
            )
            assert isinstance(node, SchedulerAgenticNode)
            assert node.get_node_name() == "scheduler"

    def test_sub_agent_resolve_node_type(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._resolve_node_type should resolve scheduler."""
        from datus.configuration.node_type import NodeType
        from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        node_type, node_name = tool._resolve_node_type("scheduler")
        assert node_type == NodeType.TYPE_SCHEDULER
        assert node_name == "scheduler"

    def test_sub_agent_create_builtin_node(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._create_builtin_node should create SchedulerAgenticNode."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
            from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

            tool = SubAgentTaskTool(agent_config=real_agent_config)
            node = tool._create_builtin_node("scheduler")
            assert isinstance(node, SchedulerAgenticNode)
            assert node.get_node_name() == "scheduler"

    def test_sub_agent_build_node_input(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._build_node_input should return SchedulerNodeInput."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
            from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput
            from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

            tool = SubAgentTaskTool(agent_config=real_agent_config)
            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            result = tool._build_node_input(node, "List all jobs")
            assert isinstance(result, SchedulerNodeInput)
            assert result.user_message == "List all jobs"
            assert result.database == real_agent_config.current_database

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        """Node.new_instance with input_data should set node.input."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType
            from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

            input_data = SchedulerNodeInput(user_message="Submit a daily job")
            node = Node.new_instance(
                node_id="test_factory_input",
                description="Factory test",
                node_type=NodeType.TYPE_SCHEDULER,
                input_data=input_data,
                agent_config=real_agent_config,
            )
            assert node.input is not None
            assert node.input.user_message == "Submit a daily job"

    def test_from_dict_input_deserialization(self, real_agent_config, mock_llm_create):
        """Node.from_dict should deserialize SchedulerNodeInput from dict."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node_dict = {
                "id": "test_from_dict",
                "description": "From dict test",
                "type": NodeType.TYPE_SCHEDULER,
                "input": {"user_message": "List jobs", "database": "test_db"},
                "result": None,
                "status": "completed",
                "start_time": None,
                "end_time": None,
                "dependencies": [],
                "metadata": {},
            }
            node = Node.from_dict(node_dict, agent_config=real_agent_config)
            assert node.input is not None
            assert node.input.user_message == "List jobs"

    def test_from_dict_result_deserialization(self, real_agent_config, mock_llm_create):
        """Node.from_dict should deserialize SchedulerNodeResult from dict."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node_dict = {
                "id": "test_from_dict_result",
                "description": "From dict result test",
                "type": NodeType.TYPE_SCHEDULER,
                "input": None,
                "result": {
                    "success": True,
                    "response": "Job submitted",
                    "scheduler_result": {"job_id": "dag_123"},
                    "tokens_used": 300,
                },
                "status": "completed",
                "start_time": None,
                "end_time": None,
                "dependencies": [],
                "metadata": {},
            }
            node = Node.from_dict(node_dict, agent_config=real_agent_config)
            assert node.result is not None
            assert node.result.response == "Job submitted"
            assert node.result.scheduler_result == {"job_id": "dag_123"}


# ---------------------------------------------------------------------------
# execute_stream Tests
# ---------------------------------------------------------------------------


class TestSchedulerExecuteStream:
    """Tests for execute_stream method."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        """execute_stream should raise DatusException when self.input is None."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
            from datus.utils.exceptions import DatusException

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.input is None

            with pytest.raises(DatusException):
                async for _ in node.execute_stream():
                    pass

    @pytest.mark.asyncio
    async def test_execute_stream_happy_path(self, real_agent_config, mock_llm_create):
        """execute_stream should yield actions and produce a successful result."""
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput
        from tests.unit_tests.mock_llm_model import build_simple_response

        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = SchedulerNodeInput(user_message="List all jobs")

            mock_llm_create.reset(responses=[build_simple_response("Here are your scheduled jobs: ...")])

            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

            assert len(actions) >= 2  # At least user action + final action
            # First action is user input
            assert actions[0].role == ActionRole.USER
            # Last action should be success
            assert actions[-1].status == ActionStatus.SUCCESS
            assert actions[-1].action_type == "scheduler_response"

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        """execute_stream should yield error action on exception."""
        from datus.schemas.action_history import ActionHistoryManager, ActionStatus
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = SchedulerNodeInput(user_message="Submit a job")

            # Force an error in the model
            mock_llm_create.reset(responses=[])
            with patch.object(node.model, "generate_with_tools_stream", side_effect=RuntimeError("LLM unavailable")):
                ahm = ActionHistoryManager()
                actions = []
                async for action in node.execute_stream(ahm):
                    actions.append(action)

                assert actions[-1].status == ActionStatus.FAILED
                assert actions[-1].action_type == "error"

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_execution_interrupted(self, real_agent_config, mock_llm_create):
        """execute_stream should propagate ExecutionInterrupted without catching it."""
        from datus.cli.execution_state import ExecutionInterrupted
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = SchedulerNodeInput(user_message="Submit a job")
            with patch.object(node.model, "generate_with_tools_stream", side_effect=ExecutionInterrupted()):
                with pytest.raises(ExecutionInterrupted):
                    async for _ in node.execute_stream():
                        pass


# ---------------------------------------------------------------------------
# Template Context Tests
# ---------------------------------------------------------------------------


class TestSchedulerTemplateContext:
    """Tests for _prepare_template_context and _fallback_system_prompt."""

    def test_context_with_tools(self, real_agent_config, mock_llm_create):
        """With scheduler tools, native_tools should list tool names."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            ctx = node._prepare_template_context()
            assert "submit_sql_job" in ctx["native_tools"]
            assert ctx["has_ask_user_tool"] is False  # workflow mode

    def test_context_without_tools(self, real_agent_config, mock_llm_create):
        """Without scheduler config, native_tools should be 'None'."""
        real_agent_config.scheduler_config = None
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        ctx = node._prepare_template_context()
        assert ctx["native_tools"] == "None"

    def test_fallback_system_prompt(self, real_agent_config, mock_llm_create):
        """Fallback prompt should mention scheduler specialist and Airflow."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            prompt = node._fallback_system_prompt({})
            assert "scheduler" in prompt.lower()
            assert "Airflow" in prompt


# ---------------------------------------------------------------------------
# Custom node_name Tests (P2-2: alias subagent config lookup)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestSchedulerCustomNodeName:
    """Tests for custom node_name support (e.g. my_scheduler: {node_class: scheduler})."""

    def test_default_node_name(self, real_agent_config, mock_llm_create):
        """Without node_name, get_node_name() returns NODE_NAME."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.get_node_name() == "scheduler"

    def test_custom_node_name(self, real_agent_config, mock_llm_create):
        """With node_name, get_node_name() returns the custom name."""
        _add_scheduler_config(real_agent_config)
        real_agent_config.agentic_nodes["my_scheduler"] = {
            "system_prompt": "scheduler",
            "max_turns": 15,
        }
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(
                agent_config=real_agent_config, execution_mode="workflow", node_name="my_scheduler"
            )
            assert node.get_node_name() == "my_scheduler"
            assert node.max_turns == 15

    def test_custom_node_name_reads_own_config(self, real_agent_config, mock_llm_create):
        """Custom node_name should read its own agentic_nodes config, not the builtin one."""
        _add_scheduler_config(real_agent_config, max_turns=25)  # scheduler max_turns=25
        real_agent_config.agentic_nodes["etl_scheduler"] = {
            "system_prompt": "scheduler",
            "max_turns": 50,
        }
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(
                agent_config=real_agent_config, execution_mode="workflow", node_name="etl_scheduler"
            )
            assert node.max_turns == 50  # from etl_scheduler, not scheduler's 25

    def test_node_factory_passes_node_name(self, real_agent_config, mock_llm_create):
        """Node.new_instance() should pass node_name to SchedulerAgenticNode."""
        _add_scheduler_config(real_agent_config)
        real_agent_config.agentic_nodes["my_jobs"] = {
            "system_prompt": "scheduler",
            "max_turns": 10,
        }
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node = Node.new_instance(
                node_id="test_factory",
                description="Factory test",
                node_type=NodeType.TYPE_SCHEDULER,
                agent_config=real_agent_config,
                node_name="my_jobs",
            )
            assert node.get_node_name() == "my_jobs"
            assert node.max_turns == 10


# ---------------------------------------------------------------------------
# prompt_version passthrough Tests (P3)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestSchedulerPromptVersion:
    """Tests for prompt_version passthrough from input to _get_system_prompt."""

    def test_prompt_version_from_input(self, real_agent_config, mock_llm_create):
        """Input prompt_version should override node_config prompt_version."""
        _add_scheduler_config(real_agent_config)
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "test prompt"
                node._get_system_prompt(prompt_version="2.0")
                call_kwargs = mock_pm.return_value.render_template.call_args
                version = call_kwargs.kwargs.get("version")
                assert version == "2.0", f"Expected version '2.0', got '{version}'"

    def test_prompt_version_fallback_to_config(self, real_agent_config, mock_llm_create):
        """Without input prompt_version, should use node_config value."""
        _add_scheduler_config(real_agent_config)
        real_agent_config.agentic_nodes["scheduler"]["prompt_version"] = "1.5"
        with patch(_SCHEDULER_TOOLS_PATCH, return_value=_make_mock_scheduler_tools()):
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            node = SchedulerAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "test prompt"
                node._get_system_prompt(prompt_version=None)
                call_kwargs = mock_pm.return_value.render_template.call_args
                version = call_kwargs.kwargs.get("version")
                assert version == "1.5", f"Expected version '1.5', got '{version}'"


# ---------------------------------------------------------------------------
# Partial Resource Collection Tests
# ---------------------------------------------------------------------------


class TestCollectSubmittedJobs:
    """Tests for SchedulerAgenticNode._collect_submitted_jobs."""

    @staticmethod
    def _make_action(action_type, status, output=None):
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        return ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type=action_type,
            messages="",
            input_data={},
            output_data=output,
            status=ActionStatus.SUCCESS if status == "success" else ActionStatus.FAILED,
        )

    def test_collects_submitted_sql_jobs(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
        from datus.schemas.action_history import ActionHistoryManager

        ahm = ActionHistoryManager()
        ahm.add_action(
            self._make_action(
                "submit_sql_job",
                "success",
                {"result": {"job_id": "daily_report", "job_name": "daily_report", "status": "active"}},
            )
        )

        result = SchedulerAgenticNode._collect_submitted_jobs(ahm)

        assert len(result["submitted_jobs"]) == 1
        assert result["submitted_jobs"][0]["job_id"] == "daily_report"

    def test_collects_both_sql_and_sparksql_jobs(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
        from datus.schemas.action_history import ActionHistoryManager

        ahm = ActionHistoryManager()
        ahm.add_action(
            self._make_action(
                "submit_sql_job",
                "success",
                {"result": {"job_id": "job1", "job_name": "job1", "status": "active"}},
            )
        )
        ahm.add_action(
            self._make_action(
                "submit_sparksql_job",
                "success",
                {"result": {"job_id": "job2", "job_name": "job2", "status": "active"}},
            )
        )

        result = SchedulerAgenticNode._collect_submitted_jobs(ahm)

        assert len(result["submitted_jobs"]) == 2

    def test_skips_failed_submissions(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
        from datus.schemas.action_history import ActionHistoryManager

        ahm = ActionHistoryManager()
        ahm.add_action(
            self._make_action(
                "submit_sql_job",
                "success",
                {"result": {"job_id": "ok_job", "job_name": "ok_job", "status": "active"}},
            )
        )
        ahm.add_action(
            self._make_action(
                "submit_sql_job",
                "failed",
                {"result": {"job_id": "bad_job", "job_name": "bad_job", "status": "error"}},
            )
        )

        result = SchedulerAgenticNode._collect_submitted_jobs(ahm)

        assert len(result["submitted_jobs"]) == 1
        assert result["submitted_jobs"][0]["job_id"] == "ok_job"

    def test_returns_empty_dict_when_no_jobs(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
        from datus.schemas.action_history import ActionHistoryManager

        ahm = ActionHistoryManager()
        ahm.add_action(self._make_action("get_scheduler_job", "success", {"result": {"status": "active"}}))

        result = SchedulerAgenticNode._collect_submitted_jobs(ahm)

        assert result == {}
