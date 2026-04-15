# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenDashboardAgenticNode.

Tests cover:
- Node creation with/without bi_platform config
- Tools setup (BI tools only, no DB tools exposed)
- Auto-detect single platform from dashboard_config
- Max turns configuration
- Node name
- Graceful handling when no BI config present

Design principle: NO mock except LLM + datus_bi_core (optional package).
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite)
- Mock datus_bi_core (optional dependency, not available in CI)
- The ONLY LLM mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import DashboardConfig

# ---- Minimal stubs for datus_bi_core (so tests run without the package) ----


class _AuthParam:
    def __init__(self, **kwargs):
        pass


class _DashboardInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class _ChartInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class _DatasetInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class MockDashboardWriteMixin:
    pass


class MockChartWriteMixin:
    pass


class MockDatasetWriteMixin:
    pass


class FullMockAdapter(MockDashboardWriteMixin, MockChartWriteMixin, MockDatasetWriteMixin):
    """Mock adapter implementing all mixins."""

    def list_dashboards(self, search="", page_size=20):
        return [_DashboardInfo(id=1, name="Test Dashboard")]

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Test", description="", chart_ids=[])

    def list_charts(self, dashboard_id):
        return [_ChartInfo(id=1, name="Chart 1", chart_type="bar")]

    def list_datasets(self, dashboard_id=""):
        return [_DatasetInfo(id=1, name="orders", dialect="postgresql")]

    def get_chart(self, chart_id, dashboard_id=None):
        return _ChartInfo(id=chart_id, name="Test Chart", chart_type="bar")

    def create_dashboard(self, spec):
        return _DashboardInfo(id=10, name=spec.title)

    def update_dashboard(self, dashboard_id, spec):
        return _DashboardInfo(id=dashboard_id, name=spec.title)

    def delete_dashboard(self, dashboard_id):
        return True

    def create_chart(self, spec, dashboard_id=None):
        return _ChartInfo(id=5, name=spec.title, chart_type=spec.chart_type)

    def update_chart(self, chart_id, spec):
        return _ChartInfo(id=chart_id, name=spec.title, chart_type=spec.chart_type)

    def delete_chart(self, chart_id):
        return True

    def add_chart_to_dashboard(self, dashboard_id, chart_id):
        return True

    def create_dataset(self, spec):
        return _DatasetInfo(id=3, name=spec.name, dialect="postgresql")

    def delete_dataset(self, dataset_id):
        return True

    def list_bi_databases(self):
        return [{"id": 1, "name": "PostgreSQL"}]


class ReadOnlyMockAdapter:
    """Mock adapter with only read operations (no write mixins)."""

    def list_dashboards(self, search="", page_size=20):
        return []

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Read Only")

    def list_charts(self, dashboard_id):
        return []

    def list_datasets(self, dashboard_id=""):
        return []


# ---- Build a mock datus_bi_core module ----

_bi_core_mock = MagicMock()
_bi_core_mock.AuthParam = _AuthParam
_bi_core_mock.DashboardWriteMixin = MockDashboardWriteMixin
_bi_core_mock.ChartWriteMixin = MockChartWriteMixin
_bi_core_mock.DatasetWriteMixin = MockDatasetWriteMixin
_bi_core_mock.adapter_registry = MagicMock()


class _MockChartSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _MockDatasetSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _MockDashboardSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_bi_core_mock.models.ChartSpec = _MockChartSpec
_bi_core_mock.models.DatasetSpec = _MockDatasetSpec
_bi_core_mock.models.DashboardSpec = _MockDashboardSpec

_BI_MODULES_PATCH = {
    "datus_bi_core": _bi_core_mock,
    "datus_bi_core.models": _bi_core_mock.models,
}


# ---- Helper: add dashboard config to agent_config ----


def _add_dashboard_config(agent_config, platform="superset"):
    """Add dashboard config and gen_dashboard agentic node config to an AgentConfig."""
    agent_config.dashboard_config[platform] = DashboardConfig(
        platform=platform,
        api_url="http://localhost:8088",
        username="admin",
        password="admin",
        extra={"provider": "db"},
    )
    agent_config.agentic_nodes["gen_dashboard"] = {
        "system_prompt": "gen_dashboard",
        "bi_platform": platform,
        "max_turns": 25,
    }
    # Make adapter_registry.get() return FullMockAdapter
    _bi_core_mock.adapter_registry.get.return_value = lambda **kwargs: FullMockAdapter()


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenDashboardAgenticNodeInit:
    """Tests for GenDashboardAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.NODE_NAME == "gen_dashboard"
            assert node.get_node_name() == "gen_dashboard"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.agentic_node import AgenticNode
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.id == "gen_dashboard_node"

    def test_max_turns_from_config(self, real_agent_config, mock_llm_create):
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.max_turns == 25

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        """Default max_turns is 30 when gen_dashboard not in agentic_nodes."""
        # Add dashboard config but no gen_dashboard agentic node
        real_agent_config.dashboard_config["superset"] = DashboardConfig(
            platform="superset",
            api_url="http://localhost:8088",
            username="admin",
            password="admin",
        )
        _bi_core_mock.adapter_registry.get.return_value = lambda **kwargs: FullMockAdapter()
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.max_turns == 30


# ---------------------------------------------------------------------------
# Tool Setup Tests
# ---------------------------------------------------------------------------


class TestGenDashboardToolSetup:
    """Tests for tool setup — BI tools only, no DB tools exposed."""

    def test_has_bi_tools_with_full_adapter(self, real_agent_config, mock_llm_create):
        """Full adapter should expose all BI tools (read + write)."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            # Read tools (always present)
            assert "list_dashboards" in tool_names
            assert "get_dashboard" in tool_names
            assert "list_charts" in tool_names
            assert "list_datasets" in tool_names

            # Write tools (from full adapter mixins)
            assert "create_dashboard" in tool_names
            assert "create_chart" in tool_names
            assert "create_dataset" in tool_names

    def test_no_db_tools_exposed(self, real_agent_config, mock_llm_create):
        """DB tools (list_tables, describe_table, read_query) should NOT be in tools list."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "list_tables" not in tool_names
            assert "describe_table" not in tool_names
            assert "read_query" not in tool_names
            assert "get_table_ddl" not in tool_names

    def test_no_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Filesystem tools should NOT be in tools list."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "read_file" not in tool_names
            assert "write_file" not in tool_names

    def test_read_only_adapter_fewer_tools(self, real_agent_config, mock_llm_create):
        """Read-only adapter should only expose read tools."""
        real_agent_config.dashboard_config["superset"] = DashboardConfig(
            platform="superset",
            api_url="http://localhost:8088",
            username="admin",
            password="admin",
        )
        real_agent_config.agentic_nodes["gen_dashboard"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "superset",
            "max_turns": 25,
        }
        _bi_core_mock.adapter_registry.get.return_value = lambda **kwargs: ReadOnlyMockAdapter()
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            # Read tools present
            assert "list_dashboards" in tool_names
            assert "get_dashboard" in tool_names

            # Write tools absent
            assert "create_dashboard" not in tool_names
            assert "create_chart" not in tool_names
            assert "create_dataset" not in tool_names

    def test_no_bi_config_no_bi_tools(self, real_agent_config, mock_llm_create):
        """Without BI config, node should have 0 BI tools (graceful no-op)."""
        # No dashboard_config, no bi_platform — just the default real_agent_config
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]

            assert "list_dashboards" not in tool_names
            assert "create_dashboard" not in tool_names

    def test_import_error_yields_no_tools(self, real_agent_config, mock_llm_create):
        """When datus_bi_core import fails, node should have no BI tools."""
        real_agent_config.dashboard_config["superset"] = DashboardConfig(
            platform="superset",
            api_url="http://localhost:8088",
            username="admin",
            password="admin",
        )
        real_agent_config.agentic_nodes["gen_dashboard"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "superset",
            "max_turns": 25,
        }
        # Setting sys.modules["datus_bi_core"] = None causes import to raise ImportError
        with patch.dict(sys.modules, {"datus_bi_core": None}):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert "list_dashboards" not in [t.name for t in node.tools]


# ---------------------------------------------------------------------------
# Auto-detect Platform Tests
# ---------------------------------------------------------------------------


class TestGenDashboardAutoDetect:
    """Tests for auto-detecting bi_platform from dashboard_config."""

    def test_auto_detect_single_platform(self, real_agent_config, mock_llm_create):
        """When only one platform in dashboard_config and no explicit bi_platform, auto-detect it."""
        real_agent_config.dashboard_config["superset"] = DashboardConfig(
            platform="superset",
            api_url="http://localhost:8088",
            username="admin",
            password="admin",
        )
        # agentic_nodes.gen_dashboard WITHOUT bi_platform
        real_agent_config.agentic_nodes["gen_dashboard"] = {
            "system_prompt": "gen_dashboard",
            "max_turns": 25,
        }
        _bi_core_mock.adapter_registry.get.return_value = lambda **kwargs: FullMockAdapter()
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            tool_names = [tool.name for tool in node.tools]
            # Should have BI tools via auto-detection
            assert "list_dashboards" in tool_names

    def test_multiple_platforms_without_explicit_yields_no_tools(self, real_agent_config, mock_llm_create):
        """When multiple platforms configured and no explicit bi_platform, no tools should be set up."""
        real_agent_config.dashboard_config["superset"] = DashboardConfig(
            platform="superset",
            api_url="http://localhost:8088",
            username="admin",
            password="admin",
        )
        real_agent_config.dashboard_config["grafana"] = DashboardConfig(
            platform="grafana",
            api_url="http://localhost:3000",
            username="admin",
            password="admin",
        )
        real_agent_config.agentic_nodes["gen_dashboard"] = {"system_prompt": "gen_dashboard", "max_turns": 25}
        # No bi_platform set
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert "list_dashboards" not in [t.name for t in node.tools]

    def test_explicit_platform_no_matching_config_yields_no_tools(self, real_agent_config, mock_llm_create):
        """When bi_platform is explicitly set but no matching dashboard_config entry, no tools should be set up."""
        real_agent_config.agentic_nodes["gen_dashboard"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "grafana",
            "max_turns": 25,
        }
        # dashboard_config has no "grafana"
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert "list_dashboards" not in [t.name for t in node.tools]


# ---------------------------------------------------------------------------
# Registration Coverage Tests (node.py, node_type.py, sub_agent_task_tool.py)
# ---------------------------------------------------------------------------


class TestGenDashboardRegistration:
    """Tests covering the registration code paths in existing files."""

    def test_node_type_input(self):
        """NodeType.type_input should return GenDashboardNodeInput for TYPE_GEN_DASHBOARD."""
        from datus.configuration.node_type import NodeType
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

        result = NodeType.type_input(NodeType.TYPE_GEN_DASHBOARD, {"user_message": "test"})
        assert isinstance(result, GenDashboardNodeInput)
        assert result.user_message == "test"

    def test_node_type_input_optional(self):
        """NodeType.type_input with ignore_require_check should accept empty input."""
        from datus.configuration.node_type import NodeType
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

        result = NodeType.type_input(NodeType.TYPE_GEN_DASHBOARD, {}, ignore_require_check=True)
        assert result is not None
        assert isinstance(result, GenDashboardNodeInput)
        assert result.database is None

    def test_node_factory_creates_gen_dashboard(self, real_agent_config, mock_llm_create):
        """Node.new_instance should create GenDashboardAgenticNode for TYPE_GEN_DASHBOARD."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node = Node.new_instance(
                node_id="test_factory",
                description="Factory test",
                node_type=NodeType.TYPE_GEN_DASHBOARD,
                agent_config=real_agent_config,
            )
            assert isinstance(node, GenDashboardAgenticNode)
            assert node.get_node_name() == "gen_dashboard"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        """Node.new_instance with input_data should set node.input."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType
            from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

            input_data = GenDashboardNodeInput(user_message="Create a sales dashboard")
            node = Node.new_instance(
                node_id="test_factory_input",
                description="Factory test",
                node_type=NodeType.TYPE_GEN_DASHBOARD,
                input_data=input_data,
                agent_config=real_agent_config,
            )
            assert node.input is not None
            assert node.input.user_message == "Create a sales dashboard"

    def test_from_dict_input_deserialization(self, real_agent_config, mock_llm_create):
        """Node.from_dict should deserialize GenDashboardNodeInput from dict."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node_dict = {
                "id": "test_from_dict",
                "description": "From dict test",
                "type": NodeType.TYPE_GEN_DASHBOARD,
                "input": {"user_message": "List dashboards", "database": "test_db"},
                "result": None,
                "status": "completed",
                "start_time": None,
                "end_time": None,
                "dependencies": [],
                "metadata": {},
            }
            node = Node.from_dict(node_dict, agent_config=real_agent_config)
            assert node.input is not None
            assert node.input.user_message == "List dashboards"

    def test_from_dict_result_deserialization(self, real_agent_config, mock_llm_create):
        """Node.from_dict should deserialize GenDashboardNodeResult from dict."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node_dict = {
                "id": "test_from_dict_result",
                "description": "From dict result test",
                "type": NodeType.TYPE_GEN_DASHBOARD,
                "input": None,
                "result": {
                    "success": True,
                    "response": "Dashboard created",
                    "dashboard_result": {"dashboard_id": 42},
                    "tokens_used": 500,
                },
                "status": "completed",
                "start_time": None,
                "end_time": None,
                "dependencies": [],
                "metadata": {},
            }
            node = Node.from_dict(node_dict, agent_config=real_agent_config)
            assert node.result is not None
            assert node.result.response == "Dashboard created"
            assert node.result.dashboard_result == {"dashboard_id": 42}

    def test_sub_agent_resolve_node_type(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._resolve_node_type should resolve gen_dashboard."""
        from datus.configuration.node_type import NodeType
        from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        node_type, node_name = tool._resolve_node_type("gen_dashboard")
        assert node_type == NodeType.TYPE_GEN_DASHBOARD
        assert node_name == "gen_dashboard"

    def test_sub_agent_create_builtin_node(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._create_builtin_node should create GenDashboardAgenticNode."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
            from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

            tool = SubAgentTaskTool(agent_config=real_agent_config)
            node = tool._create_builtin_node("gen_dashboard")
            assert isinstance(node, GenDashboardAgenticNode)
            assert node.get_node_name() == "gen_dashboard"

    def test_sub_agent_build_node_input(self, real_agent_config, mock_llm_create):
        """SubAgentTaskTool._build_node_input should return GenDashboardNodeInput."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
            from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput
            from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

            tool = SubAgentTaskTool(agent_config=real_agent_config)
            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            result = tool._build_node_input(node, "List all dashboards")
            assert isinstance(result, GenDashboardNodeInput)
            assert result.user_message == "List all dashboards"
            assert result.database == real_agent_config.current_database


# ---------------------------------------------------------------------------
# execute_stream Tests
# ---------------------------------------------------------------------------


class TestGenDashboardExecuteStream:
    """Tests for execute_stream method."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        """execute_stream should raise DatusException when self.input is None."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
            from datus.utils.exceptions import DatusException

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.input is None

            with pytest.raises(DatusException):
                async for _ in node.execute_stream():
                    pass

    @pytest.mark.asyncio
    async def test_execute_stream_happy_path(self, real_agent_config, mock_llm_create):
        """execute_stream should yield actions and produce a successful result."""
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput
        from tests.unit_tests.mock_llm_model import build_simple_response

        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = GenDashboardNodeInput(user_message="List dashboards")

            mock_llm_create.reset(responses=[build_simple_response("Here are your dashboards: ...")])

            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

            assert len(actions) >= 2  # At least user action + final action
            # First action is user input
            assert actions[0].role == ActionRole.USER
            # Last action should be success
            assert actions[-1].status == ActionStatus.SUCCESS
            assert actions[-1].action_type == "gen_dashboard_response"

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        """execute_stream should yield error action on exception."""
        from datus.schemas.action_history import ActionHistoryManager, ActionStatus
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = GenDashboardNodeInput(user_message="Create dashboard")

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
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            node.input = GenDashboardNodeInput(user_message="Create dashboard")
            with patch.object(node.model, "generate_with_tools_stream", side_effect=ExecutionInterrupted()):
                with pytest.raises(ExecutionInterrupted):
                    async for _ in node.execute_stream():
                        pass


# ---------------------------------------------------------------------------
# Template Context Tests
# ---------------------------------------------------------------------------


class TestGenDashboardTemplateContext:
    """Tests for _prepare_template_context and _fallback_system_prompt."""

    def test_context_with_full_adapter(self, real_agent_config, mock_llm_create):
        """Full adapter should set all has_*_write flags to True and report the platform."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            ctx = node._prepare_template_context()
            assert ctx["has_dashboard_write"] is True
            assert ctx["has_chart_write"] is True
            assert ctx["has_dataset_write"] is True
            assert ctx["bi_platform"] == "superset"

    def test_context_without_bi_tools(self, real_agent_config, mock_llm_create):
        """Without BI config, all has_*_write flags should be False."""
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            ctx = node._prepare_template_context()
            assert ctx["has_dashboard_write"] is False
            assert ctx["has_chart_write"] is False
            assert ctx["has_dataset_write"] is False

    def test_fallback_system_prompt(self, real_agent_config, mock_llm_create):
        """Fallback prompt should mention the BI platform and role."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            prompt = node._fallback_system_prompt({"bi_platform": "superset"})
            assert "BI dashboard specialist" in prompt
            assert "superset" in prompt


# ---------------------------------------------------------------------------
# Custom node_name Tests (P2-2: alias subagent config lookup)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestGenDashboardCustomNodeName:
    """Tests for custom node_name support (e.g. my_dashboard: {node_class: gen_dashboard})."""

    def test_default_node_name(self, real_agent_config, mock_llm_create):
        """Without node_name, get_node_name() returns NODE_NAME."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.get_node_name() == "gen_dashboard"

    def test_custom_node_name(self, real_agent_config, mock_llm_create):
        """With node_name, get_node_name() returns the custom name."""
        _add_dashboard_config(real_agent_config)
        # Add custom alias config
        real_agent_config.agentic_nodes["my_dashboard"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "superset",
            "max_turns": 15,
        }
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(
                agent_config=real_agent_config, execution_mode="workflow", node_name="my_dashboard"
            )
            assert node.get_node_name() == "my_dashboard"
            assert node.max_turns == 15

    def test_custom_node_name_reads_own_config(self, real_agent_config, mock_llm_create):
        """Custom node_name should read its own agentic_nodes config, not the builtin one."""
        _add_dashboard_config(real_agent_config)  # gen_dashboard max_turns=25
        real_agent_config.agentic_nodes["analytics_dash"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "superset",
            "max_turns": 50,
        }
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(
                agent_config=real_agent_config, execution_mode="workflow", node_name="analytics_dash"
            )
            assert node.max_turns == 50  # from analytics_dash, not gen_dashboard's 25

    def test_node_factory_passes_node_name(self, real_agent_config, mock_llm_create):
        """Node.new_instance() should pass node_name to GenDashboardAgenticNode."""
        _add_dashboard_config(real_agent_config)
        real_agent_config.agentic_nodes["my_bi"] = {
            "system_prompt": "gen_dashboard",
            "bi_platform": "superset",
            "max_turns": 10,
        }
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.node import Node
            from datus.configuration.node_type import NodeType

            node = Node.new_instance(
                node_id="test_factory",
                description="Factory test",
                node_type=NodeType.TYPE_GEN_DASHBOARD,
                agent_config=real_agent_config,
                node_name="my_bi",
            )
            assert node.get_node_name() == "my_bi"
            assert node.max_turns == 10


# ---------------------------------------------------------------------------
# prompt_version passthrough Tests (P3)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestGenDashboardPromptVersion:
    """Tests for prompt_version passthrough from input to _get_system_prompt."""

    def test_prompt_version_from_input(self, real_agent_config, mock_llm_create):
        """Input prompt_version should override node_config prompt_version."""
        _add_dashboard_config(real_agent_config)
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "test prompt"
                node._get_system_prompt(prompt_version="2.0")
                call_kwargs = mock_pm.return_value.render_template.call_args
                version = call_kwargs.kwargs.get("version")
                assert version == "2.0", f"Expected version '2.0', got '{version}'"

    def test_prompt_version_fallback_to_config(self, real_agent_config, mock_llm_create):
        """Without input prompt_version, should use node_config value."""
        _add_dashboard_config(real_agent_config)
        real_agent_config.agentic_nodes["gen_dashboard"]["prompt_version"] = "1.5"
        with patch.dict(sys.modules, _BI_MODULES_PATCH):
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            node = GenDashboardAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "test prompt"
                node._get_system_prompt(prompt_version=None)
                call_kwargs = mock_pm.return_value.render_template.call_args
                version = call_kwargs.kwargs.get("version")
                assert version == "1.5", f"Expected version '1.5', got '{version}'"
