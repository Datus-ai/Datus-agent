# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for SubAgentTaskTool (AgenticNode-based execution)."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.tools.func_tool.sub_agent_task_tool import (
    BUILTIN_SUBAGENT_DESCRIPTIONS,
    NODE_CLASS_MAP,
    SubAgentTaskTool,
)
from datus.utils.constants import SYS_SUB_AGENTS


@pytest.fixture
def mock_agent_config():
    config = Mock(spec=AgentConfig)
    config.db_type = "sqlite"
    config.current_database = "test_db"
    config.agentic_nodes = {
        "chat": {"model": "default"},
        "gen_sql": {"model": "default", "system_prompt": "gen_sql", "node_class": "gen_sql"},
        "sales_analyst": {
            "model": "default",
            "node_class": "gen_sql",
            "agent_description": "Sales data specialist",
        },
    }
    config.sub_agent_config.side_effect = lambda name: config.agentic_nodes.get(name)
    return config


@pytest.fixture
def task_tool(mock_agent_config):
    return SubAgentTaskTool(agent_config=mock_agent_config)


# ── Initialization ─────────────────────────────────────────────────


@pytest.mark.ci
class TestInit:
    def test_init(self, task_tool, mock_agent_config):
        assert task_tool.agent_config is mock_agent_config
        assert task_tool._action_bus is None
        assert task_tool._interaction_broker is None

    def test_init_only_requires_agent_config(self, mock_agent_config):
        """No model or tool params needed."""
        tool = SubAgentTaskTool(agent_config=mock_agent_config)
        assert tool.agent_config is mock_agent_config


# ── available_tools ────────────────────────────────────────────────


@pytest.mark.ci
class TestAvailableTools:
    def test_returns_task_function_tool(self, task_tool):
        tools = task_tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "task"

    def test_tool_has_correct_schema(self, task_tool):
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "type" in schema["properties"]
        assert "prompt" in schema["properties"]
        assert set(schema["required"]) == {"type", "prompt", "description"}


# ── _get_available_types ───────────────────────────────────────────


@pytest.mark.ci
class TestGetAvailableTypes:
    def test_includes_gen_sql(self, task_tool):
        types = task_tool._get_available_types()
        assert "gen_sql" in types

    def test_includes_custom_subagent(self, task_tool):
        types = task_tool._get_available_types()
        assert "sales_analyst" in types

    def test_excludes_chat(self, task_tool):
        types = task_tool._get_available_types()
        assert "chat" not in types

    def test_includes_agent_without_node_class(self):
        """Subagent without node_class should still be discovered (defaults to gen_sql)."""
        config = Mock(spec=AgentConfig)
        config.current_database = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "custom": {"model": "default"},  # no node_class
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "gen_sql" in types
        assert "custom" in types

    def test_excludes_scoped_agent_wrong_namespace(self):
        """Subagent with scoped_context bound to a different namespace should be excluded."""
        config = Mock(spec=AgentConfig)
        config.current_database = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "scoped_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "scoped_context": {"namespace": "other_ns", "tables": "t1"},
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "scoped_agent" not in types

    def test_includes_scoped_agent_matching_namespace(self):
        """Subagent with scoped_context matching current namespace should be included."""
        config = Mock(spec=AgentConfig)
        config.current_database = "sales"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "scoped_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "scoped_context": {"namespace": "sales", "tables": "orders"},
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "scoped_agent" in types

    def test_includes_agent_without_scoped_context(self):
        """Subagent without scoped_context should not be filtered by namespace."""
        config = Mock(spec=AgentConfig)
        config.current_database = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "global_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "agent_description": "A global agent",
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "global_agent" in types

    def test_explicit_list_filters_out_unknown_types(self, caplog):
        """Unknown types in explicit allowed_subagents are skipped with a warning."""
        config = Mock(spec=AgentConfig)
        config.current_database = "default"
        config.agentic_nodes = {"chat": {"model": "default"}}
        tool = SubAgentTaskTool(
            agent_config=config,
            allowed_subagents=["gen_sql", "nonexistent_foo", "explore"],
            parent_node_name="chat",
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="datus.tools.func_tool.sub_agent_task_tool"):
            types = tool._get_available_types()

        assert "gen_sql" in types
        assert "explore" in types
        assert "nonexistent_foo" not in types
        assert any("nonexistent_foo" in rec.message for rec in caplog.records)

    def test_explicit_list_excludes_self(self):
        """The parent node name is excluded even if listed in allowed_subagents."""
        config = Mock(spec=AgentConfig)
        config.current_database = "default"
        config.agentic_nodes = {"chat": {"model": "default"}}
        tool = SubAgentTaskTool(
            agent_config=config,
            allowed_subagents=["gen_sql", "explore"],
            parent_node_name="gen_sql",
        )
        types = tool._get_available_types()
        assert "gen_sql" not in types
        assert "explore" in types


# ── _resolve_node_type ─────────────────────────────────────────────


@pytest.mark.ci
class TestResolveNodeType:
    def test_gen_sql_with_config(self, task_tool):
        """gen_sql resolves to TYPE_GENSQL using config key."""
        node_type, node_name = task_tool._resolve_node_type("gen_sql")
        assert node_type == NodeType.TYPE_GENSQL
        assert node_name == "gen_sql"

    def test_gen_sql_without_config(self):
        """gen_sql falls back to TYPE_GENSQL with default name."""
        config = Mock(spec=AgentConfig)
        config.agentic_nodes = {}
        tool = SubAgentTaskTool(agent_config=config)
        node_type, node_name = tool._resolve_node_type("gen_sql")
        assert node_type == NodeType.TYPE_GENSQL
        assert node_name == "gen_sql"

    def test_custom_type_gen_sql_class(self, task_tool):
        """Custom type with node_class=gen_sql maps to TYPE_GENSQL."""
        node_type, node_name = task_tool._resolve_node_type("sales_analyst")
        assert node_type == NodeType.TYPE_GENSQL
        assert node_name == "sales_analyst"

    def test_unknown_type_raises(self, task_tool):
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown subagent type"):
            task_tool._resolve_node_type("nonexistent")

    def test_node_class_map_coverage(self):
        """NODE_CLASS_MAP contains exactly the expected key→NodeType mappings."""
        expected_map = {
            "gen_sql": NodeType.TYPE_GENSQL,
            "chat": NodeType.TYPE_CHAT,
            "gen_report": NodeType.TYPE_GEN_REPORT,
            "ext_knowledge": NodeType.TYPE_EXT_KNOWLEDGE,
            "semantic": NodeType.TYPE_SEMANTIC,
            "sql_summary": NodeType.TYPE_SQL_SUMMARY,
            "explore": NodeType.TYPE_EXPLORE,
            "gen_table": NodeType.TYPE_GEN_TABLE,
            "gen_job": NodeType.TYPE_GEN_JOB,
            "migration": NodeType.TYPE_MIGRATION,
            "gen_skill": NodeType.TYPE_GEN_SKILL,
            "gen_dashboard": NodeType.TYPE_GEN_DASHBOARD,
            "scheduler": NodeType.TYPE_SCHEDULER,
        }
        assert set(NODE_CLASS_MAP.keys()) == set(expected_map.keys()), (
            f"NODE_CLASS_MAP keys differ: got {set(NODE_CLASS_MAP.keys())}"
        )
        for key, expected_value in expected_map.items():
            assert NODE_CLASS_MAP[key] == expected_value, f"Wrong mapping for key '{key}'"


# ── _build_task_description ────────────────────────────────────────


@pytest.mark.ci
class TestBuildTaskDescription:
    def test_contains_all_types(self, task_tool):
        desc = task_tool._build_task_description()
        assert "gen_sql" in desc
        assert "sales_analyst" in desc

    def test_contains_guidelines(self, task_tool):
        desc = task_tool._build_task_description()
        assert "Guidelines" in desc

    def test_contains_custom_description(self, task_tool):
        desc = task_tool._build_task_description()
        assert "Sales data specialist" in desc

    def test_explore_description_contains_directions(self, task_tool):
        """Explore description lists 3 exploration directions."""
        desc = task_tool._build_task_description()
        assert "Schema+Sample" in desc
        assert "Knowledge" in desc
        assert "File" in desc

    def test_explore_description_contains_prompt_examples(self, task_tool):
        """Explore description includes prompt examples for each direction."""
        desc = task_tool._build_task_description()
        assert "Prompt example:" in desc

    def test_guidelines_contain_parallel_explore(self, task_tool):
        """Guidelines recommend parallel exploration with direction-specific prompts."""
        desc = task_tool._build_task_description()
        assert "PARALLEL" in desc
        assert "direction-specific prompt" in desc


# ── node creation (fresh per invocation) ──────────────────────────


@pytest.mark.ci
class TestNodeCreation:
    def test_always_creates_fresh_node(self, task_tool):
        """Each call to _create_node returns a distinct new instance (no caching).

        Uses "explore" because it is in NODE_CLASS_MAP but NOT in SYS_SUB_AGENTS,
        so it goes through the Node.new_instance factory path.
        """
        node_a = Mock(name="node_a")
        node_b = Mock(name="node_b")

        with patch("datus.agent.node.node.Node.new_instance", side_effect=[node_a, node_b]):
            node1 = task_tool._create_node("explore")
            node2 = task_tool._create_node("explore")

        assert node1 is not node2
        assert node1 is node_a
        assert node2 is node_b


# ── _build_node_input ──────────────────────────────────────────────


@pytest.mark.ci
class TestBuildNodeInput:
    def test_gen_sql_node_input(self, task_tool):
        """GenSQLAgenticNode gets GenSQLNodeInput."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        # Create a mock that is an instance of GenSQLAgenticNode
        mock_node = Mock(spec=GenSQLAgenticNode)
        mock_node.type = NodeType.TYPE_GENSQL

        result = task_tool._build_node_input(mock_node, "Show all users")

        assert isinstance(result, GenSQLNodeInput)
        assert result.user_message == "Show all users"
        assert result.database == "test_db"


# ── _convert_to_func_result ───────────────────────────────────────


@pytest.mark.ci
class TestConvertToFuncResult:
    def test_sql_result(self, task_tool):
        """GenSQLNodeResult with sql key."""
        output = {"sql": "SELECT 1", "response": "test query", "tokens_used": 100}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert result.result["response"] == "test query"
        assert result.result["tokens_used"] == 100

    def test_generic_result(self, task_tool):
        """Result without sql key."""
        output = {"response": "Here is the answer", "tokens_used": 50}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["response"] == "Here is the answer"
        assert result.result["tokens_used"] == 50

    def test_none_output(self, task_tool):
        """None output returns error."""
        result = task_tool._convert_to_func_result(None)
        assert result.success == 0
        assert "No result" in result.error

    def test_empty_dict(self, task_tool):
        """Empty dict returns error."""
        result = task_tool._convert_to_func_result({})
        assert result.success == 0

    def test_content_fallback(self, task_tool):
        """Falls back to 'content' key when no 'response'."""
        output = {"content": "Some content", "tokens_used": 0}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["response"] == "Some content"


# ── task execution ─────────────────────────────────────────────────


@pytest.mark.ci
class TestTaskExecution:
    @pytest.mark.asyncio
    async def test_execute_gen_sql_success(self, task_tool):
        """Successful gen_sql execution through node."""
        # Create mock action with SUCCESS status and GenSQLNodeResult-like output
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {
            "sql": "SELECT 1",
            "response": "test query",
            "tokens_used": 100,
            "success": True,
        }

        mock_node = MagicMock()

        # Make execute_stream an async generator
        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="Show all users")

        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert result.result["tokens_used"] == 100

    @pytest.mark.asyncio
    async def test_execute_unknown_type(self, task_tool):
        result = await task_tool.task(type="nonexistent", prompt="test")
        assert result.success == 0
        assert "disallowed subagent type" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_type(self, task_tool):
        result = await task_tool.task(type="", prompt="test")
        assert result.success == 0
        assert "Missing required parameter: type" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_prompt(self, task_tool):
        result = await task_tool.task(type="gen_sql", prompt="")
        assert result.success == 0
        assert "Missing required parameter: prompt" in result.error

    @pytest.mark.asyncio
    async def test_execute_custom_subagent(self, task_tool):
        """Custom subagent type executes through node."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "Sales report", "tokens_used": 50, "success": True}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="sales_analyst", prompt="Show sales")

        assert result.success == 1
        assert result.result["response"] == "Sales report"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, task_tool):
        """Node exception is caught and returned as error."""
        with patch.object(task_tool, "_create_node", side_effect=RuntimeError("Node init error")):
            result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 0
        assert "Task execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_successful_output(self, task_tool):
        """When stream yields no successful actions, returns error."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.PROCESSING
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = None

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 0
        assert "No result" in result.error


# ── ActionBus integration ──────────────────────────────────────────


@pytest.mark.ci
class TestActionBusIntegration:
    def test_set_action_bus(self, task_tool):
        """set_action_bus stores the bus reference."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)
        assert task_tool._action_bus is bus

    @pytest.mark.asyncio
    async def test_actions_forwarded_to_bus(self, task_tool):
        """Child actions are put into action_bus with depth=1."""
        from datus.schemas.action_bus import ActionBus
        from datus.schemas.action_history import ActionRole

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL  # Non-USER role so it's forwarded
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1
        # Forwarded action should be in the bus with depth=1
        forwarded = bus._queue.get_nowait()
        assert forwarded.depth == 1

    @pytest.mark.asyncio
    async def test_actions_have_parent_action_id(self, task_tool):
        """When call_id is provided, child actions get parent_action_id."""
        from datus.schemas.action_bus import ActionBus
        from datus.schemas.action_history import ActionRole

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL  # Non-USER role so it's forwarded
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test", call_id="parent_call_123")

        assert result.success == 1
        # Forwarded action should have parent_action_id
        forwarded = bus._queue.get_nowait()
        assert forwarded.parent_action_id == "parent_call_123"

    @pytest.mark.asyncio
    async def test_no_bus_no_error(self, task_tool):
        """Without action_bus, execution still works (no forwarding)."""
        assert task_tool._action_bus is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1


# ── InteractionBroker pass-through ─────────────────────────────────


@pytest.mark.ci
class TestInteractionBrokerPassthrough:
    def test_set_interaction_broker(self, task_tool):
        """set_interaction_broker stores the broker reference."""
        from datus.cli.execution_state import InteractionBroker

        broker = InteractionBroker()
        task_tool.set_interaction_broker(broker)
        assert task_tool._interaction_broker is broker

    def test_inject_broker_updates_node(self, task_tool):
        """_inject_broker replaces the node's interaction_broker."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        node = MagicMock()
        node.interaction_broker = InteractionBroker()  # original broker
        node.hooks = None
        node.permission_hooks = None
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)
        assert node.interaction_broker is parent_broker

    def test_inject_broker_updates_hooks(self, task_tool):
        """_inject_broker updates broker on GenerationHooks / PermissionHooks."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        original_broker = InteractionBroker()

        # Use spec to limit attributes — prevents false positive on hooks_list
        mock_hooks = Mock(spec=["broker"])
        mock_hooks.broker = original_broker

        mock_perm_hooks = Mock(spec=["broker"])
        mock_perm_hooks.broker = original_broker

        node = MagicMock()
        node.interaction_broker = original_broker
        node.hooks = mock_hooks
        node.permission_hooks = mock_perm_hooks
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)

        assert node.interaction_broker is parent_broker
        assert mock_hooks.broker is parent_broker
        assert mock_perm_hooks.broker is parent_broker

    def test_inject_broker_updates_composite_hooks(self, task_tool):
        """_inject_broker updates broker inside CompositeHooks."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        original_broker = InteractionBroker()

        inner_hook_1 = Mock()
        inner_hook_1.broker = original_broker
        inner_hook_2 = Mock()
        inner_hook_2.broker = original_broker

        composite = Mock()
        composite.broker = original_broker
        composite.hooks_list = [inner_hook_1, inner_hook_2]

        node = MagicMock()
        node.interaction_broker = original_broker
        node.hooks = composite
        node.permission_hooks = None
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)

        assert inner_hook_1.broker is parent_broker
        assert inner_hook_2.broker is parent_broker

    @pytest.mark.asyncio
    async def test_with_broker_uses_execute_stream(self, task_tool):
        """When broker is injected, _execute_node calls execute_stream (not _with_interactions)."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        task_tool.set_interaction_broker(parent_broker)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream = mock_stream
        # execute_stream_with_interactions should NOT be called
        mock_node.execute_stream_with_interactions = MagicMock(side_effect=AssertionError("should not be called"))

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_without_broker_uses_execute_stream_with_interactions(self, task_tool):
        """Without broker, _execute_node falls back to execute_stream_with_interactions."""
        assert task_tool._interaction_broker is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream
        # execute_stream should NOT be called
        mock_node.execute_stream = MagicMock(side_effect=AssertionError("should not be called"))

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1

    @pytest.mark.asyncio
    async def test_broker_injected_into_node(self, task_tool):
        """When parent broker is set, _execute_node injects it into the created node."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        task_tool.set_interaction_broker(parent_broker)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "ok", "tokens_used": 0}

        injected_broker = None

        mock_node = MagicMock()
        mock_node.hooks = None
        mock_node.permission_hooks = None
        mock_node.plan_hooks = None

        async def mock_stream(ahm):
            nonlocal injected_broker
            injected_broker = mock_node.interaction_broker
            yield mock_action

        mock_node.execute_stream = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test")

        assert injected_broker is parent_broker


# ── SQL file storage result conversion ─────────────────────────────


@pytest.mark.ci
class TestConvertToFuncResultFileStorage:
    """Tests for _convert_to_func_result with file-based SQL results."""

    def test_file_based_sql_result(self, task_tool):
        """Result with sql_file_path returns file-based format."""
        output = {
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT a\nFROM users\n-- ... (55 more lines)",
            "response": "Generated complex query",
            "tokens_used": 200,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_file_path"] == "sql/session_1/task_1.sql"
        assert result.result["sql_preview"] == output["sql_preview"]
        assert result.result["response"] == "Generated complex query"
        assert result.result["tokens_used"] == 200
        assert "sql" not in result.result
        assert "sql_diff" not in result.result

    def test_file_based_sql_result_with_diff(self, task_tool):
        """Result with sql_file_path and sql_diff includes diff."""
        output = {
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT a, b\nFROM users",
            "sql_diff": "--- a/query.sql\n+++ b/query.sql\n@@ -1 +1 @@\n-SELECT a\n+SELECT a, b",
            "response": "Modified query",
            "tokens_used": 150,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_file_path"] == "sql/session_1/task_1.sql"
        assert result.result["sql_diff"] == output["sql_diff"]

    def test_inline_sql_still_works(self, task_tool):
        """Short SQL still returns inline format (backward compatible)."""
        output = {"sql": "SELECT 1", "response": "simple query", "tokens_used": 50}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert "sql_file_path" not in result.result

    def test_file_path_takes_priority_over_sql(self, task_tool):
        """When both sql_file_path and sql are present, file path wins."""
        output = {
            "sql": "SELECT full query...",
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT ...",
            "response": "query",
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert "sql_file_path" in result.result
        assert "sql" not in result.result


@pytest.mark.ci
class TestBuildTaskDescriptionFileStorage:
    """Tests for updated _build_task_description with file storage info."""

    def test_description_mentions_file_path(self, task_tool):
        desc = task_tool._build_task_description()
        assert "sql_file_path" in desc

    def test_description_mentions_read_file(self, task_tool):
        desc = task_tool._build_task_description()
        assert "read_file" in desc

    def test_description_mentions_diff(self, task_tool):
        desc = task_tool._build_task_description()
        assert "diff" in desc.lower()


# ── Built-in subagent: _get_available_types ────────────────────────


@pytest.mark.ci
class TestGetAvailableTypesBuiltIn:
    def test_includes_all_builtin_types(self, task_tool):
        """All 4 SYS_SUB_AGENTS appear in available types."""
        types = task_tool._get_available_types()
        for name in SYS_SUB_AGENTS:
            assert name in types, f"{name} not found in available types"

    def test_no_duplicates(self, task_tool):
        """No duplicates even if builtin names appear in agentic_nodes."""
        types = task_tool._get_available_types()
        assert len(types) == len(set(types))

    def test_no_duplicates_when_in_agentic_nodes(self):
        """Builtin types in agentic_nodes are not duplicated."""
        config = Mock(spec=AgentConfig)
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "gen_sql_summary": {"model": "default", "node_class": "sql_summary"},
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert types.count("gen_sql_summary") == 1

    def test_builtin_types_sorted(self, task_tool):
        """Built-in types appear in sorted order after gen_sql."""
        types = task_tool._get_available_types()
        builtin_in_list = [t for t in types if t in SYS_SUB_AGENTS]
        assert builtin_in_list == sorted(SYS_SUB_AGENTS)


# ── Built-in subagent: _resolve_node_type ──────────────────────────


@pytest.mark.ci
class TestResolveNodeTypeBuiltIn:
    def test_gen_semantic_model(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_semantic_model")
        assert node_type == NodeType.TYPE_SEMANTIC
        assert node_name == "gen_semantic_model"

    def test_gen_metrics(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_metrics")
        assert node_type == NodeType.TYPE_SEMANTIC
        assert node_name == "gen_metrics"

    def test_gen_sql_summary(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_sql_summary")
        assert node_type == NodeType.TYPE_SQL_SUMMARY
        assert node_name == "gen_sql_summary"

    def test_gen_ext_knowledge(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_ext_knowledge")
        assert node_type == NodeType.TYPE_EXT_KNOWLEDGE
        assert node_name == "gen_ext_knowledge"

    def test_gen_table(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_table")
        assert node_type == NodeType.TYPE_GEN_TABLE
        assert node_name == "gen_table"


# ── Built-in subagent: _create_builtin_node ────────────────────────


@pytest.mark.ci
class TestCreateBuiltinNode:
    @patch("datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode.__init__", return_value=None)
    def test_gen_semantic_model(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_semantic_model")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__", return_value=None)
    def test_gen_metrics(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_metrics")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__", return_value=None)
    def test_gen_sql_summary(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_sql_summary")
        mock_init.assert_called_once_with(
            node_name="gen_sql_summary",
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.gen_ext_knowledge_agentic_node.GenExtKnowledgeAgenticNode.__init__", return_value=None)
    def test_gen_ext_knowledge(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_ext_knowledge")
        mock_init.assert_called_once_with(
            node_name="gen_ext_knowledge",
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__", return_value=None)
    def test_gen_table(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("gen_table")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
        )

    @patch("datus.agent.node.gen_job_agentic_node.GenJobAgenticNode.__init__", return_value=None)
    def test_gen_job(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_job")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.migration_agentic_node.MigrationAgenticNode.__init__", return_value=None)
    def test_migration(self, mock_init, task_tool):
        task_tool._create_builtin_node("migration")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
        )

    @patch("datus.agent.node.gen_dashboard_agentic_node.GenDashboardAgenticNode.__init__", return_value=None)
    def test_gen_dashboard(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("gen_dashboard")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
        )

    @patch("datus.agent.node.scheduler_agentic_node.SchedulerAgenticNode.__init__", return_value=None)
    def test_scheduler(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("scheduler")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
        )

    def test_unknown_builtin_raises(self, task_tool):
        with pytest.raises(ValueError, match="Unknown builtin subagent type"):
            task_tool._create_builtin_node("nonexistent")

    def test_create_node_delegates_to_builtin(self, task_tool):
        """_create_node delegates to _create_builtin_node for SYS_SUB_AGENTS."""
        with patch.object(task_tool, "_create_builtin_node", return_value=Mock()) as mock_builtin:
            task_tool._create_node("gen_semantic_model")
            mock_builtin.assert_called_once_with("gen_semantic_model")

    def test_create_node_custom_passes_is_subagent_true(self, task_tool):
        """Custom agents must receive ``is_subagent=True`` via Node.new_instance.

        This enforces 2-level depth at the source: the child never constructs a
        SubAgentTaskTool, so there is nothing to strip post-construction.
        """
        with patch("datus.agent.node.node.Node.new_instance", return_value=Mock()) as mock_new_instance:
            task_tool._create_node("sales_analyst")

        mock_new_instance.assert_called_once()
        call_kwargs = mock_new_instance.call_args.kwargs
        assert call_kwargs["is_subagent"] is True
        assert call_kwargs["node_name"] == "sales_analyst"


# ── _resolve_execution_mode ─────────────────────────────────────────


@pytest.mark.ci
class TestResolveExecutionMode:
    def test_returns_interactive_when_no_parent(self, task_tool):
        assert task_tool._parent_node is None
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_parent_mode_workflow(self, task_tool):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "workflow"

    def test_returns_parent_mode_interactive(self, task_tool):
        parent = Mock()
        parent.execution_mode = "interactive"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_interactive_when_parent_has_no_execution_mode(self, task_tool):
        parent = Mock(spec=[])  # no attributes
        task_tool._parent_node = parent
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_interactive_for_invalid_mode(self, task_tool):
        parent = Mock()
        parent.execution_mode = "unknown_mode"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "interactive"


@pytest.mark.ci
class TestBuiltinNodeInheritsExecutionMode:
    """Verify _create_builtin_node passes parent's execution_mode to subagent constructors."""

    @pytest.mark.parametrize(
        "subagent_type,init_path",
        [
            (
                "gen_semantic_model",
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode.__init__",
            ),
            ("gen_metrics", "datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__"),
            ("gen_sql_summary", "datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__"),
            (
                "gen_ext_knowledge",
                "datus.agent.node.gen_ext_knowledge_agentic_node.GenExtKnowledgeAgenticNode.__init__",
            ),
            ("gen_table", "datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__"),
            ("gen_dashboard", "datus.agent.node.gen_dashboard_agentic_node.GenDashboardAgenticNode.__init__"),
            ("scheduler", "datus.agent.node.scheduler_agentic_node.SchedulerAgenticNode.__init__"),
        ],
    )
    def test_builtin_node_uses_workflow_mode(self, task_tool, subagent_type, init_path):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)

        with patch(init_path, return_value=None) as mock_init:
            task_tool._create_builtin_node(subagent_type)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["execution_mode"] == "workflow"

    @pytest.mark.parametrize(
        "subagent_type,init_path",
        [
            ("gen_sql", "datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode.__init__"),
            ("gen_report", "datus.agent.node.gen_report_agentic_node.GenReportAgenticNode.__init__"),
            ("gen_skill", "datus.agent.node.gen_skill_agentic_node.SkillCreatorAgenticNode.__init__"),
        ],
    )
    def test_builtin_node_with_extra_params_uses_workflow_mode(self, task_tool, subagent_type, init_path):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)

        with patch(init_path, return_value=None) as mock_init:
            task_tool._create_builtin_node(subagent_type)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["execution_mode"] == "workflow"


# ── Built-in subagent: _build_node_input ───────────────────────────


@pytest.mark.ci
class TestBuildNodeInputBuiltIn:
    def test_semantic_model_node_input(self, task_tool):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenSemanticModelAgenticNode)
        result = task_tool._build_node_input(mock_node, "orders table")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "orders table"
        assert result.database == "test_db"

    def test_metrics_node_input(self, task_tool):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenMetricsAgenticNode)
        result = task_tool._build_node_input(mock_node, "SELECT SUM(amount) FROM orders")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "SELECT SUM(amount) FROM orders"
        assert result.database == "test_db"

    def test_sql_summary_node_input(self, task_tool):
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        mock_node = Mock(spec=SqlSummaryAgenticNode)
        result = task_tool._build_node_input(mock_node, "SELECT * FROM users")
        assert isinstance(result, SqlSummaryNodeInput)
        assert result.user_message == "SELECT * FROM users"
        assert result.database == "test_db"

    def test_ext_knowledge_node_input(self, task_tool):
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
        from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

        mock_node = Mock(spec=GenExtKnowledgeAgenticNode)
        result = task_tool._build_node_input(mock_node, "What is total revenue by region?")
        assert isinstance(result, ExtKnowledgeNodeInput)
        assert result.user_message == "What is total revenue by region?"

    def test_gen_table_node_input(self, task_tool):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenTableAgenticNode)
        result = task_tool._build_node_input(mock_node, "Create wide table from orders and customers")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "Create wide table from orders and customers"
        assert result.database == "test_db"


# ── Built-in subagent: _build_task_description ─────────────────────


@pytest.mark.ci
class TestBuildTaskDescriptionBuiltIn:
    def test_contains_all_builtin_types(self, task_tool):
        desc = task_tool._build_task_description()
        for name in SYS_SUB_AGENTS:
            assert name in desc, f"{name} not found in task description"

    def test_contains_builtin_descriptions(self, task_tool):
        desc = task_tool._build_task_description()
        for name, builtin_desc in BUILTIN_SUBAGENT_DESCRIPTIONS.items():
            assert builtin_desc in desc, f"Description for {name} not found"

    def test_gen_semantic_model_description_content(self, task_tool):
        desc = task_tool._build_task_description()
        assert "semantic model" in desc.lower()
        assert "semantic_models" in desc

    def test_gen_sql_summary_description_content(self, task_tool):
        desc = task_tool._build_task_description()
        assert "sql_summary_file" in desc

    def test_gen_ext_knowledge_description_content(self, task_tool):
        desc = task_tool._build_task_description()
        assert "ext_knowledge_file" in desc


# ── Built-in subagent: _convert_to_func_result ────────────────────


@pytest.mark.ci
class TestConvertToFuncResultBuiltIn:
    def test_semantic_models_result(self, task_tool):
        output = {
            "response": "Generated 2 models",
            "semantic_models": ["models/orders.yml", "models/customers.yml"],
            "tokens_used": 500,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["semantic_models"] == ["models/orders.yml", "models/customers.yml"]
        assert result.result["response"] == "Generated 2 models"
        assert result.result["tokens_used"] == 500

    def test_semantic_models_empty_list(self, task_tool):
        output = {
            "response": "No models generated",
            "semantic_models": [],
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["semantic_models"] == []

    def test_sql_summary_file_result(self, task_tool):
        output = {
            "response": "Summarized query",
            "sql_summary_file": "knowledge/summaries/query_001.yml",
            "tokens_used": 300,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_summary_file"] == "knowledge/summaries/query_001.yml"
        assert result.result["response"] == "Summarized query"

    def test_ext_knowledge_file_result(self, task_tool):
        output = {
            "response": "Extracted knowledge",
            "ext_knowledge_file": "knowledge/ext/revenue_by_region.yml",
            "tokens_used": 800,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["ext_knowledge_file"] == "knowledge/ext/revenue_by_region.yml"
        assert result.result["response"] == "Extracted knowledge"

    def test_sql_file_path_takes_priority_over_semantic_models(self, task_tool):
        """sql_file_path still takes priority (checked first)."""
        output = {
            "sql_file_path": "sql/session/task.sql",
            "sql_preview": "SELECT ...",
            "semantic_models": ["models/x.yml"],
            "response": "test",
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert "sql_file_path" in result.result
        assert "semantic_models" not in result.result

    def test_dashboard_result(self, task_tool):
        """Dashboard result should preserve dashboard_result dict."""
        output = {
            "response": "Created dashboard",
            "dashboard_result": {"dashboard_id": 42, "url": "http://superset/dashboard/42"},
            "tokens_used": 600,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["dashboard_result"] == {"dashboard_id": 42, "url": "http://superset/dashboard/42"}
        assert result.result["response"] == "Created dashboard"
        assert result.result["tokens_used"] == 600

    def test_dashboard_result_empty_dict(self, task_tool):
        """Empty dashboard_result dict should still be preserved (not fall to generic)."""
        output = {
            "response": "No dashboard changes",
            "dashboard_result": {},
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["dashboard_result"] == {}

    def test_scheduler_result(self, task_tool):
        """Scheduler result should preserve scheduler_result dict."""
        output = {
            "response": "Job submitted",
            "scheduler_result": {"job_id": "dag_123", "status": "scheduled"},
            "tokens_used": 300,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["scheduler_result"] == {"job_id": "dag_123", "status": "scheduled"}
        assert result.result["response"] == "Job submitted"
        assert result.result["tokens_used"] == 300

    def test_scheduler_result_empty_dict(self, task_tool):
        """Empty scheduler_result dict should still be preserved."""
        output = {
            "response": "No scheduler changes",
            "scheduler_result": {},
            "tokens_used": 50,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["scheduler_result"] == {}


# ── Built-in subagent: end-to-end task execution ──────────────────


@pytest.mark.ci
class TestTaskExecutionBuiltIn:
    @pytest.mark.asyncio
    async def test_execute_gen_semantic_model(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Generated semantic model for orders",
            "semantic_models": ["models/orders.yml"],
            "tokens_used": 400,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_semantic_model", prompt="orders table")

        assert result.success == 1
        assert result.result["semantic_models"] == ["models/orders.yml"]
        assert result.result["tokens_used"] == 400

    @pytest.mark.asyncio
    async def test_execute_gen_metrics(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Extracted 3 metrics",
            "tokens_used": 350,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_metrics", prompt="SELECT SUM(amount) FROM orders")

        assert result.success == 1
        assert result.result["response"] == "Extracted 3 metrics"

    @pytest.mark.asyncio
    async def test_execute_gen_sql_summary(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "SQL summarized",
            "sql_summary_file": "knowledge/summaries/query_001.yml",
            "tokens_used": 250,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql_summary", prompt="SELECT * FROM users WHERE active = 1")

        assert result.success == 1
        assert result.result["sql_summary_file"] == "knowledge/summaries/query_001.yml"

    @pytest.mark.asyncio
    async def test_execute_gen_ext_knowledge(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Knowledge extracted",
            "ext_knowledge_file": "knowledge/ext/revenue.yml",
            "tokens_used": 900,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_ext_knowledge", prompt="What is total revenue by region?")

        assert result.success == 1
        assert result.result["ext_knowledge_file"] == "knowledge/ext/revenue.yml"
        assert result.result["tokens_used"] == 900


# ── SubAgent complete action ──────────────────────────────────────


@pytest.mark.ci
class TestCompleteAction:
    """Tests for the subagent_complete action emitted by _execute_node."""

    @pytest.mark.asyncio
    async def test_complete_action_emitted_on_success(self, task_tool):
        """After a successful stream, the bus contains a subagent_complete action."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test", call_id="call_123")

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].status == ActionStatus.SUCCESS
        assert complete_actions[0].parent_action_id == "call_123"
        assert complete_actions[0].depth == 1

    @pytest.mark.asyncio
    async def test_complete_action_emitted_on_failure(self, task_tool):
        """When the stream raises an exception, a FAILED complete action is still emitted."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield Mock(
                spec=ActionHistory,
                status=ActionStatus.PROCESSING,
                output=None,
                depth=0,
                parent_action_id=None,
                role=ActionRole.TOOL,
                action_type="tool_call",
            )
            raise RuntimeError("Stream error")

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test", call_id="call_fail")

        assert result.success == 0

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].status == ActionStatus.FAILED
        assert complete_actions[0].parent_action_id == "call_fail"

    @pytest.mark.asyncio
    async def test_complete_action_not_emitted_without_bus(self, task_tool):
        """Without an ActionBus, no complete action is emitted and no error occurs."""
        assert task_tool._action_bus is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1

    @pytest.mark.asyncio
    async def test_complete_action_metadata(self, task_tool):
        """The complete action output contains subagent_type and tool_count."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        # Two TOOL actions to count
        tool_action_1 = Mock(spec=ActionHistory)
        tool_action_1.status = ActionStatus.SUCCESS
        tool_action_1.output = None
        tool_action_1.depth = 0
        tool_action_1.parent_action_id = None
        tool_action_1.role = ActionRole.TOOL
        tool_action_1.action_type = "describe_table"

        tool_action_2 = Mock(spec=ActionHistory)
        tool_action_2.status = ActionStatus.SUCCESS
        tool_action_2.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action_2.depth = 0
        tool_action_2.parent_action_id = None
        tool_action_2.role = ActionRole.TOOL
        tool_action_2.action_type = "read_query"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield tool_action_1
            yield tool_action_2

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test", call_id="call_meta")

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].output["subagent_type"] == "gen_sql"
        assert complete_actions[0].output["tool_count"] == 2


# ── Description parameter ────────────────────────────────────────


@pytest.mark.ci
class TestDescriptionParameter:
    """Tests for the optional 'description' parameter on the task tool."""

    def test_schema_contains_description_property(self, task_tool):
        """Schema includes a 'description' property."""
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "description" in schema["properties"]
        assert schema["properties"]["description"]["type"] == "string"

    def test_description_is_required(self, task_tool):
        """'description' IS in the required list."""
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "description" in schema["required"]

    @pytest.mark.asyncio
    async def test_description_injected_into_first_user_action(self, task_tool):
        """When description is provided, it is injected into the first USER action's input."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = {}
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(
                    type="gen_sql", prompt="Show all users", description="List all users from database"
                )

        # The user_action's input should now contain _task_description
        assert user_action.input["_task_description"] == "List all users from database"

    @pytest.mark.asyncio
    async def test_description_not_injected_when_empty(self, task_tool):
        """When description is empty, no _task_description is added."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = {}
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="Show all users", description="")

        # No _task_description should be injected
        assert "_task_description" not in user_action.input

    @pytest.mark.asyncio
    async def test_description_only_injected_into_first_user_action(self, task_tool):
        """Description is injected only into the first USER action, not subsequent ones."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action_1 = Mock(spec=ActionHistory)
        user_action_1.role = ActionRole.USER
        user_action_1.status = ActionStatus.SUCCESS
        user_action_1.output = None
        user_action_1.depth = 0
        user_action_1.parent_action_id = None
        user_action_1.action_type = "user_message"
        user_action_1.input = {}
        user_action_1.messages = "First message"

        user_action_2 = Mock(spec=ActionHistory)
        user_action_2.role = ActionRole.USER
        user_action_2.status = ActionStatus.SUCCESS
        user_action_2.output = None
        user_action_2.depth = 0
        user_action_2.parent_action_id = None
        user_action_2.action_type = "user_message"
        user_action_2.input = {}
        user_action_2.messages = "Second message"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action_1
            yield user_action_2
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="First message", description="Task goal")

        assert user_action_1.input["_task_description"] == "Task goal"
        assert "_task_description" not in user_action_2.input

    @pytest.mark.asyncio
    async def test_description_injected_when_input_is_none(self, task_tool):
        """When the first USER action has input=None, a dict is created for the description."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = None
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="Show all users", description="List users")

        assert user_action.input["_task_description"] == "List users"

    def test_build_task_description_mentions_description_guideline(self, task_tool):
        """The guidelines mention providing a 'description' parameter."""
        desc = task_tool._build_task_description()
        assert "description" in desc.lower()


# ── Proxy tool propagation ─────────────────────────────────────────


@pytest.mark.ci
class TestProxyToolPropagation:
    """Tests for proxy tool config propagation to sub-agent nodes via parent node reference."""

    def test_set_parent_node_stores_reference(self, task_tool):
        """set_parent_node stores the parent node reference."""
        parent_node = MagicMock()
        task_tool.set_parent_node(parent_node)
        assert task_tool._parent_node is parent_node

    def test_default_parent_node_is_none(self, task_tool):
        """By default, _parent_node is None."""
        assert task_tool._parent_node is None

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_called_when_parent_has_patterns(self, task_tool):
        """When parent node has proxy_tool_patterns, apply_proxy_tools is called on the sub-agent node."""
        from datus.tools.proxy.tool_result_channel import ToolResultChannel

        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = ["filesystem_tools.*"]
        parent_node.tool_channel = ToolResultChannel()
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_called_once_with(mock_node, parent_node.proxy_tool_patterns, channel=parent_node.tool_channel)
        assert result.success == 1

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_not_called_when_parent_has_no_patterns(self, task_tool):
        """When parent node has no proxy_tool_patterns, apply_proxy_tools is NOT called."""
        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = None
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_not_called()
        assert result.success == 1

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_not_called_when_no_parent_node(self, task_tool):
        """When _parent_node is None, apply_proxy_tools is NOT called."""
        assert task_tool._parent_node is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_not_called()
        assert result.success == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "subagent_type",
        ["gen_semantic_model", "gen_metrics", "gen_sql_summary", "gen_ext_knowledge"],
    )
    async def test_fs_dependent_types_still_call_apply_proxy(self, task_tool, subagent_type):
        """FS-dependent subagents still call apply_proxy_tools (exclusion is internal to proxy_tool)."""
        from datus.tools.proxy.tool_result_channel import ToolResultChannel

        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = ["*"]
        parent_node.tool_channel = ToolResultChannel()
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type=subagent_type, prompt="test")

        mock_apply.assert_called_once_with(mock_node, parent_node.proxy_tool_patterns, channel=parent_node.tool_channel)
        assert result.success == 1
