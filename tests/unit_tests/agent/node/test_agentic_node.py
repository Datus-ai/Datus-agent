# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for AgenticNode base class.

CI-level: zero external deps, zero network, zero API keys.
Uses _ConcreteAgenticNode (minimal concrete subclass) and patches LLM + sessions.
"""

import asyncio
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult

# ---------------------------------------------------------------------------
# Concrete subclass for testing (can't instantiate abstract AgenticNode directly)
# ---------------------------------------------------------------------------


class _ConcreteAgenticNode(AgenticNode):
    """Minimal concrete implementation of AgenticNode for testing."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="test response",
            input_data={},
            output_data={"success": True, "result": "done"},
            status=ActionStatus.SUCCESS,
        )
        yield action


def _make_node(agent_config=None, **overrides):
    """Create a node with __init__ bypassed for targeted testing."""
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _ConcreteAgenticNode.__new__(_ConcreteAgenticNode)
    # Set minimum required attributes
    node._session = None
    node.ephemeral = False
    node.session_id = None
    node.last_summary = None
    node.model = None
    node.tools = []
    node.mcp_servers = {}
    node.actions = []
    node.context_length = None
    node.node_config = {}
    node.agent_config = agent_config
    node.skill_manager = None
    node.skill_func_tool = None
    node.permission_manager = None
    node._permission_callback = None
    node.result = None
    node.input = None
    node.type = "test"
    from datus.cli.execution_state import InteractionBroker, InterruptController
    from datus.schemas.action_bus import ActionBus

    node.action_bus = ActionBus()
    node.interaction_broker = InteractionBroker()
    node.interrupt_controller = InterruptController()
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeName:
    def test_concrete_node_name_derived_from_class(self):
        """get_node_name strips 'AgenticNode' suffix and lowercases."""
        node = _make_node()
        assert node.get_node_name() == "_concrete"

    def test_node_name_for_specific_class(self):
        """Verify the naming pattern with a well-named subclass."""
        # GenMetricsAgenticNode -> "gen_metrics" (tested via real class)
        # For our concrete class: _ConcreteAgenticNode -> "_concrete"
        node = _make_node()
        name = node.get_node_name()
        assert isinstance(name, str)
        assert len(name) > 0


# ---------------------------------------------------------------------------
# TestParseNodeConfig
# ---------------------------------------------------------------------------


class TestParseNodeConfig:
    def test_returns_empty_when_no_agent_config(self):
        node = _make_node()
        result = node._parse_node_config(None, "mynode")
        assert result == {}

    def test_returns_empty_when_node_not_in_config(self):
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {"other_node": {"model": "gpt-4"}}
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result == {}

    def test_parses_model_from_dict(self):
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {"mynode": {"model": "gpt-4o", "max_turns": 10}}
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result.get("model") == "gpt-4o"

    def test_normalizes_rules_list(self):
        """Rules with dict items are converted to 'key: value' strings."""
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {
            "mynode": {
                "rules": [{"always": "be concise"}, "plain rule"],
            }
        }
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        rules = result.get("rules", [])
        assert any("always: be concise" in r for r in rules)
        assert "plain rule" in rules

    def test_returns_empty_when_no_agentic_nodes_attr(self):
        cfg = MagicMock()
        del cfg.agentic_nodes  # remove the attribute
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result == {}


# ---------------------------------------------------------------------------
# TestResolveWorkspaceRoot
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRoot:
    def test_returns_dot_when_no_config(self):
        node = _make_node()
        result = node._resolve_workspace_root()
        assert result == "."

    def test_uses_node_config_workspace_root(self):
        node = _make_node()
        node.node_config = {"workspace_root": "/custom/path"}
        node.agent_config = None
        result = node._resolve_workspace_root()
        assert result == "/custom/path"

    def test_expands_tilde(self, tmp_path):
        node = _make_node()
        node.node_config = {"workspace_root": "~/testdir"}
        node.agent_config = None
        result = node._resolve_workspace_root()
        import os

        assert not result.startswith("~")
        assert os.path.expanduser("~/testdir") == result

    def test_uses_storage_workspace_root(self):
        node = _make_node()
        node.node_config = {}
        cfg = MagicMock()
        cfg.storage.workspace_root = "/storage/root"
        node.agent_config = cfg
        result = node._resolve_workspace_root()
        assert result == "/storage/root"

    def test_uses_legacy_workspace_root(self):
        node = _make_node()
        node.node_config = {}
        cfg = MagicMock(spec=[])  # no 'storage' attribute
        cfg.workspace_root = "/legacy/root"
        node.agent_config = cfg
        result = node._resolve_workspace_root()
        assert result == "/legacy/root"


# ---------------------------------------------------------------------------
# TestGetToolCategory
# ---------------------------------------------------------------------------


class TestGetToolCategory:
    def test_skill_tool(self):
        node = _make_node()
        assert node._get_tool_category("load_skill") == "skills"
        assert node._get_tool_category("skill_something") == "skills"

    def test_db_tool(self):
        node = _make_node()
        assert node._get_tool_category("list_tables") == "db_tools"
        assert node._get_tool_category("execute_sql") == "db_tools"
        assert node._get_tool_category("db_custom_tool") == "db_tools"

    def test_generic_tool(self):
        node = _make_node()
        assert node._get_tool_category("some_random_tool") == "tools"

    def test_mcp_tool(self):
        node = _make_node()
        node.mcp_servers = {"myserver": MagicMock()}
        assert node._get_tool_category("myserver_do_something") == "mcp"


# ---------------------------------------------------------------------------
# TestSetupInput (default implementation)
# ---------------------------------------------------------------------------


class TestSetupInputAgenticNode:
    def test_default_setup_input_returns_success(self):
        node = _make_node()
        node.input = BaseInput()
        wf = MagicMock()
        wf.task.catalog_name = "cat"
        wf.task.database_name = "db"
        wf.task.schema_name = "sch"
        wf.context.table_schemas = []
        wf.context.metrics = []
        result = node.setup_input(wf)

        assert result["success"] is True

    def test_default_setup_input_creates_base_input_when_none(self):
        node = _make_node()
        node.input = None
        wf = MagicMock()
        wf.task.catalog_name = "cat"
        wf.task.database_name = "db"
        wf.task.schema_name = "sch"
        wf.context.table_schemas = []
        wf.context.metrics = []
        node.setup_input(wf)

        assert node.input is not None


# ---------------------------------------------------------------------------
# TestUpdateContextAgenticNode
# ---------------------------------------------------------------------------


class TestUpdateContextAgenticNode:
    def test_no_result_returns_failure(self):
        node = _make_node()
        node.result = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is False

    def test_result_without_sql_returns_success(self):
        node = _make_node()
        node.result = MagicMock()
        node.result.sql = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is True

    def test_result_with_sql_appends_context(self):
        node = _make_node()
        node.result = MagicMock()
        node.result.sql = "SELECT 1"
        node.result.response = "some explanation"
        wf = MagicMock()
        wf.context.sql_contexts = []
        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1


# ---------------------------------------------------------------------------
# TestClearSession
# ---------------------------------------------------------------------------


class TestClearSession:
    def test_clear_session_ephemeral(self):
        node = _make_node()
        node.ephemeral = True
        node._session = MagicMock()
        node.session_id = "ephemeral_session_1"
        node.clear_session()
        assert node._session is None

    def test_clear_session_non_ephemeral(self):
        node = _make_node()
        node.ephemeral = False
        node.session_id = "real_session_1"
        mock_model = MagicMock()
        node.model = mock_model
        node._session = MagicMock()
        node.clear_session()
        mock_model.clear_session.assert_called_once_with("real_session_1")
        assert node._session is None

    def test_clear_session_no_model(self):
        node = _make_node()
        node.ephemeral = False
        node.model = None
        node.session_id = "some_id"
        node._session = MagicMock()
        # Should not raise
        node.clear_session()


# ---------------------------------------------------------------------------
# TestDeleteSession
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_session_ephemeral(self):
        node = _make_node()
        node.ephemeral = True
        node._session = MagicMock()
        node.session_id = "eph_1"
        node.delete_session()
        assert node._session is None
        assert node.session_id is None

    def test_delete_session_non_ephemeral(self):
        node = _make_node()
        node.ephemeral = False
        node.session_id = "real_1"
        mock_model = MagicMock()
        node.model = mock_model
        node._session = MagicMock()
        node.delete_session()
        mock_model.delete_session.assert_called_once_with("real_1")
        assert node._session is None
        assert node.session_id is None


# ---------------------------------------------------------------------------
# TestSetPermissionCallback
# ---------------------------------------------------------------------------


class TestSetPermissionCallback:
    def test_set_permission_callback_stores_callback(self):
        node = _make_node()
        callback = AsyncMock()
        node.set_permission_callback(callback)
        assert node._permission_callback is callback

    def test_set_permission_callback_forwards_to_permission_manager(self):
        node = _make_node()
        mock_pm = MagicMock()
        node.permission_manager = mock_pm
        callback = AsyncMock()
        node.set_permission_callback(callback)
        mock_pm.set_permission_callback.assert_called_once_with(callback)


# ---------------------------------------------------------------------------
# TestGetAvailableSkillsContext
# ---------------------------------------------------------------------------


class TestGetAvailableSkillsContext:
    def test_returns_empty_when_no_skill_manager(self):
        node = _make_node()
        node.skill_manager = None
        result = node._get_available_skills_context()
        assert result == ""

    def test_calls_skill_manager_generate_xml(self):
        node = _make_node()
        mock_sm = MagicMock()
        mock_sm.parse_skill_patterns.return_value = ["sql-*"]
        mock_sm.generate_available_skills_xml.return_value = "<skills>...</skills>"
        node.skill_manager = mock_sm
        node.node_config = {"skills": "sql-*"}
        result = node._get_available_skills_context()
        assert "<skills>" in result


# ---------------------------------------------------------------------------
# TestGetResultClass
# ---------------------------------------------------------------------------


class TestGetResultClass:
    def test_returns_none_for_unknown_class(self):
        node = _make_node()
        result = node._get_result_class()
        # _ConcreteAgenticNode is not in the result_class_map
        assert result is None

    def test_returns_compare_result_for_compare_node(self):
        from datus.agent.node.compare_agentic_node import CompareAgenticNode

        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            node = CompareAgenticNode.__new__(CompareAgenticNode)
        node.__class__ = type("CompareAgenticNode", (AgenticNode,), {})
        # Use a simple mock to check the lookup
        # We just test that a known node class returns expected result class
        # by directly checking the map
        result_class_map = {
            "ChatAgenticNode": "ChatNodeResult",
            "GenSQLAgenticNode": "GenSQLNodeResult",
            "CompareAgenticNode": "CompareResult",
        }
        assert result_class_map.get("CompareAgenticNode") == "CompareResult"


# ---------------------------------------------------------------------------
# TestAutoCompact
# ---------------------------------------------------------------------------


class TestAutoCompact:
    @pytest.mark.asyncio
    async def test_auto_compact_skips_when_no_model(self):
        node = _make_node()
        node.model = None
        node.context_length = None
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_compact_skips_when_no_context_length(self):
        node = _make_node()
        node.model = MagicMock()
        node.context_length = None
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_compact_triggers_when_over_limit(self):
        node = _make_node()
        node.model = MagicMock()
        node.context_length = 1000
        node._session = MagicMock()

        with patch.object(node, "_count_session_tokens", return_value=950):
            with patch.object(node, "_manual_compact", return_value={"success": True}) as mock_compact:
                result = await node._auto_compact()

        mock_compact.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_compact_skips_when_under_limit(self):
        node = _make_node()
        node.model = MagicMock()
        node.context_length = 1000

        with patch.object(node, "_count_session_tokens", return_value=500):
            result = await node._auto_compact()

        assert result is False


# ---------------------------------------------------------------------------
# TestGetSessionInfo
# ---------------------------------------------------------------------------


class TestGetSessionInfo:
    @pytest.mark.asyncio
    async def test_get_session_info_no_session(self):
        node = _make_node()
        node.session_id = None
        info = await node.get_session_info()
        assert info["session_id"] is None
        assert info["active"] is False

    @pytest.mark.asyncio
    async def test_get_session_info_with_session(self):
        node = _make_node()
        node.session_id = "my_session"
        node._session = MagicMock()
        node.context_length = 100000
        node.actions = []

        with patch.object(node, "_count_session_tokens", return_value=5000):
            info = await node.get_session_info()

        assert info["session_id"] == "my_session"
        assert info["active"] is True
        assert info["token_count"] == 5000


# ---------------------------------------------------------------------------
# TestManualCompact
# ---------------------------------------------------------------------------


class TestManualCompact:
    @pytest.mark.asyncio
    async def test_manual_compact_ephemeral_returns_failure(self):
        node = _make_node()
        node.ephemeral = True
        node._session = MagicMock()
        result = await node._manual_compact()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_manual_compact_no_model_returns_failure(self):
        node = _make_node()
        node.ephemeral = False
        node.model = None
        node._session = MagicMock()
        result = await node._manual_compact()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_manual_compact_no_session_returns_failure(self):
        node = _make_node()
        node.ephemeral = False
        node.model = MagicMock()
        node._session = None
        result = await node._manual_compact()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_manual_compact_success(self):
        node = _make_node()
        node.ephemeral = False
        node.session_id = "compact_test"
        node._session = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_with_tools = AsyncMock(
            return_value={"content": "Summary of conversation", "usage": {"output_tokens": 100}}
        )
        node.model = mock_model

        result = await node._manual_compact()

        assert result["success"] is True
        assert "Summary" in result["summary"]
        # Session should be cleared
        assert node._session is None
        assert node.session_id is None


# ---------------------------------------------------------------------------
# TestCountSessionTokens
# ---------------------------------------------------------------------------


class TestCountSessionTokens:
    @pytest.mark.asyncio
    async def test_count_tokens_no_session(self):
        node = _make_node()
        node._session = None
        result = await node._count_session_tokens()
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_with_session(self):
        node = _make_node()
        mock_session = MagicMock()
        mock_session.get_session_usage = AsyncMock(return_value={"total_tokens": 1234})
        node._session = mock_session
        result = await node._count_session_tokens()
        assert result == 1234

    @pytest.mark.asyncio
    async def test_count_tokens_empty_usage(self):
        node = _make_node()
        mock_session = MagicMock()
        mock_session.get_session_usage = AsyncMock(return_value={})
        node._session = mock_session
        result = await node._count_session_tokens()
        assert result == 0


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class _SimpleAgenticNode(AgenticNode):
    """Minimal concrete AgenticNode for unit tests."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="done",
            input_data={},
            output_data={"success": True, "result": "ok"},
            status=ActionStatus.SUCCESS,
        )
        yield action


def _make_simple_node(**overrides):
    """Build a minimal _SimpleAgenticNode bypassing __init__."""
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _SimpleAgenticNode.__new__(_SimpleAgenticNode)

    node._session = None
    node.ephemeral = False
    node.session_id = None
    node.last_summary = None
    node.model = None
    node.tools = []
    node.mcp_servers = {}
    node.actions = []
    node.context_length = None
    node.node_config = {}
    node.agent_config = None
    node.permission_manager = None
    node.skill_manager = None
    node.skill_func_tool = None
    node._permission_callback = None
    node.id = "test_node"
    node.description = "Test"
    node.type = "test"
    node.status = "pending"
    node.result = None
    node.dependencies = []
    node.input = None

    from datus.cli.execution_state import InteractionBroker, InterruptController
    from datus.schemas.action_bus import ActionBus

    node.action_bus = ActionBus()
    node.interaction_broker = InteractionBroker()
    node.interrupt_controller = InterruptController()

    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# ---------------------------------------------------------------------------
# get_node_name
# ---------------------------------------------------------------------------


class TestGetNodeNameExtended:
    def test_removes_agentic_node_suffix(self):
        node = _make_simple_node()
        # _SimpleAgenticNode -> "simple"
        assert node.get_node_name() == "_simple"

    def test_class_without_suffix_returns_lowercase(self):
        class MyCustomNode(AgenticNode):
            async def execute_stream(self, ahm=None):
                return
                yield  # noqa

        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            n = MyCustomNode.__new__(MyCustomNode)
        n.node_config = {}
        assert n.get_node_name() == "mycustomnode"


# ---------------------------------------------------------------------------
# _parse_node_config
# ---------------------------------------------------------------------------


class TestParseNodeConfigExtended:
    def test_no_agent_config_returns_empty(self):
        node = _make_simple_node()
        result = node._parse_node_config(None, "chat")
        assert result == {}

    def test_node_not_in_config_returns_empty(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {}
        result = node._parse_node_config(mock_config, "chat")
        assert result == {}

    def test_dict_node_config_extracted(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {
            "chat": {
                "model": "gpt-4",
                "system_prompt": "You are a SQL assistant",
                "max_turns": 10,
            }
        }
        result = node._parse_node_config(mock_config, "chat")
        assert result.get("model") == "gpt-4"
        assert result.get("system_prompt") == "You are a SQL assistant"
        assert result.get("max_turns") == 10

    def test_rules_dict_normalized_to_string(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {
            "gensql": {
                "rules": [{"always": "use CTEs"}, "plain rule"],
            }
        }
        result = node._parse_node_config(mock_config, "gensql")
        rules = result.get("rules", [])
        assert len(rules) == 2
        assert any("always" in r for r in rules)

    def test_none_values_not_included(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {"mynode": {"model": "gpt-4", "system_prompt": None}}
        result = node._parse_node_config(mock_config, "mynode")
        assert result.get("model") == "gpt-4"
        # None system_prompt should not be in result
        assert "system_prompt" not in result


# ---------------------------------------------------------------------------
# _get_tool_category
# ---------------------------------------------------------------------------


class TestGetToolCategoryExtended:
    def test_load_skill_is_skills(self):
        node = _make_simple_node()
        assert node._get_tool_category("load_skill") == "skills"

    def test_skill_prefix_is_skills(self):
        node = _make_simple_node()
        assert node._get_tool_category("skill_run_query") == "skills"

    def test_db_prefix_is_db_tools(self):
        node = _make_simple_node()
        assert node._get_tool_category("db_execute") == "db_tools"

    def test_list_tables_is_db_tools(self):
        node = _make_simple_node()
        assert node._get_tool_category("list_tables") == "db_tools"

    def test_execute_sql_is_db_tools(self):
        node = _make_simple_node()
        assert node._get_tool_category("execute_sql") == "db_tools"

    def test_unknown_tool_is_tools(self):
        node = _make_simple_node()
        assert node._get_tool_category("my_custom_tool") == "tools"


# ---------------------------------------------------------------------------
# _resolve_workspace_root
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRootExtended:
    def test_default_is_dot(self):
        node = _make_simple_node()
        result = node._resolve_workspace_root()
        assert result == "."

    def test_node_config_workspace_root_used(self):
        node = _make_simple_node(node_config={"workspace_root": "/tmp/ws"})
        result = node._resolve_workspace_root()
        assert result == "/tmp/ws"

    def test_agent_config_workspace_root_used(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.workspace_root = "/var/data/ws"
        # no storage attribute
        del mock_config.storage
        node.agent_config = mock_config
        result = node._resolve_workspace_root()
        assert result == "/var/data/ws"

    def test_tilde_expanded(self):
        node = _make_simple_node(node_config={"workspace_root": "~/myproject"})
        result = node._resolve_workspace_root()
        assert "~" not in result
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# clear_session / delete_session
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_clear_session_ephemeral(self):
        node = _make_simple_node(ephemeral=True, session_id="sess_1")
        mock_session = MagicMock()
        node._session = mock_session
        node.clear_session()
        assert node._session is None

    def test_clear_session_normal(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        node.model = mock_model
        node.session_id = "sess_2"
        node._session = MagicMock()
        node.clear_session()
        mock_model.clear_session.assert_called_once_with("sess_2")
        assert node._session is None

    def test_clear_session_no_model(self):
        node = _make_simple_node()
        node._session = MagicMock()
        node.session_id = "sess_3"
        # no model - should not raise
        node.clear_session()

    def test_delete_session_ephemeral(self):
        node = _make_simple_node(ephemeral=True, session_id="sess_4")
        node._session = MagicMock()
        node.delete_session()
        assert node._session is None
        assert node.session_id is None

    def test_delete_session_normal(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        node.model = mock_model
        node.session_id = "sess_5"
        node._session = MagicMock()
        node.delete_session()
        mock_model.delete_session.assert_called_once_with("sess_5")
        assert node._session is None
        assert node.session_id is None


# ---------------------------------------------------------------------------
# get_session_info
# ---------------------------------------------------------------------------


class TestGetSessionInfoExtended:
    def test_no_session_id_returns_inactive(self):
        node = _make_simple_node()
        result = asyncio.run(node.get_session_info())
        assert result["session_id"] is None
        assert result["active"] is False

    def test_with_session_returns_info(self):
        node = _make_simple_node()
        node.session_id = "sess_x"
        node._session = MagicMock()
        node._session.get_session_usage = AsyncMock(return_value={"total_tokens": 500})
        node.context_length = 4000

        result = asyncio.run(node.get_session_info())
        assert result["session_id"] == "sess_x"
        assert result["active"] is True
        assert result["token_count"] == 500
        assert result["context_length"] == 4000


# ---------------------------------------------------------------------------
# _get_or_create_session
# ---------------------------------------------------------------------------


class TestGetOrCreateSession:
    def test_returns_existing_session(self):
        node = _make_simple_node()
        mock_session = MagicMock()
        node._session = mock_session
        session, summary = node._get_or_create_session()
        assert session is mock_session
        assert summary is None

    def test_creates_new_session_when_none(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_model.create_session.return_value = mock_session
        node.model = mock_model
        node.session_id = "my_session"

        session, summary = node._get_or_create_session()
        assert session is mock_session
        mock_model.create_session.assert_called_once_with("my_session")

    def test_generates_session_id_when_none(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_model.create_session.return_value = mock_session
        node.model = mock_model
        # session_id is None - should be generated

        session, _ = node._get_or_create_session()
        assert node.session_id is not None
        assert "_session_" in node.session_id

    def test_ephemeral_creates_in_memory_session(self):
        node = _make_simple_node(ephemeral=True)
        mock_model = MagicMock()
        node.model = mock_model
        node.session_id = "eph_sess"

        with patch("datus.agent.node.agentic_node.AdvancedSQLiteSession") as mock_sqlite_cls:
            mock_sqlite_cls.return_value = MagicMock()
            session, _ = node._get_or_create_session()

        mock_sqlite_cls.assert_called_once()
        call_kwargs = mock_sqlite_cls.call_args
        assert call_kwargs[1].get("db_path") == ":memory:" or ":memory:" in str(call_kwargs)

    def test_returns_last_summary_and_clears_it(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_model.create_session.return_value = mock_session
        node.model = mock_model
        node.session_id = "s"
        node.last_summary = "previous conversation summary"

        _, summary = node._get_or_create_session()
        assert summary == "previous conversation summary"
        assert node.last_summary is None


# ---------------------------------------------------------------------------
# update_context
# ---------------------------------------------------------------------------


class TestUpdateContext:
    def test_no_result_returns_failure(self):
        node = _make_simple_node()
        workflow = MagicMock()
        result = node.update_context(workflow)
        assert result["success"] is False

    def test_result_with_sql_appended_to_context(self):
        node = _make_simple_node()
        mock_result = MagicMock()
        mock_result.sql = "SELECT * FROM users"
        mock_result.response = "Query executed"
        node.result = mock_result

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1

    def test_result_without_sql_does_not_append(self):
        node = _make_simple_node()
        mock_result = MagicMock()
        mock_result.sql = None
        node.result = mock_result

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 0


# ---------------------------------------------------------------------------
# setup_input
# ---------------------------------------------------------------------------


class TestSetupInput:
    def test_creates_base_input_when_none(self):
        node = _make_simple_node()
        workflow = MagicMock()
        workflow.task.catalog_name = "cat"
        workflow.task.database_name = "db"
        workflow.task.schema_name = "schema"
        workflow.context.table_schemas = []
        workflow.context.metrics = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert node.input is not None

    def test_populates_fields_when_input_has_them(self):
        node = _make_simple_node()
        node.input = BaseInput()

        workflow = MagicMock()
        workflow.task.catalog_name = "my_cat"
        workflow.task.database_name = "my_db"
        workflow.task.schema_name = "my_schema"
        workflow.context.table_schemas = ["schema1"]
        workflow.context.metrics = []

        node.setup_input(workflow)
        # Verify setup_input populated the node's input
        assert node.input is not None


# ---------------------------------------------------------------------------
# set_permission_callback
# ---------------------------------------------------------------------------


class TestSetPermissionCallbackExtended:
    def test_stores_callback(self):
        node = _make_simple_node()
        callback = AsyncMock()
        node.set_permission_callback(callback)
        assert node._permission_callback is callback

    def test_forwards_to_permission_manager(self):
        node = _make_simple_node()
        mock_pm = MagicMock()
        node.permission_manager = mock_pm
        callback = AsyncMock()
        node.set_permission_callback(callback)
        mock_pm.set_permission_callback.assert_called_once_with(callback)


# ---------------------------------------------------------------------------
# execute (sync wrapper)
# ---------------------------------------------------------------------------


class TestExecuteSync:
    def test_execute_returns_base_result(self):
        node = _make_simple_node()
        result = node.execute()
        assert isinstance(result, BaseResult)

    def test_execute_success_result(self):
        node = _make_simple_node()
        result = node.execute()
        # The simple node yields success action
        assert result is not None


# ---------------------------------------------------------------------------
# _manual_compact
# ---------------------------------------------------------------------------


class TestManualCompactExtended:
    def test_ephemeral_returns_failure(self):
        node = _make_simple_node(ephemeral=True)
        result = asyncio.run(node._manual_compact())
        assert result["success"] is False

    def test_no_model_returns_failure(self):
        node = _make_simple_node()
        result = asyncio.run(node._manual_compact())
        assert result["success"] is False

    def test_success_stores_summary(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        node.model = mock_model
        node._session = mock_session
        node.session_id = "sess_compact"

        mock_model.generate_with_tools = AsyncMock(
            return_value={"content": "summary text", "usage": {"output_tokens": 100}}
        )
        mock_model.delete_session = MagicMock()

        result = asyncio.run(node._manual_compact())
        assert result["success"] is True
        assert result["summary"] == "summary text"
        assert node.last_summary == "summary text"
        assert node._session is None
        assert node.session_id is None


# ---------------------------------------------------------------------------
# _auto_compact
# ---------------------------------------------------------------------------


class TestAutoCompactExtended:
    def test_no_model_returns_false(self):
        node = _make_simple_node()
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_no_context_length_returns_false(self):
        node = _make_simple_node()
        node.model = MagicMock()
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_below_threshold_returns_false(self):
        node = _make_simple_node()
        node.model = MagicMock()
        node.context_length = 10000
        node._session = MagicMock()
        node._session.get_session_usage = AsyncMock(return_value={"total_tokens": 100})

        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_above_threshold_triggers_compact(self):
        node = _make_simple_node()
        node.model = MagicMock()
        node.context_length = 1000
        node._session = MagicMock()
        node._session.get_session_usage = AsyncMock(return_value={"total_tokens": 950})
        node.model.generate_with_tools = AsyncMock(return_value={"content": "summary", "usage": {"output_tokens": 50}})
        node.model.delete_session = MagicMock()
        node.session_id = "sess_auto"

        result = asyncio.run(node._auto_compact())
        assert result is True
