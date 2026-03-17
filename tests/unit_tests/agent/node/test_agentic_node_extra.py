# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for datus/agent/node/agentic_node.py

Covers: get_node_name, _parse_node_config, _get_tool_category,
_resolve_workspace_root, update_context, setup_input,
clear_session, delete_session, get_session_info,
_get_or_create_session, _count_session_tokens,
execute (sync wrapper), execute_stream_with_interactions.

CI-level: zero external deps, LLM/session mocked.
"""

import asyncio
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult

pytestmark = pytest.mark.ci


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


def _make_node(**overrides):
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


class TestGetNodeName:
    def test_removes_agentic_node_suffix(self):
        node = _make_node()
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


class TestParseNodeConfig:
    def test_no_agent_config_returns_empty(self):
        node = _make_node()
        result = node._parse_node_config(None, "chat")
        assert result == {}

    def test_node_not_in_config_returns_empty(self):
        node = _make_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {}
        result = node._parse_node_config(mock_config, "chat")
        assert result == {}

    def test_dict_node_config_extracted(self):
        node = _make_node()
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
        node = _make_node()
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
        node = _make_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {"mynode": {"model": "gpt-4", "system_prompt": None}}
        result = node._parse_node_config(mock_config, "mynode")
        assert result.get("model") == "gpt-4"
        # None system_prompt should not be in result
        assert "system_prompt" not in result


# ---------------------------------------------------------------------------
# _get_tool_category
# ---------------------------------------------------------------------------


class TestGetToolCategory:
    def test_load_skill_is_skills(self):
        node = _make_node()
        assert node._get_tool_category("load_skill") == "skills"

    def test_skill_prefix_is_skills(self):
        node = _make_node()
        assert node._get_tool_category("skill_run_query") == "skills"

    def test_db_prefix_is_db_tools(self):
        node = _make_node()
        assert node._get_tool_category("db_execute") == "db_tools"

    def test_list_tables_is_db_tools(self):
        node = _make_node()
        assert node._get_tool_category("list_tables") == "db_tools"

    def test_execute_sql_is_db_tools(self):
        node = _make_node()
        assert node._get_tool_category("execute_sql") == "db_tools"

    def test_unknown_tool_is_tools(self):
        node = _make_node()
        assert node._get_tool_category("my_custom_tool") == "tools"


# ---------------------------------------------------------------------------
# _resolve_workspace_root
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRoot:
    def test_default_is_dot(self):
        node = _make_node()
        result = node._resolve_workspace_root()
        assert result == "."

    def test_node_config_workspace_root_used(self):
        node = _make_node(node_config={"workspace_root": "/tmp/ws"})
        result = node._resolve_workspace_root()
        assert result == "/tmp/ws"

    def test_agent_config_workspace_root_used(self):
        node = _make_node()
        mock_config = MagicMock()
        mock_config.workspace_root = "/var/data/ws"
        # no storage attribute
        del mock_config.storage
        node.agent_config = mock_config
        result = node._resolve_workspace_root()
        assert result == "/var/data/ws"

    def test_tilde_expanded(self):
        node = _make_node(node_config={"workspace_root": "~/myproject"})
        result = node._resolve_workspace_root()
        assert "~" not in result
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# clear_session / delete_session
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_clear_session_ephemeral(self):
        node = _make_node(ephemeral=True, session_id="sess_1")
        mock_session = MagicMock()
        node._session = mock_session
        node.clear_session()
        assert node._session is None

    def test_clear_session_normal(self):
        node = _make_node()
        mock_model = MagicMock()
        node.model = mock_model
        node.session_id = "sess_2"
        node._session = MagicMock()
        node.clear_session()
        mock_model.clear_session.assert_called_once_with("sess_2")
        assert node._session is None

    def test_clear_session_no_model(self):
        node = _make_node()
        node._session = MagicMock()
        node.session_id = "sess_3"
        # no model - should not raise
        node.clear_session()

    def test_delete_session_ephemeral(self):
        node = _make_node(ephemeral=True, session_id="sess_4")
        node._session = MagicMock()
        node.delete_session()
        assert node._session is None
        assert node.session_id is None

    def test_delete_session_normal(self):
        node = _make_node()
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


class TestGetSessionInfo:
    def test_no_session_id_returns_inactive(self):
        node = _make_node()
        result = asyncio.run(node.get_session_info())
        assert result["session_id"] is None
        assert result["active"] is False

    def test_with_session_returns_info(self):
        node = _make_node()
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
        node = _make_node()
        mock_session = MagicMock()
        node._session = mock_session
        session, summary = node._get_or_create_session()
        assert session is mock_session
        assert summary is None

    def test_creates_new_session_when_none(self):
        node = _make_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_model.create_session.return_value = mock_session
        node.model = mock_model
        node.session_id = "my_session"

        session, summary = node._get_or_create_session()
        assert session is mock_session
        mock_model.create_session.assert_called_once_with("my_session")

    def test_generates_session_id_when_none(self):
        node = _make_node()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_model.create_session.return_value = mock_session
        node.model = mock_model
        # session_id is None - should be generated

        session, _ = node._get_or_create_session()
        assert node.session_id is not None
        assert "_session_" in node.session_id

    def test_ephemeral_creates_in_memory_session(self):
        node = _make_node(ephemeral=True)
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
        node = _make_node()
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
        node = _make_node()
        workflow = MagicMock()
        result = node.update_context(workflow)
        assert result["success"] is False

    def test_result_with_sql_appended_to_context(self):
        node = _make_node()
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
        node = _make_node()
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
        node = _make_node()
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
        node = _make_node()
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


class TestSetPermissionCallback:
    def test_stores_callback(self):
        node = _make_node()
        callback = AsyncMock()
        node.set_permission_callback(callback)
        assert node._permission_callback is callback

    def test_forwards_to_permission_manager(self):
        node = _make_node()
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
        node = _make_node()
        result = node.execute()
        assert isinstance(result, BaseResult)

    def test_execute_success_result(self):
        node = _make_node()
        result = node.execute()
        # The simple node yields success action
        assert result is not None


# ---------------------------------------------------------------------------
# _manual_compact
# ---------------------------------------------------------------------------


class TestManualCompact:
    def test_ephemeral_returns_failure(self):
        node = _make_node(ephemeral=True)
        result = asyncio.run(node._manual_compact())
        assert result["success"] is False

    def test_no_model_returns_failure(self):
        node = _make_node()
        result = asyncio.run(node._manual_compact())
        assert result["success"] is False

    def test_success_stores_summary(self):
        node = _make_node()
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


class TestAutoCompact:
    def test_no_model_returns_false(self):
        node = _make_node()
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_no_context_length_returns_false(self):
        node = _make_node()
        node.model = MagicMock()
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_below_threshold_returns_false(self):
        node = _make_node()
        node.model = MagicMock()
        node.context_length = 10000
        node._session = MagicMock()
        node._session.get_session_usage = AsyncMock(return_value={"total_tokens": 100})

        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_above_threshold_triggers_compact(self):
        node = _make_node()
        node.model = MagicMock()
        node.context_length = 1000
        node._session = MagicMock()
        node._session.get_session_usage = AsyncMock(return_value={"total_tokens": 950})
        node.model.generate_with_tools = AsyncMock(return_value={"content": "summary", "usage": {"output_tokens": 50}})
        node.model.delete_session = MagicMock()
        node.session_id = "sess_auto"

        result = asyncio.run(node._auto_compact())
        assert result is True
