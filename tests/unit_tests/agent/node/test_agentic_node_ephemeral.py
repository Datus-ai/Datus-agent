# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for AgenticNode ephemeral (in-memory) session support."""

from typing import AsyncGenerator, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from agents.extensions.memory import AdvancedSQLiteSession

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool


class _ConcreteAgenticNode(AgenticNode):
    """Minimal concrete subclass of AgenticNode for testing."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        yield  # pragma: no cover


def _make_node(**overrides):
    """Create a _ConcreteAgenticNode with __init__ bypassed, setting only the attributes needed."""
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _ConcreteAgenticNode.__new__(_ConcreteAgenticNode)
    # Set default attributes expected by _get_or_create_session
    node._session = None
    node.ephemeral = False
    node.session_id = None
    node.model = None
    # Apply overrides
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


@pytest.fixture
def mock_agent_config():
    """Minimal AgentConfig mock for testing."""
    config = Mock(spec=AgentConfig)
    config.db_type = "sqlite"
    config.current_database = "test_db"
    config.current_database = "default"
    config.agentic_nodes = {
        "chat": {"model": "default"},
        "gen_sql": {"model": "default", "system_prompt": "gen_sql", "node_class": "gen_sql"},
    }
    config.sub_agent_config.side_effect = lambda name: config.agentic_nodes.get(name)
    return config


# ── Ephemeral session on AgenticNode ──────────────────────────────


@pytest.mark.ci
class TestEphemeralSession:
    def test_ephemeral_default_false(self):
        """AgenticNode.ephemeral defaults to False."""
        node = _make_node()
        assert node.ephemeral is False

    def test_ephemeral_session_is_memory(self):
        """When ephemeral=True, _get_or_create_session() creates an in-memory AdvancedSQLiteSession."""
        node = _make_node(ephemeral=True, model=MagicMock())
        node._generate_session_id = lambda: "test_ephemeral_001"

        session, summary = node._get_or_create_session()

        assert session is not None
        assert isinstance(session, AdvancedSQLiteSession)
        assert session.db_path == ":memory:"
        assert summary is None

    def test_ephemeral_session_not_in_session_manager(self):
        """In-memory session is NOT registered in SessionManager (bypasses model.create_session)."""
        mock_model = MagicMock()
        node = _make_node(ephemeral=True, session_id="eph_session_123", model=mock_model)

        node._get_or_create_session()

        # model.create_session should NOT have been called
        mock_model.create_session.assert_not_called()

    def test_non_ephemeral_uses_model_create_session(self):
        """When ephemeral=False (default), _get_or_create_session() delegates to model.create_session()."""
        mock_model = MagicMock()
        mock_session = MagicMock(spec=AdvancedSQLiteSession)
        mock_model.create_session.return_value = mock_session

        node = _make_node(ephemeral=False, session_id="normal_session_456", model=mock_model)

        session, summary = node._get_or_create_session()

        mock_model.create_session.assert_called_once_with("normal_session_456")
        assert session is mock_session

    def test_ephemeral_session_summary_slot_is_none(self):
        """After the compact-persistence refactor, _get_or_create_session
        no longer carries forward a compacted summary through a node
        attribute; the summary slot must always be None."""
        node = _make_node(
            ephemeral=True,
            session_id="eph_summary_001",
            model=MagicMock(),
        )

        session, summary = node._get_or_create_session()

        assert summary is None
        assert session is not None


# ── SubAgentTaskTool sets ephemeral ───────────────────────────────


@pytest.mark.ci
class TestSubAgentTaskSetsEphemeral:
    @pytest.mark.asyncio
    async def test_execute_node_sets_ephemeral(self, mock_agent_config):
        """SubAgentTaskTool._execute_node() marks the created node as ephemeral."""
        tool = SubAgentTaskTool(agent_config=mock_agent_config)

        mock_action = Mock(spec=ActionHistory)
        mock_action.role = ActionRole.TOOL
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()
        mock_node.ephemeral = False  # start as non-ephemeral

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(tool, "_create_node", return_value=mock_node):
            with patch.object(tool, "_build_node_input", return_value=Mock()):
                await tool._execute_node("gen_sql", "test query")

        # After _execute_node, node.ephemeral should have been set to True
        assert mock_node.ephemeral is True
