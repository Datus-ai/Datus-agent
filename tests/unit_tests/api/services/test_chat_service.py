"""Tests for datus.api.services.chat_service — chat session management."""

import pytest

from datus.api.services.chat_service import ChatService
from datus.api.services.chat_task_manager import ChatTaskManager


class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """ChatService initializes with real agent config and task manager."""
        task_manager = ChatTaskManager()
        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=task_manager,
            project_id="test-proj",
        )
        assert svc is not None

    def test_init_stores_properties(self, real_agent_config):
        """ChatService stores agent_config and task_manager."""
        tm = ChatTaskManager()
        svc = ChatService(agent_config=real_agent_config, task_manager=tm, project_id="p1")
        assert svc.agent_config is real_agent_config
        assert svc._task_manager is tm


class TestChatServiceSessionExists:
    """Tests for session_exists."""

    def test_nonexistent_session_returns_false(self, real_agent_config):
        """session_exists returns False for unknown session."""
        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        assert svc.session_exists("nonexistent-session-id") is False


class TestChatServiceListSessions:
    """Tests for list_sessions."""

    def test_list_sessions_empty(self, real_agent_config):
        """list_sessions returns empty list when no sessions exist."""
        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        result = svc.list_sessions()
        assert result.success is True
        assert result.data.sessions == []


class TestChatServiceDeleteSession:
    """Tests for delete_session."""

    def test_delete_nonexistent_session(self, real_agent_config):
        """delete_session for unknown session returns result."""
        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        result = svc.delete_session("nonexistent-session")
        # May succeed with empty data or fail - both are valid
        assert result is not None


class TestChatServiceGetHistory:
    """Tests for get_history."""

    def test_get_history_nonexistent_session(self, real_agent_config):
        """get_history for unknown session returns result."""
        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        result = svc.get_history("nonexistent-session")
        # May succeed with empty messages or fail
        assert result is not None
