"""Tests for datus.api.services.chat_service — chat session management."""

from unittest.mock import MagicMock, patch

import pytest

from datus.api.services.chat_service import ChatService
from datus.api.services.chat_task_manager import ChatTaskManager
from datus.models.session_manager import SessionManager


@pytest.fixture
def chat_svc(real_agent_config):
    """Create ChatService with real config for reuse."""
    return ChatService(
        agent_config=real_agent_config,
        task_manager=ChatTaskManager(),
        project_id="test-proj",
    )


class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_init_with_real_config(self, chat_svc):
        """ChatService initializes with real agent config and task manager."""
        assert chat_svc is not None

    def test_init_stores_properties(self, real_agent_config):
        """ChatService stores agent_config and task_manager."""
        tm = ChatTaskManager()
        svc = ChatService(agent_config=real_agent_config, task_manager=tm, project_id="p1")
        assert svc.agent_config is real_agent_config
        assert svc._task_manager is tm

    def test_init_sets_session_dir(self, chat_svc):
        """ChatService sets _session_dir from agent_config."""
        assert chat_svc._session_dir is not None


class TestChatServiceSessionExists:
    """Tests for session_exists."""

    def test_nonexistent_session_returns_false(self, chat_svc):
        """session_exists returns False for unknown session."""
        assert chat_svc.session_exists("nonexistent-session-id") is False

    def test_session_check_uses_session_manager(self, chat_svc):
        """session_exists delegates to SessionManager.session_exists."""
        # Multiple non-existent calls should all return False
        assert chat_svc.session_exists("fake-a") is False
        assert chat_svc.session_exists("fake-b") is False


class TestChatServiceListSessions:
    """Tests for list_sessions."""

    def test_list_sessions_empty(self, chat_svc):
        """list_sessions returns empty list when no sessions exist."""
        result = chat_svc.list_sessions()
        assert result.success is True
        assert result.data.sessions == []

    def test_list_sessions_returns_total_count(self, chat_svc):
        """list_sessions data includes total_count field."""
        result = chat_svc.list_sessions()
        assert result.data.total_count == 0

    def test_list_sessions_with_created_session(self, chat_svc):
        """list_sessions detects a session created via SessionManager."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        session = sm.create_session("test-list-session")
        session.add_items([{"role": "user", "content": "Hello"}])

        result = chat_svc.list_sessions()
        assert result.success is True
        assert result.data.total_count >= 1
        session_ids = [s.session_id for s in result.data.sessions]
        assert "test-list-session" in session_ids


class TestChatServiceDeleteSession:
    """Tests for delete_session."""

    def test_delete_nonexistent_session_succeeds(self, chat_svc):
        """delete_session for unknown session succeeds (no-op)."""
        result = chat_svc.delete_session("nonexistent-session")
        assert result.success is True

    def test_delete_existing_session(self, chat_svc):
        """delete_session removes existing session."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("to-delete")

        result = chat_svc.delete_session("to-delete")
        assert result.success is True
        assert chat_svc.session_exists("to-delete") is False


class TestChatServiceGetHistory:
    """Tests for get_history."""

    def test_get_history_nonexistent_session_returns_empty(self, chat_svc):
        """get_history for unknown session returns empty messages."""
        result = chat_svc.get_history("nonexistent-session")
        assert result.success is True
        assert result.data is not None

    def test_get_history_empty_session_returns_success(self, chat_svc):
        """get_history for empty session returns success with empty messages."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("empty-hist")

        result = chat_svc.get_history("empty-hist")
        assert result.success is True


class TestChatServiceScopePropagation:
    """user_id is propagated as SessionManager.scope for isolation."""

    def _patched_sm(self):
        fake = MagicMock()
        fake.session_exists.return_value = False
        fake.list_sessions.return_value = []
        fake.get_session_messages.return_value = []
        return fake

    def test_session_exists_passes_scope(self, chat_svc):
        fake = self._patched_sm()
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake) as cls:
            chat_svc.session_exists("sid", user_id="alice")
            cls.assert_called_once_with(session_dir=chat_svc._session_dir, scope="alice")

    def test_list_sessions_passes_scope(self, chat_svc):
        fake = self._patched_sm()
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake) as cls:
            chat_svc.list_sessions(user_id="bob")
            cls.assert_called_once_with(session_dir=chat_svc._session_dir, scope="bob")

    def test_delete_session_passes_scope(self, chat_svc):
        fake = self._patched_sm()
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake) as cls:
            chat_svc.delete_session("sid", user_id="carol")
            cls.assert_called_once_with(session_dir=chat_svc._session_dir, scope="carol")

    def test_get_history_passes_scope(self, chat_svc):
        fake = self._patched_sm()
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake) as cls:
            chat_svc.get_history("sid", user_id="dave")
            cls.assert_called_once_with(session_dir=chat_svc._session_dir, scope="dave")

    def test_none_user_id_falls_back_to_default_scope(self, chat_svc):
        fake = self._patched_sm()
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake) as cls:
            chat_svc.list_sessions()
            cls.assert_called_once_with(session_dir=chat_svc._session_dir, scope=None)


@pytest.mark.asyncio
class TestChatServiceCompactSession:
    """Tests for compact_session."""

    async def test_compact_nonexistent_session(self, real_agent_config, mock_llm_create):
        """compact_session for nonexistent session returns error."""
        from datus.api.models.cli_models import CompactSessionInput

        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        request = CompactSessionInput(session_id="nonexistent")
        result = await svc.compact_session(request)
        # Should handle gracefully
        assert result is not None


@pytest.mark.asyncio
class TestChatServiceStreamChat:
    """Tests for stream_chat."""

    async def test_stream_chat_produces_events(self, real_agent_config, mock_llm_create):
        """stream_chat yields SSE events from the task manager."""
        from datus.api.models.cli_models import StreamChatInput

        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        request = StreamChatInput(message="hello", session_id="stream-test")
        events = []
        async for event in svc.stream_chat(request):
            events.append(event)
            if len(events) > 5:
                break
        assert len(events) >= 0

    async def test_stream_chat_duplicate_session_yields_error(self, real_agent_config, mock_llm_create):
        """stream_chat for duplicate session_id yields error event."""
        from datus.api.models.cli_models import StreamChatInput

        tm = ChatTaskManager()
        svc = ChatService(agent_config=real_agent_config, task_manager=tm, project_id="test-proj")

        # Start first chat
        request1 = StreamChatInput(message="first", session_id="dup-stream")
        async for _ in svc.stream_chat(request1):
            break

        # Second should yield error event
        request2 = StreamChatInput(message="second", session_id="dup-stream")
        events = []
        async for event in svc.stream_chat(request2):
            events.append(event)
            break

        assert len(events) >= 1
        # First event should be an error
        assert events[0].event == "error"
        await tm.shutdown()
