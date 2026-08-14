"""Tests for datus.api.services.chat_service — chat session management."""

import asyncio
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import pytest

from datus.api.services.chat_service import (
    _DEFAULT_MAX_SESSION_PAGE_SIZE,
    _DEFAULT_SESSION_PAGE_SIZE,
    ChatService,
)
from datus.api.services.chat_task_manager import ChatTaskManager
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.exceptions import ErrorCode


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

    def test_init_with_real_config(self, chat_svc, real_agent_config):
        """ChatService initializes with real agent config and task manager."""
        assert chat_svc.agent_config is real_agent_config
        assert isinstance(chat_svc._task_manager, ChatTaskManager)

    def test_init_stores_properties(self, real_agent_config):
        """ChatService stores agent_config and task_manager."""
        tm = ChatTaskManager()
        svc = ChatService(agent_config=real_agent_config, task_manager=tm, project_id="p1")
        assert svc.agent_config is real_agent_config
        assert svc._task_manager is tm

    def test_init_sets_session_dir(self, chat_svc, real_agent_config):
        """ChatService sets _session_dir from agent_config."""
        assert chat_svc._session_dir == real_agent_config.session_dir


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

    def test_list_sessions_filters_by_subagent_id(self, chat_svc):
        """subagent_id='gen_metrics' keeps only sessions whose prefix matches."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("chat_session_a")
        sm.create_session("gen_metrics_session_a")
        sm.create_session("gen_metrics_session_b")

        result = chat_svc.list_sessions(subagent_id="gen_metrics")
        assert result.success is True
        session_ids = {s.session_id for s in result.data.sessions}
        assert session_ids == {"gen_metrics_session_a", "gen_metrics_session_b"}

    def test_list_sessions_filter_chat_includes_legacy(self, chat_svc):
        """subagent_id='chat' returns chat-prefixed and legacy (no-prefix) ids, but not subagents."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("chat_session_a")
        sm.create_session("legacy-id-1")
        sm.create_session("gen_metrics_session_a")

        result = chat_svc.list_sessions(subagent_id="chat")
        assert result.success is True
        session_ids = {s.session_id for s in result.data.sessions}
        assert session_ids == {"chat_session_a", "legacy-id-1"}

    def test_list_sessions_no_filter_returns_all(self, chat_svc):
        """subagent_id=None returns sessions for every agent."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("chat_session_a")
        sm.create_session("gen_metrics_session_a")

        result = chat_svc.list_sessions()
        assert result.success is True
        session_ids = {s.session_id for s in result.data.sessions}
        assert {"chat_session_a", "gen_metrics_session_a"} <= session_ids

    def test_list_sessions_timestamps_use_iso_z_format(self, chat_svc):
        """created_at / last_updated must be ISO-8601 UTC with 'Z' suffix.

        Regression guard: previously these fields were emitted as bare SQLite
        ``YYYY-MM-DD HH:MM:SS`` strings (no timezone), so clients could not
        convert them to local time correctly.
        """
        import os
        import re
        import sqlite3
        from datetime import datetime

        # Build a session DB with explicit naive UTC timestamps in agent_sessions
        # and one user message, mirroring the schema OpenAI Agents SDK creates.
        session_id = "ts-format-session"
        db_path = os.path.join(chat_svc._session_dir, f"{session_id}.db")
        os.makedirs(chat_svc._session_dir, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE agent_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE agent_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "message_data TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO agent_sessions (session_id, created_at, updated_at) "
                "VALUES (?, '2026-04-30 12:34:56', '2026-04-30 12:35:10')",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) "
                "VALUES (?, ?, '2026-04-30 12:34:56')",
                (session_id, '{"role": "user", "content": "Hi"}'),
            )

        result = chat_svc.list_sessions()
        target = next(s for s in result.data.sessions if s.session_id == session_id)

        assert target.created_at == "2026-04-30T12:34:56Z"
        assert target.last_updated == "2026-04-30T12:35:10Z"
        iso_z_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        assert iso_z_pattern.match(target.created_at), target.created_at
        assert iso_z_pattern.match(target.last_updated), target.last_updated
        # And they must round-trip through fromisoformat as aware UTC datetimes.
        from datetime import timezone

        parsed = datetime.fromisoformat(target.created_at.replace("Z", "+00:00"))
        assert parsed == datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)


class TestChatServiceListSessionsPagination:
    """offset/limit must slice *before* the expensive per-session get_session_info()
    call, so cost stops scaling with a user's total lifetime session count
    (regression test for the timeout reported in Datus-ai/Datus-agent#1222)."""

    @staticmethod
    def _fake_session_manager(ids):
        fake = MagicMock()
        fake.list_sessions.return_value = list(ids)
        fake.get_session_info.side_effect = lambda sid: {
            "exists": True,
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "first_user_message": None,
            "message_count": 0,
            "total_tokens": 0,
        }
        return fake

    def test_uses_cheap_mtime_sort_instead_of_full_scan(self, chat_svc):
        """Ordering comes from the stat-based sort, not an eager per-file open."""
        fake = self._fake_session_manager([])
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            chat_svc.list_sessions()

        fake.list_sessions.assert_called_once_with(sort_by_modified=True)

    def test_total_count_reflects_all_matching_sessions_not_just_the_page(self, chat_svc):
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(offset=0, limit=5)

        assert result.success is True
        assert result.data.total_count == 50
        assert len(result.data.sessions) == 5

    def test_offset_and_limit_select_the_requested_page(self, chat_svc):
        ids = [f"session-{i}" for i in range(10)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(offset=4, limit=3)

        assert [s.session_id for s in result.data.sessions] == ids[4:7]

    def test_only_the_requested_page_is_enriched(self, chat_svc):
        """The actual performance fix: get_session_info must not be called for
        sessions outside the page, however many the user has accumulated."""
        ids = [f"session-{i}" for i in range(300)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            chat_svc.list_sessions(offset=0, limit=20)

        assert fake.get_session_info.call_count == 20

    def test_omitted_limit_still_honours_offset(self, chat_svc):
        """A page smaller than the default page size is returned whole."""
        ids = [f"session-{i}" for i in range(5)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(offset=2)

        assert [s.session_id for s in result.data.sessions] == ids[2:]
        assert result.data.total_count == 5

    def test_omitted_limit_falls_back_to_finite_default_page_size(self, chat_svc):
        """Clients that omit ``limit`` are the ones that timed out in #1222, so
        the default path must be bounded too — not just the opt-in one."""
        ids = [f"session-{i}" for i in range(300)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert len(result.data.sessions) == _DEFAULT_SESSION_PAGE_SIZE
        assert fake.get_session_info.call_count == _DEFAULT_SESSION_PAGE_SIZE
        assert result.data.total_count == 300

    def test_default_page_size_reads_from_api_config(self, chat_svc):
        chat_svc.agent_config.api_config = {"default_session_page_size": 7}
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert [s.session_id for s in result.data.sessions] == ids[:7]
        assert fake.get_session_info.call_count == 7

    def test_explicit_limit_above_max_is_clamped(self, chat_svc):
        """An over-large explicit limit must not re-open the unbounded path."""
        ids = [f"session-{i}" for i in range(500)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(limit=10_000)

        assert len(result.data.sessions) == _DEFAULT_MAX_SESSION_PAGE_SIZE
        assert fake.get_session_info.call_count == _DEFAULT_MAX_SESSION_PAGE_SIZE
        assert result.data.total_count == 500

    def test_max_page_size_reads_from_api_config(self, chat_svc):
        chat_svc.agent_config.api_config = {"max_session_page_size": 3}
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(offset=1, limit=100)

        assert [s.session_id for s in result.data.sessions] == ids[1:4]

    def test_explicit_limit_below_max_is_honoured_verbatim(self, chat_svc):
        chat_svc.agent_config.api_config = {"max_session_page_size": 100}
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(limit=4)

        assert len(result.data.sessions) == 4

    def test_configured_default_cannot_exceed_configured_max(self, chat_svc):
        """A misconfigured default larger than the max must not defeat the cap."""
        chat_svc.agent_config.api_config = {
            "default_session_page_size": 100,
            "max_session_page_size": 10,
        }
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert len(result.data.sessions) == 10

    @pytest.mark.parametrize("bad_value", [0, -1, None, "abc", "", []])
    def test_junk_page_size_config_falls_back_to_shipped_defaults(self, chat_svc, bad_value):
        """Operator YAML is untrusted: junk must not yield an empty or reversed page."""
        chat_svc.agent_config.api_config = {
            "default_session_page_size": bad_value,
            "max_session_page_size": bad_value,
        }
        ids = [f"session-{i}" for i in range(300)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert len(result.data.sessions) == _DEFAULT_SESSION_PAGE_SIZE

    def test_fractional_page_size_config_truncates_to_a_bounded_int(self, chat_svc):
        """A float still coerces to a positive int, so it bounds rather than falls back."""
        chat_svc.agent_config.api_config = {"default_session_page_size": 3.5}
        ids = [f"session-{i}" for i in range(50)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert [s.session_id for s in result.data.sessions] == ids[:3]

    @pytest.mark.parametrize("bad_limit", [0, -1, -3])
    def test_non_positive_limit_falls_back_to_the_default_page(self, chat_svc, bad_limit):
        """A negative limit would slice all_ids[offset:-n] and enrich nearly every
        session — the exact sqlite cost this pagination exists to bound."""
        ids = [f"session-{i}" for i in range(300)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(limit=bad_limit)

        assert len(result.data.sessions) == _DEFAULT_SESSION_PAGE_SIZE
        assert fake.get_session_info.call_count == _DEFAULT_SESSION_PAGE_SIZE

    def test_infinite_page_size_config_does_not_break_the_endpoint(self, chat_svc):
        """YAML ``.inf`` parses to float infinity, which int() refuses to convert;
        an uncaught OverflowError would fail every session-list request."""
        chat_svc.agent_config.api_config = {
            "default_session_page_size": float("inf"),
            "max_session_page_size": float("inf"),
        }
        ids = [f"session-{i}" for i in range(300)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions()

        assert result.success is True
        assert len(result.data.sessions) == _DEFAULT_SESSION_PAGE_SIZE

    def test_negative_offset_is_clamped_to_the_first_page(self, chat_svc):
        """The route pins ge=0; a direct caller must not slice from the tail."""
        ids = [f"session-{i}" for i in range(10)]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(offset=-5, limit=3)

        assert [s.session_id for s in result.data.sessions] == ids[:3]

    def test_subagent_filter_applied_before_pagination(self, chat_svc):
        ids = ["chat_session_a", "gen_metrics_session_a", "chat_session_b", "gen_metrics_session_b", "chat_session_c"]
        fake = self._fake_session_manager(ids)
        with patch("datus.api.services.chat_service.SessionManager", return_value=fake):
            result = chat_svc.list_sessions(subagent_id="chat", offset=1, limit=1)

        assert [s.session_id for s in result.data.sessions] == ["chat_session_b"]
        assert result.data.total_count == 3


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
        assert result.data.messages == []

    def test_get_history_empty_session_returns_success(self, chat_svc):
        """get_history for empty session returns success with empty messages."""
        sm = SessionManager(session_dir=chat_svc._session_dir)
        sm.create_session("empty-hist")

        result = chat_svc.get_history("empty-hist")
        assert result.success is True

    def _assistant_response(self, action_id: str, text: str) -> ActionHistory:
        return ActionHistory(
            action_id=action_id,
            role=ActionRole.ASSISTANT,
            action_type="response",
            status=ActionStatus.SUCCESS,
            output={"is_thinking": False, "response": text},
        )

    def _patch_messages(self, raw_messages: list[dict]) -> AbstractContextManager[MagicMock]:
        fake_sm = MagicMock()
        fake_sm.get_session_messages.return_value = raw_messages
        return patch("datus.api.services.chat_service.SessionManager", return_value=fake_sm)

    def test_get_history_renders_assistant_actions(self, chat_svc):
        """Assistant actions convert to SSE payloads instead of erroring out."""
        raw = [{"role": "assistant", "actions": [self._assistant_response("a1", "hello")]}]
        with self._patch_messages(raw):
            result = chat_svc.get_history("sid")

        assert result.success is True
        assert len(result.data.messages) == 1
        assert result.data.messages[0].content[0].payload["content"] == "hello"

    def test_get_history_dedupes_repeated_assistant_text(self, chat_svc):
        """The fingerprint store is a dict of text -> owning message_id, not a set.

        Passing a set made ``_should_skip_duplicate_assistant_message`` raise
        ``'set' object has no attribute 'get'`` and failed the whole endpoint.
        """
        # Distinct action_ids so only the repeated text can drive the dedup.
        raw = [
            {
                "role": "assistant",
                "actions": [self._assistant_response("a1", "same"), self._assistant_response("a2", "same")],
            }
        ]
        with self._patch_messages(raw):
            result = chat_svc.get_history("sid")

        assert result.success is True
        assert len(result.data.messages) == 1


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
        """compact_session auto-creates and compacts a fresh empty session — the call must
        never raise, and the typed Result must round-trip the requested session_id."""
        from datus.api.models.cli_models import CompactSessionInput

        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        request = CompactSessionInput(session_id="nonexistent")
        result = await svc.compact_session(request)

        assert result.success is True
        assert result.data.session_id == "nonexistent"

    async def test_compact_persists_summary_into_session(self, real_agent_config, mock_llm_create):
        """End-to-end: compact must keep the .db alive and write a
        user-marker + assistant-summary pair back into the same session.

        This is the regression coverage for the original bug where compact
        deleted the .db and stored the summary only in the discarded node's
        memory, so UI history reads got an empty session and the next
        chat turn had no summary context.
        """
        from datus.api.models.cli_models import CompactSessionInput

        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )

        # Route the mock LLM's session manager to the real on-disk session_dir
        # so that ChatAgenticNode._get_or_create_session loads the same .db
        # we pre-populate below.
        real_sm = SessionManager(session_dir=svc._session_dir)
        mock_llm_create._session_manager = real_sm

        # Pre-create a real chat session with two Q/A pairs.
        session_id = "chat_session_compact_test"
        seeded = real_sm.create_session(session_id)
        await seeded.add_items(
            [
                {"role": "user", "content": "What tables are there?"},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Tables: schools, frpm"}]},
                {"role": "user", "content": "Describe schools."},
                {"role": "assistant", "content": [{"type": "output_text", "text": "schools has cols a, b, c"}]},
            ]
        )

        # Patch the mock LLM's generate_with_tools to return a deterministic
        # summary for the summarization prompt issued inside _manual_compact.
        from unittest.mock import AsyncMock as _AsyncMock

        mock_llm_create.generate_with_tools = _AsyncMock(
            return_value={"content": "Summary of conversation", "usage": {"output_tokens": 42}}
        )

        request = CompactSessionInput(session_id=session_id)
        result = await svc.compact_session(request)

        assert result.success is True
        assert result.data.success is True

        # The .db file must still exist — compact no longer deletes it.
        import os

        db_path = os.path.join(svc._session_dir, f"{session_id}.db")
        assert os.path.exists(db_path), "Session .db must be preserved after compact"

        # Re-open the session via a fresh SessionManager to bypass any
        # in-memory caches and verify on-disk state.
        verify_sm = SessionManager(session_dir=svc._session_dir)
        verify_session = verify_sm.get_session(session_id)
        items = await verify_session.get_items()

        # After the compact refactor, the session contains a single assistant
        # message carrying the summary + a JSONL recovery pointer appended by
        # the host. Storing as ``assistant`` (not ``user``) makes the next
        # turn see the summary as a prior assistant utterance — the natural
        # shape for "I summarized previously, now answer the next question",
        # and avoids /chat/history rendering a phantom user message.
        assert len(items) == 1
        assert items[0]["role"] == "assistant"
        content_blocks = items[0]["content"]
        assert isinstance(content_blocks, list) and len(content_blocks) == 1
        assert content_blocks[0]["type"] == "output_text"
        body = content_blocks[0]["text"]
        assert "Summary of conversation" in body


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
        assert len(events) >= 1
        assert events[0].event == "session"

    async def test_stream_chat_unknown_model_yields_model_not_configured(self, real_agent_config, mock_llm_create):
        """A model param naming a connection this project no longer has is
        reported under its own error_type, which is what remote front-ends
        match on to refresh their model list instead of parsing the message."""
        from datus.api.models.cli_models import StreamChatInput

        svc = ChatService(
            agent_config=real_agent_config,
            task_manager=ChatTaskManager(),
            project_id="test-proj",
        )
        request = StreamChatInput(
            message="hello",
            session_id="unknown-model",
            model="custom/detached-connection",
        )

        events = [event async for event in svc.stream_chat(request)]

        assert len(events) == 1
        assert events[0].event == "error"
        assert events[0].data.error_type == ErrorCode.MODEL_NOT_CONFIGURED.name
        assert "detached-connection" in events[0].data.error

    async def test_stream_chat_duplicate_session_yields_error(self, real_agent_config, mock_llm_create):
        """stream_chat for duplicate session_id yields error event."""
        from datus.api.models.cli_models import StreamChatInput

        tm = ChatTaskManager()
        svc = ChatService(agent_config=real_agent_config, task_manager=tm, project_id="test-proj")

        release_first_task = asyncio.Event()

        class BlockingNode:
            """Keep the first task running so duplicate-session handling is deterministic."""

            def __init__(self, session_id: str):
                self.session_id = session_id

            async def execute_stream_with_interactions(self, action_history):
                if False:
                    yield None
                await release_first_task.wait()

            async def get_last_turn_usage(self):
                return None

        # Mock _create_node to avoid real storage initialization and keep the task active.
        with patch.object(tm, "_create_node", return_value=BlockingNode("dup-stream")):
            request1 = StreamChatInput(message="first", session_id="dup-stream")
            stream1 = svc.stream_chat(request1)
            stream2 = None
            try:
                first_event = await asyncio.wait_for(anext(stream1), timeout=2)
                assert first_event.event == "session"
                assert "dup-stream" in tm._tasks

                request2 = StreamChatInput(message="second", session_id="dup-stream")
                stream2 = svc.stream_chat(request2)
                duplicate_event = await asyncio.wait_for(anext(stream2), timeout=2)
                assert duplicate_event.event == "error"
            finally:
                release_first_task.set()
                await stream1.aclose()
                if stream2 is not None:
                    await stream2.aclose()
                await tm.shutdown()
