# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Comprehensive unit tests for datus/cli/web/chatbot.py."""

import asyncio
import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level streamlit mock – must be installed before importing chatbot
# ---------------------------------------------------------------------------
_mock_st = MagicMock()
_mock_st.session_state = {}
_mock_st.query_params = {}
_mock_st.get_option.return_value = "localhost"
sys.modules.setdefault("streamlit", _mock_st)
sys.modules.setdefault("streamlit.components", MagicMock())
sys.modules.setdefault("streamlit.components.v1", MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SessionState(dict):
    """Dict subclass that also supports attribute access (like Streamlit's real SessionState)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


class _QueryParams(dict):
    """Dict subclass used for st.query_params."""


def _fresh_session_state():
    """Return a SessionState with the defaults StreamlitChatbot.__init__ would set."""
    return _SessionState(
        {
            "messages": [],
            "current_actions": [],
            "chat_session_initialized": False,
            "cli_instance": None,
            "current_chat_id": None,
            "subagent_name": None,
            "view_session_id": None,
            "session_readonly_mode": False,
        }
    )


def _make_mock_st(session_state=None, query_params=None):
    """Create a properly configured MagicMock for streamlit."""
    mock = MagicMock()
    mock.session_state = session_state if session_state is not None else _fresh_session_state()
    mock.query_params = query_params if query_params is not None else _QueryParams()
    mock.get_option.return_value = "localhost"
    return mock


_COMPONENT_PATCHES = [
    "datus.cli.web.chatbot.SessionManager",
    "datus.cli.web.chatbot.ChatExecutor",
    "datus.cli.web.chatbot.ConfigManager",
    "datus.cli.web.chatbot.UIComponents",
]


def _create_chatbot():
    """Create chatbot with all dependencies mocked.

    Caller must patch ``datus.cli.web.chatbot.st`` before calling this.
    """
    import contextlib

    from datus.cli.web.chatbot import StreamlitChatbot

    with contextlib.ExitStack() as stack:
        for p in _COMPONENT_PATCHES:
            stack.enter_context(patch(p))
        return StreamlitChatbot()


# ═══════════════════════════════════════════════════════════════════════════
# 1. _run_async
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestRunAsync:
    """Tests for the module-level _run_async helper."""

    def test_run_async_normal_coroutine(self):
        from datus.cli.web.chatbot import _run_async

        async def coro():
            return 42

        assert _run_async(coro()) == 42

    def test_run_async_returns_result(self):
        from datus.cli.web.chatbot import _run_async

        async def greeting():
            return "hello"

        assert _run_async(greeting()) == "hello"

    def test_run_async_closed_loop(self):
        """When the current loop is closed a new one should be created."""
        from datus.cli.web.chatbot import _run_async

        loop = asyncio.new_event_loop()
        loop.close()
        asyncio.set_event_loop(loop)

        async def value():
            return "ok"

        assert _run_async(value()) == "ok"

    def test_run_async_no_loop(self):
        """When there is no current loop a new one should be created."""
        from datus.cli.web.chatbot import _run_async

        try:
            asyncio.set_event_loop(None)
        except (RuntimeError, ValueError):
            # Some platforms/versions don't allow setting loop to None
            pass

        async def value():
            return "no_loop"

        assert _run_async(value()) == "no_loop"

    def test_run_async_running_loop(self):
        """When called inside a running loop, should use thread pool."""
        from datus.cli.web.chatbot import _run_async

        async def inner():
            return "from_thread"

        async def outer():
            return _run_async(inner())

        result = asyncio.run(outer())
        assert result == "from_thread"


# ═══════════════════════════════════════════════════════════════════════════
# 2. initialize_logging
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestInitializeLogging:
    """Tests for the module-level initialize_logging function."""

    def _reset(self):
        import datus.cli.web.chatbot as mod

        mod._LOGGING_INITIALIZED = False

    def test_initialize_logging_default(self):
        self._reset()
        from datus.cli.web.chatbot import initialize_logging

        with (
            patch("datus.cli.web.chatbot.configure_logging") as mock_conf,
            patch("datus.cli.web.chatbot.setup_web_chatbot_logging") as mock_setup,
            patch("datus.utils.path_manager.get_path_manager") as mock_pm,
        ):
            mock_pm.return_value.logs_dir = Path("/tmp/logs")
            mock_setup.return_value = MagicMock()
            initialize_logging()

            mock_conf.assert_called_once_with(debug=False, log_dir="/tmp/logs", console_output=False)
            mock_setup.assert_called_once()

    def test_initialize_logging_idempotent(self):
        self._reset()
        from datus.cli.web.chatbot import initialize_logging

        with (
            patch("datus.cli.web.chatbot.configure_logging") as mock_conf,
            patch("datus.cli.web.chatbot.setup_web_chatbot_logging") as mock_setup,
            patch("datus.utils.path_manager.get_path_manager") as mock_pm,
        ):
            mock_pm.return_value.logs_dir = Path("/tmp/logs")
            mock_setup.return_value = MagicMock()
            initialize_logging()
            initialize_logging()  # second call should be no-op
            assert mock_conf.call_count == 1

    def test_initialize_logging_custom_dir(self):
        self._reset()
        from datus.cli.web.chatbot import initialize_logging

        with (
            patch("datus.cli.web.chatbot.configure_logging") as mock_conf,
            patch("datus.cli.web.chatbot.setup_web_chatbot_logging") as mock_setup,
        ):
            mock_setup.return_value = MagicMock()
            initialize_logging(debug=True, log_dir="/custom/dir")
            mock_conf.assert_called_once_with(debug=True, log_dir="/custom/dir", console_output=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. sanitize_csv_field
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestSanitizeCsvField:
    """Tests for StreamlitChatbot.sanitize_csv_field (static method)."""

    def test_none_returns_none(self):
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field(None) is None

    def test_normal_string_unchanged(self):
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field("hello world") == "hello world"

    def test_empty_string_unchanged(self):
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field("") == ""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("=CMD()", "'=CMD()"),
            ("+1", "'+1"),
            ("-1", "'-1"),
            ("@SUM(A1)", "'@SUM(A1)"),
        ],
        ids=["equals", "plus", "minus", "at"],
    )
    def test_formula_prefix(self, input_val, expected):
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field(input_val) == expected

    def test_non_string_converted(self):
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field(12345) == "12345"

    def test_non_string_with_formula_prefix(self):
        """Negative numbers converted to str get sanitized."""
        from datus.cli.web.chatbot import StreamlitChatbot

        assert StreamlitChatbot.sanitize_csv_field(-99) == "'-99"


# ═══════════════════════════════════════════════════════════════════════════
# 4. StreamlitChatbot – properties
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestStreamlitChatbotProperties:
    """Tests for StreamlitChatbot property accessors."""

    def test_cli_getter(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            mock_cli = MagicMock()
            mock_st.session_state.cli_instance = mock_cli
            assert chatbot.cli is mock_cli

    def test_cli_setter(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            mock_cli = MagicMock()
            chatbot.cli = mock_cli
            assert mock_st.session_state.cli_instance is mock_cli

    def test_current_subagent(self):
        qp = _QueryParams({"subagent": "baisheng"})
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            assert chatbot.current_subagent == "baisheng"

    def test_current_subagent_none(self):
        mock_st = _make_mock_st(query_params=_QueryParams())
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            assert chatbot.current_subagent is None

    def test_should_hide_sidebar_true(self):
        state = _fresh_session_state()
        state["embed_mode"] = True
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            assert chatbot.should_hide_sidebar is True

    def test_should_hide_sidebar_default_false(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            assert chatbot.should_hide_sidebar is False


# ═══════════════════════════════════════════════════════════════════════════
# 5. StreamlitChatbot – __init__
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestStreamlitChatbotInit:
    """Tests for StreamlitChatbot.__init__."""

    def test_init_creates_components(self):
        mock_st = _make_mock_st()
        with (
            patch("datus.cli.web.chatbot.st", mock_st),
            patch("datus.cli.web.chatbot.SessionManager") as mock_sm,
            patch("datus.cli.web.chatbot.ChatExecutor") as mock_ce,
            patch("datus.cli.web.chatbot.ConfigManager") as mock_cm,
            patch("datus.cli.web.chatbot.UIComponents") as mock_ui,
        ):
            from datus.cli.web.chatbot import StreamlitChatbot

            StreamlitChatbot()
            mock_sm.assert_called_once()
            mock_ce.assert_called_once()
            mock_cm.assert_called_once()
            mock_ui.assert_called_once_with("localhost", "localhost")

    def test_init_sets_session_state_defaults(self):
        state = _SessionState()
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            _create_chatbot()
            assert state["messages"] == []
            assert state["current_actions"] == []
            assert state["chat_session_initialized"] is False
            assert state["cli_instance"] is None
            assert state["current_chat_id"] is None
            assert state["subagent_name"] is None
            assert state["view_session_id"] is None
            assert state["session_readonly_mode"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 6. clear_chat
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestClearChat:
    """Tests for StreamlitChatbot.clear_chat."""

    def test_clear_chat_resets_state(self):
        state = _fresh_session_state()
        state["messages"] = [{"role": "user", "content": "hi"}]
        state["current_actions"] = ["action1"]
        state["current_chat_id"] = "abc-123"
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.clear_chat()
            assert state.messages == []
            assert state.current_actions == []
            assert state.current_chat_id is None

    def test_clear_chat_calls_cli_cmd_clear(self):
        state = _fresh_session_state()
        mock_cli = MagicMock()
        state["cli_instance"] = mock_cli
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.clear_chat()
            mock_cli.chat_commands.cmd_clear_chat.assert_called_once_with("")


# ═══════════════════════════════════════════════════════════════════════════
# 7. setup_config
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestSetupConfig:
    """Tests for StreamlitChatbot.setup_config."""

    def test_setup_config_success(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            mock_cli = MagicMock()
            chatbot.config_manager.setup_config.return_value = mock_cli

            result = chatbot.setup_config("conf/agent.yml")
            assert result is True
            assert mock_st.session_state.chat_session_initialized is True

    def test_setup_config_already_initialized(self):
        state = _fresh_session_state()
        state["cli_instance"] = MagicMock()
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            result = chatbot.setup_config()
            assert result is True
            chatbot.config_manager.setup_config.assert_not_called()

    def test_setup_config_failure(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.config_manager.setup_config.side_effect = RuntimeError("bad config")

            result = chatbot.setup_config()
            assert result is False
            mock_st.error.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Delegation methods
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestDelegationMethods:
    """Tests for simple delegation methods."""

    def test_get_session_messages_delegates(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.get_session_messages.return_value = [{"role": "user", "content": "hi"}]
            result = chatbot.get_session_messages("sid-123")
            chatbot.session_manager.get_session_messages.assert_called_once_with("sid-123")
            assert result == [{"role": "user", "content": "hi"}]

    def test_get_current_session_id_with_node(self):
        state = _fresh_session_state()
        mock_cli = MagicMock()
        mock_node = MagicMock()
        mock_node.session_id = "session-abc"
        mock_cli.chat_commands.current_node = mock_node
        mock_cli.chat_commands.chat_node = None
        state["cli_instance"] = mock_cli
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            result = chatbot.get_current_session_id()
            assert result == "session-abc"

    def test_get_current_session_id_no_cli(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            result = chatbot.get_current_session_id()
            assert result is None

    def test_extract_sql_and_response_delegates(self):
        state = _fresh_session_state()
        state["cli_instance"] = MagicMock()
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.chat_executor.extract_sql_and_response.return_value = ("SELECT 1", "Result")
            actions = [MagicMock()]
            sql, resp = chatbot.extract_sql_and_response(actions)
            assert sql == "SELECT 1"
            assert resp == "Result"

    def test_format_action_for_stream_delegates(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.chat_executor.format_action_for_stream.return_value = "formatted"
            action = MagicMock()
            result = chatbot._format_action_for_stream(action)
            assert result == "formatted"


# ═══════════════════════════════════════════════════════════════════════════
# 9. _store_session_id
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestStoreSessionId:
    """Tests for StreamlitChatbot._store_session_id."""

    def test_store_session_id_success(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value="sid-999"):
                chatbot._store_session_id()
            assert mock_st.session_state.current_session_id == "sid-999"

    def test_store_session_id_none(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value=None):
                chatbot._store_session_id()
            assert "current_session_id" not in mock_st.session_state


# ═══════════════════════════════════════════════════════════════════════════
# 10. save_success_story
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestSaveSuccessStory:
    """Tests for StreamlitChatbot.save_success_story."""

    def test_save_success_story_no_session(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value=None):
                chatbot.save_success_story("SELECT 1", "test question")
            mock_st.warning.assert_called_once()

    def test_save_success_story_creates_csv(self, tmp_path):
        state = _fresh_session_state()
        state["subagent_name"] = "test_agent"
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value="sid-001"):
                mock_pm = MagicMock()
                mock_pm.benchmark_dir = tmp_path
                with patch("datus.utils.path_manager.get_path_manager", return_value=mock_pm):
                    chatbot.save_success_story("SELECT 1", "What is the count?")

            csv_path = tmp_path / "test_agent" / "success_story.csv"
            assert csv_path.exists()
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["sql"] == "SELECT 1"
            assert rows[0]["user_message"] == "What is the count?"
            assert rows[0]["session_id"] == "sid-001"

    def test_save_success_story_appends(self, tmp_path):
        state = _fresh_session_state()
        state["subagent_name"] = "agent2"
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value="sid-002"):
                mock_pm = MagicMock()
                mock_pm.benchmark_dir = tmp_path
                with patch("datus.utils.path_manager.get_path_manager", return_value=mock_pm):
                    chatbot.save_success_story("SELECT 1", "q1")
                    chatbot.save_success_story("SELECT 2", "q2")

            csv_path = tmp_path / "agent2" / "success_story.csv"
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2

    def test_save_success_story_unsafe_name(self, tmp_path):
        state = _fresh_session_state()
        state["subagent_name"] = ".."
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value="sid-003"):
                mock_pm = MagicMock()
                mock_pm.benchmark_dir = tmp_path
                with patch("datus.utils.path_manager.get_path_manager", return_value=mock_pm):
                    chatbot.save_success_story("SELECT 1", "evil")

            mock_st.error.assert_called_once()

    def test_save_success_story_sanitizes_fields(self, tmp_path):
        state = _fresh_session_state()
        state["subagent_name"] = "safe_agent"
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value="sid-004"):
                mock_pm = MagicMock()
                mock_pm.benchmark_dir = tmp_path
                with patch("datus.utils.path_manager.get_path_manager", return_value=mock_pm):
                    chatbot.save_success_story("=CMD()", "=HYPERLINK()")

            csv_path = tmp_path / "safe_agent" / "success_story.csv"
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows[0]["sql"] == "'=CMD()"
            assert rows[0]["user_message"] == "'=HYPERLINK()"


# ═══════════════════════════════════════════════════════════════════════════
# 11. load_session_from_url
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestLoadSessionFromUrl:
    """Tests for StreamlitChatbot.load_session_from_url."""

    def test_load_no_session_param(self):
        mock_st = _make_mock_st(query_params=_QueryParams())
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.load_session_from_url()
            chatbot.session_manager.session_exists.assert_not_called()

    def test_load_already_loaded(self):
        state = _fresh_session_state()
        state["view_session_id"] = "sid-100"
        state["_loaded_session_mode"] = "readonly"
        qp = _QueryParams({"session": "sid-100", "mode": "readonly"})
        mock_st = _make_mock_st(session_state=state, query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.load_session_from_url()
            chatbot.session_manager.session_exists.assert_not_called()

    def test_load_readonly_mode(self):
        qp = _QueryParams({"session": "sid-200"})
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = True
            chatbot.session_manager.get_session_messages.return_value = [{"role": "user", "content": "hi"}]

            chatbot.load_session_from_url()

            assert mock_st.session_state.session_readonly_mode is True
            assert mock_st.session_state.view_session_id == "sid-200"
            assert mock_st.session_state.messages == [{"role": "user", "content": "hi"}]

    def test_load_resume_mode(self):
        qp = _QueryParams({"session": "sid-300", "mode": "resume"})
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = True

            with patch.object(chatbot, "resume_session", return_value=True) as mock_resume:
                chatbot.load_session_from_url()
                mock_resume.assert_called_once_with("sid-300")
                assert mock_st.session_state._loaded_session_mode == "resume"

    def test_load_nonexistent_session(self):
        qp = _QueryParams({"session": "sid-404"})
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = False

            chatbot.load_session_from_url()
            mock_st.error.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 12. resume_session
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestResumeSession:
    """Tests for StreamlitChatbot.resume_session."""

    def test_resume_success(self):
        state = _fresh_session_state()
        mock_cli = MagicMock()
        state["cli_instance"] = mock_cli
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = True
            chatbot.session_manager.get_session_messages.return_value = [{"role": "user", "content": "hi"}]

            with patch("datus.cli.chat_commands.ChatCommands._extract_node_type_from_session_id", return_value="chat"):
                mock_cli.chat_commands._create_new_node.return_value = MagicMock()
                result = chatbot.resume_session("sid-resume")
                assert result is True
                assert state.view_session_id == "sid-resume"
                assert state.session_readonly_mode is False

    def test_resume_nonexistent(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = False

            result = chatbot.resume_session("sid-gone")
            assert result is False
            mock_st.error.assert_called_once()

    def test_resume_cli_not_initialized(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = True
            # cli_instance is None by default from _fresh_session_state

            result = chatbot.resume_session("sid-no-cli")
            assert result is False
            mock_st.error.assert_called()

    def test_resume_exception(self):
        state = _fresh_session_state()
        mock_cli = MagicMock()
        state["cli_instance"] = mock_cli
        mock_st = _make_mock_st(session_state=state)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.session_exists.return_value = True
            chatbot.session_manager.get_session_messages.side_effect = RuntimeError("db error")

            with patch("datus.cli.chat_commands.ChatCommands._extract_node_type_from_session_id", return_value="chat"):
                result = chatbot.resume_session("sid-err")
                assert result is False
                mock_st.error.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# 13. execute_chat_stream
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestExecuteChatStream:
    """Tests for StreamlitChatbot.execute_chat_stream."""

    def test_yields_from_chat_executor(self):
        qp = _QueryParams()
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            action1, action2 = MagicMock(), MagicMock()
            chatbot.chat_executor.execute_chat_stream.return_value = iter([action1, action2])
            with patch.object(chatbot, "get_current_session_id", return_value=None):
                results = list(chatbot.execute_chat_stream("test query"))
            assert results == [action1, action2]

    def test_stores_session_id_after_stream(self):
        qp = _QueryParams()
        mock_st = _make_mock_st(query_params=qp)
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.chat_executor.execute_chat_stream.return_value = iter([])
            with patch.object(chatbot, "get_current_session_id", return_value="sid-after"):
                list(chatbot.execute_chat_stream("test"))
            assert mock_st.session_state.current_session_id == "sid-after"


# ═══════════════════════════════════════════════════════════════════════════
# 14. _handle_rewind
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestHandleRewind:
    """Tests for StreamlitChatbot._handle_rewind."""

    def test_rewind_success(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            chatbot.session_manager.rewind_session.return_value = "sid-rewound"

            with (
                patch.object(chatbot, "get_current_session_id", return_value="sid-orig"),
                patch.object(chatbot, "resume_session", return_value=True),
            ):
                chatbot._handle_rewind(2, True)
                chatbot.session_manager.rewind_session.assert_called_once_with(
                    "sid-orig", 2, include_assistant_response=True
                )
                chatbot.ui.safe_update_query_params.assert_called_once()
                mock_st.rerun.assert_called_once()

    def test_rewind_no_session(self):
        mock_st = _make_mock_st()
        with patch("datus.cli.web.chatbot.st", mock_st):
            chatbot = _create_chatbot()
            with patch.object(chatbot, "get_current_session_id", return_value=None):
                chatbot._handle_rewind(1, False)
            mock_st.error.assert_called_once_with("No active session to rewind.")
