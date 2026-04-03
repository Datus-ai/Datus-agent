# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/print_mode.py
"""

import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.schemas.message_content import MessageContent, MessagePayload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = dict(
        print_mode="hello",
        resume=None,
        subagent=None,
        proxy_tools=None,
        namespace="test_ns",
        db_type="sqlite",
        db_path=None,
        config=None,
        debug=False,
        no_color=False,
        database="",
        history_file=None,
        save_llm_trace=False,
        web=False,
        port=8501,
        host="localhost",
        catalog=None,
        schema=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_runner(**overrides):
    """Create a PrintModeRunner with mocked dependencies."""
    with (
        patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
        patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
    ):
        mock_cfg.return_value = MagicMock(namespaces=[])
        mock_completer.return_value.parse_at_context.return_value = ([], [], [])
        from datus.cli.print_mode import PrintModeRunner

        return PrintModeRunner(_make_args(**overrides))


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner._write_payload
# ---------------------------------------------------------------------------


class TestWritePayload:
    def test_writes_json_line(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            payload = MessagePayload(
                message_id="m1",
                role="assistant",
                content=[MessageContent(type="markdown", payload={"content": "hi"})],
            )
            runner._write_payload(payload)

        line = buf.getvalue().strip()
        data = json.loads(line)
        assert data["message_id"] == "m1"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "markdown"


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner._read_interaction_input
# ---------------------------------------------------------------------------


class TestReadInteractionInput:
    def test_parses_valid_payload(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        payload = MessagePayload(
            message_id="m1",
            role="user",
            content=[MessageContent(type="user-interaction", payload={"content": "y"})],
        )
        with patch("sys.stdin", io.StringIO(payload.model_dump_json() + "\n")):
            result = runner._read_interaction_input()
        assert result == "y"

    def test_empty_input(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        with patch("sys.stdin", io.StringIO("\n")):
            result = runner._read_interaction_input()
        assert result == ""

    def test_invalid_json_returns_raw(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        with patch("sys.stdin", io.StringIO("raw text\n")):
            result = runner._read_interaction_input()
        assert result == "raw text"


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner.run (mocked end-to-end)
# ---------------------------------------------------------------------------


class TestPrintModeRun:
    @pytest.mark.asyncio
    async def test_stream_chat_writes_payloads(self):
        """Test that _stream_chat writes payloads to stdout."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages="thinking",
            action_type="llm_generation",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
        )

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            yield mock_action

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        assert output  # at least one JSON line
        data = json.loads(output)
        assert data["message_id"] == "a1"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "thinking"
        assert data["depth"] == 0
        assert data["parent_action_id"] is None

    @pytest.mark.asyncio
    async def test_stream_chat_subagent_hierarchy(self):
        """Test that _stream_chat propagates depth and parent_action_id from subagent actions."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_action = ActionHistory(
            action_id="sub_a1",
            role=ActionRole.ASSISTANT,
            messages="subagent thinking",
            action_type="llm_generation",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
            depth=1,
            parent_action_id="call_parent_123",
        )

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            yield mock_action

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message_id"] == "sub_a1"
        assert data["depth"] == 1
        assert data["parent_action_id"] == "call_parent_123"


# ---------------------------------------------------------------------------
# Tests: --resume sets session_id
# ---------------------------------------------------------------------------


class TestResumeSessionId:
    def test_resume_sets_session_id(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="session_abc"))

        assert runner.session_id == "session_abc"

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield  # make it an async generator

        mock_node.execute_stream_with_interactions = fake_stream

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = True

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node),
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
        ):
            runner.run()

        # session_id should be set on the node
        assert mock_node.session_id == "session_abc"

    def test_resume_nonexistent_session_continues(self):
        """Verify that a non-existent session_id logs a warning and continues (creates new session)."""
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="no_such_session"))

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = False

        with (
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node),
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
        ):
            runner.run()  # should not raise

        # session_id is still set on the node even for new sessions
        assert mock_node.session_id == "no_such_session"

    def test_resume_subagent_name_from_args(self):
        """Verify that subagent_name comes from args, not derived from session_id."""
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="session_uuid123", subagent="gen_sql"))

        assert runner.subagent_name == "gen_sql"

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = True

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node) as mock_create_node,
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
        ):
            runner.run()

        mock_create_node.assert_called_once_with("gen_sql", runner.agent_config, node_id_suffix="_print")
        assert mock_node.session_id == "session_uuid123"


# ---------------------------------------------------------------------------
# Tests: run delegates to factory functions
# ---------------------------------------------------------------------------


class TestRunUsesFactory:
    def test_run_calls_factory(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream
        mock_input = MagicMock()

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node) as mock_create_node,
            patch("datus.cli.print_mode.create_node_input", return_value=mock_input) as mock_create_input,
        ):
            runner.run()

        mock_create_node.assert_called_once_with(None, runner.agent_config, node_id_suffix="_print")
        mock_create_input.assert_called_once()
        assert mock_node.input == mock_input


# ---------------------------------------------------------------------------
# Tests: run() with proxy_tool_patterns
# ---------------------------------------------------------------------------


class TestRunProxyTools:
    def test_run_applies_proxy_tools_when_patterns_set(self):
        """Verify that apply_proxy_tools is called when proxy_tool_patterns is set."""
        runner = _make_runner(proxy_tools="filesystem_tools.*,db_tools.*")

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node),
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply,
        ):
            runner.run()

        mock_apply.assert_called_once_with(mock_node, ["filesystem_tools.*", "db_tools.*"])

    def test_run_does_not_apply_proxy_tools_when_none(self):
        """Verify that apply_proxy_tools is NOT called when proxy_tool_patterns is None."""
        runner = _make_runner()

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node),
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply,
        ):
            runner.run()

        mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _stream_chat with different action types
# ---------------------------------------------------------------------------


class TestStreamChatActions:
    @pytest.mark.asyncio
    async def test_stream_chat_interaction_without_proxy(self):
        """Verify INTERACTION actions write payload and read user input when not in proxy mode."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        interaction_action = ActionHistory(
            action_id="int_1",
            role=ActionRole.INTERACTION,
            messages="confirm?",
            action_type="user_confirm",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
        )

        mock_node = MagicMock()
        mock_node.session_id = None
        mock_node.interaction_broker = AsyncMock()

        async def fake_stream(actions):
            yield interaction_action

        mock_node.execute_stream_with_interactions = fake_stream

        runner = _make_runner()

        buf = io.StringIO()
        with (
            patch("sys.stdout", buf),
            patch("asyncio.to_thread", return_value="user_response"),
        ):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message_id"] == "int_1"
        mock_node.interaction_broker.submit.assert_awaited_once_with("int_1", "user_response")

    @pytest.mark.asyncio
    async def test_stream_chat_interaction_with_proxy_skips_stdin(self):
        """Verify INTERACTION actions in proxy mode do NOT read from stdin."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        interaction_action = ActionHistory(
            action_id="int_2",
            role=ActionRole.INTERACTION,
            messages="confirm?",
            action_type="user_confirm",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
        )

        mock_node = MagicMock()
        mock_node.session_id = None
        mock_node.tool_channel = MagicMock()
        mock_node.interaction_broker = AsyncMock()

        async def fake_stream(actions):
            yield interaction_action

        mock_node.execute_stream_with_interactions = fake_stream

        runner = _make_runner(proxy_tools="*")

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        # Should write payload but NOT submit interaction
        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message_id"] == "int_2"
        mock_node.interaction_broker.submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_chat_response_action(self):
        """Verify actions with action_type ending in '_response' write response content."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        response_action = ActionHistory(
            action_id="resp_1",
            role=ActionRole.ASSISTANT,
            messages="SQL generated",
            action_type="gen_sql_response",
            input=None,
            output={"sql": "SELECT 1"},
            status=ActionStatus.SUCCESS,
        )

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            yield response_action

        mock_node.execute_stream_with_interactions = fake_stream

        runner = _make_runner()

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message_id"] == "resp_1"
        assert data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_stream_chat_finally_cancels_channel_in_proxy_mode(self):
        """Verify that tool_channel.cancel_all is called in finally block when proxy mode is active."""
        mock_node = MagicMock()
        mock_node.session_id = None
        mock_node.tool_channel = MagicMock()

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        runner = _make_runner(proxy_tools="*")

        # Patch _stdin_dispatch_loop to return immediately
        async def noop_loop(node):
            return

        with patch.object(runner, "_stdin_dispatch_loop", noop_loop):
            await runner._stream_chat(mock_node)

        mock_node.tool_channel.cancel_all.assert_called_once_with("stream ended")


# ---------------------------------------------------------------------------
# Tests: _stdin_dispatch_loop
# ---------------------------------------------------------------------------


class TestStdinDispatchLoop:
    @pytest.mark.asyncio
    async def test_dispatch_call_tool_result(self):
        """Verify that call-tool-result messages are dispatched to tool_channel."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()
        mock_node.tool_channel.publish = AsyncMock()
        mock_node.interaction_broker = AsyncMock()

        payload = MessagePayload(
            message_id="m1",
            role="user",
            content=[
                MessageContent(
                    type="call-tool-result",
                    payload={"callToolId": "call_xyz", "result": {"success": 1}},
                )
            ],
        )

        lines = [payload.model_dump_json(), None]
        line_iter = iter(lines)

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return next(line_iter)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        mock_node.tool_channel.publish.assert_awaited_once_with("call_xyz", {"success": 1})

    @pytest.mark.asyncio
    async def test_dispatch_user_interaction(self):
        """Verify that user-interaction messages are dispatched to interaction_broker."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()
        mock_node.tool_channel.publish = AsyncMock()
        mock_node.interaction_broker = AsyncMock()

        payload = MessagePayload(
            message_id="int_abc",
            role="user",
            content=[
                MessageContent(
                    type="user-interaction",
                    payload={"content": "yes"},
                )
            ],
        )

        lines = [payload.model_dump_json(), None]
        line_iter = iter(lines)

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return next(line_iter)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        mock_node.interaction_broker.submit.assert_awaited_once_with("int_abc", "yes")

    @pytest.mark.asyncio
    async def test_dispatch_invalid_json_logs_warning(self):
        """Verify that invalid JSON on stdin is logged and skipped."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()
        mock_node.tool_channel.publish = AsyncMock()

        lines = ["not valid json", None]
        line_iter = iter(lines)

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return next(line_iter)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        # Should not crash, and tool_channel.publish should not be called
        mock_node.tool_channel.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_call_tool_result_skips_empty_call_id(self):
        """Verify that call-tool-result with empty callToolId is skipped."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()
        mock_node.tool_channel.publish = AsyncMock()

        payload = MessagePayload(
            message_id="m1",
            role="user",
            content=[
                MessageContent(
                    type="call-tool-result",
                    payload={"callToolId": "", "result": {"success": 1}},
                )
            ],
        )

        lines = [payload.model_dump_json(), None]
        line_iter = iter(lines)

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return next(line_iter)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        mock_node.tool_channel.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_eof_cancels_all(self):
        """Verify that EOF (None) on stdin cancels all tool channel futures."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()  # cancel_all is sync

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return None

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        mock_node.tool_channel.cancel_all.assert_called_once_with("stdin EOF")

    @pytest.mark.asyncio
    async def test_dispatch_skips_empty_lines(self):
        """Verify that blank lines on stdin are skipped."""
        mock_node = MagicMock()
        mock_node.tool_channel = MagicMock()
        mock_node.tool_channel.publish = AsyncMock()

        lines = ["", "  ", None]
        line_iter = iter(lines)

        runner = _make_runner(proxy_tools="*")
        runner._stdin_stop_event = threading.Event()

        async def mock_read_line(fn, *args):
            return next(line_iter)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_read_line
            await runner._stdin_dispatch_loop(mock_node)

        mock_node.tool_channel.publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: _read_stdin_line
# ---------------------------------------------------------------------------


class TestReadStdinLine:
    def test_read_stdin_line_returns_line_unix(self):
        """Verify _read_stdin_line reads a complete line on Unix (via select)."""
        from datus.cli.print_mode import PrintModeRunner

        stop_event = threading.Event()

        # Simulate select returning ready and os.read returning chars
        calls = [b"h", b"i", b"\n"]
        call_iter = iter(calls)

        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin") as mock_stdin,
            patch("select.select", return_value=([mock_stdin.fileno()], [], [])),
            patch("os.read", side_effect=lambda fd, n: next(call_iter)),
        ):
            result = PrintModeRunner._read_stdin_line(stop_event)

        assert result == "hi"

    def test_read_stdin_line_returns_none_on_eof(self):
        """Verify _read_stdin_line returns None on EOF."""
        from datus.cli.print_mode import PrintModeRunner

        stop_event = threading.Event()

        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin") as mock_stdin,
            patch("select.select", return_value=([mock_stdin.fileno()], [], [])),
            patch("os.read", return_value=b""),
        ):
            result = PrintModeRunner._read_stdin_line(stop_event)

        assert result is None

    def test_read_stdin_line_returns_none_when_stopped(self):
        """Verify _read_stdin_line returns None when stop_event is set."""
        from datus.cli.print_mode import PrintModeRunner

        stop_event = threading.Event()
        stop_event.set()

        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.fileno.return_value = 0
            result = PrintModeRunner._read_stdin_line(stop_event)

        assert result is None

    def test_read_stdin_line_windows_path(self):
        """Verify _read_stdin_line works on Windows path."""
        from datus.cli.print_mode import PrintModeRunner

        stop_event = threading.Event()

        # Simulate Windows: readable returns True, then read chars
        chars = ["h", "e", "l", "l", "o", "\n"]
        char_iter = iter(chars)

        with (
            patch("sys.platform", "win32"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.fileno.return_value = 0
            mock_stdin.readable.return_value = True
            mock_stdin.read.side_effect = lambda n: next(char_iter)
            result = PrintModeRunner._read_stdin_line(stop_event)

        assert result == "hello"

    def test_read_stdin_line_windows_eof(self):
        """Verify _read_stdin_line returns None on EOF in Windows mode."""
        from datus.cli.print_mode import PrintModeRunner

        stop_event = threading.Event()

        with (
            patch("sys.platform", "win32"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.fileno.return_value = 0
            mock_stdin.readable.return_value = True
            mock_stdin.read.return_value = ""
            result = PrintModeRunner._read_stdin_line(stop_event)

        assert result is None
