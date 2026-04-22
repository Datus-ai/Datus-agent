# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.gateway.bridge.ChannelBridge."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEndData,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
)
from datus.gateway.bridge import _MAX_BOT_MSG_MAP, ChannelBridge
from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.models import (
    DONE_EMOJI,
    ERROR_EMOJI,
    PROCESSING_EMOJI,
    ChannelConfig,
    InboundMessage,
    OutboundMessage,
    ReactionEvent,
    Verbose,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAdapter(ChannelAdapter):
    """Minimal adapter that records sent messages and reactions (no streaming)."""

    def __init__(self, bridge=None):
        super().__init__("test_channel", {}, bridge=bridge or MagicMock())
        self.sent: list[OutboundMessage] = []
        self.reactions_added: list[tuple[str, str, str]] = []  # (conversation_id, message_id, emoji)
        self.reactions_removed: list[tuple[str, str, str]] = []
        self._send_counter = 0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_message(self, message: OutboundMessage) -> str:
        self.sent.append(message)
        self._send_counter += 1
        return f"bot_msg_{self._send_counter}"

    async def add_reaction(self, conversation_id, message_id, emoji, thread_id=None):
        self.reactions_added.append((conversation_id, message_id, emoji))

    async def remove_reaction(self, conversation_id, message_id, emoji, thread_id=None):
        self.reactions_removed.append((conversation_id, message_id, emoji))


class _StreamingStubAdapter(_StubAdapter):
    """Stub adapter that supports streaming (like Feishu)."""

    @property
    def supports_streaming(self) -> bool:
        return True


def _make_inbound(
    text="hello", thread_id=None, chat_type=None, mentions_bot=False, message_id="msg1"
) -> InboundMessage:
    return InboundMessage(
        channel_id="ch1",
        sender_id="user1",
        conversation_id="conv1",
        message_id=message_id,
        text=text,
        mentions_bot=mentions_bot,
        chat_type=chat_type,
        thread_id=thread_id,
    )


def _make_sse_message(content_text: str, content_type: str = "markdown", message_id: str = "m1") -> SSEEvent:
    return SSEEvent(
        id=1,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id=message_id,
                role="assistant",
                content=[IMessageContent(type=content_type, payload={"content": content_text})],
            ),
        ),
        timestamp=datetime.now().isoformat() + "Z",
    )


def _make_sse_sql(sql_text: str) -> SSEEvent:
    return SSEEvent(
        id=2,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m2",
                role="assistant",
                content=[IMessageContent(type="code", payload={"content": sql_text, "codeType": "sql"})],
            ),
        ),
        timestamp=datetime.now().isoformat() + "Z",
    )


def _make_sse_tool_call(tool_name: str) -> SSEEvent:
    return SSEEvent(
        id=3,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m3",
                role="assistant",
                content=[
                    IMessageContent(
                        type="call-tool",
                        payload={"callToolId": "t1", "toolName": tool_name, "toolParams": {}},
                    )
                ],
            ),
        ),
        timestamp=datetime.now().isoformat() + "Z",
    )


def _make_sse_tool_result(tool_name: str, short_desc: str = "", duration: float = 1.2) -> SSEEvent:
    return SSEEvent(
        id=4,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m4",
                role="assistant",
                content=[
                    IMessageContent(
                        type="call-tool-result",
                        payload={
                            "callToolId": "t1",
                            "toolName": tool_name,
                            "duration": duration,
                            "shortDesc": short_desc,
                            "result": {},
                        },
                    )
                ],
            ),
        ),
        timestamp=datetime.now().isoformat() + "Z",
    )


def _make_sse_thinking(text: str) -> SSEEvent:
    return SSEEvent(
        id=5,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m5",
                role="assistant",
                content=[IMessageContent(type="thinking", payload={"content": text})],
            ),
        ),
        timestamp=datetime.now().isoformat() + "Z",
    )


def _make_sse_end() -> SSEEvent:
    return SSEEvent(
        id=99,
        event="end",
        data=SSEEndData(session_id="s", llm_session_id="ls", total_events=3, action_count=1, duration=0.5),
        timestamp=datetime.now().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChannelBridge:
    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StreamingStubAdapter()
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    def test_build_session_id_without_thread(self, setup):
        bridge, _, _ = setup
        msg = _make_inbound()
        sid = bridge.build_session_id(msg)
        assert sid == "gateway_ch1--conv1"

    def test_build_session_id_with_thread(self, setup):
        bridge, _, _ = setup
        msg = _make_inbound(thread_id="t42")
        sid = bridge.build_session_id(msg)
        assert sid == "gateway_ch1--conv1--t42"

    @pytest.mark.asyncio
    async def test_streams_each_event(self, setup):
        """Each SSE message event should produce a separate outbound message.

        With call-tool caching, call-tool + call-tool-result merge into one message.
        So: thinking(1) + merged-tool(1) + markdown(1) + sql(1) = 4 messages.
        """
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        events = [
            _make_sse_thinking("analyzing the question"),
            _make_sse_tool_call("schema_linking"),
            _make_sse_tool_result("schema_linking", "Found 3 tables"),
            _make_sse_message("Here is the answer"),
            _make_sse_sql("SELECT COUNT(*) FROM orders"),
            _make_sse_end(),
        ]

        async def _fake_consume(task, start_from=None):
            for e in events:
                yield e

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("show sales"), adapter)

        # 4 messages: thinking + merged tool + markdown + sql
        assert len(adapter.sent) == 4

        # Check thinking
        assert "analyzing the question" in adapter.sent[0].text

        # Check merged tool call + result
        assert "schema_linking" in adapter.sent[1].text

        # Check markdown
        assert "Here is the answer" in adapter.sent[2].text

        # Check SQL
        assert adapter.sent[3].sql == "SELECT COUNT(*) FROM orders"

    @pytest.mark.asyncio
    async def test_session_busy(self, setup):
        bridge, adapter, task_manager = setup

        task_manager.start_chat = AsyncMock(side_effect=ValueError("A task is already running"))

        await bridge.handle_message(_make_inbound("another question"), adapter)

        assert len(adapter.sent) == 1
        assert "still being processed" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_no_response_fallback(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        # Only an end event, no content
        async def _fake_consume(task, start_from=None):
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("hello"), adapter)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].text == "(No response generated)"

    @pytest.mark.asyncio
    async def test_channel_config_applied(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("done")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", datasource="prod", subagent_id="agent_x")
        await bridge.handle_message(_make_inbound("test"), adapter, channel_config=cfg)

        call_args = task_manager.start_chat.call_args
        request = call_args.args[1]
        assert request.database == "prod"
        assert call_args.kwargs.get("sub_agent_id") == "agent_x"

    @pytest.mark.asyncio
    async def test_thread_id_propagated(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q", thread_id="thread_99"), adapter)

        assert adapter.sent[0].thread_id == "thread_99"

    @pytest.mark.asyncio
    async def test_thinking_not_truncated(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        long_text = "x" * 1000

        async def _fake_consume(task, start_from=None):
            yield _make_sse_thinking(long_text)
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 1
        # Full thinking text should be preserved, not truncated
        assert long_text in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_new_command_clears_session(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("first answer")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        # First message — establishes session
        await bridge.handle_message(_make_inbound("hello", message_id="m1"), adapter)
        first_call = task_manager.start_chat.call_args
        first_session = first_call.args[1].session_id

        # Send /new to clear session
        with patch("datus.gateway.bridge.SessionManager") as mock_sm_cls:
            mock_sm = MagicMock()
            mock_sm.session_exists.return_value = True
            mock_sm_cls.return_value = mock_sm
            await bridge.handle_message(_make_inbound("/new", message_id="m2"), adapter)
            mock_sm.clear_session.assert_called_once_with(first_session)
        assert "Session cleared" in adapter.sent[-1].text

        # Next message — should use the SAME session id (no suffix rotation)
        await bridge.handle_message(_make_inbound("hello again", message_id="m3"), adapter)
        second_call = task_manager.start_chat.call_args
        second_session = second_call.args[1].session_id
        assert first_session == second_session

    @pytest.mark.asyncio
    async def test_new_command_case_insensitive(self, setup):
        bridge, adapter, _ = setup

        for idx, cmd in enumerate(["/new", "/NEW", "new", "reset", "/Reset"], start=1):
            await bridge.handle_message(_make_inbound(cmd, message_id=f"cmd_{idx}"), adapter)

        assert len(adapter.sent) == 5
        assert all("Session cleared" in m.text for m in adapter.sent)

    @pytest.mark.asyncio
    async def test_error_event(self, setup):
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        error_event = SSEEvent(
            id=1,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m1",
                    role="assistant",
                    content=[IMessageContent(type="error", payload={"content": "Something went wrong"})],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )

        async def _fake_consume(task, start_from=None):
            yield error_event
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 1
        assert "Something went wrong" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_group_message_without_mention_ignored(self, setup):
        """Group chat message without @bot should be silently ignored."""
        bridge, adapter, task_manager = setup
        task_manager.start_chat = AsyncMock()

        msg = _make_inbound("hello", chat_type="group", mentions_bot=False)
        await bridge.handle_message(msg, adapter)

        task_manager.start_chat.assert_not_called()
        assert len(adapter.sent) == 0

    @pytest.mark.asyncio
    async def test_group_message_with_mention_processed(self, setup):
        """Group chat message with @bot should be processed normally."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("group reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("show sales", chat_type="group", mentions_bot=True)
        await bridge.handle_message(msg, adapter)

        assert len(adapter.sent) == 1
        assert "group reply" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_group_mention_creates_thread(self, setup):
        """Group @bot message without thread_id should auto-set thread_id = message_id."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("threaded reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("question", chat_type="group", mentions_bot=True, message_id="msg_abc")
        await bridge.handle_message(msg, adapter)

        # Reply should be in the auto-created thread
        assert adapter.sent[0].thread_id == "msg_abc"
        # Session ID should include the thread_id
        session_id = task_manager.start_chat.call_args.args[1].session_id
        assert "msg_abc" in session_id

    @pytest.mark.asyncio
    async def test_group_mention_existing_thread_preserved(self, setup):
        """Group @bot message with existing thread_id should keep it."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("thread reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("follow up", chat_type="group", mentions_bot=True, thread_id="existing_thread")
        await bridge.handle_message(msg, adapter)

        assert adapter.sent[0].thread_id == "existing_thread"

    @pytest.mark.asyncio
    async def test_group_thread_reply_without_mention_processed(self, setup):
        """Thread replies in bot-active threads should be processed without @bot."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("follow up reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("more details please", chat_type="group", mentions_bot=False, thread_id="existing_thread")
        with patch("datus.gateway.bridge.SessionManager") as mock_sm_cls:
            mock_sm = MagicMock()
            mock_sm.session_exists.return_value = True
            mock_sm_cls.return_value = mock_sm
            await bridge.handle_message(msg, adapter)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].thread_id == "existing_thread"

    @pytest.mark.asyncio
    async def test_group_thread_reply_to_unknown_thread_ignored(self, setup):
        """Thread replies in non-bot threads should be ignored."""
        bridge, adapter, task_manager = setup
        task_manager.start_chat = AsyncMock()

        msg = _make_inbound("more details please", chat_type="group", mentions_bot=False, thread_id="other_thread")
        with patch("datus.gateway.bridge.SessionManager") as mock_sm_cls:
            mock_sm = MagicMock()
            mock_sm.session_exists.return_value = False
            mock_sm_cls.return_value = mock_sm
            await bridge.handle_message(msg, adapter)

        task_manager.start_chat.assert_not_called()
        assert len(adapter.sent) == 0

    @pytest.mark.asyncio
    async def test_dm_always_processed(self, setup):
        """DM (p2p) messages should always be processed, even without @bot."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("dm reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("hello", chat_type="p2p", mentions_bot=False)
        await bridge.handle_message(msg, adapter)

        assert len(adapter.sent) == 1
        assert "dm reply" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_none_chat_type_passes_through(self, setup):
        """Messages with chat_type=None should be processed (backward compat)."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("compat reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        msg = _make_inbound("hello", chat_type=None, mentions_bot=False)
        await bridge.handle_message(msg, adapter)

        assert len(adapter.sent) == 1


class TestEventToOutbound:
    @pytest.fixture
    def bridge(self, tmp_path):
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        return ChannelBridge(agent_config, task_manager)

    def test_markdown_content(self, bridge):
        msg = _make_inbound()
        data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[IMessageContent(type="markdown", payload={"content": "Hello"})],
            ),
        )
        result = bridge._event_to_outbound(data, msg)
        assert result is not None
        assert result.text == "Hello"
        assert result.sql is None

    def test_sql_content(self, bridge):
        msg = _make_inbound()
        data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[IMessageContent(type="code", payload={"content": "SELECT 1", "codeType": "sql"})],
            ),
        )
        result = bridge._event_to_outbound(data, msg)
        assert result is not None
        assert result.sql == "SELECT 1"

    def test_skip_non_message_data(self, bridge):
        msg = _make_inbound()
        result = bridge._event_to_outbound("not_an_sse_data", msg)
        assert result is None

    def test_skip_empty_content(self, bridge):
        msg = _make_inbound()
        data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[IMessageContent(type="markdown", payload={"content": ""})],
            ),
        )
        result = bridge._event_to_outbound(data, msg)
        assert result is None

    def test_delta_preserves_leading_trailing_whitespace(self, bridge):
        """Delta (thinking) chunks must preserve leading/trailing whitespace for correct accumulation."""
        msg = _make_inbound()
        data = SSEMessageData(
            type=SSEDataType.APPEND_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[IMessageContent(type="thinking", payload={"content": " again! How are you?"})],
            ),
        )
        result = bridge._event_to_outbound(data, msg)
        assert result is not None
        assert result.is_delta is True
        # Leading space must be preserved so accumulation works: "Hi" + " again!" = "Hi again!"
        assert result.text == " again! How are you?"

    def test_delta_whitespace_only_not_dropped(self, bridge):
        """A delta chunk that is whitespace-only (e.g. ' ') must not be discarded."""
        msg = _make_inbound()
        data = SSEMessageData(
            type=SSEDataType.APPEND_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[IMessageContent(type="thinking", payload={"content": " "})],
            ),
        )
        result = bridge._event_to_outbound(data, msg)
        assert result is not None
        assert result.is_delta is True
        assert result.text == " "

    def test_tool_call_cached_not_output(self, bridge):
        """call-tool alone should NOT produce output; it gets cached."""
        msg = _make_inbound()
        pending = {}
        data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[
                    IMessageContent(
                        type="call-tool",
                        payload={"callToolId": "t1", "toolName": "execute_sql", "toolParams": {}},
                    )
                ],
            ),
        )
        result = bridge._event_to_outbound(data, msg, pending_tool_calls=pending)
        assert result is None
        assert "t1" in pending

    def test_tool_result_merges_with_cached_call(self, bridge):
        """call-tool-result should merge with cached call-tool payload."""
        msg = _make_inbound()
        # First, cache a call-tool
        pending = {
            "t1": {
                "callToolId": "t1",
                "toolName": "execute_sql",
                "toolParams": {"sql": "SELECT 1"},
            }
        }
        data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id="m1",
                role="assistant",
                content=[
                    IMessageContent(
                        type="call-tool-result",
                        payload={
                            "callToolId": "t1",
                            "toolName": "execute_sql",
                            "duration": 2.5,
                            "shortDesc": "3 rows returned",
                            "result": {},
                        },
                    )
                ],
            ),
        )
        result = bridge._event_to_outbound(data, msg, pending_tool_calls=pending)
        assert result is not None
        assert "execute_sql" in result.text
        assert "2.5s" in result.text
        # Cached entry should be consumed
        assert "t1" not in pending


class TestReactionLifecycle:
    """Tests for the reaction emoji lifecycle in handle_message."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StubAdapter()
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    @pytest.mark.asyncio
    async def test_success_adds_processing_then_done(self, setup):
        """Normal flow: hourglass added, then removed, then checkmark added."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("answer")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        # Processing emoji added then removed
        assert (("conv1", "msg1", PROCESSING_EMOJI)) in adapter.reactions_added
        assert (("conv1", "msg1", PROCESSING_EMOJI)) in adapter.reactions_removed
        # Done emoji added
        assert (("conv1", "msg1", DONE_EMOJI)) in adapter.reactions_added
        # No error emoji
        assert all(r[2] != ERROR_EMOJI for r in adapter.reactions_added)

    @pytest.mark.asyncio
    async def test_error_adds_error_emoji(self, setup):
        """When streaming raises, error emoji should be added."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            raise RuntimeError("boom")
            yield  # make it a generator  # noqa: E501

        task_manager.consume_events = _fake_consume

        with pytest.raises(RuntimeError, match="boom"):
            await bridge.handle_message(_make_inbound("q"), adapter)

        # Processing emoji added then removed
        assert ("conv1", "msg1", PROCESSING_EMOJI) in adapter.reactions_added
        assert ("conv1", "msg1", PROCESSING_EMOJI) in adapter.reactions_removed
        # Error emoji added
        assert ("conv1", "msg1", ERROR_EMOJI) in adapter.reactions_added

    @pytest.mark.asyncio
    async def test_add_reaction_failure_does_not_break_message(self, setup):
        """If add_reaction raises, message processing should still proceed."""
        bridge, adapter, task_manager = setup

        call_count = 0
        original_add = adapter.add_reaction

        async def _failing_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("reaction API down")
            await original_add(*args, **kwargs)

        adapter.add_reaction = _failing_add

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("works")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        # Message was still sent despite reaction failure
        assert len(adapter.sent) == 1
        assert "works" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_bot_message_tracked(self, setup):
        """Bot messages should be tracked in _bot_message_map."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("reply")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(bridge._bot_message_map) == 1
        bot_msg_id = list(bridge._bot_message_map.keys())[0]
        assert bot_msg_id.startswith("bot_msg_")

    @pytest.mark.asyncio
    async def test_handle_reaction_known_message_triggers_feedback(self, setup):
        """A reaction on a tracked bot message should start a feedback subagent task."""
        bridge, adapter, task_manager = setup

        bridge._bot_message_map["bot_123"] = {
            "session_id": "sess_1",
            "channel_id": "ch1",
            "conversation_id": "conv1",
            "sender_id": "user1",
            "text": "Here is your result",
        }

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            if False:
                yield
            return

        task_manager.consume_events = _fake_consume

        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user2",
            conversation_id="conv1",
            target_message_id="bot_123",
            emoji="thumbsup",
        )

        await bridge.handle_reaction(event, adapter)

        task_manager.start_chat.assert_awaited_once()
        kwargs = task_manager.start_chat.await_args.kwargs
        assert kwargs.get("sub_agent_id") == "feedback"
        # Request is the second positional arg
        request = task_manager.start_chat.await_args.args[1]
        assert request.source_session_id == "sess_1"
        assert request.session_id is None
        assert request.subagent_id is None  # subagent_id passes via sub_agent_id kwarg
        assert "thumbsup" in request.message
        assert "Here is your result" in request.message

    @pytest.mark.asyncio
    async def test_handle_reaction_dedupes_same_message(self, setup):
        """Only the first reaction on a bot message should trigger feedback."""
        bridge, adapter, task_manager = setup

        bridge._bot_message_map["bot_dup"] = {
            "session_id": "sess_1",
            "channel_id": "ch1",
            "conversation_id": "conv1",
            "sender_id": "user1",
            "text": "Reply body",
        }

        task_manager.start_chat = AsyncMock(return_value=MagicMock())

        async def _fake_consume(task, start_from=None):
            if False:
                yield
            return

        task_manager.consume_events = _fake_consume

        base = dict(
            channel_id="ch1",
            sender_id="user2",
            conversation_id="conv1",
            target_message_id="bot_dup",
        )
        await bridge.handle_reaction(ReactionEvent(**base, emoji="thumbsup"), adapter)
        await bridge.handle_reaction(ReactionEvent(**base, emoji="thumbsdown"), adapter)
        await bridge.handle_reaction(ReactionEvent(**base, emoji="heart"), adapter)

        assert task_manager.start_chat.await_count == 1

    @pytest.mark.asyncio
    async def test_handle_reaction_unknown_message(self, setup):
        """Unknown bot messages should silently return without starting a task."""
        bridge, adapter, task_manager = setup
        task_manager.start_chat = AsyncMock()

        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user2",
            conversation_id="conv1",
            target_message_id="unknown_msg",
            emoji="thumbsup",
        )

        await bridge.handle_reaction(event, adapter)

        task_manager.start_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tracked_reference_includes_sql_when_text_empty(self, setup):
        """SQL-only replies (empty text, SQL in outbound.sql) must still populate reference text."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_sql("SELECT 1")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(bridge._bot_message_map) == 1
        tracked = next(iter(bridge._bot_message_map.values()))
        assert tracked["text"] == "```sql\nSELECT 1\n```"

    @pytest.mark.asyncio
    async def test_tracked_reference_combines_text_and_sql(self, setup):
        """When both markdown text and SQL are present, tracked reference should concat both."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        combined_event = SSEEvent(
            id=10,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m10",
                    role="assistant",
                    content=[
                        IMessageContent(type="markdown", payload={"content": "Here is the query"}),
                        IMessageContent(type="code", payload={"content": "SELECT 2", "codeType": "sql"}),
                    ],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )

        async def _fake_consume(task, start_from=None):
            yield combined_event
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        tracked = next(iter(bridge._bot_message_map.values()))
        assert tracked["text"] == "Here is the query\n\n```sql\nSELECT 2\n```"

    @pytest.mark.asyncio
    async def test_track_bot_message_stores_text(self, setup):
        """_track_bot_message should persist the bot reply text for later reference."""
        bridge, _, _ = setup
        msg = _make_inbound()

        await bridge._track_bot_message("bot_1", "sess_1", msg, text="Hello world")

        assert bridge._bot_message_map["bot_1"]["text"] == "Hello world"
        assert bridge._bot_message_map["bot_1"]["session_id"] == "sess_1"

    @pytest.mark.asyncio
    async def test_track_bot_message_accumulates_streamed_text(self, setup):
        """Repeated tracking for the same bot_msg_id should concat streamed fragments."""
        bridge, _, _ = setup
        msg = _make_inbound()

        await bridge._track_bot_message("bot_1", "sess_1", msg, text="Hello ")
        await bridge._track_bot_message("bot_1", "sess_1", msg, text="world")

        assert bridge._bot_message_map["bot_1"]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_handle_reaction_rolls_back_dedup_on_start_chat_value_error(self, setup):
        """If start_chat raises ValueError, the message should NOT stay marked as reacted."""
        bridge, adapter, task_manager = setup

        bridge._bot_message_map["bot_retry"] = {
            "session_id": "sess_1",
            "channel_id": "ch1",
            "conversation_id": "conv1",
            "sender_id": "user1",
            "text": "Reply body",
        }

        task_manager.start_chat = AsyncMock(side_effect=ValueError("transient"))

        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user2",
            conversation_id="conv1",
            target_message_id="bot_retry",
            emoji="thumbsup",
        )

        await bridge.handle_reaction(event, adapter)

        # Dedup mark rolled back so a later reaction on the same message can retry
        assert "bot_retry" not in bridge._reacted_bot_messages

        # Second reaction should attempt start_chat again, not be silently skipped
        task_manager.start_chat.reset_mock(return_value=True, side_effect=True)
        task_manager.start_chat = AsyncMock(return_value=MagicMock())

        async def _empty(task, start_from=None):
            if False:
                yield
            return

        task_manager.consume_events = _empty

        await bridge.handle_reaction(event, adapter)
        task_manager.start_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_reaction_rolls_back_dedup_on_start_chat_unexpected_error(self, setup):
        """Unexpected errors from start_chat must also roll back the dedup mark."""
        bridge, adapter, task_manager = setup

        bridge._bot_message_map["bot_boom"] = {
            "session_id": "sess_1",
            "channel_id": "ch1",
            "conversation_id": "conv1",
            "sender_id": "user1",
            "text": "Reply body",
        }

        task_manager.start_chat = AsyncMock(side_effect=RuntimeError("boom"))

        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user2",
            conversation_id="conv1",
            target_message_id="bot_boom",
            emoji="thumbsup",
        )

        await bridge.handle_reaction(event, adapter)

        assert "bot_boom" not in bridge._reacted_bot_messages

    @pytest.mark.asyncio
    async def test_bot_message_map_eviction(self, setup):
        """_bot_message_map should not exceed _MAX_BOT_MSG_MAP."""
        bridge, _, _ = setup

        msg = _make_inbound()
        for i in range(_MAX_BOT_MSG_MAP + 100):
            await bridge._track_bot_message(f"msg_{i}", "sess", msg)

        assert len(bridge._bot_message_map) == _MAX_BOT_MSG_MAP
        # Oldest entries should be evicted
        assert "msg_0" not in bridge._bot_message_map
        assert f"msg_{_MAX_BOT_MSG_MAP + 99}" in bridge._bot_message_map


class TestVerboseLevels:
    """Tests that verbose level correctly filters SSE events."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StreamingStubAdapter()
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    def _events(self):
        return [
            _make_sse_thinking("thinking about it"),
            _make_sse_tool_call("search_table"),
            _make_sse_tool_result("search_table", "Found 3 tables", 1.5),
            _make_sse_message("Final answer"),
            _make_sse_end(),
        ]

    @pytest.mark.asyncio
    async def test_off_shows_thinking_hides_tools(self, setup):
        """OFF mode: thinking + markdown/code/error pass through, tools hidden."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            for e in self._events():
                yield e

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.OFF)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        # thinking(1) + markdown(1) = 2, tools hidden
        assert len(adapter.sent) == 2
        assert "thinking about it" in adapter.sent[0].text
        assert "Final answer" in adapter.sent[1].text

    @pytest.mark.asyncio
    async def test_on_shows_thinking_and_tool_summary(self, setup):
        """ON mode: thinking + tool summary + markdown."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            for e in self._events():
                yield e

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.ON)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        # thinking(1) + merged tool(1) + markdown(1) = 3
        assert len(adapter.sent) == 3
        assert "thinking about it" in adapter.sent[0].text
        assert "search_table" in adapter.sent[1].text
        assert "Final answer" in adapter.sent[2].text

    @pytest.mark.asyncio
    async def test_full_shows_params_and_results(self, setup):
        """FULL mode: tool output includes params and result details."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        # Use events with params in call-tool
        tool_call = SSEEvent(
            id=3,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m3",
                    role="assistant",
                    content=[
                        IMessageContent(
                            type="call-tool",
                            payload={
                                "callToolId": "t1",
                                "toolName": "search_table",
                                "toolParams": {"query": "revenue", "database": "analytics"},
                            },
                        )
                    ],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )
        tool_result = SSEEvent(
            id=4,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m4",
                    role="assistant",
                    content=[
                        IMessageContent(
                            type="call-tool-result",
                            payload={
                                "callToolId": "t1",
                                "toolName": "search_table",
                                "duration": 1.5,
                                "shortDesc": "",
                                "result": {"metadata": [1], "sample_data": [1, 2]},
                            },
                        )
                    ],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )

        events = [tool_call, tool_result, _make_sse_message("done"), _make_sse_end()]

        async def _fake_consume(task, start_from=None):
            for e in events:
                yield e

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.FULL)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        # merged tool(1) + markdown(1) = 2
        assert len(adapter.sent) == 2
        tool_msg = adapter.sent[0].text
        assert "search_table" in tool_msg
        # "query" is a SQL key so it's wrapped in a code block, not inline
        assert "revenue" in tool_msg
        assert "database: analytics" in tool_msg

    @pytest.mark.asyncio
    async def test_off_still_shows_error(self, setup):
        """OFF mode should still show error events."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        error_event = SSEEvent(
            id=1,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m1",
                    role="assistant",
                    content=[IMessageContent(type="error", payload={"content": "Something broke"})],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )

        async def _fake_consume(task, start_from=None):
            yield error_event
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.OFF)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        assert len(adapter.sent) == 1
        assert "Something broke" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_off_still_shows_code(self, setup):
        """OFF mode should still show non-SQL code blocks."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        code_event = SSEEvent(
            id=1,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m1",
                    role="assistant",
                    content=[IMessageContent(type="code", payload={"content": "print('hi')", "codeType": "python"})],
                ),
            ),
            timestamp=datetime.now().isoformat() + "Z",
        )

        async def _fake_consume(task, start_from=None):
            yield code_event
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.OFF)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        assert len(adapter.sent) == 1
        assert "print('hi')" in adapter.sent[0].text

    @pytest.mark.asyncio
    async def test_default_verbose_is_on(self, setup):
        """Without channel_config, default verbose should be ON."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            for e in self._events():
                yield e

        task_manager.consume_events = _fake_consume

        # No channel_config passed
        await bridge.handle_message(_make_inbound("q"), adapter)

        # ON mode: thinking + merged tool + markdown = 3
        assert len(adapter.sent) == 3

    @pytest.mark.asyncio
    async def test_pending_tool_calls_scoped_per_request(self, setup):
        """pending_tool_calls is local per request and does not leak across messages."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_tool_call("orphan_tool")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        # Should complete without error; orphan tool call stays in local dict only
        await bridge.handle_message(_make_inbound("q"), adapter)
        # No instance-level _pending_tool_calls attribute should exist
        assert not hasattr(bridge, "_pending_tool_calls")


class TestStreamIdAndFinalize:
    """Tests that stream_id is set on outbound messages and finalize_stream is called."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StreamingStubAdapter()
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    @pytest.mark.asyncio
    async def test_stream_id_set_on_outbound(self, setup):
        """Each outbound message should have stream_id when adapter supports streaming."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("first")
            yield _make_sse_message("second")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 2
        expected_stream_id = "ch1_msg1"
        assert adapter.sent[0].stream_id == expected_stream_id
        assert adapter.sent[1].stream_id == expected_stream_id

    @pytest.mark.asyncio
    async def test_finalize_stream_called_on_adapter_with_method(self, setup):
        """finalize_stream should be called in finally block."""
        bridge, adapter, task_manager = setup

        adapter.finalize_stream = AsyncMock()

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("done")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        adapter.finalize_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalize_stream_called_even_on_error(self, setup):
        """finalize_stream should be called even when streaming raises."""
        bridge, adapter, task_manager = setup

        adapter.finalize_stream = AsyncMock()

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            raise RuntimeError("boom")
            yield  # noqa: E501

        task_manager.consume_events = _fake_consume

        with pytest.raises(RuntimeError, match="boom"):
            await bridge.handle_message(_make_inbound("q"), adapter)

        adapter.finalize_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_finalize_on_adapter_without_override(self, setup):
        """Adapters with default no-op finalize_stream should not cause errors."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("ok")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)
        assert len(adapter.sent) == 1

    @pytest.mark.asyncio
    async def test_fallback_message_has_no_stream_id(self, setup):
        """The '(No response generated)' fallback should not have stream_id."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].stream_id is None

    @pytest.mark.asyncio
    async def test_busy_reply_has_no_stream_id(self, setup):
        """The busy reply should not have stream_id."""
        bridge, adapter, task_manager = setup

        task_manager.start_chat = AsyncMock(side_effect=ValueError("session is still being processed"))

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].stream_id is None


class TestNonStreamingAdapter:
    """Tests that non-streaming adapters skip deltas and don't get stream_id."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StubAdapter()  # supports_streaming = False
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    @pytest.mark.asyncio
    async def test_no_stream_id_on_non_streaming_adapter(self, setup):
        """Non-streaming adapters should not get stream_id on outbound messages."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("result")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].stream_id is None

    @pytest.mark.asyncio
    async def test_delta_messages_skipped(self, setup):
        """Non-streaming adapters should skip delta (thinking) messages."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_thinking("analyzing...")
            yield _make_sse_message("final result")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        # Only the non-delta markdown message should be sent, thinking is skipped
        assert len(adapter.sent) == 1
        assert "final result" in adapter.sent[0].text


class TestStreamResponseConfig:
    """Tests that channel_config.stream_response=False disables streaming on capable adapters."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StreamingStubAdapter()  # supports_streaming = True
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    @pytest.mark.asyncio
    async def test_stream_response_false_disables_stream_id(self, setup):
        """When channel_config.stream_response=False, stream_id should not be set."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("result")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", stream_response=False)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        assert len(adapter.sent) == 1
        assert adapter.sent[0].stream_id is None

    @pytest.mark.asyncio
    async def test_stream_response_false_skips_deltas(self, setup):
        """When channel_config.stream_response=False, delta messages should be skipped."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_thinking("analyzing...")
            yield _make_sse_message("final result")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        cfg = ChannelConfig(adapter="feishu", stream_response=False)
        await bridge.handle_message(_make_inbound("q"), adapter, channel_config=cfg)

        assert len(adapter.sent) == 1
        assert "final result" in adapter.sent[0].text


class TestStreamMessageIdSeparator:
    """Tests that \\n\\n separator is prepended when SSE message_id changes during streaming."""

    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        adapter = _StreamingStubAdapter()
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        bridge = ChannelBridge(agent_config, task_manager)
        return bridge, adapter, task_manager

    @pytest.mark.asyncio
    async def test_separator_prepended_on_message_id_change(self, setup):
        """When message_id changes, the outbound text should be prepended with \\n\\n."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("first part", message_id="msg_a")
            yield _make_sse_message("second part", message_id="msg_b")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 2
        # First message: no separator (first message_id seen)
        assert adapter.sent[0].text == "first part"
        # Second message: prepended with \n\n because message_id changed
        assert adapter.sent[1].text == "\n\nsecond part"

    @pytest.mark.asyncio
    async def test_no_separator_when_message_id_unchanged(self, setup):
        """When message_id stays the same, no separator should be added."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("chunk1", message_id="msg_a")
            yield _make_sse_message("chunk2", message_id="msg_a")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 2
        assert adapter.sent[0].text == "chunk1"
        assert adapter.sent[1].text == "chunk2"

    @pytest.mark.asyncio
    async def test_separator_only_in_streaming_mode(self, setup):
        """Non-streaming adapters should not get separators."""
        bridge, _, task_manager = setup
        non_stream_adapter = _StubAdapter()

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("first", message_id="msg_a")
            yield _make_sse_message("second", message_id="msg_b")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), non_stream_adapter)

        assert len(non_stream_adapter.sent) == 2
        assert non_stream_adapter.sent[0].text == "first"
        assert non_stream_adapter.sent[1].text == "second"

    @pytest.mark.asyncio
    async def test_multiple_message_id_changes(self, setup):
        """Multiple message_id transitions should each get a separator."""
        bridge, adapter, task_manager = setup

        mock_task = MagicMock()
        task_manager.start_chat = AsyncMock(return_value=mock_task)

        async def _fake_consume(task, start_from=None):
            yield _make_sse_message("a", message_id="id1")
            yield _make_sse_message("b", message_id="id2")
            yield _make_sse_message("c", message_id="id3")
            yield _make_sse_end()

        task_manager.consume_events = _fake_consume

        await bridge.handle_message(_make_inbound("q"), adapter)

        assert len(adapter.sent) == 3
        assert adapter.sent[0].text == "a"
        assert adapter.sent[1].text == "\n\nb"
        assert adapter.sent[2].text == "\n\nc"


class TestVerboseOverride:
    """Tests for set_verbose / get_verbose per-conversation override."""

    @pytest.fixture
    def bridge(self, tmp_path):
        agent_config = MagicMock()
        agent_config.session_dir = str(tmp_path / "sessions")
        task_manager = MagicMock()
        return ChannelBridge(agent_config, task_manager)

    def test_get_verbose_default_is_on(self, bridge):
        msg = _make_inbound()
        assert bridge.get_verbose(msg) == Verbose.ON

    def test_get_verbose_fallback_to_channel_config(self, bridge):
        msg = _make_inbound()
        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.FULL)
        assert bridge.get_verbose(msg, cfg) == Verbose.FULL

    def test_set_verbose_overrides_channel_config(self, bridge):
        msg = _make_inbound()
        cfg = ChannelConfig(adapter="feishu", verbose=Verbose.FULL)
        bridge.set_verbose(msg, Verbose.OFF)
        assert bridge.get_verbose(msg, cfg) == Verbose.OFF

    def test_different_conversations_independent(self, bridge):
        msg1 = _make_inbound()
        msg2 = InboundMessage(
            channel_id="ch2",
            sender_id="user2",
            conversation_id="conv2",
            message_id="msg2",
            text="hello",
        )
        bridge.set_verbose(msg1, Verbose.OFF)
        # msg2 should still use default
        assert bridge.get_verbose(msg2) == Verbose.ON
        assert bridge.get_verbose(msg1) == Verbose.OFF
