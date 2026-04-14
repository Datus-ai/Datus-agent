# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the Slack adapter — bot mention detection, chat type mapping, and core adapter methods."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.claw.adapters.slack import SlackAdapter
from datus.claw.models import InboundMessage, OutboundMessage, ReactionEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_slack_adapter(bot_user_id: str = "") -> SlackAdapter:
    """Create a SlackAdapter with minimal config and no real connections."""
    bridge = MagicMock()
    bridge.handle_message = AsyncMock()
    bridge.handle_reaction = AsyncMock()
    adapter = SlackAdapter(
        channel_id="test-slack",
        config={"app_token": "test-app-token", "bot_token": "test-bot-token"},
        bridge=bridge,
    )
    adapter._bot_user_id = bot_user_id
    return adapter


def _make_socket_event(
    text: str = "hello",
    user: str = "U123",
    channel: str = "C456",
    ts: str = "1234567890.123456",
    channel_type: str = "",
    thread_ts: str | None = None,
    bot_id: str | None = None,
    subtype: str | None = None,
    event_type: str = "message",
) -> dict:
    """Build a minimal Slack events_api payload."""
    event: dict = {
        "type": event_type,
        "text": text,
        "user": user,
        "channel": channel,
        "ts": ts,
        "channel_type": channel_type,
    }
    if thread_ts:
        event["thread_ts"] = thread_ts
    if bot_id:
        event["bot_id"] = bot_id
    if subtype:
        event["subtype"] = subtype
    return {"event": event}


def _make_socket_request(payload: dict, envelope_id: str = "env-1") -> SimpleNamespace:
    """Build a mock SocketModeRequest."""
    return SimpleNamespace(
        type="events_api",
        payload=payload,
        envelope_id=envelope_id,
    )


# ---------------------------------------------------------------------------
# Tests: Bot mention detection
# ---------------------------------------------------------------------------
class TestSlackMentionDetection:
    """Tests for bot mention detection in Slack message text."""

    def test_mentions_bot_when_present(self):
        adapter = _make_slack_adapter("U_BOT")
        text = "<@U_BOT> what is the revenue?"
        assert f"<@{adapter._bot_user_id}>" in text

    def test_no_mention_when_absent(self):
        adapter = _make_slack_adapter("U_BOT")
        text = "what is the revenue?"
        assert f"<@{adapter._bot_user_id}>" not in text

    def test_no_mention_when_bot_id_empty(self):
        adapter = _make_slack_adapter("")
        text = "<@U_BOT> hello"
        mentions_bot = bool(adapter._bot_user_id and f"<@{adapter._bot_user_id}>" in text)
        assert mentions_bot is False


class TestSlackStripBotMention:
    """Tests for removing @bot mention text from messages."""

    def test_strip_bot_mention(self):
        bot_user_id = "U_BOT"
        text = "<@U_BOT> what is the revenue?"
        cleaned = text.replace(f"<@{bot_user_id}>", "").strip()
        assert cleaned == "what is the revenue?"

    def test_strip_bot_mention_multiple(self):
        bot_user_id = "U_BOT"
        text = "<@U_BOT> hey <@U_BOT>"
        cleaned = text.replace(f"<@{bot_user_id}>", "").strip()
        assert cleaned == "hey"

    def test_strip_preserves_other_mentions(self):
        bot_user_id = "U_BOT"
        text = "<@U_BOT> cc <@U_OTHER>"
        cleaned = text.replace(f"<@{bot_user_id}>", "").strip()
        assert "<@U_OTHER>" in cleaned
        assert "<@U_BOT>" not in cleaned


class TestSlackChatTypeMapping:
    """Tests for Slack channel_type -> chat_type mapping."""

    @pytest.mark.parametrize(
        "channel_type,expected",
        [
            ("im", "p2p"),
            ("channel", "group"),
            ("group", "group"),
            ("mpim", "group"),
            ("", None),
        ],
    )
    def test_chat_type_mapping(self, channel_type, expected):
        chat_type = "p2p" if channel_type == "im" else ("group" if channel_type else None)
        assert chat_type == expected


# ---------------------------------------------------------------------------
# Tests: Adapter send_message
# ---------------------------------------------------------------------------
class TestSlackSendMessage:
    """Tests for SlackAdapter.send_message."""

    @pytest.mark.asyncio
    async def test_send_message_plain_text(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.chat_postMessage = AsyncMock(return_value={"ts": "1111.2222"})
        adapter._web_client = mock_web

        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="hello")
        ts = await adapter.send_message(msg)

        assert ts == "1111.2222"
        mock_web.chat_postMessage.assert_called_once()
        call_kwargs = mock_web.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C456"
        assert call_kwargs["text"] == "hello"
        assert "thread_ts" not in call_kwargs

    @pytest.mark.asyncio
    async def test_send_message_with_thread_id(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.chat_postMessage = AsyncMock(return_value={"ts": "1111.3333"})
        adapter._web_client = mock_web

        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="reply", thread_id="1111.0000")
        ts = await adapter.send_message(msg)

        assert ts == "1111.3333"
        call_kwargs = mock_web.chat_postMessage.call_args[1]
        assert call_kwargs["thread_ts"] == "1111.0000"

    @pytest.mark.asyncio
    async def test_send_message_with_sql(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.chat_postMessage = AsyncMock(return_value={"ts": "1111.4444"})
        adapter._web_client = mock_web

        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="Result:", sql="SELECT 1")
        await adapter.send_message(msg)

        call_kwargs = mock_web.chat_postMessage.call_args[1]
        assert "```SELECT 1```" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_message_returns_none_when_web_client_missing(self):
        adapter = _make_slack_adapter()
        adapter._web_client = None

        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="hello")
        result = await adapter.send_message(msg)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_message_returns_none_on_api_error(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.chat_postMessage = AsyncMock(side_effect=Exception("Slack API error"))
        adapter._web_client = mock_web

        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="hello")
        result = await adapter.send_message(msg)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_message_with_ir_renders_richtext(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.chat_postMessage = AsyncMock(return_value={"ts": "1111.5555"})
        adapter._web_client = mock_web

        from datus.claw.richtext.ir import MarkdownIR, StyleSpan, StyleType

        ir = MarkdownIR(text="hello world", styles=[StyleSpan(style=StyleType.BOLD, start=0, end=5)])
        msg = OutboundMessage(channel_id="test-slack", conversation_id="C456", text="hello world", ir=ir)
        await adapter.send_message(msg)

        call_kwargs = mock_web.chat_postMessage.call_args[1]
        # Rendered IR should contain bold markers
        assert "*" in call_kwargs["text"]


# ---------------------------------------------------------------------------
# Tests: Adapter reactions
# ---------------------------------------------------------------------------
class TestSlackReactions:
    """Tests for add_reaction and remove_reaction."""

    @pytest.mark.asyncio
    async def test_add_reaction(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        adapter._web_client = mock_web

        await adapter.add_reaction("C456", "1111.2222", "thumbsup")
        mock_web.reactions_add.assert_called_once_with(channel="C456", name="thumbsup", timestamp="1111.2222")

    @pytest.mark.asyncio
    async def test_remove_reaction(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        adapter._web_client = mock_web

        await adapter.remove_reaction("C456", "1111.2222", "thumbsup")
        mock_web.reactions_remove.assert_called_once_with(channel="C456", name="thumbsup", timestamp="1111.2222")

    @pytest.mark.asyncio
    async def test_add_reaction_no_client(self):
        adapter = _make_slack_adapter()
        adapter._web_client = None
        # Should not raise
        await adapter.add_reaction("C456", "1111.2222", "thumbsup")

    @pytest.mark.asyncio
    async def test_remove_reaction_no_client(self):
        adapter = _make_slack_adapter()
        adapter._web_client = None
        await adapter.remove_reaction("C456", "1111.2222", "thumbsup")

    @pytest.mark.asyncio
    async def test_add_reaction_api_error_does_not_raise(self):
        adapter = _make_slack_adapter()
        mock_web = AsyncMock()
        mock_web.reactions_add = AsyncMock(side_effect=Exception("rate_limited"))
        adapter._web_client = mock_web

        # Should not propagate
        await adapter.add_reaction("C456", "1111.2222", "thumbsup")


# ---------------------------------------------------------------------------
# Tests: Adapter stop
# ---------------------------------------------------------------------------
class TestSlackStop:
    """Tests for SlackAdapter.stop."""

    @pytest.mark.asyncio
    async def test_stop_disconnects_socket_client(self):
        adapter = _make_slack_adapter()
        mock_socket = AsyncMock()
        mock_socket.disconnect = AsyncMock()
        adapter._socket_client = mock_socket
        adapter._web_client = AsyncMock()

        await adapter.stop()

        mock_socket.disconnect.assert_called_once()
        assert adapter._socket_client is None
        assert adapter._web_client is None

    @pytest.mark.asyncio
    async def test_stop_cancels_listen_task(self):
        adapter = _make_slack_adapter()
        adapter._socket_client = None
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        adapter._listen_task = mock_task

        await adapter.stop()

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped(self):
        adapter = _make_slack_adapter()
        adapter._socket_client = None
        adapter._web_client = None
        adapter._listen_task = None

        # Should not raise
        await adapter.stop()


# ---------------------------------------------------------------------------
# Tests: Handle socket request (message dispatch)
# ---------------------------------------------------------------------------
class TestSlackHandleSocketRequest:
    """Tests for SlackAdapter._handle_socket_request."""

    @pytest.fixture(autouse=True)
    def _patch_socket_mode_response(self):
        """Patch SocketModeResponse to avoid importing slack_sdk."""
        with patch("datus.claw.adapters.slack.SocketModeResponse", create=True) as mock_cls:
            mock_cls.return_value = SimpleNamespace()
            # The import is inside the method, so we patch the module-level import path
            with patch.dict("sys.modules", {"slack_sdk.socket_mode.response": MagicMock(SocketModeResponse=mock_cls)}):
                yield mock_cls

    @pytest.mark.asyncio
    async def test_dispatch_regular_dm_message(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = _make_socket_event(text="hello", channel_type="im")
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_message.assert_called_once()
        call_args = adapter._bridge.handle_message.call_args
        msg: InboundMessage = call_args[0][0]
        assert msg.text == "hello"
        assert msg.chat_type == "p2p"
        assert msg.mentions_bot is False

    @pytest.mark.asyncio
    async def test_dispatch_message_with_bot_mention(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = _make_socket_event(text="<@U_BOT> show revenue", channel_type="channel")
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_message.assert_called_once()
        msg: InboundMessage = adapter._bridge.handle_message.call_args[0][0]
        assert msg.mentions_bot is True
        assert "<@U_BOT>" not in msg.text
        assert msg.text == "show revenue"

    @pytest.mark.asyncio
    async def test_skip_bot_messages(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = _make_socket_event(text="bot reply", bot_id="B123")
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_subtype_messages(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = _make_socket_event(text="edited", subtype="message_changed")
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_non_events_api(self):
        adapter = _make_slack_adapter("U_BOT")
        req = SimpleNamespace(type="slash_commands", payload={}, envelope_id="env-1")

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_thread_ts_propagated(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = _make_socket_event(text="thread reply", thread_ts="1111.0000", channel_type="channel")
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        msg: InboundMessage = adapter._bridge.handle_message.call_args[0][0]
        assert msg.thread_id == "1111.0000"

    @pytest.mark.asyncio
    async def test_reaction_added_dispatched(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = {
            "event": {
                "type": "reaction_added",
                "user": "U123",
                "reaction": "thumbsup",
                "item": {"channel": "C456", "ts": "1111.2222"},
            }
        }
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_reaction.assert_called_once()
        event: ReactionEvent = adapter._bridge.handle_reaction.call_args[0][0]
        assert event.emoji == "thumbsup"
        assert event.action == "added"
        assert event.target_message_id == "1111.2222"

    @pytest.mark.asyncio
    async def test_reaction_removed_dispatched(self):
        adapter = _make_slack_adapter("U_BOT")
        payload = {
            "event": {
                "type": "reaction_removed",
                "user": "U123",
                "reaction": "thumbsup",
                "item": {"channel": "C456", "ts": "1111.2222"},
            }
        }
        req = _make_socket_request(payload)

        mock_client = AsyncMock()
        await adapter._handle_socket_request(mock_client, req)

        adapter._bridge.handle_reaction.assert_called_once()
        event: ReactionEvent = adapter._bridge.handle_reaction.call_args[0][0]
        assert event.action == "removed"


# ---------------------------------------------------------------------------
# Tests: Render IR
# ---------------------------------------------------------------------------
class TestSlackRenderIr:
    """Tests for SlackAdapter._render_ir."""

    def test_render_bold(self):
        from datus.claw.richtext.ir import MarkdownIR, StyleSpan, StyleType

        adapter = _make_slack_adapter()
        ir = MarkdownIR(text="hello world", styles=[StyleSpan(style=StyleType.BOLD, start=0, end=5)])
        result = adapter._render_ir(ir)
        assert result.startswith("*")
        assert "hello" in result

    def test_render_plain_text(self):
        from datus.claw.richtext.ir import MarkdownIR

        adapter = _make_slack_adapter()
        ir = MarkdownIR(text="no styles here")
        result = adapter._render_ir(ir)
        assert result == "no styles here"


# ---------------------------------------------------------------------------
# Tests: supports_streaming
# ---------------------------------------------------------------------------
class TestSlackSupportsStreaming:
    """Slack adapter should not support streaming."""

    def test_supports_streaming_is_false(self):
        adapter = _make_slack_adapter()
        assert adapter.supports_streaming is False
