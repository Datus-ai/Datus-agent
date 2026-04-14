# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the Feishu adapter — interactive card message sending."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from datus.claw.models import OutboundMessage


# ---------------------------------------------------------------------------
# Stub the lark_oapi imports so we don't need the real SDK installed
# ---------------------------------------------------------------------------
def _build_lark_stubs():
    """Create minimal stubs for lark_oapi so the adapter can be imported."""
    lark = ModuleType("lark_oapi")
    lark_im = ModuleType("lark_oapi.api")
    lark_im_v1 = ModuleType("lark_oapi.api.im")
    lark_im_v1_mod = ModuleType("lark_oapi.api.im.v1")
    lark_cardkit = ModuleType("lark_oapi.api.cardkit")
    lark_cardkit_v1 = ModuleType("lark_oapi.api.cardkit.v1")

    class _Builder:
        def __getattr__(self, name):
            def _setter(*args, **kwargs):
                return self

            return _setter

        def build(self):
            return SimpleNamespace()

    class CreateMessageRequestBody:
        @staticmethod
        def builder():
            return _Builder()

    class CreateMessageRequest:
        @staticmethod
        def builder():
            return _Builder()

    class IdConvertCardRequestBody:
        @staticmethod
        def builder():
            return _Builder()

    class IdConvertCardRequest:
        @staticmethod
        def builder():
            return _Builder()

    class ContentCardElementRequestBody:
        @staticmethod
        def builder():
            return _Builder()

    class ContentCardElementRequest:
        @staticmethod
        def builder():
            return _Builder()

    class SettingsCardRequestBody:
        @staticmethod
        def builder():
            return _Builder()

    class SettingsCardRequest:
        @staticmethod
        def builder():
            return _Builder()

    class ReplyMessageRequestBody:
        @staticmethod
        def builder():
            return _Builder()

    class ReplyMessageRequest:
        @staticmethod
        def builder():
            return _Builder()

    lark_im_v1_mod.CreateMessageRequest = CreateMessageRequest
    lark_im_v1_mod.CreateMessageRequestBody = CreateMessageRequestBody
    lark_im_v1_mod.ReplyMessageRequest = ReplyMessageRequest
    lark_im_v1_mod.ReplyMessageRequestBody = ReplyMessageRequestBody

    lark_cardkit_v1.IdConvertCardRequest = IdConvertCardRequest
    lark_cardkit_v1.IdConvertCardRequestBody = IdConvertCardRequestBody
    lark_cardkit_v1.ContentCardElementRequest = ContentCardElementRequest
    lark_cardkit_v1.ContentCardElementRequestBody = ContentCardElementRequestBody
    lark_cardkit_v1.SettingsCardRequest = SettingsCardRequest
    lark_cardkit_v1.SettingsCardRequestBody = SettingsCardRequestBody

    for mod_name, mod_obj in [
        ("lark_oapi", lark),
        ("lark_oapi.api", lark_im),
        ("lark_oapi.api.im", lark_im_v1),
        ("lark_oapi.api.im.v1", lark_im_v1_mod),
        ("lark_oapi.api.cardkit", lark_cardkit),
        ("lark_oapi.api.cardkit.v1", lark_cardkit_v1),
    ]:
        sys.modules.setdefault(mod_name, mod_obj)

    return lark_im_v1_mod


_lark_v1 = _build_lark_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_adapter():
    """Instantiate a FeishuAdapter with a dummy bridge."""
    from datus.claw.adapters.feishu import FeishuAdapter

    bridge = MagicMock()
    adapter = FeishuAdapter(
        channel_id="test-feishu",
        config={"app_id": "fake_id", "app_secret": "fake_secret"},
        bridge=bridge,
    )
    return adapter


def _make_message(text: str = "hello", sql: Optional[str] = None) -> OutboundMessage:
    return OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        text=text,
        sql=sql,
    )


# ---------------------------------------------------------------------------
# Capture helpers — intercept what gets passed to the SDK
# ---------------------------------------------------------------------------
class _BodyCapture:
    """Capture the arguments passed to CreateMessageRequestBody.builder()."""

    def __init__(self):
        self.receive_id_val = None
        self.msg_type_val = None
        self.content_val = None

    def builder(self):
        return self

    def receive_id(self, val):
        self.receive_id_val = val
        return self

    def msg_type(self, val):
        self.msg_type_val = val
        return self

    def content(self, val):
        self.content_val = val
        return self

    def build(self):
        return SimpleNamespace()


def _make_streaming_message(
    text: str = "hello", sql: Optional[str] = None, stream_id: str = "stream_1"
) -> OutboundMessage:
    return OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        text=text,
        sql=sql,
        stream_id=stream_id,
    )


def _mock_lark_client(response_success=True, message_id="msg_ok_001", card_id="card_001"):
    """Create a mock lark client whose im.v1.message.create is synchronous."""
    create_resp = SimpleNamespace(
        success=lambda: response_success,
        code=0,
        msg="ok",
        data=SimpleNamespace(message_id=message_id),
    )
    id_convert_resp = SimpleNamespace(
        success=lambda: response_success,
        code=0,
        msg="ok",
        data=SimpleNamespace(card_id=card_id),
    )
    content_resp = SimpleNamespace(success=lambda: True, code=0, msg="ok")
    settings_resp = SimpleNamespace(success=lambda: True, code=0, msg="ok")

    client = MagicMock()
    client.im.v1.message.create = MagicMock(return_value=create_resp)
    client.cardkit.v1.card.id_convert = MagicMock(return_value=id_convert_resp)
    client.cardkit.v1.card_element.content = MagicMock(return_value=content_resp)
    client.cardkit.v1.card.settings = MagicMock(return_value=settings_resp)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_message_uses_interactive_card():
    """msg_type should be 'interactive', not 'text'."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(_make_message("hi"))

    assert capture.msg_type_val == "interactive"


@pytest.mark.asyncio
async def test_send_message_card_structure():
    """The content JSON should be a valid interactive card with markdown element."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(_make_message("hello world"))

    card = json.loads(capture.content_val)
    assert card["config"]["wide_screen_mode"] is True
    assert len(card["elements"]) == 1
    assert card["elements"][0]["tag"] == "markdown"
    assert card["elements"][0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_send_message_appends_sql():
    """SQL should be appended as a fenced code block inside the card markdown."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(_make_message("Result:", sql="SELECT 1"))

    card = json.loads(capture.content_val)
    content = card["elements"][0]["content"]
    assert "```sql\nSELECT 1\n```" in content
    assert content.startswith("Result:")


@pytest.mark.asyncio
async def test_send_message_plain_text_no_sql():
    """Without SQL, the card markdown should be just the text."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(_make_message("only text"))

    card = json.loads(capture.content_val)
    assert card["elements"][0]["content"] == "only text"


@pytest.mark.asyncio
async def test_send_message_uses_text_not_ir():
    """Even if ir is set, send_message should use message.text directly (IR is ignored)."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    msg = _make_message("raw markdown **bold**")

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(msg)

    card = json.loads(capture.content_val)
    assert card["elements"][0]["content"] == "raw markdown **bold**"


@pytest.mark.asyncio
async def test_send_message_returns_message_id():
    """send_message should return the message_id from the response."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_12345")

    result = await adapter.send_message(_make_message("test"))
    assert result == "msg_12345"


@pytest.mark.asyncio
async def test_send_message_returns_none_on_failure():
    """send_message should return None when the API call fails."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(response_success=False)

    result = await adapter.send_message(_make_message("test"))
    assert result is None


# ---------------------------------------------------------------------------
# Streaming card tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_streaming_first_message_creates_card():
    """First streaming message should create a streaming card and return message_id."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_stream_1", card_id="card_stream_1")

    result = await adapter.send_message(_make_streaming_message("first chunk"))

    assert result == "msg_stream_1"
    assert adapter._streams["stream_1"].card_id == "card_stream_1"
    assert "stream_1" in adapter._streams
    assert adapter._streams["stream_1"].seq == 1
    assert adapter._streams["stream_1"].accumulated == "first chunk"
    # im.v1.message.create should have been called
    adapter._lark_client.im.v1.message.create.assert_called_once()
    # cardkit id_convert should have been called
    adapter._lark_client.cardkit.v1.card.id_convert.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_card_json_has_streaming_mode():
    """The initial card JSON should have streaming_mode enabled."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    capture = _BodyCapture()
    with patch("lark_oapi.api.im.v1.CreateMessageRequestBody", capture):
        await adapter.send_message(_make_streaming_message("hello"))

    card = json.loads(capture.content_val)
    assert card["config"]["streaming_mode"] is True
    assert card["config"]["update_multi"] is True
    assert card["body"]["elements"][0]["element_id"] == "content_md"


@pytest.mark.asyncio
async def test_streaming_subsequent_message_updates_card():
    """Subsequent messages with same stream_id should update the card, not create a new one."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_s1", card_id="card_s1")

    # First message — creates the card
    result1 = await adapter.send_message(_make_streaming_message("chunk1"))
    assert result1 == "msg_s1"

    # Second message — updates existing card
    result2 = await adapter.send_message(_make_streaming_message("chunk2"))
    assert result2 is None  # No new message created

    # Card element content should have been called for the update
    assert adapter._lark_client.cardkit.v1.card_element.content.call_count == 1
    assert "chunk1\n\nchunk2" in adapter._streams["stream_1"].accumulated
    assert adapter._streams["stream_1"].seq == 2


@pytest.mark.asyncio
async def test_streaming_delta_concatenates_directly():
    """Delta messages (is_delta=True) should be concatenated without separator."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_d1", card_id="card_d1")

    # First message creates the card
    msg1 = _make_streaming_message("Hel")
    await adapter.send_message(msg1)

    # Subsequent delta messages — should concatenate directly
    msg2 = OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        text="lo ",
        stream_id="stream_1",
        is_delta=True,
    )
    await adapter.send_message(msg2)

    msg3 = OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        text="world",
        stream_id="stream_1",
        is_delta=True,
    )
    await adapter.send_message(msg3)

    # Should be directly concatenated: "Hel" + "lo " + "world" = "Hello world"
    assert adapter._streams["stream_1"].accumulated == "Hello world"


@pytest.mark.asyncio
async def test_streaming_non_delta_uses_separator():
    """Non-delta messages should use \\n\\n separator."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_nd", card_id="card_nd")

    # First message creates the card
    await adapter.send_message(_make_streaming_message("first"))

    # Non-delta follow-up (is_delta defaults to False)
    msg2 = _make_streaming_message("second")
    await adapter.send_message(msg2)

    assert adapter._streams["stream_1"].accumulated == "first\n\nsecond"


@pytest.mark.asyncio
async def test_streaming_with_sql_appended():
    """SQL should be appended to text in streaming messages."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_sql", card_id="card_sql")

    await adapter.send_message(_make_streaming_message("Result:", sql="SELECT 1"))

    assert "```sql\nSELECT 1\n```" in adapter._streams["stream_1"].accumulated


@pytest.mark.asyncio
async def test_finalize_stream_disables_streaming():
    """_finalize_stream should call cardkit settings to disable streaming_mode."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_f", card_id="card_f")

    await adapter.send_message(_make_streaming_message("content"))
    await adapter.finalize_stream("stream_1")

    adapter._lark_client.cardkit.v1.card.settings.assert_called_once()
    # State should be cleaned up
    assert "stream_1" not in adapter._streams


@pytest.mark.asyncio
async def test_finalize_stream_noop_when_no_stream():
    """_finalize_stream should be a no-op when there is no active stream."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    # No stream started — should not raise or call any API
    await adapter.finalize_stream("nonexistent_stream")
    adapter._lark_client.cardkit.v1.card.settings.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_different_stream_id_creates_new_card():
    """A message with a different stream_id should start a new streaming card."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_a", card_id="card_a")

    await adapter.send_message(_make_streaming_message("first", stream_id="stream_a"))
    assert "stream_a" in adapter._streams

    # New stream_id — should create a new card
    await adapter.send_message(_make_streaming_message("second", stream_id="stream_b"))
    assert "stream_b" in adapter._streams
    # im.v1.message.create should have been called twice
    assert adapter._lark_client.im.v1.message.create.call_count == 2


@pytest.mark.asyncio
async def test_non_streaming_message_uses_simple_path():
    """Messages without stream_id should use the simple (non-streaming) path."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client()

    result = await adapter.send_message(_make_message("no stream"))
    assert result == "msg_ok_001"
    # cardkit APIs should NOT be called
    adapter._lark_client.cardkit.v1.card.id_convert.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_fallback_on_id_convert_failure():
    """If id_convert fails, should fall back gracefully and still return message_id."""
    adapter = _make_adapter()
    client = _mock_lark_client(message_id="msg_fallback")
    # Make id_convert fail
    fail_resp = SimpleNamespace(success=lambda: False, code=99, msg="convert failed", data=None)
    client.cardkit.v1.card.id_convert = MagicMock(return_value=fail_resp)
    adapter._lark_client = client

    result = await adapter.send_message(_make_streaming_message("test"))

    assert result == "msg_fallback"
    # Stream state should NOT be set since id_convert failed
    assert "stream_1" not in adapter._streams


# ---------------------------------------------------------------------------
# Bot mention detection and text stripping tests
# ---------------------------------------------------------------------------
class TestBotMentionDetection:
    def _make_mention(self, open_id: str, key: str = "@_user_1"):
        """Create a mock mention object."""
        mention = SimpleNamespace()
        mention.id = SimpleNamespace(open_id=open_id)
        mention.key = key
        return mention

    def test_is_bot_mentioned_with_matching_open_id(self):
        adapter = _make_adapter()
        adapter._bot_open_id = "bot_123"
        mentions = [self._make_mention("bot_123")]
        assert adapter._is_bot_mentioned(mentions) is True

    def test_is_bot_mentioned_without_match(self):
        adapter = _make_adapter()
        adapter._bot_open_id = "bot_123"
        mentions = [self._make_mention("other_user")]
        assert adapter._is_bot_mentioned(mentions) is False

    def test_is_bot_mentioned_no_mentions(self):
        adapter = _make_adapter()
        adapter._bot_open_id = "bot_123"
        assert adapter._is_bot_mentioned(None) is False
        assert adapter._is_bot_mentioned([]) is False

    def test_is_bot_mentioned_fallback_when_no_bot_id(self):
        adapter = _make_adapter()
        adapter._bot_open_id = ""
        mentions = [self._make_mention("anyone")]
        assert adapter._is_bot_mentioned(mentions) is True

    def test_strip_bot_mention_removes_placeholder(self):
        adapter = _make_adapter()
        adapter._bot_open_id = "bot_123"
        mentions = [self._make_mention("bot_123", "@_user_1")]
        result = adapter._strip_bot_mention("@_user_1 hello world", mentions)
        assert result == "hello world"

    def test_strip_bot_mention_keeps_other_mentions(self):
        adapter = _make_adapter()
        adapter._bot_open_id = "bot_123"
        bot_mention = self._make_mention("bot_123", "@_user_1")
        other_mention = self._make_mention("other_user", "@_user_2")
        mentions = [bot_mention, other_mention]
        result = adapter._strip_bot_mention("@_user_1 @_user_2 hello", mentions)
        assert "@_user_2" in result
        assert "@_user_1" not in result

    def test_strip_bot_mention_no_bot_id(self):
        adapter = _make_adapter()
        adapter._bot_open_id = ""
        mentions = [self._make_mention("bot_123", "@_user_1")]
        result = adapter._strip_bot_mention("@_user_1 hello", mentions)
        assert result == "@_user_1 hello"


# ---------------------------------------------------------------------------
# Reply-in-thread tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reply_in_thread_uses_reply_api():
    """When thread_id is present, send_simple should use reply API."""
    adapter = _make_adapter()
    client = _mock_lark_client(message_id="msg_reply_1")
    client.im.v1.message.reply = MagicMock(return_value=client.im.v1.message.create.return_value)
    adapter._lark_client = client

    msg = OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        thread_id="om_thread_1",
        text="reply text",
    )
    result = await adapter.send_message(msg)

    assert result == "msg_reply_1"
    client.im.v1.message.reply.assert_called_once()
    client.im.v1.message.create.assert_not_called()


@pytest.mark.asyncio
async def test_no_thread_id_uses_create_api():
    """When thread_id is absent, send_simple should use create API."""
    adapter = _make_adapter()
    adapter._lark_client = _mock_lark_client(message_id="msg_create_1")

    msg = OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        text="direct text",
    )
    result = await adapter.send_message(msg)

    assert result == "msg_create_1"
    adapter._lark_client.im.v1.message.create.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_reply_in_thread():
    """Streaming with thread_id should use reply API for the initial card."""
    adapter = _make_adapter()
    client = _mock_lark_client(message_id="msg_stream_reply", card_id="card_stream_reply")
    reply_resp = SimpleNamespace(
        success=lambda: True,
        code=0,
        msg="ok",
        data=SimpleNamespace(message_id="msg_stream_reply"),
    )
    client.im.v1.message.reply = MagicMock(return_value=reply_resp)
    adapter._lark_client = client

    msg = OutboundMessage(
        channel_id="test-feishu",
        conversation_id="oc_test123",
        thread_id="om_thread_2",
        text="stream reply",
        stream_id="stream_thread",
    )
    result = await adapter.send_message(msg)

    assert result == "msg_stream_reply"
    client.im.v1.message.reply.assert_called_once()
    client.im.v1.message.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: supports_streaming
# ---------------------------------------------------------------------------
class TestFeishuSupportsStreaming:
    """Feishu adapter should support streaming."""

    def test_supports_streaming_is_true(self):
        adapter = _make_adapter()
        assert adapter.supports_streaming is True
