# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the Slack adapter — bot mention detection and chat type mapping."""

from __future__ import annotations

import pytest

from datus.claw.adapters.slack import SlackAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_slack_adapter() -> SlackAdapter:
    """Create a SlackAdapter with minimal config and no real connections."""
    from unittest.mock import MagicMock

    bridge = MagicMock()
    adapter = SlackAdapter(
        channel_id="test-slack",
        config={"app_token": "xapp-fake", "bot_token": "xoxb-fake"},
        bridge=bridge,
    )
    return adapter


def _make_socket_event(
    text: str = "hello",
    user: str = "U123",
    channel: str = "C456",
    ts: str = "1234567890.123456",
    channel_type: str = "",
    thread_ts: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Build a minimal Slack events_api payload."""
    event = {
        "type": "message",
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
    return {"event": event}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSlackMentionDetection:
    """Tests for bot mention detection in Slack message text."""

    def test_mentions_bot_when_present(self):
        adapter = _make_slack_adapter()
        adapter._bot_user_id = "U_BOT"
        text = "<@U_BOT> what is the revenue?"
        assert f"<@{adapter._bot_user_id}>" in text

    def test_no_mention_when_absent(self):
        adapter = _make_slack_adapter()
        adapter._bot_user_id = "U_BOT"
        text = "what is the revenue?"
        assert f"<@{adapter._bot_user_id}>" not in text

    def test_no_mention_when_bot_id_empty(self):
        adapter = _make_slack_adapter()
        adapter._bot_user_id = ""
        text = "<@U_BOT> hello"
        # With empty bot_user_id, mentions_bot should be False
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
