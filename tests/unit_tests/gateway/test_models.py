# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.gateway.models."""

import pytest
from pydantic import ValidationError

from datus.gateway.models import (
    DONE_EMOJI,
    ERROR_EMOJI,
    FEEDBACK_NEGATIVE,
    FEEDBACK_POSITIVE,
    PROCESSING_EMOJI,
    ChannelConfig,
    InboundMessage,
    OutboundMessage,
    ReactionEvent,
    Verbose,
)


class TestInboundMessage:
    """Tests for InboundMessage serialization and defaults."""

    def test_minimal_construction(self):
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            conversation_id="conv1",
            message_id="msg1",
            text="hello",
        )
        assert msg.channel_id == "ch1"
        assert msg.sender_name == ""
        assert msg.mentions_bot is False
        assert msg.thread_id is None
        assert msg.raw_payload is None

    def test_full_construction(self):
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            sender_name="Alice",
            conversation_id="conv1",
            message_id="msg1",
            text="@bot what is the revenue?",
            mentions_bot=True,
            thread_id="thread_42",
            raw_payload={"key": "value"},
        )
        assert msg.sender_name == "Alice"
        assert msg.mentions_bot is True
        assert msg.thread_id == "thread_42"
        assert msg.raw_payload == {"key": "value"}

    def test_serialization_roundtrip(self):
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="u",
            conversation_id="c",
            message_id="m",
            text="hi",
        )
        data = msg.model_dump()
        restored = InboundMessage(**data)
        assert restored == msg

    def test_json_roundtrip(self):
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="u",
            conversation_id="c",
            message_id="m",
            text="hi",
            raw_payload={"nested": {"a": 1}},
        )
        json_str = msg.model_dump_json()
        restored = InboundMessage.model_validate_json(json_str)
        assert restored == msg


class TestOutboundMessage:
    """Tests for OutboundMessage serialization and defaults."""

    def test_minimal_construction(self):
        msg = OutboundMessage(channel_id="ch1", conversation_id="conv1")
        assert msg.text == ""
        assert msg.sql is None
        assert msg.thread_id is None
        assert msg.is_delta is False

    def test_full_construction(self):
        msg = OutboundMessage(
            channel_id="ch1",
            conversation_id="conv1",
            thread_id="t1",
            text="Here is your result",
            sql="SELECT 1",
            is_delta=True,
        )
        assert msg.sql == "SELECT 1"
        assert msg.thread_id == "t1"
        assert msg.is_delta is True

    def test_serialization_roundtrip(self):
        msg = OutboundMessage(
            channel_id="ch1",
            conversation_id="conv1",
            text="resp",
            sql="SELECT 1",
        )
        data = msg.model_dump()
        restored = OutboundMessage(**data)
        assert restored == msg


class TestChannelConfig:
    """Tests for ChannelConfig serialization and defaults."""

    def test_minimal_construction(self):
        cfg = ChannelConfig(adapter="feishu")
        assert cfg.enabled is True
        assert cfg.namespace is None
        assert cfg.subagent_id is None
        assert cfg.verbose == Verbose.ON
        assert cfg.stream_response is True
        assert cfg.extra == {}

    def test_full_construction(self):
        cfg = ChannelConfig(
            adapter="slack",
            enabled=False,
            namespace="production",
            subagent_id="agent_42",
            extra={"app_token": "test-app-token", "bot_token": "test-bot-token"},
        )
        assert cfg.adapter == "slack"
        assert cfg.enabled is False
        assert cfg.namespace == "production"
        assert cfg.extra["app_token"] == "test-app-token"

    def test_serialization_roundtrip(self):
        cfg = ChannelConfig(adapter="feishu", extra={"app_id": "id123"})
        data = cfg.model_dump()
        restored = ChannelConfig(**data)
        assert restored == cfg

    @pytest.mark.parametrize("adapter_name", ["feishu", "slack"])
    def test_valid_adapter_names(self, adapter_name):
        cfg = ChannelConfig(adapter=adapter_name)
        assert cfg.adapter == adapter_name

    def test_disabled_channel(self):
        cfg = ChannelConfig(adapter="feishu", enabled=False)
        assert cfg.enabled is False

    def test_stream_response_disabled(self):
        cfg = ChannelConfig(adapter="feishu", stream_response=False)
        assert cfg.stream_response is False


class TestReactionEvent:
    """Tests for ReactionEvent model."""

    def test_minimal_construction(self):
        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user1",
            conversation_id="conv1",
            target_message_id="msg1",
            emoji="thumbsup",
        )
        assert event.action == "added"
        assert event.thread_id is None

    def test_full_construction(self):
        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user1",
            conversation_id="conv1",
            target_message_id="msg1",
            emoji="thumbsup",
            action="removed",
            thread_id="thread_1",
        )
        assert event.action == "removed"
        assert event.thread_id == "thread_1"

    def test_serialization_roundtrip(self):
        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user1",
            conversation_id="conv1",
            target_message_id="msg1",
            emoji="hourglass_flowing_sand",
            action="added",
        )
        data = event.model_dump()
        restored = ReactionEvent(**data)
        assert restored == event

    def test_json_roundtrip(self):
        event = ReactionEvent(
            channel_id="ch1",
            sender_id="user1",
            conversation_id="conv1",
            target_message_id="msg1",
            emoji="white_check_mark",
        )
        json_str = event.model_dump_json()
        restored = ReactionEvent.model_validate_json(json_str)
        assert restored == event


class TestEmojiConstants:
    """Tests for emoji constants."""

    def test_processing_emoji(self):
        assert PROCESSING_EMOJI == "hourglass_flowing_sand"

    def test_done_emoji(self):
        assert DONE_EMOJI == "white_check_mark"

    def test_error_emoji(self):
        assert ERROR_EMOJI == "x"

    def test_feedback_positive(self):
        assert "thumbsup" in FEEDBACK_POSITIVE
        assert "+1" in FEEDBACK_POSITIVE
        assert "THUMBSUP" in FEEDBACK_POSITIVE

    def test_feedback_negative(self):
        assert "thumbsdown" in FEEDBACK_NEGATIVE
        assert "-1" in FEEDBACK_NEGATIVE
        assert "THUMBSDOWN" in FEEDBACK_NEGATIVE


class TestVerbose:
    """Tests for Verbose enum."""

    def test_default_value_is_on(self):
        cfg = ChannelConfig(adapter="feishu")
        assert cfg.verbose == Verbose.ON

    @pytest.mark.parametrize("value", ["quiet", "brief", "detail"])
    def test_valid_string_parsing(self, value):
        cfg = ChannelConfig(adapter="feishu", verbose=value)
        assert cfg.verbose == Verbose(value)

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            ChannelConfig(adapter="feishu", verbose="invalid")

    def test_enum_values(self):
        assert Verbose.OFF == "quiet"
        assert Verbose.ON == "brief"
        assert Verbose.FULL == "detail"

    def test_verbose_in_serialization(self):
        cfg = ChannelConfig(adapter="slack", verbose=Verbose.FULL)
        data = cfg.model_dump()
        assert data["verbose"] == "detail"
        restored = ChannelConfig(**data)
        assert restored.verbose == Verbose.FULL
