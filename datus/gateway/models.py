# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Pydantic models for the Datus Gateway IM gateway."""

from enum import Enum
from typing import Dict, Optional, Set

from pydantic import BaseModel, Field

from datus.gateway.richtext.ir import MarkdownIR


class Verbose(str, Enum):
    """Output verbosity level for IM channels.

    - OFF: Only final output (markdown, code, error).
    - ON: Brief output including thinking and tool call summaries.
    - FULL: Detailed output including tool parameters and full results.
    """

    OFF = "quiet"
    ON = "brief"
    FULL = "detail"


# Emoji constants (normalized names used by Slack / Feishu)
PROCESSING_EMOJI = "hourglass_flowing_sand"
DONE_EMOJI = "white_check_mark"
ERROR_EMOJI = "x"
FEEDBACK_POSITIVE: Set[str] = {"thumbsup", "+1", "THUMBSUP"}
FEEDBACK_NEGATIVE: Set[str] = {"thumbsdown", "-1", "THUMBSDOWN"}


class InboundMessage(BaseModel):
    """Unified inbound message from an IM platform."""

    channel_id: str = Field(..., description="Adapter instance identifier")
    sender_id: str = Field(..., description="IM platform user ID")
    sender_name: str = Field("", description="Display name")
    conversation_id: str = Field(..., description="IM group/DM/channel ID")
    message_id: str = Field(..., description="IM platform message ID")
    text: str = Field(..., description="Plain text content")
    mentions_bot: bool = Field(False, description="Whether the message mentions the bot")
    chat_type: Optional[str] = Field(None, description="Chat type: 'group' or 'p2p'")
    thread_id: Optional[str] = Field(None, description="Thread ID for threaded conversations")
    raw_payload: Optional[dict] = Field(None, description="Raw platform-specific payload")


class OutboundMessage(BaseModel):
    """Unified outbound message to an IM platform."""

    channel_id: str = Field(..., description="Adapter instance identifier")
    conversation_id: str = Field(..., description="IM group/DM/channel ID")
    thread_id: Optional[str] = Field(None, description="Thread ID for threaded replies")
    text: str = Field("", description="Markdown/plain text response")
    ir: Optional[MarkdownIR] = Field(None, description="Structured IR for platform-specific rendering")
    sql: Optional[str] = Field(None, description="Generated SQL code block")
    stream_id: Optional[str] = Field(None, description="Groups outbound messages from the same request into one stream")
    is_delta: bool = Field(False, description="True for token-level delta chunks that should be concatenated directly")


class ReactionEvent(BaseModel):
    """A reaction added/removed on a message."""

    channel_id: str = Field(..., description="Adapter instance identifier")
    sender_id: str = Field(..., description="IM platform user ID")
    conversation_id: str = Field(..., description="IM group/DM/channel ID")
    target_message_id: str = Field(..., description="Message the reaction is on")
    emoji: str = Field(..., description="Normalized emoji name")
    action: str = Field("added", description="'added' or 'removed'")
    thread_id: Optional[str] = Field(None, description="Thread ID if applicable")


class ChannelConfig(BaseModel):
    """Configuration for a single IM channel adapter."""

    adapter: str = Field(..., description="Adapter type: feishu, slack")
    enabled: bool = Field(True, description="Whether this channel is enabled")
    datasource: Optional[str] = Field(None, description="Override default datasource")
    subagent_id: Optional[str] = Field(None, description="Route to a specific sub-agent")
    verbose: Verbose = Field(Verbose.ON, description="Output verbosity level: quiet, brief, detail")
    stream_response: bool = Field(True, description="Enable token-level streaming for chat responses")
    extra: Dict = Field(default_factory=dict, description="Adapter-specific config (tokens, app_id, etc.)")
