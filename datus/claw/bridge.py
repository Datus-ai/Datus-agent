# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""ChannelBridge — routes IM messages through ChatTaskManager and sends responses back."""

import asyncio
from collections import OrderedDict
from typing import Dict, Optional

from datus.api.models.cli_models import SSEDataType, StreamChatInput
from datus.api.services.chat_task_manager import ChatTaskManager, is_thinking_only_content
from datus.claw.channel.base import ChannelAdapter
from datus.claw.commands import CommandContext, match_command, register_builtin_commands
from datus.claw.formatters import ToolOutputFormatter
from datus.claw.models import (
    DONE_EMOJI,
    ERROR_EMOJI,
    PROCESSING_EMOJI,
    ChannelConfig,
    InboundMessage,
    OutboundMessage,
    ReactionEvent,
    Verbose,
)
from datus.claw.richtext.parser import markdown_to_ir
from datus.configuration.agent_config import AgentConfig
from datus.models.session_manager import SessionManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Content types that should be streamed back to the IM channel
_STREAMABLE_TYPES = {"markdown", "code", "thinking", "error", "call-tool", "call-tool-result"}

# Maximum number of bot message IDs to track
_MAX_BOT_MSG_MAP = 10000


class ChannelBridge:
    """Converts InboundMessage -> ChatTaskManager -> OutboundMessage.

    Each SSE event is forwarded to the IM channel as it arrives, giving
    the user real-time feedback (thinking, tool calls, results, etc.).
    """

    def __init__(self, agent_config: AgentConfig, task_manager: ChatTaskManager) -> None:
        register_builtin_commands()
        self._agent_config = agent_config
        self._task_manager = task_manager
        self._formatter = ToolOutputFormatter()
        # Lock protects mutable state accessed from concurrent handle_message calls.
        self._lock = asyncio.Lock()
        # Per-conversation verbose override set via /verbose command
        self._verbose_overrides: Dict[str, Verbose] = {}
        # Track bot message IDs for reaction feedback: {bot_msg_id -> context}
        self._bot_message_map: OrderedDict[str, dict] = OrderedDict()
        # Deduplication: recently seen message IDs
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._max_seen_ids: int = 1000

    def _conversation_key(self, msg: InboundMessage) -> str:
        """Stable key for a conversation (before session suffix)."""
        parts = [msg.channel_id, msg.conversation_id]
        if msg.thread_id:
            parts.append(msg.thread_id)
        return "--".join(parts)

    def build_session_id(self, msg: InboundMessage) -> str:
        """Deterministic session id from channel + conversation (+ thread)."""
        conv_key = self._conversation_key(msg)
        return f"claw_{conv_key}"

    def clear_session(self, msg: InboundMessage) -> None:
        """Clear the conversation history for the current session."""
        session_id = self.build_session_id(msg)
        session_mgr = SessionManager(session_dir=self._agent_config.session_dir)
        if session_mgr.session_exists(session_id):
            session_mgr.clear_session(session_id)

    def set_verbose(self, msg: InboundMessage, level: Verbose) -> None:
        """Override the verbose level for the conversation that *msg* belongs to."""
        self._verbose_overrides[self._conversation_key(msg)] = level

    def get_verbose(self, msg: InboundMessage, channel_config: Optional[ChannelConfig] = None) -> Verbose:
        """Resolve verbose level: per-conversation override > channel config > default ON."""
        override = self._verbose_overrides.get(self._conversation_key(msg))
        if override is not None:
            return override
        if channel_config:
            return channel_config.verbose
        return Verbose.ON

    async def handle_message(
        self,
        msg: InboundMessage,
        adapter: ChannelAdapter,
        channel_config: Optional[ChannelConfig] = None,
    ) -> None:
        """Process an inbound IM message end-to-end, streaming each action back."""
        # Deduplicate: IM platforms may retry delivery of the same message
        dedup_key = f"{msg.channel_id}_{msg.message_id}"
        async with self._lock:
            if dedup_key in self._seen_message_ids:
                logger.debug("Ignoring duplicate message %s", msg.message_id)
                return
            self._seen_message_ids[dedup_key] = None
            while len(self._seen_message_ids) > self._max_seen_ids:
                self._seen_message_ids.popitem(last=False)

        # In group chats, only respond to @bot messages or thread replies
        if msg.chat_type == "group" and not msg.mentions_bot and not msg.thread_id:
            logger.debug("Ignoring non-mention group message in %s", msg.conversation_id)
            return

        # In group chats without an existing thread, use message_id as thread_id
        # so each @bot message gets its own thread (independent session)
        if msg.chat_type == "group" and not msg.thread_id:
            msg = msg.model_copy(update={"thread_id": msg.message_id})

        # Handle chat commands (e.g. /new, /reset, /verbose)
        match = match_command(msg.text)
        if match is not None:
            ctx = CommandContext(msg=msg, adapter=adapter, bridge=self, args=match.args)
            await match.command.execute(ctx)
            return

        # Add processing reaction to the user's message
        try:
            await adapter.add_reaction(msg.conversation_id, msg.message_id, PROCESSING_EMOJI, msg.thread_id)
        except Exception as e:
            logger.warning("Failed to add processing reaction: %s", e)

        session_id = self.build_session_id(msg)
        subagent_id = channel_config.subagent_id if channel_config else None

        request = StreamChatInput(
            message=msg.text,
            session_id=session_id,
            source="claw",
        )

        # Apply channel-level namespace override
        if channel_config and channel_config.namespace:
            request.database = channel_config.namespace

        has_error = False
        # Start the agentic loop
        try:
            task = await self._task_manager.start_chat(
                self._agent_config,
                request,
                sub_agent_id=subagent_id,
            )
        except ValueError as exc:
            if "still being processed" not in str(exc) and "already running" not in str(exc):
                raise
            busy_reply = OutboundMessage(
                channel_id=msg.channel_id,
                conversation_id=msg.conversation_id,
                thread_id=msg.thread_id,
                text="The previous request is still being processed. Please wait.",
            )
            await adapter.send_message(busy_reply)
            has_error = True
            task = None

        verbose = self.get_verbose(msg, channel_config)
        stream_id = f"{msg.channel_id}_{msg.message_id}"
        # Per-request pending tool calls to avoid cross-message contamination
        pending_tool_calls: Dict[str, dict] = {}

        try:
            # Stream each SSE event to the IM channel as it arrives
            any_sent = task is None  # Busy path already sent a reply
            if task is not None:
                async for event in self._task_manager.consume_events(task):
                    if event.event == "error":
                        error_text = ""
                        if hasattr(event.data, "error"):
                            error_text = event.data.error
                        error_reply = OutboundMessage(
                            channel_id=msg.channel_id,
                            conversation_id=msg.conversation_id,
                            thread_id=msg.thread_id,
                            text=f"\u274c Error: {error_text}" if error_text else "\u274c An error occurred.",
                        )
                        await adapter.send_message(error_reply)
                        has_error = True
                        any_sent = True
                    elif event.event == "message":
                        outbound = self._event_to_outbound(event.data, msg, verbose, pending_tool_calls)
                        if outbound:
                            stream_enabled = adapter.supports_streaming and (
                                not channel_config or channel_config.stream_response
                            )
                            if stream_enabled:
                                outbound.stream_id = stream_id
                            elif outbound.is_delta:
                                continue
                            bot_msg_id = await adapter.send_message(outbound)
                            if bot_msg_id:
                                await self._track_bot_message(bot_msg_id, session_id, msg)
                            any_sent = True

            if not any_sent:
                fallback = OutboundMessage(
                    channel_id=msg.channel_id,
                    conversation_id=msg.conversation_id,
                    thread_id=msg.thread_id,
                    text="(No response generated)",
                )
                bot_msg_id = await adapter.send_message(fallback)
                if bot_msg_id:
                    await self._track_bot_message(bot_msg_id, session_id, msg)
        except Exception:
            has_error = True
            raise
        finally:
            # Finalize card stream (no-op for adapters that don't support streaming)
            try:
                await adapter.finalize_stream(stream_id)
            except Exception as e:
                logger.debug("Failed to finalize stream: %s", e)
            try:
                await adapter.remove_reaction(msg.conversation_id, msg.message_id, PROCESSING_EMOJI, msg.thread_id)
            except Exception as e:
                logger.debug("Failed to remove processing reaction: %s", e)
            done_emoji = ERROR_EMOJI if has_error else DONE_EMOJI
            try:
                await adapter.add_reaction(msg.conversation_id, msg.message_id, done_emoji, msg.thread_id)
            except Exception as e:
                logger.debug("Failed to add done reaction: %s", e)

    async def _track_bot_message(self, bot_msg_id: str, session_id: str, msg: InboundMessage) -> None:
        """Record a bot message ID for future reaction feedback tracking."""
        async with self._lock:
            self._bot_message_map[bot_msg_id] = {
                "session_id": session_id,
                "channel_id": msg.channel_id,
                "conversation_id": msg.conversation_id,
                "sender_id": msg.sender_id,
            }
            # Evict oldest entries if over limit
            while len(self._bot_message_map) > _MAX_BOT_MSG_MAP:
                self._bot_message_map.popitem(last=False)

    async def handle_reaction(self, event: ReactionEvent, adapter: ChannelAdapter) -> None:
        """Process a reaction event on a bot message (feedback tracking)."""
        context = self._bot_message_map.get(event.target_message_id)
        if not context:
            return
        logger.info(
            "Feedback: emoji=%s sender=%s session=%s msg=%s",
            event.emoji,
            event.sender_id,
            context.get("session_id"),
            event.target_message_id,
        )

    def _event_to_outbound(
        self,
        data,
        msg: InboundMessage,
        verbose: Verbose = Verbose.ON,
        pending_tool_calls: Optional[Dict[str, dict]] = None,
    ) -> Optional[OutboundMessage]:
        """Convert a single SSE message event to an OutboundMessage.

        Returns None if the event contains no displayable content.
        """
        if pending_tool_calls is None:
            pending_tool_calls = {}

        if not hasattr(data, "type") or not hasattr(data, "payload"):
            return None
        if data.type not in (SSEDataType.CREATE_MESSAGE, SSEDataType.APPEND_MESSAGE, SSEDataType.UPDATE_MESSAGE):
            return None

        text_parts: list[str] = []
        sql: Optional[str] = None

        for item in getattr(data.payload, "content", []):
            content_type = getattr(item, "type", "")
            payload = item.payload if hasattr(item, "payload") else {}
            if not isinstance(payload, dict):
                continue

            if content_type not in _STREAMABLE_TYPES:
                continue

            if content_type == "code" and payload.get("codeType") == "sql":
                sql = payload.get("content", "")
            elif content_type == "call-tool":
                if verbose != Verbose.OFF:
                    call_tool_id = payload.get("callToolId", "")
                    pending_tool_calls[call_tool_id] = payload
                # Never output call-tool directly; wait for call-tool-result
            elif content_type == "call-tool-result":
                if verbose != Verbose.OFF:
                    call_tool_id = payload.get("callToolId", "")
                    call_payload = pending_tool_calls.pop(call_tool_id, {})
                    formatted = self._formatter.format_tool_complete(call_payload, payload, verbose)
                    if formatted:
                        text_parts.append(formatted)
            elif content_type == "thinking":
                md_text = payload.get("content", "")
                if md_text:
                    text_parts.append(md_text)
            elif content_type == "markdown":
                md_text = payload.get("content", "")
                if md_text:
                    text_parts.append(md_text)
            elif content_type == "error":
                error_text = payload.get("content", "Unknown error")
                text_parts.append(f"\u274c Error: {error_text}")
            elif content_type == "code":
                code_text = payload.get("content", "")
                code_type = payload.get("codeType", "")
                if code_text:
                    text_parts.append(f"```{code_type}\n{code_text}\n```")

        combined_text = "\n".join(text_parts).strip()
        if not combined_text and not sql:
            return None

        # Detect delta messages: events whose content is only thinking chunks.
        # Delta text is partial and cannot be parsed into rich-text IR.
        content_items = getattr(data.payload, "content", [])
        is_delta = is_thinking_only_content(content_items)

        ir = None
        if combined_text and not is_delta:
            ir = markdown_to_ir(combined_text, heading_style="bold", table_mode="bullets")

        return OutboundMessage(
            channel_id=msg.channel_id,
            conversation_id=msg.conversation_id,
            thread_id=msg.thread_id,
            text=combined_text,
            ir=ir,
            sql=sql,
            is_delta=is_delta,
        )
