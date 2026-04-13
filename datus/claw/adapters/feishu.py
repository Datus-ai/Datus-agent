# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Feishu (Lark) adapter using the official lark-oapi SDK long connection."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import threading
import uuid
from typing import TYPE_CHECKING, Optional

from datus.claw.channel.base import ChannelAdapter
from datus.claw.models import ChannelConfig, InboundMessage, OutboundMessage, ReactionEvent

if TYPE_CHECKING:
    from datus.claw.bridge import ChannelBridge
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Mapping from normalized (Slack-style) emoji names to Feishu emoji_type values.
# Feishu uses its own naming scheme (e.g. "DONE" instead of "white_check_mark").
# Reference: https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce
_EMOJI_TO_FEISHU: dict[str, str] = {
    "hourglass_flowing_sand": "OnIt",
    "white_check_mark": "DONE",
    "x": "Oppose",
    "thumbsup": "THUMBSUP",
    "thumbsdown": "THUMBSDOWN",
    "+1": "JIAYI",
}

# Reverse mapping: Feishu emoji_type -> normalized name (for inbound reaction events)
_FEISHU_TO_EMOJI: dict[str, str] = {v: k for k, v in _EMOJI_TO_FEISHU.items()}

# CardKit v2 streaming card template
_STREAMING_CARD_TEMPLATE: dict = {
    "schema": "2.0",
    "config": {
        "update_multi": True,
        "streaming_mode": True,
        "streaming_config": {
            "print_frequency_ms": {"default": 50},
            "print_step": {"default": 2},
            "print_strategy": "fast",
        },
    },
    "body": {
        "direction": "vertical",
        "padding": "12px 12px 12px 12px",
        "elements": [
            {"tag": "markdown", "content": "", "element_id": "content_md"},
        ],
    },
}


@dataclasses.dataclass
class _StreamState:
    """Per-stream state for a single streaming card session."""

    card_id: str
    msg_id: str
    seq: int = 1
    accumulated: str = ""


class FeishuAdapter(ChannelAdapter):
    """Feishu IM adapter using the official ``lark-oapi`` WebSocket long connection.

    Relies on ``lark_oapi.ws.Client`` which internally handles authentication,
    encryption, heartbeat, and reconnection.
    """

    def __init__(
        self,
        channel_id: str,
        config: dict,
        bridge: ChannelBridge,
        channel_config: Optional[ChannelConfig] = None,
    ) -> None:
        super().__init__(channel_id, config, bridge, channel_config)
        self._app_id: str = config.get("app_id", "")
        self._app_secret: str = config.get("app_secret", "")
        self._bot_open_id: Optional[str] = config.get("bot_open_id", "")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_client: Optional[object] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lark_client: Optional[object] = None
        # Cache reaction_id for remove_reaction: {(msg_id, emoji) -> reaction_id}
        self._processing_reactions: dict[tuple[str, str], str] = {}
        # Streaming card state — keyed by stream_id for concurrency safety
        self._streams: dict[str, _StreamState] = {}

    def _dispatch_to_loop(self, coro) -> None:
        """Schedule *coro* on the main event loop from a sync SDK callback thread.

        Safely handles cases where the loop is not yet set or already closed.
        """
        loop = self._loop
        if not loop or not loop.is_running():
            logger.warning("Main event loop not available; dropping dispatched coroutine.")
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.add_done_callback(self._dispatch_done_callback)
        except RuntimeError:
            logger.warning("Failed to dispatch coroutine: event loop is closed.")

    @staticmethod
    def _dispatch_done_callback(future) -> None:
        """Log exceptions from dispatched coroutines so they are not silently lost."""
        try:
            future.result()
        except Exception as e:
            logger.error("Dispatched coroutine raised an exception: %s", e)

    def _build_event_handler(self):
        """Build a lark_oapi EventDispatcherHandler that dispatches to our message handler."""
        import lark_oapi as lark

        def _on_receive(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
            """Synchronous callback invoked by the SDK in its internal thread."""
            event = data.event
            message = event.message
            sender = event.sender

            content_str = message.content or "{}"
            try:
                content = json.loads(content_str)
            except json.JSONDecodeError:
                content = {}

            raw_text = content.get("text", "").strip()
            chat_type = getattr(message, "chat_type", None)

            # Auto-learn bot_open_id from the first group @mention.
            # With im:message.group_at_msg:readonly, the bot only receives
            # messages that @mention it, so the mention whose open_id differs
            # from the sender is the bot itself.
            if chat_type == "group" and not self._bot_open_id and message.mentions:
                sender_oid = sender.sender_id.open_id if sender and sender.sender_id else ""
                for m in message.mentions:
                    m_oid = self._get_mention_open_id(m)
                    if m_oid and m_oid != sender_oid:
                        self._bot_open_id = m_oid
                        logger.info("Feishu bot_open_id learned from mention: %s", m_oid)
                        break

            mentions_bot = self._is_bot_mentioned(message.mentions)
            clean_text = self._strip_bot_mention(raw_text, message.mentions) if mentions_bot else raw_text

            msg = InboundMessage(
                channel_id=self.channel_id,
                sender_id=sender.sender_id.open_id if sender and sender.sender_id else "",
                sender_name="",
                conversation_id=message.chat_id or "",
                message_id=message.message_id or "",
                text=clean_text,
                mentions_bot=mentions_bot,
                chat_type=chat_type,
                thread_id=message.root_id,
                raw_payload=None,
            )

            self._dispatch_to_loop(self.dispatch_message(msg))

        def _on_reaction_created(data) -> None:
            """Handle reaction_created events from Feishu."""
            try:
                event = data.event
                message_id = event.message_id if hasattr(event, "message_id") else ""
                operator_id = ""
                if hasattr(event, "operator_id") and event.operator_id:
                    operator_id = event.operator_id.open_id if hasattr(event.operator_id, "open_id") else ""
                emoji_type = ""
                if hasattr(event, "reaction_type") and event.reaction_type:
                    emoji_type = getattr(event.reaction_type, "emoji_type", "")

                reaction_event = ReactionEvent(
                    channel_id=self.channel_id,
                    sender_id=operator_id,
                    conversation_id="",
                    target_message_id=message_id,
                    emoji=_FEISHU_TO_EMOJI.get(emoji_type, emoji_type),
                    action="added",
                )

                self._dispatch_to_loop(self.dispatch_reaction(reaction_event))
            except Exception as e:
                logger.warning("Failed to handle Feishu reaction event: %s", e)

        def _on_reaction_deleted(data) -> None:
            """Handle reaction_deleted events from Feishu."""
            try:
                event = data.event
                message_id = event.message_id if hasattr(event, "message_id") else ""
                operator_id = ""
                if hasattr(event, "operator_id") and event.operator_id:
                    operator_id = event.operator_id.open_id if hasattr(event.operator_id, "open_id") else ""
                emoji_type = ""
                if hasattr(event, "reaction_type") and event.reaction_type:
                    emoji_type = getattr(event.reaction_type, "emoji_type", "")

                reaction_event = ReactionEvent(
                    channel_id=self.channel_id,
                    sender_id=operator_id,
                    conversation_id="",
                    target_message_id=message_id,
                    emoji=_FEISHU_TO_EMOJI.get(emoji_type, emoji_type),
                    action="removed",
                )

                self._dispatch_to_loop(self.dispatch_reaction(reaction_event))
            except Exception as e:
                logger.warning("Failed to handle Feishu reaction deleted event: %s", e)

        builder = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(_on_receive)
        try:
            builder = builder.register_p2_im_message_reaction_created_v1(_on_reaction_created)
            builder = builder.register_p2_im_message_reaction_deleted_v1(_on_reaction_deleted)
        except AttributeError:
            logger.debug("Feishu SDK does not support reaction events; skipping registration.")
        handler = builder.build()
        return handler

    async def start(self) -> None:
        """Connect to Feishu via the official SDK long connection."""
        try:
            import lark_oapi as lark
        except ImportError as exc:
            raise ImportError(
                "lark-oapi package is required for the Feishu adapter. Install it with: pip install lark-oapi"
            ) from exc

        logger.info("Feishu adapter '%s' connecting...", self.channel_id)

        self._loop = asyncio.get_running_loop()

        self._lark_client = lark.Client.builder().app_id(self._app_id).app_secret(self._app_secret).build()

        # Build the event handler on the main thread (no loop dependency).
        # The ws.Client is created inside the worker thread so the SDK
        # initialises its internal event loop and asyncio.Lock in the
        # correct thread context — no monkey-patching required.
        event_handler = self._build_event_handler()

        self._ws_thread = threading.Thread(
            target=self._start_ws_in_new_loop,
            args=(event_handler,),
            name=f"feishu-ws-{self.channel_id}",
            daemon=True,
        )
        self._ws_thread.start()
        logger.info("Feishu adapter '%s' started.", self.channel_id)

    def _get_mention_open_id(self, mention) -> Optional[str]:
        """Extract open_id from a Feishu mention object."""
        if hasattr(mention, "id") and hasattr(mention.id, "open_id"):
            return mention.id.open_id
        return None

    def _is_bot_mentioned(self, mentions) -> bool:
        """Check whether the bot itself is mentioned (not just any user).

        On the first call with mentions, if bot_open_id is unknown, we learn it
        from the mention whose ``name`` matches the bot app name pattern.
        Until bot_open_id is resolved, any mention is treated as a bot mention.
        """
        if not mentions:
            return False
        if not self._bot_open_id:
            return bool(mentions)  # fallback: treat any mention as bot mention
        for m in mentions:
            if self._get_mention_open_id(m) == self._bot_open_id:
                return True
        return False

    def _strip_bot_mention(self, text: str, mentions) -> str:
        """Remove the bot's @mention placeholder (e.g. @_user_1) from the text."""
        if not mentions or not self._bot_open_id:
            return text
        for m in mentions:
            if self._get_mention_open_id(m) == self._bot_open_id and hasattr(m, "key"):
                text = text.replace(m.key, "").strip()
        return text

    def _start_ws_in_new_loop(self, event_handler) -> None:
        """Create the ws.Client and run it in a brand-new event loop on this thread.

        Creating the client here (rather than in ``start()``) ensures
        the SDK's internal ``asyncio.Lock`` is bound to this thread's
        event loop — no need to replace the private ``_lock`` attribute.

        We still set ``ws_module.loop`` because the SDK's ``start()``
        calls ``loop.run_until_complete()`` on that module-level
        reference instead of using ``asyncio.get_event_loop()``.
        """
        import lark_oapi as lark
        import lark_oapi.ws.client as ws_module

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        ws_module.loop = new_loop
        self._ws_loop = new_loop

        self._ws_client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        try:
            self._ws_client.start()
        except Exception as e:
            logger.error("Feishu ws client exited with error: %s", e)
        finally:
            new_loop.close()

    async def stop(self) -> None:
        """Stop the Feishu WebSocket client and clean up resources."""
        # Stop the WS event loop running in the daemon thread so it exits
        # gracefully instead of lingering until process termination.
        ws_loop = self._ws_loop
        if ws_loop and ws_loop.is_running():
            ws_loop.call_soon_threadsafe(ws_loop.stop)

        # Wait briefly for the daemon thread to finish
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3.0)
            if self._ws_thread.is_alive():
                logger.warning("Feishu WS thread for '%s' did not exit within timeout.", self.channel_id)

        self._processing_reactions.clear()
        self._streams.clear()
        self._ws_client = None
        self._lark_client = None
        self._ws_thread = None
        self._ws_loop = None
        self._loop = None
        logger.info("Feishu adapter '%s' stopped.", self.channel_id)

    async def send_message(self, message: OutboundMessage) -> Optional[str]:
        """Send a reply message via the Feishu OpenAPI (lark-oapi SDK). Returns the message ID."""
        if message.stream_id:
            return await self._send_streaming(message)
        return await self._send_simple(message)

    async def _send_simple(self, message: OutboundMessage) -> Optional[str]:
        """Send a standalone interactive card (no streaming)."""
        try:
            text = message.text
            if message.sql:
                text += f"\n\n```sql\n{message.sql}\n```"

            card = {
                "config": {"wide_screen_mode": True},
                "elements": [{"tag": "markdown", "content": text}],
            }

            if not self._lark_client:
                logger.error("Feishu lark client not initialized for '%s'", self.channel_id)
                return None

            loop = asyncio.get_running_loop()

            if message.thread_id:
                from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

                body = ReplyMessageRequestBody.builder().msg_type("interactive").content(json.dumps(card)).build()
                request = ReplyMessageRequest.builder().message_id(message.thread_id).request_body(body).build()
                response = await loop.run_in_executor(
                    None,
                    self._lark_client.im.v1.message.reply,
                    request,
                )
            else:
                from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

                body = (
                    CreateMessageRequestBody.builder()
                    .receive_id(message.conversation_id)
                    .msg_type("interactive")
                    .content(json.dumps(card))
                    .build()
                )
                request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
                response = await loop.run_in_executor(
                    None,
                    self._lark_client.im.v1.message.create,
                    request,
                )

            if not response.success():
                logger.error(
                    "Failed to send Feishu message (code=%s): %s",
                    response.code,
                    response.msg,
                )
                return None
            logger.debug("Feishu message sent to conversation %s", message.conversation_id)
            if response.data and hasattr(response.data, "message_id"):
                return response.data.message_id
            return None
        except Exception as e:
            logger.error("Failed to send Feishu message: %s", e)
            return None

    async def _send_streaming(self, message: OutboundMessage) -> Optional[str]:
        """Send or update a streaming card for incremental content delivery."""
        text = message.text
        if message.sql:
            text += f"\n\n```sql\n{message.sql}\n```"

        stream_id = message.stream_id
        state = self._streams.get(stream_id)
        if state is None:
            # First message in this stream — create the streaming card
            return await self._start_stream(message, text)
        else:
            # Subsequent message — append content to existing card
            state.accumulated += f"\n\n{text}"
            await self._update_card_content(stream_id)
            return None

    async def _start_stream(self, message: OutboundMessage, text: str) -> Optional[str]:
        """Create a new streaming card and send the initial content."""
        try:
            card = copy.deepcopy(_STREAMING_CARD_TEMPLATE)
            card["body"]["elements"][0]["content"] = text

            if not self._lark_client:
                logger.error("Feishu lark client not initialized for '%s'", self.channel_id)
                return None

            loop = asyncio.get_running_loop()

            if message.thread_id:
                from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

                body = ReplyMessageRequestBody.builder().msg_type("interactive").content(json.dumps(card)).build()
                request = ReplyMessageRequest.builder().message_id(message.thread_id).request_body(body).build()
                response = await loop.run_in_executor(
                    None,
                    self._lark_client.im.v1.message.reply,
                    request,
                )
            else:
                from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

                body = (
                    CreateMessageRequestBody.builder()
                    .receive_id(message.conversation_id)
                    .msg_type("interactive")
                    .content(json.dumps(card))
                    .build()
                )
                request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
                response = await loop.run_in_executor(
                    None,
                    self._lark_client.im.v1.message.create,
                    request,
                )
            if not response.success():
                logger.error(
                    "Failed to send Feishu streaming card (code=%s): %s",
                    response.code,
                    response.msg,
                )
                return None

            msg_id = response.data.message_id if response.data and hasattr(response.data, "message_id") else None
            if not msg_id:
                return None

            # Convert message_id to card_id via CardKit
            card_id = await self._convert_to_card_id(msg_id)
            if not card_id:
                logger.warning("Failed to convert message_id to card_id; falling back to non-streaming")
                return msg_id

            self._streams[message.stream_id] = _StreamState(
                card_id=card_id,
                msg_id=msg_id,
                seq=1,
                accumulated=text,
            )

            logger.debug("Feishu streaming card created: card_id=%s, message_id=%s", card_id, msg_id)
            return msg_id
        except Exception as e:
            logger.error("Failed to create Feishu streaming card: %s", e)
            return None

    async def _convert_to_card_id(self, message_id: str) -> Optional[str]:
        """Convert a Feishu message_id to a CardKit card_id."""
        try:
            from lark_oapi.api.cardkit.v1 import IdConvertCardRequest, IdConvertCardRequestBody

            body = IdConvertCardRequestBody.builder().message_id(message_id).build()
            request = IdConvertCardRequest.builder().request_body(body).build()

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._lark_client.cardkit.v1.card.id_convert,
                request,
            )
            if not response.success():
                logger.error(
                    "Failed to convert message_id to card_id (code=%s): %s",
                    response.code,
                    response.msg,
                )
                return None
            if response.data and hasattr(response.data, "card_id"):
                return response.data.card_id
            return None
        except Exception as e:
            logger.error("Failed to convert message_id to card_id: %s", e)
            return None

    async def _update_card_content(self, stream_id: str) -> None:
        """Push accumulated content to the streaming card element."""
        state = self._streams.get(stream_id)
        if not state or not self._lark_client:
            return
        try:
            from lark_oapi.api.cardkit.v1 import ContentCardElementRequest, ContentCardElementRequestBody

            body = (
                ContentCardElementRequestBody.builder()
                .uuid(str(uuid.uuid4()))
                .content(state.accumulated)
                .sequence(state.seq)
                .build()
            )
            request = (
                ContentCardElementRequest.builder()
                .card_id(state.card_id)
                .element_id("content_md")
                .request_body(body)
                .build()
            )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._lark_client.cardkit.v1.card_element.content,
                request,
            )
            state.seq += 1
            if not response.success():
                logger.warning(
                    "Failed to update streaming card content (code=%s): %s",
                    response.code,
                    response.msg,
                )
        except Exception as e:
            logger.warning("Failed to update streaming card content: %s", e)

    async def finalize_stream(self, stream_id: str) -> None:
        """Disable streaming_mode on the card identified by *stream_id* and clean up."""
        state = self._streams.get(stream_id)
        if not state or not self._lark_client:
            self._streams.pop(stream_id, None)
            return
        try:
            from lark_oapi.api.cardkit.v1 import SettingsCardRequest, SettingsCardRequestBody

            settings = json.dumps({"config": {"streaming_mode": False}})
            body = (
                SettingsCardRequestBody.builder().settings(settings).uuid(str(uuid.uuid4())).sequence(state.seq).build()
            )
            request = SettingsCardRequest.builder().card_id(state.card_id).request_body(body).build()

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._lark_client.cardkit.v1.card.settings,
                request,
            )
            if not response.success():
                logger.warning(
                    "Failed to finalize streaming card (code=%s): %s",
                    response.code,
                    response.msg,
                )
        except Exception as e:
            logger.warning("Failed to finalize streaming card: %s", e)
        finally:
            self._streams.pop(stream_id, None)

    async def add_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Add an emoji reaction to a Feishu message."""
        if not self._lark_client:
            return
        feishu_emoji = _EMOJI_TO_FEISHU.get(emoji, emoji)
        try:
            from lark_oapi.api.im.v1 import CreateMessageReactionRequest, CreateMessageReactionRequestBody, Emoji

            body = (
                CreateMessageReactionRequestBody.builder()
                .reaction_type(Emoji.builder().emoji_type(feishu_emoji).build())
                .build()
            )
            request = CreateMessageReactionRequest.builder().message_id(message_id).request_body(body).build()

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self._lark_client.im.v1.message_reaction.create, request)
            if response.success() and response.data:
                reaction_id = getattr(response.data, "reaction_id", None)
                if reaction_id:
                    self._processing_reactions[(message_id, emoji)] = reaction_id
            else:
                logger.warning(
                    "Failed to add Feishu reaction '%s' (feishu='%s'): code=%s %s",
                    emoji,
                    feishu_emoji,
                    response.code,
                    response.msg,
                )
        except Exception as e:
            logger.warning("Failed to add Feishu reaction '%s': %s", emoji, e)

    async def remove_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Remove an emoji reaction from a Feishu message."""
        reaction_id = self._processing_reactions.pop((message_id, emoji), None)
        if not reaction_id or not self._lark_client:
            return
        try:
            from lark_oapi.api.im.v1 import DeleteMessageReactionRequest

            request = DeleteMessageReactionRequest.builder().message_id(message_id).reaction_id(reaction_id).build()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self._lark_client.im.v1.message_reaction.delete, request)
            if not response.success():
                logger.warning("Failed to remove Feishu reaction '%s': code=%s %s", emoji, response.code, response.msg)
        except Exception as e:
            logger.warning("Failed to remove Feishu reaction '%s': %s", emoji, e)
