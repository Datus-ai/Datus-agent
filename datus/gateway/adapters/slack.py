# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Slack Socket Mode adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.models import ChannelConfig, InboundMessage, OutboundMessage, ReactionEvent
from datus.gateway.richtext.chunker import chunk_text
from datus.gateway.richtext.escape import slack_escape
from datus.gateway.richtext.ir import MarkdownIR, StyleType
from datus.gateway.richtext.render import RenderOptions, StyleMarker, render_ir

if TYPE_CHECKING:
    from datus.gateway.bridge import ChannelBridge
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_SLACK_STYLES: dict[StyleType, StyleMarker] = {
    StyleType.BOLD: StyleMarker("*", "*"),
    StyleType.ITALIC: StyleMarker("_", "_"),
    StyleType.STRIKETHROUGH: StyleMarker("~", "~"),
    StyleType.CODE: StyleMarker("`", "`"),
    StyleType.CODE_BLOCK: StyleMarker("```\n", "\n```"),
    StyleType.BLOCKQUOTE: StyleMarker("> ", "\n"),
}


class SlackAdapter(ChannelAdapter):
    """Slack IM adapter using Socket Mode (long connection)."""

    def __init__(
        self,
        channel_id: str,
        config: dict,
        bridge: ChannelBridge,
        channel_config: Optional[ChannelConfig] = None,
    ) -> None:
        super().__init__(channel_id, config, bridge, channel_config)
        self._app_token: str = config.get("app_token", "")
        self._bot_token: str = config.get("bot_token", "")
        self._bot_user_id: str = ""
        self._socket_client: Optional[object] = None
        self._web_client: Optional[object] = None
        self._listen_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Connect via Slack Socket Mode."""
        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as exc:
            raise ImportError(
                "slack_sdk is required for the Slack adapter. Install it with: pip install slack-sdk[socket_mode]"
            ) from exc

        logger.info("Slack adapter '%s' connecting...", self.channel_id)

        self._web_client = AsyncWebClient(token=self._bot_token)

        try:
            auth_resp = await self._web_client.auth_test()
            data = auth_resp.data if hasattr(auth_resp, "data") else auth_resp
            self._bot_user_id = data.get("user_id", "") if isinstance(data, dict) else ""
            logger.info("Slack bot user ID: %s", self._bot_user_id)
        except Exception as e:
            logger.warning("Failed to get Slack bot user ID: %s", e)

        self._socket_client = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        self._socket_client.socket_mode_request_listeners.append(self._handle_socket_request)

        self._listen_task = asyncio.create_task(self._socket_client.connect())
        logger.info("Slack adapter '%s' started.", self.channel_id)

    async def _handle_socket_request(self, client, req) -> None:
        """Handle incoming Socket Mode requests."""
        # Acknowledge the request immediately
        from slack_sdk.socket_mode.response import SocketModeResponse

        response = SocketModeResponse(envelope_id=req.envelope_id)
        await client.send_socket_mode_response(response)

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        event_type = event.get("type", "")

        # Handle reaction events
        if event_type in ("reaction_added", "reaction_removed"):
            reaction_event = ReactionEvent(
                channel_id=self.channel_id,
                sender_id=event.get("user", ""),
                conversation_id=event.get("item", {}).get("channel", ""),
                target_message_id=event.get("item", {}).get("ts", ""),
                emoji=event.get("reaction", ""),
                action="added" if event_type == "reaction_added" else "removed",
                thread_id=None,
            )
            await self.dispatch_reaction(reaction_event)
            return

        if event_type != "message" or event.get("subtype"):
            return  # Skip non-message events and subtypes (edits, deletes, etc.)

        # Skip messages from bots (including ourselves) to avoid infinite loops
        if event.get("bot_id") or event.get("bot_profile"):
            return

        text = event.get("text", "").strip()
        mentions_bot = bool(self._bot_user_id and f"<@{self._bot_user_id}>" in text)

        # Remove @bot mention text to avoid confusing the LLM
        if mentions_bot and self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        # Slack channel_type: "im" = DM, "channel"/"group"/"mpim" = group
        channel_type = event.get("channel_type", "")
        chat_type = "p2p" if channel_type == "im" else ("group" if channel_type else None)

        msg = InboundMessage(
            channel_id=self.channel_id,
            sender_id=event.get("user", ""),
            sender_name="",
            conversation_id=event.get("channel", ""),
            message_id=event.get("ts", ""),
            text=text,
            mentions_bot=mentions_bot,
            chat_type=chat_type,
            thread_id=event.get("thread_ts"),
            raw_payload=req.payload,
        )
        await self.dispatch_message(msg)

    async def stop(self) -> None:
        """Disconnect Socket Mode client."""
        if self._socket_client and hasattr(self._socket_client, "disconnect"):
            await self._socket_client.disconnect()
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        self._socket_client = None
        self._web_client = None
        logger.info("Slack adapter '%s' stopped.", self.channel_id)

    def _render_ir(self, ir: MarkdownIR) -> str:
        """Render a MarkdownIR to Slack mrkdwn format."""
        opts = RenderOptions(
            style_markers=_SLACK_STYLES,
            escape_fn=slack_escape,
            link_builder=lambda text, href: f"<{href}|{text}>",
        )
        return render_ir(ir, opts)

    async def send_message(self, message: OutboundMessage) -> Optional[str]:
        """Send a reply via Slack Web API. Returns the message timestamp (ID)."""
        if not self._web_client:
            logger.error("Slack web client not initialized for '%s'", self.channel_id)
            return None

        try:
            text = self._render_ir(message.ir) if message.ir else message.text
            if message.sql:
                text += f"\n\n```{message.sql}```"

            chunks = chunk_text(text)
            last_ts: Optional[str] = None
            for chunk in chunks:
                kwargs = {
                    "channel": message.conversation_id,
                    "text": chunk,
                }
                if message.thread_id:
                    kwargs["thread_ts"] = message.thread_id

                response = await self._web_client.chat_postMessage(**kwargs)
                last_ts = response.get("ts") if isinstance(response, dict) else getattr(response, "ts", None)

            logger.debug("Slack message sent to channel %s", message.conversation_id)
            return last_ts
        except Exception as e:
            logger.error("Failed to send Slack message: %s", e)
            return None

    async def add_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Add an emoji reaction to a Slack message."""
        if not self._web_client:
            return
        try:
            await self._web_client.reactions_add(channel=conversation_id, name=emoji, timestamp=message_id)
        except Exception as e:
            logger.warning("Failed to add Slack reaction '%s': %s", emoji, e)

    async def remove_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Remove an emoji reaction from a Slack message."""
        if not self._web_client:
            return
        try:
            await self._web_client.reactions_remove(channel=conversation_id, name=emoji, timestamp=message_id)
        except Exception as e:
            logger.warning("Failed to remove Slack reaction '%s': %s", emoji, e)
