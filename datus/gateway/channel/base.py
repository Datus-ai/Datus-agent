# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base class for IM channel adapters (long-connection only)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from datus.gateway.models import ChannelConfig, InboundMessage, OutboundMessage, ReactionEvent

if TYPE_CHECKING:
    from datus.gateway.bridge import ChannelBridge


class ChannelAdapter(ABC):
    """Base class for all IM channel adapters.

    Adapters actively connect to the IM platform via long-lived connections
    (WebSocket, Stream SDK, Socket Mode, etc.) — no inbound webhooks.
    """

    def __init__(
        self,
        channel_id: str,
        config: dict,
        bridge: ChannelBridge,
        channel_config: Optional[ChannelConfig] = None,
    ) -> None:
        self.channel_id = channel_id
        self.config = config
        self._bridge = bridge
        self._channel_config = channel_config

    @property
    def supports_streaming(self) -> bool:
        """Whether this adapter supports streaming cards (delta accumulation).

        Adapters that return False will not receive delta messages — only
        complete, fully-rendered messages are forwarded by the bridge.
        """
        return False

    async def dispatch_message(self, msg: InboundMessage) -> None:
        """Forward an inbound message to the bridge for processing."""
        await self._bridge.handle_message(msg, self, self._channel_config)

    async def dispatch_reaction(self, event: ReactionEvent) -> None:
        """Forward a reaction event to the bridge for processing."""
        await self._bridge.handle_reaction(event, self)

    @abstractmethod
    async def start(self) -> None:
        """Connect to the IM platform and start listening."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and clean up resources."""

    @abstractmethod
    async def send_message(self, message: OutboundMessage) -> Optional[str]:
        """Send a response message to the IM platform.

        Returns the platform message ID of the sent message, or None.
        """

    @abstractmethod
    async def add_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Add an emoji reaction to a message."""

    @abstractmethod
    async def remove_reaction(
        self, conversation_id: str, message_id: str, emoji: str, thread_id: Optional[str] = None
    ) -> None:
        """Remove an emoji reaction from a message."""

    async def finalize_stream(self, stream_id: str) -> None:
        """Finalize a streaming session and clean up associated resources.

        Subclasses that support streaming cards (e.g. Feishu) should override
        this method. The default implementation is a no-op.
        """
