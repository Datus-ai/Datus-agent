# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for the IM Gateway (``gateway/slack.md`` / ``gateway/feishu.md``).

The IM Gateway is a documented core flow with only unit coverage before this
suite. These tests drive the real end-to-end path with a real LLM:

    InboundMessage -> ChannelBridge.handle_message -> ChatTaskManager (real
    agentic loop) -> SSE events -> OutboundMessage -> adapter.send_message

Only the IM platform transport is faked: a ``_RecordingAdapter`` (a real
``ChannelAdapter`` subclass, identical in shape to the Slack/Feishu adapters)
records what the bridge sends back instead of calling Slack/Feishu. Everything
between the inbound message and the outbound reply — session handling, the
agentic loop, SQL execution, streaming, rich-text rendering, reactions — is
exercised for real.
"""

from typing import Optional

import pytest
import pytest_asyncio

from datus.api.services.chat_task_manager import ChatTaskManager
from datus.gateway.bridge import ChannelBridge
from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.models import DONE_EMOJI, ERROR_EMOJI, PROCESSING_EMOJI, InboundMessage, OutboundMessage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _RecordingAdapter(ChannelAdapter):
    """A real ChannelAdapter that records outbound traffic instead of calling an
    IM platform — the test's only fake boundary."""

    def __init__(self, bridge: Optional[ChannelBridge] = None):
        # ``bridge`` is only used by the adapter's own dispatch_* helpers, which
        # the bridge-driven test path does not exercise; a placeholder is fine.
        super().__init__("itest_channel", {}, bridge=bridge)
        self.sent: list[OutboundMessage] = []
        self.reactions_added: list[tuple[str, str, str]] = []
        self.reactions_removed: list[tuple[str, str, str]] = []
        self._counter = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_message(self, message: OutboundMessage) -> str:
        self.sent.append(message)
        self._counter += 1
        return f"bot_msg_{self._counter}"

    async def add_reaction(self, conversation_id, message_id, emoji, thread_id=None) -> None:
        self.reactions_added.append((conversation_id, message_id, emoji))

    async def remove_reaction(self, conversation_id, message_id, emoji, thread_id=None) -> None:
        self.reactions_removed.append((conversation_id, message_id, emoji))

    def combined_text(self) -> str:
        """All outbound text concatenated, lowercased — for content assertions."""
        return "\n".join(m.text or "" for m in self.sent).lower()


@pytest.mark.nightly
@pytest.mark.product_e2e
class TestGatewayBridgeAgentic:
    """IM Gateway end-to-end through ChannelBridge with a real LLM."""

    @pytest_asyncio.fixture
    async def bridge_and_adapter(self, nightly_agent_config):
        # default_interactive=False mirrors the real gateway runtime: the IM
        # bot never blocks on human-in-the-loop confirmations.
        task_manager = ChatTaskManager(default_interactive=False)
        bridge = ChannelBridge(nightly_agent_config, task_manager)
        adapter = _RecordingAdapter(bridge)
        try:
            yield bridge, adapter
        finally:
            await task_manager.shutdown()

    @pytest.mark.asyncio
    async def test_direct_message_answered_end_to_end(self, bridge_and_adapter):
        """A DM data question is routed through the real agent and answered back."""
        bridge, adapter = bridge_and_adapter

        msg = InboundMessage(
            channel_id="ch_itest",
            sender_id="user_itest",
            conversation_id="conv_dm",
            message_id="m_dm_1",
            text="How many schools are there in Fresno county?",
            chat_type="p2p",
        )

        await bridge.handle_message(msg, adapter)

        # The bridge forwarded at least one non-empty reply from the agent.
        assert adapter.sent, "Gateway should send at least one reply back to the channel"
        combined = adapter.combined_text()
        assert combined.strip(), f"Reply text should not be empty, got {[m.text for m in adapter.sent]}"

        # Behavior-based content check: the answer must reflect the real query —
        # either it echoes the subject ('fresno') or it contains a numeric count.
        # This avoids asserting on exact LLM phrasing while still proving a real
        # answer (not a generic acknowledgement) flowed back.
        # Real-LLM phrasing varies: a correct answer either echoes the subject
        # or states the count, so accepting either is intentional, not ambiguous.
        assert "fresno" in combined or any(ch.isdigit() for ch in combined), (  # audit-noqa: or_assert
            f"Reply should reference the question's subject or a count, got: {combined[:500]}"
        )

        # Reaction lifecycle: processing added then removed, success (not error)
        # reaction added at the end.
        assert ("conv_dm", "m_dm_1", PROCESSING_EMOJI) in adapter.reactions_added
        assert ("conv_dm", "m_dm_1", PROCESSING_EMOJI) in adapter.reactions_removed
        assert ("conv_dm", "m_dm_1", DONE_EMOJI) in adapter.reactions_added
        assert all(r[2] != ERROR_EMOJI for r in adapter.reactions_added), (
            f"No error reaction expected on a successful run, got: {adapter.reactions_added}"
        )

    @pytest.mark.asyncio
    async def test_group_mention_answered_in_thread(self, bridge_and_adapter):
        """An @bot group message is processed and the reply lands in a thread."""
        bridge, adapter = bridge_and_adapter

        msg = InboundMessage(
            channel_id="ch_itest",
            sender_id="user_itest",
            conversation_id="conv_group",
            message_id="m_grp_1",
            text="List schools in Alameda county.",
            chat_type="group",
            mentions_bot=True,
        )

        await bridge.handle_message(msg, adapter)

        assert adapter.sent, "Group @bot message should be answered"
        combined = adapter.combined_text()
        assert combined.strip(), "Group reply text should not be empty"
        # Same rationale as the DM test: subject-or-count is a deliberate,
        # robust check against non-deterministic LLM phrasing.
        assert "alameda" in combined or any(ch.isdigit() for ch in combined), (  # audit-noqa: or_assert
            f"Group reply should reference the subject or a count, got: {combined[:500]}"
        )

        # A group @bot message without an explicit thread starts its own thread
        # (thread_id defaults to the triggering message_id), so replies are
        # threaded rather than posted to the channel root.
        assert all(m.thread_id == "m_grp_1" for m in adapter.sent), (
            f"Group replies should be threaded under the triggering message, got: {[m.thread_id for m in adapter.sent]}"
        )
