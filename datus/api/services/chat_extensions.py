# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Constructor contracts for trusted host chat extensions, never client DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode
    from datus.api.services.chat_task_manager import ChatTaskManager
    from datus.configuration.agent_config import AgentConfig


class ChatNodeFactory(Protocol):
    """Construct a host node for an already resolved session identity."""

    def __call__(
        self,
        *,
        agent_config: AgentConfig,
        subagent_id: str | None,
        node_id: str,
        user_id: str | None,
        interactive: bool,
        session_id: str,
    ) -> AgenticNode | None:
        """Return a host node, or None to use the standard Agent factory.

        Missing session IDs are resolved to node_id before this call.
        Called on a worker thread with a per-turn configuration clone. Raising
        fails this turn; failures must never fall back to a less restricted node.
        """
        ...


class ChatTaskManagerFactory(Protocol):
    """Construct a project manager from trusted startup defaults."""

    def __call__(
        self,
        *,
        default_source: str | None,
        default_interactive: bool,
        stream_thinking: bool,
    ) -> ChatTaskManager:
        """Return a fresh manager owned by one project's DatusService."""
        ...
