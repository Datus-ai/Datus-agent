# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Retry strategy driven by ``verify_sql`` tool verification status.

Used by :class:`GenExtKnowledgeAgenticNode`. The node owns the
``_verification_passed`` flag and the ``_get_retry_prompt`` helper; this
policy is a thin adapter that lets the template's retry loop consume them.

When gold_sql is absent (``node._gold_sql`` falsy), verification is
considered passed and the loop exits after the first attempt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

logger = get_logger(__name__)


class VerifySqlRetryPolicy:
    """Re-prompt the model when ``verify_sql`` reports a mismatch."""

    def __init__(self, node: "GenExtKnowledgeAgenticNode"):
        self.node = node
        # max_verification_retries is the number of *retries* allowed; the
        # total attempt count is one larger (initial + retries).
        self.max_attempts = max(1, node.max_verification_retries + 1)

    def reset(self, ctx: "StreamRunContext") -> None:
        # Verification state is owned by the node so the ``verify_sql`` tool's
        # ``on_end`` hook can keep updating it during the stream.
        self.node._reset_verification_state()

    def should_retry(self, ctx: "StreamRunContext") -> bool:
        if self.node._verification_passed:
            return False
        # No gold_sql means there is nothing to verify against — treat as passed.
        if not getattr(self.node, "_gold_sql", None):
            return False
        logger.info(
            "Verification failed for %s (attempt %d/%d), scheduling retry",
            self.node.get_node_name(),
            ctx.attempt,
            self.max_attempts,
        )
        return True

    def next_prompt(self, ctx: "StreamRunContext") -> Optional[str]:
        # ``ctx.attempt`` is the iteration we just finished; the next attempt
        # uses ``ctx.attempt`` as the retry index (1-based for the user-facing
        # "(N/max)" suffix the node's prompt builder embeds).
        return self.node._get_retry_prompt(ctx.attempt)

    def on_retry_actions(self, ctx: "StreamRunContext") -> Iterable["ActionHistory"]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="verification_retry",
            messages=(f"Verification failed, retrying ({ctx.attempt}/{self.node.max_verification_retries})..."),
            input_data={"attempt": ctx.attempt},
            status=ActionStatus.PROCESSING,
        )
        return (action,)

    def finalise(self, ctx: "StreamRunContext") -> None:
        return None
