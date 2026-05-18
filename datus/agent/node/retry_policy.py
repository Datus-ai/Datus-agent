# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Pluggable validate/retry strategies for ``AgenticNode.execute_stream``.

Most nodes execute a single LLM stream and stop. ``DeliverableAgenticNode``
and ``GenExtKnowledgeAgenticNode`` need to re-prompt the model when an
out-of-band validator reports failure. Rather than embedding that loop in
the template method, ``AgenticNode._get_retry_policy()`` returns a
``RetryPolicy`` and the template drives the loop generically.

The default :class:`NoRetryPolicy` runs once and never retries — node
subclasses only override ``_get_retry_policy`` when they need the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.schemas.action_history import ActionHistory


@runtime_checkable
class RetryPolicy(Protocol):
    """Contract that drives the retry loop inside ``execute_stream``.

    The template invokes hooks in this order each iteration:

    1. ``reset(ctx)`` — clear per-attempt accumulators (e.g. validation
       hook's ``final_report``) before the stream begins.
    2. Stream runs to completion (every action is yielded to the caller).
    3. ``should_retry(ctx)`` — if False (or ``ctx.attempt == max_attempts``),
       the loop breaks.
    4. ``on_retry_actions(ctx)`` — yielded by the template before the next
       attempt; lets the policy surface a user-visible "retrying…" action.
    5. ``next_prompt(ctx)`` — returned string replaces ``ctx.user_prompt``
       for the next attempt. ``None`` means keep the current prompt.

    After the loop exits, ``finalise(ctx)`` runs once — strategies that
    need to project per-iteration state into ``ctx.extras`` for the
    subclass's ``_build_success_result`` do it here.
    """

    max_attempts: int

    def reset(self, ctx: "StreamRunContext") -> None: ...
    def should_retry(self, ctx: "StreamRunContext") -> bool: ...
    def next_prompt(self, ctx: "StreamRunContext") -> Optional[str]: ...
    def on_retry_actions(self, ctx: "StreamRunContext") -> Iterable["ActionHistory"]: ...
    def finalise(self, ctx: "StreamRunContext") -> None: ...


class NoRetryPolicy:
    """Default policy: single execution, never retry.

    Used by every node that does not override ``_get_retry_policy``.
    """

    max_attempts: int = 1

    def reset(self, ctx: "StreamRunContext") -> None:
        return None

    def should_retry(self, ctx: "StreamRunContext") -> bool:
        return False

    def next_prompt(self, ctx: "StreamRunContext") -> Optional[str]:
        return None

    def on_retry_actions(self, ctx: "StreamRunContext") -> Iterable["ActionHistory"]:
        return ()

    def finalise(self, ctx: "StreamRunContext") -> None:
        return None
