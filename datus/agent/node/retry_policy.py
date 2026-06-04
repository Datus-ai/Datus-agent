# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Retry-policy contract for ``AgenticNode.execute_stream``.

Most nodes execute a single LLM stream and stop. ``DeliverableAgenticNode``
re-prompts the model when an out-of-band validator reports failure. Rather than embedding the loop in each subclass,
``AgenticNode._get_retry_policy()`` returns a :class:`RetryPolicy` and the
template's generic loop drives it.

This module exposes only the contract (:class:`RetryPolicy` protocol) and
the default :class:`NoRetryPolicy` (single-shot). Concrete policies live in
the node module that uses them since each is bound to its node's internal state
and would not be reused elsewhere.
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

    1. ``reset(ctx)`` — clear per-attempt accumulators (validation hook's
       ``final_report``, verification flags) before the stream begins.
    2. Stream runs to completion (every action is yielded to the caller).
    3. ``should_retry(ctx)`` — when False (or ``ctx.attempt == max_attempts``)
       the loop breaks. Implementations may stash per-attempt state here
       even when no further retry will fire.
    4. ``on_retry_actions(ctx)`` — yielded by the template before the next
       attempt; lets the policy surface a user-visible "retrying…" action.
    5. ``next_prompt(ctx)`` — returned string replaces ``ctx.user_prompt``
       for the next attempt. ``None`` keeps the current prompt.

    After the loop exits, ``finalise(ctx)`` runs once. Strategies that need
    to project per-iteration state into ``ctx.extras`` for the subclass's
    ``_build_success_result`` do it here.
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
