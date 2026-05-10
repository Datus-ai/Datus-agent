"""Pre / post hooks around the streaming chat endpoint.

The chat route invokes ``ChatHooks.pre_chat`` *before* spinning up the
agentic loop, and ``ChatHooks.post_chat`` after the loop completes (or
errors). The hook is optional — if no host has called ``set_chat_hooks``,
the route runs in passthrough mode.

Design notes:

* The pre-hook returns a structured outcome (allow / deny + user-facing
  message) instead of raising, so the route can convert a denial into a
  well-formed SSE error payload that frontends already render.

* The post-hook is **fire-and-forget**: the route schedules it as a
  background task and never awaits it. Hosts must catch their own
  exceptions; an unhandled error is logged but never propagates back to
  the client. This keeps token-usage reporting off the response path.

* Both hooks receive the original ``fastapi.Request`` so hosts can lift
  headers like ``x-trace-id`` (or read the OTel trace id) without us
  having to thread bespoke context through the agent layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

from fastapi import Request

from datus.api.models.cli_models import StreamChatInput


@dataclass
class ChatPreCheckOutcome:
    """Result returned by :meth:`ChatHooks.pre_chat`."""

    allow: bool
    """When ``False`` the route emits an SSE ``error`` event and stops."""

    error: Optional[str] = None
    """User-facing message used as ``SSEErrorData.error`` on denial."""

    error_type: Optional[str] = None
    """Machine-readable code used as ``SSEErrorData.error_type`` on denial."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Free-form data the host may carry over to the post hook (logging etc.)."""


@dataclass
class ChatPostUsageContext:
    """Payload passed to :meth:`ChatHooks.post_chat` after the stream ends.

    ``usage`` mirrors the fields of ``SSEEndData`` (input_tokens,
    output_tokens, total_tokens, cached_tokens, requests, ...). When the
    agent fails before reaching the end event this dict is empty.
    """

    user_id: Optional[str]
    session_id: Optional[str]
    model: Optional[str]
    usage: Dict[str, Any]
    error: Optional[str] = None
    pre_check_extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatHooks(Protocol):
    """Extension contract for chat-stream embedding hosts."""

    async def pre_chat(
        self,
        http_request: Request,
        stream_request: StreamChatInput,
        user_id: Optional[str],
    ) -> ChatPreCheckOutcome:
        """Approve or veto a streaming chat request.

        Implementations MUST be defensive: catch their own exceptions and
        translate them into a denial outcome (or re-raise to surface a
        generic server-error SSE event).
        """
        ...

    async def post_chat(
        self,
        http_request: Request,
        stream_request: StreamChatInput,
        ctx: ChatPostUsageContext,
    ) -> None:
        """Report usage / billing after the agentic loop finishes.

        Called as a fire-and-forget background task. Implementations must
        handle their own retries and logging — exceptions raised here are
        only logged, never surfaced to the client.
        """
        ...


_chat_hooks: Optional[ChatHooks] = None


def set_chat_hooks(hooks: Optional[ChatHooks]) -> None:
    """Register (or clear) the active chat hooks.

    Typically called once during application lifespan startup. Passing
    ``None`` removes any previously registered hooks.
    """
    global _chat_hooks
    _chat_hooks = hooks


def get_chat_hooks() -> Optional[ChatHooks]:
    """Return the active chat hooks, or ``None`` when unregistered."""
    return _chat_hooks


# Convenience factory for hosts that prefer plain callables over a class.
PreCheckFn = Callable[[Request, StreamChatInput, Optional[str]], Awaitable[ChatPreCheckOutcome]]
PostUsageFn = Callable[[Request, StreamChatInput, ChatPostUsageContext], Awaitable[None]]


def make_chat_hooks(
    *,
    pre_chat: Optional[PreCheckFn] = None,
    post_chat: Optional[PostUsageFn] = None,
) -> ChatHooks:
    """Build a :class:`ChatHooks` from optional callables.

    Missing callbacks fall back to permissive defaults: pre allows by
    default, post is a no-op.
    """

    async def _default_pre(_req: Request, _input: StreamChatInput, _uid: Optional[str]) -> ChatPreCheckOutcome:
        return ChatPreCheckOutcome(allow=True)

    async def _default_post(_req: Request, _input: StreamChatInput, _ctx: ChatPostUsageContext) -> None:
        return None

    pre = pre_chat or _default_pre
    post = post_chat or _default_post

    class _LambdaHooks:
        async def pre_chat(
            self,
            http_request: Request,
            stream_request: StreamChatInput,
            user_id: Optional[str],
        ) -> ChatPreCheckOutcome:
            return await pre(http_request, stream_request, user_id)

        async def post_chat(
            self,
            http_request: Request,
            stream_request: StreamChatInput,
            ctx: ChatPostUsageContext,
        ) -> None:
            await post(http_request, stream_request, ctx)

    return _LambdaHooks()
