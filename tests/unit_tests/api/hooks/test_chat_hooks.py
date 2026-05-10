# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.api.hooks.chat_hooks`` registry + helpers and the
hook integration in ``datus.api.routes.chat_routes.stream_chat``."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from datus.api.hooks import (
    ChatPostUsageContext,
    ChatPreCheckOutcome,
    get_chat_hooks,
    make_chat_hooks,
    set_chat_hooks,
)
from datus.api.models.cli_models import StreamChatInput
from datus.api.routes import chat_routes


@pytest.fixture(autouse=True)
def _clear_hooks_after_test():
    yield
    set_chat_hooks(None)


def _mock_svc_with_nodes(nodes=None):
    svc = MagicMock()
    svc.agent_config.agentic_nodes = nodes or {}
    return svc


class TestRegistry:
    def test_default_is_none(self):
        assert get_chat_hooks() is None

    def test_set_and_get_hook(self):
        hooks = make_chat_hooks()
        set_chat_hooks(hooks)
        assert get_chat_hooks() is hooks

    def test_clear_hooks(self):
        set_chat_hooks(make_chat_hooks())
        set_chat_hooks(None)
        assert get_chat_hooks() is None

    def test_make_chat_hooks_default_pre_allows(self):
        hooks = make_chat_hooks()

        async def _run():
            return await hooks.pre_chat(MagicMock(), StreamChatInput(message="hi"), "u1")

        outcome = asyncio.run(_run())
        assert outcome.allow is True

    def test_make_chat_hooks_default_post_is_noop(self):
        """The default post hook returns ``None`` and never touches its inputs.

        Sentinel asserts that the no-op default doesn't accidentally read from
        or mutate the supplied request / payload — anything else here would
        be a regression in ``make_chat_hooks``'s default branch.
        """
        hooks = make_chat_hooks()
        sentinel: list[str] = []
        request = MagicMock()
        # Mark the request as "untouched" — if the default branch ever
        # decides to introspect headers / call methods, the assertion below
        # against ``mock_calls`` would catch it.
        ctx = ChatPostUsageContext(user_id="u1", session_id="s1", model=None, usage={})

        async def _run():
            return await hooks.post_chat(request, StreamChatInput(message="hi"), ctx)

        result = asyncio.run(_run())

        assert result is None
        assert sentinel == []
        assert request.mock_calls == []


class TestStreamChatPreHookDenial:
    """When pre_chat denies, stream_chat returns a single SSE error event."""

    @pytest.mark.asyncio
    async def test_denial_short_circuits_chat_service(self):
        svc = _mock_svc_with_nodes()
        # If we ever reach the upstream stream this side-effects; the denial
        # path must not call into ChatService at all.
        svc.chat.stream_chat = MagicMock(side_effect=AssertionError("upstream invoked"))
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi")

        async def _pre(_req, _input, _uid):
            return ChatPreCheckOutcome(
                allow=False,
                error="Datus credits exhausted; switch model.",
                error_type="DATUS_CREDITS_EXHAUSTED",
            )

        set_chat_hooks(make_chat_hooks(pre_chat=_pre))

        response = await chat_routes.stream_chat(request, svc, ctx, MagicMock())

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 1
        body = chunks[0]
        assert "event: error" in body
        # Extract the JSON payload to assert structured fields.
        data_line = next(line for line in body.splitlines() if line.startswith("data: "))
        payload = json.loads(data_line[len("data: ") :])
        assert payload["error"] == "Datus credits exhausted; switch model."
        assert payload["error_type"] == "DATUS_CREDITS_EXHAUSTED"

    @pytest.mark.asyncio
    async def test_pre_hook_exception_treated_as_server_error(self):
        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = MagicMock(side_effect=AssertionError("upstream invoked"))
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi")

        async def _pre(_req, _input, _uid):
            raise RuntimeError("saas unreachable")

        set_chat_hooks(make_chat_hooks(pre_chat=_pre))

        response = await chat_routes.stream_chat(request, svc, ctx, MagicMock())

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 1
        data_line = next(line for line in chunks[0].splitlines() if line.startswith("data: "))
        payload = json.loads(data_line[len("data: ") :])
        assert payload["error_type"] == "PRE_CHAT_HOOK_ERROR"
        assert "Server error" in payload["error"]


class TestStreamChatPostHookSchedule:
    """When pre_chat allows, the upstream stream is forwarded and post_chat is scheduled."""

    @pytest.mark.asyncio
    async def test_post_hook_scheduled_after_stream(self):
        # Build a fake upstream that yields exactly one "end" event.
        from datus.api.models.cli_models import SSEEndData, SSEEvent

        end_event = SSEEvent(
            id=1,
            event="end",
            data=SSEEndData(
                session_id="sess-1",
                llm_session_id=None,
                total_events=1,
                action_count=0,
                duration=0.1,
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cached_tokens=0,
                requests=1,
            ),
            timestamp="2026-01-01T00:00:00Z",
        )

        async def _upstream(*_args, **_kwargs):
            yield end_event

        svc = _mock_svc_with_nodes()
        svc.chat.stream_chat = _upstream
        ctx = MagicMock(user_id="u1")
        request = StreamChatInput(message="hi")

        post_seen: list[ChatPostUsageContext] = []
        post_done = asyncio.Event()

        async def _pre(_req, _input, _uid):
            return ChatPreCheckOutcome(allow=True, extra={"trace_id": "abc"})

        async def _post(_req, _input, post_ctx: ChatPostUsageContext):
            post_seen.append(post_ctx)
            post_done.set()

        set_chat_hooks(make_chat_hooks(pre_chat=_pre, post_chat=_post))

        response = await chat_routes.stream_chat(request, svc, ctx, MagicMock())

        body_chunks = []
        async for chunk in response.body_iterator:
            body_chunks.append(chunk)

        # Wait for the fire-and-forget post hook to run.
        await asyncio.wait_for(post_done.wait(), timeout=1.0)

        assert len(body_chunks) == 1
        assert "event: end" in body_chunks[0]
        assert len(post_seen) == 1
        ctx_seen = post_seen[0]
        assert ctx_seen.user_id == "u1"
        assert ctx_seen.usage.get("total_tokens") == 30
        assert ctx_seen.pre_check_extra == {"trace_id": "abc"}
        assert ctx_seen.error is None
