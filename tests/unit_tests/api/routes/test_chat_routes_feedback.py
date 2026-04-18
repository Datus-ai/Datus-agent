# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for POST /api/v1/chat/feedback endpoint."""

from unittest.mock import MagicMock

import pytest

from datus.api.models.cli_models import FeedbackChatInput, StreamChatInput
from datus.api.routes.chat_routes import stream_chat_feedback


def _build_svc():
    svc = MagicMock()

    async def _empty_stream(*args, **kwargs):
        if False:
            yield
        return

    svc.chat.stream_chat = MagicMock(side_effect=_empty_stream)
    return svc


def _build_ctx(user_id="tester"):
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


async def _drain(response):
    """Iterate a StreamingResponse body_iterator so the inner generator runs."""
    async for _ in response.body_iterator:
        pass


@pytest.mark.asyncio
async def test_feedback_endpoint_renders_prompt_and_routes_to_feedback_subagent():
    svc = _build_svc()
    request = FeedbackChatInput(
        source_session_id="chat_session_abc",
        reaction_emoji="thumbsup",
        reference_msg="Here is your SQL result",
        database="sales_db",
    )

    response = await stream_chat_feedback(request, svc, _build_ctx())
    await _drain(response)

    svc.chat.stream_chat.assert_called_once()
    call_args = svc.chat.stream_chat.call_args
    stream_input: StreamChatInput = call_args.args[0]
    assert isinstance(stream_input, StreamChatInput)
    assert stream_input.subagent_id == "feedback"
    assert stream_input.source_session_id == "chat_session_abc"
    assert stream_input.database == "sales_db"
    assert call_args.kwargs["sub_agent_id"] == "feedback"
    assert call_args.kwargs["user_id"] == "tester"
    assert stream_input.message == '[The user reacted to this message "Here is your SQL result" with [thumbsup]]'


@pytest.mark.parametrize(
    "field",
    ["source_session_id", "reaction_emoji", "reference_msg"],
)
@pytest.mark.parametrize("blank_value", ["", "   ", "\t\n"])
def test_feedback_input_rejects_blank_required_field(field, blank_value):
    """Required feedback fields must reject empty / whitespace-only strings."""
    kwargs = dict(
        source_session_id="sess_1",
        reaction_emoji="thumbsup",
        reference_msg="hi",
    )
    kwargs[field] = blank_value
    with pytest.raises(ValueError):
        FeedbackChatInput(**kwargs)


def test_feedback_input_strips_whitespace_on_required_fields():
    """Surrounding whitespace on required fields should be stripped, not retained."""
    inp = FeedbackChatInput(
        source_session_id="  sess_1  ",
        reaction_emoji="  thumbsup  ",
        reference_msg="  hi  ",
    )
    assert inp.source_session_id == "sess_1"
    assert inp.reaction_emoji == "thumbsup"
    assert inp.reference_msg == "hi"


@pytest.mark.asyncio
async def test_feedback_endpoint_appends_optional_reaction_msg():
    svc = _build_svc()
    request = FeedbackChatInput(
        source_session_id="chat_session_abc",
        reaction_emoji="thumbsdown",
        reference_msg="Wrong answer",
        reaction_msg="Please recheck the metric definition",
    )

    response = await stream_chat_feedback(request, svc, _build_ctx())
    await _drain(response)

    stream_input: StreamChatInput = svc.chat.stream_chat.call_args.args[0]
    assert stream_input.message.endswith("Please recheck the metric definition")
    assert "[thumbsdown]" in stream_input.message
