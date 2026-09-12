# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

import pytest

from datus.models.sdk_patches import apply_sdk_patches, remove_sdk_patches


@pytest.fixture(autouse=True)
def unpatched_baseline():
    remove_sdk_patches()
    yield
    remove_sdk_patches()


@pytest.mark.asyncio
async def test_stream_patch_preserves_only_correlation_headers(monkeypatch):
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    async def fake_stream_helper(self, *args, **kwargs):
        return object(), {"Trace-ID": "remote-trace", "Set-Cookie": "secret"}

    monkeypatch.setattr(BaseLLMHTTPHandler, "make_async_call_stream_helper", fake_stream_helper)
    apply_sdk_patches()
    logging_obj = SimpleNamespace(model_call_details={})
    try:
        await BaseLLMHTTPHandler.make_async_call_stream_helper(object(), logging_obj=logging_obj)
    finally:
        remove_sdk_patches()

    assert logging_obj.model_call_details["_datus_response_headers"] == {"trace-id": "remote-trace"}
    assert "secret" not in str(logging_obj.model_call_details)


def test_stream_patch_preserves_glm_request_id(monkeypatch):
    from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler

    transformed = SimpleNamespace()
    monkeypatch.setattr(OpenAIChatCompletionStreamingHandler, "chunk_parser", lambda self, chunk: transformed)
    apply_sdk_patches()
    try:
        result = OpenAIChatCompletionStreamingHandler.chunk_parser(
            object(), {"id": "completion-id", "request_id": "glm-request"}
        )
    finally:
        remove_sdk_patches()

    assert result is transformed
    assert result.request_id == "glm-request"
