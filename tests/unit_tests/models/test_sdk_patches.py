# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest

from datus.models.sdk_patches import apply_sdk_patches, remove_sdk_patches


@pytest.fixture(autouse=True)
def unpatched_baseline():
    remove_sdk_patches()
    yield
    remove_sdk_patches()


def test_stream_patch_preserves_glm_request_id(monkeypatch):
    from types import SimpleNamespace

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
