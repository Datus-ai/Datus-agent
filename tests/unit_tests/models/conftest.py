# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared fixtures for model-layer unit tests.

The ``two_turn_agents_model`` factory below builds a real agents-SDK
:class:`~agents.models.interface.Model` so a test can drive the SDK's genuine
turn loop (``Runner.run_streamed``) without any network. That matters for the
mid-run-insert tests: the injection point is a per-turn SDK hook, so a mocked
``Runner`` would assert nothing about when a staged message is actually
flushed.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, List

import pytest
from agents.models.interface import Model
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


def _response(output: List[Any], response_id: str) -> Response:
    return Response(
        id=response_id,
        created_at=0.0,
        model="stub",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class RecordingTwoTurnModel(Model):
    """Two-turn stub: turn 1 calls ``tool_name``, turn 2 returns final text.

    Every turn's input list is snapshotted into :attr:`inputs`, which is what
    lets a test assert *which* LLM turn first saw a given message.
    """

    def __init__(self, tool_name: str = "probe_tool", final_text: str = "done") -> None:
        self.inputs: List[Any] = []
        self.turn = 0
        self._tool_name = tool_name
        self._final_text = final_text

    def saw_on_turn(self, index: int, needle: str) -> bool:
        """Whether ``needle`` appears in the input of turn ``index`` (0-based)."""
        if index >= len(self.inputs):
            return False
        return needle in json.dumps(self.inputs[index])

    async def get_response(self, *args, **kwargs):  # pragma: no cover - streaming only
        raise NotImplementedError("RecordingTwoTurnModel is streaming-only")

    async def stream_response(self, system_instructions, input, *args, **kwargs) -> AsyncIterator[Any]:
        self.turn += 1
        self.inputs.append(json.loads(json.dumps(input, default=str)))

        if self.turn == 1:
            output: List[Any] = [
                ResponseFunctionToolCall(
                    id="fc_1",
                    call_id="call_1",
                    name=self._tool_name,
                    arguments="{}",
                    type="function_call",
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="msg_1",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(text=self._final_text, type="output_text", annotations=[])],
                )
            ]

        yield ResponseCompletedEvent(
            response=_response(output, f"resp_{self.turn}"),
            type="response.completed",
            sequence_number=0,
        )


@pytest.fixture
def two_turn_agents_model():
    """Factory for :class:`RecordingTwoTurnModel`."""
    return RecordingTwoTurnModel
