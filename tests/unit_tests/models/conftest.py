# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared fixtures for model-layer unit tests.

The ``two_turn_agents_model`` factory below builds a real agents-SDK
:class:`~agents.models.interface.Model` so a test can drive the SDK's genuine
turn loop (``Runner.run_streamed``) without any network. That matters for the
mid-run-insert and mid-turn-compaction tests: both injection points are
per-model-call SDK hooks, so a mocked ``Runner`` would assert nothing about
what a given LLM call actually receives.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Optional, Sequence

import pytest
from agents.models.interface import Model
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails


def _response(output: List[Any], response_id: str, usage: Optional[ResponseUsage] = None) -> Response:
    return Response(
        id=response_id,
        created_at=0.0,
        model="stub",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


def _usage(input_tokens: int, output_tokens: int = 10) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class RecordingScriptedModel(Model):
    """Scripted stub: every turn either calls ``tool_name`` or returns final text.

    ``script`` is a sequence of ``"tool"`` / ``"final"`` steps consumed one per
    turn; once exhausted the model keeps returning final text. Tool calls use
    ``call_id="call_{turn}"`` so a test can tell which turn's tool round a
    given item belongs to. Every turn's input list is snapshotted into
    :attr:`inputs`, which is what lets a test assert *which* LLM turn first
    saw (or stopped seeing) a given message. ``input_tokens_per_turn`` attaches
    a real ``usage`` block to each response so ``on_llm_end`` hooks fire with
    numbers.
    """

    def __init__(
        self,
        script: Sequence[str] = ("tool", "final"),
        tool_name: str = "probe_tool",
        final_text: str = "done",
        input_tokens_per_turn: Optional[Sequence[int]] = None,
    ) -> None:
        self.inputs: List[Any] = []
        self.turn = 0
        self._script = list(script)
        self._tool_name = tool_name
        self._final_text = final_text
        self._input_tokens = list(input_tokens_per_turn or [])

    def saw_on_turn(self, index: int, needle: str) -> bool:
        """Whether ``needle`` appears in the input of turn ``index`` (0-based)."""
        if index >= len(self.inputs):
            return False
        return needle in json.dumps(self.inputs[index])

    async def get_response(self, *args, **kwargs):  # pragma: no cover - streaming only
        raise NotImplementedError("RecordingScriptedModel is streaming-only")

    async def stream_response(self, system_instructions, input, *args, **kwargs) -> AsyncIterator[Any]:
        self.turn += 1
        self.inputs.append(json.loads(json.dumps(input, default=str)))

        step = self._script[self.turn - 1] if self.turn - 1 < len(self._script) else "final"
        if step == "tool":
            output: List[Any] = [
                ResponseFunctionToolCall(
                    id=f"fc_{self.turn}",
                    call_id=f"call_{self.turn}",
                    name=self._tool_name,
                    arguments="{}",
                    type="function_call",
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id=f"msg_{self.turn}",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(text=self._final_text, type="output_text", annotations=[])],
                )
            ]

        usage = None
        if self.turn - 1 < len(self._input_tokens):
            usage = _usage(self._input_tokens[self.turn - 1])

        yield ResponseCompletedEvent(
            response=_response(output, f"resp_{self.turn}", usage),
            type="response.completed",
            sequence_number=0,
        )


class RecordingTwoTurnModel(RecordingScriptedModel):
    """Two-turn stub: turn 1 calls ``tool_name``, turn 2 returns final text."""

    def __init__(self, tool_name: str = "probe_tool", final_text: str = "done") -> None:
        super().__init__(("tool", "final"), tool_name=tool_name, final_text=final_text)


@pytest.fixture
def two_turn_agents_model():
    """Factory for :class:`RecordingTwoTurnModel`."""
    return RecordingTwoTurnModel


@pytest.fixture
def scripted_agents_model():
    """Factory for :class:`RecordingScriptedModel`."""
    return RecordingScriptedModel
