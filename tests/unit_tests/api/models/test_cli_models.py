# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/models/cli_models.py."""

import pytest
from pydantic import ValidationError

from datus.api.models.cli_models import SSEEndData, StreamChatInput, UserInteractionInput


class TestSSEEndDataTokenFields:
    def test_backward_compatible_defaults(self):
        """New token fields default to 0 — existing callers unaffected."""
        data = SSEEndData(
            session_id="s1",
            total_events=5,
            action_count=3,
            duration=1.5,
        )
        assert data.requests == 0
        assert data.input_tokens == 0
        assert data.output_tokens == 0
        assert data.total_tokens == 0
        assert data.cached_tokens == 0
        assert data.session_total_tokens == 0
        assert data.context_length == 0

    def test_token_fields_populated(self):
        data = SSEEndData(
            session_id="s1",
            total_events=5,
            action_count=3,
            duration=1.5,
            requests=3,
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            cached_tokens=500,
            session_total_tokens=1000,
            context_length=128000,
        )
        assert data.requests == 3
        assert data.input_tokens == 1000
        assert data.output_tokens == 200
        assert data.total_tokens == 1200
        assert data.cached_tokens == 500
        assert data.session_total_tokens == 1000
        assert data.context_length == 128000

    def test_serialization_includes_token_fields(self):
        data = SSEEndData(
            session_id="s1",
            total_events=1,
            action_count=1,
            duration=0.1,
            requests=2,
            input_tokens=42,
        )
        d = data.model_dump()
        assert "requests" in d
        assert d["requests"] == 2
        assert "input_tokens" in d
        assert d["input_tokens"] == 42
        assert d["output_tokens"] == 0


class TestUserInteractionInput:
    """Tests for UserInteractionInput with List[List[str]] and legacy format support."""

    def test_single_select(self):
        """Single-select: input=[['2']]."""
        obj = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"]])
        assert obj.input == [["2"]]

    def test_multi_select(self):
        """Multi-select: input=[['1','3']]."""
        obj = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["1", "3"]])
        assert obj.input == [["1", "3"]]

    def test_batch_mixed(self):
        """Batch: single + multi select."""
        obj = UserInteractionInput(session_id="s1", interaction_key="k1", input=[["2"], ["1", "3"]])
        assert len(obj.input) == 2
        assert obj.input[0] == ["2"]
        assert obj.input[1] == ["1", "3"]

    def test_legacy_format_list_of_strings(self):
        """Legacy List[str] format should be auto-normalized to List[List[str]]."""
        data = UserInteractionInput(
            session_id="s1",
            interaction_key="k1",
            input=["answer1", "answer2"],
        )
        assert data.input == [["answer1"], ["answer2"]]

    def test_single_legacy_answer(self):
        """Single legacy string answer should normalize correctly."""
        data = UserInteractionInput(
            session_id="s1",
            interaction_key="k1",
            input=["yes"],
        )
        assert data.input == [["yes"]]

    def test_rejects_empty_input(self):
        """Missing input raises validation error."""
        with pytest.raises(ValidationError):
            UserInteractionInput(session_id="s1", interaction_key="k1")


class TestStreamChatInputLanguage:
    """``language`` is an optional per-request override for the configured
    ``agent.language``. Unset means "inherit yaml default".
    """

    def test_language_defaults_to_none(self):
        obj = StreamChatInput(message="hi")
        assert obj.language is None

    def test_language_accepts_code(self):
        obj = StreamChatInput(message="hi", language="zh")
        assert obj.language == "zh"

    def test_language_independent_from_prompt_language(self):
        """``prompt_language`` selects prompt template variant (unchanged);
        ``language`` controls user-facing response language (new). Both must
        travel as separate fields."""
        obj = StreamChatInput(message="hi", prompt_language="zh", language="en")
        assert obj.prompt_language == "zh"
        assert obj.language == "en"
