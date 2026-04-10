# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for SSEEndData token fields in datus/api/models/cli_models.py."""

from datus.api.models.cli_models import SSEEndData


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
