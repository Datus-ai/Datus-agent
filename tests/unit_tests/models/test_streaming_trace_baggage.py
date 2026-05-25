# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from contextlib import contextmanager

import pytest


class FakeResult:
    def __init__(self, active):
        self.active = active

    async def stream_events(self):
        assert self.active["value"] is True
        yield "event"


class FakeObservabilityManager:
    def __init__(self, active):
        self.active = active

    @contextmanager
    def trace_baggage(self, name, attributes):
        self.active["value"] = True
        try:
            yield
        finally:
            self.active["value"] = False


@pytest.mark.asyncio
async def test_codex_stream_events_keep_trace_baggage_active(monkeypatch):
    from datus.models import codex_model

    active = {"value": False}
    manager = FakeObservabilityManager(active)
    monkeypatch.setattr(codex_model, "_agents_trace_baggage", lambda agent_name: manager.trace_baggage("chat", {}))

    events = [event async for event in codex_model._stream_events_with_trace_baggage(FakeResult(active), "chat")]

    assert events == ["event"]
    assert active["value"] is False


@pytest.mark.asyncio
async def test_openai_compatible_stream_events_keep_trace_baggage_active(monkeypatch):
    from datus.models import openai_compatible

    active = {"value": False}
    manager = FakeObservabilityManager(active)
    monkeypatch.setattr(
        openai_compatible, "_agents_trace_baggage", lambda agent_name: manager.trace_baggage("chat", {})
    )

    events = [event async for event in openai_compatible._stream_events_with_trace_baggage(FakeResult(active), "chat")]

    assert events == ["event"]
    assert active["value"] is False
