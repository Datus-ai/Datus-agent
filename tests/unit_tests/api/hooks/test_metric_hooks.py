# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.api.hooks.metric_hooks`` registry + helpers and the
``MetricRAG._emit_retrieval`` integration."""

from types import SimpleNamespace

import pytest

from datus.api.hooks import (
    MetricRetrievalEvent,
    get_metric_retrieval_hook,
    make_metric_retrieval_hook,
    set_metric_retrieval_hook,
)
from datus.storage.metric.store import MetricRAG


@pytest.fixture(autouse=True)
def _clear_hook_after_test():
    yield
    set_metric_retrieval_hook(None)


def test_registry_set_get_clear():
    assert get_metric_retrieval_hook() is None
    seen = []
    set_metric_retrieval_hook(make_metric_retrieval_hook(seen.append))
    get_metric_retrieval_hook().on_retrieval(MetricRetrievalEvent(query_text="q", metrics=[]))
    assert len(seen) == 1 and seen[0].query_text == "q"
    set_metric_retrieval_hook(None)
    assert get_metric_retrieval_hook() is None


def test_emit_retrieval_forwards_uid_and_context():
    seen = []
    set_metric_retrieval_hook(make_metric_retrieval_hook(seen.append))

    # Call the unbound method with a minimal fake self — no real storage needed.
    fake = SimpleNamespace(
        datasource_id="ds1",
        sub_agent_name="sub",
        agent_config=SimpleNamespace(project_name="proj"),
    )
    results = [
        {"id": "metric:dau", "name": "dau", "uid": "u1", "vector": [0.1]},
        {"id": "metric:rev", "name": "rev", "uid": ""},
    ]
    MetricRAG._emit_retrieval(fake, "daily active", results)

    assert len(seen) == 1
    ev = seen[0]
    assert ev.datasource_id == "ds1"
    assert ev.project_name == "proj"
    assert ev.sub_agent_name == "sub"
    # Only id/name/uid are forwarded — heavy fields like the vector are dropped.
    assert ev.metrics == [
        {"id": "metric:dau", "name": "dau", "uid": "u1"},
        {"id": "metric:rev", "name": "rev", "uid": ""},
    ]


def test_emit_retrieval_never_raises_when_hook_errors():
    calls = []

    def _boom(_ev):
        calls.append(1)
        raise RuntimeError("host blew up")

    set_metric_retrieval_hook(make_metric_retrieval_hook(_boom))
    fake = SimpleNamespace(datasource_id="ds1", sub_agent_name=None, agent_config=None)
    # The hook raises, but _emit_retrieval must swallow it (a failing host can
    # never break a search) — it reaches the hook yet still returns cleanly.
    result = MetricRAG._emit_retrieval(fake, "q", [{"id": "metric:x", "name": "x", "uid": "u"}])
    assert result is None
    assert calls == [1]  # error path was actually exercised, not skipped


def test_emit_retrieval_noop_without_hook():
    seen = []
    set_metric_retrieval_hook(make_metric_retrieval_hook(seen.append))
    set_metric_retrieval_hook(None)  # no hook registered
    assert get_metric_retrieval_hook() is None

    fake = SimpleNamespace(datasource_id="ds1", sub_agent_name=None, agent_config=None)
    MetricRAG._emit_retrieval(fake, "q", [{"id": "metric:x", "name": "x", "uid": "u"}])
    assert seen == []  # nothing emitted when no hook is registered
