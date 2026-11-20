"""Semantic draft lock contention must not freeze the backend event loop."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from datus.agent.node.configurable_agentic_node import ConfigurableAgenticNode


def _node():
    return SimpleNamespace(
        definition=SimpleNamespace(plugin=object()),
        _semantic_draft=None,
        _semantic_authoring_session=None,
        execution_scope=object(),
        _retire_database_tools=AsyncMock(),
        _refresh_native_tools=AsyncMock(side_effect=RuntimeError("draft constructed")),
    )


@pytest.mark.asyncio
async def test_begin_draft_does_not_block_event_loop_while_waiting_for_file_lock():
    loop_responded = threading.Event()
    release = threading.Event()
    saw_responsive_loop = []
    draft = SimpleNamespace(close=lambda: None)

    def construct(_scope):
        release.wait(timeout=0.5)
        saw_responsive_loop.append(loop_responded.is_set())
        return draft

    async def respond():
        await asyncio.sleep(0.01)
        loop_responded.set()
        release.set()

    response = asyncio.create_task(respond())
    node = _node()
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", side_effect=construct):
        with pytest.raises(RuntimeError, match="draft constructed"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
    await response
    assert saw_responsive_loop == [True]
    assert node._semantic_draft is draft


@pytest.mark.asyncio
async def test_cancelled_begin_closes_draft_created_after_cancellation():
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def construct(_scope):
        started.set()
        release.wait(timeout=1)
        return SimpleNamespace(close=closed.set)

    node = _node()
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", side_effect=construct):
        task = asyncio.create_task(ConfigurableAgenticNode._begin_semantic_draft(node))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        for _ in range(100):
            if closed.is_set():
                break
            await asyncio.sleep(0.005)
    assert closed.is_set()
    assert node._semantic_draft is None
