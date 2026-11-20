"""Semantic draft lock contention must not freeze the backend event loop."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datus.agent.node.configurable_agentic_node import ConfigurableAgenticNode


def _node():
    return SimpleNamespace(
        definition=SimpleNamespace(plugin=object()),
        _semantic_draft=None,
        _semantic_draft_creating=False,
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
    retry_draft = SimpleNamespace(close=lambda: None)
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", return_value=retry_draft):
        with pytest.raises(RuntimeError, match="draft constructed"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
    assert node._semantic_draft is retry_draft


@pytest.mark.asyncio
async def test_abandoned_draft_cleanup_keeps_event_loop_responsive():
    started = threading.Event()
    construct_release = threading.Event()
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    closed = threading.Event()
    heartbeat = threading.Event()
    cleanup_saw_heartbeat = []

    def close():
        cleanup_started.set()
        cleanup_release.wait(timeout=0.5)
        cleanup_saw_heartbeat.append(heartbeat.is_set())
        closed.set()

    def construct(_scope):
        started.set()
        construct_release.wait(timeout=1)
        return SimpleNamespace(close=close)

    async def respond():
        for _ in range(100):
            if cleanup_started.is_set():
                break
            await asyncio.sleep(0.001)
        assert cleanup_started.is_set()
        heartbeat.set()
        cleanup_release.set()

    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", side_effect=construct):
        node = _node()
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
            construct_release.set()
            response = asyncio.create_task(respond())
            for _ in range(200):
                if closed.is_set():
                    break
                await asyncio.sleep(0.005)
            await response
        finally:
            construct_release.set()
            cleanup_release.set()
    assert closed.is_set()
    assert cleanup_saw_heartbeat == [True]
    assert node._semantic_draft is None


@pytest.mark.asyncio
async def test_overlapping_begin_rejects_second_request_without_constructing_another_draft():
    started = threading.Event()
    release = threading.Event()
    draft = SimpleNamespace(close=lambda: None)

    def construct(_scope):
        started.set()
        release.wait(timeout=1)
        return draft

    node = _node()
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", side_effect=construct) as factory:
        first = asyncio.create_task(ConfigurableAgenticNode._begin_semantic_draft(node))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set()
            with pytest.raises(RuntimeError, match="already active|being created"):
                await ConfigurableAgenticNode._begin_semantic_draft(node)
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="draft constructed"):
                await first
    assert factory.call_count == 1
    assert node._semantic_draft is draft
    with pytest.raises(RuntimeError, match="already active"):
        await ConfigurableAgenticNode._begin_semantic_draft(node)
    assert node._semantic_draft is draft


@pytest.mark.asyncio
async def test_begin_reserves_creation_before_retiring_database_tools():
    started = asyncio.Event()
    release = asyncio.Event()

    async def retire():
        started.set()
        await release.wait()

    node = _node()
    node._retire_database_tools = retire
    draft = SimpleNamespace(close=lambda: None)
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", return_value=draft) as factory:
        first = asyncio.create_task(ConfigurableAgenticNode._begin_semantic_draft(node))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            with pytest.raises(RuntimeError, match="already active"):
                await ConfigurableAgenticNode._begin_semantic_draft(node)
            factory.assert_not_called()
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="draft constructed"):
                await first
    factory.assert_called_once_with(node.execution_scope)
    assert node._semantic_draft is draft


@pytest.mark.asyncio
async def test_begin_can_retry_after_retiring_database_tools_fails():
    node = _node()
    node._retire_database_tools.side_effect = [ValueError("retirement failed"), None]
    draft = SimpleNamespace(close=lambda: None)
    with patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", return_value=draft) as factory:
        with pytest.raises(ValueError, match="retirement failed"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
        factory.assert_not_called()
        assert node._semantic_draft is None
        with pytest.raises(RuntimeError, match="draft constructed"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
    factory.assert_called_once_with(node.execution_scope)
    assert node._semantic_draft is draft


@pytest.mark.asyncio
async def test_begin_can_retry_after_constructor_failure():
    node = _node()
    draft = SimpleNamespace(close=lambda: None)
    with patch(
        "datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft",
        side_effect=[ValueError("invalid artifact"), draft],
    ) as factory:
        with pytest.raises(ValueError, match="invalid artifact"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
        assert node._semantic_draft is None
        with pytest.raises(RuntimeError, match="draft constructed"):
            await ConfigurableAgenticNode._begin_semantic_draft(node)
    assert factory.call_count == 2
    assert node._semantic_draft is draft


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["construct", "close"])
async def test_abandoned_draft_errors_are_logged(failure_stage):
    started = threading.Event()
    release = threading.Event()

    def construct(_scope):
        started.set()
        release.wait(timeout=1)
        if failure_stage == "construct":
            raise ValueError("constructor failed")
        return SimpleNamespace(close=Mock(side_effect=ValueError("cleanup failed")))

    with (
        patch("datus.agent.node.configurable_agentic_node.SemanticAuthoringDraft", side_effect=construct),
        patch("datus.agent.node.configurable_agentic_node.logger") as logger,
    ):
        node = _node()
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
        for _ in range(200):
            if logger.exception.called:
                break
            await asyncio.sleep(0.005)
        logger.exception.assert_called_once_with("Failed to construct or close an abandoned semantic draft")
    assert node._semantic_draft is None
