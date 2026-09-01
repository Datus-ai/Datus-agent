# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.models.base.LLMBaseModel.test_connection``."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel

pytestmark = pytest.mark.ci


class _StubModel(LLMBaseModel):
    """Minimal concrete subclass exposing a controllable ``generate``.

    Inheriting from :class:`LLMBaseModel` without implementing every
    abstract method would trigger TypeError on instantiation, so the
    stub satisfies the full interface with no-op coroutines for the
    unused branches.
    """

    def __init__(self, responder):
        self.model_config = ModelConfig(type="stub", api_key="", model="stub", base_url=None)
        self._responder = responder

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs):  # type: ignore[override]
        return self._responder(prompt, **kwargs)

    def generate_with_json_output(self, prompt: Any, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools_stream(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def token_count(self, prompt: str) -> int:  # type: ignore[override]
        return 0

    def context_length(self):  # type: ignore[override]
        return None


class TestTestConnection:
    def test_returns_ok_on_successful_generate(self):
        model = _StubModel(lambda _p, **_k: "pong")
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is True
        assert err == ""

    def test_returns_false_on_empty_response(self):
        model = _StubModel(lambda _p, **_k: "   ")
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is False
        assert "empty" in err.lower()

    def test_returns_false_on_exception(self):
        def _fail(_p, **_k):
            raise RuntimeError("down")

        model = _StubModel(_fail)
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is False
        assert "down" in err

    def test_returns_false_on_timeout(self):
        model = _StubModel(MagicMock(return_value="ignored"))
        with patch("datus.models.base.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            ok, err = asyncio.run(model.test_connection(timeout=0.05))
        assert ok is False
        assert "timed out" in err.lower()


class TestCreateModelCache:
    """``create_model`` keeps an LRU cache so ``/model`` switches stay cheap."""

    def _agent_config(self, model_config: ModelConfig):
        cfg = MagicMock()
        cfg.active_model.return_value = model_config
        cfg.model_config.return_value = model_config
        cfg.resolve_model_ref.return_value = model_config
        cfg.models = {"custom": model_config}
        cfg.session_dir = None
        return cfg

    def _patch_constructor(self):
        sentinel = object()
        module = MagicMock()
        instance = MagicMock(name="LLMInstance")
        module.OpenAIModel = MagicMock(return_value=instance)
        return sentinel, module, instance

    def _fresh_cache(self):
        """Reset the process-wide model cache between tests."""
        LLMBaseModel._MODEL_CACHE.clear()

    def test_same_config_returns_cached_instance(self):
        self._fresh_cache()
        cfg_a = ModelConfig(type="openai", api_key="k", model="gpt-4.1", base_url="https://a")
        agent_cfg = self._agent_config(cfg_a)
        _, module, instance = self._patch_constructor()
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            first = LLMBaseModel.create_model(agent_cfg)
            second = LLMBaseModel.create_model(agent_cfg)
        assert first is second
        module.OpenAIModel.assert_called_once()

    def test_different_model_yields_new_instance(self):
        self._fresh_cache()
        cfg_a = ModelConfig(type="openai", api_key="k", model="gpt-4.1", base_url="https://a")
        cfg_b = ModelConfig(type="openai", api_key="k", model="gpt-4o", base_url="https://a")
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            a1 = LLMBaseModel.create_model(self._agent_config(cfg_a))
            b1 = LLMBaseModel.create_model(self._agent_config(cfg_b))
            a2 = LLMBaseModel.create_model(self._agent_config(cfg_a))
        assert a1 is not b1
        assert a1 is a2, "switching back to the original model should hit the cache"
        assert module.OpenAIModel.call_count == 2

    def test_cache_eviction_respects_maxsize(self):
        self._fresh_cache()
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            for i in range(LLMBaseModel._MODEL_CACHE_MAXSIZE + 2):
                cfg = ModelConfig(type="openai", api_key="k", model=f"m{i}", base_url="https://a")
                LLMBaseModel.create_model(self._agent_config(cfg))
        assert len(LLMBaseModel._MODEL_CACHE) == LLMBaseModel._MODEL_CACHE_MAXSIZE

    def test_different_reasoning_effort_yields_new_instance(self):
        """Changing ``reasoning_effort`` must bust the cache so the new adapter
        picks up the fresh effort level instead of reusing a stale binding."""
        self._fresh_cache()
        cfg_low = ModelConfig(type="openai", api_key="k", model="gpt-4.1", reasoning_effort="low")
        cfg_high = ModelConfig(type="openai", api_key="k", model="gpt-4.1", reasoning_effort="high")
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            low = LLMBaseModel.create_model(self._agent_config(cfg_low))
            high = LLMBaseModel.create_model(self._agent_config(cfg_high))
        assert low is not high
        assert module.OpenAIModel.call_count == 2

    def test_different_enable_thinking_yields_new_instance(self):
        """Toggling ``enable_thinking`` must bust the cache (previously a bug)."""
        self._fresh_cache()
        cfg_off = ModelConfig(type="openai", api_key="k", model="gpt-4.1", enable_thinking=False)
        cfg_on = ModelConfig(type="openai", api_key="k", model="gpt-4.1", enable_thinking=True)
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            off = LLMBaseModel.create_model(self._agent_config(cfg_off))
            on = LLMBaseModel.create_model(self._agent_config(cfg_on))
        assert off is not on
        assert module.OpenAIModel.call_count == 2


# ---------------------------------------------------------------------------
# Mid-run message insertion — shared drain contract + SDK input filter
# ---------------------------------------------------------------------------


class _RecordingBroker:
    def __init__(self, explode: bool = False):
        self.emitted: list[str] = []
        self._explode = explode

    def emit_user_insert(self, text: str) -> None:
        if self._explode:
            raise RuntimeError("broker down")
        self.emitted.append(text)


class TestDrainPendingUserInserts:
    """Contract for the single definition of "flush staged user messages".

    Every provider path (agents-SDK filter, Claude's native Anthropic loop)
    routes through this helper, so the invariants below are what keep the
    three paths behaving identically.
    """

    def test_no_queue_returns_empty(self):
        assert LLMBaseModel.drain_pending_user_inserts(None) == []

    def test_drains_fifo_and_empties_queue(self):
        from datus.cli.execution_state import PendingInputQueue

        queue = PendingInputQueue()
        queue.push("first")
        queue.push("second")

        assert LLMBaseModel.drain_pending_user_inserts(queue) == ["first", "second"]
        assert len(queue) == 0

    def test_interrupted_run_leaves_queue_intact(self):
        """An aborting run must not consume text the user will re-send.

        Draining here would delete the message from the queue *and* never
        deliver it: the interrupted run makes no further LLM call, so the
        text would vanish silently.
        """
        from datus.cli.execution_state import InterruptController, PendingInputQueue

        queue = PendingInputQueue()
        queue.push("steer me")
        controller = InterruptController()
        controller.interrupt()

        assert LLMBaseModel.drain_pending_user_inserts(queue, interrupt_controller=controller) == []
        assert queue.snapshot() == ["steer me"]

    def test_emits_each_message_to_the_broker(self):
        from datus.cli.execution_state import PendingInputQueue

        queue = PendingInputQueue()
        queue.push("a")
        queue.push("b")
        broker = _RecordingBroker()

        drained = LLMBaseModel.drain_pending_user_inserts(queue, interaction_broker=broker)

        assert drained == ["a", "b"]
        assert broker.emitted == ["a", "b"]

    def test_broker_failure_still_delivers_text_to_the_model(self):
        """Transcript rendering is best-effort; the model input is not."""
        from datus.cli.execution_state import PendingInputQueue

        queue = PendingInputQueue()
        queue.push("a")

        assert LLMBaseModel.drain_pending_user_inserts(queue, interaction_broker=_RecordingBroker(explode=True)) == [
            "a"
        ]

    def test_broken_queue_does_not_propagate(self):
        """A queue hiccup must never abort the agent run."""
        broken = MagicMock()
        broken.drain.side_effect = RuntimeError("queue exploded")

        assert LLMBaseModel.drain_pending_user_inserts(broken) == []


class TestBuildRunConfigDeliversInsertAtTurnBoundary:
    """End-to-end through the real SDK turn loop.

    ``_build_run_config`` lives on the base class precisely so every
    agents-SDK-driven provider gets the filter. These tests drive
    ``Runner.run_streamed`` for real (stub model, no network) and assert the
    turn on which the model first sees an inserted message — the thing a
    mocked Runner cannot check.
    """

    @staticmethod
    async def _run(model, queue, tool_pushes: str | None):
        from agents import Agent, Runner, function_tool

        @function_tool
        def probe_tool() -> str:
            """Probe tool; the user types while it runs."""
            if tool_pushes is not None:
                queue.push(tool_pushes)
            return "tool finished"

        agent = Agent(name="probe_agent", instructions="", model=model, tools=[probe_tool])
        run_config = LLMBaseModel._build_run_config(
            MagicMock(),
            pending_input_queue=queue,
            session=None,
            agent_name="probe_agent",
        )
        result = Runner.run_streamed(agent, input="start", max_turns=5, run_config=run_config)
        async for _ in result.stream_events():
            pass

    def test_message_typed_during_tool_call_reaches_the_next_turn(self, two_turn_agents_model):
        """The whole point of insertion: steer the run already in progress.

        Before the fix on the codex/native paths this text only arrived after
        the run had finished, via the chat layer's auto-continuation.
        """
        from datus.cli.execution_state import PendingInputQueue

        model = two_turn_agents_model()
        queue = PendingInputQueue()

        asyncio.run(self._run(model, queue, tool_pushes="STOP_AND_CHECK"))

        assert model.turn == 2
        assert not model.saw_on_turn(0, "STOP_AND_CHECK")
        assert model.saw_on_turn(1, "STOP_AND_CHECK")
        assert len(queue) == 0

    def test_inserted_text_arrives_as_a_user_role_item(self, two_turn_agents_model):
        """Injection must be a user message, not an edit of the tool result."""
        from datus.cli.execution_state import PendingInputQueue

        model = two_turn_agents_model()
        queue = PendingInputQueue()

        asyncio.run(self._run(model, queue, tool_pushes="STOP_AND_CHECK"))

        injected = [
            item
            for item in model.inputs[1]
            if isinstance(item, dict) and item.get("role") == "user" and "STOP_AND_CHECK" in json.dumps(item)
        ]
        assert len(injected) == 1
        assert injected[0]["content"] == [{"type": "input_text", "text": "STOP_AND_CHECK"}]

    def test_idle_queue_leaves_turn_input_untouched(self, two_turn_agents_model):
        """No staged text → the filter is a no-op, not an empty user message."""
        from datus.cli.execution_state import PendingInputQueue

        model = two_turn_agents_model()
        queue = PendingInputQueue()

        asyncio.run(self._run(model, queue, tool_pushes=None))

        assert model.turn == 2
        for turn_input in model.inputs:
            assert all(
                not (isinstance(item, dict) and item.get("role") == "user" and item.get("content") == [])
                for item in turn_input
            )
