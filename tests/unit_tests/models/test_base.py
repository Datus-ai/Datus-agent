# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.models.base.LLMBaseModel.test_connection``."""

from __future__ import annotations

import asyncio
import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestSummarizeItemsDefault:
    """``LLMBaseModel.summarize_items``: a session-less, tool-less single call
    over an explicit transcript. This is what keeps the compaction summary from
    writing into (or reading from) the live session mid-run."""

    @pytest.mark.asyncio
    async def test_appends_prompt_as_user_item_and_runs_without_session_or_tools(self):
        model = MagicMock()
        model.generate_with_tools = AsyncMock(return_value={"content": "S", "usage": {"output_tokens": 3}})
        items = [
            {"role": "user", "content": "start"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "big"},
        ]

        result = await LLMBaseModel.summarize_items(
            model, items, instruction="sys", prompt="SUMMARIZE", agent_name="chat"
        )

        assert result == {"content": "S", "usage": {"output_tokens": 3}}
        kwargs = model.generate_with_tools.await_args.kwargs
        assert kwargs["prompt"] == items + [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "SUMMARIZE"}]}
        ]
        assert kwargs["session"] is None
        assert kwargs["tools"] is None
        assert kwargs["mcp_servers"] is None
        assert kwargs["max_turns"] == 1
        assert kwargs["instruction"] == "sys"
        assert kwargs["agent_name"] == "chat"
        # The transcript passed in is not mutated.
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_rejects_non_responses_transcripts(self):
        from datus.utils.exceptions import DatusException

        model = MagicMock()
        model.generate_with_tools = AsyncMock()
        with pytest.raises(DatusException):
            await LLMBaseModel.summarize_items(model, [], instruction="", prompt="P", item_format="anthropic")
        model.generate_with_tools.assert_not_awaited()


class _CompactingFakeNode:
    """Node double for the SDK-loop tests below.

    Mirrors the real ``AgenticNode.compact_mid_turn`` contract: decide from
    ``base_tokens + estimate(tail)`` against a token budget, rebuild the view
    with ``build_mid_turn_view`` and persist it to the session when one is
    wired. ``results`` can override the dict returned on each call.
    """

    def __init__(self, *, budget_tokens: int, session=None, results=None, summary: str = "SUMMARY", once: bool = False):
        from datus.configuration.agent_config import CompactConfig

        self.context_length = 1000
        self.running_turn_usage = None
        self._compact_cfg = CompactConfig()
        self._budget = budget_tokens
        self._session = session
        self._results = list(results or [])
        self._summary = summary
        # ``once``: after one major rewrite report noop forever, standing in for
        # the real node's ratio dropping once the context has been compacted.
        self._once = once
        self._done = False
        self.calls: List[dict] = []

    def _mid_turn_output_reserve(self) -> int:
        return 0

    async def compact_mid_turn(self, items, *, item_format, base_tokens, tail_start, instruction, turn_request, reason):
        from datus.agent.node.context_rewriter import build_mid_turn_view, estimate_items_tokens

        self.calls.append({"items": list(items), "base_tokens": base_tokens, "tail_start": tail_start})
        if self._results:
            return self._results.pop(0)
        if self._done or base_tokens + estimate_items_tokens(items[tail_start:]) < self._budget:
            return {"mode": "noop", "success": True, "items": items}
        view = build_mid_turn_view(items, self._summary, item_format=item_format, turn_request=turn_request)
        if self._session is not None:
            await self._session.clear_session()
            await self._session.add_items(view)
        if self._once:
            self._done = True
        return {"mode": "major", "success": True, "items": view, "summary": self._summary}


class TestBuildRunConfigCompactsMidRun:
    """End-to-end through the real SDK turn loop.

    A mid-turn compaction that only rewrote SQLite never changed what the
    in-flight run sent to the model (the SDK rebuilds each call's input from
    memory). These tests drive ``Runner.run_streamed`` for real (stub model,
    no network) and assert what each LLM call actually receives.
    """

    @staticmethod
    async def _run(model, rewriter, *, queue=None, session=None, tool_pushes=None, hooks=True, max_turns=8):
        from agents import Agent, Runner, function_tool

        @function_tool
        def probe_tool() -> str:
            """Probe tool returning a bulky result."""
            if tool_pushes is not None and queue is not None:
                queue.push(tool_pushes)
            return "result " + "x" * 400

        agent = Agent(
            name="probe_agent",
            instructions="",
            model=model,
            tools=[probe_tool],
            hooks=rewriter if hooks else None,
        )
        run_config = LLMBaseModel._build_run_config(
            MagicMock(),
            pending_input_queue=queue,
            session=session,
            agent_name="probe_agent",
            context_rewriter=rewriter,
        )
        result = Runner.run_streamed(agent, input="start", max_turns=max_turns, session=session, run_config=run_config)
        async for _ in result.stream_events():
            pass

    @staticmethod
    def _rewriter(node):
        from datus.agent.node.context_rewriter import MidTurnCompactor

        return MidTurnCompactor(node)

    @staticmethod
    def _call_ids(turn_input):
        return [item.get("call_id") for item in turn_input if isinstance(item, dict) and item.get("call_id")]

    @pytest.mark.asyncio
    async def test_turn_two_receives_the_compacted_view(self, scripted_agents_model):
        """The regression: a compaction decided before turn 2 must change what
        turn 2 is sent. On the pre-fix code turn 2 still carried ``call_1``."""
        from datus.agent.node.compact_prompts import MID_TURN_RESUME_PREFIX

        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=1)  # anything crosses the line
        await self._run(model, self._rewriter(node))

        assert model.turn == 3
        turn2 = model.inputs[1]
        assert "call_1" not in self._call_ids(turn2)
        assert turn2[0]["role"] == "user" and "start" in json.dumps(turn2[0])
        assert any(item.get("role") == "assistant" and "SUMMARY" in json.dumps(item) for item in turn2)
        assert turn2[-1]["role"] == "user"
        assert turn2[-1]["content"][0]["text"].startswith(MID_TURN_RESUME_PREFIX)
        # Turn 1 was untouched: the very first call of a run never compacts.
        assert model.inputs[0] == [{"role": "user", "content": "start"}]

    @pytest.mark.asyncio
    async def test_the_view_is_replayed_on_every_later_call(self, scripted_agents_model):
        """The SDK does not write the filter's output back, so turn 3 must be
        the same view plus turn 2's tool round — never the raw history."""
        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=1, once=True)
        rewriter = self._rewriter(node)
        await self._run(model, rewriter)

        turn2, turn3 = model.inputs[1], model.inputs[2]
        assert turn3[: len(turn2)] == turn2
        tail_ids = self._call_ids(turn3[len(turn2) :])
        assert tail_ids == ["call_2", "call_2"]
        assert "call_1" not in self._call_ids(turn3)
        assert rewriter.compactions == 1  # compacted once, replayed after

    @pytest.mark.asyncio
    async def test_tail_estimate_pushes_the_decision_over_the_line(self, scripted_agents_model):
        """Usage reported by the last call is stale by the time the tool output
        lands; the estimate of the new items must count."""
        from datus.schemas.token_usage import TokenUsage

        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=150)
        # Last call's input alone is under budget; with the ~100-token tool
        # output appended it is over.
        node.running_turn_usage = TokenUsage(requests=1, session_total_tokens=120, context_length=1000)
        await self._run(model, self._rewriter(node))

        assert node.calls[0]["base_tokens"] == 120
        assert node.calls[0]["tail_start"] == 1
        assert "call_1" not in self._call_ids(model.inputs[1])

    @pytest.mark.asyncio
    async def test_below_budget_leaves_the_run_untouched(self, scripted_agents_model):
        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=10_000)
        rewriter = self._rewriter(node)
        await self._run(model, rewriter)

        assert rewriter.compactions == 0
        assert self._call_ids(model.inputs[2]) == ["call_1", "call_1", "call_2", "call_2"]

    @pytest.mark.asyncio
    async def test_session_mirrors_the_view_after_the_run(self, scripted_agents_model, tmp_path):
        """SQLite must equal the view plus everything appended afterwards — no
        pre-compaction tool rounds, no duplicated prompt."""
        from agents.extensions.memory import AdvancedSQLiteSession

        session = AdvancedSQLiteSession(session_id="sid_compact", db_path=str(tmp_path / "s.db"), create_tables=True)
        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=1, session=session, once=True)
        await self._run(model, self._rewriter(node), session=session)

        stored = await session.get_items()
        turn3 = model.inputs[2]
        # The persisted history is exactly what the last call saw, plus the
        # final assistant message the model produced on that call.
        assert stored[: len(turn3)] == turn3
        assert len(stored) == len(turn3) + 1
        assert stored[-1]["role"] == "assistant" and "done" in json.dumps(stored[-1])
        assert "call_1" not in self._call_ids(stored)
        assert sum(1 for it in stored if it.get("role") == "user" and "start" in json.dumps(it)) == 1

    @pytest.mark.asyncio
    async def test_insert_in_the_same_call_lands_after_the_resume_and_stays_visible(self, scripted_agents_model):
        """A message typed while tool 1 runs is drained in the same filter call
        that compacts. It must trail the resume instruction (compaction first,
        then inserts) and still be present on turn 3 — on the pre-fix code an
        insert was visible for exactly one call and then vanished."""
        from datus.cli.execution_state import PendingInputQueue

        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=1, once=True)
        queue = PendingInputQueue()
        await self._run(model, self._rewriter(node), queue=queue, tool_pushes="STOP_AND_CHECK")

        turn2, turn3 = model.inputs[1], model.inputs[2]
        assert "STOP_AND_CHECK" in json.dumps(turn2[-1])
        assert turn2[-2]["content"][0]["text"].startswith("[DATUS_COMPACT_RESUME]")
        assert model.saw_on_turn(2, "STOP_AND_CHECK")
        assert turn3[: len(turn2)] == turn2
        assert len(queue) == 0

    @pytest.mark.asyncio
    async def test_insert_without_compaction_stays_visible_on_later_calls(self, scripted_agents_model):
        """Pinning fixes the pre-existing one-call visibility of inserts."""
        from datus.cli.execution_state import PendingInputQueue

        model = scripted_agents_model(["tool", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=10_000)
        queue = PendingInputQueue()
        await self._run(model, self._rewriter(node), queue=queue, tool_pushes="STOP_AND_CHECK")

        assert not model.saw_on_turn(0, "STOP_AND_CHECK")
        assert model.saw_on_turn(1, "STOP_AND_CHECK")
        assert model.saw_on_turn(2, "STOP_AND_CHECK")

    @pytest.mark.asyncio
    async def test_failures_trip_the_breaker_and_never_abort_the_run(self, scripted_agents_model):
        model = scripted_agents_model(["tool", "tool", "tool", "tool", "tool", "final"])
        failing = [{"mode": "major", "success": False}] * 10
        node = _CompactingFakeNode(budget_tokens=1, results=failing)
        await self._run(model, self._rewriter(node))

        assert model.turn == 6  # the run completed
        assert len(node.calls) == 3  # 3 strikes, then pass-through
        assert self._call_ids(model.inputs[-1])[:2] == ["call_1", "call_1"]

    @pytest.mark.asyncio
    async def test_a_new_run_starts_without_the_previous_overlay(self, scripted_agents_model):
        """``on_start`` resets the compactor, so a retried / follow-up run that
        re-reads the session is never spliced with a stale prefix."""
        model = scripted_agents_model(["tool", "final", "tool", "final"])
        node = _CompactingFakeNode(budget_tokens=1, once=True)
        rewriter = self._rewriter(node)
        await self._run(model, rewriter)
        assert rewriter.compactions == 1
        await self._run(model, rewriter)
        # Run 2, call 1: plain prompt, no overlay, and no compaction attempt
        # (the first call of a run never compacts); the per-run counter was
        # reset by ``on_start``.
        assert model.inputs[2] == [{"role": "user", "content": "start"}]
        assert rewriter.compactions == 0
        assert "SUMMARY" not in json.dumps(model.inputs[3])

    def test_rewriter_alone_installs_the_filter(self):
        node = _CompactingFakeNode(budget_tokens=1)
        rc = LLMBaseModel._build_run_config(MagicMock(), context_rewriter=self._rewriter(node))
        assert rc is not None and rc.call_model_input_filter is not None
