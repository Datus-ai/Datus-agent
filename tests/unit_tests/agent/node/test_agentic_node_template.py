# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the unified ``AgenticNode.execute_stream`` template method.

These tests target the template skeleton (hook ordering, initial / final
action emission, ``execution_mode`` session gating, ``_build_error_result``
adapter, ``user_message_override`` handling) using a minimal fake subclass
that only implements ``result_class`` + ``_build_success_result``.

External LLM calls are stubbed out via ``model.generate_with_tools_stream``
so the test stays deterministic and does not require API keys.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional

import pytest
from pydantic import BaseModel

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class FakeInput(BaseInput):
    user_message: str = ""
    prompt_version: Optional[str] = None


class FakeResult(BaseModel):
    success: bool = True
    error: Optional[str] = None
    response: str = ""
    tokens_used: int = 0


class FakeResultWithHistory(BaseModel):
    success: bool = True
    error: Optional[str] = None
    response: str = ""
    tokens_used: int = 0
    action_history: List[Any] = []


class _StreamModel:
    """Minimal stand-in for ``LLMBaseModel`` that yields a scripted stream."""

    def __init__(self, actions_per_call: List[List[ActionHistory]]):
        self._scripts = list(actions_per_call)
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_kwargs: dict = {}

    async def generate_with_tools_stream(self, *args, **kwargs) -> AsyncGenerator[ActionHistory, None]:
        self.call_count += 1
        self.last_prompt = kwargs.get("prompt")
        self.last_kwargs = kwargs
        script = self._scripts.pop(0) if self._scripts else []
        ahm = kwargs.get("action_history_manager")
        for action in script:
            if ahm is not None:
                ahm.add_action(action)
            yield action


class FakeAgenticNode(AgenticNode):
    """Concrete subclass: declares ``result_class`` + ``_build_success_result``.

    Bypasses the real ``Node`` / ``AgenticNode`` constructors so we can drop
    a stub into ``self.model`` without dragging in the full agent_config /
    permission stack.
    """

    result_class = FakeResult

    def __init__(self, model: _StreamModel, execution_mode: str = "workflow"):
        # Skip the heavy AgenticNode.__init__ — we only need the attributes the
        # template touches.
        self.execution_mode = execution_mode
        self.input: Optional[FakeInput] = None
        self.tools: List[Any] = []
        self.mcp_servers: dict = {}
        self.actions: List[ActionHistory] = []
        self.result: Optional[FakeResult] = None
        self.max_turns = 5
        self.interrupt_controller = _NoopInterrupt()
        self.permission_hooks = None
        self.skill_manager = None
        self.before_calls: List[str] = []
        self._pinned_model = model

    @property
    def model(self):
        return self._pinned_model

    @model.setter
    def model(self, value):
        self._pinned_model = value

    def get_node_name(self) -> str:
        return "fake"

    def _auto_compact(self):
        async def _noop():
            return False

        return _noop()

    def _get_or_create_session(self):
        return "fake_session"

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        return f"sys[{prompt_version or '-'}]"

    def _build_enhanced_message(self, user_input: FakeInput, extra_enhanced_parts=None) -> str:
        return f"enhanced({user_input.user_message})"

    def _compose_hooks(self, extra=None):
        return extra

    def _format_execution_error(self, exc: BaseException) -> str:
        return f"err[{type(exc).__name__}]: {exc}"

    @staticmethod
    def _extract_total_tokens(actions) -> int:
        return 0

    def _build_success_result(self, ctx: StreamRunContext) -> FakeResult:
        return FakeResult(
            success=True,
            response=str(ctx.response_content) if ctx.response_content else "",
            tokens_used=0,
        )

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        self.before_calls.append(ctx.user_input.user_message)


class _NoopInterrupt:
    def reset(self):
        return None

    def check(self):
        return None


def _ok_action(content: str) -> ActionHistory:
    return ActionHistory(
        action_id="fake_assistant",
        role=ActionRole.ASSISTANT,
        action_type="response",
        messages="ok",
        input={},
        output={"content": content},
        status=ActionStatus.SUCCESS,
    )


def _thinking_action(content: str) -> ActionHistory:
    return ActionHistory(
        action_id="fake_thinking",
        role=ActionRole.ASSISTANT,
        action_type="response",
        messages="thinking",
        input={},
        output={"content": content, "is_thinking": True},
        status=ActionStatus.SUCCESS,
    )


def _tool_action(summary: str) -> ActionHistory:
    return ActionHistory(
        action_id="fake_tool",
        role=ActionRole.TOOL,
        action_type="my_tool",
        messages="tool",
        input={"function_name": "my_tool", "arguments": "{}"},
        output={"success": True, "summary": summary},
        status=ActionStatus.SUCCESS,
    )


# ---------------------------------------------------------------------------
# Template-method behavioural tests
# ---------------------------------------------------------------------------


class TestTemplateMethodHappyPath:
    @pytest.mark.asyncio
    async def test_initial_and_final_actions_have_canonical_types(self):
        model = _StreamModel([[_ok_action("done")]])
        node = FakeAgenticNode(model)
        node.input = FakeInput(user_message="hello")

        ahm = ActionHistoryManager()
        emitted: List[ActionHistory] = []
        async for action in node.execute_stream(ahm):
            emitted.append(action)

        assert emitted[0].role == ActionRole.USER
        assert emitted[0].action_type == "fake_request"
        assert emitted[-1].role == ActionRole.ASSISTANT
        assert emitted[-1].action_type == "fake_response"
        assert emitted[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_response_content_collected_from_assistant(self):
        model = _StreamModel([[_ok_action("hello world")]])
        node = FakeAgenticNode(model)
        node.input = FakeInput(user_message="hello")

        ahm = ActionHistoryManager()
        actions = [a async for a in node.execute_stream(ahm)]

        assert node.result is not None
        assert node.result.response == "hello world"
        assert actions[-1].output["response"] == "hello world"

    @pytest.mark.asyncio
    async def test_thinking_chunks_are_dropped_from_response(self):
        # Two thinking chunks followed by a real tool result — only the tool
        # summary should fall through to the response (via _build_success_result's
        # last_tool_summary fallback, which our fake here ignores). We assert
        # the thinking chunks did not poison ctx.response_content.
        model = _StreamModel([[_thinking_action("inner monologue"), _tool_action("done"), _thinking_action("more")]])
        node = FakeAgenticNode(model)
        node.input = FakeInput(user_message="hi")

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        # Fake's _build_success_result returns ctx.response_content directly;
        # thinking-filtered means it stays empty.
        assert node.result is not None
        assert node.result.response == ""


class TestExecutionModeGuard:
    @pytest.mark.asyncio
    async def test_workflow_mode_also_builds_session(self):
        """Workflow mode must also build a session so ``--resume`` / API chat
        callers receive prior items via SDK; only the interactive *capability*
        (broker prompts, ask_user tool) is gated on ``execution_mode``."""
        model = _StreamModel([[_ok_action("x")]])
        node = FakeAgenticNode(model, execution_mode="workflow")
        node.input = FakeInput(user_message="m")
        calls = {"count": 0}

        original = node._get_or_create_session

        def _spy():
            calls["count"] += 1
            return original()

        node._get_or_create_session = _spy  # type: ignore[assignment]

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_interactive_mode_builds_session_once(self):
        model = _StreamModel([[_ok_action("x")]])
        node = FakeAgenticNode(model, execution_mode="interactive")
        node.input = FakeInput(user_message="m")
        calls = {"count": 0}

        original = node._get_or_create_session

        def _spy():
            calls["count"] += 1
            return original()

        node._get_or_create_session = _spy  # type: ignore[assignment]

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        assert calls["count"] == 1


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_missing_input_raises_datus_exception(self):
        node = FakeAgenticNode(_StreamModel([]))
        node.input = None

        with pytest.raises(DatusException, match="Missing required field"):
            async for _ in node.execute_stream():
                pass


class TestUserMessageOverride:
    """``ctx.user_message_override`` set in _before_stream redirects the user prompt."""

    @pytest.mark.asyncio
    async def test_override_swaps_user_message_for_enhanced_message(self):
        model = _StreamModel([[_ok_action("ok")]])

        class _OverrideNode(FakeAgenticNode):
            async def _before_stream(self, ctx):
                ctx.user_message_override = "INJECTED PROMPT"

        node = _OverrideNode(model)
        node.input = FakeInput(user_message="ORIGINAL")

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        # ``_build_enhanced_message`` was called with the override text in
        # place — and the original message was restored afterwards.
        assert model.last_prompt == "enhanced(INJECTED PROMPT)"
        assert node.input.user_message == "ORIGINAL"


class TestErrorPath:
    @pytest.mark.asyncio
    async def test_unhandled_exception_emits_error_action_and_failed_result(self):
        class _BoomModel:
            async def generate_with_tools_stream(self, *args, **kwargs):
                raise RuntimeError("model exploded")
                yield  # unreachable; preserves async-generator type

        node = FakeAgenticNode(_BoomModel())
        node.input = FakeInput(user_message="m")

        emitted = [a async for a in node.execute_stream(ActionHistoryManager())]

        assert emitted[-1].action_type == "error"
        assert emitted[-1].status == ActionStatus.FAILED
        assert node.result is not None
        assert node.result.success is False
        assert "model exploded" in (node.result.error or "")
        # Error response is the unified empty string, not a node-specific apology.
        assert node.result.response == ""

    @pytest.mark.asyncio
    async def test_error_result_fills_action_history_when_field_exists(self):
        class _BoomModel:
            async def generate_with_tools_stream(self, *args, **kwargs):
                raise ValueError("bang")
                yield

        class _NodeWithHistory(FakeAgenticNode):
            result_class = FakeResultWithHistory

            def _build_success_result(self, ctx):
                return FakeResultWithHistory(success=True, response="")

        node = _NodeWithHistory(_BoomModel())
        node.input = FakeInput(user_message="m")

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        assert isinstance(node.result, FakeResultWithHistory)
        # action_history populated with the actions emitted before the failure
        # (initial USER action at minimum).
        assert isinstance(node.result.action_history, list)
        assert len(node.result.action_history) >= 1

    @pytest.mark.asyncio
    async def test_missing_result_class_raises_not_implemented(self):
        class _NodeWithoutResultClass(FakeAgenticNode):
            result_class = None

            def _build_success_result(self, ctx):
                return None  # unreachable for this test

        class _BoomModel:
            async def generate_with_tools_stream(self, *args, **kwargs):
                raise RuntimeError("model exploded")
                yield

        node = _NodeWithoutResultClass(_BoomModel())
        node.input = FakeInput(user_message="m")

        # The template catches the model failure and calls _build_error_result,
        # which raises NotImplementedError because result_class is unset.
        with pytest.raises(NotImplementedError, match="result_class"):
            async for _ in node.execute_stream(ActionHistoryManager()):
                pass


class TestHookCallOrder:
    @pytest.mark.asyncio
    async def test_before_stream_runs_before_session_setup(self):
        model = _StreamModel([[_ok_action("ok")]])
        order: List[str] = []

        class _OrderNode(FakeAgenticNode):
            async def _before_stream(self, ctx):
                order.append("before")

            def _get_or_create_session(self):
                order.append("session")
                return "s"

            def _build_template_context(self, ctx):
                order.append("template_ctx")
                return None

            def _build_success_result(self, ctx):
                order.append("success_result")
                return FakeResult()

        node = _OrderNode(model, execution_mode="interactive")
        node.input = FakeInput(user_message="m")

        async for _ in node.execute_stream(ActionHistoryManager()):
            pass

        # Strict ordering contract: side-effect hooks run before session,
        # template context after session, success result at the very end.
        assert order == ["before", "session", "template_ctx", "success_result"]
