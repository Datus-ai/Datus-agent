# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for the openai_compatible overflow → compact_callback fallback.

When ``MaxTurnsExceeded`` fires and the caller wires in a ``compact_callback``
(typically ``AgenticNode.compact``), the model layer must:

1. Invoke the callback in ``mode="major"`` so the session shrinks before retry.
2. Retry the run exactly once — never enter a loop where compact and retry
   keep failing.
3. If no callback is wired, propagate the original failure unchanged.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.exceptions import MaxTurnsExceeded

from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.exceptions import DatusException


def _make_model() -> OpenAICompatibleModel:
    """Build a model with __init__ bypassed — we only exercise the overflow
    branch in ``_generate_with_tools_internal`` and don't need a real client.
    """
    with patch.object(OpenAICompatibleModel, "__init__", lambda self, *a, **kw: None):
        m = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
    m.model_name = "test-model"
    return m


def _success_run_result(text: str = "ok"):
    """Build a duck-typed ``Runner.run`` result that satisfies the post-success
    accessors (`final_output`, `to_input_list`, `context_wrapper.usage`,
    `turn_count`)."""
    result = MagicMock()
    result.final_output = text
    result.to_input_list = MagicMock(return_value=[])
    result.context_wrapper = MagicMock()
    result.context_wrapper.usage = MagicMock()
    result.turn_count = 1
    return result


@asynccontextmanager
async def _noop_mcp_cm(*args, **kwargs):
    """Async context manager substitute for ``multiple_mcp_servers``."""
    yield {}


@pytest.fixture
def model_with_safe_helpers():
    """Stub the non-overflow helpers so the test focuses on the exception path."""
    m = _make_model()
    m._setup_custom_json_encoder = MagicMock()
    m._build_agent = MagicMock(return_value=MagicMock())
    m._build_run_config = MagicMock(return_value=None)

    # Bypass retry wrapper: invoke the inner coroutine directly so a single
    # MaxTurnsExceeded path doesn't get wrapped in retries that interfere
    # with our test assertions about call counts.
    async def _direct_call(fn, _label):
        return await fn()

    m._with_retry_async = _direct_call
    m._extract_usage_info = MagicMock(return_value={})
    # Stub the success-path trace hook so post-success accessors don't trip
    # on missing ``model_config`` (the real init wires that up).
    m._save_llm_trace = MagicMock()
    return m


def _patch_module():
    """Common patcher: stub the MCP CM + extract_sql_contexts so the
    ``async with multiple_mcp_servers(...)`` block runs cleanly.
    """
    return patch.multiple(
        "datus.models.openai_compatible",
        multiple_mcp_servers=_noop_mcp_cm,
        extract_sql_contexts=MagicMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_overflow_invokes_compact_callback_and_retries(model_with_safe_helpers):
    m = model_with_safe_helpers
    compact_cb = AsyncMock()
    success = _success_run_result("ok")
    # First Runner.run raises overflow; second succeeds.
    with _patch_module(), patch("datus.models.openai_compatible.Runner") as runner:
        runner.run = AsyncMock(side_effect=[MaxTurnsExceeded("boom"), success])
        result = await m._generate_with_tools_internal(
            prompt="hi",
            mcp_servers={},
            tools=None,
            instruction="",
            output_type=str,
            strict_json_schema=False,
            max_turns=2,
            session=None,
            compact_callback=compact_cb,
        )
    assert result["content"] == "ok"
    # The callback must be invoked once, with mode="major" and reason="overflow"
    # — the contract the model layer promises to its agent caller.
    compact_cb.assert_awaited_once_with(mode="major", reason="overflow")
    # Runner.run was called twice: first failed, then retry succeeded.
    assert runner.run.await_count == 2


@pytest.mark.asyncio
async def test_overflow_without_callback_raises(model_with_safe_helpers):
    m = model_with_safe_helpers
    with _patch_module(), patch("datus.models.openai_compatible.Runner") as runner:
        runner.run = AsyncMock(side_effect=MaxTurnsExceeded("boom"))
        with pytest.raises(DatusException) as exc_info:
            await m._generate_with_tools_internal(
                prompt="hi",
                mcp_servers={},
                tools=None,
                instruction="",
                output_type=str,
                strict_json_schema=False,
                max_turns=2,
                session=None,
            )
    # Without a callback there's no recovery — propagate the original failure.
    # DatusException renders as "error_code=300022, error_message=...";
    # check the code prefix rather than the enum to keep the test resilient.
    assert "300022" in str(exc_info.value)


@pytest.mark.asyncio
async def test_overflow_retry_failure_raises(model_with_safe_helpers):
    """If compact succeeds but the retry still overflows, the error must
    surface — we never silently swallow a persistent overflow.
    """
    m = model_with_safe_helpers
    compact_cb = AsyncMock()
    with _patch_module(), patch("datus.models.openai_compatible.Runner") as runner:
        runner.run = AsyncMock(side_effect=[MaxTurnsExceeded("boom"), MaxTurnsExceeded("boom2")])
        with pytest.raises(DatusException) as exc_info:
            await m._generate_with_tools_internal(
                prompt="hi",
                mcp_servers={},
                tools=None,
                instruction="",
                output_type=str,
                strict_json_schema=False,
                max_turns=2,
                session=None,
                compact_callback=compact_cb,
            )
    # DatusException renders as "error_code=300022, error_message=...";
    # check the code prefix rather than the enum to keep the test resilient.
    assert "300022" in str(exc_info.value)
    # Retried exactly once — three or more calls would mean the loop guard is broken.
    assert runner.run.await_count == 2


@pytest.mark.asyncio
async def test_overflow_compact_callback_failure_raises(model_with_safe_helpers):
    """If compact_callback itself raises, surface the original MaxTurnsExceeded
    rather than masking it. The user should see "context overflow", not "compact
    crashed" — the latter would mislead the operator about the root cause.
    """
    m = model_with_safe_helpers
    compact_cb = AsyncMock(side_effect=RuntimeError("compact broke"))
    with _patch_module(), patch("datus.models.openai_compatible.Runner") as runner:
        runner.run = AsyncMock(side_effect=MaxTurnsExceeded("boom"))
        with pytest.raises(DatusException) as exc_info:
            await m._generate_with_tools_internal(
                prompt="hi",
                mcp_servers={},
                tools=None,
                instruction="",
                output_type=str,
                strict_json_schema=False,
                max_turns=2,
                session=None,
                compact_callback=compact_cb,
            )
    # DatusException renders as "error_code=300022, error_message=...";
    # check the code prefix rather than the enum to keep the test resilient.
    assert "300022" in str(exc_info.value)
    compact_cb.assert_awaited_once()
    # Runner.run NOT retried because the callback failed first.
    assert runner.run.await_count == 1
