# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``RetryPolicy`` implementations.

The retry-policy contract is what ``AgenticNode.execute_stream`` uses to
drive its generic retry loop. Two concrete policies ship today:

* :class:`NoRetryPolicy` — single-shot, default for every node that does
  not override ``_get_retry_policy``.
* :class:`ValidationHookRetryPolicy` — re-prompts the model when the
  Deliverable node's ``ValidationHook.final_report`` flags a blocking
  failure; lives next to the node that owns it.
These are pure-function tests; no LLM, no DB, no agent_config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from datus.agent.node.deliverable_node import ValidationHookRetryPolicy
from datus.agent.node.retry_policy import NoRetryPolicy, RetryPolicy
from datus.agent.node.stream_run_context import StreamRunContext
from datus.schemas.action_history import ActionHistoryManager

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCheck:
    name: str
    passed: bool
    error: str = ""


@dataclass
class _FakeReport:
    """Just enough of ``ValidationReport`` for the policy under test."""

    checks: List[_FakeCheck] = field(default_factory=list)

    def has_blocking_failure(self) -> bool:
        return any(not c.passed for c in self.checks)

    def model_dump(self, **_kwargs) -> dict:
        return {"checks": [c.__dict__ for c in self.checks]}


class _FakeValidationHook:
    """Stand-in for ``ValidationHook`` that replays a scripted sequence of reports.

    The policy under test calls ``final_report`` after each attempt, then
    ``reset_session`` when scheduling a retry. We mirror the real hook's
    contract: ``reset_session`` clears the current report; ``final_report``
    advances to the next scripted entry on every fresh call.
    """

    def __init__(self, reports: List[Optional[_FakeReport]]):
        self._scripted = list(reports)
        self._cursor = 0
        self._current: Optional[_FakeReport] = None
        self.reset_calls = 0
        self._advance()

    def _advance(self) -> None:
        if self._cursor < len(self._scripted):
            self._current = self._scripted[self._cursor]
            self._cursor += 1
        else:
            self._current = None

    @property
    def final_report(self) -> Optional[_FakeReport]:
        return self._current

    @property
    def session_targets(self) -> List[object]:
        return []

    def reset_session(self) -> None:
        self.reset_calls += 1
        self._advance()


# Tiny ``StreamRunContext`` factory — the policies only touch ``attempt`` /
# ``extras``, so we sidestep the dataclass's mandatory user_input field.
def _make_ctx(attempt: int = 1) -> StreamRunContext:
    return StreamRunContext(
        user_input=None,  # type: ignore[arg-type]
        action_history_manager=ActionHistoryManager(),
        attempt=attempt,
    )


# ---------------------------------------------------------------------------
# NoRetryPolicy
# ---------------------------------------------------------------------------


class TestNoRetryPolicy:
    def test_max_attempts_is_one(self):
        assert NoRetryPolicy().max_attempts == 1

    def test_reset_is_noop(self):
        ctx = _make_ctx()
        ctx.extras["sentinel"] = "kept"
        NoRetryPolicy().reset(ctx)
        assert ctx.extras == {"sentinel": "kept"}

    def test_should_retry_always_false(self):
        ctx = _make_ctx()
        assert NoRetryPolicy().should_retry(ctx) is False

    def test_next_prompt_is_none(self):
        ctx = _make_ctx()
        assert NoRetryPolicy().next_prompt(ctx) is None

    def test_on_retry_actions_is_empty(self):
        ctx = _make_ctx()
        assert list(NoRetryPolicy().on_retry_actions(ctx)) == []

    def test_finalise_is_noop(self):
        ctx = _make_ctx()
        NoRetryPolicy().finalise(ctx)
        assert ctx.extras == {}

    def test_satisfies_retry_policy_protocol(self):
        # ``runtime_checkable`` Protocol — instance check confirms the
        # default policy still fits the template's contract.
        assert isinstance(NoRetryPolicy(), RetryPolicy)


# ---------------------------------------------------------------------------
# ValidationHookRetryPolicy
# ---------------------------------------------------------------------------


class TestValidationHookRetryPolicy:
    def test_should_retry_returns_false_when_no_report(self):
        hook = _FakeValidationHook([None])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert policy.should_retry(_make_ctx()) is False

    def test_should_retry_returns_false_when_report_passes(self):
        hook = _FakeValidationHook([_FakeReport(checks=[_FakeCheck("ok", passed=True)])])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert policy.should_retry(_make_ctx()) is False

    def test_should_retry_returns_true_on_blocking_failure(self):
        failing = _FakeReport(checks=[_FakeCheck("bad", passed=False, error="boom")])
        hook = _FakeValidationHook([failing])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert policy.should_retry(_make_ctx()) is True
        # Internal report captured (as a serialised dict) so finalise can
        # surface the same content to ``ctx.extras["validation_report"]``.
        assert policy._blocking_report == {"checks": [{"name": "bad", "passed": False, "error": "boom"}]}

    def test_reset_clears_blocking_report_and_session(self):
        failing = _FakeReport(checks=[_FakeCheck("bad", passed=False, error="boom")])
        hook = _FakeValidationHook([failing])
        policy = ValidationHookRetryPolicy(hook=hook)
        policy.should_retry(_make_ctx())
        assert policy._blocking_report == {"checks": [{"name": "bad", "passed": False, "error": "boom"}]}
        policy.reset(_make_ctx())
        assert policy._blocking_report is None
        # ``reset`` must also clear hook state so the next attempt starts fresh.
        assert hook.reset_calls == 1

    def test_next_prompt_returns_none_when_no_report(self):
        # Defensive branch: ``next_prompt`` may be invoked even when
        # ``should_retry`` saw no report. The policy returns ``None`` rather
        # than building a prompt against missing state.
        hook = _FakeValidationHook([None])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert policy.next_prompt(_make_ctx()) is None

    def test_next_prompt_builds_retry_text(self, monkeypatch):
        # ``build_retry_prompt`` requires a richer report than our minimal
        # fake; stub it out — the contract is that the policy delegates and
        # returns the string.
        from datus.agent.node import deliverable_node

        monkeypatch.setattr(deliverable_node, "build_retry_prompt", lambda report, targets: "STUB RETRY PROMPT")
        failing = _FakeReport(checks=[_FakeCheck("bad", passed=False, error="boom")])
        hook = _FakeValidationHook([failing])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert policy.next_prompt(_make_ctx()) == "STUB RETRY PROMPT"

    def test_on_retry_actions_yields_nothing(self):
        # Pre-refactor Deliverable emitted no user-visible action between
        # retries — the policy preserves that.
        hook = _FakeValidationHook([None])
        policy = ValidationHookRetryPolicy(hook=hook)
        assert list(policy.on_retry_actions(_make_ctx())) == []

    def test_finalise_records_blocking_report_in_ctx_extras(self):
        failing = _FakeReport(checks=[_FakeCheck("bad", passed=False, error="boom")])
        hook = _FakeValidationHook([failing, failing, failing])
        policy = ValidationHookRetryPolicy(hook=hook, max_attempts=2)
        ctx = _make_ctx()
        policy.should_retry(ctx)  # populates _blocking_report
        policy.finalise(ctx)
        assert ctx.extras["blocked"] is True
        assert ctx.extras["validation_report"]["checks"][0]["error"] == "boom"

    def test_finalise_records_passing_report_when_no_block(self):
        passing = _FakeReport(checks=[_FakeCheck("ok", passed=True)])
        hook = _FakeValidationHook([passing])
        policy = ValidationHookRetryPolicy(hook=hook)
        ctx = _make_ctx()
        policy.should_retry(ctx)
        policy.finalise(ctx)
        assert ctx.extras["blocked"] is False
        assert ctx.extras["validation_report"]["checks"][0]["passed"] is True

    def test_finalise_with_missing_report(self):
        # When the hook never produced a report, ``validation_report`` is
        # ``None`` and ``blocked`` is ``False``.
        hook = _FakeValidationHook([None])
        policy = ValidationHookRetryPolicy(hook=hook)
        ctx = _make_ctx()
        policy.finalise(ctx)
        assert ctx.extras["blocked"] is False
        assert ctx.extras["validation_report"] is None

    def test_min_max_attempts_floor(self):
        # Even when caller passes ``0``, the policy floors to 1 so the
        # template loop runs the initial attempt.
        hook = _FakeValidationHook([None])
        policy = ValidationHookRetryPolicy(hook=hook, max_attempts=0)
        assert policy.max_attempts == 1


# ---------------------------------------------------------------------------
# Protocol surface — regression-guard against renames the template depends on.
# ---------------------------------------------------------------------------


class TestPolicyProtocolSurface:
    @pytest.mark.parametrize(
        "policy",
        [
            NoRetryPolicy(),
            ValidationHookRetryPolicy(hook=_FakeValidationHook([None])),
        ],
    )
    def test_required_methods_present(self, policy):
        assert hasattr(policy, "max_attempts")
        assert callable(policy.reset)
        assert callable(policy.should_retry)
        assert callable(policy.next_prompt)
        assert callable(policy.on_retry_actions)
        assert callable(policy.finalise)
