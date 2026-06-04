# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Direct unit tests for ``_build_success_result`` fallback paths.

After the ``execute_stream`` template refactor, every subclass owns a
``_build_success_result(ctx)`` hook that constructs the typed
:class:`NodeResult`. The end-to-end execution tests cover the happy path
where ``ctx.response_content`` is already populated by the streaming
loop. The fallback branches — used when the assistant's content channel
is empty but a prior tool call left material in
``ctx.last_successful_output`` — only fire when the model returns no
final assistant message.

Building a full node + LLM mock for each of those edge cases is heavy.
This module exercises each subclass's ``_build_success_result`` and
``_extract_total_tokens`` directly: we sidestep the heavy ``__init__``
via ``__new__`` and only set the attributes the hook actually touches.
"""

from __future__ import annotations

from typing import Any

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.node.compare_agentic_node import CompareAgenticNode
from datus.agent.node.explore_agentic_node import ExploreAgenticNode
from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_node(cls: type, **attrs: Any):
    """Construct ``cls`` without running ``__init__``.

    ``_build_success_result`` only reads a handful of attributes
    (``execution_mode`` and a few tool slots). Bypassing the real
    constructor keeps these tests independent of agent_config, datasource
    connectivity, and prompt-template files.
    """
    obj = cls.__new__(cls)
    obj.execution_mode = "workflow"
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _ctx(
    *,
    response_content: Any = "",
    last_successful_output: Any = None,
    last_tool_summary: str = "",
    actions: list | None = None,
    extras: dict | None = None,
) -> StreamRunContext:
    ahm = ActionHistoryManager()
    if actions:
        for a in actions:
            ahm.add_action(a)
    ctx = StreamRunContext(
        user_input=None,  # type: ignore[arg-type]
        action_history_manager=ahm,
    )
    ctx.response_content = response_content
    ctx.last_successful_output = last_successful_output
    ctx.last_tool_summary = last_tool_summary
    if extras:
        ctx.extras.update(extras)
    return ctx


def _assistant_action(content: str = "", *, depth: int = 0, usage: dict | None = None) -> ActionHistory:
    output: dict = {"content": content}
    if usage is not None:
        output["usage"] = usage
    return ActionHistory(
        action_id="a",
        role=ActionRole.ASSISTANT,
        action_type="response",
        messages="ok",
        input={},
        output=output,
        status=ActionStatus.SUCCESS,
        depth=depth,
    )


def _user_action(*, depth: int = 0) -> ActionHistory:
    return ActionHistory(
        action_id="u",
        role=ActionRole.USER,
        action_type="request",
        messages="user",
        input={},
        output={},
        status=ActionStatus.SUCCESS,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# AgenticNode._extract_total_tokens — static helper edge cases
# ---------------------------------------------------------------------------


class TestExtractTotalTokens:
    """The reverse-scan helper has several "skip this action" branches that
    keep the per-turn total robust against malformed usage payloads."""

    def test_returns_zero_for_empty(self):
        assert AgenticNode._extract_total_tokens([]) == 0

    def test_skips_root_user_break(self):
        # Stops at root user action — earlier assistant usage is out of scope.
        actions = [
            _assistant_action("earlier", usage={"total_tokens": 99}),
            _user_action(),
            _assistant_action("later", usage={"total_tokens": 100}),
        ]
        assert AgenticNode._extract_total_tokens(actions) == 100

    def test_skips_non_dict_output(self):
        # Assistant action whose ``output`` is not a dict must not crash.
        act = ActionHistory(
            action_id="a",
            role=ActionRole.ASSISTANT,
            action_type="response",
            messages="ok",
            input={},
            output="raw string output",  # not a dict
            status=ActionStatus.SUCCESS,
            depth=0,
        )
        # Followed by a valid one so we know the scan kept walking.
        good = _assistant_action("x", usage={"total_tokens": 42})
        assert AgenticNode._extract_total_tokens([good, act]) == 42

    def test_skips_action_with_no_total_tokens(self):
        # ``usage`` present but ``total_tokens`` missing → keep scanning.
        no_total = _assistant_action("x", usage={"prompt_tokens": 5})
        good = _assistant_action("y", usage={"total_tokens": 17})
        assert AgenticNode._extract_total_tokens([good, no_total]) == 17

    def test_skips_action_with_zero_total(self):
        # ``total_tokens == 0`` is falsy — skipped so a later (older) entry can win.
        zero = _assistant_action("x", usage={"total_tokens": 0})
        good = _assistant_action("y", usage={"total_tokens": 7})
        assert AgenticNode._extract_total_tokens([good, zero]) == 7

    def test_skips_non_numeric_total(self):
        # Some providers emit ``"NaN"`` / objects — int cast fails and the
        # loop continues. Ensures the TypeError/ValueError branch executes.
        bad = _assistant_action("x", usage={"total_tokens": "not-a-number"})
        good = _assistant_action("y", usage={"total_tokens": 11})
        assert AgenticNode._extract_total_tokens([good, bad]) == 11

    def test_accepts_string_total(self):
        # Numeric strings ("1234") should still count — the cast succeeds.
        act = _assistant_action("x", usage={"total_tokens": "55"})
        assert AgenticNode._extract_total_tokens([act]) == 55

    def test_returns_zero_when_only_unusable_entries(self):
        actions = [_assistant_action("x", usage={"total_tokens": None})]
        assert AgenticNode._extract_total_tokens(actions) == 0


# ---------------------------------------------------------------------------
# ChatAgenticNode — last_successful_output content fallback (line 509)
# ---------------------------------------------------------------------------


class TestChatBuildSuccessResultFallback:
    def test_falls_back_to_content_string_from_last_output(self):
        node = _bare_node(ChatAgenticNode)
        ctx = _ctx(last_successful_output={"content": "tool said hi"})
        result = node._build_success_result(ctx)
        assert result.response == "tool said hi"

    def test_falls_back_to_str_when_candidate_non_string(self):
        node = _bare_node(ChatAgenticNode)
        ctx = _ctx(last_successful_output={"content": {"key": "value"}})
        result = node._build_success_result(ctx)
        # Non-string candidates are stringified.
        assert "key" in result.response and "value" in result.response

    def test_falls_back_to_last_tool_summary(self):
        node = _bare_node(ChatAgenticNode)
        ctx = _ctx(last_tool_summary="tool summary text")
        result = node._build_success_result(ctx)
        assert result.response == "tool summary text"

    def test_falls_back_to_summary_report_action(self):
        node = _bare_node(ChatAgenticNode)
        summary_action = ActionHistory(
            action_id="sr",
            role=ActionRole.ASSISTANT,
            action_type="summary_report",
            messages="summary",
            input={},
            output={"markdown": "**Summary** content"},
            status=ActionStatus.SUCCESS,
        )
        ctx = _ctx(actions=[summary_action])
        result = node._build_success_result(ctx)
        assert "Summary" in result.response


# ---------------------------------------------------------------------------
# FeedbackAgenticNode (lines 209-223)
# ---------------------------------------------------------------------------


class TestFeedbackBuildSuccessResultFallback:
    def test_falls_back_to_content_string(self):
        node = _bare_node(FeedbackAgenticNode)
        ctx = _ctx(last_successful_output={"content": "feedback text"})
        result = node._build_success_result(ctx)
        assert result.response == "feedback text"

    def test_falls_back_to_response_key(self):
        # ``content`` empty, ``response`` carries the body.
        node = _bare_node(FeedbackAgenticNode)
        ctx = _ctx(last_successful_output={"content": "", "response": "via response key"})
        result = node._build_success_result(ctx)
        assert result.response == "via response key"

    def test_falls_back_to_raw_output_when_dict_candidate(self):
        # Non-string candidate hits the ``str(candidate)`` branch.
        node = _bare_node(FeedbackAgenticNode)
        ctx = _ctx(last_successful_output={"raw_output": {"k": "v"}})
        result = node._build_success_result(ctx)
        assert "k" in result.response and "v" in result.response

    def test_coerces_non_string_response_content(self):
        # ``ctx.response_content`` is a non-string, no last_successful_output.
        node = _bare_node(FeedbackAgenticNode)
        ctx = _ctx(response_content={"shape": "dict"})
        result = node._build_success_result(ctx)
        assert isinstance(result.response, str)
        assert "shape" in result.response

    def test_interactive_mode_counts_tokens(self):
        node = _bare_node(FeedbackAgenticNode, execution_mode="interactive")
        actions = [_assistant_action("x", usage={"total_tokens": 23})]
        ctx = _ctx(response_content="hi", actions=actions)
        result = node._build_success_result(ctx)
        assert result.tokens_used == 23


# ---------------------------------------------------------------------------
# ExploreAgenticNode (lines 267-275)
# ---------------------------------------------------------------------------


class TestExploreBuildSuccessResultFallback:
    def test_falls_back_to_content(self):
        node = _bare_node(ExploreAgenticNode)
        ctx = _ctx(last_successful_output={"content": "explore result"})
        result = node._build_success_result(ctx)
        assert result.response == "explore result"

    def test_coerces_non_string_response_content(self):
        node = _bare_node(ExploreAgenticNode)
        ctx = _ctx(response_content=123)
        result = node._build_success_result(ctx)
        assert result.response == "123"


# ---------------------------------------------------------------------------
# CompareAgenticNode (lines 217, 237)
# ---------------------------------------------------------------------------


class TestCompareBuildSuccessResultFallback:
    def test_falls_back_to_raw_output(self):
        # ``raw_output`` from a tool result — taken verbatim when
        # ``ctx.response_content`` is empty. ``_parse_comparison_output``
        # expects a JSON dict; route the string through it to mirror the
        # production flow.
        node = _bare_node(CompareAgenticNode)
        ctx = _ctx(last_successful_output={"raw_output": {"explanation": "ok", "suggest": "do X"}})
        # When ``raw_output`` is a dict, ``_parse_comparison_output`` returns it as-is.
        result = node._build_success_result(ctx)
        assert result.explanation == "ok"
        assert result.suggest == "do X"

    def test_unparseable_raw_output_yields_failure_explanation(self):
        # Plain-text raw_output fails JSON parsing — Compare surfaces a
        # diagnostic explanation rather than crashing.
        node = _bare_node(CompareAgenticNode)
        ctx = _ctx(last_successful_output={"raw_output": "not json"})
        result = node._build_success_result(ctx)
        assert "Failed to parse" in result.explanation


# ---------------------------------------------------------------------------
# GenSemanticModelAgenticNode (lines 373-377, 386)
# ---------------------------------------------------------------------------


class TestGenSemanticModelBuildSuccessResultFallback:
    def _build_ctx_with_input(self, **kwargs):
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        ctx = _ctx(**kwargs)
        ctx.user_input = SemanticNodeInput(user_message="m")
        return ctx

    def test_falls_back_to_raw_output_dict(self):
        # ``raw_output`` is a dict — taken verbatim, then stringified for output.
        node = _bare_node(GenSemanticModelAgenticNode, agent_config=None)
        ctx = self._build_ctx_with_input(last_successful_output={"raw_output": {"semantic_model_files": []}})

        # Stub the parser + storage side effect so the test focuses on the
        # response_content fallback chain.
        node._extract_semantic_model_and_output_from_response = lambda payload: ([], None)  # type: ignore[assignment]
        node._finalize_semantic_model_generation = lambda **_kw: None  # type: ignore[assignment]

        result = node._build_success_result(ctx)
        # raw_output dict gets stringified by the trailing ``isinstance`` check.
        assert "semantic_model_files" in result.response

    def test_falls_back_to_str_of_last_successful_output(self):
        # ``raw_output`` falsy and not a dict — ``str(last_successful_output)``.
        node = _bare_node(GenSemanticModelAgenticNode, agent_config=None)
        ctx = self._build_ctx_with_input(last_successful_output={"raw_output": ""})
        node._extract_semantic_model_and_output_from_response = lambda payload: ([], None)  # type: ignore[assignment]
        node._finalize_semantic_model_generation = lambda **_kw: None  # type: ignore[assignment]

        result = node._build_success_result(ctx)
        assert "raw_output" in result.response


# ---------------------------------------------------------------------------
# GenMetricsAgenticNode (lines 385-389, 398)
# ---------------------------------------------------------------------------


class TestGenMetricsBuildSuccessResultFallback:
    def test_falls_back_to_raw_output_dict(self):
        node = _bare_node(GenMetricsAgenticNode, agent_config=None)
        node._extract_metric_and_output_from_response = lambda payload: (None, None, None, None)  # type: ignore[assignment]
        node._finalize_metric_generation = lambda **_kw: None  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"raw_output": {"metric_file": "x.yml"}})
        result = node._build_success_result(ctx)
        assert "metric_file" in result.response

    def test_falls_back_to_str_of_last_successful_output(self):
        node = _bare_node(GenMetricsAgenticNode, agent_config=None)
        node._extract_metric_and_output_from_response = lambda payload: (None, None, None, None)  # type: ignore[assignment]
        node._finalize_metric_generation = lambda **_kw: None  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"raw_output": ""})
        result = node._build_success_result(ctx)
        assert "raw_output" in result.response


# ---------------------------------------------------------------------------
# SqlSummaryAgenticNode (lines 370-374, 383)
# ---------------------------------------------------------------------------


class TestSqlSummaryBuildSuccessResultFallback:
    def test_falls_back_to_raw_output_dict(self):
        node = _bare_node(SqlSummaryAgenticNode, agent_config=None)
        node._extract_sql_summary_and_output_from_response = lambda payload: (None, None)  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"raw_output": {"summary_file": "s.yml"}})
        result = node._build_success_result(ctx)
        assert "summary_file" in result.response

    def test_falls_back_to_str_of_last_successful_output(self):
        node = _bare_node(SqlSummaryAgenticNode, agent_config=None)
        node._extract_sql_summary_and_output_from_response = lambda payload: (None, None)  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"raw_output": ""})
        result = node._build_success_result(ctx)
        assert "raw_output" in result.response


# ---------------------------------------------------------------------------
# GenSQLAgenticNode (lines 728-733, 892-893)
# ---------------------------------------------------------------------------


class TestGenSQLBuildSuccessResultFallback:
    def test_reads_sql_file_when_response_looks_like_path(self, monkeypatch):
        # ``response_content.strip().endswith('.sql')`` triggers the
        # ``_read_existing_sql_file`` branch — the file contents replace
        # ``result.sql`` and a preview is generated.
        node = _bare_node(GenSQLAgenticNode, agent_config=None, filesystem_func_tool=None, node_config={})
        sample_sql = "SELECT 1\nFROM t\nLIMIT 1;"
        node._collect_final_response = lambda _ahm: ("queries/q.sql", "queries/q.sql")  # type: ignore[assignment]
        node._read_existing_sql_file = lambda _p: sample_sql  # type: ignore[assignment]
        ctx = _ctx()
        result = node._build_success_result(ctx)
        assert result.sql_file_path == "queries/q.sql"
        assert result.sql == sample_sql
        assert "SELECT 1" in (result.sql_preview or "")

    def test_no_file_read_when_response_inline_sql(self):
        node = _bare_node(GenSQLAgenticNode, agent_config=None, filesystem_func_tool=None, node_config={})
        node._collect_final_response = lambda _ahm: ("hi", "SELECT 2;")  # type: ignore[assignment]
        ctx = _ctx()
        result = node._build_success_result(ctx)
        assert result.sql == "SELECT 2;"
        assert result.sql_file_path is None

    def test_no_file_read_when_read_returns_none(self):
        # Filesystem read fails (path missing) — keep the raw SQL string.
        node = _bare_node(GenSQLAgenticNode, agent_config=None, filesystem_func_tool=None, node_config={})
        node._collect_final_response = lambda _ahm: ("missing.sql", "missing.sql")  # type: ignore[assignment]
        node._read_existing_sql_file = lambda _p: None  # type: ignore[assignment]
        ctx = _ctx()
        result = node._build_success_result(ctx)
        assert result.sql == "missing.sql"
        assert result.sql_file_path is None
        assert result.sql_preview is None


class TestGenSQLCollectFinalResponseFallback:
    def test_collects_non_string_candidate_via_str(self):
        # When an assistant action's ``content`` is a dict, the
        # ``elif candidate and not isinstance(candidate, str)`` branch
        # stringifies it instead of dropping the response.
        node = _bare_node(GenSQLAgenticNode, agent_config=None)
        ahm = ActionHistoryManager()
        action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            action_type="response",
            messages="ok",
            input={},
            output={"content": {"nested": "value"}},
            status=ActionStatus.SUCCESS,
        )
        ahm.add_action(action)
        response, _ = node._collect_final_response(ahm)
        assert "nested" in response and "value" in response


# ---------------------------------------------------------------------------
# GenReportAgenticNode (lines 481, 487, 499, 517, 522)
# ---------------------------------------------------------------------------


class TestGenReportBuildSuccessResultFallback:
    def test_falls_back_to_content_key(self):
        node = _bare_node(GenReportAgenticNode, agent_config=None)
        # No JSON report → ``_extract_report_from_response`` returns (None, None).
        node._extract_report_from_response = lambda _out: (None, None)  # type: ignore[assignment]
        node._extract_report_result = lambda _actions: None  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"content": "plain report body"})
        result = node._build_success_result(ctx)
        assert "plain report body" in result.response

    def test_uses_extracted_report_when_parser_surfaces_one(self):
        node = _bare_node(GenReportAgenticNode, agent_config=None)
        node._extract_report_from_response = lambda _out: ("parsed report", {"title": "T"})  # type: ignore[assignment]
        node._extract_report_result = lambda _actions: None  # type: ignore[assignment]
        ctx = _ctx(last_successful_output={"content": "raw json"})
        result = node._build_success_result(ctx)
        assert result.response == "parsed report"
        # When ``_extract_report_result`` returns nothing, the parser's
        # metadata fills ``report_result``.
        assert result.report_result == {"title": "T"}

    def test_coerces_non_string_response_content(self):
        node = _bare_node(GenReportAgenticNode, agent_config=None)
        node._extract_report_from_response = lambda _out: (None, None)  # type: ignore[assignment]
        node._extract_report_result = lambda _actions: None  # type: ignore[assignment]
        ctx = _ctx(response_content={"shape": "dict"})
        result = node._build_success_result(ctx)
        assert isinstance(result.response, str)
        assert "shape" in result.response


# ---------------------------------------------------------------------------
# SkillCreatorAgenticNode (lines 351, 359)
# ---------------------------------------------------------------------------


class TestGenSkillBuildSuccessResultFallback:
    def test_falls_back_to_content_key(self):
        node = _bare_node(SkillCreatorAgenticNode)
        ctx = _ctx(last_successful_output={"content": "skill summary"})
        result = node._build_success_result(ctx)
        assert "skill summary" in result.response

    def test_coerces_non_string_response_content(self):
        node = _bare_node(SkillCreatorAgenticNode)
        ctx = _ctx(response_content={"shape": "dict"})
        result = node._build_success_result(ctx)
        assert isinstance(result.response, str)
        assert "shape" in result.response

    def test_extracts_skill_name_and_path_from_last_output(self):
        node = _bare_node(SkillCreatorAgenticNode)
        ctx = _ctx(
            response_content="ok",
            last_successful_output={"skill_name": "search_logs", "skill_path": "skills/search_logs.md"},
        )
        result = node._build_success_result(ctx)
        assert result.skill_name == "search_logs"
        assert result.skill_path == "skills/search_logs.md"


# ---------------------------------------------------------------------------
# Smoke parametrised: every node returns its declared result_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,extra_attrs",
    [
        (ChatAgenticNode, {}),
        (FeedbackAgenticNode, {}),
        (ExploreAgenticNode, {}),
        (SkillCreatorAgenticNode, {}),
    ],
)
def test_build_success_result_returns_declared_result_class(cls, extra_attrs):
    node = _bare_node(cls, **extra_attrs)
    ctx = _ctx(response_content="ok")
    result = node._build_success_result(ctx)
    assert isinstance(result, cls.result_class)
    assert result.success is True
