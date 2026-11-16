# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Results must not recursively snapshot earlier runs in a shared manager."""

from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.node.explore_agentic_node import ExploreAgenticNode
from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode
from datus.agent.node.retry_policy import NoRetryPolicy
from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.cli.execution_state import ExecutionInterrupted
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeResult
from datus.tools.func_tool.semantic_tools import SemanticTools
from tests.unit_tests.agent.node.test_agentic_node_template import FakeAgenticNode, FakeInput, _StreamModel


class _ChatHistoryNode(FakeAgenticNode):
    """Run the real chat result builder through the common streaming template."""

    result_class = ChatNodeResult
    _build_success_result = ChatAgenticNode._build_success_result
    _extract_total_tokens = staticmethod(AgenticNode._extract_total_tokens)

    def _setup_ask_user_tool(self):
        pass


def _action(action_id, *, role=ActionRole.ASSISTANT, action_type="response", output=None):
    return ActionHistory(
        action_id=action_id,
        role=role,
        action_type=action_type,
        messages="done",
        input={},
        output=output or {},
        status=ActionStatus.SUCCESS,
    )


def _query(turn):
    return _action(
        f"query-{turn}",
        role=ActionRole.TOOL,
        action_type="execute_sql",
        output={"success": True, "result": {"columns": ["id"], "data": [[turn]]}},
    )


def _response(turn):
    return _action(
        f"reply-{turn}",
        output={"content": f"answer {turn}", "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_mode", ["interactive", "workflow"])
async def test_shared_manager_stays_linear_over_100_runs(execution_mode):
    manager = ActionHistoryManager()
    node = _ChatHistoryNode(_StreamModel([[_query(i), _response(i)] for i in range(100)]), execution_mode)
    sizes = []
    first_history_size = 0

    for turn in range(100):
        start = manager.checkpoint()
        node.input = FakeInput(user_message=f"question {turn}")
        emitted = [a async for a in node.execute_stream(manager)]
        result = node.result

        assert result.success
        assert result.response == f"answer {turn}"
        assert [a["action_id"] for a in result.action_history] == [a.action_id for a in emitted[:-1]]
        assert result.action_history[1]["output"] == _query(turn).output
        assert result.execution_stats == {
            "total_actions": 3,
            "tool_calls_count": 1,
            "tools_used": ["execute_sql"],
            "total_tokens": 10,
        }
        assert result.tokens_used == 10
        assert emitted[-1].output["action_history"] == result.action_history
        assert manager.get_actions()[start:] == emitted
        assert len(node.actions) == len({a.action_id for a in node.actions}) == 3 * (turn + 1)
        sizes.append(len(json.dumps(emitted[-1].output, default=str)))
        if turn == 0:
            first_history_size = len(json.dumps([a.model_dump() for a in manager.get_actions()], default=str))

    assert len(manager.get_actions()) == 400
    assert max(sizes) < min(sizes) * 1.1
    assert len(json.dumps([a.model_dump() for a in manager.get_actions()], default=str)) < first_history_size * 110


@pytest.mark.asyncio
async def test_failed_run_keeps_its_partial_trace_without_previous_results():
    manager = ActionHistoryManager()
    node = _ChatHistoryNode(_StreamModel([[_query(0), _response(0)]]))
    node.input = FakeInput(user_message="first")
    previous = [a async for a in node.execute_stream(manager)]

    class _FailureModel(_StreamModel):
        async def generate_with_tools_stream(self, *args, **kwargs):
            async for action in super().generate_with_tools_stream(*args, **kwargs):
                yield action
            raise RuntimeError("model failed")

    node.model = _FailureModel([[_query(1)]])
    node.input = FakeInput(user_message="second")
    emitted = [a async for a in node.execute_stream(manager)]

    assert not node.result.success
    assert "model failed" in node.result.error
    assert [a["action_id"] for a in node.result.action_history] == [a.action_id for a in emitted[:-1]]
    assert node.result.action_history[1]["output"] == _query(1).output
    assert emitted[-1].action_type == "error"
    assert emitted[-1].output["action_history"] == node.result.action_history
    assert manager.get_actions()[: len(previous)] == previous
    assert len(node.actions) == len({a.action_id for a in node.actions}) == 6


@pytest.mark.asyncio
async def test_post_build_failure_does_not_append_the_same_actions_twice():
    class _FinalizeFailureNode(_ChatHistoryNode):
        async def _stream_post_build(self, ctx, result):
            yield _action("progress", action_type="finalize_progress")
            raise RuntimeError("finalize failed")

    manager = ActionHistoryManager()
    manager.add_action(_response("old"))
    node = _FinalizeFailureNode(_StreamModel([[_query(1), _response(1)]]))
    node.input = FakeInput(user_message="current")
    emitted = [a async for a in node.execute_stream(manager)]

    assert not node.result.success
    assert "finalize failed" in node.result.error
    assert node.actions == emitted
    assert len(node.actions) == len({a.action_id for a in node.actions}) == 5
    assert [a["action_id"] for a in node.result.action_history] == [a.action_id for a in emitted[:-1]]


@pytest.mark.asyncio
async def test_retry_attempts_share_one_run_boundary():
    class _RetryOnce(NoRetryPolicy):
        max_attempts = 2

        def should_retry(self, ctx):
            return ctx.attempt == 1

        def on_retry_actions(self, ctx):
            return [_action("retry", action_type="retry")]

    manager = ActionHistoryManager()
    old = _response("old")
    manager.add_action(old)
    node = _ChatHistoryNode(_StreamModel([[_query(1), _response(1)], [_query(2), _response(2)]]))
    node.input = FakeInput(user_message="retry question")
    node._get_retry_policy = lambda: _RetryOnce()
    emitted = [a async for a in node.execute_stream(manager)]

    assert node.result.response == "answer 2"
    assert [a["action_id"] for a in node.result.action_history] == [a.action_id for a in emitted[:-1]]
    assert node.result.execution_stats["tool_calls_count"] == 2
    assert manager.get_actions()[0] is old
    assert node.actions == emitted[:-1]


@pytest.mark.asyncio
async def test_cancel_rollback_then_resume_preserves_prior_actions():
    manager = ActionHistoryManager()
    node = _ChatHistoryNode(_StreamModel([[_response(0)]]))
    node.input = FakeInput(user_message="first")
    previous = [a async for a in node.execute_stream(manager)]
    manager_checkpoint = manager.checkpoint()
    node_checkpoint = len(node.actions)

    class _InterruptedModel(_StreamModel):
        async def generate_with_tools_stream(self, *args, **kwargs):
            async for action in super().generate_with_tools_stream(*args, **kwargs):
                yield action
            raise ExecutionInterrupted("cancelled")

    node.model = _InterruptedModel([[_query(1)]])
    node.input = FakeInput(user_message="cancel me")
    with pytest.raises(ExecutionInterrupted):
        async for _ in node.execute_stream(manager):
            pass
    manager.rollback_to(manager_checkpoint)
    del node.actions[node_checkpoint:]
    assert manager.get_actions() == previous

    node.model = _StreamModel([[_response(2)]])
    node.input = FakeInput(user_message="resume")
    resumed = [a async for a in node.execute_stream(manager)]
    assert [a["action_id"] for a in node.result.action_history] == [a.action_id for a in resumed[:-1]]
    assert len(node.actions) == len({a.action_id for a in node.actions}) == 4


@pytest.mark.asyncio
async def test_empty_new_run_does_not_reuse_previous_summary():
    manager = ActionHistoryManager()
    manager.add_action(_action("old-summary", action_type="summary_report", output={"markdown": "old answer"}))
    node = _ChatHistoryNode(_StreamModel([[]]))
    node.input = FakeInput(user_message="new question")
    emitted = [a async for a in node.execute_stream(manager)]
    assert node.result.response == ""
    assert [a["action_id"] for a in node.result.action_history] == [emitted[0].action_id]


def test_action_mutations_do_not_format_payloads_for_logging():
    class _Payload:
        def __repr__(self):
            raise AssertionError("action payload must not be formatted for logging")

    manager = ActionHistoryManager()
    action = _action("opaque", output=_Payload())
    manager.add_action(action)
    replacement = _Payload()
    assert manager.update_action_by_id(action.action_id, output=replacement)
    assert action.output is replacement


@pytest.mark.parametrize(
    "node_class",
    [ChatAgenticNode, AskMetricsAgenticNode, ExploreAgenticNode, SkillCreatorAgenticNode, GenSQLAgenticNode],
)
def test_result_builders_exclude_prior_tools_and_keep_current_query_data(node_class):
    manager = ActionHistoryManager()
    manager.add_action(_query("old"))
    manager.add_action(_response("old"))
    start = manager.checkpoint()
    current = [_query("new"), _response("new")]
    for action in current:
        manager.add_action(action)
    ctx = StreamRunContext(
        user_input=FakeInput(user_message="new question"),
        action_history_manager=manager,
        action_start=start,
        response_content="answer new",
    )
    node = node_class.__new__(node_class)
    node.execution_mode = "workflow"
    result = node._build_success_result(ctx)

    assert result.response == "answer new"
    assert result.action_history == [a.model_dump() for a in current]
    assert result.execution_stats["total_actions"] == 2
    assert result.execution_stats["tool_calls_count"] == 1
    assert result.tokens_used == 10


@pytest.mark.parametrize("has_current_summary", [False, True])
def test_gen_sql_fallback_does_not_return_sql_from_a_prior_run(has_current_summary):
    manager = ActionHistoryManager()
    manager.add_action(_action("old", action_type="summary_report", output={"sql": "SELECT 1", "content": "old"}))
    start = manager.checkpoint()
    if has_current_summary:
        manager.add_action(_action("new", action_type="summary_report", output={"sql": "SELECT 2", "content": "new"}))
    ctx = StreamRunContext(FakeInput(), manager, action_start=start)
    node = GenSQLAgenticNode.__new__(GenSQLAgenticNode)
    result = node._build_success_result(ctx)
    assert result.sql == ("SELECT 2" if has_current_summary else None)
    assert result.response == ("new" if has_current_summary else "")


@pytest.mark.parametrize(
    "node_class,tool_name,payload,result_field,expected",
    [
        (
            GenDashboardAgenticNode,
            "create_dashboard",
            {"id": "current", "name": "new dashboard"},
            "dashboard_result",
            {"dashboards": [{"id": "current", "name": "new dashboard"}]},
        ),
        (
            SchedulerAgenticNode,
            "submit_sql_job",
            {"job_id": "current", "job_name": "new job", "status": "submitted"},
            "scheduler_result",
            {"submitted_jobs": [{"job_id": "current", "job_name": "new job", "status": "submitted"}]},
        ),
    ],
)
def test_error_partial_resources_belong_to_current_run(node_class, tool_name, payload, result_field, expected):
    manager = ActionHistoryManager()
    manager.add_action(_action("old-resource", role=ActionRole.TOOL, action_type=tool_name, output={"result": payload}))
    start = manager.checkpoint()
    manager.add_action(_action("new-resource", role=ActionRole.TOOL, action_type=tool_name, output={"result": payload}))
    ctx = StreamRunContext(FakeInput(), manager, action_start=start)
    node = node_class.__new__(node_class)
    result = node._build_error_result(RuntimeError("failed"), ctx)
    assert not result.success
    assert getattr(result, result_field) == expected


@pytest.mark.asyncio
async def test_prior_authorization_and_benchmark_consumers_keep_their_data():
    from datus.utils.benchmark_trajectory import _build_node_payload

    manager = ActionHistoryManager()
    queries = [_query(0), _query(1)]
    for turn, action in enumerate(queries):
        action.input = {"function_name": "execute_sql", "arguments": {"sql": f"SELECT {turn}"}}
    node = _ChatHistoryNode(_StreamModel([[queries[0], _response(0)], [queries[1], _response(1)]]))
    for message in ["create the test tables", "continue the test"]:
        node.input = FakeInput(user_message=message)
        async for _ in node.execute_stream(manager):
            pass

    node.session_id = "permission-history"
    node._current_action_history = manager
    evidence = node._build_permission_review_context()
    assert evidence["trusted_user_messages"] == ["create the test tables", "continue the test"]
    assert evidence["prior_actions"] == [
        {"tool": "execute_sql", "arguments": {"sql": "SELECT 0"}},
        {"tool": "execute_sql", "arguments": {"sql": "SELECT 1"}},
    ]

    trajectory = _build_node_payload(node, {}, None, "2026-11-13T00:00:00Z")
    tool_actions = [a for a in trajectory["actions"] if a["role"] == "tool"]
    assert tool_actions == [
        {"role": "tool", "status": "success", "name": "execute_sql", "content": queries[1].output["result"]}
    ]
    assert trajectory["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert trajectory["execution_stats"]["tool_calls_count"] == 1


def _cross_run_metric_selection(node_class, rows):
    # Use the real cache and its scope validation without initializing a Dosi runtime.
    semantic = SemanticTools.__new__(SemanticTools)
    semantic.agent_config = None
    semantic._query_metrics_result_cache = OrderedDict()
    semantic._evicted_query_metrics_scopes = OrderedDict()
    result_id = semantic._cache_query_metrics_result(["order_count"], rows, {"sql": "SELECT 42 AS order_count"})
    cached = semantic.get_cached_query_metrics_result(result_id)
    node = node_class.__new__(node_class)
    node.execution_mode = "workflow"
    node.require_final_result_selection = True
    node.semantic_tools = semantic
    manager = ActionHistoryManager()
    manager.add_action(
        _action(
            "previous-query",
            role=ActionRole.TOOL,
            action_type="query_metrics" if node_class is AskMetricsAgenticNode else "select_records",
            output={
                "raw_output": {
                    "success": 1,
                    "result": f"result_id: {result_id}\nrows: {len(rows)}\n\n{cached['csv']}",
                }
            },
        )
    )
    start = manager.checkpoint()
    manager.add_action(_action("current-user", role=ActionRole.USER, action_type="request"))
    selection = node.select_final_metric_result(result_id)
    assert selection.success == 1
    manager.add_action(
        _action(
            "current-selection",
            role=ActionRole.TOOL,
            action_type="select_final_metric_result",
            output={"raw_output": selection.model_dump()},
        )
    )
    ctx = StreamRunContext(FakeInput(), manager, action_start=start, response_content="Use the previous result")
    node.result = node._build_success_result(ctx)
    return node, result_id, cached


@pytest.mark.parametrize("node_class", [AskMetricsAgenticNode, OntologyAnalystAgenticNode])
@pytest.mark.parametrize("rows", [[], [[7], [42]]])
def test_cross_run_metric_selection_captures_cache_without_copying_old_trace(node_class, rows):
    node, _, cached = _cross_run_metric_selection(node_class, rows)
    workflow = SimpleNamespace(context=SimpleNamespace(sql_contexts=[]))
    trace_before = node.result.model_dump_json()

    assert node.update_context(workflow) == {"success": True, "message": "query_metrics result captured"}
    [captured] = workflow.context.sql_contexts
    assert captured.sql_return == cached["csv"]
    assert captured.sql_query == "SELECT 42 AS order_count"
    assert captured.row_count == len(rows)
    assert [a["action_id"] for a in node.result.action_history] == ["current-user", "current-selection"]
    assert node.result.model_dump_json() == trace_before


@pytest.mark.parametrize("node_class", [AskMetricsAgenticNode, OntologyAnalystAgenticNode])
@pytest.mark.parametrize("cache_change", ["evicted", "scope_changed"])
def test_cross_run_metric_selection_rechecks_cache_validity(node_class, cache_change):
    node, result_id, cached = _cross_run_metric_selection(node_class, [[42]])
    if cache_change == "evicted":
        node.semantic_tools._query_metrics_result_cache.pop(result_id)
    else:
        cached["policy_scope"] = "a-different-policy-scope"
    assert node.semantic_tools.get_cached_query_metrics_result(result_id) is None
    workflow = SimpleNamespace(context=SimpleNamespace(sql_contexts=[]))

    result = node.update_context(workflow)

    assert result["success"] is False
    assert result_id in result["message"]
    assert workflow.context.sql_contexts == []
