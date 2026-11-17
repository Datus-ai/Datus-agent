"""Execution evidence survives async task boundaries and excludes sensitive content."""

import asyncio
from types import SimpleNamespace

import pytest

from ci.bi_diagnostics import install_capture, trace_rows

pytestmark = pytest.mark.acceptance


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_child_task_trace_is_bound_to_owning_session_even_after_failure(monkeypatch, fail):
    from datus.agent.node.agentic_node import AgenticNode
    from datus.observability.manager import ObservabilityManager
    from datus.observability.reference import TraceReference

    manager = ObservabilityManager()
    monkeypatch.setattr(
        ObservabilityManager,
        "_trace_reference_from_span",
        lambda *a, **k: TraceReference("trace-1", run_id="session-1", provider="langfuse"),
    )

    async def original(node, *args, **kwargs):
        async def child():
            manager._trace_reference_from_span(None)

        await asyncio.create_task(child())
        yield SimpleNamespace()
        if fail:
            raise RuntimeError("model failed")

    monkeypatch.setattr(AgenticNode, "execute_stream", original)
    case = {}
    install_capture(monkeypatch, case)
    node = SimpleNamespace(session_id="session-1", get_node_name=lambda: "gen_dashboard", result=None)
    try:
        async for _ in AgenticNode.execute_stream(node):
            pass
    except RuntimeError:
        assert fail
    assert trace_rows(case) == [
        {
            "trace_id": "trace-1",
            "trace_run_id": "session-1",
            "trace_provider": "langfuse",
            "session_id": "session-1",
            "node_name": "gen_dashboard",
        }
    ]
    assert case["runs"][0]["success"] is False


def test_capture_omits_sql_rows_and_secrets_and_supports_no_trace(monkeypatch):
    from datus.tools.func_tool.base import FuncToolResult
    from datus.tools.func_tool.bi_tools import BIFuncTool

    def original(self, chart_id, sql="", dimensions=None):
        return FuncToolResult(
            result={"columns": ["region"], "rows": [{"secret": "sensitive"}], "sql": sql, "password": "secret"}
        )

    monkeypatch.setattr(BIFuncTool, "update_chart", original)
    case = {}
    install_capture(monkeypatch, case)
    BIFuncTool().update_chart("3", sql="select private_data", dimensions="region")
    assert case["tool_calls"] == [
        {
            "name": "original",
            "arguments": {"chart_id": "3", "dimensions": "region"},
            "success": 1,
            "result": {"columns": ["region"]},
        }
    ]
    assert trace_rows(case) == []


@pytest.mark.parametrize("has_trace", [True, False])
def test_pytest_report_persists_failed_execution_and_child_trace(monkeypatch, tmp_path, has_trace):
    import json

    from ci import pytest_trace_reference_plugin as plugin

    trace_file = tmp_path / "traces.jsonl"
    evidence = tmp_path / "cases"
    for key, value in {
        "DATUS_TEST_LAYER": "nightly",
        "DATUS_NIGHTLY_TRACE_REFERENCES_FILE": str(trace_file),
        "DATUS_NIGHTLY_BI_EVIDENCE_DIR": str(evidence),
        "DATUS_NIGHTLY_TRACE_EXPECTED": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(plugin, "_current_trace_metadata", lambda: None)
    case = {
        "nodeid": "test_dashboard",
        "runs": [
            {
                "session_id": "session-1",
                "node_name": "gen_dashboard",
                "trace_references": [{"trace_id": "trace-1", "trace_run_id": "session-1"}] if has_trace else [],
            }
        ],
    }
    stash = pytest.Stash()
    stash[plugin._bi_case_key] = case
    item = SimpleNamespace(stash=stash, nodeid="test_dashboard")
    report = SimpleNamespace(when="call", outcome="failed", duration=1.0)
    hook = plugin.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    (saved,) = evidence.glob("*.json")
    assert json.loads(saved.read_text())["outcome"] == "failed"
    row = json.loads(trace_file.read_text())
    assert row["outcome"] == "failed"
    assert row.get("trace_id") == ("trace-1" if has_trace else None)
    assert row["trace_expected"] is True
    assert row["evidence_file"] == str(saved)


def test_tool_checkpoint_keeps_pending_call_and_exception_without_pytest_report(monkeypatch):
    import json

    from datus.tools.func_tool.bi_tools import BIFuncTool

    def original(self, chart_id):
        raise ValueError("private error content")

    monkeypatch.setattr(BIFuncTool, "get_chart", original)
    case, checkpoints = {}, []
    install_capture(monkeypatch, case, checkpoint=lambda: checkpoints.append(json.loads(json.dumps(case))))
    with pytest.raises(ValueError):
        BIFuncTool().get_chart("3")
    assert checkpoints[0]["tool_calls"] == [{"name": "original", "arguments": {"chart_id": "3"}}]
    assert checkpoints[-1]["tool_calls"][0]["exception_type"] == "ValueError"
    assert "private error content" not in json.dumps(checkpoints)
