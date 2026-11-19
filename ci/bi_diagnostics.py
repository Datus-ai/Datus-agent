"""Test-owned BI evidence capture, including references created in child async tasks.

Only chart coordinates, grouping controls and result shape are retained. Credentials,
SQL text, Bash arguments, prompts and data rows are excluded from CI artifacts.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from contextvars import ContextVar

SAFE_ARGUMENTS = {"chart_id", "dashboard_id", "dataset_id", "chart_type", "dimensions", "x_axis", "metrics", "limit"}
SAFE_RESULTS = {
    "id",
    "chart_id",
    "dashboard_id",
    "deleted",
    "columns",
    "row_count",
    "deliverable_target",
    "deleted_deliverable",
}


def install_capture(monkeypatch, case: dict, checkpoint: Callable[[], None] | None = None) -> None:
    from datus.agent.node.agentic_node import AgenticNode
    from datus.observability.manager import ObservabilityManager
    from datus.tools.func_tool.bi_tools import BIFuncTool
    from datus.validation import ValidationHook

    save = checkpoint or (lambda: None)
    active = ContextVar("bi_test_run", default=case)
    original_stream = AgenticNode.execute_stream
    original_reference = ObservabilityManager._trace_reference_from_span
    original_tool_end = ValidationHook.on_tool_end
    original_end = ValidationHook.on_end

    @functools.wraps(original_stream)
    async def stream(node, *args, **kwargs):
        run = {"session_id": getattr(node, "session_id", None), "node_name": node.get_node_name()}
        case.setdefault("runs", []).append(run)
        save()
        token = active.set(run)
        try:
            async for action in original_stream(node, *args, **kwargs):
                yield action
        finally:
            run["session_id"] = getattr(node, "session_id", None)
            result = getattr(node, "result", None)
            run["success"] = getattr(result, "success", False)
            active.reset(token)
            save()

    @functools.wraps(original_reference)
    def reference(manager, *args, **kwargs):
        ref = original_reference(manager, *args, **kwargs)
        if ref is not None:
            references = active.get().setdefault("trace_references", [])
            metadata = ref.to_metadata()
            if metadata not in references and len(references) < 64:
                references.append(metadata)
                save()
        return ref

    @functools.wraps(original_tool_end)
    async def tool_end(hook, *args, **kwargs):
        try:
            return await original_tool_end(hook, *args, **kwargs)
        finally:
            active.get()["validation_targets"] = [t.model_dump(exclude_none=True) for t in hook.session_targets]
            save()

    @functools.wraps(original_end)
    async def end(hook, *args, **kwargs):
        try:
            return await original_end(hook, *args, **kwargs)
        finally:
            report = hook.final_report
            if report is not None:
                # Check names/outcomes and resource IDs suffice for repair diagnosis.
                active.get()["validation_checks"] = [
                    {"name": check.name, "passed": check.passed, "severity": check.severity} for check in report.checks
                ]

    def capture_tool(original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            record = {
                "name": original.__name__,
                "arguments": {k: v for k, v in bound.arguments.items() if k in SAFE_ARGUMENTS},
            }
            records = active.get().setdefault("tool_calls", [])
            if len(records) < 100:
                records.append(record)
            save()  # Keep the pending call even if pytest-timeout terminates the process.
            try:
                result = original(*args, **kwargs)
                record["success"] = result.success
                if isinstance(result.result, dict):
                    record["result"] = {k: v for k, v in result.result.items() if k in SAFE_RESULTS}
                return result
            except Exception as exc:
                record["success"] = False
                record["exception_type"] = type(exc).__name__
                raise
            finally:
                save()

        return wrapped

    monkeypatch.setattr(AgenticNode, "execute_stream", stream)
    monkeypatch.setattr(ObservabilityManager, "_trace_reference_from_span", reference)
    monkeypatch.setattr(ValidationHook, "on_tool_end", tool_end)
    monkeypatch.setattr(ValidationHook, "on_end", end)
    for name in [
        "create_dashboard",
        "create_chart",
        "update_chart",
        "add_chart_to_dashboard",
        "delete_chart",
        "get_chart",
        "get_chart_data",
    ]:
        monkeypatch.setattr(BIFuncTool, name, capture_tool(getattr(BIFuncTool, name)))


def trace_rows(case: dict) -> list[dict]:
    rows = []
    for run in [case, *case.get("runs", [])]:
        refs = run.get("trace_references", [])
        # Prefer a reference explicitly attached to this Agent session.
        matching = [ref for ref in refs if ref.get("trace_run_id") == run.get("session_id")]
        if refs:
            rows.append(
                {**(matching or refs)[0], "session_id": run.get("session_id"), "node_name": run.get("node_name")}
            )
    return rows
