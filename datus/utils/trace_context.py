# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime trace identity shared by Datus workflow, chat, and benchmark paths."""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Optional, Sequence

_CURRENT_TRACE_CONTEXT: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "datus_trace_context",
    default=None,
)


def _compact_slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text.strip("_") or fallback


def _as_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, Sequence):
        return [str(item) for item in values if item is not None and str(item) != ""]
    return [str(values)]


def _string_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value)
    return text[:200]


def _timestamped_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}:{uuid.uuid4().hex[:8]}"


def create_workflow_run_id(action: str) -> str:
    return f"workflow:{_compact_slug(action)}:{_timestamped_id()}"


@dataclass(frozen=True)
class TraceContext:
    """Stable trace identity for one logical Datus operation."""

    name: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def langfuse_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trace_name": self.name}
        if self.session_id:
            kwargs["session_id"] = self.session_id
        if self.user_id:
            kwargs["user_id"] = self.user_id
        if self.tags:
            kwargs["tags"] = list(dict.fromkeys(self.tags))
        if self.metadata:
            kwargs["metadata"] = {
                str(key): _string_metadata_value(value) for key, value in self.metadata.items() if value is not None
            }
        return kwargs

    def agents_run_config_kwargs(self, agent_name: Optional[str] = None) -> dict[str, Any]:
        child_name = self.name
        if agent_name:
            agent_slug = _compact_slug(agent_name)
            trace_leaf = self.name.rstrip("/").rsplit("/", 1)[-1]
            if agent_slug != trace_leaf:
                child_name = f"{self.name}/{agent_slug}"

        metadata = dict(self.metadata)
        metadata.setdefault("trace_name", self.name)
        if self.session_id:
            metadata.setdefault("trace_session_id", self.session_id)
        if self.tags:
            metadata.setdefault("trace_tags", list(dict.fromkeys(self.tags)))
        if agent_name:
            metadata.setdefault("agent_name", agent_name)

        kwargs: dict[str, Any] = {
            "workflow_name": child_name,
            "trace_metadata": metadata,
        }
        if self.session_id:
            kwargs["group_id"] = self.session_id
        return kwargs


def get_trace_context() -> Optional[TraceContext]:
    return _CURRENT_TRACE_CONTEXT.get()


def set_trace_context(ctx: TraceContext) -> contextvars.Token[Optional[TraceContext]]:
    return _CURRENT_TRACE_CONTEXT.set(ctx)


def reset_trace_context(token: contextvars.Token[Optional[TraceContext]]) -> None:
    _CURRENT_TRACE_CONTEXT.reset(token)


@contextlib.contextmanager
def trace_context(ctx: Optional[TraceContext], *, replace: bool = False) -> Iterator[None]:
    if ctx is None or (get_trace_context() is not None and not replace):
        yield
        return

    token = set_trace_context(ctx)
    try:
        yield
    finally:
        reset_trace_context(token)


def build_agents_run_config_kwargs(agent_name: Optional[str] = None) -> dict[str, Any]:
    ctx = get_trace_context()
    if ctx is None:
        return {}
    return ctx.agents_run_config_kwargs(agent_name=agent_name)


def build_trace_span_attributes(
    *,
    operation: str,
    run_type: str = "chain",
    ctx: Optional[TraceContext] = None,
) -> dict[str, Any]:
    """Build provider-neutral attributes for Datus observability propagation."""
    trace_ctx = get_trace_context() if ctx is None else ctx
    attrs: dict[str, Any] = {
        "datus.operation": operation,
        "datus.run_type": str(run_type),
    }
    if trace_ctx is None:
        return attrs

    attrs["datus.trace.name"] = trace_ctx.name
    if trace_ctx.session_id:
        attrs["datus.session_id"] = trace_ctx.session_id
    if trace_ctx.user_id:
        attrs["datus.user_id"] = trace_ctx.user_id
    if trace_ctx.tags:
        attrs["datus.tags"] = ",".join(trace_ctx.tags)
    for key, value in (trace_ctx.metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            attrs[f"datus.metadata.{key}"] = value
        else:
            attrs[f"datus.metadata.{key}"] = _string_metadata_value(value)
    run_id = (trace_ctx.metadata or {}).get("run_id") or (trace_ctx.metadata or {}).get("benchmark_run_id")
    if run_id:
        attrs["datus.run_id"] = run_id
    return attrs


def _metadata_base(
    *,
    component: str,
    datasource: Optional[str] = None,
    workflow: Optional[str] = None,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_home: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"datus_component": component}
    if datasource:
        metadata["datasource"] = datasource
    if workflow:
        metadata["workflow"] = workflow
    if run_id:
        metadata["run_id"] = run_id
    if task_id:
        metadata["task_id"] = task_id
    if agent_home:
        metadata["agent_home"] = agent_home
    if extra:
        metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def build_workflow_trace_context_from_runner(runner: Any, *args: Any, **kwargs: Any) -> TraceContext:
    sql_task = kwargs.get("sql_task")
    if sql_task is None and args:
        sql_task = args[0]

    runner_args = getattr(runner, "args", None)
    config = getattr(runner, "global_config", None)
    workflow = getattr(runner_args, "workflow", None) or "default"
    datasource = getattr(config, "current_datasource", None) or getattr(runner_args, "datasource", None)
    run_id = getattr(runner, "run_id", None) or getattr(runner_args, "run_id", None)
    task_id = getattr(sql_task, "id", None)
    session_id = f"workflow:run:{run_id}" if run_id else create_workflow_run_id("run")
    workflow_slug = _compact_slug(workflow)

    tags = ["workflow", f"workflow:{workflow_slug}"]
    if datasource:
        tags.append(f"datasource:{_compact_slug(datasource)}")

    return TraceContext(
        name=f"workflow/{workflow_slug}",
        session_id=session_id,
        tags=tuple(tags),
        metadata=_metadata_base(
            component="workflow_run",
            datasource=datasource,
            workflow=workflow,
            run_id=run_id,
            task_id=str(task_id) if task_id else None,
            agent_home=getattr(config, "home", None),
        ),
    )


def _infer_context_type(benchmark: Optional[str], workflow: Optional[str], run_id: Optional[str]) -> Optional[str]:
    if benchmark and workflow and workflow.startswith(f"{benchmark}_"):
        return workflow[len(benchmark) + 1 :]
    if run_id:
        match = re.match(r"(.+)_\d{8}_\d{6}$", run_id)
        if match:
            value = match.group(1)
            return value.rsplit("_", 1)[-1] if value.startswith("default_") else value
    return None


def build_benchmark_trace_context(
    *,
    benchmark: str,
    run_id: str,
    task_id: str,
    workflow: Optional[str] = None,
    context_type: Optional[str] = None,
    datasource: Optional[str] = None,
    agent_home: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> TraceContext:
    run_id = run_id or _timestamped_id()
    context = context_type or _infer_context_type(benchmark, workflow, run_id) or "default"
    benchmark_slug = _compact_slug(benchmark)
    context_slug = _compact_slug(context)
    task_slug = _compact_slug(task_id)
    tags = [
        "benchmark",
        benchmark_slug,
        context_slug,
        f"task:{task_slug}",
    ]
    if datasource:
        tags.append(f"datasource:{_compact_slug(datasource)}")

    metadata = _metadata_base(
        component="benchmark",
        datasource=datasource,
        workflow=workflow,
        run_id=run_id,
        task_id=task_id,
        agent_home=agent_home,
        extra={
            "benchmark": benchmark,
            "benchmark_run_id": run_id,
            "context_type": context,
            **(dict(extra) if extra else {}),
        },
    )
    return TraceContext(
        name=f"benchmark/{benchmark_slug}/{context_slug}/task-{task_slug}",
        session_id=f"benchmark:{run_id}",
        tags=tuple(tags),
        metadata=metadata,
    )


def build_bootstrap_trace_context(
    *,
    datasource: Optional[str],
    components: Sequence[str],
    strategy: Optional[str] = None,
    run_id: Optional[str] = None,
    stream_id: Optional[str] = None,
    agent_home: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> TraceContext:
    component_values = _as_list(components) or ["metadata"]
    component_slug = "+".join(_compact_slug(component) for component in component_values)
    datasource_slug = _compact_slug(datasource, "default")
    raw_session_id = stream_id or run_id or _timestamped_id()
    session_id = raw_session_id if str(raw_session_id).startswith("bootstrap:") else f"bootstrap:{raw_session_id}"
    tags = [
        "bootstrap-kb",
        f"datasource:{datasource_slug}",
        *(f"component:{_compact_slug(c)}" for c in component_values),
    ]
    if strategy:
        tags.append(f"strategy:{_compact_slug(strategy)}")

    metadata = _metadata_base(
        component="bootstrap_kb",
        datasource=datasource,
        run_id=run_id,
        agent_home=agent_home,
        extra={
            "components": component_values,
            "strategy": strategy,
            "stream_id": stream_id,
            **(dict(extra) if extra else {}),
        },
    )
    return TraceContext(
        name=f"bootstrap-kb/{datasource_slug}/{component_slug}",
        session_id=session_id,
        tags=tuple(tags),
        metadata=metadata,
    )


def build_bootstrap_trace_context_from_agent(agent: Any, *_args: Any, **_kwargs: Any) -> TraceContext:
    agent_args = getattr(agent, "args", None)
    config = getattr(agent, "global_config", None)
    return build_bootstrap_trace_context(
        datasource=getattr(config, "current_datasource", None) or getattr(agent_args, "datasource", None),
        components=_as_list(getattr(agent_args, "components", None)),
        strategy=getattr(agent_args, "kb_update_strategy", None),
        run_id=getattr(agent_args, "run_id", None),
        agent_home=getattr(config, "home", None),
        extra={
            "catalog": getattr(agent_args, "catalog", None),
            "database_name": getattr(agent_args, "database_name", None),
            "subject_path": getattr(agent_args, "subject_path", None),
        },
    )


def build_chat_trace_context(
    *,
    session_id: str,
    node_name: Optional[str] = None,
    subagent_id: Optional[str] = None,
    llm_session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    datasource: Optional[str] = None,
    source_session_id: Optional[str] = None,
    source: Optional[str] = None,
    model: Optional[str] = None,
    agent_home: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> TraceContext:
    display_name = node_name or subagent_id or "chat"
    display_slug = _compact_slug(display_name, "chat")
    tags = ["chat", f"agent:{display_slug}"]
    if datasource:
        tags.append(f"datasource:{_compact_slug(datasource)}")
    if source:
        tags.append(f"source:{_compact_slug(source)}")

    metadata = _metadata_base(
        component="chat",
        datasource=datasource,
        agent_home=agent_home,
        extra={
            "service_session_id": session_id,
            "llm_session_id": llm_session_id,
            "subagent_id": subagent_id,
            "node_name": node_name,
            "source_session_id": source_session_id,
            "source": source,
            "model": model,
            **(dict(extra) if extra else {}),
        },
    )
    return TraceContext(
        name=f"agent/{display_slug}",
        session_id=session_id,
        user_id=user_id,
        tags=tuple(tags),
        metadata=metadata,
    )
