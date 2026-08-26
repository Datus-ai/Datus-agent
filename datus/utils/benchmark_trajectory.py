# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Native v1 benchmark trajectory artifacts.

Builds the consumer-contract trajectory envelope (``artifact_type: trajectory``,
``schema_version: 1``) from an executed workflow: identity and lifecycle fields,
aggregate model/usage, a deduplicated table-schema registry with per-node
references, normalized per-node timing/actions/errors, and a trace reference.
The contract lives in ``datus-benchmark`` (``contracts/v1/trajectory.schema.json``).
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from datus.utils.benchmark_artifacts import (
    BenchmarkAttempt,
    _aggregate_usage,
    _as_mapping,
    _model_identity,
    _node_sequence,
    _node_usage,
    _rfc3339,
    _structured_error,
    _utc_now,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

TRAJECTORY_ARTIFACT_TYPE = "trajectory"
TRAJECTORY_SCHEMA_VERSION = 1

# Node statuses the contract accepts; anything else has not durably executed.
_CONTRACT_NODE_STATUSES = {"running", "completed", "failed"}


def _epoch_rfc3339(value: Any) -> Optional[str]:
    """Render an epoch-seconds timestamp as RFC 3339, or ``None`` when unset."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return _rfc3339(datetime.fromtimestamp(float(value), tz=timezone.utc))


def _node_duration_seconds(node: Any) -> Optional[float]:
    """Duration between a node's start/end epoch timestamps when both exist."""
    start = getattr(node, "start_time", None)
    end = getattr(node, "end_time", None)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start > 0:
        return float(end) - float(start)
    return None


def _schema_key(identifier: str) -> str:
    """Stable registry key for one table schema body."""
    return f"table:{identifier}"


def _iter_table_schemas(container: Any) -> list[Mapping[str, Any]]:
    """Collect table-schema mappings from a node input/result-like object."""
    data = _as_mapping(container)
    entries = data.get("table_schemas")
    if not isinstance(entries, list):
        return []
    collected = []
    for entry in entries:
        entry_data = _as_mapping(entry)
        if entry_data.get("definition") and (entry_data.get("identifier") or entry_data.get("table_name")):
            collected.append(entry_data)
    return collected


def _register_schema(registry: dict[str, dict[str, Any]], entry: Mapping[str, Any]) -> str:
    """Store one schema body once and return its registry key."""
    identifier = str(entry.get("identifier") or entry.get("table_name"))
    key = _schema_key(identifier)
    if key not in registry:
        metadata = {
            field: value
            for field, value in entry.items()
            if field not in {"identifier", "definition"} and isinstance(value, (str, int, float, bool))
        }
        registry[key] = {
            "identifier": identifier,
            "ddl": str(entry["definition"]),
            "metadata": metadata,
        }
    return key


def _action_content(entry: Mapping[str, Any]) -> Any:
    """Extract the action payload the consumer reads from ``content`` directly.

    Collapses the historical ``output.raw_output.result`` / ``output.result`` /
    raw ``output`` probing into the single native shape.
    """
    content = entry.get("content")
    if content is not None:
        return content
    output = entry.get("output")
    output_data = _as_mapping(output)
    if output_data:
        raw_output = _as_mapping(output_data.get("raw_output"))
        if "result" in raw_output:
            return raw_output.get("result")
        if "result" in output_data:
            return output_data.get("result")
        return dict(output_data)
    return output


def _normalize_action(entry: Any) -> Optional[dict[str, Any]]:
    """Normalize one action-history entry into the native action shape."""
    data = _as_mapping(entry)
    if not data:
        return None
    action: dict[str, Any] = {
        "role": str(data.get("role") or "workflow"),
        "status": str(data.get("status") or "completed"),
        "content": _action_content(data),
    }
    name = data.get("tool_name") or data.get("name") or data.get("action_type")
    if name:
        action["name"] = str(name)
    error = data.get("error")
    if error:
        action["error"] = str(error)
    usage = _as_mapping(data.get("output")).get("usage")
    if isinstance(usage, Mapping) and usage:
        action["usage"] = dict(usage)
    return action


def _node_error(node: Any) -> Optional[dict[str, Any]]:
    """Structured error for a failed node, or ``None``."""
    if str(getattr(node, "status", "")) != "failed":
        return None
    result_error = getattr(getattr(node, "result", None), "error", None)
    node_type = str(getattr(node, "type", "") or "unknown")
    return {
        "type": "node_failure",
        "message": str(result_error) or f"Benchmark node {getattr(node, 'id', node_type)} failed",
        "code": None,
        "retryable": None,
        "details": {"node_type": node_type},
    }


def _build_node_payload(
    node: Any,
    schemas: dict[str, dict[str, Any]],
    agent_config: Any,
    fallback_started_at: str,
) -> dict[str, Any]:
    """Serialize one executed node without inlining schema bodies."""
    result = getattr(node, "result", None)
    schema_refs: list[str] = []
    for container in (getattr(node, "input", None), result):
        for entry in _iter_table_schemas(container):
            key = _register_schema(schemas, entry)
            if key not in schema_refs:
                schema_refs.append(key)

    actions = []
    action_history = _as_mapping(result).get("action_history") or []
    if isinstance(action_history, list):
        for entry in action_history:
            action = _normalize_action(entry)
            if action is not None:
                actions.append(action)

    execution_stats = _as_mapping(result).get("execution_stats")

    # Planner-completed nodes (e.g. the start node) never call Node.start(), so
    # backfill their start from the end timestamp or the attempt creation time.
    started_at = _epoch_rfc3339(getattr(node, "start_time", None))
    completed_at = _epoch_rfc3339(getattr(node, "end_time", None))
    if started_at is None:
        started_at = completed_at or fallback_started_at

    return {
        "id": str(getattr(node, "id", "") or "unknown"),
        "type": str(getattr(node, "type", "") or "unknown"),
        "status": str(getattr(node, "status", "") or "completed"),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": _node_duration_seconds(node),
        "model": _model_identity_for_node(node, agent_config),
        "usage": _node_usage(node),
        "schema_refs": schema_refs,
        "actions": actions,
        "execution_stats": dict(execution_stats) if isinstance(execution_stats, Mapping) else {},
        "error": _node_error(node),
        "metadata": {},
    }


def _model_identity_for_node(node: Any, agent_config: Any) -> Optional[dict[str, Any]]:
    """Model identity for one node only; ``None`` when the node ran no model."""
    node_model = getattr(node, "model", None)
    if getattr(node_model, "model_config", None) is None:
        return None
    single_node_workflow = type("_W", (), {"nodes": {"n": node}, "node_order": ["n"]})()
    return _model_identity(single_node_workflow, agent_config)


def _trace_reference(workflow: Any) -> dict[str, Any]:
    """Trace reference from workflow metadata; empty when tracing was off."""
    metadata = _as_mapping(getattr(workflow, "metadata", None))
    trace_id = metadata.get("trace_id")
    if not trace_id:
        return {}
    trace: dict[str, Any] = {"trace_id": str(trace_id)}
    for source_key, target_key in (
        ("trace_span_id", "span_id"),
        ("trace_run_id", "run_id"),
        ("trace_provider", "provider"),
    ):
        value = metadata.get(source_key)
        if value:
            trace[target_key] = str(value)
    return trace


def build_trajectory_payload(
    attempt: BenchmarkAttempt,
    *,
    workflow: Any,
    agent_config: Any,
    exception: Optional[BaseException] = None,
) -> dict[str, Any]:
    """Build the schema-valid native v1 trajectory payload for one attempt."""
    completed_at = _utc_now()
    duration_seconds = max(0.0, time.monotonic() - attempt.started_monotonic)

    schemas: dict[str, dict[str, Any]] = {}
    created_at_text = _rfc3339(attempt.created_at)
    nodes = [
        _build_node_payload(node, schemas, agent_config, created_at_text)
        for node in _node_sequence(workflow)
        if str(getattr(node, "status", "")) in _CONTRACT_NODE_STATUSES
    ]
    # Workflow-context schemas repeat node schemas; register them so the
    # registry stays complete even when node payloads were trimmed upstream.
    for entry in _iter_table_schemas(getattr(workflow, "context", None)):
        _register_schema(schemas, entry)

    completed = exception is None and str(getattr(workflow, "status", "")) == "completed"
    error = None if completed else _structured_error(workflow, exception, outputs_exist=True)
    failure_types = [str(error["type"])] if error else []

    return {
        "artifact_type": TRAJECTORY_ARTIFACT_TYPE,
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "run_id": attempt.run_id,
        "task_id": attempt.task_id,
        "attempt_id": attempt.attempt_id,
        "status": "completed" if completed else "failed",
        "partial": False,
        "created_at": created_at_text,
        "completed_at": _rfc3339(completed_at),
        "duration_seconds": duration_seconds,
        "model": _model_identity(workflow, agent_config),
        "usage": _aggregate_usage(workflow),
        "failure_types": failure_types,
        "error": error,
        "schemas": schemas,
        "nodes": nodes,
        "trace": _trace_reference(workflow),
        "execution_stats": {
            "total_nodes": len(nodes),
            "completed_nodes": sum(1 for node in nodes if node["status"] == "completed"),
        },
        "metadata": {"capture_level": "standard"},
    }


class _TrajectoryDumper(yaml.SafeDumper):
    """SafeDumper that keeps long SQL/DDL readable via block scalars."""


def _str_representer(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    """Represent multi-line strings as block scalars per the contract."""
    if "\n" in value:
        # Block scalars cannot carry trailing spaces on lines; normalize them.
        cleaned = "\n".join(line.rstrip() for line in value.splitlines())
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_TrajectoryDumper.add_representer(str, _str_representer)


def trajectory_file_path(attempt: BenchmarkAttempt) -> Path:
    """Deterministic trajectory path for one attempt (no timestamp authority)."""
    return attempt.trajectory_run_root / f"task_{attempt.task_id}.{attempt.attempt_id}.yaml"


def write_benchmark_trajectory(
    attempt: BenchmarkAttempt,
    *,
    workflow: Any,
    agent_config: Any,
    exception: Optional[BaseException] = None,
) -> Path:
    """Atomically write the native v1 trajectory YAML and return its path."""
    payload = build_trajectory_payload(
        attempt,
        workflow=workflow,
        agent_config=agent_config,
        exception=exception,
    )
    path = trajectory_file_path(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.dump(
                payload,
                handle,
                Dumper=_TrajectoryDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
