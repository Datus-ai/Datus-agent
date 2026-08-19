# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Versioned task-output artifacts for benchmark runs.

The consumer-owned contract lives in ``datus-benchmark``. This module keeps
producer-specific lifecycle and compatibility handling out of the generic
interactive output tool.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

TASK_OUTPUT_SCHEMA_VERSION = 1
TASK_OUTPUT_ARTIFACT_TYPE = "task_output"
BENCHMARK_ARTIFACT_PROFILE = "benchmark_v1"
TASK_OUTPUT_MANIFEST = "task-output.json"


def _utc_now() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(timezone.utc)


def _rfc3339(value: datetime) -> str:
    """Render a datetime as an RFC 3339 UTC timestamp with a ``Z`` suffix."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_path_segment(value: str, field_name: str) -> str:
    """Reject identifiers that cannot be used as a single portable path segment."""
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text) or text in {".", ".."}:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"benchmark {field_name} must be a non-empty portable path segment: {value!r}",
        )
    return text


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string, rejecting escapes."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"benchmark artifact path is outside its run root: {path}",
        ) from exc
    value = relative.as_posix()
    if not value or any(part == ".." for part in relative.parts):
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"invalid benchmark artifact relative path: {value!r}",
        )
    return value


@dataclass(frozen=True)
class BenchmarkAttempt:
    """One immutable task attempt under a benchmark save run root."""

    save_run_root: Path
    trajectory_run_root: Path
    run_id: str
    task_id: str
    task_type: str
    attempt_id: str
    output_dir: Path
    created_at: datetime
    started_monotonic: float

    @property
    def task_root(self) -> Path:
        """Directory that holds this task's manifest and attempts."""
        return self.save_run_root / "tasks" / self.task_id

    @property
    def manifest_path(self) -> Path:
        """Path of the task's authoritative ``task-output.json`` manifest."""
        return self.task_root / TASK_OUTPUT_MANIFEST


def allocate_benchmark_attempt(
    save_run_root: Path | str,
    trajectory_run_root: Path | str,
    *,
    run_id: str,
    task_id: str,
    task_type: str = "query",
) -> BenchmarkAttempt:
    """Atomically allocate ``attempt-N`` so retries never overwrite output."""
    safe_task_id = _validate_path_segment(task_id, "task_id")
    safe_run_id = str(run_id).strip() if run_id is not None else ""
    if not safe_run_id:
        raise DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="benchmark run_id must be non-empty",
        )

    save_root = Path(save_run_root)
    trajectory_root = Path(trajectory_run_root)
    attempts_root = save_root / "tasks" / safe_task_id / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)

    attempt_number = 1
    while True:
        attempt_id = f"attempt-{attempt_number}"
        output_dir = attempts_root / attempt_id
        try:
            output_dir.mkdir()
            break
        except FileExistsError:
            attempt_number += 1

    return BenchmarkAttempt(
        save_run_root=save_root,
        trajectory_run_root=trajectory_root,
        run_id=safe_run_id,
        task_id=safe_task_id,
        task_type=str(task_type or "query"),
        attempt_id=attempt_id,
        output_dir=output_dir,
        created_at=_utc_now(),
        started_monotonic=time.monotonic(),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a same-directory temp file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce dataclass-like, pydantic, or mapping values into a plain mapping."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _node_sequence(workflow: Any) -> list[Any]:
    """Return workflow nodes in execution order, tolerating partial workflows."""
    if workflow is None:
        return []
    nodes = getattr(workflow, "nodes", {}) or {}
    order = getattr(workflow, "node_order", []) or []
    if isinstance(nodes, Mapping):
        ordered = [nodes[node_id] for node_id in order if node_id in nodes]
        return ordered or list(nodes.values())
    return list(nodes) if isinstance(nodes, (list, tuple)) else []


def _normalize_usage_key(key: str) -> str:
    """Map provider-specific token usage keys onto the contract's canonical names."""
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "cached_tokens": "cached_input_tokens",
    }
    return aliases.get(key, key)


def _merge_usage(target: dict[str, int], usage: Any) -> bool:
    """Accumulate non-negative numeric usage counters into ``target``; report if any merged."""
    if not isinstance(usage, Mapping):
        return False
    found = False
    for raw_key, raw_value in usage.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or raw_value < 0:
            continue
        key = _normalize_usage_key(str(raw_key))
        target[key] += int(raw_value)
        found = True
    return found


def _node_usage(node: Any) -> dict[str, int]:
    """Extract one node's token usage, preferring per-call assistant usage over snapshots."""
    result = _as_mapping(getattr(node, "result", None))
    usage: defaultdict[str, int] = defaultdict(int)
    token_event_usage: defaultdict[str, int] = defaultdict(int)
    found_action_usage = False
    found_token_event_usage = False

    action_history = result.get("action_history") or []
    if isinstance(action_history, list):
        for action in action_history:
            action_data = _as_mapping(action)
            if str(action_data.get("action_type") or "") == "token_usage":
                # These records carry cumulative snapshots and would double count
                # the per-call usage attached to assistant actions. Keep deltas
                # only as a fallback when the assistant action has no usage.
                output = _as_mapping(action_data.get("output"))
                found_token_event_usage = (
                    _merge_usage(token_event_usage, _as_mapping(output.get("delta"))) or found_token_event_usage
                )
                continue
            output = _as_mapping(action_data.get("output"))
            found_action_usage = _merge_usage(usage, output.get("usage")) or found_action_usage

    if not found_action_usage and found_token_event_usage:
        _merge_usage(usage, token_event_usage)
        found_action_usage = True

    if not found_action_usage:
        stats = _as_mapping(result.get("execution_stats"))
        token_stats = {key: value for key, value in stats.items() if "token" in str(key)}
        found_action_usage = _merge_usage(usage, token_stats)

    if not found_action_usage:
        tokens_used = result.get("tokens_used")
        if isinstance(tokens_used, (int, float)) and not isinstance(tokens_used, bool) and tokens_used >= 0:
            usage["total_tokens"] += int(tokens_used)

    return dict(usage)


def _aggregate_usage(workflow: Any) -> dict[str, int]:
    """Sum per-node usage across the workflow and derive ``total_tokens`` when absent."""
    aggregate: defaultdict[str, int] = defaultdict(int)
    for node in _node_sequence(workflow):
        _merge_usage(aggregate, _node_usage(node))
    if "total_tokens" not in aggregate and (aggregate.get("input_tokens") or aggregate.get("output_tokens")):
        aggregate["total_tokens"] = aggregate.get("input_tokens", 0) + aggregate.get("output_tokens", 0)
    return dict(aggregate)


def _model_identity(workflow: Any, agent_config: Any) -> Optional[dict[str, Any]]:
    """Describe the executing model from workflow nodes, falling back to the active config."""
    model_config = None
    for node in reversed(_node_sequence(workflow)):
        node_model = getattr(node, "model", None)
        candidate = getattr(node_model, "model_config", None)
        if candidate is not None:
            model_config = candidate
            break

    if model_config is None:
        active_model = getattr(agent_config, "active_model", None)
        if callable(active_model):
            try:
                model_config = active_model()
            except Exception:
                model_config = None

    name = getattr(model_config, "model", None)
    if not isinstance(name, str) or not name.strip():
        return None

    provider = getattr(model_config, "type", None)
    configuration: dict[str, Any] = {}
    for field_name in ("reasoning_effort", "temperature", "top_p", "enable_thinking"):
        value = getattr(model_config, field_name, None)
        if value is not None:
            configuration[field_name] = value

    return {
        "provider": provider if isinstance(provider, str) and provider else None,
        "name": name,
        "configuration": configuration,
    }


def _last_sql_context(workflow: Any) -> Any:
    """Return the most recent SQL context recorded on the workflow, if any."""
    context = getattr(workflow, "context", None)
    sql_contexts = getattr(context, "sql_contexts", None) or []
    return sql_contexts[-1] if sql_contexts else None


def _failed_node(workflow: Any) -> Any:
    """Return the first failed node in execution order, or ``None``."""
    for node in _node_sequence(workflow):
        if str(getattr(node, "status", "")) == "failed":
            return node
    return None


def _output_completed(workflow: Any) -> bool:
    """Report whether the workflow's output node completed successfully."""
    for node in reversed(_node_sequence(workflow)):
        if str(getattr(node, "type", "")) != "output":
            continue
        result = getattr(node, "result", None)
        return str(getattr(node, "status", "")) == "completed" and bool(getattr(result, "success", False))
    return False


def _structured_error(workflow: Any, exception: Optional[BaseException], outputs_exist: bool) -> dict[str, Any]:
    """Build the manifest's structured error from an exception or the failed workflow state."""
    if exception is not None:
        raw_code = getattr(exception, "code", None)
        return {
            "type": "benchmark_execution",
            "message": str(exception) or exception.__class__.__name__,
            "code": str(raw_code) if raw_code is not None else None,
            "retryable": None,
            "details": {"exception_type": exception.__class__.__name__},
        }

    node = _failed_node(workflow)
    sql_context = _last_sql_context(workflow)
    sql_error = getattr(sql_context, "sql_error", None)
    result_error = getattr(getattr(node, "result", None), "error", None) if node is not None else None
    message = str(sql_error or result_error or "")

    if node is not None:
        node_type = str(getattr(node, "type", "") or "unknown")
        return {
            "type": "sql_execution" if node_type == "execute_sql" or sql_error else "node_failure",
            "message": message or f"Benchmark node {getattr(node, 'id', node_type)} failed",
            "code": None,
            "retryable": None,
            "details": {
                "node_id": str(getattr(node, "id", "") or ""),
                "node_type": node_type,
            },
        }

    if workflow is None:
        return {
            "type": "benchmark_execution",
            "message": "Workflow did not initialize",
            "code": None,
            "retryable": None,
            "details": {},
        }

    return {
        "type": "missing_output" if not outputs_exist else "workflow_incomplete",
        "message": "Benchmark workflow did not produce a completed output",
        "code": None,
        "retryable": None,
        "details": {"workflow_status": str(getattr(workflow, "status", "unknown"))},
    }


def _replace_with_hardlink(source: Path, target: Path) -> None:
    """Atomically point ``target`` at ``source`` via hardlink, copying when linking fails."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_legacy_compatibility(
    attempt: BenchmarkAttempt,
    *,
    task: Any,
    completed: bool,
    error: Optional[dict[str, Any]],
    sql_path: Path,
    result_path: Path,
    row_count: int,
) -> None:
    """Refresh the flat legacy alias files so pre-v1 consumers keep working."""
    legacy_sql = attempt.save_run_root / f"{attempt.task_id}.sql"
    legacy_result = attempt.save_run_root / f"{attempt.task_id}.csv"
    legacy_json = attempt.save_run_root / f"{attempt.task_id}.json"

    if completed:
        _replace_with_hardlink(sql_path, legacy_sql)
        _replace_with_hardlink(result_path, legacy_result)
        legacy_payload = {
            "finished": True,
            "instance_id": attempt.task_id,
            "database_name": str(getattr(task, "database_name", "") or ""),
            "result": legacy_result.name,
            "row_count": row_count,
            "instruction": str(getattr(task, "task", "") or ""),
            "metadata": {"source": "benchmark_v1_compatibility", "attempt_id": attempt.attempt_id},
        }
    else:
        legacy_sql.unlink(missing_ok=True)
        legacy_result.unlink(missing_ok=True)
        legacy_payload = {
            "finished": False,
            "instance_id": attempt.task_id,
            "database_name": str(getattr(task, "database_name", "") or ""),
            "error": (error or {}).get("message", "Benchmark execution failed"),
            "metadata": {"source": "benchmark_v1_compatibility", "attempt_id": attempt.attempt_id},
        }
    _atomic_write_json(legacy_json, legacy_payload)


def finalize_benchmark_attempt(
    attempt: BenchmarkAttempt,
    *,
    task: Any,
    workflow: Any,
    trajectory_path: Optional[Path | str],
    agent_config: Any,
    exception: Optional[BaseException] = None,
) -> Path:
    """Write the authoritative manifest, then refresh legacy flat aliases."""
    completed_at = _utc_now()
    duration_seconds = max(0.0, time.monotonic() - attempt.started_monotonic)
    sql_path = attempt.output_dir / f"{attempt.task_id}.sql"
    result_path = attempt.output_dir / f"{attempt.task_id}.csv"
    outputs_exist = sql_path.is_file() and result_path.is_file()
    completed = exception is None and outputs_exist and _output_completed(workflow)
    structured_error = None if completed else _structured_error(workflow, exception, outputs_exist)

    outputs: list[dict[str, Any]] = []
    row_count = int(getattr(_last_sql_context(workflow), "row_count", 0) or 0)
    if sql_path.is_file():
        outputs.append(
            {
                "name": "generated_sql",
                "kind": "file",
                "format": "sql",
                "root": "save",
                "path": _relative_posix(sql_path, attempt.save_run_root),
                "metadata": {"canonical": True},
            }
        )
    if result_path.is_file():
        outputs.append(
            {
                "name": "sql_result",
                "kind": "file",
                "format": "csv",
                "root": "save",
                "path": _relative_posix(result_path, attempt.save_run_root),
                "metadata": {"row_count": row_count},
            }
        )

    trajectory_reference = None
    if trajectory_path:
        trajectory_file = Path(trajectory_path)
        if trajectory_file.is_file():
            trajectory_reference = {
                "root": "trajectory",
                "path": _relative_posix(trajectory_file, attempt.trajectory_run_root),
                "format": "yaml",
                "schema_version": 1,
                "contract_profile": "compatibility_v1",
                "attempt_id": attempt.attempt_id,
            }

    payload = {
        "artifact_type": TASK_OUTPUT_ARTIFACT_TYPE,
        "schema_version": TASK_OUTPUT_SCHEMA_VERSION,
        "run_id": attempt.run_id,
        "task_id": attempt.task_id,
        "task_type": attempt.task_type,
        "attempt_id": attempt.attempt_id,
        "status": "completed" if completed else "failed",
        "created_at": _rfc3339(attempt.created_at),
        "completed_at": _rfc3339(completed_at),
        "duration_seconds": duration_seconds,
        "model": _model_identity(workflow, agent_config),
        "usage": _aggregate_usage(workflow),
        "outputs": outputs,
        "trajectory": trajectory_reference,
        "error": structured_error,
        "metadata": {
            "capture_level": "standard",
            "datasource": str(getattr(task, "datasource", "") or ""),
            "database_name": str(getattr(task, "database_name", "") or ""),
            "legacy_compatibility": True,
        },
    }

    _atomic_write_json(attempt.manifest_path, payload)
    _publish_legacy_compatibility(
        attempt,
        task=task,
        completed=completed,
        error=structured_error,
        sql_path=sql_path,
        result_path=result_path,
        row_count=row_count,
    )
    return attempt.manifest_path


def load_task_output_manifest(run_root: Path | str, task_id: str) -> Optional[dict[str, Any]]:
    """Load and validate one task's v1 output manifest, or return ``None`` when absent."""
    safe_task_id = _validate_path_segment(task_id, "task_id")
    path = Path(run_root) / "tasks" / safe_task_id / TASK_OUTPUT_MANIFEST
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED,
            message=f"invalid benchmark task output manifest: {path}",
        )
    if payload.get("artifact_type") != TASK_OUTPUT_ARTIFACT_TYPE or payload.get("schema_version") != 1:
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED,
            message=f"unsupported benchmark task output manifest: {path}",
        )
    if str(payload.get("task_id")) != safe_task_id:
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED,
            message=f"benchmark task output manifest identity mismatch: {path}",
        )
    return payload


def resolve_task_output_path(run_root: Path | str, task_id: str, output_name: str) -> Optional[Path]:
    """Resolve a manifest-declared save output to an absolute path inside the run root."""
    root = Path(run_root)
    manifest = load_task_output_manifest(root, task_id)
    if manifest is None:
        return None
    for output in manifest.get("outputs") or []:
        if not isinstance(output, Mapping) or output.get("name") != output_name or output.get("root") != "save":
            continue
        raw_path = output.get("path")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=f"invalid benchmark output path: {raw_path!r}",
            )
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=f"invalid benchmark output path: {raw_path!r}",
            )
        resolved = (root / path).resolve()
        resolved.relative_to(root.resolve())
        return resolved
    return None


def resolve_task_trajectory_path(
    save_run_root: Path | str,
    trajectory_run_root: Path | str,
    task_id: str,
) -> Optional[Path]:
    """Resolve the manifest-referenced trajectory file inside the trajectory run root."""
    manifest = load_task_output_manifest(save_run_root, task_id)
    if manifest is None or not isinstance(manifest.get("trajectory"), Mapping):
        return None
    raw_path = manifest["trajectory"].get("path")
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"invalid benchmark trajectory path: {raw_path!r}",
        )
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"invalid benchmark trajectory path: {raw_path!r}",
        )
    root = Path(trajectory_run_root).resolve()
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved
